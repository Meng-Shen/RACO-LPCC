#!/usr/bin/env python3
"""Plot fixed G-PCC and true-loss oracle SUN RGB-D curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmengine
import numpy as np
from mmengine.logging import MMLogger
from mmdet3d.datasets import SUNRGBDDataset
from mmdet3d.evaluation import indoor_eval
from mmdet3d.structures import get_box_type


NUM_LEVELS = 6


def load_loss(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    scene_ids = []
    values = {}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            sid = row['scene_id']
            if sid in values:
                raise RuntimeError(f'Duplicate loss scene {sid} in {path}')
            scene_ids.append(sid)
            values[sid] = np.asarray(
                [float(row[f'L{i}_total_loss']) for i in range(NUM_LEVELS)],
                dtype=np.float64,
            )
    return scene_ids, values


def load_gpcc(path: Path) -> dict[tuple[str, int], tuple[int, int]]:
    values = {}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = (row['scene_id'], int(row['rate_id']))
            if key in values:
                raise RuntimeError(f'Duplicate G-PCC key {key} in {path}')
            values[key] = (int(row['bits']), int(row['num_points']))
    return values


def aligned_arrays(scene_ids, losses, gpcc):
    loss = np.stack([losses[sid] for sid in scene_ids])
    bits = np.asarray([
        [gpcc[(sid, level)][0] for level in range(NUM_LEVELS)]
        for sid in scene_ids
    ], dtype=np.float64)
    points = np.asarray([
        [gpcc[(sid, level)][1] for level in range(NUM_LEVELS)]
        for sid in scene_ids
    ], dtype=np.float64)
    if np.any(points != points[:, :1]):
        raise RuntimeError('Original point count changes across G-PCC levels')
    return loss, bits / points, bits, points[:, 0]


def aggregate_bpp(bits, points, levels):
    selected_bits = bits[np.arange(len(levels)), levels]
    return float(selected_bits.sum() / points.sum())


def calibrate_lambdas(train_loss, train_bpp, train_bits, train_points,
                      calibration_split, evaluation_split_used):
    loss_span = max(float(np.median(np.ptp(train_loss, axis=1))), 1e-9)
    bpp_span = max(float(np.median(np.ptp(train_bpp, axis=1))), 1e-9)
    base = loss_span / bpp_span
    candidates = np.concatenate([
        [0.0],
        np.logspace(np.log10(base) - 6.0, np.log10(base) + 8.0, 3000),
    ])
    selections = np.empty((len(candidates), len(train_loss)), dtype=np.uint8)
    achieved_bpp = np.empty(len(candidates), dtype=np.float64)
    for index, value in enumerate(candidates):
        levels = np.argmin(train_loss + value * train_bpp, axis=1).astype(np.uint8)
        selections[index] = levels
        achieved_bpp[index] = aggregate_bpp(
            train_bits, train_points, levels.astype(np.int64))

    order = np.argsort(achieved_bpp)
    unique_indices = []
    previous = None
    for index in order:
        current = achieved_bpp[index]
        if previous is None or abs(current - previous) > 1e-12:
            unique_indices.append(int(index))
            previous = current
    if len(unique_indices) < NUM_LEVELS:
        raise RuntimeError(
            f'Only {len(unique_indices)} distinct oracle BPP points are available')

    unique_indices = np.asarray(unique_indices, dtype=np.int64)
    unique_bpp = achieved_bpp[unique_indices]
    targets = np.geomspace(unique_bpp[0], unique_bpp[-1], NUM_LEVELS)
    chosen = []
    used = set()
    for target in targets:
        distances = np.abs(np.log(unique_bpp) - np.log(target))
        for position in np.argsort(distances):
            candidate_index = int(unique_indices[position])
            if candidate_index not in used:
                chosen.append(candidate_index)
                used.add(candidate_index)
                break
    chosen.sort(key=lambda index: achieved_bpp[index])
    lambdas = candidates[chosen]
    selected = selections[chosen].T.astype(np.int64)
    return lambdas, selected, {
        'status': 'complete',
        'calibration_split': calibration_split,
        'evaluation_split_used_for_lambda_selection': evaluation_split_used,
        'routing_rule': 'argmin true_absolute_task_loss + lambda * true_G-PCC_bpp',
        'lambda_scale_base': base,
        'lambdas_low_rate_to_high_rate': lambdas.tolist(),
        'calibration_aggregate_bpp': [
            float(achieved_bpp[index]) for index in chosen],
        'calibration_selection_counts': [
            np.bincount(selected[:, rate], minlength=NUM_LEVELS).tolist()
            for rate in range(NUM_LEVELS)
        ],
    }


def select_levels(loss, bpp, lambdas):
    scores = loss[:, None, :] + lambdas[None, :, None] * bpp[:, None, :]
    return np.argmin(scores, axis=2)


def evaluate(records, levels, logger):
    annotations = [record['eval_ann_info'] for record in records]
    predictions = [
        record['predictions'][int(level)] for record, level in zip(records, levels)
    ]
    _, box_mode = get_box_type('depth')
    metrics = indoor_eval(
        annotations, predictions, [0.25, 0.5],
        SUNRGBDDataset.METAINFO['classes'], logger=logger,
        box_mode_3d=box_mode,
    )
    return {key: float(value) for key, value in metrics.items()}


def load_predictions(root: Path, expected_scenes: int, expected_shards: int,
                     split_name: str):
    records = []
    manifests = []
    paths = sorted(root.glob('shard_*/predictions.pkl'))
    if len(paths) != expected_shards:
        raise RuntimeError(f'Expected {expected_shards} prediction shards, got {len(paths)}')
    for path in paths:
        records.extend(mmengine.load(path))
        manifests.append(json.loads(path.with_suffix('.manifest.json').read_text()))
    records.sort(key=lambda row: int(row['dataset_index']))
    if len(records) != expected_scenes:
        raise RuntimeError(f'Expected {expected_scenes} records, got {len(records)}')
    indices = [int(row['dataset_index']) for row in records]
    if indices != list(range(expected_scenes)):
        raise RuntimeError(
            f'Prediction indices do not exactly cover SUN RGB-D {split_name}')
    manifest_splits = {manifest.get('split') for manifest in manifests}
    if manifest_splits != {split_name}:
        raise RuntimeError(
            f'Prediction manifests contain splits {manifest_splits}, '
            f'expected only {split_name}')
    return records, manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction-root', required=True, type=Path)
    parser.add_argument('--loss-csv', required=True, type=Path)
    parser.add_argument('--gpcc-csv', required=True, type=Path)
    parser.add_argument('--calibration-loss-csv', type=Path)
    parser.add_argument('--calibration-gpcc-csv', type=Path)
    parser.add_argument('--calibration-split-name')
    parser.add_argument('--split-name', choices=('train', 'val'), required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--expected-scenes', type=int, default=5050)
    parser.add_argument('--expected-shards', type=int, default=7)
    args = parser.parse_args()

    records, manifests = load_predictions(
        args.prediction_root, args.expected_scenes, args.expected_shards,
        args.split_name)
    prediction_scene_ids = [row['scene_id'] for row in records]
    scene_ids, losses = load_loss(args.loss_csv)
    if prediction_scene_ids != scene_ids:
        raise RuntimeError('Prediction and task-loss scene order differs')
    gpcc = load_gpcc(args.gpcc_csv)
    eval_loss, eval_bpp, eval_bits, eval_points = aligned_arrays(
        scene_ids, losses, gpcc)
    calibration_loss_path = args.calibration_loss_csv or args.loss_csv
    calibration_gpcc_path = args.calibration_gpcc_csv or args.gpcc_csv
    calibration_scene_ids, calibration_losses = load_loss(
        calibration_loss_path)
    calibration_gpcc = load_gpcc(calibration_gpcc_path)
    calibration_loss, calibration_bpp, calibration_bits, calibration_points = (
        aligned_arrays(
            calibration_scene_ids, calibration_losses, calibration_gpcc))
    same_split_calibration = (
        calibration_loss_path.resolve() == args.loss_csv.resolve()
        and calibration_gpcc_path.resolve() == args.gpcc_csv.resolve())
    calibration_split_name = (
        args.calibration_split_name or args.split_name)
    lambdas, _, calibration = calibrate_lambdas(
        calibration_loss, calibration_bpp, calibration_bits,
        calibration_points,
        f'official SUN RGB-D {calibration_split_name}',
        same_split_calibration)
    oracle_levels = select_levels(eval_loss, eval_bpp, lambdas)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = MMLogger.get_instance(
        'sunrgbd_oracle_gpcc_ap_bpp',
        log_file=str(args.output_dir / 'evaluation.log'), log_level='INFO')
    qsteps = manifests[0]['qsteps_mm_coarse_to_fine']
    rows = []
    for level, qstep in enumerate(qsteps):
        levels = np.full(args.expected_scenes, level, dtype=np.int64)
        metrics = evaluate(records, levels, logger)
        rows.append({
            'series': 'fixed_gpcc_baseline', 'rate_point': level,
            'qstep_mm': qstep, 'lambda': '',
            'bpp': aggregate_bpp(eval_bits, eval_points, levels),
            'mAP_0.25': metrics['mAP_0.25'],
            'mAP_0.50': metrics['mAP_0.50'],
            'selection_counts': ';'.join(
                str(args.expected_scenes if i == level else 0)
                for i in range(NUM_LEVELS)),
        })
    for rate in range(NUM_LEVELS):
        levels = oracle_levels[:, rate]
        metrics = evaluate(records, levels, logger)
        rows.append({
            'series': 'true_loss_oracle', 'rate_point': rate,
            'qstep_mm': '', 'lambda': float(lambdas[rate]),
            'bpp': aggregate_bpp(eval_bits, eval_points, levels),
            'mAP_0.25': metrics['mAP_0.25'],
            'mAP_0.50': metrics['mAP_0.50'],
            'selection_counts': ';'.join(
                str(int(value)) for value in
                np.bincount(levels, minlength=NUM_LEVELS)),
        })

    csv_path = args.output_dir / 'sunrgbd_true_loss_oracle_vs_gpcc_ap_bpp.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=180)
    for axis, key, title in (
        (axes[0], 'mAP_0.25', f'SUN RGB-D {args.split_name}: mAP@0.25'),
        (axes[1], 'mAP_0.50', f'SUN RGB-D {args.split_name}: mAP@0.50'),
    ):
        for series, label, color, marker in (
            ('fixed_gpcc_baseline', 'Fixed G-PCC', '#4C78A8', 'o'),
            ('true_loss_oracle', 'True-loss oracle', '#E45756', 's'),
        ):
            selected = sorted(
                [row for row in rows if row['series'] == series],
                key=lambda row: float(row['bpp']))
            axis.plot(
                [float(row['bpp']) for row in selected],
                [100.0 * float(row[key]) for row in selected],
                color=color, marker=marker, linewidth=2.2, markersize=5.5,
                label=label,
            )
        axis.set_xlabel('BPP (total G-PCC bits / total original points)')
        axis.set_ylabel('mAP (%)')
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    png_path = args.output_dir / 'sunrgbd_true_loss_oracle_vs_gpcc_ap_bpp.png'
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)

    payload = {
        'status': 'complete',
        'dataset': f'SUN RGB-D {args.split_name}',
        'evaluation_scope': (
            'exploratory same-split oracle preview' if same_split_calibration
            else 'held-out evaluation with separate calibration split'),
        'curves_shown': ['fixed G-PCC baseline', 'true-loss oracle'],
        'qsteps_mm_coarse_to_fine': qsteps,
        'bpp_definition': 'sum selected encoded geometry bits / sum original points',
        'calibration': calibration, 'rows': rows,
        'csv': str(csv_path.resolve()), 'png': str(png_path.resolve()),
    }
    (args.output_dir / 'ORACLE_BASELINE_COMPLETE.json').write_text(
        json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
