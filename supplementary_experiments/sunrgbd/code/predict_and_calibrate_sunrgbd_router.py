#!/usr/bin/env python3
"""Calibrate six train-only lambdas and predict SUN RGB-D route selections."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from gpu_voxelizer import voxelize_batch_gpu
from lite_s3_absolute_loss_monotonic_rate_proxy import LiteS3AbsoluteLossMonotonicRateProxy
from train_sunrgbd_lite_s3_router_ddp import (
    QSTEPS_MM,
    SUNRGBDRouterDataset,
    collate_raw,
    read_ids,
)


class SUNRGBDInferenceDataset(Dataset):
    def __init__(self, points_dir: Path, split_file: Path, bpp_csv: Path):
        self.points_dir = points_dir
        self.ids = read_ids(split_file)
        rates = {}
        with bpp_csv.open(newline='') as handle:
            for row in csv.DictReader(handle):
                sid = row['scene_id']
                rates.setdefault(sid, np.full(6, np.nan, np.float32))[
                    int(row['rate_id'])] = float(row['bpp'])
        missing = [sid for sid in self.ids if sid not in rates]
        if missing:
            raise RuntimeError(f'Missing BPP for {len(missing)} scenes')
        self.rates = rates

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]
        points = np.fromfile(
            self.points_dir / f'{sid}.bin', dtype=np.float32).reshape(-1, 6)
        return {
            'scene_id': sid,
            'points': torch.from_numpy(points[:, :3].copy()),
            'bpp': torch.from_numpy(self.rates[sid].copy()),
        }


def collate_inference(batch):
    return {
        'scene_ids': [item['scene_id'] for item in batch],
        'points': [item['points'] for item in batch],
        'bpp': torch.stack([item['bpp'] for item in batch]),
    }


def predict(model, dataset, batch_size, workers, device, model_args, collate_fn):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, collate_fn=collate_fn,
    )
    scene_ids, predicted_loss, predicted_bpp, true_loss, true_bpp = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            points = [cloud.to(device, non_blocking=True) for cloud in batch['points']]
            features, coords = voxelize_batch_gpu(
                points, model_args['voxel_size'], model_args['point_cloud_range'],
                model_args['max_voxels'], use_abs_xyz=True,
                include_intensity=False, random_subsample=False,
            )
            output = model(features, coords, len(points))
            scene_ids.extend(batch['scene_ids'])
            predicted_loss.append(output['loss_pred'].cpu().numpy())
            predicted_bpp.append(output['bpp_pred'].cpu().numpy())
            if 'loss' in batch:
                true_loss.append(batch['loss'].numpy())
            if 'raw_bpp' in batch:
                true_bpp.append(batch['raw_bpp'].numpy())
            elif 'bpp' in batch:
                true_bpp.append(batch['bpp'].numpy())
            if batch_index == 0 or (batch_index + 1) % 50 == 0:
                print(json.dumps({
                    'first_or_periodic_batch': batch_index + 1,
                    'scenes_seen': len(scene_ids), 'active_voxels': int(features.shape[0]),
                }), flush=True)
    return {
        'scene_ids': scene_ids,
        'predicted_loss': np.concatenate(predicted_loss),
        'predicted_bpp': np.concatenate(predicted_bpp),
        'true_loss': np.concatenate(true_loss) if true_loss else None,
        'true_bpp': np.concatenate(true_bpp) if true_bpp else None,
    }


def selected_levels(loss: np.ndarray, bpp: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    scores = loss[:, None, :] + lambdas[None, :, None] * bpp[:, None, :]
    return scores.argmin(axis=2)


def calibrate_lambdas(train_predictions: dict) -> tuple[np.ndarray, dict]:
    loss = train_predictions['predicted_loss'].astype(np.float64)
    bpp = train_predictions['predicted_bpp'].astype(np.float64)
    loss_span = np.median(np.ptp(loss, axis=1))
    bpp_span = max(np.median(np.ptp(bpp, axis=1)), 1e-6)
    base = max(loss_span / bpp_span, 1e-6)
    candidates = np.concatenate([[0.0], np.logspace(
        np.log10(base) - 5.0, np.log10(base) + 5.0, 2000)])
    levels = selected_levels(loss, bpp, candidates)
    true_bpp = train_predictions['true_bpp'].astype(np.float64)
    achieved_bpp = np.take_along_axis(
        true_bpp[:, None, :], levels[:, :, None], axis=2).squeeze(2).mean(axis=0)
    order = np.argsort(achieved_bpp)
    minimum = max(float(achieved_bpp[order[0]]), 1e-12)
    maximum = max(float(achieved_bpp[order[-1]]), minimum)
    targets = np.geomspace(minimum, maximum, 6)
    chosen = []
    used = set()
    for target in targets:
        distances = np.abs(np.log(np.maximum(achieved_bpp, 1e-12)) - np.log(target))
        for candidate_index in np.argsort(distances):
            index = int(candidate_index)
            signature = tuple(np.bincount(levels[:, index], minlength=6).tolist())
            if index not in used and signature not in {
                tuple(np.bincount(levels[:, old], minlength=6).tolist()) for old in chosen
            }:
                chosen.append(index)
                used.add(index)
                break
    if len(chosen) != 6:
        raise RuntimeError(f'Could only calibrate {len(chosen)} distinct routing points')
    chosen.sort(key=lambda index: achieved_bpp[index])
    lambdas = candidates[chosen]
    chosen_levels = levels[:, chosen]
    report = {
        'lambda_scale_base': base,
        'lambdas_low_rate_to_high_rate': lambdas.tolist(),
        'train_aggregate_true_bpp': achieved_bpp[chosen].tolist(),
        'train_mean_selected_level': chosen_levels.mean(axis=0).tolist(),
        'train_selection_counts': [
            np.bincount(chosen_levels[:, i], minlength=6).tolist() for i in range(6)
        ],
        'calibration_uses_test': False,
        'calibration_split': 'full official SUN RGB-D train (5285 scenes)',
    }
    return lambdas, report


def write_predictions(path: Path, predictions: dict, lambdas: np.ndarray) -> dict:
    levels = selected_levels(
        predictions['predicted_loss'], predictions['predicted_bpp'], lambdas)
    fields = ['scene_id']
    fields += [f'pred_loss_L{i}' for i in range(6)]
    fields += [f'pred_bpp_L{i}' for i in range(6)]
    fields += [f'selected_level_R{i}' for i in range(6)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index, sid in enumerate(predictions['scene_ids']):
            row = {'scene_id': sid}
            row.update({f'pred_loss_L{i}': predictions['predicted_loss'][row_index, i]
                        for i in range(6)})
            row.update({f'pred_bpp_L{i}': predictions['predicted_bpp'][row_index, i]
                        for i in range(6)})
            row.update({f'selected_level_R{i}': int(levels[row_index, i])
                        for i in range(6)})
            writer.writerow(row)
    report = {
        'scenes': len(predictions['scene_ids']),
        'bpp_monotonic_violation_rate': float(
            (np.diff(predictions['predicted_bpp'], axis=1) < 0).mean()),
        'selection_counts': [
            np.bincount(levels[:, i], minlength=6).tolist() for i in range(6)
        ],
        'output': str(path.resolve()),
    }
    if predictions['true_loss'] is not None:
        report['loss_mae'] = float(np.abs(
            predictions['predicted_loss'] - predictions['true_loss']).mean())
    if predictions['true_bpp'] is not None:
        report['bpp_mae'] = float(np.abs(
            predictions['predicted_bpp'] - predictions['true_bpp']).mean())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--points-dir', required=True, type=Path)
    parser.add_argument('--train-split', required=True, type=Path)
    parser.add_argument('--val-split', required=True, type=Path)
    parser.add_argument('--train-loss-csv', required=True, type=Path)
    parser.add_argument('--train-bpp-csv', required=True, type=Path)
    parser.add_argument('--val-bpp-csv', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    train_set = SUNRGBDRouterDataset(
        args.points_dir, args.train_split, args.train_loss_csv,
        args.train_bpp_csv, training=False)
    val_set = SUNRGBDInferenceDataset(
        args.points_dir, args.val_split, args.val_bpp_csv)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_args = checkpoint['args']
    grid_xyz = np.floor(
        (np.asarray(model_args['point_cloud_range'][3:])
         - np.asarray(model_args['point_cloud_range'][:3]))
        / np.asarray(model_args['voxel_size'])
    ).astype(int)
    model = LiteS3AbsoluteLossMonotonicRateProxy(
        grid_xyz[::-1].tolist(), 256, checkpoint['loss_scales'],
        train_set.mean_log_bpp,
    )
    state = {
        (key[7:] if key.startswith('module.') else key): value
        for key, value in checkpoint['model'].items()
    }
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model.to(device)

    train_predictions = predict(
        model, train_set, args.batch_size, args.workers, device, model_args,
        collate_raw)
    lambdas, calibration = calibrate_lambdas(train_predictions)
    val_predictions = predict(
        model, val_set, args.batch_size, args.workers, device, model_args,
        collate_inference)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_report = write_predictions(
        args.output_dir / 'train_router_predictions.csv', train_predictions, lambdas)
    val_report = write_predictions(
        args.output_dir / 'val_router_predictions.csv', val_predictions, lambdas)
    payload = {
        'status': 'complete', 'qsteps_mm_coarse_to_fine': QSTEPS_MM,
        'routing_rule': 'argmin predicted_loss + lambda * predicted_bpp',
        'calibration': calibration, 'train': train_report, 'val': val_report,
        'checkpoint': str(args.checkpoint.resolve()),
    }
    (args.output_dir / 'lambda_calibration_and_metrics.json').write_text(
        json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
