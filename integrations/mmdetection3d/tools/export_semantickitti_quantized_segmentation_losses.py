#!/usr/bin/env python3
"""Export per-frame SemanticKITTI loss at six uniform geometry steps.

Each quantized cloud is segmented by an XYZ-only MinkUNet. Its hard labels
are transferred to every original point with a 3-D nearest-neighbour query.
The primary loss is the point-wise 0/1 error after ignoring label 19. Per-frame
mIoU is also recorded for auditing. Eight independent torchrun workers write
resumable shards; ``--merge-only`` validates and combines them.
"""

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

from _bootstrap import MMDET_ROOT, bootstrap_paths

bootstrap_paths()

import numpy as np
import torch
from scipy.spatial import cKDTree


DEFAULT_TRAIN_SEQUENCES = ('00', '01', '02', '03', '04', '05', '06',
                           '07', '09', '10')
DEFAULT_STEPS_MM = (2048, 1024, 512, 256, 128, 64)
IGNORE_LABEL = 19
NUM_CLASSES = 19


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--router-data-dir', type=Path)
    parser.add_argument('--steps-mm', type=int, nargs=6,
                        default=DEFAULT_STEPS_MM)
    parser.add_argument('--loss-thresholds', type=float, nargs=6,
                        default=(0.0, 0.005, 0.01, 0.02, 0.04, 0.08))
    parser.add_argument('--merge-only', action='store_true')
    parser.add_argument('--expected-world-size', type=int, default=8)
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Only for a smoke test; zero means the full set.')
    return parser.parse_args()


def log(message):
    rank = int(os.environ.get('RANK', '0'))
    print(f'[{time.strftime("%F %T")}][rank {rank}] {message}', flush=True)


def load_train_items(dataset_root, max_frames=0):
    info_path = dataset_root / 'semantickitti_infos_train.pkl'
    if not info_path.is_file():
        raise FileNotFoundError(f'Missing training info file: {info_path}')
    with info_path.open('rb') as handle:
        payload = pickle.load(handle)
    items = []
    for raw in payload['data_list']:
        frame_id = str(raw['sample_idx'])
        sequence = frame_id.split('_', 1)[0]
        if sequence not in DEFAULT_TRAIN_SEQUENCES:
            continue
        point_path = dataset_root / raw['lidar_points']['lidar_path']
        label_path = dataset_root / raw['pts_semantic_mask_path']
        items.append({
            'frame_id': frame_id,
            'point_path': point_path,
            'label_path': label_path,
        })
    items.sort(key=lambda item: item['frame_id'])
    if max_frames > 0:
        items = items[:max_frames]
    if not items:
        raise RuntimeError('No SemanticKITTI training frames were found')
    frame_ids = [item['frame_id'] for item in items]
    if len(frame_ids) != len(set(frame_ids)):
        raise RuntimeError('Duplicate SemanticKITTI frame IDs in info file')
    return items


def load_points_and_labels(item):
    points = np.fromfile(item['point_path'], dtype=np.float32)
    if points.size % 4:
        raise ValueError(f'Invalid point cloud: {item["point_path"]}')
    points = points.reshape(-1, 4)
    raw_labels = np.fromfile(item['label_path'], dtype=np.uint32)
    if len(points) != len(raw_labels):
        raise ValueError(
            f'Point/label count mismatch for {item["frame_id"]}: '
            f'{len(points)} vs {len(raw_labels)}')
    return points[:, :3].astype(np.float32, copy=False), raw_labels


def build_label_lookup(config_path):
    repo_root = MMDET_ROOT
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from mmengine.config import Config

    cfg = Config.fromfile(str(config_path))
    mapping = dict(cfg.metainfo['seg_label_mapping'])
    lookup = np.full(1 << 16, IGNORE_LABEL, dtype=np.int64)
    for source, target in mapping.items():
        lookup[int(source)] = int(target)
    return lookup


def quantize_xyz(xyz_m, step_mm):
    """Match the KITTI detector experiment's millimetre-origin quantizer."""
    if len(xyz_m) == 0:
        return np.empty((0, 3), dtype=np.float32)
    xyz_mm = np.rint(xyz_m.astype(np.float64) * 1000.0).astype(np.int64)
    offset_mm = xyz_mm.min(axis=0, keepdims=True)
    lattice = np.rint(
        (xyz_mm - offset_mm).astype(np.float64) / float(step_mm)
    ).astype(np.int64)
    unique_lattice = np.unique(lattice, axis=0)
    decoded_mm = unique_lattice * int(step_mm) + offset_mm
    return (decoded_mm.astype(np.float64) / 1000.0).astype(np.float32)


def build_model(config_path, checkpoint_path, device):
    repo_root = MMDET_ROOT
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from mmengine.config import Config
    from mmengine.runner.checkpoint import load_checkpoint
    from mmdet3d.registry import MODELS
    from mmdet3d.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config_path))
    if int(cfg.model.backbone.in_channels) != 3:
        raise ValueError('Loss export requires the XYZ-only 3-channel model')
    if int(cfg.model.decode_head.num_classes) != NUM_CLASSES:
        raise ValueError('Loss export requires the standard 19-class head')
    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(checkpoint_path), map_location='cpu')
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_quantized_labels(model, quantized_xyz, device):
    from mmdet3d.structures import Det3DDataSample, PointData

    points = torch.from_numpy(quantized_xyz).to(device=device,
                                                  dtype=torch.float32)
    data_sample = Det3DDataSample()
    # The MinkUNet preprocessor checks for this field even in test mode.
    data_sample.gt_pts_seg = PointData(
        pts_semantic_mask=torch.full(
            (len(points),), IGNORE_LABEL, dtype=torch.long, device=device))
    batch = {
        'inputs': {'points': [points]},
        'data_samples': [data_sample],
    }
    output = model.test_step(batch)[0]
    prediction = output.pred_pts_seg.pts_semantic_mask
    prediction = prediction.detach().cpu().numpy().astype(np.int64, copy=False)
    if len(prediction) != len(quantized_xyz):
        raise RuntimeError(
            f'Prediction/quantized point mismatch: {len(prediction)} vs '
            f'{len(quantized_xyz)}')
    return prediction


def transfer_with_nearest_neighbor(original_xyz, quantized_xyz,
                                   quantized_prediction):
    if len(quantized_xyz) == 0:
        raise RuntimeError('Cannot transfer labels from an empty cloud')
    tree = cKDTree(quantized_xyz.astype(np.float64, copy=False))
    _, nearest = tree.query(
        original_xyz.astype(np.float64, copy=False), k=1, workers=1)
    return quantized_prediction[np.asarray(nearest, dtype=np.int64)]


def segmentation_metrics(prediction, ground_truth):
    valid = ground_truth != IGNORE_LABEL
    prediction = prediction[valid]
    ground_truth = ground_truth[valid]
    if len(ground_truth) == 0:
        raise RuntimeError('Frame has no valid semantic labels')
    if prediction.min() < 0 or prediction.max() >= NUM_CLASSES:
        raise ValueError('Predicted label lies outside the 19 valid classes')
    point_error = float(np.mean(prediction != ground_truth))
    confusion = np.bincount(
        ground_truth * NUM_CLASSES + prediction,
        minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - intersection
    present = union > 0
    miou = float(np.mean(intersection[present] / union[present]))
    return point_error, miou, int(len(ground_truth))


def fieldnames(steps_mm):
    fields = ['frame_id', 'finest_label', 'finest_quant_step_mm',
              'finest_point_error']
    for label, _ in enumerate(steps_mm):
        fields.extend([
            f'L{label}_quant_step_mm',
            f'L{label}_point_error',
            f'L{label}_loss_delta',
            f'L{label}_miou',
            f'L{label}_miou_loss',
            f'L{label}_valid_points',
            f'L{label}_quantized_points',
        ])
    return fields


def repair_partial_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open('rb+') as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            return
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b'\n':
            return
        handle.seek(0)
        content = handle.read()
        last_newline = content.rfind(b'\n')
        handle.seek(0)
        handle.truncate(last_newline + 1 if last_newline >= 0 else 0)


def completed_frame_ids(path, expected_fields):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields:
            raise ValueError(f'Unexpected shard header in {path}')
        completed = set()
        for row in reader:
            try:
                for key in expected_fields[1:]:
                    float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            completed.add(row['frame_id'])
    return completed


def evaluate_worker(args, items):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', str(rank)))
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f'Expected {args.expected_world_size} workers, got {world_size}')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    worker_items = items[rank::world_size]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = args.output_dir / f'loss_shard_{rank:02d}_of_{world_size:02d}.csv'
    expected_fields = fieldnames(args.steps_mm)
    repair_partial_csv(shard_path)
    completed = completed_frame_ids(shard_path, expected_fields)
    pending = [item for item in worker_items
               if item['frame_id'] not in completed]
    log(f'Assigned {len(worker_items)} frames; {len(completed)} complete; '
        f'{len(pending)} pending on cuda:{local_rank}')

    label_lookup = build_label_lookup(args.config)
    model = build_model(args.config, args.checkpoint, device)
    exists = shard_path.exists() and shard_path.stat().st_size > 0
    with shard_path.open('a', newline='', buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=expected_fields)
        if not exists:
            writer.writeheader()
        start = time.time()
        for local_index, item in enumerate(pending, 1):
            original_xyz, raw_labels = load_points_and_labels(item)
            ground_truth = label_lookup[(raw_labels & 0xFFFF).astype(np.int64)]
            step_stats = []
            for step_mm in args.steps_mm:
                quantized_xyz = quantize_xyz(original_xyz, step_mm)
                quantized_prediction = predict_quantized_labels(
                    model, quantized_xyz, device)
                original_prediction = transfer_with_nearest_neighbor(
                    original_xyz, quantized_xyz, quantized_prediction)
                point_error, miou, valid_points = segmentation_metrics(
                    original_prediction, ground_truth)
                step_stats.append({
                    'point_error': point_error,
                    'miou': miou,
                    'valid_points': valid_points,
                    'quantized_points': len(quantized_xyz),
                })

            finest_label = len(args.steps_mm) - 1
            finest_error = step_stats[finest_label]['point_error']
            row = {
                'frame_id': item['frame_id'],
                'finest_label': finest_label,
                'finest_quant_step_mm': args.steps_mm[finest_label],
                'finest_point_error': f'{finest_error:.10f}',
            }
            for label, (step_mm, stats) in enumerate(
                    zip(args.steps_mm, step_stats)):
                row.update({
                    f'L{label}_quant_step_mm': step_mm,
                    f'L{label}_point_error': f'{stats["point_error"]:.10f}',
                    f'L{label}_loss_delta':
                    f'{stats["point_error"] - finest_error:.10f}',
                    f'L{label}_miou': f'{stats["miou"]:.10f}',
                    f'L{label}_miou_loss': f'{1.0 - stats["miou"]:.10f}',
                    f'L{label}_valid_points': stats['valid_points'],
                    f'L{label}_quantized_points': stats['quantized_points'],
                })
            writer.writerow(row)
            if local_index % 10 == 0:
                handle.flush()
                os.fsync(handle.fileno())
            if local_index == 1 or local_index % 25 == 0:
                elapsed = time.time() - start
                rate = local_index / max(elapsed, 1e-6)
                remaining = (len(pending) - local_index) / max(rate, 1e-6)
                log(f'{local_index}/{len(pending)} new frames; '
                    f'ETA {remaining / 60.0:.1f} min')
        handle.flush()
        os.fsync(handle.fileno())
    log(f'Worker shard complete: {shard_path}')


def number_name(value):
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_router_data(items, router_data_dir):
    velodyne_dir = router_data_dir / 'velodyne'
    velodyne_dir.mkdir(parents=True, exist_ok=True)
    split_path = router_data_dir / 'train.txt'
    for item in items:
        link = velodyne_dir / f'{item["frame_id"]}.bin'
        target = item['point_path'].resolve()
        if link.is_symlink():
            if link.resolve() != target:
                raise RuntimeError(f'Wrong existing symlink: {link}')
        elif link.exists():
            raise RuntimeError(f'Refusing to replace non-symlink: {link}')
        else:
            link.symlink_to(target)
    with split_path.open('w') as handle:
        for item in items:
            handle.write(item['frame_id'] + '\n')
    return velodyne_dir, split_path


def merge_shards(args, items):
    expected_fields = fieldnames(args.steps_mm)
    by_frame = {}
    shard_paths = []
    for rank in range(args.expected_world_size):
        path = args.output_dir / (
            f'loss_shard_{rank:02d}_of_{args.expected_world_size:02d}.csv')
        if not path.is_file():
            raise FileNotFoundError(f'Missing loss shard: {path}')
        shard_paths.append(path)
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_fields:
                raise ValueError(f'Unexpected shard header: {path}')
            for row in reader:
                frame_id = row['frame_id']
                if frame_id in by_frame:
                    raise RuntimeError(f'Duplicate loss row for {frame_id}')
                for label, step_mm in enumerate(args.steps_mm):
                    if int(row[f'L{label}_quant_step_mm']) != step_mm:
                        raise ValueError(f'Wrong step for {frame_id}, L{label}')
                    point_error = float(row[f'L{label}_point_error'])
                    if not 0.0 <= point_error <= 1.0:
                        raise ValueError(f'Invalid point error for {frame_id}')
                if abs(float(row['L5_loss_delta'])) > 1e-8:
                    raise ValueError(f'L5 is not the zero baseline: {frame_id}')
                by_frame[frame_id] = row

    expected_ids = [item['frame_id'] for item in items]
    missing = [frame_id for frame_id in expected_ids
               if frame_id not in by_frame]
    extras = sorted(set(by_frame) - set(expected_ids))
    if missing or extras:
        raise RuntimeError(
            f'Shard coverage mismatch: missing={len(missing)}, '
            f'extras={len(extras)}, first_missing={missing[:3]}')

    merged_path = args.output_dir / 'train_segmentation_loss_sensitivity.csv'
    with merged_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=expected_fields)
        writer.writeheader()
        for frame_id in expected_ids:
            writer.writerow(by_frame[frame_id])

    label_dir = args.output_dir / 'labels'
    label_dir.mkdir(parents=True, exist_ok=True)
    distribution = []
    label_paths = []
    for rate_id, threshold in enumerate(args.loss_thresholds):
        path = label_dir / (
            f'segmentation_loss_rate_{rate_id}_{number_name(threshold)}.csv')
        counts = np.zeros(len(args.steps_mm), dtype=np.int64)
        with path.open('w', newline='') as handle:
            fields = ['frame_id', 'jucp_label', 'rate_id', 'threshold',
                      'quant_step_mm', 'selected_point_error',
                      'selected_loss_delta']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for frame_id in expected_ids:
                row = by_frame[frame_id]
                chosen = len(args.steps_mm) - 1
                for label in range(len(args.steps_mm)):
                    if float(row[f'L{label}_loss_delta']) <= threshold:
                        chosen = label
                        break
                counts[chosen] += 1
                writer.writerow({
                    'frame_id': frame_id,
                    'jucp_label': chosen,
                    'rate_id': rate_id,
                    'threshold': threshold,
                    'quant_step_mm': args.steps_mm[chosen],
                    'selected_point_error': row[f'L{chosen}_point_error'],
                    'selected_loss_delta': row[f'L{chosen}_loss_delta'],
                })
        label_paths.append(str(path))
        distribution.append({
            'rate_id': rate_id,
            'threshold': threshold,
            'counts': counts.tolist(),
        })

    router_paths = None
    if args.router_data_dir is not None:
        velodyne_dir, split_path = prepare_router_data(
            items, args.router_data_dir)
        router_paths = {
            'velodyne_dir': str(velodyne_dir),
            'train_split': str(split_path),
        }

    manifest = {
        'mode': 'semantic_kitti_xyz19_hard_label_nearest_neighbor_loss',
        'dataset_root': str(args.dataset_root.resolve()),
        'config': str(args.config.resolve()),
        'checkpoint': str(args.checkpoint.resolve()),
        'quant_steps_mm_coarse_to_fine': list(args.steps_mm),
        'baseline_label': len(args.steps_mm) - 1,
        'baseline_quant_step_mm': args.steps_mm[-1],
        'loss_definition': (
            'mean(predicted_label_after_3d_nearest_neighbor_transfer != '
            'ground_truth_label) over original non-ignore points'),
        'loss_delta_definition': 'candidate_point_error - 64mm_point_error',
        'ignore_label': IGNORE_LABEL,
        'num_classes': NUM_CLASSES,
        'num_frames': len(items),
        'loss_thresholds': list(args.loss_thresholds),
        'loss_csv': str(merged_path),
        'loss_csv_sha256': sha256(merged_path),
        'label_csvs': label_paths,
        'label_distribution': distribution,
        'shards': [str(path) for path in shard_paths],
        'router_data': router_paths,
    }
    manifest_path = args.output_dir / 'segmentation_loss_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    complete_path = args.output_dir / 'LOSS_LABELS_COMPLETE.json'
    complete_path.write_text(json.dumps({
        'completed_at': time.strftime('%F %T'),
        'num_frames': len(items),
        'loss_csv': str(merged_path),
        'sha256': manifest['loss_csv_sha256'],
    }, indent=2))
    log(f'Merged and verified {len(items)} loss rows: {merged_path}')
    log(f'Manifest: {manifest_path}')


def validate_args(args):
    args.dataset_root = args.dataset_root.resolve()
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.router_data_dir is not None:
        args.router_data_dir = args.router_data_dir.resolve()
    if tuple(args.steps_mm) != DEFAULT_STEPS_MM:
        raise ValueError(
            f'Expected KITTI detection steps {DEFAULT_STEPS_MM}, got '
            f'{tuple(args.steps_mm)}')
    if list(args.loss_thresholds) != sorted(args.loss_thresholds):
        raise ValueError('--loss-thresholds must be nondecreasing')
    for path in (args.dataset_root, args.config, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)


def main():
    args = parse_args()
    validate_args(args)
    items = load_train_items(args.dataset_root, args.max_frames)
    if args.merge_only:
        merge_shards(args, items)
    else:
        evaluate_worker(args, items)


if __name__ == '__main__':
    main()
