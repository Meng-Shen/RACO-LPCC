#!/usr/bin/env python3
"""Benchmark the irreducible GPU forward cost of a MinkUNet model.

The benchmark deliberately excludes source-point quantization, coordinate
sorting, MMDetection3D's generic inference pipeline, host-to-device copies,
device-to-host copies, file I/O, and model/checkpoint initialization.  Input
features and Minkowski coordinates are prepared once and kept on the GPU.

Each timed invocation still creates a fresh MinkowskiEngine SparseTensor inside
the backbone.  Its coordinate manager and per-layer kernel maps are therefore
part of the measurement; those are required for a new point-cloud frame.
"""

import argparse
import csv
import json
import platform
import statistics
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings

from mmdet3d.apis import inference_detector, init_model


DEFAULT_SCALES = (64, 128, 256, 512, 1024, 2048)


def sorted_coords(coords):
    coords = np.asarray(coords, dtype=np.int32).reshape(-1, 3)
    if len(coords) == 0:
        return np.ascontiguousarray(coords)
    order = np.lexsort((coords[:, 0], coords[:, 1], coords[:, 2]))
    return np.ascontiguousarray(coords[order])


def quantize_coarse(ref_xyz, quant_step_mm):
    xyz_mm = np.rint(
        np.asarray(ref_xyz, dtype=np.float64) * 1000.0).astype(np.int64)
    offset_mm = xyz_mm.min(axis=0)
    point_coords = np.rint(
        (xyz_mm - offset_mm) / float(quant_step_mm)).astype(np.int32)
    return sorted_coords(np.unique(point_coords, axis=0))


def load_frame_ids(split_file, max_frames=None):
    frame_ids = [
        line.strip() for line in Path(split_file).read_text().splitlines()
        if line.strip()
    ]
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    if not frame_ids:
        raise ValueError(f'No frame IDs in {split_file}')
    return frame_ids


def load_xyz(testdata, frame_id):
    path = Path(testdata) / f'{frame_id}.bin'
    points = np.fromfile(path, dtype=np.float32)
    if points.size % 4:
        raise ValueError(f'Invalid KITTI point file: {path}')
    return points.reshape(-1, 4)[:, :3]


def build_gpu_inputs(coarse_coords, device):
    """Reproduce the model's post-voxelization tensors without its pipeline."""
    coarse_coords = np.ascontiguousarray(coarse_coords, dtype=np.int32)
    if len(coarse_coords) == 0:
        raise ValueError('Cannot benchmark an empty point cloud')

    features_np = np.zeros((len(coarse_coords), 4), dtype=np.float32)
    features_np[:, :3] = coarse_coords.astype(np.float32)

    shifted = coarse_coords - coarse_coords.min(axis=0, keepdims=True)
    coors_np = np.zeros((len(coarse_coords), 4), dtype=np.int32)
    # batch_first=True in the active model data preprocessor.
    coors_np[:, 1:] = shifted

    features = torch.from_numpy(features_np).to(device=device)
    coors = torch.from_numpy(coors_np).to(device=device)
    return {
        'voxels': {
            'voxels': features,
            'coors': coors,
        }
    }, features_np


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


@torch.inference_mode()
def one_timed_forward(model, inputs, device):
    """Measure device timeline and synchronized wall latency for one call."""
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    forward_event = torch.cuda.Event(enable_timing=True)
    mask_event = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    start_event.record()
    logits = model._forward(inputs)
    forward_event.record()
    mask = torch.argmax(logits, dim=1)
    mask_event.record()
    mask_event.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    forward_cuda_ms = start_event.elapsed_time(forward_event)
    with_mask_cuda_ms = start_event.elapsed_time(mask_event)
    foreground_count = int(mask.sum().item())
    return forward_cuda_ms, with_mask_cuda_ms, wall_ms, foreground_count


@torch.inference_mode()
def validate_direct_path(model, coarse_coords, direct_inputs, features_np):
    """Verify direct logits produce the same labels as the public API once."""
    inference_cfg = deepcopy(model.cfg)
    inference_cfg.test_dataloader.dataset = dict(
        box_type_3d='LiDAR',
        pipeline=[
            dict(
                type='LoadPointsFromDict',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(type='Pack3DDetInputs', keys=['points']),
        ])
    model.cfg = inference_cfg

    public_result, _ = inference_detector(model, features_np)
    public_labels = public_result.pred_pts_seg.pts_semantic_mask
    public_labels = public_labels.detach().cpu().numpy().reshape(-1)

    direct_labels = model._forward(direct_inputs).argmax(dim=1)
    direct_labels = direct_labels.detach().cpu().numpy().reshape(-1)
    if not np.array_equal(public_labels, direct_labels):
        mismatch = int(np.count_nonzero(public_labels != direct_labels))
        raise RuntimeError(
            'Direct GPU input does not reproduce public inference labels: '
            f'{mismatch}/{len(public_labels)} mismatches')
    return int(np.count_nonzero(direct_labels == 1))


def summarize_case(frame_id, quant_step_mm, num_voxels, first_call, samples):
    forward_values = [sample[0] for sample in samples]
    with_mask_values = [sample[1] for sample in samples]
    wall_values = [sample[2] for sample in samples]
    foreground_counts = [sample[3] for sample in samples]
    if len(set(foreground_counts)) != 1:
        raise RuntimeError('Foreground prediction changed across repetitions')
    return {
        'frame_id': frame_id,
        'quant_step_mm': int(quant_step_mm),
        'num_voxels': int(num_voxels),
        'foreground_voxels': int(foreground_counts[0]),
        'first_call_forward_cuda_ms': round(float(first_call[0]), 6),
        'first_call_with_mask_cuda_ms': round(float(first_call[1]), 6),
        'first_call_wall_ms': round(float(first_call[2]), 6),
        'forward_cuda_mean_ms': round(
            statistics.fmean(forward_values), 6),
        'forward_cuda_median_ms': round(
            statistics.median(forward_values), 6),
        'forward_cuda_p90_ms': round(percentile(forward_values, 90), 6),
        'with_mask_cuda_mean_ms': round(
            statistics.fmean(with_mask_values), 6),
        'with_mask_cuda_median_ms': round(
            statistics.median(with_mask_values), 6),
        'with_mask_cuda_p90_ms': round(
            percentile(with_mask_values, 90), 6),
        'wall_mean_ms': round(statistics.fmean(wall_values), 6),
        'wall_median_ms': round(statistics.median(wall_values), 6),
        'wall_p90_ms': round(percentile(wall_values, 90), 6),
    }


def aggregate_cases(cases):
    aggregated = []
    for quant_step_mm in sorted({row['quant_step_mm'] for row in cases}):
        group = [
            row for row in cases
            if row['quant_step_mm'] == quant_step_mm
        ]
        aggregated.append({
            'quant_step_mm': quant_step_mm,
            'num_frames': len(group),
            'mean_voxels': round(statistics.fmean(
                row['num_voxels'] for row in group), 3),
            'mean_foreground_voxels': round(statistics.fmean(
                row['foreground_voxels'] for row in group), 3),
            'forward_cuda_ms': round(statistics.fmean(
                row['forward_cuda_median_ms'] for row in group), 6),
            'with_mask_cuda_ms': round(statistics.fmean(
                row['with_mask_cuda_median_ms'] for row in group), 6),
            'wall_ms': round(statistics.fmean(
                row['wall_median_ms'] for row in group), 6),
            'first_call_wall_ms': round(statistics.fmean(
                row['first_call_wall_ms'] for row in group), 6),
        })
    return aggregated


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--testdata', required=True)
    parser.add_argument('--split_file', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_label', default='W16')
    parser.add_argument('--output_prefix', default='w16')
    parser.add_argument(
        '--scales', default=','.join(map(str, DEFAULT_SCALES)))
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--max_frames', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--skip_validation', action='store_true')
    args = parser.parse_args()

    if args.warmup < 1 or args.repeats < 1:
        raise ValueError('warmup and repeats must both be positive')
    scales = tuple(int(value) for value in args.scales.split(','))
    frame_ids = load_frame_ids(args.split_file, args.max_frames)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires a CUDA device')

    model_cfg = Config.fromfile(args.config)
    if model_cfg.get('custom_imports'):
        import_modules_from_strings(**model_cfg.custom_imports)
    model = init_model(model_cfg, args.checkpoint, device=str(device))
    model.eval()

    cases = []
    validation_done = False
    total = len(frame_ids) * len(scales)
    case_index = 0
    for frame_id in frame_ids:
        ref_xyz = load_xyz(args.testdata, frame_id)
        for quant_step_mm in scales:
            case_index += 1
            coarse_coords = quantize_coarse(ref_xyz, quant_step_mm)
            direct_inputs, features_np = build_gpu_inputs(
                coarse_coords, device)

            if not validation_done and not args.skip_validation:
                foreground_count = validate_direct_path(
                    model, coarse_coords, direct_inputs, features_np)
                print(
                    'Validated direct GPU path against inference_detector: '
                    f'{len(coarse_coords)} voxels, '
                    f'{foreground_count} foreground')
                validation_done = True

            first_call = one_timed_forward(model, direct_inputs, device)
            for _ in range(args.warmup):
                model._forward(direct_inputs)
            torch.cuda.synchronize(device)

            samples = [
                one_timed_forward(model, direct_inputs, device)
                for _ in range(args.repeats)
            ]
            row = summarize_case(
                frame_id, quant_step_mm, len(coarse_coords),
                first_call, samples)
            cases.append(row)
            print(
                f'[{case_index}/{total}] {frame_id} Q={quant_step_mm}: '
                f'N={len(coarse_coords)}, '
                f"forward={row['forward_cuda_median_ms']:.3f} ms, "
                f"+mask={row['with_mask_cuda_median_ms']:.3f} ms, "
                f"wall={row['wall_median_ms']:.3f} ms")

    aggregated = aggregate_cases(cases)
    output_dir = Path(args.output_dir)
    write_csv(
        output_dir / f'{args.output_prefix}_forward_per_frame.csv', cases)
    write_csv(
        output_dir / f'{args.output_prefix}_forward_by_scale.csv', aggregated)
    metadata = {
        'model_label': args.model_label,
        'gpu': torch.cuda.get_device_name(device),
        'cuda_device': str(device),
        'torch': torch.__version__,
        'python': platform.python_version(),
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'frames': frame_ids,
        'scales_mm': scales,
        'warmup': args.warmup,
        'repeats': args.repeats,
        'measurement_mode': (
            'Warmed steady-state. first_call fields mean the first measured '
            'call for that prepared input, not process cold start.'),
        'timing_scope': {
            'included': [
                'MinkowskiEngine SparseTensor creation inside backbone',
                'backbone',
                'decode head',
                'GPU argmax in with_mask metric',
            ],
            'excluded': [
                'model initialization and checkpoint loading',
                'source quantization and coordinate sorting',
                'MMDetection3D inference pipeline',
                'input allocation and host-to-device transfer',
                'label device-to-host transfer',
                'file I/O',
            ],
        },
    }
    (output_dir / f'{args.output_prefix}_forward_metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')

    print('\nPer-scale averages of per-frame medians:')
    print(
        'Q(mm)  voxels    forward_cuda(ms)  +mask_cuda(ms)  wall(ms)')
    for row in aggregated:
        print(
            f"{row['quant_step_mm']:>5}  "
            f"{row['mean_voxels']:>8.1f}  "
            f"{row['forward_cuda_ms']:>16.3f}  "
            f"{row['with_mask_cuda_ms']:>14.3f}  "
            f"{row['wall_ms']:>8.3f}")


if __name__ == '__main__':
    main()
