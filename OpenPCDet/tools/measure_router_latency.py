#!/usr/bin/env python3
import argparse
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from export_router_jucp import RouterInferenceDataset, load_calibrator, load_train_args
from train_cost_proxy import SparseCostProxyNet, read_split_file, sparse_collate_fn


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'n': int(values.size),
        'mean_s': float(values.mean()),
        'median_s': float(np.median(values)),
        'min_s': float(values.min()),
        'max_s': float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(description='Measure router proxy latency.')
    parser.add_argument('--velodyne_dir', required=True)
    parser.add_argument('--split_file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--calibration', default=None)
    parser.add_argument('--out_csv', required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()

    train_args, checkpoint = load_train_args(args.ckpt)
    ns = SimpleNamespace(**train_args)
    voxel_size = getattr(ns, 'voxel_size', [0.16, 0.16, 0.16])
    pc_range = getattr(ns, 'point_cloud_range', [0.0, -40.0, -3.0, 70.4, 40.0, 1.0])
    max_voxels = getattr(ns, 'max_voxels', 50000)
    num_cost_heads = getattr(ns, 'num_cost_heads', 6)
    num_targets = getattr(ns, 'num_targets', 3)
    feat_dim = getattr(ns, 'feat_dim', 256)
    use_abs_xyz = not getattr(ns, 'no_abs_xyz', False)
    allow_negative_cost = getattr(ns, 'allow_negative_cost', False)
    monotonic_cost = not getattr(ns, 'no_monotonic_cost', False)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    dataset = RouterInferenceDataset(args.velodyne_dir, args.split_file, voxel_size, pc_range, max_voxels, use_abs_xyz)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=sparse_collate_fn,
    )

    model = SparseCostProxyNet(
        input_channels=dataset.num_point_features,
        spatial_shape=dataset.spatial_shape,
        feat_dim=feat_dim,
        num_cost_heads=num_cost_heads,
        num_targets=num_targets,
        cost_nonnegative=not allow_negative_cost,
        monotonic_cost=monotonic_cost,
    ).to(device)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    calibrator = load_calibrator(args.calibration, device, num_targets, allow_negative=allow_negative_cost)

    batches = list(loader)
    if not batches:
        raise RuntimeError('No router timing batches found.')

    with torch.no_grad():
        for batch in batches[:max(0, args.warmup)]:
            voxel_features = batch['voxel_features'].to(device, non_blocking=True)
            voxel_coords = batch['voxel_coords'].to(device, non_blocking=True)
            cost = model(voxel_features, voxel_coords, batch['batch_size'])['cost_pred']
            if calibrator is not None:
                cost = calibrator(cost)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    rows = []
    frame_count = len(read_split_file(args.split_file))
    start_total = time.perf_counter()
    with torch.no_grad():
        for batch in batches:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_batch = time.perf_counter()
            voxel_features = batch['voxel_features'].to(device, non_blocking=True)
            voxel_coords = batch['voxel_coords'].to(device, non_blocking=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_forward = time.perf_counter()
            cost = model(voxel_features, voxel_coords, batch['batch_size'])['cost_pred']
            if calibrator is not None:
                cost = calibrator(cost)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()

            batch_size = int(batch['batch_size'])
            rows.append({
                'frame_ids': ';'.join(batch['frame_id']),
                'batch_size': batch_size,
                'model_forward_s': (end - start_forward) / batch_size,
                'h2d_plus_forward_s': (end - start_batch) / batch_size,
            })
    total_elapsed = time.perf_counter() - start_total

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        fieldnames = ['frame_ids', 'batch_size', 'model_forward_s', 'h2d_plus_forward_s']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'frame_ids': row['frame_ids'],
                'batch_size': row['batch_size'],
                'model_forward_s': f"{row['model_forward_s']:.8f}",
                'h2d_plus_forward_s': f"{row['h2d_plus_forward_s']:.8f}",
            })

    forward = summarize([row['model_forward_s'] for row in rows])
    h2d = summarize([row['h2d_plus_forward_s'] for row in rows])
    print(f'Frames: {frame_count}')
    print(f'Batch size: {args.batch_size}')
    print(f'Model forward mean: {forward["mean_s"]:.8f}s = {forward["mean_s"] * 1000:.3f} ms/frame')
    print(f'Model forward median: {forward["median_s"] * 1000:.3f} ms/frame')
    print(f'Model forward range: {forward["min_s"] * 1000:.3f}-{forward["max_s"] * 1000:.3f} ms/frame')
    print(f'H2D + forward mean: {h2d["mean_s"]:.8f}s = {h2d["mean_s"] * 1000:.3f} ms/frame')
    print(f'Total timed loop per frame: {total_elapsed / frame_count:.8f}s = {total_elapsed / frame_count * 1000:.3f} ms/frame')
    print(f'CSV: {out_csv}')


if __name__ == '__main__':
    main()
