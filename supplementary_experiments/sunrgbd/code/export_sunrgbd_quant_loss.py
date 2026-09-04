#!/usr/bin/env python3
"""Resumable per-scene VoteNet loss export for six SUN RGB-D geometry steps."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmdet3d.registry import DATASETS, MODELS, TRANSFORMS


QSTEPS_MM = (160.0, 80.0, 40.0, 20.0, 10.0, 5.0)


@TRANSFORMS.register_module(force=True)
class SUNRGBDUniformQuantize(BaseTransform):
    def __init__(self, qstep_mm: float):
        self.qstep_mm = float(qstep_mm)
        self.last_stats = None

    def transform(self, results: dict) -> dict:
        points = results['points']
        xyz = points.tensor[:, :3].detach().cpu().numpy().astype(np.float64)
        raw_count = len(xyz)
        coords_mm = np.rint(xyz * 1000.0).astype(np.int64)
        offset_mm = coords_mm.min(axis=0)
        grid = np.rint((coords_mm - offset_mm) / self.qstep_mm).astype(np.int64)
        unique_grid = np.unique(grid, axis=0)
        decoded_xyz = (
            unique_grid.astype(np.float64) * self.qstep_mm + offset_mm
        ) / 1000.0
        floor_height = np.percentile(decoded_xyz[:, 2], 0.99)
        height = decoded_xyz[:, 2:3] - floor_height
        decoded = np.concatenate([decoded_xyz, height], axis=1).astype(np.float32)
        results['points'] = points.new_point(decoded)
        self.last_stats = {
            'raw_num_points': raw_count,
            'unique_num_points': len(decoded),
            'retention': len(decoded) / raw_count,
        }
        return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--max-scenes', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20260828)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--qsteps-mm', type=float, nargs='+', default=QSTEPS_MM)
    return parser.parse_args()


def make_dataset_cfg(cfg, data_root: str, split: str, qstep_mm: float):
    source = cfg.val_dataloader.dataset
    return dict(
        type='SUNRGBDDataset',
        data_root=str(Path(data_root).resolve()) + '/',
        ann_file=f'sunrgbd_infos_{split}.pkl',
        metainfo=source.metainfo,
        box_type_3d='Depth',
        filter_empty_gt=False,
        test_mode=False,
        backend_args=None,
        pipeline=[
            dict(
                type='LoadPointsFromFile', coord_type='DEPTH', shift_height=True,
                load_dim=6, use_dim=[0, 1, 2], backend_args=None,
            ),
            dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
            dict(type=SUNRGBDUniformQuantize, qstep_mm=qstep_mm),
            dict(type='PointSample', num_points=20000),
            dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
        ],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scene_id(dataset, index: int) -> str:
    info = dataset.get_data_info(index)
    return Path(info['lidar_points']['lidar_path']).stem


def evaluate_loss(model, item: dict) -> dict[str, float]:
    batch = pseudo_collate([item])
    data = model.data_preprocessor(batch, training=False)
    with torch.no_grad():
        losses = model._run_forward(data, mode='loss')
        _, log_vars = model.parse_losses(losses)
    return {
        key: float(value.detach().cpu() if torch.is_tensor(value) else value)
        for key, value in log_vars.items()
    }


def load_completed(path: Path, num_levels: int) -> tuple[list[dict], set[str]]:
    if not path.is_file():
        return [], set()
    rows = []
    seen = set()
    try:
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                sid = row.get('scene_id', '')
                try:
                    values = [
                        float(row[f'L{i}_total_loss'])
                        for i in range(num_levels)
                    ]
                except (KeyError, ValueError):
                    continue
                if sid and sid not in seen and all(np.isfinite(values)):
                    rows.append(row)
                    seen.add(sid)
    except csv.Error:
        pass
    return rows, seen


def atomic_write(path: Path, rows: list[dict]) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    fields = list(rows[0])
    with temp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError('Invalid shard specification')
    init_default_scope('mmdet3d')
    cfg = Config.fromfile(str(Path(args.config).resolve()))
    datasets = [
        DATASETS.build(make_dataset_cfg(cfg, args.data_root, args.split, qstep))
        for qstep in args.qsteps_mm
    ]
    for dataset in datasets:
        dataset.full_init()
    total = len(datasets[0])
    if any(len(dataset) != total for dataset in datasets):
        raise RuntimeError('Dataset lengths differ across quantization levels')
    indices = list(range(args.shard_id, total, args.num_shards))
    if args.max_scenes > 0:
        indices = indices[:args.max_scenes]

    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(Path(args.checkpoint).resolve()), map_location='cpu')
    model.to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, completed = load_completed(output, len(args.qsteps_mm))
    started = time.time()
    for ordinal, index in enumerate(indices, 1):
        sid = scene_id(datasets[0], index)
        if sid in completed:
            continue
        row = {'scene_id': sid, 'dataset_index': index, 'split': args.split}
        for level, (qstep, dataset) in enumerate(zip(args.qsteps_mm, datasets)):
            set_seed(args.seed + index)
            item = dataset[index]
            stats = evaluate_loss(model, item)
            quantizer = next(
                transform for transform in dataset.pipeline.transforms
                if isinstance(transform, SUNRGBDUniformQuantize)
            )
            if quantizer.last_stats is None:
                raise RuntimeError('Quantizer statistics are missing')
            row[f'L{level}_qstep_mm'] = qstep
            row[f'L{level}_raw_num_points'] = quantizer.last_stats['raw_num_points']
            row[f'L{level}_unique_num_points'] = quantizer.last_stats['unique_num_points']
            row[f'L{level}_retention'] = quantizer.last_stats['retention']
            row[f'L{level}_total_loss'] = stats['loss']
            for key, value in stats.items():
                if key != 'loss':
                    row[f'L{level}_{key}'] = value
        rows.append(row)
        rows.sort(key=lambda item: int(item['dataset_index']))
        atomic_write(output, rows)
        completed.add(sid)
        if ordinal == 1 or ordinal % args.log_every == 0 or ordinal == len(indices):
            print(json.dumps({
                'split': args.split, 'shard': args.shard_id,
                'visited': ordinal, 'assigned': len(indices), 'scene_id': sid,
                'rows': len(rows), 'elapsed_seconds': time.time() - started,
                'losses': [
                    row[f'L{i}_total_loss']
                    for i in range(len(args.qsteps_mm))
                ],
            }), flush=True)

    output.with_suffix('.manifest.json').write_text(json.dumps({
        'status': 'complete', 'dataset': 'SUN RGB-D', 'split': args.split,
        'qsteps_mm_coarse_to_fine': args.qsteps_mm,
        'absolute_task_losses': True,
        'geometry_only': True, 'num_levels': len(args.qsteps_mm),
        'assigned_scenes': len(indices),
        'completed_rows': len(rows), 'shard_id': args.shard_id,
        'num_shards': args.num_shards, 'checkpoint': str(Path(args.checkpoint).resolve()),
    }, indent=2))


if __name__ == '__main__':
    main()
