#!/usr/bin/env python3
"""Cache six fixed-quantization VoteNet predictions for a SUN RGB-D split."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mmengine
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmdet3d.registry import DATASETS, MODELS

from export_sunrgbd_quant_loss import QSTEPS_MM, SUNRGBDUniformQuantize


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataset_cfg(cfg, data_root, split, qstep):
    source = cfg.val_dataloader.dataset
    return dict(
        type='SUNRGBDDataset',
        data_root=str(Path(data_root).resolve()) + '/',
        ann_file=f'sunrgbd_infos_{split}.pkl', metainfo=source.metainfo,
        box_type_3d='Depth', test_mode=True, backend_args=None,
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=True,
                 load_dim=6, use_dim=[0, 1, 2], backend_args=None),
            dict(type=SUNRGBDUniformQuantize, qstep_mm=qstep),
            dict(type='PointSample', num_points=20000),
            dict(type='Pack3DDetInputs', keys=['points']),
        ],
    )


def cpu_prediction(instances):
    return {key: (value.to('cpu') if hasattr(value, 'to') else value)
            for key, value in instances.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--max-scenes', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20260828)
    parser.add_argument('--qsteps-mm', type=float, nargs='+', default=QSTEPS_MM)
    args = parser.parse_args()

    init_default_scope('mmdet3d')
    cfg = Config.fromfile(str(Path(args.config).resolve()))
    datasets = [
        DATASETS.build(dataset_cfg(cfg, args.data_root, args.split, q))
        for q in args.qsteps_mm
    ]
    for dataset in datasets:
        dataset.full_init()
    total = len(datasets[0])
    indices = list(range(args.shard_id, total, args.num_shards))
    if args.max_scenes > 0:
        indices = indices[:args.max_scenes]

    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(Path(args.checkpoint).resolve()), map_location='cpu')
    model.to(args.device).eval()
    records, started = [], time.time()
    for ordinal, index in enumerate(indices, 1):
        predictions, eval_ann_info, sid = [], None, None
        for dataset in datasets:
            set_seed(args.seed + index)
            with torch.no_grad():
                result = model.test_step(pseudo_collate([dataset[index]]))[0]
            if eval_ann_info is None:
                eval_ann_info = result.eval_ann_info
                sid = Path(result.metainfo.get('lidar_path', '')).stem
                if not sid:
                    sid = Path(dataset.get_data_info(index)['lidar_points']['lidar_path']).stem
            predictions.append(cpu_prediction(result.pred_instances_3d))
        records.append({
            'scene_id': sid, 'dataset_index': index,
            'eval_ann_info': eval_ann_info, 'predictions': predictions,
        })
        if ordinal == 1 or ordinal % 20 == 0 or ordinal == len(indices):
            print(json.dumps({
                'shard': args.shard_id, 'visited': ordinal,
                'assigned': len(indices), 'scene_id': sid,
                'elapsed_seconds': time.time() - started,
            }), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mmengine.dump(records, args.output)
    args.output.with_suffix('.manifest.json').write_text(json.dumps({
        'status': 'complete', 'dataset': 'SUN RGB-D', 'split': args.split,
        'qsteps_mm_coarse_to_fine': args.qsteps_mm,
        'shard_id': args.shard_id,
        'num_shards': args.num_shards, 'records': len(records),
        'checkpoint': str(Path(args.checkpoint).resolve()),
    }, indent=2))


if __name__ == '__main__':
    main()
