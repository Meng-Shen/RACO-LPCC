#!/usr/bin/env python3
"""Build an XYZ-only detector and execute one real nuScenes loss step."""

import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import DATASETS, MODELS


def scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().cpu())
    if isinstance(value, (list, tuple)):
        return sum(scalar(item) for item in value)
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope('mmdet3d')
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    dataset.full_init()
    item = dataset[0]
    points = item['inputs']['points']
    point_tensor = points.tensor if hasattr(points, 'tensor') else points
    point_features = int(point_tensor.shape[1])
    if point_features != 3:
        raise RuntimeError(f'Expected strict XYZ input, got {point_features} channels')
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda().train()
    batch = pseudo_collate([item])
    data = model.data_preprocessor(batch, training=True)
    with torch.no_grad():
        losses = model._run_forward(data, mode='loss')
        total, log_vars = model.parse_losses(losses)
    payload = {
        'status': 'passed',
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'dataset_frames': len(dataset),
        'input_channels': point_features,
        'total_loss': float(total.detach().cpu()),
        'loss_terms': {key: scalar(value) for key, value in log_vars.items()},
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
