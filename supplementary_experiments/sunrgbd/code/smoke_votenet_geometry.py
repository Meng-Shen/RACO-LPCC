#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmdet3d.registry import DATASETS, MODELS


CONFIG = Path('/home/sm/sunrgbd_lite_s3_20260828/configs/votenet_sunrgbd_geometry_finetune.py')
CHECKPOINT = Path('/home/sm/sunrgbd_lite_s3_20260828/checkpoints/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth')
OUTPUT = Path('/home/sm/sunrgbd_lite_s3_20260828/state/VOTENET_SMOKE_COMPLETE.json')


def main() -> None:
    init_default_scope('mmdet3d')
    cfg = Config.fromfile(str(CONFIG))
    dataset_cfg = cfg.train_dataloader.dataset.dataset
    dataset = DATASETS.build(dataset_cfg)
    dataset.full_init()
    batch = pseudo_collate([dataset[0], dataset[1]])

    model = MODELS.build(cfg.model).cuda()
    checkpoint_info = load_checkpoint(model, str(CHECKPOINT), map_location='cpu')
    model.train()
    data = model.data_preprocessor(batch, training=True)
    losses = model._run_forward(data, mode='loss')
    total_loss, log_vars = model.parse_losses(losses)
    total_loss.backward()

    point_dims = [int(points.shape[1]) for points in data['inputs']['points']]
    payload = {
        'status': 'complete',
        'dataset_samples': len(dataset),
        'batch_size': 2,
        'point_feature_dims': point_dims,
        'geometry_only_xyz_plus_derived_height': all(dim == 4 for dim in point_dims),
        'total_loss': float(total_loss.detach().cpu()),
        'loss_terms': {key: float(value) for key, value in log_vars.items()},
        'checkpoint_meta_keys': sorted(checkpoint_info.get('meta', {}).keys()),
        'finite_gradients': all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        ),
    }
    if not payload['geometry_only_xyz_plus_derived_height']:
        raise RuntimeError(f'Unexpected point feature dimensions: {point_dims}')
    if not payload['finite_gradients']:
        raise RuntimeError('Non-finite gradients in VoteNet smoke test')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
