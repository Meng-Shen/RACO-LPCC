#!/usr/bin/env python3
"""Static validation for the XYZ-only CenterPoint configuration."""

import argparse
from pathlib import Path

import mmcv
import mmdet
import mmdet3d
import mmengine
import torch
from mmengine.config import Config
from mmengine.runner.checkpoint import load_checkpoint

from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules


def transform_types(pipeline):
    found = []
    for item in pipeline:
        found.append(item['type'])
        found.extend(transform_types(item.get('transforms', [])))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--load-checkpoint', action='store_true')
    args = parser.parse_args()

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(args.config.resolve()))
    assert cfg.model.pts_voxel_encoder.num_features == 3
    assert cfg.model.pts_middle_encoder.in_channels == 3
    assert cfg.train_dataloader.batch_size == 4
    assert 'LoadPointsFromMultiSweeps' not in transform_types(
        cfg.train_pipeline)
    assert 'LoadPointsFromMultiSweeps' not in transform_types(
        cfg.test_pipeline)
    assert cfg.train_pipeline[0].use_dim == [0, 1, 2]
    assert cfg.test_pipeline[0].use_dim == [0, 1, 2]
    assert cfg.train_cfg.max_epochs == 12

    model = MODELS.build(cfg.model)
    if args.load_checkpoint:
        load_checkpoint(
            model,
            str(args.checkpoint.resolve()),
            map_location='cpu',
            strict=False)
    print(
        'VALID '
        f'python_torch={torch.__version__} '
        f'cuda={torch.version.cuda} '
        f'mmcv={mmcv.__version__} '
        f'mmengine={mmengine.__version__} '
        f'mmdet={mmdet.__version__} '
        f'mmdet3d={mmdet3d.__version__} '
        'input=XYZ sweeps=0 gpus=4 batch_per_gpu=4 epochs=12')


if __name__ == '__main__':
    main()
