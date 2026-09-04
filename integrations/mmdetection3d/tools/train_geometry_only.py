# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import sys
from ast import literal_eval

from _bootstrap import MMDET_ROOT, PROJECT_ROOT, bootstrap_paths

bootstrap_paths()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune geometry-only MinkUNet from an xyz+intensity checkpoint')
    parser.add_argument(
        'config',
        nargs='?',
        default=str(
            PROJECT_ROOT / 'integrations' / 'mmdetection3d' / 'configs' /
            'minkunet' /
            'minkunet34_w32_minkowski_geometry_8xb2-laser-polar-mix-3x_semantickitti.py'),
        help='geometry-only train config file path')
    parser.add_argument(
        '--pretrained',
        default=str(
            MMDET_ROOT / 'checkpoints' /
            'minkunet34_w32_minkowski_8xb2-laser-polar-mix-3x_semantickitti_20230514_202236-839847a8.pth'),
        help='4-channel xyz+intensity checkpoint used as pretrained weights')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--sync_bn',
        choices=['none', 'torch', 'mmcv'],
        default='none',
        help='convert all BatchNorm layers in the model to SyncBatchNorm '
        '(SyncBN) or mmcv.ops.sync_bn.SyncBatchNorm (MMSyncBN) layers.')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        help='override config settings in key=value format')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def _get_checkpoint_state_dict(checkpoint):
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    if 'model_state' in checkpoint:
        return checkpoint['model_state']
    return checkpoint


def _strip_module_prefix(state_dict, target_state):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict))
    if first_key.startswith('module.') and not next(iter(target_state)).startswith('module.'):
        return {key[7:]: val for key, val in state_dict.items()}
    return state_dict


def _adapt_geometry_only_weight(key, val, target_shape):
    if not hasattr(val, 'shape') or len(val.shape) != len(target_shape):
        return None
    mismatch_dims = [
        dim for dim, (src, dst) in enumerate(zip(val.shape, target_shape))
        if src != dst
    ]
    if len(mismatch_dims) != 1:
        return None
    dim = mismatch_dims[0]
    if val.shape[dim] == 4 and target_shape[dim] == 3:
        slices = [slice(None)] * val.dim()
        slices[dim] = slice(0, 3)
        return val[tuple(slices)].contiguous()
    return None


def parse_cfg_options(options):
    cfg_options = {}
    for option in options:
        if '=' not in option:
            raise ValueError(f'Invalid cfg option: {option}')
        key, value = option.split('=', 1)
        try:
            value = literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        cfg_options[key] = value
    return cfg_options


def load_geometry_pretrained(model, filename):
    import torch
    from mmengine.logging import print_log

    checkpoint = torch.load(filename, map_location='cpu')
    source_state = _get_checkpoint_state_dict(checkpoint)
    # Runner wraps the model before this function is called under a
    # distributed launcher. Loading against the wrapper would make every
    # target key start with ``module.`` and silently skip the whole
    # checkpoint. Always adapt and load against the underlying segmentor.
    target_model = model.module if hasattr(model, 'module') else model
    target_state = target_model.state_dict()
    source_state = _strip_module_prefix(source_state, target_state)

    adapted_state = {}
    adapted_keys = []
    skipped_keys = []
    for key, val in source_state.items():
        if key not in target_state:
            skipped_keys.append(key)
            continue
        if target_state[key].shape == val.shape:
            adapted_state[key] = val
            continue
        adapted_val = _adapt_geometry_only_weight(key, val, target_state[key].shape)
        if adapted_val is not None:
            adapted_state[key] = adapted_val
            adapted_keys.append(key)
        else:
            skipped_keys.append(key)

    missing, unexpected = target_model.load_state_dict(
        adapted_state, strict=False)
    print_log(
        f'Loaded geometry pretrained checkpoint: {filename}; '
        f'loaded={len(adapted_state)}, adapted={len(adapted_keys)}, '
        f'skipped={len(skipped_keys)}, missing={len(missing)}, '
        f'unexpected={len(unexpected)}',
        logger='current')
    if adapted_keys:
        print_log('Adapted 4-channel weights to 3 channels: ' + ', '.join(adapted_keys),
                  logger='current')


def main():
    args = parse_args()

    from mmengine.config import Config
    from mmengine.logging import print_log
    from mmengine.registry import RUNNERS
    from mmengine.runner import Runner

    from mmdet3d.utils import replace_ceph_backend

    cfg = Config.fromfile(args.config)

    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(parse_cfg_options(args.cfg_options))

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            './work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    if args.sync_bn != 'none':
        cfg.sync_bn = args.sync_bn

    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find auto_scale_lr settings.')

    cfg.resume = False
    cfg.load_from = None

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    if args.pretrained:
        load_geometry_pretrained(runner.model, args.pretrained)

    runner.train()


if __name__ == '__main__':
    main()
