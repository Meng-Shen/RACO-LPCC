#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ROOT_DIR = Path(__file__).resolve().parents[1]
OPENPCDET_TOOLS = ROOT_DIR / 'OpenPCDet' / 'tools'
sys.path.insert(0, str(OPENPCDET_TOOLS))
import _init_path  # noqa: F401,E402

from eval_utils import eval_utils  # noqa: E402
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402


DEFAULT_SCALES = '1/64,1.5/128,1/128,1.5/256,1/256,1.5/512,1/512,1/2048'


def parse_number(value):
    value = str(value).strip()
    if '/' in value:
        num, den = value.split('/', 1)
        return float(num) / float(den)
    return float(value)


def parse_scales(value):
    rates = []
    for rate_id, item in enumerate(str(value).replace(';', ',').split(',')):
        label = item.strip()
        if not label:
            continue
        scale = parse_number(label)
        rates.append({'rate_id': rate_id, 'label': label, 'scale': scale, 'posQ': 1.0 / scale})
    if not rates:
        raise ValueError('--scales must contain at least one scale')
    return rates


def parse_config():
    parser = argparse.ArgumentParser(description='Evaluate OpenPCDet AP under RENO-equivalent quantization.')
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--extra_tag', type=str, default='reno')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888)
    parser.add_argument('--local_rank', type=int, default=None)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--eval_tag', type=str, default='default')
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--save_to_file', action='store_true', default=False)
    parser.add_argument('--infer_time', action='store_true')
    parser.add_argument('--scales', default=DEFAULT_SCALES)
    args = parser.parse_args()
    args.rates = parse_scales(args.scales)

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg


def reno_quantize_points(points, posq):
    coords = points[:, :3]
    coords_mm = np.round(coords.astype(np.float64) * 1000.0).astype(np.int64)
    offset = coords_mm.min(axis=0)
    coords_scaled = coords_mm - offset
    qcoords = np.round(coords_scaled.astype(np.float64) / float(posq)).astype(np.int64)
    qcoords = np.unique(qcoords, axis=0)
    coords_dec = (qcoords.astype(np.float64) * float(posq) + offset) * 0.001
    zeros = np.zeros((coords_dec.shape[0], 1), dtype=np.float32)
    return np.concatenate([coords_dec.astype(np.float32), zeros], axis=1)


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    model.load_params_from_file(
        filename=args.ckpt, logger=logger, to_cpu=dist_test,
        pre_trained_path=args.pretrained_model
    )
    model.cuda()
    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
    )


def progress_iter(items, **kwargs):
    if tqdm is None:
        return items
    return tqdm(items, **kwargs)


def main():
    args, cfg = parse_config()
    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, f'init_dist_{args.launcher}')(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = output_dir / 'eval'

    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if len(num_list) > 0 else 'no_number'
    eval_output_dir = eval_output_dir / f'epoch_{epoch_id}' / cfg.DATA_CONFIG.DATA_SPLIT['test']
    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = eval_output_dir / f'log_eval_reno_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt'
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    for key, val in vars(args).items():
        if key == 'rates':
            continue
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    orig_get_lidar = test_set.__class__.get_lidar

    def patched_get_lidar(self, idx):
        points = orig_get_lidar(self, idx)
        posq = getattr(self.__class__, 'current_reno_posq', None)
        if posq is None:
            return points
        return reno_quantize_points(points, posq)

    test_set.__class__.get_lidar = patched_get_lidar
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    with torch.no_grad():
        for rate in progress_iter(args.rates, desc='RENO AP scales', unit='scale'):
            logger.info('=================================================================================')
            logger.info(
                f"===================== Start RENO Evaluation rate_id={rate['rate_id']} "
                f"scale={rate['label']} posQ={rate['posQ']} ====================="
            )
            logger.info('=================================================================================')
            test_set.__class__.current_reno_posq = rate['posQ']

            _, cur_test_loader, _ = build_dataloader(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                batch_size=args.batch_size,
                dist=dist_test,
                workers=args.workers,
                logger=logger,
                training=False,
            )

            scale_eval_dir = eval_output_dir / f"rate_{rate['rate_id']}"
            scale_eval_dir.mkdir(parents=True, exist_ok=True)
            eval_single_ckpt(
                model=model,
                test_loader=cur_test_loader,
                args=args,
                eval_output_dir=scale_eval_dir,
                logger=logger,
                epoch_id=epoch_id,
                dist_test=dist_test,
            )
    logger.info('********************** All RENO scales evaluated successfully **********************')


if __name__ == '__main__':
    main()
