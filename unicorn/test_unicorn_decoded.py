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


def parse_rate_ids(value):
    ids = []
    for item in str(value).replace(',', ' ').split():
        if item.strip():
            ids.append(int(item))
    if not ids:
        raise ValueError('--rate_ids must contain at least one rate id')
    return ids


def parse_config():
    parser = argparse.ArgumentParser(description='Evaluate OpenPCDet AP from Unicorn decoded .bin files.')
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--extra_tag', type=str, default='unicorn')
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
    parser.add_argument('--decoded_dir', required=True)
    parser.add_argument('--rate_ids', required=True)
    args = parser.parse_args()
    args.rate_ids = parse_rate_ids(args.rate_ids)

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg


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

    log_file = eval_output_dir / f'log_eval_unicorn_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt'
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    for key, val in vars(args).items():
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

    decoded_root = Path(args.decoded_dir)
    orig_get_lidar = test_set.__class__.get_lidar

    def patched_get_lidar(self, idx):
        rate_id = getattr(self.__class__, 'current_unicorn_rate_id', None)
        if rate_id is None:
            return orig_get_lidar(self, idx)
        decoded_file = decoded_root / f'rate_{rate_id}' / f'{idx}.bin'
        if not decoded_file.exists():
            raise FileNotFoundError(decoded_file)
        points = np.fromfile(str(decoded_file), dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 0.0
        return points

    test_set.__class__.get_lidar = patched_get_lidar
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    with torch.no_grad():
        for rate_id in progress_iter(args.rate_ids, desc='Unicorn AP rates', unit='rate'):
            logger.info('=================================================================================')
            logger.info(f'===================== Start Unicorn Evaluation rate_id={rate_id} =====================')
            logger.info('=================================================================================')
            test_set.__class__.current_unicorn_rate_id = rate_id
            _, cur_test_loader, _ = build_dataloader(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                batch_size=args.batch_size,
                dist=dist_test,
                workers=args.workers,
                logger=logger,
                training=False,
            )
            rate_eval_dir = eval_output_dir / f'rate_{rate_id}'
            rate_eval_dir.mkdir(parents=True, exist_ok=True)
            eval_single_ckpt(
                model=model,
                test_loader=cur_test_loader,
                args=args,
                eval_output_dir=rate_eval_dir,
                logger=logger,
                epoch_id=epoch_id,
                dist_test=dist_test,
            )
    logger.info('********************** All Unicorn rates evaluated successfully **********************')


if __name__ == '__main__':
    main()
