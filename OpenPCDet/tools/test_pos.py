import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test, 
                                pre_trained_path=args.pretrained_model)
    model.cuda()
    
    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
    )


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

        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_pos_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    # =========================================================================
    # 首次构建 Dataset 以获取类，应用全局量化钩子 (Monkey Patch)
    # =========================================================================
    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    orig_get_lidar = test_set.__class__.get_lidar

    def patched_get_lidar(self, idx):
        points = orig_get_lidar(self, idx)
        
        # 注意：使用 __class__ 访问变量，确保 DataLoader workers 能继承主进程状态
        scale = getattr(self.__class__, 'current_scale', 1.0)
        
        if scale >= 1.0:
            return points
            
        coords = points[:, :3]
        coords_mm = np.round(coords.astype(np.float64) * 1000).astype(np.int32)
        offset = coords_mm.min(axis=0)
        coords_scaled = coords_mm - offset
        
        quantized_coords = np.round(coords_scaled * scale).astype(np.int32)
        unique_quantized = np.unique(quantized_coords, axis=0)
        
        coords_dec = unique_quantized.astype(np.float64) / scale
        coords_dec = (coords_dec + offset) / 1000.0
        
        # 保留原打标脚本的反射率补 0 逻辑
        zeros = np.zeros((coords_dec.shape[0], 1), dtype=np.float32)
        reconstructed_points = np.concatenate([coords_dec, zeros], axis=1).astype(np.float32)
        
        return reconstructed_points

    # 替换 Dataset 类的点云读取方法
    test_set.__class__.get_lidar = patched_get_lidar
    # =========================================================================

    # 构建并加载网络模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    
    # 设定 10 个量化挡位
    scales = [1/64, 1.5/128, 1/128, 1.5/256, 1/256, 1.5/512, 1/512]

    with torch.no_grad():
        for scale in scales:
            logger.info('=================================================================================')
            logger.info(f'===================== Start Evaluation for Scale: {scale} =====================')
            logger.info('=================================================================================')
            
            # 将新的 scale 值绑定给 Dataset 类，供 patched_get_lidar 获取
            test_set.__class__.current_scale = scale
            
            # 【核心】：重新构建 DataLoader
            # 这一步非常重要，因为 Dataloader 在 fork 出 workers 后状态可能会独立。
            # 每次循环重新建立 Iterator 能够确保 workers 读取到我们刚刚更新的 current_scale
            _, cur_test_loader, _ = build_dataloader(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                batch_size=args.batch_size,
                dist=dist_test, workers=args.workers, logger=logger, training=False
            )

            # 为当前量化步长创建专用的结果输出目录 (如 eval/epoch_xxxx/val/scale_0.125/)
            scale_eval_dir = eval_output_dir / f'scale_{scale}'
            scale_eval_dir.mkdir(parents=True, exist_ok=True)
            
            # 进行单轮完整验证集的测试
            eval_single_ckpt(
                model=model, 
                test_loader=cur_test_loader, 
                args=args, 
                eval_output_dir=scale_eval_dir, 
                logger=logger, 
                epoch_id=epoch_id, 
                dist_test=dist_test
            )
            
    logger.info('********************** All 10 Scales Evaluated Successfully **********************')


if __name__ == '__main__':
    main()