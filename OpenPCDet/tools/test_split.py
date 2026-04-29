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

# 设定前景类别 (根据你的需求)
FG_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18]

def parse_config():
    parser = argparse.ArgumentParser(description='Split Quantization mAP Evaluation')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')
    
    # 指向刚才第一步生成的 mask 目录
    parser.add_argument('--mask_dir', type=str, default='../output/eval/seg_masks', help='Directory of pre-computed .npy masks')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    np.random.seed(1024)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test, pre_trained_path=args.pretrained_model)
    model.cuda()
    eval_utils.eval_one_epoch(cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test, result_dir=eval_output_dir)


def main():
    args, cfg = parse_config()

    if args.infer_time: os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        if args.local_rank is None: args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(args.tcp_port, args.local_rank, backend='nccl')
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

    if args.eval_tag is not None: eval_output_dir = eval_output_dir / args.eval_tag
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = eval_output_dir / ('log_eval_split_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start logging**********************')
    
    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    orig_get_lidar = test_set.__class__.get_lidar
    np.seterr(divide='ignore', invalid='ignore')
    mask_dir_path = Path(args.mask_dir)

    def patched_get_lidar(self, idx):
        points = orig_get_lidar(self, idx)
        scale_fg = getattr(self.__class__, 'current_scale_fg', 1.0)
        scale_bg = getattr(self.__class__, 'current_scale_bg', 1.0)
        
        if scale_fg >= 1.0 and scale_bg >= 1.0:
            return points
            
        coords_raw = points[:, :3]
        
        # 直接读取脱机跑好的 NPY 掩码，不用推理！
        # 兼容两种 Dataset 中 idx 的类型 (有的是数字，有的是字符串)
        try:
            info = self.kitti_infos[idx]
            lidar_idx = info['point_cloud']['lidar_idx']
        except TypeError:
            lidar_idx = idx

        mask_path = mask_dir_path / f"{lidar_idx}.npy"
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {lidar_idx}. Please run generate_masks.py first!")
            
        seg_labels = np.load(mask_path)[:len(coords_raw)]
        fg_mask = np.isin(seg_labels, FG_CLASSES)
        bg_mask = ~fg_mask
        
        coords_mm = np.round(coords_raw.astype(np.float64) * 1000).astype(np.int32)
        offset = coords_mm.min(axis=0)
        coords_scaled = coords_mm - offset
        
        def quantize_subset(mask, scale):
            c_sub = coords_scaled[mask]
            if len(c_sub) == 0:
                return np.empty((0, 3), dtype=np.float64)
            if scale >= 1.0:
                return c_sub.astype(np.float64)
            q = np.round(c_sub * scale).astype(np.int32)
            u = np.unique(q, axis=0)
            return u.astype(np.float64) / scale

        # 核心：前后景分离量化
        dec_fg = quantize_subset(fg_mask, scale_fg)
        dec_bg = quantize_subset(bg_mask, scale_bg)
        
        if len(dec_fg) > 0 and len(dec_bg) > 0:
            merged_dec = np.concatenate([dec_fg, dec_bg], axis=0)
        else:
            merged_dec = dec_fg if len(dec_fg) > 0 else dec_bg
            
        merged_dec = (merged_dec + offset) / 1000.0
        
        # 补回 0 强度供检测模型使用
        zeros = np.zeros((merged_dec.shape[0], 1), dtype=np.float32)
        reconstructed_points = np.concatenate([merged_dec, zeros], axis=1).astype(np.float32)
        
        return reconstructed_points

    # 挂载补丁
    test_set.__class__.get_lidar = patched_get_lidar

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    
    quant_map = [
        (1.5/256, 1/512), (2/256, 1/512),(3/256, 1/512),(4/256, 1/512), 
        (1/64, 1.25/512), (1/64, 1.5/512)
    ]

    with torch.no_grad():
        for combo_idx, (scale_fg, scale_bg) in enumerate(quant_map):
            logger.info('=================================================================================')
            logger.info(f'========== Start Evaluation | Combo {combo_idx} | FG Scale: {scale_fg:.6f} | BG Scale: {scale_bg:.6f} ==========')
            logger.info('=================================================================================')
            
            test_set.__class__.current_scale_fg = scale_fg
            test_set.__class__.current_scale_bg = scale_bg
            
            _, cur_test_loader, _ = build_dataloader(
                dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=args.batch_size,
                dist=dist_test, workers=args.workers, logger=logger, training=False
            )

            scale_eval_dir = eval_output_dir / f'combo_{combo_idx}_fg_{scale_fg:.6f}_bg_{scale_bg:.6f}'
            scale_eval_dir.mkdir(parents=True, exist_ok=True)

            os.environ['CURRENT_EVAL_DIR'] = str(scale_eval_dir)
            
            eval_single_ckpt(
                model=model, test_loader=cur_test_loader, args=args, 
                eval_output_dir=scale_eval_dir, logger=logger, epoch_id=epoch_id, dist_test=dist_test
            )
            
    logger.info('********************** All 12 Split Combinations Evaluated Successfully **********************')

if __name__ == '__main__':
    main()