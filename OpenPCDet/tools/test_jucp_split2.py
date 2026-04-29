import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

# 与生成脚本保持一致的前景类别
FG_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18]

def parse_config():
    parser = argparse.ArgumentParser(description='Auto JUCP Split Compression Eval')
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

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    # JUCP Split 特有参数
    parser.add_argument('--jucp_csv', type=str, required=True, help='Path to the JUCP split labels CSV file')
    parser.add_argument('--mask_dir', type=str, required=True, help='Path to the pre-computed semantic segmentation masks')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test, 
                                pre_trained_path=args.pretrained_model)
    model.cuda()
    
    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, args, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir
        )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


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

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    # 初次构建 Dataloader（加载完整的 test split）
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    # =========================================================================
    # 动态注入 JUCP Split 双量化逻辑 & 数据集子集截断 (Subset Filtering)
    # =========================================================================
    if not os.path.exists(args.jucp_csv):
        raise FileNotFoundError(f"JUCP CSV not found at {args.jucp_csv}")
    if not os.path.exists(args.mask_dir):
        raise FileNotFoundError(f"Mask directory not found at {args.mask_dir}")
    
    logger.info(f"Loading JUCP split labels from {args.jucp_csv}")
    
    # 1. 解析 CSV，提取必须评估的 valid_frame_ids
    jucp_df = pd.read_csv(args.jucp_csv, dtype={'frame_id': str})
    jucp_map = {
        str(row['frame_id']).zfill(6): (row['scale_fg'], row['scale_bg']) 
        for _, row in jucp_df.iterrows()
    }
    valid_frame_ids = set(jucp_map.keys())

    # 2. 截断数据集 (只保留 CSV 里有的帧)
    info_list_name = None
    for attr in ['kitti_infos', 'waymo_infos', 'infos']:
        if hasattr(test_set, attr):
            info_list_name = attr
            break

    if info_list_name:
        original_infos = getattr(test_set, info_list_name)
        filtered_infos = []
        for info in original_infos:
            frame_id = None
            if 'point_cloud' in info and 'lidar_idx' in info['point_cloud']:
                frame_id = str(info['point_cloud']['lidar_idx']).zfill(6)
            elif 'frame_id' in info:
                frame_id = str(info['frame_id']).zfill(6)
            
            if frame_id in valid_frame_ids:
                filtered_infos.append(info)
        
        # 覆盖原有 infos
        setattr(test_set, info_list_name, filtered_infos)
        logger.info(f"Filtered dataset based on CSV: {len(original_infos)} -> {len(filtered_infos)} frames.")
        
        # 因为 dataset 的长度变了，必须重新实例化 DataLoader
        import torch.utils.data
        sampler = torch.utils.data.distributed.DistributedSampler(test_set) if dist_test else None
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, pin_memory=True, num_workers=args.workers,
            shuffle=False, collate_fn=test_set.collate_batch,
            drop_last=False, sampler=sampler, timeout=0
        )
    else:
        logger.warning("Could not find the dataset infos list to filter. Proceeding with full dataset.")

    # 3. 替换 DataLoader 的 Lidar 读取逻辑 (Monkey Patch)
    test_set.__class__.jucp_map = jucp_map
    test_set.__class__.mask_dir = Path(args.mask_dir)
    orig_get_lidar = test_set.__class__.get_lidar
    
    def patched_get_lidar(self, idx):
        points = orig_get_lidar(self, idx)
        
        str_idx = str(idx).zfill(6)
        scale_fg, scale_bg = self.jucp_map.get(str_idx, (1.0, 1.0))

        if getattr(self, '_print_once', True):
            print(f"\n[DEBUG] Success Hook! Frame {str_idx} is using JUCP Split scales: FG={scale_fg:.5f}, BG={scale_bg:.5f}")
            self._print_once = False
        
        if scale_fg >= 1.0 and scale_bg >= 1.0:
            return points
            
        mask_path = self.mask_dir / f"{idx}.npy"
        if not mask_path.exists():
            print(f"[WARNING] Mask not found for {idx}. Using original points.")
            return points
            
        seg_labels = np.load(mask_path)[:len(points)]
        fg_mask = np.isin(seg_labels, FG_CLASSES)
        
        coords = points[:, :3]
        coords_mm = np.round(coords.astype(np.float64) * 1000).astype(np.int32)
        offset = coords_mm.min(axis=0)
        coords_scaled = coords_mm - offset
        
        def quantize_subset(mask, scale):
            c_sub = coords_scaled[mask]
            i_sub = points[mask, 3:4]
            if len(c_sub) == 0:
                return np.empty((0, 4), dtype=np.float32)
                
            if scale >= 1.0:
                c_dec = c_sub.astype(np.float64)
                return np.concatenate([(c_dec + offset)/1000.0, i_sub], axis=1).astype(np.float32)
                
            q = np.round(c_sub * scale).astype(np.int32)
            u, indices = np.unique(q, axis=0, return_index=True)
            c_dec = u.astype(np.float64) / scale
            i_dec = i_sub[indices] 
            
            return np.concatenate([(c_dec + offset)/1000.0, i_dec], axis=1).astype(np.float32)

        pts_fg = quantize_subset(fg_mask, scale_fg)
        pts_bg = quantize_subset(~fg_mask, scale_bg)
        points_recon = np.concatenate([pts_fg, pts_bg], axis=0)
        
        return points_recon

    test_set.__class__.get_lidar = patched_get_lidar
    logger.info(f"Successfully injected JUCP Split dynamic dual-quantization.")
    # =========================================================================

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    with torch.no_grad():
        if args.eval_all:
            repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:
            eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)

if __name__ == '__main__':
    main()