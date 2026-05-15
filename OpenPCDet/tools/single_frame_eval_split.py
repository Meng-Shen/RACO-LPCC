import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# 设定前景类别
FG_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18]

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = Path(root_path)
        self.ext = ext
        data_file_list = glob.glob(str(self.root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        pass # 我们在 main 中直接处理

def parse_config():
    parser = argparse.ArgumentParser(description='Single Frame Split Quantization & Evaluation')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--data_path', type=str, default=None, help='specify the point cloud data file or directory')
    
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory of pre-computed .npy masks')
    parser.add_argument('--save_dir', type=str, default='.', help='directory to save quantized point clouds')
    parser.add_argument('--frame_id', type=str, required=True, help='KITTI frame ID, e.g., 000008')
    parser.add_argument('--save_path', type=str, default=None, help='specify the path to save the prediction pkl file')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def split_and_quantize_with_mask(points, seg_labels, scale_fg, scale_bg):
    """
    基于给定的 seg_labels (mask) 分离前景和背景，并执行独特的量化与去重。
    """
    coords_raw = points[:, :3]
    
    # 按照提供的类别划分前景和背景
    fg_mask = np.isin(seg_labels, FG_CLASSES)
    bg_mask = ~fg_mask
    
    # 转为毫米级坐标并减去偏移量
    coords_mm = np.round(coords_raw.astype(np.float64) * 1000).astype(np.int32)
    offset = coords_mm.min(axis=0)
    coords_scaled = coords_mm - offset
    
    def quantize_subset(mask, scale):
        c_sub = coords_scaled[mask]
        if len(c_sub) == 0:
            return np.empty((0, 3), dtype=np.float64)
        if scale >= 1.0:
            return c_sub.astype(np.float64)
        # 乘以 scale 后取整，再通过 unique 剔除合并的点
        q = np.round(c_sub * scale).astype(np.int32)
        u = np.unique(q, axis=0)
        return u.astype(np.float64) / scale

    dec_fg = quantize_subset(fg_mask, scale_fg)
    dec_bg = quantize_subset(bg_mask, scale_bg)
    
    # 拼合前后景
    if len(dec_fg) > 0 and len(dec_bg) > 0:
        merged_dec = np.concatenate([dec_fg, dec_bg], axis=0)
    else:
        merged_dec = dec_fg if len(dec_fg) > 0 else dec_bg
        
    # 恢复物理坐标
    merged_dec = (merged_dec + offset) / 1000.0
    dec_fg_final = (dec_fg + offset) / 1000.0 if len(dec_fg) > 0 else np.empty((0,3))
    dec_bg_final = (dec_bg + offset) / 1000.0 if len(dec_bg) > 0 else np.empty((0,3))
    
    return merged_dec, dec_fg_final, dec_bg_final

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Single Frame Seg-based Split Quantization-------------------------')
    
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_path is None:
        args.save_path = args.frame_id + '.pkl'
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=args.data_path, ext=args.ext, logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    all_predictions = []  
    
    # 你指定的量化步长
    scale_fg = 1.5/256
    scale_bg = 1/512

    with torch.no_grad():
        for idx in range(len(dataset.sample_file_list)):
            file_path = dataset.sample_file_list[idx]
            frame_id = Path(file_path).stem
            
            # --- 1. 读取原始点云 ---
            if args.ext == '.bin':
                full_raw_points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            elif args.ext == '.npy':
                full_raw_points = np.load(file_path)
                
            # --- 2. 读取事先生成的 mask 文件 ---
            mask_path = Path(args.mask_dir) / f"{frame_id}.npy"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {frame_id}. Please run generate_masks.py first!")
            
            seg_labels = np.load(mask_path)[:len(full_raw_points)]

            # --- 3. 基于 mask 分离并量化点云 ---
            logger.info(f"Quantizing {frame_id} with FG Scale: {scale_fg:.6f}, BG Scale: {scale_bg:.6f}")
            quantized_full, points_fg, points_bg = split_and_quantize_with_mask(
                full_raw_points, seg_labels, scale_fg, scale_bg
            )
            
            # --- 4. 保存量化后的纯 3D 坐标用于 Visualize ---
            save_path_full = os.path.join(args.save_dir, f'{frame_id}_quantized.npy')
            save_path_fg = os.path.join(args.save_dir, f'{frame_id}_quantized_fg.npy')
            save_path_bg = os.path.join(args.save_dir, f'{frame_id}_quantized_bg.npy')

            np.save(save_path_full, quantized_full)
            np.save(save_path_fg, points_fg)
            np.save(save_path_bg, points_bg)
            
            # --- 5. 补齐强度(Intensity=0)，送入检测模型做推理 ---
            zeros = np.zeros((quantized_full.shape[0], 1), dtype=np.float32)
            input_points = np.concatenate([quantized_full, zeros], axis=1).astype(np.float32)
            
            input_dict_quantized = {
                'points': input_points,
                'frame_id': idx,
            }
            
            data_dict_quantized = dataset.prepare_data(data_dict=input_dict_quantized)
            batch_dict_quantized = dataset.collate_batch([data_dict_quantized])
            load_data_to_gpu(batch_dict_quantized)
            
            final_pred_dicts, _ = model.forward(batch_dict_quantized)
            
            # 收集预测框并转为 NumPy
            final_pred_boxes = final_pred_dicts[0]['pred_boxes'].cpu().numpy()
            final_pred_scores = final_pred_dicts[0]['pred_scores'].cpu().numpy()
            final_pred_labels = final_pred_dicts[0]['pred_labels'].cpu().numpy()

            all_predictions.append({
                'frame_id': frame_id,
                'pred_boxes': final_pred_boxes,
                'pred_scores': final_pred_scores,
                'pred_labels': final_pred_labels
            })
            
            logger.info(f"Frame {frame_id} processing complete. Saved to {save_path_full}")
            logger.info(f"Points After Quantization: FG={len(points_fg)}, BG={len(points_bg)}, Total={len(quantized_full)}")

    if args.save_path is not None:
        with open(args.save_path, 'wb') as f:
            pickle.dump(all_predictions, f)
        logger.info(f"Prediction results successfully saved to: {args.save_path}")

    logger.info('-----------------Evaluation Finished-------------------------')

if __name__ == '__main__':
    main()