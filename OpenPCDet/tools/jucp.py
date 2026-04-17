import os
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils

def parse_config():
    parser = argparse.ArgumentParser(description='Auto JUCP Label Generation for Routing Network')
    parser.add_argument('--cfg_file', type=str, required=True, help='Path to model config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split_file', type=str, required=True, help='Path to val.txt or train.txt')
    parser.add_argument('--iou_thresh', type=float, default=0.7, help='IoU threshold for TP (Car defaults to 0.7)')
    parser.add_argument('--out_csv', type=str, default='jucp_labels.csv', help='Output CSV file name')
    
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def evaluate_single_dict(model, data_dict, iou_thresh):
    """提取单帧预测结果并计算 TP 和 mIoU，以此替代脆弱的 F1-Score"""
    with torch.no_grad():
        # 必须使用 model() 而非 model.forward()，以触发完整的 PyTorch forward hooks
        pred_dicts, _ = model(data_dict)
    
    pred_dict = pred_dicts[0]
    pred_boxes = pred_dict['pred_boxes']
    pred_labels = pred_dict['pred_labels']
    pred_scores = pred_dict['pred_scores']
    
    # 【关键过滤】：滤除低置信度噪点，防止原图误检过多导致基线数据失真
    mask_conf = pred_scores >= 0.3
    pred_boxes = pred_boxes[mask_conf]
    pred_labels = pred_labels[mask_conf]
    
    gt_boxes_with_classes = data_dict['gt_boxes'][0]
    mask = gt_boxes_with_classes[:, :7].sum(dim=1) != 0
    gt_boxes = gt_boxes_with_classes[mask, :7]
    gt_labels = gt_boxes_with_classes[mask, 7]

    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)

    # 如果没有 GT 也没有预测，算作满分；如果有其中一方为0，则得0分
    if num_preds == 0 and num_gts == 0:
        return num_gts, 1.0  # tp=0, miou=1.0 (完美)
    elif num_preds == 0 or num_gts == 0:
        return 0, 0.0
        
    ious = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes, gt_boxes)
    max_ious, gt_match_indices = ious.max(dim=1)
    
    tp = 0
    iou_sum = 0.0
    matched_gts = set() 
    
    for i in range(num_preds):
        iou = max_ious[i].item()
        matched_gt_idx = gt_match_indices[i].item()
        
        if iou >= iou_thresh:
            pred_cls = pred_labels[i].item()
            gt_cls = int(gt_labels[matched_gt_idx].item())
            # 类别一致且该 GT 未被其他更高分框占用
            if pred_cls == gt_cls and matched_gt_idx not in matched_gts:
                tp += 1
                iou_sum += iou
                matched_gts.add(matched_gt_idx)
                
    miou = iou_sum / tp if tp > 0 else 0.0
    return tp, miou

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    
    # 10个量化挡位 (0对应无损，9对应最低压缩精度/最大压缩率)
    scales = [1/64, 1.5/128, 1/128, 1.5/256, 1/256, 1.5/512, 1/512]
    
    with open(args.split_file, 'r') as f:
        frame_ids = [line.strip() for line in f.readlines() if line.strip()]
    logger.info(f"Loaded {len(frame_ids)} frames for JUCP labeling.")

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=1, training=False, logger=logger
    )
    
    # =========================================================================
    # 动态注入量化钩子：从源头量化，确保 OpenPCDet 会重新生成正确的体素
    # 由于你的模型是专门按反射率全0训练的，这里保留你原本补0的逻辑
    # =========================================================================
    orig_get_lidar = dataset.__class__.get_lidar
    
    def patched_get_lidar(self, idx):
        points = orig_get_lidar(self, idx)
        
        scale = getattr(self, 'current_scale', 1.0)
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
        
        zeros = np.zeros((coords_dec.shape[0], 1), dtype=np.float32)
        reconstructed_points = np.concatenate([coords_dec, zeros], axis=1).astype(np.float32)
        
        return reconstructed_points

    dataset.__class__.get_lidar = patched_get_lidar
    # =========================================================================

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    results_list = []

    for frame_id in tqdm(frame_ids, desc="Generating JUCP Labels"):
        frame_index = -1
        for i, info in enumerate(dataset.kitti_infos):
            if info['point_cloud']['lidar_idx'] == frame_id:
                frame_index = i
                break
                
        if frame_index == -1:
            continue
            
        # --- A. 计算基线(无损)表现 ---
        dataset.current_scale = scales[0] 
        
        base_data_dict = dataset.collate_batch([dataset[frame_index]]) 
        load_data_to_gpu(base_data_dict)
        
        base_tp, base_miou = evaluate_single_dict(model, base_data_dict, args.iou_thresh)
        
        # 如果模型连无损状态下都没测出物体，直接打为极难帧，赋予最高压缩率
        if base_tp == 0:
            results_list.append({
                'frame_id': frame_id, 'jucp_label': 6, 'scale': scales[6],
                'base_tp': 0, 'cur_tp': 0, 'base_miou': 0.0, 'cur_miou': 0.0, 
                'drop_reason': 'Hard Frame (Base TP=0)'
            })
            continue

        jucp_label = 0
        best_scale = scales[0]
        final_cur_tp = base_tp
        final_cur_miou = base_miou
        drop_reason = 'Failed to compress'
        
        # --- B. 从最低精度(9)向最高精度(0)搜索 ---
        for label_idx in range(6, -1, -1):
            if label_idx == 0:
                jucp_label = 0
                best_scale = scales[0]
                break
                
            cur_scale = scales[label_idx]
            
            dataset.current_scale = cur_scale
            eval_dict = dataset.collate_batch([dataset[frame_index]])
            load_data_to_gpu(eval_dict)
            
            cur_tp, cur_miou = evaluate_single_dict(model, eval_dict, args.iou_thresh)
            
            # 【核心无损条件】：正确识别的目标数(TP)不能少于基线，且回归框平均IoU下降不超过2%
            if cur_tp >= base_tp and (base_miou - cur_miou) <= 0.02:
                jucp_label = label_idx
                best_scale = cur_scale
                final_cur_tp = cur_tp
                final_cur_miou = cur_miou
                drop_reason = 'Pass'
                break 
            else:
                # 记录阻挡该档位通过的具体原因
                if cur_tp < base_tp:
                    drop_reason = f'TP Drop: {cur_tp} < {base_tp}'
                else:
                    drop_reason = f'IoU Drop: {cur_miou:.3f} < {base_miou:.3f}'

        results_list.append({
            'frame_id': frame_id,
            'jucp_label': jucp_label,
            'scale': best_scale,
            'base_tp': base_tp,
            'cur_tp': final_cur_tp,
            'base_miou': round(base_miou, 4),
            'cur_miou': round(final_cur_miou, 4),
            'drop_reason': drop_reason
        })

        if len(results_list) % 10 == 0:
            df = pd.DataFrame(results_list)
            df.to_csv(args.out_csv, index=False)

    df = pd.DataFrame(results_list)
    df.to_csv(args.out_csv, index=False)
    logger.info(f"Done! JUCP labels saved to {args.out_csv}")

if __name__ == '__main__':
    main()