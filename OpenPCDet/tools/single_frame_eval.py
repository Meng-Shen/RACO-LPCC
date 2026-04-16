import argparse
import torch
import numpy as np
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils

def parse_config():
    parser = argparse.ArgumentParser(description='Single frame evaluation script for OpenPCDet')
    parser.add_argument('--cfg_file', type=str, required=True, help='Path to model config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--frame_id', type=str, required=True, help='KITTI frame ID, e.g., 000008')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='3D IoU threshold for True Positive')
    
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('----------------- Single Frame Evaluation -----------------')

    # 1. 初始化验证集 Dataset（不加载完整的 Dataloader，只用 Dataset 类）
    val_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=1, training=False, logger=logger
    )

    # 2. 在 kitti_infos 中寻找指定的 frame_id
    frame_index = -1
    for i, info in enumerate(val_set.kitti_infos):
        if info['point_cloud']['lidar_idx'] == args.frame_id:
            frame_index = i
            break
            
    if frame_index == -1:
        logger.error(f"Frame ID {args.frame_id} not found in the validation set infos!")
        return

    # 3. 获取该帧的 data_dict 并添加 batch 维度
    data_dict = val_set[frame_index]
    data_dict = val_set.collate_batch([data_dict])
    load_data_to_gpu(data_dict)

    # 4. 构建模型并加载权重
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=val_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    # 5. 模型推理
    logger.info(f"Running inference on frame {args.frame_id}...")
    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)
    
    pred_dict = pred_dicts[0]
    pred_boxes = pred_dict['pred_boxes']    # (N, 7) [x, y, z, dx, dy, dz, heading]
    pred_scores = pred_dict['pred_scores']  # (N,)
    pred_labels = pred_dict['pred_labels']  # (N,)

    # 6. 获取真实标签 (GT Boxes)
    gt_boxes_with_classes = data_dict['gt_boxes'][0] # (M, 8) 最后一列是 class label
    # 过滤掉全为0的 padding GT boxes
    mask = gt_boxes_with_classes[:, :7].sum(dim=1) != 0
    gt_boxes = gt_boxes_with_classes[mask, :7]
    gt_labels = gt_boxes_with_classes[mask, 7]

    logger.info(f"Found {len(gt_boxes)} GT objects and predicted {len(pred_boxes)} objects.")

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        logger.info("No predictions or no GT boxes. Skipping IoU calculation.")
        return

    # 7. 计算 3D IoU (调用 CUDA 算子)
    # ious shape: (num_preds, num_gts)
    ious = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes, gt_boxes)
    
    # 8. 匹配与评估逻辑 (基于匈牙利匹配或简单的贪心匹配，这里展示最高IoU贪心匹配)
    max_ious, gt_match_indices = ious.max(dim=1)
    
    tp_count = 0
    fp_count = 0
    
    logger.info(f"--- Prediction Analysis (IoU Threshold: {args.iou_thresh}) ---")
    for i in range(len(pred_boxes)):
        iou = max_ious[i].item()
        matched_gt_idx = gt_match_indices[i].item()
        pred_cls = cfg.CLASS_NAMES[pred_labels[i].item() - 1]
        score = pred_scores[i].item()
        
        if iou >= args.iou_thresh:
            gt_cls = cfg.CLASS_NAMES[int(gt_labels[matched_gt_idx].item()) - 1]
            if pred_cls == gt_cls:
                logger.info(f"[TP] Pred {i} ({pred_cls}, Score: {score:.3f}) matched GT {matched_gt_idx} ({gt_cls}) with IoU: {iou:.4f}")
                tp_count += 1
            else:
                logger.info(f"[FP - Class Error] Pred {i} ({pred_cls}) matched GT {matched_gt_idx} ({gt_cls}) with IoU: {iou:.4f}")
                fp_count += 1
        else:
            logger.info(f"[FP - Low IoU/Background] Pred {i} ({pred_cls}, Score: {score:.3f}). Max GT IoU: {iou:.4f}")
            fp_count += 1

    fn_count = len(gt_boxes) - tp_count

    logger.info("----------------- Summary -----------------")
    logger.info(f"True Positives (TP): {tp_count}")
    logger.info(f"False Positives (FP): {fp_count}")
    logger.info(f"False Negatives (Missed GTs) (FN): {fn_count}")
    
    if (tp_count + fp_count) > 0:
        precision = tp_count / (tp_count + fp_count)
        logger.info(f"Precision: {precision:.4f}")
    if (tp_count + fn_count) > 0:
        recall = tp_count / (tp_count + fn_count)
        logger.info(f"Recall:    {recall:.4f}")

if __name__ == '__main__':
    main()