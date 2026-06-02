# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import time
import argparse
import torch
import numpy as np
import open3d as o3d
from mmengine.logging import print_log
from mmdet3d.apis import LidarSeg3DInferencer

# 导入 GPCC 工具 (请确保这些工具在您的 PYTHONPATH 或当前目录下)
from gpcc_geo import gpcc_encode, gpcc_decode
from data_utils.geometry.inout import write_ply_o3d

# 定义前景标签
FG_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18]

def read_kitti_bin(filedir):
    """读取 KITTI 格式的 bin 文件，返回 (N, 3) 坐标"""
    coords = np.fromfile(filedir, dtype=np.float32).reshape(-1, 4)
    return coords[:, :3]

def write_kitti_bin(filedir, coords):
    """保存 KITTI 格式的 bin 文件"""
    coords.astype(np.float32).tofile(filedir)

def compress_partition(coords, pos_quant_scale, prefix, out_dir, cfg_path):
    """
    针对单独的点云划分(前景或背景)进行 GPCC 压缩和解压缩
    返回: bits (比特数), enc_time (编码时间), dec_time (解码时间), coords_restored (重建坐标)
    """
    if len(coords) == 0:
        return 0, 0.0, 0.0, np.empty((0, 3))
    
    # 1. 坐标偏移与放大处理 (转为毫米级整数)
    coords_scaled = np.round(coords.astype(np.float64) * 1000).astype(np.int32) 
    offset = coords_scaled.min(axis=0)
    coords_scaled = coords_scaled - offset
    
    # 2. 保存中间 PLY 文件
    tmp_in_ply = os.path.join(out_dir, f"{prefix}_in.ply")
    tmp_bin = os.path.join(out_dir, f"{prefix}_stream.bin")
    tmp_out_ply = os.path.join(out_dir, f"{prefix}_out.ply")
    
    write_ply_o3d(tmp_in_ply, coords_scaled, normal=True, knn=16)
    
    # 3. 编码 (Encode)
    t0 = time.time()
    log_enc = gpcc_encode(tmp_in_ply, tmp_bin, posQuantscale=pos_quant_scale, cfgdir=cfg_path)
    enc_time = log_enc.get('Processing time (wall)', time.time() - t0)
    
    # 获取压缩后的大小
    file_size_bits = os.path.getsize(tmp_bin) * 8 if os.path.exists(tmp_bin) else 0
    
    # 4. 解码 (Decode)
    t1 = time.time()
    log_dec = gpcc_decode(tmp_bin, tmp_out_ply)
    dec_time = log_dec.get('Processing time (wall)', time.time() - t1)
    
    # 5. 还原坐标尺度与偏移
    pcd_dec = o3d.io.read_point_cloud(tmp_out_ply)
    coords_dec = np.asarray(pcd_dec.points) + offset
    coords_restored = (coords_dec.astype(np.float64) / 1000.0)
    
    # 清理中间文件
    for tmp_f in [tmp_in_ply, tmp_bin, tmp_out_ply]:
        if os.path.exists(tmp_f):
            os.remove(tmp_f)
            
    return file_size_bits, enc_time, dec_time, coords_restored

def parse_args():
    parser = argparse.ArgumentParser(description="Segment, Split, Compress and Merge Point Cloud")
    # 分割模型相关参数
    parser.add_argument('pcd', help='Input Point cloud file (.bin)')
    parser.add_argument('model', help='Config file for LidarSeg3D')
    parser.add_argument('weights', help='Checkpoint file for LidarSeg3D')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    
    # 压缩相关参数
    parser.add_argument('--fg-quant', type=float, default=1/64, help='posQuantscale for Foreground')
    parser.add_argument('--bg-quant', type=float, default=1/8, help='posQuantscale for Background')
    parser.add_argument('--gpcc-cfg', type=str, default='kitti.cfg', help='Path to GPCC cfg file')
    
    # 输出相关参数
    parser.add_argument('--out-dir', type=str, default='output', help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    filename = os.path.split(args.pcd)[-1].split('.')[0]

    # ==========================================
    # 1. 语义分割推理
    # ==========================================
    print_log(f'Initializing model and predicting semantics for {args.pcd}...', logger='current')
    
    inferencer = LidarSeg3DInferencer(model=args.model, weights=args.weights, device=args.device)
    _ = inferencer(inputs=dict(points=args.pcd), no_save_vis=True, no_save_pred=True)
    torch.cuda.synchronize()
    t_seg_start = time.time()
    result = inferencer(inputs=dict(points=args.pcd), no_save_vis=True, no_save_pred=True)
    
    # 提取预测的语义掩码 (适配 mmdet3d 的输出结构)
    predictions = result['predictions'][0]
    if 'pred_sem_seg' in predictions:
        seg_labels = predictions['pred_sem_seg']['pts_semantic_mask']
    else:
        seg_labels = predictions['pts_semantic_mask']
        
    if hasattr(seg_labels, 'cpu'):
        seg_labels = seg_labels.cpu().numpy()
    seg_labels = np.asarray(seg_labels)
    torch.cuda.synchronize()
    t_seg_end = time.time()
    seg_time = t_seg_end - t_seg_start

    # ==========================================
    # 2. 读取原始点云并分离前景和背景
    # ==========================================
    coords_raw = read_kitti_bin(args.pcd)
    num_points_raw = len(coords_raw)
    
    if len(seg_labels) != num_points_raw:
        print_log(f'Warning: Points count ({num_points_raw}) != labels count ({len(seg_labels)})', logger='current')
        # 防止使用带有额外特征维度的长度，截断至相同长度
        min_len = min(num_points_raw, len(seg_labels))
        coords_raw = coords_raw[:min_len]
        seg_labels = seg_labels[:min_len]
        num_points_raw = min_len

    fg_mask = np.isin(seg_labels, FG_CLASSES)
    bg_mask = ~fg_mask
    
    fg_points = coords_raw[fg_mask]
    bg_points = coords_raw[bg_mask]
    
    print_log(f'Total points: {num_points_raw} | FG points: {len(fg_points)} | BG points: {len(bg_points)}', logger='current')

    # ==========================================
    # 3. 独立压缩前景和背景
    # ==========================================
    gpcc_cfg_path = os.path.abspath(args.gpcc_cfg)
    
    print_log(f'Compressing Foreground (posQuantscale={args.fg_quant})...', logger='current')
    fg_bits, fg_enc_t, fg_dec_t, fg_restored = compress_partition(
        fg_points, args.fg_quant, f"{filename}_fg", args.out_dir, gpcc_cfg_path
    )
    
    print_log(f'Compressing Background (posQuantscale={args.bg_quant})...', logger='current')
    bg_bits, bg_enc_t, bg_dec_t, bg_restored = compress_partition(
        bg_points, args.bg_quant, f"{filename}_bg", args.out_dir, gpcc_cfg_path
    )

    # ==========================================
    # 4. 合并与统计
    # ==========================================
    # 合并重建点云
    merged_restored_coords = np.vstack((fg_restored, bg_restored)) if len(fg_restored) > 0 and len(bg_restored) > 0 \
                             else (fg_restored if len(fg_restored) > 0 else bg_restored)
                             
    # 保存最终合并后的点云
    out_bin_file = os.path.join(args.out_dir, f"{filename}_reconstructed.bin")
    write_kitti_bin(out_bin_file, merged_restored_coords)
    
    # 统计指标
    total_bits = fg_bits + bg_bits
    total_bpp = total_bits / num_points_raw if num_points_raw > 0 else 0
    fg_bpp = fg_bits / len(fg_points) if len(fg_points) > 0 else 0
    bg_bpp = bg_bits / len(bg_points) if len(bg_points) > 0 else 0
    
    total_enc_time = fg_enc_t + bg_enc_t
    total_dec_time = fg_dec_t + bg_dec_t
    
    print("-" * 50)
    print(" Compression & Reconstruction Results ")
    print("-" * 50)
    print(f"Segmentation Time : {seg_time:.3f} s")
    print(f"FG Encoding Time  : {fg_enc_t:.3f} s  | FG Decoding Time: {fg_dec_t:.3f} s")
    print(f"BG Encoding Time  : {bg_enc_t:.3f} s  | BG Decoding Time: {bg_dec_t:.3f} s")
    print(f"Total GPCC Enc/Dec: {total_enc_time:.3f} s / {total_dec_time:.3f} s")
    print(f"Total Pipeline T  : {(seg_time + total_enc_time + total_dec_time):.3f} s")
    print("-" * 50)
    print(f"Foreground bpp    : {fg_bpp:.4f} bits/point (posQuantscale={args.fg_quant})")
    print(f"Background bpp    : {bg_bpp:.4f} bits/point (posQuantscale={args.bg_quant})")
    print(f"Overall bpp       : {total_bpp:.4f} bits/point")
    print(f"Reconstructed bin : {out_bin_file}")
    print("-" * 50)

if __name__ == '__main__':
    main()