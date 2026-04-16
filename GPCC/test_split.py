# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import glob
import time
import argparse
import torch
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
from mmengine.logging import print_log
from mmdet3d.apis import LidarSeg3DInferencer

# 导入 GPCC 工具
from gpcc_geo import gpcc_encode, gpcc_decode
from data_utils.geometry.inout import write_ply_o3d
from pc_error_geo import pc_error

# 定义前景标签
FG_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18]

def read_kitti_bin(filedir):
    coords = np.fromfile(filedir, dtype=np.float32).reshape(-1, 4)
    return coords[:, :3]

def write_kitti_bin(filedir, coords):
    coords.astype(np.float32).tofile(filedir)

def compress_partition(coords, pos_quant_scale, prefix, out_dir, cfg_path):
    """针对分区进行压缩和解压"""
    if len(coords) == 0:
        return 0, 0.0, 0.0, np.empty((0, 3))
    
    tmp_in_ply = os.path.join(out_dir, f"{prefix}_in.ply")
    tmp_bin = os.path.join(out_dir, f"{prefix}_stream.bin")
    tmp_out_ply = os.path.join(out_dir, f"{prefix}_out.ply")
    
    write_ply_o3d(tmp_in_ply, coords, normal=True, knn=16)
    
    # 2. 编码
    t0 = time.time()
    log_enc = gpcc_encode(tmp_in_ply, tmp_bin, posQuantscale=pos_quant_scale, cfgdir=cfg_path)
    enc_time = log_enc.get('Processing time (wall)', time.time() - t0)
    bits = os.path.getsize(tmp_bin) * 8 if os.path.exists(tmp_bin) else 0
    
    # 3. 解码
    t1 = time.time()
    log_dec = gpcc_decode(tmp_bin, tmp_out_ply)
    dec_time = log_dec.get('Processing time (wall)', time.time() - t1)
    
    # 4. 还原
    pcd_dec = o3d.io.read_point_cloud(tmp_out_ply)
    coords_restored = np.asarray(pcd_dec.points)
    
    # 清理
    for f in [tmp_in_ply, tmp_bin, tmp_out_ply]:
        if os.path.exists(f): os.remove(f)
            
    return bits, enc_time, dec_time, coords_restored

class SementicCompressor:
    def __init__(self, args):
        self.args = args
        self.inferencer = LidarSeg3DInferencer(model=args.model, weights=args.weights, device=args.device)
        _ = self.inferencer(inputs=dict(points=args.file_list[0]), no_save_vis=True, no_save_pred=True)
        self.gpcc_cfg = os.path.abspath(args.gpcc_cfg)

    def get_quant_params(self, r):
        quant_map = [
            (1/2, 1/64),   # 0
            (1/2, 1/128),  # 1
            (1/2, 1/256),  # 2
            (1/2, 1/32),   # 3
            (1/2, 1/16),  # 4
            (1/2, 1/8),  # 5
            (1, 1),   # 6
            (1/8, 1/128),  # 7
            (1/8, 1/256),  # 8
        ]
        if 0 <= r <= 8:
            return quant_map[r]
        return 1/2, 1/2

    def process_single(self, pcd_path):
        filename = os.path.basename(pcd_path).replace('.bin', '')
        results_list = []

        print(f"\n>> Segmenting {filename}...")
        torch.cuda.synchronize()
        t_s = time.time()
        result = self.inferencer(inputs=dict(points=pcd_path), no_save_vis=True, no_save_pred=True)
        seg_labels = result['predictions'][0].get('pred_sem_seg', result['predictions'][0])['pts_semantic_mask']
        if hasattr(seg_labels, 'cpu'): seg_labels = seg_labels.cpu().numpy()
        torch.cuda.synchronize()
        seg_time = time.time() - t_s

        coords_raw = read_kitti_bin(pcd_path)
        # 准备 pc_error 参考文件 (毫米级)
        tmp_ref_ply = os.path.join(self.args.output, f"{filename}_ref.ply")
        ref_coords_scaled = np.round(coords_raw.astype(np.float64) * 1000).astype(np.int32)
        ref_offset = ref_coords_scaled.min(axis=0)
        ref_coords = ref_coords_scaled - ref_offset
        write_ply_o3d(tmp_ref_ply, ref_coords, normal=True, knn=16)

        # 分离点云
        fg_mask = np.isin(seg_labels[:len(ref_coords)], FG_CLASSES)
        fg_pts = ref_coords[fg_mask]
        bg_pts = ref_coords[~fg_mask]

        # 2. 遍历 9 种精度
        for r in [6]:
            fg_q, bg_q = self.get_quant_params(r)
            print(f"   Rate r={r}: fg_q={fg_q}, bg_q={bg_q}")

            # 压缩前景
            f_bits, f_et, f_dt, f_res = compress_partition(fg_pts, fg_q, f"{filename}_r{r}_fg", self.args.output, self.gpcc_cfg)
            # 压缩背景
            b_bits, b_et, b_dt, b_res = compress_partition(bg_pts, bg_q, f"{filename}_r{r}_bg", self.args.output, self.gpcc_cfg)

            # 合并重建
            merged = np.vstack((f_res, b_res)) if len(f_res)>0 and len(b_res)>0 else (f_res if len(f_res)>0 else b_res)            
            tmp_dec_ply = os.path.join(self.args.output, f"{filename}_r{r}_dec.ply")
            write_ply_o3d(tmp_dec_ply, merged)            

            # 计算误差 (PC Error)
            coords_dec = merged + ref_offset
            coords_restored = (coords_dec.astype(np.float64) / 1000.0)
            out_bin = os.path.join(self.args.output, f"{filename}_r{r}.bin")
            write_kitti_bin(out_bin, coords_restored)
            
            err_res = pc_error(tmp_ref_ply, tmp_dec_ply, resolution=self.args.resolution, normal=True, show=False)
            if os.path.exists(tmp_dec_ply): os.remove(tmp_dec_ply)

            # 统计
            total_bits = f_bits + b_bits
            bpp = total_bits / len(coords_raw)
            fg_bpp = f_bits / len(fg_pts) if len(fg_pts) > 0 else 0
            bg_bpp = b_bits / len(bg_pts) if len(bg_pts) > 0 else 0
            
            results_list.append({
                'filename': filename, 'r': r, 'fg_quant': fg_q, 'bg_quant': bg_q,
                'fg_bpp':fg_bpp, 'bg_bpp':bg_bpp, 
                'bpp': bpp, 'seg_time': seg_time, 
                'enc_time': f_et + b_et, 'dec_time': f_dt + b_dt,
                **err_res
            })

        if os.path.exists(tmp_ref_ply): os.remove(tmp_ref_ply)
        
        df = pd.DataFrame(results_list)
        df.to_csv(os.path.join(self.args.results, f"{filename}.csv"), index=False)
        return df

def main():
    parser = argparse.ArgumentParser(description="Batch Semantic GPCC Compression")
    parser.add_argument('--testdata', type=str, required=True, help='Input folder with .bin files')
    parser.add_argument('--model', required=True, help='MMSeg3D config')
    parser.add_argument('--weights', required=True, help='MMSeg3D checkpoint')
    parser.add_argument('--gpcc-cfg', default='kitti.cfg')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resolution', type=int, default=80000)
    parser.add_argument('--output', default='./output')
    parser.add_argument('--results', default='./results')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(args.testdata, '**', '*.bin'), recursive=True))
    print(f"Found {len(file_list)} files.")

    args.file_list = file_list
    compressor = SementicCompressor(args)
    all_dfs = []
    
    for fpath in tqdm(file_list, desc="Total Progress"):
        df = compressor.process_single(fpath)
        all_dfs.append(df)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.to_csv(os.path.join(args.results, 'all_details.csv'), index=False)
        # 按 r 计算平均值
        avg_df = full_df.groupby('r').mean(numeric_only=True).reset_index()
        avg_df.to_csv(os.path.join(args.results, 'average_results.csv'), index=False)
        print("\nProcessing Complete. Average Results Saved.")

if __name__ == '__main__':
    main()