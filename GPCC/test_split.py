import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager

# 获取当前目录 (GPCC) 和项目根目录，并将根目录加入环境变量
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 导入 mmdet3d 语义分割推理器
from mmdet3d.apis import LidarSeg3DInferencer

from data_utils.geometry.inout import write_ply_o3d
from extention.gpcc_geo import gpcc_encode

FG_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18]

@contextmanager
def suppress_stderr():
    """系统底层 stderr 黑洞，用于屏蔽 TMC13 C++ 核心库打印的 Warning"""
    fd = sys.stderr.fileno()
    saved_fd = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    try:
        yield
    finally:
        os.dup2(saved_fd, fd)
        os.close(devnull)
        os.close(saved_fd)

def read_kitti_bin(filedir):
    """读取 KITTI 格式的 bin 文件，返回 (N, 3) 坐标"""
    coords = np.fromfile(filedir, dtype=np.float32).reshape(-1, 4)
    return coords[:, :3]

class Tester():
    def __init__(self, args):
        self.args = args
        
        # ==========================================
        # 全局仅加载一次语义分割模型，常驻显卡
        # ==========================================
        print(f"Loading Semantic Segmentation model from {args.ckpt}...")
        self.inferencer = LidarSeg3DInferencer(model=args.cfg_file, weights=args.ckpt, device='cuda:0')
        print("Semantic model loaded successfully!")

    def test_bitrates(self, filedir, idx_file):
        filename = os.path.split(filedir)[-1].split('.')[0]
        
        # 用户指定的 6 种 (前景量化步长, 背景量化步长) 组合
        scale_combinations = [
            (1.5/256, 1/512), 
            (2/256, 1/512),
            (3/256, 1/512),
            (4/256, 1/512), 
            (1/64, 1.25/512), 
            (1/64, 1.5/512)
        ]
        
        results_list = []
        
        # 1. 读取原始 bin 点云以获取真实点数 (计算 BPP 的分母)
        coords_raw = read_kitti_bin(filedir)
        num_points_raw = len(coords_raw)
        
        if num_points_raw == 0:
            return pd.DataFrame()
            
        # 2. 坐标偏移与放大处理 (放大 1000 倍转到毫米级整数尺度)
        coords = np.round(coords_raw.astype(np.float64) * 1000).astype(np.int32) 
        offset = coords.min(axis=0)
        coords_scaled = coords - offset

        # 3. 运行语义分割模型推理
        result = self.inferencer(inputs=dict(points=filedir), no_save_vis=True, no_save_pred=True)
        seg_labels = result['predictions'][0].get('pred_sem_seg', result['predictions'][0])['pts_semantic_mask']
        if hasattr(seg_labels, 'cpu'): seg_labels = seg_labels.cpu().numpy()
        
        # 获取每个点的预测标签。背景类通常是 0，前景类大于 0
        fg_mask = np.isin(seg_labels[:len(coords_raw)], FG_CLASSES)
        
        coords_fg = coords_scaled[fg_mask]
        coords_bg = coords_scaled[~fg_mask]

        # 4. 将分割好的前景和背景分别保存为 ply 临时文件
        tmp_fg_ply = os.path.join(self.args.tmp_dir, f"{filename}_tmp_fg.ply")
        tmp_bg_ply = os.path.join(self.args.tmp_dir, f"{filename}_tmp_bg.ply")
        
        if len(coords_fg) > 0:
            write_ply_o3d(tmp_fg_ply, coords_fg, normal=True, knn=16)
        if len(coords_bg) > 0:
            write_ply_o3d(tmp_bg_ply, coords_bg, normal=True, knn=16)

        cfg_path = os.path.join(root_dir, "extention", "kitti.cfg")

        # 遍历 12 种组合
        for combo_idx, (posQ_fg, posQ_bg) in enumerate(scale_combinations):
            tmp_fg_bin = os.path.join(self.args.tmp_dir, f"{filename}_tmp_fg_{combo_idx}.bin")
            tmp_bg_bin = os.path.join(self.args.tmp_dir, f"{filename}_tmp_bg_{combo_idx}.bin")
            
            enc_time_total = 0.0
            file_size_bits_total = 0
            
            # A. 压缩前景 (Foreground) - 使用组合中的 posQ_fg
            if len(coords_fg) > 0:
                with suppress_stderr():
                    log_fg = gpcc_encode(tmp_fg_ply, tmp_fg_bin, posQuantscale=posQ_fg, cfgdir=cfg_path)
                if os.path.exists(tmp_fg_bin):
                    file_size_bits_total += os.path.getsize(tmp_fg_bin) * 8
                    enc_time_total += log_fg.get('Processing time (wall)', 0) if isinstance(log_fg, dict) else 0
                    os.remove(tmp_fg_bin)

            # B. 压缩背景 (Background) - 使用组合中的 posQ_bg
            if len(coords_bg) > 0:
                with suppress_stderr():
                    log_bg = gpcc_encode(tmp_bg_ply, tmp_bg_bin, posQuantscale=posQ_bg, cfgdir=cfg_path)
                if os.path.exists(tmp_bg_bin):
                    file_size_bits_total += os.path.getsize(tmp_bg_bin) * 8
                    enc_time_total += log_bg.get('Processing time (wall)', 0) if isinstance(log_bg, dict) else 0
                    os.remove(tmp_bg_bin)
            
            # C. 计算该帧在该组合下的总体 BPP
            bpp = round(file_size_bits_total / num_points_raw, 6)
                
            results = {
                'filename': filename,
                'combo_id': combo_idx,       # 记录是第几个组合，方便之后按此分组求平均
                'posQ_fg': posQ_fg,
                'posQ_bg': posQ_bg,
                'bpp': bpp,
                'enc_time': round(enc_time_total, 4)
            }
            results_list.append(results)
            
        # 清理输入的 ply 临时文件
        if os.path.exists(tmp_fg_ply):
            os.remove(tmp_fg_ply)
        if os.path.exists(tmp_bg_ply):
            os.remove(tmp_bg_ply)

        # 保存该点云独立 CSV 结果
        df = pd.DataFrame(results_list)
        csvfile = os.path.join(self.args.results, f"{filename}.csv")
        df.to_csv(csvfile, index=False)

        return df

    def test_seqs(self, filedir_list):
        all_results_list = []
        
        for idx_file, filedir in enumerate(tqdm(filedir_list, desc="Processing Split Compression")):
            results_df = self.test_bitrates(filedir=filedir, idx_file=idx_file)
            if not results_df.empty:
                all_results_list.append(results_df)

        if all_results_list:
            all_results_df = pd.concat(all_results_list, ignore_index=True)
            detail_csv = os.path.join(self.args.results, 'split_all_details.csv')
            all_results_df.to_csv(detail_csv, index=False)
            
            # 这里按照 combo_id 分组求平均值，以确保不同组合能够单独展示平均结果
            mean_results = all_results_df.groupby(['combo_id', 'posQ_fg', 'posQ_bg']).mean(numeric_only=True).reset_index()
            # 删除 filename 等无意义的平均值列
            if 'filename' in mean_results.columns:
                 mean_results = mean_results.drop(columns=['filename'])
            
            avg_csv = os.path.join(self.args.results, 'split_average_results.csv')
            mean_results.to_csv(avg_csv, index=False)
            
        return all_results_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Foreground-Background Split Compression Rate Test Script')
    
    parser.add_argument('--testdata', type=str, required=True, help='Path to the test point cloud .bin file or directory')
    parser.add_argument('--cfg_file', type=str, required=True, help='Path to mmdet3d segmentation config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to mmdet3d segmentation checkpoint')
    
    parser.add_argument('--results', type=str, default='./results_split', help='Directory to save the results .csv files')
    parser.add_argument('--tmp_dir', type=str, default='./tmp', help='Directory for temporary files (will be cleaned)')
    
    args = parser.parse_args()

    os.makedirs(args.results, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    # ---------------- 依据 val.txt 构建文件列表 ----------------
    if os.path.isdir(args.testdata):
        norm_testdata = os.path.normpath(args.testdata)
        parent_dir = os.path.dirname(norm_testdata)
        grandparent_dir = os.path.dirname(parent_dir)
        
        val_txt_standard = os.path.join(grandparent_dir, 'ImageSets', 'val.txt')
        val_txt_direct = os.path.join(grandparent_dir, 'val.txt')
        
        if os.path.exists(val_txt_standard):
            val_txt_path = val_txt_standard
        elif os.path.exists(val_txt_direct):
            val_txt_path = val_txt_direct
        else:
            print(f"Error: 找不到 val.txt，已尝试查找以下路径：\n1. {val_txt_standard}\n2. {val_txt_direct}")
            sys.exit(1)
            
        with open(val_txt_path, 'r') as f:
            val_ids = [line.strip() for i, line in enumerate(f)]
            
        filedir_list = []
        for vid in val_ids:
            bin_file = os.path.join(args.testdata, f"{vid}.bin")
            if os.path.exists(bin_file):
                filedir_list.append(bin_file)
    else:
        filedir_list = [args.testdata]
    # -----------------------------------------------------------

    if not filedir_list:
        print(f"Error: No valid .bin files found to process.")
        sys.exit(1)

    tester = Tester(args)
    tester.test_seqs(filedir_list=filedir_list)