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

from data_utils.geometry.inout import write_ply_o3d
from extention.gpcc_geo import gpcc_encode

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

    def test_bitrates(self, filedir, idx_file):
        filename = os.path.split(filedir)[-1].split('.')[0]
        
        # 10 种量化步长
        posQuantscale_list = [1/64, 1.5/128, 1/128, 1.5/256, 1/256, 1.5/512, 1/512]
        
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
        
        # 3. 将缩放后的坐标保存为带法线的 ply 临时文件
        tmp_ref_normal_ply = os.path.join(self.args.tmp_dir, f"{filename}_tmp_ref_normal.ply")
        write_ply_o3d(tmp_ref_normal_ply, coords_scaled, normal=True, knn=16)

        # 修正 cfg 配置文件路径：指向 ../extention/kitti.cfg
        cfg_path = os.path.join(root_dir, "extention", "kitti.cfg")

        for posQuantscale in posQuantscale_list:
            tmp_bitstream_bin = os.path.join(self.args.tmp_dir, f"{filename}_tmp_bitstream.bin")
            
            # A. 仅压缩 (Encode) - 使用黑洞拦截 C++ 警告
            with suppress_stderr():
                log_enc = gpcc_encode(tmp_ref_normal_ply, tmp_bitstream_bin, posQuantscale=posQuantscale, cfgdir=cfg_path)
            
            # B. 计算 bpp (Bits Per Point)
            bpp = 0.0
            if os.path.exists(tmp_bitstream_bin):
                file_size_bits = os.path.getsize(tmp_bitstream_bin) * 8
                bpp = round(file_size_bits / num_points_raw, 6)
                
                # C. 阅后即焚：立刻删除生成的码流文件
                os.remove(tmp_bitstream_bin)
            else:
                print(f"[Warning] {filename} 压缩失败 (posQuantscale={posQuantscale})，未生成码流文件。")
                
            # 汇总本次结果
            results = {
                'filename': filename,
                'posQuantscale': posQuantscale,
                'bpp': bpp,
                'enc_time': log_enc.get('Processing time (wall)', 0) if isinstance(log_enc, dict) else 0
            }
            results_list.append(results)
            
        # 清理外层的输入 ply 临时文件
        if os.path.exists(tmp_ref_normal_ply):
            os.remove(tmp_ref_normal_ply)

        # 保存该点云独立 CSV 结果
        df = pd.DataFrame(results_list)
        csvfile = os.path.join(self.args.results, f"{filename}.csv")
        df.to_csv(csvfile, index=False)

        return df

    def test_seqs(self, filedir_list):
        all_results_list = []
        
        for idx_file, filedir in enumerate(tqdm(filedir_list, desc="Processing Point Clouds")):
            results_df = self.test_bitrates(filedir=filedir, idx_file=idx_file)
            if not results_df.empty:
                all_results_list.append(results_df)

        if all_results_list:
            all_results_df = pd.concat(all_results_list, ignore_index=True)
            detail_csv = os.path.join(self.args.results, 'gpcc_all_details.csv')
            all_results_df.to_csv(detail_csv, index=False)
            
            # 求按量化步长统计的平均值
            mean_results = all_results_df.groupby('posQuantscale').mean(numeric_only=True).reset_index()
            avg_csv = os.path.join(self.args.results, 'gpcc_average_results.csv')
            mean_results.to_csv(avg_csv, index=False)
            
        return all_results_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPCC Point Cloud Compression Rate Test Script')
    
    parser.add_argument('--testdata', type=str, required=True, help='Path to the test point cloud .bin file or directory')
    parser.add_argument('--results', type=str, default='./results_gpcc', help='Directory to save the results .csv files')
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