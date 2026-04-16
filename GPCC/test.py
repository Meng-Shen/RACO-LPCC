# GPCC/test.py
import os
import sys
import glob
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import open3d as o3d
from data_utils.geometry.inout import write_ply_o3d

# 导入 GPCC 工具和评测工具
from gpcc_geo import gpcc_encode, gpcc_decode
from pc_error_geo import pc_error


def read_kitti_bin(filedir):
    """读取 KITTI 格式的 bin 文件，返回 (N, 3) 坐标"""
    coords = np.fromfile(filedir, dtype=np.float32).reshape(-1, 4)
    return coords[:, :3]

def write_kitti_bin(filedir, coords):
    zeros = np.zeros((coords.shape[0], 1), dtype=np.float32)
    coords_4d = np.concatenate([coords, zeros], axis=1)
    coords_4d.astype(np.float32).tofile(filedir)

class Tester():
    def __init__(self, args):
        self.args = args

    def test_bitrates(self, filedir, idx_file):
        filename = os.path.split(filedir)[-1].split('.')[0]
        
        # 10 种量化步长
        posQuantscale_list = [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512]
        #posQuantscale_list = [1/8] # 
        print(f'=====set bitrates=====\nposQuantscale_list: {posQuantscale_list}')
        
        results_list = []
        
        # 1. 读取原始 bin 点云 (米为单位)
        coords_raw = read_kitti_bin(filedir)
        num_points_raw = len(coords_raw)
        
        # 2. 坐标偏移与放大处理
        # 偏移到 (0,0,0) 并放大 1000 倍转到毫米级整数尺度，以满足 GPCC 要求
        coords = np.round(coords_raw.astype(np.float64) * 1000).astype(np.int32) 
        offset = coords.min(axis=0)
        coords_scaled = coords - offset
        
        # 3. 将缩放后的坐标保存为带法线的 ply 临时文件
        # 该文件将作为 gpcc_encode 的输入 和 pc_error 的参考点云
        tmp_ref_normal_ply = os.path.join(self.args.output, f"{filename}_tmp_ref_normal.ply")
        write_ply_o3d(tmp_ref_normal_ply, coords_scaled, normal=True, knn=16)

        for idx_rate, posQuantscale in enumerate(posQuantscale_list):
            true_rate = idx_rate# + 2
            out_bin_file = os.path.join(self.args.output, f"{filename}_R{true_rate}.bin")
            
            # 定义每次压缩的临时文件
            tmp_bitstream_bin = os.path.join(self.args.output, f"{filename}_tmp_bitstream.bin")
            tmp_dec_out_ply = os.path.join(self.args.output, f"{filename}_tmp_dec_out.ply")
            
            # A. 压缩 (Encode)
            print(f'[{filename}] Encoding with posQuantscale={posQuantscale} ...')
            gpcc_dir = os.path.dirname(os.path.abspath(__file__))
            cfg_path = os.path.join(gpcc_dir, "kitti.cfg")
            log_enc = gpcc_encode(tmp_ref_normal_ply, tmp_bitstream_bin, posQuantscale=posQuantscale , cfgdir=cfg_path)
            
            # B. 解压缩 (Decode)
            # 解码出来的 tmp_dec_out_ply 也会处在放大 1000 倍的整数尺度下
            print(f'[{filename}] Decoding ...')
            log_dec = gpcc_decode(tmp_bitstream_bin, tmp_dec_out_ply)
            
            # C. 计算 bpp
            if os.path.exists(tmp_bitstream_bin):
                file_size = os.path.getsize(tmp_bitstream_bin) * 8
                bpp = round(file_size / num_points_raw, 6) if num_points_raw > 0 else 0
            else:
                bpp = 0
                
            # D. 计算 PC Error (都在 1000倍尺度 下比对)
            print(f'[{filename}] Calculating PC Error ...')
            psnr_results = pc_error(tmp_ref_normal_ply, tmp_dec_out_ply, resolution=self.args.resolution, normal=True, show=False)

            # E. 将解码后的整数坐标还原回原位，并保存为目标 bin 格式
            pcd_dec = o3d.io.read_point_cloud(tmp_dec_out_ply)
            coords_dec = np.asarray(pcd_dec.points) + offset
            coords_dec = coords_dec.astype(np.float64)
            coords_dec_restored = coords_dec / 1000.0
            
            write_kitti_bin(out_bin_file, coords_dec_restored)
            
            # 汇总本次结果
            results = {
                'filename': filename,
                'rate': true_rate,
                'posQuantscale': posQuantscale,
                'bpp': bpp,
                'enc_time': log_enc.get('Processing time (wall)', 0),
                'dec_time': log_dec.get('Processing time (wall)', 0)
            }
            results.update(psnr_results)
            print('DBG!!! results:', results)
            results_list.append(results)
            
            # F. 清理本次循环产生的中间文件（码流 bin，gpcc 解码 ply）
            for tmp_f in [tmp_bitstream_bin, tmp_dec_out_ply]:
                if os.path.exists(tmp_f):
                    os.remove(tmp_f)
                    
        # 遍历完所有码率档位后，清理外层的参考 ply 临时文件
        if os.path.exists(tmp_ref_normal_ply):
            os.remove(tmp_ref_normal_ply)

        # 保存为独立 CSV
        df = pd.DataFrame(results_list)
        csvfile = os.path.join(self.args.results, f"{filename}.csv")
        df.to_csv(csvfile, index=False)

        return df

    def test_seqs(self, filedir_list):
        all_results_list = []
        
        # 遍历所有点云
        for idx_file, filedir in enumerate(tqdm(filedir_list, desc="Processing Point Clouds")):
            print(f'\n--- Processing File {idx_file}: {filedir} ---')
            results_df = self.test_bitrates(filedir=filedir, idx_file=idx_file)
            all_results_list.append(results_df)

        if all_results_list:
            all_results_df = pd.concat(all_results_list, ignore_index=True)
            detail_csv = os.path.join(self.args.results, 'all_details.csv')
            all_results_df.to_csv(detail_csv, index=False)
            
            # 求平均值
            mean_results = all_results_df.groupby('rate').mean(numeric_only=True).reset_index()
            avg_csv = os.path.join(self.args.results, 'average_results.csv')
            mean_results.to_csv(avg_csv, index=False)
            print('\nDBG!!! Final Average Results:\n', mean_results)
            
        return all_results_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPCC Point Cloud Compression Test Script')
    
    # 命令行参数配置
    parser.add_argument('--testdata', type=str, required=True, help='Path to the test point cloud .bin file or directory')
    parser.add_argument('--resolution', type=int, default=80000, help='Resolution for pc_error (peak value)')
    parser.add_argument('--output', type=str, default='./output', help='Directory to save the decoded .bin files')
    parser.add_argument('--results', type=str, default='./results', help='Directory to save the results .csv files')
    
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    # 匹配 .bin 格式的测试文件
    if os.path.isdir(args.testdata):
        filedir_list = sorted(glob.glob(os.path.join(args.testdata, '**', '*.bin'), recursive=True))
    else:
        filedir_list = [args.testdata]

    if not filedir_list:
        print(f"Error: No .bin files found in {args.testdata}")
        sys.exit(1)

    print(f'Found {len(filedir_list)} test files.')

    tester = Tester(args)
    tester.test_seqs(filedir_list=filedir_list)