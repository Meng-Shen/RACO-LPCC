# GPCC/encode_only.py
import os
import sys
import glob
import argparse
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement

# 导入 GPCC 工具 (只需编码器)
from gpcc_attr import gpcc_encode

def read_kitti_bin(filedir):
    """读取 KITTI 格式的 bin 文件，返回 (N, 4) 坐标和反射率"""
    coords = np.fromfile(filedir, dtype=np.float32).reshape(-1, 4)
    return coords  # [x, y, z, reflectance]

def write_ply_with_reflectance(filename, xyz, reflectance):
    """使用 plyfile 写入仅包含坐标和 reflectance 的 PLY 文件（无需求法线）"""
    vertex = np.empty(len(xyz), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('reflectance', 'u1') # uchar 类型
    ])
    vertex['x'] = xyz[:, 0]
    vertex['y'] = xyz[:, 1]
    vertex['z'] = xyz[:, 2]
    vertex['reflectance'] = reflectance

    ply = PlyData([PlyElement.describe(vertex, 'vertex')], text=False)
    ply.write(filename)

class Tester():
    def __init__(self, args):
        self.args = args

    def test_bitrates(self, filedir, idx_file):
        
        filename = os.path.split(filedir)[-1].split('.')[0]
        
        # 定义几何与属性的量化档位
        posQuantscale_list = [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512]
        attrQP_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        
        print(f'=====set bitrates=====\nposQuantscale: {posQuantscale_list}\nattrQP: {attrQP_list}')
        
        results_list = []
        
        # 1. 读取原始 bin 点云 (N, 4)
        coords_raw = read_kitti_bin(filedir)
        xyz_raw = coords_raw[:, :3]
        ref_raw = coords_raw[:, 3]
        num_points_raw = len(xyz_raw)
        
        # 2. 坐标与属性缩放处理
        # 几何: 放大 1000 倍转到毫米级整数尺度

        xyz_scaled = np.round(xyz_raw.astype(np.float64) * 1000).astype(np.int32) 
        offset = xyz_scaled.min(axis=0)
        xyz_scaled_shifted = xyz_scaled - offset
        
        # 属性: 浮点数转换为 0~255 的 uint8
        ref_scaled = np.round(ref_raw.astype(np.float64) * 100).astype(np.uint8)

        # 3. 将缩放后的数据保存为带 reflectance 的 ply 临时文件
        tmp_ref_ply = os.path.join(self.args.bitstream, f"{filename}_tmp_ref.ply")
        write_ply_with_reflectance(tmp_ref_ply, xyz_scaled_shifted, ref_scaled)
        

        # 遍历所有码率档位
        for idx_rate, (posQuantscale, attrQP) in enumerate(zip(posQuantscale_list, attrQP_list)):
            
            true_rate = idx_rate
            
            # 临时和最终输出文件路径
            tmp_gpcc_bin = os.path.join(self.args.bitstream, f"{filename}_tmp_gpcc.bin")
            final_bitstream_file = os.path.join(self.args.bitstream, f"{filename}_bitstream_R{true_rate}.bin")
            
            # A. 压缩 (Encode)
            print(f'[{filename}] Encoding (posQ={posQuantscale}, attrQP={attrQP}) ...')
            gpcc_dir = os.path.dirname(os.path.abspath(__file__))
            cfg_path = os.path.join(gpcc_dir, "kitti.cfg")
            start = time.time()
            log_enc = gpcc_encode(tmp_ref_ply, tmp_gpcc_bin, posQuantscale=posQuantscale, attrQP=attrQP, cfgdir=cfg_path)
            encode_time = time.time() - start
            # B. 拼接偏移量和比特流
            if os.path.exists(tmp_gpcc_bin):
                with open(tmp_gpcc_bin, 'rb') as f:
                    gpcc_bytes = f.read()
                
                # 将 offset 转换为 bytes (3个 int32 = 12 bytes)
                offset_bytes = offset.astype(np.int32).tobytes()

                
                # 写入最终比特流文件：首部是 offset，后面是 GPCC 比特流
                with open(final_bitstream_file, 'wb') as f:
                    f.write(offset_bytes)
                    f.write(gpcc_bytes)
                
                # C. 计算 bpp (基于拼接后的总文件大小)
                total_file_size_bytes = len(offset_bytes) + len(gpcc_bytes)
                bpp = round((total_file_size_bytes * 8) / num_points_raw, 6) if num_points_raw > 0 else 0
            else:
                bpp = 0
                print(f"⚠️ [Error] GPCC encoding failed for {filename}, bitstream not found.")

            

            # 汇总本次结果
            results = {
                'filename': filename,
                'rate': true_rate,
                'posQuantscale': posQuantscale,
                'attrQP': attrQP,
                'bpp': bpp,
                'enc_time': encode_time
            }
            print('DBG!!! results:', results)
            results_list.append(results)
            
            # 清理本次循环产生的 GPCC 临时比特流
            if os.path.exists(tmp_gpcc_bin):
                os.remove(tmp_gpcc_bin)
                    
        # 清理外层参考 ply 文件
        if os.path.exists(tmp_ref_ply):
            os.remove(tmp_ref_ply)

        # 保存为独立 CSV
        df = pd.DataFrame(results_list)
        csvfile = os.path.join(self.args.result, f"{filename}.csv")
        df.to_csv(csvfile, index=False)

        return df

    def test_seqs(self, filedir_list):
        all_results_list = []
        
        for idx_file, filedir in enumerate(tqdm(filedir_list, desc="Processing Point Clouds")):
            print(f'\n--- Processing File {idx_file}: {filedir} ---')
            results_df = self.test_bitrates(filedir=filedir, idx_file=idx_file)
            all_results_list.append(results_df)

        if all_results_list:
            all_results_df = pd.concat(all_results_list, ignore_index=True)
            detail_csv = os.path.join(self.args.result, 'all_details.csv')
            all_results_df.to_csv(detail_csv, index=False)
            
            # 求平均值
            mean_results = all_results_df.groupby('rate').mean(numeric_only=True).reset_index()
            avg_csv = os.path.join(self.args.result, 'average_results.csv')
            mean_results.to_csv(avg_csv, index=False)
            print('\nDBG!!! Final Average Results:\n', mean_results)
            
        return all_results_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPCC Point Cloud Encoder Script')
    
    parser.add_argument('--testdata', type=str, required=True, help='Path to the test point cloud .bin file or directory')
    parser.add_argument('--bitstream', type=str, default='./bitstream', help='Directory to save the final encoded .bin bitstream files')
    parser.add_argument('--result', type=str, default='./result', help='Directory to save the results .csv files')
    
    args = parser.parse_args()

    os.makedirs(args.bitstream, exist_ok=True)
    os.makedirs(args.result, exist_ok=True)

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