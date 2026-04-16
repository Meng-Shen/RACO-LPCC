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
from plyfile import PlyData, PlyElement

# 导入 GPCC 工具和评测工具 (注: 你的 gpcc_encode 需要支持接收 attrQP 参数)
from gpcc_attr import gpcc_encode, gpcc_decode
from pc_error_attr import pc_error


def read_kitti_bin(filedir):
    """读取 KITTI 格式的 bin 文件，返回 (N, 4) 坐标和反射率"""
    coords = np.fromfile(filedir, dtype=np.float32).reshape(-1, 4)
    return coords  # [x, y, z, reflectance]

def write_kitti_bin(filedir, coords):
    """将 (N, 4) 的数据写回 bin"""
    coords.astype(np.float32).tofile(filedir)

def write_ply_with_reflectance_and_normal(filename, xyz, reflectance):
    """使用 plyfile 写入包含法线和 reflectance 的 PLY 文件"""
    # 1. 使用 Open3D 计算法线 (为了 pc_error 计算 point-to-plane)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
    normals = np.asarray(pcd.normals)

    # 2. 构建包含自定义属性的复合数据类型
    vertex = np.empty(len(xyz), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('reflectance', 'u1') # uchar 类型，G-PCC 必备
    ])
    vertex['x'] = xyz[:, 0]
    vertex['y'] = xyz[:, 1]
    vertex['z'] = xyz[:, 2]
    vertex['nx'] = normals[:, 0]
    vertex['ny'] = normals[:, 1]
    vertex['nz'] = normals[:, 2]
    vertex['reflectance'] = reflectance

    # 3. 保存为二进制 PLY
    ply = PlyData([PlyElement.describe(vertex, 'vertex')], text=False)
    ply.write(filename)

def read_ply_with_reflectance(filename):
    """使用 plyfile 读取解码后的 PLY 文件，并自适应匹配属性名"""
    plydata = PlyData.read(filename)
    v = plydata['vertex']
    xyz = np.vstack([v['x'], v['y'], v['z']]).T
    
    # 1. 获取解码后 PLY 文件中的所有字段名
    prop_names = [p.name for p in v.properties]
    
    # 2. 常见 G-PCC 解码后的属性名称映射
    ref_keys = ['reflectance', 'intensity', 'val', 'red', 'refc'] 
    
    reflectance = None
    for key in ref_keys:
        if key in prop_names:
            reflectance = v[key]
            # print(f"  [Debug] 成功在解码文件中读取到属性字段: '{key}'")
            break
            
    # 3. 如果还是找不到，打印出文件里到底有什么字段，方便我们排查
    if reflectance is None:
        print(f"\n⚠️ [致命错误] 解码后的 PLY 文件中找不到反射率属性！")
        print(f"⚠️ 当前 PLY 文件中实际包含的字段有: {prop_names}")
        print(f"⚠️ 这意味着 G-PCC 在编码时根本没有把反射率压缩进去。")
        raise ValueError(f"No reflectance/intensity field found! Available fields: {prop_names}")
        
    return xyz, reflectance

def write_ply_for_pcerror(filename, xyz, ref, compute_normals=False):
    """
    专门为 pc_error_d 伪造一个 RGB 彩色 PLY。
    将反射率 ref 强行赋值给 red, green, blue，骗过评测软件。
    """
    dtypes = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    if compute_normals:
        dtypes.extend([('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
        
    vertex = np.empty(len(xyz), dtype=dtypes)
    vertex['x'] = xyz[:, 0]
    vertex['y'] = xyz[:, 1]
    vertex['z'] = xyz[:, 2]
    # 【核心伪装】把反射率变成颜色
    vertex['red'] = ref
    vertex['green'] = ref
    vertex['blue'] = ref
    
    if compute_normals:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
        normals = np.asarray(pcd.normals)
        vertex['nx'] = normals[:, 0]
        vertex['ny'] = normals[:, 1]
        vertex['nz'] = normals[:, 2]
        
    ply = PlyData([PlyElement.describe(vertex, 'vertex')], text=False)
    ply.write(filename)

class Tester():
    def __init__(self, args):
        self.args = args

    def test_bitrates(self, filedir, idx_file):
        filename = os.path.split(filedir)[-1].split('.')[0]
        
        # 定义几何与属性的量化档位 (需一一对应)
        posQuantscale_list = [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512]
        attrQP_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        posQuantscale_list = [1]
        attrQP_list = [1]
        
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
        
        # 属性: KITTI 通常是 0~1 的浮点数，转换为 0~255 的 uint8
        ref_scaled = np.clip(np.round(ref_raw * 255.0), 0, 255).astype(np.uint8)
        ref_scaled = np.round(ref_raw.astype(np.float64)  * 100).astype(np.uint8)
        
        # 3. 将缩放后的数据保存为带 reflectance 的 ply 临时文件
        tmp_ref_ply = os.path.join(self.args.output, f"{filename}_tmp_ref.ply")
        write_ply_with_reflectance_and_normal(tmp_ref_ply, xyz_scaled_shifted, ref_scaled)

        # 遍历所有码率档位
        for idx_rate, (posQuantscale, attrQP) in enumerate(zip(posQuantscale_list, attrQP_list)):
            true_rate = idx_rate
            out_bin_file = os.path.join(self.args.output, f"{filename}_R{true_rate}.bin")
            
            tmp_bitstream_bin = os.path.join(self.args.output, f"{filename}_tmp_bitstream.bin")
            tmp_dec_out_ply = os.path.join(self.args.output, f"{filename}_tmp_dec_out.ply")
            
            # A. 压缩 (Encode)
            print(f'[{filename}] Encoding (posQ={posQuantscale}, attrQP={attrQP}) ...')
            gpcc_dir = os.path.dirname(os.path.abspath(__file__))
            cfg_path = os.path.join(gpcc_dir, "kitti.cfg")
            
            # 注意: 此处假设你已将上文提供的 attrQP 参数加入了 gpcc_encode 函数中
            log_enc = gpcc_encode(tmp_ref_ply, tmp_bitstream_bin, posQuantscale=posQuantscale, attrQP=attrQP, cfgdir=cfg_path)
            
            # B. 解压缩 (Decode)
            print(f'[{filename}] Decoding ...')
            log_dec = gpcc_decode(tmp_bitstream_bin, tmp_dec_out_ply)
            
            # C. 计算 bpp
            if os.path.exists(tmp_bitstream_bin):
                file_size = os.path.getsize(tmp_bitstream_bin) * 8
                bpp = round(file_size / num_points_raw, 6) if num_points_raw > 0 else 0
            else:
                bpp = 0
                
            # ================= D. 评测环节：制作伪装文件 =================
            print(f'[{filename}] Preparing Eval Files & Calculating PC Error ...')
            
            # 1. 抓取解压出来的几何和属性
            xyz_dec, ref_dec = read_ply_with_reflectance(tmp_dec_out_ply)
            
            # 2. 生成给 pc_error 专用的“伪彩色” PLY 
            tmp_ref_eval_ply = os.path.join(self.args.output, f"{filename}_tmp_ref_eval.ply")
            tmp_dec_eval_ply = os.path.join(self.args.output, f"{filename}_tmp_dec_eval.ply")
            
            # 参考点云需要算法线(为了算 p2plane)，解压点云不需要
            write_ply_for_pcerror(tmp_ref_eval_ply, xyz_scaled_shifted, ref_scaled, compute_normals=True)
            write_ply_for_pcerror(tmp_dec_eval_ply, xyz_dec, ref_dec, compute_normals=False)
            
            # 3. 计算 PC Error (喂给它我们伪造的 eval 文件)
            psnr_results = pc_error(tmp_ref_eval_ply, tmp_dec_eval_ply, resolution=self.args.resolution, normal=True, attr=True, show=False)
            # =============================================================

            # E. 还原数据并保存为目标 bin
            xyz_dec_restored = (xyz_dec.astype(np.float64) + offset) / 1000.0
            ref_dec_restored = ref_dec.astype(np.float64) / 100.0
            coords_dec_restored = np.hstack([xyz_dec_restored, ref_dec_restored.reshape(-1, 1)])
            write_kitti_bin(out_bin_file, coords_dec_restored)
            
            # 汇总本次结果
            results = {
                'filename': filename,
                'rate': true_rate,
                'posQuantscale': posQuantscale,
                'attrQP': attrQP,
                'bpp': bpp,
                'enc_time': log_enc.get('Processing time (wall)', 0),
                'dec_time': log_dec.get('Processing time (wall)', 0)
            }
            results.update(psnr_results)
            print('DBG!!! results:', results)
            results_list.append(results)
            
            # F. 清理本次循环产生的中间文件
            for tmp_f in [tmp_bitstream_bin, tmp_dec_out_ply, tmp_ref_eval_ply, tmp_dec_eval_ply]:
                if os.path.exists(tmp_f):
                    os.remove(tmp_f)
                    
        # 清理外层参考 ply 文件
        if os.path.exists(tmp_ref_ply):
            os.remove(tmp_ref_ply)

        # 保存为独立 CSV
        df = pd.DataFrame(results_list)
        csvfile = os.path.join(self.args.results, f"{filename}.csv")
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
    
    parser.add_argument('--testdata', type=str, required=True, help='Path to the test point cloud .bin file or directory')
    parser.add_argument('--resolution', type=int, default=30000, help='Resolution for pc_error (peak value)')
    parser.add_argument('--output', type=str, default='./output', help='Directory to save the decoded .bin files')
    parser.add_argument('--results', type=str, default='./results', help='Directory to save the results .csv files')
    
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

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