# GPCC/eval_psnr.py
import os
import sys
import glob
import argparse
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import open3d as o3d
from plyfile import PlyData, PlyElement

# 导入评测工具
from pc_error_attr import pc_error

def read_kitti_bin(filedir):
    """读取 KITTI 格式的 bin 文件，返回 (N, 4) 坐标和反射率"""
    coords = np.fromfile(filedir, dtype=np.float32).reshape(-1, 4)
    return coords  # [x, y, z, reflectance]

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

class PSNREvaluator():
    def __init__(self, args):
        self.args = args

    def evaluate_all(self):
        # 1. 建立原始点云的文件映射字典 {basename: full_path}
        orig_files = glob.glob(os.path.join(self.args.origin, '**', '*.bin'), recursive=True)
        if not orig_files:
            print(f"⚠️ [Error] 在 {self.args.origin} 中找不到任何原始 .bin 文件。")
            sys.exit(1)
        orig_map = {os.path.basename(f): f for f in orig_files}
        
        # 2. 获取所有重建点云
        recon_files = sorted(glob.glob(os.path.join(self.args.recon, '**', '*.bin'), recursive=True))
        if not recon_files:
            print(f"⚠️ [Error] 在 {self.args.recon} 中找不到任何重建 .bin 文件。")
            sys.exit(1)
            
        print(f"Found {len(recon_files)} reconstructed files to evaluate.")
        
        all_results_list = []
        
        for recon_file in tqdm(recon_files, desc="Calculating PSNR"):
            basename = os.path.basename(recon_file)  # 例如: 000000.bin
            filename = basename.split('.')[0]        # 例如: 000000
            
            # ================= 3. 提取码率信息 =================
            # 获取重建点云所在的父文件夹名字
            parent_dir = os.path.basename(os.path.dirname(recon_file))
            match = re.search(r'R(\d+)', parent_dir)
            if match:
                rate = int(match.group(1))
                rate_str = f"R{rate}"
            else:
                rate = parent_dir
                rate_str = parent_dir
                print(f"⚠️ 警告: 无法从路径 {recon_file} 中解析出标准 R{{rate}}，将使用文件夹名: {rate}")
            
            # ================= 4. 匹配原始点云 =================
            # 因为名字一模一样，直接用 basename 去字典里找
            orig_file = orig_map.get(basename)
            if orig_file is None:
                print(f"\n⚠️ 跳过 {recon_file}: 在 origin 目录中找不到同名文件 {basename}。")
                continue
                
            # ================= 5. 读取数据 =================
            coords_orig = read_kitti_bin(orig_file)
            coords_recon = read_kitti_bin(recon_file)
            
            xyz_orig = coords_orig[:, :3]
            ref_orig = coords_orig[:, 3]
            xyz_recon = coords_recon[:, :3]
            ref_recon = coords_recon[:, 3]
            
            # ================= 6. 数据对齐与预处理 =================
            # 【注意】：统一使用原始点云的 offset 确保坐标系一致！
            
            # A. 原始点云: 放大 1000 倍，化为整数，并平移到正坐标
            xyz_orig_scaled = np.round(xyz_orig.astype(np.float64) * 1000).astype(np.int32) 
            offset = xyz_orig_scaled.min(axis=0)
            xyz_orig_shifted = xyz_orig_scaled - offset
            ref_orig_scaled = np.round(ref_orig.astype(np.float64) * 100).astype(np.uint8)
            
            # B. 重建点云: 放大 1000 倍，化为整数，减去【原始点云】的 offset
            xyz_recon_scaled = np.round(xyz_recon.astype(np.float64) * 1000).astype(np.int32)
            xyz_recon_shifted = xyz_recon_scaled - offset
            ref_recon_scaled = np.clip(np.round(ref_recon.astype(np.float64) * 100), 0, 255).astype(np.uint8)
            
            # ================= 7. 生成伪装 PLY 用于评测 =================
            tmp_orig_ply = os.path.join(self.args.result, f"tmp_{filename}_orig.ply")
            tmp_recon_ply = os.path.join(self.args.result, f"tmp_{filename}_recon.ply")
            
            # 原始点云需要计算法线用于 p2plane，重建点云不需要
            write_ply_for_pcerror(tmp_orig_ply, xyz_orig_shifted, ref_orig_scaled, compute_normals=True)
            write_ply_for_pcerror(tmp_recon_ply, xyz_recon_shifted, ref_recon_scaled, compute_normals=False)
            
            # ================= 8. 调用 pc_error 计算 =================
            psnr_results = pc_error(
                tmp_orig_ply, 
                tmp_recon_ply, 
                resolution=self.args.resolution, 
                normal=True, 
                attr=True, 
                show=False
            )
            
            # ================= 9. 保存当前单次评测结果 =================
            result_dict = {
                'filename': filename,
                'rate': rate
            }
            result_dict.update(psnr_results)
            all_results_list.append(result_dict)
            
            # 直接保存带有码率信息的独立 CSV (满足“保存的csv文件里也要带有码率”需求)
            single_csv_name = os.path.join(self.args.result, f"{filename}_{rate_str}_psnr.csv")
            pd.DataFrame([result_dict]).to_csv(single_csv_name, index=False)
            
            # 清理中间文件
            for tmp_f in [tmp_orig_ply, tmp_recon_ply]:
                if os.path.exists(tmp_f):
                    os.remove(tmp_f)

        # ================= 10. 保存汇总汇总表 =================
        if all_results_list:
            df = pd.DataFrame(all_results_list)
            
            # 保存所有文件的明细总表
            detail_csv = os.path.join(self.args.result, 'all_psnr_details.csv')
            df.sort_values(by=['filename', 'rate']).to_csv(detail_csv, index=False)
            
            # 尝试按数值码率求平均并保存
            try:
                # 过滤出 rate 是数字的行计算均值
                numeric_df = df[pd.to_numeric(df['rate'], errors='coerce').notnull()].copy()
                numeric_df['rate'] = numeric_df['rate'].astype(int)
                
                mean_results = numeric_df.groupby('rate').mean(numeric_only=True).reset_index()
                mean_results = mean_results.sort_values('rate')
                
                avg_csv = os.path.join(self.args.result, 'average_psnr_results.csv')
                mean_results.to_csv(avg_csv, index=False)
                print('\n✅ [Summary] Average PSNR Results:\n', mean_results)
            except Exception as e:
                print("\n⚠️ 无法计算平均值:", e)
                
        print(f"\n✅ 所有评估完成，结果已存入: {self.args.result}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud PSNR Evaluation Script')
    
    parser.add_argument('--origin', type=str, required=True, help='Directory containing the original .bin point clouds')
    parser.add_argument('--recon', type=str, required=True, help='Directory containing the reconstructed .bin point clouds')
    parser.add_argument('--result', type=str, default='./result', help='Directory to save the PSNR result .csv files')
    parser.add_argument('--resolution', type=int, default=80000, help='Resolution peak value for pc_error (default: 80000)')
    
    args = parser.parse_args()

    os.makedirs(args.result, exist_ok=True)

    evaluator = PSNREvaluator(args)
    evaluator.evaluate_all()