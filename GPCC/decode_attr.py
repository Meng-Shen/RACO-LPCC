# GPCC/decode_only.py
import os
import sys
import glob
import argparse
import re
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from plyfile import PlyData

# 导入 GPCC 工具 (只需解码器)
from gpcc_attr import gpcc_decode

def read_ply_with_reflectance(filename):
    """使用 plyfile 读取解码后的 PLY 文件，并提取坐标和反射率"""
    plydata = PlyData.read(filename)
    v = plydata['vertex']
    xyz = np.vstack([v['x'], v['y'], v['z']]).T
    
    # 获取解码后 PLY 文件中的所有字段名
    prop_names = [p.name for p in v.properties]
    
    # 常见 G-PCC 解码后的属性名称映射
    ref_keys = ['reflectance', 'intensity', 'val', 'red', 'refc'] 
    
    reflectance = None
    for key in ref_keys:
        if key in prop_names:
            reflectance = v[key]
            break
            
    if reflectance is None:
        print(f"\n⚠️ [致命错误] 解码后的 PLY 文件中找不到反射率属性！")
        print(f"⚠️ 当前 PLY 文件中实际包含的字段有: {prop_names}")
        raise ValueError(f"No reflectance/intensity field found! Available fields: {prop_names}")
        
    return xyz, reflectance

def write_kitti_bin(filedir, coords):
    """将 (N, 4) 的数据写回 KITTI 格式的 bin 文件"""
    coords.astype(np.float32).tofile(filedir)

class DecoderTester():
    def __init__(self, args):
        self.args = args

    def decode_seqs(self, filedir_list):
        all_results_list = []
        
        for idx_file, bitstream_file in enumerate(tqdm(filedir_list, desc="Decoding Point Clouds")):
            
            basename = os.path.basename(bitstream_file)
            
            # 1. 使用正则解析文件名: 提取 filename 和 rate
            match = re.search(r'(.*)_bitstream_R(\d+)\.bin$', basename)
            if not match:
                print(f"\n⚠️ 跳过不符合命名规则的文件: {basename}")
                continue
                
            filename = match.group(1)
            rate = int(match.group(2))
            
            # 2. 读取码流文件
            with open(bitstream_file, 'rb') as f:
                # 提取前12个字节恢复 offset (3 个 int32)
                offset_bytes = f.read(12)
                offset = np.frombuffer(offset_bytes, dtype=np.int32)
                
                # 读取剩余的真正 G-PCC 码流
                gpcc_bytes = f.read()
                
            # 3. 将真实的 G-PCC 码流暂存为临时文件供 gpcc_decode 调用
            tmp_gpcc_bin = os.path.join(self.args.output, f"{filename}_tmp_gpcc.bin")
            tmp_dec_ply = os.path.join(self.args.output, f"{filename}_tmp_dec.ply")
            
            with open(tmp_gpcc_bin, 'wb') as f:
                f.write(gpcc_bytes)
                
            # 4. 调用 G-PCC 解码并记录时间
            start = time.time()
            log_dec = gpcc_decode(tmp_gpcc_bin, tmp_dec_ply)
            dec_time = time.time() - start
            # 5. 读取解码后的 PLY
            if not os.path.exists(tmp_dec_ply):
                print(f"\n⚠️ 找不到解码输出文件 {tmp_dec_ply}，跳过。")
                continue
                
            xyz_dec, ref_dec = read_ply_with_reflectance(tmp_dec_ply)
            
            # 6. 数据还原 (逆向映射)
            # 几何: 加上 offset 并除以 1000 转换回浮点数尺度
            xyz_restored = (xyz_dec.astype(np.float64) + offset) / 1000.0
            # 属性: 除以 100 转换回原始浮点范围
            ref_restored = ref_dec.astype(np.float64) / 100.0
            
            # 拼接为 (N, 4)
            coords_restored = np.hstack([xyz_restored, ref_restored.reshape(-1, 1)])
            
            
            # 7. 保存到对应 rate 的文件夹
            rate_dir = os.path.join(self.args.output, f"R{rate}")
            os.makedirs(rate_dir, exist_ok=True)
            
            out_bin_file = os.path.join(rate_dir, f"{filename}.bin")
            write_kitti_bin(out_bin_file, coords_restored)

            
            
            # 8. 汇总当前文件解码结果
            results = {
                'filename': filename,
                'rate': rate,
                'dec_time': dec_time
            }
            all_results_list.append(results)
            
            # 9. 清理临时文件
            for tmp_f in [tmp_gpcc_bin, tmp_dec_ply]:
                if os.path.exists(tmp_f):
                    os.remove(tmp_f)

        # ================= 10. 保存评测结果到 CSV =================
        if all_results_list:
            df = pd.DataFrame(all_results_list)
            
            # 按文件名分别保存独立的 CSV
            for fname, group in df.groupby('filename'):
                csvfile = os.path.join(self.args.result, f"{fname}_dec.csv")
                group.sort_values(by='rate').to_csv(csvfile, index=False)
            
            # 保存汇总所有明细的 CSV
            detail_csv = os.path.join(self.args.result, 'all_decode_details.csv')
            df.sort_values(by=['filename', 'rate']).to_csv(detail_csv, index=False)
            
            # 求各码率档位的平均耗时
            mean_results = df.groupby('rate').mean(numeric_only=True).reset_index()
            avg_csv = os.path.join(self.args.result, 'average_decode_results.csv')
            mean_results.to_csv(avg_csv, index=False)
            print('\nDBG!!! Final Average Decode Results:\n', mean_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPCC Point Cloud Decoder Script')
    
    parser.add_argument('--bitstream', type=str, required=True, help='Directory containing the compressed _bitstream_R{rate}.bin files')
    parser.add_argument('--output', type=str, default='./output', help='Directory to save the reconstructed .bin files')
    parser.add_argument('--result', type=str, default='./result', help='Directory to save the decoding results .csv files')
    
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.result, exist_ok=True)

    # 搜索码流目录下的所有 bin 文件
    filedir_list = sorted(glob.glob(os.path.join(args.bitstream, '**', '*.bin'), recursive=True))

    if not filedir_list:
        print(f"Error: No .bin files found in {args.bitstream}")
        sys.exit(1)

    print(f'Found {len(filedir_list)} bitstream files to decode.')

    decoder = DecoderTester(args)
    decoder.decode_seqs(filedir_list=filedir_list)
    print("\n✅ 所有文件解码并统计完成！")