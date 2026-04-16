import os
import argparse
import subprocess

import numpy as np
from glob import glob
from tqdm import tqdm

from multiprocessing import Pool

parser = argparse.ArgumentParser(
    prog='eval.py',
    description='Eval geometry PSNR.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input_glob', type=str, help='Glob pattern to load point clouds.', default='./data/examples/*.ply')
parser.add_argument('--decompressed_path', type=str, help='Path to save decompressed files.', default='./data/kitt_decompressed')
parser.add_argument('--pcc_metric_path', type=str, help='Path for pc_error_d.', default='./third_party/pc_error_d')
parser.add_argument('--resolution', type=float, help='Point cloud resolution (peak signal).', default=59.70)

args = parser.parse_args()

files = np.array(glob(args.input_glob))

# check if decode file exists
checked_files = []
for file in files:
    filename_wo_ext = os.path.split(file)[-1].split('.ply')[0] # 000001
    dec_f = os.path.join(os.path.abspath(args.decompressed_path), filename_wo_ext+'.ply.bin.ply')
    if os.path.exists(dec_f):
        checked_files.append(file)
files = checked_files

def process(input_f):
    filename_wo_ext = os.path.split(input_f)[-1].split('.ply')[0]
    if 'vox10' in filename_wo_ext:
        res = 2**10
    elif 'vox11' in filename_wo_ext:
        res = 2**11
    elif 'vox9' in filename_wo_ext:
        res = 2**9
    elif 'vox12' in filename_wo_ext:
        res = 2**12
    dec_f = os.path.join(os.path.abspath(args.decompressed_path), filename_wo_ext+'.ply.bin.ply')
    # 先给路径加双引号（避免空格问题，提前规避常见错误）
    cmd = f'"{args.pcc_metric_path}" \
    --fileA="{input_f}" --fileB="{dec_f}" \
    --resolution={res} --inputNorm="{input_f}"'
    
    d1_psnr, d2_psnr = -1, -1
    output = None  # 初始化，避免局部变量未定义
    
    try:
        # 执行命令，捕获输出（字节流）
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        # 解码字节流为字符串（处理编码问题）
        output_str = output.decode('utf-8', errors='ignore').strip()
        
        # 解析 p2point PSNR
        if 'mseF,PSNR (p2point):' not in output_str:
            raise ValueError(f"工具输出中未找到 'mseF,PSNR (p2point):' 字段")
        p2point_part = output_str.split('mseF,PSNR (p2point):')[1].split('\n')[0].strip()
        d1_psnr = float(p2point_part)
        
        # 解析 p2plane PSNR
        if 'mseF,PSNR (p2plane):' not in output_str:
            raise ValueError(f"工具输出中未找到 'mseF,PSNR (p2plane):' 字段")
        p2plane_part = output_str.split('mseF,PSNR (p2plane):')[1].split('\n')[0].strip()
        d2_psnr = float(p2plane_part)
        
    except subprocess.CalledProcessError as e:
        # 场景1：命令执行失败（工具返回非0退出码）
        print(f'\n==================================================')
        print(f'!!! 命令执行失败 !!!')
        print(f'文件：{filename_wo_ext}')
        print(f'异常类型：{type(e).__name__}')
        print(f'退出码：{e.returncode}')
        print(f'执行命令：{cmd}')
        print(f'工具错误输出：\n{e.output.decode("utf-8", errors="ignore")}')
        print(f'==================================================\n')
        
    except ValueError as e:
        # 场景2：解析失败（字段缺失、无法转float）
        print(f'\n==================================================')
        print(f'!!! 输出解析失败 !!!')
        print(f'文件：{filename_wo_ext}')
        print(f'异常信息：{str(e)}')
        print(f'执行命令：{cmd}')
        if output:
            print(f'工具原始输出：\n{output.decode("utf-8", errors="ignore")}')
        print(f'==================================================\n')
        
    except PermissionError:
        # 场景3：无执行权限（Linux/Mac 常见）
        print(f'\n==================================================')
        print(f'!!! 权限不足 !!!')
        print(f'文件：{filename_wo_ext}')
        print(f'问题：{args.pcc_metric_path} 无执行权限，请运行 chmod +x 赋予权限')
        print(f'执行命令：{cmd}')
        print(f'==================================================\n')
        
    except FileNotFoundError as e:
        # 场景4：文件不存在（工具/输入文件/解压文件）
        print(f'\n==================================================')
        print(f'!!! 文件未找到 !!!')
        print(f'文件：{filename_wo_ext}')
        print(f'缺失文件：{str(e.filename)}')
        print(f'执行命令：{cmd}')
        print(f'检查：1. pc_error_d 路径是否正确 2. 输入文件/解压文件是否存在')
        print(f'==================================================\n')
        
    except Exception as e:
        # 场景5：其他未知异常（兜底捕获）
        print(f'\n==================================================')
        print(f'!!! 未知异常 !!!')
        print(f'文件：{filename_wo_ext}')
        print(f'异常类型：{type(e).__name__}')
        print(f'异常信息：{str(e)}')
        print(f'执行命令：{cmd}')
        if output:
            print(f'工具原始输出：\n{output.decode("utf-8", errors="ignore")}')
        print(f'==================================================\n')
    
    return np.array([filename_wo_ext, d1_psnr, d2_psnr])


with Pool(32) as p:
    arr = list(tqdm(p.imap(process, files), total=len(files)))

# process(files[0])

arr = np.array(arr)

fnames, d1_PSNRs, d2_PSNRs = arr[:, 0], arr[:, 1].astype(float), arr[:, 2].astype(float)

print('Avg. D1 PSNR:', round(d1_PSNRs.mean(), 3))
print('Avg. D2 PSNR:', round(d2_PSNRs.mean(), 3))
