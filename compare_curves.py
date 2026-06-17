import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Comparison of Baseline, Split, and JUCP GPCC.")
    # Baseline GPCC 的文件路径
    parser.add_argument('--gpcc_log', type=str, default=None, help='Path to Baseline GPCC log file')
    parser.add_argument('--gpcc_csv', type=str, default=None, help='Path to Baseline gpcc_average_results.csv')
    
    # Split GPCC 的文件路径
    parser.add_argument('--split_log', type=str, required=True, help='Path to Split test_split.py log file')
    parser.add_argument('--split_csv', type=str, required=True, help='Path to Split average_results.csv')
    
    # JUCP 的文件夹路径
    parser.add_argument('--jucp_txt_dir', type=str, required=True, help='Dir containing jucp*.txt log files')
    parser.add_argument('--jucp_csv_dir', type=str, required=True, help='Dir containing jucp*.csv label files')
    
    # 输出图片路径 (将会作为基础路径被修改，例如 _Car.png)
    parser.add_argument('--out', type=str, default='Compare_mAP_BPP.png', help='Base output image path')
    return parser.parse_args()

def extract_map_from_log(log_path, is_split=False):
    """ 提取 Baseline 或 Split 这种单个文件包含多个评测挡位的日志 """
    map_data = {}
    current_idx = None
    target_class = None

    with open(log_path, 'r') as f:
        for line in f:
            if is_split:
                m_idx = re.search(r'Combo\s+(\d+)', line)
                if m_idx:
                    current_idx = int(m_idx.group(1))
                    if current_idx not in map_data: map_data[current_idx] = {}
                    continue
            else:
                m_idx = re.search(r'Start Evaluation for Scale:\s*([0-9\.]+)', line)
                if m_idx:
                    current_idx = float(m_idx.group(1))
                    if current_idx not in map_data: map_data[current_idx] = {}
                    continue

            if current_idx is None: continue

            m_car = re.search(r'(Car)\s+AP_R40@0\.70,\s*0\.70,\s*0\.70', line)
            if m_car:
                target_class = 'Car'
                continue
                
            m_other = re.search(r'(Pedestrian|Cyclist)\s+AP_R40@0\.50,\s*0\.50,\s*0\.50', line)
            if m_other:
                target_class = m_other.group(1)
                continue

            if target_class is not None:
                m_3d = re.search(r'3d\s+AP:\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)', line)
                if m_3d:
                    map_data[current_idx][target_class] = float(m_3d.group(2))
                    target_class = None
    return map_data

def extract_single_map(log_path):
    """ 提取 JUCP 单个日志文件中的 AP 结果 """
    aps = {}
    target_class = None
    with open(log_path, 'r') as f:
        for line in f:
            m_car = re.search(r'(Car)\s+AP_R40@0\.70,\s*0\.70,\s*0\.70', line)
            if m_car: target_class = 'Car'; continue
            
            m_other = re.search(r'(Pedestrian|Cyclist)\s+AP_R40@0\.50,\s*0\.50,\s*0\.50', line)
            if m_other: target_class = m_other.group(1); continue

            if target_class is not None:
                m_3d = re.search(r'3d\s+AP:\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)', line)
                if m_3d:
                    aps[target_class] = float(m_3d.group(2)) # Moderate Difficulty
                    target_class = None
    return aps

def extract_jucp_csv_name(log_path):
    """ 从 test_jucp_split.py 的日志中提取本次使用的 JUCP CSV 文件名 """
    with open(log_path, 'r') as f:
        for line in f:
            m = re.search(r'jucp_csv\s+(\S+\.csv)', line)
            if m:
                return m.group(1)
    return None

def find_jucp_log_files(jucp_txt_dir):
    """ 同时兼容旧版 jucp*.txt 和新版 OpenPCDet log_eval_*.txt 日志 """
    patterns = ['jucp*.txt', 'log_eval_*.txt']
    files = []
    seen = set()
    for pattern in patterns:
        for file_path in glob.glob(os.path.join(jucp_txt_dir, pattern)):
            if file_path in seen:
                continue
            seen.add(file_path)
            files.append(file_path)
    return files

def get_closest_bpp(target_scale, csv_df):
    scales = csv_df['posQuantscale'].values
    idx = np.argmin(np.abs(scales - target_scale))
    return csv_df['bpp'].iloc[idx]

def get_adaptive_ylim(*curves):
    values = []
    for curve in curves:
        values.extend([v for v in curve if not pd.isna(v)])

    if not values:
        return 0, 1

    y_min = min(values)
    y_max = max(values)

    if np.isclose(y_min, y_max):
        margin = max(abs(y_max) * 0.1, 1.0)
    else:
        margin = (y_max - y_min) * 0.1

    bottom = max(0, y_min - margin)
    top = y_max + margin
    return bottom, top

def main():
    args = parse_args()

    # ==================== 1. 处理 Baseline 数据 ====================
    use_baseline = bool(args.gpcc_log and args.gpcc_csv)
    gpcc_bpps, gpcc_cars, gpcc_peds, gpcc_cycs = [], [], [], []

    if use_baseline:
        print("[*] Parsing Baseline Data...")
        gpcc_map = extract_map_from_log(args.gpcc_log, is_split=False)
        gpcc_df = pd.read_csv(args.gpcc_csv)
        gpcc_pts = []
        for scale, aps in gpcc_map.items():
            if not aps: continue
            bpp = get_closest_bpp(scale, gpcc_df)
            gpcc_pts.append((bpp, aps.get('Car', np.nan), aps.get('Pedestrian', np.nan), aps.get('Cyclist', np.nan)))
        gpcc_pts.sort(key=lambda x: x[0])
        
        gpcc_bpps = [p[0] for p in gpcc_pts]
        gpcc_cars, gpcc_peds, gpcc_cycs = [p[1] for p in gpcc_pts], [p[2] for p in gpcc_pts], [p[3] for p in gpcc_pts]
    else:
        print("[*] Skipping Baseline Data: --gpcc_log and --gpcc_csv were not both provided.")

    # ==================== 2. 处理 Split 数据 ====================
    print("[*] Parsing Semantic Split Data...")
    split_map = extract_map_from_log(args.split_log, is_split=True)
    split_df = pd.read_csv(args.split_csv)
    
    # 建立 combo_id 到 bpp 的映射字典，后续 JUCP 也要用
    split_bpp_dict = dict(zip(split_df['combo_id'], split_df['bpp']))
    
    split_pts = []
    for _, row in split_df.iterrows():
        if 'combo_id' not in row: continue
        cid = int(row['combo_id'])
        bpp = row['bpp']
        aps = split_map.get(cid)
        if not aps: continue
        split_pts.append((bpp, aps.get('Car', np.nan), aps.get('Pedestrian', np.nan), aps.get('Cyclist', np.nan)))
    split_pts.sort(key=lambda x: x[0])

    split_bpps = [p[0] for p in split_pts]
    split_cars, split_peds, split_cycs = [p[1] for p in split_pts], [p[2] for p in split_pts], [p[3] for p in split_pts]

    # ==================== 3. 处理 JUCP 数据 ====================
    print("[*] Parsing JUCP Method Data...")
    jucp_pts = []
    jucp_txt_files = find_jucp_log_files(args.jucp_txt_dir)
    
    if not jucp_txt_files:
        print(f"[!] Warning: No JUCP log files found in {args.jucp_txt_dir}")
    
    for txt_file in jucp_txt_files:
        base_name = Path(txt_file).stem  # 例如：jucp0 或 log_eval_20260601-123000
        csv_name = extract_jucp_csv_name(txt_file)
        csv_file = os.path.join(args.jucp_csv_dir, csv_name) if csv_name else os.path.join(args.jucp_csv_dir, f"{base_name}.csv")
        
        if not os.path.exists(csv_file):
            print(f"[-] Skipping {base_name}: CSV file not found.")
            continue
            
        # 3.1 提取该点(该日志)的 mAP
        aps = extract_single_map(txt_file)
        if not aps: continue
        
        # 3.2 提取每个点的标签分布并计算加权平均 BPP
        df_labels = pd.read_csv(csv_file)
        total_frames = len(df_labels)
        if total_frames == 0: continue
        
        total_bpp = 0.0
        label_counts = df_labels['jucp_label'].value_counts()
        
        for label_val, count in label_counts.items():
            # JUCP Label (0~5) 映射到 Split Combo ID (5~0)
            mapped_combo_id = 5 - int(label_val)
            
            if mapped_combo_id in split_bpp_dict:
                bpp_for_this_label = split_bpp_dict[mapped_combo_id]
                total_bpp += bpp_for_this_label * count
            else:
                print(f"[!] Warning: Mapped combo_id {mapped_combo_id} not found in split_csv!")
        
        avg_bpp = total_bpp / total_frames
        jucp_pts.append((avg_bpp, aps.get('Car', np.nan), aps.get('Pedestrian', np.nan), aps.get('Cyclist', np.nan)))

    jucp_pts.sort(key=lambda x: x[0])
    jucp_bpps = [p[0] for p in jucp_pts]
    jucp_cars, jucp_peds, jucp_cycs = [p[1] for p in jucp_pts], [p[2] for p in jucp_pts], [p[3] for p in jucp_pts]


    # ==================== 4. 分别绘制并保存三张图表 ====================
    print(f"[*] Plotting Individual Class Data...")
    
    method_colors = {'Split': '#1f77b4', 'Baseline': '#ff7f0e', 'JUCP': '#2ca02c'}
    
    classes = ['Car', 'Pedestrian', 'Cyclist']
    
    # 构造方便索引的数据字典
    data_dict = {
        'Car': {'split': split_cars, 'gpcc': gpcc_cars, 'jucp': jucp_cars},
        'Pedestrian': {'split': split_peds, 'gpcc': gpcc_peds, 'jucp': jucp_peds},
        'Cyclist': {'split': split_cycs, 'gpcc': gpcc_cycs, 'jucp': jucp_cycs}
    }

    base_out_path = Path(args.out)

    for cls in classes:
        plt.figure(figsize=(10, 7))
        
        # 绘制 Split
        plt.plot(split_bpps, data_dict[cls]['split'], color=method_colors['Split'], 
                 marker='s', linestyle='-.', markersize=6, linewidth=2.5, label='Semantic-Split')
                 
        # 绘制 Baseline
        if use_baseline:
            plt.plot(gpcc_bpps, data_dict[cls]['gpcc'], color=method_colors['Baseline'], 
                     marker='X', linestyle='--', markersize=8, linewidth=2, alpha=0.7, label='Baseline GPCC')
                 
        # 绘制 JUCP
        if jucp_bpps:
            plt.plot(jucp_bpps, data_dict[cls]['jucp'], color=method_colors['JUCP'], 
                     marker='o', markersize=6, linewidth=2.5, label='JUCP')

        plt.xlabel('Bitrate / Bits Per Point (BPP)', fontsize=14)
        plt.ylabel(f'{cls} 3D AP (Moderate Difficulty) [%]', fontsize=14)
        title_methods = 'Baseline vs. Split vs. JUCP' if use_baseline else 'Split vs. JUCP'
        plt.title(f'Performance Comparison: {title_methods} ({cls})', fontsize=16, pad=15)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right', fontsize=12) 
        
        y_curves = [data_dict[cls]['split'], data_dict[cls]['jucp']]
        if use_baseline:
            y_curves.append(data_dict[cls]['gpcc'])
        y_bottom, y_top = get_adaptive_ylim(*y_curves)
        plt.ylim(bottom=y_bottom, top=y_top)
        
        plt.tight_layout()
        
        # 动态生成图片文件名
        out_name = f"{base_out_path.stem}_{cls}{base_out_path.suffix}"
        final_out_path = base_out_path.parent / out_name
        
        plt.savefig(str(final_out_path), dpi=300)
        plt.close()
        print(f"[+] 成功保存 {cls} 类别的曲线图至: {final_out_path}")

    print("\n[+] 三个目标的单独对比曲线图已全部生成！")

if __name__ == '__main__':
    main()
