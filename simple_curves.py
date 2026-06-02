import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Comparison of Baseline, Split, and JUCP GPCC.")
    # Baseline GPCC 的文件路径
    parser.add_argument('--gpcc_log', type=str, required=True, help='Path to Baseline GPCC log file')
    parser.add_argument('--gpcc_csv', type=str, required=True, help='Path to Baseline gpcc_average_results.csv')
    
    # Split GPCC 的文件路径
    parser.add_argument('--split_log', type=str, required=True, help='Path to Split test_split.py log file')
    parser.add_argument('--split_csv', type=str, required=True, help='Path to Split average_results.csv')
    
    # JUCP 的文件夹路径 (可选)
    parser.add_argument('--jucp_txt_dir', type=str, default=None, help='Dir containing jucp*.txt log files')
    parser.add_argument('--jucp_csv_dir', type=str, default=None, help='Dir containing jucp*.csv label files')
    
    # 输出图片路径
    parser.add_argument('--out', type=str, default='Compare_mAP_BPP.png', help='Base output image path')
    return parser.parse_args()

def extract_map_from_log(log_path, is_split=False):
    """ 
    提取常规日志中的 mAP 数据 (用于 Baseline)
    使用严格的 IoU 阈值进行匹配
    """
    map_dict = {'Car': [], 'Pedestrian': [], 'Cyclist': []}
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    target_class = None
    for i, line in enumerate(lines):
        # 1. 严格匹配 Car 的 IoU 阈值
        m_car = re.search(r'(Car)\s+AP_R40@0\.70,\s*0\.70,\s*0\.70', line)
        # 2. 严格匹配 Pedestrian/Cyclist 的 IoU 阈值 (兼容 0.25 和 0.50)
        m_other = re.search(r'(Pedestrian|Cyclist)\s+AP_R40@0\.50,\s*0\.50,\s*0\.50', line)
        
        if m_car:
            target_class = 'Car'
        elif m_other:
            target_class = m_other.group(1)
            
        if target_class is not None:
            # 向下寻找 3d AP 行
            for j in range(1, 6):
                if i + j < len(lines) and '3d   AP:' in lines[i+j]:
                    ap_str = lines[i+j].split('3d   AP:')[1]
                    moderate_ap = float(ap_str.split(',')[1].strip())
                    map_dict[target_class].append(moderate_ap)
                    target_class = None  # 提取成功后重置
                    break
    return map_dict

def parse_split_data(log_path, csv_path):
    """
    专门针对带有 Combo ID 的 split log 和 csv 进行联合解析
    通过 Combo ID 精确映射 BPP 和 mAP，防止数据错位，并使用严格正则提取
    """
    # 1. 解析 CSV 提取 bpp
    df = pd.read_csv(csv_path)
    bpp_dict = dict(zip(df['combo_id'], df['bpp']))
    
    # 2. 解析 TXT 日志提取 mAP
    map_dict = {'Car': {}, 'Pedestrian': {}, 'Cyclist': {}}
    current_combo = None
    target_class = None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        # 捕捉当前的 Combo 编号
        combo_match = re.search(r'Combo (\d+)', line)
        if combo_match:
            current_combo = int(combo_match.group(1))
            
        if current_combo is not None:
            # 严格匹配阈值
            m_car = re.search(r'(Car)\s+AP_R40@0\.70,\s*0\.70,\s*0\.70', line)
            m_other = re.search(r'(Pedestrian|Cyclist)\s+AP_R40@0\.50,\s*0\.50,\s*0\.50', line)
            
            if m_car:
                target_class = 'Car'
            elif m_other:
                target_class = m_other.group(1)
                
            if target_class is not None:
                # 向下寻找 3d AP 行
                for j in range(1, 6):
                    if i + j < len(lines) and '3d   AP:' in lines[i+j]:
                        ap_str = lines[i+j].split('3d   AP:')[1]
                        moderate_ap = float(ap_str.split(',')[1].strip())
                        map_dict[target_class][current_combo] = moderate_ap
                        target_class = None  # 提取成功后重置
                        break
                            
    # 3. 合并数据：找到在 csv 和 log 中都存在的 combo_id
    valid_combos = [c for c in bpp_dict.keys() if c in map_dict['Car']]
    
    split_bpps_raw = [bpp_dict[c] for c in valid_combos]
    split_maps_raw = {
        'Car': [map_dict['Car'].get(c, 0) for c in valid_combos],
        'Pedestrian': [map_dict['Pedestrian'].get(c, 0) for c in valid_combos],
        'Cyclist': [map_dict['Cyclist'].get(c, 0) for c in valid_combos]
    }
    
    # 4. 按 BPP 从小到大排序 (防止画出的曲线折返乱穿)
    sorted_indices = np.argsort(split_bpps_raw)
    
    split_bpps = np.array(split_bpps_raw)[sorted_indices].tolist()
    split_maps = {cls: np.array(split_maps_raw[cls])[sorted_indices].tolist() for cls in split_maps_raw}
        
    return split_bpps, split_maps

def main():
    args = parse_args()
    
    # === 解析 Baseline 数据 ===
    gpcc_df = pd.read_csv(args.gpcc_csv)
    gpcc_bpps_raw = gpcc_df['bpp'].tolist() if 'bpp' in gpcc_df.columns else []
    gpcc_maps_raw = extract_map_from_log(args.gpcc_log, is_split=False)
    
    # 对 Baseline 按 bpp 排序
    sorted_gpcc_indices = np.argsort(gpcc_bpps_raw)
    gpcc_bpps = np.array(gpcc_bpps_raw)[sorted_gpcc_indices].tolist()
    
    # === 解析 Split 数据 ===
    split_bpps, split_maps = parse_split_data(args.split_log, args.split_csv)
    
    # === 预留的 JUCP 数据（根据需要自行补充遍历逻辑） ===
    jucp_bpps = []
    jucp_maps = {'Car': [], 'Pedestrian': [], 'Cyclist': []}
    
    # === 整合绘图字典 ===
    classes = ['Car', 'Pedestrian', 'Cyclist']
    data_dict = {cls: {} for cls in classes}
    method_colors = {'Split': 'red', 'Baseline': 'blue', 'JUCP': 'green'}
    
    for cls in classes:
        # 准备数据
        data_dict[cls]['gpcc'] = np.array(gpcc_maps_raw[cls])[sorted_gpcc_indices].tolist() if gpcc_maps_raw[cls] else []
        data_dict[cls]['split'] = split_maps[cls] 
        data_dict[cls]['jucp'] = jucp_maps[cls]
        
        # ==================== 开始绘图 ====================
        plt.figure(figsize=(10, 6))
        
        # 绘制 Split 曲线
        if split_bpps and data_dict[cls]['split']:
            plt.plot(split_bpps, data_dict[cls]['split'], color=method_colors['Split'], 
                     marker='s', linestyle='-.', markersize=6, linewidth=2.5, label='Split-LPCC')
                 
        # 绘制 Baseline 曲线
        if gpcc_bpps and data_dict[cls]['gpcc']:
            plt.plot(gpcc_bpps, data_dict[cls]['gpcc'], color=method_colors['Baseline'], 
                     marker='X', linestyle='--', markersize=8, linewidth=2, alpha=0.7, label='Baseline G-PCC')
                 
        # 绘制 JUCP 曲线
        if jucp_bpps and data_dict[cls]['jucp']:
            plt.plot(jucp_bpps, data_dict[cls]['jucp'], color=method_colors['JUCP'], 
                     marker='o', markersize=6, linewidth=2.5, label='JUCP')

        # 图表修饰
        plt.xlabel('Bits Per Point (BPP)', fontsize=14)
        plt.ylabel(f'{cls} 3D AP (Moderate Difficulty) [%]', fontsize=14)
        plt.title(f'Performance Comparison: G-PCC vs. Split-LPCC({cls})', fontsize=16, pad=15)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right', fontsize=12) 
        
        # 核心修改：取消顶部硬编码限制，让 matplotlib 自动适应数据高度
        plt.ylim(bottom=0)
        
        plt.tight_layout()
        
        # 动态生成图片文件名 (如 Compare_mAP_BPP_Car.png)
        base_name, ext = os.path.splitext(args.out)
        out_name = f"{base_name}_{cls}{ext}"
        
        plt.savefig(out_name, dpi=300, facecolor='white')
        plt.close()
        print(f"✅ Saved plot to {out_name}")

if __name__ == '__main__':
    main()