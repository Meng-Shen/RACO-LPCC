import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Plot mAP vs BPP curve from log and CSV files.")
    parser.add_argument('--log', type=str, required=True, help='Path to the test_pos.py log file')
    parser.add_argument('--csv', type=str, required=True, help='Path to the gpcc_average_results.csv file')
    parser.add_argument('--out', type=str, default='mAP_bpp_curve.png', help='Output image path')
    return parser.parse_args()

def extract_map_from_log(log_path):
    """
    使用严格的状态机正则表达式，精准提取 R40 标准下的 Moderate 3D AP
    - Car: AP_R40@0.70, 0.50, 0.50
    - Pedestrian: AP_R40@0.50, 0.50, 0.50
    - Cyclist: AP_R40@0.50, 0.50, 0.50
    """
    map_data = {}
    current_scale = None
    target_class = None

    with open(log_path, 'r') as f:
        for line in f:
            # 1. 匹配当前正在评测的量化步长 Scale
            m_scale = re.search(r'Start Evaluation for Scale:\s*([0-9\.]+)', line)
            if m_scale:
                current_scale = float(m_scale.group(1))
                if current_scale not in map_data:
                    map_data[current_scale] = {}
                continue
                
            if current_scale is None:
                continue

            # 2. 【核心修复】：极度严格地匹配带有 0.50, 0.50 后缀的 AP_R40 行
            
            # 匹配: "Car AP_R40@0.70, 0.50, 0.50:"
            m_car = re.search(r'Car\s+AP_R40@0\.70,\s*0\.50,\s*0\.50', line)
            if m_car:
                target_class = 'Car'
                continue
                
            # 匹配: "Pedestrian AP_R40@0.50, 0.50, 0.50:" 或 "Cyclist AP_R40@0..."
            m_other = re.search(r'(Pedestrian|Cyclist)\s+AP_R40@0\.50,\s*0\.50,\s*0\.50', line)
            if m_other:
                target_class = m_other.group(1)
                continue

            # 3. 提取紧接着的 '3d AP' 里面的 Moderate 难度得分
            if target_class is not None:
                # 匹配格式: "3d   AP: 90.00, 80.00, 70.00" -> 提取 Easy, Moderate, Hard
                m_3d = re.search(r'3d\s+AP:\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)', line)
                if m_3d:
                    moderate_ap = float(m_3d.group(2)) # 取第二个值 (Moderate)
                    map_data[current_scale][target_class] = moderate_ap
                    target_class = None # 提取成功后重置，防止误捕获其他指标

    return map_data

def get_closest_bpp(target_scale, csv_df):
    """防止浮点数精度误差，寻找 CSV 中最接近的量化步长对应的 bpp"""
    scales = csv_df['posQuantscale'].values
    idx = np.argmin(np.abs(scales - target_scale))
    return csv_df['bpp'].iloc[idx]

def main():
    args = parse_args()

    if not os.path.exists(args.log):
        print(f"Error: Log file not found at {args.log}")
        return
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found at {args.csv}")
        return

    print("Parsing Log File...")
    map_data = extract_map_from_log(args.log)
    
    print("Parsing CSV File...")
    bpp_df = pd.read_csv(args.csv)

    # 汇总对齐数据
    points = []
    for scale, aps in map_data.items():
        if not aps:
            continue # 如果这个 scale 没有任何 AP 数据，跳过
            
        bpp = get_closest_bpp(scale, bpp_df)
        car_ap = aps.get('Car', np.nan)
        ped_ap = aps.get('Pedestrian', np.nan)
        cyc_ap = aps.get('Cyclist', np.nan)
        
        points.append((bpp, car_ap, ped_ap, cyc_ap, scale))

    if not points:
        print("Error: No matching AP data found in the log file. Please check the log format.")
        return

    # 按 bpp 从小到大排序 (即从高压缩率到低压缩率)
    points.sort(key=lambda x: x[0])

    bpps = [p[0] for p in points]
    car_aps = [p[1] for p in points]
    ped_aps = [p[2] for p in points]
    cyc_aps = [p[3] for p in points]
    scales = [p[4] for p in points]

    print("\n========= Extracted Data (AP_R40 Moderate) =========")
    print(f"{'Scale':<10} | {'BPP':<10} | {'Car AP':<10} | {'Ped AP':<10} | {'Cyc AP':<10}")
    for i in range(len(bpps)):
        print(f"{scales[i]:<10.6f} | {bpps[i]:<10.4f} | {car_aps[i]:<10.2f} | {ped_aps[i]:<10.2f} | {cyc_aps[i]:<10.2f}")

    # ================= 绘图 =================
    plt.figure(figsize=(10, 6))
    
    # 绘制三条曲线，更新图例
    plt.plot(bpps, car_aps, marker='o', markersize=6, linewidth=2, label='Car (AP_R40@0.70,0.50,0.50)')
    plt.plot(bpps, ped_aps, marker='s', markersize=6, linewidth=2, label='Pedestrian (AP_R40@0.50,0.50,0.50)')
    plt.plot(bpps, cyc_aps, marker='^', markersize=6, linewidth=2, label='Cyclist (AP_R40@0.50,0.50,0.50)')

    # 设置图表格式
    plt.xlabel('Bitrate / Bits Per Point (BPP)', fontsize=12)
    plt.ylabel('3D mAP (Moderate Difficulty) [%]', fontsize=12)
    plt.title('Point Cloud Compression Rate-Distortion (mAP vs. BPP)', fontsize=14, pad=15)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=10)
    
    # 限制 y 轴最高为 105 留出顶部空间
    plt.ylim(bottom=0, top=105)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"\n[+] Successfully saved the plot to: {args.out}")

if __name__ == '__main__':
    main()