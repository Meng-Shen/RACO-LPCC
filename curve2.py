import os
import glob
import argparse
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

def extract_map_from_log(log_path):
    """
    使用严格的状态机正则表达式，精准提取 R40 标准下的 Moderate 3D AP
    - Car: AP_R40@0.70, 0.50, 0.50
    - Pedestrian: AP_R40@0.50, 0.50, 0.50
    - Cyclist: AP_R40@0.50, 0.50, 0.50
    """
    map_data = {}
    current_combo = None
    target_class = None

    with open(log_path, 'r') as f:
        for line in f:
            # 1. 匹配当前正在评测的 Combo ID
            m_combo = re.search(r'Combo\s+(\d+)', line)
            if m_combo:
                current_combo = int(m_combo.group(1))
                if current_combo not in map_data:
                    map_data[current_combo] = {}
                continue

            if current_combo is None:
                continue

            # 2. 【核心修复】：极度严格地匹配带有 0.50, 0.50 后缀的 AP_R40 行
            
            # 匹配: "Car AP_R40@0.70, 0.50, 0.50:"
            m_car = re.search(r'(Car)\s+AP_R40@0\.70,\s*0\.50,\s*0\.50', line)
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
                # 匹配格式: "3d   AP:98.3521, 94.0958, 93.7405" -> 提取 Easy, Moderate, Hard
                m_3d = re.search(r'3d\s+AP:\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)', line)
                if m_3d:
                    moderate_ap = float(m_3d.group(2)) # 取第二个值 (Moderate)
                    map_data[current_combo][target_class] = moderate_ap
                    target_class = None # 提取成功后重置，防止误捕获其他指标

    return map_data


def main():
    parser = argparse.ArgumentParser(description="Calculate average BPP/Time and Plot mAP vs BPP Curve.")
    parser.add_argument('--results_dir', type=str, default='./results_gpcc', help='Directory containing the individual CSV files')
    parser.add_argument('--log', type=str, default=None, help='Path to the test_split.py log file for plotting (e.g., log_eval_split_xxx.txt)')
    parser.add_argument('--out_img', type=str, default='mAP_bpp_curve.png', help='Output image filename')
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: 找不到目录 {args.results_dir}")
        return

    # ---------------- 1. 查找并处理 CSV 文件 ----------------
    search_pattern = os.path.join(args.results_dir, '*.csv')
    all_csv_files = glob.glob(search_pattern)

    exclude_files = ['gpcc_all_details.csv', 'gpcc_average_results.csv']
    valid_csv_files = [f for f in all_csv_files if os.path.basename(f) not in exclude_files]

    if not valid_csv_files:
        print(f"Error: 在 {args.results_dir} 下没有找到有效的单帧测试 CSV 文件。")
        return

    print(f"找到了 {len(valid_csv_files)} 个单帧结果文件，正在进行汇总计算...")

    df_list = []
    for f in valid_csv_files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                df_list.append(df)
        except Exception as e:
            print(f"读取文件 {f} 时出错: {e}")

    if not df_list:
        print("Error: 所有文件均为空或无法读取。")
        return

    all_results_df = pd.concat(df_list, ignore_index=True)

    # 保存完整的明细大表
    detail_csv = os.path.join(args.results_dir, 'gpcc_all_details.csv')
    all_results_df.to_csv(detail_csv, index=False)
    print(f"[*] 完整明细已保存至: {detail_csv}")

    # 按 combo_id 分组并求平均值
    if 'combo_id' in all_results_df.columns:
        mean_results = all_results_df.groupby('combo_id').mean(numeric_only=True).reset_index()
        mean_results = mean_results.sort_values(by='combo_id', ascending=False)
    else:
        print("[Warning] CSV 中没有找到 combo_id 列，可能不是 test_split.py 跑出的数据。")
        mean_results = all_results_df.groupby('posQuantscale').mean(numeric_only=True).reset_index()
        mean_results = mean_results.sort_values(by='posQuantscale', ascending=False)

    avg_csv = os.path.join(args.results_dir, 'gpcc_average_results.csv')
    mean_results.to_csv(avg_csv, index=False)
    print(f"[*] 平均结果已保存至: {avg_csv}")

    print("\n==================== 最终平均结果 (Final Average Results) ====================")
    print(mean_results.to_string(index=False))
    print("==============================================================================\n")

    # ---------------- 2. 画图逻辑 (如果有 --log 参数) ----------------
    if args.log:
        if not os.path.exists(args.log):
            print(f"[Error] 找不到 Log 文件: {args.log}")
            return

        print(f"[*] 正在解析 Log 文件: {args.log}")
        map_data = extract_map_from_log(args.log)

        if not map_data:
            print("[Error] 未能在 Log 文件中匹配到有效的 AP 数据，请检查 Log 格式！")
            return

        points = []
        for _, row in mean_results.iterrows():
            if 'combo_id' not in row:
                continue
                
            combo_id = int(row['combo_id'])
            bpp = row['bpp']
            
            aps = map_data.get(combo_id)
            if aps is None:
                continue
                
            car_ap = aps.get('Car', np.nan)
            ped_ap = aps.get('Pedestrian', np.nan)
            cyc_ap = aps.get('Cyclist', np.nan)

            points.append((bpp, car_ap, ped_ap, cyc_ap, combo_id))

        if not points:
            print("[Error] CSV 数据与 Log 数据中的 combo_id 无法对齐！")
            return

        # 按 BPP 从小到大排序作图
        points.sort(key=lambda x: x[0])
        
        bpps = [p[0] for p in points]
        car_aps = [p[1] for p in points]
        ped_aps = [p[2] for p in points]
        cyc_aps = [p[3] for p in points]
        combos = [p[4] for p in points]

        print("========= Plotting Data (AP_R40 Moderate) =========")
        print(f"{'Combo ID':<10} | {'BPP':<10} | {'Car AP':<10} | {'Ped AP':<10} | {'Cyc AP':<10}")
        for i in range(len(bpps)):
            print(f"{combos[i]:<10} | {bpps[i]:<10.4f} | {car_aps[i]:<10.2f} | {ped_aps[i]:<10.2f} | {cyc_aps[i]:<10.2f}")

        # ======== 绘图 ========
        plt.figure(figsize=(10, 6))
        plt.plot(bpps, car_aps, marker='o', markersize=6, linewidth=2, label='Car (AP_R40@0.70,0.50,0.50)')
        plt.plot(bpps, ped_aps, marker='s', markersize=6, linewidth=2, label='Pedestrian (AP_R40@0.50,0.50,0.50)')
        plt.plot(bpps, cyc_aps, marker='^', markersize=6, linewidth=2, label='Cyclist (AP_R40@0.50,0.50,0.50)')

        plt.xlabel('Bitrate / Bits Per Point (BPP)', fontsize=12)
        plt.ylabel('3D mAP (Moderate Difficulty) [%]', fontsize=12)
        plt.title('Split Point Cloud Compression Rate-Distortion (mAP vs. BPP)', fontsize=14, pad=15)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='lower right', fontsize=10)
        plt.ylim(bottom=0, top=105)

        plt.tight_layout()
        out_img_path = os.path.join(args.results_dir, args.out_img)
        plt.savefig(out_img_path, dpi=300)
        print(f"\n[+] 曲线图已成功保存至: {out_img_path}")

if __name__ == '__main__':
    main()