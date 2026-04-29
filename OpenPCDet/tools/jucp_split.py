import pandas as pd
import argparse
import math

def main():
    parser = argparse.ArgumentParser(description='Final JUCP Decision based on Global AP')
    # 填入你上一步跑出来的 CSV 文件名
    parser.add_argument('--ap_csv', type=str, default='jucp_ap_sensitivity.csv', help='Input CSV file with AP metrics')
    parser.add_argument('--out_csv', type=str, default='final_jucp_labels.csv', help='Final output CSV with labels')
    args = parser.parse_args()

    print(f"Loading AP sensitivity data from {args.ap_csv}...")
    try:
        df = pd.read_csv(args.ap_csv)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {args.ap_csv}，请检查文件名是否正确。")
        return
    
    # 确保 frame_id 格式统一（补充前导 0）
    if 'frame_id' in df.columns:
        df['frame_id'] = df['frame_id'].astype(str).str.zfill(6)
    
    results = []
    
    for idx, row in df.iterrows():
        frame_id = row['frame_id']
        
        # 1. 计算最高码率 (L0) 作为基准 AP 总和
        baseline_sum = row['L0_Car_AP'] + row['L0_Ped_AP'] + row['L0_Cyc_AP']
        
        best_label = 0  # 默认保底为最大码率 L0
        chosen_sum = baseline_sum
        reason = "Fallback to L0 (Baseline)"
        
        # 2. 从压缩最狠的 L5 开始，倒序往上尝试到 L1
        for l in range(5, 0, -1):
            cur_sum = row[f'L{l}_Car_AP'] + row[f'L{l}_Ped_AP'] + row[f'L{l}_Cyc_AP']
            
            # 使用 1e-6 的极小容差防止浮点数精度截断导致的误判
            # 判断条件：当前码率的 AP 总和 >= 基准 AP 总和
            if row[f'L{l}_Car_AP']>=row['L0_Car_AP']-0.0045 and row[f'L{l}_Ped_AP']>=row['L0_Ped_AP']-0.05 and row[f'L{l}_Cyc_AP']>=row['L0_Cyc_AP']-0.075:
                best_label = l
                chosen_sum = cur_sum
                reason = f"Passed at L{l} (Diff: {cur_sum - baseline_sum:+.4f})"
                break  # 找到第一个满足条件的最强压缩挡位，立刻停止搜索
                
        results.append({
            'frame_id': frame_id,
            'jucp_label': best_label,
            'baseline_ap_sum': round(baseline_sum, 6),
            'chosen_ap_sum': round(chosen_sum, 6),
            'reason': reason
        })
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_csv, index=False)
    
    print(f"\n✨ 筛选完毕！成功处理了 {len(out_df)} 帧。")
    print(f"最终判定结果已保存至: {args.out_csv}\n")
    
    # 打印最终筛选出的码率挡位分布
    print("📊 JUCP 挡位分布统计:")
    distribution = out_df['jucp_label'].value_counts().sort_index(ascending=False)
    for label, count in distribution.items():
        if label == 5:
            desc = "(压缩最狠)"
        elif label == 0:
            desc = "(不压缩/基准)"
        else:
            desc = ""
        print(f"  Label {label} {desc}: {count} 帧")

if __name__ == "__main__":
    main()