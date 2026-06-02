import pandas as pd
import argparse
import re


def detect_max_level(df):
    """
    从 AP CSV 的列名中自动检测最大 L 编号。
    例如存在 L0_Car_AP ... L6_Cyc_AP，则返回 6。
    """
    levels = []
    for col in df.columns:
        m = re.match(r"L(\d+)_Car_AP$", col)
        if m:
            levels.append(int(m.group(1)))
    if not levels:
        raise ValueError("Cannot detect AP levels from CSV columns. Expected columns like L0_Car_AP")
    return max(levels)


def main():
    parser = argparse.ArgumentParser(description='Final JUCP Decision based on Global AP')
    parser.add_argument('--ap_csv', type=str, default='jucp_ap_sensitivity.csv', help='Input CSV file with AP metrics')
    parser.add_argument('--out_csv', type=str, default='final_jucp_labels.csv', help='Final output CSV with labels')
    parser.add_argument('--car_threshold', type=float, default=0.0045, help='Allowed AP drop threshold for Car')
    parser.add_argument('--ped_threshold', type=float, default=0.05, help='Allowed AP drop threshold for Pedestrian')
    parser.add_argument('--cyc_threshold', type=float, default=0.075, help='Allowed AP drop threshold for Cyclist')
    args = parser.parse_args()

    print(f"Loading AP sensitivity data from {args.ap_csv}...")
    print(
        "JUCP thresholds: "
        f"Car={args.car_threshold}, "
        f"Pedestrian={args.ped_threshold}, "
        f"Cyclist={args.cyc_threshold}"
    )

    try:
        df = pd.read_csv(args.ap_csv)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {args.ap_csv}，请检查文件名是否正确。")
        return

    if 'frame_id' in df.columns:
        df['frame_id'] = df['frame_id'].astype(str).str.zfill(6)

    max_level = detect_max_level(df)
    print(f"Detected AP levels: L0 ~ L{max_level}")

    required_cols = [
        'L0_Car_AP', 'L0_Ped_AP', 'L0_Cyc_AP',
    ]
    for l in range(1, max_level + 1):
        required_cols.extend([f'L{l}_Car_AP', f'L{l}_Ped_AP', f'L{l}_Cyc_AP'])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"AP CSV missing required columns: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    results = []

    for _, row in df.iterrows():
        frame_id = row['frame_id']

        baseline_sum = row['L0_Car_AP'] + row['L0_Ped_AP'] + row['L0_Cyc_AP']

        best_label = 0
        chosen_sum = baseline_sum
        reason = "Fallback to L0 (Baseline)"

        # 从压缩最狠的 Label 开始，倒序往上尝试。
        # Label 0 是最高码率/基准；Label 越大表示压缩越狠。
        for l in range(max_level, 0, -1):
            cur_sum = row[f'L{l}_Car_AP'] + row[f'L{l}_Ped_AP'] + row[f'L{l}_Cyc_AP']

            if (
                row[f'L{l}_Car_AP'] >= row['L0_Car_AP'] - args.car_threshold and
                row[f'L{l}_Ped_AP'] >= row['L0_Ped_AP'] - args.ped_threshold and
                row[f'L{l}_Cyc_AP'] >= row['L0_Cyc_AP'] - args.cyc_threshold
            ):
                best_label = l
                chosen_sum = cur_sum
                reason = (
                    f"Passed at L{l} "
                    f"with thresholds "
                    f"Car={args.car_threshold}, "
                    f"Ped={args.ped_threshold}, "
                    f"Cyc={args.cyc_threshold}; "
                    f"SumDiff={cur_sum - baseline_sum:+.4f}"
                )
                break

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

    print("📊 JUCP 挡位分布统计:")
    distribution = out_df['jucp_label'].value_counts().sort_index(ascending=False)
    for label, count in distribution.items():
        if label == max_level:
            desc = "(压缩最狠)"
        elif label == 0:
            desc = "(不压缩/基准)"
        else:
            desc = ""
        print(f"  Label {label} {desc}: {count} 帧")


if __name__ == "__main__":
    main()
