import os
import glob
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Calculate average BPP and Encoding Time from individual CSVs.")
    parser.add_argument('--results_dir', type=str, default='./results_gpcc', help='Directory containing the individual CSV files')
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: 找不到目录 {args.results_dir}")
        return

    # 查找目录下所有的 .csv 文件
    search_pattern = os.path.join(args.results_dir, '*.csv')
    all_csv_files = glob.glob(search_pattern)

    # 排除可能已经生成的汇总文件，防止重复计算
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

    # 将所有单个 dataframe 拼接成一个大表
    all_results_df = pd.concat(df_list, ignore_index=True)

    # 保存完整的明细大表
    detail_csv = os.path.join(args.results_dir, 'gpcc_all_details.csv')
    all_results_df.to_csv(detail_csv, index=False)
    print(f"[*] 完整明细已保存至: {detail_csv}")

    # 按量化步长 (posQuantscale) 分组并求平均值
    mean_results = all_results_df.groupby('posQuantscale').mean(numeric_only=True).reset_index()

    # 按照 posQuantscale 从大到小排序 (1.0 -> 0.001953)
    mean_results = mean_results.sort_values(by='posQuantscale', ascending=False)

    # 保存平均值表
    avg_csv = os.path.join(args.results_dir, 'gpcc_average_results.csv')
    mean_results.to_csv(avg_csv, index=False)
    print(f"[*] 平均结果已保存至: {avg_csv}")

    print("\n==================== 最终平均结果 (Final Average Results) ====================")
    # 格式化打印，方便在终端查看
    print(mean_results.to_string(index=False))
    print("==============================================================================")


if __name__ == '__main__':
    main()