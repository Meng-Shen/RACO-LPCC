import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # ==================== 硬编码数据录入 ====================
    # 数据来源于提供的表格图片，对应同一组 BPP 序列下的不同指标
    
    # Split-LPCC 数据
    split_data = {
        'compress_time': [0.427, 0.407, 0.385, 0.286, 0.277, 0.264, 0.248],
        'enc_time':      [0.478, 0.458, 0.436, 0.337, 0.328, 0.315, 0.299],
        'dec_time':      [0.153, 0.136, 0.119, 0.078, 0.071, 0.062, 0.052],
        'Car':           [79.99, 79.61, 79.03, 80.12, 79.73, 72.08, 33.28],
        'Cyclist':       [68.16, 65.28, 62.29, 64.98, 60.69, 45.28, 13.24],
        'Pedestrian':    [50.30, 50.37, 49.59, 47.53, 39.97, 13.24,  1.02]
    }

    # Baseline G-PCC 数据
    gpcc_data = {
        'compress_time': [0.919, 0.774, 0.613, 0.515, 0.420, 0.369, 0.320],
        'enc_time':      [0.919, 0.774, 0.613, 0.515, 0.420, 0.369, 0.320],
        'dec_time':      [0.477, 0.380, 0.262, 0.193, 0.124, 0.090, 0.060],
        'Car':           [81.87, 81.18, 76.69, 67.51, 39.59, 15.73,  1.98],
        'Cyclist':       [67.74, 63.43, 48.50, 32.03, 13.60,  3.23,  0.00],
        'Pedestrian':    [49.34, 34.61, 11.58,  3.57,  1.40,  0.60,  0.03]
    }

    # ==================== 绘图配置 ====================
    classes = ['Car', 'Cyclist', 'Pedestrian']
    metrics = {
        'compress_time': 'Compression Time (s)',
        'enc_time': 'Total Encoding Time (s)',
        'dec_time': 'Decoding Time (s)'
    }
    
    method_colors = {'Split': 'red', 'Baseline': 'blue'}
    out_dir = './output_plots'
    os.makedirs(out_dir, exist_ok=True)

    # ==================== 遍历生成图表 ====================
    for metric_key, metric_label in metrics.items():
        for cls in classes:
            plt.figure(figsize=(10, 6))

            # 1. 处理 Split-LPCC 数据 (根据 X 轴从小到大排序防止曲线乱穿)
            split_x = split_data[metric_key]
            split_y = split_data[cls]
            split_sorted = sorted(zip(split_x, split_y))
            split_x_sorted = [val[0] for val in split_sorted]
            split_y_sorted = [val[1] for val in split_sorted]

            # 2. 处理 Baseline G-PCC 数据
            gpcc_x = gpcc_data[metric_key]
            gpcc_y = gpcc_data[cls]
            gpcc_sorted = sorted(zip(gpcc_x, gpcc_y))
            gpcc_x_sorted = [val[0] for val in gpcc_sorted]
            gpcc_y_sorted = [val[1] for val in gpcc_sorted]

            # 3. 绘制曲线
            plt.plot(split_x_sorted, split_y_sorted, color=method_colors['Split'], 
                     marker='s', linestyle='-.', markersize=6, linewidth=2.5, label='Split-LPCC')
                     
            plt.plot(gpcc_x_sorted, gpcc_y_sorted, color=method_colors['Baseline'], 
                     marker='X', linestyle='--', markersize=8, linewidth=2, alpha=0.7, label='Baseline G-PCC')

            # 4. 图表修饰
            plt.xlabel(metric_label, fontsize=14)
            plt.ylabel(f'{cls} 3D AP (Moderate) [%]', fontsize=14)
            plt.title(f'Performance Comparison: AP vs {metric_label.split(" (")[0]} ({cls})', fontsize=16, pad=15)
            
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='lower right', fontsize=12) 
            plt.ylim(bottom=0)
            plt.tight_layout()
            
            # 5. 保存图表
            out_name = os.path.join(out_dir, f'Compare_AP_{metric_key}_{cls}.png')
            plt.savefig(out_name, dpi=300, facecolor='white')
            plt.close()
            print(f"✅ Saved plot to {out_name}")

if __name__ == '__main__':
    main()