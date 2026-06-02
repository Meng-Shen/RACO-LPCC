import math
import numpy
import numpy as np

def bdrate(metric_set1, metric_set2):
    """
    BJONTEGAARD    Bjontegaard metric calculation
    Bjontegaard's metric allows to compute the average % saving in bitrate
    (or time in this case) between two rate-distortion curves [1].

    rate1,psnr1 - RD points for curve 1 (Baseline / G-PCC)
    rate2,psnr2 - RD points for curve 2 (Proposed / Split-LPCC)
    """
    rate1 = [x[0] for x in metric_set1]
    psnr1 = [x[1] for x in metric_set1]
    rate2 = [x[0] for x in metric_set2]
    psnr2 = [x[1] for x in metric_set2]

    log_rate1 = list(map(math.log, rate1))
    log_rate2 = list(map(math.log, rate2))

    # 注意：将拟合阶数从 7 改为 3 (Cubic poly fit)。
    # 7个数据点使用7阶多项式会引发严重的过拟合和数值震荡(Runge's phenomenon)。
    poly1 = numpy.polyfit(psnr1, log_rate1, 3)
    poly2 = numpy.polyfit(psnr2, log_rate2, 3)

    # Integration interval.
    min_int = max([min(psnr1), min(psnr2)])
    max_int = min([max(psnr1), max(psnr2)])

    # find integral
    p_int1 = numpy.polyint(poly1)
    p_int2 = numpy.polyint(poly2)

    # Calculate the integrated value over the interval we care about.
    int1 = numpy.polyval(p_int1, max_int) - numpy.polyval(p_int1, min_int)
    int2 = numpy.polyval(p_int2, max_int) - numpy.polyval(p_int2, min_int)

    # Calculate the average improvement.
    avg_exp_diff = (int2 - int1) / (max_int - min_int)

    # In really bad formed data the exponent can grow too large.
    # clamp it.
    if avg_exp_diff > 200:
        avg_exp_diff = 200

    # Convert to a percentage.
    avg_diff = (math.exp(avg_exp_diff) - 1) * 100

    return avg_diff

def robust_bdrate(metric_set1, metric_set2):
    """
    使用分段线性插值 + 梯形积分的鲁棒版 BD-Rate 计算。
    完美解决 AP 扎堆、非单调导致的“多项式爆炸”问题。
    """
    # 1. 必须先按照 AP (Y轴) 从小到大排序，这是积分的X轴
    s1 = sorted(metric_set1, key=lambda x: x[1])
    s2 = sorted(metric_set2, key=lambda x: x[1])

    rate1 = np.array([x[0] for x in s1])
    psnr1 = np.array([x[1] for x in s1])
    rate2 = np.array([x[0] for x in s2])
    psnr2 = np.array([x[1] for x in s2])

    log_rate1 = np.log(rate1)
    log_rate2 = np.log(rate2)

    # 2. 找到两条曲线在 AP 上的公共重叠区间
    min_int = max(min(psnr1), min(psnr2))
    max_int = min(max(psnr1), max(psnr2))

    if min_int >= max_int:
        return float('nan')

    # 3. 在重叠区间内生成 1000 个密集网格点
    grid = np.linspace(min_int, max_int, 1000)

    # 4. 线性插值：打败多项式震荡的核心（允许非单调数据的折线连接）
    interp_log_r1 = np.interp(grid, psnr1, log_rate1)
    interp_log_r2 = np.interp(grid, psnr2, log_rate2)

    # 5. 梯形法则求面积积分
    int1 = np.trapz(interp_log_r1, grid)
    int2 = np.trapz(interp_log_r2, grid)

    # 计算平均指数差
    avg_exp_diff = (int2 - int1) / (max_int - min_int)

    # 限制范围防止 math.exp 溢出报错
    avg_exp_diff = max(min(avg_exp_diff, 200), -200)

    # 换算回百分比
    avg_diff = (math.exp(avg_exp_diff) - 1) * 100

    return avg_diff


def main():
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

    classes = ['Car', 'Cyclist', 'Pedestrian']
    time_metrics = {
        'enc_time': 'BD-enctime',
        'dec_time': 'BD-dectime'
    }

    print("=== Bjontegaard Delta Metric: Split-LPCC vs G-PCC ===")
    print("*(负数表示节省的时间百分比)*\n")

    for t_key, t_label in time_metrics.items():
        print(f"--- {t_label} ---")
        for cls in classes:
            # 构建 G-PCC 的 metric_set1: 格式为 [(rate1, psnr1), (rate2, psnr2), ...]
            # 这里 rate 就是 time，psnr 就是 AP
            metric_set1 = list(zip(gpcc_data[t_key], gpcc_data[cls]))
            
            # 构建 Split-LPCC 的 metric_set2
            metric_set2 = list(zip(split_data[t_key], split_data[cls]))
            
            # 计算 BD-Rate，set1为基准，set2为对比项
            bd_value = robust_bdrate(metric_set1, metric_set2)
            
            print(f"{cls:12s}: {bd_value:>7.2f} %")
        print()

if __name__ == "__main__":
    main()