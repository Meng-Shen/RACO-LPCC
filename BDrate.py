import math
import numpy as np

def robust_bdrate(metric_set1, metric_set2):

    s1 = sorted(metric_set1, key=lambda x: x[1])
    s2 = sorted(metric_set2, key=lambda x: x[1])

    rate1 = np.array([x[0] for x in s1])
    psnr1 = np.array([x[1] for x in s1])
    rate2 = np.array([x[0] for x in s2])
    psnr2 = np.array([x[1] for x in s2])

    log_rate1 = np.log(rate1)
    log_rate2 = np.log(rate2)

    # 找到两条曲线在 AP 上的公共重叠区间
    min_int = max(min(psnr1), min(psnr2))
    max_int = min(max(psnr1), max(psnr2))

    if min_int >= max_int:
        return float('nan')

    # 在重叠区间内生成 1000 个密集网格点
    grid = np.linspace(min_int, max_int, 1000)

    # 线性插值
    interp_log_r1 = np.interp(grid, psnr1, log_rate1)
    interp_log_r2 = np.interp(grid, psnr2, log_rate2)

    # 梯形法求面积积分
    int1 = np.trapz(interp_log_r1, grid)
    int2 = np.trapz(interp_log_r2, grid)

    avg_exp_diff = (int2 - int1) / (max_int - min_int)

    avg_exp_diff = max(min(avg_exp_diff, 200), -200)

    avg_diff = (math.exp(avg_exp_diff) - 1) * 100

    return avg_diff


# === 数据区 ===

curve1_car = [
    (0.671, 79.99), (0.580, 79.61), (0.493, 79.03), 
    (0.258, 80.12), (0.195, 79.73), (0.128, 72.08), (0.062, 33.28)
]
curve2_car = [
    (3.028, 81.87), # 帮你补上了原图中遗漏的第一个点
    (2.301, 81.18), (1.497, 76.69), (1.076, 67.51), 
    (0.650, 39.59), (0.446, 15.73), (0.225, 1.98)
]

curve1_biker = [
    (0.671, 68.16), (0.580, 65.28), (0.493, 62.29), 
    (0.258, 64.98), (0.195, 60.69), (0.128, 45.28), (0.062, 13.24)
]
curve2_biker = [
    (3.028, 67.74), (2.301, 63.43), (1.497, 48.50), 
    (1.076, 32.03), (0.650, 13.60), (0.446, 3.23), (0.225, 0.00)
]

curve1_ped = [
    (0.671, 50.30), (0.580, 50.37), (0.493, 49.59), 
    (0.258, 47.53), (0.195, 39.97), (0.128, 13.24), (0.062, 1.02)
]
curve2_ped = [
    (3.028, 49.34), (2.301, 34.61), (1.497, 11.58), 
    (1.076, 3.57), (0.650, 1.40), (0.446, 0.60), (0.225, 0.03)
]

# === 计算与输出区 ===
# 以前者为对比项(Split)，后者为基准项(GPCC)，若为负数代表节约了码率

bdrate_car = robust_bdrate(curve2_car, curve1_car)
bdrate_biker = robust_bdrate(curve2_biker, curve1_biker)
bdrate_ped = robust_bdrate(curve2_ped, curve1_ped)

print("=== BD-Rate (BPP) 计算结果 (Split 相比 G-PCC) ===")
print("  汽车(Car)    : {:.2f} %".format(bdrate_car))
print("  骑行者(Biker): {:.2f} %".format(bdrate_biker))
print("  行人(Ped)    : {:.2f} %".format(bdrate_ped))