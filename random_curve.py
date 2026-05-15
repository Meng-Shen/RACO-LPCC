import matplotlib.pyplot as plt
import os

def main():
    # ==========================================
    # 1. 直接定义 12 组 (BPP, mAP) 数据点
    # 数据模拟了典型的点云压缩率失真（R-D）曲线趋势
    # ==========================================
    bpps = [
        0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 
        0.70, 0.90, 1.20, 1.50, 2.00, 3.00
    ]

    car_aps = [
        10.50, 45.60, 57.85, 73.30, 25.10, 65.15, 
        75.50, 80.05, 40.25, 72.40, 80.15, 84.50
    ]

    print("========= Plotting Data (AP_R40 Moderate) =========")
    print(f"{'BPP':<10} | {'Car AP':<10}")
    for i in range(len(bpps)):
        print(f"{bpps[i]:<10.4f} | {car_aps[i]:<10.2f}")

    # ==========================================
    # 2. 绘制曲线
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # 只画一条 Car 的曲线
    plt.plot(bpps, car_aps, marker='o', markersize=6, linewidth=2, label='Car')

    # 设置坐标轴标签和标题
    plt.xlabel('Bitrate / Bits Per Point (BPP)', fontsize=12)
    plt.ylabel('3D mAP (Moderate Difficulty) [%]', fontsize=12)
    
    # 标题可以直接根据你的方案名称进行微调
    plt.title('Split-LPCC Rate-Distortion Curve (Car mAP vs. BPP)', fontsize=14, pad=15)
    
    # 设置网格、图例和 Y 轴范围
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=10)
    plt.ylim(bottom=0, top=100) # 根据实际 AP 范围可调整 top 值

    plt.tight_layout()
    
    # ==========================================
    # 3. 保存图像
    # ==========================================
    out_img_path = 'car_mAP_bpp_curve.png'
    plt.savefig(out_img_path, dpi=300)
    print(f"\n[+] 曲线图已成功保存至: {os.path.abspath(out_img_path)}")

if __name__ == '__main__':
    main()