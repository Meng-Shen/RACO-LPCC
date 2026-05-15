import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

# ====================== SemanticKITTI 19类 颜色 ======================

SEMANTICKITTI_COLORS = np.array([
    [70, 100, 180],          # 0: unlabeled
    [0, 220, 220],    # 1: car
    [0, 220, 220],    # 2: bicycle
    [0, 220, 220],      # 3: motorcycle
    [0, 220, 220],      # 4: truck
    [0, 220, 220],    # 5: other-vehicle
    [0, 220, 220],        # 6: person
    [0, 220, 220],      # 7: bicyclist
    [255, 0, 255],        # 8: motorcyclist
    [180, 100, 180],        # 9: road
    [170, 0, 170],       # 10: parking
    [200, 110, 0],      # 11: sidewalk
    [200, 160, 0],      # 12: other-ground
    [0, 220, 220],      # 13: building
    [0, 150, 0],      # 14: fence
    [150, 60, 0],      # 15: lane marker
    [140, 200, 100],       # 16: vegetation
    [200, 190, 140],     # 17: trunk
    [0, 220, 220]      # 18: terrain
]) / 255.0

SEMANTICKITTI_COLORS = np.array([
    [255, 0, 0],    # 0  属于FG → 红
    [255, 0, 0],    # 1  属于FG → 红
    [255, 0, 0],    # 2  属于FG → 红
    [255, 0, 0],    # 3  属于FG → 红
    [255, 0, 0],    # 4  属于FG → 红
    [255, 0, 0],    # 5  属于FG → 红
    [255, 0, 0],    # 6  属于FG → 红
    [255, 0, 0],    # 7  属于FG → 红
    [0,   0, 0],    # 8  不属于FG → 黑
    [0,   0, 0],    # 9  不属于FG → 黑
    [0,   0, 0],    # 10 不属于FG → 黑
    [0,   0, 0],    # 11 不属于FG → 黑
    [0,   0, 0],    # 12 不属于FG → 黑
    [0,   0, 0],    # 13 不属于FG → 黑
    [0,   0, 0],    # 14 不属于FG → 黑
    [0,   0, 0],    # 15 不属于FG → 黑
    [0,   0, 0],    # 16 不属于FG → 黑
    [255, 0, 0],    # 17 属于FG → 红
    [255, 0, 0]     # 18 属于FG → 红
]) / 255.0

# ====================== 工具函数 ======================
def load_point_cloud(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # xyz


def load_json_labels(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data['pts_semantic_mask'], dtype=np.int32)


# 👉 修改点 1：增加 x_range 和 y_range 参数
def visualize_and_save(points, labels, save_path, x_range=None, y_range=None):
    
    # 🌟 优化与安全处理：不仅要过滤点，必须同步过滤对应的颜色标签(labels)！
    if x_range is not None:
        mask_x = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
        points = points[mask_x]
        labels = labels[mask_x]  # 同步过滤标签
        
    if y_range is not None:
        mask_y = (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
        points = points[mask_y]
        labels = labels[mask_y]  # 同步过滤标签

    x = points[:, 0]
    y = points[:, 1]
    # z = points[:, 2] # 俯视图中用不到Z轴，可以注释掉

    plt.figure(figsize=(16, 14), dpi=150)
    ax = plt.gca()
    
    ax.scatter(x, y, c=SEMANTICKITTI_COLORS[labels], s=0.05, alpha=0.8)
    
    # 🌟 修改点 2：硬性限制画布的显示范围
    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()
    print(f"✅ 保存成功: {save_path}")


# ====================== 命令行输入 ======================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用方法:")
        print("   python visualize.py 点云.bin 结果.json")
        sys.exit(1)

    bin_file = sys.argv[1]
    json_file = sys.argv[2]

    points = load_point_cloud(bin_file)
    labels = load_json_labels(json_file)
    
    # 👇👇👇 在这里手动输入你想要的 X 和 Y 坐标范围 👇👇👇
    X_RANGE = (-30, 30)  # 例如 X轴(前后) -20米 到 20米
    Y_RANGE = (-30, 30)  # 例如 Y轴(左右) -20米 到 20米
    # 如果某一个轴不想限制，可以设置为 None，例如：Y_RANGE = None
    # 👆👆👆 -------------------------------------- 👆👆👆

    # 文件名加上 cropped 后缀以区分
    img_path = os.path.splitext(json_file)[0] + "_cropped.png"

    # 将坐标范围参数传进去
    visualize_and_save(points, labels, img_path, x_range=X_RANGE, y_range=Y_RANGE)