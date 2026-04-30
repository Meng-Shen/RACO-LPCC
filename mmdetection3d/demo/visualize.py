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


def visualize_and_save(points, labels, save_path):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    plt.figure(figsize=(16, 14), dpi=150)
    plt.scatter(x, y, c=SEMANTICKITTI_COLORS[labels], s=0.05, alpha=0.8)
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
    img_path = os.path.splitext(json_file)[0] + ".png"

    visualize_and_save(points, labels, img_path)