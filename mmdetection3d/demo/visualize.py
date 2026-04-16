import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

# ====================== SemanticKITTI 19类 颜色 ======================
SEMANTICKITTI_COLORS = np.array([
    [0, 0, 0],          # 0: unlabeled
    [245, 150, 100],    # 1: car
    [245, 230, 100],    # 2: bicycle
    [150, 60, 30],      # 3: motorcycle
    [180, 30, 80],      # 4: truck
    [255, 255, 255],    # 5: other-vehicle
    [75, 0, 75],        # 6: person
    [255, 255, 0],      # 7: bicyclist
    [255, 0, 0],        # 8: motorcyclist
    [0, 255, 0],        # 9: road
    [0, 60, 255],       # 10: parking
    [0, 255, 255],      # 11: sidewalk
    [255, 0, 255],      # 12: other-ground
    [200, 200, 0],      # 13: building
    [255, 150, 0],      # 14: fence
    [0, 150, 255],      # 15: lane marker
    [0, 200, 80],       # 16: vegetation
    [150, 80, 150],     # 17: trunk
    [180, 180, 80]      # 18: terrain
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
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor='black')
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