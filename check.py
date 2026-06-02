import argparse
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

def parse_args():
    parser = argparse.ArgumentParser(description="Find and print points where nearest neighbor is different (XYZ + Reflectivity)")
    parser.add_argument("--dir1", required=True, help="dir1: 4D point cloud bin (x,y,z,reflectivity)")
    parser.add_argument("--dir2", required=True, help="dir2: 4D point cloud bin (x,y,z,reflectivity)")
    return parser.parse_args()

def main():
    args = parse_args()

    # ------------------- 加载点云：都是 4 维 -------------------
    p1 = np.fromfile(args.dir1, dtype=np.float32).reshape(-1, 4)  # XYZ + 反射率
    p2 = np.fromfile(args.dir2, dtype=np.float32).reshape(-1, 4)  # XYZ + 反射率

    # 只使用 XYZ 做最近邻搜索
    p1_xyz = p1[:, :3]
    p2_xyz = p2[:, :3]

    total_points = len(p1)
    print(f"dir1 总点数: {total_points}")
    print(f"dir2 总点数: {len(p2)}")

    # ------------------- 最近邻搜索（基于 XYZ） -------------------
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
    nn.fit(p2_xyz)
    dists, nn_indices = nn.kneighbors(p1_xyz, return_distance=True)
    nn_p = p2[nn_indices.flatten()]  # 取出完整 4 维最近邻

    # ------------------- 比较：XYZ + 反射率 全部维度 -------------------
    diff_mask = np.any(p1 != nn_p, axis=1)
    diff_points = p1[diff_mask]
    diff_nn_points = nn_p[diff_mask]
    diff_count = np.sum(diff_mask)

    # ------------------- 输出结果 -------------------
    print("\n" + "="*100)
    print(f"总点数：{total_points}")
    print(f"最近邻不完全一致的点数量：{diff_count}")
    print("="*100)

    if diff_count == 0:
        print("✅ 所有点的最近邻（XYZ + 反射率）完全一致！")
        return

    print("\n【不完全一致的点详情】")
    print(f"{'序号':<4}{'dir1 (x,y,z,refl)':<55}{'最近邻 (x,y,z,refl)':<55}{'差值 (dx,dy,dze,drefl)'}")
    print("-"*220)

    for i, (a, b) in enumerate(zip(diff_points, diff_nn_points)):
        dx, dy, dz, drefl = a - b
        print(f"{i+1:<4}"
              f"({a[0]:9.4f}, {a[1]:9.4f}, {a[2]:9.4f}, {a[3]:9.4f})   "
              f"({b[0]:9.4f}, {b[1]:9.4f}, {b[2]:9.4f}, {b[3]:9.4f})   "
              f"({dx:9.4f},{dy:9.4f},{dz:9.4f},{drefl:9.4f})")

    print(f"\n✅ 输出完成！共 {diff_count} 个点不完全一致")

if __name__ == "__main__":
    main()