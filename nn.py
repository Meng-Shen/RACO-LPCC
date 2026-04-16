import os
import argparse
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Assign reflectivity from ref.bin to input_{mode}.bin using nearest neighbor")
    parser.add_argument("--input", required=True, help="input folder with xxx_{mode}.bin (3D points)")
    parser.add_argument("--ref", required=True, help="reference folder with xxx.bin (4D points: xyz+reflectivity)")
    parser.add_argument("--output", required=True, help="output base folder, will auto add mode suffix like output_R0")
    parser.add_argument("--mode", required=True, choices=['r0','r1','r2','r3','r4','r5','r6','r7','r8','r9'], help="mode: R0 ~ R9")
    return parser.parse_args()

def main():
    args = parse_args()
    mode = args.mode

    # 输出文件夹 = output + mode
    out_dir = Path(f"{args.output}/{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 获取ref文件夹下所有bin文件
    ref_files = list(Path(args.ref).glob("*.bin"))
    print(f"Found {len(ref_files)} bin files in ref folder")

    # 遍历每个参考文件
    for ref_path in tqdm(ref_files, desc=f"assign reflectivity {mode}", unit="file", ncols=80):
        base_name = ref_path.stem  # 获取文件名(不含后缀)
        
        # 对应 input 中的文件: {base_name}_{mode}.bin
        input_path = Path(args.input) / f"{base_name}_{mode}.bin"
        if not input_path.exists():
            print(f"Skip {base_name}: {input_path} not found")
            continue

        # 输出文件名
        output_path = out_dir / f"{base_name}.bin"

        # ==================== 加载点云 ====================
        # 加载ref点云 (4维: xyz + reflectivity)
        ref_points = np.fromfile(ref_path, dtype=np.float32)
        ref_points = ref_points.reshape(-1, 4)
        ref_xyz = ref_points[:, 0:3]          # 坐标
        ref_reflectivity = ref_points[:, 3:4] # 反射率

        # 加载input点云 (3维: 只有坐标)
        input_xyz = np.fromfile(input_path, dtype=np.float32)
        input_xyz = input_xyz.reshape(-1, 3)

        # ==================== 最近邻搜索 ====================
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
        nn.fit(ref_xyz)
        indices = nn.kneighbors(input_xyz, return_distance=False).flatten()

        # ==================== 赋值反射率 ====================
        input_reflectivity = ref_reflectivity[indices]
        output_points = np.concatenate([input_xyz, input_reflectivity], axis=1).astype(np.float32)

        # ==================== 保存 ====================
        output_points.tofile(str(output_path))

    print(f"\nAll done! Output saved to: {out_dir}")

if __name__ == "__main__":
    main()