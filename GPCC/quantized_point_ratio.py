import argparse
import csv
from fractions import Fraction
from pathlib import Path

import numpy as np


DEFAULT_SCALES = [
    "1/64",
    "1.5/128",
    "1/128",
    "1.5/256",
    "1/256",
    "1.5/512",
    "1/512",
    "1/2048",
]


def parse_scale(value):
    value = str(value).strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return Fraction(numerator) / Fraction(denominator)
    return Fraction(value)


def format_scale(scale):
    return f"{scale.numerator}/{scale.denominator}" if scale.denominator != 1 else str(scale.numerator)


def collect_files(velodyne_dir, split_file):
    velodyne_dir = Path(velodyne_dir)
    if split_file:
        with open(split_file, "r") as f:
            frame_ids = [line.strip() for line in f if line.strip()]
        return [velodyne_dir / f"{frame_id}.bin" for frame_id in frame_ids if (velodyne_dir / f"{frame_id}.bin").exists()]
    return sorted(velodyne_dir.glob("*.bin"))


def read_kitti_xyz(bin_path):
    points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def quantized_count(coords_scaled, scale):
    qcoords = np.round(coords_scaled.astype(np.float64) * float(scale)).astype(np.int32)
    maxima = qcoords.max(axis=0).astype(np.int64) + 1
    strides = np.array([maxima[1] * maxima[2], maxima[2], 1], dtype=np.int64)
    keys = qcoords.astype(np.int64) @ strides
    return int(np.unique(keys).shape[0])


def main():
    parser = argparse.ArgumentParser(
        description="Count KITTI quantized-point ratios between each G-PCC scale and 2x that scale."
    )
    parser.add_argument(
        "--velodyne_dir",
        default="OpenPCDet/data/kitti_fov/training/velodyne",
        help="KITTI velodyne directory containing .bin files.",
    )
    parser.add_argument(
        "--split_file",
        default="OpenPCDet/data/kitti_fov/ImageSets/train.txt",
        help="KITTI split file. Use an empty string to scan all .bin files.",
    )
    parser.add_argument(
        "--scales",
        default=",".join(DEFAULT_SCALES),
        help="Comma-separated positionQuantizationScale values.",
    )
    parser.add_argument(
        "--output",
        default="GPCC/quantized_point_ratios_train.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    scales = [parse_scale(item) for item in args.scales.split(",") if item.strip()]
    split_file = args.split_file if args.split_file else None
    files = collect_files(args.velodyne_dir, split_file)
    if not files:
        raise FileNotFoundError("No KITTI .bin files found.")

    rows = {
        scale: {
            "sum_ratio": 0.0,
            "sum_count": 0,
            "sum_count_2x": 0,
            "frames": 0,
        }
        for scale in scales
    }

    for index, bin_path in enumerate(files, start=1):
        xyz = read_kitti_xyz(bin_path)
        if len(xyz) == 0:
            continue

        coords_mm = np.round(xyz.astype(np.float64) * 1000).astype(np.int32)
        coords_scaled = coords_mm - coords_mm.min(axis=0)
        counts = {}
        for scale in sorted(set(scales + [scale * 2 for scale in scales])):
            counts[scale] = quantized_count(coords_scaled, scale)

        for scale in scales:
            count = counts[scale]
            count_2x = counts[scale * 2]
            ratio = count / count_2x if count_2x else 0.0
            rows[scale]["sum_ratio"] += ratio
            rows[scale]["sum_count"] += count
            rows[scale]["sum_count_2x"] += count_2x
            rows[scale]["frames"] += 1

        if index % 500 == 0:
            print(f"processed {index}/{len(files)} frames", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        fieldnames = [
            "rate_id",
            "scale",
            "scale_float",
            "scale_2x",
            "scale_2x_float",
            "num_frames",
            "mean_frame_ratio",
            "total_count",
            "total_count_2x",
            "total_count_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rate_id, scale in enumerate(scales):
            item = rows[scale]
            mean_frame_ratio = item["sum_ratio"] / item["frames"] if item["frames"] else 0.0
            total_ratio = item["sum_count"] / item["sum_count_2x"] if item["sum_count_2x"] else 0.0
            row = {
                "rate_id": rate_id,
                "scale": format_scale(scale),
                "scale_float": f"{float(scale):.12g}",
                "scale_2x": format_scale(scale * 2),
                "scale_2x_float": f"{float(scale * 2):.12g}",
                "num_frames": item["frames"],
                "mean_frame_ratio": f"{mean_frame_ratio:.12f}",
                "total_count": item["sum_count"],
                "total_count_2x": item["sum_count_2x"],
                "total_count_ratio": f"{total_ratio:.12f}",
            }
            writer.writerow(row)
            print(
                f"{rate_id}: scale={row['scale']} vs {row['scale_2x']}, "
                f"mean_frame_ratio={row['mean_frame_ratio']}, "
                f"total_count_ratio={row['total_count_ratio']}"
            )

    print(f"wrote {output}")


if __name__ == "__main__":
    main()
