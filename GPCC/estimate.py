import argparse
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from data_utils.geometry.inout import write_ply_o3d
from extention.gpcc_geo import gpcc_decode, gpcc_encode


FG_CLASSES = [1]
DEFAULT_FG_CLASSES = ",".join(str(x) for x in FG_CLASSES)


@contextmanager
def suppress_stderr():
    fd = sys.stderr.fileno()
    saved_fd = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    try:
        yield
    finally:
        os.dup2(saved_fd, fd)
        os.close(devnull)
        os.close(saved_fd)


def parse_scale_value(text):
    text = str(text).strip()
    if not text:
        raise ValueError("empty quantization scale value")
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)


def parse_quant_map(quant_map_str):
    quant_map = []
    for item in str(quant_map_str).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid quant map item '{item}'. Expected format: fg,bg")
        quant_map.append((parse_scale_value(parts[0]), parse_scale_value(parts[1])))
    if not quant_map:
        raise ValueError("--quant_map must contain at least one fg,bg pair")
    return quant_map


def parse_class_ids(text):
    class_ids = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            class_ids.append(int(item))
    if not class_ids:
        raise ValueError("--fg_classes must contain at least one class id")
    return class_ids


def read_kitti_bin(path):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def resolve_mask_dir(args):
    if args.mask_dir:
        return Path(args.mask_dir).resolve()

    candidates = []
    if args.split == "train":
        candidates.append(root_dir / "OpenPCDet" / "output" / "eval" / "train_seg_masks")
    else:
        candidates.append(root_dir / "OpenPCDet" / "output" / "eval" / "test_seg_masks")

    for path in candidates:
        if path.exists():
            return path.resolve()

    return candidates[0].resolve()


def build_sample_list(data_root, split, sample_count, seed):
    split_file = data_root / "ImageSets" / f"{split}.txt"
    velodyne_dir = data_root / "training" / "velodyne"

    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    if not velodyne_dir.exists():
        raise FileNotFoundError(f"Missing velodyne directory: {velodyne_dir}")

    with split_file.open("r") as f:
        frame_ids = [line.strip() for line in f if line.strip()]

    files = []
    for frame_id in frame_ids:
        bin_path = velodyne_dir / f"{frame_id}.bin"
        if bin_path.exists():
            files.append((frame_id, bin_path))

    if not files:
        raise RuntimeError(f"No valid .bin files found for split '{split}' under {velodyne_dir}")

    rng = random.Random(seed)
    if len(files) > sample_count:
        files = rng.sample(files, sample_count)
    else:
        rng.shuffle(files)
    return files


def load_fg_mask(mask_dir, frame_id, num_points, fg_classes):
    mask_path = mask_dir / f"{frame_id}.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing semantic mask: {mask_path}")

    seg_labels = np.load(mask_path)
    if len(seg_labels) < num_points:
        raise ValueError(
            f"Mask length is shorter than point count for {frame_id}: "
            f"mask={len(seg_labels)}, points={num_points}"
        )

    seg_labels = seg_labels[:num_points]
    return np.isin(seg_labels, fg_classes)


def encode_decode_partition(ply_path, bitstream_path, dec_ply_path, pos_quant_scale, cfg_path):
    bits = 0
    enc_time = 0.0
    dec_time = 0.0

    with suppress_stderr():
        enc_log = gpcc_encode(
            str(ply_path),
            str(bitstream_path),
            posQuantscale=pos_quant_scale,
            cfgdir=str(cfg_path),
        )
    if isinstance(enc_log, dict):
        enc_time = float(enc_log.get("Processing time (wall)", 0.0))

    if bitstream_path.exists():
        bits = bitstream_path.stat().st_size * 8
        with suppress_stderr():
            dec_log = gpcc_decode(str(bitstream_path), str(dec_ply_path))
        if isinstance(dec_log, dict):
            dec_time = float(dec_log.get("Processing time (wall)", 0.0))

    return bits, enc_time, dec_time


def estimate_file(frame_id, bin_path, mask_dir, fg_classes, quant_map, tmp_dir, cfg_path):
    coords_raw = read_kitti_bin(bin_path)
    num_points = len(coords_raw)
    if num_points == 0:
        return []

    coords_mm = np.round(coords_raw.astype(np.float64) * 1000).astype(np.int32)
    offset = coords_mm.min(axis=0)
    coords_scaled = coords_mm - offset

    fg_mask = load_fg_mask(mask_dir, frame_id, num_points, fg_classes)
    coords_fg = coords_scaled[fg_mask]
    coords_bg = coords_scaled[~fg_mask]

    fg_ply = tmp_dir / f"{frame_id}_fg_in.ply"
    bg_ply = tmp_dir / f"{frame_id}_bg_in.ply"

    if len(coords_fg) > 0:
        write_ply_o3d(str(fg_ply), coords_fg, normal=True, knn=16)
    if len(coords_bg) > 0:
        write_ply_o3d(str(bg_ply), coords_bg, normal=True, knn=16)

    rows = []
    try:
        for combo_id, (posq_fg, posq_bg) in enumerate(quant_map):
            fg_bits = bg_bits = 0
            fg_enc_time = bg_enc_time = 0.0
            fg_dec_time = bg_dec_time = 0.0

            fg_bitstream = tmp_dir / f"{frame_id}_combo{combo_id}_fg.bin"
            bg_bitstream = tmp_dir / f"{frame_id}_combo{combo_id}_bg.bin"
            fg_dec_ply = tmp_dir / f"{frame_id}_combo{combo_id}_fg_dec.ply"
            bg_dec_ply = tmp_dir / f"{frame_id}_combo{combo_id}_bg_dec.ply"

            if len(coords_fg) > 0:
                fg_bits, fg_enc_time, fg_dec_time = encode_decode_partition(
                    fg_ply, fg_bitstream, fg_dec_ply, posq_fg, cfg_path
                )
            if len(coords_bg) > 0:
                bg_bits, bg_enc_time, bg_dec_time = encode_decode_partition(
                    bg_ply, bg_bitstream, bg_dec_ply, posq_bg, cfg_path
                )

            total_bits = fg_bits + bg_bits
            rows.append(
                {
                    "frame_id": frame_id,
                    "combo_id": combo_id,
                    "posQ_fg": posq_fg,
                    "posQ_bg": posq_bg,
                    "num_points": num_points,
                    "fg_points": len(coords_fg),
                    "bg_points": len(coords_bg),
                    "fg_bits": fg_bits,
                    "bg_bits": bg_bits,
                    "total_bits": total_bits,
                    "bpp": round(total_bits / num_points, 6),
                    "fg_enc_time": round(fg_enc_time, 6),
                    "bg_enc_time": round(bg_enc_time, 6),
                    "enc_time": round(fg_enc_time + bg_enc_time, 6),
                    "fg_dec_time": round(fg_dec_time, 6),
                    "bg_dec_time": round(bg_dec_time, 6),
                    "dec_time": round(fg_dec_time + bg_dec_time, 6),
                }
            )

            for path in [fg_bitstream, bg_bitstream, fg_dec_ply, bg_dec_ply]:
                if path.exists():
                    path.unlink()
    finally:
        for path in [fg_ply, bg_ply]:
            if path.exists():
                path.unlink()

    return rows


def write_results(rows, results_dir):
    detail_df = pd.DataFrame(rows)
    detail_csv = results_dir / "estimate_all_details.csv"
    detail_df.to_csv(detail_csv, index=False)

    group_cols = ["combo_id", "posQ_fg", "posQ_bg"]
    avg_df = detail_df.groupby(group_cols).mean(numeric_only=True).reset_index()
    avg_csv = results_dir / "estimate_average_results.csv"
    avg_df.to_csv(avg_csv, index=False)

    return detail_csv, avg_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate split G-PCC geometry bitrate and codec time on random KITTI samples."
    )
    parser.add_argument("--split", choices=["train", "val"], required=True, help="Dataset split to sample.")
    parser.add_argument(
        "--quant_map",
        type=str,
        required=True,
        help="Quantization combinations. Format: fg,bg;fg,bg. Example: '1/256,1/1024;1/64,1/512'",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(root_dir / "OpenPCDet" / "data" / "kitti"),
        help="KITTI data root containing ImageSets and training/velodyne.",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Directory containing pre-computed semantic masks named <frame_id>.npy.",
    )
    parser.add_argument("--sample_count", type=int, default=100, help="Number of random point clouds to sample.")
    parser.add_argument("--seed", type=int, default=1024, help="Random seed for sampling.")
    parser.add_argument(
        "--results",
        type=str,
        default=str(current_dir / "estimate_results"),
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default=str(current_dir / "tmp_estimate"),
        help="Directory for temporary PLY and bitstream files.",
    )
    parser.add_argument(
        "--gpcc_cfg",
        type=str,
        default=str(root_dir / "extention" / "kitti.cfg"),
        help="G-PCC geometry config file.",
    )
    parser.add_argument(
        "--fg_classes",
        type=str,
        default=DEFAULT_FG_CLASSES,
        help=f"Comma-separated semantic class ids treated as foreground. Default: {DEFAULT_FG_CLASSES}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    quant_map = parse_quant_map(args.quant_map)
    fg_classes = parse_class_ids(args.fg_classes)

    data_root = Path(args.data_root).resolve()
    mask_dir = resolve_mask_dir(args)
    results_dir = Path(args.results).resolve()
    tmp_dir = Path(args.tmp_dir).resolve()
    cfg_path = Path(args.gpcc_cfg).resolve()

    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing G-PCC cfg file: {cfg_path}")

    results_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sample_files = build_sample_list(data_root, args.split, args.sample_count, args.seed)

    print(f"Split       : {args.split}")
    print(f"Data root   : {data_root}")
    print(f"Mask dir    : {mask_dir}")
    print(f"Samples     : {len(sample_files)}")
    print(f"Quant combos: {len(quant_map)}")
    for combo_id, (posq_fg, posq_bg) in enumerate(quant_map):
        print(f"  combo_{combo_id}: fg={posq_fg}, bg={posq_bg}")

    all_rows = []
    for frame_id, bin_path in tqdm(sample_files, desc="Estimating G-PCC rates"):
        rows = estimate_file(frame_id, bin_path, mask_dir, fg_classes, quant_map, tmp_dir, cfg_path)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No results were produced.")

    detail_csv, avg_csv = write_results(all_rows, results_dir)
    print(f"Details CSV : {detail_csv}")
    print(f"Average CSV : {avg_csv}")


if __name__ == "__main__":
    main()
