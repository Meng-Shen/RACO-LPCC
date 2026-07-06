import argparse
import csv
import os
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(CURRENT_DIR))

import MinkowskiEngine as ME

from data_utils.geometry.inout import write_ply_o3d
from extention.gpcc_geo import gpcc_decode, gpcc_encode
from extention.pc_error_geo import pc_error
from train_sparse_sr import SparseOccupancySRNet, format_scale, parse_scale


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
FG_CLASSES = [1]


def suppress_fd2():
    class _Suppress:
        def __enter__(self):
            self.saved = os.dup(2)
            self.devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self.devnull, 2)

        def __exit__(self, exc_type, exc, tb):
            os.dup2(self.saved, 2)
            os.close(self.devnull)
            os.close(self.saved)

    return _Suppress()


def read_kitti_xyz(bin_path):
    points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def read_ply_points(path):
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points, dtype=np.float64)


def unique_rows(array):
    if len(array) == 0:
        return array.reshape(0, array.shape[-1])
    return np.unique(array, axis=0)


def quantized_count(coords, scale):
    qcoords = np.round(coords.astype(np.float64) * float(scale)).astype(np.int32)
    return len(unique_rows(qcoords))


def quantized_points(coords, scale):
    qcoords = np.round(coords.astype(np.float64) * float(scale)).astype(np.int32)
    qcoords = unique_rows(qcoords)
    return qcoords.astype(np.float64) / float(scale)


def load_split_mask(mask_dir, frame_id, num_points):
    if not mask_dir:
        return None
    mask_path = Path(mask_dir) / f"{frame_id}.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"Split-GPCC mask not found: {mask_path}")
    labels = np.load(mask_path)[:num_points]
    if len(labels) != num_points:
        raise ValueError(
            f"Mask length mismatch for {frame_id}: mask has {len(labels)}, point cloud has {num_points}"
        )
    return np.isin(labels, FG_CLASSES)


def write_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_ratio_map(path, column):
    ratios = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratios[row["scale"]] = float(row[column])
    return ratios


def encode_decode(coords, name, scale, tmp_dir, cfg_path):
    if len(coords) == 0:
        return 0, np.empty((0, 3), dtype=np.float64), None
    in_ply = tmp_dir / f"{name}_in.ply"
    bitstream = tmp_dir / f"{name}.bin"
    out_ply = tmp_dir / f"{name}_dec.ply"
    write_ply_o3d(str(in_ply), coords, dtype="int32")
    with suppress_fd2():
        gpcc_encode(str(in_ply), str(bitstream), posQuantscale=float(scale), cfgdir=str(cfg_path))
        gpcc_decode(str(bitstream), str(out_ply))
    bits = bitstream.stat().st_size * 8
    decoded = read_ply_points(out_ply)
    return bits, decoded, out_ply


def psnr_metrics(ref_ply, dec_ply, resolution):
    result = pc_error(str(ref_ply), str(dec_ply), resolution=resolution, normal=True, show=False)
    return {
        "psnr_d1": float(result["mseF,PSNR (p2point)"]),
        "psnr_d2": float(result["mseF,PSNR (p2plane)"]),
    }


@torch.no_grad()
def sparse_sr_quota(model, coarse_points, scale, target_count, device):
    coarse = np.round(coarse_points.astype(np.float64) * float(scale)).astype(np.int32)
    coarse = unique_rows(coarse)
    if len(coarse) == 0:
        return np.empty((0, 3), dtype=np.float64), 0, 0

    coords = torch.from_numpy(
        np.concatenate([np.zeros((len(coarse), 1), dtype=np.int32), coarse], axis=1)
    ).int()
    feats = torch.ones((len(coarse), 1), dtype=torch.float32)
    stensor = ME.SparseTensor(features=feats.to(device), coordinates=coords.to(device), device=device)
    probs = torch.sigmoid(model(stensor).F).detach().cpu().numpy()

    child_offsets = np.array(
        [[i // 4, (i // 2) % 2, i % 2] for i in range(8)], dtype=np.int32
    )
    target_count = max(len(coarse), min(int(target_count), len(coarse) * 8))

    selected = np.zeros_like(probs, dtype=bool)
    best_child = np.argmax(probs, axis=1)
    selected[np.arange(len(coarse)), best_child] = True

    remaining = target_count - len(coarse)
    if remaining > 0:
        flat_probs = probs.reshape(-1).copy()
        flat_selected = selected.reshape(-1)
        flat_probs[flat_selected] = -np.inf
        remaining = min(remaining, flat_probs.size - int(flat_selected.sum()))
        if remaining > 0:
            top = np.argpartition(flat_probs, -remaining)[-remaining:]
            flat_selected[top] = True

    parent_idx, child_idx = np.nonzero(selected)
    fine = coarse[parent_idx] * 2 + child_offsets[child_idx]
    fine = unique_rows(fine.astype(np.int32))
    return fine.astype(np.float64) / float(scale * 2), int(selected.sum()), target_count


@torch.no_grad()
def sparse_sr_threshold(model, coarse_points, scale, threshold, device):
    coarse = np.round(coarse_points.astype(np.float64) * float(scale)).astype(np.int32)
    coarse = unique_rows(coarse)
    if len(coarse) == 0:
        return np.empty((0, 3), dtype=np.float64), 0, 0

    coords = torch.from_numpy(
        np.concatenate([np.zeros((len(coarse), 1), dtype=np.int32), coarse], axis=1)
    ).int()
    feats = torch.ones((len(coarse), 1), dtype=torch.float32)
    stensor = ME.SparseTensor(features=feats.to(device), coordinates=coords.to(device), device=device)
    probs = torch.sigmoid(model(stensor).F).detach().cpu().numpy()

    child_offsets = np.array(
        [[i // 4, (i // 2) % 2, i % 2] for i in range(8)], dtype=np.int32
    )
    child_mask = probs >= threshold
    fine_chunks = []
    for child_idx in range(8):
        keep = child_mask[:, child_idx]
        if np.any(keep):
            fine_chunks.append(coarse[keep] * 2 + child_offsets[child_idx])
    if not fine_chunks:
        fine_chunks.append(coarse * 2)
    fine = unique_rows(np.concatenate(fine_chunks, axis=0).astype(np.int32))
    return fine.astype(np.float64) / float(scale * 2), int(child_mask.sum()), int(child_mask.sum())


def load_model(ckpt_path, device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    args = ckpt.get("args", {})
    model = SparseOccupancySRNet(
        channels=int(args.get("channels", 64)),
        num_blocks=int(args.get("blocks", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="000001.bin", help="KITTI frame name or id, e.g. 000001.bin")
    parser.add_argument("--frame_id", default="", help="Deprecated alias. If set, overrides --frame.")
    parser.add_argument("--velodyne_dir", default="OpenPCDet/data/kitti_fov/training/velodyne")
    parser.add_argument("--bin", default="", help="Optional explicit KITTI .bin path.")
    parser.add_argument("--ckpt", default="GPCC/work_dirs/sparse_sr/latest.pth")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--cfg", default="extention/kitti.cfg")
    parser.add_argument("--scales", default=",".join(DEFAULT_SCALES))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--selection",
        choices=["oracle_points", "oracle_count", "quota", "threshold"],
        default="oracle_count",
    )
    parser.add_argument("--ratio_csv", default="GPCC/quantized_point_ratios_train.csv")
    parser.add_argument("--ratio_column", default="mean_frame_ratio")
    parser.add_argument("--metadata_bits", type=int, default=32)
    parser.add_argument(
        "--split_mask_dir",
        default="point_pairs/split_gpcc_fov/seg_masks",
        help="Directory containing <frame_id>.npy foreground/background masks for the Split-GPCC curve.",
    )
    parser.add_argument(
        "--split_fg_scale",
        default="1/64",
        help="Fixed foreground quantization scale used by the Split-GPCC curve.",
    )
    parser.add_argument(
        "--split_metadata_bits",
        type=int,
        default=32,
        help="Fixed metadata bits added to every Split-GPCC point.",
    )
    parser.add_argument(
        "--no_split_gpcc",
        action="store_true",
        help="Disable the Split-GPCC + background SR curve.",
    )
    parser.add_argument("--resolution", type=int, default=80000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    frame_id = args.frame_id or Path(args.frame).stem
    bin_path = Path(args.bin) if args.bin else Path(args.velodyne_dir) / f"{frame_id}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"KITTI frame not found: {bin_path}")
    if not args.out_dir:
        ckpt_name = Path(args.ckpt).stem
        args.out_dir = f"GPCC/outputs_sr_eval/{frame_id}_{ckpt_name}"

    out_dir = Path(args.out_dir)
    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    scales = [parse_scale(item) for item in args.scales.split(",") if item.strip()]
    ratio_map = load_ratio_map(args.ratio_csv, args.ratio_column) if args.selection == "quota" else {}
    model, ckpt = load_model(Path(args.ckpt), args.device)

    xyz = read_kitti_xyz(bin_path)
    coords_mm = np.round(xyz.astype(np.float64) * 1000).astype(np.int32)
    offset = coords_mm.min(axis=0)
    coords_scaled = (coords_mm - offset).astype(np.int32)

    ref_ply = tmp_dir / f"{frame_id}_ref.ply"
    write_ply_o3d(str(ref_ply), coords_scaled, dtype="int32", normal=True, knn=16)

    num_points = len(coords_scaled)
    fg_mask = None
    split_fg_scale = parse_scale(args.split_fg_scale)
    if not args.no_split_gpcc:
        fg_mask = load_split_mask(args.split_mask_dir, frame_id, num_points)

    sr_rows = []
    baseline_rows = []
    split_rows = []
    for rate_idx, scale in enumerate(scales):
        scale_name = format_scale(scale)
        input_bits, dec_points, _ = encode_decode(
            coords_scaled,
            f"{frame_id}_sr_input_{rate_idx}_{scale_name.replace('/', '_')}",
            scale,
            tmp_dir,
            Path(args.cfg).resolve(),
        )
        if args.selection == "quota":
            quota_ratio = ratio_map[scale_name]
            metadata_bits = 0
            target_children = int(round(len(unique_rows(np.round(dec_points.astype(np.float64) * float(scale)).astype(np.int32))) / quota_ratio))
            sr_points, pred_children, target_children = sparse_sr_quota(
                model, dec_points, scale, target_children, args.device
            )
        elif args.selection == "oracle_count":
            quota_ratio = 0.0
            metadata_bits = args.metadata_bits
            target_children = quantized_count(coords_scaled, scale * 2)
            sr_points, pred_children, target_children = sparse_sr_quota(
                model, dec_points, scale, target_children, args.device
            )
        elif args.selection == "oracle_points":
            quota_ratio = 0.0
            metadata_bits = args.metadata_bits
            sr_points = quantized_points(coords_scaled, scale * 2)
            target_children = len(sr_points)
            pred_children = target_children
        else:
            quota_ratio = 0.0
            metadata_bits = 0
            sr_points, pred_children, target_children = sparse_sr_threshold(
                model, dec_points, scale, args.threshold, args.device
            )
        sr_points = unique_rows(sr_points)
        sr_ply = out_dir / f"{frame_id}_sr_{rate_idx}_{scale_name.replace('/', '_')}.ply"
        write_ply_o3d(str(sr_ply), sr_points, dtype="float32")
        sr_psnr = psnr_metrics(ref_ply, sr_ply, args.resolution)
        sr_rows.append(
            {
                "filename": frame_id,
                "rate_id": rate_idx,
                "scale": scale_name,
                "input_bits": input_bits,
                "metadata_bits": metadata_bits,
                "bits": input_bits + metadata_bits,
                "bpp": (input_bits + metadata_bits) / num_points,
                "psnr_d1": sr_psnr["psnr_d1"],
                "psnr_d2": sr_psnr["psnr_d2"],
                "points_dec": len(dec_points),
                "points_sr": len(sr_points),
                "pred_children": pred_children,
                "target_children": target_children,
                "quota_ratio": quota_ratio,
            }
        )

        base_bits, _, base_ply = encode_decode(
            coords_scaled,
            f"{frame_id}_gpcc_{rate_idx}_{scale_name.replace('/', '_')}",
            scale,
            tmp_dir,
            Path(args.cfg).resolve(),
        )
        base_psnr = psnr_metrics(ref_ply, base_ply, args.resolution)
        baseline_rows.append(
            {
                "filename": frame_id,
                "rate_id": rate_idx,
                "scale": scale_name,
                "bits": base_bits,
                "bpp": base_bits / num_points,
                "psnr_d1": base_psnr["psnr_d1"],
                "psnr_d2": base_psnr["psnr_d2"],
            }
        )

        split_log = ""
        if fg_mask is not None:
            coords_fg = coords_scaled[fg_mask]
            coords_bg = coords_scaled[~fg_mask]
            split_fg_bits, split_fg_points, _ = encode_decode(
                coords_fg,
                f"{frame_id}_split_fg_{rate_idx}_{format_scale(split_fg_scale).replace('/', '_')}",
                split_fg_scale,
                tmp_dir,
                Path(args.cfg).resolve(),
            )
            split_bg_bits, split_bg_dec_points, _ = encode_decode(
                coords_bg,
                f"{frame_id}_split_bg_{rate_idx}_{scale_name.replace('/', '_')}",
                scale,
                tmp_dir,
                Path(args.cfg).resolve(),
            )
            split_bg_target_children = quantized_count(coords_bg, scale * 2)
            split_bg_sr_points, split_bg_pred_children, split_bg_target_children = sparse_sr_quota(
                model, split_bg_dec_points, scale, split_bg_target_children, args.device
            )
            split_points = unique_rows(np.concatenate([split_fg_points, split_bg_sr_points], axis=0))
            split_ply = out_dir / f"{frame_id}_split_gpcc_sr_{rate_idx}_{scale_name.replace('/', '_')}.ply"
            write_ply_o3d(str(split_ply), split_points, dtype="float32")
            split_psnr = psnr_metrics(ref_ply, split_ply, args.resolution)
            split_bits = split_fg_bits + split_bg_bits + args.split_metadata_bits
            split_rows.append(
                {
                    "filename": frame_id,
                    "rate_id": rate_idx,
                    "fg_scale": format_scale(split_fg_scale),
                    "bg_scale": scale_name,
                    "fg_bits": split_fg_bits,
                    "bg_bits": split_bg_bits,
                    "metadata_bits": args.split_metadata_bits,
                    "bits": split_bits,
                    "bpp": split_bits / num_points,
                    "psnr_d1": split_psnr["psnr_d1"],
                    "psnr_d2": split_psnr["psnr_d2"],
                    "fg_points": int(fg_mask.sum()),
                    "bg_points": int((~fg_mask).sum()),
                    "fg_points_dec": len(split_fg_points),
                    "bg_points_dec": len(split_bg_dec_points),
                    "bg_points_sr": len(split_bg_sr_points),
                    "bg_pred_children": split_bg_pred_children,
                    "bg_target_children": split_bg_target_children,
                }
            )
            split_log = (
                f" Split-GPCC+BG-SR bpp={split_rows[-1]['bpp']:.6f} "
                f"D1={split_psnr['psnr_d1']:.4f} D2={split_psnr['psnr_d2']:.4f} "
                f"bg_points={len(split_bg_sr_points)}/{split_bg_target_children} "
                f"metadata_bits={args.split_metadata_bits}"
            )
        print(
            f"{frame_id} scale={scale_name} "
            f"SR bpp={sr_rows[-1]['bpp']:.6f} D1={sr_psnr['psnr_d1']:.4f} D2={sr_psnr['psnr_d2']:.4f} "
            f"points={len(sr_points)}/{target_children} metadata_bits={metadata_bits} "
            f"GPCC bpp={baseline_rows[-1]['bpp']:.6f} D1={base_psnr['psnr_d1']:.4f} D2={base_psnr['psnr_d2']:.4f}"
            f"{split_log}",
            flush=True,
        )

    write_rows(
        out_dir / "sr_psnr_bpp.csv",
        sr_rows,
        [
            "filename",
            "rate_id",
            "scale",
            "input_bits",
            "metadata_bits",
            "bits",
            "bpp",
            "psnr_d1",
            "psnr_d2",
            "points_dec",
            "points_sr",
            "pred_children",
            "target_children",
            "quota_ratio",
        ],
    )
    write_rows(
        out_dir / "gpcc_psnr_bpp.csv",
        baseline_rows,
        ["filename", "rate_id", "scale", "bits", "bpp", "psnr_d1", "psnr_d2"],
    )
    if split_rows:
        write_rows(
            out_dir / "split_gpcc_sr_psnr_bpp.csv",
            split_rows,
            [
                "filename",
                "rate_id",
                "fg_scale",
                "bg_scale",
                "fg_bits",
                "bg_bits",
                "metadata_bits",
                "bits",
                "bpp",
                "psnr_d1",
                "psnr_d2",
                "fg_points",
                "bg_points",
                "fg_points_dec",
                "bg_points_dec",
                "bg_points_sr",
                "bg_pred_children",
                "bg_target_children",
            ],
        )

    def plot_curve(metric_key, ylabel, filename):
        sr_sorted = sorted(sr_rows, key=lambda r: r["bpp"])
        base_sorted = sorted(baseline_rows, key=lambda r: r["bpp"])
        split_sorted = sorted(split_rows, key=lambda r: r["bpp"])
        plt.figure(figsize=(7.5, 5.2), dpi=160)
        plt.plot(
            [r["bpp"] for r in base_sorted],
            [r[metric_key] for r in base_sorted],
            marker="o",
            linewidth=2,
            label="GPCC baseline",
        )
        plt.plot(
            [r["bpp"] for r in sr_sorted],
            [r[metric_key] for r in sr_sorted],
            marker="s",
            linewidth=2,
            label="GPCC + SR",
        )
        if split_sorted:
            plt.plot(
                [r["bpp"] for r in split_sorted],
                [r[metric_key] for r in split_sorted],
                marker="^",
                linewidth=2,
                label="Split-GPCC + BG SR",
            )
        for row in sr_rows:
            plt.annotate(row["scale"], (row["bpp"], row[metric_key]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        for row in split_rows:
            plt.annotate(
                row["bg_scale"],
                (row["bpp"], row[metric_key]),
                fontsize=7,
                xytext=(3, -8),
                textcoords="offset points",
            )
        plt.xlabel("bpp")
        plt.ylabel(ylabel)
        plt.title(f"{frame_id} {ylabel}-bpp")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename)
        plt.close()

    plot_curve("psnr_d1", "D1 PSNR (dB)", "d1_psnr_bpp_curve.png")
    plot_curve("psnr_d2", "D2 PSNR (dB)", "d2_psnr_bpp_curve.png")

    summary = {
        "frame_id": frame_id,
        "bin": str(bin_path.resolve()),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "checkpoint_epoch": ckpt.get("epoch"),
        "threshold": args.threshold,
        "selection": args.selection,
        "ratio_csv": str(Path(args.ratio_csv).resolve()),
        "ratio_column": args.ratio_column,
        "metadata_bits": args.metadata_bits,
        "split_gpcc_enabled": bool(split_rows),
        "split_mask_dir": str(Path(args.split_mask_dir).resolve()) if args.split_mask_dir else "",
        "split_fg_scale": format_scale(split_fg_scale),
        "split_metadata_bits": args.split_metadata_bits,
        "num_points": num_points,
        "out_dir": str(out_dir.resolve()),
    }
    write_rows(out_dir / "summary.csv", [summary], list(summary.keys()))


if __name__ == "__main__":
    main()
