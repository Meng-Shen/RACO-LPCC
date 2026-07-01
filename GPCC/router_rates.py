#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from split_rates import FG_CLASSES, encode_subset, parse_quant_map, read_kitti_bin, read_seg_times


def read_split(split_file):
    with open(split_file) as f:
        return [line.strip().zfill(6) for line in f if line.strip()]


def read_labels(path):
    labels = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            labels[str(row["frame_id"]).zfill(6)] = int(row["jucp_label"])
    return labels


def load_label_jobs(label_csvs, manifest):
    jobs = []
    if manifest:
        data = json.loads(Path(manifest).read_text())
        for item in data["label_csvs"]:
            jobs.append((int(item["rate_id"]), item.get("threshold", ""), Path(item["path"])))
    for item in label_csvs or []:
        rate_id, threshold, path = item.split(":", 2)
        jobs.append((int(rate_id), threshold, Path(path)))
    if not jobs:
        raise ValueError("Provide --manifest or one or more --label_csv rate_id:threshold:path entries")
    return sorted(jobs, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(description="Measure adaptive router-assisted Split-GPCC bpp/time.")
    parser.add_argument("--testdata", required=True, help="KITTI velodyne directory")
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--seg_time_csv", default=None)
    parser.add_argument("--quant_map", required=True)
    parser.add_argument("--manifest", default=None, help="router_manifest.json from export_router_jucp.py")
    parser.add_argument("--label_csv", action="append", default=[], help="rate_id:threshold:path")
    parser.add_argument("--results", required=True)
    parser.add_argument("--tmp_dir", required=True)
    parser.add_argument("--cfg", default="extention/kitti.cfg")
    args = parser.parse_args()

    combos = parse_quant_map(args.quant_map)
    frame_ids = read_split(args.split_file)
    testdata = Path(args.testdata)
    mask_dir = Path(args.mask_dir)
    result_dir = Path(args.results)
    tmp_dir = Path(args.tmp_dir)
    cfg_path = Path(args.cfg).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    seg_times = read_seg_times(args.seg_time_csv)
    jobs = load_label_jobs(args.label_csv, args.manifest)

    detail_rows = []
    progress = tqdm(total=len(frame_ids) * len(jobs), desc="Router-GPCC", unit="job", dynamic_ncols=True)
    for rate_id, threshold, label_csv in jobs:
        labels_by_frame = read_labels(label_csv)
        for frame_id in frame_ids:
            bin_path = testdata / f"{frame_id}.bin"
            mask_path = mask_dir / f"{frame_id}.npy"
            if not bin_path.exists():
                raise FileNotFoundError(bin_path)
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)

            coords_raw = read_kitti_bin(bin_path)
            num_points = len(coords_raw)
            labels = np.load(mask_path)[:num_points]
            fg_mask = np.isin(labels, FG_CLASSES)
            coords_mm = np.round(coords_raw.astype(np.float64) * 1000).astype(np.int32)
            coords_scaled = coords_mm - coords_mm.min(axis=0)
            coords_fg = coords_scaled[fg_mask]
            coords_bg = coords_scaled[~fg_mask]
            jucp_label = int(labels_by_frame.get(frame_id, 0))
            scale_fg, scale_bg = combos[jucp_label]
            fg_stats = encode_subset(frame_id, f"router_r{rate_id}_fg", coords_fg, scale_fg, tmp_dir, cfg_path)
            bg_stats = encode_subset(frame_id, f"router_r{rate_id}_bg", coords_bg, scale_bg, tmp_dir, cfg_path)
            total_bits = fg_stats["bits"] + bg_stats["bits"]
            gpcc_enc = fg_stats["enc_time"] + bg_stats["enc_time"]
            seg_time = float(seg_times.get(frame_id, 0.0))
            detail_rows.append({
                "filename": frame_id,
                "rate_id": rate_id,
                "threshold": threshold,
                "jucp_label": jucp_label,
                "posQ_fg": scale_fg,
                "posQ_bg": scale_bg,
                "num_points": num_points,
                "fg_points": int(fg_mask.sum()),
                "bg_points": int((~fg_mask).sum()),
                "bits": total_bits,
                "bpp": round(total_bits / num_points, 6) if num_points else 0.0,
                "seg_time": round(seg_time, 6),
                "fg_enc_time": round(fg_stats["enc_time"], 6),
                "bg_enc_time": round(bg_stats["enc_time"], 6),
                "gpcc_enc_time": round(gpcc_enc, 6),
                "enc_time": round(seg_time + gpcc_enc, 6),
                "dec_time": round(fg_stats["dec_time"] + bg_stats["dec_time"], 6),
            })
            progress.update(1)
    progress.close()

    detail_csv = result_dir / "router_all_details.csv"
    with open(detail_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    avg_rows = []
    for rate_id, threshold, _ in jobs:
        rows = [r for r in detail_rows if int(r["rate_id"]) == rate_id]
        total_bits = sum(int(r["bits"]) for r in rows)
        total_points = sum(int(r["num_points"]) for r in rows)
        avg_rows.append({
            "rate_id": rate_id,
            "threshold": threshold,
            "num_frames": len(rows),
            "total_points": total_points,
            "total_bits": total_bits,
            "bpp": round(total_bits / total_points, 6) if total_points else 0.0,
            "seg_time": round(sum(float(r["seg_time"]) for r in rows) / len(rows), 6),
            "fg_enc_time": round(sum(float(r["fg_enc_time"]) for r in rows) / len(rows), 6),
            "bg_enc_time": round(sum(float(r["bg_enc_time"]) for r in rows) / len(rows), 6),
            "gpcc_enc_time": round(sum(float(r["gpcc_enc_time"]) for r in rows) / len(rows), 6),
            "enc_time": round(sum(float(r["enc_time"]) for r in rows) / len(rows), 6),
            "dec_time": round(sum(float(r["dec_time"]) for r in rows) / len(rows), 6),
        })

    avg_csv = result_dir / "router_average_results.csv"
    with open(avg_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(avg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(avg_rows)
    print(f"Detail CSV: {detail_csv}")
    print(f"Average CSV: {avg_csv}")


if __name__ == "__main__":
    main()
