#!/usr/bin/env python3
import argparse
import os
import csv
import pickle
import sys
from pathlib import Path
from contextlib import contextmanager

ROOT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = ROOT_DIR / "OpenPCDet" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import _init_path

from new_split import parse_quant_map


CLASSES = ["Car", "Pedestrian", "Cyclist"]
CLASS_TO_COL = {
    "Car": "Car",
    "Pedestrian": "Ped",
    "Cyclist": "Cyc",
}


@contextmanager
def pushd(path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute an oracle router AP-bpp upper bound from true per-frame AP sensitivity."
    )
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--eval_dir", required=True, help="Directory containing combo_*/result.pkl files.")
    parser.add_argument("--ap_csv", required=True, help="Per-frame AP sensitivity CSV from new_split.py.")
    parser.add_argument("--split_details_csv", required=True, help="split_all_details.csv from Split-GPCC.")
    parser.add_argument("--quant_map", required=True)
    parser.add_argument("--out_dir", default="point_pairs/oracle_router_fov")
    parser.add_argument("--objective", choices=CLASSES, default="Car")
    parser.add_argument(
        "--lambdas",
        default="0,0.00025,0.0005,0.001,0.002,0.004,0.008,0.016,0.032",
        help="Comma-separated Lagrange multipliers for AP_drop + lambda * per-frame bpp.",
    )
    parser.add_argument("--save_mixed_pkls_dir", default=None)
    return parser.parse_args()


def norm_frame_id(value):
    return str(value).strip().zfill(6)


def combo_dir_name(combo_idx, scale_fg, scale_bg):
    return f"combo_{combo_idx}_fg_{scale_fg:.6f}_bg_{scale_bg:.6f}"


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_lambdas(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("--lambdas must contain at least one value")
    return values


def load_dataset(cfg_file):
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader

    cfg_path = Path(cfg_file)
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path) if cfg_path.exists() else (TOOLS_DIR / cfg_path)
    cfg_path = cfg_path.resolve()
    with pushd(TOOLS_DIR):
        cfg_from_yaml_file(str(cfg_path), cfg)
        dataset, _, _ = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=1,
            dist=False,
            workers=1,
            training=False,
            logger=None,
        )
        gt_annos = [info["annos"] for info in dataset.kitti_infos]
        frame_ids = [norm_frame_id(info["point_cloud"]["lidar_idx"]) for info in dataset.kitti_infos]
    return gt_annos, frame_ids


def load_combo_annos(eval_dir, quant_map, frame_ids):
    eval_dir = Path(eval_dir)
    combo_annos = {}
    for combo_idx, (scale_fg, scale_bg) in enumerate(quant_map):
        pkl_path = eval_dir / combo_dir_name(combo_idx, scale_fg, scale_bg) / "result.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Missing combo result.pkl: {pkl_path}")
        with open(pkl_path, "rb") as f:
            annos = pickle.load(f)
        if len(annos) != len(frame_ids):
            raise ValueError(f"{pkl_path} has {len(annos)} frames, expected {len(frame_ids)}")
        for idx, frame_id in enumerate(frame_ids):
            anno_frame = norm_frame_id(annos[idx].get("frame_id", ""))
            if anno_frame != frame_id:
                raise ValueError(
                    f"Frame mismatch in combo_{combo_idx} at index {idx}: "
                    f"result.pkl frame_id={anno_frame}, dataset frame_id={frame_id}"
                )
        combo_annos[combo_idx] = annos
    return combo_annos


def load_ap_sensitivity(path, num_levels):
    table = {}
    for row in read_csv_rows(path):
        frame_id = norm_frame_id(row["frame_id"])
        table[frame_id] = row
        for level in range(num_levels):
            for short_name in ("Car", "Ped", "Cyc"):
                key = f"L{level}_{short_name}_AP"
                if key not in row:
                    raise KeyError(f"{path} missing column {key}")
    return table


def load_split_details(path):
    table = {}
    for row in read_csv_rows(path):
        frame_id = norm_frame_id(row.get("filename") or row.get("frame_id"))
        label = int(row.get("combo_id") or row["rate_id"])
        table[(frame_id, label)] = row
    return table


def ap_drop(ap_row, label, objective):
    short = CLASS_TO_COL[objective]
    base = float(ap_row[f"L0_{short}_AP"])
    current = float(ap_row[f"L{label}_{short}_AP"])
    return max(base - current, 0.0)


def frame_bpp(detail_row):
    bits = float(detail_row["bits"])
    num_points = float(detail_row["num_points"])
    return bits / num_points if num_points else 0.0


def choose_labels(frame_ids, ap_table, details, num_levels, objective, lam):
    labels = {}
    for frame_id in frame_ids:
        if frame_id not in ap_table:
            raise KeyError(f"Missing AP sensitivity row for frame_id={frame_id}")
        best_label = 0
        best_score = None
        for label in range(num_levels):
            key = (frame_id, label)
            if key not in details:
                raise KeyError(f"Missing Split-GPCC detail for frame={frame_id}, combo/label={label}")
            score = ap_drop(ap_table[frame_id], label, objective) + float(lam) * frame_bpp(details[key])
            if best_score is None or score < best_score:
                best_score = score
                best_label = label
        labels[frame_id] = best_label
    return labels


def parse_ap_dict(ap_dict):
    return {
        "Car_3d_AP_R40_moderate": float(ap_dict.get("Car_3d/moderate_R40", 0.0)),
        "Pedestrian_3d_AP_R40_moderate": float(ap_dict.get("Pedestrian_3d/moderate_R40", 0.0)),
        "Cyclist_3d_AP_R40_moderate": float(ap_dict.get("Cyclist_3d/moderate_R40", 0.0)),
    }


def aggregate_details(frame_ids, labels, details):
    rows = []
    for frame_id in frame_ids:
        row = dict(details[(frame_id, labels[frame_id])])
        row["oracle_label"] = labels[frame_id]
        rows.append(row)
    total_bits = sum(int(float(row["bits"])) for row in rows)
    total_points = sum(int(float(row["num_points"])) for row in rows)
    return rows, {
        "num_frames": len(rows),
        "total_points": total_points,
        "total_bits": total_bits,
        "bpp": round(total_bits / total_points, 6) if total_points else 0.0,
        "seg_time": round(sum(float(row.get("seg_time", 0.0)) for row in rows) / len(rows), 6),
        "fg_enc_time": round(sum(float(row.get("fg_enc_time", 0.0)) for row in rows) / len(rows), 6),
        "bg_enc_time": round(sum(float(row.get("bg_enc_time", 0.0)) for row in rows) / len(rows), 6),
        "gpcc_enc_time": round(sum(float(row.get("gpcc_enc_time", 0.0)) for row in rows) / len(rows), 6),
        "enc_time": round(sum(float(row.get("enc_time", 0.0)) for row in rows) / len(rows), 6),
        "dec_time": round(sum(float(row.get("dec_time", 0.0)) for row in rows) / len(rows), 6),
    }


def main():
    args = parse_args()
    from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quant_map = parse_quant_map(args.quant_map)
    lambdas = parse_lambdas(args.lambdas)
    gt_annos, frame_ids = load_dataset(args.cfg_file)
    combo_annos = load_combo_annos(args.eval_dir, quant_map, frame_ids)
    ap_table = load_ap_sensitivity(args.ap_csv, num_levels=len(quant_map))
    details = load_split_details(args.split_details_csv)

    if args.save_mixed_pkls_dir:
        Path(args.save_mixed_pkls_dir).mkdir(parents=True, exist_ok=True)

    curve_rows = []
    avg_rows = []
    all_detail_rows = []
    for rate_id, lam in enumerate(lambdas):
        labels = choose_labels(frame_ids, ap_table, details, len(quant_map), args.objective, lam)
        mixed_annos = [combo_annos[labels[frame_id]][idx] for idx, frame_id in enumerate(frame_ids)]
        _, ap_dict = kitti_eval.get_official_eval_result(gt_annos, mixed_annos, CLASSES)
        detail_rows, avg = aggregate_details(frame_ids, labels, details)

        label_counts = {label: 0 for label in range(len(quant_map))}
        for label in labels.values():
            label_counts[label] += 1

        ap_values = parse_ap_dict(ap_dict)
        row = {
            "rate_id": rate_id,
            "lambda": lam,
            "objective": args.objective,
            **avg,
            **ap_values,
            "label_counts": ";".join(f"L{label}:{count}" for label, count in sorted(label_counts.items())),
        }
        curve_rows.append(row)
        avg_rows.append({k: row[k] for k in (
            "rate_id", "lambda", "objective", "num_frames", "total_points", "total_bits",
            "bpp", "seg_time", "fg_enc_time", "bg_enc_time", "gpcc_enc_time", "enc_time", "dec_time",
            "label_counts",
        )})

        for detail in detail_rows:
            detail["rate_id"] = rate_id
            detail["lambda"] = lam
            detail["objective"] = args.objective
            all_detail_rows.append(detail)

        label_csv = out_dir / f"oracle_rate_{rate_id}_labels.csv"
        write_csv(label_csv, [
            {"frame_id": frame_id, "jucp_label": labels[frame_id], "rate_id": rate_id, "lambda": lam}
            for frame_id in frame_ids
        ])

        if args.save_mixed_pkls_dir:
            with open(Path(args.save_mixed_pkls_dir) / f"oracle_rate_{rate_id}_result.pkl", "wb") as f:
                pickle.dump(mixed_annos, f)

    fieldnames = [
        "rate_id", "lambda", "objective", "num_frames", "total_points", "total_bits",
        "bpp", "seg_time", "fg_enc_time", "bg_enc_time", "gpcc_enc_time", "enc_time", "dec_time",
        "Car_3d_AP_R40_moderate", "Pedestrian_3d_AP_R40_moderate", "Cyclist_3d_AP_R40_moderate",
        "label_counts",
    ]
    write_csv(out_dir / "oracle_router_curve.csv", curve_rows, fieldnames)
    write_csv(out_dir / "oracle_average_results.csv", avg_rows)
    write_csv(out_dir / "oracle_all_details.csv", all_detail_rows, list(all_detail_rows[0].keys()) if all_detail_rows else [])
    print(f"Oracle curve CSV: {out_dir / 'oracle_router_curve.csv'}")
    print(f"Oracle average CSV: {out_dir / 'oracle_average_results.csv'}")
    print(f"Oracle detail CSV: {out_dir / 'oracle_all_details.csv'}")


if __name__ == "__main__":
    main()
