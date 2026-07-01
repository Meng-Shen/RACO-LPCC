#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
from pathlib import Path

import pandas as pd

import _init_path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

from new_split import parse_quant_map


CLASSES = ["Car", "Pedestrian", "Cyclist"]


def combo_dir_name(combo_idx, scale_fg, scale_bg):
    return f"combo_{combo_idx}_fg_{scale_fg:.6f}_bg_{scale_bg:.6f}"


def load_combo_annos(eval_dir, quant_map):
    eval_dir = Path(eval_dir)
    combo_annos = {}
    for combo_idx, (scale_fg, scale_bg) in enumerate(quant_map):
        pkl_path = eval_dir / combo_dir_name(combo_idx, scale_fg, scale_bg) / "result.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Missing combo result.pkl: {pkl_path}")
        with open(pkl_path, "rb") as f:
            combo_annos[combo_idx] = pickle.load(f)
    return combo_annos


def normalize_frame_id(x):
    return str(x).strip().zfill(6)


def load_label_csv(path):
    df = pd.read_csv(path, dtype={"frame_id": str})
    if "frame_id" not in df.columns or "jucp_label" not in df.columns:
        raise KeyError(f"{path} must contain frame_id and jucp_label")
    return {normalize_frame_id(row.frame_id): int(row.jucp_label) for row in df.itertuples(index=False)}


def parse_ap_dict(ap_dict):
    return {
        "Car_3d_AP_R40_moderate": float(ap_dict.get("Car_3d/moderate_R40", 0.0)),
        "Pedestrian_3d_AP_R40_moderate": float(ap_dict.get("Pedestrian_3d/moderate_R40", 0.0)),
        "Cyclist_3d_AP_R40_moderate": float(ap_dict.get("Cyclist_3d/moderate_R40", 0.0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate adaptive router AP by selecting existing combo result.pkl annos.")
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--eval_dir", required=True, help="Directory containing combo_*/result.pkl from test_split.py")
    parser.add_argument("--quant_map", required=True)
    parser.add_argument("--manifest", required=True, help="router_manifest.json from export_router_jucp.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--save_mixed_pkls_dir", default=None)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
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
    frame_ids = [normalize_frame_id(info["point_cloud"]["lidar_idx"]) for info in dataset.kitti_infos]

    quant_map = parse_quant_map(args.quant_map)
    combo_annos = load_combo_annos(args.eval_dir, quant_map)
    for combo_idx, annos in combo_annos.items():
        if len(annos) != len(frame_ids):
            raise ValueError(
                f"combo_{combo_idx} result.pkl has {len(annos)} frames, "
                f"but dataset config has {len(frame_ids)} frames"
            )
        for idx, frame_id in enumerate(frame_ids):
            anno_frame = normalize_frame_id(annos[idx].get("frame_id", ""))
            if anno_frame != frame_id:
                raise ValueError(
                    f"Frame mismatch in combo_{combo_idx} at index {idx}: "
                    f"result.pkl frame_id={anno_frame}, dataset frame_id={frame_id}. "
                    "Use result.pkl files generated with the same cfg/split."
                )
    manifest = json.loads(Path(args.manifest).read_text())
    if args.save_mixed_pkls_dir:
        Path(args.save_mixed_pkls_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for item in manifest["label_csvs"]:
        rate_id = int(item["rate_id"])
        threshold = item.get("threshold", "")
        label_by_frame = load_label_csv(item["path"])
        mixed_annos = []
        for idx, frame_id in enumerate(frame_ids):
            if frame_id not in label_by_frame:
                raise KeyError(
                    f"Missing router label for frame_id={frame_id} in {item['path']}. "
                    "Generate router labels with the same split used by cfg/result.pkl."
                )
            label = label_by_frame[frame_id]
            if label not in combo_annos:
                raise ValueError(f"Invalid jucp_label={label} for frame_id={frame_id}")
            mixed_annos.append(combo_annos[label][idx])

        _, ap_dict = kitti_eval.get_official_eval_result(gt_annos, mixed_annos, CLASSES)
        row = {
            "rate_id": rate_id,
            "threshold": threshold,
            "label_csv": item["path"],
            **parse_ap_dict(ap_dict),
        }
        rows.append(row)

        if args.save_mixed_pkls_dir:
            with open(Path(args.save_mixed_pkls_dir) / f"router_rate_{rate_id}_result.pkl", "wb") as f:
                pickle.dump(mixed_annos, f)

    rows = sorted(rows, key=lambda x: int(x["rate_id"]))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Router AP CSV from existing result.pkl files: {args.out}")


if __name__ == "__main__":
    main()
