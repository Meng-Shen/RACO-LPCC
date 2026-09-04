#!/usr/bin/env python3
"""Wait for prepared infos, then smoke-test production detector datasets."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import DATASETS

from export_nuscenes_centerpoint_quant import make_dataset_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    base = Path(args.base).resolve()
    data = base / "data" / "nuscenes"
    out = base / "preflight" / "dataset_after_prepare"
    out.mkdir(parents=True, exist_ok=True)
    marker = data / ".xyz_singleframe_prepared"
    while not marker.is_file():
        time.sleep(args.poll_seconds)
    try:
        cfg = Config.fromfile(
            str(base / "configs" / "centerpoint" /
                "centerpoint_voxel01_xyz_singleframe_recovery_12e_nus-3d.py")
        )
        init_default_scope("mmdet3d")
        train = DATASETS.build(cfg.train_dataloader.dataset)
        train.full_init()
        train_item = train[0]
        val = DATASETS.build(cfg.val_dataloader.dataset)
        val.full_init()
        val_item = val[0]
        export_train = DATASETS.build(make_dataset_cfg(
            cfg, str(data), "train", 64.0, True
        ))
        export_train.full_init()
        export_item = export_train[0]
        for name, item in (
            ("train", train_item), ("val", val_item),
            ("loss_export", export_item),
        ):
            points = item["inputs"]["points"]
            if points.shape[-1] != 3 or len(points) == 0:
                raise RuntimeError(f"Invalid {name} points: {tuple(points.shape)}")
        result = {
            "status": "PASS",
            "train_samples": len(train),
            "val_samples": len(val),
            "train_points": list(train_item["inputs"]["points"].shape),
            "val_points": list(val_item["inputs"]["points"].shape),
            "loss_export_points": list(export_item["inputs"]["points"].shape),
        }
        (out / "PASS.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    except Exception:
        (out / "FAILED.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
