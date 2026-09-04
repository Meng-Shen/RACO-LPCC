#!/usr/bin/env python3
"""Build PV-RCNN and verify complete geometry-only checkpoint loading on CPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "OpenPCDet",
             PROJECT_ROOT / "OpenPCDet" / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from integrations.openpcdet import install_openpcdet_compat

install_openpcdet_compat()

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "integrations" / "openpcdet" / "configs" /
        "kitti_models" / "pv_rcnn_fov_geometry.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "OpenPCDet" / "tools" / "ckpt" /
        "model_non_reflectance.pth",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()

    old_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT / "OpenPCDet" / "tools")
        cfg_from_yaml_file(str(config_path), cfg)
    finally:
        os.chdir(old_cwd)

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=None,
        training=False,
    )
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model_state, loaded_state = model._load_state_dict(
        checkpoint["model_state"], strict=False)
    missing = sorted(set(model_state) - set(loaded_state))
    result = {
        "model": cfg.MODEL.NAME,
        "checkpoint": str(checkpoint_path),
        "model_tensors": len(model_state),
        "loaded_tensors": len(loaded_state),
        "missing_tensors": missing,
    }
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
