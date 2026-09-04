#!/usr/bin/env python3
"""Smoke-test project integrations without training or writing artifacts."""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENPCDET_ROOT = PROJECT_ROOT / "OpenPCDet"
OPENPCDET_TOOLS = OPENPCDET_ROOT / "tools"
MMDET_ROOT = PROJECT_ROOT / "mmdetection3d"
ROUTER_ROOT = PROJECT_ROOT / "routing" / "lrproxy"
for path in (PROJECT_ROOT, OPENPCDET_ROOT, OPENPCDET_TOOLS, MMDET_ROOT, ROUTER_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def check_openpcdet():
    from integrations.openpcdet import install_openpcdet_compat

    install_openpcdet_compat()
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets.processor.data_processor import DataProcessor
    from pcdet.models.backbones_3d.pfe.voxel_set_abstraction import (
        VoxelSetAbstraction,
    )
    from pcdet.models.detectors.detector3d_template import Detector3DTemplate

    old_cwd = Path.cwd()
    try:
        os.chdir(OPENPCDET_TOOLS)
        cfg_from_yaml_file(
            str(PROJECT_ROOT / "integrations" / "openpcdet" / "configs" /
                "kitti_models" / "pv_rcnn_fov_geometry.yaml"),
            cfg,
        )
    finally:
        os.chdir(old_cwd)

    return {
        "model": cfg.MODEL.NAME,
        "data_path": str(cfg.DATA_CONFIG.DATA_PATH),
        "coarse_sampling_wrapper": bool(
            getattr(DataProcessor, "_raco_coarse_sampling_compat", False)),
        "xyz_only_wrapper": bool(
            getattr(VoxelSetAbstraction, "_raco_xyz_only_compat", False)),
        "checkpoint_wrapper": bool(
            getattr(Detector3DTemplate, "_raco_geometry_checkpoint_compat", False)),
    }


def check_mmdetection3d():
    from mmengine.config import Config

    config = Config.fromfile(
        str(PROJECT_ROOT / "integrations" / "mmdetection3d" / "configs" /
            "minkunet" /
            "minkunet34_w32_minkowski_geometry_8xb2-laser-polar-mix-3x_semantickitti.py"))
    from mmdet3d.registry import MODELS
    from mmdet3d.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    model = MODELS.build(config.model)

    return {
        "backbone": type(model.backbone).__name__,
        "in_channels": config.model.backbone.in_channels,
        "built_head": type(model.decode_head).__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def check_router():
    from lrproxy import (
        LRProxy,
    )

    return {
        "class": LRProxy.__name__,
        "constructor": str(inspect.signature(LRProxy)),
    }


def main():
    result = {
        "openpcdet": check_openpcdet(),
        "mmdetection3d": check_mmdetection3d(),
        "router": check_router(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
