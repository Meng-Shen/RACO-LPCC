"""Install the small compatibility hooks required by geometry-only inputs.

The detector, voxel encoder, point heads, and dataset implementations remain
in OpenPCDet.  These wrappers call those implementations and only handle edge
cases introduced by very coarse XYZ-only point clouds.
"""

from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prepend_framework_paths() -> None:
    root = project_root()
    for path in (root, root / "OpenPCDet", root / "OpenPCDet" / "tools"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def _adapt_weight(key, value, target_shape):
    """Return a geometry-only checkpoint tensor in the model's layout."""
    if not hasattr(value, "shape") or len(value.shape) != len(target_shape):
        return None

    candidates = [value]
    if value.dim() == 5:
        candidates.extend((value.transpose(-1, -2), value.permute(4, 0, 1, 2, 3)))

    for candidate in candidates:
        if tuple(candidate.shape) == tuple(target_shape):
            return candidate.contiguous()

        mismatch = [
            dim for dim, (source, target) in enumerate(zip(candidate.shape, target_shape))
            if source != target
        ]
        if len(mismatch) != 1:
            continue
        dim = mismatch[0]

        if candidate.shape[dim] == 4 and target_shape[dim] == 3:
            return candidate.narrow(dim, 0, 3).contiguous()
        if (key.endswith("backbone_3d.FP_modules.0.mlp.0.weight")
                and candidate.shape[dim] == 257 and target_shape[dim] == 256):
            return candidate.narrow(dim, 0, 256).contiguous()
        if (key.endswith("vfe.pfn_layers.0.linear.weight")
                and candidate.shape[dim] == 10 and target_shape[dim] == 9):
            import torch
            xyz = candidate.narrow(dim, 0, 3)
            derived_xyz = candidate.narrow(dim, 4, 6)
            return torch.cat((xyz, derived_xyz), dim=dim).contiguous()
    return None


def install_openpcdet_compat() -> None:
    """Install idempotent wrappers around upstream OpenPCDet methods."""
    _prepend_framework_paths()

    import numpy as np
    from pcdet.datasets.processor.data_processor import DataProcessor
    from pcdet.models.backbones_3d.pfe.voxel_set_abstraction import (
        VoxelSetAbstraction,
    )
    from pcdet.models.detectors.detector3d_template import Detector3DTemplate

    if not getattr(DataProcessor, "_raco_coarse_sampling_compat", False):
        original_sample_points = DataProcessor.sample_points

        def sample_points_with_coarse_padding(self, data_dict=None, config=None):
            if data_dict is not None:
                requested = config.NUM_POINTS[self.mode]
                points = data_dict["points"]
                if requested > len(points) and requested - len(points) > len(points):
                    if len(points) == 0:
                        raise ValueError("cannot sample from an empty point cloud")
                    extra = np.random.choice(
                        np.arange(len(points), dtype=np.int32),
                        requested - len(points),
                        replace=True,
                    )
                    data_dict["points"] = np.concatenate((points, points[extra]), axis=0)
            return original_sample_points(self, data_dict=data_dict, config=config)

        DataProcessor.sample_points = sample_points_with_coarse_padding
        DataProcessor._raco_coarse_sampling_compat = True

    if not getattr(VoxelSetAbstraction, "_raco_xyz_only_compat", False):
        original_aggregate = VoxelSetAbstraction.aggregate_keypoint_features_from_one_source

        def aggregate_xyz_only(*args, **kwargs):
            if "xyz_features" in kwargs and kwargs["xyz_features"] is None:
                xyz = kwargs["xyz"]
                kwargs["xyz_features"] = xyz.new_empty((xyz.shape[0], 0))
            return original_aggregate(*args, **kwargs)

        VoxelSetAbstraction.aggregate_keypoint_features_from_one_source = staticmethod(
            aggregate_xyz_only
        )
        VoxelSetAbstraction._raco_xyz_only_compat = True

    if not getattr(Detector3DTemplate, "_raco_geometry_checkpoint_compat", False):
        original_load_state_dict = Detector3DTemplate._load_state_dict

        def load_geometry_state_dict(self, model_state_disk, *, strict=True):
            target_state = self.state_dict()
            adapted_state = dict(model_state_disk)
            for key, value in model_state_disk.items():
                if key not in target_state or value.shape == target_state[key].shape:
                    continue
                adapted = _adapt_weight(key, value, target_state[key].shape)
                if adapted is not None:
                    adapted_state[key] = adapted
            return original_load_state_dict(self, adapted_state, strict=strict)

        Detector3DTemplate._load_state_dict = load_geometry_state_dict
        Detector3DTemplate._raco_geometry_checkpoint_compat = True


__all__ = ["install_openpcdet_compat", "project_root"]
