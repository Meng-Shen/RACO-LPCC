"""Differentiable OpenPCDet adapter for direct decoded XYZ supervision.

OpenPCDet normally voxelizes NumPy points before the model is called.  That
would sever the coordinate-restoration gradient.  This adapter keeps the
discrete voxel assignment, but constructs voxel values from the predicted XYZ
tensor with ``index_add``.  The detector's raw input remains
``batch_dict['points'][:, 1:4] == decoded_xyz``; no residual is appended as a
feature.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def _norm_frame_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def _freeze_detector(model: nn.Module) -> None:
    """Keep detector weights fixed while retaining input-coordinate gradients."""

    model.train()
    for module in model.modules():
        if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout)):
            module.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _differentiable_voxelize(
    xyz: torch.Tensor,
    point_cloud_range: torch.Tensor,
    voxel_size: torch.Tensor,
    max_points_per_voxel: int,
    max_voxels: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voxelize XYZ with detached integer assignment and differentiable values.

    Returns ``voxels, voxel_coords_zyx, voxel_num_points, valid_indices``.
    ``voxel_coords_zyx`` is discrete by design; ``voxels`` is built from the
    original floating-point XYZ tensor and therefore carries gradients.
    """

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    xyz_min = point_cloud_range[:3]
    xyz_max = point_cloud_range[3:6]
    valid = torch.all((xyz >= xyz_min) & (xyz < xyz_max), dim=1)
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError(
            "decoded points contain no point in the detector range"
        )

    xyz_valid = xyz[valid_indices]
    grid_xyz = torch.floor(
        (xyz_valid - xyz_min) / voxel_size
    ).to(torch.long)
    grid_size = torch.ceil((xyz_max - xyz_min) / voxel_size).to(torch.long)
    linear = (
        grid_xyz[:, 0]
        + grid_size[0] * (grid_xyz[:, 1] + grid_size[1] * grid_xyz[:, 2])
    )
    # A stable sort groups equal voxel keys while retaining point order within
    # each voxel. OpenPCDet/spconv assigns voxel ids in first-occurrence order,
    # so reorder the groups by their first input point before max_voxels clips.
    _, sort_order = torch.sort(linear, stable=True)
    sorted_linear = linear[sort_order]
    unique_linear, counts = torch.unique_consecutive(sorted_linear, return_counts=True)
    voxel_count_all = int(unique_linear.numel())
    sorted_group_ids = torch.repeat_interleave(
        torch.arange(voxel_count_all, device=xyz.device, dtype=torch.long), counts
    )
    starts = torch.cumsum(counts, dim=0) - counts
    rank = torch.arange(sorted_linear.numel(), device=xyz.device) - torch.repeat_interleave(
        starts, counts
    )
    first_input_index = sort_order[starts]
    old_group_order = torch.argsort(first_input_index)
    old_to_new = torch.empty_like(old_group_order)
    old_to_new[old_group_order] = torch.arange(
        voxel_count_all, device=xyz.device, dtype=torch.long
    )
    group_ids = old_to_new[sorted_group_ids]
    keep = rank < int(max_points_per_voxel)
    keep &= group_ids < int(max_voxels)
    kept_group = group_ids[keep]
    kept_rank = rank[keep]
    kept_values = xyz_valid[sort_order[keep]]
    voxel_count = min(voxel_count_all, int(max_voxels))

    flat = xyz.new_zeros((voxel_count * int(max_points_per_voxel), 3))
    flat_index = kept_group * int(max_points_per_voxel) + kept_rank
    flat = flat.index_add(0, flat_index, kept_values)
    voxels = flat.view(voxel_count, int(max_points_per_voxel), 3)

    retained_old_groups = old_group_order[:voxel_count]
    counts_kept = counts[retained_old_groups].clamp_max(
        int(max_points_per_voxel)
    ).to(torch.int32)
    first_sorted = first_input_index[retained_old_groups]
    coords_xyz = grid_xyz[first_sorted]
    voxel_coords_zyx = coords_xyz[:, [2, 1, 0]].to(torch.int32)
    return voxels, voxel_coords_zyx, counts_kept, valid_indices


class CoordinateDetectorLoss:
    """Frozen PV-RCNN loss whose differentiable input is decoded XYZ."""

    def __init__(
        self,
        cfg_file: str | Path,
        checkpoint: str | Path,
        device: torch.device | str = "cuda",
        max_frames: int | None = None,
        split: str = "train",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda" and torch.cuda.is_available():
            # The supplied OpenPCDet checkpoint and custom ops are CUDA-first.
            raise ValueError("CoordinateDetectorLoss requires a CUDA device for model_non_reflectance")

        cfg_file = Path(cfg_file).resolve()
        checkpoint = Path(checkpoint).resolve()
        if not cfg_file.is_file():
            raise FileNotFoundError(cfg_file)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        # Expected layout: RACO-LPCC/OpenPCDet/tools/<config>, and this file
        # is normally copied into RENO.  Resolve the repository from config.
        self.openpcdet_tools = cfg_file.parents[2]
        if not (self.openpcdet_tools / "_init_path.py").is_file():
            # The canonical config lives in OpenPCDet/tools/cfgs/..., whose
            # parents[2] is OpenPCDet.  Keep a clear fallback for overrides.
            for parent in [cfg_file.parent, *cfg_file.parents]:
                candidate = parent / "_init_path.py"
                if candidate.is_file():
                    self.openpcdet_tools = parent
                    break
        if str(self.openpcdet_tools) not in sys.path:
            sys.path.insert(0, str(self.openpcdet_tools))
        raco_root = self.openpcdet_tools.parent
        if str(raco_root) not in sys.path:
            sys.path.insert(0, str(raco_root))

        import _init_path  # noqa: F401,E402
        from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
        from pcdet.datasets import build_dataloader  # noqa: E402
        from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
        from pcdet.utils import common_utils  # noqa: E402

        self.cfg = cfg
        launch_dir = Path.cwd()
        os.chdir(self.openpcdet_tools)
        try:
            cfg_from_yaml_file(str(cfg_file), cfg)
            cfg.TAG = cfg_file.stem
            cfg.EXP_GROUP_PATH = "/".join(str(cfg_file).split("/")[1:-1])
            # build_dataloader(training=False) is intentional: it disables
            # random data augmentation, but the evaluation-mode dataset would
            # otherwise load the val infos.  Point restoration training needs
            # the requested split's labels without augmentation.
            if split:
                cfg.DATA_CONFIG.DATA_SPLIT["test"] = str(split)
                cfg.DATA_CONFIG.INFO_PATH["test"] = [f"kitti_infos_{split}.pkl"]
            logger = common_utils.create_logger()
            dataset, _, _ = build_dataloader(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                batch_size=1,
                dist=False,
                workers=0,
                logger=logger,
                training=False,
            )
            dataset_root = Path(dataset.root_path).resolve()
        finally:
            os.chdir(launch_dir)

        # The config uses paths relative to OpenPCDet/tools.  Normalize them
        # before retaining the dataset for repeated __getitem__ calls.
        dataset.root_path = dataset_root
        dataset.root_split_path = dataset.root_path / (
            "training" if dataset.split != "test" else "testing"
        )
        self.dataset = dataset
        self.load_data_to_gpu = load_data_to_gpu
        self.fov_points_only = bool(cfg.DATA_CONFIG.FOV_POINTS_ONLY)
        self._original_get_lidar = dataset.__class__.get_lidar
        self.frame_to_index = {
            _norm_frame_id(info["point_cloud"]["lidar_idx"]): index
            for index, info in enumerate(dataset.kitti_infos)
        }
        if max_frames is not None:
            self.frame_ids = list(self.frame_to_index)[:int(max_frames)]
        else:
            self.frame_ids = list(self.frame_to_index)

        model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=dataset,
        )
        model.load_params_from_file(filename=str(checkpoint), logger=logger, to_cpu=False)
        self.model = model.to(self.device)
        _freeze_detector(self.model)

        self.point_cloud_range = torch.tensor(
            dataset.point_cloud_range, dtype=torch.float32, device=self.device
        )
        self.voxel_size = torch.tensor(
            dataset.voxel_size, dtype=torch.float32, device=self.device
        )
        self.max_points_per_voxel = int(
            cfg.DATA_CONFIG.DATA_PROCESSOR[-1].MAX_POINTS_PER_VOXEL
        )
        # The detector is built in test/preprocessing mode, matching the
        # existing model_non_reflectance evaluation path.
        self.max_voxels = int(
            cfg.DATA_CONFIG.DATA_PROCESSOR[-1].MAX_NUMBER_OF_VOXELS["test"]
        )

    def indices_for_frames(self, frame_ids: Sequence[str]) -> List[int]:
        missing = [_norm_frame_id(frame) for frame in frame_ids if _norm_frame_id(frame) not in self.frame_to_index]
        if missing:
            raise KeyError(f"frames missing from detector infos: {missing}")
        return [self.frame_to_index[_norm_frame_id(frame)] for frame in frame_ids]

    def source_points(self, dataset_index: int) -> np.ndarray:
        """Return the same range/FOV-filtered XYZ used by the detector."""

        data = self.dataset[int(dataset_index)]
        points = np.asarray(data["points"])
        if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
            raise ValueError(f"empty or malformed detector source at index {dataset_index}")
        return np.ascontiguousarray(points[:, :3].astype(np.float32, copy=True))

    def _filter_predicted_xyz(
        self,
        xyz: torch.Tensor,
        base: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the exact FOV and point mask from OpenPCDet's dataset path.

        KITTI's ``mask_points_by_range`` deliberately checks x/y only and
        uses inclusive upper bounds.  The voxel generator applies the full
        half-open x/y/z range separately.  Keeping those two masks distinct is
        important because PV-RCNN's raw-point branch sees the former set.
        """

        valid = (
            (xyz[:, 0] >= self.point_cloud_range[0])
            & (xyz[:, 0] <= self.point_cloud_range[3])
            & (xyz[:, 1] >= self.point_cloud_range[1])
            & (xyz[:, 1] <= self.point_cloud_range[4])
        )
        if self.fov_points_only:
            calib = base.get("calib")
            image_shape = base.get("image_shape")
            if calib is None or image_shape is None:
                raise ValueError(
                    "FOV filtering requires calib and image_shape"
                )
            xyz_np = xyz.detach().cpu().numpy()
            pts_rect = calib.lidar_to_rect(xyz_np)
            fov_np = self.dataset.get_fov_flag(
                pts_rect, np.asarray(image_shape), calib
            )
            fov = torch.from_numpy(
                np.asarray(fov_np, dtype=np.bool_)
            ).to(xyz.device)
            valid &= fov
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError(
                "decoded points contain no detector-valid point"
            )
        return xyz[indices], indices

    def _batch_from_xyz(self, xyz: torch.Tensor, dataset_index: int) -> Dict[str, Any]:
        """Replace all geometry tensors while retaining this frame's labels."""

        base = copy.deepcopy(self.dataset[int(dataset_index)])
        detector_xyz, _ = self._filter_predicted_xyz(
            xyz, base
        )
        voxels, voxel_coords, voxel_num_points, voxel_valid_indices = _differentiable_voxelize(
            xyz=detector_xyz,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            max_points_per_voxel=self.max_points_per_voxel,
            max_voxels=self.max_voxels,
        )
        # ``points`` retains the exact FOV + x/y range-filtered set consumed
        # by PV-RCNN's raw-point branch.  Voxel values use the stricter 3-D
        # half-open range applied by spconv, represented by
        # ``voxel_valid_indices``.

        # This is the detector's direct point-coordinate input.  Column 0 is
        # the OpenPCDet batch id; columns 1:4 are the decoded absolute XYZ.
        points = torch.cat(
            [
                torch.zeros(
                    (detector_xyz.shape[0], 1),
                    device=xyz.device,
                    dtype=xyz.dtype,
                ),
                detector_xyz,
            ],
            dim=1,
        )
        batch_coords = torch.cat(
            [torch.zeros((voxel_coords.shape[0], 1), device=xyz.device, dtype=torch.int32), voxel_coords],
            dim=1,
        )

        batch: Dict[str, Any] = {}
        for key, value in base.items():
            if key in {"points", "voxels", "voxel_coords", "voxel_num_points", "gt_boxes"}:
                continue
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
                if tensor.dtype == torch.float64:
                    tensor = tensor.float()
                batch[key] = tensor.to(self.device)
            else:
                batch[key] = value

        batch["points"] = points
        batch["voxels"] = voxels
        batch["voxel_coords"] = batch_coords
        batch["voxel_num_points"] = voxel_num_points
        gt_boxes = base.get("gt_boxes")
        if gt_boxes is None:
            raise ValueError("detector source has no gt_boxes for training loss")
        gt_tensor = torch.from_numpy(np.asarray(gt_boxes)).float().to(self.device)
        batch["gt_boxes"] = gt_tensor.unsqueeze(0)
        batch["batch_size"] = 1
        batch["use_lead_xyz"] = True
        # The PFE uses raw_points and derives its coordinates from this exact
        # tensor, so do not substitute voxel-centre features here.
        return batch

    def __call__(self, decoded_xyz: torch.Tensor, dataset_index: int) -> torch.Tensor:
        if decoded_xyz.ndim != 2 or decoded_xyz.shape[1] != 3:
            raise ValueError("decoded_xyz must have shape (N, 3) in metres")
        batch = self._batch_from_xyz(decoded_xyz, dataset_index)
        ret_dict, _, _ = self.model(batch)
        loss = ret_dict["loss"] if isinstance(ret_dict, dict) else ret_dict
        return loss.mean()

    def standard_loss(
        self,
        decoded_xyz: torch.Tensor,
        dataset_index: int,
    ) -> torch.Tensor:
        """Evaluate with OpenPCDet's exact non-differentiable preprocessing."""

        if decoded_xyz.ndim != 2 or decoded_xyz.shape[1] != 3:
            raise ValueError(
                "decoded_xyz must have shape (N, 3) in metres"
            )
        info = self.dataset.kitti_infos[int(dataset_index)]
        frame_id = _norm_frame_id(
            info["point_cloud"]["lidar_idx"]
        )
        points_np = decoded_xyz.detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        points4 = np.concatenate(
            [
                points_np,
                np.zeros((len(points_np), 1), dtype=np.float32),
            ],
            axis=1,
        )
        dataset_class = self.dataset.__class__
        original = dataset_class.get_lidar

        def patched_get_lidar(instance, sample_idx):
            if _norm_frame_id(sample_idx) == frame_id:
                return points4
            return self._original_get_lidar(instance, sample_idx)

        dataset_class.get_lidar = patched_get_lidar
        try:
            data_dict = self.dataset[int(dataset_index)]
        finally:
            dataset_class.get_lidar = original
        batch = self.dataset.collate_batch([data_dict])
        self.load_data_to_gpu(batch)
        ret_dict, _, _ = self.model(batch)
        loss = (
            ret_dict["loss"]
            if isinstance(ret_dict, dict)
            else ret_dict
        )
        return loss.mean()
