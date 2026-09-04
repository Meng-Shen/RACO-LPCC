"""Current-scale sparse coordinate restoration with constant input features.

The network consumes only the active q-level coordinates already reconstructed
by G-PCC.  Every active voxel receives a scalar feature equal to one, so local
occupancy geometry is represented implicitly by sparse-coordinate adjacency.
No parent-scale tensor, occupancy code, FCG expansion, or side information is
used.  The scale-conditioned residual heads remain decoder-side operations and
add no payload to the G-PCC bitstream.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse import SparseTensor
from torchsparse import nn as spnn

Q_STEPS_MM = (2048, 1024, 512, 256, 128, 64)
N_BY_Q = {
    2048: 10,
    1024: 10,
    512: 7,
    256: 3,
    128: 1,
    64: 1,
}
N_MAX = max(N_BY_Q.values())
# Maximum absolute decoder-side displacement around each quantized anchor.
# The original model used q/2 for every scale.  This experiment keeps that
# contract except at q=64 mm, where movement is deliberately limited to 8 mm.
MAX_ABS_OFFSET_MM_BY_Q = {
    2048: 1024.0,
    1024: 512.0,
    512: 256.0,
    256: 128.0,
    128: 64.0,
    64: 8.0,
}
LEGACY_ARCHITECTURE = "reno_final_2q_to_q_target_heads"
SCALE_CONDITIONED_ARCHITECTURE = (
    "reno_final_2q_to_q_scale_conditioned_lite_v1"
)
CURRENT_SCALE_ONES_ARCHITECTURE = (
    "current_q_ones_sparse_scale_conditioned_v1"
)
Q_TO_SCALE_INDEX = {
    q_step_mm: index
    for index, q_step_mm in enumerate(Q_STEPS_MM)
}
DEFAULT_NND_PATH = Path("/public/DATA/sm/GRASP-Net/third_party/nndistance")


@dataclass
class DecoderTarget:
    """Current q-level geometry and full-resolution target for one frame."""

    origin_mm: np.ndarray
    anchor_coords: np.ndarray
    target_points_m: np.ndarray


@dataclass
class ResidualBatch:
    """Batched current-scale sparse input used by the network and losses."""

    input_coords: torch.Tensor
    input_features: torch.Tensor
    anchor_coords: torch.Tensor
    origins_mm: torch.Tensor
    anchor_batch: torch.Tensor
    target_points_m: List[torch.Tensor]
    q_step_mm: int
    n: int


def _sort_xyz_like_reno(coords: np.ndarray) -> np.ndarray:
    """Sort XYZ deterministically by z, then y, then x."""

    if len(coords) == 0:
        return coords
    order = np.lexsort((coords[:, 0], coords[:, 1], coords[:, 2]))
    return coords[order]


def build_decoder_target(
    points_xyz_m: np.ndarray,
    q_step_mm: int,
) -> DecoderTarget:
    """Build the q-level sparse coordinates available after G-PCC decoding.

    This mirrors the project's geometry quantization exactly:

    1. round source XYZ to integer millimetres;
    2. subtract the per-frame minimum offset stored in the bitstream;
    3. round by q_step_mm to obtain the decoded q-level anchors.

    The network sees these anchors directly and receives no explicit occupancy
    symbol.  Their sparse-coordinate pattern is the only geometry input.
    """

    points = np.asarray(points_xyz_m)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points_xyz_m must have shape (N, >=3)")
    if len(points) == 0:
        raise ValueError("a point cloud must contain at least one point")
    if q_step_mm <= 0:
        raise ValueError("q_step_mm must be positive")
    if not np.isfinite(points[:, :3]).all():
        raise ValueError("point cloud contains non-finite XYZ")

    coords_mm = np.rint(
        points[:, :3].astype(np.float64) * 1000.0
    ).astype(np.int64)
    origin_mm = coords_mm.min(axis=0)
    anchors_per_point = np.rint(
        (coords_mm - origin_mm[None, :]).astype(np.float64)
        / float(q_step_mm)
    ).astype(np.int64)
    anchor_coords = np.unique(anchors_per_point, axis=0)
    anchor_coords = _sort_xyz_like_reno(anchor_coords)

    return DecoderTarget(
        origin_mm=origin_mm,
        anchor_coords=anchor_coords.astype(np.int64, copy=False),
        target_points_m=np.ascontiguousarray(
            points[:, :3], dtype=np.float32
        ),
    )


def build_residual_batch(
    point_clouds_xyz_m: Sequence[np.ndarray],
    q_step_mm: int,
    n: int,
    device: torch.device | str,
) -> ResidualBatch:
    """Build q-level coordinates plus all-one features and retain targets."""

    if not point_clouds_xyz_m:
        raise ValueError("point_clouds_xyz_m cannot be empty")
    if n <= 0:
        raise ValueError("n must be positive")

    targets = [
        build_decoder_target(points, q_step_mm)
        for points in point_clouds_xyz_m
    ]
    input_coords: List[np.ndarray] = []
    input_features: List[np.ndarray] = []
    anchor_coords: List[np.ndarray] = []
    anchor_batch: List[np.ndarray] = []
    origins: List[np.ndarray] = []
    target_points: List[torch.Tensor] = []

    for batch_id, target in enumerate(targets):
        anchor_count = len(target.anchor_coords)
        current_coords = np.concatenate(
            [
                np.full(
                    (anchor_count, 1), batch_id, dtype=np.int64
                ),
                target.anchor_coords,
            ],
            axis=1,
        )
        input_coords.append(current_coords)
        input_features.append(
            np.ones((anchor_count, 1), dtype=np.float32)
        )
        anchor_coords.append(
            np.concatenate(
                [
                    np.full(
                        (anchor_count, 1), batch_id, dtype=np.int64
                    ),
                    target.anchor_coords,
                ],
                axis=1,
            )
        )
        anchor_batch.append(
            np.full((anchor_count,), batch_id, dtype=np.int64)
        )
        origins.append(target.origin_mm)
        target_points.append(
            torch.from_numpy(target.target_points_m).to(
                device=device, dtype=torch.float32
            )
        )

    return ResidualBatch(
        input_coords=torch.from_numpy(
            np.concatenate(input_coords)
        ).to(device=device, dtype=torch.int32),
        input_features=torch.from_numpy(
            np.concatenate(input_features)
        ).to(device=device, dtype=torch.float32),
        anchor_coords=torch.from_numpy(
            np.concatenate(anchor_coords)
        ).to(device=device, dtype=torch.int32),
        origins_mm=torch.from_numpy(np.stack(origins)).to(
            device=device, dtype=torch.float32
        ),
        anchor_batch=torch.from_numpy(
            np.concatenate(anchor_batch)
        ).to(device=device, dtype=torch.long),
        target_points_m=target_points,
        q_step_mm=int(q_step_mm),
        n=int(n),
    )


def build_inference_batch_from_anchors(
    anchor_coords_xyz: np.ndarray | torch.Tensor,
    origin_mm: np.ndarray | torch.Tensor,
    q_step_mm: int,
    n: int,
    device: torch.device | str,
) -> ResidualBatch:
    """Build the all-one sparse input directly from decoder integer outputs.

    G-PCC already produces unique integer coordinates and the frame origin.
    This path therefore performs no CPU re-quantization, sorting, or unique.
    """

    if q_step_mm <= 0 or n <= 0:
        raise ValueError("q_step_mm and n must be positive")
    anchors = torch.as_tensor(anchor_coords_xyz, device=device)
    if anchors.ndim != 2 or anchors.shape[1] != 3 or len(anchors) == 0:
        raise ValueError("anchor_coords_xyz must have shape (N, 3)")
    anchors = anchors.to(dtype=torch.int32)
    origin = torch.as_tensor(origin_mm, device=device, dtype=torch.float32)
    if origin.numel() != 3:
        raise ValueError("origin_mm must contain three values")
    origin = origin.reshape(1, 3)
    batch_ids_i32 = torch.zeros(
        (anchors.shape[0], 1), device=device, dtype=torch.int32
    )
    sparse_coords = torch.cat((batch_ids_i32, anchors), dim=1)
    batch_ids = torch.zeros(
        anchors.shape[0], device=device, dtype=torch.long
    )
    return ResidualBatch(
        input_coords=sparse_coords,
        input_features=torch.ones(
            (anchors.shape[0], 1), device=device, dtype=torch.float32
        ),
        anchor_coords=sparse_coords,
        origins_mm=origin,
        anchor_batch=batch_ids,
        target_points_m=[],
        q_step_mm=int(q_step_mm),
        n=int(n),
    )


def build_inference_batch_from_decoded_xyz(
    decoded_xyz_m: np.ndarray | torch.Tensor,
    q_step_mm: int,
    n: int,
    device: torch.device | str,
    origin_mm: np.ndarray | torch.Tensor | None = None,
) -> ResidualBatch:
    """Build inference input on GPU from already-decoded absolute XYZ.

    Unlike ``build_residual_batch``, this function assumes decoder geometry is
    already unique and skips all training-target construction and CPU unique.
    Passing the bitstream origin avoids even the per-axis GPU minimum.
    """

    xyz = torch.as_tensor(decoded_xyz_m, device=device, dtype=torch.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("decoded_xyz_m must have shape (N, 3)")
    coords_mm = torch.round(xyz * 1000.0)
    if origin_mm is None:
        origin = torch.amin(coords_mm, dim=0)
    else:
        origin = torch.as_tensor(origin_mm, device=device, dtype=torch.float32)
        if origin.numel() != 3:
            raise ValueError("origin_mm must contain three values")
        origin = origin.reshape(3)
    anchors = torch.round(
        (coords_mm - origin.reshape(1, 3)) / float(q_step_mm)
    ).to(dtype=torch.int32)
    return build_inference_batch_from_anchors(
        anchors,
        origin,
        q_step_mm,
        n,
        device,
    )


class CoordinateResidualNet(nn.Module):
    """All-one q-level sparse backbone with scale-conditioned restoration.

    Four stride-one sparse convolutions extract local geometry directly from
    the active q-level coordinate pattern.  The existing scale embedding,
    FiLM, low-rank expert, output calibration, and XYZ heads are retained.
    """

    def __init__(
        self,
        channels: int = 32,
        kernel_size: int = 3,
        n_max: int = N_MAX,
        scale_rank: int = 8,
    ) -> None:
        super().__init__()
        if channels != 32:
            raise ValueError(
                "coordinate residual heads require 32 input channels"
            )
        if n_max <= 0:
            raise ValueError("n_max must be positive")
        if scale_rank <= 0 or scale_rank > channels:
            raise ValueError(
                "scale_rank must be in [1, channels]"
            )
        self.channels = int(channels)
        self.n_max = int(n_max)
        self.scale_rank = int(scale_rank)

        self.feature_extractor = nn.Sequential(
            spnn.Conv3d(1, channels, kernel_size),
            spnn.ReLU(True),
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
        )

        # Per-scale conditioning.  All branches are identity at initialization
        # so a decoder-aligned legacy checkpoint can be migrated without
        # changing its predictions before fine-tuning.
        scale_count = len(Q_STEPS_MM)
        self.scale_embedding = nn.Embedding(
            scale_count, channels
        )
        self.scale_film = nn.Embedding(
            scale_count, 2 * channels
        )
        self.scale_expert_down = nn.Parameter(
            torch.empty(
                scale_count, channels, self.scale_rank
            )
        )
        self.scale_expert_up = nn.Parameter(
            torch.zeros(
                scale_count, self.scale_rank, channels
            )
        )
        self.scale_head_affine = nn.Embedding(
            scale_count, self.n_max * 3 * 2
        )

        self.head_x = spnn.Conv3d(
            channels, self.n_max, kernel_size
        )
        self.head_y = spnn.Conv3d(
            channels, self.n_max, kernel_size
        )
        self.head_z = spnn.Conv3d(
            channels, self.n_max, kernel_size
        )
        self._reset_scale_parameters()

    def _reset_scale_parameters(self) -> None:
        nn.init.zeros_(self.scale_embedding.weight)
        nn.init.zeros_(self.scale_film.weight)
        nn.init.normal_(
            self.scale_expert_down, mean=0.0, std=0.02
        )
        nn.init.zeros_(self.scale_expert_up)
        nn.init.zeros_(self.scale_head_affine.weight)

    @staticmethod
    def _scale_index(q_step_mm: int) -> int:
        try:
            return Q_TO_SCALE_INDEX[int(q_step_mm)]
        except KeyError as error:
            raise ValueError(
                f"unsupported q_step_mm={q_step_mm}; "
                f"expected one of {Q_STEPS_MM}"
            ) from error

    def load_coordinate_checkpoint(
        self,
        checkpoint: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> tuple[str, int]:
        """Resume this model or import unchanged downstream tensors."""

        state = torch.load(
            str(checkpoint), map_location=map_location
        )
        if not isinstance(state, dict) or (
            "model_state_dict" not in state
        ):
            raise TypeError(
                "coordinate checkpoint must contain model_state_dict"
            )
        architecture = str(state.get("architecture", ""))
        if architecture not in {
            SCALE_CONDITIONED_ARCHITECTURE,
            CURRENT_SCALE_ONES_ARCHITECTURE,
        }:
            raise ValueError(
                "unsupported coordinate checkpoint architecture: "
                f"{architecture!r}"
            )
        source = {
            key.removeprefix("module."): value
            for key, value in state["model_state_dict"].items()
        }
        current = self.state_dict()
        compatible = {
            key: value
            for key, value in source.items()
            if key in current
            and tuple(value.shape) == tuple(current[key].shape)
        }
        if architecture == CURRENT_SCALE_ONES_ARCHITECTURE:
            required = list(current)
        else:
            required = [
                key
                for key in current
                if key.startswith(("scale_", "head_x.", "head_y.", "head_z."))
            ]
        missing = [key for key in required if key not in compatible]
        if missing:
            raise RuntimeError(
                "coordinate checkpoint is missing required tensors: "
                + ", ".join(missing)
            )
        self.load_state_dict(compatible, strict=False)
        return architecture, len(compatible)

    def forward(
        self,
        input_coords: torch.Tensor,
        input_features: torch.Tensor,
        n: int,
        q_step_mm: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return scale-conditioned offsets at the input q-level coordinates."""

        if n <= 0 or n > self.n_max:
            raise ValueError(
                f"n must be in [1, {self.n_max}], got {n}"
            )
        input_coords = input_coords.int()
        input_features = input_features.float()
        if input_coords.ndim != 2 or input_coords.shape[1] != 4:
            raise ValueError("input_coords must have shape (P, 4)")
        if input_features.ndim != 2 or input_features.shape[1] != 1:
            raise ValueError("input_features must have shape (P, 1)")
        if len(input_coords) != len(input_features):
            raise ValueError("coordinate and feature counts do not match")
        if not torch.all(input_features == 1):
            raise ValueError("current-scale input features must all equal one")

        target = self.feature_extractor(
            SparseTensor(
                coords=input_coords, feats=input_features
            )
        )

        scale_index = self._scale_index(q_step_mm)
        index = torch.tensor(
            scale_index,
            device=target.feats.device,
            dtype=torch.long,
        )
        scale_embedding = self.scale_embedding(index).view(
            1, self.channels
        )
        film_gamma, film_beta = self.scale_film(index).chunk(2)
        conditioned = target.feats + scale_embedding
        conditioned = conditioned * (
            1.0 + torch.tanh(film_gamma).view(1, -1)
        ) + film_beta.view(1, -1)

        expert_down = self.scale_expert_down[scale_index]
        expert_up = self.scale_expert_up[scale_index]
        conditioned = conditioned + F.silu(
            conditioned @ expert_down
        ) @ expert_up

        conditioned_sparse = SparseTensor(
            coords=target.coords, feats=conditioned
        )

        x = self.head_x(conditioned_sparse).feats
        y = self.head_y(conditioned_sparse).feats
        z = self.head_z(conditioned_sparse).feats
        logits = torch.stack((x, y, z), dim=-1)
        affine = self.scale_head_affine(index).view(
            self.n_max, 3, 2
        )
        gain = 1.0 + 0.25 * torch.tanh(
            affine[..., 0]
        )
        bias = 0.25 * affine[..., 1]
        logits = logits * gain.unsqueeze(0) + bias.unsqueeze(0)
        offsets = torch.sigmoid(
            logits[:, :n]
        )
        return offsets, target.coords.int()


def assert_anchor_alignment(
    actual_coords: torch.Tensor,
    expected_coords: torch.Tensor,
) -> None:
    """Fail if the sparse backbone changes q-level coordinate order."""

    if actual_coords.shape != expected_coords.shape:
        raise RuntimeError(
            f"anchor count mismatch: actual={tuple(actual_coords.shape)} "
            f"expected={tuple(expected_coords.shape)}"
        )
    if not torch.equal(
        actual_coords.int(), expected_coords.int()
    ):
        raise RuntimeError(
            "RENO final q-level anchor coordinates are misaligned"
        )


def decode_coordinates(
    normalized_offsets: torch.Tensor,
    anchor_coords: torch.Tensor,
    origins_mm: torch.Tensor,
    q_step_mm: int,
) -> torch.Tensor:
    """Convert q-anchor residuals to absolute XYZ metres."""

    if (
        normalized_offsets.ndim != 3
        or normalized_offsets.shape[-1] != 3
    ):
        raise ValueError(
            "normalized_offsets must have shape (P, n, 3)"
        )
    if len(anchor_coords) != len(normalized_offsets):
        raise ValueError(
            "anchor and prediction counts do not match"
        )
    batch_id = anchor_coords[:, 0].long()
    anchor = anchor_coords[:, 1:].to(
        normalized_offsets.dtype
    ).unsqueeze(1)
    origin = origins_mm[batch_id].to(
        normalized_offsets.dtype
    ).unsqueeze(1)
    try:
        max_abs_offset_mm = float(
            MAX_ABS_OFFSET_MM_BY_Q[int(q_step_mm)]
        )
    except KeyError as error:
        raise ValueError(
            f"unsupported q_step_mm={q_step_mm}; "
            f"expected one of {Q_STEPS_MM}"
        ) from error
    anchor_mm = anchor * float(q_step_mm)
    residual_mm = (
        normalized_offsets - 0.5
    ) * (2.0 * max_abs_offset_mm)
    xyz_mm = origin + anchor_mm + residual_mm
    return xyz_mm * 0.001


_NND_MODULE = None


def _get_nnd_module():
    global _NND_MODULE
    if _NND_MODULE is not None:
        return _NND_MODULE
    nnd_root = Path(
        os.environ.get(
            "GRASP_NND_PATH", str(DEFAULT_NND_PATH)
        )
    )
    if not nnd_root.is_dir():
        raise FileNotFoundError(
            "compiled GRASP NND extension not found: "
            f"{nnd_root}"
        )
    if str(nnd_root) not in sys.path:
        sys.path.insert(0, str(nnd_root))
    from modules.nnd import NNDModule

    _NND_MODULE = NNDModule()
    return _NND_MODULE


def global_chamfer_loss(
    decoded_points_m: Sequence[torch.Tensor],
    target_points_m: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Full-cloud symmetric squared nearest-neighbour loss."""

    if (
        len(decoded_points_m) != len(target_points_m)
        or not decoded_points_m
    ):
        raise ValueError(
            "decoded and target point-cloud batches must match"
        )
    nndistance = _get_nnd_module()
    losses = []
    for decoded, target in zip(
        decoded_points_m, target_points_m
    ):
        if (
            decoded.ndim != 2
            or decoded.shape[1] != 3
            or len(decoded) == 0
        ):
            raise ValueError(
                "decoded point clouds must have shape (N, 3)"
            )
        if (
            target.ndim != 2
            or target.shape[1] != 3
            or len(target) == 0
        ):
            raise ValueError(
                "target point clouds must have shape (N, 3)"
            )
        pred_to_target, target_to_pred, _, _ = nndistance(
            decoded.unsqueeze(0).contiguous(),
            target.unsqueeze(0).contiguous(),
        )
        losses.append(
            torch.maximum(
                pred_to_target.mean(),
                target_to_pred.mean(),
            )
        )
    return torch.stack(losses).mean()


def decoded_point_clouds(
    normalized_pred: torch.Tensor,
    anchor_coords: torch.Tensor,
    origins_mm: torch.Tensor,
    q_step_mm: int,
) -> List[torch.Tensor]:
    """Decode and split fixed-n outputs into one XYZ tensor per frame."""

    decoded = decode_coordinates(
        normalized_pred,
        anchor_coords,
        origins_mm,
        q_step_mm,
    )
    batch_id = anchor_coords[:, 0].long()
    return [
        decoded[batch_id == index].reshape(-1, 3)
        for index in range(origins_mm.shape[0])
    ]


def decreasing_scale_weight(
    q_step_mm: int,
    exponent: float = 1.0,
    reference_mm: int = 64,
) -> float:
    """Return the requested monotone coarse-to-fine scale weight."""

    if q_step_mm <= 0 or exponent < 0:
        raise ValueError(
            "q_step_mm must be positive and exponent non-negative"
        )
    return (
        float(reference_mm) / float(q_step_mm)
    ) ** float(exponent)
