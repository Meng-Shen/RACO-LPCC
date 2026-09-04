#!/usr/bin/env python3
"""Train current-scale all-one sparse coordinate restoration."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchsparse.nn import functional as sparse_functional


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from coordinate_residual import (  # noqa: E402
    DEFAULT_NND_PATH,
    CURRENT_SCALE_ONES_ARCHITECTURE,
    MAX_ABS_OFFSET_MM_BY_Q,
    N_BY_Q,
    N_MAX,
    Q_STEPS_MM,
    CoordinateResidualNet,
    assert_anchor_alignment,
    build_residual_batch,
    decoded_point_clouds,
    decreasing_scale_weight,
    global_chamfer_loss,
)
from detector_coordinate_loss import CoordinateDetectorLoss  # noqa: E402


DEFAULT_DET_CFG = Path(
    "/public/DATA/sm/RACO-LPCC/OpenPCDet/tools/cfgs/"
    "kitti_models/pv_rcnn_fov_geometry.yaml"
)
DEFAULT_DET_CKPT = Path(
    "/public/DATA/sm/RACO-LPCC/OpenPCDet/tools/ckpt/"
    "model_non_reflectance.pth"
)
DEFAULT_SPLIT = Path(
    "/public/DATA/sm/RACO-LPCC/OpenPCDet/data/kitti_fov/"
    "ImageSets/train.txt"
)
DEFAULT_POINTS = Path(
    "/public/DATA/sm/RACO-LPCC/OpenPCDet/data/kitti_fov/"
    "training/velodyne"
)
DEFAULT_OUTPUT = Path(
    "/public/DATA/sm/RACO-LPCC/reno/"
    "current_q_ones_coordinate_runs_v1"
)

OPTIMIZER_POLICY = "current_q_ones_scale_conditioned_diff_lr_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--det_cfg", type=Path, default=DEFAULT_DET_CFG
    )
    parser.add_argument(
        "--det_ckpt", type=Path, default=DEFAULT_DET_CKPT
    )
    parser.add_argument(
        "--split", type=Path, default=DEFAULT_SPLIT
    )
    parser.add_argument(
        "--points_dir", type=Path, default=DEFAULT_POINTS
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--shared_lr",
        type=float,
        default=1e-5,
        help="base LR for the current-scale sparse backbone and spatial heads",
    )
    parser.add_argument(
        "--scale_lr",
        type=float,
        default=3e-4,
        help="base LR for parameters whose names start with scale_",
    )
    parser.add_argument(
        "--freeze_shared_epochs",
        type=int,
        default=2,
        help="freeze shared parameters for the first N epochs",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-6
    )
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument(
        "--dist_scale_exponent", type=float, default=1.0
    )
    parser.add_argument(
        "--task_scale_exponent", type=float, default=1.0
    )
    parser.add_argument(
        "--dist_weight", type=float, default=1.0
    )
    parser.add_argument(
        "--task_weight", type=float, default=1.0
    )
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument(
        "--train_q_steps",
        default=",".join(str(q) for q in Q_STEPS_MM),
        help=(
            "comma-separated quantization steps to optimize; all model "
            "scales remain available for inference"
        ),
    )
    parser.add_argument(
        "--dist_q_multipliers",
        default="",
        help="per-scale distortion multipliers, for example 128:100,64:1",
    )
    parser.add_argument(
        "--task_q_multipliers",
        default="",
        help="per-scale task-loss multipliers, for example 64:4",
    )
    parser.add_argument(
        "--detector_max_frames", type=int, default=-1
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=340)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init_coordinate_checkpoint",
        type=Path,
        default=None,
        help=(
            "initialize unchanged scale modules and XYZ heads from the old "
            "scale-conditioned checkpoint, or all tensors from this architecture"
        ),
    )
    parser.add_argument(
        "--skip_detector",
        action="store_true",
        help="geometry-only smoke test",
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="disable deterministic epoch shuffling",
    )
    parser.add_argument(
        "--nnd_path", type=Path, default=DEFAULT_NND_PATH
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="single-process CUDA ordinal",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=-1,
        dest="local_rank",
    )
    parser.add_argument(
        "--ddp_master_addr",
        default=None,
        help="single-node rendezvous address override",
    )
    return parser.parse_args()


def parse_q_multipliers(value: str, name: str) -> Dict[int, float]:
    multipliers = {q: 1.0 for q in Q_STEPS_MM}
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"{name} entry must be q:value, got {item!r}")
        q_text, multiplier_text = item.split(":", 1)
        q_step_mm = int(q_text.strip())
        multiplier = float(multiplier_text.strip())
        if q_step_mm not in Q_STEPS_MM:
            raise ValueError(f"unsupported {name} scale: {q_step_mm}")
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError(f"{name} multiplier must be positive: {item!r}")
        multipliers[q_step_mm] = multiplier
    return multipliers


def configure_torchsparse() -> None:
    config = sparse_functional.conv_config.get_default_conv_config()
    config.kmap_mode = "hashmap"
    sparse_functional.conv_config.set_global_conv_config(config)


def setup_distributed(args: argparse.Namespace):
    distributed = (
        "RANK" in os.environ and "WORLD_SIZE" in os.environ
    )
    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(
            os.environ.get("LOCAL_RANK", args.local_rank)
        )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for distributed training"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if args.ddp_master_addr:
            os.environ["MASTER_ADDR"] = args.ddp_master_addr
        dist.init_process_group(
            backend="nccl", init_method="env://"
        )
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            if device.index is None:
                device = torch.device("cuda", 0)
            torch.cuda.set_device(device)
    return distributed, rank, world_size, local_rank, device


def shard_frame_ids(
    frame_ids: List[str],
    rank: int,
    world_size: int,
) -> List[str]:
    if not frame_ids:
        raise ValueError("cannot shard an empty frame list")
    total = (
        (len(frame_ids) + world_size - 1) // world_size
    ) * world_size
    padded = list(frame_ids)
    padded.extend(
        frame_ids[index % len(frame_ids)]
        for index in range(total - len(frame_ids))
    )
    return padded[rank:total:world_size]


def ordered_frames(
    frame_ids: List[str],
    epoch: int,
    seed: int,
    shuffle: bool,
) -> List[str]:
    ordered = list(frame_ids)
    if shuffle:
        random.Random(seed + epoch).shuffle(ordered)
    return ordered


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def read_frame_ids(path: Path, max_frames: int) -> List[str]:
    frame_ids = [
        line.strip().zfill(6)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]
    if not frame_ids:
        raise ValueError(f"no frame ids found in {path}")
    return frame_ids


def load_points(
    points_dir: Path,
    frame_id: str,
) -> np.ndarray:
    path = points_dir / f"{frame_id}.bin"
    values = np.fromfile(str(path), dtype=np.float32)
    if values.size == 0 or values.size % 4 != 0:
        raise ValueError(
            f"expected Nx4 point data in {path}"
        )
    return np.ascontiguousarray(
        values.reshape(-1, 4)[:, :3], dtype=np.float32
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": unwrap_model(
                model
            ).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": int(epoch),
            "architecture": CURRENT_SCALE_ONES_ARCHITECTURE,
            "optimizer_policy": OPTIMIZER_POLICY,
            "backbone": {
                "input": "current_q_sparse_coords_plus_all_one_feature",
                "feature_channels": int(unwrap_model(model).channels),
                "sparse_conv_layers": 4,
                "parent_scale_input": False,
                "occupancy_code_input": False,
            },
            "scale_conditioning": {
                "scales_mm": list(Q_STEPS_MM),
                "rank": int(unwrap_model(model).scale_rank),
                "components": [
                    "embedding",
                    "film",
                    "low_rank_expert",
                    "head_affine",
                ],
            },
            "geometry_target": "full_raw_fov_cloud",
            "distortion": "full_cloud_symmetric_nnd_m2",
            "q_steps_mm": list(Q_STEPS_MM),
            "n_by_q": {
                str(q): int(n)
                for q, n in N_BY_Q.items()
            },
            "n_max": int(N_MAX),
            "max_abs_offset_mm_by_q": {
                str(q): float(MAX_ABS_OFFSET_MM_BY_Q[q])
                for q in Q_STEPS_MM
            },
            "train_q_steps_mm": [
                int(value)
                for value in str(args.train_q_steps).split(",")
                if value.strip()
            ],
            "dist_q_multipliers": parse_q_multipliers(
                args.dist_q_multipliers, "dist_q_multipliers"
            ),
            "task_q_multipliers": parse_q_multipliers(
                args.task_q_multipliers, "task_q_multipliers"
            ),
            "args": vars(args),
        },
        path,
    )


def validate_resume_state(state: dict) -> None:
    if int(state.get("n_max", -1)) != N_MAX:
        raise ValueError("resume checkpoint n_max mismatch")
    expected = {
        str(q): int(N_BY_Q[q]) for q in Q_STEPS_MM
    }
    if state.get("n_by_q") != expected:
        raise ValueError("resume checkpoint n_by_q mismatch")
    if state.get("architecture") != CURRENT_SCALE_ONES_ARCHITECTURE:
        raise ValueError(
            "resume requires a current-scale all-one checkpoint; "
            "use --init_coordinate_checkpoint to import only unchanged "
            "downstream tensors from the previous architecture"
        )
    if state.get("optimizer_policy") != OPTIMIZER_POLICY:
        raise ValueError(
            "resume checkpoint optimizer policy mismatch; "
            "use --init_coordinate_checkpoint to start a fresh "
            "differential-LR optimizer"
        )


def main() -> None:
    args = parse_args()
    if (
        args.resume is not None
        and args.init_coordinate_checkpoint is not None
    ):
        raise ValueError(
            "--resume and --init_coordinate_checkpoint are mutually exclusive"
        )
    if args.shared_lr <= 0 or args.scale_lr <= 0:
        raise ValueError("learning rates must be positive")
    if args.freeze_shared_epochs < 0:
        raise ValueError("--freeze_shared_epochs must be non-negative")
    train_q_steps = tuple(
        int(value.strip())
        for value in str(args.train_q_steps).split(",")
        if value.strip()
    )
    if not train_q_steps or len(set(train_q_steps)) != len(train_q_steps):
        raise ValueError("--train_q_steps must contain unique scales")
    unsupported_q_steps = [
        q for q in train_q_steps if q not in Q_STEPS_MM
    ]
    if unsupported_q_steps:
        raise ValueError(
            f"unsupported --train_q_steps: {unsupported_q_steps}"
        )
    dist_q_multipliers = parse_q_multipliers(
        args.dist_q_multipliers, "dist_q_multipliers"
    )
    task_q_multipliers = parse_q_multipliers(
        args.task_q_multipliers, "task_q_multipliers"
    )
    os.environ["GRASP_NND_PATH"] = str(
        args.nnd_path.resolve()
    )
    distributed, rank, world_size, local_rank, device = (
        setup_distributed(args)
    )
    configure_torchsparse()
    set_seed(args.seed)
    if args.batch_size != 1:
        raise ValueError(
            "detector adapter currently requires batch_size=1"
        )

    all_frame_ids = read_frame_ids(
        args.split, args.max_frames
    )
    model = CoordinateResidualNet(
        channels=args.channels,
        kernel_size=args.kernel_size,
        n_max=N_MAX,
    ).to(device)
    initialized_coordinate_architecture = None
    initialized_coordinate_tensors = 0
    if args.init_coordinate_checkpoint is not None:
        (
            initialized_coordinate_architecture,
            initialized_coordinate_tensors,
        ) = model.load_coordinate_checkpoint(
            args.init_coordinate_checkpoint,
            map_location="cpu",
        )

    detector = None
    frame_to_detector_index: Dict[str, int] = {}
    if not args.skip_detector:
        detector = CoordinateDetectorLoss(
            cfg_file=args.det_cfg,
            checkpoint=args.det_ckpt,
            device=device,
            max_frames=(
                args.detector_max_frames
                if args.detector_max_frames > 0
                else None
            ),
            split="train",
        )
        indices = detector.indices_for_frames(
            all_frame_ids
        )
        frame_to_detector_index = dict(
            zip(all_frame_ids, indices)
        )

    named_parameters = list(model.named_parameters())
    shared_parameters = [
        parameter
        for name, parameter in named_parameters
        if not name.startswith("scale_")
    ]
    scale_parameters = [
        parameter
        for name, parameter in named_parameters
        if name.startswith("scale_")
    ]
    if not shared_parameters or not scale_parameters:
        raise RuntimeError(
            "failed to split shared and scale-conditioned parameters"
        )
    shared_parameter_count = sum(
        parameter.numel() for parameter in shared_parameters
    )
    scale_parameter_count = sum(
        parameter.numel() for parameter in scale_parameters
    )

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": shared_parameters,
                "lr": args.shared_lr,
                "name": "shared",
            },
            {
                "params": scale_parameters,
                "lr": args.scale_lr,
                "name": "scale",
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            lambda schedule_epoch: (
                0.0
                if schedule_epoch < args.freeze_shared_epochs
                else 0.1 ** (schedule_epoch // 20)
            ),
            lambda schedule_epoch: 0.1 ** (schedule_epoch // 20),
        ],
    )
    start_epoch = 0
    if args.resume is not None:
        state = torch.load(
            str(args.resume), map_location="cpu"
        )
        validate_resume_state(state)
        unwrap_model(model).load_state_dict(
            state["model_state_dict"]
        )
        optimizer.load_state_dict(
            state["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            state["scheduler_state_dict"]
        )
        start_epoch = int(state["epoch"]) + 1

    if rank == 0:
        args.output_dir.mkdir(
            parents=True, exist_ok=True
        )
    if distributed:
        dist.barrier()
    config = {
        "architecture": CURRENT_SCALE_ONES_ARCHITECTURE,
        "optimizer_policy": {
            "name": OPTIMIZER_POLICY,
            "shared_lr": args.shared_lr,
            "scale_lr": args.scale_lr,
            "freeze_shared_epochs": args.freeze_shared_epochs,
            "decay_every_epochs": 20,
            "decay_gamma": 0.1,
            "shared_parameter_count": shared_parameter_count,
            "scale_parameter_count": scale_parameter_count,
        },
        "scale_conditioning": {
            "scales_mm": list(Q_STEPS_MM),
            "rank": int(unwrap_model(model).scale_rank),
            "components": [
                "embedding",
                "film",
                "low_rank_expert",
                "head_affine",
            ],
            "identity_initialized": True,
        },
        "input_contract": (
            "decoded current-q sparse coordinates with one scalar feature "
            "equal to one at every active voxel"
        ),
        "backbone": {
            "type": "four_layer_stride1_torchsparse_conv",
            "input_channels": 1,
            "hidden_channels": int(unwrap_model(model).channels),
            "parent_scale_input": False,
            "occupancy_code_input": False,
            "fcg": False,
            "target_embedding": False,
        },
        "q_steps_mm": list(Q_STEPS_MM),
        "n_by_q": {
            str(q): int(N_BY_Q[q]) for q in Q_STEPS_MM
        },
        "n_max": int(N_MAX),
        "max_abs_offset_mm_by_q": {
            str(q): float(MAX_ABS_OFFSET_MM_BY_Q[q])
            for q in Q_STEPS_MM
        },
        "train_q_steps_mm": list(train_q_steps),
        "dist_q_multipliers": {
            str(q): float(dist_q_multipliers[q])
            for q in Q_STEPS_MM
        },
        "task_q_multipliers": {
            str(q): float(task_q_multipliers[q])
            for q in Q_STEPS_MM
        },
        "geometry_source": str(args.points_dir),
        "geometry_target": "complete preprocessed FOV bin",
        "distortion": "GRASP-style full-cloud symmetric NND",
        "detector_preprocessing": (
            "detached FOV/range/voxel assignment, "
            "differentiable XYZ values"
        ),
        "init_coordinate_checkpoint": (
            str(args.init_coordinate_checkpoint)
            if args.init_coordinate_checkpoint is not None
            else None
        ),
        "init_coordinate_architecture": (
            initialized_coordinate_architecture
        ),
        "init_coordinate_tensors": (
            initialized_coordinate_tensors
        ),
        "num_frames": len(all_frame_ids),
        "split": str(args.split),
        "world_size": world_size,
        "shuffle_each_epoch": not args.no_shuffle,
        "loss_scales": {
            "distortion_exponent": args.dist_scale_exponent,
            "task_exponent": args.task_scale_exponent,
            "reference_mm": 64,
        },
    }
    if rank == 0:
        (args.output_dir / "config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )
        print(
            "CONFIG " + json.dumps(config, sort_keys=True),
            flush=True,
        )

    best_train_total = float("inf")
    best_train_epoch = -1
    for epoch in range(start_epoch, args.epochs):
        model.train()
        shared_frozen = epoch < args.freeze_shared_epochs
        current_lrs = {
            group["name"]: float(group["lr"])
            for group in optimizer.param_groups
        }
        if rank == 0:
            print(
                "EPOCH_START "
                + json.dumps(
                    {
                        "epoch": epoch,
                        "shared_frozen": shared_frozen,
                        "shared_lr": current_lrs["shared"],
                        "scale_lr": current_lrs["scale"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if detector is not None:
            detector.model.zero_grad(set_to_none=True)
        epoch_order = ordered_frames(
            all_frame_ids,
            epoch,
            args.seed,
            not args.no_shuffle,
        )
        frame_ids = shard_frame_ids(
            epoch_order, rank, world_size
        )
        running: Dict[int, Dict[str, float]] = {
            q: {
                "weighted": 0.0,
                "dist": 0.0,
                "task": 0.0,
            }
            for q in train_q_steps
        }
        epoch_started = time.time()

        for frame_index, frame_id in enumerate(frame_ids):
            # Geometry uses the exact FOV-only source point cloud.
            points = load_points(args.points_dir, frame_id)
            detector_index = (
                frame_to_detector_index.get(frame_id)
            )
            optimizer.zero_grad(set_to_none=True)
            total_value = 0.0

            for scale_index, q_step_mm in enumerate(
                train_q_steps
            ):
                n = N_BY_Q[q_step_mm]
                if (
                    distributed
                    and scale_index < len(train_q_steps) - 1
                ):
                    sync_context = model.no_sync()
                else:
                    sync_context = nullcontext()

                with sync_context:
                    batch = build_residual_batch(
                        [points],
                        q_step_mm,
                        n,
                        device,
                    )
                    pred_all, anchor_coords = model(
                        batch.input_coords,
                        batch.input_features,
                        N_MAX,
                        q_step_mm,
                    )
                    assert_anchor_alignment(
                        anchor_coords,
                        batch.anchor_coords,
                    )
                    pred = pred_all[:, :n]
                    decoded_clouds = decoded_point_clouds(
                        pred,
                        anchor_coords,
                        batch.origins_mm,
                        q_step_mm,
                    )
                    distortion = global_chamfer_loss(
                        decoded_clouds,
                        batch.target_points_m,
                    )
                    distortion_term = (
                        args.dist_weight
                        * dist_q_multipliers[q_step_mm]
                        * decreasing_scale_weight(
                            q_step_mm,
                            args.dist_scale_exponent,
                        )
                        * distortion
                    )

                    if detector is None:
                        task = pred.sum() * 0.0
                    else:
                        if detector_index is None:
                            raise KeyError(
                                f"missing detector index: {frame_id}"
                            )
                        task = detector(
                            decoded_clouds[0],
                            detector_index,
                        )
                    task_term = (
                        args.task_weight
                        * task_q_multipliers[q_step_mm]
                        * decreasing_scale_weight(
                            q_step_mm,
                            args.task_scale_exponent,
                        )
                        * task
                    )
                    scale_total = (
                        distortion_term + task_term
                    )
                    scale_total = scale_total + (
                        pred_all[:, n:].sum() * 0.0
                    )
                    scale_total.backward()

                total_value += float(
                    scale_total.detach()
                )
                running[q_step_mm]["weighted"] += float(
                    scale_total.detach()
                )
                running[q_step_mm]["dist"] += float(
                    distortion.detach()
                )
                running[q_step_mm]["task"] += float(
                    task.detach()
                )

            if shared_frozen:
                # A zero LR alone would still accumulate Adam moments.
                # Dropping these gradients keeps the shared state truly frozen.
                for parameter in shared_parameters:
                    parameter.grad = None
                parameters_to_clip = scale_parameters
            else:
                parameters_to_clip = list(model.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters_to_clip, 5.0
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"non-finite grad norm at {frame_id}"
                )
            optimizer.step()
            if rank == 0:
                print(
                    f"epoch={epoch} "
                    f"frame={frame_index + 1}/{len(frame_ids)} "
                    f"/{len(all_frame_ids)} "
                    f"id={frame_id} total={total_value:.6f} "
                    f"grad_norm={float(grad_norm):.6f}",
                    flush=True,
                )

        scheduler.step()
        metric_tensor = torch.tensor(
            [
                *[
                    running[q]["weighted"]
                    for q in train_q_steps
                ],
                *[
                    running[q]["dist"]
                    for q in train_q_steps
                ],
                *[
                    running[q]["task"]
                    for q in train_q_steps
                ],
            ],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.all_reduce(
                metric_tensor, op=dist.ReduceOp.SUM
            )
        denom = float(
            sum(
                len(
                    shard_frame_ids(
                        epoch_order, current_rank, world_size
                    )
                )
                for current_rank in range(world_size)
            )
        )
        values = metric_tensor.cpu().tolist()
        count = len(train_q_steps)
        weighted_values = values[:count]
        dist_values = values[count : 2 * count]
        task_values = values[2 * count :]
        summary = {
            "epoch": epoch,
            "lr": current_lrs["shared"],
            "shared_lr": current_lrs["shared"],
            "scale_lr": current_lrs["scale"],
            "shared_frozen": shared_frozen,
            "seconds": time.time() - epoch_started,
            "world_size": world_size,
            "total": sum(weighted_values) / denom,
            "scales": {
                str(q): {
                    "weighted_total": (
                        weighted_values[index] / denom
                    ),
                    "dist_m2": (
                        dist_values[index] / denom
                    ),
                    "task": (
                        task_values[index] / denom
                    ),
                }
                for index, q in enumerate(train_q_steps)
            },
        }
        if rank == 0:
            print(
                "EPOCH_SUMMARY "
                + json.dumps(summary, sort_keys=True),
                flush=True,
            )
            save_checkpoint(
                args.output_dir / f"epoch_{epoch}.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                args,
            )
            if float(summary["total"]) < best_train_total:
                best_train_total = float(summary["total"])
                best_train_epoch = int(epoch)
                save_checkpoint(
                    args.output_dir / "best_train_loss.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    args,
                )
                (args.output_dir / "best_train_loss.json").write_text(
                    json.dumps(
                        {
                            "epoch": best_train_epoch,
                            "train_total": best_train_total,
                            "selection": "minimum full-training-set weighted loss",
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    "BEST_TRAIN_LOSS "
                    + json.dumps(
                        {
                            "epoch": best_train_epoch,
                            "train_total": best_train_total,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            save_checkpoint(
                args.output_dir / "epoch_newest.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                args,
            )
        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
