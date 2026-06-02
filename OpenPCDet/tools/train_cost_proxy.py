#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a SparseConv cost-only JUQP/JUCP proxy network.

This script is designed for the RACO-LPCC / OpenPCDet environment.
It replaces the previous PointNet-style encoder with a sparse 3D convolutional
encoder based on spconv, which is already used by OpenPCDet backbones.

Core idea
---------
The network does NOT directly predict JUQP labels. It predicts AP drops:

    cost_pred: [B, 6, 3]

where:
    cost_pred[:, 0, :] = L0_AP - L1_AP
    cost_pred[:, 1, :] = L0_AP - L2_AP
    cost_pred[:, 2, :] = L0_AP - L3_AP
    cost_pred[:, 3, :] = L0_AP - L4_AP
    cost_pred[:, 4, :] = L0_AP - L5_AP
    cost_pred[:, 5, :] = L0_AP - L6_AP

Last dimension 3 means:
    0: Car AP drop
    1: Pedestrian AP drop
    2: Cyclist AP drop

Then JUQP/JUCP labels are derived by the same threshold rule as jucp_split.py:
    for l in L6 -> L1:
        if Car/Ped/Cyc AP drop <= corresponding thresholds:
            label = l
            break
    if none is valid:
        label = 0

Recommended location
--------------------
Save this file as:
    /public/DATA/sm/RACO-LPCC/OpenPCDet/tools/train_sparse_cost_proxy.py

Run from:
    /public/DATA/sm/RACO-LPCC/OpenPCDet/tools

Example
-------
python train_sparse_cost_proxy.py \
  --velodyne_dir ../data/kitti/training/velodyne \
  --train_split ../data/kitti/ImageSets/train.txt \
  --ap_csv split_AP_train.csv \
  --test_split ../data/kitti/ImageSets/val.txt \
  --test_ap_csv test/split_AP.csv \
  --thresholds "0,0,0;0.001,0.01,0.02;0.0015,0.02,0.035;0.0025,0.03,0.045;0.0035,0.04,0.06;0.0045,0.05,0.075" \
  --test_every 5 \
  --out_dir router_work_dirs/sparse_cost_proxy \
  --epochs 120 \
  --batch_size 4 \
  --workers 4 \
  --voxel_size 0.16 0.16 0.16 \
  --max_voxels 50000 \
  --feat_dim 256 \
  --ap_drop_scale 100 \
  --lambda_threshold 0.1
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from pcdet.utils.spconv_utils import spconv, replace_feature
except Exception:
    try:
        import spconv.pytorch as spconv
    except Exception as e:
        raise ImportError(
            "Cannot import spconv. Please run this script inside the OpenPCDet environment "
            "where spconv is installed."
        ) from e

    def replace_feature(out, new_features):
        out.features = new_features
        return out


# ============================================================
# Basic utilities
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_frame_id(x) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(6)


def read_split_file(split_file: str) -> List[str]:
    with open(split_file, "r") as f:
        return [normalize_frame_id(line) for line in f if line.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class TeeLogger:
    """Write stdout/stderr to both terminal and a log file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()


def setup_file_logger(out_dir: Path, log_file_arg: Optional[str] = None) -> Path:
    ensure_dir(out_dir)
    if log_file_arg is None or log_file_arg == "":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = out_dir / f"train_sparse_cost_proxy_{timestamp}.log"
    else:
        log_path = Path(log_file_arg)
        if not log_path.is_absolute():
            log_path = out_dir / log_path
        ensure_dir(log_path.parent)

    f = open(log_path, "a", buffering=1)
    sys.stdout = TeeLogger(sys.__stdout__, f)
    sys.stderr = TeeLogger(sys.__stderr__, f)
    print(f"[INFO] Full log will be saved to: {log_path}")
    return log_path


def parse_thresholds(thresholds: Optional[str], scale: float = 1.0) -> Optional[torch.Tensor]:
    """Parse 'car,ped,cyc;car,ped,cyc;...' into [K,3].

    If AP drops are scaled by --ap_drop_scale, thresholds are scaled by the same
    factor to keep threshold-based JUQP derivation consistent.
    """
    if thresholds is None or thresholds.strip() == "":
        return None

    out = []
    for item in thresholds.split(";"):
        item = item.strip()
        if not item:
            continue
        vals = [float(x.strip()) * float(scale) for x in item.split(",")]
        if len(vals) != 3:
            raise ValueError(f"Invalid threshold triple: {item}. Expected car,ped,cyc")
        out.append(vals)
    if not out:
        return None
    return torch.tensor(out, dtype=torch.float32)


# ============================================================
# AP CSV loading
# ============================================================


def detect_ap_levels(ap_df: pd.DataFrame) -> List[int]:
    levels = []
    for col in ap_df.columns:
        if col.startswith("L") and col.endswith("_Car_AP"):
            level_str = col[1:].split("_")[0]
            if level_str.isdigit():
                levels.append(int(level_str))
    levels = sorted(set(levels))
    if not levels:
        raise ValueError("Cannot detect AP levels. Expected columns like L0_Car_AP, L1_Car_AP, ...")
    return levels


def load_ap_drop_gt(ap_csv: str, num_cost_heads: int, ap_drop_scale: float = 1.0) -> Dict[str, np.ndarray]:
    """Load split_AP.csv and convert AP matrix to AP drop targets.

    Return:
        ap_drop[fid]: [6, 3]
    """
    df = pd.read_csv(ap_csv)
    if "frame_id" not in df.columns:
        raise KeyError(f"{ap_csv} missing column: frame_id")

    df["frame_id"] = df["frame_id"].map(normalize_frame_id)
    levels = detect_ap_levels(df)
    required_levels = list(range(0, num_cost_heads + 1))
    for lv in required_levels:
        if lv not in levels:
            raise ValueError(f"AP csv missing level L{lv}; detected levels: {levels}")

    required_cols = []
    for lv in required_levels:
        required_cols.extend([f"L{lv}_Car_AP", f"L{lv}_Ped_AP", f"L{lv}_Cyc_AP"])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"AP csv missing columns: {missing[:20]}")

    ap_drop: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        fid = row["frame_id"]
        base = np.asarray(
            [float(row["L0_Car_AP"]), float(row["L0_Ped_AP"]), float(row["L0_Cyc_AP"])],
            dtype=np.float32,
        )

        drops = []
        for h in range(num_cost_heads):
            lv = h + 1
            cur = np.asarray(
                [float(row[f"L{lv}_Car_AP"]), float(row[f"L{lv}_Ped_AP"]), float(row[f"L{lv}_Cyc_AP"])],
                dtype=np.float32,
            )
            drop = np.maximum(base - cur, 0.0)
            drops.append(drop)
        ap_drop[fid] = np.stack(drops, axis=0).astype(np.float32) * float(ap_drop_scale)
    return ap_drop


# ============================================================
# Voxelization
# ============================================================


def augment_points(raw: np.ndarray, use_rotation_aug: bool, jitter_std: float) -> np.ndarray:
    raw = raw.copy()
    if use_rotation_aug:
        theta = np.random.uniform(-np.pi, np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        raw[:, :3] = raw[:, :3] @ rot.T

    raw[:, :3] *= np.random.uniform(0.98, 1.02)
    if jitter_std > 0:
        raw[:, :3] += np.random.normal(0.0, jitter_std, size=raw[:, :3].shape).astype(np.float32)
    return raw


def voxelize_points(
    points: np.ndarray,
    voxel_size: np.ndarray,
    pc_range: np.ndarray,
    max_voxels: int,
    use_abs_xyz: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple CPU voxelization with mean feature aggregation.

    Args:
        points: [N,4], xyz + intensity
        voxel_size: [3], x/y/z voxel size
        pc_range: [6], x_min,y_min,z_min,x_max,y_max,z_max

    Returns:
        voxel_features: [M,C]
        voxel_coords_zyx: [M,3], int32 coords in z,y,x order for spconv
    """
    xyz = points[:, :3]
    mask = (
        (xyz[:, 0] >= pc_range[0]) & (xyz[:, 0] < pc_range[3]) &
        (xyz[:, 1] >= pc_range[1]) & (xyz[:, 1] < pc_range[4]) &
        (xyz[:, 2] >= pc_range[2]) & (xyz[:, 2] < pc_range[5])
    )
    points = points[mask]
    if points.shape[0] == 0:
        # Return one dummy voxel. This should be rare for KITTI.
        return np.zeros((1, 8 if use_abs_xyz else 5), dtype=np.float32), np.zeros((1, 3), dtype=np.int32)

    xyz = points[:, :3]
    coords_xyz = np.floor((xyz - pc_range[:3]) / voxel_size).astype(np.int32)

    grid_size = np.floor((pc_range[3:] - pc_range[:3]) / voxel_size).astype(np.int32)
    valid = (
        (coords_xyz[:, 0] >= 0) & (coords_xyz[:, 0] < grid_size[0]) &
        (coords_xyz[:, 1] >= 0) & (coords_xyz[:, 1] < grid_size[1]) &
        (coords_xyz[:, 2] >= 0) & (coords_xyz[:, 2] < grid_size[2])
    )
    coords_xyz = coords_xyz[valid]
    points = points[valid]
    if points.shape[0] == 0:
        return np.zeros((1, 8 if use_abs_xyz else 5), dtype=np.float32), np.zeros((1, 3), dtype=np.int32)

    unique_coords, inverse, counts = np.unique(coords_xyz, axis=0, return_inverse=True, return_counts=True)

    if unique_coords.shape[0] > max_voxels:
        # Keep random active voxels to control memory.
        keep = np.random.choice(unique_coords.shape[0], max_voxels, replace=False)
        keep = np.sort(keep)
        old_to_new = -np.ones(unique_coords.shape[0], dtype=np.int32)
        old_to_new[keep] = np.arange(keep.shape[0], dtype=np.int32)
        point_keep = old_to_new[inverse] >= 0
        inverse = old_to_new[inverse[point_keep]]
        points = points[point_keep]
        unique_coords = unique_coords[keep]
        counts = np.bincount(inverse, minlength=unique_coords.shape[0])

    M = unique_coords.shape[0]
    sum_xyz = np.zeros((M, 3), dtype=np.float32)
    sum_i = np.zeros((M, 1), dtype=np.float32)
    np.add.at(sum_xyz, inverse, points[:, :3].astype(np.float32))
    np.add.at(sum_i, inverse, points[:, 3:4].astype(np.float32))
    counts_f = counts.reshape(-1, 1).astype(np.float32)

    mean_xyz = sum_xyz / np.maximum(counts_f, 1.0)
    mean_i = sum_i / np.maximum(counts_f, 1.0)

    voxel_centers = pc_range[:3] + (unique_coords.astype(np.float32) + 0.5) * voxel_size
    rel_xyz = (mean_xyz - voxel_centers) / voxel_size
    density = np.log1p(counts_f) / np.log(64.0)
    density = np.clip(density, 0.0, 1.0)

    if use_abs_xyz:
        # Normalize absolute xyz roughly to [-1,1] range.
        abs_xyz_norm = (mean_xyz - pc_range[:3]) / (pc_range[3:] - pc_range[:3] + 1e-6)
        abs_xyz_norm = abs_xyz_norm * 2.0 - 1.0
        voxel_features = np.concatenate([rel_xyz, mean_i, density, abs_xyz_norm], axis=1).astype(np.float32)
    else:
        voxel_features = np.concatenate([rel_xyz, mean_i, density], axis=1).astype(np.float32)

    # spconv expects coords in z,y,x order, while we computed x,y,z.
    coords_zyx = unique_coords[:, [2, 1, 0]].astype(np.int32)
    return voxel_features, coords_zyx


# ============================================================
# Dataset and collate
# ============================================================


class SparseCostProxyDataset(Dataset):
    def __init__(
        self,
        velodyne_dir: str,
        split_file: str,
        ap_csv: str,
        voxel_size: List[float],
        pc_range: List[float],
        max_voxels: int,
        num_cost_heads: int,
        training: bool,
        ap_drop_scale: float,
        use_rotation_aug: bool,
        jitter_std: float,
        use_abs_xyz: bool,
    ):
        self.velodyne_dir = Path(velodyne_dir)
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.pc_range = np.asarray(pc_range, dtype=np.float32)
        self.max_voxels = int(max_voxels)
        self.training = bool(training)
        self.use_rotation_aug = bool(use_rotation_aug)
        self.jitter_std = float(jitter_std)
        self.use_abs_xyz = bool(use_abs_xyz)

        split_ids = read_split_file(split_file)
        ap_drop_gt = load_ap_drop_gt(ap_csv, num_cost_heads=num_cost_heads, ap_drop_scale=ap_drop_scale)

        self.items = []
        missing_bin = 0
        missing_ap = 0
        for fid in split_ids:
            bin_path = self.velodyne_dir / f"{fid}.bin"
            if not bin_path.exists():
                missing_bin += 1
                continue
            if fid not in ap_drop_gt:
                missing_ap += 1
                continue
            self.items.append((fid, bin_path, ap_drop_gt[fid]))

        grid_size = np.floor((self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size).astype(np.int32)
        self.spatial_shape = grid_size[[2, 1, 0]].tolist()  # z,y,x
        self.num_point_features = 8 if use_abs_xyz else 5

        print(f"Dataset split: {split_file}")
        print(f"  usable samples: {len(self.items)}")
        print(f"  missing bin:    {missing_bin}")
        print(f"  missing AP:     {missing_ap}")
        print(f"  voxel_size:     {self.voxel_size.tolist()}")
        print(f"  pc_range:       {self.pc_range.tolist()}")
        print(f"  spatial_shape:  {self.spatial_shape}  # z,y,x")
        print(f"  max_voxels:     {self.max_voxels}")

        if len(self.items) == 0:
            raise RuntimeError("No usable samples found. Check velodyne_dir, split_file and ap_csv frame_id alignment.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        fid, bin_path, ap_drop = self.items[idx]
        raw = np.fromfile(str(bin_path), dtype=np.float32)
        if raw.size % 4 != 0:
            raise ValueError(f"Invalid KITTI bin file: {bin_path}")
        raw = raw.reshape(-1, 4)

        if self.training:
            raw = augment_points(raw, self.use_rotation_aug, self.jitter_std)

        voxel_features, voxel_coords = voxelize_points(
            raw,
            voxel_size=self.voxel_size,
            pc_range=self.pc_range,
            max_voxels=self.max_voxels,
            use_abs_xyz=self.use_abs_xyz,
        )

        return {
            "frame_id": fid,
            "voxel_features": torch.from_numpy(voxel_features),  # [M,C]
            "voxel_coords": torch.from_numpy(voxel_coords),      # [M,3], z,y,x
            "ap_drop": torch.from_numpy(ap_drop),                # [6,3]
        }


def sparse_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    voxel_features = []
    voxel_coords = []
    ap_drop = []
    frame_ids = []

    for b, item in enumerate(batch):
        vf = item["voxel_features"]
        vc = item["voxel_coords"]
        batch_col = torch.full((vc.shape[0], 1), b, dtype=torch.int32)
        vc_b = torch.cat([batch_col, vc.int()], dim=1)  # [M,4], batch,z,y,x

        voxel_features.append(vf.float())
        voxel_coords.append(vc_b)
        ap_drop.append(item["ap_drop"].float())
        frame_ids.append(item["frame_id"])

    return {
        "frame_id": frame_ids,
        "voxel_features": torch.cat(voxel_features, dim=0),
        "voxel_coords": torch.cat(voxel_coords, dim=0).int(),
        "ap_drop": torch.stack(ap_drop, dim=0),
        "batch_size": len(batch),
    }


# ============================================================
# SparseConv model
# ============================================================


class SparseConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, indice_key: str):
        super().__init__()
        self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = replace_feature(x, self.bn(x.features))
        x = replace_feature(x, self.relu(x.features))
        return x


class SparseDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, indice_key: str):
        super().__init__()
        self.conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False, indice_key=indice_key)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = replace_feature(x, self.bn(x.features))
        x = replace_feature(x, self.relu(x.features))
        return x


class SparseCostProxyNet(nn.Module):
    """SparseConv encoder + cost heads.

    Input sparse tensor:
        features: [num_active_voxels, C]
        indices:  [num_active_voxels, 4], batch,z,y,x

    Output:
        cost_pred: [B, 6, 3]
    """

    def __init__(
        self,
        input_channels: int,
        spatial_shape: List[int],
        feat_dim: int = 256,
        num_cost_heads: int = 6,
        num_targets: int = 3,
        cost_nonnegative: bool = True,
    ):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.num_cost_heads = int(num_cost_heads)
        self.num_targets = int(num_targets)
        self.cost_nonnegative = bool(cost_nonnegative)

        self.stem = SparseConvBlock(input_channels, 32, indice_key="subm1")
        self.stage1 = nn.Sequential(
            SparseConvBlock(32, 32, indice_key="subm1a"),
            SparseConvBlock(32, 32, indice_key="subm1b"),
        )
        self.down2 = SparseDownBlock(32, 64, indice_key="spconv2")
        self.stage2 = nn.Sequential(
            SparseConvBlock(64, 64, indice_key="subm2a"),
            SparseConvBlock(64, 64, indice_key="subm2b"),
        )
        self.down3 = SparseDownBlock(64, 128, indice_key="spconv3")
        self.stage3 = nn.Sequential(
            SparseConvBlock(128, 128, indice_key="subm3a"),
            SparseConvBlock(128, 128, indice_key="subm3b"),
        )
        self.down4 = SparseDownBlock(128, 256, indice_key="spconv4")
        self.stage4 = nn.Sequential(
            SparseConvBlock(256, 256, indice_key="subm4a"),
            SparseConvBlock(256, 256, indice_key="subm4b"),
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(256 * 2, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

        self.cost_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(feat_dim, num_targets),
            )
            for _ in range(num_cost_heads)
        ])

    @staticmethod
    def global_pool(features: torch.Tensor, batch_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        C = features.shape[1]
        device = features.device
        dtype = features.dtype

        sum_feat = torch.zeros(batch_size, C, device=device, dtype=dtype)
        cnt = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        sum_feat.index_add_(0, batch_indices, features)
        cnt.index_add_(0, batch_indices, torch.ones(features.shape[0], 1, device=device, dtype=dtype))
        avg_feat = sum_feat / cnt.clamp_min(1.0)

        # max pooling. Loop over batch is fine because batch size is small.
        max_feat = torch.zeros(batch_size, C, device=device, dtype=dtype)
        for b in range(batch_size):
            mask = batch_indices == b
            if mask.any():
                max_feat[b] = features[mask].max(dim=0).values

        return torch.cat([avg_feat, max_feat], dim=1)

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.spatial_shape,
            batch_size=batch_size,
        )

        x = self.stem(x)
        x = self.stage1(x)
        x = self.down2(x)
        x = self.stage2(x)
        x = self.down3(x)
        x = self.stage3(x)
        x = self.down4(x)
        x = self.stage4(x)

        batch_indices = x.indices[:, 0].long()
        pooled = self.global_pool(x.features, batch_indices, batch_size)
        global_feat = self.global_mlp(pooled)

        preds = []
        for head in self.cost_heads:
            pred = head(global_feat)
            if self.cost_nonnegative:
                pred = F.softplus(pred)
            preds.append(pred)

        return {"cost_pred": torch.stack(preds, dim=1)}  # [B,6,3]


# ============================================================
# JUQP derivation, loss, metrics
# ============================================================


@torch.no_grad()
def cost_to_jucp_labels(cost: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """Convert cost [B,6,3] to JUQP labels [B,K]."""
    thresholds = thresholds.to(cost.device, dtype=cost.dtype)
    B, H, _ = cost.shape
    labels_all = []
    for k in range(thresholds.shape[0]):
        thr = thresholds[k].view(1, 1, 3)
        valid = (cost <= thr).all(dim=-1)  # [B,6]
        labels = torch.zeros(B, dtype=torch.long, device=cost.device)
        assigned = torch.zeros(B, dtype=torch.bool, device=cost.device)
        for l in range(H, 0, -1):
            h = l - 1
            choose = valid[:, h] & (~assigned)
            labels = torch.where(choose, torch.full_like(labels, l), labels)
            assigned = assigned | choose
        labels_all.append(labels)
    return torch.stack(labels_all, dim=1)


def compute_loss(
    cost_pred: torch.Tensor,
    ap_drop_gt: torch.Tensor,
    ap_weights: torch.Tensor,
    head_weights: Optional[torch.Tensor] = None,
    thresholds: Optional[torch.Tensor] = None,
    lambda_threshold: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Regression loss + optional threshold-aware consistency loss."""
    B, H, T = cost_pred.shape

    loss_per = F.smooth_l1_loss(cost_pred, ap_drop_gt, reduction="none")
    loss_per = (loss_per * ap_weights.view(1, 1, T)).mean(dim=-1)  # [B,H]
    if head_weights is not None:
        reg_loss = (loss_per * head_weights.view(1, H)).mean()
    else:
        reg_loss = loss_per.mean()

    threshold_loss = cost_pred.new_tensor(0.0)
    if thresholds is not None and lambda_threshold > 0:
        thresholds = thresholds.to(cost_pred.device, dtype=cost_pred.dtype)
        bce_losses = []
        for k in range(thresholds.shape[0]):
            thr = thresholds[k].view(1, 1, T)
            # gt_valid: whether the AP drop is below threshold for each category.
            gt_valid = (ap_drop_gt <= thr).float()
            # Positive logit means predicted drop is below threshold.
            pred_logits = thr - cost_pred
            bce = F.binary_cross_entropy_with_logits(pred_logits, gt_valid, reduction="none")
            bce = (bce * ap_weights.view(1, 1, T)).mean(dim=-1)
            if head_weights is not None:
                bce = (bce * head_weights.view(1, H)).mean()
            else:
                bce = bce.mean()
            bce_losses.append(bce)
        threshold_loss = torch.stack(bce_losses).mean()

    total_loss = reg_loss + float(lambda_threshold) * threshold_loss
    return {"total_loss": total_loss, "reg_loss": reg_loss, "threshold_loss": threshold_loss}


@torch.no_grad()
def compute_metrics(cost_pred: torch.Tensor, ap_drop_gt: torch.Tensor, thresholds: Optional[torch.Tensor] = None) -> Dict[str, object]:
    mae_per_target = torch.abs(cost_pred - ap_drop_gt).mean(dim=(0, 1)).detach().cpu().numpy().tolist()
    cost_mae = float(np.mean(mae_per_target))

    metrics: Dict[str, object] = {
        "cost_mae": cost_mae,
        "mae_car": float(mae_per_target[0]),
        "mae_ped": float(mae_per_target[1]),
        "mae_cyc": float(mae_per_target[2]),
    }

    if thresholds is not None:
        pred_labels = cost_to_jucp_labels(cost_pred, thresholds)
        gt_labels = cost_to_jucp_labels(ap_drop_gt, thresholds)
        acc = (pred_labels == gt_labels).float().mean(dim=0).detach().cpu().numpy().tolist()
        metrics["mean_jucp_acc"] = float(np.mean(acc))
        metrics["jucp_acc_per_threshold"] = acc
    else:
        metrics["mean_jucp_acc"] = 0.0
        metrics["jucp_acc_per_threshold"] = []
    return metrics


# ============================================================
# Train / eval loop
# ============================================================


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool,
    ap_weights: torch.Tensor,
    head_weights: Optional[torch.Tensor],
    thresholds: Optional[torch.Tensor],
    lambda_threshold: float,
) -> Dict[str, object]:
    model.train(train)

    total_samples = 0
    total_loss_sum = 0.0
    reg_loss_sum = 0.0
    thr_loss_sum = 0.0
    cost_mae_sum = 0.0
    mae_target_sum = np.zeros(3, dtype=np.float64)
    acc_sum = None

    pbar = tqdm(loader, desc="train" if train else "eval", dynamic_ncols=True)
    for batch in pbar:
        voxel_features = batch["voxel_features"].to(device, non_blocking=True)
        voxel_coords = batch["voxel_coords"].to(device, non_blocking=True)
        ap_drop = batch["ap_drop"].to(device, non_blocking=True)
        batch_size = int(batch["batch_size"])

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(voxel_features, voxel_coords, batch_size)
            cost_pred = out["cost_pred"]
            losses = compute_loss(
                cost_pred=cost_pred,
                ap_drop_gt=ap_drop,
                ap_weights=ap_weights,
                head_weights=head_weights,
                thresholds=thresholds,
                lambda_threshold=lambda_threshold,
            )
            if train:
                losses["total_loss"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

        bs = batch_size
        total_samples += bs
        total_loss_sum += float(losses["total_loss"].detach().cpu()) * bs
        reg_loss_sum += float(losses["reg_loss"].detach().cpu()) * bs
        thr_loss_sum += float(losses["threshold_loss"].detach().cpu()) * bs

        metrics = compute_metrics(cost_pred, ap_drop, thresholds)
        cost_mae_sum += float(metrics["cost_mae"]) * bs
        mae_target_sum += np.asarray([metrics["mae_car"], metrics["mae_ped"], metrics["mae_cyc"]], dtype=np.float64) * bs

        acc = metrics.get("jucp_acc_per_threshold", [])
        if acc:
            cur = np.asarray(acc, dtype=np.float64)
            acc_sum = cur * bs if acc_sum is None else acc_sum + cur * bs

        postfix = {
            "loss": total_loss_sum / max(1, total_samples),
            "mae": cost_mae_sum / max(1, total_samples),
        }
        if acc_sum is not None:
            postfix["jucp_acc"] = float(np.mean(acc_sum / max(1, total_samples)))
        pbar.set_postfix(**postfix)

    out_metrics: Dict[str, object] = {
        "total_loss": total_loss_sum / max(1, total_samples),
        "reg_loss": reg_loss_sum / max(1, total_samples),
        "threshold_loss": thr_loss_sum / max(1, total_samples),
        "cost_mae": cost_mae_sum / max(1, total_samples),
        "mae_car": mae_target_sum[0] / max(1, total_samples),
        "mae_ped": mae_target_sum[1] / max(1, total_samples),
        "mae_cyc": mae_target_sum[2] / max(1, total_samples),
    }

    if acc_sum is not None:
        acc_per_threshold = (acc_sum / max(1, total_samples)).tolist()
        out_metrics["mean_jucp_acc"] = float(np.mean(acc_per_threshold))
        out_metrics["jucp_acc_per_threshold"] = acc_per_threshold
    else:
        out_metrics["mean_jucp_acc"] = 0.0
        out_metrics["jucp_acc_per_threshold"] = []
    return out_metrics


# ============================================================
# Save / load / logging
# ============================================================


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, epoch: int, metrics: Dict[str, object], args: argparse.Namespace) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def load_pretrained_weights(model: nn.Module, ckpt_path: Optional[str], strict: bool = False) -> None:
    if ckpt_path is None or ckpt_path == "":
        return
    print(f"[INFO] Loading pretrained checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned = {}
    for k, v in state_dict.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    print(f"[INFO] Pretrained loaded. strict={strict}")
    if missing:
        print(f"[INFO] Missing keys: {len(missing)}")
        print("       " + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
    if unexpected:
        print(f"[INFO] Unexpected keys: {len(unexpected)}")
        print("       " + ", ".join(unexpected[:20]) + (" ..." if len(unexpected) > 20 else ""))


def append_metrics_csv(csv_path: Path, epoch: int, split: str, metrics: Dict[str, object], ap_drop_scale: float) -> None:
    ensure_dir(csv_path.parent)
    row = {
        "epoch": epoch,
        "split": split,
        "total_loss": metrics.get("total_loss", 0.0),
        "reg_loss": metrics.get("reg_loss", 0.0),
        "threshold_loss": metrics.get("threshold_loss", 0.0),
        "cost_mae": metrics.get("cost_mae", 0.0),
        "mae_car": metrics.get("mae_car", 0.0),
        "mae_ped": metrics.get("mae_ped", 0.0),
        "mae_cyc": metrics.get("mae_cyc", 0.0),
        "raw_cost_mae": metrics.get("cost_mae", 0.0) / max(ap_drop_scale, 1e-12),
        "raw_mae_car": metrics.get("mae_car", 0.0) / max(ap_drop_scale, 1e-12),
        "raw_mae_ped": metrics.get("mae_ped", 0.0) / max(ap_drop_scale, 1e-12),
        "raw_mae_cyc": metrics.get("mae_cyc", 0.0) / max(ap_drop_scale, 1e-12),
        "mean_jucp_acc": metrics.get("mean_jucp_acc", 0.0),
        "jucp_acc_per_threshold": json.dumps(metrics.get("jucp_acc_per_threshold", [])),
    }
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def format_metrics(prefix: str, metrics: Dict[str, object], ap_drop_scale: float) -> str:
    acc = metrics.get("jucp_acc_per_threshold", [])
    acc_str = ",".join([f"{x:.3f}" for x in acc]) if isinstance(acc, list) else str(acc)
    raw_mae = metrics.get("cost_mae", 0.0) / max(ap_drop_scale, 1e-12)
    return (
        f"{prefix}: "
        f"loss={metrics.get('total_loss', 0.0):.6f}, "
        f"reg={metrics.get('reg_loss', 0.0):.6f}, "
        f"thr={metrics.get('threshold_loss', 0.0):.6f}, "
        f"mae={metrics.get('cost_mae', 0.0):.6f}, "
        f"raw_mae={raw_mae:.6f}, "
        f"raw=[car {metrics.get('mae_car', 0.0) / max(ap_drop_scale, 1e-12):.6f}, "
        f"ped {metrics.get('mae_ped', 0.0) / max(ap_drop_scale, 1e-12):.6f}, "
        f"cyc {metrics.get('mae_cyc', 0.0) / max(ap_drop_scale, 1e-12):.6f}], "
        f"mean_jucp_acc={metrics.get('mean_jucp_acc', 0.0):.4f}, "
        f"acc_thresholds=[{acc_str}]"
    )


# ============================================================
# Args / main
# ============================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train SparseConv cost-only JUQP/JUCP proxy network")

    # Data
    p.add_argument("--velodyne_dir", type=str, required=True)
    p.add_argument("--train_split", type=str, required=True)
    p.add_argument("--ap_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="router_work_dirs/sparse_cost_proxy")

    p.add_argument("--val_split", type=str, default=None)
    p.add_argument("--val_ap_csv", type=str, default=None)
    p.add_argument("--test_split", type=str, default=None)
    p.add_argument("--test_ap_csv", type=str, default=None)
    p.add_argument("--test_every", type=int, default=0)

    # Voxelization
    p.add_argument("--voxel_size", type=float, nargs=3, default=[0.16, 0.16, 0.16])
    p.add_argument("--point_cloud_range", type=float, nargs=6, default=[0.0, -40.0, -3.0, 70.4, 40.0, 1.0])
    p.add_argument("--max_voxels", type=int, default=50000)
    p.add_argument("--no_abs_xyz", action="store_true", help="do not include normalized absolute xyz in voxel features")

    # Thresholds for deriving JUQP labels from cost.
    p.add_argument("--thresholds", type=str, default=None)

    # Model
    p.add_argument("--num_cost_heads", type=int, default=6)
    p.add_argument("--num_targets", type=int, default=3)
    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--allow_negative_cost", action="store_true")

    # Augmentation
    p.add_argument("--use_rotation_aug", action="store_true")
    p.add_argument("--jitter_std", type=float, default=0.0)

    # Optimization
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--pretrained_ckpt", type=str, default=None)
    p.add_argument("--strict_load", action="store_true")

    # Loss
    p.add_argument("--ap_weights", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    p.add_argument("--head_weights", type=float, nargs="*", default=None)
    p.add_argument("--ap_drop_scale", type=float, default=100.0)
    p.add_argument("--lambda_threshold", type=float, default=0.0)

    # Save / logging
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--log_file", type=str, default=None)

    return p.parse_args()


def build_loader(args: argparse.Namespace, split_file: str, ap_csv: str, training: bool) -> Tuple[DataLoader, List[int], int]:
    dataset = SparseCostProxyDataset(
        velodyne_dir=args.velodyne_dir,
        split_file=split_file,
        ap_csv=ap_csv,
        voxel_size=args.voxel_size,
        pc_range=args.point_cloud_range,
        max_voxels=args.max_voxels,
        num_cost_heads=args.num_cost_heads,
        training=training,
        ap_drop_scale=args.ap_drop_scale,
        use_rotation_aug=args.use_rotation_aug,
        jitter_std=args.jitter_std,
        use_abs_xyz=not args.no_abs_xyz,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=training,
        collate_fn=sparse_collate_fn,
    )
    return loader, dataset.spatial_shape, dataset.num_point_features


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    setup_file_logger(out_dir, args.log_file)

    thresholds = parse_thresholds(args.thresholds, scale=args.ap_drop_scale)
    if thresholds is not None:
        print(f"[INFO] Using {thresholds.shape[0]} threshold settings. Values are scaled by ap_drop_scale={args.ap_drop_scale}.")
        for i, thr in enumerate(thresholds.tolist()):
            print(f"  threshold_{i}: car={thr[0]}, ped={thr[1]}, cyc={thr[2]}")

    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    train_loader, spatial_shape, input_channels = build_loader(args, args.train_split, args.ap_csv, training=True)

    val_loader = None
    if args.val_split is not None:
        val_ap_csv = args.val_ap_csv if args.val_ap_csv is not None else args.ap_csv
        val_loader, _, _ = build_loader(args, args.val_split, val_ap_csv, training=False)

    test_loader = None
    if args.test_split is not None or args.test_ap_csv is not None:
        if args.test_split is None or args.test_ap_csv is None:
            raise ValueError("To enable test evaluation, provide both --test_split and --test_ap_csv")
        test_loader, _, _ = build_loader(args, args.test_split, args.test_ap_csv, training=False)

    model = SparseCostProxyNet(
        input_channels=input_channels,
        spatial_shape=spatial_shape,
        feat_dim=args.feat_dim,
        num_cost_heads=args.num_cost_heads,
        num_targets=args.num_targets,
        cost_nonnegative=not args.allow_negative_cost,
    ).to(device)

    load_pretrained_weights(model, args.pretrained_ckpt, strict=args.strict_load)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ap_weights = torch.tensor(args.ap_weights, dtype=torch.float32, device=device)
    head_weights = None
    if args.head_weights is not None and len(args.head_weights) > 0:
        if len(args.head_weights) != args.num_cost_heads:
            raise ValueError(f"--head_weights must contain {args.num_cost_heads} values")
        head_weights = torch.tensor(args.head_weights, dtype=torch.float32, device=device)

    thresholds_device = thresholds.to(device) if thresholds is not None else None
    metrics_csv = out_dir / "metrics.csv"
    best_score = -1e18

    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.epochs} ==========")

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
            ap_weights=ap_weights,
            head_weights=head_weights,
            thresholds=thresholds_device,
            lambda_threshold=args.lambda_threshold,
        )
        scheduler.step()
        print(format_metrics("Train", train_metrics, args.ap_drop_scale))
        append_metrics_csv(metrics_csv, epoch, "train", train_metrics, args.ap_drop_scale)

        all_metrics = {"train": train_metrics}
        score = float(train_metrics["mean_jucp_acc"]) if thresholds is not None else -float(train_metrics["cost_mae"])

        if val_loader is not None:
            val_metrics = run_one_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                train=False,
                ap_weights=ap_weights,
                head_weights=head_weights,
                thresholds=thresholds_device,
                lambda_threshold=args.lambda_threshold,
            )
            print(format_metrics("Val", val_metrics, args.ap_drop_scale))
            append_metrics_csv(metrics_csv, epoch, "val", val_metrics, args.ap_drop_scale)
            all_metrics["val"] = val_metrics
            score = float(val_metrics["mean_jucp_acc"]) if thresholds is not None else -float(val_metrics["cost_mae"])

        if test_loader is not None and args.test_every > 0 and (epoch % args.test_every == 0 or epoch == args.epochs):
            test_metrics = run_one_epoch(
                model=model,
                loader=test_loader,
                optimizer=None,
                device=device,
                train=False,
                ap_weights=ap_weights,
                head_weights=head_weights,
                thresholds=thresholds_device,
                lambda_threshold=args.lambda_threshold,
            )
            print(format_metrics("Test", test_metrics, args.ap_drop_scale))
            append_metrics_csv(metrics_csv, epoch, "test", test_metrics, args.ap_drop_scale)
            all_metrics["test"] = test_metrics

        save_checkpoint(out_dir / "latest.pth", model, optimizer, scheduler, epoch, all_metrics, args)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(out_dir / f"epoch_{epoch:03d}.pth", model, optimizer, scheduler, epoch, all_metrics, args)
        if score > best_score:
            best_score = score
            save_checkpoint(out_dir / "best.pth", model, optimizer, scheduler, epoch, all_metrics, args)
            print(f"Saved best checkpoint: score={best_score:.6f}")

    print(f"\nTraining finished. Outputs saved to: {out_dir}")
    print(f"Best score: {best_score:.6f}")


if __name__ == "__main__":
    main()
