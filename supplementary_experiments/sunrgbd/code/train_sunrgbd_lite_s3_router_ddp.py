#!/usr/bin/env python3
"""Train Lite-S3 on six absolute SUN RGB-D VoteNet losses and monotonic BPP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from lite_s3_absolute_loss_monotonic_rate_proxy import (
    LiteS3AbsoluteLossMonotonicRateProxy,
    count_parameters,
)
from gpu_voxelizer import voxelize_batch_gpu


QSTEPS_MM = (160.0, 120.0, 100.0, 80.0, 60.0, 40.0)
NUM_LEVELS = 6


def read_ids(path: Path) -> list[str]:
    return [f'{int(line):06d}' for line in path.read_text().splitlines() if line.strip()]


def load_labels(
        loss_csv: Path, bpp_csv: Path
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    losses = {}
    with loss_csv.open(newline='') as handle:
        for row in csv.DictReader(handle):
            losses[row['scene_id']] = np.asarray(
                [float(row[f'L{i}_total_loss']) for i in range(NUM_LEVELS)],
                dtype=np.float32,
            )
    rates: dict[str, np.ndarray] = {}
    with bpp_csv.open(newline='') as handle:
        for row in csv.DictReader(handle):
            sid = row['scene_id']
            rates.setdefault(sid, np.full(NUM_LEVELS, np.nan, np.float32))[
                int(row['rate_id'])
            ] = float(row['bpp'])
    labels = {}
    for sid in losses.keys() & rates.keys():
        if np.isfinite(losses[sid]).all() and np.isfinite(rates[sid]).all():
            raw_bpp = rates[sid].copy()
            monotonic_bpp = np.maximum.accumulate(raw_bpp)
            labels[sid] = (losses[sid], monotonic_bpp, raw_bpp)
    return labels


class SUNRGBDRouterDataset(Dataset):
    def __init__(self, points_dir: Path, split_file: Path, loss_csv: Path,
                 bpp_csv: Path, training: bool):
        self.points_dir = points_dir
        self.ids = read_ids(split_file)
        self.labels = load_labels(loss_csv, bpp_csv)
        missing = [sid for sid in self.ids if sid not in self.labels]
        if missing:
            raise RuntimeError(f'Missing labels for {len(missing)} scenes: {missing[:5]}')
        self.training = training
        values = np.stack([self.labels[sid][1] for sid in self.ids])
        self.mean_log_bpp = np.log1p(values).mean(axis=0).astype(np.float32)
        raw_values = np.stack([self.labels[sid][2] for sid in self.ids])
        self.raw_bpp_monotonic_violation_rate = float(
            (np.diff(raw_values, axis=1) < 0).mean())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        sid = self.ids[index]
        points = np.fromfile(self.points_dir / f'{sid}.bin', dtype=np.float32).reshape(-1, 6)
        loss, bpp, raw_bpp = self.labels[sid]
        return {
            'scene_id': sid,
            'points': torch.from_numpy(points[:, :3].copy()),
            'loss': torch.from_numpy(loss.copy()),
            'bpp': torch.from_numpy(bpp.copy()),
            'raw_bpp': torch.from_numpy(raw_bpp.copy()),
        }


def collate_raw(batch):
    return {
        'scene_ids': [item['scene_id'] for item in batch],
        'points': [item['points'] for item in batch],
        'loss': torch.stack([item['loss'] for item in batch]),
        'bpp': torch.stack([item['bpp'] for item in batch]),
        'raw_bpp': torch.stack([item['raw_bpp'] for item in batch]),
    }


def set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed((seed + rank) % (2**32 - 1))
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def augment(points: list[torch.Tensor], device: torch.device, enabled: bool) -> list[torch.Tensor]:
    output = []
    for cloud in points:
        cloud = cloud.to(device, non_blocking=True)
        if enabled:
            angle = (torch.rand((), device=device) * 2.0 - 1.0) * math.pi
            cosine, sine = torch.cos(angle), torch.sin(angle)
            rotation = torch.stack([
                torch.stack([cosine, -sine, cosine.new_zeros(())]),
                torch.stack([sine, cosine, cosine.new_zeros(())]),
                torch.stack([cosine.new_zeros(()), cosine.new_zeros(()), cosine.new_ones(())]),
            ])
            cloud = cloud @ rotation.T
            cloud = cloud + torch.randn_like(cloud) * 0.003
        output.append(cloud)
    return output


def run_epoch(model, loader, optimizer, device, loss_scales, args):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(7, dtype=torch.float64, device=device)
    for batch_index, batch in enumerate(loader):
        points = augment(batch['points'], device, training)
        true_loss = batch['loss'].to(device, non_blocking=True)
        true_bpp = batch['bpp'].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            voxel_features, voxel_coords = voxelize_batch_gpu(
                points, args.voxel_size, args.point_cloud_range, args.max_voxels,
                use_abs_xyz=True, include_intensity=False,
                random_subsample=training,
            )
            output = model(voxel_features, voxel_coords, len(points))
            loss_reg = F.smooth_l1_loss(
                output['loss_pred'] / loss_scales[None, :],
                true_loss / loss_scales[None, :],
            )
            rate_reg = F.smooth_l1_loss(output['rate_log_pred'], torch.log1p(true_bpp))
            total = args.loss_weight * loss_reg + args.rate_weight * rate_reg
            if training:
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
        count = len(points)
        totals[0] += count
        totals[1] += float(total.detach()) * count
        totals[2] += float(loss_reg.detach()) * count
        totals[3] += float(rate_reg.detach()) * count
        totals[4] += float(torch.abs(output['loss_pred'] - true_loss).mean()) * count
        totals[5] += float(torch.abs(output['bpp_pred'] - true_bpp).mean()) * count
        totals[6] += float((torch.diff(output['bpp_pred'], dim=1) < 0).sum())
        if batch_index == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            print(json.dumps({
                'first_batch': True, 'training': training, 'batch_size': count,
                'active_voxels': int(voxel_features.shape[0]),
                'loss': float(total.detach()),
            }), flush=True)
    if dist.is_initialized():
        dist.all_reduce(totals)
    count = max(float(totals[0]), 1.0)
    return {
        'samples': int(totals[0]), 'total_loss': float(totals[1] / count),
        'loss_reg': float(totals[2] / count), 'rate_reg': float(totals[3] / count),
        'loss_mae': float(totals[4] / count), 'bpp_mae': float(totals[5] / count),
        'bpp_monotonic_violation_rate': float(totals[6] / (count * 5.0)),
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, args,
                    loss_scales, initialization_report):
    torch.save({
        'epoch': epoch, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'metrics': metrics, 'args': vars(args), 'qsteps_mm': QSTEPS_MM,
        'loss_scales': loss_scales.detach().cpu().tolist(),
        'model_type': 'lite_s3_six_independent_absolute_loss_plus_monotonic_bpp',
        'routing_rule': 'argmin_q predicted_loss[q] + lambda * predicted_bpp[q]',
        'checkpoint_selection': 'lowest full-training-set regression loss',
        'initialization_report': initialization_report,
        'parameter_counts': count_parameters(model),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--points-dir', required=True, type=Path)
    parser.add_argument('--split-file', required=True, type=Path)
    parser.add_argument('--loss-csv', required=True, type=Path)
    parser.add_argument('--bpp-csv', required=True, type=Path)
    parser.add_argument('--init-checkpoint', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--backbone-lr', type=float, default=5e-5)
    parser.add_argument('--head-lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--loss-weight', type=float, default=2.0)
    parser.add_argument('--rate-weight', type=float, default=1.0)
    parser.add_argument('--clip-grad-norm', type=float, default=5.0)
    parser.add_argument('--voxel-size', type=float, nargs=3, default=[0.16, 0.16, 0.16])
    parser.add_argument('--point-cloud-range', type=float, nargs=6,
                        default=[-8.0, -8.0, -2.0, 8.0, 8.0, 6.0])
    parser.add_argument('--max-voxels', type=int, default=50000)
    parser.add_argument('--max-scenes', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20260828)
    args = parser.parse_args()

    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group('nccl')
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    if world > 7:
        raise RuntimeError(f'Server GPU cap is seven, got {world}')
    device = torch.device('cuda', local_rank)
    set_seed(args.seed, rank)

    dataset = SUNRGBDRouterDataset(
        args.points_dir, args.split_file, args.loss_csv, args.bpp_csv, training=True)
    if args.max_scenes > 0:
        dataset.ids = dataset.ids[:args.max_scenes]
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if distributed else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=sampler is None,
        sampler=sampler, num_workers=args.workers, pin_memory=True,
        drop_last=False, collate_fn=collate_raw,
        persistent_workers=args.workers > 0,
    )
    all_losses = np.stack([dataset.labels[sid][0] for sid in dataset.ids])
    loss_scale_np = np.maximum(np.median(all_losses, axis=0), 1e-3).astype(np.float32)
    grid_xyz = np.floor(
        (np.asarray(args.point_cloud_range[3:]) - np.asarray(args.point_cloud_range[:3]))
        / np.asarray(args.voxel_size)
    ).astype(int)
    spatial_shape = grid_xyz[::-1].tolist()
    model = LiteS3AbsoluteLossMonotonicRateProxy(
        spatial_shape, 256, loss_scale_np, dataset.mean_log_bpp).to(device)
    initialization_report = model.load_legacy_checkpoint(args.init_checkpoint)

    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        (heads if '.cost_heads.' in name or name.startswith('rate_head.') else backbone).append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': backbone, 'lr': args.backbone_lr},
        {'params': heads, 'lr': args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    bare = model.module if distributed else model
    loss_scales = torch.tensor(loss_scale_np, dtype=torch.float32, device=device)

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'args.json').write_text(json.dumps(vars(args), default=str, indent=2))
        (args.output_dir / 'initialization_report.json').write_text(
            json.dumps(initialization_report, indent=2))
        print(json.dumps({
            'world_size': world, 'train_scenes': len(dataset),
            'spatial_shape_zyx': spatial_shape, 'loss_scales': loss_scale_np.tolist(),
            'mean_log_bpp': dataset.mean_log_bpp.tolist(),
            'raw_bpp_monotonic_violation_rate': (
                dataset.raw_bpp_monotonic_violation_rate),
            'parameters': count_parameters(bare),
        }, indent=2), flush=True)

    best_loss, best_epoch, stale = math.inf, 0, 0
    metrics_path = args.output_dir / 'metrics.csv'
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        metrics = run_epoch(model, loader, optimizer, device, loss_scales, args)
        scheduler.step()
        if rank == 0:
            improved = metrics['total_loss'] < best_loss - 1e-7
            if improved:
                best_loss, best_epoch, stale = metrics['total_loss'], epoch, 0
            else:
                stale += 1
            row = {'epoch': epoch, **metrics, 'best_epoch': best_epoch, 'best_loss': best_loss}
            with metrics_path.open('a', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=row)
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            save_checkpoint(args.output_dir / 'latest.pth', bare, optimizer, scheduler,
                            epoch, row, args, loss_scales, initialization_report)
            if improved:
                save_checkpoint(args.output_dir / 'best.pth', bare, optimizer, scheduler,
                                epoch, row, args, loss_scales, initialization_report)
            print(json.dumps(row), flush=True)
        stop = torch.tensor([int(stale >= args.patience if rank == 0 else 0)], device=device)
        if distributed:
            dist.broadcast(stop, 0)
        if stop.item():
            break

    if rank == 0:
        summary = {
            'status': 'complete', 'best_epoch': best_epoch,
            'best_all_training_regression_loss': best_loss,
            'elapsed_seconds': time.time() - started, 'gpus': world,
            'parameters': count_parameters(bare),
        }
        (args.output_dir / 'TRAINING_COMPLETE.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
