#!/usr/bin/env python3
"""Train or adapt the five-head geometry-only nuScenes loss proxy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from train_cost_proxy import SparseCostProxyNet, augment_points, voxelize_points
from train_scannet_rate_aware_proxy import flexible_load


LEVEL_ORDER = [4, 3, 2, 1, 0]  # fine-to-coarse; L5=64mm is implicit zero.
HEAD_SEMANTICS = [
    '128mm - 64mm', '256mm - 64mm', '512mm - 64mm',
    '1024mm - 64mm', '2048mm - 64mm']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-csv', required=True)
    parser.add_argument('--train-split', required=True)
    parser.add_argument('--val-split', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--pretrained-ckpt', default='')
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--thresholds', type=float, nargs=6, required=True)
    parser.add_argument('--target-scale', type=float, default=1.0)
    parser.add_argument('--voxel-size', type=float, nargs=3,
                        default=[0.16, 0.16, 0.16])
    parser.add_argument('--point-cloud-range', type=float, nargs=6,
                        default=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    parser.add_argument('--max-voxels', type=int, default=50000)
    parser.add_argument('--feat-dim', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--lambda-threshold', type=float, default=0.1)
    parser.add_argument('--jitter-std', type=float, default=0.005)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260819)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_tokens(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines()
            if line.strip()]


class NuScenesProxyDataset(Dataset):
    def __init__(self, loss_csv, split_file, voxel_size, point_cloud_range,
                 max_voxels, target_scale, training, jitter_std):
        frame = pd.read_csv(
            loss_csv, dtype={'scene_id': str, 'sample_idx': str})
        frame = frame.set_index('scene_id', drop=False)
        tokens = read_tokens(split_file)
        missing = [token for token in tokens if token not in frame.index]
        if missing:
            raise ValueError(f'{len(missing)} split tokens lack loss rows')
        self.items = []
        for token in tokens:
            row = frame.loc[token]
            target = np.asarray(
                [[float(row[f'L{level}_signed_delta'])]
                 for level in LEVEL_ORDER], dtype=np.float32)
            target *= float(target_scale)
            path = Path(str(row['lidar_path']))
            if not path.is_file():
                raise FileNotFoundError(path)
            self.items.append((token, path, target))
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.pc_range = np.asarray(point_cloud_range, dtype=np.float32)
        self.max_voxels = int(max_voxels)
        self.training = bool(training)
        self.jitter_std = float(jitter_std)
        grid = np.floor(
            (self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size
        ).astype(np.int32)
        self.spatial_shape = grid[[2, 1, 0]].tolist()
        self.num_point_features = 7
        print(
            f'Dataset {split_file}: samples={len(self.items)} '
            f'spatial_shape={self.spatial_shape}', flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        token, path, target = self.items[index]
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 5:
            raise ValueError(f'Invalid nuScenes point file: {path}')
        points = raw.reshape(-1, 5)[:, :3]
        if self.training:
            points = augment_points(points, True, self.jitter_std)
        features, coords = voxelize_points(
            points, self.voxel_size, self.pc_range, self.max_voxels,
            use_abs_xyz=True, include_intensity=False)
        return dict(
            scene_id=token, voxel_features=torch.from_numpy(features),
            voxel_coords=torch.from_numpy(coords),
            loss_delta=torch.from_numpy(target))


def collate_batch(batch):
    features, coords, targets, tokens = [], [], [], []
    for batch_index, item in enumerate(batch):
        cur = item['voxel_coords'].int()
        batch_col = torch.full((len(cur), 1), batch_index, dtype=torch.int32)
        features.append(item['voxel_features'].float())
        coords.append(torch.cat([batch_col, cur], dim=1))
        targets.append(item['loss_delta'].float())
        tokens.append(item['scene_id'])
    return dict(
        scene_id=tokens, voxel_features=torch.cat(features),
        voxel_coords=torch.cat(coords).int(), loss_delta=torch.stack(targets),
        batch_size=len(batch))


def select_levels(fine_to_coarse: torch.Tensor,
                  thresholds: torch.Tensor) -> torch.Tensor:
    outputs = []
    for threshold in thresholds:
        valid = fine_to_coarse[:, :, 0] <= threshold
        labels = torch.zeros(len(valid), dtype=torch.long, device=valid.device)
        assigned = torch.zeros(len(valid), dtype=torch.bool, device=valid.device)
        for label in range(5, 0, -1):
            choose = valid[:, label - 1] & ~assigned
            labels[choose] = label
            assigned |= choose
        outputs.append(labels)
    return torch.stack(outputs, dim=1)


def objective(prediction, target, thresholds, lambda_threshold):
    regression = F.smooth_l1_loss(prediction, target)
    terms = []
    for threshold in thresholds:
        valid = (target <= threshold).float()
        terms.append(F.binary_cross_entropy_with_logits(
            threshold - prediction, valid))
    threshold_loss = torch.stack(terms).mean()
    return regression + lambda_threshold * threshold_loss, regression, threshold_loss


def reduce_sum(value: torch.Tensor):
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)


def set_training_mode(model, training, freeze_backbone):
    module = model.module if isinstance(model, DistributedDataParallel) else model
    if training and not freeze_backbone:
        module.train()
        return
    module.eval()
    if training:
        module.cost_heads.train(True)


def run_epoch(model, loader, device, thresholds, lambda_threshold,
              optimizer, rank, freeze_backbone):
    training = optimizer is not None
    set_training_mode(model, training, freeze_backbone)
    # samples, objective, regression, threshold, absolute error, six accuracies
    sums = torch.zeros(11, dtype=torch.float64, device=device)
    progress = tqdm(loader, disable=rank != 0,
                    desc='train' if training else 'val', dynamic_ncols=True)
    for batch in progress:
        features = batch['voxel_features'].to(device, non_blocking=True)
        coords = batch['voxel_coords'].to(device, non_blocking=True)
        target = batch['loss_delta'].to(device, non_blocking=True)
        batch_size = int(batch['batch_size'])
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(features, coords, batch_size)['cost_pred']
            total, regression, threshold_loss = objective(
                prediction, target, thresholds, lambda_threshold)
            if training:
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
        with torch.no_grad():
            pred_levels = select_levels(prediction, thresholds)
            true_levels = select_levels(target, thresholds)
            sums[0] += batch_size
            sums[1] += total.double() * batch_size
            sums[2] += regression.double() * batch_size
            sums[3] += threshold_loss.double() * batch_size
            sums[4] += torch.abs(prediction - target).mean().double() * batch_size
            sums[5:11] += (pred_levels == true_levels).double().sum(dim=0)
        if rank == 0:
            progress.set_postfix(loss=float(total))
    reduce_sum(sums)
    count = max(1.0, float(sums[0]))
    accuracy = (sums[5:11] / count).cpu().tolist()
    return dict(
        samples=int(sums[0]), total_loss=float(sums[1] / count),
        regression_loss=float(sums[2] / count),
        threshold_loss=float(sums[3] / count),
        scaled_mae=float(sums[4] / count),
        threshold_accuracy=accuracy,
        mean_accuracy=float(np.mean(accuracy)))


def build_loader(args, split, training, distributed):
    dataset = NuScenesProxyDataset(
        args.loss_csv, split, args.voxel_size, args.point_cloud_range,
        args.max_voxels, args.target_scale, training, args.jitter_std)
    sampler = DistributedSampler(
        dataset, shuffle=training, drop_last=training) if distributed else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=training and sampler is None, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=training,
        collate_fn=collate_batch, persistent_workers=args.workers > 0)
    return loader, dataset, sampler


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, args,
                    source_checkpoint):
    module = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save(dict(
        epoch=epoch, model=module.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), metrics=metrics, args=vars(args),
        initialization='kitti_pretrained' if args.pretrained_ckpt else 'scratch',
        source_kitti_checkpoint=(
            str(Path(args.pretrained_ckpt).resolve())
            if args.pretrained_ckpt else None),
        source_kitti_epoch=source_checkpoint.get('epoch'),
        frozen_backbone=bool(args.freeze_backbone),
        trainable_prefixes=(['cost_heads.'] if args.freeze_backbone else ['*']),
        detector_name='centerpoint_nuscenes_xyz',
        target_type='signed_centerpoint_total_loss_delta',
        input_feature_mode='geometry_only_7d',
        head_semantics=HEAD_SEMANTICS), path)


def append_metrics(path: Path, epoch: int, split: str, metrics: dict,
                   target_scale: float):
    row = dict(
        epoch=epoch, split=split, total_loss=metrics['total_loss'],
        regression_loss=metrics['regression_loss'],
        threshold_loss=metrics['threshold_loss'],
        raw_mae=metrics['scaled_mae'] / target_scale,
        mean_label_accuracy=metrics['mean_accuracy'],
        threshold_accuracy=json.dumps(metrics['threshold_accuracy']))
    exists = path.exists()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    if args.thresholds != sorted(args.thresholds):
        raise ValueError('--thresholds must be nondecreasing')
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', timeout=timedelta(minutes=5))
        rank = dist.get_rank()
        device = torch.device('cuda', local_rank)
    else:
        rank = 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed + rank)
    out = Path(args.out_dir).resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'args.json').write_text(json.dumps(vars(args), indent=2))
    if distributed:
        dist.barrier()
    train_loader, train_dataset, train_sampler = build_loader(
        args, args.train_split, True, distributed)
    val_loader, _, val_sampler = build_loader(
        args, args.val_split, False, distributed)
    model = SparseCostProxyNet(
        input_channels=7, spatial_shape=train_dataset.spatial_shape,
        feat_dim=args.feat_dim, num_cost_heads=5, num_targets=1,
        cost_nonnegative=False, monotonic_cost=False).to(device)
    source_checkpoint = {}
    if args.pretrained_ckpt:
        source_checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu')
        source_state = source_checkpoint.get('model', source_checkpoint)
        normalized = {}
        for key, value in source_state.items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            if key.startswith('base.'):
                key = key[len('base.'):]
            if key in model.state_dict():
                normalized[key] = value
        missing = sorted(set(model.state_dict()) - set(normalized))
        if missing:
            raise RuntimeError(
                f'Incompatible KITTI checkpoint misses {len(missing)} tensors: '
                f'{missing[:8]}')
        flexible_load(model, normalized)
        if rank == 0:
            print(
                f'Loaded KITTI geometry-only proxy from {args.pretrained_ckpt} '
                f'(epoch={source_checkpoint.get("epoch")})', flush=True)
    elif args.freeze_backbone:
        raise ValueError('--freeze-backbone requires --pretrained-ckpt')
    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.cost_heads.parameters():
            parameter.requires_grad_(True)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    frozen_count = sum(
        parameter.numel() for parameter in model.parameters()
        if not parameter.requires_grad)
    if rank == 0:
        print(
            f'Parameters: trainable={trainable_count} frozen={frozen_count} '
            f'freeze_backbone={args.freeze_backbone}', flush=True)
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index,
            broadcast_buffers=False)
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    thresholds = torch.tensor(
        args.thresholds, dtype=torch.float32, device=device) * args.target_scale
    metrics_path = out / 'metrics.csv'
    best_score, best_epoch, stale = -float('inf'), 0, 0
    begin = time.time()
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, device, thresholds,
            args.lambda_threshold, optimizer, rank, args.freeze_backbone)
        val_metrics = run_epoch(
            model, val_loader, device, thresholds,
            args.lambda_threshold, None, rank, args.freeze_backbone)
        scheduler.step()
        score = val_metrics['mean_accuracy']
        if rank == 0:
            append_metrics(metrics_path, epoch, 'train', train_metrics,
                           args.target_scale)
            append_metrics(metrics_path, epoch, 'val', val_metrics,
                           args.target_scale)
            print(
                f'Epoch {epoch:03d}: train_acc={train_metrics["mean_accuracy"]:.4f} '
                f'val_acc={score:.4f} '
                f'val_raw_mae={val_metrics["scaled_mae"]/args.target_scale:.5f}',
                flush=True)
            save_checkpoint(out / 'latest.pth', model, optimizer, scheduler,
                            epoch, val_metrics, args, source_checkpoint)
            if score > best_score + 1e-6:
                best_score, best_epoch, stale = score, epoch, 0
                save_checkpoint(out / 'best.pth', model, optimizer, scheduler,
                                epoch, val_metrics, args, source_checkpoint)
            else:
                stale += 1
        stop = torch.tensor(
            int(rank == 0 and stale >= args.patience), device=device)
        if distributed:
            dist.broadcast(stop, src=0)
        if int(stop):
            break
    if rank == 0:
        summary = dict(
            best_epoch=best_epoch, best_val_mean_accuracy=best_score,
            elapsed_seconds=time.time() - begin,
            initialization=(
                'kitti_pretrained' if args.pretrained_ckpt else 'scratch'),
            source_kitti_checkpoint=(
                str(Path(args.pretrained_ckpt).resolve())
                if args.pretrained_ckpt else None),
            source_kitti_epoch=source_checkpoint.get('epoch'),
            frozen_backbone=bool(args.freeze_backbone),
            trainable_parameters=trainable_count,
            frozen_parameters=frozen_count,
            num_heads=5, head_semantics=HEAD_SEMANTICS)
        (out / 'TRAINING_COMPLETE.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
