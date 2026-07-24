#!/usr/bin/env python3
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import MinkowskiEngine as ME


def configure_unicorn_args():
    sys.argv = ['train_unicorn_kitti.py', '--only_global_topk', '0']


def add_unicorn_to_path(unicorn_root):
    unicorn_root = Path(unicorn_root).resolve()
    sys.path.insert(0, str(unicorn_root))
    return unicorn_root


def add_geometry_module_to_path(unicorn_root, module):
    geometry_dir = 'lossless_geometry' if module == 'lossless' else 'lossy_geometry'
    sys.path.insert(0, str(unicorn_root / geometry_dir))


def read_split(path):
    with open(path) as f:
        return [line.strip().zfill(6) for line in f if line.strip()]


def collect_files(velodyne, split_file):
    velodyne = Path(velodyne)
    return [velodyne / f'{frame_id}.bin' for frame_id in read_split(split_file)]


def load_sparse_from_bin(path, posq, device):
    points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)[:, :3]
    coords_mm = np.round(points.astype(np.float64) * 1000.0).astype(np.int64)
    offset = coords_mm.min(axis=0)
    coords = np.round((coords_mm - offset).astype(np.float64) / float(posq)).astype(np.int32)
    coords = np.unique(coords, axis=0).astype(np.int32)
    coords_t = torch.from_numpy(coords).int()
    feats_t = torch.ones((coords_t.shape[0], 1), dtype=torch.float32)
    coords_b, feats_b = ME.utils.sparse_collate([coords_t], [feats_t])
    return ME.SparseTensor(features=feats_b, coordinates=coords_b, device=device)


def train_sr_one(model, x, get_bce, get_bits, weight_distortion, weight_bitrate):
    out_set_list = model(x, training=True)
    loss = 0
    stats = {'bce': 0.0, 'bpp': 0.0}
    for out_set in out_set_list:
        bce = 0
        for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
            bce = bce + get_bce(out_cls, ground_truth) / float(len(x))
        if 'likelihood' in out_set:
            bpp = get_bits(out_set['likelihood']) / float(len(x))
        else:
            bpp = torch.zeros((), device=x.device)
        loss = loss + weight_distortion * bce + weight_bitrate * bpp
        stats['bce'] += float(bce.detach().cpu())
        stats['bpp'] += float(bpp.detach().cpu())
    return loss, stats


def save_checkpoint(path, model, args, epoch, global_step):
    torch.save({
        'model': model.state_dict(),
        'module': args.module,
        'epoch': epoch,
        'global_step': global_step,
        'args': vars(args),
    }, path)


def train_sr_pair(model, x_low, x_high, get_bce, weight_distortion):
    x_low = ME.SparseTensor(
        features=x_low.F,
        coordinates=x_low.C,
        tensor_stride=2,
        device=x_low.device,
    )
    out_set = model.upsampler(x_low, x_high=x_high)
    bce = 0
    for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
        bce = bce + get_bce(out_cls, ground_truth) / float(len(x_high))
    loss = weight_distortion * bce
    return loss, {'bce': float(bce.detach().cpu()), 'bpp': 0.0}


def train_sr(model, x, get_bce, get_bits, quantize_sparse_tensor,
             weight_distortion, weight_bitrate, sr_posq_scales=None):
    if not sr_posq_scales:
        return train_sr_one(model, x, get_bce, get_bits, weight_distortion, weight_bitrate)

    loss = torch.zeros((), device=x.device)
    stats = {'bce': 0.0, 'bpp': 0.0, 'sr_pairs': []}
    for posq in sr_posq_scales:
        low_posq = float(posq)
        high_posq = low_posq / 2.0
        if high_posq < 1.0:
            raise ValueError(f'SR low posQuantscale {low_posq} requires high scale {high_posq}, below 1.')
        x_low = quantize_sparse_tensor(x, factor=1 / low_posq, quant_mode='round')
        x_high = quantize_sparse_tensor(x, factor=1 / high_posq, quant_mode='round')
        curr_loss, curr_stats = train_sr_pair(model, x_low, x_high, get_bce, weight_distortion)
        loss = loss + curr_loss
        stats['bce'] += curr_stats['bce']
        stats['bpp'] += curr_stats['bpp']
        stats['sr_pairs'].append(f'{low_posq:g}->{high_posq:g}')

    denom = float(len(sr_posq_scales))
    loss = loss / denom
    stats['bce'] /= denom
    stats['bpp'] /= denom
    return loss, stats


def train_offset(model, x, posq_scales, mse_loss):
    out_set_list = model(x, posQuantscaleList=posq_scales)
    loss = 0
    mses = []
    for out_set in out_set_list:
        mse = mse_loss(out_set['out'].F, out_set['ground_truth'].F)
        loss = loss + mse
        mses.append(float(mse.detach().cpu()))
    return loss, {'mse': float(sum(mses)), 'mse_list': mses}


def main():
    parser = argparse.ArgumentParser(description='Train Unicorn geometry modules directly from KITTI .bin files.')
    parser.add_argument('--unicorn_root', default='/public/DATA/sm/Unicorn')
    parser.add_argument('--module', choices=['lossless', 'sr', 'offset'], required=True)
    parser.add_argument('--velodyne', required=True)
    parser.add_argument('--split_file', required=True)
    parser.add_argument('--model_save_folder', required=True)
    parser.add_argument('--train_posq', type=float, default=64.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=0)
    parser.add_argument('--train_num', type=int, default=1000000)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--lr_min', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1, help='Kept for CLI symmetry; direct loop uses one frame per step.')
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--offset_channels', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--block_layers', type=int, default=3)
    parser.add_argument('--block_type', default='conv')
    parser.add_argument('--stage', type=int, default=None, help='Defaults to 8 for lossless and 1 otherwise.')
    parser.add_argument('--scale', type=int, default=None, help='Defaults to 5 for lossless and 1 otherwise.')
    parser.add_argument('--posQuantscaleList', type=float, nargs='+', default=[2, 4, 8, 16, 32])
    parser.add_argument(
        '--sr_posQuantscaleList',
        type=float,
        nargs='+',
        default=None,
        help='If set, train SR as posQuantscale -> posQuantscale/2 pairs instead of directly on train_posq.',
    )
    parser.add_argument('--weight_distortion', type=float, default=1.0)
    parser.add_argument('--weight_bitrate', type=float, default=1.0)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--checkpoint_every', type=int, default=500)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=11)
    args = parser.parse_args()
    if args.stage is None:
        args.stage = 8 if args.module == 'lossless' else 1
    if args.scale is None:
        args.scale = 5 if args.module == 'lossless' else 1

    configure_unicorn_args()
    unicorn_root = add_unicorn_to_path(args.unicorn_root)
    add_geometry_module_to_path(unicorn_root, args.module)
    from basic_models.loss import get_bce, get_bits
    from data_utils.geometry.quantize import quantize_sparse_tensor
    if args.module == 'lossless':
        from lossless_geometry.model import PCCModel as LosslessPCCModel
    else:
        from lossy_geometry.model import PCCModel as LossyPCCModel
        from lossy_geometry.model_offset import OffsetModel

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    if args.module == 'lossless':
        model = LosslessPCCModel(
            channels=args.channels,
            kernel_size=args.kernel_size,
            block_layers=args.block_layers,
            stage=args.stage,
            scale=args.scale,
            block_type=args.block_type,
        ).to(device)
    elif args.module == 'sr':
        model = LossyPCCModel(
            channels=args.channels,
            kernel_size=args.kernel_size,
            block_layers=args.block_layers,
            stage=args.stage,
            scale=args.scale,
            enc_type='pooling',
            block_type=args.block_type,
        ).to(device)
    else:
        model = OffsetModel(
            channels=args.offset_channels,
            kernel_size=args.kernel_size,
            block_layers=args.block_layers,
            posQuantscaleList=args.posQuantscaleList,
        ).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    mse_loss = torch.nn.MSELoss().to(device)

    files = collect_files(args.velodyne, args.split_file)
    if not files:
        raise ValueError(f'No training frames found in split file: {args.split_file}')
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
    save_dir = Path(args.model_save_folder)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(
        f'module={args.module} files={len(files)} train_posq={args.train_posq} '
        f'epochs={args.epochs} max_steps={args.max_steps} '
        f'stage={args.stage} scale={args.scale} '
        f'sr_posq_scales={args.sr_posQuantscaleList}',
        flush=True,
    )

    global_step = 0
    current_epoch = -1
    recent = []
    for epoch in range(args.epochs):
        current_epoch = epoch
        epoch_files = random.sample(files, min(len(files), args.train_num))
        random.shuffle(epoch_files)
        for path in epoch_files:
            x = load_sparse_from_bin(path, args.train_posq, device)
            optimizer.zero_grad()
            if args.module == 'lossless':
                loss, stats = train_sr_one(
                    model, x, get_bce, get_bits,
                    args.weight_distortion, args.weight_bitrate)
            elif args.module == 'sr':
                loss, stats = train_sr(
                    model, x, get_bce, get_bits, quantize_sparse_tensor,
                    args.weight_distortion, args.weight_bitrate, args.sr_posQuantscaleList)
            else:
                loss, stats = train_offset(model, x, args.posQuantscaleList, mse_loss)
            loss.backward()
            optimizer.step()
            global_step += 1
            recent.append(float(loss.detach().cpu()))

            if global_step == 1 or (args.log_every > 0 and global_step % args.log_every == 0):
                print(
                    f'{time.strftime("%F %T")} epoch={epoch + 1} step={global_step} '
                    f'loss={np.mean(recent):.6f} stats={stats}',
                    flush=True,
                )
                recent = []

            if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                save_checkpoint(
                    save_dir / 'epoch_last.pth', model,
                    args, current_epoch, global_step)

            torch.cuda.empty_cache()
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        save_checkpoint(
            save_dir / 'epoch_last.pth', model,
            args, current_epoch, global_step)
        if epoch > 0 and epoch % 2 == 0:
            lr = max(optimizer.param_groups[0]['lr'] / 2.0, args.lr_min)
            for group in optimizer.param_groups:
                group['lr'] = lr
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    save_checkpoint(
        save_dir / 'epoch_last.pth', model,
        args, current_epoch, global_step)
    print(f'Checkpoint: {save_dir / "epoch_last.pth"}', flush=True)


if __name__ == '__main__':
    main()
