#!/usr/bin/env python3
import argparse
import datetime
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torchsparse import SparseTensor
from torchsparse.nn import functional as F
from torchsparse.utils.collate import sparse_collate_fn


DEFAULT_RENO_ROOT = Path('/public/DATA/sm/RENO')


def add_reno_to_path(reno_root):
    reno_root = Path(reno_root).resolve()
    if str(reno_root) not in sys.path:
        sys.path.insert(0, str(reno_root))
    return reno_root


def configure_torchsparse():
    conv_config = F.conv_config.get_default_conv_config()
    conv_config.kmap_mode = 'hashmap'
    F.conv_config.set_global_conv_config(conv_config)


def read_valid_samples(path):
    if not path:
        return None
    with open(path) as f:
        return {line.strip().zfill(6) for line in f if line.strip()}


def collect_files(training_data, valid_samples=None):
    files = [Path(path) for path in sorted(glob.glob(str(training_data), recursive=True))]
    if valid_samples is None:
        return [str(path) for path in files]
    selected = []
    for path in files:
        if path.stem.zfill(6) in valid_samples:
            selected.append(str(path))
    return selected


class GpccStyleKittiDataset:
    def __init__(self, files, train_posq=4.0):
        self.files = list(files)
        self.train_posq = float(train_posq)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        points = np.fromfile(self.files[idx], dtype=np.float32).reshape(-1, 4)[:, :3]
        coords_mm = np.round(points.astype(np.float64) * 1000.0).astype(np.int64)
        offset = coords_mm.min(axis=0)
        coords_scaled = coords_mm - offset
        coords = np.round(coords_scaled.astype(np.float64) / self.train_posq).astype(np.int32)
        coords = torch.from_numpy(coords).int()
        feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
        return {'input': SparseTensor(coords=coords, feats=feats)}


def main():
    parser = argparse.ArgumentParser(description='Train RENO on KITTI from RACO-LPCC.')
    parser.add_argument('--reno_root', default=str(DEFAULT_RENO_ROOT))
    parser.add_argument('--training_data', required=True, help='Glob pattern, e.g. OpenPCDet/data/kitti_fov/training/velodyne/*.bin')
    parser.add_argument('--model_save_folder', required=True)
    parser.add_argument('--valid_samples', default='', help='Optional split file. Use KITTI train.txt for training.')
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--train_posq', type=float, default=4.0,
                        help='Training lattice step in millimeters after per-frame GPCC-style offset.')
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lr_decay_steps', default='100000,150000')
    parser.add_argument('--max_steps', type=int, default=170000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=11)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    configure_torchsparse()
    add_reno_to_path(args.reno_root)

    from network import Network

    model_dir = Path(args.model_save_folder)
    model_dir.mkdir(parents=True, exist_ok=True)

    valid_samples = read_valid_samples(args.valid_samples)
    files = collect_files(args.training_data, valid_samples)
    if not files:
        raise FileNotFoundError('No training files found.')
    np.random.shuffle(files)

    loader = torch.utils.data.DataLoader(
        dataset=GpccStyleKittiDataset(files, train_posq=args.train_posq),
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=sparse_collate_fn,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    net = Network(channels=args.channels, kernel_size=args.kernel_size).to(device).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    lr_decay_steps = {int(x) for x in str(args.lr_decay_steps).replace(',', ' ').split() if x.strip()}

    losses = []
    global_step = 0
    for epoch in range(1, 9999):
        print(datetime.datetime.now())
        for data in loader:
            x = data['input'].to(device=device)
            optimizer.zero_grad()
            loss = net(x)
            loss.backward()
            optimizer.step()
            global_step += 1
            losses.append(float(loss.item()))

            if global_step % 500 == 0:
                print(f'Epoch:{epoch} | Step:{global_step} | Loss:{round(float(np.mean(losses)), 5)}')
                losses = []
                torch.save(net.state_dict(), model_dir / 'ckpt.pt')

            if global_step in lr_decay_steps:
                args.learning_rate *= args.lr_decay
                for group in optimizer.param_groups:
                    group['lr'] = args.learning_rate
                print(f'Learning rate decay triggered at step {global_step}, LR is setting to {args.learning_rate}.')

            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    torch.save(net.state_dict(), model_dir / 'ckpt.pt')
    print(f'Checkpoint: {model_dir / "ckpt.pt"}')


if __name__ == '__main__':
    main()
