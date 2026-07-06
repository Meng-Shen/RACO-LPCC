import argparse
import csv
import random
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import MinkowskiEngine as ME


DEFAULT_SCALES = [
    "1/64",
    "1.5/128",
    "1/128",
    "1.5/256",
    "1/256",
    "1.5/512",
    "1/512",
    "1/2048",
]


def parse_scale(value):
    value = str(value).strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return Fraction(numerator) / Fraction(denominator)
    return Fraction(value)


def format_scale(scale):
    if scale.denominator == 1:
        return str(scale.numerator)
    return f"{scale.numerator}/{scale.denominator}"


def read_split(split_file):
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def read_kitti_xyz(bin_path):
    points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def unique_rows(array):
    if len(array) == 0:
        return array.reshape(0, array.shape[-1])
    return np.unique(array, axis=0)


def row_membership(rows, keys):
    if len(rows) == 0:
        return np.zeros((0,), dtype=bool)
    if len(keys) == 0:
        return np.zeros((len(rows),), dtype=bool)
    rows = np.ascontiguousarray(rows)
    keys = np.ascontiguousarray(keys.astype(rows.dtype, copy=False))
    dtype = np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))
    rows_view = rows.view(dtype).reshape(-1)
    keys_view = keys.view(dtype).reshape(-1)
    return np.isin(rows_view, keys_view)


def build_supervision(coords_mm, scale, coarse_from_fine=True):
    """Build coarse occupied voxels and their 8 child occupancy labels.

    The target resolution is always 2 * scale.  By default the coarse lattice is
    derived from the fine lattice with integer division, so every fine voxel maps
    to exactly one of the 8 children of a coarse voxel.  In direct coarse mode,
    the input lattice is round(coords * scale), and child labels are generated
    only from the 8 fine candidates implied by each coarse voxel.
    """
    fine = np.round(coords_mm.astype(np.float64) * float(scale * 2)).astype(np.int32)
    fine = unique_rows(fine)

    if coarse_from_fine:
        coarse_per_fine = np.floor_divide(fine, 2).astype(np.int32)
        child = np.mod(fine, 2).astype(np.int32)
        coarse, inverse = np.unique(coarse_per_fine, axis=0, return_inverse=True)
        labels = np.zeros((len(coarse), 8), dtype=np.float32)
        child_idx = child[:, 0] * 4 + child[:, 1] * 2 + child[:, 2]
        labels[inverse, child_idx] = 1.0
    else:
        coarse = np.round(coords_mm.astype(np.float64) * float(scale)).astype(np.int32)
        coarse = unique_rows(coarse)
        labels = np.zeros((len(coarse), 8), dtype=np.float32)
        child_offsets = np.array(
            [[i // 4, (i // 2) % 2, i % 2] for i in range(8)], dtype=np.int32
        )
        for child_idx, offset in enumerate(child_offsets):
            labels[:, child_idx] = row_membership(coarse * 2 + offset, fine).astype(np.float32)

    return coarse.astype(np.int32), labels


class KittiSparseSRDataset(Dataset):
    def __init__(
        self,
        velodyne_dir,
        split_file,
        scales,
        scale_sampling="random",
        coarse_from_fine=True,
        max_points=0,
    ):
        self.velodyne_dir = Path(velodyne_dir)
        self.frame_ids = read_split(split_file)
        self.files = [self.velodyne_dir / f"{frame_id}.bin" for frame_id in self.frame_ids]
        self.files = [path for path in self.files if path.exists()]
        if not self.files:
            raise FileNotFoundError(f"No KITTI .bin files found under {self.velodyne_dir}")

        self.scales = scales
        self.scale_sampling = scale_sampling
        self.coarse_from_fine = coarse_from_fine
        self.max_points = int(max_points)

        if scale_sampling not in {"random", "cycle", "all"}:
            raise ValueError("--scale_sampling must be one of: random, cycle, all")

    def __len__(self):
        if self.scale_sampling == "all":
            return len(self.files) * len(self.scales)
        return len(self.files)

    def pick_file_and_scale(self, idx):
        if self.scale_sampling == "all":
            file_idx = idx // len(self.scales)
            scale_idx = idx % len(self.scales)
        elif self.scale_sampling == "cycle":
            file_idx = idx
            scale_idx = idx % len(self.scales)
        else:
            file_idx = idx
            scale_idx = random.randrange(len(self.scales))
        return self.files[file_idx], scale_idx, self.scales[scale_idx]

    def __getitem__(self, idx):
        bin_path, scale_idx, scale = self.pick_file_and_scale(idx)
        xyz = read_kitti_xyz(bin_path)
        coords_mm = np.round(xyz.astype(np.float64) * 1000).astype(np.int32)
        coords_mm -= coords_mm.min(axis=0)

        coords, labels = build_supervision(
            coords_mm,
            scale=scale,
            coarse_from_fine=self.coarse_from_fine,
        )

        if self.max_points > 0 and len(coords) > self.max_points:
            choice = np.random.choice(len(coords), self.max_points, replace=False)
            coords = coords[choice]
            labels = labels[choice]

        feats = np.ones((len(coords), 1), dtype=np.float32)
        return {
            "coords": torch.from_numpy(coords).int(),
            "feats": torch.from_numpy(feats).float(),
            "labels": torch.from_numpy(labels).float(),
            "scale_idx": scale_idx,
            "frame_id": bin_path.stem,
        }


def sparse_sr_collate(batch):
    coords_batch = []
    feats_batch = []
    labels_batch = []
    scale_idx_batch = []

    for batch_idx, item in enumerate(batch):
        coords = item["coords"]
        batch_col = torch.full((coords.shape[0], 1), batch_idx, dtype=torch.int32)
        coords_batch.append(torch.cat([batch_col, coords.int()], dim=1))
        feats_batch.append(item["feats"])
        labels_batch.append(item["labels"])
        scale_idx_batch.append(item["scale_idx"])

    return {
        "coords": torch.cat(coords_batch, dim=0),
        "feats": torch.cat(feats_batch, dim=0),
        "labels": torch.cat(labels_batch, dim=0),
        "scale_idx": torch.tensor(scale_idx_batch, dtype=torch.long),
    }


class SparseResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(channels, channels, kernel_size=3, dimension=3)
        self.bn1 = ME.MinkowskiBatchNorm(channels)
        self.conv2 = ME.MinkowskiConvolution(channels, channels, kernel_size=3, dimension=3)
        self.bn2 = ME.MinkowskiBatchNorm(channels)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class SparseOccupancySRNet(nn.Module):
    def __init__(self, in_channels=1, channels=64, num_blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(channels),
            ME.MinkowskiReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[SparseResidualBlock(channels) for _ in range(num_blocks)])
        self.head = ME.MinkowskiConvolution(channels, 8, kernel_size=1, dimension=3)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


def make_loader(args, split_name, shuffle, scale_sampling):
    dataset = KittiSparseSRDataset(
        velodyne_dir=args.velodyne_dir,
        split_file=getattr(args, f"{split_name}_split"),
        scales=args.scales,
        scale_sampling=scale_sampling,
        coarse_from_fine=not args.direct_coarse_quant,
        max_points=args.max_points,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=sparse_sr_collate,
        drop_last=False,
    )


def move_batch(batch, device):
    return {
        "coords": batch["coords"].to(device),
        "feats": batch["feats"].to(device),
        "labels": batch["labels"].to(device),
        "scale_idx": batch["scale_idx"].to(device),
    }


def train_one_epoch(model, loader, optimizer, device, pos_weight, log_interval):
    model.train()
    total_loss = 0.0
    total_points = 0
    start = time.time()

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        stensor = ME.SparseTensor(
            features=batch["feats"],
            coordinates=batch["coords"],
            device=device,
        )
        logits = model(stensor).F
        labels = batch["labels"]
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        num_points = labels.shape[0]
        total_loss += float(loss.item()) * num_points
        total_points += num_points

        if log_interval > 0 and step % log_interval == 0:
            elapsed = time.time() - start
            print(
                f"train step {step}/{len(loader)} "
                f"loss={total_loss / max(total_points, 1):.6f} "
                f"points={total_points} time={elapsed:.1f}s",
                flush=True,
            )

    return total_loss / max(total_points, 1)


@torch.no_grad()
def evaluate(model, loader, device, threshold):
    model.eval()
    total_loss = 0.0
    total_points = 0
    true_pos = 0
    pred_pos = 0
    hit_pos = 0

    pos_weight = torch.full((8,), 10.0, device=device)
    for batch in loader:
        batch = move_batch(batch, device)
        stensor = ME.SparseTensor(
            features=batch["feats"],
            coordinates=batch["coords"],
            device=device,
        )
        logits = model(stensor).F
        labels = batch["labels"]
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        probs = torch.sigmoid(logits)
        pred = probs >= threshold
        target = labels > 0.5

        total_loss += float(loss.item()) * labels.shape[0]
        total_points += labels.shape[0]
        true_pos += int(target.sum().item())
        pred_pos += int(pred.sum().item())
        hit_pos += int((pred & target).sum().item())

    recall = hit_pos / true_pos if true_pos else 0.0
    precision = hit_pos / pred_pos if pred_pos else 0.0
    return {
        "loss": total_loss / max(total_points, 1),
        "recall": recall,
        "precision": precision,
        "true_pos": true_pos,
        "pred_pos": pred_pos,
        "hit_pos": hit_pos,
    }


def save_checkpoint(path, model, optimizer, epoch, args, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def append_log(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a sparse-conv lossy occupancy super-resolution module on KITTI.")
    parser.add_argument("--velodyne_dir", default="OpenPCDet/data/kitti_fov/training/velodyne")
    parser.add_argument("--train_split", default="OpenPCDet/data/kitti_fov/ImageSets/train.txt")
    parser.add_argument("--val_split", default="OpenPCDet/data/kitti_fov/ImageSets/val.txt")
    parser.add_argument("--scales", default=",".join(DEFAULT_SCALES))
    parser.add_argument("--scale_sampling", default="random", choices=["random", "cycle", "all"])
    parser.add_argument(
        "--val_scale_sampling",
        default="all",
        choices=["random", "cycle", "all"],
        help="Scale sampling for validation. 'all' evaluates every frame at every scale.",
    )
    parser.add_argument(
        "--direct_coarse_quant",
        action="store_true",
        help=(
            "Use round(coords * scale) as input coarse voxels and supervise the "
            "8 children generated from coarse * 2 + offset. Default uses "
            "floor(round(coords * 2scale) / 2)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_points", type=int, default=0, help="Randomly subsample coarse voxels per frame; 0 disables.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--work_dir", default="GPCC/work_dirs/sparse_sr")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    args.scales = [parse_scale(item) for item in args.scales.split(",") if item.strip()]
    args.scale_names = [format_scale(scale) for scale in args.scales]
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"scales: {', '.join(args.scale_names)}")
    print(f"device: {args.device}")
    print(f"work_dir: {work_dir}")

    train_loader = make_loader(args, "train", shuffle=True, scale_sampling=args.scale_sampling)
    val_loader = make_loader(args, "val", shuffle=False, scale_sampling=args.val_scale_sampling)
    print(
        f"train frames: {len(train_loader.dataset.files)}, "
        f"train samples: {len(train_loader.dataset)} "
        f"(scale_sampling={args.scale_sampling})"
    )
    print(
        f"val frames: {len(val_loader.dataset.files)}, "
        f"val samples: {len(val_loader.dataset)} "
        f"(scale_sampling={args.val_scale_sampling})"
    )

    model = SparseOccupancySRNet(channels=args.channels, num_blocks=args.blocks).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    pos_weight = torch.full((8,), args.pos_weight, device=args.device)
    best_f1 = -1.0
    log_csv = work_dir / "train_log.csv"

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            pos_weight,
            args.log_interval,
        )
        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.8f}",
            "val_loss": "",
            "val_recall": "",
            "val_precision": "",
            "val_f1": "",
            "true_pos": "",
            "pred_pos": "",
            "hit_pos": "",
        }

        print(f"epoch {epoch}: train_loss={train_loss:.6f}", flush=True)

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            metrics = evaluate(model, val_loader, args.device, args.threshold)
            recall = metrics["recall"]
            precision = metrics["precision"]
            f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
            row.update(
                {
                    "val_loss": f"{metrics['loss']:.8f}",
                    "val_recall": f"{recall:.8f}",
                    "val_precision": f"{precision:.8f}",
                    "val_f1": f"{f1:.8f}",
                    "true_pos": metrics["true_pos"],
                    "pred_pos": metrics["pred_pos"],
                    "hit_pos": metrics["hit_pos"],
                }
            )
            print(
                f"epoch {epoch}: val_loss={metrics['loss']:.6f} "
                f"recall={recall:.6f} precision={precision:.6f} f1={f1:.6f}",
                flush=True,
            )

            save_checkpoint(work_dir / "latest.pth", model, optimizer, epoch, args, metrics)
            if f1 > best_f1:
                best_f1 = f1
                save_checkpoint(work_dir / "best.pth", model, optimizer, epoch, args, metrics)
        else:
            save_checkpoint(work_dir / "latest.pth", model, optimizer, epoch, args, {})

        append_log(log_csv, row)


if __name__ == "__main__":
    main()
