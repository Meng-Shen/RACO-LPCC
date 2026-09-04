#!/usr/bin/env python3
"""Train the loss+BPP routing proxy on ShapeNet55 classification labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_cost_proxy import SparseCostProxyNet, voxelize_points


LEVEL_ORDER = (4, 3, 2, 1, 0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_bpp(path: Path, indices: np.ndarray, levels: int):
    table = {(int(row["sample_index"]), int(row["level"])): float(row["bpp"])
             for row in csv.DictReader(path.open(newline=""))}
    return np.asarray([[table[(int(index), level)] for level in range(levels)]
                       for index in indices], dtype=np.float32)


class RoutingObjects(Dataset):
    def __init__(self, points_path, quant_path, bpp_csv, indices, target_scale,
                 voxel_size, point_cloud_range, max_voxels):
        self.points = np.load(points_path, mmap_mode="r")
        quant = np.load(quant_path)
        quant_indices = quant["indices"].astype(np.int64)
        row_by_index = {int(index): row for row, index in enumerate(quant_indices)}
        self.indices = np.asarray(indices, dtype=np.int64)
        rows = np.asarray([row_by_index[int(index)] for index in self.indices])
        self.loss = quant["loss_deltas"][rows].astype(np.float32)
        self.predictions = quant["predictions"][rows].astype(np.int16)
        labels = quant["labels"][rows].astype(np.int16)
        self.correct = self.predictions == labels[:, None]
        self.bpp = load_bpp(Path(bpp_csv), self.indices, self.loss.shape[1])
        self.qsteps = quant["qsteps"].astype(np.float32)
        self.target_scale = float(target_scale)
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.pc_range = np.asarray(point_cloud_range, dtype=np.float32)
        self.max_voxels = int(max_voxels)
        grid = np.floor((self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size).astype(np.int32)
        self.spatial_shape = grid[[2, 1, 0]].tolist()
        self.mean_log_bpp = np.log1p(self.bpp).mean(axis=0).astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        xyz = np.asarray(self.points[index], dtype=np.float32)
        points = np.concatenate([xyz, np.zeros((len(xyz), 1), dtype=np.float32)], axis=1)
        features, coords = voxelize_points(
            points, voxel_size=self.voxel_size, pc_range=self.pc_range,
            max_voxels=self.max_voxels, use_abs_xyz=True, include_intensity=False,
        )
        head_target = np.asarray(
            [[self.loss[item, level] * self.target_scale] for level in LEVEL_ORDER],
            dtype=np.float32,
        )
        return {
            "index": index,
            "voxel_features": torch.from_numpy(features),
            "voxel_coords": torch.from_numpy(coords),
            "head_target": torch.from_numpy(head_target),
            "loss": torch.from_numpy(self.loss[item]),
            "bpp": torch.from_numpy(self.bpp[item]),
            "correct": torch.from_numpy(self.correct[item]),
        }


def collate(batch):
    features, coords = [], []
    for batch_index, item in enumerate(batch):
        current = item["voxel_coords"].int()
        column = torch.full((len(current), 1), batch_index, dtype=torch.int32)
        features.append(item["voxel_features"].float())
        coords.append(torch.cat([column, current], dim=1))
    return {
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.int64),
        "voxel_features": torch.cat(features),
        "voxel_coords": torch.cat(coords).int(),
        "head_target": torch.stack([item["head_target"] for item in batch]),
        "loss": torch.stack([item["loss"] for item in batch]),
        "bpp": torch.stack([item["bpp"] for item in batch]),
        "correct": torch.stack([item["correct"] for item in batch]),
        "batch_size": len(batch),
    }


class RateAwareProxy(nn.Module):
    def __init__(self, spatial_shape, feat_dim, mean_log_bpp):
        super().__init__()
        self.base = SparseCostProxyNet(
            input_channels=7, spatial_shape=spatial_shape, feat_dim=feat_dim,
            num_cost_heads=5, num_targets=1, cost_nonnegative=False,
            monotonic_cost=False,
        )
        self.rate_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True), nn.Dropout(0.15),
            nn.Linear(feat_dim, 6),
        )
        mean_log_bpp = torch.as_tensor(mean_log_bpp, dtype=torch.float32)
        increments = torch.diff(torch.cat([mean_log_bpp.new_zeros(1), mean_log_bpp]))
        self.register_buffer("mean_log_increments", increments.clamp_min(1e-4))
        nn.init.zeros_(self.rate_head[-1].weight)
        nn.init.zeros_(self.rate_head[-1].bias)
        self._global_feature = None
        self.base.global_mlp.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self._global_feature = output

    def forward(self, features, coords, batch_size):
        self._global_feature = None
        output = self.base(features, coords, batch_size)
        if self._global_feature is None:
            raise RuntimeError("Sparse backbone global feature was not captured")
        residual = self.rate_head(self._global_feature)
        multiplier = torch.exp(0.9 * torch.tanh(residual))
        rate_log = torch.cumsum(self.mean_log_increments[None, :] * multiplier, dim=1)
        return output["cost_pred"], rate_log, torch.expm1(rate_log).clamp_min(0.0)


def losses_by_level(cost_pred, scale):
    result = cost_pred.new_zeros((cost_pred.shape[0], 6))
    for head, level in enumerate(LEVEL_ORDER):
        result[:, level] = cost_pred[:, head, 0] / scale
    return result


def choose_levels(loss, bpp, lambdas):
    savings = bpp[:, -1:] - bpp
    scores = loss[:, None, :] - lambdas[None, :, None] * savings[:, None, :]
    return scores.argmin(dim=-1), scores


def curve_auc(bpp, accuracy):
    order = np.argsort(bpp)
    x = np.log(np.maximum(np.asarray(bpp)[order], 1e-9))
    y = np.asarray(accuracy)[order]
    if x[-1] - x[0] < 1e-9:
        return float(y.mean())
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def run_epoch(model, loader, optimizer, device, lambdas, scale):
    training = optimizer is not None
    model.train(training)
    count = 0
    total_sum = loss_sum = rate_sum = loss_abs = rate_abs = regret_sum = 0.0
    correct_levels = np.zeros(len(lambdas), dtype=np.float64)
    chosen_correct = np.zeros(len(lambdas), dtype=np.float64)
    chosen_bpp = np.zeros(len(lambdas), dtype=np.float64)
    for batch in loader:
        for key in ("head_target", "loss", "bpp", "correct"):
            batch[key] = batch[key].to(device, non_blocking=True)
        features = batch["voxel_features"].to(device, non_blocking=True)
        coords = batch["voxel_coords"].to(device, non_blocking=True)
        n = int(batch["batch_size"])
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            cost_pred, rate_log, bpp_pred = model(features, coords, n)
            loss_reg = F.smooth_l1_loss(cost_pred, batch["head_target"])
            rate_reg = F.smooth_l1_loss(rate_log, torch.log1p(batch["bpp"]))
            total = loss_reg + rate_reg
            if training:
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        with torch.no_grad():
            predicted_loss = losses_by_level(cost_pred, scale)
            predicted_levels, _ = choose_levels(predicted_loss, bpp_pred, lambdas)
            oracle_levels, true_scores = choose_levels(batch["loss"], batch["bpp"], lambdas)
            selected_true_scores = torch.gather(true_scores, 2, predicted_levels[:, :, None]).squeeze(-1)
            optimal_scores = true_scores.min(dim=-1).values
            chosen_true_bpp = torch.gather(batch["bpp"], 1, predicted_levels)
            selected_correct = torch.gather(batch["correct"], 1, predicted_levels)
            count += n
            total_sum += float(total) * n
            loss_sum += float(loss_reg) * n
            rate_sum += float(rate_reg) * n
            loss_abs += float(torch.abs(predicted_loss - batch["loss"]).mean()) * n
            rate_abs += float(torch.abs(bpp_pred - batch["bpp"]).mean()) * n
            regret_sum += float((selected_true_scores - optimal_scores).mean()) * n
            correct_levels += (predicted_levels == oracle_levels).sum(dim=0).cpu().numpy()
            chosen_correct += selected_correct.sum(dim=0).cpu().numpy()
            chosen_bpp += chosen_true_bpp.sum(dim=0).cpu().numpy()
    accuracy = chosen_correct / max(count, 1)
    mean_bpp = chosen_bpp / max(count, 1)
    return {
        "samples": count,
        "total_loss": total_sum / count,
        "loss_regression": loss_sum / count,
        "rate_regression": rate_sum / count,
        "loss_mae": loss_abs / count,
        "bpp_mae": rate_abs / count,
        "rd_regret": regret_sum / count,
        "selection_accuracy": (correct_levels / count).tolist(),
        "mean_selection_accuracy": float(correct_levels.mean() / count),
        "curve_accuracy": accuracy.tolist(),
        "curve_bpp": mean_bpp.tolist(),
        "accuracy_bpp_auc": curve_auc(mean_bpp, accuracy),
    }


def make_loader(dataset, batch_size, workers, training):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=training, num_workers=workers,
        pin_memory=True, drop_last=training, collate_fn=collate,
        persistent_workers=workers > 0,
    )


def export_test(model, loader, device, lambdas, scale, qsteps, output):
    indices, levels, loss_pred, bpp_pred, true_bpp = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features = batch["voxel_features"].to(device)
            coords = batch["voxel_coords"].to(device)
            cost, _, rates = model(features, coords, int(batch["batch_size"]))
            loss = losses_by_level(cost, scale)
            selected, _ = choose_levels(loss, rates, lambdas)
            indices.append(batch["index"].numpy())
            levels.append(selected.cpu().numpy())
            loss_pred.append(loss.cpu().numpy())
            bpp_pred.append(rates.cpu().numpy())
            true_bpp.append(batch["bpp"].numpy())
    np.savez_compressed(
        output, indices=np.concatenate(indices), selected_levels=np.concatenate(levels),
        predicted_loss=np.concatenate(loss_pred), predicted_bpp=np.concatenate(bpp_pred),
        true_bpp=np.concatenate(true_bpp), lambdas=lambdas.cpu().numpy(), qsteps=qsteps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True)
    parser.add_argument("--quant", required=True)
    parser.add_argument("--bpp", required=True)
    parser.add_argument("--train-indices", required=True)
    parser.add_argument("--val-indices", required=True)
    parser.add_argument("--test-indices", required=True)
    parser.add_argument("--lambda-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--feat-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--max-voxels", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    device = torch.device("cuda:0")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    quant_train = np.load(args.quant)
    p95 = max(float(np.quantile(np.abs(quant_train["loss_deltas"][:, :5]), 0.95)), 1e-4)
    target_scale = min(100.0, max(0.1, 0.5 / p95))
    train_indices = np.load(args.train_indices)
    val_indices = np.load(args.val_indices)
    test_indices = np.load(args.test_indices)
    common = dict(
        target_scale=target_scale,
        voxel_size=[args.voxel_size] * 3,
        point_cloud_range=[-1.1, -1.1, -1.1, 1.1, 1.1, 1.1],
        max_voxels=args.max_voxels,
    )
    train_set = RoutingObjects(args.points, args.quant, args.bpp, train_indices, **common)
    val_set = RoutingObjects(args.points, args.quant, args.bpp, val_indices, **common)
    test_set = RoutingObjects(args.points, args.quant, args.bpp, test_indices, **common)
    train_loader = make_loader(train_set, args.batch_size, args.workers, True)
    val_loader = make_loader(val_set, args.batch_size, args.workers, False)
    test_loader = make_loader(test_set, args.batch_size, args.workers, False)
    lambda_data = json.loads(Path(args.lambda_json).read_text())
    lambdas = torch.tensor(lambda_data["lambdas_high_rate_to_low_rate"], dtype=torch.float32, device=device)
    model = RateAwareProxy(train_set.spatial_shape, args.feat_dim, train_set.mean_log_bpp).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    metrics_path = output / "metrics.csv"
    fields = None
    best_score, best_epoch, stale = -math.inf, 0, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, lambdas, target_scale)
        val_metrics = run_epoch(model, val_loader, None, device, lambdas, target_scale)
        scheduler.step()
        for split, metrics in (("train", train_metrics), ("val", val_metrics)):
            row = {"epoch": epoch, "split": split, **metrics}
            row = {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
            if fields is None:
                fields = list(row)
            with metrics_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
        score = val_metrics["accuracy_bpp_auc"]
        checkpoint = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "metrics": val_metrics,
            "args": vars(args), "target_scale": target_scale,
            "lambdas": lambdas.cpu().tolist(), "qsteps": train_set.qsteps.tolist(),
            "selection_metric": "validation Accuracy-BPP AUC",
        }
        torch.save(checkpoint, output / "latest.pth")
        if score > best_score + 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(checkpoint, output / "best.pth")
        else:
            stale += 1
        print(
            f"epoch={epoch:03d} val_auc={score:.6f} val_loss_mae={val_metrics['loss_mae']:.5f} "
            f"val_bpp_mae={val_metrics['bpp_mae']:.5f} selection={val_metrics['mean_selection_accuracy']:.4f}",
            flush=True,
        )
        if stale >= args.patience:
            break

    best = torch.load(output / "best.pth", map_location=device)
    model.load_state_dict(best["model"])
    export_test(model, test_loader, device, lambdas, target_scale, test_set.qsteps, output / "test_router_predictions.npz")
    summary = {
        "dataset": "ShapeNet55 official split; test held out",
        "best_epoch": best_epoch,
        "best_validation_accuracy_bpp_auc": best_score,
        "best_validation_metrics": best["metrics"],
        "elapsed_seconds": time.time() - started,
        "model_type": "shared sparse XYZ backbone + five CE-loss heads + one six-rate BPP head",
        "optimization_targets": "loss regression + BPP regression only",
        "checkpoint_selection": "validation Accuracy-BPP AUC",
        "test_used_for_checkpoint_selection": False,
        "target_scale": target_scale,
    }
    (output / "TRAINING_COMPLETE.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
