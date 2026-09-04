#!/usr/bin/env python3
"""Train TinyPoint six-absolute-loss + monotonic-BPP router on point objects.

The routing heads and analytical RD decision are unchanged from the current
router.  Only the sparse backbone is replaced by TinyPoint's dense point MLP.
"""

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

from tiny_point_absolute_loss_monotonic_rate_proxy import (
    NUM_LEVELS,
    TinyPointAbsoluteLossMonotonicRateProxy,
    count_parameters,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_bpp(path: Path, indices: np.ndarray) -> np.ndarray:
    table = {
        (int(row["sample_index"]), int(row["level"])): float(row["bpp"])
        for row in csv.DictReader(path.open(newline=""))
    }
    result = np.asarray(
        [[table[(int(index), level)] for level in range(NUM_LEVELS)] for index in indices],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"Invalid BPP values in {path}")
    return result


class RoutingObjects(Dataset):
    def __init__(self, points_path: Path, quant_path: Path, bpp_csv: Path,
                 indices: np.ndarray) -> None:
        self.points = np.load(points_path, mmap_mode="r")
        quant = np.load(quant_path)
        quant_indices = quant["indices"].astype(np.int64)
        row_by_index = {int(index): row for row, index in enumerate(quant_indices)}
        self.indices = np.asarray(indices, dtype=np.int64)
        rows = np.asarray([row_by_index[int(index)] for index in self.indices])
        # Six absolute task losses.  Deliberately do not use legacy loss_deltas.
        self.loss = quant["losses"][rows].astype(np.float32)
        self.predictions = quant["predictions"][rows].astype(np.int16)
        labels = quant["labels"][rows].astype(np.int16)
        self.correct = self.predictions == labels[:, None]
        self.bpp = load_bpp(Path(bpp_csv), self.indices)
        self.qsteps = quant["qsteps"].astype(np.float32)
        if self.loss.shape[1] != NUM_LEVELS or self.bpp.shape[1] != NUM_LEVELS:
            raise ValueError("Expected exactly six aligned quantization levels")
        if not np.all(np.isfinite(self.loss)) or np.any(self.loss < 0):
            raise ValueError("Absolute loss labels must be finite and nonnegative")
        if np.any(np.diff(self.bpp, axis=1).mean(axis=0) < 0):
            raise ValueError("BPP levels are not ordered coarse to fine")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        xyz = np.ascontiguousarray(
            np.asarray(self.points[index])[:, :3], dtype=np.float32
        ).copy()
        return {
            "index": index,
            "points": torch.from_numpy(xyz),
            "loss": torch.from_numpy(self.loss[item]),
            "bpp": torch.from_numpy(self.bpp[item]),
            "correct": torch.from_numpy(self.correct[item]),
        }


def collate(batch):
    point_counts = {item["points"].shape[0] for item in batch}
    if len(point_counts) != 1:
        raise ValueError("TinyPoint training expects fixed-size point sets")
    return {
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.int64),
        "points": torch.stack([item["points"] for item in batch]),
        "loss": torch.stack([item["loss"] for item in batch]),
        "bpp": torch.stack([item["bpp"] for item in batch]),
        "correct": torch.stack([item["correct"] for item in batch]),
        "batch_size": len(batch),
    }


def make_loader(dataset: Dataset, batch_size: int, workers: int, training: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
        persistent_workers=workers > 0,
    )


def choose_levels(loss: torch.Tensor, bpp: torch.Tensor, lambdas: torch.Tensor):
    # Equivalent to argmin(loss + lambda * bpp), with a per-sample constant removed.
    savings = bpp[:, -1:] - bpp
    scores = loss[:, None, :] - lambdas[None, :, None] * savings[:, None, :]
    return scores.argmin(dim=-1), scores


def curve_auc(bpp, accuracy) -> float:
    order = np.argsort(bpp)
    x = np.log(np.maximum(np.asarray(bpp, dtype=np.float64)[order], 1e-9))
    y = np.asarray(accuracy, dtype=np.float64)[order]
    if x[-1] - x[0] < 1e-9:
        return float(y.mean())
    return float(np.trapz(y, x) / (x[-1] - x[0]))


@torch.no_grad()
def load_legacy_five_delta_heads(model: nn.Module, checkpoint_path: Path, dataset_name: str):
    """Reuse safe parts of an old five-delta-loss classification router.

    The legacy loss head order is levels [4,3,2,1,0].  Hidden linear layers
    are remapped to direct level order [0..5].  Final loss linears are not
    loaded because their old target was a signed delta rather than an absolute
    positive loss.  The sixth hidden head starts from the adjacent level-4
    hidden head.  The complete six-output rate MLP is shape-compatible.
    """
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    source = checkpoint.get("model", checkpoint)
    source = {(key[7:] if key.startswith("module.") else key): value for key, value in source.items()}
    target = model.state_dict()
    loaded = {}

    for suffix in ("0.weight", "0.bias", "3.weight", "3.bias"):
        source_key = f"rate_head.{suffix}"
        if source_key in source and source_key in target and source[source_key].shape == target[source_key].shape:
            loaded[source_key] = source[source_key]

    legacy_head_for_level = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 0}
    remapped = []
    for level, old_head in legacy_head_for_level.items():
        for suffix in ("0.weight", "0.bias"):
            source_key = f"base.cost_heads.{old_head}.{suffix}"
            target_key = f"base.cost_heads.{level}.{suffix}"
            if source_key in source and source[source_key].shape == target[target_key].shape:
                loaded[target_key] = source[source_key].clone()
                remapped.append({"source": source_key, "target": target_key})

    current = model.state_dict()
    current.update(loaded)
    model.load_state_dict(current)
    parameter_keys = set(dict(model.named_parameters()))
    return {
        "source": str(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
        "mode": f"legacy_{dataset_name}_five_delta_heads_to_TinyPoint_six_absolute_heads",
        "loaded_tensor_count": len(loaded),
        "loaded_parameter_count": int(sum(target[k].numel() for k in loaded if k in parameter_keys)),
        "rate_head_loaded_completely": all(f"rate_head.{s}" in loaded for s in ("0.weight", "0.bias", "3.weight", "3.bias")),
        "loss_hidden_remapping": remapped,
        "absolute_loss_final_linears_reinitialized": True,
        "new_TinyPoint_backbone_randomly_initialized": True,
    }


@torch.no_grad()
def load_full_tiny_point_checkpoint(model: nn.Module, checkpoint_path: Path):
    """Load all learned TinyPoint tensors while preserving new train-set scales."""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    source = checkpoint.get("model", checkpoint)
    source = {(key[7:] if key.startswith("module.") else key): value for key, value in source.items()}
    loss_scales = model.loss_scales.detach().clone()
    mean_log_increments = model.mean_log_increments.detach().clone()
    model.load_state_dict(source, strict=True)
    model.loss_scales.copy_(loss_scales)
    model.mean_log_increments.copy_(mean_log_increments)
    return {
        "checkpoint": str(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
        "loaded_full_model": True,
        "loaded_tensor_count": len(source),
        "loaded_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "current_full_training_output_scaling_preserved": True,
        "new_backbone_randomly_initialized": False,
    }


def run_epoch(model, loader, optimizer, device, lambdas, loss_scales,
              loss_weight: float, rate_weight: float, clip_grad_norm: float):
    training = optimizer is not None
    model.train(training)
    count = 0
    total_sum = loss_reg_sum = rate_reg_sum = 0.0
    loss_abs_sum = rate_abs_sum = regret_sum = 0.0
    correct_levels = np.zeros(NUM_LEVELS, dtype=np.float64)
    chosen_correct = np.zeros(NUM_LEVELS, dtype=np.float64)
    chosen_bpp = np.zeros(NUM_LEVELS, dtype=np.float64)
    selection_counts = np.zeros((NUM_LEVELS, NUM_LEVELS), dtype=np.int64)
    monotonic_violations = 0
    scales = loss_scales[None, :]
    first_batch = None

    for batch_index, batch in enumerate(loader):
        points = batch["points"].to(device, non_blocking=True)
        loss_target = batch["loss"].to(device, non_blocking=True)
        bpp_target = batch["bpp"].to(device, non_blocking=True)
        correct = batch["correct"].to(device, non_blocking=True)
        n = int(batch["batch_size"])
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(points)
            loss_reg = F.smooth_l1_loss(output["loss_pred"] / scales, loss_target / scales)
            rate_reg = F.smooth_l1_loss(output["rate_log_pred"], torch.log1p(bpp_target))
            total = loss_weight * loss_reg + rate_weight * rate_reg
            if training:
                total.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError("Non-finite gradient norm")
                optimizer.step()

        with torch.no_grad():
            predicted_levels, _ = choose_levels(output["loss_pred"], output["bpp_pred"], lambdas)
            oracle_levels, true_scores = choose_levels(loss_target, bpp_target, lambdas)
            selected_true_scores = torch.gather(true_scores, 2, predicted_levels[:, :, None]).squeeze(-1)
            optimal_scores = true_scores.min(dim=-1).values
            selected_true_bpp = torch.gather(bpp_target, 1, predicted_levels)
            selected_correct = torch.gather(correct, 1, predicted_levels)
            count += n
            total_sum += float(total) * n
            loss_reg_sum += float(loss_reg) * n
            rate_reg_sum += float(rate_reg) * n
            loss_abs_sum += float(torch.abs(output["loss_pred"] - loss_target).mean()) * n
            rate_abs_sum += float(torch.abs(output["bpp_pred"] - bpp_target).mean()) * n
            regret_sum += float((selected_true_scores - optimal_scores).mean()) * n
            monotonic_violations += int((torch.diff(output["bpp_pred"], dim=1) < 0).sum())
            correct_levels += (predicted_levels == oracle_levels).sum(dim=0).cpu().numpy()
            chosen_correct += selected_correct.sum(dim=0).cpu().numpy()
            chosen_bpp += selected_true_bpp.sum(dim=0).cpu().numpy()
            for rate_id in range(NUM_LEVELS):
                selection_counts[rate_id] += np.bincount(
                    predicted_levels[:, rate_id].cpu().numpy(), minlength=NUM_LEVELS
                )
            if batch_index == 0:
                first_batch = {
                    "batch_size": n,
                    "points_shape": list(points.shape),
                    "loss_shape": list(output["loss_pred"].shape),
                    "bpp_shape": list(output["bpp_pred"].shape),
                    "loss_finite": bool(torch.isfinite(output["loss_pred"]).all()),
                    "bpp_finite": bool(torch.isfinite(output["bpp_pred"]).all()),
                    "bpp_monotonic": bool(torch.all(torch.diff(output["bpp_pred"], dim=1) >= 0)),
                }

    accuracy = chosen_correct / max(count, 1)
    mean_bpp = chosen_bpp / max(count, 1)
    return {
        "samples": count,
        "total_loss": total_sum / count,
        "loss_regression": loss_reg_sum / count,
        "rate_regression": rate_reg_sum / count,
        "loss_mae": loss_abs_sum / count,
        "bpp_mae": rate_abs_sum / count,
        "rd_regret": regret_sum / count,
        "bpp_monotonic_violation_rate": monotonic_violations / max(count * (NUM_LEVELS - 1), 1),
        "selection_accuracy": (correct_levels / count).tolist(),
        "mean_selection_accuracy": float(correct_levels.mean() / count),
        "selection_counts": selection_counts.tolist(),
        "curve_accuracy": accuracy.tolist(),
        "curve_bpp": mean_bpp.tolist(),
        "accuracy_bpp_auc": curve_auc(mean_bpp, accuracy),
        "first_batch": first_batch,
    }


def append_metrics(path: Path, epoch: int, split: str, metrics) -> None:
    row = {"epoch": epoch, "split": split, **{k: v for k, v in metrics.items() if k != "first_batch"}}
    row = {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
    fields = list(row)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics,
                    args, lambdas, loss_scales, mean_log_bpp, qsteps, init_report) -> None:
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": metrics,
        "args": vars(args),
        "lambdas": lambdas.cpu().tolist(),
        "loss_scales": loss_scales.cpu().tolist(),
        "mean_log_bpp": mean_log_bpp.tolist(),
        "qsteps": qsteps.tolist(),
        "initialization": init_report,
        "model_type": "TinyPoint backbone + six independent absolute-loss heads + monotonic six-BPP head",
        "selection_metric": (
            "minimum full-training regression total loss"
            if args.selection_mode == "train_loss"
            else "validation Accuracy-BPP AUC"
        ),
    }, path)


def export_test(model, loader, device, lambdas, output: Path) -> None:
    indices, levels, loss_pred, bpp_pred, true_loss, true_bpp = [], [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            points = batch["points"].to(device, non_blocking=True)
            result = model(points)
            selected, _ = choose_levels(result["loss_pred"], result["bpp_pred"], lambdas)
            indices.append(batch["index"].numpy())
            levels.append(selected.cpu().numpy())
            loss_pred.append(result["loss_pred"].cpu().numpy())
            bpp_pred.append(result["bpp_pred"].cpu().numpy())
            true_loss.append(batch["loss"].numpy())
            true_bpp.append(batch["bpp"].numpy())
    np.savez_compressed(
        output,
        indices=np.concatenate(indices),
        selected_levels=np.concatenate(levels),
        predicted_loss=np.concatenate(loss_pred),
        predicted_bpp=np.concatenate(bpp_pred),
        true_loss=np.concatenate(true_loss),
        true_bpp=np.concatenate(true_bpp),
        lambdas=lambdas.cpu().numpy(),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-points", type=Path, required=True)
    parser.add_argument("--test-points", type=Path, required=True)
    parser.add_argument("--train-quant", type=Path, required=True)
    parser.add_argument("--test-quant", type=Path, required=True)
    parser.add_argument("--train-bpp", type=Path, required=True)
    parser.add_argument("--test-bpp", type=Path, required=True)
    parser.add_argument("--train-indices", type=Path, required=True)
    parser.add_argument("--val-indices", type=Path, required=True)
    parser.add_argument("--test-indices", type=Path)
    parser.add_argument("--lambda-json", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--init-kind",
        choices=["legacy_heads", "tiny_point_full"],
        default="legacy_heads",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="ModelNet40")
    parser.add_argument("--task-model", default="DGCNN")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1e-3)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--loss-weight", type=float, default=2.0)
    parser.add_argument("--rate-weight", type=float, default=1.0)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument(
        "--selection-mode",
        choices=["val_auc", "train_loss"],
        default="val_auc",
    )
    parser.add_argument("--merge-val-into-train", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    train_indices = np.load(args.train_indices)
    val_indices = np.load(args.val_indices)
    test_indices = (
        np.load(args.test_indices)
        if args.test_indices is not None
        else np.arange(len(np.load(args.test_points, mmap_mode="r")))
    )
    if len(set(train_indices.tolist()) & set(val_indices.tolist())):
        raise ValueError("Router train and validation indices overlap")
    original_train_samples = int(len(train_indices))
    original_val_samples = int(len(val_indices))
    if args.merge_val_into_train:
        train_indices = np.unique(np.concatenate([train_indices, val_indices])).astype(np.int64)
        if len(train_indices) != original_train_samples + original_val_samples:
            raise ValueError("Merged router training indices are not a disjoint union")
        val_indices = np.empty(0, dtype=np.int64)
    if args.selection_mode == "train_loss" and not args.merge_val_into_train:
        raise ValueError("train_loss selection requires --merge-val-into-train")

    train_set = RoutingObjects(args.train_points, args.train_quant, args.train_bpp, train_indices)
    val_set = (
        None
        if args.selection_mode == "train_loss"
        else RoutingObjects(args.train_points, args.train_quant, args.train_bpp, val_indices)
    )
    test_set = RoutingObjects(args.test_points, args.test_quant, args.test_bpp, test_indices)
    workers = 0 if args.smoke_only else args.workers
    batch_size = min(8, args.batch_size) if args.smoke_only else args.batch_size
    train_loader = make_loader(train_set, batch_size, workers, True)
    val_loader = None if val_set is None else make_loader(val_set, batch_size, workers, False)
    test_loader = make_loader(test_set, batch_size, workers, False)

    loss_scales_np = np.maximum(np.median(train_set.loss, axis=0), np.float32(1e-3)).astype(np.float32)
    mean_log_bpp = np.mean(np.log1p(train_set.bpp), axis=0).astype(np.float32)
    loss_scales = torch.tensor(loss_scales_np, dtype=torch.float32, device=device)
    model = TinyPointAbsoluteLossMonotonicRateProxy(
        feat_dim=256,
        loss_scales=loss_scales_np,
        mean_log_bpp=mean_log_bpp,
        input_channels=3,
    ).to(device)
    init_report = (
        load_full_tiny_point_checkpoint(model, args.init_checkpoint)
        if args.init_kind == "tiny_point_full"
        else load_legacy_five_delta_heads(model, args.init_checkpoint, args.dataset_name)
    )

    lambda_data = json.loads(args.lambda_json.read_text())
    lambdas = torch.tensor(lambda_data["lambdas_high_rate_to_low_rate"], dtype=torch.float32, device=device)
    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        (heads if "cost_heads" in name or name.startswith("rate_head.") else backbone).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr},
        {"params": heads, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_info = {
        **vars(args),
        "train_samples": len(train_set),
        "val_samples": 0 if val_set is None else len(val_set),
        "test_samples": len(test_set),
        "original_router_train_samples": original_train_samples,
        "merged_router_validation_samples": original_val_samples if args.merge_val_into_train else 0,
        "all_official_training_samples_used": bool(args.merge_val_into_train),
        "qsteps_coarse_to_fine": train_set.qsteps.tolist(),
        "loss_target": f"quantization-level absolute {args.task_model} cross entropy (NPZ losses)",
        "legacy_loss_deltas_used": False,
        "loss_scales_train_median": loss_scales_np.tolist(),
        "mean_log_bpp_train_only": mean_log_bpp.tolist(),
        "parameters": count_parameters(model),
        "initialization": init_report,
        "checkpoint_selection": (
            "minimum full-training regression total loss"
            if args.selection_mode == "train_loss"
            else "validation Accuracy-BPP AUC"
        ),
        "test_used_for_selection": False,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    run_info = {k: str(v) if isinstance(v, Path) else v for k, v in run_info.items()}
    (output / "args.json").write_text(json.dumps(run_info, indent=2))
    (output / "initialization_report.json").write_text(json.dumps(init_report, indent=2))
    print(json.dumps(run_info, indent=2), flush=True)

    if args.smoke_only:
        metrics = run_epoch(
            model, [next(iter(train_loader))], optimizer, device, lambdas, loss_scales,
            args.loss_weight, args.rate_weight, args.clip_grad_norm,
        )
        smoke = {
            "status": "passed",
            "forward_backward": True,
            "six_absolute_loss_target": True,
            "legacy_loss_deltas_used": False,
            "metrics": metrics,
            "parameters": count_parameters(model),
            "initialization": init_report,
        }
        if metrics["bpp_monotonic_violation_rate"] != 0:
            raise AssertionError("TinyPoint BPP output violated monotonicity")
        (output / "SMOKE_TEST.json").write_text(json.dumps(smoke, indent=2))
        print(json.dumps(smoke, indent=2), flush=True)
        return

    metrics_path = output / "metrics.csv"
    started = time.time()
    best_score = math.inf if args.selection_mode == "train_loss" else -math.inf
    best_epoch, stale = 0, 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, lambdas, loss_scales,
            args.loss_weight, args.rate_weight, args.clip_grad_norm,
        )
        val_metrics = (
            None
            if val_loader is None
            else run_epoch(
                model, val_loader, None, device, lambdas, loss_scales,
                args.loss_weight, args.rate_weight, args.clip_grad_norm,
            )
        )
        scheduler.step()
        append_metrics(metrics_path, epoch, "train", train_metrics)
        if val_metrics is not None:
            append_metrics(metrics_path, epoch, "val", val_metrics)
        checkpoint_metrics = train_metrics if args.selection_mode == "train_loss" else val_metrics
        save_checkpoint(
            output / "latest.pth", model, optimizer, scheduler, epoch, checkpoint_metrics,
            args, lambdas, loss_scales, mean_log_bpp, train_set.qsteps, init_report,
        )
        score = (
            train_metrics["total_loss"]
            if args.selection_mode == "train_loss"
            else val_metrics["accuracy_bpp_auc"]
        )
        improved = (
            score < best_score - 1e-8
            if args.selection_mode == "train_loss"
            else score > best_score + 1e-8
        )
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            save_checkpoint(
                output / "best.pth", model, optimizer, scheduler, epoch, checkpoint_metrics,
                args, lambdas, loss_scales, mean_log_bpp, train_set.qsteps, init_report,
            )
        else:
            stale += 1
        if val_metrics is None:
            print(
                f"epoch={epoch:03d} seconds={time.time()-epoch_started:.2f} "
                f"train_total={score:.6f} train_loss_MAE={train_metrics['loss_mae']:.6f} "
                f"train_bpp_MAE={train_metrics['bpp_mae']:.6f} "
                f"RD_regret={train_metrics['rd_regret']:.6f} "
                f"monotonic_violation={train_metrics['bpp_monotonic_violation_rate']:.1f}",
                flush=True,
            )
        else:
            print(
                f"epoch={epoch:03d} seconds={time.time()-epoch_started:.2f} "
                f"train_total={train_metrics['total_loss']:.6f} val_total={val_metrics['total_loss']:.6f} "
                f"val_auc={score:.6f} val_loss_MAE={val_metrics['loss_mae']:.6f} "
                f"val_bpp_MAE={val_metrics['bpp_mae']:.6f} RD_regret={val_metrics['rd_regret']:.6f} "
                f"monotonic_violation={val_metrics['bpp_monotonic_violation_rate']:.1f}",
                flush=True,
            )
        if stale >= args.patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    best = torch.load(output / "best.pth", map_location=device)
    model.load_state_dict(best["model"])
    test_metrics = run_epoch(
        model, test_loader, None, device, lambdas, loss_scales,
        args.loss_weight, args.rate_weight, args.clip_grad_norm,
    )
    export_test(model, test_loader, device, lambdas, output / "test_router_predictions.npz")
    summary = {
        "status": "complete",
        "best_epoch": best_epoch,
        "selection_metric": (
            "minimum full-training regression total loss"
            if args.selection_mode == "train_loss"
            else "validation Accuracy-BPP AUC"
        ),
        "best_selection_score": best_score,
        "best_training_total_loss": best_score if args.selection_mode == "train_loss" else None,
        "best_validation_accuracy_bpp_auc": best_score if args.selection_mode == "val_auc" else None,
        "best_selection_metrics": best["metrics"],
        "test_metrics": test_metrics,
        "elapsed_seconds": time.time() - started,
        "model_type": "TinyPoint backbone + six independent absolute-loss heads + monotonic six-BPP head",
        "dataset": args.dataset_name,
        "task_model": args.task_model,
        "optimization_targets": f"six absolute {args.task_model} losses + true G-PCC BPP only",
        "test_used_for_checkpoint_selection": False,
        "all_official_training_samples_used": bool(args.merge_val_into_train),
        "training_samples": len(train_set),
        "validation_samples": 0 if val_set is None else len(val_set),
        "parameters": count_parameters(model),
    }
    (output / "TRAINING_COMPLETE.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
