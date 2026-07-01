#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_cost_proxy import (
    CostCalibrator,
    SparseCostProxyNet,
    cost_to_jucp_labels,
    parse_thresholds,
    read_split_file,
    sparse_collate_fn,
    voxelize_points,
)


TARGET_INDEX = {
    "car": 0,
    "ped": 1,
    "pedestrian": 1,
    "cyc": 2,
    "cyclist": 2,
}


def load_train_args(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    return args, ckpt


class RouterInferenceDataset(Dataset):
    def __init__(self, velodyne_dir, split_file, voxel_size, pc_range, max_voxels, use_abs_xyz):
        self.velodyne_dir = Path(velodyne_dir)
        self.frame_ids = read_split_file(split_file)
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.pc_range = np.asarray(pc_range, dtype=np.float32)
        self.max_voxels = int(max_voxels)
        self.use_abs_xyz = bool(use_abs_xyz)
        grid_size = np.floor((self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size).astype(np.int32)
        self.spatial_shape = grid_size[[2, 1, 0]].tolist()
        self.num_point_features = 8 if self.use_abs_xyz else 5

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, index):
        fid = self.frame_ids[index]
        bin_path = self.velodyne_dir / f"{fid}.bin"
        raw = np.fromfile(str(bin_path), dtype=np.float32)
        if raw.size % 4 != 0:
            raise ValueError(f"Invalid KITTI bin file: {bin_path}")
        raw = raw.reshape(-1, 4)
        voxel_features, voxel_coords = voxelize_points(
            raw,
            voxel_size=self.voxel_size,
            pc_range=self.pc_range,
            max_voxels=self.max_voxels,
            use_abs_xyz=self.use_abs_xyz,
        )
        return {
            "frame_id": fid,
            "voxel_features": torch.from_numpy(voxel_features),
            "voxel_coords": torch.from_numpy(voxel_coords),
            "ap_drop": torch.zeros((6, 3), dtype=torch.float32),
        }


def parse_quant_map(text):
    def parse_scale(value):
        value = str(value).strip()
        if "/" in value:
            num, den = value.split("/", 1)
            return float(num) / float(den)
        return float(value)

    combos = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        fg, bg = [x.strip() for x in item.split(",", 1)]
        combos.append((parse_scale(fg), parse_scale(bg)))
    if not combos:
        raise ValueError("--quant_map is empty")
    return combos


def parse_float_list(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def parse_int_list(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def parse_class_weights(text):
    parts = parse_float_list(text)
    if len(parts) != 3:
        raise ValueError("--lagrange_class_weights must contain exactly three values: car,ped,cyc")
    weights = np.asarray(parts, dtype=np.float32)
    if float(np.abs(weights).sum()) <= 0.0:
        raise ValueError("--lagrange_class_weights cannot all be zero")
    return weights


def norm_frame_id(x):
    return str(x).strip().zfill(6)


def load_label_bpp_table(path, mode="mean"):
    """Return bpp lookup data from Split-GPCC per-frame details.

    mode="mean" returns label -> mean bpp, which simulates inference-time rate
    estimates shared by all frames. mode="per_frame" returns
    frame_id -> label -> bpp, which is only useful for oracle-style analysis.
    """
    if not path:
        return None
    per_frame = {}
    by_label = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            frame_id = norm_frame_id(row.get("filename") or row.get("frame_id"))
            label = int(row.get("combo_id") or row["rate_id"])
            if "bpp" in row and str(row["bpp"]).strip() != "":
                bpp = float(row["bpp"])
            else:
                bits = float(row["bits"])
                num_points = float(row["num_points"])
                bpp = bits / num_points if num_points else 0.0
            per_frame.setdefault(frame_id, {})[label] = bpp
            by_label.setdefault(label, []).append(bpp)
    if mode == "per_frame":
        return per_frame
    if mode == "mean":
        return {label: float(np.mean(values)) for label, values in by_label.items()}
    raise ValueError(f"Unsupported bpp estimate mode: {mode}")


def hard_label_for_cost(cost_row, threshold):
    valid = (cost_row <= threshold.view(1, -1)).all(dim=-1)
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    return int(valid_indices[-1].item() + 1) if valid_indices.numel() else 0


def label_bpp(frame_id, label, label_bpp_table, num_labels):
    if label_bpp_table is not None:
        if label in label_bpp_table:
            return float(label_bpp_table[label])
        frame_rates = label_bpp_table.get(norm_frame_id(frame_id))
        if isinstance(frame_rates, dict) and label in frame_rates:
            return float(frame_rates[label])
    # Fallback only preserves ordering: larger labels are treated as lower rate.
    return float(num_labels - label)


def cost_to_jucp_labels_debt(
    cost,
    thresholds,
    frame_ids,
    label_bpp_table=None,
    target_index=0,
    alpha=1.0,
    beta=0.5,
    max_extra=0.0,
    min_threshold_ratio=0.5,
    min_saving_per_cost=0.0,
):
    """Sequential debt-aware label selection.

    Cost and thresholds are expected to use the same scale. The debt is tracked
    independently for each threshold row and only on the selected target class;
    the other classes remain hard-constrained by their original thresholds.
    """
    thresholds = thresholds.to(cost.device, dtype=cost.dtype)
    labels_all = []
    stats_all = []
    B, H, _ = cost.shape
    if len(frame_ids) != B:
        raise ValueError(f"frame_ids has {len(frame_ids)} rows, cost has batch size {B}")

    for k in range(thresholds.shape[0]):
        threshold = thresholds[k]
        base_target_thr = float(threshold[target_index].item())
        min_target_thr = base_target_thr * float(min_threshold_ratio)
        max_extra_for_rate = float(max_extra) if base_target_thr > 0.0 else 0.0
        debt = 0.0
        labels = []
        stats = []

        for i, frame_id in enumerate(frame_ids):
            cost_row = cost[i]
            hard_label = hard_label_for_cost(cost_row, threshold)
            hard_bpp = label_bpp(frame_id, hard_label, label_bpp_table, H)
            effective_target_thr = max(min_target_thr, base_target_thr - float(alpha) * debt)

            best_label = hard_label
            best_score = 0.0
            best_extra = 0.0
            best_saving = 0.0

            for label in range(hard_label + 1, H + 1):
                h = label - 1
                candidate_cost = cost_row[h]
                non_target_ok = True
                for target in range(candidate_cost.numel()):
                    if target == target_index:
                        continue
                    if float(candidate_cost[target].item()) > float(threshold[target].item()):
                        non_target_ok = False
                        break
                if not non_target_ok:
                    continue

                target_cost = float(candidate_cost[target_index].item())
                extra = max(0.0, target_cost - effective_target_thr)
                if extra > max_extra_for_rate:
                    continue

                candidate_bpp = label_bpp(frame_id, label, label_bpp_table, H)
                saving = hard_bpp - candidate_bpp
                if saving < 0.0:
                    continue
                if extra > 0.0 and saving / extra < float(min_saving_per_cost):
                    continue

                score = saving
                if score > best_score or (score == best_score and label > best_label):
                    best_label = label
                    best_score = score
                    best_extra = extra
                    best_saving = saving

            chosen_h = max(0, best_label - 1)
            chosen_target_cost = float(cost_row[chosen_h, target_index].item()) if best_label > 0 else 0.0
            margin = max(0.0, effective_target_thr - chosen_target_cost)
            debt = max(0.0, debt + best_extra - float(beta) * margin)

            labels.append(best_label)
            stats.append({
                "hard_label": hard_label,
                "extra": best_extra,
                "saving": best_saving,
                "debt": debt,
                "effective_threshold": effective_target_thr,
            })

        labels_all.append(labels)
        stats_all.append(stats)

    return np.asarray(labels_all, dtype=np.int64).T, stats_all


def cost_to_jucp_labels_lagrangian(
    cost,
    frame_ids,
    lambdas,
    label_bpp_table=None,
    class_weights=None,
    ap_drop_scale=1.0,
    max_labels=None,
):
    """Choose labels by minimizing weighted AP-drop + lambda * per-frame bpp."""
    B, H, T = cost.shape
    if len(frame_ids) != B:
        raise ValueError(f"frame_ids has {len(frame_ids)} rows, cost has batch size {B}")
    if class_weights is None:
        class_weights = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    class_weights = torch.as_tensor(class_weights, dtype=cost.dtype, device=cost.device).view(1, T)
    labels_all = []
    stats_all = []
    if max_labels is None:
        max_labels = [H] * len(lambdas)
    if len(max_labels) != len(lambdas):
        raise ValueError("--lagrange_max_labels must have the same number of values as --lagrange_lambdas")

    for rate_id, lam in enumerate(lambdas):
        max_label = max(0, min(H, int(max_labels[rate_id])))
        labels = []
        stats = []
        for i, frame_id in enumerate(frame_ids):
            candidates = []
            for label in range(max_label + 1):
                if label == 0:
                    weighted_cost = 0.0
                else:
                    weighted_cost = float((cost[i, label - 1].view(1, T) * class_weights).sum().item())
                    weighted_cost /= float(ap_drop_scale)
                bpp = label_bpp(frame_id, label, label_bpp_table, H)
                score = weighted_cost + float(lam) * bpp
                candidates.append((score, label, weighted_cost, bpp))
            score, label, weighted_cost, bpp = min(candidates, key=lambda item: (item[0], item[3]))
            labels.append(label)
            stats.append({
                "lagrange_score": score,
                "weighted_cost": weighted_cost,
                "label_bpp": bpp,
                "max_label": max_label,
            })
        labels_all.append(labels)
        stats_all.append(stats)

    return np.asarray(labels_all, dtype=np.int64).T, stats_all


def load_calibrator(path, device, num_targets, allow_negative=False):
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    payload = torch.load(path, map_location=device)
    payload_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    allow_negative = bool(payload_args.get("allow_negative_cost", allow_negative))
    calibrator = CostCalibrator(num_targets=num_targets, allow_negative=allow_negative).to(device)
    calibrator.load_state_dict(payload["calibrator"])
    calibrator.eval()
    return calibrator


def main():
    parser = argparse.ArgumentParser(description="Export router proxy JUQP/JUCP labels from predicted AP-drop costs.")
    parser.add_argument("--velodyne_dir", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--ckpt", required=True, help="Router best.pth/latest.pth")
    parser.add_argument("--calibration", default=None, help="Optional calibration.pth")
    parser.add_argument("--thresholds", required=True, help="car,ped,cyc;car,ped,cyc;...")
    parser.add_argument("--quant_map", required=True, help="fg,bg;fg,bg;... label order")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix", default="router")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cost_csv", default=None)
    parser.add_argument("--selection_policy", choices=["hard", "debt", "lagrangian"], default="hard")
    parser.add_argument("--split_details_csv", default=None,
                        help="Split-GPCC per-frame details for label-bpp estimates.")
    parser.add_argument("--bpp_estimate", choices=["mean", "per_frame"], default="mean",
                        help="Use one mean bpp per label, or per-frame bpp for oracle-style analysis.")
    parser.add_argument("--debt_target", default="car", choices=sorted(TARGET_INDEX.keys()))
    parser.add_argument("--debt_alpha", type=float, default=1.0,
                        help="How strongly accumulated debt tightens the target threshold.")
    parser.add_argument("--debt_beta", type=float, default=0.5,
                        help="How much below-threshold margin repays accumulated debt.")
    parser.add_argument("--debt_max_extra", type=float, default=0.0015,
                        help="Maximum per-frame target AP-drop overshoot, before ap_drop_scale.")
    parser.add_argument("--debt_min_threshold_ratio", type=float, default=0.5,
                        help="Lower bound for dynamic target threshold as a ratio of the base threshold.")
    parser.add_argument("--debt_min_saving_per_cost", type=float, default=0.0,
                        help="Minimum bpp saving per extra target AP-drop cost.")
    parser.add_argument("--lagrange_lambdas", default="0,0.0005,0.001,0.002,0.005,0.01,0.02",
                        help="Comma-separated lambda values for cost + lambda*bpp selection.")
    parser.add_argument("--lagrange_class_weights", default="1,0,0",
                        help="Class weights for predicted AP-drop cost: car,ped,cyc.")
    parser.add_argument("--lagrange_max_labels", default=None,
                        help="Optional comma-separated max allowed label per lambda.")
    args = parser.parse_args()

    train_args, checkpoint = load_train_args(args.ckpt)
    ns = SimpleNamespace(**train_args)
    voxel_size = getattr(ns, "voxel_size", [0.16, 0.16, 0.16])
    pc_range = getattr(ns, "point_cloud_range", [0.0, -40.0, -3.0, 70.4, 40.0, 1.0])
    max_voxels = getattr(ns, "max_voxels", 50000)
    num_cost_heads = getattr(ns, "num_cost_heads", 6)
    num_targets = getattr(ns, "num_targets", 3)
    feat_dim = getattr(ns, "feat_dim", 256)
    ap_drop_scale = getattr(ns, "ap_drop_scale", 100.0)
    use_abs_xyz = not getattr(ns, "no_abs_xyz", False)
    allow_negative_cost = getattr(ns, "allow_negative_cost", False)
    monotonic_cost = not getattr(ns, "no_monotonic_cost", False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combos = parse_quant_map(args.quant_map)
    thresholds = parse_thresholds(args.thresholds, scale=ap_drop_scale)
    if thresholds is None:
        raise ValueError("--thresholds produced no threshold rows")
    if args.selection_policy in {"debt", "lagrangian"} and args.split_details_csv and not Path(args.split_details_csv).exists():
        raise FileNotFoundError(args.split_details_csv)
    label_bpp_table = (
        load_label_bpp_table(args.split_details_csv, mode=args.bpp_estimate)
        if args.selection_policy in {"debt", "lagrangian"}
        else None
    )
    lagrange_lambdas = parse_float_list(args.lagrange_lambdas)
    lagrange_class_weights = parse_class_weights(args.lagrange_class_weights)
    lagrange_max_labels = parse_int_list(args.lagrange_max_labels) if args.lagrange_max_labels else None

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dataset = RouterInferenceDataset(args.velodyne_dir, args.split_file, voxel_size, pc_range, max_voxels, use_abs_xyz)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=sparse_collate_fn,
    )

    model = SparseCostProxyNet(
        input_channels=dataset.num_point_features,
        spatial_shape=dataset.spatial_shape,
        feat_dim=feat_dim,
        num_cost_heads=num_cost_heads,
        num_targets=num_targets,
        cost_nonnegative=not allow_negative_cost,
        monotonic_cost=monotonic_cost,
    ).to(device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    calibrator = load_calibrator(args.calibration, device, num_targets, allow_negative=allow_negative_cost)
    thresholds_device = thresholds.to(device)

    cost_rows = []
    all_frame_ids = []
    all_cost = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="router export", dynamic_ncols=True):
            voxel_features = batch["voxel_features"].to(device, non_blocking=True)
            voxel_coords = batch["voxel_coords"].to(device, non_blocking=True)
            cost = model(voxel_features, voxel_coords, batch["batch_size"])["cost_pred"]
            if calibrator is not None:
                cost = calibrator(cost)
            cost_np = (cost.cpu().numpy() / float(ap_drop_scale)).astype(float)
            all_frame_ids.extend(batch["frame_id"])
            all_cost.append(cost.detach().cpu())

            for b, fid in enumerate(batch["frame_id"]):
                row = {"frame_id": fid}
                for h in range(num_cost_heads):
                    for t, name in enumerate(("Car", "Ped", "Cyc")):
                        row[f"L{h + 1}_{name}_cost"] = round(float(cost_np[b, h, t]), 8)
                cost_rows.append(row)

    cost_all = torch.cat(all_cost, dim=0).to(device)
    if args.selection_policy == "hard":
        labels = cost_to_jucp_labels(cost_all, thresholds_device).cpu().numpy()
        selection_stats = None
    elif args.selection_policy == "debt":
        labels, debt_stats = cost_to_jucp_labels_debt(
            cost_all.cpu(),
            thresholds,
            all_frame_ids,
            label_bpp_table=label_bpp_table,
            target_index=TARGET_INDEX[args.debt_target],
            alpha=args.debt_alpha,
            beta=args.debt_beta,
            max_extra=args.debt_max_extra * float(ap_drop_scale),
            min_threshold_ratio=args.debt_min_threshold_ratio,
            min_saving_per_cost=args.debt_min_saving_per_cost / float(ap_drop_scale),
        )
        selection_stats = debt_stats
    else:
        labels, lagrange_stats = cost_to_jucp_labels_lagrangian(
            cost_all.cpu(),
            all_frame_ids,
            lagrange_lambdas,
            label_bpp_table=label_bpp_table,
            class_weights=lagrange_class_weights,
            ap_drop_scale=ap_drop_scale,
            max_labels=lagrange_max_labels,
        )
        selection_stats = lagrange_stats

    if args.selection_policy == "lagrangian":
        rate_count = len(lagrange_lambdas)
        rate_texts = [f"lambda={lam:g}" for lam in lagrange_lambdas]
    else:
        rate_count = thresholds.shape[0]
        rate_texts = [item.strip() for item in args.thresholds.split(";")]

    label_rows = [[] for _ in range(rate_count)]
    for b, fid in enumerate(all_frame_ids):
        for rate_id in range(rate_count):
            label = int(labels[b, rate_id])
            fg, bg = combos[label]
            row = {
                "frame_id": fid,
                "jucp_label": label,
                "rate_id": rate_id,
                "threshold": rate_texts[rate_id],
                "posQ_fg": fg,
                "posQ_bg": bg,
            }
            if args.selection_policy == "debt":
                stats = selection_stats[rate_id][b]
                row.update({
                    "hard_label": stats["hard_label"],
                    "debt_extra": round(stats["extra"] / float(ap_drop_scale), 8),
                    "debt_bpp_saving": round(stats["saving"], 8),
                    "debt_after": round(stats["debt"] / float(ap_drop_scale), 8),
                    "effective_threshold": round(stats["effective_threshold"] / float(ap_drop_scale), 8),
                })
            elif args.selection_policy == "lagrangian":
                stats = selection_stats[rate_id][b]
                row.update({
                    "lagrange_lambda": f"{lagrange_lambdas[rate_id]:g}",
                    "lagrange_score": round(stats["lagrange_score"], 8),
                    "weighted_ap_drop": round(stats["weighted_cost"], 8),
                    "label_bpp": round(stats["label_bpp"], 8),
                    "max_label": stats["max_label"],
                })
            label_rows[rate_id].append(row)

    cost_csv = Path(args.cost_csv) if args.cost_csv else out_dir / f"{args.prefix}_costs.csv"
    with open(cost_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cost_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cost_rows)

    manifest = {
        "ckpt": str(args.ckpt),
        "calibration": str(args.calibration or ""),
        "split_file": str(args.split_file),
        "velodyne_dir": str(args.velodyne_dir),
        "quant_map": args.quant_map,
        "thresholds": args.thresholds,
        "selection_policy": args.selection_policy,
        "debt_target": args.debt_target if args.selection_policy == "debt" else "",
        "debt_alpha": args.debt_alpha if args.selection_policy == "debt" else "",
        "debt_beta": args.debt_beta if args.selection_policy == "debt" else "",
        "debt_max_extra": args.debt_max_extra if args.selection_policy == "debt" else "",
        "debt_min_threshold_ratio": args.debt_min_threshold_ratio if args.selection_policy == "debt" else "",
        "debt_min_saving_per_cost": args.debt_min_saving_per_cost if args.selection_policy == "debt" else "",
        "lagrange_lambdas": args.lagrange_lambdas if args.selection_policy == "lagrangian" else "",
        "lagrange_class_weights": args.lagrange_class_weights if args.selection_policy == "lagrangian" else "",
        "lagrange_max_labels": args.lagrange_max_labels if args.selection_policy == "lagrangian" else "",
        "bpp_estimate": args.bpp_estimate if args.selection_policy in {"debt", "lagrangian"} else "",
        "split_details_csv": str(args.split_details_csv or ""),
        "ap_drop_scale": ap_drop_scale,
        "cost_csv": str(cost_csv),
        "label_csvs": [],
    }
    for rate_id, rows in enumerate(label_rows):
        rate_name = rows[0]["threshold"].replace("=", "_").replace(",", "_").replace(".", "p")
        out_csv = out_dir / f"{args.prefix}_rate_{rate_id}_{rate_name}.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        manifest["label_csvs"].append({"rate_id": rate_id, "threshold": rows[0]["threshold"], "path": str(out_csv)})

    manifest_path = out_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Cost CSV: {cost_csv}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
