#!/usr/bin/env python3
"""Convert the labelled PoinTr/Point-BERT ShapeNet55 release to fixed XYZ arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def find_dataset_root(search_root: Path) -> Path:
    matches = []
    for train_txt in search_root.rglob("train.txt"):
        if train_txt.parent.name != "ShapeNet-55":
            continue
        candidate = train_txt.parent.parent
        if (candidate / "shapenet_pc").is_dir() and (train_txt.parent / "test.txt").is_file():
            matches.append(candidate)
    if not matches:
        raise FileNotFoundError(
            f"Could not find ShapeNet-55/{{train,test}}.txt and shapenet_pc below {search_root}"
        )
    matches.sort(key=lambda path: len(path.parts))
    return matches[0]


def read_split(path: Path):
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names:
        raise RuntimeError(f"Empty split file: {path}")
    return names


def resolve_cloud(pc_root: Path, name: str) -> Path:
    candidates = [pc_root / name, pc_root / f"{name}.npy"]
    if name.endswith(".npy"):
        candidates.insert(0, pc_root / name)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Point cloud for split entry {name!r} was not found below {pc_root}")


def taxonomy(name: str) -> str:
    return Path(name).stem.split("-", 1)[0]


def normalize_and_sample(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Unexpected point array shape: {points.shape}")
    points = points[:, :3]
    points = points - points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(points, axis=1).max()
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Degenerate point cloud")
    points = points / radius
    rng = np.random.default_rng(seed)
    if len(points) >= count:
        choice = rng.choice(len(points), size=count, replace=False)
    else:
        choice = rng.choice(len(points), size=count, replace=True)
    return points[choice].astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    dataset_root = find_dataset_root(Path(args.search_root).resolve())
    split_root = dataset_root / "ShapeNet-55"
    pc_root = dataset_root / "shapenet_pc"
    full_train_names = read_split(split_root / "train.txt")
    full_test_names = read_split(split_root / "test.txt")
    available = {path.name for path in pc_root.glob("*.npy")}
    train_names = [name for name in full_train_names if Path(name).name in available]
    test_names = [name for name in full_test_names if Path(name).name in available]
    if len(train_names) + len(test_names) < 5000:
        raise RuntimeError(
            f"Too few labelled ShapeNet55 point clouds: train={len(train_names)} test={len(test_names)}"
        )
    all_names = train_names + test_names
    if len(set(all_names)) != len(all_names):
        raise RuntimeError("Official train/test split contains duplicate object IDs")

    taxonomies = sorted({taxonomy(name) for name in all_names})
    if len(taxonomies) != 55:
        raise RuntimeError(f"Expected 55 taxonomy IDs, found {len(taxonomies)}")
    taxonomy_to_label = {name: index for index, name in enumerate(taxonomies)}
    labels = np.asarray([taxonomy_to_label[taxonomy(name)] for name in all_names], dtype=np.int16)

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    points_path = output / "all_points.npy"
    points = np.lib.format.open_memmap(
        points_path, mode="w+", dtype=np.float32,
        shape=(len(all_names), args.num_points, 3),
    )
    for index, name in enumerate(all_names):
        source = resolve_cloud(pc_root, name)
        points[index] = normalize_and_sample(
            np.load(source), args.num_points, args.seed + index * 1000003
        )
        if (index + 1) % 1000 == 0 or index + 1 == len(all_names):
            points.flush()
            print(f"prepared={index + 1}/{len(all_names)}", flush=True)
    del points

    official_train = np.arange(len(train_names), dtype=np.int64)
    official_test = np.arange(len(train_names), len(all_names), dtype=np.int64)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.validation_fraction, random_state=args.seed
    )
    train_rows, val_rows = next(splitter.split(official_train, labels[official_train]))
    model_train = official_train[train_rows]
    model_val = official_train[val_rows]

    arrays = {
        "labels.npy": labels,
        "model_train_indices.npy": model_train,
        "model_val_indices.npy": model_val,
        "router_train_indices.npy": model_train,
        "router_val_indices.npy": model_val,
        "test_indices.npy": official_test,
    }
    for name, array in arrays.items():
        np.save(output / name, array)
    (output / "sample_ids.txt").write_text("\n".join(all_names) + "\n")

    class_counts = {
        tax: {
            "label": taxonomy_to_label[tax],
            "train": int(np.sum(labels[official_train] == taxonomy_to_label[tax])),
            "test": int(np.sum(labels[official_test] == taxonomy_to_label[tax])),
        }
        for tax in taxonomies
    }
    digest = hashlib.sha256()
    for name in all_names:
        digest.update(name.encode())
        digest.update(b"\n")
    full_release = len(all_names) == len(full_train_names) + len(full_test_names)
    summary = {
        "dataset": (
            "Full ShapeNet55 labelled PoinTr/Point-BERT release"
            if full_release else
            "ShapeNet55 labelled subset from the sayakpaul/PoinTr release"
        ),
        "dataset_root": str(dataset_root),
        "geometry_features": "XYZ only; centered and scaled to unit sphere",
        "num_points": args.num_points,
        "classes": 55,
        "official_train_samples": int(len(official_train)),
        "classifier_train_samples": int(len(model_train)),
        "classifier_validation_samples": int(len(model_val)),
        "official_test_samples": int(len(official_test)),
        "full_release_train_entries": int(len(full_train_names)),
        "full_release_test_entries": int(len(full_test_names)),
        "labelled_subset_samples": int(len(all_names)),
        "full_release_available": bool(full_release),
        "split_seed": args.seed,
        "test_used_for_training_or_selection": False,
        "sample_id_sha256": digest.hexdigest(),
        "taxonomy_to_label": taxonomy_to_label,
        "class_counts": class_counts,
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
