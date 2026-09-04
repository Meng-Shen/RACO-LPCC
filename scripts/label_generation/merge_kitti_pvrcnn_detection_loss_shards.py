#!/usr/bin/env python3
"""Merge disjoint KITTI PV-RCNN detection-loss frame shards."""

import argparse
import csv
import json
from pathlib import Path


def norm_frame_id(value):
    return str(value).strip().zfill(6)


def read_split(path):
    with open(path) as handle:
        return [norm_frame_id(line) for line in handle if line.strip()]


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def shard_index(path):
    try:
        return int(path.name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 10**9


def sanitized_number(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def validate_rows(rows, expected_frame_ids, description):
    actual = [norm_frame_id(row["frame_id"]) for row in rows]
    if len(actual) != len(set(actual)):
        raise ValueError(f"Duplicate frame_id values in {description}")
    if set(actual) != set(expected_frame_ids):
        missing = sorted(set(expected_frame_ids) - set(actual))
        extra = sorted(set(actual) - set(expected_frame_ids))
        raise ValueError(
            f"Frame mismatch in {description}: missing={missing[:5]} extra={extra[:5]} "
            f"actual={len(actual)} expected={len(expected_frame_ids)}"
        )
    order = {frame_id: idx for idx, frame_id in enumerate(expected_frame_ids)}
    rows.sort(key=lambda row: order[norm_frame_id(row["frame_id"])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_root", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--loss_csv", required=True)
    parser.add_argument("--prefix", default="uniform_detection_loss")
    args = parser.parse_args()

    shard_dirs = sorted(
        [path for path in Path(args.shard_root).glob("shard_*") if path.is_dir()],
        key=shard_index,
    )
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* directories under {args.shard_root}")

    manifests = []
    for shard_dir in shard_dirs:
        manifest_path = shard_dir / "labels" / f"{args.prefix}_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing shard manifest: {manifest_path}")
        manifests.append(json.loads(manifest_path.read_text()))

    reference = manifests[0]
    metadata_keys = [
        "mode", "cfg_file", "ckpt", "split_file", "mask_dir", "quant_map",
        "candidate_labels", "loss_thresholds", "baseline", "baseline_label",
        "baseline_quantization", "loss_definition",
    ]
    for shard_id, manifest in enumerate(manifests[1:], start=1):
        for key in metadata_keys:
            if manifest.get(key) != reference.get(key):
                raise ValueError(f"Shard {shard_id} metadata mismatch for {key}")

    expected_frame_ids = read_split(args.split_file)
    loss_rows = []
    for manifest in manifests:
        loss_rows.extend(read_csv(manifest["loss_csv"]))
    validate_rows(loss_rows, expected_frame_ids, "merged loss CSV")
    write_csv(args.loss_csv, loss_rows)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_label_items = []
    reference_items = sorted(reference["label_csvs"], key=lambda item: int(item["rate_id"]))
    for reference_item in reference_items:
        rate_id = int(reference_item["rate_id"])
        threshold = reference_item["threshold"]
        rows = []
        for manifest in manifests:
            item_by_rate = {int(item["rate_id"]): item for item in manifest["label_csvs"]}
            rows.extend(read_csv(item_by_rate[rate_id]["path"]))
        validate_rows(rows, expected_frame_ids, f"rate {rate_id} labels")
        output = out_dir / f"{args.prefix}_rate_{rate_id}_{sanitized_number(threshold)}.csv"
        write_csv(output, rows)
        merged_label_items.append(
            {"rate_id": rate_id, "threshold": threshold, "path": str(output)}
        )

    merged_manifest = dict(reference)
    merged_manifest["loss_csv"] = str(Path(args.loss_csv).resolve())
    merged_manifest["label_csvs"] = merged_label_items
    merged_manifest["num_frames"] = len(expected_frame_ids)
    merged_manifest["parallel_shards"] = len(manifests)
    merged_manifest["shard_elapsed_seconds"] = [
        float(manifest.get("elapsed_seconds", 0.0)) for manifest in manifests
    ]
    merged_manifest["elapsed_seconds"] = max(merged_manifest["shard_elapsed_seconds"])
    manifest_path = out_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(merged_manifest, indent=2))

    print(f"Merged {len(manifests)} loss shards and {len(expected_frame_ids)} frames")
    print(f"Loss CSV: {args.loss_csv}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
