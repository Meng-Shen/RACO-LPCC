#!/usr/bin/env python3
"""Smoke-test nuScenes loss merging, split creation, and lambda calibration."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    base = Path(args.base).resolve()
    code = base / "code"
    source = base / "preflight" / "rate_proxy" / "loss.csv"
    root = base / "preflight" / "label_pipeline"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    frame = pd.read_csv(source, dtype={"scene_id": str})
    for shard in range(2):
        folder = root / f"shard_{shard}"
        folder.mkdir()
        frame.iloc[shard::2].to_csv(folder / "loss.csv", index=False)

    merged = root / "merged.csv"
    train_split, val_split = root / "train.txt", root / "val.txt"
    run(
        code / "merge_nuscenes_quant_losses.py",
        "--shard-root", root,
        "--output-csv", merged,
        "--train-split", train_split,
        "--val-split", val_split,
        "--val-percent", 25,
    )
    generic_merged = root / "generic_merged.csv"
    run(
        code / "merge_nuscenes_loss_shards.py",
        "--shard-root", root,
        "--output-csv", generic_merged,
        "--expected-samples", len(frame),
        "--tokens-out", root / "tokens.txt",
    )
    lambdas = root / "lambdas.json"
    run(
        code / "select_scannet_rd_lambdas.py",
        "--dataset-format", "nuscenes",
        "--loss-csv", merged,
        "--bpp-csv", base / "labels" / "nuscenes_train_gpcc_per_frame_per_rate.csv",
        "--split-file", train_split,
        "--output-json", lambdas,
    )
    payload = json.loads(lambdas.read_text())
    values = payload["lambdas_high_rate_to_low_rate"]
    if len(values) != 6 or any(b <= a for a, b in zip(values, values[1:])):
        raise RuntimeError(f"Invalid calibrated lambdas: {values}")
    result = {
        "status": "PASS",
        "merged_rows": len(pd.read_csv(merged)),
        "train_tokens": len(train_split.read_text().splitlines()),
        "val_tokens": len(val_split.read_text().splitlines()),
        "lambdas": values,
    }
    (root / "PASS.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
