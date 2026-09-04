#!/usr/bin/env python3
"""Merge and validate sharded nuScenes detector-loss CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--tokens-out", default="")
    args = parser.parse_args()

    paths = sorted(Path(args.shard_root).glob("shard_*/loss.csv"))
    if not paths:
        raise FileNotFoundError(f"No loss shards below {args.shard_root}")
    frame = pd.concat(
        [pd.read_csv(path, dtype={"scene_id": str, "sample_idx": str}) for path in paths],
        ignore_index=True,
    )
    frame = frame.sort_values("dataset_index").reset_index(drop=True)
    if len(frame) != args.expected_samples:
        raise RuntimeError(
            f"Expected {args.expected_samples} rows, found {len(frame)}"
        )
    if frame["dataset_index"].astype(int).tolist() != list(range(len(frame))):
        raise RuntimeError("Merged shards do not cover every dataset index")
    if frame["scene_id"].duplicated().any():
        raise RuntimeError("Duplicate sample tokens across shards")
    for level in range(6):
        if f"L{level}_signed_delta" not in frame:
            raise RuntimeError(f"Missing L{level}_signed_delta")

    output = Path(args.output_csv).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    tokens_out = Path(args.tokens_out).resolve() if args.tokens_out else None
    if tokens_out is not None:
        tokens_out.parent.mkdir(parents=True, exist_ok=True)
        tokens_out.write_text("".join(f"{token}\n" for token in frame["scene_id"]))
    summary = {
        "output_csv": str(output),
        "samples": len(frame),
        "shards": [str(path.resolve()) for path in paths],
        "tokens_out": str(tokens_out) if tokens_out else None,
    }
    output.with_suffix(".merge.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
