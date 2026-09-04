#!/usr/bin/env python3
"""Merge disjoint quantized-loss NPZ shards into global sample-index order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    args = parser.parse_args()

    loaded = [np.load(path) for path in args.shards]
    qsteps = loaded[0]["qsteps"]
    for shard in loaded[1:]:
        if not np.array_equal(shard["qsteps"], qsteps):
            raise RuntimeError("qstep mismatch between shards")
    indices = np.concatenate([shard["indices"] for shard in loaded]).astype(np.int64)
    if len(indices) != args.expected_samples or len(np.unique(indices)) != args.expected_samples:
        raise RuntimeError(
            f"Expected {args.expected_samples} unique samples, got rows={len(indices)} unique={len(np.unique(indices))}"
        )
    order = np.argsort(indices)
    indices = indices[order]
    if not np.array_equal(indices, np.arange(args.expected_samples)):
        raise RuntimeError("Merged indices are not exactly 0..N-1")
    payload = {"indices": indices, "qsteps": qsteps}
    for key in ("labels", "losses", "loss_deltas", "predictions", "unique_counts"):
        payload[key] = np.concatenate([shard[key] for shard in loaded], axis=0)[order]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    summary = {
        "samples": args.expected_samples,
        "shards": [str(Path(path).resolve()) for path in args.shards],
        "qsteps_coarse_to_fine": qsteps.astype(float).tolist(),
        "index_order": "0..N-1",
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
