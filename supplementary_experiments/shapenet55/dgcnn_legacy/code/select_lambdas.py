#!/usr/bin/env python3
"""Select six train-only RD multipliers with well-spaced oracle rates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_bpp(path: Path, indices: np.ndarray, levels: int):
    table = {(int(row["sample_index"]), int(row["level"])): float(row["bpp"])
             for row in csv.DictReader(path.open(newline=""))}
    return np.asarray([[table[(int(index), level)] for level in range(levels)]
                       for index in indices], dtype=np.float64)


def selected(loss, bpp, lam):
    saving = bpp[:, -1:] - bpp
    return np.argmin(loss - lam * saving, axis=1)


def average_rate(loss, bpp, lam):
    levels = selected(loss, bpp, lam)
    return float(bpp[np.arange(len(bpp)), levels].mean()), levels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-npz", required=True)
    parser.add_argument("--bpp-csv", required=True)
    parser.add_argument("--train-indices", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    quant = np.load(args.quant_npz)
    all_indices = quant["indices"].astype(np.int64)
    row_by_index = {int(index): row for row, index in enumerate(all_indices)}
    train_indices = np.load(args.train_indices).astype(np.int64)
    rows = np.asarray([row_by_index[int(index)] for index in train_indices])
    loss = quant["loss_deltas"][rows].astype(np.float64)
    bpp = load_bpp(Path(args.bpp_csv), train_indices, loss.shape[1])

    high = 1.0
    rate_high, levels_high = average_rate(loss, bpp, high)
    coarse_mean = float(bpp[:, 0].mean())
    while high < 1e8 and (rate_high > coarse_mean * 1.001 or np.mean(levels_high == 0) < 0.995):
        high *= 2.0
        rate_high, levels_high = average_rate(loss, bpp, high)
    zero_rate, _ = average_rate(loss, bpp, 0.0)
    targets = np.geomspace(max(zero_rate, coarse_mean), coarse_mean, 6)
    lambdas = [0.0]
    achieved = [zero_rate]
    for target in targets[1:-1]:
        lower, upper = 0.0, high
        for _ in range(64):
            middle = (lower + upper) / 2.0
            rate, _ = average_rate(loss, bpp, middle)
            if rate > target:
                lower = middle
            else:
                upper = middle
        lam = (lower + upper) / 2.0
        rate, _ = average_rate(loss, bpp, lam)
        lambdas.append(float(lam))
        achieved.append(rate)
    lambdas.append(float(high))
    achieved.append(rate_high)

    output = {
        "selection_data": "official ModelNet40 train split; router-train subset only",
        "routing_rule": "argmin_q DeltaCE(q)-lambda*(BPP_fine-BPP_q)",
        "lambdas_high_rate_to_low_rate": lambdas,
        "target_mean_bpp_high_to_low": targets.tolist(),
        "achieved_oracle_mean_bpp": achieved,
        "lambda_upper_search_bound": high,
        "router_train_samples": int(len(train_indices)),
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
