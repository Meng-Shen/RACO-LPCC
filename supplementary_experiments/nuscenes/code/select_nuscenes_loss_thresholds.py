#!/usr/bin/env python3
"""Select six loss thresholds without enumerating every routing state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-csv', required=True)
    parser.add_argument('--split-file', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--shell', action='store_true')
    return parser.parse_args()


def select_levels(values: np.ndarray, threshold: float) -> np.ndarray:
    valid = values <= float(threshold)
    selected = np.full(len(values), 5, dtype=np.int64)
    assigned = np.zeros(len(values), dtype=bool)
    for level in range(6):
        choose = valid[:, level] & ~assigned
        selected[choose] = level
        assigned |= choose
    return selected


def state(values: np.ndarray, threshold: float):
    selected = select_levels(values, threshold)
    return float(np.mean(selected + 1)), selected


def closest_candidate(values: np.ndarray, candidates: np.ndarray,
                      target_bpp: float) -> int:
    low, high = 0, len(candidates) - 1
    while low < high:
        mid = (low + high) // 2
        mean_bpp, _ = state(values, float(candidates[mid]))
        if mean_bpp > target_bpp:
            low = mid + 1
        else:
            high = mid
    choices = sorted(set([max(0, low - 1), low,
                          min(len(candidates) - 1, low + 1)]))
    return min(choices, key=lambda index: (
        abs(state(values, float(candidates[index]))[0] - target_bpp), index))


def main():
    args = parse_args()
    frame = pd.read_csv(args.loss_csv, dtype={'scene_id': str})
    tokens = {
        line.strip() for line in Path(args.split_file).read_text().splitlines()
        if line.strip()
    }
    frame = frame[frame['scene_id'].isin(tokens)].copy()
    if len(frame) != len(tokens):
        raise ValueError('Split tokens and loss rows do not match')
    columns = [f'L{level}_signed_delta' for level in range(6)]
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite loss values')
    if np.max(np.abs(values[:, 5])) > 1e-7:
        raise ValueError('L5 must be the zero 64mm reference')

    positive = values[:, :5][values[:, :5] > 0]
    if not len(positive):
        raise ValueError('No positive non-reference loss values')
    candidates = np.unique(np.r_[0.0, positive])
    candidates[1:] = np.nextafter(candidates[1:], np.inf)
    # Append a threshold that always accepts the coarsest candidate, including
    # signed losses larger than all positive entries due to numerical ties.
    candidates = np.unique(np.r_[
        candidates, np.nextafter(np.max(values[:, :5]), np.inf)])
    start_bpp, _ = state(values, float(candidates[0]))
    end_bpp, _ = state(values, float(candidates[-1]))
    targets = np.linspace(start_bpp, end_bpp, 6)
    indices = [closest_candidate(values, candidates, target)
               for target in targets]
    # Binary searches can meet the same plateau.  Retain monotonicity and let
    # the summary expose any duplicate operating point rather than leak val.
    indices = np.maximum.accumulate(indices).tolist()
    thresholds = [float(candidates[index]) for index in indices]
    states = [state(values, threshold) for threshold in thresholds]
    payload = dict(
        method='binary_search_even_train_oracle_mean_estimated_bpp',
        loss_csv=str(Path(args.loss_csv).resolve()),
        split_file=str(Path(args.split_file).resolve()),
        num_samples=len(values),
        qsteps_mm_coarse_to_fine=[2048, 1024, 512, 256, 128, 64],
        estimated_bpp_by_level=[1, 2, 3, 4, 5, 6],
        thresholds=thresholds,
        target_mean_bpp=targets.tolist(),
        achieved_mean_bpp=[item[0] for item in states],
        selection_counts=[
            np.bincount(item[1], minlength=6).astype(int).tolist()
            for item in states])
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    if args.shell:
        print(' '.join(f'{value:.12g}' for value in thresholds))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
