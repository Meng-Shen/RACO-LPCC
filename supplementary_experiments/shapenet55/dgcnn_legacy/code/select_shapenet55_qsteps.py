#!/usr/bin/env python3
"""Select six useful ShapeNet55 quantization levels from a validation probe."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fine-tolerance", type=float, default=0.005)
    args = parser.parse_args()

    probe = json.loads(Path(args.probe_json).read_text())
    levels = probe["levels"]
    qsteps = np.asarray([row["qstep"] for row in levels], dtype=float)
    accuracy = np.asarray([row["overall_accuracy"] for row in levels], dtype=float)
    if len(qsteps) < 6 or not np.all(np.diff(qsteps) < 0):
        raise RuntimeError("Probe qsteps must contain at least six coarse-to-fine values")
    envelope = np.maximum.accumulate(accuracy)
    fine_target = envelope[-1] - args.fine_tolerance
    fine_candidates = np.flatnonzero(envelope >= fine_target)
    fine_index = int(fine_candidates[0]) if len(fine_candidates) else len(qsteps) - 1
    fine_index = max(fine_index, 5)

    low_target = max(0.05, 0.10 * envelope[fine_index])
    coarse_candidates = np.flatnonzero(envelope[: fine_index + 1] >= low_target)
    coarse_index = int(coarse_candidates[0]) if len(coarse_candidates) else 0
    if fine_index - coarse_index + 1 < 6:
        coarse_index = max(0, fine_index - 5)

    target_accuracy = np.linspace(envelope[coarse_index], envelope[fine_index], 6)
    middle = range(coarse_index + 1, fine_index)
    best = None
    for combination in itertools.combinations(middle, 4):
        indices = (coarse_index, *combination, fine_index)
        values = envelope[list(indices)]
        score = float(np.sum((values - target_accuracy) ** 2))
        if best is None or score < best[0]:
            best = (score, indices)
    if best is None:
        raise RuntimeError("Could not choose six distinct qsteps")
    selected_indices = list(best[1])
    selected_qsteps = qsteps[selected_indices].tolist()

    output = {
        "selection_data": "ShapeNet55 official-train validation subset only",
        "selection_rule": "six distinct validation-accuracy-spaced levels; finest is the coarsest candidate within tolerance of best validation accuracy",
        "fine_accuracy_tolerance": args.fine_tolerance,
        "candidate_qsteps_coarse_to_fine": qsteps.tolist(),
        "candidate_overall_accuracy": accuracy.tolist(),
        "monotonic_accuracy_envelope": envelope.tolist(),
        "selected_indices": selected_indices,
        "qsteps_coarse_to_fine": selected_qsteps,
        "selected_overall_accuracy": accuracy[selected_indices].tolist(),
        "test_used_for_selection": False,
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()

