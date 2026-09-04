#!/usr/bin/env python3
"""Combine two SUN RGB-D G-PCC CSVs into a requested qstep order."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--extra', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--target-qsteps', required=True, type=float, nargs='+')
    parser.add_argument('--expected-scenes', required=True, type=int)
    args = parser.parse_args()

    base_rows, extra_rows = load(args.base), load(args.extra)
    available = {}
    scene_indices = {}
    for row in base_rows + extra_rows:
        key = (row['scene_id'], float(row['qstep_mm']))
        if key in available:
            continue
        available[key] = row
        scene_indices[row['scene_id']] = int(row['dataset_index'])
    if len(scene_indices) != args.expected_scenes:
        raise RuntimeError(
            f'Expected {args.expected_scenes} scenes, got {len(scene_indices)}')
    rows = []
    for sid in sorted(scene_indices, key=scene_indices.get):
        for level, qstep in enumerate(args.target_qsteps):
            key = (sid, qstep)
            if key not in available:
                raise RuntimeError(f'Missing G-PCC entry {key}')
            row = dict(available[key])
            row['rate_id'] = level
            row['qstep_mm'] = qstep
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    averages = []
    for level, qstep in enumerate(args.target_qsteps):
        selected = [row for row in rows if int(row['rate_id']) == level]
        bits = sum(int(row['bits']) for row in selected)
        points = sum(int(row['num_points']) for row in selected)
        averages.append({
            'rate_id': level, 'qstep_mm': qstep,
            'scenes': len(selected), 'total_bits': bits,
            'total_points': points, 'bpp': bits / points,
        })
    average_path = args.output.with_name(args.output.stem + '_average.csv')
    with average_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(averages[0]))
        writer.writeheader()
        writer.writerows(averages)
    payload = {
        'status': 'complete', 'rows': len(rows),
        'scenes': args.expected_scenes,
        'qsteps_mm_coarse_to_fine': args.target_qsteps,
        'bpp_definition': 'total encoded geometry bits / total original points',
        'averages': averages, 'output': str(args.output.resolve()),
    }
    args.output.with_suffix('.manifest.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
