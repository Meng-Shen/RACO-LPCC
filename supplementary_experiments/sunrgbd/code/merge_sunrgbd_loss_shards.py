#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--split-file', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expected-scenes', required=True, type=int)
    parser.add_argument('--num-levels', type=int, default=6)
    args = parser.parse_args()

    ids = [f'{int(line):06d}' for line in args.split_file.read_text().splitlines() if line.strip()]
    if len(ids) != args.expected_scenes:
        raise RuntimeError(f'Expected {args.expected_scenes} split IDs, got {len(ids)}')
    rows, seen = [], set()
    fields = None
    for path in sorted(args.root.glob('shard_*/loss.csv')):
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                sid = row['scene_id']
                if sid in seen:
                    raise RuntimeError(f'Duplicate scene {sid}')
                seen.add(sid)
                rows.append(row)
                fields = fields or list(row)
    if len(rows) != args.expected_scenes:
        raise RuntimeError(f'Expected {args.expected_scenes} rows, got {len(rows)}')
    if seen != set(ids):
        raise RuntimeError('Merged scenes do not exactly match the split')
    rows.sort(key=lambda row: int(row['dataset_index']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + '.tmp')
    with temp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(args.output)
    for row in rows:
        for level in range(args.num_levels):
            float(row[f'L{level}_total_loss'])
    args.output.with_suffix('.manifest.json').write_text(json.dumps({
        'status': 'complete', 'rows': len(rows), 'absolute_task_losses': True,
        'num_levels': args.num_levels, 'output': str(args.output.resolve()),
    }, indent=2))
    print(json.dumps({'rows': len(rows), 'output': str(args.output.resolve())}, indent=2))


if __name__ == '__main__':
    main()
