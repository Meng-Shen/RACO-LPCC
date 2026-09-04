#!/usr/bin/env python3
"""Combine cached SUN RGB-D assets into the revised six-level qstep set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mmengine


DEFAULT_BASE_QSTEPS = (160.0, 80.0, 40.0, 20.0, 10.0, 5.0)
DEFAULT_EXTRA_QSTEPS = (120.0, 60.0)
DEFAULT_TARGET_QSTEPS = (160.0, 120.0, 80.0, 60.0, 40.0, 20.0)


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_source_map(base_qsteps, extra_qsteps, target_qsteps):
    if len(set(base_qsteps)) != len(base_qsteps):
        raise RuntimeError('Base qsteps contain duplicates')
    if len(set(extra_qsteps)) != len(extra_qsteps):
        raise RuntimeError('Supplemental qsteps contain duplicates')
    overlap = set(base_qsteps) & set(extra_qsteps)
    if overlap:
        raise RuntimeError(f'Qsteps appear in both sources: {sorted(overlap)}')
    available = {
        qstep: ('base', index) for index, qstep in enumerate(base_qsteps)
    }
    available.update({
        qstep: ('extra', index) for index, qstep in enumerate(extra_qsteps)
    })
    missing = [qstep for qstep in target_qsteps if qstep not in available]
    if missing:
        raise RuntimeError(f'Target qsteps are unavailable: {missing}')
    return {qstep: available[qstep] for qstep in target_qsteps}


def combine_loss(old_path, extra_path, output, source, target_qsteps):
    old = {row['scene_id']: row for row in read_csv(old_path)}
    extra = {row['scene_id']: row for row in read_csv(extra_path)}
    if old.keys() != extra.keys():
        raise RuntimeError('Old and supplemental loss scene sets differ')
    rows = []
    for sid in sorted(old, key=lambda key: int(old[key]['dataset_index'])):
        row = {
            'scene_id': sid, 'dataset_index': old[sid]['dataset_index'],
            'split': old[sid]['split'],
        }
        for target_level, qstep in enumerate(target_qsteps):
            group, source_level = source[qstep]
            source_row = old[sid] if group == 'base' else extra[sid]
            prefix = f'L{source_level}_'
            for key, value in source_row.items():
                if key.startswith(prefix):
                    row[f'L{target_level}_{key[len(prefix):]}'] = value
        rows.append(row)
    write_csv(output, rows)
    Path(output).with_suffix('.manifest.json').write_text(json.dumps({
        'status': 'complete', 'rows': len(rows),
        'qsteps_mm_coarse_to_fine': target_qsteps,
        'sources': {'base': str(old_path), 'supplemental': str(extra_path)},
    }, indent=2))


def combine_gpcc(old_path, extra_path, output, source, target_qsteps):
    old_rows = read_csv(old_path)
    extra_rows = read_csv(extra_path)
    old = {(row['scene_id'], float(row['qstep_mm'])): row for row in old_rows}
    extra = {(row['scene_id'], float(row['qstep_mm'])): row for row in extra_rows}
    scene_indices = {
        row['scene_id']: int(row['dataset_index']) for row in old_rows
    }
    scene_ids = sorted(scene_indices, key=scene_indices.get)
    rows = []
    for sid in scene_ids:
        for level, qstep in enumerate(target_qsteps):
            group, _ = source[qstep]
            source_rows = old if group == 'base' else extra
            row = dict(source_rows[(sid, qstep)])
            row['rate_id'] = level
            row['qstep_mm'] = qstep
            rows.append(row)
    write_csv(output, rows)
    Path(output).with_suffix('.manifest.json').write_text(json.dumps({
        'status': 'complete', 'rows': len(rows),
        'qsteps_mm_coarse_to_fine': target_qsteps,
        'bpp_definition': 'encoded geometry bits / original stored scene points',
    }, indent=2))


def combine_predictions(old_root, extra_root, output_root, shards, source,
                        target_qsteps):
    old_root, extra_root, output_root = map(Path, (old_root, extra_root, output_root))
    for shard in range(shards):
        old_path = old_root / f'shard_{shard}' / 'predictions.pkl'
        extra_path = extra_root / f'shard_{shard}' / 'predictions.pkl'
        old_records = {int(row['dataset_index']): row for row in mmengine.load(old_path)}
        extra_records = {
            int(row['dataset_index']): row for row in mmengine.load(extra_path)}
        if old_records.keys() != extra_records.keys():
            raise RuntimeError(f'Prediction indices differ in shard {shard}')
        combined = []
        for index in sorted(old_records):
            old_row, extra_row = old_records[index], extra_records[index]
            if old_row['scene_id'] != extra_row['scene_id']:
                raise RuntimeError(f'Prediction scene mismatch at index {index}')
            predictions = []
            for qstep in target_qsteps:
                group, source_level = source[qstep]
                record = old_row if group == 'base' else extra_row
                predictions.append(record['predictions'][source_level])
            row = dict(old_row)
            row['predictions'] = predictions
            combined.append(row)
        output = output_root / f'shard_{shard}' / 'predictions.pkl'
        output.parent.mkdir(parents=True, exist_ok=True)
        mmengine.dump(combined, output)
        output.with_suffix('.manifest.json').write_text(json.dumps({
            'status': 'complete', 'dataset': 'SUN RGB-D', 'split': 'train',
            'qsteps_mm_coarse_to_fine': target_qsteps,
            'shard_id': shard, 'num_shards': shards,
            'records': len(combined),
        }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-loss', required=True, type=Path)
    parser.add_argument('--extra-loss', required=True, type=Path)
    parser.add_argument('--output-loss', required=True, type=Path)
    parser.add_argument('--old-gpcc', required=True, type=Path)
    parser.add_argument('--extra-gpcc', required=True, type=Path)
    parser.add_argument('--output-gpcc', required=True, type=Path)
    parser.add_argument('--old-predictions', required=True, type=Path)
    parser.add_argument('--extra-predictions', required=True, type=Path)
    parser.add_argument('--output-predictions', required=True, type=Path)
    parser.add_argument('--shards', type=int, default=7)
    parser.add_argument(
        '--base-qsteps', type=float, nargs='+', default=DEFAULT_BASE_QSTEPS)
    parser.add_argument(
        '--extra-qsteps', type=float, nargs='+', default=DEFAULT_EXTRA_QSTEPS)
    parser.add_argument(
        '--target-qsteps', type=float, nargs='+', default=DEFAULT_TARGET_QSTEPS)
    args = parser.parse_args()
    source = make_source_map(
        args.base_qsteps, args.extra_qsteps, args.target_qsteps)
    combine_loss(
        args.old_loss, args.extra_loss, args.output_loss,
        source, args.target_qsteps)
    combine_gpcc(
        args.old_gpcc, args.extra_gpcc, args.output_gpcc,
        source, args.target_qsteps)
    combine_predictions(
        args.old_predictions, args.extra_predictions,
        args.output_predictions, args.shards, source, args.target_qsteps)
    print(json.dumps({
        'status': 'complete',
        'qsteps_mm_coarse_to_fine': args.target_qsteps,
        'output_loss': str(args.output_loss.resolve()),
        'output_gpcc': str(args.output_gpcc.resolve()),
        'output_predictions': str(args.output_predictions.resolve()),
    }, indent=2))


if __name__ == '__main__':
    main()
