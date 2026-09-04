#!/usr/bin/env python3
"""Resumable six-rate G-PCC measurement for SUN RGB-D point clouds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np


QSTEPS_MM = (160.0, 80.0, 40.0, 20.0, 10.0, 5.0)
FIELDS = (
    'scene_id', 'dataset_index', 'split', 'rate_id', 'qstep_mm',
    'position_quantization_scale', 'num_points', 'bits', 'bpp', 'enc_time',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--points-dir', type=Path)
    parser.add_argument('--split-file', type=Path)
    parser.add_argument('--split-name', choices=('train', 'val'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--tmp-dir', type=Path)
    parser.add_argument('--tmc3', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--max-scenes', type=int, default=0)
    parser.add_argument('--merge-root', type=Path)
    parser.add_argument('--expected-scenes', type=int, default=0)
    parser.add_argument('--qsteps-mm', type=float, nargs='+', default=QSTEPS_MM)
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    ids = [f'{int(line.strip()):06d}' for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise RuntimeError('Split contains duplicate scene IDs')
    return ids


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows, seen = [], set()
    try:
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                try:
                    key = (row['scene_id'], int(row['rate_id']))
                    bits = int(row['bits'])
                    points = int(row['num_points'])
                except (KeyError, TypeError, ValueError):
                    continue
                if key not in seen and bits > 0 and points > 0:
                    rows.append({field: row.get(field, '') for field in FIELDS})
                    seen.add(key)
    except csv.Error:
        pass
    return rows


def atomic_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def write_binary_ply(path: Path, xyz: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype='<f4', order='C')
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        f'element vertex {len(xyz)}\n'
        'property float x\nproperty float y\nproperty float z\nend_header\n'
    ).encode('ascii')
    with path.open('wb') as handle:
        handle.write(header)
        handle.write(xyz.tobytes(order='C'))


def encode_one(tmc3: Path, config: Path, tmp_dir: Path, sid: str,
               rate_id: int, qstep_mm: float,
               coords_mm: np.ndarray) -> tuple[int, float]:
    token = f'{os.getpid()}_{sid}_{rate_id}'
    ply = tmp_dir / f'{token}.ply'
    bitstream = tmp_dir / f'{token}.bin'
    write_binary_ply(ply, coords_mm)
    started = time.perf_counter()
    try:
        process = subprocess.run([
            str(tmc3), '--mode=0', f'--config={config}',
            f'--positionQuantizationScale={1.0 / qstep_mm:.15g}',
            f'--uncompressedDataPath={ply}',
            f'--compressedStreamPath={bitstream}',
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if process.returncode != 0 or not bitstream.is_file() or bitstream.stat().st_size == 0:
            raise RuntimeError(
                f'tmc3 failed for {sid} L{rate_id}: rc={process.returncode}\n'
                + process.stdout[-3000:]
            )
        return bitstream.stat().st_size * 8, time.perf_counter() - started
    finally:
        for path in (ply, bitstream):
            if path.exists():
                path.unlink()


def run_shard(args) -> None:
    required = (args.points_dir, args.split_file, args.split_name,
                args.tmp_dir, args.tmc3, args.config)
    if any(value is None for value in required):
        raise ValueError('Shard mode requires points, split, temp, tmc3 and config paths')
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError('Invalid shard specification')
    ids = read_ids(args.split_file.resolve())
    indices = list(range(args.shard_id, len(ids), args.num_shards))
    if args.max_scenes > 0:
        indices = indices[:args.max_scenes]
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    rows = load_rows(output)
    completed = {(row['scene_id'], int(row['rate_id'])) for row in rows}
    started = time.time()
    for ordinal, index in enumerate(indices, 1):
        sid = ids[index]
        missing = [
            level for level in range(len(args.qsteps_mm))
            if (sid, level) not in completed
        ]
        if missing:
            raw = np.fromfile(args.points_dir / f'{sid}.bin', dtype=np.float32)
            points = raw.reshape(-1, 6)
            xyz_mm = np.rint(points[:, :3].astype(np.float64) * 1000.0)
            xyz_mm -= xyz_mm.min(axis=0)
            for level in missing:
                bits, enc_time = encode_one(
                    args.tmc3.resolve(), args.config.resolve(), args.tmp_dir.resolve(),
                    sid, level, args.qsteps_mm[level], xyz_mm,
                )
                rows.append({
                    'scene_id': sid, 'dataset_index': index, 'split': args.split_name,
                    'rate_id': level, 'qstep_mm': args.qsteps_mm[level],
                    'position_quantization_scale': 1.0 / args.qsteps_mm[level],
                    'num_points': len(points), 'bits': bits,
                    'bpp': bits / len(points), 'enc_time': enc_time,
                })
                completed.add((sid, level))
                rows.sort(key=lambda row: (int(row['dataset_index']), int(row['rate_id'])))
                atomic_write(output, rows)
        if ordinal == 1 or ordinal % 20 == 0 or ordinal == len(indices):
            print(json.dumps({
                'split': args.split_name, 'shard': args.shard_id,
                'visited': ordinal, 'assigned': len(indices), 'scene_id': sid,
                'completed_rows': len(rows), 'elapsed_seconds': time.time() - started,
            }), flush=True)
    output.with_suffix('.manifest.json').write_text(json.dumps({
        'status': 'complete', 'dataset': 'SUN RGB-D', 'split': args.split_name,
        'qsteps_mm_coarse_to_fine': args.qsteps_mm,
        'shard_id': args.shard_id,
        'num_shards': args.num_shards, 'assigned_scenes': len(indices),
        'completed_rows': len(rows),
        'bpp_definition': 'encoded geometry bits / original stored scene points',
    }, indent=2))


def merge(args) -> None:
    if not args.merge_root or not args.split_file or not args.expected_scenes:
        raise ValueError('Merge mode requires merge-root, split-file and expected-scenes')
    ids = read_ids(args.split_file.resolve())
    if len(ids) != args.expected_scenes:
        raise RuntimeError(f'Expected {args.expected_scenes} split IDs, got {len(ids)}')
    rows, seen = [], set()
    for path in sorted(args.merge_root.resolve().glob('shard_*/gpcc.csv')):
        for row in load_rows(path):
            key = (row['scene_id'], int(row['rate_id']))
            if key in seen:
                raise RuntimeError(f'Duplicate key {key}')
            seen.add(key)
            rows.append(row)
    expected = args.expected_scenes * len(args.qsteps_mm)
    if len(rows) != expected:
        raise RuntimeError(f'Expected {expected} rows, got {len(rows)}')
    rows.sort(key=lambda row: (int(row['dataset_index']), int(row['rate_id'])))
    atomic_write(args.output.resolve(), rows)
    averages = []
    for level, qstep in enumerate(args.qsteps_mm):
        selected = [row for row in rows if int(row['rate_id']) == level]
        total_bits = sum(int(row['bits']) for row in selected)
        total_points = sum(int(row['num_points']) for row in selected)
        averages.append({
            'rate_id': level, 'qstep_mm': qstep, 'scenes': len(selected),
            'total_bits': total_bits, 'total_points': total_points,
            'bpp': total_bits / total_points,
        })
    average_path = args.output.resolve().with_name(
        f'sunrgbd_{rows[0]["split"]}_gpcc_average.csv')
    with average_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=averages[0])
        writer.writeheader()
        writer.writerows(averages)
    print(json.dumps({'rows': len(rows), 'output': str(args.output.resolve()),
                      'averages': averages}, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.merge_root:
        merge(args)
    else:
        run_shard(args)


if __name__ == '__main__':
    main()
