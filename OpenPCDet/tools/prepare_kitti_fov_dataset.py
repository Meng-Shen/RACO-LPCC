#!/usr/bin/env python3
"""Build a KITTI data root whose Velodyne files contain camera-FOV points only."""

import argparse
import csv
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--save-indices', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    return parser.parse_args()


def read_calib(path):
    values = {}
    with path.open() as handle:
        for line in handle:
            if ':' not in line:
                continue
            key, raw = line.split(':', 1)
            values[key] = np.asarray(raw.split(), dtype=np.float32)
    p2 = values['P2'].reshape(3, 4)
    rect = values['R0_rect'].reshape(3, 3)
    velo_to_cam = values['Tr_velo_to_cam'].reshape(3, 4)
    return p2, rect, velo_to_cam


def read_png_size(path):
    with path.open('rb') as handle:
        header = handle.read(24)
    if header[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f'Not a PNG file: {path}')
    return struct.unpack('>II', header[16:24])


def get_fov_mask(points, p2, rect, velo_to_cam, image_size):
    points_h = np.column_stack([
        points[:, :3],
        np.ones(points.shape[0], dtype=points.dtype)
    ])
    points_rect = points_h @ velo_to_cam.T @ rect.T
    points_rect_h = np.column_stack([
        points_rect,
        np.ones(points_rect.shape[0], dtype=points_rect.dtype)
    ])
    projected = points_rect_h @ p2.T
    depth = projected[:, 2] - p2[2, 3]
    valid = depth >= 0
    u = np.full_like(depth, -1)
    v = np.full_like(depth, -1)
    u[valid] = projected[valid, 0] / points_rect[valid, 2]
    v[valid] = projected[valid, 1] / points_rect[valid, 2]
    width, height = image_size
    return valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def relative_symlink(source, destination):
    if destination.exists() or destination.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def link_metadata(source_root, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    for source in source_root.iterdir():
        if source.name in ('training', 'testing'):
            continue
        relative_symlink(source.resolve(), output_root / source.name)

    for split_name in ('training', 'testing'):
        source_split = source_root / split_name
        if not source_split.exists():
            continue
        output_split = output_root / split_name
        output_split.mkdir(parents=True, exist_ok=True)
        for source in source_split.iterdir():
            if source.name == 'velodyne':
                continue
            relative_symlink(source.resolve(), output_split / source.name)


def process_frame(task):
    split_name, bin_path, source_root, output_root, overwrite, save_indices = task
    frame_id = bin_path.stem
    output_bin = output_root / split_name / 'velodyne' / bin_path.name
    index_path = output_root / split_name / 'fov_indices' / f'{frame_id}.npy'

    if output_bin.exists() and not overwrite:
        input_count = bin_path.stat().st_size // (4 * 4)
        output_count = output_bin.stat().st_size // (4 * 4)
        return split_name, frame_id, input_count, output_count

    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    p2, rect, velo_to_cam = read_calib(
        source_root / split_name / 'calib' / f'{frame_id}.txt')
    image_size = read_png_size(
        source_root / split_name / 'image_2' / f'{frame_id}.png')
    keep = get_fov_mask(points, p2, rect, velo_to_cam, image_size)

    output_bin.parent.mkdir(parents=True, exist_ok=True)
    points[keep].astype(np.float32).tofile(output_bin)
    if save_indices:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        np.flatnonzero(keep).astype(np.int32).tofile(index_path.with_suffix('.bin'))
    return split_name, frame_id, len(points), int(keep.sum())


def main():
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if source_root == output_root:
        raise ValueError('Source and output roots must be different')
    link_metadata(source_root, output_root)

    tasks = []
    for split_name in ('training', 'testing'):
        velodyne = source_root / split_name / 'velodyne'
        if not velodyne.exists():
            continue
        bins = sorted(velodyne.glob('*.bin'))
        if args.limit is not None:
            bins = bins[:args.limit]
        tasks.extend(
            (split_name, path, source_root, output_root, args.overwrite,
             args.save_indices) for path in bins)
    if not tasks:
        raise FileNotFoundError(f'No Velodyne files found under {source_root}')

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(process_frame, tasks), 1):
            rows.append(row)
            if index % 200 == 0 or index == len(tasks):
                print(f'Processed {index}/{len(tasks)} frames', flush=True)

    stats_path = output_root / 'fov_crop_stats.csv'
    with stats_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ['split', 'frame_id', 'input_points', 'fov_points', 'keep_ratio'])
        for split_name, frame_id, input_count, output_count in rows:
            ratio = output_count / input_count if input_count else 0
            writer.writerow(
                [split_name, frame_id, input_count, output_count, ratio])

    total_input = sum(row[2] for row in rows)
    total_output = sum(row[3] for row in rows)
    print(f'Input points: {total_input:,}')
    print(f'FOV points:   {total_output:,}')
    print(f'Keep ratio:   {total_output / total_input:.4%}')
    print(f'Output root:  {output_root}')


if __name__ == '__main__':
    main()
