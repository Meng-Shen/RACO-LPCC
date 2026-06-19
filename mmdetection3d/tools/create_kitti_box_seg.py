#!/usr/bin/env python3
"""Create weak foreground/background point labels from KITTI 3D boxes."""

import argparse
import pickle
import random
import struct
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kitti-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--source-split', default='train')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument(
        '--foreground-classes',
        nargs='+',
        default=['Car', 'Pedestrian', 'Cyclist'])
    parser.add_argument(
        '--include-outside-fov',
        action='store_true',
        help='Label points outside the camera FOV as background instead of ignore.')
    parser.add_argument(
        '--overwrite', action='store_true', help='Regenerate existing masks.')
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Only process the first N samples (for smoke tests).')
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


def read_boxes(path, foreground_classes):
    boxes = []
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0] not in foreground_classes:
                continue
            h, w, length = map(float, fields[8:11])
            x, y, z = map(float, fields[11:14])
            boxes.append((x, y, z, h, w, length, float(fields[14])))
    return boxes


def points_in_camera_fov(points_rect, p2, image_size):
    points_h = np.column_stack(
        [points_rect, np.ones(points_rect.shape[0], dtype=points_rect.dtype)])
    projected = points_h @ p2.T
    depth = projected[:, 2] - p2[2, 3]
    valid_depth = depth > 0
    u = np.full_like(depth, -1)
    v = np.full_like(depth, -1)
    u[valid_depth] = projected[valid_depth, 0] / points_rect[valid_depth, 2]
    v[valid_depth] = projected[valid_depth, 1] / points_rect[valid_depth, 2]
    width, height = image_size
    return valid_depth & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def points_in_box(points_rect, box):
    x, y, z, height, width, length, rotation_y = box
    relative = points_rect - np.array([x, y, z], dtype=points_rect.dtype)
    cosine = np.cos(rotation_y)
    sine = np.sin(rotation_y)
    local_x = cosine * relative[:, 0] - sine * relative[:, 2]
    local_z = sine * relative[:, 0] + cosine * relative[:, 2]
    return (
        (np.abs(local_x) <= length / 2)
        & (relative[:, 1] <= 0)
        & (relative[:, 1] >= -height)
        & (np.abs(local_z) <= width / 2))


def make_info(sample_id):
    return {
        'sample_idx': sample_id,
        'lidar_points': {
            'lidar_path': f'training/velodyne/{sample_id}.bin',
            'num_pts_feats': 4,
        },
        'pts_semantic_mask_path': f'box_seg_labels/{sample_id}.label',
    }


def dump_info(path, sample_ids):
    data = {
        'metainfo': {
            'dataset': 'KITTI box-supervised foreground segmentation',
            'classes': ('background', 'foreground'),
        },
        'data_list': [make_info(sample_id) for sample_id in sample_ids],
    }
    with path.open('wb') as handle:
        pickle.dump(data, handle, protocol=4)


def main():
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError('--val-ratio must be between 0 and 1')

    split_path = args.kitti_root / 'ImageSets' / f'{args.source_split}.txt'
    sample_ids = [
        line.strip() for line in split_path.read_text().splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        sample_ids = sample_ids[:args.limit]
    if len(sample_ids) < 2:
        raise ValueError('At least two samples are required for train/val split')

    labels_dir = args.output_root / 'box_seg_labels'
    labels_dir.mkdir(parents=True, exist_ok=True)
    foreground_classes = set(args.foreground_classes)
    totals = np.zeros(3, dtype=np.int64)

    for index, sample_id in enumerate(sample_ids, 1):
        mask_path = labels_dir / f'{sample_id}.label'
        if mask_path.exists() and not args.overwrite:
            mask = np.fromfile(mask_path, dtype=np.uint8)
        else:
            points = np.fromfile(
                args.kitti_root / 'training' / 'velodyne' /
                f'{sample_id}.bin',
                dtype=np.float32).reshape(-1, 4)
            p2, rect, velo_to_cam = read_calib(
                args.kitti_root / 'training' / 'calib' / f'{sample_id}.txt')
            points_h = np.column_stack([
                points[:, :3],
                np.ones(points.shape[0], dtype=points.dtype)
            ])
            points_rect = points_h @ velo_to_cam.T @ rect.T

            if args.include_outside_fov:
                mask = np.zeros(points.shape[0], dtype=np.uint8)
            else:
                width, height = read_png_size(
                    args.kitti_root / 'training' / 'image_2' /
                    f'{sample_id}.png')
                in_fov = points_in_camera_fov(
                    points_rect, p2, (width, height))
                mask = np.full(points.shape[0], 2, dtype=np.uint8)
                mask[in_fov] = 0

            boxes = read_boxes(
                args.kitti_root / 'training' / 'label_2' /
                f'{sample_id}.txt', foreground_classes)
            for box in boxes:
                in_box = points_in_box(points_rect, box)
                if not args.include_outside_fov:
                    in_box &= mask != 2
                mask[in_box] = 1
            mask.tofile(mask_path)

        totals += np.bincount(mask, minlength=3)[:3]
        if index % 200 == 0 or index == len(sample_ids):
            print(f'Generated {index}/{len(sample_ids)} masks', flush=True)

    shuffled = sample_ids.copy()
    random.Random(args.seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * args.val_ratio))
    val_ids = sorted(shuffled[:val_count])
    train_ids = sorted(shuffled[val_count:])
    dump_info(args.output_root / 'kitti_box_seg_infos_train.pkl', train_ids)
    dump_info(args.output_root / 'kitti_box_seg_infos_val.pkl', val_ids)
    (args.output_root / 'box_seg_train.txt').write_text(
        '\n'.join(train_ids) + '\n')
    (args.output_root / 'box_seg_val.txt').write_text(
        '\n'.join(val_ids) + '\n')

    print(f'Train/val samples: {len(train_ids)}/{len(val_ids)}')
    for name, count in zip(('background', 'foreground', 'ignore'), totals):
        print(f'{name}: {count:,} points')


if __name__ == '__main__':
    main()
