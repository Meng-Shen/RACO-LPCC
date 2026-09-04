#!/usr/bin/env python3
"""Python equivalent of the official SUN RGB-D extract_rgbd_data_v2.m."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import io as sio


def struct_list(value):
    if value is None:
        return []
    return list(np.asarray(value, dtype=object).reshape(-1))


def text(value) -> str:
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    return str(value)


def official_path(raw_root: Path, metadata_path: str) -> Path:
    relative = metadata_path[16:] if len(metadata_path) >= 16 else metadata_path
    return raw_root / relative.lstrip('/')


def extract_points(depth_path: Path, rgb_path: Path, intrinsic: np.ndarray,
                   rtilt: np.ndarray) -> np.ndarray:
    depth_raw = np.asarray(Image.open(depth_path), dtype=np.uint16)
    decoded = np.bitwise_or(
        np.right_shift(depth_raw, 3),
        np.left_shift(depth_raw, 13).astype(np.uint16),
    ).astype(np.float32) / 1000.0
    decoded[decoded > 8.0] = 8.0
    invalid = decoded == 0

    rgb = np.asarray(Image.open(rgb_path).convert('RGB'), dtype=np.float32) / 255.0
    if rgb.shape[:2] != decoded.shape:
        raise RuntimeError(
            f'RGB/depth shape mismatch: rgb={rgb.shape} depth={decoded.shape}')
    height, width = decoded.shape
    x, y = np.meshgrid(
        np.arange(1, width + 1, dtype=np.float32),
        np.arange(1, height + 1, dtype=np.float32),
    )
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    x3 = (x - cx) * decoded / fx
    y3 = (y - cy) * decoded / fy
    valid_f = (~invalid).ravel(order='F')
    points = np.column_stack([
        x3.ravel(order='F'), decoded.ravel(order='F'), -y3.ravel(order='F')
    ])[valid_f]
    points = (np.asarray(rtilt, dtype=np.float32) @ points.T).T
    colors = np.column_stack([
        rgb[:, :, channel].ravel(order='F') for channel in range(3)
    ])[valid_f]
    return np.concatenate([points, colors], axis=1).astype(np.float32)


def format_label(meta3, meta2) -> str:
    boxes3 = struct_list(getattr(meta3, 'groundtruth3DBB', None))
    boxes2 = struct_list(getattr(meta2, 'groundtruth2DBB', None))
    # The official MATLAB script iterates over 3D boxes.  The v2 metadata may
    # contain either extra 2D-only boxes or a few 3D boxes without a matching
    # 2D annotation, so matching must not assume equal list lengths.
    lines = []
    unused_2d = list(boxes2)
    for box3 in boxes3:
        classname = text(box3.classname)
        match = next(
            (index for index, candidate in enumerate(unused_2d)
             if text(candidate.classname) == classname), None)
        if match is None:
            # A small number of v2 3D boxes have no corresponding v2 2D box.
            # The geometry-only VoteNet pipeline never consumes bbox2d, but
            # the converter requires four placeholders in its label format.
            box = np.zeros(4, dtype=np.float64)
        else:
            box2 = unused_2d.pop(match)
            box = np.asarray(box2.gtBb2D, dtype=np.float64).reshape(-1)
        centroid = np.asarray(box3.centroid, dtype=np.float64).reshape(-1)
        coeffs = np.abs(np.asarray(box3.coeffs, dtype=np.float64).reshape(-1))
        orientation = np.asarray(box3.orientation, dtype=np.float64).reshape(-1)
        values = np.concatenate([box[:4], centroid[:3], coeffs[:3], orientation[:2]])
        lines.append(classname + ' ' + ' '.join(f'{value:.9g}' for value in values))
    return ''.join(line + '\n' for line in lines)


def complete(paths: list[Path]) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', required=True, type=Path)
    parser.add_argument('--meta3d', required=True, type=Path)
    parser.add_argument('--meta2d', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--max-scenes', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError('Invalid shard specification')

    meta3 = np.asarray(sio.loadmat(
        args.meta3d, squeeze_me=True, struct_as_record=False)['SUNRGBDMeta']).reshape(-1)
    meta2 = np.asarray(sio.loadmat(
        args.meta2d, squeeze_me=True, struct_as_record=False)['SUNRGBDMeta2DBB']).reshape(-1)
    if len(meta3) != 10335 or len(meta2) != 10335:
        raise RuntimeError(f'Unexpected metadata lengths: {len(meta3)} {len(meta2)}')
    for folder in ('depth', 'image', 'calib', 'label'):
        (args.output_root / folder).mkdir(parents=True, exist_ok=True)
    indices = list(range(args.shard_id, len(meta3), args.num_shards))
    if args.max_scenes > 0:
        indices = indices[:args.max_scenes]
    started, generated = time.time(), 0
    for ordinal, index in enumerate(indices, 1):
        image_id = index + 1
        stem = f'{image_id:06d}'
        outputs = [
            args.output_root / 'depth' / f'{stem}.mat',
            args.output_root / 'image' / f'{stem}.jpg',
            args.output_root / 'calib' / f'{stem}.txt',
            args.output_root / 'label' / f'{stem}.txt',
        ]
        if complete(outputs):
            continue
        item3, item2 = meta3[index], meta2[index]
        depth_path = official_path(args.raw_root, text(item3.depthpath))
        rgb_path = official_path(args.raw_root, text(item3.rgbpath))
        points = extract_points(
            depth_path, rgb_path,
            np.asarray(item3.K, dtype=np.float32),
            np.asarray(item3.Rtilt, dtype=np.float32),
        )
        sio.savemat(outputs[0], {'instance': points}, do_compression=True)
        shutil.copy2(rgb_path, outputs[1])
        with outputs[2].open('w') as handle:
            handle.write(' '.join(
                f'{value:.9g}' for value in np.asarray(item3.Rtilt).reshape(-1, order='F')) + '\n')
            handle.write(' '.join(
                f'{value:.9g}' for value in np.asarray(item3.K).reshape(-1, order='F')) + '\n')
        outputs[3].write_text(format_label(item3, item2))
        generated += 1
        if ordinal == 1 or ordinal % 50 == 0 or ordinal == len(indices):
            print(json.dumps({
                'shard': args.shard_id, 'visited': ordinal, 'assigned': len(indices),
                'generated': generated, 'image_id': image_id,
                'points': len(points), 'elapsed_seconds': time.time() - started,
            }), flush=True)


if __name__ == '__main__':
    main()
