#!/usr/bin/env python3
"""Octave-free replacement for the official SUN RGB-D split MATLAB script."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

import numpy as np
from scipy import io as sio


def matlab_strings(value) -> list[str]:
    output = []
    for item in np.asarray(value, dtype=object).reshape(-1):
        while isinstance(item, np.ndarray) and item.size == 1:
            item = item.reshape(-1)[0]
        output.append(str(item))
    return output


def strip_official_prefix(path: str) -> str:
    # Official paths begin with /n/fs/sun3d/data (16 characters).
    relative = path[16:] if len(path) >= 16 else path
    return relative.rstrip('/')


def metadata_scene_dir(path: str) -> str:
    # Metadata points to /scene/depth/file; the split already stores /scene.
    return str(PurePosixPath(strip_official_prefix(path)).parent.parent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--toolbox', required=True, type=Path)
    parser.add_argument('--meta3d', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()

    split = sio.loadmat(
        args.toolbox / 'traintestSUNRGBD' / 'allsplit.mat',
        squeeze_me=True, struct_as_record=False)
    train_dirs = {strip_official_prefix(path) for path in matlab_strings(split['alltrain'])}
    val_dirs = {strip_official_prefix(path) for path in matlab_strings(split['alltest'])}
    metadata = sio.loadmat(
        args.meta3d, squeeze_me=True, struct_as_record=False)['SUNRGBDMeta']

    train_ids, val_ids, unknown = [], [], []
    for image_id, item in enumerate(np.asarray(metadata).reshape(-1), 1):
        scene_dir = metadata_scene_dir(str(item.depthpath))
        if scene_dir in train_dirs:
            train_ids.append(image_id)
        elif scene_dir in val_dirs:
            val_ids.append(image_id)
        else:
            unknown.append((image_id, scene_dir))
    if len(train_ids) != 5285 or len(val_ids) != 5050 or unknown:
        raise RuntimeError(
            f'Unexpected split: train={len(train_ids)} val={len(val_ids)} '
            f'unknown={unknown[:5]} train_dir_sample={sorted(train_dirs)[:3]} '
            f'val_dir_sample={sorted(val_dirs)[:3]}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'train_data_idx.txt').write_text(
        ''.join(f'{index}\n' for index in train_ids))
    (args.output_dir / 'val_data_idx.txt').write_text(
        ''.join(f'{index}\n' for index in val_ids))
    print(f'train={len(train_ids)} val={len(val_ids)}')


if __name__ == '__main__':
    main()
