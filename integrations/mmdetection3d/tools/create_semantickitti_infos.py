#!/usr/bin/env python3
"""Create the minimal MMDetection3D info files for SemanticKITTI."""

import argparse
import pickle
from pathlib import Path


DEFAULT_TRAIN_SEQUENCES = (
    '00', '01', '02', '03', '04', '05', '06', '07', '09', '10')
DEFAULT_VAL_SEQUENCES = ('08', )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument(
        '--train-sequences', nargs='+', default=DEFAULT_TRAIN_SEQUENCES)
    parser.add_argument(
        '--val-sequences', nargs='+', default=DEFAULT_VAL_SEQUENCES)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace existing semantickitti_infos_{train,val}.pkl files.')
    return parser.parse_args()


def find_sequences_root(dataset_root: Path):
    candidates = (
        (dataset_root / 'sequences', Path('sequences')),
        (dataset_root / 'dataset' / 'sequences', Path('dataset/sequences')),
    )
    for absolute_path, relative_path in candidates:
        if absolute_path.is_dir():
            return absolute_path, relative_path
    raise FileNotFoundError(
        f'Could not find sequences/ or dataset/sequences/ under '
        f'{dataset_root}')


def collect_sequence(sequence_root: Path, relative_root: Path,
                     sequence: str):
    velodyne_dir = sequence_root / sequence / 'velodyne'
    label_dir = sequence_root / sequence / 'labels'
    if not velodyne_dir.is_dir():
        raise FileNotFoundError(f'Missing velodyne directory: {velodyne_dir}')
    if not label_dir.is_dir():
        raise FileNotFoundError(f'Missing label directory: {label_dir}')

    records = []
    missing_labels = []
    for point_path in sorted(velodyne_dir.glob('*.bin')):
        label_path = label_dir / f'{point_path.stem}.label'
        if not label_path.is_file():
            missing_labels.append(label_path)
            continue
        relative_sequence = relative_root / sequence
        records.append({
            'sample_idx': f'{sequence}_{point_path.stem}',
            'lidar_points': {
                'lidar_path':
                (relative_sequence / 'velodyne' /
                 point_path.name).as_posix(),
                'num_pts_feats':
                4,
            },
            'pts_semantic_mask_path':
            (relative_sequence / 'labels' / label_path.name).as_posix(),
        })

    if missing_labels:
        preview = ', '.join(str(path) for path in missing_labels[:3])
        raise FileNotFoundError(
            f'{len(missing_labels)} point clouds have no label file; first: '
            f'{preview}')
    if not records:
        raise FileNotFoundError(
            f'No labeled .bin files found for sequence {sequence}')
    return records


def build_split(sequence_root: Path, relative_root: Path, sequences):
    records = []
    for sequence in sequences:
        sequence = str(sequence).zfill(2)
        sequence_records = collect_sequence(
            sequence_root, relative_root, sequence)
        records.extend(sequence_records)
        print(
            f'Sequence {sequence}: {len(sequence_records)} frames',
            flush=True)
    return records


def dump_info(path: Path, split_name: str, records, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(
            f'{path} already exists; pass --overwrite to replace it')
    payload = {
        'metainfo': {
            'dataset': 'SemanticKITTI',
            'split': split_name,
        },
        'data_list': records,
    }
    with path.open('wb') as handle:
        pickle.dump(payload, handle, protocol=4)
    print(f'Wrote {len(records)} samples to {path}', flush=True)


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    sequence_root, relative_root = find_sequences_root(dataset_root)
    train_records = build_split(
        sequence_root, relative_root, args.train_sequences)
    val_records = build_split(
        sequence_root, relative_root, args.val_sequences)
    dump_info(
        dataset_root / 'semantickitti_infos_train.pkl',
        'train',
        train_records,
        args.overwrite)
    dump_info(
        dataset_root / 'semantickitti_infos_val.pkl',
        'val',
        val_records,
        args.overwrite)


if __name__ == '__main__':
    main()

