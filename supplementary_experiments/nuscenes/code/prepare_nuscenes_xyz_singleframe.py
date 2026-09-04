#!/usr/bin/env python3
"""Prepare nuScenes train/val infos and an XYZ-only single-frame GT DB."""

from __future__ import annotations

import argparse
import os
import pickle
from multiprocessing.pool import ThreadPool
from pathlib import Path

import mmengine
import torch.multiprocessing
from mmengine import ProgressBar, print_log, track_iter_progress

from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules
from tools.dataset_converters import nuscenes_converter
from tools.dataset_converters.create_gt_database import GTDatabaseCreater
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos


_GT_DB_CREATOR = None


def create_gt_record_from_index(index: int):
    """Pool worker entry point carrying only an integer through IPC.

    Linux fork workers inherit the read-only dataset and creator.  Building
    each data dictionary inside its worker avoids serializing thousands of
    torch-backed box objects through multiprocessing's eager task feeder.
    """
    creator = _GT_DB_CREATOR
    if creator is None:
        raise RuntimeError('GT database worker was not initialized')
    input_dict = creator.dataset.get_data_info(index)
    input_dict['box_type_3d'] = creator.dataset.box_type_3d
    input_dict['box_mode_3d'] = creator.dataset.box_mode_3d
    return creator.create_single(input_dict)


def info_is_v2(path: Path) -> bool:
    """Return whether an info pickle already uses MMDet3D's v2 schema."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    payload = mmengine.load(path)
    return isinstance(payload, dict) and 'data_list' in payload


def update_infos_with_real_root(root: Path, paths: list[Path]):
    """Work around MMDet3D 1.4's hard-coded ``./data/nuscenes`` root.

    ``update_nuscenes_infos`` ignores its pkl/out paths when constructing the
    nuScenes SDK and always resolves tables below the process CWD.  Use a tiny
    compatibility working directory inside the dataset root, with a symlink
    pointing back to the actual dataset.  No dataset file is copied or moved.
    """
    compat = root / '.mmdet3d_update_compat'
    data_dir = compat / 'data'
    link = data_dir / 'nuscenes'
    data_dir.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != root:
            raise RuntimeError(f'Unexpected compatibility path: {link}')
    else:
        link.symlink_to(root, target_is_directory=True)
    previous = Path.cwd()
    os.chdir(compat)
    try:
        for path in paths:
            if not info_is_v2(path):
                update_pkl_infos(
                    'nuscenes', out_dir=str(root), pkl_path=str(path))
    finally:
        os.chdir(previous)


class XYZSingleFrameGTDatabaseCreater(GTDatabaseCreater):
    """Use only XYZ from the annotated keyframe when creating DB samples."""

    def create(self):
        print_log(
            'Create XYZ single-frame GT Database of NuScenesDataset',
            logger='current')
        dataset_cfg = dict(
            type='NuScenesDataset',
            data_root=self.data_path,
            ann_file=self.info_path,
            use_valid_flag=True,
            data_prefix=dict(
                pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=[0, 1, 2]),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True)
            ])
        self.dataset = DATASETS.build(dataset_cfg)
        self.pipeline = self.dataset.pipeline
        self.database_save_path = str(
            Path(self.data_path) / f'{self.info_prefix}_gt_database')
        self.db_info_save_path = str(
            Path(self.data_path) / f'{self.info_prefix}_dbinfos_train.pkl')
        mmengine.mkdir_or_exist(self.database_save_path)

        indexed = (range(len(self.dataset)), len(self.dataset))
        global _GT_DB_CREATOR
        _GT_DB_CREATOR = self
        try:
            if self.num_worker == 0:
                multi_db_infos = mmengine.track_progress(
                    create_gt_record_from_index, indexed)
            else:
                pool = ThreadPool(self.num_worker)
                progress = ProgressBar(len(self.dataset))
                multi_db_infos = []
                try:
                    for result in pool.imap_unordered(
                            create_gt_record_from_index,
                            range(len(self.dataset)),
                            chunksize=1):
                        multi_db_infos.append(result)
                        progress.update()
                finally:
                    pool.close()
                    pool.join()
        finally:
            _GT_DB_CREATOR = None

        print_log('Make global unique group id', logger='current')
        group_counter_offset = 0
        all_db_infos = {}
        for single_db_infos in track_iter_progress(multi_db_infos):
            group_id = -1
            for name, name_db_infos in single_db_infos.items():
                for db_info in name_db_infos:
                    group_id = max(group_id, db_info['group_id'])
                    db_info['group_id'] += group_counter_offset
                all_db_infos.setdefault(name, []).extend(name_db_infos)
            group_counter_offset += group_id + 1

        for name, infos in all_db_infos.items():
            print_log(f'load {len(infos)} {name} database infos')
        print_log(f'Saving GT database infos into {self.db_info_save_path}')
        with open(self.db_info_save_path, 'wb') as handle:
            pickle.dump(all_db_infos, handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-path', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--info-prefix', default='nuscenes')
    parser.add_argument('--db-prefix', default='nuscenes_xyz')
    args = parser.parse_args()

    register_all_modules(init_default_scope=True)
    # The GT database task dictionaries contain torch-backed 3D boxes.  The
    # default file-descriptor sharing strategy lets Pool.imap's eager feeder
    # hold one descriptor per queued sample and exceeds the usual 1024 limit
    # before workers can consume 28k nuScenes tasks.  File-system sharing keeps
    # the same parallel implementation without exhausting process descriptors.
    torch.multiprocessing.set_sharing_strategy('file_system')
    root = args.root_path.resolve()
    train_info = root / f'{args.info_prefix}_infos_train.pkl'
    val_info = root / f'{args.info_prefix}_infos_val.pkl'

    if not train_info.is_file() or not val_info.is_file():
        print_log('Creating train/val infos with max_sweeps=0')
        nuscenes_converter.create_nuscenes_infos(
            str(root),
            args.info_prefix,
            version='v1.0-trainval',
            max_sweeps=0)
    else:
        print_log('Train/val source infos already exist')

    if not info_is_v2(train_info) or not info_is_v2(val_info):
        print_log('Updating train/val infos to MMDet3D v2 schema')
        update_infos_with_real_root(root, [train_info, val_info])
    else:
        print_log('Train/val infos already use MMDet3D v2 schema')

    db_info = root / f'{args.db_prefix}_dbinfos_train.pkl'
    if not db_info.is_file() or db_info.stat().st_size == 0:
        creator = XYZSingleFrameGTDatabaseCreater(
            dataset_class_name='NuScenesDataset',
            data_path=str(root),
            info_prefix=args.db_prefix,
            info_path=train_info.name,
            num_worker=args.workers)
        creator.create()
    else:
        print_log('XYZ GT database already exists; skipping creation')

    assert train_info.stat().st_size > 0
    assert val_info.stat().st_size > 0
    assert db_info.stat().st_size > 0
    marker = root / '.xyz_singleframe_prepared'
    marker.write_text('ready\n', encoding='utf-8')
    print_log(f'Preparation complete; marker={marker}')


if __name__ == '__main__':
    main()
