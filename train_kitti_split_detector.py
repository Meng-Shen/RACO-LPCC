#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parent
OPENPCDET_DIR = ROOT_DIR / 'OpenPCDet'
OPENPCDET_TOOLS = OPENPCDET_DIR / 'tools'


CLASS_HEADERS = {
    'Car': r'Car\s+AP_R40@0\.70,\s*0\.70,\s*0\.70',
    'Pedestrian': r'Pedestrian\s+AP_R40@0\.50,\s*0\.50,\s*0\.50',
    'Cyclist': r'Cyclist\s+AP_R40@0\.50,\s*0\.50,\s*0\.50',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split KITTI train frames, train detector from scratch on the first part, and select the best validation checkpoint.'
    )
    parser.add_argument('--data-root', type=Path, default=OPENPCDET_DIR / 'data/kitti_fov')
    parser.add_argument('--source-split', default='train', help='Source ImageSets split to divide, usually train.')
    parser.add_argument('--ratio', default='5:1', help='Train:val count ratio, e.g. 5:1 or 0.8333.')
    parser.add_argument(
        '--extra-val-splits',
        default='val',
        help='Comma-separated existing ImageSets splits appended to validation, default: val.')
    parser.add_argument(
        '--no-extra-val-splits',
        action='store_true',
        help='Use only the held-out part of --source-split for validation.')
    parser.add_argument('--split-name', default=None, help='Name suffix for generated split/info files.')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle source ids before splitting.')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--workers', type=int, default=8, help='Workers for GT database creation.')
    parser.add_argument(
        '--template-cfg',
        type=Path,
        default=OPENPCDET_TOOLS / 'cfgs/kitti_models/pv_rcnn_fov_geometry.yaml')
    parser.add_argument('--cfg-out', type=Path, default=None)
    parser.add_argument('--python-bin', default='/home/sm/miniconda3/envs/SparsePCGC/bin/python')
    parser.add_argument('--run-train', action='store_true', help='Launch OpenPCDet training after preparation.')
    parser.add_argument('--cuda-visible-devices', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--train-workers', type=int, default=4)
    parser.add_argument('--extra-tag', default=None, help='OpenPCDet extra_tag. Defaults to a timestamped scratch tag.')
    parser.add_argument('--best-metric', choices=['mean', 'car', 'pedestrian', 'cyclist'], default='mean')
    return parser.parse_args()


def parse_ratio(text):
    text = str(text).strip()
    if ':' in text:
        left, right = text.split(':', 1)
        train = float(left)
        val = float(right)
        if train <= 0 or val <= 0:
            raise ValueError('--ratio parts must be positive')
        return train / (train + val)
    value = float(text)
    if not 0 < value < 1:
        raise ValueError('--ratio as a float must be between 0 and 1')
    return value


def read_ids(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def write_ids(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for frame_id in ids:
            f.write(f'{frame_id}\n')


def split_ids(ids, train_fraction, shuffle=False, seed=2026):
    ids = list(ids)
    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(ids)
    split_index = int(round(len(ids) * train_fraction))
    split_index = min(max(split_index, 1), len(ids) - 1)
    return ids[:split_index], ids[split_index:]


def unique_preserve_order(values):
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def read_extra_val_ids(image_sets, split_names):
    ids = []
    for split_name in split_names:
        split_name = split_name.strip()
        if not split_name:
            continue
        path = image_sets / f'{split_name}.txt'
        if not path.exists():
            raise FileNotFoundError(path)
        ids.extend(read_ids(path))
    return ids


def frame_id_from_info(info):
    frame_id = info['point_cloud']['lidar_idx']
    return str(frame_id).zfill(6)


def filter_infos(info_path, train_ids, val_ids, train_out, val_out):
    train_set = set(train_ids)
    val_set = set(val_ids)
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    train_infos = [info for info in infos if frame_id_from_info(info) in train_set]
    val_infos = [info for info in infos if frame_id_from_info(info) in val_set]

    missing_train = train_set - {frame_id_from_info(info) for info in train_infos}
    missing_val = val_set - {frame_id_from_info(info) for info in val_infos}
    if missing_train or missing_val:
        raise RuntimeError(
            f'Missing infos for {len(missing_train)} train and {len(missing_val)} val frames. '
            f'Check source info file: {info_path}')

    with open(train_out, 'wb') as f:
        pickle.dump(train_infos, f)
    with open(val_out, 'wb') as f:
        pickle.dump(val_infos, f)

    return len(train_infos), len(val_infos)


def add_openpcdet_to_path():
    for path in (OPENPCDET_DIR, OPENPCDET_TOOLS):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def build_gt_database(data_root, dataset_cfg, train_info_path, train_split_name):
    add_openpcdet_to_path()
    import pcdet.datasets.kitti.kitti_dataset as kitti_dataset
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset

    # This local OpenPCDet copy uses Path inside create_groundtruth_database
    # without importing it at module scope.
    kitti_dataset.Path = Path

    dataset = KittiDataset(
        dataset_cfg=dataset_cfg,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        root_path=data_root,
        training=False)
    dataset.create_groundtruth_database(train_info_path, split=train_split_name)


def edict_to_plain(value):
    if isinstance(value, dict):
        return {k: edict_to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [edict_to_plain(v) for v in value]
    return value


def load_openpcdet_cfg(cfg_path):
    add_openpcdet_to_path()
    from pcdet.config import cfg, cfg_from_yaml_file

    cfg.clear()
    cfg.ROOT_DIR = OPENPCDET_DIR.resolve()
    cfg.LOCAL_RANK = 0
    old_cwd = Path.cwd()
    os.chdir(OPENPCDET_TOOLS)
    try:
        cfg_from_yaml_file(str(cfg_path.relative_to(OPENPCDET_TOOLS)), cfg)
    finally:
        os.chdir(old_cwd)
    return cfg


def update_gt_sampling_db_path(data_config, dbinfo_name):
    augmentor = data_config.get('DATA_AUGMENTOR', {})
    for item in augmentor.get('AUG_CONFIG_LIST', []):
        if item.get('NAME') == 'gt_sampling':
            item['DB_INFO_PATH'] = [dbinfo_name]


def write_generated_cfg(template_cfg, cfg_out, data_root, train_split, val_split, train_info, val_info):
    cfg = load_openpcdet_cfg(template_cfg)
    cfg.DATA_CONFIG.DATA_PATH = os.path.relpath(data_root.resolve(), OPENPCDET_TOOLS)
    cfg.DATA_CONFIG.DATA_SPLIT = {'train': train_split, 'test': val_split}
    cfg.DATA_CONFIG.INFO_PATH = {
        'train': [train_info.name],
        'test': [val_info.name],
    }
    update_gt_sampling_db_path(cfg.DATA_CONFIG, f'kitti_dbinfos_{train_split}.pkl')

    cfg_out.parent.mkdir(parents=True, exist_ok=True)
    plain = edict_to_plain(cfg)
    plain.pop('ROOT_DIR', None)
    plain.pop('LOCAL_RANK', None)
    with open(cfg_out, 'w') as f:
        yaml.safe_dump(plain, f, sort_keys=False)


def parse_eval_log(log_path, metric='mean'):
    results = []
    current_epoch = None
    current_class = None
    current_scores = {}

    def flush():
        nonlocal current_epoch, current_scores
        if current_epoch is None or not current_scores:
            return
        values = {
            'car': current_scores.get('Car'),
            'pedestrian': current_scores.get('Pedestrian'),
            'cyclist': current_scores.get('Cyclist'),
        }
        valid = [v for v in values.values() if v is not None]
        if not valid:
            return
        score = sum(valid) / len(valid) if metric == 'mean' else values[metric]
        if score is not None:
            results.append((float(score), current_epoch, values))

    with open(log_path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            epoch_match = re.search(r'EPOCH\s+([0-9.]+)\s+EVALUATION', line)
            if epoch_match:
                flush()
                current_epoch = epoch_match.group(1)
                current_scores = {}
                current_class = None
                continue

            if current_epoch is None:
                continue

            matched_header = False
            for cls_name, pattern in CLASS_HEADERS.items():
                if re.search(pattern, line):
                    if cls_name not in current_scores:
                        current_class = cls_name
                    else:
                        current_class = None
                    matched_header = True
                    break
            if matched_header:
                continue

            if current_class and line.startswith('3d') and 'AP:' in line:
                values = [float(x) for x in re.findall(r'[0-9]+(?:\.[0-9]+)?', line.split('AP:', 1)[1])]
                if len(values) >= 3:
                    current_scores[current_class] = values[1]
                current_class = None

    flush()
    if not results:
        return None
    return max(results, key=lambda item: item[0])


def run_training(args, cfg_out, extra_tag):
    cfg_arg = str(cfg_out.relative_to(OPENPCDET_TOOLS))
    cfg_stem = cfg_out.stem
    output_dir = OPENPCDET_DIR / 'output' / 'kitti_models' / cfg_stem / extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = output_dir / f'launcher_{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}.log'

    cmd = [
        args.python_bin, 'train.py',
        '--cfg_file', cfg_arg,
        '--extra_tag', extra_tag,
        '--workers', str(args.train_workers),
        '--ckpt_save_interval', '1',
    ]
    if args.epochs is not None:
        cmd.extend(['--epochs', str(args.epochs), '--num_epochs_to_eval', str(args.epochs)])
        cmd.extend(['--max_ckpt_save_num', str(max(args.epochs, 30))])
    if args.batch_size is not None:
        cmd.extend(['--batch_size', str(args.batch_size)])

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

    print('[+] Launching training:')
    print('    cd', OPENPCDET_TOOLS)
    print('   ', ' '.join(cmd))
    print(f'[+] Saving launcher stdout/stderr to {launcher_log}')
    with open(launcher_log, 'w') as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=OPENPCDET_TOOLS,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='')
            log_file.write(line)
        ret = proc.wait()
    if ret != 0:
        print(f'[!] Training failed with exit code {ret}. Full output: {launcher_log}')
        train_logs = sorted(output_dir.glob('train_*.log'), key=lambda p: p.stat().st_mtime)
        if train_logs:
            print(f'[!] Latest OpenPCDet train log: {train_logs[-1]}')
        raise subprocess.CalledProcessError(ret, cmd)

    eval_dir = output_dir / 'eval' / 'eval_with_train'
    logs = sorted(eval_dir.glob('log_eval_*.txt'), key=lambda p: p.stat().st_mtime)
    if not logs:
        print(f'[!] Training finished, but no eval log found under {eval_dir}')
        return

    best = parse_eval_log(logs[-1], metric=args.best_metric)
    if best is None:
        print(f'[!] Could not parse best checkpoint from {logs[-1]}')
        return

    score, epoch, values = best
    ckpt = output_dir / 'ckpt' / f'checkpoint_epoch_{int(float(epoch))}.pth'
    summary = output_dir / 'best_val_checkpoint.txt'
    with open(summary, 'w') as f:
        f.write(f'best_metric={args.best_metric}\n')
        f.write(f'best_score={score:.6f}\n')
        f.write(f'best_epoch={epoch}\n')
        f.write(f'best_checkpoint={ckpt}\n')
        f.write(f'car={values.get("car")}\n')
        f.write(f'pedestrian={values.get("pedestrian")}\n')
        f.write(f'cyclist={values.get("cyclist")}\n')

    print('[+] Best validation checkpoint:')
    print(f'    epoch: {epoch}')
    print(f'    {args.best_metric}: {score:.4f}')
    print(f'    checkpoint: {ckpt}')
    print(f'    summary: {summary}')


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    image_sets = data_root / 'ImageSets'
    source_split_path = image_sets / f'{args.source_split}.txt'
    if not source_split_path.exists():
        raise FileNotFoundError(source_split_path)

    ratio_text = args.ratio.replace(':', 'to').replace('.', 'p')
    split_name = args.split_name or f'{args.source_split}_{ratio_text}_seed{args.seed}'
    train_split = f'{split_name}_train'
    val_split = f'{split_name}_val'

    ids = read_ids(source_split_path)
    train_ids, heldout_val_ids = split_ids(ids, parse_ratio(args.ratio), args.shuffle, args.seed)
    extra_val_splits = [] if args.no_extra_val_splits else [
        item.strip() for item in args.extra_val_splits.split(',') if item.strip()
    ]
    extra_val_ids = read_extra_val_ids(image_sets, extra_val_splits)
    val_ids = unique_preserve_order(heldout_val_ids + extra_val_ids)

    train_txt = image_sets / f'{train_split}.txt'
    val_txt = image_sets / f'{val_split}.txt'
    write_ids(train_txt, train_ids)
    write_ids(val_txt, val_ids)

    source_info = data_root / 'kitti_infos_trainval.pkl'
    if not source_info.exists():
        source_info = data_root / f'kitti_infos_{args.source_split}.pkl'
    if not source_info.exists():
        raise FileNotFoundError(f'No source info file found under {data_root}')

    train_info = data_root / f'kitti_infos_{train_split}.pkl'
    val_info = data_root / f'kitti_infos_{val_split}.pkl'
    train_count, val_count = filter_infos(source_info, train_ids, val_ids, train_info, val_info)

    cfg_for_db = load_openpcdet_cfg(args.template_cfg)
    build_gt_database(data_root, cfg_for_db.DATA_CONFIG, train_info, train_split)

    cfg_out = args.cfg_out
    if cfg_out is None:
        cfg_out = OPENPCDET_TOOLS / 'cfgs/kitti_models' / f'pv_rcnn_fov_geometry_{split_name}.yaml'
    cfg_out = cfg_out.resolve()
    write_generated_cfg(args.template_cfg.resolve(), cfg_out, data_root, train_split, val_split, train_info, val_info)

    print('[+] Split and config prepared')
    print(f'    source split: {source_split_path} ({len(ids)} frames)')
    print(f'    held-out val: {len(heldout_val_ids)} ids from {args.source_split}')
    if extra_val_splits:
        print(f'    extra val:    {len(extra_val_ids)} ids from {",".join(extra_val_splits)}')
    print(f'    train split:  {train_txt} ({len(train_ids)} ids, {train_count} infos)')
    print(f'    val split:    {val_txt} ({len(val_ids)} ids, {val_count} infos)')
    print(f'    train info:   {train_info}')
    print(f'    val info:     {val_info}')
    print(f'    config:       {cfg_out}')
    print(f'    dbinfo:       {data_root / ("kitti_dbinfos_%s.pkl" % train_split)}')

    if args.run_train:
        extra_tag = args.extra_tag or f'{split_name}_scratch_{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        run_training(args, cfg_out, extra_tag)
    else:
        print('[i] Preparation only. Add --run-train to start training from scratch.')


if __name__ == '__main__':
    main()
