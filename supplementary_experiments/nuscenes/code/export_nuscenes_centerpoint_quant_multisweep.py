#!/usr/bin/env python3
"""Export per-frame CenterPoint losses or predictions at six XYZ quantizers.

The detector input pipeline is derived from the selected config.  In
particular, a full nuScenes detector trained with historical sweeps is evaluated
with the same number of sweeps instead of silently falling back to a single
keyframe.  The script itself remains single-GPU so independent, safely
resumable shards can be distributed over all available GPUs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import mmengine
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import DATASETS, MODELS, TRANSFORMS


def parse_qsteps(text: str) -> list[float]:
    values = [float(item) for item in text.split(',') if item.strip()]
    if len(values) != 6 or any(value <= 0 for value in values):
        raise ValueError('--qsteps-mm must contain exactly six positive values')
    if any(a <= b for a, b in zip(values, values[1:])):
        raise ValueError('--qsteps-mm must be strictly descending')
    return values


@TRANSFORMS.register_module(force=True)
class NuScenesUniformQuantize(BaseTransform):
    """Quantize/deduplicate XYZ with the same convention as earlier tasks."""

    def __init__(self, qstep_mm: float):
        self.qstep_mm = float(qstep_mm)
        self.last_stats = None

    def transform(self, results: dict) -> dict:
        points = results['points']
        tensor = points.tensor.detach().cpu().numpy()
        xyz = tensor[:, :3].astype(np.float64, copy=False)
        raw_count = len(xyz)
        if raw_count == 0:
            self.last_stats = dict(
                raw_num_points=0, unique_num_points=0, retention=0.0)
            return results

        coords_mm = np.rint(xyz * 1000.0).astype(np.int64)
        offset_mm = coords_mm.min(axis=0)
        grid = np.rint(
            (coords_mm - offset_mm).astype(np.float64) / self.qstep_mm
        ).astype(np.int64)
        unique_grid = np.unique(grid, axis=0)
        decoded_xyz = (
            unique_grid.astype(np.float64) * self.qstep_mm + offset_mm
        ) / 1000.0
        results['points'] = points.new_point(decoded_xyz.astype(np.float32))
        self.last_stats = dict(
            raw_num_points=int(raw_count),
            unique_num_points=int(len(decoded_xyz)),
            retention=float(len(decoded_xyz) / raw_count),
        )
        return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('loss', 'predictions'), required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--qsteps-mm', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20260819)
    parser.add_argument('--log-every', type=int, default=20)
    return parser.parse_args()


def multisweep_settings(cfg):
    """Return the configured multi-sweep transform, if the detector uses one."""
    for pipeline_name in ('test_pipeline', 'eval_pipeline', 'train_pipeline'):
        for transform in cfg.get(pipeline_name, []):
            transform_type = transform.get('type')
            if transform_type == 'LoadPointsFromMultiSweeps':
                return dict(
                    sweeps_num=int(transform.get('sweeps_num', 9)),
                    load_dim=int(transform.get('load_dim', 5)),
                    use_dim=list(transform.get('use_dim', [0, 1, 2])),
                    pad_empty_sweeps=bool(
                        transform.get('pad_empty_sweeps', True)),
                    remove_close=bool(transform.get('remove_close', True)),
                )
    return None


def make_dataset_cfg(cfg, data_root: str, split: str, qstep_mm: float,
                     loss_mode: bool, sweep_cfg):
    root = str(Path(data_root).resolve()) + '/'
    common = dict(
        type='NuScenesDataset',
        data_root=root,
        ann_file=f'nuscenes_infos_{split}.pkl',
        metainfo=cfg.metainfo,
        modality=cfg.input_modality,
        data_prefix=cfg.data_prefix,
        box_type_3d='LiDAR',
        backend_args=None,
        use_valid_flag=True,
        filter_empty_gt=False,
    )
    # Load all five stored nuScenes fields before sweep accumulation because the
    # MMDet3D transform temporarily writes the time-lag field.  Its ``use_dim``
    # then removes intensity and time, leaving strict XYZ detector input.
    pipeline = [dict(
        type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5,
        use_dim=5 if sweep_cfg else [0, 1, 2], backend_args=None)]
    if sweep_cfg:
        pipeline.append(dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=sweep_cfg['sweeps_num'],
            load_dim=sweep_cfg['load_dim'],
            use_dim=sweep_cfg['use_dim'],
            pad_empty_sweeps=sweep_cfg['pad_empty_sweeps'],
            remove_close=sweep_cfg['remove_close'],
            # Loss labels must be reproducible, so always take the nearest
            # configured historical sweeps rather than a random subset.
            test_mode=True,
            backend_args=None))
    if loss_mode:
        pipeline.append(dict(
            type='LoadAnnotations3D', with_bbox_3d=True,
            with_label_3d=True))
    pipeline.append(dict(type=NuScenesUniformQuantize, qstep_mm=qstep_mm))
    pipeline.append(dict(
        type='PointsRangeFilter', point_cloud_range=cfg.point_cloud_range))
    if loss_mode:
        pipeline.extend([
            dict(type='ObjectRangeFilter',
                 point_cloud_range=cfg.point_cloud_range),
            dict(type='ObjectNameFilter', classes=cfg.class_names),
            dict(type='Pack3DDetInputs',
                 keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
        ])
    else:
        pipeline.append(dict(type='Pack3DDetInputs', keys=['points']))
    common.update(pipeline=pipeline, test_mode=not loss_mode)
    return common


def find_quantizer(dataset) -> NuScenesUniformQuantize:
    for transform in dataset.pipeline.transforms:
        if isinstance(transform, NuScenesUniformQuantize):
            return transform
    raise RuntimeError('NuScenesUniformQuantize is missing')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().item()
    return round(float(value), 8)


def sample_identity(dataset, index: int) -> tuple[str, str]:
    info = dataset.get_data_info(index)
    sample_idx = str(info.get('sample_idx', info.get('token', '')))
    if not sample_idx:
        raw = dataset.data_list[index]
        sample_idx = str(raw.get('sample_idx', raw.get('token', index)))
    lidar_path = str(info['lidar_points']['lidar_path'])
    path = Path(lidar_path)
    if not path.is_absolute():
        path = Path(dataset.data_root) / path
    return sample_idx, str(path.resolve())


def evaluate_loss(model, item: dict) -> dict[str, float]:
    batch = pseudo_collate([item])
    data = model.data_preprocessor(batch, training=False)
    with torch.no_grad():
        losses = model._run_forward(data, mode='loss')
        _, log_vars = model.parse_losses(losses)
    return {key: scalar(value) for key, value in log_vars.items()}


def prediction_to_cpu(result) -> dict:
    pred_3d = result.pred_instances_3d
    for key, value in list(pred_3d.items()):
        if hasattr(value, 'to'):
            pred_3d[key] = value.to('cpu')
    pred_2d = result.pred_instances
    for key, value in list(pred_2d.items()):
        if hasattr(value, 'to'):
            pred_2d[key] = value.to('cpu')
    return dict(
        pred_instances_3d=pred_3d,
        pred_instances=pred_2d,
        sample_idx=result.sample_idx,
    )


def write_csv(path: Path, rows: list[dict]):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError('Invalid shard specification')
    qsteps = parse_qsteps(args.qsteps_mm)
    output = Path(args.output).resolve()
    cfg = Config.fromfile(str(Path(args.config).resolve()))
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope('mmdet3d')
    loss_mode = args.mode == 'loss'
    sweep_cfg = multisweep_settings(cfg)
    if sweep_cfg:
        print(
            'Detector input: keyframe + '
            f'{sweep_cfg["sweeps_num"]} sweeps, retained dimensions='
            f'{sweep_cfg["use_dim"]}', flush=True)
    else:
        print('Detector input: single keyframe XYZ', flush=True)
    # One dataset is sufficient: changing the stateless quantizer's scale before
    # each fresh ``dataset[index]`` call avoids loading the 832 MB nuScenes info
    # file six times in every GPU shard.
    dataset = DATASETS.build(make_dataset_cfg(
        cfg, args.data_root, args.split, qsteps[0], loss_mode, sweep_cfg))
    dataset.full_init()
    quantizer = find_quantizer(dataset)
    total = len(dataset)
    indices = list(range(args.shard_id, total, args.num_shards))

    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(Path(args.checkpoint).resolve()),
                    map_location='cpu')
    model.to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows = []
    records = []
    begin = time.time()
    for ordinal, index in enumerate(indices, 1):
        sample_idx, lidar_path = sample_identity(dataset, index)
        if loss_mode:
            stats_by_level = []
            for qstep in qsteps:
                set_seed(args.seed + index)
                quantizer.qstep_mm = float(qstep)
                stats = evaluate_loss(model, dataset[index])
                if quantizer.last_stats is None:
                    raise RuntimeError('Quantizer did not report statistics')
                stats_by_level.append({**quantizer.last_stats, **stats})
            finest_loss = stats_by_level[-1]['loss']
            row = dict(
                scene_id=sample_idx, sample_idx=sample_idx,
                dataset_index=index, lidar_path=lidar_path,
                finest_label=5, finest_qstep_mm=qsteps[-1],
                finest_total_loss=finest_loss)
            for level, (qstep, stats) in enumerate(
                    zip(qsteps, stats_by_level)):
                delta = stats['loss'] - finest_loss
                row[f'L{level}_qstep_mm'] = qstep
                row[f'L{level}_raw_num_points'] = stats['raw_num_points']
                row[f'L{level}_unique_num_points'] = stats['unique_num_points']
                row[f'L{level}_retention'] = round(stats['retention'], 8)
                row[f'L{level}_total_loss'] = stats['loss']
                row[f'L{level}_loss_delta'] = round(delta, 8)
                row[f'L{level}_signed_delta'] = round(delta, 8)
                for key, value in stats.items():
                    if key not in {'raw_num_points', 'unique_num_points',
                                   'retention', 'loss'}:
                        row[f'L{level}_{key}'] = value
            rows.append(row)
            detail = ','.join(f'{x["loss"]:.3f}' for x in stats_by_level)
        else:
            predictions = []
            for qstep in qsteps:
                set_seed(args.seed + index)
                quantizer.qstep_mm = float(qstep)
                with torch.no_grad():
                    result = model.test_step(
                        pseudo_collate([dataset[index]]))[0]
                predictions.append(prediction_to_cpu(result))
            records.append(dict(
                scene_id=sample_idx, sample_idx=sample_idx,
                dataset_index=index, lidar_path=lidar_path,
                predictions=predictions))
            detail = 'predictions cached'
        if ordinal == 1 or ordinal % args.log_every == 0 or ordinal == len(indices):
            print(
                f'[{ordinal}/{len(indices)}] index={index} sample={sample_idx} '
                f'elapsed={time.time()-begin:.1f}s {detail}', flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    if loss_mode:
        write_csv(output, rows)
    else:
        mmengine.dump(records, output)
    manifest = dict(
        mode=args.mode, dataset='nuScenes', split=args.split,
        geometry_only=True,
        single_keyframe=sweep_cfg is None,
        sweeps_num=0 if sweep_cfg is None else sweep_cfg['sweeps_num'],
        config=str(Path(args.config).resolve()),
        checkpoint=str(Path(args.checkpoint).resolve()),
        qsteps_mm_coarse_to_fine=qsteps,
        estimated_bpp_by_level=[1, 2, 3, 4, 5, 6],
        baseline_label=5,
        loss_definition=(
            'candidate_total_loss - 64mm_total_loss' if loss_mode else None),
        shard_id=args.shard_id, num_shards=args.num_shards,
        dataset_size=total, num_samples=len(indices),
        elapsed_seconds=time.time() - begin,
        output=str(output))
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Output: {output}', flush=True)


if __name__ == '__main__':
    main()
