#!/usr/bin/env python3
"""Drop the pretrained reflectance coefficient while retaining all XYZ weights."""

import argparse
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope('mmdet3d')
    target = MODELS.build(cfg.model).state_dict()
    checkpoint = torch.load(args.source, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    converted = {}
    changed = []
    unresolved = []
    for key, value in state.items():
        normalized = key[7:] if key.startswith('module.') else key
        if normalized not in target:
            converted[normalized] = value
            continue
        wanted = target[normalized]
        candidate = value
        if tuple(candidate.shape) != tuple(wanted.shape):
            # HardVFE decorates raw [x,y,z,intensity] with geometric offsets.
            # The XYZ model has exactly one fewer input coefficient; drop only
            # raw intensity (index 3), preserving every geometry coefficient.
            if (
                candidate.ndim == wanted.ndim == 2
                and candidate.shape[0] == wanted.shape[0]
                and candidate.shape[1] == wanted.shape[1] + 1
            ):
                candidate = torch.cat([candidate[:, :3], candidate[:, 4:]], dim=1)
            elif (
                candidate.ndim == wanted.ndim == 2
                and candidate.shape[1] == wanted.shape[1]
                and candidate.shape[0] == wanted.shape[0] + 1
            ):
                candidate = torch.cat([candidate[:3], candidate[4:]], dim=0)
            if tuple(candidate.shape) != tuple(wanted.shape):
                unresolved.append((normalized, tuple(value.shape), tuple(wanted.shape)))
            else:
                changed.append((normalized, tuple(value.shape), tuple(candidate.shape)))
        converted[normalized] = candidate
    if unresolved:
        raise RuntimeError(f'Unresolved parameter shapes: {unresolved[:10]}')

    missing = sorted(set(target) - set(converted))
    unexpected = sorted(set(converted) - set(target))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'meta': checkpoint.get('meta', {}),
        'state_dict': converted,
        'geometry_adaptation': {
            'source': str(Path(args.source).resolve()),
            'config': str(Path(args.config).resolve()),
            'changed_tensors': changed,
            'missing_model_keys': missing,
            'unexpected_checkpoint_keys': unexpected,
            'rule': 'drop raw reflectance coefficient index 3; preserve XYZ and geometric decorations',
        },
    }, output)
    print(f'Adapted checkpoint: {output}')
    print(f'Changed tensors: {changed}')
    print(f'Missing target keys: {len(missing)}; unexpected source keys: {len(unexpected)}')


if __name__ == '__main__':
    main()
