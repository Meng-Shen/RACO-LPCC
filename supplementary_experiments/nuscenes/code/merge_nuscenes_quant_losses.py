#!/usr/bin/env python3
"""Merge four loss shards and create a deterministic 90/10 proxy split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard-root', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--train-split', required=True)
    parser.add_argument('--val-split', required=True)
    parser.add_argument('--val-percent', type=int, default=10)
    return parser.parse_args()


def stable_bucket(token: str) -> int:
    digest = hashlib.sha1(token.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'big') % 100


def main():
    args = parse_args()
    if not 1 <= args.val_percent <= 50:
        raise ValueError('--val-percent must be in [1, 50]')
    root = Path(args.shard_root).resolve()
    paths = sorted(root.glob('shard_*/loss.csv'))
    if not paths:
        raise FileNotFoundError(f'No loss shards below {root}')
    frames = [pd.read_csv(path, dtype={'scene_id': str, 'sample_idx': str})
              for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    if frame['dataset_index'].duplicated().any():
        raise ValueError('Duplicate dataset_index across loss shards')
    frame = frame.sort_values('dataset_index').reset_index(drop=True)
    expected = list(range(len(frame)))
    if frame['dataset_index'].astype(int).tolist() != expected:
        raise ValueError('Merged shards do not cover every training sample')
    if frame['scene_id'].duplicated().any():
        raise ValueError('Duplicate sample token in merged losses')
    output = Path(args.output_csv).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    val_mask = frame['scene_id'].map(stable_bucket) < args.val_percent
    # Keep both subsets non-empty even for a tiny smoke-test export.
    if len(frame) > 1 and (val_mask.all() or not val_mask.any()):
        val_mask.iloc[:] = False
        val_mask.iloc[::max(2, round(100 / args.val_percent))] = True
    train_tokens = frame.loc[~val_mask, 'scene_id'].tolist()
    val_tokens = frame.loc[val_mask, 'scene_id'].tolist()
    for path_arg, tokens in [
        (args.train_split, train_tokens), (args.val_split, val_tokens)
    ]:
        path = Path(path_arg).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(f'{token}\n' for token in tokens))
    summary = dict(
        shards=[str(path) for path in paths], total_samples=len(frame),
        proxy_train_samples=len(train_tokens), proxy_val_samples=len(val_tokens),
        split='stable SHA1 token bucket', val_percent=args.val_percent,
        output_csv=str(output), train_split=str(Path(args.train_split).resolve()),
        val_split=str(Path(args.val_split).resolve()))
    output.with_suffix('.merge.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
