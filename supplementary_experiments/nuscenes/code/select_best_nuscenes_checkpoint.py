#!/usr/bin/env python3
"""Select the saved epoch with the highest official nuScenes validation mAP."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERN = re.compile(
    r'Epoch\(val\) \[(\d+)\].*?NuScenes metric/'
    r'pred_instances_3d_NuScenes/mAP:\s*([0-9.]+)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument(
        '--baseline-selection-json', default='',
        help='Optional selector JSON for the initialization checkpoint. The '
             'baseline competes with newly trained epochs by overall mAP.')
    args = parser.parse_args()
    root = Path(args.work_dir).resolve()
    scores = {}
    sources = {}
    for log_path in sorted(root.rglob('*.log')):
        text = log_path.read_text(errors='replace')
        for epoch_text, score_text in PATTERN.findall(text):
            epoch, score = int(epoch_text), float(score_text)
            if epoch not in scores or score > scores[epoch]:
                scores[epoch] = score
                sources[epoch] = str(log_path)
    available = {
        epoch: score for epoch, score in scores.items()
        if (root / f'epoch_{epoch}.pth').is_file()
    }
    if not available:
        raise RuntimeError(f'No validation mAP/checkpoint pair found below {root}')
    best_epoch = max(available, key=lambda epoch: (available[epoch], epoch))
    checkpoint = root / f'epoch_{best_epoch}.pth'
    payload = dict(
        best_epoch=best_epoch, best_mAP=available[best_epoch],
        checkpoint=str(checkpoint), source_log=sources[best_epoch],
        selected_source='continued_finetuning',
        all_saved_epoch_mAP={str(k): available[k] for k in sorted(available)})
    if args.baseline_selection_json:
        baseline_path = Path(args.baseline_selection_json).resolve()
        baseline = json.loads(baseline_path.read_text())
        baseline_checkpoint = Path(baseline['checkpoint']).resolve()
        baseline_score = float(baseline['best_mAP'])
        if not baseline_checkpoint.is_file():
            raise FileNotFoundError(baseline_checkpoint)
        payload['initialization_candidate'] = dict(
            checkpoint=str(baseline_checkpoint),
            overall_mAP=baseline_score,
            selection_json=str(baseline_path))
        if baseline_score >= float(payload['best_mAP']):
            payload.update(
                best_epoch=int(baseline['best_epoch']),
                best_mAP=baseline_score,
                checkpoint=str(baseline_checkpoint),
                source_log=baseline.get('source_log', ''),
                selected_source='epoch6_initialization')
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(payload['checkpoint'])


if __name__ == '__main__':
    main()
