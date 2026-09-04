#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

import numpy as np


path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open(newline='')))
losses = np.asarray([[float(row[f'L{i}_total_loss']) for i in range(6)] for row in rows])
retention = np.asarray([[float(row[f'L{i}_retention']) for i in range(6)] for row in rows])
payload = {
    'status': 'complete',
    'scenes': len(rows),
    'mean_absolute_loss_by_level': losses.mean(axis=0).tolist(),
    'median_absolute_loss_by_level': np.median(losses, axis=0).tolist(),
    'mean_retention_by_level': retention.mean(axis=0).tolist(),
    'median_retention_by_level': np.median(retention, axis=0).tolist(),
    'mean_per_scene_loss_span': np.ptp(losses, axis=1).mean(),
    'qsteps_mm_coarse_to_fine': [160, 80, 40, 20, 10, 5],
}
if len(rows) < 6 or not np.isfinite(losses).all():
    raise RuntimeError('Invalid quantization probe')
path.with_name('quant_probe_summary.json').write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
