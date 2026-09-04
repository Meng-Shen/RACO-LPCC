#!/usr/bin/env python3
"""Write three disjoint index shards covering all prepared samples."""

import argparse
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--points", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
count = len(np.load(args.points, mmap_mode="r"))
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)
for index, shard in enumerate(np.array_split(np.arange(count, dtype=np.int64), 3)):
    np.save(output / f"all_indices_shard{index}.npy", shard)
print(f"samples={count} shard_sizes={[len(x) for x in np.array_split(np.arange(count), 3)]}")
