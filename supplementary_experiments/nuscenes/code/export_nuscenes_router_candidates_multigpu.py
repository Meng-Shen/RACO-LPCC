#!/usr/bin/env python3
"""Export official-val routing decisions for every saved checkpoint in parallel."""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from pathlib import Path

import torch

from train_nuscenes_rate_aware_proxy_ddp import (
    RateAwareSparseProxy,
    export_predictions,
    flexible_load,
    make_loader,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    cli = parse_args()
    training_dir = Path(cli.training_dir).resolve()
    output_dir = Path(cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_args = json.loads((training_dir / "args.json").read_text())
    args = Namespace(**run_args)
    if not args.test_split or not args.test_loss_csv or not args.test_bpp_csv:
        raise ValueError("Training args do not contain a complete official-val split")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(int(args.seed) + rank)

    loader, dataset, _ = make_loader(
        args, args.test_split, args.test_loss_csv, args.test_bpp_csv, False
    )
    model = RateAwareSparseProxy(
        dataset.spatial_shape, args.feat_dim, dataset.mean_log_bpp
    ).to(device)
    lambdas = torch.tensor(args.lambdas, dtype=torch.float32, device=device)

    candidates = []
    init_candidate = training_dir / "candidate_init.pth"
    if init_candidate.is_file():
        candidates.append(init_candidate)
    candidates.extend(sorted((training_dir / "checkpoints").glob("epoch_*.pth")))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint candidates below {training_dir}")

    assigned = candidates[rank::world_size]
    print(
        f"rank={rank}/{world_size} gpu={local_rank} "
        f"assigned={[path.stem for path in assigned]}",
        flush=True,
    )
    for checkpoint_path in assigned:
        output_path = output_dir / f"{checkpoint_path.stem}.csv"
        if output_path.is_file() and output_path.stat().st_size > 0:
            print(f"skip existing {output_path}", flush=True)
            continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        flexible_load(model, state)
        export_predictions(model, loader, device, lambdas, args, output_path)
        print(
            f"exported checkpoint={checkpoint_path} predictions={output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
