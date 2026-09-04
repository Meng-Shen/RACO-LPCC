#!/usr/bin/env python3
"""Export official-val routing decisions for all full/Lite-S3 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from pathlib import Path

import torch

from train_nuscenes_sixloss_monotonic_router_ddp import (
    build_model,
    export_predictions,
    make_loader,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    cli = parser.parse_args()
    training_dir = Path(cli.training_dir).resolve()
    output_dir = Path(cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = Namespace(**json.loads((training_dir / "args.json").read_text()))

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(args.seed, rank)

    loader, dataset, _ = make_loader(
        args,
        args.test_split,
        args.test_loss_csv,
        args.test_bpp_csv,
        False,
    )
    model = build_model(
        args.model_variant,
        dataset.spatial_shape,
        args.feat_dim,
        args.loss_scales,
        args.mean_log_bpp,
    ).to(device)
    lambdas = torch.tensor(args.lambdas, dtype=torch.float32, device=device)
    candidates = sorted((training_dir / "checkpoints").glob("epoch_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints below {training_dir}")
    assigned = candidates[rank::world]
    print(
        f"rank={rank}/{world} gpu={local_rank} assigned={[path.stem for path in assigned]}",
        flush=True,
    )
    for checkpoint_path in assigned:
        output_path = output_dir / f"{checkpoint_path.stem}.csv"
        if output_path.is_file() and output_path.stat().st_size > 0:
            print(f"skip existing {output_path}", flush=True)
            continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint["model"]
        state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        export_predictions(model, loader, device, lambdas, args, output_path)
        print(f"exported {checkpoint_path.name} -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
