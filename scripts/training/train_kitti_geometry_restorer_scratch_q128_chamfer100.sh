#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/DATA/sm/RACO-LPCC
CODE_ROOT=${ROOT}/reno/current_q_ones_coordinate_v1_20260831
OUT=${ROOT}/reno/current_q_ones_coordinate_runs_scratch_q128dist100_lr1e4_5ep_20260901
PYTHONPATH=${CODE_ROOT} \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OMP_NUM_THREADS=2 \
/home/sm/miniconda3/envs/SparsePCGC/bin/torchrun \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29750 \
  ${CODE_ROOT}/train_coordinate_residual.py \
  --output_dir ${OUT} \
  --epochs 5 \
  --train_q_steps 2048,1024,512,256,128 \
  --dist_q_multipliers 128:100 \
  --dist_weight 1 \
  --task_weight 1 \
  --shared_lr 1e-4 \
  --scale_lr 1e-4 \
  --weight_decay 1e-6 \
  --freeze_shared_epochs 0 \
  > ${OUT}/train.log 2>&1

date -Is > ${OUT}/TRAINING_COMPLETE
