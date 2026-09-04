#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/public/DATA/sm/RACO-LPCC"
ROOT="${PROJECT_ROOT}/OpenPCDet"
TOOLS="${ROOT}/tools"
LAUNCHER="${PROJECT_ROOT}/integrations/openpcdet/run_tool.py"
PYTHON="/home/sm/miniconda3/envs/SparsePCGC/bin/python"
CFG="../../integrations/openpcdet/configs/kitti_models/pointpillar_fov_geometry.yaml"
PRETRAINED="ckpt/pointpillar_7728.pth"
TAG="pointpillar_geometry_from_7728_ddp8"
OUT="${ROOT}/output/kitti_models/pointpillar_fov_geometry/${TAG}"
MASTER_PORT="${MASTER_PORT:-29641}"

mkdir -p "$OUT"
exec > >(tee -a "${OUT}/launch.log") 2>&1

cd "$TOOLS"
exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  OMP_NUM_THREADS=2 \
  "$PYTHON" -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port="$MASTER_PORT" \
    "$LAUNCHER" train.py \
      --launcher pytorch \
      --tcp_port "$MASTER_PORT" \
      --cfg_file "$CFG" \
      --pretrained_model "$PRETRAINED" \
      --extra_tag "$TAG" \
      --batch_size 32 \
      --epochs 40 \
      --workers 2 \
      --use_amp \
      --fix_random_seed \
      --logger_iter_interval 20 \
      --ckpt_save_interval 1 \
      --max_ckpt_save_num 40 \
      --num_epochs_to_eval 0 \
      --set OPTIMIZATION.LR 0.001
