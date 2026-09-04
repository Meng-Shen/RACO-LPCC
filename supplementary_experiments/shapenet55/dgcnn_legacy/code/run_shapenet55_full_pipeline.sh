#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${RACO_SHAPENET55_DGCNN_ROOT:-/home/sm/raco_rate_aware_shapenet55_dgcnn_20260825}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$SCRIPT_DIR"
DATA="$ROOT/data"
CLASSIFIER="$ROOT/classifier"
ARTIFACTS="$ROOT/artifacts"
ROUTER="$ROOT/router"
RESULTS="$ROOT/results"
ARCHIVE=/home/sm/datasets/ShapeNet55-34_official/ShapeNet55.zip
SOURCE_DATA=/home/sm/datasets/ShapeNet55-34_official/ShapeNet55
SPLIT_SOURCE=/home/sm/datasets/ShapeNet55_label_recovery/ShapeNet-55
PYTHON=/home/sm/miniconda3/envs/openmmlab/bin/python
TMC3=/home/sm/raco_rate_aware_modelnet40_dgcnn_20260824/tmc3_v22
PRETRAINED=/home/sm/raco_rate_aware_modelnet40_dgcnn_20260824/checkpoints/dgcnn_modelnet40_1024_upstream.t7
LOG="$ROOT/pipeline.log"
STATUS="$ROOT/STATUS.txt"

mkdir -p "$ROOT" "$DATA" "$CLASSIFIER" "$ARTIFACTS" "$ROUTER" "$RESULTS"
exec 9>"$ROOT/.pipeline.lock"
if ! flock -n 9; then
    echo "ShapeNet55 pipeline is already active"
    exit 0
fi
exec >>"$LOG" 2>&1

record() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee "$STATUS"
}
fail() {
    local code=$?
    record "FAILED exit=$code command=$BASH_COMMAND"
    touch "$ROOT/PIPELINE_FAILED"
    exit "$code"
}
trap fail ERR
rm -f "$ROOT/PIPELINE_FAILED"

record "stage=archive_validation GPU=none"
[[ -s "$ARCHIVE" ]]
archive_bytes=$(stat -c %s "$ARCHIVE")
[[ "$archive_bytes" -eq 8367547065 ]]
archive_sha256=$(sha256sum "$ARCHIVE" | awk '{print $1}')
[[ "$archive_sha256" == '70774799990cbdc02bb7bfe9c78af940d418f404c58d9093e548222e2f01d0c1' ]]
if [[ ! -s "$SOURCE_DATA/.EXTRACTION_COMPLETE" ]]; then
    record "stage=extract archive_bytes=$archive_bytes GPU=none"
    unzip -q "$ARCHIVE" -d /home/sm/datasets/ShapeNet55-34_official
    materialized=$(find "$SOURCE_DATA/shapenet_pc" -maxdepth 1 -type f -name '*.npy' | wc -l)
    [[ "$materialized" -eq 52470 ]]
    mkdir -p "$SOURCE_DATA/ShapeNet-55"
    cp "$SPLIT_SOURCE/train.txt" "$SOURCE_DATA/ShapeNet-55/train.txt"
    cp "$SPLIT_SOURCE/test.txt" "$SOURCE_DATA/ShapeNet-55/test.txt"
    touch "$SOURCE_DATA/.EXTRACTION_COMPLETE"
fi
materialized=$(find "$SOURCE_DATA/shapenet_pc" -maxdepth 1 -type f -name '*.npy' | wc -l)
[[ "$materialized" -eq 52470 ]]
record "stage=prepare labelled_files=$materialized/52470 GPU=none"

if [[ ! -s "$DATA/manifest.json" ]]; then
    "$PYTHON" "$CODE/prepare_shapenet55.py" \
        --search-root "$SOURCE_DATA" \
        --output-dir "$DATA" \
        --num-points 1024 \
        --validation-fraction 0.10 \
        --seed 20260825
fi

record "stage=classifier_finetune GPUs=0,1,2 maximum_GPU_count=3"
if [[ ! -s "$CLASSIFIER/TRAINING_COMPLETE.json" ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2 "$PYTHON" -m torch.distributed.run \
        --nproc_per_node=3 --master_port=29655 \
        "$CODE/finetune_dgcnn_shapenet55_ddp.py" \
        --source-dir "$CODE/dgcnn.pytorch" \
        --pretrained "$PRETRAINED" \
        --data-dir "$DATA" \
        --output-dir "$CLASSIFIER" \
        --classes 55 --epochs 60 --patience 10 \
        --batch-size 48 --workers 4
fi

record "stage=qstep_probe GPU=0"
CANDIDATES='0.40,0.32,0.28,0.24,0.20,0.16,0.14,0.12,0.10,0.08,0.06,0.05,0.04,0.03,0.02,0.01'
if [[ ! -s "$ARTIFACTS/val_qstep_probe.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$CODE/eval_dgcnn_shapenet55_quant.py" \
        --source-dir "$CODE/dgcnn.pytorch" \
        --checkpoint "$CLASSIFIER/best.pth" \
        --data-dir "$DATA" \
        --indices "$DATA/model_val_indices.npy" \
        --qsteps "$CANDIDATES" \
        --output "$ARTIFACTS/val_qstep_probe.npz" \
        --batch-size 32 --workers 6
fi
"$PYTHON" "$CODE/select_shapenet55_qsteps.py" \
    --probe-json "$ARTIFACTS/val_qstep_probe.json" \
    --output "$ARTIFACTS/selected_qsteps.json"
QSTEPS=$("$PYTHON" -c "import json; print(','.join(str(x) for x in json.load(open('$ARTIFACTS/selected_qsteps.json'))['qsteps_coarse_to_fine']))")
record "stage=labels_and_bpp qsteps=$QSTEPS GPUs=0,1,2 plus CPU_G_PCC"

"$PYTHON" "$CODE/make_index_shards.py" \
    --points "$DATA/all_points.npy" --output-dir "$ARTIFACTS"
sample_count=$("$PYTHON" -c "import numpy as np; print(len(np.load('$DATA/all_points.npy', mmap_mode='r')))" )

declare -a eval_pids=()
for gpu in 0 1 2; do
    if [[ ! -s "$ARTIFACTS/quant_shard${gpu}.npz" ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$CODE/eval_dgcnn_shapenet55_quant.py" \
            --source-dir "$CODE/dgcnn.pytorch" \
            --checkpoint "$CLASSIFIER/best.pth" \
            --data-dir "$DATA" \
            --indices "$ARTIFACTS/all_indices_shard${gpu}.npy" \
            --qsteps "$QSTEPS" \
            --output "$ARTIFACTS/quant_shard${gpu}.npz" \
            --batch-size 32 --workers 5 \
            >"$ARTIFACTS/quant_shard${gpu}.log" 2>&1 &
        eval_pids+=("$!")
    fi
done

gpcc_pid=''
if [[ ! -s "$ARTIFACTS/all_bpp.manifest.json" ]]; then
    "$PYTHON" "$CODE/measure_shapenet55_gpcc.py" \
        --points "$DATA/all_points.npy" \
        --tmc3 "$TMC3" \
        --qsteps "$QSTEPS" \
        --output "$ARTIFACTS/all_bpp.csv" \
        --workers 24 --timeout 60 --log-every 250 \
        >"$ARTIFACTS/all_bpp.log" 2>&1 &
    gpcc_pid=$!
fi

for pid in "${eval_pids[@]}"; do
    wait "$pid"
done
if [[ -n "$gpcc_pid" ]]; then
    wait "$gpcc_pid"
fi

"$PYTHON" "$CODE/merge_quant_shards.py" \
    --shards "$ARTIFACTS/quant_shard0.npz" "$ARTIFACTS/quant_shard1.npz" "$ARTIFACTS/quant_shard2.npz" \
    --output "$ARTIFACTS/all_quant_labels.npz" \
    --expected-samples "$sample_count"

record "stage=lambda_selection train_only GPU=none"
"$PYTHON" "$CODE/select_shapenet55_lambdas.py" \
    --quant-npz "$ARTIFACTS/all_quant_labels.npz" \
    --bpp-csv "$ARTIFACTS/all_bpp.csv" \
    --train-indices "$DATA/router_train_indices.npy" \
    --output "$ARTIFACTS/rd_lambdas_train_only.json"

record "stage=router_training GPU=0 maximum_GPU_count=1"
if [[ ! -s "$ROUTER/TRAINING_COMPLETE.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$CODE/train_router.py" \
        --points "$DATA/all_points.npy" \
        --quant "$ARTIFACTS/all_quant_labels.npz" \
        --bpp "$ARTIFACTS/all_bpp.csv" \
        --train-indices "$DATA/router_train_indices.npy" \
        --val-indices "$DATA/router_val_indices.npy" \
        --test-indices "$DATA/test_indices.npy" \
        --lambda-json "$ARTIFACTS/rd_lambdas_train_only.json" \
        --output-dir "$ROUTER" \
        --epochs 45 --patience 10 --batch-size 48 --workers 10
fi

record "stage=test_plot GPU=none"
"$PYTHON" "$CODE/plot_results.py" \
    --test-quant "$ARTIFACTS/all_quant_labels.npz" \
    --router-predictions "$ROUTER/test_router_predictions.npz" \
    --test-indices "$DATA/test_indices.npy" \
    --output-dir "$RESULTS"

"$PYTHON" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
payload = {
    "status": "complete",
    "dataset": "Full ShapeNet55: 52,470 objects covering all 55 classes",
    "classifier": "DGCNN initialized from public ModelNet40 checkpoint and fully fine-tuned",
    "router": "five CE-loss outputs plus six-rate BPP head; no decision head",
    "gpus_used": [0, 1, 2],
    "maximum_simultaneous_gpus": 3,
    "plot": str(root / "results/dgcnn_shapenet55_accuracy_bpp.png"),
    "csv": str(root / "results/dgcnn_shapenet55_accuracy_bpp.csv"),
}
(root / "PIPELINE_COMPLETE.json").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
PY
touch "$ROOT/PIPELINE_COMPLETE"
record "ALL COMPLETE plot=$RESULTS/dgcnn_shapenet55_accuracy_bpp.png"
