#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT="${RACO_SHAPENET55_ROOT:-/home/sm/raco_rate_aware_shapenet55_pointmae_20260825}"
ARCHIVE="${RACO_SHAPENET55_ARCHIVE:-/home/sm/datasets/ShapeNet55-34_official/ShapeNet55.zip}"
WEIGHT="$ROOT/checkpoints/pointmae_shapenet55_pretrain.pth"
ARCHIVE_EXPECTED=8367547065
WEIGHT_EXPECTED=348288755
LOG="$ROOT/upload_waiter.log"

mkdir -p "$ROOT"
exec 8>"$ROOT/.upload_waiter.lock"
if ! flock -n 8; then
    exit 0
fi
while true; do
    archive_bytes=$(stat -c %s "$ARCHIVE" 2>/dev/null || echo 0)
    weight_bytes=$(stat -c %s "$WEIGHT" 2>/dev/null || echo 0)
    printf '[%s] archive=%s/%s weight=%s/%s\n' \
        "$(date '+%F %T')" "$archive_bytes" "$ARCHIVE_EXPECTED" \
        "$weight_bytes" "$WEIGHT_EXPECTED" >>"$LOG"
    if [[ "$archive_bytes" -eq "$ARCHIVE_EXPECTED" && "$weight_bytes" -eq "$WEIGHT_EXPECTED" ]]; then
        break
    fi
    if [[ "$archive_bytes" -gt "$ARCHIVE_EXPECTED" || "$weight_bytes" -gt "$WEIGHT_EXPECTED" ]]; then
        printf '[%s] ERROR input larger than expected\n' "$(date '+%F %T')" >>"$LOG"
        exit 2
    fi
    sleep 30
done
printf '[%s] uploads complete; starting verified Point-MAE pipeline\n' "$(date '+%F %T')" >>"$LOG"
exec bash "$SCRIPT_DIR/run_shapenet55_pointmae_pipeline.sh"
