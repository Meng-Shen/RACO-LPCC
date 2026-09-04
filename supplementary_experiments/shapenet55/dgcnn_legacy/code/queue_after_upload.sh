#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT="${RACO_SHAPENET55_DGCNN_ROOT:-/home/sm/raco_rate_aware_shapenet55_dgcnn_20260825}"
ARCHIVE=/home/sm/datasets/ShapeNet55-34_official/ShapeNet55.zip
EXPECTED=8367547065
LOG="$ROOT/upload_waiter.log"

mkdir -p "$ROOT"
exec 8>"$ROOT/.upload_waiter.lock"
if ! flock -n 8; then
    exit 0
fi
while true; do
    current=$(stat -c %s "$ARCHIVE" 2>/dev/null || echo 0)
    printf '[%s] upload_bytes=%s/%s\n' "$(date '+%F %T')" "$current" "$EXPECTED" >>"$LOG"
    if [[ "$current" -eq "$EXPECTED" ]]; then
        break
    fi
    if [[ "$current" -gt "$EXPECTED" ]]; then
        printf '[%s] ERROR archive larger than expected\n' "$(date '+%F %T')" >>"$LOG"
        exit 2
    fi
    sleep 30
done
printf '[%s] upload complete; starting verified full pipeline\n' "$(date '+%F %T')" >>"$LOG"
exec bash "$SCRIPT_DIR/run_shapenet55_full_pipeline.sh"
