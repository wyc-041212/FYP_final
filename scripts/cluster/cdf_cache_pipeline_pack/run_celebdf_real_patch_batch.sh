#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAME="${ENV_NAME:-fyp}"
SRC_ROOT="${SRC_ROOT:-$SCRIPT_DIR/src}"
RELATIVE_TO="${RELATIVE_TO:-/Volumes/未命名}"
MANIFEST_CACHE_ROOT="${MANIFEST_CACHE_ROOT:-$RELATIVE_TO/cache_clip}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-$RELATIVE_TO/cache_clip_patch}"
MODEL_DIR="${MODEL_DIR:-$HOME/.cache/huggingface/hub/models--openai--clip-vit-large-patch14}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TARGET_SIZE="${TARGET_SIZE:-224}"
LAYER="${LAYER:--1}"
FACER_DEVICE="${FACER_DEVICE:-cuda}"
WITH_REGIONS="${WITH_REGIONS:-1}"

SCRIPT="$SCRIPT_DIR/prepare_celebdf_real_patch_clip.py"

if [ ! -f "$SCRIPT" ]; then
  echo "Missing script: $SCRIPT" >&2
  exit 1
fi

if [ ! -d "$SRC_ROOT" ]; then
  echo "Missing src root: $SRC_ROOT" >&2
  exit 1
fi

if [ "$#" -eq 0 ]; then
  set -- \
    "$RELATIVE_TO"/Celeb-DF-v2/Celeb-real \
    "$RELATIVE_TO"/Celeb-DF-v2/YouTube-real
fi

for dataset_root in "$@"; do
  [ -d "$dataset_root" ] || continue
  echo "[RUN ] REAL PATCH $dataset_root"
  cmd=(
    conda run --no-capture-output -n "$ENV_NAME" python "$SCRIPT"
    --dataset-root "$dataset_root"
    --relative-to "$RELATIVE_TO"
    --manifest-root "$MANIFEST_CACHE_ROOT"
    --cache-root "$OUT_CACHE_ROOT"
    --src-root "$SRC_ROOT"
    --model-dir "$MODEL_DIR"
    --backbone clip
    --target-size "$TARGET_SIZE"
    --layer "$LAYER"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --facer-device "$FACER_DEVICE"
  )
  if [ "$WITH_REGIONS" = "0" ]; then
    cmd+=(--without-regions)
  fi
  "${cmd[@]}"
done
