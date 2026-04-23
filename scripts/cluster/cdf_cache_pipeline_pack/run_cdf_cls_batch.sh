#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAME="${ENV_NAME:-fyp}"
SRC_ROOT="${SRC_ROOT:-$SCRIPT_DIR/src}"
RELATIVE_TO="${RELATIVE_TO:-$SCRIPT_DIR}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-$SCRIPT_DIR/cache_clip}"
MODEL_DIR="${MODEL_DIR:-$HOME/.cache/huggingface/hub/models--openai--clip-vit-large-patch14}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TARGET_SIZE="${TARGET_SIZE:-224}"
LIMIT_FAKE="${LIMIT_FAKE:-0}"
SEED="${SEED:-42}"

SCRIPT="$SCRIPT_DIR/prepare_cdf_cls_clip.py"

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
    "$RELATIVE_TO"/DF40_test_cdf/EFS/* \
    "$RELATIVE_TO"/DF40_test_cdf/FR/* \
    "$RELATIVE_TO"/DF40_test_cdf/FS/*
fi

for method_root in "$@"; do
  [ -d "$method_root" ] || continue
  echo "[RUN ] CLS $method_root"
  conda run --no-capture-output -n "$ENV_NAME" python "$SCRIPT" \
    --method-root "$method_root" \
    --relative-to "$RELATIVE_TO" \
    --cache-root "$OUT_CACHE_ROOT" \
    --src-root "$SRC_ROOT" \
    --model-dir "$MODEL_DIR" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --target-size "$TARGET_SIZE" \
    --limit-fake "$LIMIT_FAKE" \
    --seed "$SEED"
done
