#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ENV_NAME="${ENV_NAME:-fyp}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/cache/cls}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/runs/clip_hybrid_manifold_5way_cluster}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_DIR}/summary.json}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUTPUT_DIR}"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

conda run --no-capture-output -n "${ENV_NAME}" python "${PROJECT_ROOT}/src/train/train_upstream.py" \
  --cache-root "${CACHE_ROOT}" \
  --train-split "DF40_train" \
  --test-split "DF40_test_ff" \
  --ood-split "DF40_test_ood" \
  --real-rank 40 \
  --efs-rank 24 \
  --epochs 15 \
  --linear-warmup-epochs 5 \
  --batch-size 512 \
  --eval-batch-size 4096 \
  --lr 5e-4 \
  --temperature 0.1 \
  --alpha 1.0 \
  --lambda-linear-aux 0.5 \
  --lambda-manifold-aux 0.25 \
  --lambda-hardneg 0.0 \
  --lambda-real-hardneg 0.0 \
  --lambda-score 0.0 \
  --real-class-multiplier 2.0 \
  --hardneg-margin 0.75 \
  --real-hardneg-margin 0.75 \
  --score-target-default 0.55 \
  --score-target-hard 0.35 \
  --score-hard-weight 2.0 \
  --selection-real-weight 0.5 \
  --seed 42 \
  --device "${DEVICE}" \
  --output-json "${OUTPUT_JSON}"
