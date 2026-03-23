#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ENV_NAME="${ENV_NAME:-fyp}"
CDF_PACK_ROOT="${CDF_PACK_ROOT:-/Users/wuyuchen/Desktop/FYP/Fyp_clean/outputs/experiments/balanced_3000_seed42/20260320_cdf_cache_pipeline_pack}"
SRC_ROOT="${SRC_ROOT:-${PROJECT_ROOT}/src}"
DATA_ROOT="${DATA_ROOT:-/Volumes/未命名/DF40_test_cdf}"
RELATIVE_TO="${RELATIVE_TO:-$(cd "${DATA_ROOT}/.." && pwd)}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-${PROJECT_ROOT}/cache/cls}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/clip}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TARGET_SIZE="${TARGET_SIZE:-224}"
LIMIT_FAKE="${LIMIT_FAKE:-0}"
SEED="${SEED:-42}"

SCRIPT="${CDF_PACK_ROOT}/prepare_cdf_cls_clip.py"

if [ ! -f "${SCRIPT}" ]; then
  echo "Missing CDF CLS script: ${SCRIPT}" >&2
  exit 1
fi

mkdir -p "${OUT_CACHE_ROOT}"

if [ "$#" -eq 0 ]; then
  set -- \
    "${DATA_ROOT}"/EFS/* \
    "${DATA_ROOT}"/FR/* \
    "${DATA_ROOT}"/FS/*
fi

for method_root in "$@"; do
  [ -d "${method_root}" ] || continue
  echo "[RUN ] CDF CLS ${method_root}"
  conda run --no-capture-output -n "${ENV_NAME}" python "${SCRIPT}" \
    --method-root "${method_root}" \
    --relative-to "${RELATIVE_TO}" \
    --cache-root "${OUT_CACHE_ROOT}" \
    --src-root "${SRC_ROOT}" \
    --model-dir "${MODEL_DIR}" \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}" \
    --target-size "${TARGET_SIZE}" \
    --limit-fake "${LIMIT_FAKE}" \
    --seed "${SEED}"
done
