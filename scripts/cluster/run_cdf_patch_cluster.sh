#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ENV_NAME="${ENV_NAME:-fyp}"
CDF_PACK_ROOT="${CDF_PACK_ROOT:-/Users/wuyuchen/Desktop/FYP/Fyp_clean/outputs/experiments/balanced_3000_seed42/20260320_cdf_cache_pipeline_pack}"
SRC_ROOT="${SRC_ROOT:-${PROJECT_ROOT}/src}"
DATA_ROOT="${DATA_ROOT:-/Volumes/未命名/DF40_test_cdf}"
RELATIVE_TO="${RELATIVE_TO:-$(cd "${DATA_ROOT}/.." && pwd)}"
MANIFEST_CACHE_ROOT="${MANIFEST_CACHE_ROOT:-${PROJECT_ROOT}/cache/cls}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-${PROJECT_ROOT}/cache/patch}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/clip}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TARGET_SIZE="${TARGET_SIZE:-224}"
LAYER="${LAYER:--1}"
FACER_DEVICE="${FACER_DEVICE:-cuda}"
WITH_REGIONS="${WITH_REGIONS:-1}"

SCRIPT="${CDF_PACK_ROOT}/prepare_cdf_patch_clip.py"

if [ ! -f "${SCRIPT}" ]; then
  echo "Missing CDF PATCH script: ${SCRIPT}" >&2
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
  echo "[RUN ] CDF PATCH ${method_root}"
  CMD=(
    conda run --no-capture-output -n "${ENV_NAME}" python "${SCRIPT}"
    --method-root "${method_root}"
    --relative-to "${RELATIVE_TO}"
    --manifest-root "${MANIFEST_CACHE_ROOT}"
    --cache-root "${OUT_CACHE_ROOT}"
    --src-root "${SRC_ROOT}"
    --model-dir "${MODEL_DIR}"
    --backbone clip
    --target-size "${TARGET_SIZE}"
    --layer "${LAYER}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --facer-device "${FACER_DEVICE}"
  )
  if [ "${WITH_REGIONS}" = "0" ]; then
    CMD+=(--without-regions)
  fi
  "${CMD[@]}"
done
