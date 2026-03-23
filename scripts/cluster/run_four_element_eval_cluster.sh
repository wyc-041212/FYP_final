#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ENV_NAME="${ENV_NAME:-fyp}"
PATCH_CACHE_ROOT="${PATCH_CACHE_ROOT:-${PROJECT_ROOT}/cache/patch}"
CLS_CACHE_ROOT="${CLS_CACHE_ROOT:-${PROJECT_ROOT}/cache/cls}"
COMPACT_CACHE_DIR="${COMPACT_CACHE_DIR:-${PROJECT_ROOT}/cache/compact}"
UPSTREAM_CHECKPOINT="${UPSTREAM_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/upstream/checkpoint_best_hybrid_manifold.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/runs}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_DIR}/manifold_pair_patch_fusion_cluster.json}"
OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_DIR}/manifold_pair_patch_fusion_cluster.csv}"

TRAIN_REAL_MAX="${TRAIN_REAL_MAX:-0}"
EVAL_REAL_MAX="${EVAL_REAL_MAX:-0}"
MAX_TRAIN_FAKE_PER_METHOD="${MAX_TRAIN_FAKE_PER_METHOD:-0}"
MAX_TEST_FAKE_PER_METHOD="${MAX_TEST_FAKE_PER_METHOD:-0}"
MAX_OOD_FAKE_PER_METHOD="${MAX_OOD_FAKE_PER_METHOD:-0}"
FUSION_STEP="${FUSION_STEP:-0.05}"
PAIR_ROUTE_MODE="${PAIR_ROUTE_MODE:-weighted}"
VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-holdout_method}"
PAIR_REGION_MODE="${PAIR_REGION_MODE:-all_regions}"
SEED="${SEED:-42}"
NO_TUNING_MAINLINE="${NO_TUNING_MAINLINE:-1}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUTPUT_DIR}"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::RuntimeWarning}"

CMD=(
  conda run --no-capture-output -n "${ENV_NAME}" python "${PROJECT_ROOT}/src/eval/run_manifold_pair_patch_fusion.py"
  --patch-cache-root "${PATCH_CACHE_ROOT}"
  --cls-cache-root "${CLS_CACHE_ROOT}"
  --compact-cache-dir "${COMPACT_CACHE_DIR}"
  --hybrid-checkpoint "${UPSTREAM_CHECKPOINT}"
  --train-real-max "${TRAIN_REAL_MAX}"
  --eval-real-max "${EVAL_REAL_MAX}"
  --max-train-fake-per-method "${MAX_TRAIN_FAKE_PER_METHOD}"
  --max-test-fake-per-method "${MAX_TEST_FAKE_PER_METHOD}"
  --max-ood-fake-per-method "${MAX_OOD_FAKE_PER_METHOD}"
  --fusion-step "${FUSION_STEP}"
  --pair-route-mode "${PAIR_ROUTE_MODE}"
  --val-split-mode "${VAL_SPLIT_MODE}"
  --pair-region-mode "${PAIR_REGION_MODE}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --output-json "${OUTPUT_JSON}"
  --output-csv "${OUTPUT_CSV}"
)

if [ "${NO_TUNING_MAINLINE}" = "1" ]; then
  CMD+=(--no-tuning-mainline)
fi

"${CMD[@]}"
