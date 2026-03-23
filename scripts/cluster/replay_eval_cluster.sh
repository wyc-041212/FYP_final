#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ENV_NAME="${ENV_NAME:-fyp}"
UPSTREAM_CHECKPOINT="${UPSTREAM_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/upstream/checkpoint_best_hybrid_manifold.pt}"
PATCH_BRANCH="${PATCH_BRANCH:-${PROJECT_ROOT}/checkpoints/downstream/patch_branch.joblib}"
PAIR_BRANCH="${PAIR_BRANCH:-${PROJECT_ROOT}/checkpoints/downstream/pair_branch.joblib}"
ROUTE_META_HEAD="${ROUTE_META_HEAD:-${PROJECT_ROOT}/checkpoints/heads/route_meta_head.joblib}"
HEAD_META="${HEAD_META:-${PROJECT_ROOT}/checkpoints/heads/route_meta_head_meta.json}"
CLS_CACHE_ROOT="${CLS_CACHE_ROOT:-${PROJECT_ROOT}/cache/cls}"
PATCH_CACHE_ROOT="${PATCH_CACHE_ROOT:-${PROJECT_ROOT}/cache/patch}"
COMPACT_CACHE_DIR="${COMPACT_CACHE_DIR:-${PROJECT_ROOT}/cache/compact}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/replay}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_DIR}/replay_eval.json}"
OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_DIR}/replay_eval.csv}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUTPUT_DIR}"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::RuntimeWarning}"

conda run --no-capture-output -n "${ENV_NAME}" python "${PROJECT_ROOT}/main.py" --mode replay \
  --upstream-checkpoint "${UPSTREAM_CHECKPOINT}" \
  --patch-branch "${PATCH_BRANCH}" \
  --pair-branch "${PAIR_BRANCH}" \
  --route-meta-head "${ROUTE_META_HEAD}" \
  --head-meta "${HEAD_META}" \
  --cls-cache-root "${CLS_CACHE_ROOT}" \
  --patch-cache-root "${PATCH_CACHE_ROOT}" \
  --compact-cache-dir "${COMPACT_CACHE_DIR}" \
  --device "${DEVICE}" \
  --output-json "${OUTPUT_JSON}" \
  --output-csv "${OUTPUT_CSV}"
