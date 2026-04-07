#!/bin/bash
#SBATCH --job-name=within_cdf_pipeline
#SBATCH --output=/home/comp/f2256768/%j.out
#SBATCH --error=/home/comp/f2256768/%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/comp/f2256768/FYP_final}"

if ! command -v conda >/dev/null 2>&1; then
  for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"
  do
    if [ -f "$candidate" ]; then
      # shellcheck disable=SC1090
      source "$candidate"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in this shell. Load Anaconda/Miniconda first." >&2
  exit 1
fi

ENV_NAME="${ENV_NAME:-fyp}"
SOURCE_NODE="${SOURCE_NODE:-gpu10}"
SCP_CMD=(scp -F /dev/null -q -r)

CDF_CLS_CACHE_ROOT="${CDF_CLS_CACHE_ROOT:-/home/comp/f2256768/cdf_cache/cache_clip}"
CDF_PATCH_CACHE_ROOT="${CDF_PATCH_CACHE_ROOT:-/tmp/f2256768/cdf_cache/cache_clip_patch}"
CDF_SPLIT_NAME="${CDF_SPLIT_NAME:-DF40_test_cdf}"
COMPACT_CACHE_DIR="${COMPACT_CACHE_DIR:-/tmp/f2256768/fyp_final_compact_cdf}"

REAL_CLS_ROOT="${REAL_CLS_ROOT:-/tmp/celebdf_real_clip/cache_clip/Celeb-DF-v2/real}"
REAL_COMPACT_DIR="${REAL_COMPACT_DIR:-/tmp/f2256768/fyp_final_compact_celebdf_real}"

FULL_UPSTREAM="${FULL_UPSTREAM:-${PROJECT_ROOT}/checkpoints/upstream/checkpoint_best_hybrid_manifold.pt}"
FULL_PATCH="${FULL_PATCH:-${PROJECT_ROOT}/within_checkpoints/downstream/patch_branch.joblib}"
FULL_PAIR="${FULL_PAIR:-${PROJECT_ROOT}/within_checkpoints/downstream/pair_branch.joblib}"
FULL_HEAD="${FULL_HEAD:-${PROJECT_ROOT}/within_checkpoints/heads/route_meta_head.joblib}"
FULL_HEAD_META="${FULL_HEAD_META:-${PROJECT_ROOT}/within_checkpoints/heads/route_meta_head_meta.json}"

NOFR_UPSTREAM="${NOFR_UPSTREAM:-${PROJECT_ROOT}/checkpoints/upstream/checkpoint_no_fr.pt}"
NOFR_PATCH="${NOFR_PATCH:-${PROJECT_ROOT}/within_checkpoints/downstream/patch_branch_no_fr.joblib}"
NOFR_PAIR="${NOFR_PAIR:-${PROJECT_ROOT}/within_checkpoints/downstream/pair_branch_no_fr.joblib}"
NOFR_HEAD="${NOFR_HEAD:-${PROJECT_ROOT}/within_checkpoints/heads/route_meta_head_no_fr.joblib}"
NOFR_HEAD_META="${NOFR_HEAD_META:-${PROJECT_ROOT}/within_checkpoints/heads/route_meta_head_no_fr_meta.json}"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/within_outputs}"
THRESHOLDS="${THRESHOLDS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9}"
DEVICE="${DEVICE:-cuda}"
ROUTE_BATCH_SIZE="${ROUTE_BATCH_SIZE:-2048}"
MAX_CDF_FAKE_PER_METHOD="${MAX_CDF_FAKE_PER_METHOD:-0}"
SEED="${SEED:-42}"

mkdir -p "${OUTPUT_DIR}" "${CDF_PATCH_CACHE_ROOT}" "${COMPACT_CACHE_DIR}" "${REAL_CLS_ROOT}" "${REAL_COMPACT_DIR}"

copy_if_missing() {
  local src="$1"
  local dst_parent="$2"
  local dst_name="$3"
  local marker_rel="$4"
  local dst="${dst_parent}/${dst_name}"
  if [ -e "${dst}/${marker_rel}" ]; then
    echo "[SKIP] ${dst}/${marker_rel}"
    return
  fi
  echo "[COPY] ${SOURCE_NODE}:${src} -> ${dst_parent}/"
  rm -rf "${dst}"
  "${SCP_CMD[@]}" "${SOURCE_NODE}:${src}" "${dst_parent}/"
}

echo "[INFO] node=$(hostname)"
echo "[INFO] source_node=${SOURCE_NODE}"
echo "[INFO] cdf_cls_cache_root=${CDF_CLS_CACHE_ROOT}"
echo "[INFO] cdf_patch_cache_root=${CDF_PATCH_CACHE_ROOT}"
echo "[INFO] compact_cache_dir=${COMPACT_CACHE_DIR}"
echo "[INFO] output_dir=${OUTPUT_DIR}"

copy_if_missing "/tmp/f2256768/cdf_cache/cache_clip_patch" "/tmp/f2256768/cdf_cache" "cache_clip_patch" "DF40_test_cdf/EFS/patch_DiT.npz"
copy_if_missing "/tmp/f2256768/fyp_final_compact_cdf" "/tmp/f2256768" "fyp_final_compact_cdf" "tmp__f2256768__cdf_cache__cache_clip_patch__DF40_test_cdf__EFS__patch_DiT.npz.npz"
copy_if_missing "/tmp/celebdf_real_clip" "/tmp" "celebdf_real_clip" "cache_clip/Celeb-DF-v2/real/cls_Celeb-real.npz"
copy_if_missing "/tmp/f2256768/fyp_final_compact_celebdf_real" "/tmp/f2256768" "fyp_final_compact_celebdf_real" "tmp__celebdf_real_clip__cache_clip_patch__Celeb-DF-v2__real__patch_Celeb-real.npz.npz"

COMMON_SWEEP_ARGS=(
  --cdf-cls-cache-root "${CDF_CLS_CACHE_ROOT}"
  --cdf-patch-cache-root "${CDF_PATCH_CACHE_ROOT}"
  --cdf-split-name "${CDF_SPLIT_NAME}"
  --compact-cache-dir "${COMPACT_CACHE_DIR}"
  --device "${DEVICE}"
  --route-batch-size "${ROUTE_BATCH_SIZE}"
  --max-cdf-fake-per-method "${MAX_CDF_FAKE_PER_METHOD}"
  --seed "${SEED}"
  --thresholds ${THRESHOLDS}
  --cdf-real-name Celeb-real
  --cdf-real-cls "${REAL_CLS_ROOT}/cls_Celeb-real.npz"
  --cdf-real-compact "${REAL_COMPACT_DIR}/tmp__celebdf_real_clip__cache_clip_patch__Celeb-DF-v2__real__patch_Celeb-real.npz.npz"
  --cdf-real-name YouTube-real
  --cdf-real-cls "${REAL_CLS_ROOT}/cls_YouTube-real.npz"
  --cdf-real-compact "${REAL_COMPACT_DIR}/tmp__celebdf_real_clip__cache_clip_patch__Celeb-DF-v2__real__patch_YouTube-real.npz.npz"
)

COMMON_REPLAY_ARGS=(
  --cdf-cls-cache-root "${CDF_CLS_CACHE_ROOT}"
  --cdf-patch-cache-root "${CDF_PATCH_CACHE_ROOT}"
  --cdf-split-name "${CDF_SPLIT_NAME}"
  --compact-cache-dir "${COMPACT_CACHE_DIR}"
  --device "${DEVICE}"
  --route-batch-size "${ROUTE_BATCH_SIZE}"
  --max-cdf-fake-per-method "${MAX_CDF_FAKE_PER_METHOD}"
  --seed "${SEED}"
)

conda run --no-capture-output -n "${ENV_NAME}" python "${PROJECT_ROOT}/scripts/within_run_cdf_threshold_sweep.py" \
  --upstream-checkpoint "${FULL_UPSTREAM}" \
  --patch-branch "${FULL_PATCH}" \
  --pair-branch "${FULL_PAIR}" \
  --route-meta-head "${FULL_HEAD}" \
  --head-meta "${FULL_HEAD_META}" \
  --cdf-groups EFS FR FS \
  --output-json "${OUTPUT_DIR}/full_threshold_sweep_cdf.json" \
  --output-csv "${OUTPUT_DIR}/full_threshold_sweep_cdf.csv" \
  "${COMMON_SWEEP_ARGS[@]}"

conda run --no-capture-output -n "${ENV_NAME}" python "${PROJECT_ROOT}/scripts/within_run_cdf_threshold_sweep.py" \
  --upstream-checkpoint "${NOFR_UPSTREAM}" \
  --patch-branch "${NOFR_PATCH}" \
  --pair-branch "${NOFR_PAIR}" \
  --route-meta-head "${NOFR_HEAD}" \
  --head-meta "${NOFR_HEAD_META}" \
  --cdf-groups EFS FS \
  --output-json "${OUTPUT_DIR}/no_fr_threshold_sweep_cdf.json" \
  --output-csv "${OUTPUT_DIR}/no_fr_threshold_sweep_cdf.csv" \
  "${COMMON_SWEEP_ARGS[@]}"

conda run --no-capture-output -n "${ENV_NAME}" python "${PROJECT_ROOT}/scripts/within_run_cdf_replay_eval.py" \
  --upstream-checkpoint "${FULL_UPSTREAM}" \
  --patch-branch "${FULL_PATCH}" \
  --pair-branch "${FULL_PAIR}" \
  --route-meta-head "${FULL_HEAD}" \
  --head-meta "${FULL_HEAD_META}" \
  --cdf-groups EFS FR FS \
  --output-json "${OUTPUT_DIR}/full_cdf_replay_eval.json" \
  --output-csv "${OUTPUT_DIR}/full_cdf_replay_eval.csv" \
  "${COMMON_REPLAY_ARGS[@]}"

conda run --no-capture-output -n "${ENV_NAME}" python "${PROJECT_ROOT}/scripts/within_run_cdf_replay_eval.py" \
  --upstream-checkpoint "${NOFR_UPSTREAM}" \
  --patch-branch "${NOFR_PATCH}" \
  --pair-branch "${NOFR_PAIR}" \
  --route-meta-head "${NOFR_HEAD}" \
  --head-meta "${NOFR_HEAD_META}" \
  --cdf-groups EFS FS \
  --output-json "${OUTPUT_DIR}/no_fr_cdf_replay_eval.json" \
  --output-csv "${OUTPUT_DIR}/no_fr_cdf_replay_eval.csv" \
  "${COMMON_REPLAY_ARGS[@]}"

echo "[DONE] within CDF cluster pipeline finished."
