#!/bin/bash
#SBATCH --job-name=cdf_gpu10
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=long
#SBATCH --nodelist=gpu10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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
NO_FR="${NO_FR:-0}"

# Raw CDF dataset root on the cluster. Change this if your actual dataset path differs.
DATA_ROOT="${DATA_ROOT:-/home/comp/f2256768/DF40_test_cdf}"
RELATIVE_TO="${RELATIVE_TO:-/home/comp/f2256768}"

# These two paths are the important ones already reflected in your current cluster cache layout.
CLS_CACHE_ROOT="${CLS_CACHE_ROOT:-/home/comp/f2256768/cdf_cache/cache_clip}"
PATCH_CACHE_ROOT="${PATCH_CACHE_ROOT:-/tmp/f2256768/cdf_cache/cache_clip_patch}"

# Current tested CDF cache pack still depends on the old source tree for face-region imports.
# If your cluster checkout lives elsewhere, override these two paths at submit time.
CDF_PACK_ROOT="${CDF_PACK_ROOT:-/home/comp/f2256768/FYP/Fyp_clean/outputs/experiments/balanced_3000_seed42/20260320_cdf_cache_pipeline_pack}"
SRC_ROOT="${SRC_ROOT:-/home/comp/f2256768/FYP/Fyp_clean/src}"

MODEL_DIR="${MODEL_DIR:-$HOME/.cache/huggingface/hub/models--openai--clip-vit-large-patch14}"
DEVICE="${DEVICE:-cuda}"
FACER_DEVICE="${FACER_DEVICE:-cuda}"

CLS_BATCH_SIZE="${CLS_BATCH_SIZE:-8}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-8}"
TARGET_SIZE="${TARGET_SIZE:-224}"
LAYER="${LAYER:--1}"
LIMIT_FAKE="${LIMIT_FAKE:-0}"
SEED="${SEED:-42}"
WITH_REGIONS="${WITH_REGIONS:-1}"

# Control which stage(s) to run.
RUN_CLS="${RUN_CLS:-1}"
RUN_PATCH="${RUN_PATCH:-1}"

# Default runs all available CDF fake groups on this single node.
if [ "${NO_FR}" = "1" ]; then
  DEFAULT_CDF_GROUPS="EFS FS FE"
else
  DEFAULT_CDF_GROUPS="EFS FR FS"
fi
CDF_GROUPS="${CDF_GROUPS:-${DEFAULT_CDF_GROUPS}}"
CLS_PREP_SCRIPT="${CDF_PACK_ROOT}/prepare_cdf_cls_clip.py"
PATCH_PREP_SCRIPT="${CDF_PACK_ROOT}/prepare_cdf_patch_clip.py"

if [ ! -d "${DATA_ROOT}" ]; then
  echo "Missing DATA_ROOT: ${DATA_ROOT}" >&2
  exit 1
fi

if [ ! -d "${CDF_PACK_ROOT}" ]; then
  echo "Missing CDF_PACK_ROOT: ${CDF_PACK_ROOT}" >&2
  exit 1
fi

if [ ! -f "${CLS_PREP_SCRIPT}" ]; then
  echo "Missing CDF CLS script: ${CLS_PREP_SCRIPT}" >&2
  exit 1
fi

if [ ! -f "${PATCH_PREP_SCRIPT}" ]; then
  echo "Missing CDF PATCH script: ${PATCH_PREP_SCRIPT}" >&2
  exit 1
fi

if [ ! -d "${SRC_ROOT}" ]; then
  echo "Missing SRC_ROOT: ${SRC_ROOT}" >&2
  echo "The current CDF pack still imports old after_a utilities, so this must point to the matching old src tree." >&2
  exit 1
fi

mkdir -p "${CLS_CACHE_ROOT}" "${PATCH_CACHE_ROOT}"

declare -a METHOD_ROOTS=()
if [ "$#" -gt 0 ]; then
  for method_root in "$@"; do
    [ -d "${method_root}" ] || continue
    METHOD_ROOTS+=("${method_root}")
  done
else
  CDF_GROUPS="${CDF_GROUPS//,/ }"
  for group in ${CDF_GROUPS}; do
    for method_root in "${DATA_ROOT}/${group}"/*; do
      [ -d "${method_root}" ] || continue
      METHOD_ROOTS+=("${method_root}")
    done
  done
fi

if [ "${#METHOD_ROOTS[@]}" -eq 0 ]; then
  echo "No CDF method roots found." >&2
  exit 1
fi

declare -a CLS_METHODS=()
declare -a PATCH_METHODS=()

for method_root in "${METHOD_ROOTS[@]}"; do
  group="$(basename "$(dirname "${method_root}")")"
  method="$(basename "${method_root}")"

  cls_out="${CLS_CACHE_ROOT}/DF40_test_cdf/${group}/cls_${method}.npz"
  manifest_out="${CLS_CACHE_ROOT}/DF40_test_cdf/${group}/manifest_${method}.csv"
  patch_out="${PATCH_CACHE_ROOT}/DF40_test_cdf/${group}/patch_${method}.npz"

  if [ "${RUN_CLS}" = "1" ]; then
    if [ -f "${cls_out}" ] && [ -f "${manifest_out}" ]; then
      echo "[SKIP] CLS ${group}/${method} -> ${cls_out}"
    else
      CLS_METHODS+=("${method_root}")
    fi
  fi

  if [ "${RUN_PATCH}" = "1" ]; then
    if [ -f "${patch_out}" ]; then
      echo "[SKIP] PATCH ${group}/${method} -> ${patch_out}"
    else
      PATCH_METHODS+=("${method_root}")
    fi
  fi
done

echo "[INFO] node=$(hostname)"
echo "[INFO] data_root=${DATA_ROOT}"
echo "[INFO] relative_to=${RELATIVE_TO}"
echo "[INFO] cls_cache_root=${CLS_CACHE_ROOT}"
echo "[INFO] patch_cache_root=${PATCH_CACHE_ROOT}"
echo "[INFO] cdf_pack_root=${CDF_PACK_ROOT}"
echo "[INFO] src_root=${SRC_ROOT}"
echo "[INFO] model_dir=${MODEL_DIR}"
echo "[INFO] device=${DEVICE}"
echo "[INFO] facer_device=${FACER_DEVICE}"
echo "[INFO] no_fr=${NO_FR}"
echo "[INFO] run_cls=${RUN_CLS} pending_cls=${#CLS_METHODS[@]}"
echo "[INFO] run_patch=${RUN_PATCH} pending_patch=${#PATCH_METHODS[@]}"

if [ "${RUN_CLS}" = "1" ] && [ "${#CLS_METHODS[@]}" -gt 0 ]; then
  for method_root in "${CLS_METHODS[@]}"; do
    echo "[RUN ] CDF CLS ${method_root}"
    conda run --no-capture-output -n "${ENV_NAME}" python "${CLS_PREP_SCRIPT}" \
      --method-root "${method_root}" \
      --relative-to "${RELATIVE_TO}" \
      --cache-root "${CLS_CACHE_ROOT}" \
      --src-root "${SRC_ROOT}" \
      --model-dir "${MODEL_DIR}" \
      --device "${DEVICE}" \
      --batch-size "${CLS_BATCH_SIZE}" \
      --target-size "${TARGET_SIZE}" \
      --limit-fake "${LIMIT_FAKE}" \
      --seed "${SEED}"
  done
fi

if [ "${RUN_PATCH}" = "1" ] && [ "${#PATCH_METHODS[@]}" -gt 0 ]; then
  for method_root in "${PATCH_METHODS[@]}"; do
    echo "[RUN ] CDF PATCH ${method_root}"
    CMD=(
      conda run --no-capture-output -n "${ENV_NAME}" python "${PATCH_PREP_SCRIPT}"
      --method-root "${method_root}"
      --relative-to "${RELATIVE_TO}"
      --manifest-root "${CLS_CACHE_ROOT}"
      --cache-root "${PATCH_CACHE_ROOT}"
      --src-root "${SRC_ROOT}"
      --model-dir "${MODEL_DIR}"
      --backbone clip
      --target-size "${TARGET_SIZE}"
      --layer "${LAYER}"
      --device "${DEVICE}"
      --batch-size "${PATCH_BATCH_SIZE}"
      --facer-device "${FACER_DEVICE}"
    )
    if [ "${WITH_REGIONS}" = "0" ]; then
      CMD+=(--without-regions)
    fi
    "${CMD[@]}"
  done
fi

echo "[DONE] CDF cache job finished."
