#!/bin/bash
#SBATCH --job-name=cdf_patch
#SBATCH --output=cdf_patch_%j.out
#SBATCH --error=cdf_patch_%j.err
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --nodelist=gpu20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

export CUDA_VISIBLE_DEVICES=1

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

set -euo pipefail

DEFAULT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK_ROOT_CANDIDATE="${PACK_ROOT:-}"

if [ -z "$PACK_ROOT_CANDIDATE" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  if [ -f "${SLURM_SUBMIT_DIR}/run_cdf_cls_batch.sh" ] && \
     [ -f "${SLURM_SUBMIT_DIR}/run_cdf_patch_batch.sh" ] && \
     [ -d "${SLURM_SUBMIT_DIR}/src" ]; then
    PACK_ROOT_CANDIDATE="$SLURM_SUBMIT_DIR"
  fi
fi

if [ -z "$PACK_ROOT_CANDIDATE" ]; then
  PACK_ROOT_CANDIDATE="$DEFAULT_SCRIPT_DIR"
fi

SCRIPT_DIR="$(cd "$PACK_ROOT_CANDIDATE" && pwd)"

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

# 你当前场景默认只跑 patch
RUN_CLS="${RUN_CLS:-0}"
RUN_PATCH="${RUN_PATCH:-1}"

SRC_ROOT="${SRC_ROOT:-$SCRIPT_DIR/src}"
DEVICE="${DEVICE:-cuda}"
FACER_DEVICE="${FACER_DEVICE:-cuda}"
MODEL_DIR="${MODEL_DIR:-$HOME/.cache/huggingface/hub/models--openai--clip-vit-large-patch14}"

BATCH_SIZE="${BATCH_SIZE:-12}"
CLS_BATCH_SIZE="${CLS_BATCH_SIZE:-$BATCH_SIZE}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-$BATCH_SIZE}"

TARGET_SIZE="${TARGET_SIZE:-224}"
LIMIT_FAKE="${LIMIT_FAKE:-0}"
SEED="${SEED:-42}"
LAYER="${LAYER:--1}"
WITH_REGIONS="${WITH_REGIONS:-1}"

CDF_GROUPS="${CDF_GROUPS:-EFS}"

DATA_ROOT="${DATA_ROOT:-}"
RELATIVE_TO="${RELATIVE_TO:-}"

FINAL_ROOT="${FINAL_ROOT:-}"
CLS_FINAL_ROOT="${CLS_FINAL_ROOT:-}"
PATCH_FINAL_ROOT="${PATCH_FINAL_ROOT:-}"

TMP_BASE="${TMP_BASE:-${TMPDIR:-}}"
TMP_WORK_ROOT="${TMP_WORK_ROOT:-}"
TMP_CLS_CACHE_ROOT="${TMP_CLS_CACHE_ROOT:-}"
TMP_PATCH_CACHE_ROOT="${TMP_PATCH_CACHE_ROOT:-}"

# patch 读取 manifest 的根目录
MANIFEST_CACHE_ROOT="${MANIFEST_CACHE_ROOT:-}"

if [ -z "$DATA_ROOT" ]; then
  echo "Set DATA_ROOT to the absolute path of DF40_test_cdf on the cluster." >&2
  exit 1
fi

if [ -z "$RELATIVE_TO" ]; then
  RELATIVE_TO="$(cd "$(dirname "$DATA_ROOT")" && pwd)"
fi

if [ -z "$FINAL_ROOT" ]; then
  FINAL_ROOT="$RELATIVE_TO"
fi

if [ -z "$CLS_FINAL_ROOT" ]; then
  CLS_FINAL_ROOT="$FINAL_ROOT/cache_clip"
fi

if [ -z "$PATCH_FINAL_ROOT" ]; then
  PATCH_FINAL_ROOT="$FINAL_ROOT/cache_clip_patch"
fi

if [ -z "$MANIFEST_CACHE_ROOT" ]; then
  # 当前默认：patch 从 home 里的 cls cache 读
  MANIFEST_CACHE_ROOT="$CLS_FINAL_ROOT"
fi

if [ -z "$TMP_BASE" ]; then
  TMP_BASE="/tmp/$USER"
fi

if [ -z "$TMP_WORK_ROOT" ]; then
  TMP_WORK_ROOT="$TMP_BASE/cdf_cache"
fi

if [ -z "$TMP_CLS_CACHE_ROOT" ]; then
  TMP_CLS_CACHE_ROOT="$TMP_WORK_ROOT/cache_clip"
fi

if [ -z "$TMP_PATCH_CACHE_ROOT" ]; then
  TMP_PATCH_CACHE_ROOT="$TMP_WORK_ROOT/cache_clip_patch"
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Missing DATA_ROOT: $DATA_ROOT" >&2
  exit 1
fi

if [ ! -d "$SRC_ROOT" ]; then
  echo "Missing SRC_ROOT: $SRC_ROOT" >&2
  exit 1
fi

if [ "$RUN_CLS" = "1" ] && [ ! -f "$SCRIPT_DIR/run_cdf_cls_batch.sh" ]; then
  echo "Missing $SCRIPT_DIR/run_cdf_cls_batch.sh" >&2
  exit 1
fi

if [ "$RUN_PATCH" = "1" ] && [ ! -f "$SCRIPT_DIR/run_cdf_patch_batch.sh" ]; then
  echo "Missing $SCRIPT_DIR/run_cdf_patch_batch.sh" >&2
  exit 1
fi

if [ "$RUN_PATCH" = "1" ] && [ ! -d "$MANIFEST_CACHE_ROOT" ]; then
  echo "Missing MANIFEST_CACHE_ROOT: $MANIFEST_CACHE_ROOT" >&2
  exit 1
fi

sync_tree() {
  local src="$1"
  local dst="$2"
  if [ ! -d "$src" ]; then
    return 0
  fi
  mkdir -p "$dst"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$src"/ "$dst"/
  else
    cp -a "$src"/. "$dst"/
  fi
}

on_exit() {
  local status=$?

  echo "[SYNC] cls: $TMP_CLS_CACHE_ROOT -> $CLS_FINAL_ROOT"
  sync_tree "$TMP_CLS_CACHE_ROOT" "$CLS_FINAL_ROOT" || true

  echo "[SYNC] patch: $TMP_PATCH_CACHE_ROOT -> $PATCH_FINAL_ROOT"
  sync_tree "$TMP_PATCH_CACHE_ROOT" "$PATCH_FINAL_ROOT" || true

  exit "$status"
}

trap on_exit EXIT

METHODS=("$@")
if [ "${#METHODS[@]}" -eq 0 ]; then
  CDF_GROUPS="${CDF_GROUPS//,/ }"
  METHODS=()
  shopt -s nullglob
  for group in $CDF_GROUPS; do
    METHODS+=("$DATA_ROOT"/"$group"/*)
  done
  shopt -u nullglob
fi

if [ "${#METHODS[@]}" -eq 0 ]; then
  echo "No methods found under DATA_ROOT=$DATA_ROOT for groups: $CDF_GROUPS" >&2
  exit 1
fi

mkdir -p \
  "$TMP_WORK_ROOT" \
  "$TMP_CLS_CACHE_ROOT" \
  "$TMP_PATCH_CACHE_ROOT" \
  "$CLS_FINAL_ROOT" \
  "$PATCH_FINAL_ROOT"

echo "[INFO] pack_root=$SCRIPT_DIR"
echo "[INFO] src_root=$SRC_ROOT"
echo "[INFO] data_root=$DATA_ROOT"
echo "[INFO] relative_to=$RELATIVE_TO"
echo "[INFO] final_root=$FINAL_ROOT"
echo "[INFO] cls_final_root=$CLS_FINAL_ROOT"
echo "[INFO] patch_final_root=$PATCH_FINAL_ROOT"
echo "[INFO] manifest_cache_root=$MANIFEST_CACHE_ROOT"
echo "[INFO] tmp_base=$TMP_BASE"
echo "[INFO] tmp_work_root=$TMP_WORK_ROOT"
echo "[INFO] tmp_cls_cache_root=$TMP_CLS_CACHE_ROOT"
echo "[INFO] tmp_patch_cache_root=$TMP_PATCH_CACHE_ROOT"
echo "[INFO] model_dir=$MODEL_DIR"
echo "[INFO] device=$DEVICE"
echo "[INFO] facer_device=$FACER_DEVICE"
echo "[INFO] cdf_groups=$CDF_GROUPS"
echo "[INFO] run_cls=$RUN_CLS"
echo "[INFO] run_patch=$RUN_PATCH"
echo "[INFO] methods_count=${#METHODS[@]}"

if [ "$RUN_CLS" = "1" ]; then
  ENV_NAME="$ENV_NAME" \
  SRC_ROOT="$SRC_ROOT" \
  RELATIVE_TO="$RELATIVE_TO" \
  OUT_CACHE_ROOT="$TMP_CLS_CACHE_ROOT" \
  MODEL_DIR="$MODEL_DIR" \
  DEVICE="$DEVICE" \
  BATCH_SIZE="$CLS_BATCH_SIZE" \
  TARGET_SIZE="$TARGET_SIZE" \
  LIMIT_FAKE="$LIMIT_FAKE" \
  SEED="$SEED" \
  bash "$SCRIPT_DIR/run_cdf_cls_batch.sh" "${METHODS[@]}"
fi

if [ "$RUN_PATCH" = "1" ]; then
  ENV_NAME="$ENV_NAME" \
  SRC_ROOT="$SRC_ROOT" \
  RELATIVE_TO="$RELATIVE_TO" \
  MANIFEST_CACHE_ROOT="$MANIFEST_CACHE_ROOT" \
  OUT_CACHE_ROOT="$TMP_PATCH_CACHE_ROOT" \
  MODEL_DIR="$MODEL_DIR" \
  DEVICE="$DEVICE" \
  BATCH_SIZE="$PATCH_BATCH_SIZE" \
  TARGET_SIZE="$TARGET_SIZE" \
  LAYER="$LAYER" \
  FACER_DEVICE="$FACER_DEVICE" \
  WITH_REGIONS="$WITH_REGIONS" \
  bash "$SCRIPT_DIR/run_cdf_patch_batch.sh" "${METHODS[@]}"
fi