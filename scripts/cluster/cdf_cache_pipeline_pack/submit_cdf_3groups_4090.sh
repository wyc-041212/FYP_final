#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${DATA_ROOT:-}"
FINAL_ROOT="${FINAL_ROOT:-}"
ENV_NAME="${ENV_NAME:-fyp}"
MODEL_DIR="${MODEL_DIR:-$HOME/.cache/huggingface/hub/models--openai--clip-vit-large-patch14}"
DEVICE="${DEVICE:-cuda}"
FACER_DEVICE="${FACER_DEVICE:-cuda}"
CLS_BATCH_SIZE="${CLS_BATCH_SIZE:-16}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-8}"
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
MEMORY="${MEMORY:-64G}"
CPUS="${CPUS:-8}"

if [ -z "$DATA_ROOT" ]; then
  echo "Set DATA_ROOT to the absolute path of DF40_test_cdf on the cluster." >&2
  exit 1
fi

if [ -z "$FINAL_ROOT" ]; then
  echo "Set FINAL_ROOT to a persistent directory with enough quota." >&2
  exit 1
fi

SBATCH_ARGS=(
  --gres=gpu:1
  --cpus-per-task="$CPUS"
  --mem="$MEMORY"
  --time="$TIME_LIMIT"
)

if [ -n "$PARTITION" ]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi

if [ -n "$ACCOUNT" ]; then
  SBATCH_ARGS+=(--account="$ACCOUNT")
fi

for group in EFS FR FS; do
  sbatch \
    "${SBATCH_ARGS[@]}" \
    --job-name="cdf-${group,,}" \
    --export=ALL,DATA_ROOT="$DATA_ROOT",FINAL_ROOT="$FINAL_ROOT",ENV_NAME="$ENV_NAME",MODEL_DIR="$MODEL_DIR",DEVICE="$DEVICE",FACER_DEVICE="$FACER_DEVICE",CDF_GROUPS="$group",CLS_BATCH_SIZE="$CLS_BATCH_SIZE",PATCH_BATCH_SIZE="$PATCH_BATCH_SIZE" \
    "/home/comp/f2256768/FYP_cdf_cache_pipeline_pack/launch_cdf_cache_cluster.sh"
done
