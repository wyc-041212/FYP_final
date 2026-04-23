#!/bin/bash
#SBATCH --job-name=transfer_patch
#SBATCH --output=transfer_patch_%j.out
#SBATCH --error=transfer_patch_%j.err
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodelist=gpu20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G

set -euo pipefail

SRC_DIR="/tmp/$USER/cdf_cache/cache_clip_patch/DF40_test_cdf/EFS"
DST_NODE="gpu10"
DST_DIR="/tmp/$USER/cdf_cache/cache_clip_patch/DF40_test_cdf/EFS"

echo "[INFO] Start transfer from $SRC_DIR to $DST_NODE:$DST_DIR"

# 确保目标目录存在
ssh $USER@$DST_NODE "mkdir -p $DST_DIR"

# 用 rsync 支持断点续传（推荐），如无 rsync 可改为 scp -r
if command -v rsync >/dev/null 2>&1; then
  rsync -avP "$SRC_DIR/" "$USER@$DST_NODE:$DST_DIR/"
else
  scp -r "$SRC_DIR" "$USER@$DST_NODE:$(dirname $DST_DIR)"
fi

echo "[INFO] Transfer finished."