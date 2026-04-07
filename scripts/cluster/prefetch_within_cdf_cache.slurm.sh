#!/bin/bash
#SBATCH --job-name=within_cdf_prefetch
#SBATCH --output=/home/comp/f2256768/%j.out
#SBATCH --error=/home/comp/f2256768/%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

SOURCE_NODE="${SOURCE_NODE:-gpu10}"
KEY_FILE="${KEY_FILE:-/home/comp/f2256768/.ssh/id_rsa}"
RCLONE_BIN="${RCLONE_BIN:-/home/comp/f2256768/bin/rclone}"

copy_if_missing() {
  local src="$1"
  local dst_parent="$2"
  local dst_name="$3"
  local marker_rel="$4"
  local dst="${dst_parent}/${dst_name}"
  mkdir -p "${dst_parent}"
  if [ -e "${dst}/${marker_rel}" ]; then
    echo "[SKIP] ${dst}/${marker_rel}"
    return
  fi
  echo "[COPY] ${SOURCE_NODE}:${src} -> ${dst_parent}/"
  rm -rf "${dst}"
  if command -v rsync >/dev/null 2>&1; then
    mkdir -p "${dst}"
    rsync -aP -e "ssh -F /dev/null" "${SOURCE_NODE}:${src}/" "${dst}/"
  elif [ -x "${RCLONE_BIN}" ]; then
    mkdir -p "${dst}"
    "${RCLONE_BIN}" copy \
      ":sftp,host=${SOURCE_NODE},user=${USER},key_file=${KEY_FILE}:${src}" \
      "${dst}" \
      --progress
  else
    scp -F /dev/null -q -r "${SOURCE_NODE}:${src}" "${dst_parent}/"
  fi
}

echo "[INFO] node=$(hostname)"
echo "[INFO] source_node=${SOURCE_NODE}"

copy_if_missing "/tmp/f2256768/cdf_cache/cache_clip_patch" "/tmp/f2256768/cdf_cache" "cache_clip_patch" "DF40_test_cdf/EFS/patch_DiT.npz"
copy_if_missing "/tmp/f2256768/fyp_final_compact_cdf" "/tmp/f2256768" "fyp_final_compact_cdf" "tmp__f2256768__cdf_cache__cache_clip_patch__DF40_test_cdf__EFS__patch_DiT.npz.npz"
copy_if_missing "/tmp/celebdf_real_clip" "/tmp" "celebdf_real_clip" "cache_clip/Celeb-DF-v2/real/cls_Celeb-real.npz"
copy_if_missing "/tmp/f2256768/fyp_final_compact_celebdf_real" "/tmp/f2256768" "fyp_final_compact_celebdf_real" "tmp__celebdf_real_clip__cache_clip_patch__Celeb-DF-v2__real__patch_Celeb-real.npz.npz"

echo "[DONE] within CDF prefetch finished."
