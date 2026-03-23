#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_cdf_cls_cluster.sh" "$@"
"${SCRIPT_DIR}/run_cdf_patch_cluster.sh" "$@"
