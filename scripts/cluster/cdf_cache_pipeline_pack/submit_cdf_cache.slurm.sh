#!/bin/bash
#SBATCH --job-name=cdf-cache
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

bash "$SCRIPT_DIR/launch_cdf_cache_cluster.sh" "$@"
