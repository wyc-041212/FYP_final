#!/bin/bash
#SBATCH --job-name=within_nofr_ablation_cdf
#SBATCH --output=/home/comp/f2256768/%j.out
#SBATCH --error=/home/comp/f2256768/%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/comp/f2256768/FYP_final}"

/bin/bash "${PROJECT_ROOT}/scripts/cluster/run_within_no_fr_main_ablation_with_fr_cdf.sh"
