#!/bin/bash
#SBATCH --job-name=cdf_infer
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=long
#SBATCH --nodelist=gpu10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G

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
UPSTREAM_CHECKPOINT="${UPSTREAM_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/upstream/checkpoint_best_hybrid_manifold.pt}"
PATCH_BRANCH="${PATCH_BRANCH:-${PROJECT_ROOT}/checkpoints/downstream/patch_branch.joblib}"
PAIR_BRANCH="${PAIR_BRANCH:-${PROJECT_ROOT}/checkpoints/downstream/pair_branch.joblib}"
ROUTE_META_HEAD="${ROUTE_META_HEAD:-${PROJECT_ROOT}/checkpoints/heads/route_meta_head.joblib}"

CDF_CLS_CACHE_ROOT="${CDF_CLS_CACHE_ROOT:-/home/comp/f2256768/cdf_cache/cache_clip}"
CDF_PATCH_CACHE_ROOT="${CDF_PATCH_CACHE_ROOT:-/tmp/f2256768/cdf_cache/cache_clip_patch}"
CDF_SPLIT_NAME="${CDF_SPLIT_NAME:-DF40_test_cdf}"
CDF_GROUPS="${CDF_GROUPS:-EFS FR FS FE}"
CDF_GROUPS="${CDF_GROUPS//,/ }"

SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
ROUTE_BATCH_SIZE="${ROUTE_BATCH_SIZE:-2048}"
MAX_CDF_FAKE_PER_METHOD="${MAX_CDF_FAKE_PER_METHOD:-0}"
COMPACT_CACHE_DIR="${COMPACT_CACHE_DIR:-/tmp/f2256768/fyp_final_compact_cdf}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/cdf}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_DIR}/cdf_infer.json}"
OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_DIR}/cdf_infer.csv}"

mkdir -p "${OUTPUT_DIR}" "${COMPACT_CACHE_DIR}"

if [ ! -d "${CDF_CLS_CACHE_ROOT}/${CDF_SPLIT_NAME}" ]; then
  echo "Missing CDF CLS split root: ${CDF_CLS_CACHE_ROOT}/${CDF_SPLIT_NAME}" >&2
  exit 1
fi

if [ ! -d "${CDF_PATCH_CACHE_ROOT}/${CDF_SPLIT_NAME}" ]; then
  echo "Missing CDF PATCH split root: ${CDF_PATCH_CACHE_ROOT}/${CDF_SPLIT_NAME}" >&2
  exit 1
fi

export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::RuntimeWarning}"
export PROJECT_ROOT
export UPSTREAM_CHECKPOINT
export PATCH_BRANCH
export PAIR_BRANCH
export ROUTE_META_HEAD
export CDF_CLS_CACHE_ROOT
export CDF_PATCH_CACHE_ROOT
export CDF_SPLIT_NAME
export CDF_GROUPS
export SEED
export DEVICE
export ROUTE_BATCH_SIZE
export MAX_CDF_FAKE_PER_METHOD
export COMPACT_CACHE_DIR
export OUTPUT_JSON
export OUTPUT_CSV

echo "[INFO] node=$(hostname)"
echo "[INFO] cdf_cls_cache_root=${CDF_CLS_CACHE_ROOT}"
echo "[INFO] cdf_patch_cache_root=${CDF_PATCH_CACHE_ROOT}"
echo "[INFO] cdf_split_name=${CDF_SPLIT_NAME}"
echo "[INFO] cdf_groups=${CDF_GROUPS}"
echo "[INFO] device=${DEVICE}"
echo "[INFO] compact_cache_dir=${COMPACT_CACHE_DIR}"
echo "[INFO] output_json=${OUTPUT_JSON}"
echo "[INFO] output_csv=${OUTPUT_CSV}"

conda run --no-capture-output -n "${ENV_NAME}" python - <<'PY'
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

project_root = Path(os.environ["PROJECT_ROOT"])
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from main import load_main_runtime  # noqa: E402
from train.train_downstream_head import (  # noqa: E402
    HYBRID_LABELS,
    build_branch_features,
    build_prob_map,
    collect_cls_rows,
    discover_flat_method_caches,
    iter_regular_cls_rows,
    load_compact_cache,
    predict_hybrid_prob,
    predict_route_aware_meta_fusion,
    set_seed,
)

upstream_checkpoint = Path(os.environ["UPSTREAM_CHECKPOINT"])
patch_branch = Path(os.environ["PATCH_BRANCH"])
pair_branch = Path(os.environ["PAIR_BRANCH"])
route_meta_head = Path(os.environ["ROUTE_META_HEAD"])
cdf_cls_cache_root = Path(os.environ["CDF_CLS_CACHE_ROOT"])
cdf_patch_cache_root = Path(os.environ["CDF_PATCH_CACHE_ROOT"])
cdf_split_name = os.environ["CDF_SPLIT_NAME"]
cdf_groups = [g for g in os.environ["CDF_GROUPS"].split() if g]
seed = int(os.environ["SEED"])
device = os.environ["DEVICE"]
route_batch_size = int(os.environ["ROUTE_BATCH_SIZE"])
max_cdf_fake_per_method = int(os.environ["MAX_CDF_FAKE_PER_METHOD"])
compact_cache_dir = Path(os.environ["COMPACT_CACHE_DIR"])
output_json = Path(os.environ["OUTPUT_JSON"])
output_csv = Path(os.environ["OUTPUT_CSV"])

set_seed(seed)
runtime = load_main_runtime(
    upstream_checkpoint=upstream_checkpoint,
    patch_branch=patch_branch,
    pair_branch=pair_branch,
    route_meta_head=route_meta_head,
    route_batch_size=route_batch_size,
    device_name=device,
)

patch_bundle = runtime.patch_bundle
pair_bundle = runtime.pair_bundle
head_bundle = runtime.head_bundle

cdf_cls_rows = collect_cls_rows(iter_regular_cls_rows(cdf_cls_cache_root, cdf_split_name, None))

def hybrid_predict(x: np.ndarray) -> np.ndarray:
    return predict_hybrid_prob(
        runtime.hybrid_model,
        x,
        runtime.hybrid_mean,
        runtime.hybrid_std,
        runtime.hybrid_temperature,
        runtime.hybrid_alpha,
        runtime.route_batch_size,
        runtime.device,
    )

route_map = build_prob_map(cdf_cls_rows, hybrid_predict, runtime.route_batch_size)

method_rows = []
all_fake_prob = []
all_route_fake = []
all_patch_global = []
all_patch_dynamic = []
all_pair_dynamic = []
global_route_top1 = []

patch_paths = discover_flat_method_caches(cdf_patch_cache_root, cdf_split_name, cdf_groups)
for idx, path in enumerate(patch_paths):
    cache = load_compact_cache(
        path,
        max_rows=max_cdf_fake_per_method,
        seed=seed + idx,
        compact_cache_dir=compact_cache_dir,
    )
    feats = build_branch_features(
        cache,
        route_map=route_map,
        real_pool_means=pair_bundle["real_pool_means"],
        pair_mean_dirs=pair_bundle["pair_mean_dirs"],
        pair_classifiers=pair_bundle["pair_classifiers"],
        pair_region_idx=pair_bundle["pair_region_idx"],
        pair_region_names_by_group=pair_bundle["pair_region_names_by_group"],
        patch_scaler=patch_bundle["patch_scaler"],
        patch_clf=patch_bundle["patch_clf"],
        patch_group_classifiers=patch_bundle["patch_group_classifiers"],
        pair_route_mode=runtime.pair_route_mode,
    )
    fake_prob, _ = predict_route_aware_meta_fusion(
        head_bundle["route_meta_experts"],
        feats.route_prob,
        feats.meta_x,
        runtime.pair_route_mode,
    )
    pred = (fake_prob >= runtime.route_meta_threshold).astype(np.int64)
    route_top1 = np.asarray([HYBRID_LABELS[int(i)] for i in np.argmax(feats.route_prob, axis=1)], dtype=object)

    method_rows.append(
        {
            "group": str(cache.group[0]),
            "method": str(cache.method[0]),
            "metrics": {
                "num_images": int(len(fake_prob)),
                "mean_fake_prob": float(np.mean(fake_prob)),
                "min_fake_prob": float(np.min(fake_prob)),
                "max_fake_prob": float(np.max(fake_prob)),
                "fake_positive_rate": float(np.mean(pred)),
                "mean_route_fake_prob": float(np.mean(feats.route_score)),
                "mean_patch_global_prob": float(np.mean(feats.patch_prob)),
                "mean_patch_dynamic_prob": float(np.mean(feats.route_patch_score)),
                "mean_pair_dynamic_prob": float(np.mean(feats.pair_score)),
            },
            "route_top1_counts": {
                label: int(np.sum(route_top1 == label))
                for label in HYBRID_LABELS
                if np.any(route_top1 == label)
            },
        }
    )

    all_fake_prob.append(fake_prob.astype(np.float32))
    all_route_fake.append(feats.route_score.astype(np.float32))
    all_patch_global.append(feats.patch_prob.astype(np.float32))
    all_patch_dynamic.append(feats.route_patch_score.astype(np.float32))
    all_pair_dynamic.append(feats.pair_score.astype(np.float32))
    global_route_top1.extend(route_top1.tolist())

if not method_rows:
    raise SystemExit("No CDF patch cache files found to evaluate.")

fake_prob_all = np.concatenate(all_fake_prob, axis=0)
route_fake_all = np.concatenate(all_route_fake, axis=0)
patch_global_all = np.concatenate(all_patch_global, axis=0)
patch_dynamic_all = np.concatenate(all_patch_dynamic, axis=0)
pair_dynamic_all = np.concatenate(all_pair_dynamic, axis=0)

summary = {
    "num_methods": int(len(method_rows)),
    "num_images": int(len(fake_prob_all)),
    "threshold": float(runtime.route_meta_threshold),
    "mean_fake_prob": float(np.mean(fake_prob_all)),
    "min_fake_prob": float(np.min(fake_prob_all)),
    "max_fake_prob": float(np.max(fake_prob_all)),
    "fake_positive_rate": float(np.mean(fake_prob_all >= runtime.route_meta_threshold)),
    "mean_route_fake_prob": float(np.mean(route_fake_all)),
    "mean_patch_global_prob": float(np.mean(patch_global_all)),
    "mean_patch_dynamic_prob": float(np.mean(patch_dynamic_all)),
    "mean_pair_dynamic_prob": float(np.mean(pair_dynamic_all)),
    "route_top1_counts": {
        label: int(sum(1 for x in global_route_top1 if x == label))
        for label in HYBRID_LABELS
        if any(x == label for x in global_route_top1)
    },
}

payload = {
    "mode": "cdf_fake_only_inference",
    "config": {
        "project_root": str(project_root),
        "upstream_checkpoint": str(upstream_checkpoint),
        "patch_branch": str(patch_branch),
        "pair_branch": str(pair_branch),
        "route_meta_head": str(route_meta_head),
        "cdf_cls_cache_root": str(cdf_cls_cache_root),
        "cdf_patch_cache_root": str(cdf_patch_cache_root),
        "cdf_split_name": cdf_split_name,
        "cdf_groups": cdf_groups,
        "compact_cache_dir": str(compact_cache_dir),
        "device": device,
    },
    "cdf": {
        "route_meta_fusion": {
            "summary": summary,
            "methods": sorted(method_rows, key=lambda row: (row["group"], row["method"])),
        }
    },
}

output_json.parent.mkdir(parents=True, exist_ok=True)
output_json.write_text(json.dumps(payload, indent=2))

output_csv.parent.mkdir(parents=True, exist_ok=True)
with output_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "group",
            "method",
            "num_images",
            "mean_fake_prob",
            "min_fake_prob",
            "max_fake_prob",
            "fake_positive_rate",
            "mean_route_fake_prob",
            "mean_patch_global_prob",
            "mean_patch_dynamic_prob",
            "mean_pair_dynamic_prob",
        ],
    )
    writer.writeheader()
    for row in payload["cdf"]["route_meta_fusion"]["methods"]:
        metrics = row["metrics"]
        writer.writerow(
            {
                "group": row["group"],
                "method": row["method"],
                **metrics,
            }
        )

print(json.dumps(payload, indent=2))
print(f"Saved to {output_json}")
print(f"Saved CSV to {output_csv}")
PY
