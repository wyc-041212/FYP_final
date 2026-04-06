#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from main import load_main_runtime  # noqa: E402
from train.train_downstream_head import (  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cluster-side CDF replay evaluation.")
    parser.add_argument("--upstream-checkpoint", type=Path, required=True)
    parser.add_argument("--patch-branch", type=Path, required=True)
    parser.add_argument("--pair-branch", type=Path, required=True)
    parser.add_argument("--route-meta-head", type=Path, required=True)
    parser.add_argument("--cdf-cls-cache-root", type=Path, required=True)
    parser.add_argument("--cdf-patch-cache-root", type=Path, required=True)
    parser.add_argument("--cdf-split-name", default="DF40_test_cdf")
    parser.add_argument("--cdf-groups", nargs="+", default=["EFS", "FR", "FS"])
    parser.add_argument("--compact-cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--route-batch-size", type=int, default=2048)
    parser.add_argument("--max-cdf-fake-per-method", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    runtime = load_main_runtime(
        upstream_checkpoint=args.upstream_checkpoint,
        patch_branch=args.patch_branch,
        pair_branch=args.pair_branch,
        route_meta_head=args.route_meta_head,
        route_batch_size=args.route_batch_size,
        device_name=args.device,
    )

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

    cdf_cls_rows = collect_cls_rows(iter_regular_cls_rows(args.cdf_cls_cache_root, args.cdf_split_name, None))
    route_map = build_prob_map(cdf_cls_rows, hybrid_predict, runtime.route_batch_size)

    patch_bundle = runtime.patch_bundle
    pair_bundle = runtime.pair_bundle
    head_bundle = runtime.head_bundle
    route_gating_temperature = float(head_bundle.get("route_gating_temperature", 1.0))
    route_gating_floor = float(head_bundle.get("route_gating_floor", 0.0))

    method_rows = []
    all_fake_prob = []
    all_route_fake = []
    all_patch_global = []
    all_patch_dynamic = []
    all_pair_dynamic = []
    global_route_top1: list[str] = []

    patch_paths = discover_flat_method_caches(args.cdf_patch_cache_root, args.cdf_split_name, list(args.cdf_groups))
    for idx, path in enumerate(patch_paths):
        cache = load_compact_cache(
            path,
            max_rows=args.max_cdf_fake_per_method,
            seed=args.seed + idx,
            compact_cache_dir=args.compact_cache_dir,
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
            route_gating_temperature=route_gating_temperature,
            route_gating_floor=route_gating_floor,
        )
        fake_prob, _ = predict_route_aware_meta_fusion(
            head_bundle["route_meta_experts"],
            feats.route_prob,
            feats.meta_x,
            runtime.pair_route_mode,
            route_gating_temperature=route_gating_temperature,
            route_gating_floor=route_gating_floor,
        )
        pred = (fake_prob >= runtime.route_meta_threshold).astype(np.int64)
        route_top1 = np.asarray([runtime.hybrid_labels[int(i)] for i in np.argmax(feats.route_prob, axis=1)], dtype=object)
        method_rows.append(
            {
                "group": str(cache.group[0]),
                "method": str(cache.method[0]),
                "metrics": {
                    "num_images": int(len(fake_prob)),
                    "threshold": float(runtime.route_meta_threshold),
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
                    for label in runtime.hybrid_labels
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
            for label in runtime.hybrid_labels
            if any(x == label for x in global_route_top1)
        },
    }

    payload = {
        "mode": "cdf_replay_eval",
        "config": {
            "project_root": str(ROOT),
            "upstream_checkpoint": str(args.upstream_checkpoint),
            "patch_branch": str(args.patch_branch),
            "pair_branch": str(args.pair_branch),
            "route_meta_head": str(args.route_meta_head),
            "cdf_cls_cache_root": str(args.cdf_cls_cache_root),
            "cdf_patch_cache_root": str(args.cdf_patch_cache_root),
            "cdf_split_name": args.cdf_split_name,
            "cdf_groups": list(args.cdf_groups),
            "compact_cache_dir": str(args.compact_cache_dir),
            "device": args.device,
            "route_gating_temperature": route_gating_temperature,
            "route_gating_floor": route_gating_floor,
        },
        "cdf": {
            "route_meta_fusion": {
                "summary": summary,
                "methods": sorted(method_rows, key=lambda row: (row["group"], row["method"])),
            }
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "method",
                "num_images",
                "threshold",
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
            writer.writerow({"group": row["group"], "method": row["method"], **row["metrics"]})
    print(json.dumps(payload, indent=2))
    print(f"Saved to {args.output_json}")
    print(f"Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
