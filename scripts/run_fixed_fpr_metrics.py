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

from main import build_replay_config, load_head_manifest, load_main_runtime  # noqa: E402
from prepare.cache import CompactPatchCache  # noqa: E402
from train.train_downstream_head import (  # noqa: E402
    BASE_FAKE_GROUPS,
    build_branch_features,
    build_meta_features,
    build_prob_map,
    collect_cls_rows,
    compute_real_pool_region_means,
    discover_flat_method_caches,
    discover_ood_method_entries,
    image_region_delta_features,
    iter_ood_cls_rows,
    iter_regular_cls_rows,
    load_compact_cache,
    pair_score_matrix,
    predict_group_binary_experts,
    predict_hybrid_prob,
    predict_patch_scores,
    predict_route_aware_meta_fusion,
    route_prob_from_map,
    set_seed,
    split_train_indices,
)
from scripts.run_threshold_sweep import MethodProbRow, ood_prob_rows, regular_prob_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute TPR at fixed FPR for replay splits.")
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--upstream-checkpoint", type=Path, required=True)
    parser.add_argument("--patch-branch", type=Path, required=True)
    parser.add_argument("--pair-branch", type=Path, required=True)
    parser.add_argument("--route-meta-head", type=Path, required=True)
    parser.add_argument("--head-meta", type=Path, required=True)
    parser.add_argument("--cls-cache-root", type=Path, default=ROOT / "cache" / "cls")
    parser.add_argument("--patch-cache-root", type=Path, default=ROOT / "cache" / "patch")
    parser.add_argument("--compact-cache-dir", type=Path, default=ROOT / "cache" / "compact")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exclude-group", action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def pooled_scores(rows: list[MethodProbRow], excluded_groups: set[str]) -> tuple[np.ndarray, np.ndarray]:
    kept = [row for row in rows if row.group not in excluded_groups]
    if not kept:
        raise ValueError(f"No rows left after excluding groups: {sorted(excluded_groups)}")
    fake = np.concatenate([row.fake_prob for row in kept], axis=0).astype(np.float64)
    real = np.concatenate([row.real_prob for row in kept], axis=0).astype(np.float64)
    return fake, real


def tpr_at_fixed_fpr(fake_scores: np.ndarray, real_scores: np.ndarray, target_fpr: float) -> dict[str, float]:
    if len(fake_scores) == 0 or len(real_scores) == 0:
        raise ValueError("fake_scores and real_scores must be non-empty")
    thresholds = np.unique(np.concatenate([fake_scores, real_scores], axis=0))
    thresholds = np.sort(thresholds)[::-1]
    best = None
    for thr in thresholds:
        fpr = float(np.mean(real_scores >= thr))
        if fpr <= target_fpr:
            tpr = float(np.mean(fake_scores >= thr))
            candidate = {
                "threshold": float(thr),
                "fpr": fpr,
                "tpr": tpr,
            }
            if best is None:
                best = candidate
                continue
            if candidate["tpr"] > best["tpr"] + 1e-12:
                best = candidate
                continue
            if abs(candidate["tpr"] - best["tpr"]) <= 1e-12 and candidate["threshold"] < best["threshold"]:
                best = candidate
    if best is None:
        thr = float(np.max(thresholds) + 1e-9)
        best = {
            "threshold": thr,
            "fpr": 0.0,
            "tpr": 0.0,
        }
    return best


def summarize_split(rows: list[MethodProbRow], excluded_groups: set[str]) -> dict[str, object]:
    fake_scores, real_scores = pooled_scores(rows, excluded_groups)
    out = {
        "num_fake": int(len(fake_scores)),
        "num_real": int(len(real_scores)),
        "excluded_groups": sorted(excluded_groups),
        "tpr_at_1pct_fpr": tpr_at_fixed_fpr(fake_scores, real_scores, 0.01),
        "tpr_at_0_1pct_fpr": tpr_at_fixed_fpr(fake_scores, real_scores, 0.001),
    }
    return out


def main() -> None:
    args = parse_args()
    manifest = load_head_manifest(args.head_meta)
    cfg = build_replay_config(args, manifest)
    set_seed(cfg.seed)

    runtime = load_main_runtime(
        upstream_checkpoint=args.upstream_checkpoint,
        patch_branch=args.patch_branch,
        pair_branch=args.pair_branch,
        route_meta_head=args.route_meta_head,
        route_batch_size=cfg.route_batch_size,
        device_name=args.device,
    )
    patch_bundle = runtime.patch_bundle
    pair_bundle = runtime.pair_bundle
    head_bundle = runtime.head_bundle
    route_gating_temperature = float(head_bundle.get("route_gating_temperature", cfg.route_gating_temperature))
    route_gating_floor = float(head_bundle.get("route_gating_floor", cfg.route_gating_floor))

    train_real_cache = load_compact_cache(
        cfg.train_real_patch_cache,
        max_rows=cfg.train_real_max,
        seed=cfg.seed,
        compact_cache_dir=cfg.compact_cache_dir,
    )
    test_real_cache = load_compact_cache(
        cfg.test_real_patch_cache,
        max_rows=cfg.eval_real_max,
        seed=cfg.seed,
        compact_cache_dir=cfg.compact_cache_dir,
    )
    real_pool_means = compute_real_pool_region_means(train_real_cache)

    hybrid_model = runtime.hybrid_model
    hybrid_mean = runtime.hybrid_mean
    hybrid_std = runtime.hybrid_std
    hybrid_temperature = runtime.hybrid_temperature
    hybrid_alpha = runtime.hybrid_alpha
    device = runtime.device

    def hybrid_predict(x: np.ndarray) -> np.ndarray:
        return predict_hybrid_prob(
            hybrid_model,
            x,
            hybrid_mean,
            hybrid_std,
            hybrid_temperature,
            hybrid_alpha,
            cfg.route_batch_size,
            device,
        )

    # Build the same runtime maps as replay.
    train_fake_x_parts = []
    train_fake_img_ids = []
    train_fake_methods = []
    train_fake_groups = []
    train_fake_region_vecs = []
    for idx, path in enumerate(discover_flat_method_caches(cfg.patch_cache_root, "DF40_train", list(cfg.patch_train_groups))):
        cache = load_compact_cache(
            path,
            max_rows=cfg.max_train_fake_per_method,
            seed=cfg.seed + idx,
            compact_cache_dir=cfg.compact_cache_dir,
        )
        train_fake_x_parts.append(image_region_delta_features(cache, real_pool_means))
        train_fake_img_ids.append(cache.img_id.astype(object))
        train_fake_methods.append(cache.method.astype(object))
        train_fake_groups.append(cache.group.astype(object))
        train_fake_region_vecs.append(cache.region_vecs.astype(np.float32))
    train_fake_x = np.concatenate(train_fake_x_parts, axis=0)
    train_real_x = image_region_delta_features(train_real_cache, real_pool_means)
    train_x = np.concatenate([train_fake_x, train_real_x], axis=0)
    train_y = np.concatenate(
        [np.ones(len(train_fake_x), dtype=np.int64), np.zeros(len(train_real_x), dtype=np.int64)],
        axis=0,
    )
    train_img_ids = np.concatenate([np.concatenate(train_fake_img_ids), train_real_cache.img_id.astype(object)], axis=0)
    train_methods = np.concatenate([np.concatenate(train_fake_methods), train_real_cache.method.astype(object)], axis=0)
    train_groups = np.concatenate(
        [np.concatenate(train_fake_groups), np.full(len(train_real_x), "REAL", dtype=object)],
        axis=0,
    )
    n_fake_train = len(train_fake_x)
    train_split_keys = train_methods.astype(object).copy()
    train_split_keys[:n_fake_train] = np.asarray(
        [f"{group}::{method}" for group, method in zip(train_groups[:n_fake_train], train_methods[:n_fake_train], strict=True)],
        dtype=object,
    )
    fit_idx, val_idx = split_train_indices(
        train_split_keys,
        cfg.val_ratio,
        cfg.seed,
        n_fake_train,
        split_mode=cfg.val_split_mode,
    )
    train_cls_rows = collect_cls_rows(iter_regular_cls_rows(cfg.cls_cache_root, "DF40_train", cfg.train_real_cls_cache))
    train_route_map = build_prob_map(train_cls_rows, hybrid_predict, cfg.route_batch_size)
    train_route_prob_mat = route_prob_from_map(train_route_map, train_img_ids)
    patch_train_prob = predict_patch_scores(patch_bundle["patch_scaler"], patch_bundle["patch_clf"], train_x)
    patch_train_scores = predict_group_binary_experts(patch_bundle["patch_group_classifiers"], train_x)
    train_pair_scores_mat = pair_score_matrix(
        np.concatenate([np.concatenate(train_fake_region_vecs, axis=0), train_real_cache.region_vecs.astype(np.float32)], axis=0),
        real_pool_means,
        pair_bundle["pair_mean_dirs"],
        pair_bundle["pair_classifiers"],
        pair_bundle["pair_region_idx"],
        pair_bundle["pair_region_names_by_group"],
    )
    train_meta_x = build_meta_features(
        train_route_prob_mat,
        train_pair_scores_mat,
        patch_train_scores,
        patch_train_prob,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )
    val_route_meta_prob, _ = predict_route_aware_meta_fusion(
        head_bundle["route_meta_experts"],
        train_route_prob_mat[val_idx],
        train_meta_x[val_idx],
        runtime.pair_route_mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )

    test_cls_rows = collect_cls_rows(iter_regular_cls_rows(cfg.cls_cache_root, "DF40_test_ff", cfg.test_real_cls_cache))
    ood_cls_rows = collect_cls_rows(iter_ood_cls_rows(cfg.cls_cache_root, "DF40_test_ood"))
    test_route_map = build_prob_map(test_cls_rows, hybrid_predict, cfg.route_batch_size)
    ood_route_map = build_prob_map(ood_cls_rows, hybrid_predict, cfg.route_batch_size)
    eval_groups = list(BASE_FAKE_GROUPS) if "FR" not in runtime.hybrid_labels else ["FS", "FR", "FE", "EFS"]

    test_ff_rows = regular_prob_rows(
        fake_paths=discover_flat_method_caches(cfg.patch_cache_root, "DF40_test_ff", eval_groups),
        real_cache=test_real_cache,
        cfg=cfg,
        route_map=test_route_map,
        real_pool_means=real_pool_means,
        pair_bundle=pair_bundle,
        patch_bundle=patch_bundle,
        head_bundle=head_bundle,
        pair_route_mode=runtime.pair_route_mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )
    ood_rows = ood_prob_rows(
        ood_entries=discover_ood_method_entries(cfg.patch_cache_root, eval_groups),
        cfg=cfg,
        route_map=ood_route_map,
        real_pool_means=real_pool_means,
        pair_bundle=pair_bundle,
        patch_bundle=patch_bundle,
        head_bundle=head_bundle,
        pair_route_mode=runtime.pair_route_mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )

    excluded_groups = set(args.exclude_group)
    payload = {
        "variant_name": args.variant_name,
        "config": {
            "upstream_checkpoint": str(args.upstream_checkpoint),
            "patch_branch": str(args.patch_branch),
            "pair_branch": str(args.pair_branch),
            "route_meta_head": str(args.route_meta_head),
            "head_meta": str(args.head_meta),
            "exclude_group": sorted(excluded_groups),
            "device": args.device,
        },
        "validation": {
            "num_fake": int(np.sum(train_y[val_idx] == 1)),
            "num_real": int(np.sum(train_y[val_idx] == 0)),
            "tpr_at_1pct_fpr": tpr_at_fixed_fpr(
                val_route_meta_prob[train_y[val_idx] == 1],
                val_route_meta_prob[train_y[val_idx] == 0],
                0.01,
            ),
            "tpr_at_0_1pct_fpr": tpr_at_fixed_fpr(
                val_route_meta_prob[train_y[val_idx] == 1],
                val_route_meta_prob[train_y[val_idx] == 0],
                0.001,
            ),
        },
        "test_ff": summarize_split(test_ff_rows, excluded_groups),
        "ood": summarize_split(ood_rows, excluded_groups),
    }

    csv_rows = []
    for split in ("validation", "test_ff", "ood"):
        item = payload[split]
        csv_rows.append(
            {
                "variant_name": args.variant_name,
                "split": split,
                "excluded_groups": ",".join(payload["config"]["exclude_group"]),
                "num_fake": item["num_fake"],
                "num_real": item["num_real"],
                "metric": "TPR@1%FPR",
                "threshold": item["tpr_at_1pct_fpr"]["threshold"],
                "fpr": item["tpr_at_1pct_fpr"]["fpr"],
                "tpr": item["tpr_at_1pct_fpr"]["tpr"],
            }
        )
        csv_rows.append(
            {
                "variant_name": args.variant_name,
                "split": split,
                "excluded_groups": ",".join(payload["config"]["exclude_group"]),
                "num_fake": item["num_fake"],
                "num_real": item["num_real"],
                "metric": "TPR@0.1%FPR",
                "threshold": item["tpr_at_0_1pct_fpr"]["threshold"],
                "fpr": item["tpr_at_0_1pct_fpr"]["fpr"],
                "tpr": item["tpr_at_0_1pct_fpr"]["tpr"],
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant_name", "split", "excluded_groups", "num_fake", "num_real", "metric", "threshold", "fpr", "tpr"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(payload, indent=2))
    print(f"Saved to {args.output_json}")
    print(f"Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
