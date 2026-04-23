#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from main import build_replay_config, load_head_manifest, load_main_runtime  # noqa: E402
from prepare.cache import CompactPatchCache, EmbeddingCache  # noqa: E402
from train.within_train_downstream_head import (  # noqa: E402
    BASE_FAKE_GROUPS,
    build_branch_features,
    build_prob_map,
    build_meta_features,
    collect_cls_rows,
    configure_group_scheme,
    compute_real_pool_region_means,
    discover_flat_method_caches,
    discover_ood_method_entries,
    image_region_delta_features,
    iter_ood_cls_rows,
    iter_regular_cls_rows,
    load_compact_cache,
    metrics_from_binary_prob,
    method_metrics_with_threshold,
    pair_score_matrix,
    predict_group_binary_experts,
    predict_hybrid_prob,
    predict_patch_scores,
    predict_route_aware_meta_fusion,
    route_prob_from_map,
    set_seed,
    split_train_indices,
    summarize_method_rows,
)


@dataclass(frozen=True)
class MethodProbRow:
    group: str
    method: str
    fake_prob: np.ndarray
    real_prob: np.ndarray


@dataclass(frozen=True)
class RealOnlyProbRow:
    name: str
    prob: np.ndarray
    route_fake: np.ndarray
    patch_prob: np.ndarray
    patch_dynamic: np.ndarray
    pair_dynamic: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep fixed thresholds for final replay runs.")
    parser.add_argument("--upstream-checkpoint", type=Path, required=True)
    parser.add_argument("--patch-branch", type=Path, required=True)
    parser.add_argument("--pair-branch", type=Path, required=True)
    parser.add_argument("--route-meta-head", type=Path, required=True)
    parser.add_argument("--head-meta", type=Path, required=True)
    parser.add_argument("--cls-cache-root", type=Path, default=ROOT / "cache" / "cls")
    parser.add_argument("--patch-cache-root", type=Path, default=ROOT / "cache" / "patch")
    parser.add_argument("--compact-cache-dir", type=Path, default=ROOT / "cache" / "compact")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--cdf-real-name",
        action="append",
        default=["Celeb-real", "YouTube-real"],
        help="Display name for a local real-only subset. May be repeated.",
    )
    parser.add_argument(
        "--cdf-real-cls",
        type=Path,
        action="append",
        default=[
            Path("/Users/wuyuchen/Desktop/real/cls_Celeb-real.npz"),
            Path("/Users/wuyuchen/Desktop/real/cls_YouTube-real.npz"),
        ],
        help="CLS npz for a local real-only subset. May be repeated.",
    )
    parser.add_argument(
        "--cdf-real-compact",
        type=Path,
        action="append",
        default=[
            ROOT / "cache" / "compact" / "Users__wuyuchen__Desktop__real__patch_Celeb-real.npz.npz",
            ROOT / "cache" / "compact" / "Users__wuyuchen__Desktop__real__patch_YouTube-real.npz.npz",
        ],
        help="Compact cache for a local real-only subset. May be repeated.",
    )
    return parser.parse_args()


def build_real_only_route_map(
    runtime,
    cls_cache: EmbeddingCache,
) -> dict[str, np.ndarray]:
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

    rows = collect_cls_rows(
        [
            (
                str(img_id),
                cls.astype(np.float32, copy=False),
                str(group),
                str(method),
            )
            for img_id, group, method, cls in zip(
                cls_cache.img_id.tolist(),
                cls_cache.group.tolist(),
                cls_cache.method.tolist(),
                cls_cache.cls,
                strict=True,
            )
        ]
    )
    return build_prob_map(rows, hybrid_predict, runtime.route_batch_size)


def regular_prob_rows(
    *,
    fake_paths: list[Path],
    real_cache: CompactPatchCache,
    cfg,
    route_map: dict[str, np.ndarray],
    real_pool_means: np.ndarray,
    pair_bundle: dict,
    patch_bundle: dict,
    head_bundle: dict,
    pair_route_mode: str,
    route_gating_temperature: float,
    route_gating_floor: float,
) -> list[MethodProbRow]:
    real_features = build_branch_features(
        real_cache,
        route_map=route_map,
        real_pool_means=real_pool_means,
        pair_mean_dirs=pair_bundle["pair_mean_dirs"],
        pair_classifiers=pair_bundle["pair_classifiers"],
        pair_region_idx=pair_bundle["pair_region_idx"],
        pair_region_names_by_group=pair_bundle["pair_region_names_by_group"],
        patch_scaler=patch_bundle["patch_scaler"],
        patch_clf=patch_bundle["patch_clf"],
        patch_group_classifiers=patch_bundle["patch_group_classifiers"],
        pair_route_mode=pair_route_mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )
    real_prob, _ = predict_route_aware_meta_fusion(
        head_bundle["route_meta_experts"],
        real_features.route_prob,
        real_features.meta_x,
        pair_route_mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )
    real_idx = np.arange(len(real_prob))

    rows: list[MethodProbRow] = []
    for idx, path in enumerate(fake_paths):
        cache = load_compact_cache(
            path,
            max_rows=cfg.max_test_fake_per_method,
            seed=cfg.seed + idx,
            compact_cache_dir=cfg.compact_cache_dir,
        )
        fake_features = build_branch_features(
            cache,
            route_map=route_map,
            real_pool_means=real_pool_means,
            pair_mean_dirs=pair_bundle["pair_mean_dirs"],
            pair_classifiers=pair_bundle["pair_classifiers"],
            pair_region_idx=pair_bundle["pair_region_idx"],
            pair_region_names_by_group=pair_bundle["pair_region_names_by_group"],
            patch_scaler=patch_bundle["patch_scaler"],
            patch_clf=patch_bundle["patch_clf"],
            patch_group_classifiers=patch_bundle["patch_group_classifiers"],
            pair_route_mode=pair_route_mode,
            route_gating_temperature=route_gating_temperature,
            route_gating_floor=route_gating_floor,
        )
        fake_prob, _ = predict_route_aware_meta_fusion(
            head_bundle["route_meta_experts"],
            fake_features.route_prob,
            fake_features.meta_x,
            pair_route_mode,
            route_gating_temperature=route_gating_temperature,
            route_gating_floor=route_gating_floor,
        )
        rows.append(
            MethodProbRow(
                group=str(cache.group[0]),
                method=str(cache.method[0]),
                fake_prob=fake_prob.astype(np.float32),
                real_prob=real_prob[real_idx].astype(np.float32),
            )
        )
    return rows


def ood_prob_rows(
    *,
    ood_entries: list[tuple[str, str, Path, Path]],
    cfg,
    route_map: dict[str, np.ndarray],
    real_pool_means: np.ndarray,
    pair_bundle: dict,
    patch_bundle: dict,
    head_bundle: dict,
    pair_route_mode: str,
    route_gating_temperature: float,
    route_gating_floor: float,
) -> list[MethodProbRow]:
    rows: list[MethodProbRow] = []
    for idx, (group, method, fake_path, real_path) in enumerate(ood_entries):
        fake_cache = load_compact_cache(
            fake_path,
            max_rows=cfg.max_ood_fake_per_method,
            seed=cfg.seed + idx,
            compact_cache_dir=cfg.compact_cache_dir,
        )
        real_cache = load_compact_cache(
            real_path,
            max_rows=cfg.max_ood_fake_per_method,
            seed=cfg.seed + 1000 + idx,
            compact_cache_dir=cfg.compact_cache_dir,
        )
        fake_features = build_branch_features(
            fake_cache,
            route_map=route_map,
            real_pool_means=real_pool_means,
            pair_mean_dirs=pair_bundle["pair_mean_dirs"],
            pair_classifiers=pair_bundle["pair_classifiers"],
            pair_region_idx=pair_bundle["pair_region_idx"],
            pair_region_names_by_group=pair_bundle["pair_region_names_by_group"],
            patch_scaler=patch_bundle["patch_scaler"],
            patch_clf=patch_bundle["patch_clf"],
            patch_group_classifiers=patch_bundle["patch_group_classifiers"],
            pair_route_mode=pair_route_mode,
            route_gating_temperature=route_gating_temperature,
            route_gating_floor=route_gating_floor,
        )
        real_features = build_branch_features(
            real_cache,
            route_map=route_map,
            real_pool_means=real_pool_means,
            pair_mean_dirs=pair_bundle["pair_mean_dirs"],
            pair_classifiers=pair_bundle["pair_classifiers"],
            pair_region_idx=pair_bundle["pair_region_idx"],
            pair_region_names_by_group=pair_bundle["pair_region_names_by_group"],
            patch_scaler=patch_bundle["patch_scaler"],
            patch_clf=patch_bundle["patch_clf"],
            patch_group_classifiers=patch_bundle["patch_group_classifiers"],
            pair_route_mode=pair_route_mode,
            route_gating_temperature=route_gating_temperature,
            route_gating_floor=route_gating_floor,
        )
        fake_prob, _ = predict_route_aware_meta_fusion(
            head_bundle["route_meta_experts"],
            fake_features.route_prob,
            fake_features.meta_x,
            pair_route_mode,
            route_gating_temperature=route_gating_temperature,
            route_gating_floor=route_gating_floor,
        )
        real_prob, _ = predict_route_aware_meta_fusion(
            head_bundle["route_meta_experts"],
            real_features.route_prob,
            real_features.meta_x,
            pair_route_mode,
            route_gating_temperature=route_gating_temperature,
            route_gating_floor=route_gating_floor,
        )
        rows.append(
            MethodProbRow(
                group=group,
                method=method,
                fake_prob=fake_prob.astype(np.float32),
                real_prob=real_prob.astype(np.float32),
            )
        )
    return rows


def summarize_prob_rows(rows: list[MethodProbRow], threshold: float) -> dict:
    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "group": row.group,
                "method": row.method,
                "metrics": method_metrics_with_threshold(row.fake_prob, row.real_prob, threshold),
            }
        )
    return summarize_method_rows(out_rows)


def real_only_prob_row(
    *,
    name: str,
    cls_path: Path,
    compact_path: Path,
    runtime,
    real_pool_means: np.ndarray,
    pair_bundle: dict,
    patch_bundle: dict,
    head_bundle: dict,
    pair_route_mode: str,
    route_gating_temperature: float,
    route_gating_floor: float,
) -> RealOnlyProbRow:
    cls_cache = EmbeddingCache.from_npz(cls_path)
    compact_cache = CompactPatchCache.from_npz(compact_path)
    route_map = build_real_only_route_map(runtime, cls_cache)
    feats = build_branch_features(
        compact_cache,
        route_map=route_map,
        real_pool_means=real_pool_means,
        pair_mean_dirs=pair_bundle["pair_mean_dirs"],
        pair_classifiers=pair_bundle["pair_classifiers"],
        pair_region_idx=pair_bundle["pair_region_idx"],
        pair_region_names_by_group=pair_bundle["pair_region_names_by_group"],
        patch_scaler=patch_bundle["patch_scaler"],
        patch_clf=patch_bundle["patch_clf"],
        patch_group_classifiers=patch_bundle["patch_group_classifiers"],
        pair_route_mode=pair_route_mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )
    prob, _ = predict_route_aware_meta_fusion(
        head_bundle["route_meta_experts"],
        feats.route_prob,
        feats.meta_x,
        pair_route_mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )
    return RealOnlyProbRow(
        name=name,
        prob=prob.astype(np.float32),
        route_fake=feats.route_score.astype(np.float32),
        patch_prob=feats.patch_prob.astype(np.float32),
        patch_dynamic=feats.route_patch_score.astype(np.float32),
        pair_dynamic=feats.pair_score.astype(np.float32),
    )


def summarize_real_only(row: RealOnlyProbRow, threshold: float) -> dict:
    pred = (row.prob >= threshold).astype(np.int64)
    fake_positive_rate = float(np.mean(pred))
    return {
        "num_images": int(len(row.prob)),
        "threshold": float(threshold),
        "real_accuracy": float(np.mean(pred == 0)),
        "fake_positive_rate": fake_positive_rate,
        "mean_fake_prob": float(np.mean(row.prob)),
        "mean_route_fake_prob": float(np.mean(row.route_fake)),
        "mean_patch_global_prob": float(np.mean(row.patch_prob)),
        "mean_patch_dynamic_prob": float(np.mean(row.patch_dynamic)),
        "mean_pair_dynamic_prob": float(np.mean(row.pair_dynamic)),
    }


def summarize_real_only_pooled(rows: list[RealOnlyProbRow], threshold: float) -> dict:
    all_prob = np.concatenate([row.prob for row in rows], axis=0)
    all_route = np.concatenate([row.route_fake for row in rows], axis=0)
    all_patch = np.concatenate([row.patch_prob for row in rows], axis=0)
    all_patch_dyn = np.concatenate([row.patch_dynamic for row in rows], axis=0)
    all_pair_dyn = np.concatenate([row.pair_dynamic for row in rows], axis=0)
    pred = (all_prob >= threshold).astype(np.int64)
    return {
        "num_images": int(len(all_prob)),
        "threshold": float(threshold),
        "real_accuracy": float(np.mean(pred == 0)),
        "fake_positive_rate": float(np.mean(pred)),
        "mean_fake_prob": float(np.mean(all_prob)),
        "mean_route_fake_prob": float(np.mean(all_route)),
        "mean_patch_global_prob": float(np.mean(all_patch)),
        "mean_patch_dynamic_prob": float(np.mean(all_patch_dyn)),
        "mean_pair_dynamic_prob": float(np.mean(all_pair_dyn)),
    }


def main() -> None:
    args = parse_args()
    if not (len(args.cdf_real_name) == len(args.cdf_real_cls) == len(args.cdf_real_compact)):
        raise ValueError("cdf-real names / cls / compact paths must have the same length.")

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
    configure_group_scheme("FR" not in runtime.hybrid_labels, hybrid_labels=list(runtime.hybrid_labels))
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

    real_only_rows = [
        real_only_prob_row(
            name=name,
            cls_path=cls_path,
            compact_path=compact_path,
            runtime=runtime,
            real_pool_means=real_pool_means,
            pair_bundle=pair_bundle,
            patch_bundle=patch_bundle,
            head_bundle=head_bundle,
            pair_route_mode=runtime.pair_route_mode,
            route_gating_temperature=route_gating_temperature,
            route_gating_floor=route_gating_floor,
        )
        for name, cls_path, compact_path in zip(args.cdf_real_name, args.cdf_real_cls, args.cdf_real_compact, strict=True)
    ]

    payload = {
        "mode": "threshold_sweep",
        "config": {
            "pipeline_root": str(ROOT),
            "upstream_checkpoint": str(args.upstream_checkpoint),
            "patch_branch": str(args.patch_branch),
            "pair_branch": str(args.pair_branch),
            "route_meta_head": str(args.route_meta_head),
            "head_meta": str(args.head_meta),
            "cls_cache_root": str(args.cls_cache_root),
            "patch_cache_root": str(args.patch_cache_root),
            "compact_cache_dir": str(args.compact_cache_dir),
            "device": str(args.device),
            "thresholds": [float(x) for x in args.thresholds],
        },
        "thresholds": {},
    }
    csv_rows = []
    for threshold in args.thresholds:
        threshold = float(threshold)
        val_metrics = metrics_from_binary_prob(train_y[val_idx], val_route_meta_prob, threshold)
        ff_summary = summarize_prob_rows(test_ff_rows, threshold)
        ood_summary = summarize_prob_rows(ood_rows, threshold)
        real_only_summary = {row.name: summarize_real_only(row, threshold) for row in real_only_rows}
        real_only_summary["pooled"] = summarize_real_only_pooled(real_only_rows, threshold)
        payload["thresholds"][f"{threshold:.1f}"] = {
            "validation": val_metrics,
            "test_ff": ff_summary,
            "ood": ood_summary,
            "cdf_real_subset": real_only_summary,
        }
        csv_rows.extend(
            [
                {
                    "threshold": f"{threshold:.1f}",
                    "split": "validation",
                    "scope": "pooled",
                    "balanced_accuracy": val_metrics["balanced_accuracy"],
                    "accuracy": val_metrics["accuracy"],
                    "fake_accuracy": val_metrics["fake_accuracy"],
                    "real_accuracy": val_metrics["real_accuracy"],
                    "auc": val_metrics["auc"],
                    "ap": val_metrics["ap"],
                    "eer": val_metrics["eer"],
                    "fake_positive_rate": "",
                    "mean_fake_prob": "",
                },
                {
                    "threshold": f"{threshold:.1f}",
                    "split": "test_ff",
                    "scope": "summary",
                    "balanced_accuracy": ff_summary["summary"]["mean_balanced_accuracy"],
                    "accuracy": ff_summary["summary"]["mean_accuracy"],
                    "fake_accuracy": ff_summary["summary"]["mean_fake_accuracy"],
                    "real_accuracy": ff_summary["summary"]["mean_real_accuracy"],
                    "auc": ff_summary["summary"]["mean_auc"],
                    "ap": ff_summary["summary"]["mean_ap"],
                    "eer": ff_summary["summary"]["mean_eer"],
                    "fake_positive_rate": "",
                    "mean_fake_prob": "",
                },
                {
                    "threshold": f"{threshold:.1f}",
                    "split": "ood",
                    "scope": "summary",
                    "balanced_accuracy": ood_summary["summary"]["mean_balanced_accuracy"],
                    "accuracy": ood_summary["summary"]["mean_accuracy"],
                    "fake_accuracy": ood_summary["summary"]["mean_fake_accuracy"],
                    "real_accuracy": ood_summary["summary"]["mean_real_accuracy"],
                    "auc": ood_summary["summary"]["mean_auc"],
                    "ap": ood_summary["summary"]["mean_ap"],
                    "eer": ood_summary["summary"]["mean_eer"],
                    "fake_positive_rate": "",
                    "mean_fake_prob": "",
                },
            ]
        )
        for name, summary in real_only_summary.items():
            csv_rows.append(
                {
                    "threshold": f"{threshold:.1f}",
                    "split": "cdf_real_subset",
                    "scope": name,
                    "balanced_accuracy": "",
                    "accuracy": summary["real_accuracy"],
                    "fake_accuracy": "",
                    "real_accuracy": summary["real_accuracy"],
                    "auc": "",
                    "ap": "",
                    "eer": "",
                    "fake_positive_rate": summary["fake_positive_rate"],
                    "mean_fake_prob": summary["mean_fake_prob"],
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "threshold",
                "split",
                "scope",
                "balanced_accuracy",
                "accuracy",
                "fake_accuracy",
                "real_accuracy",
                "auc",
                "ap",
                "eer",
                "fake_positive_rate",
                "mean_fake_prob",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(payload, indent=2))
    print(f"Saved to {args.output_json}")
    print(f"Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
