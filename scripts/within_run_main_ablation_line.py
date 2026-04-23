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
    build_meta_features,
    build_prob_map,
    collect_cls_rows,
    configure_group_scheme,
    compute_real_pool_region_means,
    discover_flat_method_caches,
    discover_ood_method_entries,
    dynamic_pair_score,
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
    route_fake_score,
    route_prob_from_map,
    search_threshold,
    set_seed,
    split_train_indices,
    summarize_method_rows,
)

MODEL_KEYS = ["route_only", "patch_only", "pair_only", "route_meta_fusion"]


@dataclass(frozen=True)
class MultiScoreRow:
    group: str
    method: str
    scores: dict[str, np.ndarray]
    real_scores: dict[str, np.ndarray]


@dataclass(frozen=True)
class RealOnlyMultiScore:
    name: str
    scores: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the main branch ablation line with per-model threshold selection.")
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
        "--fixed-threshold",
        type=float,
        default=None,
        help="Use one common decision threshold for all models instead of per-model validation selection.",
    )
    parser.add_argument(
        "--cdf-real-name",
        action="append",
        default=["Celeb-real", "YouTube-real"],
    )
    parser.add_argument(
        "--cdf-real-cls",
        type=Path,
        action="append",
        default=[
            Path("/Users/wuyuchen/Desktop/real/cls_Celeb-real.npz"),
            Path("/Users/wuyuchen/Desktop/real/cls_YouTube-real.npz"),
        ],
    )
    parser.add_argument(
        "--cdf-real-compact",
        type=Path,
        action="append",
        default=[
            ROOT / "cache" / "compact" / "Users__wuyuchen__Desktop__real__patch_Celeb-real.npz.npz",
            ROOT / "cache" / "compact" / "Users__wuyuchen__Desktop__real__patch_YouTube-real.npz.npz",
        ],
    )
    return parser.parse_args()


def build_real_only_route_map(runtime, cls_cache: EmbeddingCache) -> dict[str, np.ndarray]:
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


def score_map(
    *,
    route_prob: np.ndarray,
    patch_prob: np.ndarray,
    pair_score: np.ndarray,
    meta_x: np.ndarray,
    head_bundle: dict,
    pair_route_mode: str,
    route_gating_temperature: float,
    route_gating_floor: float,
) -> dict[str, np.ndarray]:
    route_meta_prob, _ = predict_route_aware_meta_fusion(
        head_bundle["route_meta_experts"],
        route_prob,
        meta_x,
        pair_route_mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )
    return {
        "route_only": route_fake_score(route_prob).astype(np.float32),
        "patch_only": patch_prob.astype(np.float32),
        "pair_only": pair_score.astype(np.float32),
        "route_meta_fusion": route_meta_prob.astype(np.float32),
    }


def build_feature_scores(
    cache,
    *,
    route_map: dict[str, np.ndarray],
    real_pool_means: np.ndarray,
    pair_bundle: dict,
    patch_bundle: dict,
    head_bundle: dict,
    pair_route_mode: str,
    route_gating_temperature: float,
    route_gating_floor: float,
) -> dict[str, np.ndarray]:
    feats = build_branch_features(
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
    return score_map(
        route_prob=feats.route_prob,
        patch_prob=feats.patch_prob,
        pair_score=feats.pair_score,
        meta_x=feats.meta_x,
        head_bundle=head_bundle,
        pair_route_mode=pair_route_mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )


def collect_regular_rows(
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
) -> list[MultiScoreRow]:
    real_scores = build_feature_scores(
        real_cache,
        route_map=route_map,
        real_pool_means=real_pool_means,
        pair_bundle=pair_bundle,
        patch_bundle=patch_bundle,
        head_bundle=head_bundle,
        pair_route_mode=pair_route_mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )
    real_idx = np.arange(len(next(iter(real_scores.values()))))

    rows: list[MultiScoreRow] = []
    for idx, path in enumerate(fake_paths):
        cache = load_compact_cache(
            path,
            max_rows=cfg.max_test_fake_per_method,
            seed=cfg.seed + idx,
            compact_cache_dir=cfg.compact_cache_dir,
        )
        fake_scores = build_feature_scores(
            cache,
            route_map=route_map,
            real_pool_means=real_pool_means,
            pair_bundle=pair_bundle,
            patch_bundle=patch_bundle,
            head_bundle=head_bundle,
            pair_route_mode=pair_route_mode,
            route_gating_temperature=route_gating_temperature,
            route_gating_floor=route_gating_floor,
        )
        rows.append(
            MultiScoreRow(
                group=str(cache.group[0]),
                method=str(cache.method[0]),
                scores=fake_scores,
                real_scores={k: v[real_idx].astype(np.float32) for k, v in real_scores.items()},
            )
        )
    return rows


def collect_ood_rows(
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
) -> list[MultiScoreRow]:
    rows: list[MultiScoreRow] = []
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
        fake_scores = build_feature_scores(
            fake_cache,
            route_map=route_map,
            real_pool_means=real_pool_means,
            pair_bundle=pair_bundle,
            patch_bundle=patch_bundle,
            head_bundle=head_bundle,
            pair_route_mode=pair_route_mode,
            route_gating_temperature=route_gating_temperature,
            route_gating_floor=route_gating_floor,
        )
        real_scores = build_feature_scores(
            real_cache,
            route_map=route_map,
            real_pool_means=real_pool_means,
            pair_bundle=pair_bundle,
            patch_bundle=patch_bundle,
            head_bundle=head_bundle,
            pair_route_mode=pair_route_mode,
            route_gating_temperature=route_gating_temperature,
            route_gating_floor=route_gating_floor,
        )
        rows.append(
            MultiScoreRow(
                group=group,
                method=method,
                scores=fake_scores,
                real_scores=real_scores,
            )
        )
    return rows


def summarize_rows(rows: list[MultiScoreRow], model_key: str, threshold: float) -> dict:
    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "group": row.group,
                "method": row.method,
                "metrics": method_metrics_with_threshold(row.scores[model_key], row.real_scores[model_key], threshold),
            }
        )
    return summarize_method_rows(out_rows)


def collect_real_only_rows(
    *,
    names: list[str],
    cls_paths: list[Path],
    compact_paths: list[Path],
    runtime,
    real_pool_means: np.ndarray,
    pair_bundle: dict,
    patch_bundle: dict,
    head_bundle: dict,
    pair_route_mode: str,
    route_gating_temperature: float,
    route_gating_floor: float,
) -> list[RealOnlyMultiScore]:
    out = []
    for name, cls_path, compact_path in zip(names, cls_paths, compact_paths, strict=True):
        cls_cache = EmbeddingCache.from_npz(cls_path)
        compact_cache = CompactPatchCache.from_npz(compact_path)
        route_map = build_real_only_route_map(runtime, cls_cache)
        scores = build_feature_scores(
            compact_cache,
            route_map=route_map,
            real_pool_means=real_pool_means,
            pair_bundle=pair_bundle,
            patch_bundle=patch_bundle,
            head_bundle=head_bundle,
            pair_route_mode=pair_route_mode,
            route_gating_temperature=route_gating_temperature,
            route_gating_floor=route_gating_floor,
        )
        out.append(RealOnlyMultiScore(name=name, scores=scores))
    return out


def summarize_real_only(rows: list[RealOnlyMultiScore], model_key: str, threshold: float) -> dict:
    per_dataset = {}
    pooled_scores = []
    for row in rows:
        score = row.scores[model_key]
        pred = (score >= threshold).astype(np.int64)
        per_dataset[row.name] = {
            "num_images": int(len(score)),
            "real_accuracy": float(np.mean(pred == 0)),
            "fake_positive_rate": float(np.mean(pred)),
            "mean_fake_prob": float(np.mean(score)),
        }
        pooled_scores.append(score)
    pooled = np.concatenate(pooled_scores, axis=0)
    pooled_pred = (pooled >= threshold).astype(np.int64)
    per_dataset["pooled"] = {
        "num_images": int(len(pooled)),
        "real_accuracy": float(np.mean(pooled_pred == 0)),
        "fake_positive_rate": float(np.mean(pooled_pred)),
        "mean_fake_prob": float(np.mean(pooled)),
    }
    return per_dataset


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

    def hybrid_predict(x: np.ndarray) -> np.ndarray:
        return predict_hybrid_prob(
            runtime.hybrid_model,
            x,
            runtime.hybrid_mean,
            runtime.hybrid_std,
            runtime.hybrid_temperature,
            runtime.hybrid_alpha,
            cfg.route_batch_size,
            runtime.device,
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
    train_groups = np.concatenate([np.concatenate(train_fake_groups), np.full(len(train_real_x), "REAL", dtype=object)], axis=0)
    train_region_vecs = np.concatenate([np.concatenate(train_fake_region_vecs, axis=0), train_real_cache.region_vecs.astype(np.float32)], axis=0)
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
    train_route_prob = route_prob_from_map(train_route_map, train_img_ids)
    patch_train_prob = predict_patch_scores(patch_bundle["patch_scaler"], patch_bundle["patch_clf"], train_x)
    patch_train_scores = predict_group_binary_experts(patch_bundle["patch_group_classifiers"], train_x)
    pair_train_scores = pair_score_matrix(
        train_region_vecs,
        real_pool_means,
        pair_bundle["pair_mean_dirs"],
        pair_bundle["pair_classifiers"],
        pair_bundle["pair_region_idx"],
        pair_bundle["pair_region_names_by_group"],
    )
    pair_train_prob = dynamic_pair_score(
        train_route_prob,
        pair_train_scores,
        runtime.pair_route_mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )
    train_meta_x = build_meta_features(
        train_route_prob,
        pair_train_scores,
        patch_train_scores,
        patch_train_prob,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )
    train_score_map = score_map(
        route_prob=train_route_prob,
        patch_prob=patch_train_prob,
        pair_score=pair_train_prob,
        meta_x=train_meta_x,
        head_bundle=head_bundle,
        pair_route_mode=runtime.pair_route_mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )
    threshold_choice = {}
    for key in MODEL_KEYS:
        if args.fixed_threshold is None:
            threshold_choice[key] = search_threshold(train_y[val_idx], train_score_map[key][val_idx])
            continue
        metrics = metrics_from_binary_prob(train_y[val_idx], train_score_map[key][val_idx], float(args.fixed_threshold))
        threshold_choice[key] = {
            "threshold": float(args.fixed_threshold),
            "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy": metrics["accuracy"],
            "fake_accuracy": metrics["fake_accuracy"],
            "real_accuracy": metrics["real_accuracy"],
            "auc": metrics["auc"],
            "ap": metrics["ap"],
            "eer": metrics["eer"],
        }

    test_cls_rows = collect_cls_rows(iter_regular_cls_rows(cfg.cls_cache_root, "DF40_test_ff", cfg.test_real_cls_cache))
    ood_cls_rows = collect_cls_rows(iter_ood_cls_rows(cfg.cls_cache_root, "DF40_test_ood"))
    test_route_map = build_prob_map(test_cls_rows, hybrid_predict, cfg.route_batch_size)
    ood_route_map = build_prob_map(ood_cls_rows, hybrid_predict, cfg.route_batch_size)

    eval_groups = list(BASE_FAKE_GROUPS) if "FR" not in runtime.hybrid_labels else ["FS", "FR", "FE", "EFS"]
    test_rows = collect_regular_rows(
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
    ood_rows = collect_ood_rows(
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
    real_only_rows = collect_real_only_rows(
        names=args.cdf_real_name,
        cls_paths=args.cdf_real_cls,
        compact_paths=args.cdf_real_compact,
        runtime=runtime,
        real_pool_means=real_pool_means,
        pair_bundle=pair_bundle,
        patch_bundle=patch_bundle,
        head_bundle=head_bundle,
        pair_route_mode=runtime.pair_route_mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )

    payload = {
        "mode": "main_ablation_line",
        "config": {
            "pipeline_root": str(ROOT),
            "upstream_checkpoint": str(args.upstream_checkpoint),
            "patch_branch": str(args.patch_branch),
            "pair_branch": str(args.pair_branch),
            "route_meta_head": str(args.route_meta_head),
            "head_meta": str(args.head_meta),
            "device": str(args.device),
            "threshold_candidates": [float(x) for x in args.thresholds],
            "fixed_threshold": None if args.fixed_threshold is None else float(args.fixed_threshold),
        },
        "threshold_choice": threshold_choice,
        "models": {},
    }
    csv_rows = []
    for model_key in MODEL_KEYS:
        threshold = float(threshold_choice[model_key]["threshold"])
        test_summary = summarize_rows(test_rows, model_key, threshold)
        ood_summary = summarize_rows(ood_rows, model_key, threshold)
        cdf_real_summary = summarize_real_only(real_only_rows, model_key, threshold)
        payload["models"][model_key] = {
            "validation": threshold_choice[model_key],
            "test_ff": test_summary,
            "ood": ood_summary,
            "cdf_real_subset": cdf_real_summary,
        }
        csv_rows.extend(
            [
                {
                    "model": model_key,
                    "threshold": threshold,
                    "split": "validation",
                    "balanced_accuracy": threshold_choice[model_key]["balanced_accuracy"],
                    "accuracy": threshold_choice[model_key]["accuracy"],
                    "fake_accuracy": threshold_choice[model_key]["fake_accuracy"],
                    "real_accuracy": threshold_choice[model_key]["real_accuracy"],
                    "auc": threshold_choice[model_key]["auc"],
                    "ap": threshold_choice[model_key]["ap"],
                    "eer": threshold_choice[model_key]["eer"],
                    "fake_positive_rate": "",
                    "mean_fake_prob": "",
                },
                {
                    "model": model_key,
                    "threshold": threshold,
                    "split": "test_ff",
                    "balanced_accuracy": test_summary["summary"]["mean_balanced_accuracy"],
                    "accuracy": test_summary["summary"]["mean_accuracy"],
                    "fake_accuracy": test_summary["summary"]["mean_fake_accuracy"],
                    "real_accuracy": test_summary["summary"]["mean_real_accuracy"],
                    "auc": test_summary["summary"]["mean_auc"],
                    "ap": test_summary["summary"]["mean_ap"],
                    "eer": test_summary["summary"]["mean_eer"],
                    "fake_positive_rate": "",
                    "mean_fake_prob": "",
                },
                {
                    "model": model_key,
                    "threshold": threshold,
                    "split": "ood",
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
                {
                    "model": model_key,
                    "threshold": threshold,
                    "split": "cdf_real_subset",
                    "balanced_accuracy": "",
                    "accuracy": cdf_real_summary["pooled"]["real_accuracy"],
                    "fake_accuracy": "",
                    "real_accuracy": cdf_real_summary["pooled"]["real_accuracy"],
                    "auc": "",
                    "ap": "",
                    "eer": "",
                    "fake_positive_rate": cdf_real_summary["pooled"]["fake_positive_rate"],
                    "mean_fake_prob": cdf_real_summary["pooled"]["mean_fake_prob"],
                },
            ]
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "threshold",
                "split",
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
