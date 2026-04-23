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

from main import load_main_runtime  # noqa: E402
from prepare.cache import CompactPatchCache, EmbeddingCache  # noqa: E402
from train.within_train_downstream_head import (  # noqa: E402
    build_branch_features,
    build_prob_map,
    collect_cls_rows,
    configure_group_scheme,
    discover_flat_method_caches,
    iter_regular_cls_rows,
    load_compact_cache,
    predict_hybrid_prob,
    predict_route_aware_meta_fusion,
    route_fake_score,
    set_seed,
)

MODEL_KEYS = ["route_only", "patch_only", "pair_only", "route_meta_fusion"]


@dataclass(frozen=True)
class MethodScoreRow:
    group: str
    method: str
    scores: dict[str, np.ndarray]
    route_top1: np.ndarray


@dataclass(frozen=True)
class RealOnlyScoreRow:
    name: str
    scores: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supplement within main ablation with CDF fake/real metrics only.")
    parser.add_argument("--upstream-checkpoint", type=Path, required=True)
    parser.add_argument("--patch-branch", type=Path, required=True)
    parser.add_argument("--pair-branch", type=Path, required=True)
    parser.add_argument("--route-meta-head", type=Path, required=True)
    parser.add_argument("--head-meta", type=Path, required=True)
    parser.add_argument("--threshold-json", type=Path, required=True)
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--cdf-cls-cache-root", type=Path, required=True)
    parser.add_argument("--cdf-patch-cache-root", type=Path, required=True)
    parser.add_argument("--cdf-split-name", default="DF40_test_cdf")
    parser.add_argument("--cdf-groups", nargs="+", required=True)
    parser.add_argument("--compact-cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--route-batch-size", type=int, default=512)
    parser.add_argument("--max-cdf-fake-per-method", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cdf-real-name", action="append", required=True)
    parser.add_argument("--cdf-real-cls", type=Path, action="append", required=True)
    parser.add_argument("--cdf-real-compact", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_threshold_choice(path: Path, fixed_threshold: float | None = None) -> dict[str, float]:
    if fixed_threshold is not None:
        return {model_key: float(fixed_threshold) for model_key in MODEL_KEYS}
    with path.open() as handle:
        payload = json.load(handle)
    source = payload["threshold_choice"] if "threshold_choice" in payload else payload["models"]
    out = {}
    for model_key in MODEL_KEYS:
        threshold = source[model_key]["threshold"] if "threshold" in source[model_key] else source[model_key]["validation"]["threshold"]
        out[model_key] = float(threshold)
    return out


def score_map(*, route_prob: np.ndarray, patch_prob: np.ndarray, pair_score: np.ndarray, meta_x: np.ndarray, runtime) -> dict[str, np.ndarray]:
    head_bundle = runtime.head_bundle
    route_gating_temperature = float(head_bundle.get("route_gating_temperature", 1.0))
    route_gating_floor = float(head_bundle.get("route_gating_floor", 0.0))
    route_meta_prob, _ = predict_route_aware_meta_fusion(
        head_bundle["route_meta_experts"],
        route_prob,
        meta_x,
        runtime.pair_route_mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )
    return {
        "route_only": route_fake_score(route_prob).astype(np.float32),
        "patch_only": patch_prob.astype(np.float32),
        "pair_only": pair_score.astype(np.float32),
        "route_meta_fusion": route_meta_prob.astype(np.float32),
    }


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
            (str(img_id), cls.astype(np.float32, copy=False), str(group), str(method))
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


def method_score_rows(args: argparse.Namespace, runtime) -> list[MethodScoreRow]:
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
    patch_paths = discover_flat_method_caches(args.cdf_patch_cache_root, args.cdf_split_name, list(args.cdf_groups))
    patch_bundle = runtime.patch_bundle
    pair_bundle = runtime.pair_bundle
    head_bundle = runtime.head_bundle
    route_gating_temperature = float(head_bundle.get("route_gating_temperature", 1.0))
    route_gating_floor = float(head_bundle.get("route_gating_floor", 0.0))

    rows: list[MethodScoreRow] = []
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
        rows.append(
            MethodScoreRow(
                group=str(cache.group[0]),
                method=str(cache.method[0]),
                scores=score_map(
                    route_prob=feats.route_prob,
                    patch_prob=feats.patch_prob,
                    pair_score=feats.pair_score,
                    meta_x=feats.meta_x,
                    runtime=runtime,
                ),
                route_top1=np.asarray([runtime.hybrid_labels[int(i)] for i in np.argmax(feats.route_prob, axis=1)], dtype=object),
            )
        )
    return rows


def real_only_score_rows(args: argparse.Namespace, runtime) -> list[RealOnlyScoreRow]:
    patch_bundle = runtime.patch_bundle
    pair_bundle = runtime.pair_bundle
    head_bundle = runtime.head_bundle
    route_gating_temperature = float(head_bundle.get("route_gating_temperature", 1.0))
    route_gating_floor = float(head_bundle.get("route_gating_floor", 0.0))
    rows: list[RealOnlyScoreRow] = []
    for name, cls_path, compact_path in zip(args.cdf_real_name, args.cdf_real_cls, args.cdf_real_compact, strict=True):
        cls_cache = EmbeddingCache.from_npz(cls_path)
        compact_cache = CompactPatchCache.from_npz(compact_path)
        route_map = build_real_only_route_map(runtime, cls_cache)
        feats = build_branch_features(
            compact_cache,
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
        rows.append(
            RealOnlyScoreRow(
                name=name,
                scores=score_map(
                    route_prob=feats.route_prob,
                    patch_prob=feats.patch_prob,
                    pair_score=feats.pair_score,
                    meta_x=feats.meta_x,
                    runtime=runtime,
                ),
            )
        )
    return rows


def summarize_fake_only(rows: list[MethodScoreRow], model_key: str, threshold: float) -> dict:
    method_rows = []
    pooled_scores = []
    pooled_route_top1: list[str] = []
    for row in rows:
        score = row.scores[model_key]
        pred = (score >= threshold).astype(np.int64)
        method_rows.append(
            {
                "group": row.group,
                "method": row.method,
                "metrics": {
                    "num_images": int(len(score)),
                    "threshold": float(threshold),
                    "mean_fake_prob": float(np.mean(score)),
                    "min_fake_prob": float(np.min(score)),
                    "max_fake_prob": float(np.max(score)),
                    "fake_positive_rate": float(np.mean(pred)),
                },
            }
        )
        pooled_scores.append(score)
        pooled_route_top1.extend(row.route_top1.tolist())
    pooled = np.concatenate(pooled_scores, axis=0)
    return {
        "summary": {
            "num_methods": int(len(method_rows)),
            "num_images": int(len(pooled)),
            "threshold": float(threshold),
            "mean_fake_prob": float(np.mean(pooled)),
            "min_fake_prob": float(np.min(pooled)),
            "max_fake_prob": float(np.max(pooled)),
            "fake_positive_rate": float(np.mean(pooled >= threshold)),
            "route_top1_counts": {
                label: int(sum(1 for x in pooled_route_top1 if x == label))
                for label in sorted(set(pooled_route_top1))
            },
        },
        "methods": sorted(method_rows, key=lambda row: (row["group"], row["method"])),
    }


def summarize_real_only(rows: list[RealOnlyScoreRow], model_key: str, threshold: float) -> dict:
    out = {}
    pooled_scores = []
    for row in rows:
        score = row.scores[model_key]
        pred = (score >= threshold).astype(np.int64)
        out[row.name] = {
            "num_images": int(len(score)),
            "threshold": float(threshold),
            "real_accuracy": float(np.mean(pred == 0)),
            "fake_positive_rate": float(np.mean(pred)),
            "mean_fake_prob": float(np.mean(score)),
        }
        pooled_scores.append(score)
    pooled = np.concatenate(pooled_scores, axis=0)
    pooled_pred = (pooled >= threshold).astype(np.int64)
    out["pooled"] = {
        "num_images": int(len(pooled)),
        "threshold": float(threshold),
        "real_accuracy": float(np.mean(pooled_pred == 0)),
        "fake_positive_rate": float(np.mean(pooled_pred)),
        "mean_fake_prob": float(np.mean(pooled)),
    }
    return out


def main() -> None:
    args = parse_args()
    if not (len(args.cdf_real_name) == len(args.cdf_real_cls) == len(args.cdf_real_compact)):
        raise ValueError("cdf-real names / cls / compact paths must have the same length.")

    set_seed(args.seed)
    thresholds = load_threshold_choice(args.threshold_json, fixed_threshold=args.fixed_threshold)
    runtime = load_main_runtime(
        upstream_checkpoint=args.upstream_checkpoint,
        patch_branch=args.patch_branch,
        pair_branch=args.pair_branch,
        route_meta_head=args.route_meta_head,
        route_batch_size=args.route_batch_size,
        device_name=args.device,
    )
    configure_group_scheme("FR" not in runtime.hybrid_labels, hybrid_labels=list(runtime.hybrid_labels))

    fake_rows = method_score_rows(args, runtime)
    real_rows = real_only_score_rows(args, runtime)

    payload = {
        "mode": "within_main_ablation_cdf_only",
        "config": {
            "pipeline_root": str(ROOT),
            "upstream_checkpoint": str(args.upstream_checkpoint),
            "patch_branch": str(args.patch_branch),
            "pair_branch": str(args.pair_branch),
            "route_meta_head": str(args.route_meta_head),
            "head_meta": str(args.head_meta),
            "threshold_json": str(args.threshold_json),
            "fixed_threshold": args.fixed_threshold,
            "cdf_cls_cache_root": str(args.cdf_cls_cache_root),
            "cdf_patch_cache_root": str(args.cdf_patch_cache_root),
            "cdf_split_name": args.cdf_split_name,
            "cdf_groups": list(args.cdf_groups),
            "compact_cache_dir": str(args.compact_cache_dir),
            "device": str(args.device),
        },
        "threshold_choice": thresholds,
        "models": {},
    }

    csv_rows = []
    for model_key in MODEL_KEYS:
        threshold = float(thresholds[model_key])
        fake_summary = summarize_fake_only(fake_rows, model_key, threshold)
        real_summary = summarize_real_only(real_rows, model_key, threshold)
        cdf_bacc = (fake_summary["summary"]["fake_positive_rate"] + real_summary["pooled"]["real_accuracy"]) / 2.0
        payload["models"][model_key] = {
            "cdf_fake_only": fake_summary,
            "cdf_real_subset": real_summary,
            "cdf": {
                "balanced_accuracy": float(cdf_bacc),
                "fake_accuracy": float(fake_summary["summary"]["fake_positive_rate"]),
                "real_accuracy": float(real_summary["pooled"]["real_accuracy"]),
                "threshold": threshold,
            },
        }
        csv_rows.extend(
            [
                {
                    "model": model_key,
                    "threshold": threshold,
                    "split": "cdf_fake_only",
                    "balanced_accuracy": "",
                    "fake_accuracy": fake_summary["summary"]["fake_positive_rate"],
                    "real_accuracy": "",
                    "fake_positive_rate": fake_summary["summary"]["fake_positive_rate"],
                    "mean_fake_prob": fake_summary["summary"]["mean_fake_prob"],
                },
                {
                    "model": model_key,
                    "threshold": threshold,
                    "split": "cdf_real_subset",
                    "balanced_accuracy": "",
                    "fake_accuracy": "",
                    "real_accuracy": real_summary["pooled"]["real_accuracy"],
                    "fake_positive_rate": real_summary["pooled"]["fake_positive_rate"],
                    "mean_fake_prob": real_summary["pooled"]["mean_fake_prob"],
                },
                {
                    "model": model_key,
                    "threshold": threshold,
                    "split": "cdf",
                    "balanced_accuracy": cdf_bacc,
                    "fake_accuracy": fake_summary["summary"]["fake_positive_rate"],
                    "real_accuracy": real_summary["pooled"]["real_accuracy"],
                    "fake_positive_rate": fake_summary["summary"]["fake_positive_rate"],
                    "mean_fake_prob": "",
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
                "fake_accuracy",
                "real_accuracy",
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
