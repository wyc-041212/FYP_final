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
from train.train_downstream_head import (  # noqa: E402
    build_branch_features,
    build_prob_map,
    collect_cls_rows,
    discover_flat_method_caches,
    load_compact_cache,
    predict_hybrid_prob,
    predict_route_aware_meta_fusion,
    set_seed,
)


@dataclass(frozen=True)
class MethodProbRow:
    group: str
    method: str
    fake_prob: np.ndarray
    route_fake: np.ndarray
    patch_prob: np.ndarray
    patch_dynamic: np.ndarray
    pair_dynamic: np.ndarray
    route_top1: np.ndarray


@dataclass(frozen=True)
class RealOnlyProbRow:
    name: str
    prob: np.ndarray
    route_fake: np.ndarray
    patch_prob: np.ndarray
    patch_dynamic: np.ndarray
    pair_dynamic: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep thresholds on the cluster-side CDF cache.")
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
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--cdf-real-name", action="append", required=True)
    parser.add_argument("--cdf-real-cls", type=Path, action="append", required=True)
    parser.add_argument("--cdf-real-compact", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
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


def method_prob_rows(args: argparse.Namespace, runtime) -> list[MethodProbRow]:
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

    cdf_cls_rows = collect_cls_rows(
        iter_regular_cls_rows(
            args.cdf_cls_cache_root,
            args.cdf_split_name,
            None,
        )
    )
    route_map = build_prob_map(cdf_cls_rows, hybrid_predict, runtime.route_batch_size)
    patch_paths = discover_flat_method_caches(args.cdf_patch_cache_root, args.cdf_split_name, list(args.cdf_groups))

    patch_bundle = runtime.patch_bundle
    pair_bundle = runtime.pair_bundle
    head_bundle = runtime.head_bundle
    route_gating_temperature = float(head_bundle.get("route_gating_temperature", 1.0))
    route_gating_floor = float(head_bundle.get("route_gating_floor", 0.0))

    rows: list[MethodProbRow] = []
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
        route_top1 = np.asarray([runtime.hybrid_labels[int(i)] for i in np.argmax(feats.route_prob, axis=1)], dtype=object)
        rows.append(
            MethodProbRow(
                group=str(cache.group[0]),
                method=str(cache.method[0]),
                fake_prob=fake_prob.astype(np.float32),
                route_fake=feats.route_score.astype(np.float32),
                patch_prob=feats.patch_prob.astype(np.float32),
                patch_dynamic=feats.route_patch_score.astype(np.float32),
                pair_dynamic=feats.pair_score.astype(np.float32),
                route_top1=route_top1,
            )
        )
    return rows


def iter_regular_cls_rows(cls_cache_root: Path, split_name: str, real_cls_cache: Path | None):
    from train.train_downstream_head import iter_regular_cls_rows as impl  # noqa: E402

    return impl(cls_cache_root, split_name, real_cls_cache)


def real_only_prob_rows(args: argparse.Namespace, runtime) -> list[RealOnlyProbRow]:
    patch_bundle = runtime.patch_bundle
    pair_bundle = runtime.pair_bundle
    head_bundle = runtime.head_bundle
    route_gating_temperature = float(head_bundle.get("route_gating_temperature", 1.0))
    route_gating_floor = float(head_bundle.get("route_gating_floor", 0.0))
    rows: list[RealOnlyProbRow] = []
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
        prob, _ = predict_route_aware_meta_fusion(
            head_bundle["route_meta_experts"],
            feats.route_prob,
            feats.meta_x,
            runtime.pair_route_mode,
            route_gating_temperature=route_gating_temperature,
            route_gating_floor=route_gating_floor,
        )
        rows.append(
            RealOnlyProbRow(
                name=name,
                prob=prob.astype(np.float32),
                route_fake=feats.route_score.astype(np.float32),
                patch_prob=feats.patch_prob.astype(np.float32),
                patch_dynamic=feats.route_patch_score.astype(np.float32),
                pair_dynamic=feats.pair_score.astype(np.float32),
            )
        )
    return rows


def summarize_fake_only(rows: list[MethodProbRow], threshold: float, hybrid_labels: list[str]) -> dict:
    method_rows = []
    all_fake = []
    all_route = []
    all_patch = []
    all_patch_dyn = []
    all_pair_dyn = []
    all_route_top1: list[str] = []
    for row in rows:
        pred = (row.fake_prob >= threshold).astype(np.int64)
        metrics = {
            "num_images": int(len(row.fake_prob)),
            "threshold": float(threshold),
            "mean_fake_prob": float(np.mean(row.fake_prob)),
            "min_fake_prob": float(np.min(row.fake_prob)),
            "max_fake_prob": float(np.max(row.fake_prob)),
            "fake_positive_rate": float(np.mean(pred)),
            "mean_route_fake_prob": float(np.mean(row.route_fake)),
            "mean_patch_global_prob": float(np.mean(row.patch_prob)),
            "mean_patch_dynamic_prob": float(np.mean(row.patch_dynamic)),
            "mean_pair_dynamic_prob": float(np.mean(row.pair_dynamic)),
        }
        route_top1_counts = {
            label: int(np.sum(row.route_top1 == label))
            for label in hybrid_labels
            if np.any(row.route_top1 == label)
        }
        method_rows.append(
            {
                "group": row.group,
                "method": row.method,
                "metrics": metrics,
                "route_top1_counts": route_top1_counts,
            }
        )
        all_fake.append(row.fake_prob)
        all_route.append(row.route_fake)
        all_patch.append(row.patch_prob)
        all_patch_dyn.append(row.patch_dynamic)
        all_pair_dyn.append(row.pair_dynamic)
        all_route_top1.extend(row.route_top1.tolist())
    fake_all = np.concatenate(all_fake, axis=0)
    route_all = np.concatenate(all_route, axis=0)
    patch_all = np.concatenate(all_patch, axis=0)
    patch_dyn_all = np.concatenate(all_patch_dyn, axis=0)
    pair_dyn_all = np.concatenate(all_pair_dyn, axis=0)
    summary = {
        "num_methods": int(len(method_rows)),
        "num_images": int(len(fake_all)),
        "threshold": float(threshold),
        "mean_fake_prob": float(np.mean(fake_all)),
        "min_fake_prob": float(np.min(fake_all)),
        "max_fake_prob": float(np.max(fake_all)),
        "fake_positive_rate": float(np.mean(fake_all >= threshold)),
        "mean_route_fake_prob": float(np.mean(route_all)),
        "mean_patch_global_prob": float(np.mean(patch_all)),
        "mean_patch_dynamic_prob": float(np.mean(patch_dyn_all)),
        "mean_pair_dynamic_prob": float(np.mean(pair_dyn_all)),
        "route_top1_counts": {
            label: int(sum(1 for x in all_route_top1 if x == label))
            for label in hybrid_labels
            if any(x == label for x in all_route_top1)
        },
    }
    return {"summary": summary, "methods": sorted(method_rows, key=lambda row: (row["group"], row["method"]))}


def summarize_real_only(row: RealOnlyProbRow, threshold: float) -> dict:
    pred = (row.prob >= threshold).astype(np.int64)
    return {
        "num_images": int(len(row.prob)),
        "threshold": float(threshold),
        "real_accuracy": float(np.mean(pred == 0)),
        "fake_positive_rate": float(np.mean(pred)),
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

    set_seed(args.seed)
    runtime = load_main_runtime(
        upstream_checkpoint=args.upstream_checkpoint,
        patch_branch=args.patch_branch,
        pair_branch=args.pair_branch,
        route_meta_head=args.route_meta_head,
        route_batch_size=args.route_batch_size,
        device_name=args.device,
    )

    fake_rows = method_prob_rows(args, runtime)
    if not fake_rows:
        raise SystemExit("No CDF patch cache files found to evaluate.")
    real_rows = real_only_prob_rows(args, runtime)

    payload = {
        "mode": "cdf_threshold_sweep",
        "config": {
            "pipeline_root": str(ROOT),
            "upstream_checkpoint": str(args.upstream_checkpoint),
            "patch_branch": str(args.patch_branch),
            "pair_branch": str(args.pair_branch),
            "route_meta_head": str(args.route_meta_head),
            "cdf_cls_cache_root": str(args.cdf_cls_cache_root),
            "cdf_patch_cache_root": str(args.cdf_patch_cache_root),
            "cdf_split_name": args.cdf_split_name,
            "cdf_groups": list(args.cdf_groups),
            "compact_cache_dir": str(args.compact_cache_dir),
            "device": str(args.device),
            "thresholds": [float(x) for x in args.thresholds],
        },
        "thresholds": {},
    }

    csv_rows = []
    for threshold in args.thresholds:
        threshold = float(threshold)
        fake_summary = summarize_fake_only(fake_rows, threshold, runtime.hybrid_labels)
        real_summary = {row.name: summarize_real_only(row, threshold) for row in real_rows}
        real_summary["pooled"] = summarize_real_only_pooled(real_rows, threshold)
        payload["thresholds"][f"{threshold:.1f}"] = {
            "cdf_fake_only": fake_summary,
            "cdf_real_subset": real_summary,
        }
        csv_rows.append(
            {
                "threshold": f"{threshold:.1f}",
                "split": "cdf_fake_only",
                "scope": "summary",
                "num_images": fake_summary["summary"]["num_images"],
                "mean_fake_prob": fake_summary["summary"]["mean_fake_prob"],
                "fake_positive_rate": fake_summary["summary"]["fake_positive_rate"],
                "real_accuracy": "",
            }
        )
        for name, summary in real_summary.items():
            csv_rows.append(
                {
                    "threshold": f"{threshold:.1f}",
                    "split": "cdf_real_subset",
                    "scope": name,
                    "num_images": summary["num_images"],
                    "mean_fake_prob": summary["mean_fake_prob"],
                    "fake_positive_rate": summary["fake_positive_rate"],
                    "real_accuracy": summary["real_accuracy"],
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
                "num_images",
                "mean_fake_prob",
                "fake_positive_rate",
                "real_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(payload, indent=2))
    print(f"Saved to {args.output_json}")
    print(f"Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
