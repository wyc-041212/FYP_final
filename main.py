#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from train.train_downstream_head import (  # noqa: E402
    HYBRID_LABELS,
    build_meta_features,
    build_prob_map,
    collect_cls_rows,
    compute_real_pool_region_means,
    discover_flat_method_caches,
    discover_ood_method_entries,
    dynamic_expert_score,
    evaluate_ood_route_meta,
    evaluate_regular_route_meta,
    iter_ood_cls_rows,
    iter_regular_cls_rows,
    load_hybrid_checkpoint,
    load_compact_cache,
    metrics_from_binary_prob,
    pair_score_matrix,
    predict_group_binary_experts,
    predict_hybrid_prob,
    predict_route_aware_meta_fusion,
    route_fake_score,
    route_prob_from_map,
    set_seed,
    split_train_indices,
)
from train.train_downstream_head import image_region_delta_features, predict_patch_scores  # noqa: E402
from prepare.cache import CompactPatchCache, EmbeddingCache, PatchCache, compute_region_means  # noqa: E402
from utils.device import resolve_device_name, resolve_torch_device  # noqa: E402


@dataclass(frozen=True)
class BundleArtifacts:
    patch_bundle: dict
    pair_bundle: dict
    head_bundle: dict


@dataclass(frozen=True)
class MainRuntime:
    artifacts: BundleArtifacts
    hybrid_model: object
    hybrid_mean: np.ndarray
    hybrid_std: np.ndarray
    hybrid_temperature: float
    hybrid_alpha: float
    device: torch.device
    route_batch_size: int

    @property
    def patch_bundle(self) -> dict:
        return self.artifacts.patch_bundle

    @property
    def pair_bundle(self) -> dict:
        return self.artifacts.pair_bundle

    @property
    def head_bundle(self) -> dict:
        return self.artifacts.head_bundle

    @property
    def region_names(self) -> list[str]:
        return [str(x) for x in self.pair_bundle["region_names"].tolist()]

    @property
    def route_meta_threshold(self) -> float:
        return float(self.head_bundle["route_meta_threshold"])

    @property
    def pair_route_mode(self) -> str:
        return str(self.head_bundle["pair_route_mode"])


@dataclass(frozen=True)
class SamplePrediction:
    route_prob: np.ndarray
    route_fake: np.ndarray
    route_top1: np.ndarray
    patch_global_prob: np.ndarray
    patch_scores: np.ndarray
    patch_dynamic: np.ndarray
    pair_scores: np.ndarray
    pair_dynamic: np.ndarray
    route_meta_prob: np.ndarray
    pred: np.ndarray


@dataclass(frozen=True)
class SampleBatch:
    sample_name: str
    rows: list[dict]
    cls_cache: EmbeddingCache
    patch_cache: CompactPatchCache


def load_bundle_artifacts(
    *,
    patch_branch: Path,
    pair_branch: Path,
    route_meta_head: Path,
) -> BundleArtifacts:
    return BundleArtifacts(
        patch_bundle=joblib.load(patch_branch),
        pair_bundle=joblib.load(pair_branch),
        head_bundle=joblib.load(route_meta_head),
    )


def load_main_runtime(
    *,
    upstream_checkpoint: Path,
    patch_branch: Path,
    pair_branch: Path,
    route_meta_head: Path,
    route_batch_size: int = 2048,
    device_name: str | None = None,
) -> MainRuntime:
    artifacts = load_bundle_artifacts(
        patch_branch=patch_branch,
        pair_branch=pair_branch,
        route_meta_head=route_meta_head,
    )
    device = resolve_torch_device(device_name)
    hybrid_model, hybrid_mean, hybrid_std, hybrid_temperature, hybrid_alpha = load_hybrid_checkpoint(
        upstream_checkpoint,
        device,
    )
    return MainRuntime(
        artifacts=artifacts,
        hybrid_model=hybrid_model,
        hybrid_mean=hybrid_mean,
        hybrid_std=hybrid_std,
        hybrid_temperature=hybrid_temperature,
        hybrid_alpha=hybrid_alpha,
        device=device,
        route_batch_size=route_batch_size,
    )


def compact_from_patch_cache(cache: PatchCache, target_region_names: list[str]) -> CompactPatchCache:
    tokens = cache.tokens.astype(np.float32)
    if cache.region_labels is None or cache.region_names is None:
        raise ValueError("Patch cache is missing region labels/names required for inference.")
    labels = cache.region_labels.astype(np.int16)
    src_region_names = [str(x) for x in cache.region_names.astype(object).tolist()]
    num_regions = len(src_region_names)
    dim = tokens.shape[-1]
    region_vecs = np.zeros((tokens.shape[0], num_regions, dim), dtype=np.float32)
    region_present = np.zeros((tokens.shape[0], num_regions), dtype=bool)
    for i in range(tokens.shape[0]):
        vec, present = compute_region_means(tokens[i], labels[i], num_regions, dim)
        region_vecs[i] = vec.astype(np.float32, copy=False)
        region_present[i] = present
    if src_region_names != target_region_names:
        src_idx = {name: idx for idx, name in enumerate(src_region_names)}
        aligned_vecs = np.zeros((tokens.shape[0], len(target_region_names), dim), dtype=np.float32)
        aligned_present = np.zeros((tokens.shape[0], len(target_region_names)), dtype=bool)
        for j, name in enumerate(target_region_names):
            if name in src_idx:
                k = src_idx[name]
                aligned_vecs[:, j] = region_vecs[:, k]
                aligned_present[:, j] = region_present[:, k]
        region_vecs = aligned_vecs
        region_present = aligned_present
    return CompactPatchCache(
        region_vecs=region_vecs,
        region_present=region_present,
        img_id=cache.img_id.astype(object),
        group=cache.group.astype(object),
        method=cache.method.astype(object),
        real_path=np.full(tokens.shape[0], "", dtype=object),
        region_names=np.asarray(target_region_names, dtype=object),
    )


def compact_from_patch_npz(path: Path, target_region_names: list[str]) -> CompactPatchCache:
    return compact_from_patch_cache(PatchCache.from_npz(path), target_region_names)


def predict_sample(
    runtime: MainRuntime,
    cls_cache: EmbeddingCache,
    patch_cache: CompactPatchCache,
) -> SamplePrediction:
    patch_bundle = runtime.patch_bundle
    pair_bundle = runtime.pair_bundle
    head_bundle = runtime.head_bundle

    route_prob = predict_hybrid_prob(
        runtime.hybrid_model,
        cls_cache.cls.astype(np.float32),
        runtime.hybrid_mean,
        runtime.hybrid_std,
        runtime.hybrid_temperature,
        runtime.hybrid_alpha,
        runtime.route_batch_size,
        runtime.device,
    )
    route_fake = route_fake_score(route_prob)
    route_top1 = np.asarray([HYBRID_LABELS[int(i)] for i in np.argmax(route_prob, axis=1)], dtype=object)

    real_pool_means = pair_bundle["real_pool_means"]
    patch_x = image_region_delta_features(patch_cache, real_pool_means)
    patch_global_prob = predict_patch_scores(
        patch_bundle["patch_scaler"],
        patch_bundle["patch_clf"],
        patch_x,
    )
    patch_scores = predict_group_binary_experts(
        patch_bundle["patch_group_classifiers"],
        patch_x,
    )
    patch_dynamic = dynamic_expert_score(route_prob, patch_scores, runtime.pair_route_mode)

    pair_scores = pair_score_matrix(
        patch_cache.region_vecs,
        real_pool_means,
        pair_bundle["pair_mean_dirs"],
        pair_bundle["pair_classifiers"],
        pair_bundle["pair_region_idx"],
        pair_bundle["pair_region_names_by_group"],
    )
    pair_dynamic = dynamic_expert_score(route_prob, pair_scores, runtime.pair_route_mode)

    meta_x = build_meta_features(route_prob, pair_scores, patch_scores, patch_global_prob)
    route_meta_prob, _ = predict_route_aware_meta_fusion(
        head_bundle["route_meta_experts"],
        route_prob,
        meta_x,
        runtime.pair_route_mode,
    )
    pred = (route_meta_prob >= runtime.route_meta_threshold).astype(np.int64)
    return SamplePrediction(
        route_prob=route_prob,
        route_fake=route_fake,
        route_top1=route_top1,
        patch_global_prob=patch_global_prob,
        patch_scores=patch_scores,
        patch_dynamic=patch_dynamic,
        pair_scores=pair_scores,
        pair_dynamic=pair_dynamic,
        route_meta_prob=route_meta_prob,
        pred=pred,
    )


def build_sample_report(
    sample_name: str,
    cls_cache: EmbeddingCache,
    prediction: SamplePrediction,
    threshold: float,
) -> dict[str, object]:
    pair_ids = cls_cache.pair_id.astype(object)
    folder_summary = []
    for folder in sorted(set(pair_ids.tolist())):
        idx = np.flatnonzero(pair_ids == folder)
        folder_summary.append(
            {
                "folder": str(folder),
                "num_images": int(len(idx)),
                "mean_route_fake_prob": float(np.mean(prediction.route_fake[idx])),
                "mean_pair_dynamic_prob": float(np.mean(prediction.pair_dynamic[idx])),
                "mean_patch_global_prob": float(np.mean(prediction.patch_global_prob[idx])),
                "mean_patch_dynamic_prob": float(np.mean(prediction.patch_dynamic[idx])),
                "mean_route_meta_prob": float(np.mean(prediction.route_meta_prob[idx])),
                "fake_positive_rate": float(np.mean(prediction.pred[idx])),
                "route_top1_counts": {
                    label: int(np.sum(prediction.route_top1[idx] == label))
                    for label in HYBRID_LABELS
                    if np.any(prediction.route_top1[idx] == label)
                },
            }
        )

    return {
        "sample_name": sample_name,
        "num_images": int(len(prediction.route_meta_prob)),
        "threshold": float(threshold),
        "mean_fake_prob": float(np.mean(prediction.route_meta_prob)),
        "min_fake_prob": float(np.min(prediction.route_meta_prob)),
        "max_fake_prob": float(np.max(prediction.route_meta_prob)),
        "fake_positive_rate": float(np.mean(prediction.pred)),
        "route_top1_counts": {
            label: int(np.sum(prediction.route_top1 == label))
            for label in HYBRID_LABELS
            if np.any(prediction.route_top1 == label)
        },
        "folder_summary": folder_summary,
    }


def parse_sample_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-root", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, nargs="+")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--num-folders", type=int, default=3)
    parser.add_argument("--max-frames-per-folder", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--save-sampled-root", type=Path, default=None)
    parser.add_argument("--clip-model-dir", type=Path, default=ROOT / "models" / "clip")
    parser.add_argument("--patch-model-dir", type=Path, default=ROOT / "models" / "clip")
    parser.add_argument("--patch-backbone", default="clip")
    parser.add_argument("--cls-device", default="auto")
    parser.add_argument("--patch-device", default="auto")
    parser.add_argument("--facer-device", default="cpu")
    parser.add_argument("--cls-batch-size", type=int, default=16)
    parser.add_argument("--patch-batch-size", type=int, default=8)
    parser.add_argument("--cls-target-size", type=int, default=224)
    parser.add_argument("--patch-target-size", type=int, default=224)
    parser.add_argument("--patch-layer", type=int, default=-1)
    parser.add_argument("--with-regions", action="store_true", default=True)
    parser.add_argument("--without-regions", dest="with_regions", action="store_false")
    parser.add_argument("--upstream-checkpoint", type=Path, default=ROOT / "checkpoints" / "upstream" / "checkpoint_best_hybrid_manifold.pt")
    parser.add_argument("--patch-branch", type=Path, default=ROOT / "checkpoints" / "downstream" / "patch_branch.joblib")
    parser.add_argument("--pair-branch", type=Path, default=ROOT / "checkpoints" / "downstream" / "pair_branch.joblib")
    parser.add_argument("--route-meta-head", type=Path, default=ROOT / "checkpoints" / "heads" / "route_meta_head.joblib")
    parser.add_argument("--route-batch-size", type=int, default=2048)
    parser.add_argument("--runtime-device", default="auto")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "probes" / "probe_result.json")
    return parser.parse_args(args=args)


def read_manifest_rows(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "img_path": row["img_path"],
                "label": int(row["label"]),
                "group": row["group"],
                "method": row["method"],
                "pair_id": row["pair_id"],
                "video_folder": row.get("video_folder", row["pair_id"]),
            }
            for row in reader
        ]


def load_sample_batches(
    args: argparse.Namespace,
    target_region_names: list[str],
) -> list[SampleBatch]:
    from prepare.extractors import extract_cls_cache, extract_patch_cache  # noqa: E402
    from prepare.folder_sampling import (  # noqa: E402
        build_backbones,
        sample_rows_from_sources,
        write_manifest,
        write_sample_summary,
    )

    if args.source_dir:
        source_dirs = [path.resolve() for path in args.source_dir]
        data_root = args.data_root.resolve()
        selected_folders, rows = sample_rows_from_sources(
            source_dirs,
            data_root=data_root,
            num_folders=args.num_folders,
            max_frames_per_folder=args.max_frames_per_folder,
            seed=args.seed,
            label=args.label,
        )
        cls_backbone, patch_backbone = build_backbones(
            clip_model_dir=args.clip_model_dir,
            patch_model_dir=args.patch_model_dir,
            patch_backbone=args.patch_backbone,
            cls_device=args.cls_device,
            patch_device=args.patch_device,
        )
        cls_cache = extract_cls_cache(
            rows,
            cls_backbone,
            batch_size=args.cls_batch_size,
            target_size=args.cls_target_size,
            progress_desc="sample-cls",
        )
        patch_cache = extract_patch_cache(
            rows,
            patch_backbone,
            batch_size=args.patch_batch_size,
            target_size=args.patch_target_size,
            layer=args.patch_layer,
            with_regions=args.with_regions,
            facer_device=args.facer_device,
            progress_desc="sample-patch",
        )
        if args.save_sampled_root is not None:
            save_root = args.save_sampled_root.resolve()
            save_root.mkdir(parents=True, exist_ok=True)
            write_manifest(rows, save_root / "manifest.csv")
            write_sample_summary(selected_folders, rows, data_root, save_root)
            cls_cache.to_npz(save_root / "cls_sampled.npz")
            patch_cache.to_npz(save_root / "patch_sampled.npz")
        sample_name = args.save_sampled_root.name if args.save_sampled_root is not None else (
            source_dirs[0].name if len(source_dirs) == 1 else "adhoc_sample"
        )
        return [
            SampleBatch(
                sample_name=sample_name,
                rows=rows,
                cls_cache=cls_cache,
                patch_cache=compact_from_patch_cache(patch_cache, target_region_names),
            )
        ]

    sample_root = args.sample_root.resolve() if args.sample_root is not None else (ROOT / "cache" / "sampled").resolve()
    sample_dirs = sorted(
        [
            path
            for path in sample_root.iterdir()
            if path.is_dir() and (path / "cls_sampled.npz").exists() and (path / "patch_sampled.npz").exists()
        ]
    )
    loaded: list[SampleBatch] = []
    for sample_dir in sample_dirs:
        cls_cache = EmbeddingCache.from_npz(sample_dir / "cls_sampled.npz")
        patch_cache = compact_from_patch_npz(sample_dir / "patch_sampled.npz", target_region_names)
        loaded.append(
            SampleBatch(
                sample_name=sample_dir.name,
                rows=read_manifest_rows(sample_dir / "manifest.csv"),
                cls_cache=cls_cache,
                patch_cache=patch_cache,
            )
        )
    return loaded


def run_sample(args: argparse.Namespace) -> dict[str, object]:
    runtime = load_main_runtime(
        upstream_checkpoint=args.upstream_checkpoint,
        patch_branch=args.patch_branch,
        pair_branch=args.pair_branch,
        route_meta_head=args.route_meta_head,
        route_batch_size=args.route_batch_size,
        device_name=args.runtime_device,
    )
    report: dict[str, object] = {
        "mode": "sample",
        "pipeline_root": str(ROOT),
        "upstream_checkpoint": str(args.upstream_checkpoint),
        "patch_branch": str(args.patch_branch),
        "pair_branch": str(args.pair_branch),
        "route_meta_head": str(args.route_meta_head),
        "sample_root": str(args.sample_root) if args.sample_root is not None else None,
        "source_dir": [str(path) for path in args.source_dir] if args.source_dir else None,
        "threshold": runtime.route_meta_threshold,
        "samples": {},
    }
    loaded_samples = load_sample_batches(args, runtime.region_names)
    for batch in loaded_samples:
        prediction = predict_sample(runtime, batch.cls_cache, batch.patch_cache)
        report["samples"][batch.sample_name] = build_sample_report(
            batch.sample_name,
            batch.cls_cache,
            prediction,
            runtime.route_meta_threshold,
        )
    return report


def load_head_manifest(head_meta: Path) -> dict:
    return json.loads(head_meta.read_text())


def manifest_config(manifest: dict) -> dict:
    config = manifest.get("config")
    if isinstance(config, dict):
        return config
    return manifest


def build_replay_config(args: argparse.Namespace, manifest: dict) -> SimpleNamespace:
    config = manifest_config(manifest)
    return SimpleNamespace(
        patch_cache_root=args.patch_cache_root,
        cls_cache_root=args.cls_cache_root,
        train_real_patch_cache=args.patch_cache_root / "DF40_train" / "REAL" / "patch_real_pool.npz",
        test_real_patch_cache=args.patch_cache_root / "DF40_test_ff" / "REAL" / "patch_real_pool.npz",
        train_real_cls_cache=args.cls_cache_root / "DF40_train" / "REAL" / "cls_real_dedup_from_existing.npz",
        test_real_cls_cache=args.cls_cache_root / "DF40_test_ff" / "REAL" / "cls_real_dedup_from_existing.npz",
        hybrid_checkpoint=args.upstream_checkpoint,
        seed=int(config["seed"]),
        train_real_max=int(config["train_real_max"]),
        eval_real_max=0,
        max_train_fake_per_method=int(config["max_train_fake_per_method"]),
        max_test_fake_per_method=0,
        max_ood_fake_per_method=0,
        patch_train_groups=list(config["patch_train_groups"]),
        route_batch_size=int(config["route_batch_size"]),
        val_ratio=float(config["val_ratio"]),
        val_split_mode=str(config["val_split_mode"]),
        pair_region_mode=str(config["pair_region_mode"]),
        no_tuning_mainline=True,
        pair_route_mode=str(config["pair_route_mode"]),
        fusion_step=0.05,
        auto_generic_pool="no_tuning",
        compact_cache_dir=args.compact_cache_dir,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )


def parse_replay_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-checkpoint", type=Path, default=ROOT / "checkpoints" / "upstream" / "checkpoint_best_hybrid_manifold.pt")
    parser.add_argument("--patch-branch", type=Path, default=ROOT / "checkpoints" / "downstream" / "patch_branch.joblib")
    parser.add_argument("--pair-branch", type=Path, default=ROOT / "checkpoints" / "downstream" / "pair_branch.joblib")
    parser.add_argument("--route-meta-head", type=Path, default=ROOT / "checkpoints" / "heads" / "route_meta_head.joblib")
    parser.add_argument("--head-meta", type=Path, default=ROOT / "checkpoints" / "heads" / "route_meta_head_meta.json")
    parser.add_argument("--cls-cache-root", type=Path, default=ROOT / "cache" / "cls")
    parser.add_argument("--patch-cache-root", type=Path, default=ROOT / "cache" / "patch")
    parser.add_argument("--compact-cache-dir", type=Path, default=ROOT / "cache" / "compact")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "replay" / "replay_eval.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "outputs" / "replay" / "replay_eval.csv")
    return parser.parse_args(args=args)


def export_route_meta_csv(output_csv: Path, report: dict) -> None:
    rows = []
    for split_name in ["test_ff", "ood"]:
        for row in report[split_name]["route_meta_fusion"]["methods"]:
            metrics = row["metrics"]
            rows.append(
                {
                    "split": split_name,
                    "model": "route_meta_fusion",
                    "group": row["group"],
                    "method": row["method"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "fake_accuracy": metrics["fake_accuracy"],
                    "real_accuracy": metrics["real_accuracy"],
                    "auc": metrics["auc"],
                }
            )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "model",
                "group",
                "method",
                "accuracy",
                "balanced_accuracy",
                "fake_accuracy",
                "real_accuracy",
                "auc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_replay(args: argparse.Namespace) -> None:
    manifest = load_head_manifest(args.head_meta)
    artifacts = load_bundle_artifacts(
        patch_branch=args.patch_branch,
        pair_branch=args.pair_branch,
        route_meta_head=args.route_meta_head,
    )
    patch_bundle = artifacts.patch_bundle
    pair_bundle = artifacts.pair_bundle
    head_bundle = artifacts.head_bundle

    cfg = build_replay_config(args, manifest)
    set_seed(cfg.seed)
    device = resolve_torch_device(args.device)

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
    train_region_vecs = np.concatenate(
        [np.concatenate(train_fake_region_vecs, axis=0), train_real_cache.region_vecs.astype(np.float32)],
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

    patch_scaler = patch_bundle["patch_scaler"]
    patch_clf = patch_bundle["patch_clf"]
    patch_group_classifiers = patch_bundle["patch_group_classifiers"]
    pair_mean_dirs = pair_bundle["pair_mean_dirs"]
    pair_classifiers = pair_bundle["pair_classifiers"]
    pair_region_idx = pair_bundle["pair_region_idx"]
    pair_region_names_by_group = pair_bundle["pair_region_names_by_group"]
    route_meta_experts = head_bundle["route_meta_experts"]
    route_meta_threshold = float(head_bundle["route_meta_threshold"])

    patch_train_prob = predict_patch_scores(patch_scaler, patch_clf, train_x)
    patch_train_scores = predict_group_binary_experts(patch_group_classifiers, train_x)

    hybrid_model, hybrid_mean, hybrid_std, hybrid_temperature, hybrid_alpha = load_hybrid_checkpoint(cfg.hybrid_checkpoint, device)
    train_cls_rows = collect_cls_rows(iter_regular_cls_rows(cfg.cls_cache_root, "DF40_train", cfg.train_real_cls_cache))
    test_cls_rows = collect_cls_rows(iter_regular_cls_rows(cfg.cls_cache_root, "DF40_test_ff", cfg.test_real_cls_cache))
    ood_cls_rows = collect_cls_rows(iter_ood_cls_rows(cfg.cls_cache_root, "DF40_test_ood"))

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

    train_route_map = build_prob_map(train_cls_rows, hybrid_predict, cfg.route_batch_size)
    test_route_map = build_prob_map(test_cls_rows, hybrid_predict, cfg.route_batch_size)
    ood_route_map = build_prob_map(ood_cls_rows, hybrid_predict, cfg.route_batch_size)
    train_route_prob_mat = route_prob_from_map(train_route_map, train_img_ids)
    train_pair_scores = pair_score_matrix(
        train_region_vecs,
        real_pool_means,
        pair_mean_dirs,
        pair_classifiers,
        pair_region_idx,
        pair_region_names_by_group,
    )
    train_meta_x = build_meta_features(train_route_prob_mat, train_pair_scores, patch_train_scores, patch_train_prob)
    val_route_meta_prob, _ = predict_route_aware_meta_fusion(
        route_meta_experts,
        train_route_prob_mat[val_idx],
        train_meta_x[val_idx],
        cfg.pair_route_mode,
    )
    threshold_value = route_meta_threshold
    validation_metrics = metrics_from_binary_prob(train_y[val_idx], val_route_meta_prob, threshold_value)
    threshold_search = {
        "threshold": threshold_value,
        "balanced_accuracy": validation_metrics["balanced_accuracy"],
        "accuracy": validation_metrics["accuracy"],
        "fake_accuracy": validation_metrics["fake_accuracy"],
        "real_accuracy": validation_metrics["real_accuracy"],
        "auc": validation_metrics["auc"],
    }

    test_ff_summary = evaluate_regular_route_meta(
        discover_flat_method_caches(cfg.patch_cache_root, "DF40_test_ff", ["FS", "FR", "FE", "EFS"]),
        test_real_cache,
        args=cfg,
        route_map=test_route_map,
        pair_mean_dirs=pair_mean_dirs,
        pair_classifiers=pair_classifiers,
        pair_region_idx=pair_region_idx,
        pair_region_names_by_group=pair_region_names_by_group,
        patch_scaler=patch_scaler,
        patch_clf=patch_clf,
        patch_group_classifiers=patch_group_classifiers,
        real_pool_means=real_pool_means,
        route_meta_experts=route_meta_experts,
        route_meta_threshold=threshold_value,
    )
    ood_summary = evaluate_ood_route_meta(
        discover_ood_method_entries(cfg.patch_cache_root),
        args=cfg,
        route_map=ood_route_map,
        pair_mean_dirs=pair_mean_dirs,
        pair_classifiers=pair_classifiers,
        pair_region_idx=pair_region_idx,
        pair_region_names_by_group=pair_region_names_by_group,
        patch_scaler=patch_scaler,
        patch_clf=patch_clf,
        patch_group_classifiers=patch_group_classifiers,
        real_pool_means=real_pool_means,
        route_meta_experts=route_meta_experts,
        route_meta_threshold=threshold_value,
    )

    payload = {
        "mode": "replay",
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
        },
        "train_summary": manifest.get(
            "counts",
            {
                "num_train_fake": int(n_fake_train),
                "num_train_real": int(len(train_real_x)),
                "num_train_total": int(len(train_x)),
            },
        ),
        "route_meta_fusion_search": manifest.get("route_meta_search", {}),
        "threshold_search": threshold_search,
        "validation": {
            "route_meta_fusion": threshold_search["balanced_accuracy"],
        },
        "test_ff": {
            "route_meta_fusion": test_ff_summary,
        },
        "ood": {
            "route_meta_fusion": ood_summary,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    export_route_meta_csv(args.output_csv, payload)
    print(json.dumps(payload, indent=2))
    print(f"Saved to {args.output_json}")
    print(f"Saved CSV to {args.output_csv}")


def parse_mode(argv: list[str] | None = None) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["sample", "replay"], default="sample")
    known, remaining = parser.parse_known_args(argv)
    return str(known.mode), remaining


def run_sample_cli(argv: list[str] | None = None) -> None:
    args = parse_sample_args(argv)
    report = run_sample(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved to {args.output_json}")


def run_replay_cli(argv: list[str] | None = None) -> None:
    args = parse_replay_args(argv)
    run_replay(args)


def main(argv: list[str] | None = None) -> None:
    mode, remaining = parse_mode(argv)
    if mode == "replay":
        run_replay_cli(remaining)
        return
    run_sample_cli(remaining)


if __name__ == "__main__":
    main()
