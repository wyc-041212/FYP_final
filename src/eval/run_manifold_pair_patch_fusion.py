#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from eval.pair_distilled_projection import (  # noqa: E402
    compute_mean_delta_directions,
    extract_cos_norm_features,
    normalize_key,
)
from eval.four_element_registry import (  # noqa: E402
    EXPORT_MODEL_KEYS,
    GENERIC_MODEL_KEYS,
    NO_TUNING_GENERIC_MODEL_KEYS,
    resolve_auto_generic_from_validation,
)
from eval.run_unified_patch_route_fusion import (  # noqa: E402
    compute_real_pool_region_means,
    discover_flat_method_caches,
    discover_ood_method_entries,
    image_region_delta_features,
    fit_patch_classifier,
    iter_ood_cls_rows,
    iter_regular_cls_rows,
    load_compact_cache,
    method_metrics,
    predict_patch_scores,
    split_train_indices,
    summarize_split,
)
from train.train_upstream import (  # noqa: E402
    ALL_GROUPS as HYBRID_LABELS,
    HybridManifoldModel,
)
from utils.device import resolve_torch_device  # noqa: E402


FAKE_GROUPS = ["EFS", "FS", "FR", "FE"]
PAIR_GROUPS = ["FS", "FR", "FE", "EFS"]
CANONICAL_REGIONS = {
    "FS": ["eyebrow", "skin"],
    "FR": ["skin", "nose"],
    "FE": ["eye", "mouth"],
    "EFS": ["eye", "mouth", "eyebrow"],
}
REAL_LABEL = "REAL"
REAL_IDX = HYBRID_LABELS.index(REAL_LABEL)
FAKE_ROUTE_IDXS = [HYBRID_LABELS.index(group) for group in FAKE_GROUPS]
GENERIC_BLEND_BASE_MODELS = [
    "full_fusion",
    "meta_fusion",
    "route_meta_fusion",
    "bucket_expert_fusion",
    "bucket_meta_fusion",
    "uncertainty_residual_fusion",
    "semantic_fusion",
    "semantic_max_fusion",
]
NO_TUNING_GENERIC_BLEND_BASE_MODELS = [
    "full_fusion",
    "meta_fusion",
    "route_meta_fusion",
]


@dataclass(frozen=True)
class ClsRows:
    keys: np.ndarray
    x: np.ndarray


@dataclass(frozen=True)
class PairMethodData:
    group: str
    method: str
    region_vecs: np.ndarray
    paired_real_vecs: np.ndarray
    has_pair: np.ndarray


@dataclass(frozen=True)
class BranchFeatures:
    x: np.ndarray
    route_prob: np.ndarray
    route_score: np.ndarray
    patch_prob: np.ndarray
    patch_scores: np.ndarray
    route_patch_score: np.ndarray
    pair_scores: np.ndarray
    pair_score: np.ndarray
    meta_x: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manifold route + pair-distilled pseudo-delta + patch-token unified fusion."
    )
    parser.add_argument("--patch-cache-root", type=Path, default=ROOT / "cache" / "patch")
    parser.add_argument("--cls-cache-root", type=Path, default=ROOT / "cache" / "cls")
    parser.add_argument(
        "--train-real-patch-cache",
        type=Path,
        default=ROOT / "cache" / "patch" / "DF40_train" / "REAL" / "patch_real_pool.npz",
    )
    parser.add_argument(
        "--test-real-patch-cache",
        type=Path,
        default=ROOT / "cache" / "patch" / "DF40_test_ff" / "REAL" / "patch_real_pool.npz",
    )
    parser.add_argument(
        "--train-real-cls-cache",
        type=Path,
        default=ROOT / "cache" / "cls" / "DF40_train" / "REAL" / "cls_real_dedup_from_existing.npz",
    )
    parser.add_argument(
        "--test-real-cls-cache",
        type=Path,
        default=ROOT / "cache" / "cls" / "DF40_test_ff" / "REAL" / "cls_real_dedup_from_existing.npz",
    )
    parser.add_argument(
        "--real-reference-cache",
        type=Path,
        default=ROOT / "cache" / "patch" / "DF40_train" / "REAL" / "patch_real_pool_region_reference.npz",
    )
    parser.add_argument(
        "--hybrid-checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "upstream" / "checkpoint_best_hybrid_manifold.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-real-max", type=int, default=15000)
    parser.add_argument("--eval-real-max", type=int, default=2000)
    parser.add_argument("--max-train-fake-per-method", type=int, default=0)
    parser.add_argument("--max-test-fake-per-method", type=int, default=0)
    parser.add_argument("--max-ood-fake-per-method", type=int, default=0)
    parser.add_argument("--patch-train-groups", nargs="*", default=["FS", "FR", "FE", "EFS"])
    parser.add_argument("--route-batch-size", type=int, default=2048)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--val-split-mode",
        choices=["within_method", "holdout_method"],
        default="holdout_method",
    )
    parser.add_argument("--pair-ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--pair-region-mode",
        choices=["canonical", "all_regions"],
        default="canonical",
    )
    parser.add_argument(
        "--no-tuning-mainline",
        action="store_true",
        help="Use the generic review path: all-region pair features, holdout validation, and non-heuristic auto-selection.",
    )
    parser.add_argument("--pair-feature-mode", choices=["cosnorm"], default="cosnorm")
    parser.add_argument("--pair-pca-dim", type=int, default=32)
    parser.add_argument(
        "--pair-route-mode",
        choices=["weighted", "weighted_norm", "top1", "top2"],
        default="weighted",
    )
    parser.add_argument("--fusion-step", type=float, default=0.05)
    parser.add_argument(
        "--auto-generic-pool",
        choices=["default", "no_tuning"],
        default="default",
    )
    parser.add_argument(
        "--compact-cache-dir",
        type=Path,
        default=ROOT / "cache" / "compact",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "outputs" / "manifold_pair_patch_fusion.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "outputs" / "manifold_pair_patch_fusion_methods.csv",
    )
    args = parser.parse_args()
    if args.no_tuning_mainline:
        args.pair_region_mode = "all_regions"
        args.val_split_mode = "holdout_method"
        args.auto_generic_pool = "no_tuning"
    return args


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_cls_rows(rows) -> ClsRows:
    keys = []
    vecs = []
    for key, vec, _, _ in rows:
        keys.append(str(key))
        vecs.append(vec.astype(np.float32, copy=False))
    return ClsRows(keys=np.asarray(keys, dtype=object), x=np.stack(vecs).astype(np.float32))


def load_hybrid_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[HybridManifoldModel, np.ndarray, np.ndarray, float, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dim = int(checkpoint["dim"])
    real_rank = int(checkpoint["real_rank"])
    efs_rank = int(checkpoint["efs_rank"])
    model = HybridManifoldModel(
        dim=dim,
        real_rank=real_rank,
        efs_rank=efs_rank,
        real_center_init=np.zeros(dim, dtype=np.float32),
        efs_center_init=np.zeros(dim, dtype=np.float32),
        real_basis_init=np.zeros((dim, real_rank), dtype=np.float32),
        efs_basis_init=np.zeros((dim, efs_rank), dtype=np.float32),
        fake_offset_init=np.zeros((3, dim), dtype=np.float32),
        delta_proto_init=np.zeros((3, dim), dtype=np.float32),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return (
        model,
        checkpoint["mean"].astype(np.float32),
        checkpoint["std"].astype(np.float32),
        float(checkpoint["temperature"]),
        float(checkpoint["alpha"]),
    )


def predict_hybrid_prob(
    model: HybridManifoldModel,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    temperature: float,
    alpha: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    x_std = ((x - mean) / std).astype(np.float32)
    probs = []
    with torch.no_grad():
        for start in range(0, len(x_std), batch_size):
            batch = torch.from_numpy(x_std[start : start + batch_size]).to(device)
            _, _, _, fused = model.fused_logits(batch, temperature, alpha)
            probs.append(torch.softmax(fused, dim=1).cpu().numpy().astype(np.float32))
    return np.concatenate(probs, axis=0)


def build_prob_map(rows: ClsRows, predict_fn, batch_size: int) -> dict[str, np.ndarray]:
    mapping: dict[str, np.ndarray] = {}
    for start in range(0, len(rows.keys), batch_size):
        end = start + batch_size
        prob = predict_fn(rows.x[start:end])
        for key, row in zip(rows.keys[start:end].tolist(), prob, strict=True):
            mapping[str(key)] = row.astype(np.float32, copy=False)
    return mapping


def route_prob_from_map(route_map: dict[str, np.ndarray], img_ids: np.ndarray) -> np.ndarray:
    return np.stack([route_map[str(img_id)] for img_id in img_ids.tolist()]).astype(np.float32)


def route_fake_score(route_prob: np.ndarray) -> np.ndarray:
    return (1.0 - route_prob[:, REAL_IDX]).astype(np.float32)


def load_real_reference_cache(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "region_names": [str(x) for x in data["region_names"].astype(object).tolist()],
        "region_vecs": data["region_vecs"].astype(np.float32),
        "normalized_img_id": data["normalized_img_id"].astype(object),
    }


def fit_binary(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x.astype(np.float64))
    x_std = np.clip(np.nan_to_num(x_std, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
    clf = LogisticRegression(
        C=0.25,
        max_iter=3000,
        solver="liblinear",
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(x_std, y)
    return scaler, clf


def predict_binary(clf, scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    x_std = scaler.transform(x.astype(np.float64))
    x_std = np.clip(np.nan_to_num(x_std, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
    return clf.predict_proba(x_std)[:, 1].astype(np.float32)


def fit_group_binary_experts(
    fake_x: np.ndarray,
    fake_groups: np.ndarray,
    real_x: np.ndarray,
    seed: int,
) -> dict[str, tuple[StandardScaler, LogisticRegression]]:
    experts: dict[str, tuple[StandardScaler, LogisticRegression]] = {}
    for offset, group in enumerate(FAKE_GROUPS):
        group_fake = fake_x[fake_groups == group]
        if len(group_fake) == 0:
            continue
        train_x = np.concatenate([group_fake, real_x], axis=0)
        train_y = np.concatenate(
            [np.ones(len(group_fake), dtype=np.int64), np.zeros(len(real_x), dtype=np.int64)],
            axis=0,
        )
        experts[group] = fit_binary(train_x, train_y, seed + offset)
    return experts


def predict_group_binary_experts(
    experts: dict[str, tuple[StandardScaler, LogisticRegression]],
    x: np.ndarray,
) -> np.ndarray:
    out = np.zeros((len(x), len(FAKE_GROUPS)), dtype=np.float32)
    for idx, group in enumerate(FAKE_GROUPS):
        scaler, clf = experts[group]
        out[:, idx] = predict_binary(clf, scaler, x)
    return out


def sanitize_feature_matrix(x: np.ndarray, clip_value: float = 25.0) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=clip_value, neginf=-clip_value)
    return np.clip(x, -clip_value, clip_value).astype(np.float32)


def pair_delta_feature_block(
    region_deltas: np.ndarray,
    mean_delta_dirs: np.ndarray,
    region_idx: dict[str, int],
    canonical_regions: list[str],
) -> np.ndarray:
    full = extract_cos_norm_features(region_deltas.astype(np.float32), mean_delta_dirs.astype(np.float32))
    col_idx = []
    for name in canonical_regions:
        ridx = region_idx[name]
        col_idx.extend([ridx * 2, ridx * 2 + 1])
    return sanitize_feature_matrix(full[:, col_idx])


def preload_pair_training_methods(
    args: argparse.Namespace,
    real_ref: dict,
) -> dict[tuple[str, str], PairMethodData]:
    real_lookup = {
        str(key): idx for idx, key in enumerate(real_ref["normalized_img_id"].tolist())
    }
    all_methods: dict[tuple[str, str], PairMethodData] = {}
    for group in FAKE_GROUPS:
        for idx, path in enumerate(discover_flat_method_caches(args.patch_cache_root, "DF40_train", [group])):
            cache = load_compact_cache(
                path,
                max_rows=args.max_train_fake_per_method,
                seed=args.seed + idx,
                compact_cache_dir=args.compact_cache_dir,
            )
            paired_real_vecs = np.zeros_like(cache.region_vecs, dtype=np.float32)
            has_pair = np.zeros(len(cache.img_id), dtype=bool)
            for row_idx, real_path in enumerate(cache.real_path.tolist()):
                key = normalize_key(str(real_path))
                real_idx = real_lookup.get(key)
                if real_idx is not None:
                    paired_real_vecs[row_idx] = real_ref["region_vecs"][real_idx]
                    has_pair[row_idx] = True
            all_methods[(group, str(cache.method[0]))] = PairMethodData(
                group=group,
                method=str(cache.method[0]),
                region_vecs=cache.region_vecs.astype(np.float32),
                paired_real_vecs=paired_real_vecs,
                has_pair=has_pair,
            )
    return all_methods


def collect_paired_data(
    all_methods: dict[tuple[str, str], PairMethodData],
    groups: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    fake_parts = []
    real_parts = []
    for (_, _), data in all_methods.items():
        if data.group not in groups:
            continue
        if not np.any(data.has_pair):
            continue
        fake_parts.append(data.region_vecs[data.has_pair])
        real_parts.append(data.paired_real_vecs[data.has_pair])
    if not fake_parts:
        raise RuntimeError("No paired fake-real training data found for pair-distilled branch.")
    return np.concatenate(fake_parts, axis=0), np.concatenate(real_parts, axis=0)


def build_pair_region_names_by_group(region_names: list[str], mode: str) -> dict[str, list[str]]:
    if mode == "all_regions":
        return {group: list(region_names) for group in PAIR_GROUPS}
    if mode == "canonical":
        return {group: list(CANONICAL_REGIONS[group]) for group in PAIR_GROUPS}
    raise ValueError(f"Unknown pair-region mode: {mode}")


def train_pair_branch(
    args: argparse.Namespace,
    all_methods: dict[tuple[str, str], PairMethodData],
    real_ref: dict,
    real_pool_means: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[StandardScaler, LogisticRegression]], dict[str, int], dict[str, list[str]]]:
    region_idx = {name: idx for idx, name in enumerate(real_ref["region_names"])}
    pair_region_names_by_group = build_pair_region_names_by_group(
        [str(name) for name in real_ref["region_names"]],
        args.pair_region_mode,
    )
    real_delta = sanitize_feature_matrix(real_ref["region_vecs"] - real_pool_means[None, :, :])
    shared_paired_fake, shared_paired_real = collect_paired_data(all_methods, ["FS", "FR", "FE"])
    shared_mean_dirs = compute_mean_delta_directions(
        shared_paired_fake.astype(np.float32),
        shared_paired_real.astype(np.float32),
        len(real_ref["region_names"]),
    )
    mean_delta_dirs_by_group: dict[str, np.ndarray] = {}
    classifiers: dict[str, tuple[StandardScaler, LogisticRegression]] = {}
    for group in PAIR_GROUPS:
        paired_fake_parts = [data.region_vecs[data.has_pair] for data in all_methods.values() if data.group == group and np.any(data.has_pair)]
        paired_real_parts = [data.paired_real_vecs[data.has_pair] for data in all_methods.values() if data.group == group and np.any(data.has_pair)]
        fake_parts = [data.region_vecs for data in all_methods.values() if data.group == group]
        if not fake_parts:
            continue
        if paired_fake_parts:
            mean_dirs = compute_mean_delta_directions(
                np.concatenate(paired_fake_parts, axis=0).astype(np.float32),
                np.concatenate(paired_real_parts, axis=0).astype(np.float32),
                len(real_ref["region_names"]),
            )
        else:
            mean_dirs = shared_mean_dirs
        mean_delta_dirs_by_group[group] = mean_dirs.astype(np.float32)
        fake_vecs = np.concatenate(fake_parts, axis=0).astype(np.float32)
        fake_delta = sanitize_feature_matrix(fake_vecs - real_pool_means[None, :, :])
        fake_feat = pair_delta_feature_block(fake_delta, mean_dirs, region_idx, pair_region_names_by_group[group])
        real_feat = pair_delta_feature_block(real_delta, mean_dirs, region_idx, pair_region_names_by_group[group])
        train_x = np.concatenate([fake_feat, real_feat], axis=0)
        train_y = np.concatenate(
            [np.ones(len(fake_feat), dtype=np.int64), np.zeros(len(real_feat), dtype=np.int64)],
            axis=0,
        )
        classifiers[group] = fit_binary(train_x, train_y, args.seed)
    return mean_delta_dirs_by_group, classifiers, region_idx, pair_region_names_by_group


def pair_score_matrix(
    region_vecs: np.ndarray,
    real_pool_means: np.ndarray,
    mean_delta_dirs_by_group: dict[str, np.ndarray],
    classifiers: dict[str, tuple[StandardScaler, LogisticRegression]],
    region_idx: dict[str, int],
    pair_region_names_by_group: dict[str, list[str]],
) -> np.ndarray:
    region_deltas = sanitize_feature_matrix(region_vecs.astype(np.float32) - real_pool_means[None, :, :])
    out = np.zeros((len(region_vecs), len(FAKE_GROUPS)), dtype=np.float32)
    for col_idx, group in enumerate(FAKE_GROUPS):
        scaler, clf = classifiers[group]
        feats = pair_delta_feature_block(
            region_deltas,
            mean_delta_dirs_by_group[group],
            region_idx,
            pair_region_names_by_group[group],
        )
        out[:, col_idx] = predict_binary(clf, scaler, feats)
    return out


def dynamic_expert_score(route_prob: np.ndarray, expert_scores: np.ndarray, mode: str) -> np.ndarray:
    weights = route_prob[:, FAKE_ROUTE_IDXS]
    if mode == "weighted":
        return np.sum(weights * expert_scores, axis=1).astype(np.float32)
    if mode == "weighted_norm":
        norm = np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
        return np.sum((weights / norm) * expert_scores, axis=1).astype(np.float32)
    top1 = np.argmax(weights, axis=1)
    if mode == "top1":
        return expert_scores[np.arange(len(expert_scores)), top1].astype(np.float32)
    if mode == "top2":
        top2 = np.argsort(-weights, axis=1)[:, :2]
        mix = np.take_along_axis(weights, top2, axis=1)
        mix = mix / np.clip(mix.sum(axis=1, keepdims=True), 1e-8, None)
        return np.sum(np.take_along_axis(expert_scores, top2, axis=1) * mix, axis=1).astype(np.float32)
    raise ValueError(f"Unknown expert route mode: {mode}")


def dynamic_pair_score(route_prob: np.ndarray, pair_scores: np.ndarray, mode: str) -> np.ndarray:
    return dynamic_expert_score(route_prob, pair_scores, mode)


def search_three_way_weights(
    y_true: np.ndarray,
    route_prob: np.ndarray,
    pair_prob: np.ndarray,
    patch_prob: np.ndarray,
    step: float,
) -> dict:
    best: dict | None = None
    grid = np.arange(0.0, 1.0 + 1e-8, step)
    for w_route in grid:
        for w_pair in grid:
            w_patch = 1.0 - w_route - w_pair
            if w_patch < -1e-8:
                continue
            w_patch = max(0.0, float(w_patch))
            fused = (
                float(w_route) * route_prob
                + float(w_pair) * pair_prob
                + w_patch * patch_prob
            )
            pred = (fused >= 0.5).astype(np.int64)
            fake_mask = y_true == 1
            real_mask = y_true == 0
            fake_acc = float(np.mean(pred[fake_mask] == 1)) if np.any(fake_mask) else 0.0
            real_acc = float(np.mean(pred[real_mask] == 0)) if np.any(real_mask) else 0.0
            bacc = 0.5 * (fake_acc + real_acc)
            acc = float(np.mean(pred == y_true))
            auc = float(roc_auc_score(y_true, fused))
            key = (bacc, auc, real_acc, acc, fake_acc)
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "weight_route": float(w_route),
                    "weight_pair": float(w_pair),
                    "weight_patch": w_patch,
                    "val_balanced_accuracy": bacc,
                    "val_accuracy": acc,
                    "val_real_accuracy": real_acc,
                    "val_auc": auc,
                }
    assert best is not None
    return {k: v for k, v in best.items() if k != "key"}


def search_four_way_weights(
    y_true: np.ndarray,
    route_prob: np.ndarray,
    pair_prob: np.ndarray,
    route_patch_prob: np.ndarray,
    route_meta_prob: np.ndarray,
    step: float,
) -> dict:
    best: dict | None = None
    grid = np.arange(0.0, 1.0 + 1e-8, step)
    for w_route in grid:
        for w_pair in grid:
            for w_route_patch in grid:
                w_route_meta = 1.0 - w_route - w_pair - w_route_patch
                if w_route_meta < -1e-8:
                    continue
                w_route_meta = max(0.0, float(w_route_meta))
                fused = (
                    float(w_route) * route_prob
                    + float(w_pair) * pair_prob
                    + float(w_route_patch) * route_patch_prob
                    + w_route_meta * route_meta_prob
                )
                pred = (fused >= 0.5).astype(np.int64)
                fake_mask = y_true == 1
                real_mask = y_true == 0
                fake_acc = float(np.mean(pred[fake_mask] == 1)) if np.any(fake_mask) else 0.0
                real_acc = float(np.mean(pred[real_mask] == 0)) if np.any(real_mask) else 0.0
                bacc = 0.5 * (fake_acc + real_acc)
                acc = float(np.mean(pred == y_true))
                auc = float(roc_auc_score(y_true, fused))
                key = (bacc, auc, real_acc, acc, fake_acc)
                if best is None or key > best["key"]:
                    best = {
                        "key": key,
                        "weight_route": float(w_route),
                        "weight_pair": float(w_pair),
                        "weight_route_patch": float(w_route_patch),
                        "weight_route_meta": w_route_meta,
                        "val_balanced_accuracy": bacc,
                        "val_accuracy": acc,
                        "val_real_accuracy": real_acc,
                        "val_auc": auc,
                    }
    assert best is not None
    return {k: v for k, v in best.items() if k != "key"}


def search_bucket_meta_fusion(
    y_true: np.ndarray,
    route_prob: np.ndarray,
    route_score: np.ndarray,
    pair_score: np.ndarray,
    route_patch_score: np.ndarray,
    route_meta_prob: np.ndarray,
    step: float,
) -> dict:
    bucket_idx = top_fake_group(route_prob)
    global_choice = search_four_way_weights(
        y_true,
        route_score,
        pair_score,
        route_patch_score,
        route_meta_prob,
        step,
    )
    weights_by_group: dict[str, dict] = {}
    search_by_group: dict[str, dict] = {}
    for group_idx, group in enumerate(FAKE_GROUPS):
        rows = np.flatnonzero(bucket_idx == group_idx)
        if len(rows) == 0 or len(np.unique(y_true[rows])) < 2:
            weights_by_group[group] = {
                "weight_route": global_choice["weight_route"],
                "weight_pair": global_choice["weight_pair"],
                "weight_route_patch": global_choice["weight_route_patch"],
                "weight_route_meta": global_choice["weight_route_meta"],
            }
            search_by_group[group] = {
                "used_global_fallback": True,
                "num_rows": int(len(rows)),
                "val_balanced_accuracy": global_choice["val_balanced_accuracy"],
                "val_accuracy": global_choice["val_accuracy"],
                "val_auc": global_choice["val_auc"],
            }
            continue
        choice = search_four_way_weights(
            y_true[rows],
            route_score[rows],
            pair_score[rows],
            route_patch_score[rows],
            route_meta_prob[rows],
            step,
        )
        weights_by_group[group] = {
            "weight_route": choice["weight_route"],
            "weight_pair": choice["weight_pair"],
            "weight_route_patch": choice["weight_route_patch"],
            "weight_route_meta": choice["weight_route_meta"],
        }
        search_by_group[group] = {
            "used_global_fallback": False,
            "num_rows": int(len(rows)),
            "val_balanced_accuracy": choice["val_balanced_accuracy"],
            "val_accuracy": choice["val_accuracy"],
            "val_auc": choice["val_auc"],
        }
    return {
        "global": global_choice,
        "weights_by_group": weights_by_group,
        "search_by_group": search_by_group,
    }


def apply_bucket_meta_fusion(
    route_prob: np.ndarray,
    route_score: np.ndarray,
    pair_score: np.ndarray,
    route_patch_score: np.ndarray,
    route_meta_prob: np.ndarray,
    bucket_choice: dict,
) -> np.ndarray:
    bucket_idx = top_fake_group(route_prob)
    fused = np.zeros(len(route_prob), dtype=np.float32)
    for group_idx, group in enumerate(FAKE_GROUPS):
        rows = bucket_idx == group_idx
        if not np.any(rows):
            continue
        weights = bucket_choice["weights_by_group"][group]
        fused[rows] = (
            weights["weight_route"] * route_score[rows]
            + weights["weight_pair"] * pair_score[rows]
            + weights["weight_route_patch"] * route_patch_score[rows]
            + weights["weight_route_meta"] * route_meta_prob[rows]
        )
    return fused.astype(np.float32)


def search_semantic_rescue(
    y_true: np.ndarray,
    route_prob: np.ndarray,
    base_prob: np.ndarray,
    pair_prob: np.ndarray,
    route_patch_prob: np.ndarray,
) -> dict:
    bucket_idx = top_fake_group(route_prob)
    margin = route_margin_vs_real(route_prob)
    best_by_group: dict[str, dict] = {}
    for group_idx, group in enumerate(FAKE_GROUPS):
        rows = np.flatnonzero(bucket_idx == group_idx)
        default_choice = {
            "margin_tau": 0.0,
            "lambda_alt": 0.0,
            "alt_source": "max",
            "val_balanced_accuracy": 0.0,
            "val_accuracy": 0.0,
            "val_real_accuracy": 0.0,
            "val_auc": 0.0,
            "num_rows": int(len(rows)),
        }
        if len(rows) == 0 or len(np.unique(y_true[rows])) < 2:
            best_by_group[group] = default_choice
            continue
        alt_source_options = ["pair", "route_patch", "max", "blend"]
        best = None
        for alt_source in alt_source_options:
            alt_prob = resolve_alt_prob(alt_source, pair_prob[rows], route_patch_prob[rows])
            for margin_tau in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
                gate = margin[rows] <= margin_tau
                for lambda_alt in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                    fused = base_prob[rows].copy()
                    fused[gate] = (1.0 - lambda_alt) * fused[gate] + lambda_alt * alt_prob[gate]
                    metrics = metrics_from_binary_prob(y_true[rows], fused, 0.5)
                    key = (
                        metrics["balanced_accuracy"],
                        metrics["auc"],
                        metrics["real_accuracy"],
                        metrics["accuracy"],
                        metrics["fake_accuracy"],
                    )
                    candidate = {
                        "key": key,
                        "margin_tau": float(margin_tau),
                        "lambda_alt": float(lambda_alt),
                        "alt_source": alt_source,
                        "val_balanced_accuracy": metrics["balanced_accuracy"],
                        "val_accuracy": metrics["accuracy"],
                        "val_real_accuracy": metrics["real_accuracy"],
                        "val_auc": metrics["auc"],
                        "num_rows": int(len(rows)),
                    }
                    if best is None or key > best["key"]:
                        best = candidate
        assert best is not None
        best_by_group[group] = {k: v for k, v in best.items() if k != "key"}
    return best_by_group


def apply_semantic_rescue(
    route_prob: np.ndarray,
    base_prob: np.ndarray,
    pair_prob: np.ndarray,
    route_patch_prob: np.ndarray,
    rescue_choice: dict,
) -> np.ndarray:
    bucket_idx = top_fake_group(route_prob)
    margin = route_margin_vs_real(route_prob)
    fused = base_prob.copy().astype(np.float32)
    for group_idx, group in enumerate(FAKE_GROUPS):
        rows = bucket_idx == group_idx
        if not np.any(rows):
            continue
        choice = rescue_choice[group]
        if choice["lambda_alt"] <= 1e-8:
            continue
        alt_prob = resolve_alt_prob(choice["alt_source"], pair_prob[rows], route_patch_prob[rows])
        gate = margin[rows] <= choice["margin_tau"]
        fused_group = fused[rows]
        fused_group[gate] = (1.0 - choice["lambda_alt"]) * fused_group[gate] + choice["lambda_alt"] * alt_prob[gate]
        fused[rows] = fused_group
    return fused.astype(np.float32)


def search_semantic_max_rescue(
    y_true: np.ndarray,
    route_prob: np.ndarray,
    base_prob: np.ndarray,
    pair_prob: np.ndarray,
    route_patch_prob: np.ndarray,
) -> dict:
    bucket_idx = top_fake_group(route_prob)
    margin = route_margin_vs_real(route_prob)
    best_by_group: dict[str, dict] = {}
    for group_idx, group in enumerate(FAKE_GROUPS):
        rows = np.flatnonzero(bucket_idx == group_idx)
        default_choice = {
            "margin_tau": 0.0,
            "alt_min": 1.0,
            "alt_source": "max",
            "val_balanced_accuracy": 0.0,
            "val_accuracy": 0.0,
            "val_real_accuracy": 0.0,
            "val_auc": 0.0,
            "num_rows": int(len(rows)),
        }
        if len(rows) == 0 or len(np.unique(y_true[rows])) < 2:
            best_by_group[group] = default_choice
            continue
        alt_source_options = ["pair", "route_patch", "max", "blend"]
        best = None
        for alt_source in alt_source_options:
            alt_prob = resolve_alt_prob(alt_source, pair_prob[rows], route_patch_prob[rows])
            for margin_tau in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
                for alt_min in [0.4, 0.5, 0.6, 0.7, 0.8]:
                    gate = (margin[rows] <= margin_tau) & (alt_prob >= alt_min)
                    fused = base_prob[rows].copy()
                    fused[gate] = np.maximum(fused[gate], alt_prob[gate])
                    metrics = metrics_from_binary_prob(y_true[rows], fused, 0.5)
                    key = (
                        metrics["balanced_accuracy"],
                        metrics["auc"],
                        metrics["real_accuracy"],
                        metrics["accuracy"],
                        metrics["fake_accuracy"],
                    )
                    candidate = {
                        "key": key,
                        "margin_tau": float(margin_tau),
                        "alt_min": float(alt_min),
                        "alt_source": alt_source,
                        "val_balanced_accuracy": metrics["balanced_accuracy"],
                        "val_accuracy": metrics["accuracy"],
                        "val_real_accuracy": metrics["real_accuracy"],
                        "val_auc": metrics["auc"],
                        "num_rows": int(len(rows)),
                    }
                    if best is None or key > best["key"]:
                        best = candidate
        assert best is not None
        best_by_group[group] = {k: v for k, v in best.items() if k != "key"}
    return best_by_group


def apply_semantic_max_rescue(
    route_prob: np.ndarray,
    base_prob: np.ndarray,
    pair_prob: np.ndarray,
    route_patch_prob: np.ndarray,
    rescue_choice: dict,
) -> np.ndarray:
    bucket_idx = top_fake_group(route_prob)
    margin = route_margin_vs_real(route_prob)
    fused = base_prob.copy().astype(np.float32)
    for group_idx, group in enumerate(FAKE_GROUPS):
        rows = bucket_idx == group_idx
        if not np.any(rows):
            continue
        choice = rescue_choice[group]
        alt_prob = resolve_alt_prob(choice["alt_source"], pair_prob[rows], route_patch_prob[rows])
        gate = (margin[rows] <= choice["margin_tau"]) & (alt_prob >= choice["alt_min"])
        fused_group = fused[rows]
        fused_group[gate] = np.maximum(fused_group[gate], alt_prob[gate])
        fused[rows] = fused_group
    return fused.astype(np.float32)


def build_meta_features(
    route_prob: np.ndarray,
    pair_scores: np.ndarray,
    patch_scores: np.ndarray,
    patch_global_prob: np.ndarray,
) -> np.ndarray:
    fake_route = route_prob[:, FAKE_ROUTE_IDXS].astype(np.float32)
    fake_sum = np.clip(fake_route.sum(axis=1, keepdims=True), 1e-8, None)
    fake_route_norm = fake_route / fake_sum
    route_fake = route_fake_score(route_prob)[:, None]
    pair_dynamic = np.sum(fake_route_norm * pair_scores, axis=1, keepdims=True).astype(np.float32)
    patch_dynamic = np.sum(fake_route_norm * patch_scores, axis=1, keepdims=True).astype(np.float32)
    route_entropy = (-(route_prob * np.log(np.clip(route_prob, 1e-8, 1.0))).sum(axis=1, keepdims=True)).astype(np.float32)
    max_fake = fake_route.max(axis=1, keepdims=True).astype(np.float32)
    real_prob = route_prob[:, [REAL_IDX]].astype(np.float32)
    patch_global_prob = patch_global_prob[:, None].astype(np.float32)
    interaction_route_pair = fake_route_norm * pair_scores
    interaction_route_patch = fake_route_norm * patch_scores
    interaction_patch_route = patch_dynamic * fake_route_norm
    interaction_patch_pair = patch_dynamic * pair_scores
    parts = [
        route_fake,
        patch_global_prob,
        patch_dynamic,
        pair_dynamic,
        real_prob,
        route_entropy,
        max_fake,
        fake_route_norm,
        pair_scores.astype(np.float32),
        patch_scores.astype(np.float32),
        interaction_route_pair.astype(np.float32),
        interaction_route_patch.astype(np.float32),
        interaction_patch_route.astype(np.float32),
        interaction_patch_pair.astype(np.float32),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_branch_features(
    cache,
    *,
    route_map: dict[str, np.ndarray],
    real_pool_means: np.ndarray,
    pair_mean_dirs: dict[str, np.ndarray],
    pair_classifiers: dict[str, tuple[StandardScaler, LogisticRegression]],
    pair_region_idx: dict[str, int],
    pair_region_names_by_group: dict[str, list[str]],
    patch_scaler,
    patch_clf,
    patch_group_classifiers: dict[str, tuple[StandardScaler, LogisticRegression]],
    pair_route_mode: str,
) -> BranchFeatures:
    x = image_region_delta_features(cache, real_pool_means)
    patch_prob = predict_patch_scores(patch_scaler, patch_clf, x)
    route_prob = route_prob_from_map(route_map, cache.img_id.astype(object))
    route_score = route_fake_score(route_prob)
    patch_scores = predict_group_binary_experts(patch_group_classifiers, x)
    route_patch_score = dynamic_expert_score(route_prob, patch_scores, pair_route_mode)
    pair_scores = pair_score_matrix(
        cache.region_vecs.astype(np.float32),
        real_pool_means,
        pair_mean_dirs,
        pair_classifiers,
        pair_region_idx,
        pair_region_names_by_group,
    )
    pair_score = dynamic_pair_score(route_prob, pair_scores, pair_route_mode)
    meta_x = build_meta_features(route_prob, pair_scores, patch_scores, patch_prob)
    return BranchFeatures(
        x=x,
        route_prob=route_prob,
        route_score=route_score,
        patch_prob=patch_prob,
        patch_scores=patch_scores,
        route_patch_score=route_patch_score,
        pair_scores=pair_scores,
        pair_score=pair_score,
        meta_x=meta_x,
    )


def fit_route_aware_meta_experts(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_groups: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
) -> tuple[dict[str, tuple[StandardScaler, LogisticRegression]], dict]:
    experts: dict[str, tuple[StandardScaler, LogisticRegression]] = {}
    search: dict[str, dict] = {}
    for offset, group in enumerate(FAKE_GROUPS):
        fit_rows = fit_idx[(train_y[fit_idx] == 0) | (train_groups[fit_idx] == group)]
        val_rows = val_idx[(train_y[val_idx] == 0) | (train_groups[val_idx] == group)]
        best = None
        for c_value in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
            scaler = StandardScaler()
            x_fit = scaler.fit_transform(train_x[fit_rows].astype(np.float64))
            x_val = scaler.transform(train_x[val_rows].astype(np.float64))
            x_fit = np.clip(np.nan_to_num(x_fit, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
            x_val = np.clip(np.nan_to_num(x_val, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
            clf = LogisticRegression(
                C=c_value,
                max_iter=3000,
                solver="liblinear",
                class_weight="balanced",
                random_state=seed + offset,
            )
            clf.fit(x_fit, train_y[fit_rows])
            val_prob = clf.predict_proba(x_val)[:, 1]
            metrics = metrics_from_binary_prob(train_y[val_rows], val_prob, 0.5)
            key = (metrics["balanced_accuracy"], metrics["accuracy"], metrics["auc"], -abs(c_value - 0.5))
            candidate = {
                "key": key,
                "c": float(c_value),
                "scaler": scaler,
                "clf": clf,
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_accuracy": metrics["accuracy"],
                "val_auc": metrics["auc"],
            }
            if best is None or key > best["key"]:
                best = candidate
        assert best is not None
        experts[group] = (best["scaler"], best["clf"])
        search[group] = {
            "c": best["c"],
            "val_balanced_accuracy": best["val_balanced_accuracy"],
            "val_accuracy": best["val_accuracy"],
            "val_auc": best["val_auc"],
        }
    return experts, search


def fit_bucket_meta_experts(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_route_prob: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
) -> tuple[dict[str, tuple[StandardScaler, LogisticRegression]], dict]:
    bucket_idx = top_fake_group(train_route_prob)
    global_scaler, global_clf, global_search = fit_meta_fusion_model(train_x, train_y, fit_idx, val_idx, seed)
    experts: dict[str, tuple[StandardScaler, LogisticRegression]] = {}
    search: dict[str, dict] = {}
    for offset, group in enumerate(FAKE_GROUPS):
        fit_rows = fit_idx[bucket_idx[fit_idx] == offset]
        val_rows = val_idx[bucket_idx[val_idx] == offset]
        if len(fit_rows) < 20 or len(val_rows) < 20 or len(np.unique(train_y[fit_rows])) < 2 or len(np.unique(train_y[val_rows])) < 2:
            experts[group] = (global_scaler, global_clf)
            search[group] = {
                "used_global_fallback": True,
                "c": global_search["c"],
                "num_fit_rows": int(len(fit_rows)),
                "num_val_rows": int(len(val_rows)),
                "val_balanced_accuracy": global_search["val_balanced_accuracy"],
                "val_accuracy": global_search["val_accuracy"],
                "val_auc": global_search["val_auc"],
            }
            continue
        best = None
        for c_value in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
            scaler = StandardScaler()
            x_fit = scaler.fit_transform(train_x[fit_rows].astype(np.float64))
            x_val = scaler.transform(train_x[val_rows].astype(np.float64))
            x_fit = np.clip(np.nan_to_num(x_fit, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
            x_val = np.clip(np.nan_to_num(x_val, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
            clf = LogisticRegression(
                C=c_value,
                max_iter=3000,
                solver="liblinear",
                class_weight="balanced",
                random_state=seed + offset,
            )
            clf.fit(x_fit, train_y[fit_rows])
            val_prob = clf.predict_proba(x_val)[:, 1]
            metrics = metrics_from_binary_prob(train_y[val_rows], val_prob, 0.5)
            key = (metrics["balanced_accuracy"], metrics["accuracy"], metrics["auc"], -abs(c_value - 0.5))
            candidate = {
                "key": key,
                "c": float(c_value),
                "scaler": scaler,
                "clf": clf,
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_accuracy": metrics["accuracy"],
                "val_auc": metrics["auc"],
                "num_fit_rows": int(len(fit_rows)),
                "num_val_rows": int(len(val_rows)),
            }
            if best is None or key > best["key"]:
                best = candidate
        assert best is not None
        experts[group] = (best["scaler"], best["clf"])
        search[group] = {
            "used_global_fallback": False,
            "c": best["c"],
            "num_fit_rows": best["num_fit_rows"],
            "num_val_rows": best["num_val_rows"],
            "val_balanced_accuracy": best["val_balanced_accuracy"],
            "val_accuracy": best["val_accuracy"],
            "val_auc": best["val_auc"],
        }
    return experts, search


def predict_bucket_meta_experts(
    experts: dict[str, tuple[StandardScaler, LogisticRegression]],
    route_prob: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    bucket_idx = top_fake_group(route_prob)
    out = np.zeros(len(x), dtype=np.float32)
    for group_idx, group in enumerate(FAKE_GROUPS):
        rows = bucket_idx == group_idx
        if not np.any(rows):
            continue
        scaler, clf = experts[group]
        out[rows] = predict_meta_fusion(scaler, clf, x[rows])
    return out.astype(np.float32)


def predict_route_aware_meta_fusion(
    experts: dict[str, tuple[StandardScaler, LogisticRegression]],
    route_prob: np.ndarray,
    x: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    expert_prob = np.zeros((len(x), len(FAKE_GROUPS)), dtype=np.float32)
    for idx, group in enumerate(FAKE_GROUPS):
        scaler, clf = experts[group]
        expert_prob[:, idx] = predict_meta_fusion(scaler, clf, x)
    mixed = dynamic_expert_score(route_prob, expert_prob, mode)
    return mixed.astype(np.float32), expert_prob.astype(np.float32)


def predict_meta_fusion(scaler: StandardScaler, clf: LogisticRegression, x: np.ndarray) -> np.ndarray:
    x_std = scaler.transform(x.astype(np.float64))
    x_std = np.clip(np.nan_to_num(x_std, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
    return clf.predict_proba(x_std)[:, 1].astype(np.float32)


def fit_meta_fusion_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression, dict]:
    best = None
    for c_value in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(train_x[fit_idx].astype(np.float64))
        x_val = scaler.transform(train_x[val_idx].astype(np.float64))
        x_fit = np.clip(np.nan_to_num(x_fit, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
        x_val = np.clip(np.nan_to_num(x_val, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
        clf = LogisticRegression(
            C=c_value,
            max_iter=3000,
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
        )
        clf.fit(x_fit, train_y[fit_idx])
        val_prob = clf.predict_proba(x_val)[:, 1]
        val_pred = (val_prob >= 0.5).astype(np.int64)
        fake_mask = train_y[val_idx] == 1
        real_mask = train_y[val_idx] == 0
        fake_acc = float(np.mean(val_pred[fake_mask] == 1)) if np.any(fake_mask) else 0.0
        real_acc = float(np.mean(val_pred[real_mask] == 0)) if np.any(real_mask) else 0.0
        bacc = 0.5 * (fake_acc + real_acc)
        acc = float(np.mean(val_pred == train_y[val_idx]))
        auc = float(roc_auc_score(train_y[val_idx], val_prob))
        key = (bacc, acc, auc, -abs(c_value - 0.5))
        candidate = {
            "key": key,
            "c": float(c_value),
            "val_balanced_accuracy": bacc,
            "val_accuracy": acc,
            "val_auc": auc,
            "scaler": scaler,
            "clf": clf,
        }
        if best is None or key > best["key"]:
            best = candidate
    assert best is not None
    return best["scaler"], best["clf"], {
        "c": best["c"],
        "val_balanced_accuracy": best["val_balanced_accuracy"],
        "val_accuracy": best["val_accuracy"],
        "val_auc": best["val_auc"],
    }


def top_fake_group(route_prob: np.ndarray) -> np.ndarray:
    return np.argmax(route_prob[:, FAKE_ROUTE_IDXS], axis=1).astype(np.int64)


def route_margin_vs_real(route_prob: np.ndarray) -> np.ndarray:
    return (np.max(route_prob[:, FAKE_ROUTE_IDXS], axis=1) - route_prob[:, REAL_IDX]).astype(np.float32)


def top2_fake_groups(route_prob: np.ndarray) -> np.ndarray:
    return np.argsort(-route_prob[:, FAKE_ROUTE_IDXS], axis=1)[:, :2].astype(np.int64)


def apply_selective_pair_fusion(
    base_prob: np.ndarray,
    pair_prob: np.ndarray,
    route_prob: np.ndarray,
    margin_tau: float,
    pair_lambda: float,
    gate_mode: str,
) -> np.ndarray:
    margin = route_margin_vs_real(route_prob)
    candidate = margin <= margin_tau
    top_group = top_fake_group(route_prob)
    if gate_mode == "fs_fr":
        candidate = candidate & ((top_group == 1) | (top_group == 2))
    elif gate_mode == "non_efs":
        candidate = candidate & (top_group != 0)
    elif gate_mode != "all":
        raise ValueError(f"Unknown gate_mode: {gate_mode}")
    fused = base_prob.copy()
    fused[candidate] = (1.0 - pair_lambda) * fused[candidate] + pair_lambda * pair_prob[candidate]
    return fused.astype(np.float32)


def apply_ambiguity_fusion(
    base_prob: np.ndarray,
    route_prob: np.ndarray,
    pair_prob: np.ndarray,
    route_patch_prob: np.ndarray,
    fsfr_margin_tau: float,
    efs_margin_tau: float,
    fsfr_lambda: float,
    efs_lambda: float,
) -> np.ndarray:
    margin = route_margin_vs_real(route_prob)
    top_group = top_fake_group(route_prob)
    top2 = top2_fake_groups(route_prob)
    fsfr_mask = (
        (margin <= fsfr_margin_tau)
        & np.all(np.isin(top2, np.array([1, 2], dtype=np.int64)), axis=1)
    )
    efs_mask = (margin <= efs_margin_tau) & (top_group == 0)
    alt_prob = np.maximum(pair_prob, route_patch_prob).astype(np.float32)
    fused = base_prob.copy().astype(np.float32)
    fused[fsfr_mask] = (1.0 - fsfr_lambda) * fused[fsfr_mask] + fsfr_lambda * alt_prob[fsfr_mask]
    fused[efs_mask] = (1.0 - efs_lambda) * fused[efs_mask] + efs_lambda * alt_prob[efs_mask]
    return fused.astype(np.float32)


def search_ambiguity_fusion(
    y_true: np.ndarray,
    route_prob: np.ndarray,
    base_prob: np.ndarray,
    pair_prob: np.ndarray,
    route_patch_prob: np.ndarray,
) -> dict:
    best = None
    for fsfr_margin_tau in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]:
        for efs_margin_tau in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]:
            for fsfr_lambda in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                for efs_lambda in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                    fused = apply_ambiguity_fusion(
                        base_prob,
                        route_prob,
                        pair_prob,
                        route_patch_prob,
                        fsfr_margin_tau,
                        efs_margin_tau,
                        fsfr_lambda,
                        efs_lambda,
                    )
                    metrics = metrics_from_binary_prob(y_true, fused, 0.5)
                    key = (
                        metrics["balanced_accuracy"],
                        metrics["auc"],
                        metrics["real_accuracy"],
                        metrics["accuracy"],
                        metrics["fake_accuracy"],
                    )
                    candidate = {
                        "key": key,
                        "fsfr_margin_tau": float(fsfr_margin_tau),
                        "efs_margin_tau": float(efs_margin_tau),
                        "fsfr_lambda": float(fsfr_lambda),
                        "efs_lambda": float(efs_lambda),
                        "val_balanced_accuracy": metrics["balanced_accuracy"],
                        "val_accuracy": metrics["accuracy"],
                        "val_auc": metrics["auc"],
                    }
                    if best is None or key > best["key"]:
                        best = candidate
    assert best is not None
    return {k: v for k, v in best.items() if k != "key"}


def route_fake_gap(route_prob: np.ndarray) -> np.ndarray:
    fake_prob = route_prob[:, FAKE_ROUTE_IDXS]
    sorted_prob = np.sort(fake_prob, axis=1)
    return (sorted_prob[:, -1] - sorted_prob[:, -2]).astype(np.float32)


def route_uncertainty(route_prob: np.ndarray, mode: str) -> np.ndarray:
    if mode == "margin_real":
        return (-route_margin_vs_real(route_prob)).astype(np.float32)
    if mode == "fake_gap":
        return (-route_fake_gap(route_prob)).astype(np.float32)
    if mode == "entropy":
        return (-(route_prob * np.log(np.clip(route_prob, 1e-8, 1.0))).sum(axis=1)).astype(np.float32)
    raise ValueError(f"Unknown uncertainty mode: {mode}")


def apply_uncertainty_residual_fusion(
    base_prob: np.ndarray,
    route_prob: np.ndarray,
    pair_prob: np.ndarray,
    route_patch_prob: np.ndarray,
    uncertainty_mode: str,
    uncertainty_quantile: float,
    pair_weight: float,
    lambda_residual: float,
    update_mode: str,
) -> np.ndarray:
    uncertainty = route_uncertainty(route_prob, uncertainty_mode)
    threshold = float(np.quantile(uncertainty, uncertainty_quantile))
    gate = uncertainty >= threshold
    residual_prob = (
        float(pair_weight) * pair_prob + (1.0 - float(pair_weight)) * route_patch_prob
    ).astype(np.float32)
    fused = base_prob.copy().astype(np.float32)
    if update_mode == "mix":
        fused[gate] = (1.0 - lambda_residual) * fused[gate] + lambda_residual * residual_prob[gate]
    elif update_mode == "max":
        fused[gate] = np.maximum(fused[gate], residual_prob[gate])
    else:
        raise ValueError(f"Unknown update_mode: {update_mode}")
    return fused.astype(np.float32)


def search_uncertainty_residual_fusion(
    y_true: np.ndarray,
    route_prob: np.ndarray,
    pair_prob: np.ndarray,
    route_patch_prob: np.ndarray,
    base_candidates: dict[str, np.ndarray],
) -> dict:
    best = None
    for base_name, base_prob in base_candidates.items():
        for uncertainty_mode in ["margin_real", "fake_gap", "entropy"]:
            for uncertainty_quantile in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                for pair_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    for lambda_residual in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                        for update_mode in ["mix", "max"]:
                            fused = apply_uncertainty_residual_fusion(
                                base_prob,
                                route_prob,
                                pair_prob,
                                route_patch_prob,
                                uncertainty_mode,
                                uncertainty_quantile,
                                pair_weight,
                                lambda_residual,
                                update_mode,
                            )
                            metrics = metrics_from_binary_prob(y_true, fused, 0.5)
                            key = (
                                metrics["balanced_accuracy"],
                                metrics["auc"],
                                metrics["real_accuracy"],
                                metrics["accuracy"],
                                metrics["fake_accuracy"],
                            )
                            candidate = {
                                "key": key,
                                "base_name": base_name,
                                "uncertainty_mode": uncertainty_mode,
                                "uncertainty_quantile": float(uncertainty_quantile),
                                "pair_weight": float(pair_weight),
                                "patch_weight": float(1.0 - pair_weight),
                                "lambda_residual": float(lambda_residual),
                                "update_mode": update_mode,
                                "val_balanced_accuracy": metrics["balanced_accuracy"],
                                "val_accuracy": metrics["accuracy"],
                                "val_auc": metrics["auc"],
                            }
                            if best is None or key > best["key"]:
                                best = candidate
    assert best is not None
    return {k: v for k, v in best.items() if k != "key"}


def route_patch_base_prob(
    route_score: np.ndarray,
    patch_prob: np.ndarray,
    fusion_choice: dict,
) -> np.ndarray:
    weight_sum = fusion_choice["weight_route"] + fusion_choice["weight_patch"]
    if weight_sum <= 1e-8:
        return route_score.astype(np.float32)
    return (
        (fusion_choice["weight_route"] * route_score + fusion_choice["weight_patch"] * patch_prob) / weight_sum
    ).astype(np.float32)


def search_selective_pair_fusion(
    y_true: np.ndarray,
    route_prob: np.ndarray,
    pair_prob: np.ndarray,
    base_candidates: dict[str, np.ndarray],
) -> dict:
    best = None
    for base_mode, base_prob in base_candidates.items():
        for gate_mode in ["all", "fs_fr", "non_efs"]:
            for margin_tau in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
                for pair_lambda in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                    fused = apply_selective_pair_fusion(base_prob, pair_prob, route_prob, margin_tau, pair_lambda, gate_mode)
                    pred = (fused >= 0.5).astype(np.int64)
                    fake_mask = y_true == 1
                    real_mask = y_true == 0
                    fake_acc = float(np.mean(pred[fake_mask] == 1)) if np.any(fake_mask) else 0.0
                    real_acc = float(np.mean(pred[real_mask] == 0)) if np.any(real_mask) else 0.0
                    bacc = 0.5 * (fake_acc + real_acc)
                    acc = float(np.mean(pred == y_true))
                    auc = float(roc_auc_score(y_true, fused))
                    key = (bacc, auc, real_acc, acc, fake_acc)
                    if best is None or key > best["key"]:
                        best = {
                            "key": key,
                            "base_mode": base_mode,
                            "gate_mode": gate_mode,
                            "margin_tau": float(margin_tau),
                            "pair_lambda": float(pair_lambda),
                            "val_balanced_accuracy": bacc,
                            "val_accuracy": acc,
                            "val_real_accuracy": real_acc,
                            "val_auc": auc,
                        }
    assert best is not None
    return {k: v for k, v in best.items() if k != "key"}


def metrics_from_binary_prob(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(np.int64)
    fake_mask = y_true == 1
    real_mask = y_true == 0
    fake_acc = float(np.mean(pred[fake_mask] == 1)) if np.any(fake_mask) else 0.0
    real_acc = float(np.mean(pred[real_mask] == 0)) if np.any(real_mask) else 0.0
    return {
        "accuracy": float(np.mean(pred == y_true)),
        "balanced_accuracy": 0.5 * (fake_acc + real_acc),
        "fake_accuracy": fake_acc,
        "real_accuracy": real_acc,
        "auc": float(roc_auc_score(y_true, prob)),
    }


def search_threshold(y_true: np.ndarray, prob: np.ndarray) -> dict:
    best = None
    for threshold in np.arange(0.2, 0.801, 0.01):
        metrics = metrics_from_binary_prob(y_true, prob, float(threshold))
        key = (
            metrics["balanced_accuracy"],
            metrics["auc"],
            metrics["real_accuracy"],
            metrics["accuracy"],
            metrics["fake_accuracy"],
        )
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "threshold": float(threshold),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "fake_accuracy": metrics["fake_accuracy"],
                "real_accuracy": metrics["real_accuracy"],
                "auc": metrics["auc"],
            }
    assert best is not None
    return {k: v for k, v in best.items() if k != "key"}


def search_bucket_thresholds(
    y_true: np.ndarray,
    prob: np.ndarray,
    route_prob: np.ndarray,
) -> dict:
    bucket_idx = top_fake_group(route_prob)
    global_choice = search_threshold(y_true, prob)
    thresholds_by_group: dict[str, float] = {}
    search_by_group: dict[str, dict] = {}
    for group_idx, group in enumerate(FAKE_GROUPS):
        rows = np.flatnonzero(bucket_idx == group_idx)
        if len(rows) == 0 or len(np.unique(y_true[rows])) < 2:
            thresholds_by_group[group] = float(global_choice["threshold"])
            search_by_group[group] = {
                "used_global_fallback": True,
                "threshold": float(global_choice["threshold"]),
                "balanced_accuracy": global_choice["balanced_accuracy"],
                "accuracy": global_choice["accuracy"],
                "real_accuracy": global_choice["real_accuracy"],
                "auc": global_choice["auc"],
                "num_rows": int(len(rows)),
            }
            continue
        choice = search_threshold(y_true[rows], prob[rows])
        thresholds_by_group[group] = float(choice["threshold"])
        search_by_group[group] = {
            "used_global_fallback": False,
            "threshold": float(choice["threshold"]),
            "balanced_accuracy": choice["balanced_accuracy"],
            "accuracy": choice["accuracy"],
            "real_accuracy": choice["real_accuracy"],
            "auc": choice["auc"],
            "num_rows": int(len(rows)),
        }
    pred = predict_with_bucket_threshold(
        route_prob,
        prob,
        {
            "thresholds_by_group": thresholds_by_group,
        },
    )
    fake_mask = y_true == 1
    real_mask = y_true == 0
    fake_acc = float(np.mean(pred[fake_mask] == 1)) if np.any(fake_mask) else 0.0
    real_acc = float(np.mean(pred[real_mask] == 0)) if np.any(real_mask) else 0.0
    current = {
        "balanced_accuracy": 0.5 * (fake_acc + real_acc),
        "accuracy": float(np.mean(pred == y_true)),
        "fake_accuracy": fake_acc,
        "real_accuracy": real_acc,
        "auc": float(roc_auc_score(y_true, prob)),
    }
    return {
        "global": global_choice,
        "current": current,
        "thresholds_by_group": thresholds_by_group,
        "search_by_group": search_by_group,
    }


def predict_with_bucket_threshold(route_prob: np.ndarray, prob: np.ndarray, bucket_threshold_choice: dict) -> np.ndarray:
    bucket_idx = top_fake_group(route_prob)
    threshold = np.zeros(len(prob), dtype=np.float32)
    for group_idx, group in enumerate(FAKE_GROUPS):
        threshold[bucket_idx == group_idx] = float(bucket_threshold_choice["thresholds_by_group"][group])
    return (prob >= threshold).astype(np.int64)


def method_metrics_with_threshold(prob_fake: np.ndarray, prob_real: np.ndarray, threshold: float) -> dict:
    y = np.concatenate([np.ones(len(prob_fake), dtype=np.int64), np.zeros(len(prob_real), dtype=np.int64)], axis=0)
    prob = np.concatenate([prob_fake, prob_real], axis=0)
    metrics = metrics_from_binary_prob(y, prob, threshold)
    return {
        "num_fake": int(len(prob_fake)),
        "num_real": int(len(prob_real)),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "fake_accuracy": metrics["fake_accuracy"],
        "real_accuracy": metrics["real_accuracy"],
        "auc": metrics["auc"],
    }


def method_metrics_with_bucket_threshold(
    prob_fake: np.ndarray,
    route_prob_fake: np.ndarray,
    prob_real: np.ndarray,
    route_prob_real: np.ndarray,
    bucket_threshold_choice: dict,
) -> dict:
    y = np.concatenate([np.ones(len(prob_fake), dtype=np.int64), np.zeros(len(prob_real), dtype=np.int64)], axis=0)
    prob = np.concatenate([prob_fake, prob_real], axis=0)
    route_prob = np.concatenate([route_prob_fake, route_prob_real], axis=0)
    pred = predict_with_bucket_threshold(route_prob, prob, bucket_threshold_choice)
    fake_mask = y == 1
    real_mask = y == 0
    fake_acc = float(np.mean(pred[fake_mask] == 1)) if np.any(fake_mask) else 0.0
    real_acc = float(np.mean(pred[real_mask] == 0)) if np.any(real_mask) else 0.0
    return {
        "num_fake": int(len(prob_fake)),
        "num_real": int(len(prob_real)),
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": 0.5 * (fake_acc + real_acc),
        "fake_accuracy": fake_acc,
        "real_accuracy": real_acc,
        "auc": float(roc_auc_score(y, prob)),
    }


def summarize_method_rows(rows: list[dict]) -> dict:
    return {"summary": summarize_split(rows), "methods": sorted(rows, key=lambda row: (row["group"], row["method"]))}


def select_auto_generic_model(
    validation: dict,
    threshold_search: dict,
    available_models: list[str],
    candidate_keys: list[str] | tuple[str, ...] | None = None,
) -> dict:
    return resolve_auto_generic_from_validation(validation, threshold_search, available_models, candidate_keys=candidate_keys)


def export_csv(output_csv: Path, report: dict) -> None:
    rows = []
    for split_name in ["test_ff", "ood"]:
        for model_name in EXPORT_MODEL_KEYS:
            for row in report[split_name][model_name]["methods"]:
                metrics = row["metrics"]
                rows.append(
                    {
                        "split": split_name,
                        "model": model_name,
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


def resolve_alt_prob(alt_source: str, pair_prob: np.ndarray, route_patch_prob: np.ndarray) -> np.ndarray:
    if alt_source == "pair":
        return pair_prob.astype(np.float32)
    if alt_source == "route_patch":
        return route_patch_prob.astype(np.float32)
    if alt_source == "max":
        return np.maximum(pair_prob, route_patch_prob).astype(np.float32)
    if alt_source == "blend":
        return (0.5 * pair_prob + 0.5 * route_patch_prob).astype(np.float32)
    raise ValueError(f"Unknown alt_source: {alt_source}")


def apply_generic_blend_fusion(prob_by_model: dict[str, np.ndarray], choice: dict) -> np.ndarray:
    fused = np.zeros_like(next(iter(prob_by_model.values())), dtype=np.float32)
    for model_name, weight in choice["weights_by_model"].items():
        fused += float(weight) * prob_by_model[model_name]
    return fused.astype(np.float32)


def search_generic_blend_fusion(
    y_true: np.ndarray,
    prob_by_model: dict[str, np.ndarray],
    threshold_search: dict[str, dict],
    base_models: list[str] | tuple[str, ...] | None = None,
) -> dict:
    pool = GENERIC_BLEND_BASE_MODELS if base_models is None else list(base_models)
    available = [model for model in pool if model in prob_by_model]
    if len(available) < 2:
        raise ValueError("generic_blend_fusion requires at least two base models.")
    ranking = sorted(
        available,
        key=lambda model: (
            threshold_search[model]["balanced_accuracy"],
            threshold_search[model]["auc"],
            threshold_search[model].get("real_accuracy", 0.0),
            threshold_search[model]["accuracy"],
        ),
        reverse=True,
    )
    best = None
    for top_k in range(2, len(ranking) + 1):
        selected = ranking[:top_k]
        for power in [1.0, 2.0, 4.0]:
            raw_weights = np.asarray(
                [
                    max(float(threshold_search[model]["balanced_accuracy"]) - 0.5, 1e-6) ** power
                    for model in selected
                ],
                dtype=np.float32,
            )
            weights = raw_weights / np.clip(raw_weights.sum(), 1e-8, None)
            choice = {
                "selected_models": selected,
                "power": float(power),
                "weights_by_model": {model: float(weight) for model, weight in zip(selected, weights, strict=True)},
            }
            fused = apply_generic_blend_fusion(prob_by_model, choice)
            metrics = search_threshold(y_true, fused)
            key = (
                metrics["balanced_accuracy"],
                metrics["auc"],
                metrics["real_accuracy"],
                metrics["accuracy"],
                metrics["fake_accuracy"],
            )
            candidate = {
                "key": key,
                "selected_models": selected,
                "power": float(power),
                "weights_by_model": choice["weights_by_model"],
                "threshold": float(metrics["threshold"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "accuracy": float(metrics["accuracy"]),
                "fake_accuracy": float(metrics["fake_accuracy"]),
                "real_accuracy": float(metrics["real_accuracy"]),
                "auc": float(metrics["auc"]),
            }
            if best is None or key > best["key"]:
                best = candidate
    assert best is not None
    return {k: v for k, v in best.items() if k != "key"}


def evaluate_regular_split(
    fake_paths: list[Path],
    real_cache,
    *,
    args: argparse.Namespace,
    route_map: dict[str, np.ndarray],
    pair_mean_dirs: dict[str, np.ndarray],
    pair_classifiers: dict[str, tuple[StandardScaler, LogisticRegression]],
    pair_region_idx: dict[str, int],
    pair_region_names_by_group: dict[str, list[str]],
    patch_scaler,
    patch_clf,
    patch_group_classifiers: dict[str, tuple[StandardScaler, LogisticRegression]],
    real_pool_means: np.ndarray,
    fusion_choice: dict,
    threshold_choice: dict,
    meta_scaler: StandardScaler,
    meta_clf: LogisticRegression,
    route_meta_experts: dict[str, tuple[StandardScaler, LogisticRegression]],
    route_meta_bucket_threshold: dict,
    bucket_meta_experts: dict[str, tuple[StandardScaler, LogisticRegression]],
    bucket_meta_choice: dict,
    ambiguity_choice: dict,
    uncertainty_choice: dict,
    semantic_choice: dict,
    semantic_max_choice: dict,
    generic_blend_choice: dict,
    selective_pair_choice: dict,
    seed_offset: int = 0,
) -> dict:
    real_features = build_branch_features(
        real_cache,
        route_map=route_map,
        real_pool_means=real_pool_means,
        pair_mean_dirs=pair_mean_dirs,
        pair_classifiers=pair_classifiers,
        pair_region_idx=pair_region_idx,
        pair_region_names_by_group=pair_region_names_by_group,
        patch_scaler=patch_scaler,
        patch_clf=patch_clf,
        patch_group_classifiers=patch_group_classifiers,
        pair_route_mode=args.pair_route_mode,
    )
    real_route_patch_prob = route_patch_base_prob(real_features.route_score, real_features.patch_prob, fusion_choice)
    real_meta_prob = predict_meta_fusion(
        meta_scaler,
        meta_clf,
        real_features.meta_x,
    )
    real_route_meta_prob, _ = predict_route_aware_meta_fusion(
        route_meta_experts,
        real_features.route_prob,
        real_features.meta_x,
        args.pair_route_mode,
    )
    real_idx = np.arange(min(len(real_features.patch_prob), 2000))

    route_rows = []
    patch_rows = []
    route_patch_rows = []
    pair_rows = []
    fusion_rows = []
    meta_rows = []
    route_meta_rows = []
    route_meta_bucket_rows = []
    bucket_expert_rows = []
    bucket_meta_rows = []
    ambiguity_rows = []
    uncertainty_rows = []
    semantic_rows = []
    semantic_max_rows = []
    generic_blend_rows = []
    selective_rows = []
    for idx, path in enumerate(fake_paths):
        cache = load_compact_cache(
            path,
            max_rows=args.max_test_fake_per_method,
            seed=args.seed + seed_offset + idx,
            compact_cache_dir=args.compact_cache_dir,
        )
        fake_features = build_branch_features(
            cache,
            route_map=route_map,
            real_pool_means=real_pool_means,
            pair_mean_dirs=pair_mean_dirs,
            pair_classifiers=pair_classifiers,
            pair_region_idx=pair_region_idx,
            pair_region_names_by_group=pair_region_names_by_group,
            patch_scaler=patch_scaler,
            patch_clf=patch_clf,
            patch_group_classifiers=patch_group_classifiers,
            pair_route_mode=args.pair_route_mode,
        )
        fake_route_patch_prob = route_patch_base_prob(fake_features.route_score, fake_features.patch_prob, fusion_choice)
        fake_meta_prob = predict_meta_fusion(
            meta_scaler,
            meta_clf,
            fake_features.meta_x,
        )
        fake_route_meta_prob, _ = predict_route_aware_meta_fusion(
            route_meta_experts,
            fake_features.route_prob,
            fake_features.meta_x,
            args.pair_route_mode,
        )
        fake_bucket_expert_prob = predict_bucket_meta_experts(
            bucket_meta_experts,
            fake_features.route_prob,
            fake_features.meta_x,
        )
        real_bucket_expert_prob = predict_bucket_meta_experts(
            bucket_meta_experts,
            real_features.route_prob[real_idx],
            real_features.meta_x[real_idx],
        )
        fake_bucket_meta_prob = apply_bucket_meta_fusion(
            fake_features.route_prob,
            fake_features.route_score,
            fake_features.pair_score,
            fake_features.route_patch_score,
            fake_route_meta_prob,
            bucket_meta_choice,
        )
        real_bucket_meta_prob = apply_bucket_meta_fusion(
            real_features.route_prob[real_idx],
            real_features.route_score[real_idx],
            real_features.pair_score[real_idx],
            real_features.route_patch_score[real_idx],
            real_route_meta_prob[real_idx],
            bucket_meta_choice,
        )
        fake_ambiguity_prob = apply_ambiguity_fusion(
            fake_route_meta_prob,
            fake_features.route_prob,
            fake_features.pair_score,
            fake_features.route_patch_score,
            ambiguity_choice["fsfr_margin_tau"],
            ambiguity_choice["efs_margin_tau"],
            ambiguity_choice["fsfr_lambda"],
            ambiguity_choice["efs_lambda"],
        )
        real_ambiguity_prob = apply_ambiguity_fusion(
            real_route_meta_prob[real_idx],
            real_features.route_prob[real_idx],
            real_features.pair_score[real_idx],
            real_features.route_patch_score[real_idx],
            ambiguity_choice["fsfr_margin_tau"],
            ambiguity_choice["efs_margin_tau"],
            ambiguity_choice["fsfr_lambda"],
            ambiguity_choice["efs_lambda"],
        )
        fake_semantic_prob = apply_semantic_rescue(
            fake_features.route_prob,
            fake_route_meta_prob,
            fake_features.pair_score,
            fake_features.route_patch_score,
            semantic_choice,
        )
        real_semantic_prob = apply_semantic_rescue(
            real_features.route_prob[real_idx],
            real_route_meta_prob[real_idx],
            real_features.pair_score[real_idx],
            real_features.route_patch_score[real_idx],
            semantic_choice,
        )
        fake_semantic_max_prob = apply_semantic_max_rescue(
            fake_features.route_prob,
            fake_route_meta_prob,
            fake_features.pair_score,
            fake_features.route_patch_score,
            semantic_max_choice,
        )
        real_semantic_max_prob = apply_semantic_max_rescue(
            real_features.route_prob[real_idx],
            real_route_meta_prob[real_idx],
            real_features.pair_score[real_idx],
            real_features.route_patch_score[real_idx],
            semantic_max_choice,
        )
        route_rows.append({"group": str(cache.group[0]), "method": str(cache.method[0]), "metrics": method_metrics_with_threshold(fake_features.route_score, real_features.route_score[real_idx], threshold_choice["route_only"])})
        patch_rows.append({"group": str(cache.group[0]), "method": str(cache.method[0]), "metrics": method_metrics_with_threshold(fake_features.patch_prob, real_features.patch_prob[real_idx], threshold_choice["patch_only"])})
        route_patch_rows.append({"group": str(cache.group[0]), "method": str(cache.method[0]), "metrics": method_metrics_with_threshold(fake_features.route_patch_score, real_features.route_patch_score[real_idx], threshold_choice["route_patch_only"])})
        pair_rows.append({"group": str(cache.group[0]), "method": str(cache.method[0]), "metrics": method_metrics_with_threshold(fake_features.pair_score, real_features.pair_score[real_idx], threshold_choice["pair_only"])})
        fused_fake = (
            fusion_choice["weight_route"] * fake_features.route_score
            + fusion_choice["weight_pair"] * fake_features.pair_score
            + fusion_choice["weight_patch"] * fake_features.route_patch_score
        )
        fused_real = (
            fusion_choice["weight_route"] * real_features.route_score[real_idx]
            + fusion_choice["weight_pair"] * real_features.pair_score[real_idx]
            + fusion_choice["weight_patch"] * real_features.route_patch_score[real_idx]
        )
        uncertainty_base_map = {
            "route_meta": fake_route_meta_prob,
            "full_fusion": fused_fake,
            "bucket_meta": fake_bucket_meta_prob,
        }
        real_uncertainty_base_map = {
            "route_meta": real_route_meta_prob[real_idx],
            "full_fusion": fused_real,
            "bucket_meta": real_bucket_meta_prob,
        }
        fake_uncertainty_prob = apply_uncertainty_residual_fusion(
            uncertainty_base_map[uncertainty_choice["base_name"]],
            fake_features.route_prob,
            fake_features.pair_score,
            fake_features.route_patch_score,
            uncertainty_choice["uncertainty_mode"],
            uncertainty_choice["uncertainty_quantile"],
            uncertainty_choice["pair_weight"],
            uncertainty_choice["lambda_residual"],
            uncertainty_choice["update_mode"],
        )
        real_uncertainty_prob = apply_uncertainty_residual_fusion(
            real_uncertainty_base_map[uncertainty_choice["base_name"]],
            real_features.route_prob[real_idx],
            real_features.pair_score[real_idx],
            real_features.route_patch_score[real_idx],
            uncertainty_choice["uncertainty_mode"],
            uncertainty_choice["uncertainty_quantile"],
            uncertainty_choice["pair_weight"],
            uncertainty_choice["lambda_residual"],
            uncertainty_choice["update_mode"],
        )
        fake_generic_prob_map = {
            "full_fusion": fused_fake.astype(np.float32),
            "meta_fusion": fake_meta_prob.astype(np.float32),
            "route_meta_fusion": fake_route_meta_prob.astype(np.float32),
            "bucket_expert_fusion": fake_bucket_expert_prob.astype(np.float32),
            "bucket_meta_fusion": fake_bucket_meta_prob.astype(np.float32),
            "uncertainty_residual_fusion": fake_uncertainty_prob.astype(np.float32),
            "semantic_fusion": fake_semantic_prob.astype(np.float32),
            "semantic_max_fusion": fake_semantic_max_prob.astype(np.float32),
        }
        real_generic_prob_map = {
            "full_fusion": fused_real.astype(np.float32),
            "meta_fusion": real_meta_prob[real_idx].astype(np.float32),
            "route_meta_fusion": real_route_meta_prob[real_idx].astype(np.float32),
            "bucket_expert_fusion": real_bucket_expert_prob.astype(np.float32),
            "bucket_meta_fusion": real_bucket_meta_prob.astype(np.float32),
            "uncertainty_residual_fusion": real_uncertainty_prob.astype(np.float32),
            "semantic_fusion": real_semantic_prob.astype(np.float32),
            "semantic_max_fusion": real_semantic_max_prob.astype(np.float32),
        }
        fake_generic_blend_prob = apply_generic_blend_fusion(fake_generic_prob_map, generic_blend_choice)
        real_generic_blend_prob = apply_generic_blend_fusion(real_generic_prob_map, generic_blend_choice)
        fusion_rows.append({"group": str(cache.group[0]), "method": str(cache.method[0]), "metrics": method_metrics_with_threshold(fused_fake, fused_real, threshold_choice["full_fusion"])})
        meta_rows.append({"group": str(cache.group[0]), "method": str(cache.method[0]), "metrics": method_metrics_with_threshold(fake_meta_prob, real_meta_prob[real_idx], threshold_choice["meta_fusion"])})
        route_meta_rows.append({"group": str(cache.group[0]), "method": str(cache.method[0]), "metrics": method_metrics_with_threshold(fake_route_meta_prob, real_route_meta_prob[real_idx], threshold_choice["route_meta_fusion"])})
        route_meta_bucket_rows.append({"group": str(cache.group[0]), "method": str(cache.method[0]), "metrics": method_metrics_with_bucket_threshold(fake_route_meta_prob, fake_features.route_prob, real_route_meta_prob[real_idx], real_features.route_prob[real_idx], route_meta_bucket_threshold)})
        bucket_expert_rows.append({"group": str(cache.group[0]), "method": str(cache.method[0]), "metrics": method_metrics_with_threshold(fake_bucket_expert_prob, real_bucket_expert_prob, threshold_choice["bucket_expert_fusion"])})
        bucket_meta_rows.append({"group": str(cache.group[0]), "method": str(cache.method[0]), "metrics": method_metrics_with_threshold(fake_bucket_meta_prob, real_bucket_meta_prob, threshold_choice["bucket_meta_fusion"])})
        ambiguity_rows.append({"group": str(cache.group[0]), "method": str(cache.method[0]), "metrics": method_metrics_with_threshold(fake_ambiguity_prob, real_ambiguity_prob, threshold_choice["ambiguity_fusion"])})
        uncertainty_rows.append({"group": str(cache.group[0]), "method": str(cache.method[0]), "metrics": method_metrics_with_threshold(fake_uncertainty_prob, real_uncertainty_prob, threshold_choice["uncertainty_residual_fusion"])})
        semantic_rows.append({"group": str(cache.group[0]), "method": str(cache.method[0]), "metrics": method_metrics_with_threshold(fake_semantic_prob, real_semantic_prob, threshold_choice["semantic_fusion"])})
        semantic_max_rows.append({"group": str(cache.group[0]), "method": str(cache.method[0]), "metrics": method_metrics_with_threshold(fake_semantic_max_prob, real_semantic_max_prob, threshold_choice["semantic_max_fusion"])})
        generic_blend_rows.append({"group": str(cache.group[0]), "method": str(cache.method[0]), "metrics": method_metrics_with_threshold(fake_generic_blend_prob, real_generic_blend_prob, threshold_choice["generic_blend_fusion"])})
        selective_fake_base = {
            "full_fusion": fused_fake,
            "route_patch": fake_route_patch_prob,
            "route_only": fake_features.route_score,
        }[selective_pair_choice["base_mode"]]
        selective_real_base = {
            "full_fusion": fused_real,
            "route_patch": real_route_patch_prob[real_idx],
            "route_only": real_features.route_score[real_idx],
        }[selective_pair_choice["base_mode"]]
        selective_fake = apply_selective_pair_fusion(
            selective_fake_base,
            fake_features.pair_score,
            fake_features.route_prob,
            selective_pair_choice["margin_tau"],
            selective_pair_choice["pair_lambda"],
            selective_pair_choice["gate_mode"],
        )
        selective_real = apply_selective_pair_fusion(
            selective_real_base,
            real_features.pair_score[real_idx],
            real_features.route_prob[real_idx],
            selective_pair_choice["margin_tau"],
            selective_pair_choice["pair_lambda"],
            selective_pair_choice["gate_mode"],
        )
        selective_rows.append({"group": str(cache.group[0]), "method": str(cache.method[0]), "metrics": method_metrics_with_threshold(selective_fake, selective_real, threshold_choice["selective_pair_fusion"])})

    return {
        "route_only": summarize_method_rows(route_rows),
        "patch_only": summarize_method_rows(patch_rows),
        "route_patch_only": summarize_method_rows(route_patch_rows),
        "pair_only": summarize_method_rows(pair_rows),
        "full_fusion": summarize_method_rows(fusion_rows),
        "meta_fusion": summarize_method_rows(meta_rows),
        "route_meta_fusion": summarize_method_rows(route_meta_rows),
        "route_meta_bucket_threshold": summarize_method_rows(route_meta_bucket_rows),
        "bucket_expert_fusion": summarize_method_rows(bucket_expert_rows),
        "bucket_meta_fusion": summarize_method_rows(bucket_meta_rows),
        "ambiguity_fusion": summarize_method_rows(ambiguity_rows),
        "uncertainty_residual_fusion": summarize_method_rows(uncertainty_rows),
        "semantic_fusion": summarize_method_rows(semantic_rows),
        "semantic_max_fusion": summarize_method_rows(semantic_max_rows),
        "generic_blend_fusion": summarize_method_rows(generic_blend_rows),
        "selective_pair_fusion": summarize_method_rows(selective_rows),
    }


def evaluate_ood_split(
    ood_entries: list[tuple[str, str, Path, Path]],
    *,
    args: argparse.Namespace,
    route_map: dict[str, np.ndarray],
    pair_mean_dirs: dict[str, np.ndarray],
    pair_classifiers: dict[str, tuple[StandardScaler, LogisticRegression]],
    pair_region_idx: dict[str, int],
    pair_region_names_by_group: dict[str, list[str]],
    patch_scaler,
    patch_clf,
    patch_group_classifiers: dict[str, tuple[StandardScaler, LogisticRegression]],
    real_pool_means: np.ndarray,
    fusion_choice: dict,
    threshold_choice: dict,
    meta_scaler: StandardScaler,
    meta_clf: LogisticRegression,
    route_meta_experts: dict[str, tuple[StandardScaler, LogisticRegression]],
    route_meta_bucket_threshold: dict,
    bucket_meta_experts: dict[str, tuple[StandardScaler, LogisticRegression]],
    bucket_meta_choice: dict,
    ambiguity_choice: dict,
    uncertainty_choice: dict,
    semantic_choice: dict,
    semantic_max_choice: dict,
    generic_blend_choice: dict,
    selective_pair_choice: dict,
) -> dict:
    route_rows = []
    patch_rows = []
    route_patch_rows = []
    pair_rows = []
    fusion_rows = []
    meta_rows = []
    route_meta_rows = []
    route_meta_bucket_rows = []
    bucket_expert_rows = []
    bucket_meta_rows = []
    ambiguity_rows = []
    uncertainty_rows = []
    semantic_rows = []
    semantic_max_rows = []
    generic_blend_rows = []
    selective_rows = []
    for idx, (group, method, fake_path, real_path) in enumerate(ood_entries):
        fake_cache = load_compact_cache(
            fake_path,
            max_rows=args.max_ood_fake_per_method,
            seed=args.seed + idx,
            compact_cache_dir=args.compact_cache_dir,
        )
        real_cache = load_compact_cache(
            real_path,
            max_rows=args.max_ood_fake_per_method,
            seed=args.seed + 1000 + idx,
            compact_cache_dir=args.compact_cache_dir,
        )
        fake_features = build_branch_features(
            fake_cache,
            route_map=route_map,
            real_pool_means=real_pool_means,
            pair_mean_dirs=pair_mean_dirs,
            pair_classifiers=pair_classifiers,
            pair_region_idx=pair_region_idx,
            pair_region_names_by_group=pair_region_names_by_group,
            patch_scaler=patch_scaler,
            patch_clf=patch_clf,
            patch_group_classifiers=patch_group_classifiers,
            pair_route_mode=args.pair_route_mode,
        )
        real_features = build_branch_features(
            real_cache,
            route_map=route_map,
            real_pool_means=real_pool_means,
            pair_mean_dirs=pair_mean_dirs,
            pair_classifiers=pair_classifiers,
            pair_region_idx=pair_region_idx,
            pair_region_names_by_group=pair_region_names_by_group,
            patch_scaler=patch_scaler,
            patch_clf=patch_clf,
            patch_group_classifiers=patch_group_classifiers,
            pair_route_mode=args.pair_route_mode,
        )
        fake_route_patch_prob = route_patch_base_prob(fake_features.route_score, fake_features.patch_prob, fusion_choice)
        real_route_patch_prob = route_patch_base_prob(real_features.route_score, real_features.patch_prob, fusion_choice)
        fake_meta_prob = predict_meta_fusion(
            meta_scaler,
            meta_clf,
            fake_features.meta_x,
        )
        real_meta_prob = predict_meta_fusion(
            meta_scaler,
            meta_clf,
            real_features.meta_x,
        )
        fake_route_meta_prob, _ = predict_route_aware_meta_fusion(route_meta_experts, fake_features.route_prob, fake_features.meta_x, args.pair_route_mode)
        real_route_meta_prob, _ = predict_route_aware_meta_fusion(route_meta_experts, real_features.route_prob, real_features.meta_x, args.pair_route_mode)
        fake_bucket_expert_prob = predict_bucket_meta_experts(
            bucket_meta_experts,
            fake_features.route_prob,
            fake_features.meta_x,
        )
        real_bucket_expert_prob = predict_bucket_meta_experts(
            bucket_meta_experts,
            real_features.route_prob,
            real_features.meta_x,
        )
        fake_bucket_meta_prob = apply_bucket_meta_fusion(
            fake_features.route_prob,
            fake_features.route_score,
            fake_features.pair_score,
            fake_features.route_patch_score,
            fake_route_meta_prob,
            bucket_meta_choice,
        )
        real_bucket_meta_prob = apply_bucket_meta_fusion(
            real_features.route_prob,
            real_features.route_score,
            real_features.pair_score,
            real_features.route_patch_score,
            real_route_meta_prob,
            bucket_meta_choice,
        )
        fused_fake = (
            fusion_choice["weight_route"] * fake_features.route_score
            + fusion_choice["weight_pair"] * fake_features.pair_score
            + fusion_choice["weight_patch"] * fake_features.route_patch_score
        )
        fused_real = (
            fusion_choice["weight_route"] * real_features.route_score
            + fusion_choice["weight_pair"] * real_features.pair_score
            + fusion_choice["weight_patch"] * real_features.route_patch_score
        )
        uncertainty_base_map = {
            "route_meta": fake_route_meta_prob,
            "full_fusion": fused_fake,
            "bucket_meta": fake_bucket_meta_prob,
        }
        real_uncertainty_base_map = {
            "route_meta": real_route_meta_prob,
            "full_fusion": fused_real,
            "bucket_meta": real_bucket_meta_prob,
        }
        fake_uncertainty_prob = apply_uncertainty_residual_fusion(
            uncertainty_base_map[uncertainty_choice["base_name"]],
            fake_features.route_prob,
            fake_features.pair_score,
            fake_features.route_patch_score,
            uncertainty_choice["uncertainty_mode"],
            uncertainty_choice["uncertainty_quantile"],
            uncertainty_choice["pair_weight"],
            uncertainty_choice["lambda_residual"],
            uncertainty_choice["update_mode"],
        )
        real_uncertainty_prob = apply_uncertainty_residual_fusion(
            real_uncertainty_base_map[uncertainty_choice["base_name"]],
            real_features.route_prob,
            real_features.pair_score,
            real_features.route_patch_score,
            uncertainty_choice["uncertainty_mode"],
            uncertainty_choice["uncertainty_quantile"],
            uncertainty_choice["pair_weight"],
            uncertainty_choice["lambda_residual"],
            uncertainty_choice["update_mode"],
        )
        fake_ambiguity_prob = apply_ambiguity_fusion(
            fake_route_meta_prob,
            fake_features.route_prob,
            fake_features.pair_score,
            fake_features.route_patch_score,
            ambiguity_choice["fsfr_margin_tau"],
            ambiguity_choice["efs_margin_tau"],
            ambiguity_choice["fsfr_lambda"],
            ambiguity_choice["efs_lambda"],
        )
        real_ambiguity_prob = apply_ambiguity_fusion(
            real_route_meta_prob,
            real_features.route_prob,
            real_features.pair_score,
            real_features.route_patch_score,
            ambiguity_choice["fsfr_margin_tau"],
            ambiguity_choice["efs_margin_tau"],
            ambiguity_choice["fsfr_lambda"],
            ambiguity_choice["efs_lambda"],
        )
        fake_semantic_prob = apply_semantic_rescue(
            fake_features.route_prob,
            fake_route_meta_prob,
            fake_features.pair_score,
            fake_features.route_patch_score,
            semantic_choice,
        )
        real_semantic_prob = apply_semantic_rescue(
            real_features.route_prob,
            real_route_meta_prob,
            real_features.pair_score,
            real_features.route_patch_score,
            semantic_choice,
        )
        fake_semantic_max_prob = apply_semantic_max_rescue(
            fake_features.route_prob,
            fake_route_meta_prob,
            fake_features.pair_score,
            fake_features.route_patch_score,
            semantic_max_choice,
        )
        real_semantic_max_prob = apply_semantic_max_rescue(
            real_features.route_prob,
            real_route_meta_prob,
            real_features.pair_score,
            real_features.route_patch_score,
            semantic_max_choice,
        )
        fake_generic_prob_map = {
            "full_fusion": fused_fake.astype(np.float32),
            "meta_fusion": fake_meta_prob.astype(np.float32),
            "route_meta_fusion": fake_route_meta_prob.astype(np.float32),
            "bucket_expert_fusion": fake_bucket_expert_prob.astype(np.float32),
            "bucket_meta_fusion": fake_bucket_meta_prob.astype(np.float32),
            "uncertainty_residual_fusion": fake_uncertainty_prob.astype(np.float32),
            "semantic_fusion": fake_semantic_prob.astype(np.float32),
            "semantic_max_fusion": fake_semantic_max_prob.astype(np.float32),
        }
        real_generic_prob_map = {
            "full_fusion": fused_real.astype(np.float32),
            "meta_fusion": real_meta_prob.astype(np.float32),
            "route_meta_fusion": real_route_meta_prob.astype(np.float32),
            "bucket_expert_fusion": real_bucket_expert_prob.astype(np.float32),
            "bucket_meta_fusion": real_bucket_meta_prob.astype(np.float32),
            "uncertainty_residual_fusion": real_uncertainty_prob.astype(np.float32),
            "semantic_fusion": real_semantic_prob.astype(np.float32),
            "semantic_max_fusion": real_semantic_max_prob.astype(np.float32),
        }
        fake_generic_blend_prob = apply_generic_blend_fusion(fake_generic_prob_map, generic_blend_choice)
        real_generic_blend_prob = apply_generic_blend_fusion(real_generic_prob_map, generic_blend_choice)
        route_rows.append({"group": group, "method": method, "metrics": method_metrics_with_threshold(fake_features.route_score, real_features.route_score, threshold_choice["route_only"])})
        patch_rows.append({"group": group, "method": method, "metrics": method_metrics_with_threshold(fake_features.patch_prob, real_features.patch_prob, threshold_choice["patch_only"])})
        route_patch_rows.append({"group": group, "method": method, "metrics": method_metrics_with_threshold(fake_features.route_patch_score, real_features.route_patch_score, threshold_choice["route_patch_only"])})
        pair_rows.append({"group": group, "method": method, "metrics": method_metrics_with_threshold(fake_features.pair_score, real_features.pair_score, threshold_choice["pair_only"])})
        fusion_rows.append({"group": group, "method": method, "metrics": method_metrics_with_threshold(fused_fake, fused_real, threshold_choice["full_fusion"])})
        meta_rows.append({"group": group, "method": method, "metrics": method_metrics_with_threshold(fake_meta_prob, real_meta_prob, threshold_choice["meta_fusion"])})
        route_meta_rows.append({"group": group, "method": method, "metrics": method_metrics_with_threshold(fake_route_meta_prob, real_route_meta_prob, threshold_choice["route_meta_fusion"])})
        route_meta_bucket_rows.append({"group": group, "method": method, "metrics": method_metrics_with_bucket_threshold(fake_route_meta_prob, fake_features.route_prob, real_route_meta_prob, real_features.route_prob, route_meta_bucket_threshold)})
        bucket_expert_rows.append({"group": group, "method": method, "metrics": method_metrics_with_threshold(fake_bucket_expert_prob, real_bucket_expert_prob, threshold_choice["bucket_expert_fusion"])})
        bucket_meta_rows.append({"group": group, "method": method, "metrics": method_metrics_with_threshold(fake_bucket_meta_prob, real_bucket_meta_prob, threshold_choice["bucket_meta_fusion"])})
        ambiguity_rows.append({"group": group, "method": method, "metrics": method_metrics_with_threshold(fake_ambiguity_prob, real_ambiguity_prob, threshold_choice["ambiguity_fusion"])})
        uncertainty_rows.append({"group": group, "method": method, "metrics": method_metrics_with_threshold(fake_uncertainty_prob, real_uncertainty_prob, threshold_choice["uncertainty_residual_fusion"])})
        semantic_rows.append({"group": group, "method": method, "metrics": method_metrics_with_threshold(fake_semantic_prob, real_semantic_prob, threshold_choice["semantic_fusion"])})
        semantic_max_rows.append({"group": group, "method": method, "metrics": method_metrics_with_threshold(fake_semantic_max_prob, real_semantic_max_prob, threshold_choice["semantic_max_fusion"])})
        generic_blend_rows.append({"group": group, "method": method, "metrics": method_metrics_with_threshold(fake_generic_blend_prob, real_generic_blend_prob, threshold_choice["generic_blend_fusion"])})
        selective_fake_base = {
            "full_fusion": fused_fake,
            "route_patch": fake_route_patch_prob,
            "route_only": fake_features.route_score,
        }[selective_pair_choice["base_mode"]]
        selective_real_base = {
            "full_fusion": fused_real,
            "route_patch": real_route_patch_prob,
            "route_only": real_features.route_score,
        }[selective_pair_choice["base_mode"]]
        selective_fake = apply_selective_pair_fusion(
            selective_fake_base,
            fake_features.pair_score,
            fake_features.route_prob,
            selective_pair_choice["margin_tau"],
            selective_pair_choice["pair_lambda"],
            selective_pair_choice["gate_mode"],
        )
        selective_real = apply_selective_pair_fusion(
            selective_real_base,
            real_features.pair_score,
            real_features.route_prob,
            selective_pair_choice["margin_tau"],
            selective_pair_choice["pair_lambda"],
            selective_pair_choice["gate_mode"],
        )
        selective_rows.append({"group": group, "method": method, "metrics": method_metrics_with_threshold(selective_fake, selective_real, threshold_choice["selective_pair_fusion"])})

    return {
        "route_only": summarize_method_rows(route_rows),
        "patch_only": summarize_method_rows(patch_rows),
        "route_patch_only": summarize_method_rows(route_patch_rows),
        "pair_only": summarize_method_rows(pair_rows),
        "full_fusion": summarize_method_rows(fusion_rows),
        "meta_fusion": summarize_method_rows(meta_rows),
        "route_meta_fusion": summarize_method_rows(route_meta_rows),
        "route_meta_bucket_threshold": summarize_method_rows(route_meta_bucket_rows),
        "bucket_expert_fusion": summarize_method_rows(bucket_expert_rows),
        "bucket_meta_fusion": summarize_method_rows(bucket_meta_rows),
        "ambiguity_fusion": summarize_method_rows(ambiguity_rows),
        "uncertainty_residual_fusion": summarize_method_rows(uncertainty_rows),
        "semantic_fusion": summarize_method_rows(semantic_rows),
        "semantic_max_fusion": summarize_method_rows(semantic_max_rows),
        "generic_blend_fusion": summarize_method_rows(generic_blend_rows),
        "selective_pair_fusion": summarize_method_rows(selective_rows),
    }


def main() -> None:
    args = parse_args()
    if args.no_tuning_mainline:
        args.pair_region_mode = "all_regions"
        args.val_split_mode = "holdout_method"
        args.auto_generic_pool = "no_tuning"
    set_seed(args.seed)
    device = resolve_torch_device(args.device)

    train_real_cache = load_compact_cache(
        args.train_real_patch_cache,
        max_rows=args.train_real_max,
        seed=args.seed,
        compact_cache_dir=args.compact_cache_dir,
    )
    test_real_cache = load_compact_cache(
        args.test_real_patch_cache,
        max_rows=args.eval_real_max,
        seed=args.seed,
        compact_cache_dir=args.compact_cache_dir,
    )
    real_pool_means = compute_real_pool_region_means(train_real_cache)

    train_fake_x_parts = []
    train_fake_img_ids = []
    train_fake_methods = []
    train_fake_groups = []
    train_fake_region_vecs = []
    for idx, path in enumerate(discover_flat_method_caches(args.patch_cache_root, "DF40_train", list(args.patch_train_groups))):
        cache = load_compact_cache(
            path,
            max_rows=args.max_train_fake_per_method,
            seed=args.seed + idx,
            compact_cache_dir=args.compact_cache_dir,
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
    train_groups = np.concatenate([np.concatenate(train_fake_groups), np.full(len(train_real_x), REAL_LABEL, dtype=object)], axis=0)
    train_region_vecs = np.concatenate([np.concatenate(train_fake_region_vecs, axis=0), train_real_cache.region_vecs.astype(np.float32)], axis=0)
    n_fake_train = len(train_fake_x)
    train_split_keys = train_methods.astype(object).copy()
    train_split_keys[:n_fake_train] = np.asarray(
        [f"{group}::{method}" for group, method in zip(train_groups[:n_fake_train], train_methods[:n_fake_train], strict=True)],
        dtype=object,
    )
    fit_idx, val_idx = split_train_indices(
        train_split_keys,
        args.val_ratio,
        args.seed,
        n_fake_train,
        split_mode=args.val_split_mode,
    )

    patch_scaler, patch_clf = fit_patch_classifier(train_x[fit_idx], train_y[fit_idx], args.seed)
    patch_train_prob = predict_patch_scores(patch_scaler, patch_clf, train_x)
    patch_group_classifiers = fit_group_binary_experts(train_fake_x, np.concatenate(train_fake_groups), train_real_x, args.seed)
    patch_train_scores = predict_group_binary_experts(patch_group_classifiers, train_x)
    route_patch_train_score = np.zeros(len(train_x), dtype=np.float32)

    hybrid_model, hybrid_mean, hybrid_std, hybrid_temperature, hybrid_alpha = load_hybrid_checkpoint(args.hybrid_checkpoint, device)
    train_cls_rows = collect_cls_rows(iter_regular_cls_rows(args.cls_cache_root, "DF40_train", args.train_real_cls_cache))
    test_cls_rows = collect_cls_rows(iter_regular_cls_rows(args.cls_cache_root, "DF40_test_ff", args.test_real_cls_cache))
    ood_cls_rows = collect_cls_rows(iter_ood_cls_rows(args.cls_cache_root, "DF40_test_ood"))

    def hybrid_predict(x: np.ndarray) -> np.ndarray:
        return predict_hybrid_prob(
            hybrid_model,
            x,
            hybrid_mean,
            hybrid_std,
            hybrid_temperature,
            hybrid_alpha,
            args.route_batch_size,
            device,
        )

    train_route_map = build_prob_map(train_cls_rows, hybrid_predict, args.route_batch_size)
    test_route_map = build_prob_map(test_cls_rows, hybrid_predict, args.route_batch_size)
    ood_route_map = build_prob_map(ood_cls_rows, hybrid_predict, args.route_batch_size)
    train_route_prob_mat = route_prob_from_map(train_route_map, train_img_ids)
    train_route_score = route_fake_score(train_route_prob_mat)
    route_patch_train_score = dynamic_expert_score(train_route_prob_mat, patch_train_scores, args.pair_route_mode)

    real_ref = load_real_reference_cache(args.real_reference_cache)
    pair_train_methods = preload_pair_training_methods(args, real_ref)
    pair_mean_dirs, pair_classifiers, pair_region_idx, pair_region_names_by_group = train_pair_branch(
        args,
        pair_train_methods,
        real_ref,
        real_pool_means,
    )
    train_pair_scores = pair_score_matrix(
        train_region_vecs,
        real_pool_means,
        pair_mean_dirs,
        pair_classifiers,
        pair_region_idx,
        pair_region_names_by_group,
    )
    pair_train_score = dynamic_pair_score(train_route_prob_mat, train_pair_scores, args.pair_route_mode)

    fusion_choice = search_three_way_weights(
        train_y[val_idx],
        train_route_score[val_idx],
        pair_train_score[val_idx],
        route_patch_train_score[val_idx],
        args.fusion_step,
    )
    val_base_prob = (
        fusion_choice["weight_route"] * train_route_score[val_idx]
        + fusion_choice["weight_pair"] * pair_train_score[val_idx]
        + fusion_choice["weight_patch"] * route_patch_train_score[val_idx]
    ).astype(np.float32)
    val_route_patch_prob = route_patch_base_prob(train_route_score[val_idx], route_patch_train_score[val_idx], fusion_choice)
    selective_pair_choice = search_selective_pair_fusion(
        train_y[val_idx],
        train_route_prob_mat[val_idx],
        pair_train_score[val_idx],
        {
            "full_fusion": val_base_prob,
            "route_patch": val_route_patch_prob,
            "route_only": train_route_score[val_idx].astype(np.float32),
        },
    )
    train_meta_x = build_meta_features(train_route_prob_mat, train_pair_scores, patch_train_scores, patch_train_prob)
    meta_scaler, meta_clf, meta_search = fit_meta_fusion_model(
        train_meta_x,
        train_y,
        fit_idx,
        val_idx,
        args.seed,
    )
    val_meta_prob = predict_meta_fusion(
        meta_scaler,
        meta_clf,
        train_meta_x[val_idx],
    )
    route_meta_experts, route_meta_search = fit_route_aware_meta_experts(
        train_meta_x,
        train_y,
        train_groups,
        fit_idx,
        val_idx,
        args.seed,
    )
    bucket_meta_experts, bucket_expert_search = fit_bucket_meta_experts(
        train_meta_x,
        train_y,
        train_route_prob_mat,
        fit_idx,
        val_idx,
        args.seed,
    )
    val_route_meta_prob, _ = predict_route_aware_meta_fusion(
        route_meta_experts,
        train_route_prob_mat[val_idx],
        train_meta_x[val_idx],
        args.pair_route_mode,
    )
    route_meta_bucket_threshold = search_bucket_thresholds(
        train_y[val_idx],
        val_route_meta_prob,
        train_route_prob_mat[val_idx],
    )
    val_bucket_expert_prob = predict_bucket_meta_experts(
        bucket_meta_experts,
        train_route_prob_mat[val_idx],
        train_meta_x[val_idx],
    )
    bucket_meta_choice = search_bucket_meta_fusion(
        train_y[val_idx],
        train_route_prob_mat[val_idx],
        train_route_score[val_idx],
        pair_train_score[val_idx],
        route_patch_train_score[val_idx],
        val_route_meta_prob,
        args.fusion_step,
    )
    val_bucket_meta_prob = apply_bucket_meta_fusion(
        train_route_prob_mat[val_idx],
        train_route_score[val_idx],
        pair_train_score[val_idx],
        route_patch_train_score[val_idx],
        val_route_meta_prob,
        bucket_meta_choice,
    )
    uncertainty_choice = search_uncertainty_residual_fusion(
        train_y[val_idx],
        train_route_prob_mat[val_idx],
        pair_train_score[val_idx],
        route_patch_train_score[val_idx],
        {
            "route_meta": val_route_meta_prob,
            "full_fusion": val_base_prob,
            "bucket_meta": val_bucket_meta_prob,
        },
    )
    val_uncertainty_prob = apply_uncertainty_residual_fusion(
        {
            "route_meta": val_route_meta_prob,
            "full_fusion": val_base_prob,
            "bucket_meta": val_bucket_meta_prob,
        }[uncertainty_choice["base_name"]],
        train_route_prob_mat[val_idx],
        pair_train_score[val_idx],
        route_patch_train_score[val_idx],
        uncertainty_choice["uncertainty_mode"],
        uncertainty_choice["uncertainty_quantile"],
        uncertainty_choice["pair_weight"],
        uncertainty_choice["lambda_residual"],
        uncertainty_choice["update_mode"],
    )
    ambiguity_choice = search_ambiguity_fusion(
        train_y[val_idx],
        train_route_prob_mat[val_idx],
        val_route_meta_prob,
        pair_train_score[val_idx],
        route_patch_train_score[val_idx],
    )
    val_ambiguity_prob = apply_ambiguity_fusion(
        val_route_meta_prob,
        train_route_prob_mat[val_idx],
        pair_train_score[val_idx],
        route_patch_train_score[val_idx],
        ambiguity_choice["fsfr_margin_tau"],
        ambiguity_choice["efs_margin_tau"],
        ambiguity_choice["fsfr_lambda"],
        ambiguity_choice["efs_lambda"],
    )
    semantic_choice = search_semantic_rescue(
        train_y[val_idx],
        train_route_prob_mat[val_idx],
        val_route_meta_prob,
        pair_train_score[val_idx],
        route_patch_train_score[val_idx],
    )
    val_semantic_prob = apply_semantic_rescue(
        train_route_prob_mat[val_idx],
        val_route_meta_prob,
        pair_train_score[val_idx],
        route_patch_train_score[val_idx],
        semantic_choice,
    )
    semantic_max_choice = search_semantic_max_rescue(
        train_y[val_idx],
        train_route_prob_mat[val_idx],
        val_route_meta_prob,
        pair_train_score[val_idx],
        route_patch_train_score[val_idx],
    )
    val_semantic_max_prob = apply_semantic_max_rescue(
        train_route_prob_mat[val_idx],
        val_route_meta_prob,
        pair_train_score[val_idx],
        route_patch_train_score[val_idx],
        semantic_max_choice,
    )
    val_selective_prob = apply_selective_pair_fusion(
        val_base_prob,
        pair_train_score[val_idx],
        train_route_prob_mat[val_idx],
        selective_pair_choice["margin_tau"],
        selective_pair_choice["pair_lambda"],
        selective_pair_choice["gate_mode"],
    )
    threshold_choice = {
        "route_only": search_threshold(train_y[val_idx], train_route_score[val_idx]),
        "pair_only": search_threshold(train_y[val_idx], pair_train_score[val_idx]),
        "patch_only": search_threshold(train_y[val_idx], patch_train_prob[val_idx]),
        "route_patch_only": search_threshold(train_y[val_idx], route_patch_train_score[val_idx]),
        "full_fusion": search_threshold(train_y[val_idx], val_base_prob),
        "meta_fusion": search_threshold(train_y[val_idx], val_meta_prob),
        "route_meta_fusion": search_threshold(train_y[val_idx], val_route_meta_prob),
        "route_meta_bucket_threshold": {
            "threshold": None,
            "balanced_accuracy": route_meta_bucket_threshold["current"]["balanced_accuracy"],
            "accuracy": route_meta_bucket_threshold["current"]["accuracy"],
            "fake_accuracy": route_meta_bucket_threshold["current"]["fake_accuracy"],
            "real_accuracy": route_meta_bucket_threshold["current"]["real_accuracy"],
            "auc": route_meta_bucket_threshold["current"]["auc"],
        },
        "bucket_expert_fusion": search_threshold(train_y[val_idx], val_bucket_expert_prob),
        "bucket_meta_fusion": search_threshold(train_y[val_idx], val_bucket_meta_prob),
        "uncertainty_residual_fusion": search_threshold(train_y[val_idx], val_uncertainty_prob),
        "ambiguity_fusion": search_threshold(train_y[val_idx], val_ambiguity_prob),
        "semantic_fusion": search_threshold(train_y[val_idx], val_semantic_prob),
        "semantic_max_fusion": search_threshold(train_y[val_idx], val_semantic_max_prob),
        "selective_pair_fusion": search_threshold(train_y[val_idx], val_selective_prob),
    }
    auto_generic_candidate_keys = (
        NO_TUNING_GENERIC_MODEL_KEYS
        if args.auto_generic_pool == "no_tuning"
        else GENERIC_MODEL_KEYS
    )
    generic_blend_base_models = (
        NO_TUNING_GENERIC_BLEND_BASE_MODELS
        if args.auto_generic_pool == "no_tuning"
        else GENERIC_BLEND_BASE_MODELS
    )
    generic_blend_choice = search_generic_blend_fusion(
        train_y[val_idx],
        {
            "full_fusion": val_base_prob,
            "meta_fusion": val_meta_prob,
            "route_meta_fusion": val_route_meta_prob,
            "bucket_expert_fusion": val_bucket_expert_prob,
            "bucket_meta_fusion": val_bucket_meta_prob,
            "uncertainty_residual_fusion": val_uncertainty_prob,
            "semantic_fusion": val_semantic_prob,
            "semantic_max_fusion": val_semantic_max_prob,
        },
        threshold_choice,
        base_models=generic_blend_base_models,
    )
    threshold_choice["generic_blend_fusion"] = {
        "threshold": generic_blend_choice["threshold"],
        "balanced_accuracy": generic_blend_choice["balanced_accuracy"],
        "accuracy": generic_blend_choice["accuracy"],
        "fake_accuracy": generic_blend_choice["fake_accuracy"],
        "real_accuracy": generic_blend_choice["real_accuracy"],
        "auc": generic_blend_choice["auc"],
    }
    auto_generic_selection = select_auto_generic_model(
        {
            key: value["balanced_accuracy"] if isinstance(value, dict) else float(value)
            for key, value in threshold_choice.items()
        },
        threshold_choice,
        list(threshold_choice.keys()),
        candidate_keys=auto_generic_candidate_keys,
    )

    test_ff = evaluate_regular_split(
        discover_flat_method_caches(args.patch_cache_root, "DF40_test_ff", ["FS", "FR", "FE", "EFS"]),
        test_real_cache,
        args=args,
        route_map=test_route_map,
        pair_mean_dirs=pair_mean_dirs,
        pair_classifiers=pair_classifiers,
        pair_region_idx=pair_region_idx,
        pair_region_names_by_group=pair_region_names_by_group,
        patch_scaler=patch_scaler,
        patch_clf=patch_clf,
        patch_group_classifiers=patch_group_classifiers,
        real_pool_means=real_pool_means,
        fusion_choice=fusion_choice,
        threshold_choice={k: v["threshold"] for k, v in threshold_choice.items()},
        meta_scaler=meta_scaler,
        meta_clf=meta_clf,
        route_meta_experts=route_meta_experts,
        route_meta_bucket_threshold=route_meta_bucket_threshold,
        bucket_meta_experts=bucket_meta_experts,
        bucket_meta_choice=bucket_meta_choice,
        ambiguity_choice=ambiguity_choice,
        uncertainty_choice=uncertainty_choice,
        semantic_choice=semantic_choice,
        semantic_max_choice=semantic_max_choice,
        generic_blend_choice=generic_blend_choice,
        selective_pair_choice=selective_pair_choice,
    )
    ood = evaluate_ood_split(
        discover_ood_method_entries(args.patch_cache_root),
        args=args,
        route_map=ood_route_map,
        pair_mean_dirs=pair_mean_dirs,
        pair_classifiers=pair_classifiers,
        pair_region_idx=pair_region_idx,
        pair_region_names_by_group=pair_region_names_by_group,
        patch_scaler=patch_scaler,
        patch_clf=patch_clf,
        patch_group_classifiers=patch_group_classifiers,
        real_pool_means=real_pool_means,
        fusion_choice=fusion_choice,
        threshold_choice={k: v["threshold"] for k, v in threshold_choice.items()},
        meta_scaler=meta_scaler,
        meta_clf=meta_clf,
        route_meta_experts=route_meta_experts,
        route_meta_bucket_threshold=route_meta_bucket_threshold,
        bucket_meta_experts=bucket_meta_experts,
        bucket_meta_choice=bucket_meta_choice,
        ambiguity_choice=ambiguity_choice,
        uncertainty_choice=uncertainty_choice,
        semantic_choice=semantic_choice,
        semantic_max_choice=semantic_max_choice,
        generic_blend_choice=generic_blend_choice,
        selective_pair_choice=selective_pair_choice,
    )

    output = {
        "config": {
            "patch_cache_root": str(args.patch_cache_root),
            "cls_cache_root": str(args.cls_cache_root),
            "real_reference_cache": str(args.real_reference_cache),
            "hybrid_checkpoint": str(args.hybrid_checkpoint),
            "pair_feature_mode": args.pair_feature_mode,
            "pair_region_mode": args.pair_region_mode,
            "pair_pca_dim": args.pair_pca_dim,
            "pair_ridge_alpha": args.pair_ridge_alpha,
            "pair_route_mode": args.pair_route_mode,
            "patch_train_groups": list(args.patch_train_groups),
            "fusion_step": args.fusion_step,
            "val_split_mode": args.val_split_mode,
            "auto_generic_pool": args.auto_generic_pool,
            "no_tuning_mainline": bool(args.no_tuning_mainline),
            "auto_generic_candidate_keys": list(auto_generic_candidate_keys),
            "seed": args.seed,
        },
        "pair_region_names_by_group": pair_region_names_by_group,
        "fusion_search": fusion_choice,
        "threshold_search": threshold_choice,
        "selective_pair_search": selective_pair_choice,
        "meta_fusion_search": meta_search,
        "route_meta_fusion_search": route_meta_search,
        "route_meta_bucket_threshold_search": route_meta_bucket_threshold,
        "bucket_expert_fusion_search": bucket_expert_search,
        "bucket_meta_fusion_search": bucket_meta_choice,
        "uncertainty_residual_fusion_search": uncertainty_choice,
        "ambiguity_fusion_search": ambiguity_choice,
        "semantic_fusion_search": semantic_choice,
        "semantic_max_fusion_search": semantic_max_choice,
        "generic_blend_fusion_search": generic_blend_choice,
        "train_summary": {
            "num_train_samples": int(len(train_y)),
            "num_fake": int(np.sum(train_y == 1)),
            "num_real": int(np.sum(train_y == 0)),
        },
        "validation": {
            "route_only": threshold_choice["route_only"]["balanced_accuracy"],
            "pair_only": threshold_choice["pair_only"]["balanced_accuracy"],
            "patch_only": threshold_choice["patch_only"]["balanced_accuracy"],
            "route_patch_only": threshold_choice["route_patch_only"]["balanced_accuracy"],
            "full_fusion": threshold_choice["full_fusion"]["balanced_accuracy"],
            "meta_fusion": threshold_choice["meta_fusion"]["balanced_accuracy"],
            "route_meta_fusion": threshold_choice["route_meta_fusion"]["balanced_accuracy"],
            "route_meta_bucket_threshold": threshold_choice["route_meta_bucket_threshold"]["balanced_accuracy"],
            "bucket_expert_fusion": threshold_choice["bucket_expert_fusion"]["balanced_accuracy"],
            "bucket_meta_fusion": threshold_choice["bucket_meta_fusion"]["balanced_accuracy"],
            "uncertainty_residual_fusion": threshold_choice["uncertainty_residual_fusion"]["balanced_accuracy"],
            "ambiguity_fusion": threshold_choice["ambiguity_fusion"]["balanced_accuracy"],
            "semantic_fusion": threshold_choice["semantic_fusion"]["balanced_accuracy"],
            "semantic_max_fusion": threshold_choice["semantic_max_fusion"]["balanced_accuracy"],
            "generic_blend_fusion": threshold_choice["generic_blend_fusion"]["balanced_accuracy"],
            "selective_pair_fusion": threshold_choice["selective_pair_fusion"]["balanced_accuracy"],
        },
        "auto_generic_candidate_keys": list(auto_generic_candidate_keys),
        "auto_generic_selection": auto_generic_selection,
        "test_ff": test_ff,
        "ood": ood,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))
    export_csv(args.output_csv, output)
    print(
        json.dumps(
            {
                "test_ff_full_fusion": output["test_ff"]["full_fusion"]["summary"],
                "test_ff_meta_fusion": output["test_ff"]["meta_fusion"]["summary"],
                "test_ff_route_meta_fusion": output["test_ff"]["route_meta_fusion"]["summary"],
                "test_ff_route_meta_bucket_threshold": output["test_ff"]["route_meta_bucket_threshold"]["summary"],
                "test_ff_bucket_expert_fusion": output["test_ff"]["bucket_expert_fusion"]["summary"],
                "test_ff_bucket_meta_fusion": output["test_ff"]["bucket_meta_fusion"]["summary"],
                "test_ff_uncertainty_residual_fusion": output["test_ff"]["uncertainty_residual_fusion"]["summary"],
                "test_ff_ambiguity_fusion": output["test_ff"]["ambiguity_fusion"]["summary"],
                "test_ff_semantic_fusion": output["test_ff"]["semantic_fusion"]["summary"],
                "test_ff_semantic_max_fusion": output["test_ff"]["semantic_max_fusion"]["summary"],
                "test_ff_generic_blend_fusion": output["test_ff"]["generic_blend_fusion"]["summary"],
                "test_ff_selective_pair_fusion": output["test_ff"]["selective_pair_fusion"]["summary"],
                "ood_full_fusion": output["ood"]["full_fusion"]["summary"],
                "ood_meta_fusion": output["ood"]["meta_fusion"]["summary"],
                "ood_route_meta_fusion": output["ood"]["route_meta_fusion"]["summary"],
                "ood_route_meta_bucket_threshold": output["ood"]["route_meta_bucket_threshold"]["summary"],
                "ood_bucket_expert_fusion": output["ood"]["bucket_expert_fusion"]["summary"],
                "ood_bucket_meta_fusion": output["ood"]["bucket_meta_fusion"]["summary"],
                "ood_uncertainty_residual_fusion": output["ood"]["uncertainty_residual_fusion"]["summary"],
                "ood_ambiguity_fusion": output["ood"]["ambiguity_fusion"]["summary"],
                "ood_semantic_fusion": output["ood"]["semantic_fusion"]["summary"],
                "ood_semantic_max_fusion": output["ood"]["semantic_max_fusion"]["summary"],
                "ood_generic_blend_fusion": output["ood"]["generic_blend_fusion"]["summary"],
                "ood_selective_pair_fusion": output["ood"]["selective_pair_fusion"]["summary"],
                "fusion_search": output["fusion_search"],
                "threshold_search": output["threshold_search"],
                "selective_pair_search": output["selective_pair_search"],
                "meta_fusion_search": output["meta_fusion_search"],
                "route_meta_fusion_search": output["route_meta_fusion_search"],
                "route_meta_bucket_threshold_search": output["route_meta_bucket_threshold_search"],
                "bucket_expert_fusion_search": output["bucket_expert_fusion_search"],
                "bucket_meta_fusion_search": output["bucket_meta_fusion_search"],
                "uncertainty_residual_fusion_search": output["uncertainty_residual_fusion_search"],
                "ambiguity_fusion_search": output["ambiguity_fusion_search"],
                "semantic_fusion_search": output["semantic_fusion_search"],
                "semantic_max_fusion_search": output["semantic_max_fusion_search"],
                "generic_blend_fusion_search": output["generic_blend_fusion_search"],
                "auto_generic_selection": output["auto_generic_selection"],
            },
            indent=2,
        )
    )
    print(f"Saved to {args.output_json}")
    print(f"Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
