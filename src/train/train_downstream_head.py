#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from prepare.cache import (  # noqa: E402
    CompactPatchCache,
    EmbeddingCache,
    build_compact_cache,
    subset_compact_cache,
)
from train.train_upstream import (  # noqa: E402
    HybridManifoldModel,
)
from utils.device import resolve_torch_device  # noqa: E402

REAL_LABEL = "REAL"
EFS_LABEL = "EFS"
BASE_HYBRID_LABELS = ["EFS", "FS", "FR", "FE", "REAL"]
NO_FR_HYBRID_LABELS = ["EFS", "FS", "FE", "REAL"]
BASE_FAKE_GROUPS = ["EFS", "FS", "FR", "FE"]
BASE_PAIR_GROUPS = ["FS", "FR", "FE", "EFS"]
NO_FR_FAKE_GROUPS = ["EFS", "FS", "FE"]
NO_FR_PAIR_GROUPS = ["FS", "FE", "EFS"]
FAKE_GROUPS = list(BASE_FAKE_GROUPS)
PAIR_GROUPS = list(BASE_PAIR_GROUPS)
EVAL_GROUPS = list(BASE_FAKE_GROUPS)
CANONICAL_REGIONS = {
    "FS": ["eyebrow", "skin"],
    "FR": ["skin", "nose"],
    "FE": ["eye", "mouth"],
    "EFS": ["eye", "mouth", "eyebrow"],
}
HYBRID_LABELS = list(BASE_HYBRID_LABELS)
REAL_IDX = HYBRID_LABELS.index(REAL_LABEL)
FAKE_ROUTE_IDXS = [HYBRID_LABELS.index(group) for group in FAKE_GROUPS]
EPS = 1e-8
VOLUME_PREFIX = "/Volumes/未命名"


def configure_group_scheme(no_fr: bool, hybrid_labels: list[str] | None = None) -> None:
    global FAKE_GROUPS, PAIR_GROUPS, EVAL_GROUPS, HYBRID_LABELS, REAL_IDX, FAKE_ROUTE_IDXS
    FAKE_GROUPS = list(NO_FR_FAKE_GROUPS if no_fr else BASE_FAKE_GROUPS)
    PAIR_GROUPS = list(NO_FR_PAIR_GROUPS if no_fr else BASE_PAIR_GROUPS)
    EVAL_GROUPS = list(FAKE_GROUPS)
    HYBRID_LABELS = list(hybrid_labels or (NO_FR_HYBRID_LABELS if no_fr else BASE_HYBRID_LABELS))
    REAL_IDX = HYBRID_LABELS.index(REAL_LABEL)
    FAKE_ROUTE_IDXS = [HYBRID_LABELS.index(group) for group in FAKE_GROUPS]


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


def normalize_key(path_like: str | Path) -> str:
    return str(path_like).replace(VOLUME_PREFIX, "")


def extract_cos_norm_features(
    pseudo_deltas: np.ndarray,
    mean_delta_dirs: np.ndarray,
) -> np.ndarray:
    n_images, n_regions, _ = pseudo_deltas.shape
    feats = np.zeros((n_images, n_regions * 2), dtype=np.float32)
    for r in range(n_regions):
        delta = pseudo_deltas[:, r, :]
        norms = np.linalg.norm(delta, axis=1).astype(np.float32)
        feats[:, r * 2] = norms
        ref = mean_delta_dirs[r]
        ref_norm = float(np.linalg.norm(ref))
        if ref_norm > EPS:
            ref_unit = ref / ref_norm
            cos = np.sum(delta * ref_unit[None, :], axis=1) / np.maximum(norms, EPS)
            feats[:, r * 2 + 1] = np.clip(cos, -1.0, 1.0).astype(np.float32)
    return feats


def compute_mean_delta_directions(
    fake_region_vecs: np.ndarray,
    paired_real_region_vecs: np.ndarray,
    n_regions: int,
) -> np.ndarray:
    deltas = fake_region_vecs - paired_real_region_vecs
    mean_dirs = deltas.mean(axis=0).astype(np.float32)
    for r in range(n_regions):
        norm = float(np.linalg.norm(mean_dirs[r]))
        if norm > EPS:
            mean_dirs[r] /= norm
    return mean_dirs


def compact_cache_file(root: Path, source_path: Path) -> Path:
    try:
        rel = source_path.relative_to(root)
        parts = rel.parts
    except ValueError:
        parts = source_path.resolve().parts[1:]
    safe = "__".join(parts)
    return root / f"{safe}.npz"


def load_compact_cache(source_path: Path, max_rows: int, seed: int, compact_cache_dir: Path) -> CompactPatchCache:
    compact_path = compact_cache_file(compact_cache_dir, source_path)
    if compact_path.exists():
        compact = CompactPatchCache.from_npz(compact_path)
    else:
        compact = build_compact_cache(source_path, compact_path)
    return subset_compact_cache(compact, max_rows, seed)


def compute_real_pool_region_means(real_cache: CompactPatchCache) -> np.ndarray:
    num_regions = len(real_cache.region_names)
    dim = real_cache.region_vecs.shape[-1]
    sums = np.zeros((num_regions, dim), dtype=np.float64)
    counts = np.zeros(num_regions, dtype=np.int64)
    for row_idx in range(len(real_cache.img_id)):
        present = real_cache.region_present[row_idx]
        sums[present] += real_cache.region_vecs[row_idx, present].astype(np.float32)
        counts[present] += 1
    counts = np.maximum(counts, 1)
    return (sums / counts[:, None]).astype(np.float32)


def image_region_delta_features(cache: CompactPatchCache, real_pool_means: np.ndarray) -> np.ndarray:
    centered = cache.region_vecs.astype(np.float32) - real_pool_means[None, :, :]
    centered[~cache.region_present] = 0.0
    x = centered.reshape(centered.shape[0], -1).astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def fit_patch_classifier(train_x: np.ndarray, train_y: np.ndarray, seed: int) -> tuple[StandardScaler, SGDClassifier]:
    scaler = StandardScaler()
    x = scaler.fit_transform(train_x)
    x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
    x = np.clip(x, -10.0, 10.0)
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=2000,
        tol=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(x, train_y)
    return scaler, clf


def predict_patch_scores(scaler: StandardScaler, clf: SGDClassifier, x: np.ndarray) -> np.ndarray:
    x = scaler.transform(x)
    x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
    x = np.clip(x, -10.0, 10.0)
    return clf.predict_proba(x)[:, 1].astype(np.float32)


def iter_regular_cls_rows(cache_root: Path, split: str, real_cache_path: Path | None = None):
    split_root = cache_root / split
    for group in ["EFS", "FS", "FR", "FE"]:
        group_dir = split_root / group
        for cache_path in sorted(group_dir.glob("cls_*.npz")):
            cache = EmbeddingCache.from_npz(cache_path)
            fake_mask = cache.label.astype(np.int64) == 1
            method = cache_path.stem.replace("cls_", "")
            for img_id, vec in zip(cache.img_id[fake_mask].tolist(), cache.cls[fake_mask], strict=True):
                yield str(img_id), vec.astype(np.float32, copy=False), group, method
    if real_cache_path is not None:
        cache = EmbeddingCache.from_npz(real_cache_path)
        for img_id, vec, method in zip(cache.img_id.tolist(), cache.cls, cache.method.tolist(), strict=True):
            yield str(img_id), vec.astype(np.float32, copy=False), "REAL", str(method)


def iter_ood_cls_rows(cache_root: Path, split: str):
    split_root = cache_root / split
    for group in ["EFS", "FS", "FR", "FE"]:
        group_dir = split_root / group
        if not group_dir.exists():
            continue
        for method_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            fake_cache = EmbeddingCache.from_npz(method_dir / "cls_fake.npz")
            real_cache = EmbeddingCache.from_npz(method_dir / "cls_real.npz")
            method = method_dir.name
            for img_id, vec in zip(fake_cache.img_id.tolist(), fake_cache.cls, strict=True):
                yield str(img_id), vec.astype(np.float32, copy=False), group, method
            for img_id, vec in zip(real_cache.img_id.tolist(), real_cache.cls, strict=True):
                yield str(img_id), vec.astype(np.float32, copy=False), "REAL", f"real::{method}"


def discover_flat_method_caches(cache_root: Path, split: str, groups: list[str]) -> list[Path]:
    paths: list[Path] = []
    for group in groups:
        group_dir = cache_root / split / group
        if not group_dir.exists():
            continue
        paths.extend(sorted(group_dir.glob("patch_*.npz")))
    return paths


def discover_ood_method_entries(
    cache_root: Path,
    groups: list[str] | None = None,
) -> list[tuple[str, str, Path, Path]]:
    entries = []
    split_root = cache_root / "DF40_test_ood"
    for group in groups or EVAL_GROUPS:
        group_dir = split_root / group
        if not group_dir.exists():
            continue
        for method_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            fake_path = method_dir / "patch_fake.npz"
            real_path = method_dir / "patch_real.npz"
            if fake_path.exists() and real_path.exists():
                entries.append((group, method_dir.name, fake_path, real_path))
    return entries


def split_train_indices(
    methods: np.ndarray,
    val_ratio: float,
    seed: int,
    n_fake_train: int,
    split_mode: str = "within_method",
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit_idx: list[int] = []
    val_idx: list[int] = []
    method_keys = methods[:n_fake_train]
    unique_methods = sorted(np.unique(method_keys).tolist())
    if split_mode == "holdout_method" and len(unique_methods) > 1:
        rng.shuffle(unique_methods)
        n_val_methods = max(1, int(round(len(unique_methods) * val_ratio)))
        n_val_methods = min(n_val_methods, len(unique_methods) - 1)
        val_method_set = set(unique_methods[:n_val_methods])
        for method in unique_methods:
            idx = np.flatnonzero(method_keys == method)
            if method in val_method_set:
                val_idx.extend(idx.tolist())
            else:
                fit_idx.extend(idx.tolist())
    else:
        for method in unique_methods:
            idx = np.flatnonzero(method_keys == method)
            rng.shuffle(idx)
            n_val = max(1, int(round(len(idx) * val_ratio)))
            n_val = min(n_val, len(idx) - 1) if len(idx) > 1 else 0
            if n_val > 0:
                val_idx.extend(idx[:n_val].tolist())
                fit_idx.extend(idx[n_val:].tolist())
            else:
                fit_idx.extend(idx.tolist())
    real_idx = np.arange(n_fake_train, len(methods))
    rng.shuffle(real_idx)
    n_val_real = max(1, int(round(len(real_idx) * val_ratio)))
    n_val_real = min(n_val_real, len(real_idx) - 1) if len(real_idx) > 1 else 0
    if n_val_real > 0:
        val_idx.extend(real_idx[:n_val_real].tolist())
        fit_idx.extend(real_idx[n_val_real:].tolist())
    else:
        fit_idx.extend(real_idx.tolist())
    return np.asarray(fit_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


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


def checkpoint_group_settings(checkpoint: dict) -> tuple[list[str], list[str]]:
    settings = checkpoint.get("settings")
    labels = None
    pair_groups = None
    if isinstance(settings, dict):
        raw_labels = settings.get("labels")
        raw_pair_groups = settings.get("pair_groups")
        if raw_labels is not None:
            labels = [str(label) for label in raw_labels]
        if raw_pair_groups is not None:
            pair_groups = [str(group) for group in raw_pair_groups]
    if labels is None:
        raw_classes = checkpoint.get("classes")
        if raw_classes is not None:
            labels = [str(label) for label in np.asarray(raw_classes, dtype=object).tolist()]
        else:
            labels = list(BASE_HYBRID_LABELS)
    if pair_groups is None:
        pair_groups = [group for group in labels if group not in {EFS_LABEL, REAL_LABEL}]
    return labels, pair_groups


def load_hybrid_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[HybridManifoldModel, np.ndarray, np.ndarray, float, float, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dim = int(checkpoint["dim"])
    real_rank = int(checkpoint["real_rank"])
    efs_rank = int(checkpoint["efs_rank"])
    labels, pair_groups = checkpoint_group_settings(checkpoint)
    model = HybridManifoldModel(
        dim=dim,
        real_rank=real_rank,
        efs_rank=efs_rank,
        real_center_init=np.zeros(dim, dtype=np.float32),
        efs_center_init=np.zeros(dim, dtype=np.float32),
        real_basis_init=np.zeros((dim, real_rank), dtype=np.float32),
        efs_basis_init=np.zeros((dim, efs_rank), dtype=np.float32),
        fake_offset_init=np.zeros((len(pair_groups), dim), dtype=np.float32),
        delta_proto_init=np.zeros((len(pair_groups), dim), dtype=np.float32),
        labels=labels,
        pair_groups=pair_groups,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return (
        model,
        checkpoint["mean"].astype(np.float32),
        checkpoint["std"].astype(np.float32),
        float(checkpoint["temperature"]),
        float(checkpoint["alpha"]),
        labels,
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
    real_lookup = {str(key): idx for idx, key in enumerate(real_ref["normalized_img_id"].tolist())}
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
    if mode == "no_background_keep_hair":
        keep = [name for name in region_names if name != "background"]
        return {group: list(keep) for group in PAIR_GROUPS}
    raise ValueError(f"Unknown pair-region mode: {mode}")


def train_pair_branch(
    args: argparse.Namespace,
    all_methods: dict[tuple[str, str], PairMethodData],
    real_ref: dict,
    real_pool_means: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, tuple[StandardScaler, LogisticRegression]],
    dict[str, int],
    dict[str, list[str]],
]:
    region_idx = {name: idx for idx, name in enumerate(real_ref["region_names"])}
    pair_region_names_by_group = build_pair_region_names_by_group(
        [str(name) for name in real_ref["region_names"]],
        args.pair_region_mode,
    )
    real_delta = sanitize_feature_matrix(real_ref["region_vecs"] - real_pool_means[None, :, :])
    shared_pair_groups = [group for group in ["FS", "FR", "FE"] if group in FAKE_GROUPS]
    shared_paired_fake, shared_paired_real = collect_paired_data(all_methods, shared_pair_groups)
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


def softened_fake_route_weights(
    route_prob: np.ndarray,
    *,
    temperature: float = 1.0,
    floor: float = 0.0,
) -> np.ndarray:
    fake_prob = np.clip(route_prob[:, FAKE_ROUTE_IDXS].astype(np.float32), 1e-8, 1.0)
    temp = max(float(temperature), 1e-4)
    log_prob = np.log(fake_prob) / temp
    log_prob = log_prob - np.max(log_prob, axis=1, keepdims=True)
    weights = np.exp(log_prob).astype(np.float32)
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
    floor_value = float(np.clip(floor, 0.0, 0.24))
    if floor_value > 0.0:
        uniform = np.full_like(weights, 1.0 / weights.shape[1], dtype=np.float32)
        weights = ((1.0 - floor_value) * weights + floor_value * uniform).astype(np.float32)
        weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
    return weights.astype(np.float32)


def dynamic_expert_score(
    route_prob: np.ndarray,
    expert_scores: np.ndarray,
    mode: str,
    *,
    route_gating_temperature: float = 1.0,
    route_gating_floor: float = 0.0,
) -> np.ndarray:
    weights = route_prob[:, FAKE_ROUTE_IDXS]
    smooth_weights = softened_fake_route_weights(
        route_prob,
        temperature=route_gating_temperature,
        floor=route_gating_floor,
    )
    if mode == "weighted":
        return np.sum(weights * expert_scores, axis=1).astype(np.float32)
    if mode == "weighted_norm":
        return np.sum(smooth_weights * expert_scores, axis=1).astype(np.float32)
    top1 = np.argmax(smooth_weights, axis=1)
    if mode == "top1":
        return expert_scores[np.arange(len(expert_scores)), top1].astype(np.float32)
    if mode == "top2":
        top2 = np.argsort(-smooth_weights, axis=1)[:, :2]
        mix = np.take_along_axis(smooth_weights, top2, axis=1)
        mix = mix / np.clip(mix.sum(axis=1, keepdims=True), 1e-8, None)
        return np.sum(np.take_along_axis(expert_scores, top2, axis=1) * mix, axis=1).astype(np.float32)
    raise ValueError(f"Unknown expert route mode: {mode}")


def dynamic_pair_score(
    route_prob: np.ndarray,
    pair_scores: np.ndarray,
    mode: str,
    *,
    route_gating_temperature: float = 1.0,
    route_gating_floor: float = 0.0,
) -> np.ndarray:
    return dynamic_expert_score(
        route_prob,
        pair_scores,
        mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )


def build_meta_features(
    route_prob: np.ndarray,
    pair_scores: np.ndarray,
    patch_scores: np.ndarray,
    patch_global_prob: np.ndarray,
    *,
    route_gating_temperature: float = 1.0,
    route_gating_floor: float = 0.0,
) -> np.ndarray:
    fake_route = route_prob[:, FAKE_ROUTE_IDXS].astype(np.float32)
    fake_route_norm = softened_fake_route_weights(
        route_prob,
        temperature=route_gating_temperature,
        floor=route_gating_floor,
    )
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
    route_gating_temperature: float,
    route_gating_floor: float,
) -> BranchFeatures:
    x = image_region_delta_features(cache, real_pool_means)
    patch_prob = predict_patch_scores(patch_scaler, patch_clf, x)
    route_prob = route_prob_from_map(route_map, cache.img_id.astype(object))
    route_score = route_fake_score(route_prob)
    patch_scores = predict_group_binary_experts(patch_group_classifiers, x)
    route_patch_score = dynamic_expert_score(
        route_prob,
        patch_scores,
        pair_route_mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )
    pair_scores = pair_score_matrix(
        cache.region_vecs.astype(np.float32),
        real_pool_means,
        pair_mean_dirs,
        pair_classifiers,
        pair_region_idx,
        pair_region_names_by_group,
    )
    pair_score = dynamic_pair_score(
        route_prob,
        pair_scores,
        pair_route_mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )
    meta_x = build_meta_features(
        route_prob,
        pair_scores,
        patch_scores,
        patch_prob,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )
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


def predict_meta_fusion(scaler: StandardScaler, clf: LogisticRegression, x: np.ndarray) -> np.ndarray:
    x_std = scaler.transform(x.astype(np.float64))
    x_std = np.clip(np.nan_to_num(x_std, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
    return clf.predict_proba(x_std)[:, 1].astype(np.float32)


def predict_route_aware_meta_fusion(
    experts: dict[str, tuple[StandardScaler, LogisticRegression]],
    route_prob: np.ndarray,
    x: np.ndarray,
    mode: str,
    *,
    route_gating_temperature: float = 1.0,
    route_gating_floor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    expert_prob = np.zeros((len(x), len(FAKE_GROUPS)), dtype=np.float32)
    for idx, group in enumerate(FAKE_GROUPS):
        scaler, clf = experts[group]
        expert_prob[:, idx] = predict_meta_fusion(scaler, clf, x)
    mixed = dynamic_expert_score(
        route_prob,
        expert_prob,
        mode,
        route_gating_temperature=route_gating_temperature,
        route_gating_floor=route_gating_floor,
    )
    return mixed.astype(np.float32), expert_prob.astype(np.float32)


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


def summarize_split(rows: list[dict]) -> dict:
    if not rows:
        return {
            "num_methods": 0,
            "mean_accuracy": 0.0,
            "mean_balanced_accuracy": 0.0,
            "mean_fake_accuracy": 0.0,
            "mean_real_accuracy": 0.0,
            "mean_auc": 0.0,
        }
    return {
        "num_methods": int(len(rows)),
        "mean_accuracy": float(np.mean([row["metrics"]["accuracy"] for row in rows])),
        "mean_balanced_accuracy": float(np.mean([row["metrics"]["balanced_accuracy"] for row in rows])),
        "mean_fake_accuracy": float(np.mean([row["metrics"]["fake_accuracy"] for row in rows])),
        "mean_real_accuracy": float(np.mean([row["metrics"]["real_accuracy"] for row in rows])),
        "mean_auc": float(np.mean([row["metrics"]["auc"] for row in rows])),
    }


def summarize_method_rows(rows: list[dict]) -> dict:
    return {"summary": summarize_split(rows), "methods": sorted(rows, key=lambda row: (row["group"], row["method"]))}


def summarize_method_rows_for_groups(rows: list[dict], groups: list[str]) -> dict:
    group_set = set(groups)
    return summarize_method_rows([row for row in rows if row["group"] in group_set])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-head downstream pipeline: patch + pair + route_meta_fusion."
    )
    parser.add_argument("--patch-cache-root", type=Path, default=ROOT / "cache" / "patch")
    parser.add_argument("--cls-cache-root", type=Path, default=ROOT / "cache" / "cls")
    parser.add_argument("--train-real-patch-cache", type=Path, default=ROOT / "cache" / "patch" / "DF40_train" / "REAL" / "patch_real_pool.npz")
    parser.add_argument("--test-real-patch-cache", type=Path, default=ROOT / "cache" / "patch" / "DF40_test_ff" / "REAL" / "patch_real_pool.npz")
    parser.add_argument("--train-real-cls-cache", type=Path, default=ROOT / "cache" / "cls" / "DF40_train" / "REAL" / "cls_real_dedup_from_existing.npz")
    parser.add_argument("--test-real-cls-cache", type=Path, default=ROOT / "cache" / "cls" / "DF40_test_ff" / "REAL" / "cls_real_dedup_from_existing.npz")
    parser.add_argument("--real-reference-cache", type=Path, default=ROOT / "cache" / "patch" / "DF40_train" / "REAL" / "patch_real_pool_region_reference.npz")
    parser.add_argument("--hybrid-checkpoint", type=Path, default=ROOT / "checkpoints" / "upstream" / "checkpoint_best_hybrid_manifold.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-real-max", type=int, default=0)
    parser.add_argument("--eval-real-max", type=int, default=2000)
    parser.add_argument("--max-train-fake-per-method", type=int, default=0)
    parser.add_argument("--max-test-fake-per-method", type=int, default=0)
    parser.add_argument("--max-ood-fake-per-method", type=int, default=0)
    parser.add_argument("--patch-train-groups", nargs="*", default=None)
    parser.add_argument("--route-batch-size", type=int, default=2048)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-split-mode", choices=["within_method", "holdout_method"], default="holdout_method")
    parser.add_argument("--pair-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-fr",
        action="store_true",
        help="Train the downstream pipeline without FR, using EFS/FS/FE fake targets.",
    )
    parser.add_argument(
        "--pair-region-mode",
        choices=["canonical", "all_regions", "no_background_keep_hair"],
        default="no_background_keep_hair",
    )
    parser.add_argument("--pair-feature-mode", choices=["cosnorm"], default="cosnorm")
    parser.add_argument("--pair-pca-dim", type=int, default=32)
    parser.add_argument("--pair-route-mode", choices=["weighted", "weighted_norm", "top1", "top2"], default="weighted")
    parser.add_argument("--route-gating-temperature", type=float, default=1.0)
    parser.add_argument("--route-gating-floor", type=float, default=0.0)
    parser.add_argument("--compact-cache-dir", type=Path, default=ROOT / "cache" / "compact")
    parser.add_argument("--output-patch-branch", type=Path)
    parser.add_argument("--output-pair-branch", type=Path)
    parser.add_argument("--output-route-meta-head", type=Path)
    parser.add_argument("--output-head-meta", type=Path)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "route_meta_pipeline.json")
    args = parser.parse_args()
    configure_group_scheme(args.no_fr)
    if args.patch_train_groups is None:
        args.patch_train_groups = list(FAKE_GROUPS)
    if args.output_patch_branch is None:
        args.output_patch_branch = ROOT / "checkpoints" / "downstream" / ("patch_branch_no_fr.joblib" if args.no_fr else "patch_branch.joblib")
    if args.output_pair_branch is None:
        args.output_pair_branch = ROOT / "checkpoints" / "downstream" / ("pair_branch_no_fr.joblib" if args.no_fr else "pair_branch.joblib")
    if args.output_route_meta_head is None:
        args.output_route_meta_head = ROOT / "checkpoints" / "heads" / ("route_meta_head_no_fr.joblib" if args.no_fr else "route_meta_head.joblib")
    if args.output_head_meta is None:
        args.output_head_meta = ROOT / "checkpoints" / "heads" / ("route_meta_head_no_fr_meta.json" if args.no_fr else "route_meta_head_meta.json")
    return args


def evaluate_regular_route_meta(
    fake_paths: list[Path],
    real_cache,
    *,
    args: argparse.Namespace,
    route_map: dict[str, np.ndarray],
    real_pool_means: np.ndarray,
    pair_mean_dirs: dict[str, np.ndarray],
    pair_classifiers,
    pair_region_idx: dict[str, int],
    pair_region_names_by_group: dict[str, list[str]],
    patch_scaler,
    patch_clf,
    patch_group_classifiers,
    route_meta_experts,
    route_meta_threshold: float,
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
        route_gating_temperature=args.route_gating_temperature,
        route_gating_floor=args.route_gating_floor,
    )
    real_prob, _ = predict_route_aware_meta_fusion(
        route_meta_experts,
        real_features.route_prob,
        real_features.meta_x,
        args.pair_route_mode,
        route_gating_temperature=args.route_gating_temperature,
        route_gating_floor=args.route_gating_floor,
    )
    real_idx = np.arange(min(len(real_prob), 2000))

    rows = []
    for idx, path in enumerate(fake_paths):
        cache = load_compact_cache(
            path,
            max_rows=args.max_test_fake_per_method,
            seed=args.seed + idx,
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
            route_gating_temperature=args.route_gating_temperature,
            route_gating_floor=args.route_gating_floor,
        )
        fake_prob, _ = predict_route_aware_meta_fusion(
            route_meta_experts,
            fake_features.route_prob,
            fake_features.meta_x,
            args.pair_route_mode,
            route_gating_temperature=args.route_gating_temperature,
            route_gating_floor=args.route_gating_floor,
        )
        rows.append(
            {
                "group": str(cache.group[0]),
                "method": str(cache.method[0]),
                "metrics": method_metrics_with_threshold(fake_prob, real_prob[real_idx], route_meta_threshold),
            }
        )
    return summarize_method_rows(rows)


def evaluate_ood_route_meta(
    ood_entries: list[tuple[str, str, Path, Path]],
    *,
    args: argparse.Namespace,
    route_map: dict[str, np.ndarray],
    real_pool_means: np.ndarray,
    pair_mean_dirs: dict[str, np.ndarray],
    pair_classifiers,
    pair_region_idx: dict[str, int],
    pair_region_names_by_group: dict[str, list[str]],
    patch_scaler,
    patch_clf,
    patch_group_classifiers,
    route_meta_experts,
    route_meta_threshold: float,
) -> dict:
    rows = []
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
            route_gating_temperature=args.route_gating_temperature,
            route_gating_floor=args.route_gating_floor,
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
            route_gating_temperature=args.route_gating_temperature,
            route_gating_floor=args.route_gating_floor,
        )
        fake_prob, _ = predict_route_aware_meta_fusion(
            route_meta_experts,
            fake_features.route_prob,
            fake_features.meta_x,
            args.pair_route_mode,
            route_gating_temperature=args.route_gating_temperature,
            route_gating_floor=args.route_gating_floor,
        )
        real_prob, _ = predict_route_aware_meta_fusion(
            route_meta_experts,
            real_features.route_prob,
            real_features.meta_x,
            args.pair_route_mode,
            route_gating_temperature=args.route_gating_temperature,
            route_gating_floor=args.route_gating_floor,
        )
        rows.append(
            {
                "group": group,
                "method": method,
                "metrics": method_metrics_with_threshold(fake_prob, real_prob, route_meta_threshold),
            }
        )
    return summarize_method_rows(rows)


def main() -> None:
    args = parse_args()
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
        args.val_ratio,
        args.seed,
        n_fake_train,
        split_mode=args.val_split_mode,
    )

    patch_scaler, patch_clf = fit_patch_classifier(train_x[fit_idx], train_y[fit_idx], args.seed)
    patch_train_prob = predict_patch_scores(patch_scaler, patch_clf, train_x)
    patch_group_classifiers = fit_group_binary_experts(train_fake_x, np.concatenate(train_fake_groups), train_real_x, args.seed)
    patch_train_scores = predict_group_binary_experts(patch_group_classifiers, train_x)

    hybrid_model, hybrid_mean, hybrid_std, hybrid_temperature, hybrid_alpha, hybrid_labels = load_hybrid_checkpoint(args.hybrid_checkpoint, device)
    if list(hybrid_labels) != list(HYBRID_LABELS):
        raise ValueError(
            f"Hybrid checkpoint labels {hybrid_labels} do not match downstream group scheme {HYBRID_LABELS}. "
            "Pass --no-fr when using a no-FR upstream checkpoint."
        )
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

    train_meta_x = build_meta_features(
        train_route_prob_mat,
        train_pair_scores,
        patch_train_scores,
        patch_train_prob,
        route_gating_temperature=args.route_gating_temperature,
        route_gating_floor=args.route_gating_floor,
    )
    route_meta_experts, route_meta_search = fit_route_aware_meta_experts(
        train_meta_x,
        train_y,
        train_groups,
        fit_idx,
        val_idx,
        args.seed,
    )
    val_route_meta_prob, _ = predict_route_aware_meta_fusion(
        route_meta_experts,
        train_route_prob_mat[val_idx],
        train_meta_x[val_idx],
        args.pair_route_mode,
        route_gating_temperature=args.route_gating_temperature,
        route_gating_floor=args.route_gating_floor,
    )
    threshold_choice = search_threshold(train_y[val_idx], val_route_meta_prob)

    eval_groups = list(BASE_FAKE_GROUPS) if args.no_fr else list(EVAL_GROUPS)
    test_ff = evaluate_regular_route_meta(
        discover_flat_method_caches(args.patch_cache_root, "DF40_test_ff", eval_groups),
        test_real_cache,
        args=args,
        route_map=test_route_map,
        real_pool_means=real_pool_means,
        pair_mean_dirs=pair_mean_dirs,
        pair_classifiers=pair_classifiers,
        pair_region_idx=pair_region_idx,
        pair_region_names_by_group=pair_region_names_by_group,
        patch_scaler=patch_scaler,
        patch_clf=patch_clf,
        patch_group_classifiers=patch_group_classifiers,
        route_meta_experts=route_meta_experts,
        route_meta_threshold=float(threshold_choice["threshold"]),
    )
    ood = evaluate_ood_route_meta(
        discover_ood_method_entries(args.patch_cache_root, eval_groups),
        args=args,
        route_map=ood_route_map,
        real_pool_means=real_pool_means,
        pair_mean_dirs=pair_mean_dirs,
        pair_classifiers=pair_classifiers,
        pair_region_idx=pair_region_idx,
        pair_region_names_by_group=pair_region_names_by_group,
        patch_scaler=patch_scaler,
        patch_clf=patch_clf,
        patch_group_classifiers=patch_group_classifiers,
        route_meta_experts=route_meta_experts,
        route_meta_threshold=float(threshold_choice["threshold"]),
    )
    output_test_ff = test_ff
    output_ood = ood
    extra_summaries: dict[str, dict] = {}
    if args.no_fr:
        extra_summaries = {
            "test_ff_nonfr": summarize_method_rows_for_groups(test_ff["methods"], list(FAKE_GROUPS)),
            "test_ff_fr": summarize_method_rows_for_groups(test_ff["methods"], ["FR"]),
            "ood_nonfr": summarize_method_rows_for_groups(ood["methods"], list(FAKE_GROUPS)),
            "ood_fr": summarize_method_rows_for_groups(ood["methods"], ["FR"]),
        }

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
            "route_gating_temperature": args.route_gating_temperature,
            "route_gating_floor": args.route_gating_floor,
            "patch_train_groups": list(args.patch_train_groups),
            "val_split_mode": args.val_split_mode,
            "no_fr": bool(args.no_fr),
            "labels": list(HYBRID_LABELS),
            "seed": args.seed,
            "train_real_max": args.train_real_max,
            "eval_real_max": args.eval_real_max,
            "max_train_fake_per_method": args.max_train_fake_per_method,
            "route_batch_size": args.route_batch_size,
            "val_ratio": args.val_ratio,
        },
        "pair_region_names_by_group": pair_region_names_by_group,
        "route_meta_fusion_search": route_meta_search,
        "threshold_search": threshold_choice,
        "train_summary": {
            "num_train_samples": int(len(train_y)),
            "num_fake": int(np.sum(train_y == 1)),
            "num_real": int(np.sum(train_y == 0)),
        },
        "validation": {
            "route_meta_fusion": threshold_choice["balanced_accuracy"],
        },
        "test_ff": {
            "route_meta_fusion": output_test_ff,
        },
        "ood": {
            "route_meta_fusion": output_ood,
        },
    }
    for key, value in extra_summaries.items():
        output[key] = {"route_meta_fusion": value}
    patch_bundle = {
        "patch_scaler": patch_scaler,
        "patch_clf": patch_clf,
        "patch_group_classifiers": patch_group_classifiers,
    }
    pair_bundle = {
        "pair_mean_dirs": pair_mean_dirs,
        "pair_classifiers": pair_classifiers,
        "pair_region_idx": pair_region_idx,
        "pair_region_names_by_group": pair_region_names_by_group,
        "real_pool_means": real_pool_means,
        "region_names": np.asarray(real_ref["region_names"], dtype=object),
    }
    head_bundle = {
        "route_meta_experts": route_meta_experts,
        "route_meta_threshold": float(threshold_choice["threshold"]),
        "pair_route_mode": args.pair_route_mode,
        "route_gating_temperature": float(args.route_gating_temperature),
        "route_gating_floor": float(args.route_gating_floor),
    }
    head_meta = {
        "date": str(date.today()),
        "variant": "no_fr" if args.no_fr else "full",
        "default_threshold": float(threshold_choice["threshold"]),
        "pair_route_mode": args.pair_route_mode,
        "route_gating_temperature": float(args.route_gating_temperature),
        "route_gating_floor": float(args.route_gating_floor),
        "config": output["config"],
        "source": "src/train/train_downstream_head.py",
    }
    for path in [
        args.output_patch_branch,
        args.output_pair_branch,
        args.output_route_meta_head,
        args.output_head_meta,
        args.output_json,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(patch_bundle, args.output_patch_branch)
    joblib.dump(pair_bundle, args.output_pair_branch)
    joblib.dump(head_bundle, args.output_route_meta_head)
    args.output_head_meta.write_text(json.dumps(head_meta, indent=2))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                "validation_route_meta_fusion": output["validation"]["route_meta_fusion"],
                "test_ff_route_meta_fusion": output["test_ff"]["route_meta_fusion"]["summary"],
                "ood_route_meta_fusion": output["ood"]["route_meta_fusion"]["summary"],
                "threshold_search": output["threshold_search"],
            },
            indent=2,
        )
    )
    print(f"Saved to {args.output_json}")
    print(f"Saved patch branch to {args.output_patch_branch}")
    print(f"Saved pair branch to {args.output_pair_branch}")
    print(f"Saved route_meta head to {args.output_route_meta_head}")
    print(f"Saved head meta to {args.output_head_meta}")


if __name__ == "__main__":
    main()
