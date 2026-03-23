#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from prepare.cache import CompactPatchCache, EmbeddingCache, PatchCache, build_compact_cache, subset_compact_cache

TRAIN_GROUPS = ["FS", "FR", "FE"]
EVAL_GROUPS = ["FS", "FR", "FE", "EFS"]
LABELS_5WAY = ["EFS", "FS", "FR", "FE", "REAL"]
LABEL_TO_IDX_5WAY = {label: idx for idx, label in enumerate(LABELS_5WAY)}


class LayerNormProbe(torch.nn.Module):
    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.ln = torch.nn.LayerNorm(dim)
        self.linear = torch.nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.ln(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified patch binary classifier + route-score weighted fusion.")
    parser.add_argument("--patch-cache-root", type=Path, default=ROOT / "cache" / "patch")
    parser.add_argument("--cls-cache-root", type=Path, default=ROOT / "cache" / "cls")
    parser.add_argument("--train-real-patch-cache", type=Path, default=ROOT / "cache" / "patch" / "DF40_train" / "REAL" / "patch_real_pool.npz")
    parser.add_argument("--test-real-patch-cache", type=Path, default=ROOT / "cache" / "patch" / "DF40_test_ff" / "REAL" / "patch_real_pool.npz")
    parser.add_argument("--train-real-cls-cache", type=Path, default=ROOT / "cache" / "cls" / "DF40_train" / "REAL" / "cls_real_dedup_from_existing.npz")
    parser.add_argument("--test-real-cls-cache", type=Path, default=ROOT / "cache" / "cls" / "DF40_test_ff" / "REAL" / "cls_real_dedup_from_existing.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-real-max", type=int, default=15000)
    parser.add_argument("--eval-real-max", type=int, default=2000)
    parser.add_argument("--max-train-fake-per-method", type=int, default=0)
    parser.add_argument("--max-test-fake-per-method", type=int, default=0)
    parser.add_argument("--max-ood-fake-per-method", type=int, default=0)
    parser.add_argument("--route-batch-size", type=int, default=1024)
    parser.add_argument("--route-epochs", type=int, default=40)
    parser.add_argument("--route-lr", type=float, default=1e-3)
    parser.add_argument("--route-weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--compact-cache-dir",
        type=Path,
        default=ROOT / "cache" / "compact",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "outputs" / "unified_patch_route_fusion_v1.json",
    )
    return parser.parse_args()


def load_patch_cache(path: Path, max_rows: int = 0, seed: int = 42) -> PatchCache:
    cache = PatchCache.from_npz(path)
    if cache.region_labels is None or cache.region_names is None:
        raise ValueError(f"Patch cache is missing region labels/names: {path}")
    n_rows = int(cache.tokens.shape[0])
    if max_rows > 0 and n_rows > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_rows, size=max_rows, replace=False))
    else:
        idx = slice(None)
    return PatchCache(
        tokens=cache.tokens[idx],
        img_id=cache.img_id[idx],
        label=cache.label[idx],
        group=cache.group[idx],
        method=cache.method[idx],
        pair_id=cache.pair_id[idx],
        region_labels=cache.region_labels[idx],
        region_names=cache.region_names,
    )


def compact_cache_file(compact_cache_dir: Path, source_path: Path) -> Path:
    try:
        rel = source_path.relative_to(ROOT)
        parts = rel.parts
    except ValueError:
        parts = source_path.resolve().parts[1:]
    safe = "__".join(parts)
    return compact_cache_dir / f"{safe}.npz"


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


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


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


def load_route_training_arrays(cache_root: Path, train_real_cache: Path) -> tuple[np.ndarray, np.ndarray]:
    x = []
    y = []
    for _, vec, group, _ in iter_regular_cls_rows(cache_root, "DF40_train", train_real_cache):
        x.append(vec)
        y.append(LABEL_TO_IDX_5WAY[group])
    return np.stack(x).astype(np.float32), np.asarray(y, dtype=np.int64)


def train_route_model(args: argparse.Namespace) -> tuple[LayerNormProbe, np.ndarray, np.ndarray]:
    train_x, train_y = load_route_training_arrays(args.cls_cache_root, args.train_real_cls_cache)
    mean, std = standardize_fit(train_x)
    train_std = ((train_x - mean) / std).astype(np.float32)
    model = LayerNormProbe(train_std.shape[1], len(LABELS_5WAY))
    counts = np.bincount(train_y, minlength=len(LABELS_5WAY))
    class_weight = len(train_y) / (len(LABELS_5WAY) * np.maximum(counts, 1))
    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=args.route_lr, weight_decay=args.route_weight_decay)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(train_std), torch.from_numpy(train_y)),
        batch_size=args.route_batch_size,
        shuffle=True,
    )
    for _ in range(args.route_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model, mean, std


def build_route_prob_map(model: LayerNormProbe, mean: np.ndarray, std: np.ndarray, rows: Iterable[tuple[str, np.ndarray, str, str]]) -> dict[str, np.ndarray]:
    mapping: dict[str, np.ndarray] = {}
    batch_keys: list[str] = []
    batch_vecs: list[np.ndarray] = []

    def flush() -> None:
        nonlocal batch_keys, batch_vecs
        if not batch_keys:
            return
        x = np.stack(batch_vecs).astype(np.float32)
        x = ((x - mean) / std).astype(np.float32)
        with torch.no_grad():
            prob = torch.softmax(model(torch.from_numpy(x)), dim=1).cpu().numpy().astype(np.float32)
        for key, row in zip(batch_keys, prob, strict=True):
            mapping[key] = row
        batch_keys = []
        batch_vecs = []

    for key, vec, _, _ in rows:
        batch_keys.append(key)
        batch_vecs.append(vec)
        if len(batch_keys) >= 4096:
            flush()
    flush()
    return mapping


def route_score(route_map: dict[str, np.ndarray], img_ids: np.ndarray) -> np.ndarray:
    rows = []
    for img_id in img_ids.tolist():
        prob = route_map[str(img_id)]
        rows.append(float(1.0 - prob[LABEL_TO_IDX_5WAY["REAL"]]))
    return np.asarray(rows, dtype=np.float32)


def discover_flat_method_caches(cache_root: Path, split: str, groups: list[str]) -> list[Path]:
    paths: list[Path] = []
    for group in groups:
        group_dir = cache_root / split / group
        if not group_dir.exists():
            continue
        paths.extend(sorted(group_dir.glob("patch_*.npz")))
    return paths


def discover_ood_method_entries(cache_root: Path) -> list[tuple[str, str, Path, Path]]:
    entries = []
    split_root = cache_root / "DF40_test_ood"
    for group in EVAL_GROUPS:
        group_dir = split_root / group
        if not group_dir.exists():
            continue
        for method_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()]):
            fake_path = method_dir / "patch_fake.npz"
            real_path = method_dir / "patch_real.npz"
            if fake_path.exists() and real_path.exists():
                entries.append((group, method_dir.name, fake_path, real_path))
    return entries


def split_train_indices(
    methods: np.ndarray,
    val_ratio: float,
    seed: int,
    n_real_start: int,
    split_mode: str = "within_method",
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit_idx: list[int] = []
    val_idx: list[int] = []
    method_keys = methods[:n_real_start]
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
    real_idx = np.arange(n_real_start, len(methods))
    rng.shuffle(real_idx)
    n_val_real = max(1, int(round(len(real_idx) * val_ratio)))
    n_val_real = min(n_val_real, len(real_idx) - 1) if len(real_idx) > 1 else 0
    if n_val_real > 0:
        val_idx.extend(real_idx[:n_val_real].tolist())
        fit_idx.extend(real_idx[n_val_real:].tolist())
    else:
        fit_idx.extend(real_idx.tolist())
    return np.asarray(fit_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def method_metrics(prob_fake: np.ndarray, prob_real: np.ndarray) -> dict:
    y = np.concatenate([np.ones(len(prob_fake), dtype=np.int64), np.zeros(len(prob_real), dtype=np.int64)], axis=0)
    prob = np.concatenate([prob_fake, prob_real], axis=0)
    pred = (prob >= 0.5).astype(np.int64)
    fake_acc = float(np.mean(pred[: len(prob_fake)] == 1)) if len(prob_fake) else 0.0
    real_acc = float(np.mean(pred[len(prob_fake) :] == 0)) if len(prob_real) else 0.0
    return {
        "num_fake": int(len(prob_fake)),
        "num_real": int(len(prob_real)),
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": 0.5 * (fake_acc + real_acc),
        "fake_accuracy": fake_acc,
        "real_accuracy": real_acc,
        "auc": float(roc_auc_score(y, prob)),
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


def choose_weight(val_y: np.ndarray, patch_prob: np.ndarray, route_prob: np.ndarray) -> dict:
    best: dict | None = None
    for w in np.linspace(0.0, 1.0, 21):
        fused = w * route_prob + (1.0 - w) * patch_prob
        pred = (fused >= 0.5).astype(np.int64)
        fake_mask = val_y == 1
        real_mask = val_y == 0
        fake_acc = float(np.mean(pred[fake_mask] == 1)) if np.any(fake_mask) else 0.0
        real_acc = float(np.mean(pred[real_mask] == 0)) if np.any(real_mask) else 0.0
        bacc = 0.5 * (fake_acc + real_acc)
        acc = float(np.mean(pred == val_y))
        auc = float(roc_auc_score(val_y, fused))
        key = (bacc, acc, auc, -abs(w - 0.5))
        if best is None or key > best["key"]:
            best = {"key": key, "weight_route": float(w), "weight_patch": float(1.0 - w), "val_balanced_accuracy": bacc, "val_accuracy": acc, "val_auc": auc}
    assert best is not None
    return {k: v for k, v in best.items() if k != "key"}


def build_train_patch_arrays(args: argparse.Namespace, real_pool_means: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_parts = []
    y_parts = []
    img_ids = []
    methods = []
    for path in discover_flat_method_caches(args.patch_cache_root, "DF40_train", TRAIN_GROUPS):
        print(f"[TRAIN-FAKE] {path}", flush=True)
        cache = load_compact_cache(path, max_rows=args.max_train_fake_per_method, seed=args.seed, compact_cache_dir=args.compact_cache_dir)
        x = image_region_delta_features(cache, real_pool_means)
        x_parts.append(x)
        y_parts.append(np.ones(len(x), dtype=np.int64))
        img_ids.append(cache.img_id.astype(object))
        methods.append(cache.method.astype(object))
    return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(img_ids), np.concatenate(methods)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("[LOAD] train real pool", flush=True)
    train_real_cache = load_compact_cache(args.train_real_patch_cache, max_rows=args.train_real_max, seed=args.seed, compact_cache_dir=args.compact_cache_dir)
    region_names = [str(x) for x in train_real_cache.region_names.tolist()]
    real_pool_means = compute_real_pool_region_means(train_real_cache)

    print("[PATCH] build training arrays", flush=True)
    train_fake_x, train_fake_y, train_fake_img_ids, train_fake_methods = build_train_patch_arrays(args, real_pool_means)
    train_real_x = image_region_delta_features(train_real_cache, real_pool_means)
    train_real_y = np.zeros(len(train_real_x), dtype=np.int64)
    train_real_img_ids = train_real_cache.img_id.astype(object)
    train_real_methods = train_real_cache.method.astype(object)

    train_x = np.concatenate([train_fake_x, train_real_x], axis=0)
    train_y = np.concatenate([train_fake_y, train_real_y], axis=0)
    train_img_ids = np.concatenate([train_fake_img_ids, train_real_img_ids], axis=0)
    train_methods = np.concatenate([train_fake_methods, train_real_methods], axis=0)
    n_fake_train = len(train_fake_x)

    fit_idx, val_idx = split_train_indices(train_methods, args.val_ratio, args.seed, n_fake_train)
    print(f"[FIT] patch train_x={train_x.shape} fit={len(fit_idx)} val={len(val_idx)}", flush=True)
    patch_scaler, patch_clf = fit_patch_classifier(train_x[fit_idx], train_y[fit_idx], args.seed)
    patch_train_prob = predict_patch_scores(patch_scaler, patch_clf, train_x)

    print("[ROUTE] train route model", flush=True)
    route_model, route_mean, route_std = train_route_model(args)
    train_route_map = build_route_prob_map(route_model, route_mean, route_std, iter_regular_cls_rows(args.cls_cache_root, "DF40_train", args.train_real_cls_cache))
    test_route_map = build_route_prob_map(route_model, route_mean, route_std, iter_regular_cls_rows(args.cls_cache_root, "DF40_test_ff", args.test_real_cls_cache))
    ood_route_map = build_route_prob_map(route_model, route_mean, route_std, iter_ood_cls_rows(args.cls_cache_root, "DF40_test_ood"))
    route_train_prob = route_score(train_route_map, train_img_ids)

    weight_choice = choose_weight(train_y[val_idx], patch_train_prob[val_idx], route_train_prob[val_idx])

    print("[LOAD] test_ff real pool", flush=True)
    test_real_cache = load_compact_cache(args.test_real_patch_cache, max_rows=args.eval_real_max, seed=args.seed, compact_cache_dir=args.compact_cache_dir)
    test_real_x = image_region_delta_features(test_real_cache, real_pool_means)
    test_real_patch_prob_full = predict_patch_scores(patch_scaler, patch_clf, test_real_x)
    test_real_route_prob_full = route_score(test_route_map, test_real_cache.img_id.astype(object))

    test_patch_rows = []
    test_route_rows = []
    test_fusion_rows = []
    for path in discover_flat_method_caches(args.patch_cache_root, "DF40_test_ff", EVAL_GROUPS):
        print(f"[TEST-FF] {path}", flush=True)
        cache = load_compact_cache(path, max_rows=args.max_test_fake_per_method, seed=args.seed, compact_cache_dir=args.compact_cache_dir)
        fake_x = image_region_delta_features(cache, real_pool_means)
        patch_fake = predict_patch_scores(patch_scaler, patch_clf, fake_x)
        route_fake = route_score(test_route_map, cache.img_id.astype(object))
        n_real = min(len(test_real_x), len(fake_x))
        real_idx = np.sort(rng.choice(len(test_real_x), size=n_real, replace=False))
        patch_real = test_real_patch_prob_full[real_idx]
        route_real = test_real_route_prob_full[real_idx]
        fused_fake = weight_choice["weight_route"] * route_fake + weight_choice["weight_patch"] * patch_fake
        fused_real = weight_choice["weight_route"] * route_real + weight_choice["weight_patch"] * patch_real
        group = str(cache.group[0])
        method = str(cache.method[0])
        test_patch_rows.append({"group": group, "method": method, "metrics": method_metrics(patch_fake, patch_real)})
        test_route_rows.append({"group": group, "method": method, "metrics": method_metrics(route_fake, route_real)})
        test_fusion_rows.append({"group": group, "method": method, "metrics": method_metrics(fused_fake, fused_real)})

    ood_patch_rows = []
    ood_route_rows = []
    ood_fusion_rows = []
    for group, method, fake_path, real_path in discover_ood_method_entries(args.patch_cache_root):
        print(f"[OOD] {fake_path}", flush=True)
        fake_cache = load_compact_cache(fake_path, max_rows=args.max_ood_fake_per_method, seed=args.seed, compact_cache_dir=args.compact_cache_dir)
        real_cache = load_compact_cache(real_path, max_rows=args.eval_real_max, seed=args.seed, compact_cache_dir=args.compact_cache_dir)
        fake_x = image_region_delta_features(fake_cache, real_pool_means)
        real_x = image_region_delta_features(real_cache, real_pool_means)
        patch_fake = predict_patch_scores(patch_scaler, patch_clf, fake_x)
        patch_real = predict_patch_scores(patch_scaler, patch_clf, real_x)
        route_fake = route_score(ood_route_map, fake_cache.img_id.astype(object))
        route_real = route_score(ood_route_map, real_cache.img_id.astype(object))
        n_real = min(len(real_x), len(fake_x))
        real_idx = np.sort(rng.choice(len(real_x), size=n_real, replace=False))
        patch_real = patch_real[real_idx]
        route_real = route_real[real_idx]
        fused_fake = weight_choice["weight_route"] * route_fake + weight_choice["weight_patch"] * patch_fake
        fused_real = weight_choice["weight_route"] * route_real + weight_choice["weight_patch"] * patch_real
        ood_patch_rows.append({"group": group, "method": method, "metrics": method_metrics(patch_fake, patch_real)})
        ood_route_rows.append({"group": group, "method": method, "metrics": method_metrics(route_fake, route_real)})
        ood_fusion_rows.append({"group": group, "method": method, "metrics": method_metrics(fused_fake, fused_real)})

    output = {
        "config": {
            "patch_cache_root": str(args.patch_cache_root),
            "cls_cache_root": str(args.cls_cache_root),
            "train_real_patch_cache": str(args.train_real_patch_cache),
            "test_real_patch_cache": str(args.test_real_patch_cache),
            "train_real_cls_cache": str(args.train_real_cls_cache),
            "test_real_cls_cache": str(args.test_real_cls_cache),
            "seed": args.seed,
            "train_real_max": args.train_real_max,
            "eval_real_max": args.eval_real_max,
            "max_train_fake_per_method": args.max_train_fake_per_method,
            "max_test_fake_per_method": args.max_test_fake_per_method,
            "max_ood_fake_per_method": args.max_ood_fake_per_method,
            "route_batch_size": args.route_batch_size,
            "route_epochs": args.route_epochs,
            "route_lr": args.route_lr,
            "route_weight_decay": args.route_weight_decay,
            "val_ratio": args.val_ratio,
            "compact_cache_dir": str(args.compact_cache_dir),
        },
        "region_names": region_names,
        "patch_train_summary": {
            "num_train_samples": int(len(train_y)),
            "num_fake": int(n_fake_train),
            "num_real": int(len(train_real_x)),
            "feature_dim": int(train_x.shape[1]),
            "groups_used": TRAIN_GROUPS,
        },
        "fusion_search": weight_choice,
        "validation": {
            "patch_only": method_metrics(patch_train_prob[val_idx][train_y[val_idx] == 1], patch_train_prob[val_idx][train_y[val_idx] == 0]),
            "route_only": method_metrics(route_train_prob[val_idx][train_y[val_idx] == 1], route_train_prob[val_idx][train_y[val_idx] == 0]),
            "weighted_fusion": method_metrics(
                (weight_choice["weight_route"] * route_train_prob[val_idx] + weight_choice["weight_patch"] * patch_train_prob[val_idx])[train_y[val_idx] == 1],
                (weight_choice["weight_route"] * route_train_prob[val_idx] + weight_choice["weight_patch"] * patch_train_prob[val_idx])[train_y[val_idx] == 0],
            ),
        },
        "test_ff": {
            "patch_only": {"summary": summarize_split(test_patch_rows), "methods": sorted(test_patch_rows, key=lambda row: (row["group"], row["method"]))},
            "route_only": {"summary": summarize_split(test_route_rows), "methods": sorted(test_route_rows, key=lambda row: (row["group"], row["method"]))},
            "weighted_fusion": {"summary": summarize_split(test_fusion_rows), "methods": sorted(test_fusion_rows, key=lambda row: (row["group"], row["method"]))},
        },
        "ood": {
            "patch_only": {"summary": summarize_split(ood_patch_rows), "methods": sorted(ood_patch_rows, key=lambda row: (row["group"], row["method"]))},
            "route_only": {"summary": summarize_split(ood_route_rows), "methods": sorted(ood_route_rows, key=lambda row: (row["group"], row["method"]))},
            "weighted_fusion": {"summary": summarize_split(ood_fusion_rows), "methods": sorted(ood_fusion_rows, key=lambda row: (row["group"], row["method"]))},
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                "weight_route": weight_choice["weight_route"],
                "weight_patch": weight_choice["weight_patch"],
                "ood_patch_mean_bacc": output["ood"]["patch_only"]["summary"]["mean_balanced_accuracy"],
                "ood_route_mean_bacc": output["ood"]["route_only"]["summary"]["mean_balanced_accuracy"],
                "ood_fusion_mean_bacc": output["ood"]["weighted_fusion"]["summary"]["mean_balanced_accuracy"],
            },
            indent=2,
        )
    )
    print(f"[DONE] wrote report to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
