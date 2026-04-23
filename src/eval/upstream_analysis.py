from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.prepare.cache import EmbeddingCache

DEFAULT_ACTIVE_LOSS_WEIGHTS: dict[str, float] = {
    "lambda_real": 1.0,
    "lambda_efs": 1.0,
    "lambda_global": 1.0,
    "lambda_pair": 0.75,
    "lambda_sep": 0.1,
    "lambda_ortho": 0.01,
    "lambda_linear_aux": 0.5,
    "lambda_manifold_aux": 0.25,
}


@dataclass(frozen=True)
class LoadedSplit:
    cls: np.ndarray
    labels: np.ndarray
    methods: np.ndarray


def build_loss_sensitivity_jobs(
    base_weights: dict[str, float],
    *,
    loss_names: list[str] | None = None,
    scales: list[float] | tuple[float, ...] = (0.0, 0.25, 1.0, 2.0),
) -> list[dict[str, Any]]:
    ordered_losses = loss_names or list(base_weights)
    jobs: list[dict[str, Any]] = [
        {
            "job_name": "baseline",
            "kind": "baseline",
            "loss_name": None,
            "scale": 1.0,
            "weights": dict(base_weights),
        }
    ]
    for loss_name in ordered_losses:
        if loss_name not in base_weights:
            raise KeyError(f"Unknown loss coefficient: {loss_name}")
        for scale in scales:
            if np.isclose(scale, 1.0):
                continue
            weights = dict(base_weights)
            weights[loss_name] = float(base_weights[loss_name] * scale)
            jobs.append(
                {
                    "job_name": f"{loss_name}_x{scale:.2f}".replace(".", "p"),
                    "kind": "perturbation",
                    "loss_name": loss_name,
                    "scale": float(scale),
                    "weights": weights,
                }
            )
    return jobs


def choose_balanced_indices(labels: np.ndarray, *, max_per_label: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in sorted(np.unique(labels).tolist()):
        idx = np.flatnonzero(labels == label)
        if len(idx) > max_per_label:
            idx = rng.choice(idx, size=max_per_label, replace=False)
        selected.append(np.sort(idx.astype(np.int64)))
    if not selected:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(selected).astype(np.int64)


def append_anchor_points(
    *,
    sample_points: np.ndarray,
    sample_labels: np.ndarray,
    anchor_points: np.ndarray,
    anchor_labels: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    merged_points = np.concatenate([sample_points, anchor_points.astype(np.float32)], axis=0)
    merged_labels = np.concatenate([sample_labels.astype(object), np.asarray(anchor_labels, dtype=object)], axis=0)
    merged_kinds = np.concatenate(
        [
            np.full(len(sample_points), "sample", dtype=object),
            np.full(len(anchor_points), "anchor", dtype=object),
        ],
        axis=0,
    )
    return merged_points, merged_labels, merged_kinds


def subset_loaded_split(
    *,
    cls: np.ndarray,
    labels: np.ndarray,
    methods: np.ndarray,
    indices: np.ndarray,
) -> LoadedSplit:
    return LoadedSplit(
        cls=cls[indices].astype(np.float32, copy=False),
        labels=labels[indices].astype(object, copy=False),
        methods=methods[indices].astype(object, copy=False),
    )


def filter_supported_labels(split: LoadedSplit, *, supported_labels: list[str]) -> LoadedSplit:
    mask = np.isin(split.labels, np.asarray(supported_labels, dtype=object))
    indices = np.flatnonzero(mask).astype(np.int64)
    return subset_loaded_split(
        cls=split.cls,
        labels=split.labels,
        methods=split.methods,
        indices=indices,
    )


def compute_anchor_tree_layout(
    *,
    anchor_points: np.ndarray,
    anchor_labels: list[str],
    root_label: str = "REAL",
) -> dict[str, tuple[float, float]]:
    label_to_idx = {label: idx for idx, label in enumerate(anchor_labels)}
    root = anchor_points[label_to_idx[root_label]].astype(np.float32)
    fake_labels = [label for label in anchor_labels if label != root_label]
    if not fake_labels:
        return {root_label: (0.0, 0.0)}
    deltas = np.stack([anchor_points[label_to_idx[label]].astype(np.float32) - root for label in fake_labels], axis=0)
    centered = deltas - deltas.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T.astype(np.float32, copy=False)
    if basis.shape[1] < 2:
        pad = np.zeros((basis.shape[0], 2 - basis.shape[1]), dtype=np.float32)
        basis = np.concatenate([basis, pad], axis=1)
    coords = deltas @ basis
    layout = {root_label: (0.0, 0.0)}
    for label, xy in zip(fake_labels, coords, strict=True):
        layout[label] = (float(xy[0]), float(xy[1]))
    return layout


def load_embedding_split(cache_root: Path, split: str) -> LoadedSplit:
    from src.train.train_upstream import load_split

    cls, labels, methods = load_split(cache_root, split)
    return subset_loaded_split(
        cls=cls.astype(np.float32, copy=False),
        labels=labels.astype(object),
        methods=methods.astype(object),
        indices=np.arange(len(cls), dtype=np.int64),
    )


def load_validation_embedding_split(
    cache_root: Path,
    *,
    train_split: str,
    val_ratio: float,
    seed: int,
) -> LoadedSplit:
    from src.train.train_upstream import load_split, stratified_split

    cls, labels, methods = load_split(cache_root, train_split)
    _, val_idx = stratified_split(np.asarray(labels, dtype=object), val_ratio, seed)
    return subset_loaded_split(
        cls=cls.astype(np.float32, copy=False),
        labels=labels.astype(object),
        methods=methods.astype(object),
        indices=val_idx,
    )


def load_embedding_cache(path: Path) -> EmbeddingCache:
    return EmbeddingCache.from_npz(path)


def transform_embeddings(
    *,
    checkpoint_path: Path,
    cls: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, list[str], Any]:
    from src.train.train_downstream_head import load_hybrid_checkpoint

    model, mean, std, temperature, alpha, labels = load_hybrid_checkpoint(checkpoint_path, device)
    x_std = ((cls.astype(np.float32) - mean) / std).astype(np.float32)
    transformed: list[np.ndarray] = []
    route_prob: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_std), batch_size):
            batch = torch.from_numpy(x_std[start : start + batch_size]).to(device)
            y, _, _, fused = model.fused_logits(batch, temperature, alpha)
            transformed.append(y.cpu().numpy().astype(np.float32))
            route_prob.append(torch.softmax(fused, dim=1).cpu().numpy().astype(np.float32))
    return (
        np.concatenate(transformed, axis=0),
        np.concatenate(route_prob, axis=0),
        labels,
        model,
    )


def extract_anchor_points(model: Any) -> tuple[np.ndarray, list[str]]:
    anchor_points = model.all_offsets().detach().cpu().numpy().astype(np.float32)
    anchor_labels = list(model.labels)
    return anchor_points, anchor_labels


def summarize_true_residuals(model: Any, transformed: np.ndarray, labels: np.ndarray) -> list[dict[str, float | str | int]]:
    with torch.no_grad():
        dists = model.manifold_dist_sq(torch.from_numpy(transformed.astype(np.float32))).cpu().numpy().astype(np.float32)
    return summarize_residual_rows(labels=labels, dists=dists, model_labels=list(model.labels))


def compute_residual_diagnostics(
    *,
    labels: np.ndarray,
    dists: np.ndarray,
    model_labels: list[str],
) -> list[dict[str, float | str | int]]:
    label_to_idx = {label: idx for idx, label in enumerate(model_labels)}
    rows: list[dict[str, float | str | int]] = []
    for idx, label in enumerate(labels.tolist()):
        true_col = label_to_idx[str(label)]
        true_residual = float(dists[idx, true_col])
        other = np.delete(dists[idx], true_col)
        nearest_other = float(np.min(other))
        rows.append(
            {
                "label": str(label),
                "true_residual": true_residual,
                "nearest_other_residual": nearest_other,
                "margin": nearest_other - true_residual,
            }
        )
    return rows


def summarize_residual_rows(
    *,
    labels: np.ndarray,
    dists: np.ndarray,
    model_labels: list[str],
) -> list[dict[str, float | str | int]]:
    label_to_idx = {label: idx for idx, label in enumerate(model_labels)}
    rows: list[dict[str, float | str | int]] = []
    for label in model_labels:
        idx = np.flatnonzero(labels == label)
        if len(idx) == 0:
            continue
        true_col = label_to_idx[label]
        true_dist = dists[idx, true_col]
        other_dist = np.delete(dists[idx], true_col, axis=1)
        nearest_other = np.min(other_dist, axis=1)
        rows.append(
            {
                "label": str(label),
                "count": int(len(idx)),
                "true_residual_mean": float(np.mean(true_dist)),
                "true_residual_median": float(np.median(true_dist)),
                "nearest_other_mean": float(np.mean(nearest_other)),
                "margin_mean": float(np.mean(nearest_other - true_dist)),
            }
        )
    return rows


def choose_tsne_perplexity(num_points: int) -> float:
    if num_points <= 3:
        return 2.0
    return float(max(5, min(30, (num_points - 1) // 3)))


def run_tsne(points: np.ndarray, *, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE

    points = points.astype(np.float32, copy=False)
    mean = points.mean(axis=0, keepdims=True)
    std = points.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    points_std = np.clip((points - mean) / std, -8.0, 8.0).astype(np.float32)
    perplexity = choose_tsne_perplexity(len(points))
    tsne = TSNE(
        n_components=2,
        init="random",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    )
    return tsne.fit_transform(points_std).astype(np.float32)


def run_pca(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32, copy=False)
    mean = points.mean(axis=0, keepdims=True)
    std = points.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    points_std = ((points - mean) / std).astype(np.float32)
    _, _, vt = np.linalg.svd(points_std, full_matrices=False)
    basis = vt[:2].T.astype(np.float32, copy=False)
    return (points_std @ basis).astype(np.float32)


def run_umap(points: np.ndarray, *, seed: int) -> np.ndarray:
    import umap

    points = points.astype(np.float32, copy=False)
    mean = points.mean(axis=0, keepdims=True)
    std = points.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    points_std = np.clip((points - mean) / std, -8.0, 8.0).astype(np.float32)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.15,
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(points_std).astype(np.float32)


def project_points(points: np.ndarray, *, method: str, seed: int) -> np.ndarray:
    method_key = method.lower()
    if method_key == "tsne":
        return run_tsne(points, seed=seed)
    if method_key == "umap":
        return run_umap(points, seed=seed)
    if method_key == "pca":
        return run_pca(points)
    raise ValueError(f"Unsupported projection method: {method}")
