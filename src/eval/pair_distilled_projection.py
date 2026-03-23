#!/usr/bin/env python3
"""Pair-Distilled Linear Projection.

Per-region Ridge 回归，学习从 region embedding 预测 pair delta。
训练时用配对数据（fake-real pairs），推理时不需要配对 real。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

EPS = 1e-8
VOLUME_PREFIX = "/Volumes/未命名"
INTERIOR_REGIONS = ["skin", "eye", "eyebrow", "nose", "lip", "mouth"]


def normalize_key(path_like: str | Path) -> str:
    return str(path_like).replace(VOLUME_PREFIX, "")


def mean_region_vectors(tokens: np.ndarray, labels: np.ndarray, num_regions: int, dim: int) -> np.ndarray:
    """从 patch tokens 计算 per-region 的 mean-pooled embedding."""
    out = np.zeros((num_regions, dim), dtype=np.float32)
    for r in range(num_regions):
        mask = labels == r
        if np.any(mask):
            out[r] = tokens[mask].mean(axis=0)
    return out


def train_region_projections(
    fake_region_vecs: np.ndarray,
    paired_real_region_vecs: np.ndarray,
    real_region_vecs: np.ndarray,
    n_regions: int,
    alpha: float = 1.0,
) -> list[Ridge]:
    """训练 per-region Ridge 回归：region_embedding → 预测 pair_delta。

    参数:
        fake_region_vecs: (n_fake, n_regions, dim) fake 图像的 region embeddings
        paired_real_region_vecs: (n_fake, n_regions, dim) 配对 real 的 region embeddings
        real_region_vecs: (n_real, n_regions, dim) 训练集 real 的 region embeddings
        n_regions: region 数量
        alpha: Ridge 正则化参数

    返回:
        per-region Ridge 模型列表
    """
    projections = []
    for r in range(n_regions):
        X_fake = fake_region_vecs[:, r, :].astype(np.float64)
        Y_fake = (fake_region_vecs[:, r, :] - paired_real_region_vecs[:, r, :]).astype(np.float64)

        X_real = real_region_vecs[:, r, :].astype(np.float64)
        Y_real = np.zeros_like(X_real, dtype=np.float64)

        X = np.concatenate([X_fake, X_real], axis=0)
        Y = np.concatenate([Y_fake, Y_real], axis=0)

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X, Y)
        projections.append(model)
    return projections


def apply_projection_batch(
    region_vecs: np.ndarray,
    projections: list[Ridge],
) -> np.ndarray:
    """批量应用 Ridge 投影，返回 pseudo-deltas。

    参数:
        region_vecs: (n_images, n_regions, dim)
        projections: per-region Ridge 模型列表

    返回:
        pseudo_deltas: (n_images, n_regions, dim)
    """
    n_images, n_regions, dim = region_vecs.shape
    pseudo_deltas = np.zeros_like(region_vecs, dtype=np.float32)
    for r in range(n_regions):
        pseudo_deltas[:, r, :] = projections[r].predict(
            region_vecs[:, r, :].astype(np.float64)
        ).astype(np.float32)
    return pseudo_deltas


def extract_projected_features(
    pseudo_deltas: np.ndarray,
    region_idx: dict[str, int],
    canonical_regions: list[str],
) -> np.ndarray:
    """从 pseudo-deltas 提取分类特征（拼接 canonical regions 的 delta）。

    参数:
        pseudo_deltas: (n_images, n_regions, dim)
        region_idx: region name → index 映射
        canonical_regions: 当前 type 的 canonical region 名称列表

    返回:
        features: (n_images, len(canonical_regions) * dim)
    """
    parts = []
    for name in canonical_regions:
        r = region_idx[name]
        parts.append(pseudo_deltas[:, r, :])
    return np.concatenate(parts, axis=1).astype(np.float32)


def extract_cos_norm_features(
    pseudo_deltas: np.ndarray,
    mean_delta_dirs: np.ndarray,
) -> np.ndarray:
    """从 pseudo-deltas 提取紧凑的 cos+norm 特征。

    参数:
        pseudo_deltas: (n_images, n_regions, dim)
        mean_delta_dirs: (n_regions, dim) 训练集 fake 的平均 delta 方向

    返回:
        features: (n_images, n_regions * 2)  [cos, norm] per region
    """
    n_images, n_regions, dim = pseudo_deltas.shape
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
    """计算训练集 fake 各 region 的平均 pair delta 方向。"""
    deltas = fake_region_vecs - paired_real_region_vecs
    mean_dirs = deltas.mean(axis=0).astype(np.float32)
    for r in range(n_regions):
        norm = float(np.linalg.norm(mean_dirs[r]))
        if norm > EPS:
            mean_dirs[r] /= norm
    return mean_dirs


def precompute_method_region_vecs(
    cache_tokens: np.ndarray,
    cache_region_labels: np.ndarray,
    cache_img_id: np.ndarray,
    cache_real_path: np.ndarray,
    real_ref: dict,
    real_lookup: dict[str, int],
    num_regions: int,
    dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """预计算一个 method 的所有 (fake_region_vecs, paired_real_region_vecs, has_pair)。

    返回:
        fake_vecs: (n_images, num_regions, dim)
        paired_real_vecs: (n_images, num_regions, dim)
        has_pair: (n_images,) bool
    """
    n = len(cache_img_id)
    fake_vecs = np.zeros((n, num_regions, dim), dtype=np.float32)
    paired_real_vecs = np.zeros((n, num_regions, dim), dtype=np.float32)
    has_pair = np.zeros(n, dtype=bool)

    for i in range(n):
        fake_vecs[i] = mean_region_vectors(
            cache_tokens[i], cache_region_labels[i], num_regions, dim
        )
        real_path = str(cache_real_path[i])
        if real_path:
            real_idx = real_lookup.get(normalize_key(real_path))
            if real_idx is not None:
                paired_real_vecs[i] = real_ref["region_vecs"][real_idx]
                has_pair[i] = True

    return fake_vecs, paired_real_vecs, has_pair


# ── PCA 压缩 ─────────────────────────────────────────────────────────────────


def fit_pca_on_features(
    features: np.ndarray,
    n_components: int,
) -> PCA:
    """对拼接后的 pseudo-delta features 做 PCA 降维。

    参数:
        features: (n_samples, high_dim) 训练集特征
        n_components: 目标维度

    返回:
        fitted PCA model
    """
    n_components = min(n_components, features.shape[1], features.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(features.astype(np.float64))
    return pca


def apply_pca_to_features(
    features: np.ndarray,
    pca: PCA,
) -> np.ndarray:
    """用训练好的 PCA 降维。"""
    return pca.transform(features.astype(np.float64)).astype(np.float32)


def extract_norm_features(
    pseudo_deltas: np.ndarray,
    region_idx: dict[str, int],
    canonical_regions: list[str],
) -> np.ndarray:
    """从 pseudo-deltas 提取每个 canonical region 的 L2 norm 作为特征。

    最简单的特征：fake 的 pseudo-delta norm 应该大于 real 的。
    维度 = len(canonical_regions)，不可能过拟合。

    参数:
        pseudo_deltas: (n_images, n_regions, dim)
        region_idx: region name → index
        canonical_regions: canonical region 名称列表

    返回:
        features: (n_images, len(canonical_regions))
    """
    parts = []
    for name in canonical_regions:
        r = region_idx[name]
        norms = np.linalg.norm(pseudo_deltas[:, r, :], axis=1, keepdims=True)
        parts.append(norms.astype(np.float32))
    return np.concatenate(parts, axis=1)

