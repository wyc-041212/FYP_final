from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class EmbeddingCache:
    cls: np.ndarray
    img_id: np.ndarray
    label: np.ndarray
    group: np.ndarray
    method: np.ndarray
    pair_id: np.ndarray

    @classmethod
    def from_npz(cls, path: str | Path) -> "EmbeddingCache":
        data = np.load(path, allow_pickle=True)
        group = data["group"] if "group" in data.files else data["fake_type"]
        method = data["method"] if "method" in data.files else data["fake_type"]
        return cls(
            cls=data["cls"],
            img_id=data["img_id"],
            label=data["label"].astype(np.int64),
            group=group,
            method=method,
            pair_id=data["pair_id"],
        )

    def to_npz(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            cls=self.cls,
            img_id=self.img_id,
            label=self.label,
            group=self.group,
            method=self.method,
            pair_id=self.pair_id,
        )


@dataclass
class PatchCache:
    tokens: np.ndarray
    img_id: np.ndarray
    label: np.ndarray
    group: np.ndarray
    method: np.ndarray
    pair_id: np.ndarray
    real_path: np.ndarray | None = None
    region_labels: np.ndarray | None = None
    region_names: np.ndarray | None = None

    @classmethod
    def from_npz(cls, path: str | Path) -> "PatchCache":
        data = np.load(path, allow_pickle=True)
        region_labels = data["region_labels"].astype(np.int16) if "region_labels" in data.files else None
        region_names = data["region_names"].astype(object) if "region_names" in data.files else None
        return cls(
            tokens=data["tokens"].astype(np.float32),
            img_id=data["img_id"].astype(object),
            label=data["label"].astype(np.int64),
            group=data["group"].astype(object),
            method=data["method"].astype(object),
            pair_id=data["pair_id"].astype(object),
            real_path=(
                data["real_path"].astype(object)
                if "real_path" in data.files
                else np.full(data["tokens"].shape[0], "", dtype=object)
            ),
            region_labels=region_labels,
            region_names=region_names,
        )

    def to_npz(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tokens": self.tokens,
            "img_id": self.img_id,
            "label": self.label,
            "group": self.group,
            "method": self.method,
            "pair_id": self.pair_id,
        }
        if self.real_path is not None:
            payload["real_path"] = self.real_path
        if self.region_labels is not None:
            payload["region_labels"] = self.region_labels
        if self.region_names is not None:
            payload["region_names"] = self.region_names
        np.savez_compressed(out_path, **payload)


@dataclass
class CompactPatchCache:
    region_vecs: np.ndarray
    region_present: np.ndarray
    img_id: np.ndarray
    group: np.ndarray
    method: np.ndarray
    real_path: np.ndarray
    region_names: np.ndarray

    @classmethod
    def from_npz(cls, path: str | Path) -> "CompactPatchCache":
        data = np.load(path, allow_pickle=True)
        return cls(
            region_vecs=data["region_vecs"].astype(np.float16),
            region_present=data["region_present"].astype(bool),
            img_id=data["img_id"].astype(object),
            group=data["group"].astype(object),
            method=data["method"].astype(object),
            real_path=data["real_path"].astype(object),
            region_names=data["region_names"].astype(object),
        )

    def to_npz(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            region_vecs=self.region_vecs,
            region_present=self.region_present,
            img_id=self.img_id,
            group=self.group,
            method=self.method,
            real_path=self.real_path,
            region_names=self.region_names,
        )


def compute_region_means(tokens: np.ndarray, labels: np.ndarray, num_regions: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((num_regions, dim), dtype=np.float32)
    present = np.zeros(num_regions, dtype=bool)
    for region_idx in np.unique(labels):
        mask = labels == region_idx
        out[region_idx] = tokens[mask].mean(axis=0)
        present[region_idx] = True
    return out, present


def build_cls_cache(
    cls_embeddings: np.ndarray,
    img_ids: Iterable[str],
    labels: Iterable[int],
    groups: Iterable[str],
    methods: Iterable[str],
    pair_ids: Iterable[str],
) -> EmbeddingCache:
    img_id_arr = np.asarray(list(img_ids), dtype=object)
    label_arr = np.asarray(list(labels), dtype=np.int64)
    group_arr = np.asarray(list(groups), dtype=object)
    method_arr = np.asarray(list(methods), dtype=object)
    pair_id_arr = np.asarray(list(pair_ids), dtype=object)
    if cls_embeddings.shape[0] != img_id_arr.shape[0]:
        raise ValueError("CLS embedding count does not match metadata count.")
    return EmbeddingCache(
        cls=cls_embeddings.astype(np.float32),
        img_id=img_id_arr,
        label=label_arr,
        group=group_arr,
        method=method_arr,
        pair_id=pair_id_arr,
    )


def build_patch_cache(
    patch_tokens: np.ndarray,
    img_ids: Iterable[str],
    labels: Iterable[int],
    groups: Iterable[str],
    methods: Iterable[str],
    pair_ids: Iterable[str],
    *,
    real_paths: Iterable[str] | np.ndarray | None = None,
    region_labels: np.ndarray | None = None,
    region_names: Iterable[str] | np.ndarray | None = None,
) -> PatchCache:
    img_id_arr = np.asarray(list(img_ids), dtype=object)
    label_arr = np.asarray(list(labels), dtype=np.int64)
    group_arr = np.asarray(list(groups), dtype=object)
    method_arr = np.asarray(list(methods), dtype=object)
    pair_id_arr = np.asarray(list(pair_ids), dtype=object)
    token_arr = patch_tokens.astype(np.float32, copy=False)
    if token_arr.shape[0] != img_id_arr.shape[0]:
        raise ValueError("Patch token count does not match metadata count.")

    real_path_arr = None
    if real_paths is not None:
        real_path_arr = np.asarray(list(real_paths), dtype=object)
        if real_path_arr.shape[0] != token_arr.shape[0]:
            raise ValueError("Real-path row count does not match patch token count.")
    else:
        real_path_arr = np.full(token_arr.shape[0], "", dtype=object)

    region_label_arr = None
    if region_labels is not None:
        region_label_arr = region_labels.astype(np.int16, copy=False)
        if region_label_arr.shape[0] != token_arr.shape[0]:
            raise ValueError("Region-label row count does not match patch token count.")

    region_name_arr = None
    if region_names is not None:
        region_name_arr = np.asarray(list(region_names), dtype=object)

    return PatchCache(
        tokens=token_arr,
        img_id=img_id_arr,
        label=label_arr,
        group=group_arr,
        method=method_arr,
        pair_id=pair_id_arr,
        real_path=real_path_arr,
        region_labels=region_label_arr,
        region_names=region_name_arr,
    )


def build_compact_cache(source_path: str | Path, compact_path: str | Path | None = None) -> CompactPatchCache:
    cache = PatchCache.from_npz(source_path)
    if cache.region_labels is None or cache.region_names is None:
        raise ValueError(f"Patch cache is missing region labels/names: {source_path}")

    num_regions = len(cache.region_names)
    dim = cache.tokens.shape[-1]
    region_vecs = np.zeros((cache.tokens.shape[0], num_regions, dim), dtype=np.float16)
    region_present = np.zeros((cache.tokens.shape[0], num_regions), dtype=bool)
    for row_idx in range(cache.tokens.shape[0]):
        vec, present = compute_region_means(cache.tokens[row_idx], cache.region_labels[row_idx], num_regions, dim)
        region_vecs[row_idx] = vec.astype(np.float16)
        region_present[row_idx] = present

    compact = CompactPatchCache(
        region_vecs=region_vecs,
        region_present=region_present,
        img_id=cache.img_id.astype(object),
        group=cache.group.astype(object),
        method=cache.method.astype(object),
        real_path=(
            cache.real_path.astype(object)
            if cache.real_path is not None
            else np.full(cache.tokens.shape[0], "", dtype=object)
        ),
        region_names=cache.region_names.astype(object),
    )
    if compact_path is not None:
        compact.to_npz(compact_path)
    return compact


def subset_compact_cache(cache: CompactPatchCache, max_rows: int, seed: int) -> CompactPatchCache:
    n_rows = len(cache.img_id)
    if max_rows > 0 and n_rows > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_rows, size=max_rows, replace=False))
    else:
        idx = np.arange(n_rows, dtype=np.int64)
    return CompactPatchCache(
        region_vecs=cache.region_vecs[idx],
        region_present=cache.region_present[idx],
        img_id=cache.img_id[idx],
        group=cache.group[idx],
        method=cache.method[idx],
        real_path=cache.real_path[idx],
        region_names=cache.region_names,
    )
