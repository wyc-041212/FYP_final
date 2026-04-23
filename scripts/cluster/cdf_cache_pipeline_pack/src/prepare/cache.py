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


def build_cache(
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
