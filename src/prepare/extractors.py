from __future__ import annotations

import sys
from typing import Iterator

import numpy as np
from tqdm import tqdm

from prepare.backbone import FrozenBackbone
from prepare.cache import EmbeddingCache, PatchCache, build_cls_cache, build_patch_cache
from prepare.face_regions import MERGED_ORDER, load_facer_models, parse_images_to_regions


def batch_rows(rows: list[dict], batch_size: int) -> Iterator[list[dict]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def extract_cls_cache(
    rows: list[dict],
    backbone: FrozenBackbone,
    batch_size: int,
    target_size: int,
    progress_desc: str = "cls",
) -> EmbeddingCache:
    cls_chunks = []
    img_ids = []
    labels = []
    groups = []
    methods = []
    pair_ids = []

    progress = tqdm(
        total=len(rows),
        desc=progress_desc,
        unit="img",
        file=sys.stdout,
        dynamic_ncols=True,
    )
    for batch in batch_rows(rows, batch_size):
        image_paths = [row["img_path"] for row in batch]
        encoded = backbone.encode_cls_batch(image_paths, target_size=target_size, layers=[-1])
        cls_chunks.append(encoded[-1])
        img_ids.extend(row["img_path"] for row in batch)
        labels.extend(int(row["label"]) for row in batch)
        groups.extend(row["group"] for row in batch)
        methods.extend(row["method"] for row in batch)
        pair_ids.extend(row["pair_id"] for row in batch)
        progress.update(len(batch))
    progress.close()

    cls_all = np.concatenate(cls_chunks, axis=0).astype(np.float32, copy=False)
    return build_cls_cache(cls_all, img_ids, labels, groups, methods, pair_ids)


def extract_patch_cache(
    rows: list[dict],
    backbone: FrozenBackbone,
    batch_size: int,
    target_size: int,
    layer: int = -1,
    with_regions: bool = True,
    facer_device: str = "cpu",
    progress_desc: str = "patch",
) -> PatchCache:
    detector = None
    parser = None
    region_name_arr = np.asarray(["background", *MERGED_ORDER], dtype=object)
    region_to_idx = {name: idx for idx, name in enumerate(region_name_arr.tolist())}
    if with_regions:
        detector, parser = load_facer_models(facer_device)

    token_chunks = []
    region_chunks = []
    img_ids = []
    labels = []
    groups = []
    methods = []
    pair_ids = []

    progress = tqdm(
        total=len(rows),
        desc=progress_desc,
        unit="img",
        file=sys.stdout,
        dynamic_ncols=True,
    )
    for batch in batch_rows(rows, batch_size):
        image_paths = [row["img_path"] for row in batch]
        patch_tokens = backbone.encode_patch_batch(
            image_paths,
            target_size=target_size,
            layer=layer,
        )
        token_chunks.append(patch_tokens)
        if with_regions:
            from PIL import Image

            images = [Image.open(path).convert("RGB") for path in image_paths]
            patch_h = patch_w = int(round(patch_tokens.shape[1] ** 0.5))
            region_names = parse_images_to_regions(
                images,
                (patch_h, patch_w),
                detector,
                parser,
                facer_device,
            )
            region_idx = np.stack(
                [
                    np.asarray([region_to_idx.get(str(name), 0) for name in region_vec], dtype=np.int16)
                    for region_vec in region_names
                ],
                axis=0,
            )
            region_chunks.append(region_idx)
        img_ids.extend(row["img_path"] for row in batch)
        labels.extend(int(row["label"]) for row in batch)
        groups.extend(row["group"] for row in batch)
        methods.extend(row["method"] for row in batch)
        pair_ids.extend(row["pair_id"] for row in batch)
        progress.update(len(batch))
    progress.close()

    region_labels = None
    region_names = None
    if with_regions:
        region_labels = np.concatenate(region_chunks, axis=0).astype(np.int16, copy=False)
        region_names = region_name_arr
    return build_patch_cache(
        np.concatenate(token_chunks, axis=0).astype(np.float32, copy=False),
        img_ids,
        labels,
        groups,
        methods,
        pair_ids,
        region_labels=region_labels,
        region_names=region_names,
    )
