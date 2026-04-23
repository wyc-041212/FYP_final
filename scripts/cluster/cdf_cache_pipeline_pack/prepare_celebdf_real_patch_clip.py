#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_SRC_ROOT = Path(os.environ.get("SRC_ROOT", PACKAGE_ROOT / "src")).expanduser()

_orig_version = importlib.metadata.version


def _patched_version(package: str) -> str:
    version = _orig_version(package)
    if package == "huggingface-hub":
        if version.startswith("1."):
            return "0.34.0"
        if version.startswith("0."):
            return "1.3.0"
    return version


importlib.metadata.version = _patched_version


def load_src_modules(src_root: Path):
    sys.path.insert(0, str(src_root))
    from after_a.face_regions import MERGED_ORDER, load_facer_models, parse_images_to_regions
    from prepare.backbone import FrozenBackbone

    return MERGED_ORDER, load_facer_models, parse_images_to_regions, FrozenBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLIP patch cache for Celeb-DF-v2 real datasets.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--relative-to", type=Path, default=Path(os.environ.get("RELATIVE_TO", PACKAGE_ROOT)).expanduser())
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "MODEL_DIR",
                Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--clip-vit-large-patch14",
            )
        ).expanduser(),
    )
    parser.add_argument("--backbone", default="clip")
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--with-regions", action="store_true", default=True)
    parser.add_argument("--without-regions", dest="with_regions", action="store_false")
    parser.add_argument("--facer-device", default=os.environ.get("FACER_DEVICE", "cuda"))
    return parser.parse_args()


def infer_spec(dataset_root: Path, relative_to: Path) -> tuple[str, str, str]:
    parts = dataset_root.relative_to(relative_to).parts
    if len(parts) < 2:
        raise ValueError(f"Expected <split>/<dataset>, got {dataset_root}")
    return parts[0], "real", parts[1]


def batch_items(items: list[dict], batch_size: int) -> Iterator[list[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    args = parse_args()
    MERGED_ORDER, load_facer_models, parse_images_to_regions, FrozenBackbone = load_src_modules(args.src_root)
    split, group, method = infer_spec(args.dataset_root, args.relative_to)

    manifest_path = args.manifest_root / split / group / f"manifest_{method}.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    output_dir = args.cache_root / split / group
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"patch_{method}.npz"

    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest_path}")

    print(
        f"[START] prepare_celebdf_real_patch_clip\n"
        f"  split={split}\n"
        f"  group={group}\n"
        f"  method={method}\n"
        f"  manifest={manifest_path}\n"
        f"  output={cache_path}\n"
        f"  rows={len(rows)}\n"
        f"  with_regions={args.with_regions}",
        flush=True,
    )

    backbone = FrozenBackbone(backbone_name=args.backbone, model_dir=args.model_dir, device=args.device)
    detector = None
    parser = None
    region_name_arr = np.asarray(["background", *MERGED_ORDER], dtype=object)
    region_to_idx = {name: idx for idx, name in enumerate(region_name_arr.tolist())}
    if args.with_regions:
        detector, parser = load_facer_models(args.facer_device)

    token_memmap = None
    region_memmap = None
    img_ids = []
    labels = []
    groups = []
    methods = []
    pair_ids = []
    write_offset = 0

    progress = tqdm(
        total=len(rows),
        desc=f"real-patch:{method}",
        unit="img",
        file=sys.stdout,
        dynamic_ncols=True,
    )
    with tempfile.TemporaryDirectory(prefix=f"real_patch_{method}_", dir=output_dir) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        token_path = tmp_dir / "tokens.npy"
        region_path = tmp_dir / "regions.npy"

        for batch in batch_items(rows, args.batch_size):
            image_paths = [str(args.relative_to / row["img_path"].lstrip("/")) for row in batch]
            patch_tokens = backbone.encode_patch_tokens_batch(image_paths, target_size=args.target_size, layer=args.layer)

            if token_memmap is None:
                token_memmap = np.lib.format.open_memmap(
                    token_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(rows), patch_tokens.shape[1], patch_tokens.shape[2]),
                )
            token_memmap[write_offset : write_offset + len(batch)] = patch_tokens.astype(np.float32, copy=False)

            if args.with_regions:
                from PIL import Image

                images = [Image.open(path).convert("RGB") for path in image_paths]
                patch_h = patch_w = int(round(patch_tokens.shape[1] ** 0.5))
                region_names = parse_images_to_regions(images, (patch_h, patch_w), detector, parser, args.facer_device)
                region_idx = np.stack(
                    [
                        np.asarray([region_to_idx.get(str(name), 0) for name in region_vec], dtype=np.int16)
                        for region_vec in region_names
                    ],
                    axis=0,
                )
                if region_memmap is None:
                    region_memmap = np.lib.format.open_memmap(
                        region_path,
                        mode="w+",
                        dtype=np.int16,
                        shape=(len(rows), region_idx.shape[1]),
                    )
                region_memmap[write_offset : write_offset + len(batch)] = region_idx.astype(np.int16, copy=False)

            img_ids.extend(row["img_id"] for row in batch)
            labels.extend(int(row["label"]) for row in batch)
            groups.extend(row["group"] for row in batch)
            methods.extend(row["method"] for row in batch)
            pair_ids.extend(row["pair_id"] for row in batch)
            write_offset += len(batch)
            progress.update(len(batch))
    progress.close()

    payload = {
        "tokens": np.asarray(token_memmap, dtype=np.float32),
        "img_id": np.asarray(img_ids, dtype=object),
        "label": np.asarray(labels, dtype=np.int64),
        "group": np.asarray(groups, dtype=object),
        "method": np.asarray(methods, dtype=object),
        "pair_id": np.asarray(pair_ids, dtype=object),
    }
    if args.with_regions:
        payload["region_labels"] = np.asarray(region_memmap, dtype=np.int16)
        payload["region_names"] = region_name_arr
    np.savez_compressed(cache_path, **payload)
    print(f"Saved {payload['tokens'].shape[0]} patch token tensors to {cache_path}", flush=True)


if __name__ == "__main__":
    main()
