#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, UnidentifiedImageError
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
    from prepare.cache import build_cache
    from prepare.backbone import FrozenBackbone

    return build_cache, FrozenBackbone


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class RealDatasetSpec:
    split: str
    group: str
    method: str
    dataset_root: Path
    frames_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLIP CLS cache for Celeb-DF-v2 real datasets.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--relative-to", type=Path, default=Path(os.environ.get("RELATIVE_TO", PACKAGE_ROOT)).expanduser())
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
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    return parser.parse_args()


def infer_spec(dataset_root: Path, relative_to: Path) -> RealDatasetSpec:
    parts = dataset_root.relative_to(relative_to).parts
    if len(parts) < 2:
        raise ValueError(f"Expected <split>/<dataset>, got {dataset_root}")
    frames_root = dataset_root / "frames" if (dataset_root / "frames").exists() else dataset_root
    return RealDatasetSpec(
        split=parts[0],
        group="real",
        method=parts[1],
        dataset_root=dataset_root,
        frames_root=frames_root,
    )


def iter_image_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMAGE_EXTS
    ]


def natural_sort_key(path: Path) -> tuple:
    stem = path.stem
    digits = []
    current = ""
    for ch in stem:
        if ch.isdigit():
            current += ch
        elif current:
            digits.append(int(current))
            current = ""
    if current:
        digits.append(int(current))
    return tuple(digits) if digits else (stem,)


def discover_video_dirs(frames_root: Path) -> list[Path]:
    video_dirs: list[Path] = []
    for path in frames_root.rglob("*"):
        if not path.is_dir():
            continue
        names = [
            child.name
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_EXTS and not child.name.startswith(".")
        ]
        if names:
            video_dirs.append(path)
    if not video_dirs and frames_root.exists():
        if any(path.is_file() and path.suffix.lower() in IMAGE_EXTS for path in frames_root.iterdir()):
            video_dirs.append(frames_root)
    return sorted(set(video_dirs))


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return False


def to_portable_path(path: Path, relative_to: Path) -> str:
    return f"/{path.relative_to(relative_to).as_posix()}"


def build_real_rows(spec: RealDatasetSpec, relative_to: Path) -> list[dict]:
    rows: list[dict] = []
    for video_dir in discover_video_dirs(spec.frames_root):
        image_paths = sorted(iter_image_files(video_dir), key=natural_sort_key)
        for image_path in image_paths:
            if not is_valid_image(image_path):
                continue
            pair_key = video_dir.relative_to(spec.frames_root).as_posix() or video_dir.name
            rows.append(
                {
                    "img_id": to_portable_path(image_path, relative_to),
                    "img_path": to_portable_path(image_path, relative_to),
                    "label": 0,
                    "group": spec.group,
                    "method": spec.method,
                    "pair_id": f"{spec.method}:{pair_key}",
                }
            )
    return rows


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["img_id", "img_path", "label", "group", "method", "pair_id"])
        writer.writeheader()
        writer.writerows(rows)


def batch_items(items: list[dict], batch_size: int) -> Iterator[list[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    args = parse_args()
    build_cache, FrozenBackbone = load_src_modules(args.src_root)

    spec = infer_spec(args.dataset_root, args.relative_to)
    output_dir = args.cache_root / spec.split / spec.group
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"manifest_{spec.method}.csv"
    cache_path = output_dir / f"cls_{spec.method}.npz"

    rows = build_real_rows(spec, args.relative_to)
    write_manifest(rows, manifest_path)
    print(
        f"[START] prepare_celebdf_real_cls_clip\n"
        f"  split={spec.split}\n"
        f"  group={spec.group}\n"
        f"  method={spec.method}\n"
        f"  dataset_root={args.dataset_root}\n"
        f"  rows={len(rows)}\n"
        f"  manifest={manifest_path}\n"
        f"  cache={cache_path}",
        flush=True,
    )
    if not rows:
        print(f"[WARN] No rows found for {spec.method}")
        return

    backbone = FrozenBackbone(backbone_name="clip", model_dir=args.model_dir, device=args.device)
    cls_chunks = []
    img_ids: list[str] = []
    labels: list[int] = []
    groups: list[str] = []
    methods: list[str] = []
    pair_ids: list[str] = []

    progress = tqdm(
        total=len(rows),
        desc=f"real-cls:{spec.method}",
        unit="img",
        file=sys.stdout,
        dynamic_ncols=True,
    )
    for batch in batch_items(rows, args.batch_size):
        image_paths = [str(args.relative_to / row["img_path"].lstrip("/")) for row in batch]
        encoded = backbone.encode_batch(image_paths, target_size=args.target_size, layers=[-1])
        cls_chunks.append(encoded[-1])
        img_ids.extend(row["img_id"] for row in batch)
        labels.extend(int(row["label"]) for row in batch)
        groups.extend(row["group"] for row in batch)
        methods.extend(row["method"] for row in batch)
        pair_ids.extend(row["pair_id"] for row in batch)
        progress.update(len(batch))
    progress.close()

    cls_all = np.concatenate(cls_chunks, axis=0).astype(np.float32, copy=False)
    cache = build_cache(cls_all, img_ids, labels, groups, methods, pair_ids)
    cache.to_npz(cache_path)
    print(f"Saved {cls_all.shape[0]} embeddings to {cache_path}", flush=True)


if __name__ == "__main__":
    main()
