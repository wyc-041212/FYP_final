#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

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
class CdfMethodSpec:
    split: str
    group: str
    method: str
    method_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fake-only CLIP CLS cache for DF40_test_cdf.")
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--relative-to", type=Path, default=Path(os.environ.get("RELATIVE_TO", PACKAGE_ROOT)).expanduser())
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("MODEL_DIR", Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--clip-vit-large-patch14")).expanduser(),
    )
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--limit-fake", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def infer_spec(method_root: Path, relative_to: Path) -> CdfMethodSpec:
    parts = method_root.relative_to(relative_to).parts
    if len(parts) < 3:
        raise ValueError(f"Expected <split>/<group>/<method>, got {method_root}")
    return CdfMethodSpec(split=parts[0], group=parts[1], method=parts[2], method_root=method_root)


def iter_image_files(root: Path) -> list[Path]:
    return [
        p for p in root.rglob("*")
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS
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


def discover_fake_leaf_dirs(method_root: Path, group: str) -> list[Path]:
    if group in {"FS", "FR"} and (method_root / "frames").exists():
        search_root = method_root / "frames"
    else:
        search_root = method_root

    leaf_dirs: list[Path] = []
    for d in search_root.rglob("*"):
        if not d.is_dir():
            continue
        names = [f.name for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS and not f.name.startswith(".")]
        if names:
            leaf_dirs.append(d)
    if not leaf_dirs and any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in search_root.iterdir()):
        leaf_dirs.append(search_root)
    return sorted(set(leaf_dirs))


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return False


def to_portable_path(path: Path, relative_to: Path) -> str:
    return f"/{path.relative_to(relative_to).as_posix()}"


def build_fake_rows(
    method_root: Path,
    relative_to: Path,
    group: str,
    method: str,
    limit_fake: int,
    seed: int,
    is_valid_image,
    to_portable_path,
) -> list[dict]:
    leaf_dirs = discover_fake_leaf_dirs(method_root, group)
    rows: list[dict] = []
    for leaf_dir in leaf_dirs:
        image_paths = sorted(iter_image_files(leaf_dir), key=natural_sort_key)
        for image_path in image_paths:
            if not is_valid_image(image_path):
                continue
            pair_key = leaf_dir.relative_to(method_root).as_posix() or leaf_dir.name
            rows.append(
                {
                    "img_id": to_portable_path(image_path, relative_to),
                    "img_path": to_portable_path(image_path, relative_to),
                    "label": 1,
                    "group": group,
                    "method": method,
                    "pair_id": f"{method}:{pair_key}",
                }
            )
    if limit_fake > 0 and len(rows) > limit_fake:
        rng = np.random.default_rng(seed)
        keep = sorted(rng.choice(len(rows), size=limit_fake, replace=False).tolist())
        rows = [rows[i] for i in keep]
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

    spec = infer_spec(args.method_root, args.relative_to)
    if spec.split != "DF40_test_cdf":
        raise ValueError(f"Expected split DF40_test_cdf, got {spec.split}")

    output_dir = args.cache_root / spec.split / spec.group
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"manifest_{spec.method}.csv"
    cache_path = output_dir / f"cls_{spec.method}.npz"

    rows = build_fake_rows(
        args.method_root,
        args.relative_to,
        spec.group,
        spec.method,
        args.limit_fake,
        args.seed,
        is_valid_image,
        to_portable_path,
    )
    write_manifest(rows, manifest_path)
    print(
        f"[START] prepare_cdf_cls_clip\n"
        f"  split={spec.split}\n"
        f"  group={spec.group}\n"
        f"  method={spec.method}\n"
        f"  method_root={args.method_root}\n"
        f"  rows={len(rows)}\n"
        f"  manifest={manifest_path}\n"
        f"  cache={cache_path}",
        flush=True,
    )
    if not rows:
        print(f"[WARN] No rows found for {spec.group}/{spec.method}")
        return

    backbone = FrozenBackbone(backbone_name="clip", model_dir=args.model_dir, device=args.device)
    cls_chunks = []
    img_ids: list[str] = []
    labels: list[int] = []
    groups: list[str] = []
    methods: list[str] = []
    pair_ids: list[str] = []

    progress = tqdm(total=len(rows), desc=f"cdf-cls:{spec.group}/{spec.method}", unit="img", file=sys.stdout, dynamic_ncols=True)
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
