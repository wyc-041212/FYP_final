from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from prepare.backbone import FrozenBackbone

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def discover_video_folders(source_dir: Path) -> list[Path]:
    folders: list[Path] = []
    if list_image_files(source_dir):
        folders.append(source_dir)
    for path in sorted(source_dir.rglob("*")):
        if not path.is_dir():
            continue
        if list_image_files(path):
            folders.append(path)
    return folders


def choose_frames(image_paths: list[Path], max_frames: int) -> list[Path]:
    if max_frames <= 0 or len(image_paths) <= max_frames:
        return image_paths
    idx = np.linspace(0, len(image_paths) - 1, num=max_frames)
    idx = np.unique(np.round(idx).astype(np.int64))
    return [image_paths[int(i)] for i in idx.tolist()]


def infer_metadata(img_path: Path, data_root: Path, folder_path: Path, label: int) -> dict:
    rel = img_path.relative_to(data_root)
    parts = rel.parts
    if len(parts) < 3:
        raise RuntimeError(f"Path is too shallow to infer group/method: {img_path}")
    group = parts[0]
    method = parts[1]
    folder_key = str(folder_path.relative_to(data_root))
    return {
        "img_path": str(img_path),
        "label": int(label),
        "group": group,
        "method": method,
        "pair_id": folder_key,
        "video_folder": folder_key,
    }


def sample_rows_from_sources(
    source_dirs: list[Path],
    *,
    data_root: Path,
    num_folders: int,
    max_frames_per_folder: int,
    seed: int,
    label: int,
) -> tuple[list[Path], list[dict]]:
    rng = np.random.default_rng(seed)
    discovered: list[Path] = []
    for source_dir in source_dirs:
        folders = discover_video_folders(source_dir)
        if not folders:
            raise RuntimeError(f"No image folders found under {source_dir}")
        discovered.extend(folders)
    discovered = sorted(set(discovered))
    order = rng.permutation(len(discovered))
    chosen_folders = [discovered[int(idx)] for idx in order[: min(num_folders, len(discovered))]]

    rows: list[dict] = []
    for folder in chosen_folders:
        image_paths = choose_frames(list_image_files(folder), max_frames_per_folder)
        rows.extend(
            infer_metadata(img_path, data_root, folder, label)
            for img_path in image_paths
        )
    if not rows:
        raise RuntimeError("No sampled images were collected.")
    return chosen_folders, rows


def build_backbones(
    *,
    clip_model_dir: Path,
    patch_model_dir: Path,
    patch_backbone: str,
    cls_device: str,
    patch_device: str,
) -> tuple[FrozenBackbone, FrozenBackbone]:
    cls_backbone = FrozenBackbone(
        backbone_name="clip",
        model_dir=clip_model_dir,
        device=cls_device,
    )
    same_model = patch_backbone.lower() == "clip" and clip_model_dir.resolve() == patch_model_dir.resolve()
    same_device = cls_device == patch_device
    if same_model and same_device:
        return cls_backbone, cls_backbone
    patch_backbone_obj = FrozenBackbone(
        backbone_name=patch_backbone,
        model_dir=patch_model_dir,
        device=patch_device,
    )
    return cls_backbone, patch_backbone_obj


def write_manifest(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["img_path", "label", "group", "method", "pair_id", "video_folder"]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_sample_summary(
    selected_folders: list[Path],
    rows: list[dict],
    data_root: Path,
) -> dict:
    return {
        "num_folders": len(selected_folders),
        "num_images": len(rows),
        "folders": [str(path.relative_to(data_root)) for path in selected_folders],
    }


def write_sample_summary(
    selected_folders: list[Path],
    rows: list[dict],
    data_root: Path,
    out_dir: Path,
) -> None:
    summary = build_sample_summary(selected_folders, rows, data_root)
    (out_dir / "sample_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "sampled_folders.txt").write_text("\n".join(summary["folders"]) + "\n")
