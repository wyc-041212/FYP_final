from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from utils.device import resolve_device_name

LAPA_LABELS = [
    "background",
    "face",
    "rb",
    "lb",
    "re",
    "le",
    "nose",
    "ulip",
    "imouth",
    "llip",
    "hair",
]

MERGE_MAP = {
    "background": "background",
    "face": "skin",
    "rb": "eyebrow",
    "lb": "eyebrow",
    "re": "eye",
    "le": "eye",
    "nose": "nose",
    "ulip": "lip",
    "imouth": "mouth",
    "llip": "lip",
    "hair": "hair",
}

MERGED_ORDER = ["skin", "eye", "eyebrow", "nose", "lip", "mouth", "hair"]

MERGED_COLORS = {
    "background": "#888888",
    "skin": "#FFD8A8",
    "eyebrow": "#8B4513",
    "eye": "#1E90FF",
    "nose": "#FFA500",
    "lip": "#DC143C",
    "mouth": "#B22222",
    "hair": "#4B0082",
}

LAPA_RGB = {
    "background": (0, 0, 0),
    "face": (255, 224, 189),
    "rb": (139, 69, 19),
    "lb": (160, 82, 45),
    "re": (0, 128, 255),
    "le": (30, 144, 255),
    "nose": (255, 165, 0),
    "ulip": (220, 20, 60),
    "imouth": (178, 34, 34),
    "llip": (255, 69, 0),
    "hair": (75, 0, 130),
}


def seg_to_patch_labels(seg_pred: np.ndarray, patch_h: int, patch_w: int) -> np.ndarray:
    height, width = seg_pred.shape
    block_h = height / patch_h
    block_w = width / patch_w
    patch_labels = np.zeros((patch_h, patch_w), dtype=np.int64)
    for row in range(patch_h):
        row_start = int(round(row * block_h))
        row_end = int(round((row + 1) * block_h))
        for col in range(patch_w):
            col_start = int(round(col * block_w))
            col_end = int(round((col + 1) * block_w))
            block = seg_pred[row_start:row_end, col_start:col_end]
            if block.size == 0:
                continue
            patch_labels[row, col] = np.bincount(block.ravel(), minlength=len(LAPA_LABELS)).argmax()
    return patch_labels


def patch_labels_to_regions(patch_labels: np.ndarray) -> np.ndarray:
    merged = np.empty(patch_labels.size, dtype=object)
    for idx, class_idx in enumerate(patch_labels.ravel()):
        label = LAPA_LABELS[class_idx] if int(class_idx) < len(LAPA_LABELS) else "background"
        merged[idx] = MERGE_MAP.get(label, "other")
    return merged


def load_facer_models(device: str, parser_model: str = "farl/lapa/448") -> tuple[object, object]:
    import facer

    resolved = resolve_device_name(device)
    detector = facer.face_detector("retinaface/mobilenet", device=resolved)
    parser = facer.face_parser(parser_model, device=resolved)
    return detector, parser


def parse_pil_image(
    image: Image.Image,
    detector: object,
    parser: object,
    device: str,
) -> np.ndarray:
    image_np = np.array(image.convert("RGB"))
    resolved = resolve_device_name(device)
    image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(resolved)
    try:
        with torch.inference_mode():
            faces = detector(image_t)
        if faces["rects"].shape[0] == 0:
            return np.zeros(image_np.shape[:2], dtype=np.int64)
        with torch.inference_mode():
            faces = parser(image_t, faces)
        return faces["seg"]["logits"][0].argmax(dim=0).cpu().numpy()
    except Exception:
        return np.zeros(image_np.shape[:2], dtype=np.int64)


def parse_images_to_regions(
    images: Sequence[Image.Image],
    patch_hw: tuple[int, int],
    detector: object,
    parser: object,
    device: str,
) -> List[np.ndarray]:
    patch_h, patch_w = patch_hw
    regions: List[np.ndarray] = []
    for image in images:
        seg_pred = parse_pil_image(image, detector, parser, device)
        patch_labels = seg_to_patch_labels(seg_pred, patch_h, patch_w)
        regions.append(patch_labels_to_regions(patch_labels))
    return regions


def build_overlay(image: Image.Image, seg_pred: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    original = np.array(image.convert("RGB"), dtype=np.float32)
    overlay = original.copy()
    for class_idx, label in enumerate(LAPA_LABELS):
        if label == "background":
            continue
        mask = seg_pred == class_idx
        if not mask.any():
            continue
        color = np.array(LAPA_RGB.get(label, (200, 200, 200)), dtype=np.float32)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)
