from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class FramePrediction:
    frame_index: int
    fake_prob: float
    pred_label: str


def crop_to_vertical_canvas(
    image: np.ndarray,
    *,
    aspect_ratio: tuple[int, int] = (9, 16),
) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    target_h = int(round(width * aspect_ratio[1] / aspect_ratio[0]))

    if height <= target_h:
        return image.copy(), (0, 0)

    extra_h = height - target_h
    crop_top = int(round(extra_h * 0.3))
    crop_bottom = crop_top + target_h
    return image[crop_top:crop_bottom, :, :].copy(), (0, crop_top)


def build_sparse_predictions(
    *,
    img_paths: np.ndarray,
    fake_probs: np.ndarray,
    threshold: float,
) -> dict[int, FramePrediction]:
    sparse: dict[int, FramePrediction] = {}
    for img_path_value, fake_prob_value in zip(img_paths.astype(object).tolist(), fake_probs.tolist()):
        frame_index = int(Path(str(img_path_value)).stem)
        fake_prob = float(fake_prob_value)
        sparse[frame_index] = FramePrediction(
            frame_index=frame_index,
            fake_prob=fake_prob,
            pred_label="FAKE" if fake_prob >= threshold else "REAL",
        )
    return sparse


def propagate_predictions(
    *,
    frame_count: int,
    sparse_predictions: dict[int, FramePrediction],
) -> list[FramePrediction | None]:
    if frame_count < 0:
        raise ValueError("frame_count must be >= 0")
    if frame_count == 0:
        return []
    if not sparse_predictions:
        return [None] * frame_count

    sorted_indices = sorted(sparse_predictions)
    dense: list[FramePrediction | None] = [None] * frame_count
    current = sparse_predictions[sorted_indices[0]]
    next_pos = 1

    for frame_index in range(frame_count):
        while next_pos < len(sorted_indices) and sorted_indices[next_pos] <= frame_index:
            current = sparse_predictions[sorted_indices[next_pos]]
            next_pos += 1
        dense[frame_index] = current
    return dense


def draw_prediction_overlay(
    image: np.ndarray,
    *,
    prediction: FramePrediction | None,
    face_box: tuple[int, int, int, int] | None,
    font_scale: float,
    thickness: int,
) -> np.ndarray:
    frame = image.copy()
    if prediction is None or face_box is None:
        return frame

    color = (0, 0, 255) if prediction.pred_label == "FAKE" else (0, 200, 0)
    left, top, right, bottom = face_box
    cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
    pred_anchor = (left, max(top - 10, 24))
    prob_anchor = (left, min(bottom + 24, frame.shape[0] - 12))

    cv2.putText(
        frame,
        prediction.pred_label,
        pred_anchor,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{prediction.fake_prob:.4f}",
        prob_anchor,
        cv2.FONT_HERSHEY_SIMPLEX,
        max(font_scale * 0.9, 0.5),
        color,
        max(thickness - 1, 1),
        cv2.LINE_AA,
    )
    return frame
