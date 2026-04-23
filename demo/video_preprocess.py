from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from demo.alignment import build_face_detector, build_shape_predictor, extract_aligned_face_dlib
from demo.logging_utils import create_logger


@dataclass(slots=True)
class VideoProcessSummary:
    video_path: Path
    frame_count: int
    sampled_indices: list[int]
    saved_count: int
    failed_count: int
    frames_dir: Path
    landmarks_dir: Path
    raw_frames_dir: Path | None


def build_uncertainty_refined_indices(
    *,
    frame_count: int,
    sampled_indices: list[int],
    fake_probs: np.ndarray,
    target_num_frames: int,
    threshold: float,
) -> list[int]:
    if frame_count <= 0:
        return []

    sampled = sorted({int(index) for index in sampled_indices if 0 <= int(index) < frame_count})
    if not sampled:
        return sample_frame_indices(frame_count=frame_count, mode="fixed_num_frames", num_frames=target_num_frames)
    if target_num_frames <= len(sampled):
        return sampled
    if len(fake_probs) != len(sampled):
        raise ValueError("fake_probs length must match sampled_indices length.")

    fake_probs = np.asarray(fake_probs, dtype=np.float32)
    scale = max(0.05, min(float(threshold), float(1.0 - threshold)) * 0.5)
    candidates: list[tuple[float, int]] = []

    for pos, (left_idx, right_idx) in enumerate(zip(sampled[:-1], sampled[1:])):
        gap = right_idx - left_idx
        if gap <= 1:
            continue
        left_prob = float(fake_probs[pos])
        right_prob = float(fake_probs[pos + 1])
        jump = abs(right_prob - left_prob)
        flip_bonus = 1.0 if (left_prob >= threshold) != (right_prob >= threshold) else 0.0
        for frame_index in range(left_idx + 1, right_idx):
            alpha = (frame_index - left_idx) / gap
            interp_prob = (1.0 - alpha) * left_prob + alpha * right_prob
            uncertainty = float(np.exp(-abs(interp_prob - threshold) / scale))
            score = (2.0 * flip_bonus) + jump + uncertainty
            candidates.append((score, frame_index))

    if not candidates:
        return sampled

    candidates.sort(key=lambda item: (-item[0], item[1]))
    refined = set(sampled)
    for _, frame_index in candidates:
        refined.add(frame_index)
        if len(refined) >= min(target_num_frames, frame_count):
            break
    return sorted(refined)


def sample_frame_indices(
    frame_count: int,
    mode: str = "fixed_num_frames",
    num_frames: int = 32,
    stride: int = 1,
    change_scores: np.ndarray | None = None,
) -> list[int]:
    if frame_count <= 0:
        return []

    if mode == "fixed_num_frames":
        if num_frames <= 0:
            raise ValueError("num_frames must be > 0 for fixed_num_frames mode.")
        return np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=int).tolist()

    if mode == "fixed_stride":
        if stride <= 0:
            raise ValueError("stride must be > 0 for fixed_stride mode.")
        return np.arange(0, frame_count, stride, dtype=int).tolist()

    if mode == "content_change":
        if num_frames <= 0:
            raise ValueError("num_frames must be > 0 for content_change mode.")
        if frame_count == 1:
            return [0]
        if change_scores is None:
            raise ValueError("change_scores are required for content_change mode.")
        if len(change_scores) != frame_count - 1:
            raise ValueError("change_scores length must equal frame_count - 1.")

        change_scores = np.maximum(np.asarray(change_scores, dtype=np.float32), 0.0)
        positive_scores = change_scores[change_scores > 0]
        if positive_scores.size == 0:
            return np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=int).tolist()

        # Keep a time baseline so static regions still receive coverage while
        # allocating more samples to segments with larger visual changes.
        effective_steps = 1.0 + (change_scores / float(np.mean(positive_scores)))
        cumulative = np.concatenate([[0.0], np.cumsum(effective_steps, dtype=np.float64)])
        target_positions = np.linspace(0.0, cumulative[-1], num_frames, endpoint=True)
        return np.searchsorted(cumulative, target_positions, side="left").astype(int).tolist()

    raise ValueError(f"Unsupported mode: {mode}")


def compute_frame_change_scores(
    video_path: str | Path,
    *,
    resize_to: tuple[int, int] = (64, 64),
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for change scoring: {video_path}")

    previous: np.ndarray | None = None
    scores: list[float] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if resize_to is not None:
                gray = cv2.resize(gray, resize_to, interpolation=cv2.INTER_AREA)
            current = gray.astype(np.float32)
            if previous is not None:
                scores.append(float(np.mean(np.abs(current - previous))))
            previous = current
    finally:
        cap.release()

    return np.asarray(scores, dtype=np.float32)


def process_video(
    video_path: str | Path,
    output_root: str | Path,
    predictor_path: str | Path,
    mode: str = "fixed_num_frames",
    num_frames: int = 32,
    stride: int = 1,
    resolution: int = 256,
    scale: float = 1.3,
    verify_aligned_face: bool = True,
    save_uncropped_frames: bool = False,
    log_path: str | Path | None = None,
    sampled_indices_override: list[int] | None = None,
) -> VideoProcessSummary:
    video_path = Path(video_path)
    output_root = Path(output_root)
    frames_dir = output_root / "frames" / video_path.stem
    landmarks_dir = output_root / "landmarks" / video_path.stem
    raw_frames_dir = output_root / "frames_wocropface" / video_path.stem if save_uncropped_frames else None

    frames_dir.mkdir(parents=True, exist_ok=True)
    landmarks_dir.mkdir(parents=True, exist_ok=True)
    if raw_frames_dir is not None:
        raw_frames_dir.mkdir(parents=True, exist_ok=True)

    logger = create_logger(log_path)
    face_detector = build_face_detector()
    predictor = build_shape_predictor(predictor_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_indices = (
        sorted({int(index) for index in sampled_indices_override if 0 <= int(index) < frame_count})
        if sampled_indices_override is not None
        else sample_frame_indices(
            frame_count=frame_count,
            mode=mode,
            num_frames=num_frames,
            stride=stride,
            change_scores=compute_frame_change_scores(video_path) if mode == "content_change" else None,
        )
    )
    sampled_index_set = set(sampled_indices)
    saved_count = 0
    failed_count = 0

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame %s of %s", frame_idx, video_path)
            break

        if raw_frames_dir is not None and frame_idx in sampled_index_set:
            cv2.imwrite(str(raw_frames_dir / f"{frame_idx:03d}.png"), frame)

        if frame_idx not in sampled_index_set:
            continue

        result = extract_aligned_face_dlib(
            face_detector=face_detector,
            predictor=predictor,
            image=frame,
            res=resolution,
            scale=scale,
            verify_aligned_face=verify_aligned_face,
        )
        if result is None or (result.landmarks is None and verify_aligned_face):
            failed_count += 1
            logger.info("Skipped frame %s of %s because face alignment failed.", frame_idx, video_path)
            continue

        cv2.imwrite(str(frames_dir / f"{frame_idx:03d}.png"), result.image)
        if result.landmarks is not None:
            np.save(str(landmarks_dir / f"{frame_idx:03d}.npy"), result.landmarks)
        saved_count += 1

    cap.release()
    return VideoProcessSummary(
        video_path=video_path,
        frame_count=frame_count,
        sampled_indices=sampled_indices,
        saved_count=saved_count,
        failed_count=failed_count,
        frames_dir=frames_dir,
        landmarks_dir=landmarks_dir,
        raw_frames_dir=raw_frames_dir,
    )


def process_video_raw(
    video_path: str | Path,
    output_root: str | Path,
    mode: str = "fixed_num_frames",
    num_frames: int = 32,
    stride: int = 1,
    log_path: str | Path | None = None,
    sampled_indices_override: list[int] | None = None,
) -> VideoProcessSummary:
    video_path = Path(video_path)
    output_root = Path(output_root)
    frames_dir = output_root / "frames" / video_path.stem
    landmarks_dir = output_root / "landmarks" / video_path.stem

    frames_dir.mkdir(parents=True, exist_ok=True)
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    logger = create_logger(log_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_indices = (
        sorted({int(index) for index in sampled_indices_override if 0 <= int(index) < frame_count})
        if sampled_indices_override is not None
        else sample_frame_indices(
            frame_count=frame_count,
            mode=mode,
            num_frames=num_frames,
            stride=stride,
            change_scores=compute_frame_change_scores(video_path) if mode == "content_change" else None,
        )
    )
    sampled_index_set = set(sampled_indices)
    saved_count = 0

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame %s of %s", frame_idx, video_path)
            break
        if frame_idx not in sampled_index_set:
            continue
        cv2.imwrite(str(frames_dir / f"{frame_idx:06d}.png"), frame)
        saved_count += 1

    cap.release()
    return VideoProcessSummary(
        video_path=video_path,
        frame_count=frame_count,
        sampled_indices=sampled_indices,
        saved_count=saved_count,
        failed_count=0,
        frames_dir=frames_dir,
        landmarks_dir=landmarks_dir,
        raw_frames_dir=frames_dir,
    )


def iter_videos(input_dir: str | Path, suffixes: Iterable[str]) -> list[Path]:
    input_dir = Path(input_dir)
    suffix_set = {suffix.lower() for suffix in suffixes}
    videos: list[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffix_set:
            videos.append(path)
    return videos


def process_videos(
    video_paths: list[Path],
    *,
    output_root: Path,
    predictor_path: Path,
    mode: str,
    num_frames: int,
    stride: int,
    resolution: int,
    scale: float,
    verify_aligned_face: bool,
    save_uncropped_frames: bool,
    log_path: Path | None,
) -> list[VideoProcessSummary]:
    summaries: list[VideoProcessSummary] = []
    for video_path in tqdm(video_paths, desc="demo-preprocess", unit="video"):
        summaries.append(
            process_video(
                video_path=video_path,
                output_root=output_root,
                predictor_path=predictor_path,
                mode=mode,
                num_frames=num_frames,
                stride=stride,
                resolution=resolution,
                scale=scale,
                verify_aligned_face=verify_aligned_face,
                save_uncropped_frames=save_uncropped_frames,
                log_path=log_path,
            )
        )
    return summaries


def process_videos_raw(
    video_paths: list[Path],
    *,
    output_root: Path,
    mode: str,
    num_frames: int,
    stride: int,
    log_path: Path | None,
) -> list[VideoProcessSummary]:
    summaries: list[VideoProcessSummary] = []
    for video_path in tqdm(video_paths, desc="demo-preprocess-raw", unit="video"):
        summaries.append(
            process_video_raw(
                video_path=video_path,
                output_root=output_root,
                mode=mode,
                num_frames=num_frames,
                stride=stride,
                log_path=log_path,
            )
        )
    return summaries
