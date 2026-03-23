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


def sample_frame_indices(
    frame_count: int,
    mode: str = "fixed_num_frames",
    num_frames: int = 32,
    stride: int = 1,
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

    raise ValueError(f"Unsupported mode: {mode}")


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
    sampled_indices = sample_frame_indices(
        frame_count=frame_count,
        mode=mode,
        num_frames=num_frames,
        stride=stride,
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
