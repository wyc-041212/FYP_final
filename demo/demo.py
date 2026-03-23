#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from demo.rearrange import build_frame_index  # noqa: E402
from demo.video_preprocess import VideoProcessSummary, iter_videos, process_videos  # noqa: E402
from main import (  # noqa: E402
    build_sample_report,
    load_main_runtime,
    load_sample_batches,
    parse_sample_args,
    predict_sample,
)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=Path, nargs="+", default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--video-suffixes", nargs="+", default=[".mp4", ".avi", ".mov", ".mkv"])
    parser.add_argument("--predictor-path", type=Path, required=True)
    parser.add_argument("--mode", default="fixed_num_frames", choices=["fixed_num_frames", "fixed_stride"])
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--scale", type=float, default=1.3)
    parser.add_argument("--disable-verify-aligned-face", action="store_true")
    parser.add_argument("--save-uncropped-frames", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "demos" / "latest")
    parser.add_argument("--video-name", default="annotated_aligned.mp4")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--font-scale", type=float, default=0.8)
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--preprocess-log", type=Path, default=None)
    demo_args, sample_argv = parser.parse_known_args(argv)
    if not demo_args.video_path and demo_args.input_dir is None:
        parser.error("One of --video-path or --input-dir is required.")
    sample_args = parse_sample_args(sample_argv)
    return demo_args, sample_args


def resolve_video_paths(args: argparse.Namespace) -> list[Path]:
    video_paths: list[Path] = []
    if args.video_path:
        video_paths.extend(path.resolve() for path in args.video_path)
    if args.input_dir is not None:
        video_paths.extend(iter_videos(args.input_dir.resolve(), args.video_suffixes))
    unique_paths = sorted({path.resolve() for path in video_paths})
    if not unique_paths:
        raise RuntimeError("No input videos were found for demo preprocessing.")
    return unique_paths


def build_per_video_sample_args(
    base_args: argparse.Namespace,
    *,
    frames_dir: Path,
    data_root: Path,
    sampled_root: Path,
) -> argparse.Namespace:
    payload = vars(base_args).copy()
    payload["sample_root"] = None
    payload["source_dir"] = [frames_dir]
    payload["data_root"] = data_root
    payload["num_folders"] = 1
    payload["max_frames_per_folder"] = 0
    payload["save_sampled_root"] = sampled_root
    return argparse.Namespace(**payload)


def draw_overlay(
    image: np.ndarray,
    *,
    fake_prob: float,
    threshold: float,
    img_label: str,
    folder_name: str,
    font_scale: float,
    thickness: int,
) -> np.ndarray:
    frame = image.copy()
    color = (0, 0, 255) if fake_prob >= threshold else (0, 200, 0)
    category = "FAKE" if fake_prob >= threshold else "REAL"
    cv2.rectangle(frame, (12, 12), (frame.shape[1] - 12, 112), color, thickness)
    cv2.putText(
        frame,
        f"{category}  prob={fake_prob:.3f}",
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"label={img_label}",
        (24, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(font_scale * 0.8, 0.5),
        color,
        max(thickness - 1, 1),
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"folder={folder_name}",
        (24, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(font_scale * 0.8, 0.5),
        color,
        max(thickness - 1, 1),
        cv2.LINE_AA,
    )
    return frame


def write_video(frames: list[np.ndarray], out_path: Path, fps: float) -> None:
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        current = frame
        if current.shape[:2] != (height, width):
            current = cv2.resize(current, (width, height), interpolation=cv2.INTER_LINEAR)
        writer.write(current)
    writer.release()


def render_video_demo(
    demo_root: Path,
    summary: VideoProcessSummary,
    sample_args: argparse.Namespace,
    runtime,
    *,
    threshold: float,
    video_name: str,
    fps: float,
    font_scale: float,
    thickness: int,
) -> dict[str, object]:
    per_video_args = build_per_video_sample_args(
        sample_args,
        frames_dir=summary.frames_dir,
        data_root=demo_root / "preprocessed",
        sampled_root=demo_root / "sampled" / summary.video_path.stem,
    )
    batch = load_sample_batches(per_video_args, runtime.region_names)[0]
    prediction = predict_sample(runtime, batch.cls_cache, batch.patch_cache)

    sample_dir = demo_root / "rendered" / batch.sample_name
    frames_dir = sample_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    row_by_path = {str(row["img_path"]): row for row in batch.rows}

    rendered_frames: list[np.ndarray] = []
    image_rows = []
    for idx, img_path_value in enumerate(batch.cls_cache.img_id.astype(object).tolist()):
        img_path = Path(str(img_path_value))
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        row = row_by_path.get(str(img_path), {})
        fake_prob = float(prediction.route_meta_prob[idx])
        frame = draw_overlay(
            image,
            fake_prob=fake_prob,
            threshold=threshold,
            img_label=str(row.get("method", batch.cls_cache.method[idx])),
            folder_name=str(row.get("pair_id", batch.cls_cache.pair_id[idx])),
            font_scale=font_scale,
            thickness=thickness,
        )
        out_name = f"{idx:04d}_{img_path.name}"
        cv2.imwrite(str(frames_dir / out_name), frame)
        rendered_frames.append(frame)
        image_rows.append(
            {
                "img_path": str(img_path),
                "pair_id": str(batch.cls_cache.pair_id[idx]),
                "group": str(batch.cls_cache.group[idx]),
                "method": str(batch.cls_cache.method[idx]),
                "route_meta_prob": fake_prob,
                "pred": int(prediction.pred[idx]),
                "route_top1": str(prediction.route_top1[idx]),
            }
        )

    video_path = sample_dir / video_name
    write_video(rendered_frames, video_path, fps)
    sample_report = build_sample_report(
        batch.sample_name,
        batch.cls_cache,
        prediction,
        threshold,
    )
    sample_report["images"] = image_rows
    sample_report["source_video"] = str(summary.video_path)
    sample_report["sampled_indices"] = summary.sampled_indices
    sample_report["saved_count"] = summary.saved_count
    sample_report["failed_count"] = summary.failed_count
    report_path = sample_dir / "report.json"
    report_path.write_text(json.dumps(sample_report, indent=2))
    return {
        "sample_name": batch.sample_name,
        "source_video": str(summary.video_path),
        "processed_frames_dir": str(summary.frames_dir),
        "landmarks_dir": str(summary.landmarks_dir),
        "report_json": str(report_path),
        "video_path": str(video_path),
        "num_rendered_frames": len(rendered_frames),
        "mean_fake_prob": sample_report["mean_fake_prob"],
    }


def main(argv: list[str] | None = None) -> None:
    demo_args, sample_args = parse_args(argv)
    demo_root = demo_args.output_dir.resolve()
    demo_root.mkdir(parents=True, exist_ok=True)

    video_paths = resolve_video_paths(demo_args)
    preprocess_root = demo_root / "preprocessed" / "DEMO" / "ADHOC"
    summaries = process_videos(
        video_paths,
        output_root=preprocess_root,
        predictor_path=demo_args.predictor_path.resolve(),
        mode=demo_args.mode,
        num_frames=demo_args.num_frames,
        stride=demo_args.stride,
        resolution=demo_args.resolution,
        scale=demo_args.scale,
        verify_aligned_face=not demo_args.disable_verify_aligned_face,
        save_uncropped_frames=demo_args.save_uncropped_frames,
        log_path=demo_args.preprocess_log.resolve() if demo_args.preprocess_log is not None else None,
    )

    build_frame_index(
        preprocess_root / "frames",
        demo_root / "frame_index.json",
        landmarks_root=preprocess_root / "landmarks",
        label=str(sample_args.label),
    )

    runtime = load_main_runtime(
        upstream_checkpoint=sample_args.upstream_checkpoint,
        patch_branch=sample_args.patch_branch,
        pair_branch=sample_args.pair_branch,
        route_meta_head=sample_args.route_meta_head,
        route_batch_size=sample_args.route_batch_size,
        device_name=sample_args.runtime_device,
    )
    threshold = runtime.route_meta_threshold if demo_args.threshold is None else float(demo_args.threshold)

    summary_report: dict[str, object] = {
        "pipeline_root": str(PROJECT_ROOT),
        "threshold": threshold,
        "num_videos": len(summaries),
        "samples": {},
    }
    for summary in summaries:
        sample_summary = render_video_demo(
            demo_root,
            summary,
            sample_args,
            runtime,
            threshold=threshold,
            video_name=demo_args.video_name,
            fps=demo_args.fps,
            font_scale=demo_args.font_scale,
            thickness=demo_args.thickness,
        )
        summary_report["samples"][summary.video_path.stem] = sample_summary

    summary_path = demo_root / "demo_summary.json"
    summary_path.write_text(json.dumps(summary_report, indent=2))
    print(json.dumps(summary_report, indent=2))
    print(f"Saved demo outputs to {demo_root}")


if __name__ == "__main__":
    main()
