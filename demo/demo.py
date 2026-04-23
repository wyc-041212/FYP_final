#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from demo.alignment import build_face_detector, detect_primary_face_box  # noqa: E402
from demo.annotate import FramePrediction, build_sparse_predictions, crop_to_vertical_canvas, draw_prediction_overlay, propagate_predictions  # noqa: E402
from demo.rearrange import build_frame_index  # noqa: E402
from demo.video_preprocess import (  # noqa: E402
    VideoProcessSummary,
    build_uncertainty_refined_indices,
    iter_videos,
    process_video_raw,
    process_videos,
    process_videos_raw,
)
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
    parser.add_argument("--predictor-path", type=Path, default=None)
    parser.add_argument("--frame-scores-json", type=Path, default=None)
    parser.add_argument(
        "--mode",
        default="fixed_num_frames",
        choices=["fixed_num_frames", "fixed_stride", "content_change", "uncertainty_refine"],
    )
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--coarse-num-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--scale", type=float, default=1.3)
    parser.add_argument("--disable-verify-aligned-face", action="store_true")
    parser.add_argument("--save-uncropped-frames", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "demos" / "latest")
    parser.add_argument("--video-name", default="annotated_source.mp4")
    parser.add_argument("--fps", type=float, default=None)
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


def predict_video_summary(
    *,
    data_root: Path,
    sampled_root: Path,
    summary: VideoProcessSummary,
    sample_args: argparse.Namespace,
    runtime,
    threshold: float,
):
    per_video_args = build_per_video_sample_args(
        sample_args,
        frames_dir=summary.frames_dir,
        data_root=data_root,
        sampled_root=sampled_root,
    )
    batch = load_sample_batches(per_video_args, runtime.region_names)[0]
    prediction = predict_sample(runtime, batch.cls_cache, batch.patch_cache)
    row_by_path = {str(row["img_path"]): row for row in batch.rows}
    sparse_predictions = build_sparse_predictions(
        img_paths=batch.cls_cache.img_id,
        fake_probs=prediction.route_meta_prob,
        threshold=threshold,
    )
    image_rows = []
    for idx, img_path_value in enumerate(batch.cls_cache.img_id.astype(object).tolist()):
        img_path = Path(str(img_path_value))
        sampled_prediction = sparse_predictions[int(img_path.stem)]
        image_rows.append(
            {
                "img_path": str(img_path),
                "frame_index": int(img_path.stem),
                "pair_id": str(batch.cls_cache.pair_id[idx]),
                "group": str(batch.cls_cache.group[idx]),
                "method": str(batch.cls_cache.method[idx]),
                "route_meta_prob": float(prediction.route_meta_prob[idx]),
                "pred": int(prediction.pred[idx]),
                "pred_label": sampled_prediction.pred_label,
                "route_top1": str(prediction.route_top1[idx]),
            }
        )
    return batch, prediction, sparse_predictions, image_rows


def resolve_render_fps(video_path: Path, requested_fps: float | None) -> float:
    if requested_fps is not None and requested_fps > 0:
        return float(requested_fps)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for fps lookup: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps if fps > 0 else 25.0


def read_video_metadata(video_path: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return frame_count, width, height, fps if fps > 0 else 25.0


def load_sparse_predictions_from_frame_scores(
    frame_scores_json: Path,
    *,
    threshold: float,
) -> dict[int, FramePrediction]:
    rows = json.loads(frame_scores_json.read_text())
    sparse: dict[int, FramePrediction] = {}
    for row in rows:
        frame_name = str(row["frame"])
        frame_index = max(int(Path(frame_name).stem) - 1, 0)
        fake_prob = float(row["final_prob"])
        sparse[frame_index] = FramePrediction(
            frame_index=frame_index,
            fake_prob=fake_prob,
            pred_label="FAKE" if fake_prob >= threshold else "REAL",
        )
    return sparse


def render_source_video(
    *,
    video_path: Path,
    video_out_path: Path,
    render_fps: float,
    dense_predictions,
    font_scale: float,
    thickness: int,
) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video for rendering: {video_path}")

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError(f"Failed to read first frame from {video_path}")
    first_canvas, first_offset = crop_to_vertical_canvas(first_frame)
    height, width = first_canvas.shape[:2]
    writer = cv2.VideoWriter(
        str(video_out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        render_fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video writer: {video_out_path}")

    face_detector = build_face_detector()
    rendered_count = 0
    detected_face_count = 0
    frame_count = len(dense_predictions)

    try:
        for frame_index in range(frame_count):
            if frame_index == 0:
                frame = first_frame
                canvas = first_canvas
                offset_x, offset_y = first_offset
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                canvas, (offset_x, offset_y) = crop_to_vertical_canvas(frame)
            face_box = detect_primary_face_box(face_detector, frame)
            if face_box is not None:
                detected_face_count += 1
                left, top, right, bottom = face_box
                face_box = (left + offset_x, top + offset_y, right + offset_x, bottom + offset_y)
            annotated = draw_prediction_overlay(
                canvas,
                prediction=dense_predictions[frame_index],
                face_box=face_box,
                font_scale=font_scale,
                thickness=thickness,
            )
            writer.write(annotated)
            rendered_count += 1
    finally:
        cap.release()
        writer.release()

    return rendered_count, width, height, render_fps


def render_historical_video_demo(
    *,
    video_path: Path,
    frame_scores_json: Path,
    output_dir: Path,
    threshold: float,
    video_name: str,
    fps: float | None,
    font_scale: float,
    thickness: int,
) -> dict[str, object]:
    frame_count, width, height, source_fps = read_video_metadata(video_path)
    sparse_predictions = load_sparse_predictions_from_frame_scores(
        frame_scores_json,
        threshold=threshold,
    )
    if not sparse_predictions:
        raise RuntimeError(f"No frame predictions found in {frame_scores_json}")
    dense_predictions = propagate_predictions(
        frame_count=frame_count,
        sparse_predictions=sparse_predictions,
    )
    sample_dir = output_dir / "rendered" / video_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    video_out_path = sample_dir / video_name
    render_fps = resolve_render_fps(video_path, fps)
    rendered_count, _, _, actual_fps = render_source_video(
        video_path=video_path,
        video_out_path=video_out_path,
        render_fps=render_fps,
        dense_predictions=dense_predictions,
        font_scale=font_scale,
        thickness=thickness,
    )
    report = {
        "sample_name": video_path.stem,
        "source_video": str(video_path),
        "frame_scores_json": str(frame_scores_json),
        "threshold": float(threshold),
        "num_images": len(sparse_predictions),
        "rendered_frame_count": rendered_count,
        "source_frame_count": frame_count,
        "source_resolution": [width, height],
        "source_fps": source_fps,
        "render_fps": actual_fps,
        "sparse_predictions": [
            {
                "frame_index": frame_index,
                "pred_label": sparse_predictions[frame_index].pred_label,
                "fake_prob": sparse_predictions[frame_index].fake_prob,
            }
            for frame_index in sorted(sparse_predictions)
        ],
    }
    report_path = sample_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return {
        "sample_name": video_path.stem,
        "source_video": str(video_path),
        "report_json": str(report_path),
        "video_path": str(video_out_path),
        "num_rendered_frames": rendered_count,
        "render_fps": actual_fps,
        "source_resolution": [width, height],
    }


def render_video_demo(
    demo_root: Path,
    summary: VideoProcessSummary,
    sample_args: argparse.Namespace,
    runtime,
    *,
    threshold: float,
    video_name: str,
    fps: float | None,
    font_scale: float,
    thickness: int,
) -> dict[str, object]:
    if summary.saved_count <= 0:
        raise RuntimeError(f"No aligned frames were produced for {summary.video_path}.")

    batch, prediction, sparse_predictions, image_rows = predict_video_summary(
        data_root=demo_root / "preprocessed",
        sampled_root=demo_root / "sampled" / summary.video_path.stem,
        summary=summary,
        sample_args=sample_args,
        runtime=runtime,
        threshold=threshold,
    )
    sample_dir = demo_root / "rendered" / batch.sample_name
    frames_dir = sample_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if not sparse_predictions:
        raise RuntimeError(f"No sampled predictions were generated for {summary.video_path}.")
    dense_predictions = propagate_predictions(
        frame_count=summary.frame_count,
        sparse_predictions=sparse_predictions,
    )

    video_path = sample_dir / video_name
    render_fps = resolve_render_fps(summary.video_path, fps)
    rendered_count, width, height, actual_fps = render_source_video(
        video_path=summary.video_path,
        video_out_path=video_path,
        render_fps=render_fps,
        dense_predictions=dense_predictions,
        font_scale=font_scale,
        thickness=thickness,
    )
    sample_report = build_sample_report(
        batch.sample_name,
        batch.cls_cache,
        prediction,
        threshold,
        runtime.hybrid_labels,
    )
    sample_report["images"] = image_rows
    sample_report["source_video"] = str(summary.video_path)
    sample_report["sampled_indices"] = summary.sampled_indices
    sample_report["saved_count"] = summary.saved_count
    sample_report["failed_count"] = summary.failed_count
    sample_report["rendered_frame_count"] = rendered_count
    sample_report["source_frame_count"] = summary.frame_count
    sample_report["source_resolution"] = [width, height]
    sample_report["render_fps"] = actual_fps
    sample_report["sparse_predictions"] = [
        {
            "frame_index": frame_index,
            "pred_label": sparse_predictions[frame_index].pred_label,
            "fake_prob": sparse_predictions[frame_index].fake_prob,
        }
        for frame_index in sorted(sparse_predictions)
    ]
    report_path = sample_dir / "report.json"
    report_path.write_text(json.dumps(sample_report, indent=2))
    return {
        "sample_name": batch.sample_name,
        "source_video": str(summary.video_path),
        "processed_frames_dir": str(summary.frames_dir),
        "landmarks_dir": str(summary.landmarks_dir),
        "report_json": str(report_path),
        "video_path": str(video_path),
        "num_rendered_frames": rendered_count,
        "mean_fake_prob": sample_report["mean_fake_prob"],
        "render_fps": actual_fps,
        "source_resolution": [width, height],
    }


def main(argv: list[str] | None = None) -> None:
    demo_args, sample_args = parse_args(argv)
    demo_root = demo_args.output_dir.resolve()
    demo_root.mkdir(parents=True, exist_ok=True)

    video_paths = resolve_video_paths(demo_args)
    if demo_args.frame_scores_json is not None:
        if len(video_paths) != 1:
            raise RuntimeError("--frame-scores-json currently supports exactly one input video.")
        threshold = 0.5 if demo_args.threshold is None else float(demo_args.threshold)
        sample_summary = render_historical_video_demo(
            video_path=video_paths[0],
            frame_scores_json=demo_args.frame_scores_json.resolve(),
            output_dir=demo_root,
            threshold=threshold,
            video_name=demo_args.video_name,
            fps=demo_args.fps,
            font_scale=demo_args.font_scale,
            thickness=demo_args.thickness,
        )
        summary_report = {
            "pipeline_root": str(PROJECT_ROOT),
            "threshold": threshold,
            "num_videos": 1,
            "mode": "historical_frame_scores",
            "samples": {video_paths[0].stem: sample_summary},
        }
        summary_path = demo_root / "demo_summary.json"
        summary_path.write_text(json.dumps(summary_report, indent=2))
        print(json.dumps(summary_report, indent=2))
        print(f"Saved demo outputs to {demo_root}")
        return

    preprocess_root = demo_root / "preprocessed" / "DEMO" / "ADHOC"
    if demo_args.mode == "uncertainty_refine":
        if demo_args.predictor_path is not None:
            raise RuntimeError("--mode uncertainty_refine currently supports raw-frame preprocessing only.")
        coarse_preprocess_root = demo_root / "preprocessed_coarse" / "DEMO" / "ADHOC"
        coarse_num_frames = min(
            demo_args.num_frames,
            demo_args.coarse_num_frames if demo_args.coarse_num_frames is not None else max(24, demo_args.num_frames // 2),
        )
        coarse_summaries = process_videos_raw(
            video_paths,
            output_root=coarse_preprocess_root,
            mode="fixed_num_frames",
            num_frames=coarse_num_frames,
            stride=demo_args.stride,
            log_path=demo_args.preprocess_log.resolve() if demo_args.preprocess_log is not None else None,
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
        summaries = []
        for coarse_summary in coarse_summaries:
            coarse_batch, coarse_prediction, _, _ = predict_video_summary(
                data_root=demo_root / "preprocessed_coarse",
                sampled_root=demo_root / "sampled_coarse" / coarse_summary.video_path.stem,
                summary=coarse_summary,
                sample_args=sample_args,
                runtime=runtime,
                threshold=threshold,
            )
            coarse_indices = [int(Path(str(img_path)).stem) for img_path in coarse_batch.cls_cache.img_id.astype(object).tolist()]
            refined_indices = build_uncertainty_refined_indices(
                frame_count=coarse_summary.frame_count,
                sampled_indices=coarse_indices,
                fake_probs=coarse_prediction.route_meta_prob,
                target_num_frames=demo_args.num_frames,
                threshold=threshold,
            )
            summaries.append(
                process_video_raw(
                    video_path=coarse_summary.video_path,
                    output_root=preprocess_root,
                    mode="fixed_num_frames",
                    num_frames=demo_args.num_frames,
                    stride=demo_args.stride,
                    log_path=demo_args.preprocess_log.resolve() if demo_args.preprocess_log is not None else None,
                    sampled_indices_override=refined_indices,
                )
            )
    elif demo_args.predictor_path is not None:
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
    else:
        summaries = process_videos_raw(
            video_paths,
            output_root=preprocess_root,
            mode=demo_args.mode,
            num_frames=demo_args.num_frames,
            stride=demo_args.stride,
            log_path=demo_args.preprocess_log.resolve() if demo_args.preprocess_log is not None else None,
        )

    build_frame_index(
        preprocess_root / "frames",
        demo_root / "frame_index.json",
        landmarks_root=preprocess_root / "landmarks",
        label=str(sample_args.label),
    )

    if demo_args.mode != "uncertainty_refine":
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
        "mode": demo_args.mode,
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
