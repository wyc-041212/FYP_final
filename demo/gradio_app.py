#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from demo.demo import main as run_demo_cli  # noqa: E402


DEFAULT_FULL_UPSTREAM = PROJECT_ROOT / "checkpoints" / "upstream" / "checkpoint_best_hybrid_manifold.pt"
DEFAULT_WITHIN_PATCH = PROJECT_ROOT / "within_checkpoints" / "downstream" / "patch_branch.joblib"
DEFAULT_WITHIN_PAIR = PROJECT_ROOT / "within_checkpoints" / "downstream" / "pair_branch.joblib"
DEFAULT_WITHIN_HEAD = PROJECT_ROOT / "within_checkpoints" / "heads" / "route_meta_head.joblib"


def run_demo_pipeline(
    *,
    video_path: Path,
    output_root: Path,
    mode: str,
    num_frames: int,
    threshold_override: float | None,
) -> dict[str, object]:
    argv = [
        "--video-path",
        str(video_path),
        "--output-dir",
        str(output_root),
        "--mode",
        mode,
        "--num-frames",
        str(num_frames),
        "--upstream-checkpoint",
        str(DEFAULT_FULL_UPSTREAM),
        "--patch-branch",
        str(DEFAULT_WITHIN_PATCH),
        "--pair-branch",
        str(DEFAULT_WITHIN_PAIR),
        "--route-meta-head",
        str(DEFAULT_WITHIN_HEAD),
        "--runtime-device",
        "cpu",
        "--cls-device",
        "cpu",
        "--patch-device",
        "cpu",
        "--facer-device",
        "cpu",
    ]
    if threshold_override is not None:
        argv.extend(["--threshold", str(threshold_override)])

    run_demo_cli(argv)
    summary_path = output_root / "demo_summary.json"
    return json.loads(summary_path.read_text())


def summarize_demo_output(summary: dict[str, object]) -> dict[str, object]:
    samples = summary.get("samples", {})
    if not isinstance(samples, dict) or not samples:
        raise RuntimeError("Demo summary does not contain any rendered samples.")
    sample_name, payload = next(iter(samples.items()))
    if not isinstance(payload, dict):
        raise RuntimeError("Demo sample payload is malformed.")

    mean_fake_prob = payload.get("mean_fake_prob")
    rendered_frames = payload.get("num_rendered_frames")
    render_fps = payload.get("render_fps")
    threshold = summary.get("threshold")

    summary_markdown = "\n".join(
        [
            "## Demo Summary",
            f"- Sample: `{sample_name}`",
            f"- Threshold: `{threshold}`",
            f"- Mean fake probability: `{mean_fake_prob}`",
            f"- Rendered frames: `{rendered_frames}`",
            f"- Render FPS: `{render_fps}`",
        ]
    )
    return {
        "sample_name": sample_name,
        "video_path": str(payload["video_path"]),
        "report_json_path": str(payload["report_json"]),
        "summary_markdown": summary_markdown,
        "raw_summary_json": json.dumps(summary, indent=2),
    }


def run_demo_for_uploaded_video(
    *,
    video_path: str,
    output_root: Path | None = None,
    mode: str = "fixed_num_frames",
    num_frames: int = 32,
    threshold_override: float | None = None,
) -> dict[str, object]:
    input_path = Path(video_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Uploaded video was not found: {input_path}")

    if output_root is None:
        temp_root = tempfile.mkdtemp(prefix="gradio-demo-")
        run_root = Path(temp_root)
    else:
        run_root = Path(output_root).resolve()
        run_root.mkdir(parents=True, exist_ok=True)

    summary = run_demo_pipeline(
        video_path=input_path,
        output_root=run_root,
        mode=mode,
        num_frames=num_frames,
        threshold_override=threshold_override,
    )
    return summarize_demo_output(summary)


def gradio_predict(video_path: str, mode: str, num_frames: int, threshold_override: float | None):
    result = run_demo_for_uploaded_video(
        video_path=video_path,
        mode=mode,
        num_frames=int(num_frames),
        threshold_override=threshold_override,
    )
    return (
        result["video_path"],
        result["summary_markdown"],
        result["raw_summary_json"],
        result["report_json_path"],
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Within Demo Video Detector") as demo:
        gr.Markdown(
            """
# Within Video Demo

Upload one video, run the standardized within pipeline, and review the annotated output video plus a compact summary.
            """.strip()
        )
        with gr.Row():
            with gr.Column(scale=1):
                input_video = gr.Video(label="Input Video", sources=["upload"])
                mode = gr.Dropdown(
                    choices=["fixed_num_frames", "content_change", "uncertainty_refine"],
                    value="fixed_num_frames",
                    label="Sampling Mode",
                )
                num_frames = gr.Slider(minimum=8, maximum=128, step=8, value=32, label="Frames to Sample")
                threshold_override = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.2, label="Threshold Override")
                run_button = gr.Button("Run Demo", variant="primary")
            with gr.Column(scale=1):
                output_video = gr.Video(label="Annotated Output", interactive=False)
                summary_md = gr.Markdown()
                raw_json = gr.Code(label="Raw Summary JSON", language="json")
                report_file = gr.File(label="report.json")

        run_button.click(
            fn=gradio_predict,
            inputs=[input_video, mode, num_frames, threshold_override],
            outputs=[output_video, summary_md, raw_json, report_file],
            show_progress="full",
        )
    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
