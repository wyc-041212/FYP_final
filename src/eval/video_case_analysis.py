from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_variant_name(path: str | Path) -> str:
    return Path(path).resolve().parents[2].name


def load_report(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as handle:
        return json.load(handle)


def load_case_reports(video_id: str, *, reports_root: Path) -> list[dict[str, Any]]:
    pattern = f"{video_id}*/rendered/{video_id}/report.json"
    reports: list[dict[str, Any]] = []
    for path in sorted(reports_root.glob(pattern)):
        report = load_report(path)
        report["report_path"] = str(path)
        report["variant"] = parse_variant_name(path)
        reports.append(report)
    return reports


def normalize_frame_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    variant = str(report["variant"])
    rows = []
    for row in sorted(report.get("images", []), key=lambda item: int(item["frame_index"])):
        rows.append(
            {
                "variant": variant,
                "frame_index": int(row["frame_index"]),
                "route_top1": str(row["route_top1"]),
                "route_meta_prob": float(row["route_meta_prob"]),
                "pred": int(row.get("pred", 0)),
                "pred_label": str(row.get("pred_label", "unknown")),
                "pair_id": str(row.get("pair_id", "")),
                "group": str(row.get("group", "")),
                "method": str(row.get("method", "")),
                "img_path": str(row.get("img_path", "")),
            }
        )
    return rows


def build_route_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda item: int(item["frame_index"]))
    segments: list[dict[str, Any]] = []
    current_route = str(ordered[0]["route_top1"])
    start_frame = int(ordered[0]["frame_index"])
    probs = [float(ordered[0]["route_meta_prob"])]
    frames = [start_frame]
    for row in ordered[1:]:
        route = str(row["route_top1"])
        frame = int(row["frame_index"])
        prob = float(row["route_meta_prob"])
        if route == current_route:
            probs.append(prob)
            frames.append(frame)
            continue
        segments.append(
            {
                "route_top1": current_route,
                "start_frame": int(frames[0]),
                "end_frame": int(frames[-1]),
                "length": int(len(frames)),
                "mean_prob": float(np.mean(np.asarray(probs, dtype=np.float32))),
                "max_prob": float(np.max(np.asarray(probs, dtype=np.float32))),
            }
        )
        current_route = route
        probs = [prob]
        frames = [frame]
    segments.append(
        {
            "route_top1": current_route,
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "length": int(len(frames)),
            "mean_prob": float(np.mean(np.asarray(probs, dtype=np.float32))),
            "max_prob": float(np.max(np.asarray(probs, dtype=np.float32))),
        }
    )
    return segments


def summarize_variant(report: dict[str, Any], *, threshold: float = 0.8) -> dict[str, Any]:
    rows = normalize_frame_rows(report)
    probs = np.asarray([row["route_meta_prob"] for row in rows], dtype=np.float32)
    segments = build_route_segments(rows)
    route_counts = {str(key): int(value) for key, value in report.get("route_top1_counts", {}).items()}
    dominant_route = max(route_counts.items(), key=lambda item: item[1])[0] if route_counts else "unknown"
    high_idx = np.flatnonzero(probs >= threshold).astype(np.int64)
    return {
        "variant": str(report["variant"]),
        "report_path": str(report["report_path"]),
        "num_images": int(report.get("num_images", len(rows))),
        "mean_fake_prob": float(report.get("mean_fake_prob", float(np.mean(probs)) if len(probs) else 0.0)),
        "fake_positive_rate": float(report.get("fake_positive_rate", float(np.mean(probs >= 0.5)) if len(probs) else 0.0)),
        "max_fake_prob": float(report.get("max_fake_prob", float(np.max(probs)) if len(probs) else 0.0)),
        "dominant_route": dominant_route,
        "route_top1_counts": route_counts,
        "num_route_segments": int(len(segments)),
        "high_conf_frame_count": int(len(high_idx)),
        "high_conf_frame_indices": [int(rows[idx]["frame_index"]) for idx in high_idx.tolist()],
        "rows": rows,
        "segments": segments,
    }


def compare_variant_routes(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        route_counts = dict(summary["route_top1_counts"])
        total = max(int(summary["num_images"]), 1)
        for route_name in sorted(route_counts):
            count = int(route_counts[route_name])
            rows.append(
                {
                    "variant": str(summary["variant"]),
                    "route_top1": route_name,
                    "count": count,
                    "fraction": float(count / total),
                }
            )
    return rows


def select_focus_variants(summaries: list[dict[str, Any]]) -> list[str]:
    names = {str(row["variant"]) for row in summaries}
    preferred = [
        "579_1775983303_no_fr_64f_t02_content",
        "579_1775983303_no_fr_64f_t02_uniform",
        "579_1775983303_no_fr_64f_t02_uncertainty",
        "579_1775983303_no_fr_24f_t02",
        "579_1775983303_no_fr_128f_t02",
    ]
    selected = [name for name in preferred if name in names]
    if selected:
        return selected
    ordered = sorted(
        summaries,
        key=lambda row: (float(row["fake_positive_rate"]), float(row["mean_fake_prob"])),
        reverse=True,
    )
    return [str(row["variant"]) for row in ordered[: min(5, len(ordered))]]
