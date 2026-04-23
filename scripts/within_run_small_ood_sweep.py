#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from main import (  # noqa: E402
    SampleBatch,
    build_sample_report,
    compact_from_patch_npz,
    load_main_runtime,
    predict_sample,
)
from prepare.cache import EmbeddingCache  # noqa: E402
from train.within_train_downstream_head import configure_group_scheme  # noqa: E402


@dataclass(frozen=True)
class ProbeBundle:
    probe_name: str
    sample_name: str
    cls_path: Path
    patch_path: Path
    manifest_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep thresholds over small OOD/sample probes.")
    parser.add_argument("--upstream-checkpoint", type=Path, required=True)
    parser.add_argument("--patch-branch", type=Path, required=True)
    parser.add_argument("--pair-branch", type=Path, required=True)
    parser.add_argument("--route-meta-head", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--route-batch-size", type=int, default=2048)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--animations-root",
        type=Path,
        default=Path("/Users/wuyuchen/Desktop/FYP_final_backup/outputs/demos/animations_20260324_1955/sampled"),
    )
    parser.add_argument(
        "--animations-repaired-root",
        type=Path,
        default=Path("/Users/wuyuchen/Desktop/FYP_final_backup/outputs/reports/domain_shift_audit_20260324/animation_patch_reextract"),
    )
    parser.add_argument(
        "--kobe-root",
        type=Path,
        default=Path("/Users/wuyuchen/Desktop/kobe_test/kobe_prediction_samples/kobe_sample"),
    )
    parser.add_argument(
        "--tiktok-root",
        type=Path,
        default=Path("/Users/wuyuchen/Desktop/FYP_final_backup/outputs/probes/tiktok_topk_no_fr_20260329/sampled_bundle"),
    )
    return parser.parse_args()


def discover_animations(sampled_root: Path, repaired_root: Path) -> list[ProbeBundle]:
    bundles: list[ProbeBundle] = []
    if not sampled_root.exists():
        return bundles
    for sample_dir in sorted(p for p in sampled_root.iterdir() if p.is_dir()):
        if sample_dir.name == "smoke_test":
            continue
        cls_path = sample_dir / "cls_sampled.npz"
        manifest_path = sample_dir / "manifest.csv"
        repaired_patch = repaired_root / sample_dir.name / "patch_sampled_reextract.npz"
        patch_path = repaired_patch if repaired_patch.exists() else sample_dir / "patch_sampled.npz"
        if cls_path.exists() and patch_path.exists():
            bundles.append(
                ProbeBundle(
                    probe_name="animations",
                    sample_name=sample_dir.name,
                    cls_path=cls_path,
                    patch_path=patch_path,
                    manifest_path=manifest_path if manifest_path.exists() else None,
                )
            )
    return bundles


def discover_single_bundle(probe_name: str, root: Path) -> list[ProbeBundle]:
    cls_path = root / "cls_sampled.npz"
    patch_path = root / "patch_sampled.npz"
    manifest_path = root / "manifest.csv"
    if cls_path.exists() and patch_path.exists():
        return [
            ProbeBundle(
                probe_name=probe_name,
                sample_name=root.name,
                cls_path=cls_path,
                patch_path=patch_path,
                manifest_path=manifest_path if manifest_path.exists() else None,
            )
        ]
    return []


def load_probe_batches(args: argparse.Namespace, target_region_names: list[str]) -> list[ProbeBundle]:
    bundles: list[ProbeBundle] = []
    bundles.extend(discover_animations(args.animations_root, args.animations_repaired_root))
    bundles.extend(discover_single_bundle("kobe_test", args.kobe_root))
    bundles.extend(discover_single_bundle("tiktok", args.tiktok_root))
    if not bundles:
        raise FileNotFoundError("No probe bundles were discovered.")
    return bundles


def load_sample_batch(bundle: ProbeBundle, target_region_names: list[str]) -> SampleBatch:
    cls_cache = EmbeddingCache.from_npz(bundle.cls_path)
    patch_cache = compact_from_patch_npz(bundle.patch_path, target_region_names)
    return SampleBatch(
        sample_name=bundle.sample_name,
        rows=[],
        cls_cache=cls_cache,
        patch_cache=patch_cache,
    )


def fake_positive_rate(prob: np.ndarray, threshold: float) -> float:
    return float(np.mean(prob >= threshold))


def summarize_probe(
    *,
    probe_name: str,
    reports: list[dict[str, object]],
    thresholds: list[float],
) -> dict[str, object]:
    pooled_prob = np.concatenate([np.asarray(row["route_meta_prob"], dtype=np.float32) for row in reports], axis=0)
    sample_means = np.asarray([float(row["mean_fake_prob"]) for row in reports], dtype=np.float32)
    sample_sizes = np.asarray([int(row["num_images"]) for row in reports], dtype=np.int64)
    threshold_rows = []
    for threshold in thresholds:
        sample_fprs = np.asarray(
            [fake_positive_rate(np.asarray(row["route_meta_prob"], dtype=np.float32), threshold) for row in reports],
            dtype=np.float32,
        )
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "pooled_fake_positive_rate": fake_positive_rate(pooled_prob, threshold),
                "mean_sample_fake_positive_rate": float(np.mean(sample_fprs)),
                "min_sample_fake_positive_rate": float(np.min(sample_fprs)),
                "max_sample_fake_positive_rate": float(np.max(sample_fprs)),
            }
        )
    videos: list[dict[str, object]] = []
    for row in reports:
        pair_ids = np.asarray(row["pair_id"], dtype=object)
        probs = np.asarray(row["route_meta_prob"], dtype=np.float32)
        for video_name in sorted(set(pair_ids.tolist())):
            idx = np.flatnonzero(pair_ids == video_name)
            video_prob = probs[idx]
            videos.append(
                {
                    "sample_name": row["sample_name"],
                    "video_name": str(video_name),
                    "num_images": int(len(idx)),
                    "mean_fake_prob": float(np.mean(video_prob)),
                    "min_fake_prob": float(np.min(video_prob)),
                    "max_fake_prob": float(np.max(video_prob)),
                    "threshold_rows": [
                        {
                            "threshold": float(threshold),
                            "fake_positive_rate": fake_positive_rate(video_prob, threshold),
                        }
                        for threshold in thresholds
                    ],
                }
            )
    return {
        "probe_name": probe_name,
        "num_samples": int(len(reports)),
        "num_images": int(np.sum(sample_sizes)),
        "mean_fake_prob": float(np.mean(pooled_prob)),
        "min_fake_prob": float(np.min(pooled_prob)),
        "max_fake_prob": float(np.max(pooled_prob)),
        "mean_sample_fake_prob": float(np.mean(sample_means)),
        "samples": [
            {
                "sample_name": row["sample_name"],
                "num_images": int(row["num_images"]),
                "mean_fake_prob": float(row["mean_fake_prob"]),
                "min_fake_prob": float(row["min_fake_prob"]),
                "max_fake_prob": float(row["max_fake_prob"]),
                "threshold_rows": [
                    {
                        "threshold": float(threshold),
                        "fake_positive_rate": fake_positive_rate(np.asarray(row["route_meta_prob"], dtype=np.float32), threshold),
                    }
                    for threshold in thresholds
                ],
            }
            for row in reports
        ],
        "videos": videos,
        "threshold_rows": threshold_rows,
    }


def main() -> None:
    args = parse_args()
    thresholds = [float(x) for x in args.thresholds]
    runtime = load_main_runtime(
        upstream_checkpoint=args.upstream_checkpoint,
        patch_branch=args.patch_branch,
        pair_branch=args.pair_branch,
        route_meta_head=args.route_meta_head,
        route_batch_size=args.route_batch_size,
        device_name=args.device,
    )
    configure_group_scheme("FR" not in runtime.hybrid_labels, hybrid_labels=list(runtime.hybrid_labels))
    bundles = load_probe_batches(args, runtime.region_names)

    sample_rows: list[dict[str, object]] = []
    video_rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = {}

    for bundle in bundles:
        batch = load_sample_batch(bundle, runtime.region_names)
        prediction = predict_sample(runtime, batch.cls_cache, batch.patch_cache)
        report = build_sample_report(
            batch.sample_name,
            batch.cls_cache,
            prediction,
            runtime.route_meta_threshold,
            runtime.hybrid_labels,
        )
        report["route_meta_prob"] = prediction.route_meta_prob.astype(np.float32).tolist()
        report["pair_id"] = batch.cls_cache.pair_id.astype(object).tolist()
        grouped.setdefault(bundle.probe_name, []).append(report)

        for threshold in thresholds:
            sample_rows.append(
                {
                    "probe_name": bundle.probe_name,
                    "sample_name": batch.sample_name,
                    "num_images": int(report["num_images"]),
                    "threshold": float(threshold),
                    "mean_fake_prob": float(report["mean_fake_prob"]),
                    "min_fake_prob": float(report["min_fake_prob"]),
                    "max_fake_prob": float(report["max_fake_prob"]),
                    "fake_positive_rate": fake_positive_rate(prediction.route_meta_prob, threshold),
                }
            )
        pair_ids = batch.cls_cache.pair_id.astype(object)
        for video_name in sorted(set(pair_ids.tolist())):
            idx = np.flatnonzero(pair_ids == video_name)
            video_prob = prediction.route_meta_prob[idx]
            mean_fake_prob = float(np.mean(video_prob))
            for threshold in thresholds:
                video_rows.append(
                    {
                        "probe_name": bundle.probe_name,
                        "sample_name": batch.sample_name,
                        "video_name": video_name,
                        "num_images": int(len(idx)),
                        "threshold": float(threshold),
                        "mean_fake_prob": mean_fake_prob,
                        "fake_positive_rate": fake_positive_rate(video_prob, threshold),
                    }
                )

    probe_summary = {
        probe_name: summarize_probe(probe_name=probe_name, reports=reports, thresholds=thresholds)
        for probe_name, reports in grouped.items()
    }

    output = {
        "upstream_checkpoint": str(args.upstream_checkpoint),
        "patch_branch": str(args.patch_branch),
        "pair_branch": str(args.pair_branch),
        "route_meta_head": str(args.route_meta_head),
        "device": args.device,
        "route_batch_size": int(args.route_batch_size),
        "thresholds": thresholds,
        "probe_bundle_sources": {
            "animations_root": str(args.animations_root),
            "animations_repaired_root": str(args.animations_repaired_root),
            "kobe_root": str(args.kobe_root),
            "tiktok_root": str(args.tiktok_root),
        },
        "probes": probe_summary,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "probe_name",
                "sample_name",
                "num_images",
                "threshold",
                "mean_fake_prob",
                "min_fake_prob",
                "max_fake_prob",
                "fake_positive_rate",
            ],
        )
        writer.writeheader()
        for row in sample_rows:
            writer.writerow(row)

    video_csv = args.output_csv.with_name(args.output_csv.stem + "_video_level.csv")
    with video_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "probe_name",
                "sample_name",
                "video_name",
                "num_images",
                "threshold",
                "mean_fake_prob",
                "fake_positive_rate",
            ],
        )
        writer.writeheader()
        for row in video_rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
