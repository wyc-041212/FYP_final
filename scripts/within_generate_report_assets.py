#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "within_outputs"
OUT = DATA / "report_assets"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_threshold_csv(path: Path) -> dict[float, dict[tuple[str, str], dict[str, str]]]:
    rows = read_csv_rows(path)
    by_t: dict[float, dict[tuple[str, str], dict[str, str]]] = {}
    for row in rows:
        t = float(row["threshold"])
        by_t.setdefault(t, {})[(row["split"], row["scope"])] = row
    return by_t


def read_cdf_csv(path: Path) -> tuple[dict[float, dict[str, str]], dict[float, dict[str, str]]]:
    rows = read_csv_rows(path)
    real = {float(r["threshold"]): r for r in rows if r["scope"] == "pooled"}
    fake = {float(r["threshold"]): r for r in rows if r["split"] == "cdf_fake_only"}
    return real, fake


def nofr_ff_exfr_map(path: Path) -> dict[float, dict[str, float]]:
    data = json.loads(path.read_text())
    out: dict[float, dict[str, float]] = {}
    for th_key, item in data["thresholds"].items():
        methods = item["test_ff"]["methods"]
        non_fr = [m for m in methods if m["group"] != "FR"]
        vals = ["accuracy", "balanced_accuracy", "fake_accuracy", "real_accuracy", "auc"]
        agg = {k: float(np.mean([m["metrics"][k] for m in non_fr])) for k in vals}
        out[float(th_key)] = agg
    return out


def nofr_ood_exfr_map(path: Path) -> dict[float, dict[str, float]]:
    data = json.loads(path.read_text())
    out: dict[float, dict[str, float]] = {}
    for th_key, item in data["thresholds"].items():
        methods = item["ood"]["methods"]
        non_fr = [m for m in methods if m["group"] != "FR"]
        vals = ["accuracy", "balanced_accuracy", "fake_accuracy", "real_accuracy", "auc"]
        agg = {k: float(np.mean([m["metrics"][k] for m in non_fr])) for k in vals}
        out[float(th_key)] = agg
    return out


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def draw_table_png(
    title: str,
    columns: list[str],
    rows: list[list[str]],
    out_path: Path,
    col_widths: list[float] | None = None,
) -> None:
    nrows = len(rows) + 1
    ncols = len(columns)
    fig_w = max(10, ncols * 1.6)
    fig_h = max(2.6, nrows * 0.55 + 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor("#E9EEF6")
            cell.set_text_props(weight="bold", color="#111111")
        else:
            if r % 2 == 1:
                cell.set_facecolor("#F8FAFD")
            else:
                cell.set_facecolor("#FFFFFF")
    ax.set_title(title, fontsize=14, weight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_main_results_assets() -> None:
    full_local = read_threshold_csv(DATA / "full_threshold_sweep.csv")
    no_local = read_threshold_csv(DATA / "no_fr_threshold_sweep.csv")
    full_cdf_real, full_cdf_fake = read_cdf_csv(DATA / "full_threshold_sweep_cdf.csv")
    no_cdf_real, no_cdf_fake = read_cdf_csv(DATA / "no_fr_threshold_sweep_cdf.csv")
    no_exfr = nofr_ff_exfr_map(DATA / "no_fr_threshold_sweep.json")
    no_ood_exfr = nofr_ood_exfr_map(DATA / "no_fr_threshold_sweep.json")

    threshold = 0.8
    full_ff = full_local[threshold][("test_ff", "summary")]
    full_ood = full_local[threshold][("ood", "summary")]
    no_ff = no_local[threshold][("test_ff", "summary")]
    no_ood = no_local[threshold][("ood", "summary")]

    rows_csv = [
        {
            "variant": "Full corrected",
            "threshold": "0.8",
            "test_ff_bacc": fmt(float(full_ff["balanced_accuracy"])),
            "ood_bacc": fmt(float(full_ood["balanced_accuracy"])),
            "cdf_real_acc": fmt(float(full_cdf_real[threshold]["real_accuracy"])),
            "cdf_fake_acc": fmt(float(full_cdf_fake[threshold]["fake_positive_rate"])),
        },
        {
            "variant": "no-FR corrected",
            "threshold": "0.8",
            "test_ff_bacc": fmt(float(no_ff["balanced_accuracy"])),
            "ood_bacc": fmt(float(no_ood["balanced_accuracy"])),
            "cdf_real_acc": fmt(float(no_cdf_real[threshold]["real_accuracy"])),
            "cdf_fake_acc": fmt(float(no_cdf_fake[threshold]["fake_positive_rate"])),
        },
        {
            "variant": "no-FR corrected (excluding FR in test_ff)",
            "threshold": "0.8",
            "test_ff_bacc": fmt(float(no_exfr[threshold]["balanced_accuracy"])),
            "ood_bacc": fmt(float(no_ood["balanced_accuracy"])),
            "cdf_real_acc": fmt(float(no_cdf_real[threshold]["real_accuracy"])),
            "cdf_fake_acc": fmt(float(no_cdf_fake[threshold]["fake_positive_rate"])),
        },
        {
            "variant": "no-FR corrected (excluding FR in test_ff and ood)",
            "threshold": "0.8",
            "test_ff_bacc": fmt(float(no_exfr[threshold]["balanced_accuracy"])),
            "ood_bacc": fmt(float(no_ood_exfr[threshold]["balanced_accuracy"])),
            "cdf_real_acc": fmt(float(no_cdf_real[threshold]["real_accuracy"])),
            "cdf_fake_acc": fmt(float(no_cdf_fake[threshold]["fake_positive_rate"])),
        },
    ]
    write_csv(
        OUT / "main_results_summary.csv",
        ["variant", "threshold", "test_ff_bacc", "ood_bacc", "cdf_real_acc", "cdf_fake_acc"],
        rows_csv,
    )
    draw_table_png(
        "Main Results Summary at Threshold 0.8",
        ["Variant", "Thr", "Test-FF BAcc", "OOD BAcc", "CDF Real Acc", "CDF Fake Acc"],
        [[r["variant"], r["threshold"], r["test_ff_bacc"], r["ood_bacc"], r["cdf_real_acc"], r["cdf_fake_acc"]] for r in rows_csv],
        OUT / "main_results_summary.png",
        col_widths=[0.42, 0.08, 0.13, 0.13, 0.12, 0.12],
    )


def build_ablation_assets() -> None:
    rows = []
    for setting, path in [
        ("Full corrected", DATA / "full_main_ablation_line.csv"),
        ("no-FR corrected", DATA / "no_fr_main_ablation_line.csv"),
    ]:
        for row in read_csv_rows(path):
            if row["split"] != "test_ff":
                continue
            model = row["model"]
            threshold = row["threshold"]
            ff_bacc = row["balanced_accuracy"]
            rows.append((setting, model, threshold, ff_bacc))

    full_data = {r["model"]: r for r in read_csv_rows(DATA / "full_main_ablation_line.csv") if r["split"] == "ood"}
    full_cdf = {r["model"]: r for r in read_csv_rows(DATA / "full_main_ablation_line.csv") if r["split"] == "cdf_real_subset"}
    no_data = {r["model"]: r for r in read_csv_rows(DATA / "no_fr_main_ablation_line.csv") if r["split"] == "ood"}
    no_cdf = {r["model"]: r for r in read_csv_rows(DATA / "no_fr_main_ablation_line.csv") if r["split"] == "cdf_real_subset"}

    rows_csv: list[dict[str, str]] = []
    for setting, base_path, ood_map, cdf_map in [
        ("Full corrected", DATA / "full_main_ablation_line.csv", full_data, full_cdf),
        ("no-FR corrected", DATA / "no_fr_main_ablation_line.csv", no_data, no_cdf),
    ]:
        ff_map = {r["model"]: r for r in read_csv_rows(base_path) if r["split"] == "test_ff"}
        for model in ["route_only", "patch_only", "pair_only", "route_meta_fusion"]:
            ff = ff_map[model]
            ood = ood_map[model]
            cdf = cdf_map[model]
            rows_csv.append(
                {
                    "setting": setting,
                    "model": model,
                    "threshold": ff["threshold"],
                    "test_ff_bacc": fmt(float(ff["balanced_accuracy"])),
                    "ood_bacc": fmt(float(ood["balanced_accuracy"])),
                    "cdf_real_acc": fmt(float(cdf["accuracy"])),
                }
            )

    write_csv(
        OUT / "main_ablation_summary.csv",
        ["setting", "model", "threshold", "test_ff_bacc", "ood_bacc", "cdf_real_acc"],
        rows_csv,
    )
    draw_table_png(
        "Main Ablation Summary",
        ["Setting", "Model", "Thr", "Test-FF BAcc", "OOD BAcc", "CDF Real Acc"],
        [[r["setting"], r["model"], r["threshold"], r["test_ff_bacc"], r["ood_bacc"], r["cdf_real_acc"]] for r in rows_csv],
        OUT / "main_ablation_summary.png",
        col_widths=[0.22, 0.2, 0.08, 0.16, 0.16, 0.16],
    )


def build_threshold_figure() -> None:
    full_local = read_threshold_csv(DATA / "full_threshold_sweep.csv")
    no_local = read_threshold_csv(DATA / "no_fr_threshold_sweep.csv")
    full_cdf_real, _ = read_cdf_csv(DATA / "full_threshold_sweep_cdf.csv")
    no_cdf_real, _ = read_cdf_csv(DATA / "no_fr_threshold_sweep_cdf.csv")
    no_exfr = nofr_ff_exfr_map(DATA / "no_fr_threshold_sweep.json")
    no_ood_exfr = nofr_ood_exfr_map(DATA / "no_fr_threshold_sweep.json")

    th = sorted(full_local.keys())
    full_ff = [float(full_local[t][("test_ff", "summary")]["balanced_accuracy"]) for t in th]
    full_ood = [float(full_local[t][("ood", "summary")]["balanced_accuracy"]) for t in th]
    full_real = [float(full_cdf_real[t]["real_accuracy"]) for t in th]
    no_ff_ex = [float(no_exfr[t]["balanced_accuracy"]) for t in th]
    no_ood_ex = [float(no_ood_exfr[t]["balanced_accuracy"]) for t in th]
    no_real = [float(no_cdf_real[t]["real_accuracy"]) for t in th]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    ax = axes[0]
    ax.plot(th, full_ff, marker="o", color="#1f77b4", label="Full Test-FF BAcc")
    ax.plot(th, full_ood, marker="o", color="#ff7f0e", label="Full OOD BAcc")
    ax.plot(th, full_real, marker="o", color="#2ca02c", label="Full CDF Real Acc")
    ax.set_title("Full Threshold Sensitivity")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.45, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(th, no_ff_ex, marker="o", color="#1f77b4", label="no-FR Test-FF BAcc (ex-FR)")
    ax.plot(th, no_ood_ex, marker="o", color="#ff7f0e", label="no-FR OOD BAcc (ex-FR)")
    ax.plot(th, no_real, marker="o", color="#2ca02c", label="no-FR CDF Real Acc")
    ax.set_title("no-FR Threshold Sensitivity")
    ax.set_xlabel("Threshold")
    ax.set_ylim(0.45, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Threshold Sensitivity: Full vs no-FR", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "threshold_sensitivity_full_vs_no_fr.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_cdf_tradeoff_figure() -> None:
    full_cdf_real, full_cdf_fake = read_cdf_csv(DATA / "full_threshold_sweep_cdf.csv")
    no_cdf_real, no_cdf_fake = read_cdf_csv(DATA / "no_fr_threshold_sweep_cdf.csv")
    th = sorted(full_cdf_real.keys())
    full_x = [float(full_cdf_real[t]["real_accuracy"]) for t in th]
    full_y = [float(full_cdf_fake[t]["fake_positive_rate"]) for t in th]
    no_x = [float(no_cdf_real[t]["real_accuracy"]) for t in th]
    no_y = [float(no_cdf_fake[t]["fake_positive_rate"]) for t in th]

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.plot(full_x, full_y, marker="o", color="#d62728", label="Full corrected")
    ax.plot(no_x, no_y, marker="o", color="#1f77b4", label="no-FR corrected")
    for x, y, t in zip(full_x, full_y, th):
        ax.annotate(f"{t:.1f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color="#d62728")
    for x, y, t in zip(no_x, no_y, th):
        ax.annotate(f"{t:.1f}", (x, y), textcoords="offset points", xytext=(4, -10), fontsize=8, color="#1f77b4")
    ax.set_xlabel("CDF Real Accuracy")
    ax.set_ylabel("CDF Fake Accuracy")
    ax.set_title("CDF Real/Fake Trade-Off")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "cdf_real_fake_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme() -> None:
    text = """# Report Assets

Generated visual assets for the final report.

Files:
- `main_results_summary.csv`
- `main_results_summary.png`
- `main_ablation_summary.csv`
- `main_ablation_summary.png`
- `threshold_sensitivity_full_vs_no_fr.png`
- `cdf_real_fake_tradeoff.png`

Notes:
- Main-result tables currently use threshold `0.8` for both full and no-FR to provide a consistent report-facing operating point.
- The no-FR table includes a row with `test_ff` recalculated after excluding `FR`, which is important for fair interpretation of the redesign.
"""
    (OUT / "README.md").write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_main_results_assets()
    build_ablation_assets()
    build_threshold_figure()
    build_cdf_tradeoff_figure()
    write_readme()


if __name__ == "__main__":
    main()
