#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict

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
        vals = ["accuracy", "balanced_accuracy", "fake_accuracy", "real_accuracy", "auc", "ap", "eer"]
        agg = {k: float(np.mean([m["metrics"][k] for m in non_fr])) for k in vals}
        out[float(th_key)] = agg
    return out


def nofr_ood_exfr_map(path: Path) -> dict[float, dict[str, float]]:
    data = json.loads(path.read_text())
    out: dict[float, dict[str, float]] = {}
    for th_key, item in data["thresholds"].items():
        methods = item["ood"]["methods"]
        non_fr = [m for m in methods if m["group"] != "FR"]
        vals = ["accuracy", "balanced_accuracy", "fake_accuracy", "real_accuracy", "auc", "ap", "eer"]
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


def apply_clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA4B2")
    ax.spines["bottom"].set_color("#9AA4B2")
    ax.tick_params(colors="#263238", labelsize=10)
    ax.grid(axis="x", alpha=0.18, color="#9AA4B2", linewidth=0.8)
    ax.set_axisbelow(True)


def add_value_labels(ax, bars, dy: float = 0.008):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + dy,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1F2933",
        )


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
            "variant": "Full",
            "threshold": "0.8",
            "test_ff_bacc": fmt(float(full_ff["balanced_accuracy"])),
            "test_ff_ap": fmt(float(full_ff["ap"])),
            "test_ff_eer": fmt(float(full_ff["eer"])),
            "ood_bacc": fmt(float(full_ood["balanced_accuracy"])),
            "ood_ap": fmt(float(full_ood["ap"])),
            "ood_eer": fmt(float(full_ood["eer"])),
            "cdf_real_acc": fmt(float(full_cdf_real[threshold]["real_accuracy"])),
            "cdf_fake_acc": fmt(float(full_cdf_fake[threshold]["fake_positive_rate"])),
        },
        {
            "variant": "no-FR",
            "threshold": "0.8",
            "test_ff_bacc": fmt(float(no_ff["balanced_accuracy"])),
            "test_ff_ap": fmt(float(no_ff["ap"])),
            "test_ff_eer": fmt(float(no_ff["eer"])),
            "ood_bacc": fmt(float(no_ood["balanced_accuracy"])),
            "ood_ap": fmt(float(no_ood["ap"])),
            "ood_eer": fmt(float(no_ood["eer"])),
            "cdf_real_acc": fmt(float(no_cdf_real[threshold]["real_accuracy"])),
            "cdf_fake_acc": fmt(float(no_cdf_fake[threshold]["fake_positive_rate"])),
        },
        {
            "variant": "no-FR (excluding FR in test_ff)",
            "threshold": "0.8",
            "test_ff_bacc": fmt(float(no_exfr[threshold]["balanced_accuracy"])),
            "test_ff_ap": fmt(float(no_exfr[threshold]["ap"])),
            "test_ff_eer": fmt(float(no_exfr[threshold]["eer"])),
            "ood_bacc": fmt(float(no_ood["balanced_accuracy"])),
            "ood_ap": fmt(float(no_ood["ap"])),
            "ood_eer": fmt(float(no_ood["eer"])),
            "cdf_real_acc": fmt(float(no_cdf_real[threshold]["real_accuracy"])),
            "cdf_fake_acc": fmt(float(no_cdf_fake[threshold]["fake_positive_rate"])),
        },
        {
            "variant": "no-FR (fair scope)",
            "threshold": "0.8",
            "test_ff_bacc": fmt(float(no_exfr[threshold]["balanced_accuracy"])),
            "test_ff_ap": fmt(float(no_exfr[threshold]["ap"])),
            "test_ff_eer": fmt(float(no_exfr[threshold]["eer"])),
            "ood_bacc": fmt(float(no_ood_exfr[threshold]["balanced_accuracy"])),
            "ood_ap": fmt(float(no_ood_exfr[threshold]["ap"])),
            "ood_eer": fmt(float(no_ood_exfr[threshold]["eer"])),
            "cdf_real_acc": fmt(float(no_cdf_real[threshold]["real_accuracy"])),
            "cdf_fake_acc": fmt(float(no_cdf_fake[threshold]["fake_positive_rate"])),
        },
    ]
    write_csv(
        OUT / "main_results_summary.csv",
        [
            "variant",
            "threshold",
            "test_ff_bacc",
            "test_ff_ap",
            "test_ff_eer",
            "ood_bacc",
            "ood_ap",
            "ood_eer",
            "cdf_real_acc",
            "cdf_fake_acc",
        ],
        rows_csv,
    )
    metrics = [
        ("test_ff_bacc", "test_ff bAcc"),
        ("ood_bacc", "OOD bAcc"),
        ("cdf_real_acc", "CDF real"),
        ("cdf_fake_acc", "CDF fake"),
    ]
    full_row = rows_csv[0]
    nofr_row = rows_csv[3]
    labels = [m[1] for m in metrics]
    full_vals = [float(full_row[m[0]]) for m in metrics]
    nofr_vals = [float(nofr_row[m[0]]) for m in metrics]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(labels))
    width = 0.34
    bars1 = ax.bar(x - width / 2, full_vals, width, color="#A9B8D0", label="Full", edgecolor="#65748B", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, nofr_vals, width, color="#355C7D", label="no-FR (fair scope)", edgecolor="#24425B", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Main Results Summary at Threshold 0.8", fontsize=15, weight="bold")
    apply_clean_axes(ax)
    ax.legend(frameon=False, loc="upper left")
    add_value_labels(ax, bars1)
    add_value_labels(ax, bars2)
    ax.text(
        0.99,
        0.02,
        "Higher is better for all metrics shown",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    fig.savefig(OUT / "main_results_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_ablation_assets() -> None:
    rows = []
    for setting, path in [
        ("Full", DATA / "full_main_ablation_line.csv"),
        ("no-FR", DATA / "no_fr_main_ablation_line.csv"),
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
        ("Full", DATA / "full_main_ablation_line.csv", full_data, full_cdf),
        ("no-FR", DATA / "no_fr_main_ablation_line.csv", no_data, no_cdf),
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
                    "test_ff_ap": fmt(float(ff["ap"])),
                    "test_ff_eer": fmt(float(ff["eer"])),
                    "ood_bacc": fmt(float(ood["balanced_accuracy"])),
                    "ood_ap": fmt(float(ood["ap"])),
                    "ood_eer": fmt(float(ood["eer"])),
                    "cdf_real_acc": fmt(float(cdf["accuracy"])),
                }
            )

    write_csv(
        OUT / "main_ablation_summary.csv",
        ["setting", "model", "threshold", "test_ff_bacc", "test_ff_ap", "test_ff_eer", "ood_bacc", "ood_ap", "ood_eer", "cdf_real_acc"],
        rows_csv,
    )
    order = ["route_only", "patch_only", "pair_only", "route_meta_fusion"]
    pretty = {
        "route_only": "route_only",
        "patch_only": "patch_only",
        "pair_only": "pair_only",
        "route_meta_fusion": "route_meta_fusion",
    }
    colors = {
        "route_only": "#90A4AE",
        "patch_only": "#F4A261",
        "pair_only": "#2A9D8F",
        "route_meta_fusion": "#355C7D",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), sharey=True)
    panels = [
        ("test_ff_bacc", "test_ff bAcc"),
        ("ood_bacc", "OOD bAcc"),
        ("cdf_real_acc", "CDF real"),
    ]
    settings = ["Full", "no-FR"]
    offsets = [-0.18, 0.18]
    y = np.arange(len(order))

    for ax, (metric_key, title) in zip(axes, panels):
        for s_idx, setting in enumerate(settings):
            vals = []
            for model in order:
                row = next(r for r in rows_csv if r["setting"] == setting and r["model"] == model)
                vals.append(float(row[metric_key]))
            ax.barh(
                y + offsets[s_idx],
                vals,
                height=0.34,
                color=[colors[m] for m in order],
                alpha=0.95 if setting == "no-FR" else 0.45,
                edgecolor="#4B5563",
                linewidth=0.6,
                label=setting if metric_key == "test_ff_bacc" else None,
            )
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([pretty[m] for m in order], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0.2 if metric_key == "cdf_real_acc" else 0.7, 1.0)
        apply_clean_axes(ax)
        for s_idx, setting in enumerate(settings):
            vals = []
            for model in order:
                row = next(r for r in rows_csv if r["setting"] == setting and r["model"] == model)
                vals.append(float(row[metric_key]))
            for yy, vv in zip(y + offsets[s_idx], vals):
                ax.text(vv + 0.01, yy, f"{vv:.3f}", va="center", fontsize=8.5, color="#263238")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#355C7D", alpha=0.45, ec="#4B5563", lw=0.6),
        plt.Rectangle((0, 0), 1, 1, color="#355C7D", alpha=0.95, ec="#4B5563", lw=0.6),
    ]
    fig.legend(handles, settings, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Main Ablation Summary", fontsize=15, weight="bold", y=1.06)
    fig.tight_layout()
    fig.savefig(OUT / "main_ablation_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


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

    colors = {
        "test_ff": "#355C7D",
        "ood": "#F4A261",
        "cdf_real": "#2A9D8F",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True, sharey=True)
    series = [
        ("Full", full_ff, full_ood, full_real),
        ("no-FR (fair scope)", no_ff_ex, no_ood_ex, no_real),
    ]
    for ax, (title, ff_vals, ood_vals, real_vals) in zip(axes, series):
        ax.plot(th, ff_vals, marker="o", markersize=5.5, linewidth=2.2, color=colors["test_ff"], label="test_ff bAcc")
        ax.plot(th, ood_vals, marker="s", markersize=5.2, linewidth=2.2, color=colors["ood"], label="OOD bAcc")
        ax.plot(th, real_vals, marker="^", markersize=5.6, linewidth=2.2, color=colors["cdf_real"], label="CDF real")
        ax.axvline(0.8, color="#7B8794", linestyle="--", linewidth=1.2, alpha=0.9)
        ax.annotate("0.8", xy=(0.8, 0.455), xytext=(4, 2), textcoords="offset points", fontsize=9, color="#616E7C")
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel("Threshold")
        ax.set_ylim(0.20, 1.0)
        ax.set_xlim(min(th) - 0.02, max(th) + 0.02)
        apply_clean_axes(ax)
    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
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

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot(full_x, full_y, marker="o", markersize=5.5, linewidth=2.2, color="#A9B8D0", label="Full", markeredgecolor="#65748B")
    ax.plot(no_x, no_y, marker="o", markersize=5.5, linewidth=2.4, color="#355C7D", label="no-FR", markeredgecolor="#24425B")
    for x, y, t in zip(full_x, full_y, th):
        if abs(t - 0.8) < 1e-9:
            ax.annotate("0.8", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9, color="#65748B", weight="bold")
    for x, y, t in zip(no_x, no_y, th):
        if abs(t - 0.8) < 1e-9:
            ax.annotate("0.8", (x, y), textcoords="offset points", xytext=(6, -14), fontsize=9, color="#24425B", weight="bold")
    ax.scatter([full_x[0], full_x[-1]], [full_y[0], full_y[-1]], s=42, color="#A9B8D0", edgecolors="#65748B", zorder=3)
    ax.scatter([no_x[0], no_x[-1]], [no_y[0], no_y[-1]], s=42, color="#355C7D", edgecolors="#24425B", zorder=3)
    ax.set_xlabel("CDF real accuracy")
    ax.set_ylabel("CDF fake accuracy")
    ax.set_title("CDF Real/Fake Trade-Off", fontsize=15, weight="bold")
    ax.set_xlim(0.45, 0.85)
    ax.set_ylim(0.80, 0.96)
    apply_clean_axes(ax)
    ax.grid(axis="y", alpha=0.18, color="#9AA4B2", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.02,
        0.03,
        "Upper-right is better",
        transform=ax.transAxes,
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    fig.savefig(OUT / "cdf_real_fake_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_ch9_synthesis_figure() -> None:
    labels = ["test_ff bAcc", "OOD bAcc", "CDF real"]
    full_vals = [0.9507, 0.7881, 0.6051]
    nofr_vals = [0.9548, 0.8261, 0.7892]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(labels))
    width = 0.34
    bars1 = ax.bar(
        x - width / 2,
        full_vals,
        width,
        color="#A9B8D0",
        label="Full",
        edgecolor="#65748B",
        linewidth=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        nofr_vals,
        width,
        color="#355C7D",
        label="no-FR (fair scope)",
        edgecolor="#24425B",
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Final Synthesis: Full vs no-FR", fontsize=15, weight="bold")
    apply_clean_axes(ax)
    ax.grid(axis="y", alpha=0.18, color="#9AA4B2", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    add_value_labels(ax, bars1)
    add_value_labels(ax, bars2)
    ax.text(
        0.99,
        0.02,
        "no-FR preserves closed-set strength while improving robustness",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    fig.savefig(OUT / "ch9_final_synthesis.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_appendix_threshold_profiles() -> None:
    full_local = read_threshold_csv(DATA / "full_threshold_sweep.csv")
    no_local = read_threshold_csv(DATA / "no_fr_threshold_sweep.csv")
    full_cdf_real, _ = read_cdf_csv(DATA / "full_threshold_sweep_cdf.csv")
    no_cdf_real, _ = read_cdf_csv(DATA / "no_fr_threshold_sweep_cdf.csv")
    no_exfr = nofr_ff_exfr_map(DATA / "no_fr_threshold_sweep.json")
    no_ood_exfr = nofr_ood_exfr_map(DATA / "no_fr_threshold_sweep.json")
    th = sorted(full_local.keys())

    configs = [
        (
            "appendix_full_threshold_profile.png",
            "Full Operating Profile",
            [float(full_local[t][("test_ff", "summary")]["balanced_accuracy"]) for t in th],
            [float(full_local[t][("ood", "summary")]["balanced_accuracy"]) for t in th],
            [float(full_cdf_real[t]["real_accuracy"]) for t in th],
        ),
        (
            "appendix_no_fr_threshold_profile.png",
            "no-FR Operating Profile (fair scope)",
            [float(no_exfr[t]["balanced_accuracy"]) for t in th],
            [float(no_ood_exfr[t]["balanced_accuracy"]) for t in th],
            [float(no_cdf_real[t]["real_accuracy"]) for t in th],
        ),
    ]
    colors = {"test_ff": "#355C7D", "ood": "#F4A261", "cdf_real": "#2A9D8F"}
    for out_name, title, ff_vals, ood_vals, real_vals in configs:
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        ax.plot(th, ff_vals, marker="o", markersize=5.4, linewidth=2.2, color=colors["test_ff"], label="test_ff bAcc")
        ax.plot(th, ood_vals, marker="s", markersize=5.2, linewidth=2.2, color=colors["ood"], label="OOD bAcc")
        ax.plot(th, real_vals, marker="^", markersize=5.6, linewidth=2.2, color=colors["cdf_real"], label="CDF real")
        ax.axvline(0.8, color="#7B8794", linestyle="--", linewidth=1.2, alpha=0.9)
        ax.annotate("0.8", xy=(0.8, 0.215), xytext=(4, 2), textcoords="offset points", fontsize=9, color="#616E7C")
        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(0.20, 1.0)
        ax.set_xlim(min(th) - 0.02, max(th) + 0.02)
        apply_clean_axes(ax)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT / out_name, dpi=220, bbox_inches="tight")
        plt.close(fig)


def read_probe_rows(path: Path, threshold: float = 0.8) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    return [r for r in rows if float(r["threshold"]) == threshold]


def build_probe_figures() -> None:
    full_rows = read_probe_rows(DATA / "full_small_ood_probe_sweep.csv", 0.8)
    nofr_rows = read_probe_rows(DATA / "no_fr_small_ood_probe_sweep.csv", 0.8)

    probe_order = sorted({r["probe_name"] for r in full_rows} | {r["probe_name"] for r in nofr_rows})
    grouped = {}
    for label, rows in [("Full", full_rows), ("no-FR", nofr_rows)]:
        by = defaultdict(list)
        for r in rows:
            by[r["probe_name"]].append(float(r["fake_positive_rate"]))
        grouped[label] = {k: float(np.mean(v)) for k, v in by.items()}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(probe_order))
    width = 0.34
    full_vals = [grouped["Full"].get(p, np.nan) for p in probe_order]
    nofr_vals = [grouped["no-FR"].get(p, np.nan) for p in probe_order]
    bars1 = ax.bar(x - width / 2, full_vals, width, color="#A9B8D0", label="Full", edgecolor="#65748B", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, nofr_vals, width, color="#355C7D", label="no-FR", edgecolor="#24425B", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(probe_order)
    ax.set_ylabel("Fake positive rate at 0.8")
    ax.set_title("Probe Bundle Summary", fontsize=15, weight="bold")
    apply_clean_axes(ax)
    ax.grid(axis="y", alpha=0.18, color="#9AA4B2", linewidth=0.8)
    add_value_labels(ax, bars1, dy=0.01)
    add_value_labels(ax, bars2, dy=0.01)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "appendix_probe_bundle_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    probe_markers = {"animations": "o", "kobe_test": "s", "tiktok": "^"}
    palette = {"Full": "#A9B8D0", "no-FR": "#355C7D"}
    for label, rows in [("Full", full_rows), ("no-FR", nofr_rows)]:
        for probe in probe_order:
            pts = [r for r in rows if r["probe_name"] == probe]
            xs = [float(r["mean_fake_prob"]) for r in pts]
            ys = [float(r["fake_positive_rate"]) for r in pts]
            ax.scatter(
                xs,
                ys,
                s=52,
                marker=probe_markers.get(probe, "o"),
                color=palette[label],
                edgecolors="#44546A",
                alpha=0.85 if label == "no-FR" else 0.55,
                label=f"{label} / {probe}",
            )
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_h.append(h)
            uniq_l.append(l)
            seen.add(l)
    ax.set_xlabel("Mean fake probability")
    ax.set_ylabel("Fake positive rate at 0.8")
    ax.set_title("Probe Sample-Level Behavior", fontsize=15, weight="bold")
    apply_clean_axes(ax)
    ax.grid(axis="y", alpha=0.18, color="#9AA4B2", linewidth=0.8)
    ax.legend(uniq_h, uniq_l, frameon=False, fontsize=8.5, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "appendix_probe_sample_behavior.png", dpi=220, bbox_inches="tight")
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
- `ch9_final_synthesis.png`
- `appendix_full_threshold_profile.png`
- `appendix_no_fr_threshold_profile.png`
- `appendix_probe_bundle_summary.png`
- `appendix_probe_sample_behavior.png`

Notes:
- Main-result tables currently use threshold `0.8` for both full and no-FR to provide a consistent report-facing operating point.
- The no-FR table includes a row with `test_ff` and OOD recalculated after excluding `FR`, which is important for fair interpretation of the redesign.
- AP and EER are reported for `test_ff` and OOD only; the CDF lines remain real-only or fake-only robustness views rather than balanced binary evaluation tables.
"""
    (OUT / "README.md").write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_main_results_assets()
    build_ablation_assets()
    build_threshold_figure()
    build_cdf_tradeoff_figure()
    build_ch9_synthesis_figure()
    build_appendix_threshold_profiles()
    build_probe_figures()
    write_readme()


if __name__ == "__main__":
    main()
