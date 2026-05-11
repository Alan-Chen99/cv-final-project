"""Metric comparison plots from evaluation results.

Produces bar charts comparing CRPS, MAE, RMSE, and mass violation
across methods. Supports single-dataset and dual-dataset comparison layouts.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def load_results(path: str | Path) -> dict[str, dict[str, float]]:
    """Load evaluation results from JSON file.

    Returns:
        Dict mapping method name -> metric dict (crps, mae, rmse, mass_violation).
    """
    with open(path) as f:
        data = json.load(f)
    return data["results"]


DISPLAY_NAMES: dict[str, str] = {
    "flow-wide96-amp (28M)": "Wide96 (28M)",
    "flow-uniform-amp (13M)": "Uniform (13M)",
    "flow-logitnorm-ema (13M)": "LogitNorm (13M)",
    "flow-v2-zscore (13M)": "ZScore (13M)",
    "bicubic+addcl": "Bicubic+AddCL",
    "bilinear+addcl": "Bilinear+AddCL",
    "bicubic": "Bicubic",
    "bilinear": "Bilinear",
    "harder-cnn": "Harder CNN",
    "harder-cnn+smcl": "Harder CNN+SmCL",
    "harder-gan+smcl": "Harder GAN+SmCL",
    "swinir-zeroshot": "SwinIR Zero-Shot",
    "swinir-zeroshot+addcl": "SwinIR ZS+AddCL",
    "swinir-finetuned": "SwinIR Finetuned",
    "swinir-finetuned+addcl": "SwinIR FT+AddCL",
}

COLOR_MAP: dict[str, str] = {
    "flow-wide96-amp (28M)": "#1f77b4",
    "flow-uniform-amp (13M)": "#4393c3",
    "flow-logitnorm-ema (13M)": "#92c5de",
    "flow-v2-zscore (13M)": "#6baed6",
    "bicubic+addcl": "#ff7f0e",
    "bilinear+addcl": "#ffbb78",
    "bicubic": "#aec7e8",
    "bilinear": "#c7c7c7",
    "harder-cnn": "#d62728",
    "harder-cnn+smcl": "#e377c2",
    "harder-gan+smcl": "#9467bd",
    "swinir-zeroshot": "#8c564b",
    "swinir-zeroshot+addcl": "#bcbd22",
    "swinir-finetuned": "#17becf",
    "swinir-finetuned+addcl": "#2ca02c",
}


def _method_display_names(names: list[str]) -> list[str]:
    return [DISPLAY_NAMES.get(n, n) for n in names]


def _method_colors(names: list[str]) -> list[str]:
    return [COLOR_MAP.get(n, "#888888") for n in names]


def plot_crps_comparison(
    results: dict[str, dict[str, float]],
    output_path: str | Path | None = None,
    title: str = "CRPS Comparison — ERA5 TCW 4x Downscaling",
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Bar chart comparing CRPS across methods, sorted best-to-worst."""
    sorted_methods = sorted(results.keys(), key=lambda m: results[m]["crps"])
    crps_vals = [results[m]["crps"] for m in sorted_methods]
    labels = _method_display_names(sorted_methods)
    colors = _method_colors(sorted_methods)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(sorted_methods))
    bars = ax.bar(x, crps_vals, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, crps_vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=40, ha="right")
    ax.set_ylabel("CRPS (lower is better)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, max(crps_vals) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics_panel(
    results: dict[str, dict[str, float]],
    output_path: str | Path | None = None,
    title: str = "ERA5 TCW 4x Downscaling \u2014 Method Comparison",
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """2x2 panel: CRPS, MAE, RMSE, Mass Violation across all methods."""
    sorted_methods = sorted(results.keys(), key=lambda m: results[m]["crps"])
    labels = _method_display_names(sorted_methods)
    colors = _method_colors(sorted_methods)
    x = np.arange(len(sorted_methods))

    metrics = [
        ("crps", "CRPS", False),
        ("mae", "MAE", False),
        ("rmse", "RMSE", False),
        ("mass_violation", "Mass Violation", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, (key, ylabel, use_log) in zip(axes.flat, metrics, strict=True):
        vals = [results[m][key] for m in sorted_methods]
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)

        for bar, val in zip(bars, vals, strict=True):
            label_text = f"{val:.4f}" if val > 0.001 else f"{val:.1e}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label_text,
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45 if use_log else 0,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=40, ha="right")
        ax.set_ylabel(ylabel, fontsize=10)
        if use_log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, max(vals) * 1.2)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_flow_vs_baseline(
    results: dict[str, dict[str, float]],
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Grouped bar chart: flow models vs best baseline on CRPS/MAE/RMSE."""
    flow_methods = [m for m in results if m.startswith("flow-")]
    baseline_methods = [m for m in results if not m.startswith("flow-")]

    if not flow_methods or not baseline_methods:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need both flow and baseline methods", ha="center", va="center")
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    best_flow = min(flow_methods, key=lambda m: results[m]["crps"])
    best_baseline = min(baseline_methods, key=lambda m: results[m]["crps"])

    metric_keys = ["crps", "mae", "rmse"]
    metric_labels = ["CRPS", "MAE", "RMSE"]

    flow_vals = [results[best_flow][k] for k in metric_keys]
    baseline_vals = [results[best_baseline][k] for k in metric_keys]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(metric_keys))
    width = 0.35

    ax.bar(
        x - width / 2,
        flow_vals,
        width,
        label=best_flow,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        baseline_vals,
        width,
        label=best_baseline,
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.5,
    )

    for i, (fv, bv) in enumerate(zip(flow_vals, baseline_vals, strict=True)):
        ax.text(i - width / 2, fv + 0.005, f"{fv:.4f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, bv + 0.005, f"{bv:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Value (lower is better)", fontsize=11)
    ax.set_title("Best Flow Model vs Best Baseline", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Dual-dataset comparison
# ---------------------------------------------------------------------------

DATASET_LABELS: dict[str, str] = {
    "era5": "ERA5 TCW (4x, 32\u2192128)",
    "noresm": "NorESM TAS (2x, 32\u219264)",
}


def _bar_panel(
    ax: plt.Axes,
    methods: list[str],
    values: list[float],
    *,
    ylabel: str,
    use_log: bool = False,
    show_values: bool = True,
    fontsize_val: int = 7,
) -> None:
    """Draw a single bar-chart sub-panel (shared helper)."""
    labels = _method_display_names(methods)
    colors = _method_colors(methods)
    x = np.arange(len(methods))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    if show_values:
        for bar, val in zip(bars, values, strict=True):
            text = f"{val:.4f}" if val > 0.001 else f"{val:.1e}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=fontsize_val,
                rotation=45 if use_log else 0,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=40, ha="right")
    ax.set_ylabel(ylabel, fontsize=10)
    if use_log:
        ax.set_yscale("log")
    elif values:
        ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", alpha=0.3)


def plot_dual_crps(
    results_a: dict[str, dict[str, float]],
    results_b: dict[str, dict[str, float]],
    label_a: str = "ERA5 TCW (4x)",
    label_b: str = "NorESM TAS (2x)",
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (16, 5),
) -> plt.Figure:
    """Side-by-side CRPS bar charts for two datasets.

    Only shows methods present in both datasets so the comparison is fair.
    """
    shared = sorted(set(results_a) & set(results_b))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize)

    methods_a = sorted(shared, key=lambda m: results_a[m]["crps"])
    methods_b = sorted(shared, key=lambda m: results_b[m]["crps"])

    _bar_panel(ax_a, methods_a, [results_a[m]["crps"] for m in methods_a], ylabel="CRPS")
    ax_a.set_title(label_a, fontsize=12)

    _bar_panel(ax_b, methods_b, [results_b[m]["crps"] for m in methods_b], ylabel="CRPS")
    ax_b.set_title(label_b, fontsize=12)

    fig.suptitle("CRPS Comparison Across Datasets (lower is better)", fontsize=14, y=1.02)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_dual_metrics_panel(
    results_a: dict[str, dict[str, float]],
    results_b: dict[str, dict[str, float]],
    label_a: str = "ERA5 TCW (4x)",
    label_b: str = "NorESM TAS (2x)",
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (18, 24),
) -> plt.Figure:
    """8-row x 2-col panel: all 8 metrics for two datasets side-by-side."""
    shared = sorted(set(results_a) & set(results_b))

    # All 8 metrics: (key, ylabel, use_log, direction)
    metrics: list[tuple[str, str, bool, str]] = [
        ("crps", "CRPS", False, "lower"),
        ("mae", "MAE", False, "lower"),
        ("rmse", "RMSE", False, "lower"),
        ("ssim", "SSIM", False, "higher"),
        ("psnr", "PSNR (dB)", False, "higher"),
        ("ralsd", "RALSD (dB)", False, "lower"),
        ("emd", "EMD", False, "lower"),
        ("mass_violation", "Mass Violation", True, "lower"),
    ]

    # Filter to metrics that exist in the data
    available_metrics = [
        (key, ylabel, use_log, direction)
        for key, ylabel, use_log, direction in metrics
        if any(key in results_a.get(m, {}) for m in shared)
        or any(key in results_b.get(m, {}) for m in shared)
    ]

    n_rows = len(available_metrics)
    fig, axes = plt.subplots(n_rows, 2, figsize=(figsize[0], 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (key, ylabel, use_log, direction) in enumerate(available_metrics):
        better = "lower" if direction == "lower" else "higher"
        label_with_dir = f"{ylabel} ({better} is better)"
        for col, (results, label) in enumerate([(results_a, label_a), (results_b, label_b)]):
            methods = sorted(shared, key=lambda m, r=results: r[m]["crps"])
            vals = [results[m].get(key, 0.0) for m in methods]
            ax = axes[row, col]
            _bar_panel(ax, methods, vals, ylabel=label_with_dir, use_log=use_log)
            if row == 0:
                ax.set_title(label, fontsize=12)

    fig.suptitle("Method Comparison \u2014 ERA5 vs NorESM (All Metrics)", fontsize=14, y=1.01)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_constraint_impact(
    results_a: dict[str, dict[str, float]],
    results_b: dict[str, dict[str, float]],
    label_a: str = "ERA5 TCW (4x)",
    label_b: str = "NorESM TAS (2x)",
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Grouped bars showing CRPS change when constraints are applied.

    Highlights how AddCL/SmCL affect each dataset differently:
    constraints help ERA5 (same-source LR/HR) but hurt NorESM (cross-simulation).
    """
    constraint_pairs = [
        ("swinir-finetuned", "swinir-finetuned+addcl", "SwinIR FT\n+AddCL"),
        ("harder-cnn", "harder-cnn+smcl", "Harder CNN\n+SmCL"),
        ("bicubic", "bicubic+addcl", "Bicubic\n+AddCL"),
        ("bilinear", "bilinear+addcl", "Bilinear\n+AddCL"),
    ]

    valid_pairs = [
        (base, constr, lbl)
        for base, constr, lbl in constraint_pairs
        if base in results_a and constr in results_a and base in results_b and constr in results_b
    ]

    x = np.arange(len(valid_pairs))
    width = 0.35

    delta_a = [results_a[c]["crps"] - results_a[b]["crps"] for b, c, _ in valid_pairs]
    delta_b = [results_b[c]["crps"] - results_b[b]["crps"] for b, c, _ in valid_pairs]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        x - width / 2,
        delta_a,
        width,
        label=label_a,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        delta_b,
        width,
        label=label_b,
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.5,
    )

    for i, (da, db) in enumerate(zip(delta_a, delta_b, strict=True)):
        ax.text(
            i - width / 2,
            da + (0.002 if da >= 0 else -0.015),
            f"{da:+.4f}",
            ha="center",
            va="bottom" if da >= 0 else "top",
            fontsize=8,
        )
        ax.text(
            i + width / 2,
            db + (0.002 if db >= 0 else -0.015),
            f"{db:+.4f}",
            ha="center",
            va="bottom" if db >= 0 else "top",
            fontsize=8,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, _, lbl in valid_pairs], fontsize=10)
    ax.set_ylabel("\u0394 CRPS (constrained \u2212 unconstrained)", fontsize=11)
    ax.set_title("Constraint Impact on CRPS\n(positive = constraint hurts)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
