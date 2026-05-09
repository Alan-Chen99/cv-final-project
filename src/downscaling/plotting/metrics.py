"""Metric comparison plots from evaluation results.

Produces bar charts comparing CRPS, MAE, RMSE, and mass violation
across methods. Reads from eval results dicts (or JSON files).
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


def _method_display_names(names: list[str]) -> list[str]:
    """Shorten method names for plot labels."""
    replacements = {
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
    return [replacements.get(n, n) for n in names]


def _method_colors(names: list[str]) -> list[str]:
    """Assign colors: flow models = blue family, baselines = gray/orange."""
    color_map = {
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
    return [color_map.get(n, "#888888") for n in names]


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

    fig.suptitle("ERA5 TCW 4x Downscaling — Method Comparison", fontsize=14, y=1.01)
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
