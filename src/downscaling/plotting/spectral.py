"""Spectral power and extended metric visualization.

PSD comparison plots, spectral bias charts, and updated metric panels
that include RALSD, SSIM, PSNR, and EMD alongside the original four metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

from downscaling.plotting.metrics import COLOR_MAP, DISPLAY_NAMES


def _display(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def _color(name: str) -> str:
    return COLOR_MAP.get(name, "#888888")


def plot_psd_comparison(
    freq: NDArray[np.floating],
    psd_truth: NDArray[np.floating],
    method_psds: dict[str, NDArray[np.floating]],
    output_path: str | Path | None = None,
    title: str = "Radially-Averaged Power Spectral Density",
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """Log-log PSD curves for ground truth vs multiple methods.

    Args:
        freq: Frequency bin centers, shape (n_bins,).
        psd_truth: Ground truth PSD, shape (n_bins,).
        method_psds: Dict mapping method name -> PSD array (n_bins,).
        output_path: Optional save path.
        title: Plot title.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.loglog(freq, psd_truth, "k-", linewidth=2.5, label="Ground Truth", zorder=10)

    for name, psd in method_psds.items():
        ax.loglog(
            freq,
            psd,
            linewidth=1.5,
            color=_color(name),
            label=_display(name),
            alpha=0.85,
        )

    ax.set_xlabel("Spatial Frequency (cycles/pixel)", fontsize=11)
    ax.set_ylabel("Power Spectral Density", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_spectral_bias(
    freq: NDArray[np.floating],
    method_biases: dict[str, NDArray[np.floating]],
    output_path: str | Path | None = None,
    title: str = "Spectral Bias (positive = underestimates power, too smooth)",
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Per-frequency spectral bias for each method.

    Args:
        freq: Frequency bin centers, shape (n_bins,).
        method_biases: Dict mapping method name -> bias array (n_bins,) in dB.
            Positive = prediction too smooth, negative = too noisy.
        output_path: Optional save path.
        title: Plot title.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for name, bias in method_biases.items():
        valid = ~np.isnan(bias)
        ax.plot(
            freq[valid],
            bias[valid],
            linewidth=1.5,
            color=_color(name),
            label=_display(name),
            alpha=0.85,
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("Spatial Frequency (cycles/pixel)", fontsize=11)
    ax.set_ylabel("Spectral Bias (dB)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_extended_metrics_panel(
    results: dict[str, dict[str, float]],
    output_path: str | Path | None = None,
    title: str = "ERA5 TCW 4x Downscaling — Full Metric Comparison",
    figsize: tuple[float, float] = (18, 14),
) -> plt.Figure:
    """3-row x 3-col panel: all 8 metrics.

    Layout:
        Row 1: CRPS, MAE, RMSE
        Row 2: SSIM, PSNR, RALSD
        Row 3: EMD, Mass Violation, (empty)
    """
    sorted_methods = sorted(results.keys(), key=lambda m: results[m].get("crps", float("inf")))
    labels = [DISPLAY_NAMES.get(n, n) for n in sorted_methods]
    colors = [COLOR_MAP.get(n, "#888888") for n in sorted_methods]
    x = np.arange(len(sorted_methods))

    metrics = [
        ("crps", "CRPS", False, "lower"),
        ("mae", "MAE", False, "lower"),
        ("rmse", "RMSE", False, "lower"),
        ("ssim", "SSIM", False, "higher"),
        ("psnr", "PSNR (dB)", False, "higher"),
        ("ralsd", "RALSD (dB)", False, "lower"),
        ("emd", "EMD", False, "lower"),
        ("mass_violation", "Mass Violation", True, "lower"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=figsize)

    for idx, (key, ylabel, use_log, direction) in enumerate(metrics):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        vals = [results[m].get(key, 0.0) for m in sorted_methods]
        has_data = any(v != 0.0 for v in vals)
        if not has_data:
            ax.text(0.5, 0.5, f"No {key} data", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel(ylabel, fontsize=10)
            continue

        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)

        for bar, val in zip(bars, vals, strict=True):
            text = f"{val:.4f}" if val > 0.001 else f"{val:.1e}"
            if key == "ralsd":
                text = f"{val:.2f}"
            elif key == "psnr":
                text = f"{val:.1f}"
            elif key == "emd":
                text = f"{val:.4f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45 if use_log else 0,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=40, ha="right")
        better = "lower" if direction == "lower" else "higher"
        ax.set_ylabel(f"{ylabel} ({better} is better)", fontsize=9)
        if use_log:
            ax.set_yscale("log")
        elif vals:
            ax.set_ylim(0, max(vals) * 1.2)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for idx in range(len(metrics), 9):
        row, col = divmod(idx, 3)
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ralsd_comparison(
    results: dict[str, dict[str, float]],
    output_path: str | Path | None = None,
    title: str = "RALSD Comparison (lower is better)",
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Bar chart of RALSD values, sorted best-to-worst.

    Only includes methods that have RALSD computed.
    """
    methods_with_ralsd = {
        m: r for m, r in results.items() if "ralsd" in r and r["ralsd"] is not None
    }
    if not methods_with_ralsd:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No RALSD data available", ha="center", va="center", transform=ax.transAxes
        )
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    sorted_methods = sorted(methods_with_ralsd.keys(), key=lambda m: methods_with_ralsd[m]["ralsd"])
    vals = [methods_with_ralsd[m]["ralsd"] for m in sorted_methods]
    labels = [DISPLAY_NAMES.get(n, n) for n in sorted_methods]
    colors = [COLOR_MAP.get(n, "#888888") for n in sorted_methods]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(sorted_methods))
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=40, ha="right")
    ax.set_ylabel("RALSD (dB, lower is better)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
