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


def _rank_methods_by_spectral_distance(
    psd_truth: NDArray[np.floating],
    method_psds: dict[str, NDArray[np.floating]],
) -> list[str]:
    """Rank methods by mean absolute log-spectral distance from ground truth."""
    eps = 1e-30
    log_truth = np.log10(np.maximum(psd_truth, eps))
    distances: dict[str, float] = {}
    for name, psd in method_psds.items():
        log_pred = np.log10(np.maximum(psd, eps))
        distances[name] = float(np.mean(np.abs(log_truth - log_pred)))
    return sorted(distances, key=lambda n: distances[n])


def plot_psd_comparison(
    freq: NDArray[np.floating],
    psd_truth: NDArray[np.floating],
    method_psds: dict[str, NDArray[np.floating]],
    output_path: str | Path | None = None,
    title: str = "Radially-Averaged Power Spectral Density",
    figsize: tuple[float, float] = (10, 8),
) -> plt.Figure:
    """Log-log PSD curves with ratio subplot for readability.

    When >5 methods are present, uses a 2-panel layout:
    - Top: PSD curves with 3 best + 1 worst highlighted, others dimmed
    - Bottom: PSD ratio (pred/truth) in dB, clearly showing deviations

    Args:
        freq: Frequency bin centers, shape (n_bins,).
        psd_truth: Ground truth PSD, shape (n_bins,).
        method_psds: Dict mapping method name -> PSD array (n_bins,).
        output_path: Optional save path.
        title: Plot title.
        figsize: Figure size.
    """
    n_methods = len(method_psds)
    use_ratio = n_methods > 5

    if use_ratio:
        fig, (ax_psd, ax_ratio) = plt.subplots(
            2, 1, figsize=figsize, height_ratios=[1.2, 1], sharex=True
        )
    else:
        fig, ax_psd = plt.subplots(figsize=(figsize[0], figsize[1] * 0.6))
        ax_ratio = None

    # Identify highlight methods: 3 best + 1 worst by spectral distance
    ranked = _rank_methods_by_spectral_distance(psd_truth, method_psds)
    if n_methods > 5:
        highlight = set(ranked[:3] + ranked[-1:])
    else:
        highlight = set(ranked)

    # Top panel: PSD
    ax_psd.loglog(freq, psd_truth, "k-", linewidth=2.5, label="Ground Truth", zorder=10)

    # Dimmed methods first (behind)
    for name, psd in method_psds.items():
        if name not in highlight:
            ax_psd.loglog(
                freq, psd, linewidth=0.8, color="#cccccc", alpha=0.5, zorder=1
            )

    # Highlighted methods on top
    for name in ranked:
        if name in highlight:
            ax_psd.loglog(
                freq,
                method_psds[name],
                linewidth=2.0,
                color=_color(name),
                label=_display(name),
                alpha=0.9,
                zorder=5,
            )

    ax_psd.set_ylabel("Power Spectral Density", fontsize=11)
    ax_psd.set_title(title, fontsize=13)
    ax_psd.legend(fontsize=8, loc="lower left", ncol=2)
    ax_psd.grid(True, alpha=0.3, which="both")

    # Bottom panel: PSD ratio in dB
    if ax_ratio is not None:
        eps = 1e-30
        ax_ratio.axhline(0, color="black", linewidth=0.8, linestyle="--")

        for name in ranked:
            ratio_db = 10.0 * np.log10(
                np.maximum(method_psds[name], eps) / np.maximum(psd_truth, eps)
            )
            lw = 1.8 if name in highlight else 0.7
            alpha = 0.85 if name in highlight else 0.35
            color = _color(name) if name in highlight else "#aaaaaa"
            zorder = 5 if name in highlight else 1
            label = _display(name) if name in highlight else None
            ax_ratio.plot(
                freq, ratio_db, linewidth=lw, color=color, alpha=alpha,
                zorder=zorder, label=label,
            )

        ax_ratio.set_xlabel("Spatial Frequency (cycles/pixel)", fontsize=11)
        ax_ratio.set_ylabel("PSD Ratio (dB)\npred / truth", fontsize=10)
        ax_ratio.set_xscale("log")
        ax_ratio.grid(True, alpha=0.3)
        ax_ratio.legend(fontsize=7, loc="best", ncol=2)
    else:
        ax_psd.set_xlabel("Spatial Frequency (cycles/pixel)", fontsize=11)

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
