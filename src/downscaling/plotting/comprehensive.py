"""Consistent diagnostic plots from comprehensive evaluation results.

Reads cached JSON results and generates side-by-side plots for NorESM (2x SR)
and ERA5 (4x SR) datasets with the full 11-metric set.

Run: python -m downscaling.plotting.comprehensive [--output-dir DIR]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POOL = Path("/home/chenxy/orcd/pool/datasets")

NORESM_RESULTS = POOL / "metrics" / "noresm" / "comprehensive_results.json"
ERA5_RESULTS = POOL / "metrics" / "era5" / "era5_comprehensive_results.json"

# Metrics displayed in summary. (key, display_name, higher_is_better)
SCALAR_METRICS: list[tuple[str, str, bool]] = [
    ("crps", "CRPS", False),
    ("mae", "MAE", False),
    ("rmse", "RMSE", False),
    ("mass_violation", "Mass Violation", False),
    ("ssim", "SSIM", True),
    ("kl_divergence", "KL Divergence", False),
    ("psd_log_ratio", "PSD Log-Ratio", False),
    ("ralsd", "RALSD", False),
    ("spectral_coherence", "Spectral Coherence", True),
]


def _load(path: Path) -> dict[str, dict[str, object]]:
    with open(path) as f:
        return json.load(f)


def _filter_broken(results: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """Remove models with clearly diverged metrics (e.g. RALSD > 10)."""
    return {
        name: r
        for name, r in results.items()
        if "ralsd" in r and float(r["ralsd"]) < 10  # type: ignore[arg-type]
    }


# ---------------------------------------------------------------------------
# 1. PSD comparison — side by side
# ---------------------------------------------------------------------------


def plot_psd_comparison(
    noresm: dict[str, dict[str, object]],
    era5: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    fig, (ax_n, ax_e) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, results, title in [
        (ax_n, noresm, "NorESM TAS 2x SR"),
        (ax_e, era5, "ERA5 TCW 4x SR"),
    ]:
        first = next(iter(results.values()))
        k = np.array(first["psd_k"])
        truth_power = np.array(first["psd_truth_power"])
        ax.loglog(k, truth_power, "k-", linewidth=2.5, label="Truth", zorder=10)

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))  # type: ignore[attr-defined]
        for (name, r), color in zip(results.items(), colors, strict=False):
            power = np.array(r["psd_power"])
            ralsd_val = r["ralsd"]
            ax.loglog(
                k,
                power,
                "-",
                color=color,
                linewidth=1.5,
                label=f"{name} (RALSD={ralsd_val:.2f})",
            )

        ax.set_xlabel("Wavenumber k", fontsize=11)
        ax.set_ylabel("Power", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Radially Averaged Power Spectral Density", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# 2. Rank histograms — side by side
# ---------------------------------------------------------------------------


def plot_rank_histograms(
    noresm: dict[str, dict[str, object]],
    era5: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    n_ens = {name: r for name, r in noresm.items() if "rank_histogram" in r}
    e_ens = {name: r for name, r in era5.items() if "rank_histogram" in r}

    n_cols = max(len(n_ens), len(e_ens))
    if n_cols == 0:
        return

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for row, (ens_models, ds_label) in enumerate(
        [(n_ens, "NorESM"), (e_ens, "ERA5")]
    ):
        for col, (name, r) in enumerate(ens_models.items()):
            ax = axes[row, col]
            rh = np.array(r["rank_histogram"])
            n_bins = len(rh)
            uniform = rh.sum() / n_bins
            ax.bar(
                range(n_bins),
                rh,
                color="steelblue",
                edgecolor="white",
                linewidth=0.5,
            )
            ax.axhline(
                y=uniform, color="red", linestyle="--", linewidth=1.5, label="Uniform"
            )
            ax.set_xlabel("Rank")
            ax.set_ylabel("Count")
            ssr = r.get("ssr", float("nan"))
            ax.set_title(f"{ds_label}: {name}\nSSR={ssr:.3f}", fontsize=10)
            ax.legend(fontsize=8)

        # hide unused columns
        for col in range(len(ens_models), n_cols):
            axes[row, col].set_visible(False)

    fig.suptitle("Rank Histograms (SSR=1.0 is well-calibrated)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# 3. Full metrics summary — side-by-side bar charts
# ---------------------------------------------------------------------------


def plot_metrics_summary(
    noresm: dict[str, dict[str, object]],
    era5: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    metrics = SCALAR_METRICS
    n_metrics = len(metrics)

    fig, axes = plt.subplots(n_metrics, 2, figsize=(16, 3 * n_metrics))

    for col, (results, ds_label) in enumerate(
        [(noresm, "NorESM TAS 2x SR"), (era5, "ERA5 TCW 4x SR")]
    ):
        names = list(results.keys())
        n_models = len(names)

        for row, (key, display, higher_better) in enumerate(metrics):
            ax = axes[row, col]
            for name in names:
                if key not in results[name]:
                    raise KeyError(f"Metric '{key}' missing from results for model '{name}'")
            vals = [float(results[name][key]) for name in names]  # type: ignore[index]
            direction = "higher is better" if higher_better else "lower is better"

            # Color best model green
            best_idx = int(np.argmax(vals)) if higher_better else int(np.argmin(vals))
            colors = ["steelblue"] * n_models
            colors[best_idx] = "#2ca02c"

            ax.barh(range(n_models), vals, color=colors, edgecolor="white")
            ax.set_yticks(range(n_models))
            ax.set_yticklabels(names, fontsize=8)
            ax.invert_yaxis()

            # Value labels
            for i, v in enumerate(vals):
                ax.text(v, i, f" {v:.4f}", va="center", fontsize=7)

            if col == 0:
                ax.set_ylabel(f"{display}\n({direction})", fontsize=9)
            if row == 0:
                ax.set_title(ds_label, fontsize=12)

    fig.suptitle(
        "Comprehensive Metric Comparison — All Models",
        fontsize=14,
        y=1.005,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# 4. Spectral quality panel — PSD-LR, RALSD, Coherence
# ---------------------------------------------------------------------------


def plot_spectral_panel(
    noresm: dict[str, dict[str, object]],
    era5: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    spectral_keys = [
        ("psd_log_ratio", "PSD Log-Ratio", False),
        ("ralsd", "RALSD (dB)", False),
        ("spectral_coherence", "Spectral Coherence", True),
    ]
    fig, axes = plt.subplots(len(spectral_keys), 2, figsize=(14, 3.5 * len(spectral_keys)))

    for col, (results, ds_label) in enumerate(
        [(noresm, "NorESM TAS 2x"), (era5, "ERA5 TCW 4x")]
    ):
        names = list(results.keys())
        n_models = len(names)
        for row, (key, display, higher_better) in enumerate(spectral_keys):
            ax = axes[row, col]
            for name in names:
                if key not in results[name]:
                    raise KeyError(f"Metric '{key}' missing from results for model '{name}'")
            vals = [float(results[name][key]) for name in names]  # type: ignore[index]
            best_idx = int(np.argmax(vals)) if higher_better else int(np.argmin(vals))
            colors = ["steelblue"] * n_models
            colors[best_idx] = "#2ca02c"

            ax.barh(range(n_models), vals, color=colors, edgecolor="white")
            ax.set_yticks(range(n_models))
            ax.set_yticklabels(names, fontsize=8)
            ax.invert_yaxis()
            for i, v in enumerate(vals):
                ax.text(v, i, f" {v:.4f}", va="center", fontsize=7)
            if col == 0:
                direction = "higher=better" if higher_better else "lower=better"
                ax.set_ylabel(f"{display}\n({direction})", fontsize=9)
            if row == 0:
                ax.set_title(ds_label, fontsize=12)

    fig.suptitle(
        "Spectral Quality Metrics — NorESM vs ERA5",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# 5. Calibration panel — SSR for ensemble models
# ---------------------------------------------------------------------------


def plot_calibration_panel(
    noresm: dict[str, dict[str, object]],
    era5: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    fig, (ax_n, ax_e) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, results, ds_label in [
        (ax_n, noresm, "NorESM TAS 2x"),
        (ax_e, era5, "ERA5 TCW 4x"),
    ]:
        ens_models = {k: v for k, v in results.items() if "ssr" in v}
        if not ens_models:
            ax.text(0.5, 0.5, "No ensemble models", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(ds_label)
            continue

        names = list(ens_models.keys())
        ssrs = [float(ens_models[n]["ssr"]) for n in names]  # type: ignore[arg-type]
        colors = ["#2ca02c" if 0.8 <= s <= 1.2 else "#d62728" for s in ssrs]

        ax.barh(range(len(names)), ssrs, color=colors, edgecolor="white")
        ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1.5, label="Perfect (1.0)")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        for i, v in enumerate(ssrs):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        ax.set_xlabel("Spread-Skill Ratio")
        ax.set_title(ds_label, fontsize=12)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Ensemble Calibration: Spread-Skill Ratio\n(1.0 = well-calibrated, <1 = underdispersive, >1 = overdispersive)",
        fontsize=13,
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_all_figures(output_dir: Path) -> None:
    """Load cached results and generate all diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cached results...")
    noresm = _load(NORESM_RESULTS)
    era5_raw = _load(ERA5_RESULTS)

    # Filter broken models (e.g. ResFlow-Heun with RALSD > 10)
    era5 = _filter_broken(era5_raw)
    n_filtered = len(era5_raw) - len(era5)
    if n_filtered:
        removed = set(era5_raw) - set(era5)
        print(f"  Filtered {n_filtered} broken model(s): {removed}")

    print(f"  NorESM: {len(noresm)} models, ERA5: {len(era5)} models")

    print("\nGenerating plots...")
    plot_psd_comparison(noresm, era5, output_dir / "psd_comparison.png")
    plot_rank_histograms(noresm, era5, output_dir / "rank_histograms.png")
    plot_metrics_summary(noresm, era5, output_dir / "metrics_summary.png")
    plot_spectral_panel(noresm, era5, output_dir / "spectral_metrics.png")
    plot_calibration_panel(noresm, era5, output_dir / "calibration.png")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    output_dir = Path("/workspace/figures")
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--output-dir":
            output_dir = Path(args[i + 1])
            i += 2
        else:
            i += 1

    generate_all_figures(output_dir)
