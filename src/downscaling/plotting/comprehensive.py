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

# Canonical model families — both panels show the same set in the same order/color
MODEL_FAMILIES = [
    "Bicubic",
    "CNN",
    "CNN+SmCL (train)",
    "GAN",
    "GAN+SmCL (train)",
    "Flow",
    "SwinIR",
    "Truth+AddCL",
]
MODEL_COLORS = {
    "Bicubic": "#666666",
    "CNN": "#e41a1c",
    "CNN+SmCL (train)": "#984ea3",
    "GAN": "#a6761d",
    "GAN+SmCL (train)": "#66a61e",
    "Flow": "#377eb8",
    "SwinIR": "#ff7f00",
    "Truth+AddCL": "#999999",
}

# Models with RALSD above this threshold are considered diverged and excluded from plots
BROKEN_MODEL_RALSD_THRESHOLD = 10.0


def _format_val(v: float, precision: int = 4) -> str:
    """Format value, collapsing -0.0000 to 0.0000."""
    if abs(v) < 0.5 * 10 ** (-precision):
        v = 0.0
    return f"{v:.{precision}f}"

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


# Map legacy result names to canonical MODEL_FAMILIES names
_NAME_MAP: dict[str, str] = {
    "CNN(none)": "CNN",
    "Flow(none)": "Flow",
}


def _load(path: Path) -> dict[str, dict[str, object]]:
    with open(path) as f:
        raw: dict[str, dict[str, object]] = json.load(f)
    return {_NAME_MAP.get(k, k): v for k, v in raw.items()}


def _filter_broken(results: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """Remove models with clearly diverged metrics (RALSD > threshold)."""
    for name, r in results.items():
        if "ralsd" not in r:
            raise KeyError(f"Model '{name}' is missing 'ralsd' metric — results may be stale")
    return {
        name: r
        for name, r in results.items()
        if float(r["ralsd"]) < BROKEN_MODEL_RALSD_THRESHOLD  # type: ignore[arg-type]
    }


# ---------------------------------------------------------------------------
# 1. PSD comparison — side by side
# ---------------------------------------------------------------------------


def _plot_psd_comparison_axes(
    noresm: dict[str, dict[str, object]],
    era5: dict[str, dict[str, object]],
    log_x: bool,
) -> plt.Figure:
    fig, (ax_n, ax_e) = plt.subplots(1, 2, figsize=(18, 7))

    for ax, results, title in [
        (ax_n, noresm, "NorESM TAS 2x SR"),
        (ax_e, era5, "ERA5 TCW 4x SR"),
    ]:
        if not results:
            ax.text(0.5, 0.5, "No models", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        first = next(iter(results.values()))
        k = np.array(first["psd_k"])
        truth_power = np.array(first["psd_truth_power"])

        ax.axhline(y=1.0, color="k", linewidth=2, linestyle="-", label="Truth", zorder=10)

        for name in MODEL_FAMILIES:
            if name not in results:
                continue
            r = results[name]
            power = np.array(r["psd_power"])
            ralsd_val = r["ralsd"]
            ratio = power / truth_power
            plot_fn = ax.semilogx if log_x else ax.plot
            plot_fn(
                k,
                ratio,
                "-",
                color=MODEL_COLORS[name],
                linewidth=1.5,
                label=f"{name} (RALSD={ralsd_val:.2f})",
            )

        ax.set_xlabel("Wavenumber k", fontsize=11)
        ax.set_ylabel("Power / Truth", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    scale_label = "Log" if log_x else "Linear"
    fig.suptitle(
        f"Radially Averaged PSD Ratio (Model / Truth) — {scale_label} Scale",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_psd_comparison(
    noresm: dict[str, dict[str, object]],
    era5: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    stem = output_path.stem
    parent = output_path.parent
    suffix = output_path.suffix

    for log_x, tag in [(True, "_log"), (False, "_linear")]:
        fig = _plot_psd_comparison_axes(noresm, era5, log_x=log_x)
        path = parent / f"{stem}{tag}{suffix}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


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

    for row, (ens_models, ds_label) in enumerate([(n_ens, "NorESM"), (e_ens, "ERA5")]):
        for col, (name, r) in enumerate(ens_models.items()):
            ax = axes[row, col]
            rh = np.array(r["rank_histogram"], dtype=float)
            n_bins = len(rh)
            uniform = rh.sum() / n_bins
            ax.bar(
                range(n_bins),
                rh / uniform,
                color="steelblue",
                edgecolor="white",
                linewidth=0.5,
            )
            ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="Uniform")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Count / Expected")
            ssr = r.get("ssr", float("nan"))
            ax.set_title(f"{ds_label}: {name}\nSSR={ssr:.3f}", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")

        # hide unused columns
        for col in range(len(ens_models), n_cols):
            axes[row, col].set_visible(False)

    fig.suptitle("Rank Histograms (SSR=1.0 is well-calibrated)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


QUALITY_METRICS: list[tuple[str, str, bool]] = [
    ("crps", "CRPS", False),
    ("mae", "MAE", False),
    ("rmse", "RMSE", False),
    ("mass_violation", "Mass Violation", False),
    ("ssim", "SSIM", True),
    ("kl_divergence", "KL Divergence", False),
]


# ---------------------------------------------------------------------------
# 6. Calibration panel — SSR for ensemble models
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
            ax.text(
                0.5, 0.5, "No ensemble models", ha="center", va="center", transform=ax.transAxes
            )
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
# 7. Individual metric plots — one per metric, both datasets as grouped bars
# ---------------------------------------------------------------------------

_DATASET_COLORS = {"NorESM TAS 2x": "#377eb8", "ERA5 TCW 4x": "#e41a1c"}


def plot_single_metric(
    noresm: dict[str, dict[str, object]],
    era5: dict[str, dict[str, object]],
    metric_key: str,
    metric_display: str,
    higher_better: bool,
    output_path: Path,
) -> None:
    """Side-by-side vertical bar charts (left=NorESM, right=ERA5), each sorted independently."""
    direction = "higher is better" if higher_better else "lower is better"

    def _sort_key(m: str, results: dict[str, dict[str, object]]) -> float:
        v = float(results[m][metric_key])  # type: ignore[index]
        return -v if higher_better else v

    fig, (ax_n, ax_e) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, results, ds_label, color in [
        (ax_n, noresm, "NorESM TAS 2x", _DATASET_COLORS["NorESM TAS 2x"]),
        (ax_e, era5, "ERA5 TCW 4x", _DATASET_COLORS["ERA5 TCW 4x"]),
    ]:
        methods = sorted(results.keys(), key=lambda m, r=results: _sort_key(m, r))
        vals = [float(results[m][metric_key]) for m in methods]  # type: ignore[index]
        x = np.arange(len(methods))

        ax.bar(x, vals, color=color, edgecolor="white", linewidth=0.5)

        for i, v in enumerate(vals):
            ax.text(i, v, f" {_format_val(v)}", ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9, rotation=30, ha="right")
        ax.set_title(ds_label, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        if all(v >= 0 for v in vals):
            ax.set_ylim(0, max(vals) * 1.25)

    ax_n.set_ylabel(f"{metric_display} ({direction})", fontsize=11)
    fig.suptitle(metric_display, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_individual_metrics(
    noresm: dict[str, dict[str, object]],
    era5: dict[str, dict[str, object]],
    output_dir: Path,
) -> None:
    """Generate one plot per metric in output_dir/metrics/."""
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = QUALITY_METRICS + [
        ("psd_log_ratio", "PSD Log-Ratio", False),
        ("ralsd", "RALSD (dB)", False),
        ("spectral_coherence", "Spectral Coherence", True),
    ]

    for key, display, higher_better in all_metrics:
        slug = key.replace("_", "-")
        plot_single_metric(
            noresm, era5, key, display, higher_better,
            metrics_dir / f"{slug}.png",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_all_figures(output_dir: Path) -> None:
    """Load cached results and generate all diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cached results...")
    noresm_raw = _load(NORESM_RESULTS)
    era5_raw = _load(ERA5_RESULTS)

    # Filter broken models from both datasets (RALSD > 10 threshold)
    noresm = _filter_broken(noresm_raw)
    era5 = _filter_broken(era5_raw)
    for label, raw, filtered in [("NorESM", noresm_raw, noresm), ("ERA5", era5_raw, era5)]:
        n_filtered = len(raw) - len(filtered)
        if n_filtered:
            removed = set(raw) - set(filtered)
            print(f"  Filtered {n_filtered} broken {label} model(s): {removed}")

    print(f"  NorESM: {len(noresm)} models, ERA5: {len(era5)} models")

    print("\nGenerating plots...")
    plot_psd_comparison(noresm, era5, output_dir / "psd_comparison.png")
    plot_rank_histograms(noresm, era5, output_dir / "rank_histograms.png")
    plot_individual_metrics(noresm, era5, output_dir)
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
