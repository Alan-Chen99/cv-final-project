"""Visualization functions for downscaling results.

Two categories:
  - Output artifact plots: show what models produce (sample grids, ensemble members)
  - Data plots: compare methods quantitatively (metrics bar charts, constraint effects)
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_sample_grid(
    lr_up: np.ndarray,
    hr: np.ndarray,
    ensemble_preds: np.ndarray,
    indices: list[int],
    save_path: str | Path,
    method_name: str = "Flow Matching",
) -> None:
    """Plot grid comparing LR, HR, prediction, error, and spread.

    Columns: LR bilinear | HR ground truth | Ensemble mean | |Error| | Spread

    Args:
        lr_up: (N, 1, H, W) bilinear-upsampled LR
        hr: (N, 1, H, W) HR ground truth
        ensemble_preds: (N, M, 1, H, W) ensemble predictions
        indices: which samples to plot (row indices into the arrays)
        save_path: output PNG path
        method_name: display name for the method
    """
    n = len(indices)
    _fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    # Global color range from HR for consistent comparison
    vmin_global = min(float(hr[i, 0].min()) for i in indices)
    vmax_global = max(float(hr[i, 0].max()) for i in indices)

    for row, idx in enumerate(indices):
        lr_img = lr_up[idx, 0]
        hr_img = hr[idx, 0]
        ens = ensemble_preds[idx, :, 0]  # (M, H, W)
        mean_pred = ens.mean(axis=0)
        error = np.abs(hr_img - mean_pred)
        spread = ens.std(axis=0)

        axes[row, 0].imshow(lr_img, cmap="viridis", vmin=vmin_global, vmax=vmax_global)
        axes[row, 0].set_title(f"LR bilinear (#{idx})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(hr_img, cmap="viridis", vmin=vmin_global, vmax=vmax_global)
        axes[row, 1].set_title("HR ground truth")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(mean_pred, cmap="viridis", vmin=vmin_global, vmax=vmax_global)
        axes[row, 2].set_title(f"{method_name} mean")
        axes[row, 2].axis("off")

        im3 = axes[row, 3].imshow(error, cmap="hot")
        axes[row, 3].set_title(f"|Error| (MAE={error.mean():.3f})")
        axes[row, 3].axis("off")
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046)

        im4 = axes[row, 4].imshow(spread, cmap="YlOrRd")
        axes[row, 4].set_title(f"Spread (std={spread.mean():.4f})")
        axes[row, 4].axis("off")
        plt.colorbar(im4, ax=axes[row, 4], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ensemble_members(
    lr_up: np.ndarray,
    hr: np.ndarray,
    ensemble_preds: np.ndarray,
    sample_idx: int,
    save_path: str | Path,
    n_show: int = 5,
) -> None:
    """Show individual ensemble members for one sample.

    Columns: LR | HR | Member 1 | Member 2 | ... | Member N

    Args:
        lr_up: (N, 1, H, W) bilinear-upsampled LR
        hr: (N, 1, H, W) HR ground truth
        ensemble_preds: (N, M, 1, H, W) ensemble predictions
        sample_idx: which sample to visualize
        save_path: output PNG path
        n_show: how many ensemble members to display
    """
    ens = ensemble_preds[sample_idx, :, 0]  # (M, H, W)
    hr_img = hr[sample_idx, 0]
    lr_img = lr_up[sample_idx, 0]

    n_cols = min(n_show, ens.shape[0]) + 2  # +2 for LR and HR
    _fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    vmin = float(hr_img.min())
    vmax = float(hr_img.max())

    axes[0].imshow(lr_img, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("LR bilinear")
    axes[0].axis("off")

    axes[1].imshow(hr_img, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("HR truth")
    axes[1].axis("off")

    for i in range(min(n_show, ens.shape[0])):
        member_mae = float(np.mean(np.abs(ens[i] - hr_img)))
        axes[i + 2].imshow(ens[i], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i + 2].set_title(f"Member {i + 1}\nMAE={member_mae:.3f}")
        axes[i + 2].axis("off")

    plt.suptitle(f"Sample #{sample_idx}: Ensemble Members", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(
    results_path: str | Path,
    save_path: str | Path,
    metric: str = "crps",
    only_addcl: bool = True,
) -> None:
    """Bar chart comparing methods on a given metric.

    Args:
        results_path: path to eval results JSON
        save_path: output PNG path
        metric: which metric to plot ("crps", "mae", "rmse")
        only_addcl: if True, only show +AddCL variants (cleaner comparison)
    """
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    if only_addcl:
        results = [r for r in results if r["constraint"] == "addcl"]

    # Filter out NaN
    results = [r for r in results if r[metric] == r[metric]]  # filter NaN

    methods = [r["method"] for r in results]
    values = [r[metric] for r in results]

    # Color: deterministic baselines gray, flow models blue
    colors = []
    for m in methods:
        if m in ("bilinear", "bicubic"):
            colors.append("#999999")
        else:
            colors.append("#4C72B0")

    _fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(methods)), values, color=colors)

    # Add value labels
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    constraint_label = "+AddCL" if only_addcl else "(mixed)"
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Method Comparison: {metric.upper()} {constraint_label}")
    ax.grid(axis="x", alpha=0.3)

    n_samples = data["config"].get("max_samples", "all")
    n_ens = data["config"].get("n_ensemble", "?")
    ax.text(
        0.98,
        0.02,
        f"N={n_samples}, M={n_ens}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_constraint_effect(
    results_path: str | Path,
    save_path: str | Path,
) -> None:
    """Grouped bar chart showing effect of AddCL constraint on each method.

    Shows CRPS with and without constraint side by side.

    Args:
        results_path: path to eval results JSON
        save_path: output PNG path
    """
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]

    # Group by method
    methods: dict[str, dict[str, float]] = {}
    for r in results:
        if r["crps"] != r["crps"]:
            continue
        method = r["method"]
        if method not in methods:
            methods[method] = {}
        methods[method][r["constraint"]] = r["crps"]

    # Only include methods that have both none and addcl
    paired = {m: v for m, v in methods.items() if "none" in v and "addcl" in v}
    if not paired:
        return

    method_names = list(paired.keys())
    none_vals = [paired[m]["none"] for m in method_names]
    addcl_vals = [paired[m]["addcl"] for m in method_names]

    x = np.arange(len(method_names))
    width = 0.35

    _fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, none_vals, width, label="No constraint", color="#D65F5F")
    bars2 = ax.bar(x + width / 2, addcl_vals, width, label="+AddCL", color="#4C72B0")

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_ylabel("CRPS")
    ax.set_title("Effect of AddCL Constraint on CRPS")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mass_violation(
    results_path: str | Path,
    save_path: str | Path,
) -> None:
    """Bar chart showing mass violation with and without constraints.

    Args:
        results_path: path to eval results JSON
        save_path: output PNG path
    """
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]

    # Group by method
    methods: dict[str, dict[str, float]] = {}
    for r in results:
        mv = r["mass_violation"]
        if mv != mv:
            continue
        method = r["method"]
        if method not in methods:
            methods[method] = {}
        methods[method][r["constraint"]] = mv

    paired = {m: v for m, v in methods.items() if "none" in v and "addcl" in v}
    if not paired:
        return

    method_names = list(paired.keys())
    none_vals = [paired[m]["none"] for m in method_names]
    addcl_vals = [paired[m]["addcl"] for m in method_names]

    x = np.arange(len(method_names))
    width = 0.35

    _fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, none_vals, width, label="No constraint", color="#D65F5F")
    ax.bar(x + width / 2, addcl_vals, width, label="+AddCL", color="#4C72B0")

    ax.set_ylabel("Mass Violation (mean abs)")
    ax.set_title("Mass Conservation: Effect of AddCL")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
