"""Sample prediction visualization for downscaling models.

Produces side-by-side comparisons of LR input, HR ground truth,
and predictions from various methods for individual samples.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from downscaling.constraints.layers import apply_addcl


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert to 2D numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.squeeze(x)


def plot_sample_comparison(
    lr: torch.Tensor | np.ndarray,
    hr: torch.Tensor | np.ndarray,
    predictions: dict[str, torch.Tensor | np.ndarray],
    sample_idx: int = 0,
    output_path: str | None = None,
    figsize_per_panel: tuple[float, float] = (3.5, 3.5),
) -> plt.Figure:
    """Plot LR, HR, and multiple predictions side-by-side for one sample.

    Args:
        lr: Low-resolution input, shape (N, 1, H_lr, W_lr) or (H_lr, W_lr).
        hr: High-resolution ground truth, shape (N, 1, H_hr, W_hr) or (H_hr, W_hr).
        predictions: Dict mapping method name to prediction tensor.
            Each should be shape (N, 1, H_hr, W_hr) or (H_hr, W_hr).
        sample_idx: Which sample to visualize.
        output_path: Optional path to save figure.
        figsize_per_panel: Size of each subplot panel.
    """
    lr_np = _to_numpy(lr[sample_idx] if lr.ndim > 2 else lr)
    hr_np = _to_numpy(hr[sample_idx] if hr.ndim > 2 else hr)
    hr_h, hr_w = hr_np.shape[-2], hr_np.shape[-1]
    lr_h, lr_w = lr_np.shape[-2], lr_np.shape[-1]

    n_panels = 2 + len(predictions)  # LR, HR, + each prediction
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
    )

    vmin = hr_np.min()
    vmax = hr_np.max()

    # LR (upsampled to HR size for visual comparison)
    lr_up = F.interpolate(
        torch.from_numpy(lr_np).float().unsqueeze(0).unsqueeze(0),
        size=(hr_h, hr_w),
        mode="nearest",
    )
    axes[0].imshow(lr_up.squeeze().numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"LR ({lr_h}x{lr_w}\nnearest-up)", fontsize=10)
    axes[0].axis("off")

    # HR ground truth
    axes[1].imshow(hr_np, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"HR Ground Truth\n({hr_h}x{hr_w})", fontsize=10)
    axes[1].axis("off")

    # Predictions
    for i, (name, pred) in enumerate(predictions.items()):
        pred_np = _to_numpy(pred[sample_idx] if pred.ndim > 2 else pred)
        axes[i + 2].imshow(pred_np, cmap="viridis", vmin=vmin, vmax=vmax)
        mae = float(np.mean(np.abs(hr_np - pred_np)))
        axes[i + 2].set_title(f"{name}\nMAE={mae:.4f}", fontsize=10)
        axes[i + 2].axis("off")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_error_maps(
    hr: torch.Tensor | np.ndarray,
    predictions: dict[str, torch.Tensor | np.ndarray],
    sample_idx: int = 0,
    output_path: str | None = None,
    figsize_per_panel: tuple[float, float] = (3.5, 3.5),
) -> plt.Figure:
    """Plot absolute error maps for each prediction method.

    Args:
        hr: High-resolution ground truth, shape (N, 1, H, W) or (H, W).
        predictions: Dict mapping method name to prediction tensor.
        sample_idx: Which sample to visualize.
        output_path: Optional path to save figure.
        figsize_per_panel: Size of each subplot panel.
    """
    hr_np = _to_numpy(hr[sample_idx] if hr.ndim > 2 else hr)

    n_panels = len(predictions)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
    )
    if n_panels == 1:
        axes = [axes]

    # Compute all errors first for shared colorscale
    errors = {}
    for name, pred in predictions.items():
        pred_np = _to_numpy(pred[sample_idx] if pred.ndim > 2 else pred)
        errors[name] = np.abs(hr_np - pred_np)

    err_max = max(e.max() for e in errors.values())

    im = None
    for i, (name, err) in enumerate(errors.items()):
        im = axes[i].imshow(err, cmap="hot_r", vmin=0, vmax=err_max)
        mae = float(np.mean(err))
        axes[i].set_title(f"{name}\nMAE={mae:.4f}", fontsize=10)
        axes[i].axis("off")

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.8, label="Absolute Error")
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ensemble_spread(
    hr: torch.Tensor | np.ndarray,
    ensemble_preds: np.ndarray,
    sample_idx: int = 0,
    output_path: str | None = None,
    figsize: tuple[float, float] = (14, 4),
) -> plt.Figure:
    """Plot ensemble mean, std, and individual members for one sample.

    Args:
        hr: Ground truth, shape (N, 1, H, W) or (H, W).
        ensemble_preds: Ensemble predictions, shape (N, M, H, W) or (M, H, W).
            M = number of ensemble members.
        sample_idx: Which sample to visualize.
        output_path: Optional path to save figure.
        figsize: Figure size.
    """
    hr_np = _to_numpy(hr[sample_idx] if hr.ndim > 2 else hr)

    if ensemble_preds.ndim == 4:
        ens = ensemble_preds[sample_idx]  # (M, 128, 128)
    else:
        ens = ensemble_preds  # (M, 128, 128)

    ens_mean = ens.mean(axis=0)
    ens_std = ens.std(axis=0)

    vmin, vmax = hr_np.min(), hr_np.max()

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    axes[0].imshow(hr_np, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(ens_mean, cmap="viridis", vmin=vmin, vmax=vmax)
    mae = float(np.mean(np.abs(hr_np - ens_mean)))
    axes[1].set_title(f"Ensemble Mean\nMAE={mae:.4f}", fontsize=10)
    axes[1].axis("off")

    im_std = axes[2].imshow(ens_std, cmap="magma")
    axes[2].set_title(f"Ensemble Std\nmean={ens_std.mean():.4f}", fontsize=10)
    axes[2].axis("off")
    fig.colorbar(im_std, ax=axes[2], shrink=0.8)

    error = np.abs(hr_np - ens_mean)
    im_err = axes[3].imshow(error, cmap="hot_r")
    axes[3].set_title(f"Abs Error\nMAE={mae:.4f}", fontsize=10)
    axes[3].axis("off")
    fig.colorbar(im_err, ax=axes[3], shrink=0.8)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_output_grid(
    lr: torch.Tensor | np.ndarray,
    hr: torch.Tensor | np.ndarray,
    predictions: dict[str, torch.Tensor | np.ndarray],
    sample_indices: list[int] | None = None,
    n_samples: int = 3,
    output_path: str | None = None,
    figsize_per_cell: tuple[float, float] = (2.5, 2.5),
    cmap: str = "viridis",
    crop: tuple[int, int, int, int] | None = None,
    title: str = "Output Comparison",
) -> plt.Figure:
    """Horizontal grid: rows are samples, columns are methods.

    Columns: LR (nearest-upsampled), HR ground truth, then one per method.
    Each row uses per-sample color range (from that sample's HR) so contrast
    is preserved across samples with different value ranges.

    Args:
        lr: Low-resolution input, shape (N, 1, H_lr, W_lr) or (N, H_lr, W_lr).
        hr: High-resolution ground truth, shape (N, 1, H_hr, W_hr) or (N, H_hr, W_hr).
        predictions: Dict mapping method name to prediction tensor.
            Each should be shape (N, ..., H_hr, W_hr). For ensembles (N, M, H, W),
            the ensemble mean is displayed.
        sample_indices: Explicit list of sample indices to show. Overrides n_samples.
        n_samples: Number of samples (rows) if sample_indices is not given.
        output_path: Optional path to save figure.
        figsize_per_cell: Size of each cell in the grid.
        cmap: Matplotlib colormap name.
        crop: Optional (row_start, row_end, col_start, col_end) in HR pixel coords.
            When set, all cells show only this region (zoomed in).
        title: Figure title.
    """
    if sample_indices is None:
        n_avail = hr.shape[0] if hasattr(hr, "shape") else len(hr)
        sample_indices = list(range(min(n_samples, n_avail)))
    n_rows = len(sample_indices)

    # Resolve spatial sizes
    hr_0 = _to_numpy(hr[sample_indices[0]] if hr.ndim > 2 else hr)
    hr_h, hr_w = hr_0.shape[-2], hr_0.shape[-1]
    lr_0 = _to_numpy(lr[sample_indices[0]] if lr.ndim > 2 else lr)
    lr_h, lr_w = lr_0.shape[-2], lr_0.shape[-1]

    method_names = list(predictions.keys())

    if crop is not None:
        r0, r1, c0, c1 = crop
        lr_label = f"LR (crop {r1 - r0}x{c1 - c0})"
        hr_label = f"HR (crop {r1 - r0}x{c1 - c0})"
    else:
        lr_label = f"LR ({lr_h}x{lr_w})"
        hr_label = f"HR ({hr_h}x{hr_w})"

    col_labels = [lr_label, hr_label] + method_names
    n_cols = len(col_labels)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False,
    )

    def _maybe_crop(img: np.ndarray) -> np.ndarray:
        if crop is None:
            return img
        r0, r1, c0, c1 = crop
        return img[r0:r1, c0:c1]

    for row, si in enumerate(sample_indices):
        lr_np = _to_numpy(lr[si] if lr.ndim > 2 else lr)
        hr_np = _to_numpy(hr[si] if hr.ndim > 2 else hr)

        # Per-sample color range from full HR (before crop) for consistent mapping
        vmin, vmax = float(hr_np.min()), float(hr_np.max())

        hr_np = _maybe_crop(hr_np)

        # Col 0: LR (nearest-upsampled to HR size, then crop)
        lr_up = F.interpolate(
            torch.from_numpy(lr_np).float().unsqueeze(0).unsqueeze(0),
            size=(hr_h, hr_w),
            mode="nearest",
        )
        axes[row, 0].imshow(_maybe_crop(lr_up.squeeze().numpy()), cmap=cmap, vmin=vmin, vmax=vmax)

        # Col 1: HR ground truth
        axes[row, 1].imshow(hr_np, cmap=cmap, vmin=vmin, vmax=vmax)

        # Remaining cols: predictions
        for col, name in enumerate(method_names, start=2):
            pred = predictions[name]
            pred_np = _to_numpy(pred[si] if pred.ndim > 2 else pred)
            # Ensemble: average over member dimension
            if pred_np.ndim == 3:
                pred_np = pred_np.mean(axis=0)
            pred_np = _maybe_crop(pred_np)
            axes[row, col].imshow(pred_np, cmap=cmap, vmin=vmin, vmax=vmax)
            mae = float(np.mean(np.abs(hr_np - pred_np)))
            axes[row, col].text(
                0.98,
                0.02,
                f"{mae:.3f}",
                transform=axes[row, col].transAxes,
                fontsize=7,
                color="white",
                ha="right",
                va="bottom",
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1.5, "linewidth": 0},
            )

    # Column headers
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=9)

    # Row labels
    for row, si in enumerate(sample_indices):
        axes[row, 0].set_ylabel(f"Sample {si}", fontsize=9)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.suptitle(title, fontsize=13, y=1.01)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def generate_baseline_predictions(
    lr_orig: torch.Tensor,
    n_samples: int = 5,
    upsampling_factor: int = 4,
) -> dict[str, torch.Tensor]:
    """Generate baseline predictions (no GPU needed).

    Args:
        lr_orig: Original LR, shape (N, 1, H, W).
        n_samples: Number of samples to generate for.
        upsampling_factor: SR scale factor (4 for ERA5, 2 for NorESM).

    Returns:
        Dict mapping method name to predictions (N, 1, H*factor, W*factor).
    """
    lr = lr_orig[:n_samples]

    bilinear = F.interpolate(
        lr, scale_factor=upsampling_factor, mode="bilinear", align_corners=False
    )
    bicubic = F.interpolate(lr, scale_factor=upsampling_factor, mode="bicubic", align_corners=False)
    bicubic_addcl = apply_addcl(bicubic, lr, upsampling_factor=upsampling_factor)

    return {
        "Bilinear": bilinear,
        "Bicubic": bicubic,
        "Bicubic+AddCL": bicubic_addcl,
    }
