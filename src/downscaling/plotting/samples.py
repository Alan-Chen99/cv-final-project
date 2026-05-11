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
