"""Baseline evaluation methods that require no training.

Provides bicubic and bilinear interpolation baselines for 4x downscaling.
These are deterministic: CRPS = MAE (no ensemble spread term).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from downscaling.constraints.layers import apply_addcl
from downscaling.metrics.crps import crps_energy


def upsample_bilinear(lr: torch.Tensor, scale_factor: int = 4) -> torch.Tensor:
    """Bilinear 4x upsampling.

    Args:
        lr: Low-resolution input, shape (N, 1, H, W).
        scale_factor: Upsampling factor.

    Returns:
        Upsampled tensor, shape (N, 1, H*f, W*f).
    """
    return F.interpolate(lr, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def upsample_bicubic(lr: torch.Tensor, scale_factor: int = 4) -> torch.Tensor:
    """Bicubic 4x upsampling.

    Args:
        lr: Low-resolution input, shape (N, 1, H, W).
        scale_factor: Upsampling factor.

    Returns:
        Upsampled tensor, shape (N, 1, H*f, W*f).
    """
    return F.interpolate(lr, scale_factor=scale_factor, mode="bicubic", align_corners=False)


def evaluate_deterministic(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    lr_orig: torch.Tensor,
    upsampling_factor: int = 4,
) -> dict[str, float]:
    """Evaluate a deterministic (non-ensemble) model.

    For deterministic models, CRPS = MAE since E|X-X'| = 0.

    Args:
        predictions: Predicted HR, shape (N, 1, H, W).
        targets: Ground truth HR, shape (N, 1, H, W).
        lr_orig: Original LR for mass violation, shape (N, 1, H/f, W/f).
        upsampling_factor: SR factor.

    Returns:
        Dict with CRPS, MAE, RMSE, mass_violation.
    """
    pool = nn.AvgPool2d(kernel_size=upsampling_factor)
    N = predictions.shape[0]

    mae_sum = 0.0
    rmse_sq_sum = 0.0
    mass_viol_sum = 0.0
    crps_sum = 0.0

    for i in range(N):
        pred = predictions[i, 0].numpy()
        gt = targets[i, 0].numpy()
        # For deterministic, ensemble is just 1 member
        crps_sum += crps_energy(gt, pred[None, ...])
        mae_sum += float(np.mean(np.abs(gt - pred)))
        rmse_sq_sum += float(np.mean((gt - pred) ** 2))
        pooled = pool(predictions[i : i + 1]).squeeze()
        lr_i = lr_orig[i, 0]
        mass_viol_sum += float(torch.mean(torch.abs(pooled - lr_i)).item())

    return {
        "crps": crps_sum / N,
        "mae": mae_sum / N,
        "rmse": float(np.sqrt(rmse_sq_sum / N)),
        "mass_violation": mass_viol_sum / N,
    }


def eval_bicubic(
    hr: torch.Tensor,
    lr_orig: torch.Tensor,
    with_addcl: bool = False,
    upsampling_factor: int = 4,
) -> dict[str, float]:
    """Evaluate bicubic interpolation baseline.

    Args:
        hr: Ground truth HR, shape (N, 1, H, W).
        lr_orig: Original LR, shape (N, 1, H/f, W/f).
        with_addcl: Whether to apply AddCL constraint.
        upsampling_factor: SR factor.

    Returns:
        Dict with CRPS, MAE, RMSE, mass_violation.
    """
    pred = upsample_bicubic(lr_orig, upsampling_factor)
    if with_addcl:
        pred = apply_addcl(pred, lr_orig, upsampling_factor)
    return evaluate_deterministic(pred, hr, lr_orig, upsampling_factor)


def eval_bilinear(
    hr: torch.Tensor,
    lr_orig: torch.Tensor,
    with_addcl: bool = False,
    upsampling_factor: int = 4,
) -> dict[str, float]:
    """Evaluate bilinear interpolation baseline.

    Args:
        hr: Ground truth HR, shape (N, 1, H, W).
        lr_orig: Original LR, shape (N, 1, H/f, W/f).
        with_addcl: Whether to apply AddCL constraint.
        upsampling_factor: SR factor.

    Returns:
        Dict with CRPS, MAE, RMSE, mass_violation.
    """
    pred = upsample_bilinear(lr_orig, upsampling_factor)
    if with_addcl:
        pred = apply_addcl(pred, lr_orig, upsampling_factor)
    return evaluate_deterministic(pred, hr, lr_orig, upsampling_factor)
