"""Structural Similarity Index (SSIM) for 2D fields.

SSIM captures perceptual quality beyond pointwise error by comparing local
luminance, contrast, and structure between predicted and reference fields.
Standard in image quality assessment (Wang et al., 2004) and used in climate
downscaling evaluation for detecting structural artifacts.

Local statistics are computed using Gaussian-weighted windows via
scipy.ndimage.gaussian_filter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from numpy.typing import NDArray


def ssim(
    truth: NDArray[np.floating],
    prediction: NDArray[np.floating],
    *,
    window_sigma: float = 1.5,
    data_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Compute mean Structural Similarity Index (SSIM) between two 2D fields.

    Args:
        truth: Reference field of shape (H, W).
        prediction: Predicted field of shape (H, W).
        window_sigma: Std dev of Gaussian weighting window. Default 1.5
            (corresponds to effective 11x11 window at 4-sigma truncation).
        data_range: Dynamic range of the data (max - min). If None,
            computed from the joint range of truth and prediction.
        k1: Luminance stabilization constant. Default 0.01.
        k2: Contrast stabilization constant. Default 0.03.

    Returns:
        Mean SSIM score in [-1, 1]. Higher is better; 1.0 means identical.

    Raises:
        ValueError: If inputs are not 2D or have mismatched shapes.
    """
    if truth.ndim != 2:
        raise ValueError(f"Expected 2D truth, got shape {truth.shape}")
    if prediction.ndim != 2:
        raise ValueError(f"Expected 2D prediction, got shape {prediction.shape}")
    if truth.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: truth {truth.shape} vs prediction {prediction.shape}")

    truth = truth.astype(np.float64)
    prediction = prediction.astype(np.float64)

    if data_range is None:
        joint_min = min(float(truth.min()), float(prediction.min()))
        joint_max = max(float(truth.max()), float(prediction.max()))
        data_range = joint_max - joint_min
        if data_range < 1e-30:
            # Both fields constant and equal — SSIM is 1.0
            if np.allclose(truth, prediction):
                return 1.0
            data_range = 1.0

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_x = gaussian_filter(truth, sigma=window_sigma)
    mu_y = gaussian_filter(prediction, sigma=window_sigma)

    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_x_sq = gaussian_filter(truth**2, sigma=window_sigma) - mu_x_sq
    sigma_y_sq = gaussian_filter(prediction**2, sigma=window_sigma) - mu_y_sq
    sigma_xy = gaussian_filter(truth * prediction, sigma=window_sigma) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def ensemble_mean_ssim(
    truth: NDArray[np.floating],
    ensemble: NDArray[np.floating],
    **kwargs: float | None,
) -> float:
    """Compute mean SSIM across ensemble members.

    Args:
        truth: Reference field of shape (H, W).
        ensemble: Ensemble predictions of shape (M, H, W).
        **kwargs: Passed to ssim() (window_sigma, data_range, k1, k2).

    Returns:
        Mean SSIM averaged over all M ensemble members.

    Raises:
        ValueError: If ensemble is not 3D.
    """
    if ensemble.ndim != 3:
        raise ValueError(f"Expected 3D array (M, H, W), got shape {ensemble.shape}")

    m = ensemble.shape[0]
    total = sum(ssim(truth, ensemble[i], **kwargs) for i in range(m))  # type: ignore[arg-type]
    return total / m
