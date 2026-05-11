"""Structural similarity metrics for climate downscaling evaluation.

SSIM is standard in image super-resolution literature (SwinIR, SR3, HAT).
Including it enables direct comparison with pretrained SR baselines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from skimage.metrics import structural_similarity

if TYPE_CHECKING:
    from numpy.typing import NDArray


def ssim(
    truth: NDArray[np.floating],
    prediction: NDArray[np.floating],
) -> float:
    """Compute mean SSIM between ground truth and predictions.

    Args:
        truth: Ground truth fields, shape (N, H, W) or (H, W).
        prediction: Predicted fields, same shape as truth.
            For ensembles, pass ensemble mean.

    Returns:
        Mean SSIM across all samples. Higher is better (max 1.0).
    """
    if truth.ndim == 2:
        truth = truth[np.newaxis]
        prediction = prediction[np.newaxis]

    data_range = float(truth.max() - truth.min())
    if data_range == 0:
        return 1.0

    scores = []
    for i in range(truth.shape[0]):
        s = structural_similarity(
            truth[i],
            prediction[i],
            data_range=data_range,
        )
        scores.append(s)

    return float(np.mean(scores))


def psnr(
    truth: NDArray[np.floating],
    prediction: NDArray[np.floating],
) -> float:
    """Compute mean PSNR between ground truth and predictions.

    Args:
        truth: Ground truth fields, shape (N, H, W) or (H, W).
        prediction: Predicted fields, same shape as truth.

    Returns:
        Mean PSNR in dB across all samples. Higher is better.
    """
    if truth.ndim == 2:
        truth = truth[np.newaxis]
        prediction = prediction[np.newaxis]

    data_range = float(truth.max() - truth.min())
    if data_range == 0:
        return float("inf")

    psnr_vals = []
    for i in range(truth.shape[0]):
        mse = float(np.mean((truth[i] - prediction[i]) ** 2))
        if mse == 0:
            psnr_vals.append(float("inf"))
        else:
            psnr_vals.append(10.0 * np.log10(data_range**2 / mse))

    return float(np.mean(psnr_vals))
