"""Batch-level metrics that require all predictions collected.

RALSD needs dataset-averaged PSDs before comparison, so it cannot
be computed per-sample like CRPS/MAE/RMSE. EMD compares full
distributions and benefits from large sample counts. SSIM and PSNR
are per-sample but included here for a unified interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from downscaling.metrics.distribution import emd
from downscaling.metrics.spectral import radial_psd_batch, ralsd, spectral_bias
from downscaling.metrics.structural import psnr, ssim

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def compute_batch_metrics(
    ground_truth: NDArray[np.floating],
    predictions: NDArray[np.floating],
    n_bins: int = 26,
) -> dict[str, float]:
    """Compute RALSD, SSIM, PSNR, and EMD from collected predictions.

    Args:
        ground_truth: Shape (N, H, W).
        predictions: Shape (N, H, W) — ensemble mean for stochastic models.
        n_bins: Frequency bins for spectral metrics.

    Returns:
        Dict with ralsd, ssim, psnr, emd keys.
    """
    return {
        "ralsd": ralsd(ground_truth, predictions, n_bins=n_bins),
        "ssim": ssim(ground_truth, predictions),
        "psnr": psnr(ground_truth, predictions),
        "emd": emd(ground_truth, predictions),
    }


def compute_spectral_curves(
    ground_truth: NDArray[np.floating],
    predictions: NDArray[np.floating],
    n_bins: int = 26,
) -> dict[str, NDArray[np.floating]]:
    """Compute PSD curves and per-frequency bias for plotting.

    Args:
        ground_truth: Shape (N, H, W).
        predictions: Shape (N, H, W).
        n_bins: Frequency bins.

    Returns:
        Dict with freq (bin centers), psd_truth, psd_pred, bias arrays.
    """
    freq, psd_truth = radial_psd_batch(ground_truth, n_bins=n_bins)
    _, psd_pred = radial_psd_batch(predictions, n_bins=n_bins)
    bias = spectral_bias(ground_truth, predictions, n_bins=n_bins)

    return {
        "freq": freq,
        "psd_truth": psd_truth,
        "psd_pred": psd_pred,
        "bias": bias,
    }
