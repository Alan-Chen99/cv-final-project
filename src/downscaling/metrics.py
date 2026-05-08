"""Evaluation metrics for ensemble climate downscaling.

CRPS (Continuous Ranked Probability Score) implementations:
- crps_energy: Correct energy score formula E|X-y| - 0.5*E|X-X'|
- crps_paper: Buggy version from Harder et al. for backward compatibility
"""

import numpy as np
import torch


def crps_energy(observation: np.ndarray, forecasts: np.ndarray) -> float:
    """Correct energy CRPS: E|X-y| - 0.5*E|X-X'|.

    Uses the Gneiting & Raftery (2007) energy score formulation with
    the sorted-ensemble O(M log M) computation.

    Args:
        observation: Ground truth, shape (H, W)
        forecasts: Ensemble predictions, shape (M, H, W)

    Returns:
        Scalar CRPS averaged over all spatial locations.
    """
    M = forecasts.shape[0]
    # E|X - y|: mean absolute error across ensemble members
    mae_term = np.mean(np.abs(forecasts - observation[None, ...]), axis=0)

    # E|X - X'|: spread term via sorted ensemble (Gneiting formula)
    fc_sorted = np.sort(forecasts, axis=0)
    spread = np.zeros_like(observation)
    for j in range(M):
        w = (2.0 * (j + 1) - M - 1.0) / (M * M)
        spread += w * fc_sorted[j]

    crps = mae_term - spread
    return float(np.mean(crps))


def crps_paper(observation: np.ndarray, forecasts: np.ndarray) -> float:
    """CRPS from Harder et al. codebase (has known bug for backward compatibility).

    Bug: first loop normalizes by fc.shape[-1]**2 (spatial width) instead of
    fc.shape[0]**2 (ensemble size). Kept for reproducing published numbers.
    """
    fc = forecasts.copy()
    fc.sort(axis=0)
    obs = observation
    fc_below = fc < obs[None, ...]
    crps = np.zeros_like(obs)
    for i in range(fc.shape[0]):
        below = fc_below[i, ...]
        weight = ((i + 1) ** 2 - i**2) / fc.shape[-1] ** 2
        crps[below] += weight * (obs[below] - fc[i, ...][below])
    for i in range(fc.shape[0] - 1, -1, -1):
        above = ~fc_below[i, ...]
        k = fc.shape[0] - 1 - i
        weight = ((k + 1) ** 2 - k**2) / fc.shape[0] ** 2
        crps[above] += weight * (fc[i, ...][above] - obs[above])
    return float(np.mean(crps))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(prediction - target)))


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def mass_violation(
    pred_hr: torch.Tensor, lr_orig: torch.Tensor, upsampling_factor: int = 4
) -> float:
    """Mean absolute mass conservation violation.

    Measures how well the prediction's block means match the LR input.

    Args:
        pred_hr: HR prediction (B, 1, H, W)
        lr_orig: Original LR input (B, 1, H/f, W/f)
        upsampling_factor: SR factor (default 4)

    Returns:
        Mean absolute difference between downsampled prediction and LR input.
    """
    pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
    pooled = pool(pred_hr)
    return float(torch.mean(torch.abs(pooled - lr_orig)).item())


def ensemble_spread(forecasts: np.ndarray) -> float:
    """Mean ensemble standard deviation (spread/calibration diagnostic).

    Args:
        forecasts: shape (M, H, W)

    Returns:
        Mean pixelwise standard deviation across ensemble members.
    """
    return float(np.mean(np.std(forecasts, axis=0)))
