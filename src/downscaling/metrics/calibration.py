"""Ensemble calibration diagnostics: rank histogram and spread-skill ratio.

Rank histogram (Talagrand diagram): tests whether the truth is equally likely
to fall in any rank position among M ensemble members. A well-calibrated
ensemble produces a uniform histogram over M+1 bins.

Spread-skill ratio (SSR): bias-corrected ratio of ensemble spread to RMSE.
SSR ≈ 1 indicates well-calibrated uncertainty. Formula from CDSI (2603.03838)
and standard practice (Fortin et al., 2014):

    SSR = sqrt((M+1)/M) * Spread / RMSE

where Spread = std of ensemble members averaged over space,
and RMSE = error of ensemble mean vs truth averaged over space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def rank_histogram(
    truth: NDArray[np.floating],
    ensemble: NDArray[np.floating],
) -> NDArray[np.intp]:
    """Compute rank histogram counts from ensemble forecasts.

    For each spatial point, finds the rank of the truth value among the
    M ensemble members. Ranks range from 0 (truth < all members) to M
    (truth > all members), giving M+1 bins total.

    Args:
        truth: Ground truth array of shape (..., H, W). The trailing
            spatial dims are flattened for ranking.
        ensemble: Ensemble predictions of shape (M, ..., H, W) where M
            is ensemble size. Leading dim is the member axis.

    Returns:
        1D array of length M+1 containing the count of observations
        falling into each rank bin.
    """
    m = ensemble.shape[0]
    # Count how many ensemble members are below the truth at each point
    below = np.sum(ensemble < truth[None, ...], axis=0)
    # below is an int array in [0, M]; histogram over M+1 bins
    counts = np.bincount(below.ravel(), minlength=m + 1)
    return counts[: m + 1]


def spread_skill_ratio(
    truth: NDArray[np.floating],
    ensemble: NDArray[np.floating],
) -> float:
    """Compute bias-corrected spread-skill ratio (SSR).

    SSR = sqrt((M+1)/M) * Spread / RMSE

    Spread = sqrt(mean over space of ensemble variance)
    RMSE = sqrt(mean over space of (ensemble_mean - truth)^2)

    SSR ≈ 1 indicates well-calibrated ensemble uncertainty.
    SSR < 1 indicates under-dispersion (overconfident).
    SSR > 1 indicates over-dispersion (underconfident).

    Args:
        truth: Ground truth array of shape (..., H, W).
        ensemble: Ensemble predictions of shape (M, ..., H, W).

    Returns:
        Scalar SSR value.

    Raises:
        ValueError: If ensemble has fewer than 2 members.
    """
    m = ensemble.shape[0]
    if m < 2:
        raise ValueError(f"SSR requires at least 2 ensemble members, got {m}")

    ens_mean = np.mean(ensemble, axis=0)

    # Spread: sqrt of spatially averaged ensemble variance
    ens_var = np.mean((ensemble - ens_mean[None, ...]) ** 2, axis=0)
    spread = np.sqrt(np.mean(ens_var))

    # Skill: RMSE of ensemble mean vs truth
    mse = np.mean((ens_mean - truth) ** 2)
    rmse = np.sqrt(mse)

    if rmse < 1e-30:
        return float("inf")

    # Finite-size bias correction: sqrt((M+1)/M)
    correction = np.sqrt((m + 1) / m)
    return float(correction * spread / rmse)
