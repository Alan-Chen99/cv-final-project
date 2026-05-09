"""CRPS (Continuous Ranked Probability Score) computation.

Two implementations:
- crps_energy: Correct energy score formula: E|X-y| - 0.5*E|X-X'|
- crps_paper: Paper-compatible version with asymmetric weighting (has known bug
  in Harder et al. codebase; this is a clean reimplementation)

The energy CRPS (Gneiting & Raftery, 2007) is the standard correct formula.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def crps_energy(
    observation: NDArray[np.floating],
    forecasts: NDArray[np.floating],
) -> float:
    """Correct energy CRPS: E|X-y| - 0.5 * E|X-X'|.

    Args:
        observation: Ground truth array of shape (H, W) or (C, H, W).
        forecasts: Ensemble predictions of shape (M, H, W) or (M, C, H, W),
            where M is the number of ensemble members.

    Returns:
        Scalar CRPS averaged over all spatial points.
    """
    M = forecasts.shape[0]
    # E|X - y|: mean absolute error across ensemble members
    mae_term = np.mean(np.abs(forecasts - observation[None, ...]), axis=0)
    # E|X - X'|: mean pairwise absolute difference among ensemble members
    if M > 1:
        spread = 0.0
        for i in range(M):
            for j in range(i + 1, M):
                spread += np.abs(forecasts[j] - forecasts[i])
        spread = spread * 2.0 / (M * (M - 1))
    else:
        spread = 0.0
    crps = mae_term - 0.5 * spread
    return float(np.mean(crps))


def crps_paper(
    observation: NDArray[np.floating],
    forecasts: NDArray[np.floating],
) -> float:
    """Paper-compatible CRPS with asymmetric weighting.

    Note: The original Harder et al. crps_ensemble() has a bug where
    fc.shape[-1]**2 is used instead of fc.shape[0]**2 in the first loop.
    This implementation uses fc.shape[0]**2 consistently (correct behavior).

    Args:
        observation: Ground truth array of shape (H, W).
        forecasts: Ensemble predictions of shape (M, H, W).

    Returns:
        Scalar CRPS averaged over all spatial points.
    """
    fc = forecasts.copy()
    fc.sort(axis=0)
    M = fc.shape[0]
    fc_below = fc < observation[None, ...]
    crps = np.zeros_like(observation)
    for i in range(M):
        below = fc_below[i, ...]
        weight = ((i + 1) ** 2 - i**2) / M**2
        crps[below] += weight * (observation[below] - fc[i, ...][below])
    for i in range(M - 1, -1, -1):
        above = ~fc_below[i, ...]
        k = M - 1 - i
        weight = ((k + 1) ** 2 - k**2) / M**2
        crps[above] += weight * (fc[i, ...][above] - observation[above])
    return float(np.mean(crps))
