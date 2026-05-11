"""Distribution-level metrics for climate downscaling evaluation.

EMD (Earth Mover Distance / Wasserstein-1) measures the minimum cost
to transform one distribution into another. Used by:
- Rampal et al. (2025-12-16, arXiv:2512.13987) — intercomparison paper
- Price & Rasp (2023-12-11, arXiv:2312.06071) — STVD

Unlike pixelwise metrics, EMD captures whether the overall intensity
distribution is reproduced — a model may have low MAE but systematically
miss rare extremes, which EMD penalizes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import wasserstein_distance

if TYPE_CHECKING:
    from numpy.typing import NDArray


def emd(
    truth: NDArray[np.floating],
    prediction: NDArray[np.floating],
) -> float:
    """Earth Mover Distance between flattened field distributions.

    Computes the 1D Wasserstein distance between the empirical distributions
    of all pixel values across the dataset.

    Args:
        truth: Ground truth fields, shape (N, H, W).
        prediction: Predicted fields, same shape.

    Returns:
        Wasserstein-1 distance. Lower is better. Units match the field values.
    """
    return float(wasserstein_distance(truth.ravel(), prediction.ravel()))


def emd_per_sample(
    truth: NDArray[np.floating],
    prediction: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Per-sample EMD for analyzing distribution fidelity across the test set.

    Args:
        truth: Ground truth fields, shape (N, H, W).
        prediction: Predicted fields, same shape.

    Returns:
        Array of shape (N,) with per-sample EMD values.
    """
    n = truth.shape[0]
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        scores[i] = wasserstein_distance(truth[i].ravel(), prediction[i].ravel())
    return scores
