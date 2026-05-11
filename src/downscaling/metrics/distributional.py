"""Distribution comparison metrics: KL divergence between empirical distributions.

Compares the marginal distribution of predicted vs true field values via
histogram-based KL divergence. Useful for detecting distribution shifts in
climate downscaling — e.g., models that produce correct spatial patterns
but wrong value distributions (shifted mean, compressed tails, etc.).

Binning uses shared edges from the joint range of both inputs. Laplace
smoothing prevents log(0) in empty bins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def histogram_kl_divergence(
    truth: NDArray[np.floating],
    prediction: NDArray[np.floating],
    *,
    n_bins: int = 100,
) -> float:
    """Compute KL divergence D_KL(truth || prediction) from histograms.

    Flattens both fields, bins into shared histogram edges, applies Laplace
    smoothing (add 1 to all bins), normalizes, and computes:

        D_KL(P || Q) = sum P(x) * log(P(x) / Q(x))

    Args:
        truth: Reference field of arbitrary shape.
        prediction: Predicted field of same shape.
        n_bins: Number of histogram bins. Default 100.

    Returns:
        KL divergence in nats (natural log). Non-negative; 0 means identical
        distributions. Lower is better.

    Raises:
        ValueError: If shapes don't match or n_bins < 2.
    """
    if truth.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: truth {truth.shape} vs prediction {prediction.shape}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    t_flat = truth.ravel().astype(np.float64)
    p_flat = prediction.ravel().astype(np.float64)

    # Shared bin edges from joint range
    lo = min(float(t_flat.min()), float(p_flat.min()))
    hi = max(float(t_flat.max()), float(p_flat.max()))
    if hi - lo < 1e-30:
        # Both fields constant and equal
        if np.allclose(truth, prediction):
            return 0.0
        # Same constant value but somehow not close — degenerate case
        return 0.0

    edges = np.linspace(lo, hi, n_bins + 1)

    counts_t = np.histogram(t_flat, bins=edges)[0].astype(np.float64)
    counts_p = np.histogram(p_flat, bins=edges)[0].astype(np.float64)

    # Laplace smoothing: add 1 to every bin
    counts_t += 1.0
    counts_p += 1.0

    # Normalize to probability distributions
    p_dist = counts_t / counts_t.sum()
    q_dist = counts_p / counts_p.sum()

    # KL divergence: sum p * log(p/q)
    kl = float(np.sum(p_dist * np.log(p_dist / q_dist)))
    return kl


def ensemble_mean_kl_divergence(
    truth: NDArray[np.floating],
    ensemble: NDArray[np.floating],
    *,
    n_bins: int = 100,
) -> float:
    """Compute mean KL divergence across ensemble members.

    Args:
        truth: Reference field of shape (H, W) or (..., H, W).
        ensemble: Ensemble predictions of shape (M, ..., H, W).
        n_bins: Number of histogram bins.

    Returns:
        Mean KL divergence averaged over all M members.

    Raises:
        ValueError: If ensemble is not at least 2D or shapes don't match.
    """
    if ensemble.ndim < 2:
        raise ValueError(f"Expected at least 2D array (M, ...), got shape {ensemble.shape}")

    m = ensemble.shape[0]
    total = sum(
        histogram_kl_divergence(truth, ensemble[i], n_bins=n_bins) for i in range(m)
    )
    return total / m
