"""Spectral metrics for 2D fields: PSD and spectral coherence.

PSD (radially averaged power spectral density) shows whether models preserve
fine-scale structure vs over-smoothing.

Spectral coherence measures phase alignment between prediction and truth at
each spatial frequency — complementary to PSD which only captures power
magnitude. A model can match the truth PSD perfectly while having zero
coherence (right amount of texture, wrong spatial placement).

Both metrics use 2D FFT with azimuthal averaging into 1D curves indexed
by isotropic wavenumber.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def radial_psd(
    field: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute radially averaged power spectral density of a 2D field.

    Args:
        field: 2D array of shape (H, W).

    Returns:
        Tuple of (wavenumbers, power) where:
        - wavenumbers: 1D array of integer wavenumber bins [1, 2, ..., k_max].
        - power: 1D array of mean power at each wavenumber bin.
          k_max = floor(min(H, W) / 2).
    """
    if field.ndim != 2:
        raise ValueError(f"Expected 2D field, got shape {field.shape}")

    h, w = field.shape

    # 2D FFT, shift zero-frequency to center
    f_transform = np.fft.fftshift(np.fft.fft2(field))
    power_2d = np.abs(f_transform) ** 2

    # Radial wavenumber for each pixel (distance from center)
    cy, cx = h / 2, w / 2
    y_grid = np.arange(h) - cy
    x_grid = np.arange(w) - cx
    yy, xx = np.meshgrid(y_grid, x_grid, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)

    # Bin into integer wavenumber shells
    k_max = min(h, w) // 2
    k_bins = np.arange(1, k_max + 1, dtype=np.float64)
    power_1d = np.empty(k_max, dtype=np.float64)

    radius_int = np.round(radius).astype(int)
    for i, k in enumerate(k_bins):
        mask = radius_int == int(k)
        if np.any(mask):
            power_1d[i] = np.mean(power_2d[mask])
        else:
            power_1d[i] = 0.0

    return k_bins, power_1d


def ensemble_mean_psd(
    fields: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute mean PSD across an ensemble of 2D fields.

    Computes PSD for each member independently, then averages the power
    spectra. This preserves fine-scale variability present in individual
    samples (unlike computing PSD on the ensemble mean, which would show
    the smoothed spectrum).

    Args:
        fields: 3D array of shape (M, H, W) where M is ensemble size.

    Returns:
        Tuple of (wavenumbers, mean_power).
    """
    if fields.ndim != 3:
        raise ValueError(f"Expected 3D array (M, H, W), got shape {fields.shape}")

    m = fields.shape[0]

    wavenumbers, power_sum = radial_psd(fields[0])
    power_sum = power_sum.copy()
    for i in range(1, m):
        _, p = radial_psd(fields[i])
        power_sum += p

    return wavenumbers, power_sum / m


def psd_log_ratio(
    wavenumbers_pred: NDArray[np.floating],
    power_pred: NDArray[np.floating],
    wavenumbers_truth: NDArray[np.floating],
    power_truth: NDArray[np.floating],
) -> float:
    """Compute mean absolute log-ratio of predicted vs true PSD.

    Scalar summary: mean |log10(P_pred/P_truth)| across wavenumbers.
    A value of 0 means perfect spectral match. Higher means worse.

    Requires that both PSD curves share the same wavenumber grid.

    Args:
        wavenumbers_pred: Wavenumber array for predictions.
        power_pred: Power array for predictions.
        wavenumbers_truth: Wavenumber array for ground truth.
        power_truth: Power array for ground truth.

    Returns:
        Mean absolute log-ratio (scalar).
    """
    if not np.array_equal(wavenumbers_pred, wavenumbers_truth):
        raise ValueError("Wavenumber grids must match")

    # Avoid log(0) by masking zero-power bins
    valid = (power_pred > 0) & (power_truth > 0)
    if not np.any(valid):
        return float("inf")

    log_ratio = np.abs(np.log10(power_pred[valid]) - np.log10(power_truth[valid]))
    return float(np.mean(log_ratio))


def _wavenumber_grid(h: int, w: int) -> tuple[NDArray[np.floating], NDArray[np.signedinteger]]:
    """Shared helper: radial wavenumber map and integer bin assignments."""
    cy, cx = h / 2, w / 2
    yy, xx = np.meshgrid(np.arange(h) - cy, np.arange(w) - cx, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)
    radius_int = np.round(radius).astype(int)
    return radius, radius_int


def spectral_coherence(
    predictions: NDArray[np.floating],
    truths: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Radially averaged spectral coherence between prediction and truth fields.

    Coherence γ²(k) = |<S_xy(k)>|² / (<S_xx(k)> · <S_yy(k)>) where <·> is
    the average over samples AND frequencies within wavenumber bin k.

    For a single sample pair, coherence is trivially 1.0 at every frequency
    (no averaging to reduce variance). Meaningful estimation requires N >> 1.

    Args:
        predictions: (N, H, W) batch of predicted 2D fields.
        truths: (N, H, W) batch of ground-truth 2D fields.

    Returns:
        Tuple of (wavenumbers, coherence) where:
        - wavenumbers: 1D array [1, 2, ..., k_max], k_max = min(H, W) // 2.
        - coherence: 1D array in [0, 1] at each wavenumber bin.
    """
    if predictions.ndim != 3 or truths.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays (N, H, W), got {predictions.shape} and {truths.shape}"
        )
    if predictions.shape != truths.shape:
        raise ValueError(f"Shape mismatch: {predictions.shape} vs {truths.shape}")
    n, h, w = predictions.shape
    if n < 1:
        raise ValueError("Need at least 1 sample")

    _, radius_int = _wavenumber_grid(h, w)
    k_max = min(h, w) // 2
    k_bins = np.arange(1, k_max + 1, dtype=np.float64)

    # Accumulate cross-spectrum and auto-spectra per wavenumber bin
    sxy_accum = np.zeros(k_max, dtype=np.complex128)
    sxx_accum = np.zeros(k_max, dtype=np.float64)
    syy_accum = np.zeros(k_max, dtype=np.float64)
    counts = np.zeros(k_max, dtype=np.int64)

    for i in range(n):
        fx = np.fft.fftshift(np.fft.fft2(predictions[i]))
        fy = np.fft.fftshift(np.fft.fft2(truths[i]))
        cross = fx * np.conj(fy)
        pxx = np.abs(fx) ** 2
        pyy = np.abs(fy) ** 2

        for j in range(k_max):
            k = int(k_bins[j])
            mask = radius_int == k
            if np.any(mask):
                sxy_accum[j] += np.sum(cross[mask])
                sxx_accum[j] += np.sum(pxx[mask])
                syy_accum[j] += np.sum(pyy[mask])
                counts[j] += int(np.sum(mask))

    # γ²(k) = |mean(S_xy)|² / (mean(S_xx) · mean(S_yy))
    coherence = np.zeros(k_max, dtype=np.float64)
    valid = (sxx_accum > 0) & (syy_accum > 0) & (counts > 0)
    mean_sxy = np.where(counts > 0, sxy_accum / counts, 0)
    mean_sxx = np.where(counts > 0, sxx_accum / counts, 0)
    mean_syy = np.where(counts > 0, syy_accum / counts, 0)
    coherence[valid] = np.abs(mean_sxy[valid]) ** 2 / (mean_sxx[valid] * mean_syy[valid])

    return k_bins, coherence


def mean_spectral_coherence(
    predictions: NDArray[np.floating],
    truths: NDArray[np.floating],
) -> float:
    """Scalar summary: mean coherence across all wavenumber bins.

    Values in [0, 1]. Higher is better (1 = perfect phase alignment).

    Args:
        predictions: (N, H, W) batch of predicted 2D fields.
        truths: (N, H, W) batch of ground-truth 2D fields.

    Returns:
        Mean spectral coherence (scalar).
    """
    _, coherence = spectral_coherence(predictions, truths)
    return float(np.mean(coherence))
