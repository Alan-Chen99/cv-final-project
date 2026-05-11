"""Spectral evaluation metrics for climate downscaling.

Implements radially-averaged power spectral density (PSD) and RALSD
(Radially Averaged Logarithmic Spectral Distance) following:
- Harris et al. (2022)
- Rampal, Gibson, Sherwood, Abramowitz, et al. (2025)
- Intercomparison paper (Rampal et al., 2025-12-16, arXiv:2512.13987)

RALSD weights errors at all spatial scales equally in log space,
so fine-scale structure errors are penalized as much as large-scale ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def radial_psd(
    field: NDArray[np.floating],
    n_bins: int = 26,
    max_freq: float = 0.5,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute radially-averaged power spectral density of a 2D field.

    Takes the 2D FFT, computes power, then bins radially to produce
    a 1D power spectrum as a function of spatial frequency.

    Args:
        field: 2D array of shape (H, W).
        n_bins: Number of radial frequency bins.
        max_freq: Maximum normalized frequency (0.5 = Nyquist).

    Returns:
        Tuple of (bin_centers, psd) where bin_centers are normalized
        frequencies and psd is power at each frequency.
    """
    h, w = field.shape
    fft2 = np.fft.fft2(field)
    power = np.abs(fft2) ** 2 / (h * w)

    # Frequency grids normalized to [0, 0.5]
    freq_y = np.fft.fftfreq(h)
    freq_x = np.fft.fftfreq(w)
    fy, fx = np.meshgrid(freq_y, freq_x, indexing="ij")
    freq_r = np.sqrt(fx**2 + fy**2)

    bin_edges = np.linspace(0, max_freq, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    psd = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (freq_r >= bin_edges[i]) & (freq_r < bin_edges[i + 1])
        if mask.any():
            psd[i] = np.mean(power[mask])

    return bin_centers, psd


def radial_psd_batch(
    fields: NDArray[np.floating],
    n_bins: int = 26,
    max_freq: float = 0.5,
    normalize: bool = False,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute mean radially-averaged PSD over a batch of 2D fields.

    Args:
        fields: 3D array of shape (N, H, W).
        n_bins: Number of radial frequency bins.
        max_freq: Maximum normalized frequency.
        normalize: If True, normalize each field to zero mean unit variance
            before computing FFT (isolates spatial structure from intensity).

    Returns:
        Tuple of (bin_centers, mean_psd).
    """
    n = fields.shape[0]
    all_psd = []

    for i in range(n):
        f = fields[i].astype(np.float64)
        if normalize:
            std = f.std()
            if std > 0:
                f = (f - f.mean()) / std
            else:
                f = f - f.mean()
        _, psd = radial_psd(f, n_bins=n_bins, max_freq=max_freq)
        all_psd.append(psd)

    return radial_psd(fields[0], n_bins=n_bins, max_freq=max_freq)[0], np.mean(all_psd, axis=0)


def ralsd(
    truth: NDArray[np.floating],
    prediction: NDArray[np.floating],
    n_bins: int = 26,
    max_freq: float = 0.5,
    normalize: bool = False,
) -> float:
    """Compute RALSD (Radially Averaged Log Spectral Distance).

    RALSD(dB) = sqrt(1/N * sum_i (10*log10(F_true_i / F_pred_i))^2)

    Measures how well predictions match the true power spectral density
    across all spatial frequencies. Lower is better.

    Args:
        truth: Ground truth fields, shape (N, H, W).
        prediction: Predicted fields, shape (N, H, W) or (M, N, H, W)
            for ensemble. If 4D, ensemble mean is used.
        n_bins: Number of radial frequency bins.
        max_freq: Maximum normalized frequency.
        normalize: If True, normalize each field before FFT.

    Returns:
        RALSD in dB. Lower is better.
    """
    if prediction.ndim == 4:
        prediction = prediction.mean(axis=0)

    _, psd_true = radial_psd_batch(truth, n_bins=n_bins, max_freq=max_freq, normalize=normalize)
    _, psd_pred = radial_psd_batch(
        prediction, n_bins=n_bins, max_freq=max_freq, normalize=normalize
    )

    # Exclude bins where either PSD is zero (avoid log(0))
    valid = (psd_true > 0) & (psd_pred > 0)
    if not valid.any():
        return float("inf")

    log_ratio = 10.0 * np.log10(psd_true[valid] / psd_pred[valid])
    return float(np.sqrt(np.mean(log_ratio**2)))


def spectral_bias(
    truth: NDArray[np.floating],
    prediction: NDArray[np.floating],
    n_bins: int = 26,
    max_freq: float = 0.5,
    normalize: bool = False,
) -> NDArray[np.floating]:
    """Per-frequency spectral bias: 10*log10(F_true / F_pred) at each bin.

    Positive values mean prediction underestimates power (too smooth).
    Negative values mean prediction overestimates power (too noisy).

    Args:
        truth: Ground truth fields, shape (N, H, W).
        prediction: Predicted fields, shape (N, H, W) or (M, N, H, W).
        n_bins: Number of radial frequency bins.
        max_freq: Maximum normalized frequency.
        normalize: If True, normalize each field before FFT.

    Returns:
        Array of shape (n_bins,) with spectral bias in dB at each frequency.
    """
    if prediction.ndim == 4:
        prediction = prediction.mean(axis=0)

    _, psd_true = radial_psd_batch(truth, n_bins=n_bins, max_freq=max_freq, normalize=normalize)
    _, psd_pred = radial_psd_batch(
        prediction, n_bins=n_bins, max_freq=max_freq, normalize=normalize
    )

    valid = (psd_true > 0) & (psd_pred > 0)
    bias = np.full(n_bins, np.nan)
    bias[valid] = 10.0 * np.log10(psd_true[valid] / psd_pred[valid])
    return bias
