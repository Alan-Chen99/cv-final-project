"""Integration tests for spectral and structural metrics."""

import numpy as np
import pytest

from downscaling.metrics.spectral import radial_psd, radial_psd_batch, ralsd, spectral_bias
from downscaling.metrics.structural import psnr, ssim


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestRadialPSD:
    """Test radially-averaged power spectral density."""

    def test_white_noise_flat_spectrum(self, rng):
        """White noise should have approximately flat PSD."""
        field = rng.standard_normal((128, 128))
        freqs, psd = radial_psd(field, n_bins=13)
        # Exclude DC bin (index 0) — flat within factor of 3
        assert psd[1:].max() / psd[1:].min() < 3.0

    def test_low_freq_signal_concentrated(self, rng):
        """Low-frequency sinusoid should have power concentrated at low frequencies."""
        y, x = np.meshgrid(np.linspace(0, 2 * np.pi, 128), np.linspace(0, 2 * np.pi, 128))
        field = np.sin(x) + np.sin(y)
        freqs, psd = radial_psd(field, n_bins=26)
        # Most power should be in the first few bins
        assert psd[:5].sum() > 0.9 * psd.sum()

    def test_output_shape(self, rng):
        """Output shape matches n_bins."""
        field = rng.standard_normal((64, 64))
        freqs, psd = radial_psd(field, n_bins=20)
        assert freqs.shape == (20,)
        assert psd.shape == (20,)


class TestRadialPSDBatch:
    """Test batch PSD computation."""

    def test_single_field_matches(self, rng):
        """Batch with N=1 should match single-field result."""
        field = rng.standard_normal((64, 64))
        _, psd_single = radial_psd(field, n_bins=13)
        _, psd_batch = radial_psd_batch(field[np.newaxis], n_bins=13)
        np.testing.assert_allclose(psd_single, psd_batch, rtol=1e-10)

    def test_normalize_removes_intensity(self, rng):
        """Normalization should remove intensity differences."""
        fields_a = rng.standard_normal((10, 64, 64))
        fields_b = fields_a * 100  # Same structure, different intensity
        _, psd_a = radial_psd_batch(fields_a, n_bins=13, normalize=True)
        _, psd_b = radial_psd_batch(fields_b, n_bins=13, normalize=True)
        np.testing.assert_allclose(psd_a, psd_b, rtol=1e-10)


class TestRALSD:
    """Test RALSD metric."""

    def test_identical_fields_zero(self, rng):
        """RALSD of identical fields should be 0."""
        fields = rng.standard_normal((10, 64, 64))
        result = ralsd(fields, fields)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_smoother_prediction_positive(self, rng):
        """Smoother predictions should have positive RALSD (underestimate power)."""
        from scipy.ndimage import gaussian_filter

        truth = rng.standard_normal((20, 64, 64))
        # Smooth predictions lose high-frequency content
        pred = np.stack([gaussian_filter(truth[i], sigma=2) for i in range(20)])
        result = ralsd(truth, pred)
        assert result > 0.1  # Should be meaningfully different from 0

    def test_ensemble_input(self, rng):
        """4D ensemble input should work (uses mean)."""
        truth = rng.standard_normal((10, 64, 64))
        ensemble = rng.standard_normal((5, 10, 64, 64))
        result = ralsd(truth, ensemble)
        assert np.isfinite(result)

    def test_lower_is_better(self, rng):
        """Better match should give lower RALSD."""
        from scipy.ndimage import gaussian_filter

        truth = rng.standard_normal((20, 64, 64))
        # Mild smoothing vs heavy smoothing
        mild = np.stack([gaussian_filter(truth[i], sigma=0.5) for i in range(20)])
        heavy = np.stack([gaussian_filter(truth[i], sigma=3) for i in range(20)])
        assert ralsd(truth, mild) < ralsd(truth, heavy)


class TestSpectralBias:
    """Test per-frequency spectral bias."""

    def test_identical_fields_zero_bias(self, rng):
        """Identical fields should have zero bias everywhere."""
        fields = rng.standard_normal((10, 64, 64))
        bias = spectral_bias(fields, fields)
        np.testing.assert_allclose(bias[~np.isnan(bias)], 0.0, atol=1e-10)

    def test_smooth_prediction_positive_bias(self, rng):
        """Smoothed predictions should show positive bias at high frequencies."""
        from scipy.ndimage import gaussian_filter

        truth = rng.standard_normal((20, 64, 64))
        pred = np.stack([gaussian_filter(truth[i], sigma=2) for i in range(20)])
        bias = spectral_bias(truth, pred)
        # High-frequency bins (latter half) should be positive (underestimated power)
        high_freq_bias = bias[len(bias) // 2 :]
        assert np.nanmean(high_freq_bias) > 0


class TestSSIM:
    """Test SSIM metric."""

    def test_identical_images_perfect(self, rng):
        """Identical images should have SSIM = 1."""
        img = rng.standard_normal((5, 64, 64))
        assert ssim(img, img) == pytest.approx(1.0, abs=1e-6)

    def test_noisy_image_lower(self, rng):
        """Noisy version should have lower SSIM."""
        img = rng.standard_normal((5, 64, 64))
        noisy = img + rng.standard_normal(img.shape) * 0.5
        assert ssim(img, noisy) < 1.0

    def test_2d_input(self, rng):
        """Single 2D image should work."""
        img = rng.standard_normal((64, 64))
        assert ssim(img, img) == pytest.approx(1.0, abs=1e-6)


class TestPSNR:
    """Test PSNR metric."""

    def test_identical_images_infinite(self, rng):
        """Identical images should have infinite PSNR."""
        img = rng.standard_normal((5, 64, 64))
        assert psnr(img, img) == float("inf")

    def test_noisy_lower(self, rng):
        """Noisier version should have lower PSNR."""
        img = rng.standard_normal((5, 64, 64))
        mild_noise = img + rng.standard_normal(img.shape) * 0.1
        heavy_noise = img + rng.standard_normal(img.shape) * 1.0
        assert psnr(img, mild_noise) > psnr(img, heavy_noise)
