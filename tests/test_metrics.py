"""Integration tests for CRPS and spectral metrics.

Tests verify mathematical correctness of CRPS implementations
using known analytical properties and cross-validation between formulas.
Tests verify PSD implementation against analytical Fourier results.
"""

import numpy as np
import pytest

from downscaling.metrics.crps import crps_energy, crps_paper
from downscaling.metrics.spectral import (
    ensemble_mean_psd,
    psd_log_ratio,
    radial_psd,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestCRPSEnergy:
    """Test crps_energy with known analytical properties."""

    def test_perfect_ensemble_zero_crps(self):
        """Perfect deterministic predictions => CRPS equals 0."""
        obs = np.ones((128, 128), dtype=np.float64)
        # All 10 members predict exactly the observation
        forecasts = np.stack([obs] * 10, axis=0)
        crps = crps_energy(obs, forecasts)
        assert crps == pytest.approx(0.0, abs=1e-10)

    def test_single_member_equals_mae(self):
        """Single member ensemble: CRPS = MAE (spread term is 0)."""
        rng = np.random.default_rng(42)
        obs = rng.standard_normal((128, 128))
        forecasts = rng.standard_normal((1, 128, 128))
        crps = crps_energy(obs, forecasts)
        mae = float(np.mean(np.abs(obs - forecasts[0])))
        assert crps == pytest.approx(mae, rel=1e-10)

    def test_wider_ensemble_higher_spread(self, rng):
        """Wider ensemble should have larger spread (lower CRPS if centered)."""
        obs = np.zeros((64, 64), dtype=np.float64)
        # Narrow ensemble: small perturbations
        narrow = rng.normal(0, 0.01, (20, 64, 64))
        # Wide ensemble: large perturbations
        wide = rng.normal(0, 1.0, (20, 64, 64))

        crps_narrow = crps_energy(obs, narrow)
        crps_wide = crps_energy(obs, wide)
        # Both centered at obs, but narrow should have lower CRPS
        assert crps_narrow < crps_wide

    def test_crps_nonnegative(self, rng):
        """CRPS should be non-negative for any inputs."""
        obs = rng.standard_normal((32, 32))
        forecasts = rng.standard_normal((5, 32, 32))
        crps = crps_energy(obs, forecasts)
        assert crps >= -1e-10  # allow small numerical noise

    def test_multichannel_correct(self, rng):
        """CRPS on (M, C, H, W) agrees with per-channel mean."""
        obs = rng.standard_normal((3, 32, 32))
        forecasts = rng.standard_normal((8, 3, 32, 32))
        crps_joint = crps_energy(obs, forecasts)
        # Per-channel CRPS should average to the joint result
        per_channel = [crps_energy(obs[c], forecasts[:, c]) for c in range(3)]
        assert crps_joint == pytest.approx(np.mean(per_channel), rel=1e-10)

    def test_two_member_analytical(self):
        """Two-member ensemble: spread = mean |x1 - x2|."""
        obs = np.array([[0.0]], dtype=np.float64)
        f1 = np.array([[[1.0]]], dtype=np.float64)
        f2 = np.array([[[-1.0]]], dtype=np.float64)
        forecasts = np.stack([f1[0], f2[0]], axis=0)
        crps = crps_energy(obs, forecasts)
        # E|X-y| = (|1-0| + |-1-0|)/2 = 1.0
        # E|X-X'| = |1-(-1)|/1 = 2.0 (only 1 pair, normalization 2/(2*1) = 1)
        # CRPS = 1.0 - 0.5*2.0 = 0.0
        assert crps == pytest.approx(0.0, abs=1e-10)


class TestCRPSPaper:
    """Test crps_paper implementation."""

    def test_perfect_prediction_zero(self):
        """Perfect predictions should give zero CRPS."""
        obs = np.ones((32, 32), dtype=np.float64)
        forecasts = np.stack([obs] * 5, axis=0)
        crps = crps_paper(obs, forecasts)
        assert crps == pytest.approx(0.0, abs=1e-10)

    def test_single_member_matches_energy(self):
        """Single member: both CRPS formulations should equal MAE."""
        rng = np.random.default_rng(123)
        obs = rng.standard_normal((32, 32))
        forecasts = rng.standard_normal((1, 32, 32))
        crps_e = crps_energy(obs, forecasts)
        crps_p = crps_paper(obs, forecasts)
        # Both should equal MAE for single member
        mae = float(np.mean(np.abs(obs - forecasts[0])))
        assert crps_e == pytest.approx(mae, rel=1e-6)
        assert crps_p == pytest.approx(mae, rel=1e-6)

    def test_nonnegative(self):
        """Paper CRPS should be non-negative."""
        rng = np.random.default_rng(99)
        obs = rng.standard_normal((32, 32))
        forecasts = rng.standard_normal((10, 32, 32))
        crps = crps_paper(obs, forecasts)
        assert crps >= -1e-10


class TestCRPSConsistency:
    """Cross-validate energy and paper CRPS implementations."""

    def test_both_formulas_agree_approximately(self):
        """Both CRPS formulations should produce similar values for same inputs."""
        rng = np.random.default_rng(77)
        obs = rng.standard_normal((32, 32))
        forecasts = rng.standard_normal((20, 32, 32))
        crps_e = crps_energy(obs, forecasts)
        crps_p = crps_paper(obs, forecasts)
        # Both formulations should agree closely for M=20
        assert crps_e == pytest.approx(crps_p, rel=0.05)


class TestRadialPSD:
    """Test radially averaged power spectral density."""

    def test_white_noise_flat_spectrum(self):
        """White noise should have approximately flat PSD across wavenumbers."""
        rng = np.random.default_rng(42)
        field = rng.standard_normal((128, 128))
        k, power = radial_psd(field)

        # Flat = low coefficient of variation (std/mean)
        # White noise PSD variance decreases with more bins, but for 128x128
        # the CV should be well below 1.0
        cv = np.std(power) / np.mean(power)
        assert cv < 0.5, f"White noise CV={cv:.2f}, expected roughly flat spectrum"

    def test_single_frequency_peak(self):
        """A pure sinusoid should produce a peak at the correct wavenumber."""
        n = 128
        target_k = 10
        y = np.arange(n)
        x = np.arange(n)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        field = np.sin(2 * np.pi * target_k * xx / n)

        k, power = radial_psd(field)
        peak_k = k[np.argmax(power)]
        assert abs(peak_k - target_k) <= 1, f"Peak at k={peak_k}, expected ~{target_k}"

    def test_high_frequency_more_power(self):
        """Higher-frequency content should show more power at higher wavenumbers."""
        n = 128
        y = np.arange(n)
        x = np.arange(n)
        yy, xx = np.meshgrid(y, x, indexing="ij")

        low_freq = np.sin(2 * np.pi * 3 * xx / n)
        high_freq = np.sin(2 * np.pi * 30 * xx / n)

        _, power_low = radial_psd(low_freq)
        _, power_high = radial_psd(high_freq)

        # High-frequency field should have more power in high-k range
        mid = len(power_low) // 2
        ratio_low = np.sum(power_low[mid:]) / (np.sum(power_low) + 1e-30)
        ratio_high = np.sum(power_high[mid:]) / (np.sum(power_high) + 1e-30)
        assert ratio_high > ratio_low

    def test_constant_field_no_power(self):
        """Constant field should have zero power at all nonzero wavenumbers."""
        field = np.ones((64, 64)) * 5.0
        k, power = radial_psd(field)
        # k starts at 1 (DC excluded), so all power should be ~0
        assert np.max(power) < 1e-10

    def test_output_shapes(self):
        """Verify output shapes are correct."""
        field = np.random.default_rng(0).standard_normal((64, 128))
        k, power = radial_psd(field)
        k_max = min(64, 128) // 2
        assert k.shape == (k_max,)
        assert power.shape == (k_max,)
        assert k[0] == 1.0
        assert k[-1] == k_max

    def test_rejects_non_2d(self):
        """Should raise on non-2D input."""
        with pytest.raises(ValueError, match="2D"):
            radial_psd(np.zeros((3, 4, 5)))

    def test_square_vs_rectangular(self):
        """Should handle both square and rectangular fields."""
        rng = np.random.default_rng(99)
        for shape in [(64, 64), (64, 128), (128, 64)]:
            field = rng.standard_normal(shape)
            k, power = radial_psd(field)
            assert len(k) == min(shape) // 2
            assert np.all(power >= 0)


class TestEnsembleMeanPSD:
    """Test ensemble PSD averaging."""

    def test_single_member_matches_radial_psd(self):
        """Single-member ensemble should match direct radial_psd."""
        rng = np.random.default_rng(42)
        field = rng.standard_normal((128, 128))
        k_single, p_single = radial_psd(field)
        k_ens, p_ens = ensemble_mean_psd(field[np.newaxis, ...])
        np.testing.assert_array_equal(k_single, k_ens)
        np.testing.assert_allclose(p_single, p_ens, rtol=1e-12)

    def test_ensemble_reduces_variance(self):
        """Averaging PSD over ensemble should reduce variance vs single member."""
        rng = np.random.default_rng(42)
        fields = rng.standard_normal((20, 64, 64))
        _, p_ens = ensemble_mean_psd(fields)
        _, p_single = radial_psd(fields[0])

        # Ensemble-averaged white noise PSD should be flatter (lower CV)
        cv_ens = np.std(p_ens) / np.mean(p_ens)
        cv_single = np.std(p_single) / np.mean(p_single)
        assert cv_ens < cv_single

    def test_rejects_non_3d(self):
        """Should raise on non-3D input."""
        with pytest.raises(ValueError, match="3D"):
            ensemble_mean_psd(np.zeros((4, 5)))


class TestPSDLogRatio:
    """Test PSD log-ratio scalar metric."""

    def test_identical_spectra_zero(self):
        """Identical spectra should give log-ratio of 0."""
        k = np.arange(1.0, 33)
        p = np.random.default_rng(0).uniform(1, 100, size=32)
        assert psd_log_ratio(k, p, k, p) == pytest.approx(0.0, abs=1e-12)

    def test_factor_of_10_gives_1(self):
        """Power scaled by 10x should give log-ratio of 1.0."""
        k = np.arange(1.0, 33)
        p = np.ones(32) * 100.0
        p_scaled = p * 10.0
        assert psd_log_ratio(k, p_scaled, k, p) == pytest.approx(1.0, abs=1e-12)

    def test_mismatched_grids_raises(self):
        """Should raise if wavenumber grids differ."""
        k1 = np.arange(1.0, 10)
        k2 = np.arange(1.0, 11)
        with pytest.raises(ValueError, match="match"):
            psd_log_ratio(k1, np.ones(9), k2, np.ones(10))

    def test_all_zero_power_returns_inf(self):
        """All-zero power should return inf."""
        k = np.arange(1.0, 5)
        result = psd_log_ratio(k, np.zeros(4), k, np.zeros(4))
        assert result == float("inf")
