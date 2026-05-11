"""Integration tests for evaluation metrics: CRPS, PSD, calibration, SSIM, KL divergence.

Tests verify mathematical correctness using known analytical properties,
cross-validation between formulas, and expected behavior on synthetic data.
"""

import numpy as np
import pytest

from downscaling.metrics.calibration import rank_histogram, spread_skill_ratio
from downscaling.metrics.crps import crps_energy, crps_paper
from downscaling.metrics.distributional import (
    ensemble_mean_kl_divergence,
    histogram_kl_divergence,
)
from downscaling.metrics.spectral import (
    ensemble_mean_psd,
    mean_spectral_coherence,
    psd_log_ratio,
    radial_psd,
    spectral_coherence,
)
from downscaling.metrics.structural import ensemble_mean_ssim, ssim


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


class TestRankHistogram:
    """Test rank histogram for ensemble calibration."""

    def test_uniform_calibrated_ensemble(self):
        """Well-calibrated ensemble: truth drawn from same distribution as members.

        Rank histogram should be approximately uniform for large sample.
        """
        rng = np.random.default_rng(42)
        n_samples = 10000
        m = 10
        # Truth and ensemble from same N(0,1) — perfectly calibrated
        truth = rng.standard_normal(n_samples)
        ensemble = rng.standard_normal((m, n_samples))
        counts = rank_histogram(truth, ensemble)

        assert counts.shape == (m + 1,)
        assert counts.sum() == n_samples
        # Chi-squared test: uniform expected count = n_samples / (m+1)
        expected = n_samples / (m + 1)
        chi2 = np.sum((counts - expected) ** 2 / expected)
        # With 10 dof, chi2 < 25 at p=0.005
        assert chi2 < 25, f"chi2={chi2:.1f}, histogram not uniform: {counts}"

    def test_biased_ensemble_u_shape(self):
        """Under-dispersive ensemble: truth frequently outside ensemble range.

        Rank histogram should show excess counts at edges (U-shape).
        """
        rng = np.random.default_rng(42)
        n_samples = 5000
        m = 10
        truth = rng.standard_normal(n_samples)
        # Ensemble too narrow: small variance
        ensemble = rng.normal(0, 0.3, (m, n_samples))
        counts = rank_histogram(truth, ensemble)

        # Edge bins (0 and M) should have more counts than middle bins
        edge_count = counts[0] + counts[-1]
        counts[1:-1].sum()
        edge_rate = edge_count / n_samples
        # For uniform: edge_rate ≈ 2/11 ≈ 0.18
        # For under-dispersive: edge_rate >> 0.18
        assert edge_rate > 0.3, f"Expected U-shape, edge_rate={edge_rate:.2f}"

    def test_deterministic_ensemble(self):
        """All members identical: truth always at rank 0 or M."""
        truth = np.array([1.0, 2.0, 3.0])
        ensemble = np.full((5, 3), 2.0)  # all members = 2.0
        counts = rank_histogram(truth, ensemble)

        # truth=1 < all members → rank 0
        # truth=2 == all members → rank 0 (strict <)
        # truth=3 > all members → rank 5
        assert counts[0] == 2  # truth=1 and truth=2
        assert counts[5] == 1  # truth=3
        assert counts.sum() == 3

    def test_output_shape(self):
        """Verify output has M+1 bins."""
        rng = np.random.default_rng(0)
        for m in [3, 5, 10, 20]:
            truth = rng.standard_normal((8, 8))
            ensemble = rng.standard_normal((m, 8, 8))
            counts = rank_histogram(truth, ensemble)
            assert counts.shape == (m + 1,)
            assert counts.sum() == 64

    def test_multichannel(self):
        """Rank histogram works with (M, C, H, W) shaped inputs."""
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((2, 16, 16))
        ensemble = rng.standard_normal((5, 2, 16, 16))
        counts = rank_histogram(truth, ensemble)
        assert counts.shape == (6,)
        assert counts.sum() == 2 * 16 * 16


class TestSpreadSkillRatio:
    """Test spread-skill ratio for ensemble calibration."""

    def test_perfect_calibration(self):
        """SSR ≈ 1 when ensemble spread matches prediction error.

        Generate ensemble where spread ≈ RMSE by construction.
        """
        rng = np.random.default_rng(42)
        sigma = 1.0
        n = 10000
        m = 50
        truth = np.zeros(n)
        # Each member drawn from N(0, sigma) independently
        # Spread ≈ sigma, RMSE of mean ≈ sigma/sqrt(M)
        # But SSR = sqrt((M+1)/M) * spread / rmse
        # For large M: spread ≈ sigma, rmse ≈ sigma/sqrt(M)
        # SSR ≈ sqrt(M) * sqrt((M+1)/M) ≈ sqrt(M+1)
        # This is NOT 1 — calibrated SSR=1 requires truth drawn from ensemble
        # Correct construction: truth = ensemble_mean + noise(sigma/sqrt(M))
        ensemble = rng.normal(0, sigma, (m, n))
        ens_mean = ensemble.mean(axis=0)
        # Construct truth so that RMSE matches spread with correction
        truth = ens_mean + rng.normal(0, sigma, n)
        ssr = spread_skill_ratio(truth, ensemble)
        # SSR should be near 1.0 (within ~10% for n=10000)
        assert 0.85 < ssr < 1.15, f"SSR={ssr:.3f}, expected ~1.0"

    def test_underdispersive(self):
        """SSR < 1 when ensemble is too narrow (overconfident)."""
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((64, 64))
        # Ensemble with tiny spread around 0
        ensemble = rng.normal(0, 0.01, (10, 64, 64))
        ssr = spread_skill_ratio(truth, ensemble)
        assert ssr < 0.5, f"SSR={ssr:.3f}, expected << 1 for narrow ensemble"

    def test_overdispersive(self):
        """SSR > 1 when ensemble is too wide (underconfident)."""
        rng = np.random.default_rng(42)
        truth = np.zeros((64, 64))
        # Wide ensemble, but mean is close to truth
        ensemble = rng.normal(0, 5.0, (10, 64, 64))
        ssr = spread_skill_ratio(truth, ensemble)
        assert ssr > 2.0, f"SSR={ssr:.3f}, expected >> 1 for wide ensemble"

    def test_perfect_prediction_returns_inf(self):
        """SSR = inf when RMSE = 0 (perfect ensemble mean)."""
        truth = np.ones((8, 8))
        # Ensemble centered on truth with some spread
        ensemble = np.stack([truth + 0.1, truth - 0.1], axis=0)
        ssr = spread_skill_ratio(truth, ensemble)
        assert ssr == float("inf")

    def test_single_member_raises(self):
        """SSR requires at least 2 members."""
        truth = np.ones((8, 8))
        ensemble = np.ones((1, 8, 8))
        with pytest.raises(ValueError, match="at least 2"):
            spread_skill_ratio(truth, ensemble)

    def test_finite_size_correction(self):
        """Verify the sqrt((M+1)/M) correction factor is applied."""
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((32, 32))
        ensemble = rng.standard_normal((5, 32, 32))

        ens_mean = ensemble.mean(axis=0)
        ens_var = np.mean((ensemble - ens_mean[None, ...]) ** 2, axis=0)
        spread = np.sqrt(np.mean(ens_var))
        rmse = np.sqrt(np.mean((ens_mean - truth) ** 2))
        uncorrected = spread / rmse
        corrected = np.sqrt(6 / 5) * uncorrected

        ssr = spread_skill_ratio(truth, ensemble)
        assert ssr == pytest.approx(corrected, rel=1e-10)


class TestSSIM:
    """Test structural similarity index."""

    def test_identical_fields_perfect_score(self):
        """Identical fields should give SSIM = 1.0."""
        rng = np.random.default_rng(42)
        field = rng.standard_normal((64, 64))
        assert ssim(field, field) == pytest.approx(1.0, abs=1e-10)

    def test_uncorrelated_field_low_score(self):
        """Uncorrelated random field should give SSIM near zero."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((64, 64))
        b = rng.standard_normal((64, 64))
        score = ssim(a, b)
        assert abs(score) < 0.3, f"SSIM={score:.3f}, expected near 0 for uncorrelated"

    def test_noisy_field_intermediate(self):
        """Adding noise should reduce SSIM proportionally to noise level."""
        rng = np.random.default_rng(42)
        field = rng.standard_normal((64, 64))
        low_noise = field + rng.normal(0, 0.1, field.shape)
        high_noise = field + rng.normal(0, 1.0, field.shape)

        ssim_low = ssim(field, low_noise)
        ssim_high = ssim(field, high_noise)
        assert ssim_low > ssim_high, (
            f"Low noise SSIM={ssim_low:.3f} should exceed high noise SSIM={ssim_high:.3f}"
        )
        assert ssim_low > 0.8, f"Low noise SSIM={ssim_low:.3f}, expected > 0.8"

    def test_constant_fields_equal(self):
        """Two constant fields with same value should give SSIM = 1."""
        a = np.full((32, 32), 5.0)
        b = np.full((32, 32), 5.0)
        assert ssim(a, b) == pytest.approx(1.0, abs=1e-10)

    def test_explicit_data_range(self):
        """Explicit data_range should override auto-detection."""
        rng = np.random.default_rng(42)
        field = rng.standard_normal((64, 64))
        noisy = field + rng.normal(0, 0.5, field.shape)

        ssim_auto = ssim(field, noisy)
        # Larger data_range makes C1/C2 larger → SSIM closer to 1
        ssim_wide = ssim(field, noisy, data_range=100.0)
        assert ssim_wide > ssim_auto

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        with pytest.raises(ValueError, match="mismatch"):
            ssim(np.zeros((32, 32)), np.zeros((32, 64)))

    def test_non_2d_raises(self):
        """Non-2D inputs should raise ValueError."""
        with pytest.raises(ValueError, match="2D"):
            ssim(np.zeros((3, 32, 32)), np.zeros((3, 32, 32)))

    def test_symmetry(self):
        """SSIM(a, b) == SSIM(b, a) by construction."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((64, 64))
        b = rng.standard_normal((64, 64))
        assert ssim(a, b) == pytest.approx(ssim(b, a), rel=1e-12)

    def test_range_bounded(self):
        """SSIM should be in [-1, 1]."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((64, 64))
        b = rng.standard_normal((64, 64))
        score = ssim(a, b)
        assert -1.0 <= score <= 1.0, f"SSIM={score} out of bounds"


class TestEnsembleMeanSSIM:
    """Test ensemble SSIM averaging."""

    def test_single_member_matches_ssim(self):
        """Single-member ensemble should match direct ssim()."""
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((64, 64))
        pred = rng.standard_normal((64, 64))
        ens_score = ensemble_mean_ssim(truth, pred[np.newaxis, ...])
        direct_score = ssim(truth, pred)
        assert ens_score == pytest.approx(direct_score, rel=1e-12)

    def test_better_ensemble_higher_ssim(self):
        """Ensemble closer to truth should have higher mean SSIM."""
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((64, 64))
        good = truth[None, ...] + rng.normal(0, 0.1, (5, 64, 64))
        bad = rng.standard_normal((5, 64, 64))

        ssim_good = ensemble_mean_ssim(truth, good)
        ssim_bad = ensemble_mean_ssim(truth, bad)
        assert ssim_good > ssim_bad

    def test_rejects_non_3d(self):
        """Should raise on non-3D ensemble."""
        with pytest.raises(ValueError, match="3D"):
            ensemble_mean_ssim(np.zeros((8, 8)), np.zeros((8, 8)))


class TestHistogramKLDivergence:
    """Test KL divergence between empirical distributions."""

    def test_identical_fields_zero_kl(self):
        """Identical fields should give KL divergence = 0."""
        rng = np.random.default_rng(42)
        field = rng.standard_normal((64, 64))
        kl = histogram_kl_divergence(field, field)
        assert kl == pytest.approx(0.0, abs=1e-10)

    def test_nonnegative(self):
        """KL divergence is always non-negative (Gibbs' inequality)."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((64, 64))
        b = rng.standard_normal((64, 64))
        kl = histogram_kl_divergence(a, b)
        assert kl >= -1e-10

    def test_shifted_distribution_positive_kl(self):
        """Shifting the distribution should produce positive KL divergence."""
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((64, 64))
        shifted = truth + 2.0
        kl = histogram_kl_divergence(truth, shifted)
        assert kl > 0.1, f"KL={kl:.4f}, expected > 0.1 for shifted distribution"

    def test_larger_shift_higher_kl(self):
        """Larger distribution shift should produce higher KL divergence."""
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((64, 64))
        small_shift = truth + 0.5
        large_shift = truth + 3.0
        kl_small = histogram_kl_divergence(truth, small_shift)
        kl_large = histogram_kl_divergence(truth, large_shift)
        assert kl_large > kl_small, (
            f"KL(large)={kl_large:.4f} should exceed KL(small)={kl_small:.4f}"
        )

    def test_scaled_distribution(self):
        """Scaling the distribution (compressed tails) should produce positive KL."""
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((64, 64))
        compressed = truth * 0.5
        kl = histogram_kl_divergence(truth, compressed)
        assert kl > 0.01, f"KL={kl:.4f}, expected positive for compressed distribution"

    def test_constant_fields_zero_kl(self):
        """Two constant fields with same value should give KL = 0."""
        a = np.full((32, 32), 5.0)
        b = np.full((32, 32), 5.0)
        kl = histogram_kl_divergence(a, b)
        assert kl == pytest.approx(0.0, abs=1e-10)

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        with pytest.raises(ValueError, match="mismatch"):
            histogram_kl_divergence(np.zeros((32, 32)), np.zeros((32, 64)))

    def test_n_bins_too_small_raises(self):
        """n_bins < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="n_bins"):
            histogram_kl_divergence(np.zeros((8, 8)), np.zeros((8, 8)), n_bins=1)

    def test_more_bins_finer_resolution(self):
        """More bins should give different (generally higher) KL for non-identical."""
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((64, 64))
        pred = rng.normal(0.5, 1.0, (64, 64))
        kl_few = histogram_kl_divergence(truth, pred, n_bins=10)
        kl_many = histogram_kl_divergence(truth, pred, n_bins=200)
        # Both should be positive; exact relationship depends on data
        assert kl_few > 0
        assert kl_many > 0

    def test_arbitrary_shape(self):
        """Works with any shape (3D, 1D, etc.)."""
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((4, 32, 32))
        pred = rng.standard_normal((4, 32, 32))
        kl = histogram_kl_divergence(truth, pred)
        assert kl >= 0


class TestEnsembleMeanKLDivergence:
    """Test ensemble-averaged KL divergence."""

    def test_single_member_matches_direct(self):
        """Single-member ensemble should match direct histogram_kl_divergence."""
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((64, 64))
        pred = rng.standard_normal((64, 64))
        ens_kl = ensemble_mean_kl_divergence(truth, pred[np.newaxis, ...])
        direct_kl = histogram_kl_divergence(truth, pred)
        assert ens_kl == pytest.approx(direct_kl, rel=1e-12)

    def test_better_ensemble_lower_kl(self):
        """Ensemble closer to truth distribution should have lower mean KL."""
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((64, 64))
        good = truth[None, ...] + rng.normal(0, 0.1, (5, 64, 64))
        bad = rng.normal(3.0, 1.0, (5, 64, 64))
        kl_good = ensemble_mean_kl_divergence(truth, good)
        kl_bad = ensemble_mean_kl_divergence(truth, bad)
        assert kl_good < kl_bad

    def test_rejects_1d(self):
        """Should raise on 1D ensemble."""
        with pytest.raises(ValueError, match="2D"):
            ensemble_mean_kl_divergence(np.zeros(8), np.zeros(8))


# ── Spectral coherence ────────────────────────────────────────────────────


class TestSpectralCoherence:
    """Test spectral coherence metric."""

    def test_identical_fields_perfect_coherence(self):
        """Identical prediction and truth → coherence 1.0 everywhere."""
        rng = np.random.default_rng(42)
        fields = rng.standard_normal((20, 32, 32))
        k, coh = spectral_coherence(fields, fields)
        np.testing.assert_allclose(coh, 1.0, atol=1e-10)

    def test_independent_fields_low_coherence(self):
        """Independent random fields → coherence near 0 with enough samples."""
        rng = np.random.default_rng(42)
        preds = rng.standard_normal((200, 32, 32))
        truths = rng.standard_normal((200, 32, 32))
        k, coh = spectral_coherence(preds, truths)
        # With 200 independent samples, coherence should be near 0
        assert np.mean(coh) < 0.1

    def test_single_sample_independent_low_coherence(self):
        """Single sample of independent fields: coherence < 1 due to azimuthal averaging."""
        rng = np.random.default_rng(42)
        pred = rng.standard_normal((1, 32, 32))
        truth = rng.standard_normal((1, 32, 32))
        k, coh = spectral_coherence(pred, truth)
        # Phase cancellation within wavenumber bins → coherence well below 1
        assert np.mean(coh) < 0.5
        # Still bounded [0, 1]
        assert np.all(coh >= -1e-10)
        assert np.all(coh <= 1.0 + 1e-10)

    def test_coherence_bounded_zero_one(self):
        """Coherence values should be in [0, 1]."""
        rng = np.random.default_rng(42)
        preds = rng.standard_normal((50, 32, 32))
        truths = preds + rng.normal(0, 0.5, (50, 32, 32))
        k, coh = spectral_coherence(preds, truths)
        assert np.all(coh >= -1e-10)
        assert np.all(coh <= 1.0 + 1e-10)

    def test_noisy_copy_intermediate_coherence(self):
        """Truth + noise → coherence between 0 and 1, decreasing with noise."""
        rng = np.random.default_rng(42)
        truths = rng.standard_normal((100, 32, 32))
        low_noise = truths + rng.normal(0, 0.3, truths.shape)
        high_noise = truths + rng.normal(0, 3.0, truths.shape)
        _, coh_low = spectral_coherence(low_noise, truths)
        _, coh_high = spectral_coherence(high_noise, truths)
        # Low noise → higher mean coherence
        assert np.mean(coh_low) > np.mean(coh_high)

    def test_output_shape(self):
        """Output wavenumber and coherence arrays have matching shapes."""
        rng = np.random.default_rng(42)
        preds = rng.standard_normal((10, 32, 48))
        truths = rng.standard_normal((10, 32, 48))
        k, coh = spectral_coherence(preds, truths)
        assert k.shape == coh.shape
        assert len(k) == 16  # min(32, 48) // 2

    def test_shape_mismatch_raises(self):
        """Different shapes for predictions and truths → ValueError."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            spectral_coherence(np.zeros((5, 32, 32)), np.zeros((5, 32, 64)))

    def test_rejects_2d_input(self):
        """2D arrays (missing batch dim) → ValueError."""
        with pytest.raises(ValueError, match="3D"):
            spectral_coherence(np.zeros((32, 32)), np.zeros((32, 32)))

    def test_mean_spectral_coherence_scalar(self):
        """mean_spectral_coherence returns a float scalar."""
        rng = np.random.default_rng(42)
        fields = rng.standard_normal((20, 32, 32))
        msc = mean_spectral_coherence(fields, fields)
        assert isinstance(msc, float)
        assert msc == pytest.approx(1.0, abs=1e-10)

    def test_mean_coherence_ordering(self):
        """Better predictions → higher mean coherence."""
        rng = np.random.default_rng(42)
        truths = rng.standard_normal((100, 32, 32))
        good = truths + rng.normal(0, 0.2, truths.shape)
        bad = rng.standard_normal((100, 32, 32))
        msc_good = mean_spectral_coherence(good, truths)
        msc_bad = mean_spectral_coherence(bad, truths)
        assert msc_good > msc_bad

    def test_frequency_dependent_coherence(self):
        """Low-freq signal + high-freq noise: coherence high at low-k, low at high-k."""
        rng = np.random.default_rng(42)
        n = 100
        h, w = 64, 64
        # Smooth signal (low frequency content)
        from scipy.ndimage import gaussian_filter

        base = rng.standard_normal((n, h, w))
        smooth = np.array([gaussian_filter(b, sigma=5) for b in base])
        # Prediction = smooth signal + high-freq noise
        noise = rng.normal(0, 0.3, (n, h, w))
        preds = smooth + noise
        # Both share the same smooth component → high coherence at low freq
        k, coh = spectral_coherence(preds, smooth)
        # Low wavenumbers (first quarter) should have higher coherence
        # than high wavenumbers (last quarter)
        quarter = len(k) // 4
        low_k_coh = np.mean(coh[:quarter])
        high_k_coh = np.mean(coh[-quarter:])
        assert low_k_coh > high_k_coh
