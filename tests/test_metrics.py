"""Integration tests for CRPS metrics.

Tests verify mathematical correctness of CRPS implementations
using known analytical properties and cross-validation between formulas.
"""

import numpy as np
import pytest

from downscaling.metrics.crps import crps_energy, crps_paper


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

    def test_multichannel_shape(self, rng):
        """CRPS handles (M, C, H, W) shape."""
        obs = rng.standard_normal((3, 32, 32))
        forecasts = rng.standard_normal((8, 3, 32, 32))
        crps = crps_energy(obs, forecasts)
        assert isinstance(crps, float)
        assert np.isfinite(crps)

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
        # They use different formulations, so allow some numerical difference
        # but they should be in the same ballpark
        assert crps_e == pytest.approx(crps_p, rel=0.15)
