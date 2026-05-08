"""Integration tests for metrics module."""

import numpy as np
import torch

from downscaling.metrics import crps_energy, crps_paper, ensemble_spread, mae, mass_violation, rmse


class TestCRPSEnergy:
    def test_perfect_ensemble_has_zero_crps(self):
        """If all ensemble members exactly match observation, CRPS ~ 0."""
        obs = np.random.rand(32, 32).astype(np.float32)
        forecasts = np.stack([obs] * 10, axis=0)
        assert abs(crps_energy(obs, forecasts)) < 1e-6

    def test_wider_ensemble_has_higher_crps(self):
        """A wider spread ensemble should have higher CRPS when centered on truth."""
        np.random.seed(0)
        obs = np.zeros((32, 32), dtype=np.float32)
        narrow = np.random.randn(10, 32, 32).astype(np.float32) * 0.1
        wide = np.random.randn(10, 32, 32).astype(np.float32) * 10.0
        assert crps_energy(obs, narrow) < crps_energy(obs, wide)

    def test_single_member_equals_mae(self):
        """With M=1, spread term is 0, so CRPS = MAE."""
        obs = np.ones((32, 32), dtype=np.float32)
        forecast = np.zeros((1, 32, 32), dtype=np.float32)
        crps = crps_energy(obs, forecast)
        expected_mae = np.mean(np.abs(obs - forecast[0]))
        np.testing.assert_allclose(crps, expected_mae, rtol=1e-5)

    def test_crps_nonnegative(self):
        """CRPS should be non-negative for any ensemble."""
        np.random.seed(42)
        obs = np.random.rand(16, 16).astype(np.float32) * 100
        forecasts = np.random.rand(10, 16, 16).astype(np.float32) * 100
        assert crps_energy(obs, forecasts) >= 0

    def test_symmetric_ensemble_is_zero(self):
        """obs=1, forecasts=[0, 2] → CRPS = 0 (ensemble brackets truth symmetrically)."""
        obs = np.ones((4, 4), dtype=np.float32)
        forecasts = np.array([np.zeros((4, 4)), np.full((4, 4), 2.0)], dtype=np.float32)
        np.testing.assert_allclose(crps_energy(obs, forecasts), 0.0, atol=1e-6)

    def test_matches_pairwise_reference(self):
        """O(M log M) sorted formula matches O(M^2) pairwise reference."""
        np.random.seed(99)
        obs = np.random.rand(16, 16).astype(np.float32) * 10
        forecasts = np.random.rand(10, 16, 16).astype(np.float32) * 10
        # Reference: pairwise computation from experiment code
        M = forecasts.shape[0]
        abs_diff = np.mean(np.abs(forecasts - obs[None, ...]), axis=0)
        fc_sorted = np.sort(forecasts, axis=0)
        spread = np.zeros_like(obs)
        for i in range(M):
            for j in range(i + 1, M):
                spread += np.abs(fc_sorted[j] - fc_sorted[i])
        spread = spread * 2.0 / (M * (M - 1))
        ref_crps = float(np.mean(abs_diff - 0.5 * spread))
        np.testing.assert_allclose(crps_energy(obs, forecasts), ref_crps, rtol=1e-5)

    def test_paper_version_differs(self):
        """Paper version (buggy) should differ from correct energy CRPS."""
        np.random.seed(1)
        obs = np.random.rand(128, 128).astype(np.float32)
        forecasts = np.random.rand(10, 128, 128).astype(np.float32)
        # The bug in crps_paper uses shape[-1]**2 in first loop
        # which is 128**2 = 16384 instead of 10**2 = 100
        correct = crps_energy(obs, forecasts)
        buggy = crps_paper(obs, forecasts)
        assert correct != buggy


class TestPointMetrics:
    def test_mae_identical_is_zero(self):
        x = np.ones((10, 10))
        assert mae(x, x) == 0.0

    def test_rmse_identical_is_zero(self):
        x = np.ones((10, 10))
        assert rmse(x, x) == 0.0

    def test_rmse_geq_mae(self):
        """RMSE >= MAE by Jensen's inequality."""
        np.random.seed(0)
        pred = np.random.rand(32, 32)
        target = np.random.rand(32, 32)
        assert rmse(pred, target) >= mae(pred, target)


class TestMassViolation:
    def test_perfect_conservation(self):
        """If avgpool(hr) == lr, violation is 0."""
        lr = torch.ones(1, 1, 32, 32) * 5.0
        hr = torch.ones(1, 1, 128, 128) * 5.0  # uniform = perfect conservation
        assert mass_violation(hr, lr, upsampling_factor=4) < 1e-5

    def test_violation_detects_mismatch(self):
        """Non-conserving prediction has positive violation."""
        lr = torch.ones(1, 1, 32, 32) * 5.0
        hr = torch.ones(1, 1, 128, 128) * 10.0  # 2x too high
        assert mass_violation(hr, lr, upsampling_factor=4) > 0


class TestEnsembleSpread:
    def test_identical_members_zero_spread(self):
        x = np.ones((10, 32, 32))
        assert ensemble_spread(x) == 0.0

    def test_spread_increases_with_diversity(self):
        narrow = np.random.randn(10, 32, 32) * 0.1
        wide = np.random.randn(10, 32, 32) * 10.0
        assert ensemble_spread(narrow) < ensemble_spread(wide)
