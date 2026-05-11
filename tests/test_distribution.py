"""Integration tests for distribution metrics (EMD)."""

import numpy as np
import pytest

from downscaling.metrics.distribution import emd, emd_per_sample


class TestEMD:
    """Earth Mover Distance tests."""

    def test_identical_distributions_zero_distance(self) -> None:
        rng = np.random.default_rng(42)
        fields = rng.standard_normal((50, 32, 32))
        assert emd(fields, fields) == pytest.approx(0.0, abs=1e-10)

    def test_shifted_distribution_detects_bias(self) -> None:
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((50, 32, 32))
        # Shift by 1.0 — EMD should be approximately 1.0
        pred = truth + 1.0
        distance = emd(truth, pred)
        assert distance == pytest.approx(1.0, abs=0.05)

    def test_scaled_distribution_detects_variance_change(self) -> None:
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((100, 32, 32))
        # Scale by 2x — spreads the distribution
        pred = truth * 2.0
        distance = emd(truth, pred)
        assert distance > 0.3  # noticeable distance from variance change

    def test_lower_is_better(self) -> None:
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((50, 32, 32))
        good_pred = truth + rng.normal(0, 0.1, truth.shape)
        bad_pred = truth + rng.normal(0, 1.0, truth.shape)
        assert emd(truth, good_pred) < emd(truth, bad_pred)

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((20, 16, 16))
        pred = rng.standard_normal((20, 16, 16))
        assert emd(truth, pred) >= 0.0

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.standard_normal((30, 16, 16))
        b = rng.standard_normal((30, 16, 16))
        assert emd(a, b) == pytest.approx(emd(b, a), abs=1e-10)

    def test_2d_input_single_field(self) -> None:
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((1, 32, 32))
        pred = truth + 0.5
        distance = emd(truth, pred)
        assert distance == pytest.approx(0.5, abs=0.05)


class TestEMDPerSample:
    """Per-sample EMD tests."""

    def test_shape_matches_n_samples(self) -> None:
        rng = np.random.default_rng(42)
        n = 25
        truth = rng.standard_normal((n, 16, 16))
        pred = rng.standard_normal((n, 16, 16))
        scores = emd_per_sample(truth, pred)
        assert scores.shape == (n,)

    def test_identical_gives_zero(self) -> None:
        rng = np.random.default_rng(42)
        fields = rng.standard_normal((10, 16, 16))
        scores = emd_per_sample(fields, fields)
        np.testing.assert_allclose(scores, 0.0, atol=1e-10)

    def test_all_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((10, 16, 16))
        pred = rng.standard_normal((10, 16, 16))
        scores = emd_per_sample(truth, pred)
        assert np.all(scores >= 0.0)

    def test_mean_consistent_with_batch_emd(self) -> None:
        """Per-sample mean should be in same ballpark as batch EMD.

        Not exactly equal since batch EMD uses all pixels jointly.
        """
        rng = np.random.default_rng(42)
        truth = rng.standard_normal((50, 16, 16))
        pred = truth + rng.normal(0, 0.5, truth.shape)
        batch_distance = emd(truth, pred)
        per_sample_mean = float(np.mean(emd_per_sample(truth, pred)))
        # Should be in same order of magnitude
        assert abs(batch_distance - per_sample_mean) / batch_distance < 1.0


class TestEMDInBatchMetrics:
    """EMD integration with batch_metrics.py."""

    def test_compute_batch_metrics_includes_emd(self) -> None:
        from downscaling.evaluation.batch_metrics import compute_batch_metrics

        rng = np.random.default_rng(42)
        truth = rng.standard_normal((30, 32, 32))
        pred = truth + rng.normal(0, 0.1, truth.shape)
        result = compute_batch_metrics(truth, pred)
        assert "emd" in result
        assert isinstance(result["emd"], float)
        assert result["emd"] >= 0.0
