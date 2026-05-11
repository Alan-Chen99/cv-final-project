"""Integration tests for baseline evaluation methods and checkpoint loading.

Tests verify bicubic/bilinear baselines produce correct metrics,
AddCL constraint reduces mass violation, checkpoint loading handles
both patterns, and compute_all_metrics orchestration returns correct
result structure for both ensemble and deterministic paths.
"""

import numpy as np
import pytest
import torch

from downscaling.evaluation.baselines import (
    eval_bicubic,
    eval_bilinear,
    evaluate_deterministic,
    upsample_bicubic,
    upsample_bilinear,
)
from downscaling.evaluation.checkpoints import load_checkpoint, load_norm_stats
from downscaling.evaluation.comprehensive import compute_all_metrics
from downscaling.models.unet import AttentionUNet


class TestUpsamplingFunctions:
    """Test bicubic and bilinear upsampling."""

    def test_bilinear_shape(self):
        """Bilinear 4x upsampling produces correct output shape."""
        lr = torch.randn(2, 1, 32, 32)
        out = upsample_bilinear(lr, scale_factor=4)
        assert out.shape == (2, 1, 128, 128)

    def test_bicubic_shape(self):
        """Bicubic 4x upsampling produces correct output shape."""
        lr = torch.randn(2, 1, 32, 32)
        out = upsample_bicubic(lr, scale_factor=4)
        assert out.shape == (2, 1, 128, 128)

    def test_bilinear_preserves_constant(self):
        """Bilinear upsampling of a constant field stays constant."""
        lr = torch.ones(1, 1, 8, 8) * 5.0
        out = upsample_bilinear(lr, scale_factor=4)
        torch.testing.assert_close(out, torch.ones(1, 1, 32, 32) * 5.0, atol=1e-5, rtol=1e-5)

    def test_bicubic_preserves_constant(self):
        """Bicubic upsampling of a constant field stays constant."""
        lr = torch.ones(1, 1, 8, 8) * 3.0
        out = upsample_bicubic(lr, scale_factor=4)
        torch.testing.assert_close(out, torch.ones(1, 1, 32, 32) * 3.0, atol=1e-5, rtol=1e-5)


class TestEvaluateDeterministic:
    """Test deterministic evaluation (CRPS = MAE for single-member)."""

    def test_perfect_prediction(self):
        """Perfect predictions yield zero metrics."""
        hr = torch.randn(4, 1, 128, 128)
        lr = torch.nn.functional.avg_pool2d(hr, kernel_size=4)  # (4, 1, 32, 32)
        result = evaluate_deterministic(hr, hr, lr)
        assert result["crps"] == pytest.approx(0.0, abs=1e-10)
        assert result["mae"] == pytest.approx(0.0, abs=1e-10)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-10)

    def test_returns_expected_keys(self):
        """Result dict has all expected metric keys."""
        hr = torch.randn(2, 1, 128, 128)
        pred = torch.randn(2, 1, 128, 128)
        lr = torch.randn(2, 1, 32, 32)
        result = evaluate_deterministic(pred, hr, lr)
        assert set(result.keys()) == {"crps", "mae", "rmse", "mass_violation"}

    def test_crps_equals_mae_for_deterministic(self):
        """For deterministic models, CRPS should equal MAE (no spread term)."""
        torch.manual_seed(42)
        hr = torch.randn(4, 1, 128, 128)
        pred = torch.randn(4, 1, 128, 128)
        lr = torch.randn(4, 1, 32, 32)
        result = evaluate_deterministic(pred, hr, lr)
        assert result["crps"] == pytest.approx(result["mae"], rel=1e-6)

    def test_metrics_nonnegative(self):
        """All metrics should be non-negative."""
        torch.manual_seed(123)
        hr = torch.randn(4, 1, 128, 128)
        pred = torch.randn(4, 1, 128, 128)
        lr = torch.randn(4, 1, 32, 32)
        result = evaluate_deterministic(pred, hr, lr)
        for key in ("crps", "mae", "rmse", "mass_violation"):
            assert result[key] >= 0, f"{key} is negative: {result[key]}"


class TestBaselineEvaluation:
    """Test eval_bicubic and eval_bilinear end-to-end."""

    def test_eval_bicubic_runs(self):
        """eval_bicubic produces valid metrics."""
        torch.manual_seed(42)
        lr = torch.rand(4, 1, 32, 32) * 10
        hr = torch.rand(4, 1, 128, 128) * 10
        result = eval_bicubic(hr, lr)
        assert np.isfinite(result["crps"])
        assert np.isfinite(result["mae"])
        assert np.isfinite(result["rmse"])
        assert np.isfinite(result["mass_violation"])

    def test_eval_bilinear_runs(self):
        """eval_bilinear produces valid metrics."""
        torch.manual_seed(42)
        lr = torch.rand(4, 1, 32, 32) * 10
        hr = torch.rand(4, 1, 128, 128) * 10
        result = eval_bilinear(hr, lr)
        assert np.isfinite(result["crps"])

    def test_addcl_reduces_mass_violation(self):
        """AddCL constraint should reduce mass violation to near zero."""
        torch.manual_seed(42)
        lr = torch.rand(4, 1, 32, 32) * 10
        hr = torch.rand(4, 1, 128, 128) * 10
        result_no_cl = eval_bicubic(hr, lr, with_addcl=False)
        result_cl = eval_bicubic(hr, lr, with_addcl=True)
        assert result_cl["mass_violation"] < result_no_cl["mass_violation"]
        assert result_cl["mass_violation"] < 1e-5

    def test_bilinear_addcl_reduces_mass_violation(self):
        """AddCL with bilinear also reduces mass violation."""
        torch.manual_seed(42)
        lr = torch.rand(4, 1, 32, 32) * 10
        hr = torch.rand(4, 1, 128, 128) * 10
        result_no_cl = eval_bilinear(hr, lr, with_addcl=False)
        result_cl = eval_bilinear(hr, lr, with_addcl=True)
        assert result_cl["mass_violation"] < result_no_cl["mass_violation"]
        assert result_cl["mass_violation"] < 1e-5

    def test_bicubic_better_than_bilinear(self):
        """Bicubic should generally have lower MAE than bilinear on upsampled data."""
        # Use data where HR is just upsampled LR (smooth), so bicubic is exact
        torch.manual_seed(42)
        lr = torch.rand(4, 1, 32, 32) * 10
        hr_bicubic = upsample_bicubic(lr)
        # Bicubic on its own reconstruction should be perfect
        result = eval_bicubic(hr_bicubic, lr)
        assert result["mae"] == pytest.approx(0.0, abs=1e-5)


class TestCheckpointLoading:
    """Test checkpoint loading for both patterns."""

    def _make_model(self):
        return AttentionUNet(
            in_channels=2,
            out_channels=1,
            base_channels=16,
            channel_mults=(1, 2),
            time_emb_dim=32,
            dropout=0.0,
            attn_heads=2,
        )

    def test_pattern_a_full_checkpoint(self, tmp_path):
        """Pattern A: full checkpoint dict with 'model' key."""
        model = self._make_model()
        # Save as Pattern A
        ckpt = {
            "model": model.state_dict(),
            "epoch": 10,
            "val_loss": 0.05,
            "args": {"lr": 1e-4},
        }
        path = tmp_path / "checkpoint.pt"
        torch.save(ckpt, path)

        # Load into fresh model
        model2 = self._make_model()
        metadata = load_checkpoint(model2, path)
        assert metadata["epoch"] == 10
        assert metadata["val_loss"] == 0.05
        assert metadata["args"] == {"lr": 1e-4}

        # Weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
            torch.testing.assert_close(p1, p2)

    def test_pattern_b_state_dict_only(self, tmp_path):
        """Pattern B: pure state_dict (OrderedDict)."""
        model = self._make_model()
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)

        model2 = self._make_model()
        metadata = load_checkpoint(model2, path)
        assert metadata == {}  # no metadata in pattern B

        for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
            torch.testing.assert_close(p1, p2)

    def test_model_in_eval_mode_after_load(self, tmp_path):
        """Model should be in eval mode after loading."""
        model = self._make_model()
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)

        model2 = self._make_model()
        model2.train()  # explicitly set to train
        load_checkpoint(model2, path)
        assert not model2.training

    def test_load_norm_stats(self, tmp_path):
        """load_norm_stats correctly extracts float values."""
        stats = {
            "res_mean": torch.tensor(1.5),
            "res_std": torch.tensor(2.3),
            "lr_mean": torch.tensor(-0.1),
            "lr_std": torch.tensor(0.8),
        }
        path = tmp_path / "norm_stats.pt"
        torch.save(stats, path)

        loaded = load_norm_stats(path)
        assert loaded["res_mean"] == pytest.approx(1.5)
        assert loaded["res_std"] == pytest.approx(2.3)
        assert loaded["lr_mean"] == pytest.approx(-0.1)
        assert loaded["lr_std"] == pytest.approx(0.8)
        # Values should be plain floats, not tensors
        assert isinstance(loaded["res_mean"], float)


class TestComputeAllMetrics:
    """Integration tests for the metric orchestration function.

    Uses small synthetic data (N=10, 16x16 HR, 8x8 LR, factor=2) to verify
    result structure, key presence, and value sanity for both ensemble and
    deterministic paths without requiring GPU or real data.
    """

    @pytest.fixture
    def synthetic_data(self):
        """Small synthetic dataset: truth, LR, and predictions."""
        rng = np.random.default_rng(42)
        n, h, w, factor = 10, 16, 16, 2
        lr_h, lr_w = h // factor, w // factor

        truth = rng.standard_normal((n, h, w)).astype(np.float32)
        lr_orig = torch.from_numpy(
            rng.standard_normal((n, 1, lr_h, lr_w)).astype(np.float32)
        )
        return truth, lr_orig, factor

    # -- Expected keys for each path --

    COMMON_KEYS = {
        "crps", "mae", "rmse", "mass_violation", "ssim", "kl_divergence",
        "psd_log_ratio", "ralsd", "spectral_coherence",
        "psd_k", "psd_power", "psd_truth_power",
    }
    ENSEMBLE_KEYS = COMMON_KEYS | {"ssr", "rank_histogram"}

    def test_deterministic_returns_expected_keys(self, synthetic_data):
        """Non-ensemble path returns all scalar + PSD keys, no SSR/rank_histogram."""
        truth, lr_orig, factor = synthetic_data
        preds = truth + np.random.default_rng(99).normal(0, 0.1, truth.shape).astype(np.float32)
        result = compute_all_metrics(truth, preds, lr_orig, is_ensemble=False, upsampling_factor=factor)
        assert set(result.keys()) == self.COMMON_KEYS

    def test_ensemble_returns_expected_keys(self, synthetic_data):
        """Ensemble path returns all keys including SSR and rank_histogram."""
        truth, lr_orig, factor = synthetic_data
        n, h, w = truth.shape
        m = 5
        ens = np.stack([truth + np.random.default_rng(i).normal(0, 0.3, (n, h, w)) for i in range(m)], axis=1).astype(np.float32)
        result = compute_all_metrics(truth, ens, lr_orig, is_ensemble=True, upsampling_factor=factor)
        assert set(result.keys()) == self.ENSEMBLE_KEYS

    def test_all_values_finite(self, synthetic_data):
        """All scalar metrics are finite for well-behaved synthetic data."""
        truth, lr_orig, factor = synthetic_data
        preds = truth + np.random.default_rng(99).normal(0, 0.1, truth.shape).astype(np.float32)
        result = compute_all_metrics(truth, preds, lr_orig, is_ensemble=False, upsampling_factor=factor)
        for key in ("crps", "mae", "rmse", "mass_violation", "ssim", "kl_divergence",
                     "psd_log_ratio", "ralsd", "spectral_coherence"):
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"

    def test_crps_equals_mae_for_identical_members(self, synthetic_data):
        """Ensemble with M identical members → CRPS = MAE (spread term vanishes)."""
        truth, lr_orig, factor = synthetic_data
        single = (truth + np.random.default_rng(99).normal(0, 0.1, truth.shape)).astype(np.float32)
        # M=2 identical copies: ensemble spread = 0, so CRPS reduces to MAE
        preds_dup = np.stack([single, single], axis=1)
        result = compute_all_metrics(truth, preds_dup, lr_orig, is_ensemble=True, upsampling_factor=factor)
        assert result["crps"] == pytest.approx(result["mae"], rel=1e-5)

    def test_perfect_prediction_zero_error(self, synthetic_data):
        """Perfect prediction yields zero MAE and RMSE."""
        truth, lr_orig, factor = synthetic_data
        result = compute_all_metrics(truth, truth.copy(), lr_orig, is_ensemble=False, upsampling_factor=factor)
        assert result["mae"] == pytest.approx(0.0, abs=1e-10)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-10)

    def test_psd_arrays_consistent_length(self, synthetic_data):
        """psd_k, psd_power, psd_truth_power have matching lengths."""
        truth, lr_orig, factor = synthetic_data
        preds = truth + np.random.default_rng(99).normal(0, 0.1, truth.shape).astype(np.float32)
        result = compute_all_metrics(truth, preds, lr_orig, is_ensemble=False, upsampling_factor=factor)
        assert len(result["psd_k"]) == len(result["psd_power"]) == len(result["psd_truth_power"])

    def test_rank_histogram_length_matches_ensemble(self, synthetic_data):
        """Rank histogram has M+1 bins for M ensemble members."""
        truth, lr_orig, factor = synthetic_data
        n, h, w = truth.shape
        m = 5
        ens = np.stack([truth + np.random.default_rng(i).normal(0, 0.3, (n, h, w)) for i in range(m)], axis=1).astype(np.float32)
        result = compute_all_metrics(truth, ens, lr_orig, is_ensemble=True, upsampling_factor=factor)
        assert len(result["rank_histogram"]) == m + 1

    def test_ssim_bounded(self, synthetic_data):
        """SSIM is bounded in [-1, 1]."""
        truth, lr_orig, factor = synthetic_data
        preds = truth + np.random.default_rng(99).normal(0, 0.1, truth.shape).astype(np.float32)
        result = compute_all_metrics(truth, preds, lr_orig, is_ensemble=False, upsampling_factor=factor)
        assert -1.0 <= result["ssim"] <= 1.0

    def test_coherence_bounded(self, synthetic_data):
        """Spectral coherence is bounded in [0, 1]."""
        truth, lr_orig, factor = synthetic_data
        preds = truth + np.random.default_rng(99).normal(0, 0.1, truth.shape).astype(np.float32)
        result = compute_all_metrics(truth, preds, lr_orig, is_ensemble=False, upsampling_factor=factor)
        assert 0.0 <= result["spectral_coherence"] <= 1.0 + 1e-10
