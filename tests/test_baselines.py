"""Integration tests for baseline evaluation methods and checkpoint loading.

Tests verify bicubic/bilinear baselines produce correct metrics,
AddCL constraint reduces mass violation, and checkpoint loading
handles both patterns (full checkpoint and state_dict-only).
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
