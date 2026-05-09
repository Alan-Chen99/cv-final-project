"""Integration tests for SwinIR evaluation and channel adaptation.

Tests verify:
- 3ch->1ch channel adaptation preserves model functionality
- Zero-shot prediction with per-sample normalization
- Finetuned prediction with global normalization
- Deterministic evaluation metrics (CRPS=MAE, AddCL constraint)
- Finetuned checkpoint loading
"""

import numpy as np
import pytest
import torch

from downscaling.constraints.layers import apply_addcl
from downscaling.evaluation.swinir import (
    _eval_deterministic,
    eval_swinir_finetuned,
    eval_swinir_zeroshot,
    load_swinir_1ch,
    load_swinir_finetuned,
    load_swinir_pretrained,
    predict_swinir_finetuned,
    predict_swinir_zeroshot,
)

POOL_DIR = "/home/chenxy/orcd/pool/datasets"
PRETRAINED_WEIGHTS = (
    f"{POOL_DIR}/research5/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
)
FINETUNED_CHECKPOINT = f"{POOL_DIR}/spatial-4x-add-v2/models/swinir_ft/best_swinir.pt"


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def swinir_3ch():
    """Load pretrained 3ch SwinIR once for all tests in this module."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_swinir_pretrained(PRETRAINED_WEIGHTS, device=device)


@pytest.fixture(scope="module")
def swinir_1ch():
    """Load channel-adapted 1ch SwinIR once for all tests in this module."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_swinir_1ch(PRETRAINED_WEIGHTS, device=device)


class TestChannelAdaptation:
    """Test 3ch -> 1ch channel adaptation logic."""

    def test_conv_first_1ch(self, swinir_1ch):
        """Adapted model's conv_first should accept 1 input channel."""
        assert swinir_1ch.conv_first.in_channels == 1

    def test_conv_last_1ch(self, swinir_1ch):
        """Adapted model's conv_last should output 1 channel."""
        assert swinir_1ch.conv_last.out_channels == 1

    def test_mean_buffer_zeroed(self, swinir_1ch):
        """Mean buffer should be zeroed for external normalization."""
        if hasattr(swinir_1ch, "mean"):
            expected = torch.zeros(1, 1, 1, 1, device=swinir_1ch.mean.device)
            torch.testing.assert_close(swinir_1ch.mean, expected)

    def test_img_range(self, swinir_1ch):
        """img_range should be 1.0 for [0,1] normalized input."""
        if hasattr(swinir_1ch, "img_range"):
            assert swinir_1ch.img_range == 1.0

    def test_forward_shape(self, swinir_1ch, device):
        """1ch model produces (1, 1, 128, 128) from (1, 1, 32, 32)."""
        x = torch.randn(1, 1, 32, 32, device=device)
        with torch.no_grad():
            out = swinir_1ch(x)
        assert out.shape == (1, 1, 128, 128)

    def test_forward_finite(self, swinir_1ch, device):
        """Output values should be finite (no NaN/Inf)."""
        x = torch.rand(2, 1, 32, 32, device=device)
        with torch.no_grad():
            out = swinir_1ch(x)
        assert torch.isfinite(out).all()


class TestZeroShotPrediction:
    """Test zero-shot prediction with per-sample normalization."""

    def test_output_shape(self, swinir_3ch, device):
        """Output shape is (N, 1, 128, 128)."""
        torch.manual_seed(42)
        lr = torch.rand(2, 1, 32, 32) * 10 + 1
        preds = predict_swinir_zeroshot(swinir_3ch, lr, device=str(device))
        assert preds.shape == (2, 1, 128, 128)

    def test_constant_field(self, swinir_3ch, device):
        """Constant input -> constant output (per-sample norm collapses to x_min)."""
        lr = torch.ones(1, 1, 32, 32) * 5.0
        preds = predict_swinir_zeroshot(swinir_3ch, lr, device=str(device))
        # x_max - x_min < 1e-8 -> output = x_min = 5.0
        torch.testing.assert_close(preds, torch.ones(1, 1, 128, 128) * 5.0, atol=1e-5, rtol=1e-5)

    def test_output_finite(self, swinir_3ch, device):
        """All values should be finite."""
        torch.manual_seed(99)
        lr = torch.rand(3, 1, 32, 32) * 20
        preds = predict_swinir_zeroshot(swinir_3ch, lr, device=str(device))
        assert torch.isfinite(preds).all()

    def test_output_on_cpu(self, swinir_3ch, device):
        """Predictions should always be returned on CPU."""
        torch.manual_seed(42)
        lr = torch.rand(2, 1, 32, 32) * 10
        preds = predict_swinir_zeroshot(swinir_3ch, lr, device=str(device))
        assert preds.device == torch.device("cpu")


class TestFinetunedPrediction:
    """Test finetuned prediction with global normalization."""

    def test_output_shape(self, swinir_1ch, device):
        """Output shape is (N, 1, 128, 128)."""
        torch.manual_seed(42)
        lr = torch.rand(2, 1, 32, 32) * 10
        preds = predict_swinir_finetuned(swinir_1ch, lr, vmin=0.0, vmax=50.0, device=str(device))
        assert preds.shape == (2, 1, 128, 128)

    def test_output_finite(self, swinir_1ch, device):
        """All values should be finite."""
        torch.manual_seed(42)
        lr = torch.rand(4, 1, 32, 32) * 10
        preds = predict_swinir_finetuned(swinir_1ch, lr, vmin=0.0, vmax=50.0, device=str(device))
        assert torch.isfinite(preds).all()

    def test_denormalization(self, swinir_1ch, device):
        """Output is in physical units (denormalized), not [0,1]."""
        torch.manual_seed(42)
        vmin, vmax = 10.0, 50.0
        lr = torch.rand(2, 1, 32, 32) * (vmax - vmin) + vmin
        preds = predict_swinir_finetuned(swinir_1ch, lr, vmin=vmin, vmax=vmax, device=str(device))
        # Denormalized values should exceed [0,1] range
        assert preds.max() > 1.0

    def test_output_on_cpu(self, swinir_1ch, device):
        """Predictions should always be returned on CPU."""
        torch.manual_seed(42)
        lr = torch.rand(2, 1, 32, 32) * 10
        preds = predict_swinir_finetuned(swinir_1ch, lr, vmin=0.0, vmax=50.0, device=str(device))
        assert preds.device == torch.device("cpu")

    def test_batch_processing(self, swinir_1ch, device):
        """Large input is handled correctly across batches."""
        torch.manual_seed(42)
        lr = torch.rand(10, 1, 32, 32) * 10
        preds = predict_swinir_finetuned(
            swinir_1ch, lr, vmin=0.0, vmax=50.0, device=str(device), batch_size=3
        )
        assert preds.shape == (10, 1, 128, 128)


class TestFinetunedCheckpointLoading:
    """Test finetuned checkpoint loading."""

    def test_returns_model_and_stats(self, device):
        """load_swinir_finetuned returns (model, vmin, vmax)."""
        model, vmin, vmax = load_swinir_finetuned(
            PRETRAINED_WEIGHTS, FINETUNED_CHECKPOINT, device=str(device)
        )
        assert isinstance(vmin, float)
        assert isinstance(vmax, float)
        assert vmin < vmax

    def test_model_eval_mode(self, device):
        """Loaded model should be in eval mode."""
        model, _, _ = load_swinir_finetuned(
            PRETRAINED_WEIGHTS, FINETUNED_CHECKPOINT, device=str(device)
        )
        assert not model.training

    def test_1ch_convs(self, device):
        """Finetuned model should have 1ch conv layers."""
        model, _, _ = load_swinir_finetuned(
            PRETRAINED_WEIGHTS, FINETUNED_CHECKPOINT, device=str(device)
        )
        assert model.conv_first.in_channels == 1
        assert model.conv_last.out_channels == 1

    def test_forward_pass(self, device):
        """Finetuned model produces correct output shape."""
        model, vmin, vmax = load_swinir_finetuned(
            PRETRAINED_WEIGHTS, FINETUNED_CHECKPOINT, device=str(device)
        )
        x = torch.rand(1, 1, 32, 32, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 128, 128)


class TestEvalDeterministic:
    """Test _eval_deterministic metric computation."""

    def test_perfect_prediction(self):
        """Perfect predictions yield zero metrics."""
        hr = torch.randn(4, 1, 128, 128)
        lr = torch.nn.functional.avg_pool2d(hr, kernel_size=4)
        result = _eval_deterministic(hr, hr, lr)
        assert result["crps"] == pytest.approx(0.0, abs=1e-10)
        assert result["mae"] == pytest.approx(0.0, abs=1e-10)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-10)

    def test_expected_keys(self):
        """Result dict has all expected metric keys."""
        hr = torch.randn(2, 1, 128, 128)
        pred = torch.randn(2, 1, 128, 128)
        lr = torch.randn(2, 1, 32, 32)
        result = _eval_deterministic(pred, hr, lr)
        assert set(result.keys()) == {"crps", "mae", "rmse", "mass_violation"}

    def test_crps_equals_mae(self):
        """For single-member (deterministic), CRPS = MAE."""
        torch.manual_seed(42)
        hr = torch.randn(4, 1, 128, 128)
        pred = torch.randn(4, 1, 128, 128)
        lr = torch.randn(4, 1, 32, 32)
        result = _eval_deterministic(pred, hr, lr)
        assert result["crps"] == pytest.approx(result["mae"], rel=1e-5)

    def test_metrics_nonnegative(self):
        """All metrics should be non-negative."""
        torch.manual_seed(123)
        hr = torch.randn(4, 1, 128, 128)
        pred = torch.randn(4, 1, 128, 128)
        lr = torch.randn(4, 1, 32, 32)
        result = _eval_deterministic(pred, hr, lr)
        for key in ("crps", "mae", "rmse", "mass_violation"):
            assert result[key] >= 0, f"{key} is negative: {result[key]}"

    def test_addcl_eliminates_mass_violation(self):
        """AddCL should reduce mass violation to near zero."""
        torch.manual_seed(42)
        lr = torch.rand(4, 1, 32, 32) * 10
        pred = torch.randn(4, 1, 128, 128) * 5
        hr = torch.randn(4, 1, 128, 128)
        result_raw = _eval_deterministic(pred, hr, lr)
        pred_cl = apply_addcl(pred, lr, upsampling_factor=4)
        result_cl = _eval_deterministic(pred_cl, hr, lr)
        assert result_cl["mass_violation"] < 1e-5
        assert result_cl["mass_violation"] < result_raw["mass_violation"]

    def test_metrics_finite(self):
        """All metrics should be finite."""
        torch.manual_seed(42)
        hr = torch.randn(4, 1, 128, 128)
        pred = torch.randn(4, 1, 128, 128)
        lr = torch.randn(4, 1, 32, 32)
        result = _eval_deterministic(pred, hr, lr)
        for key in ("crps", "mae", "rmse", "mass_violation"):
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"


class TestEndToEnd:
    """End-to-end SwinIR evaluation (zero-shot + finetuned)."""

    def test_eval_zeroshot(self, device):
        """eval_swinir_zeroshot produces valid metrics."""
        torch.manual_seed(42)
        lr = torch.rand(4, 1, 32, 32) * 10
        hr = torch.rand(4, 1, 128, 128) * 10
        result = eval_swinir_zeroshot(hr, lr, PRETRAINED_WEIGHTS, device=str(device))
        for key in ("crps", "mae", "rmse", "mass_violation"):
            assert np.isfinite(result[key])

    def test_eval_zeroshot_addcl(self, device):
        """Zero-shot + AddCL has near-zero mass violation."""
        torch.manual_seed(42)
        lr = torch.rand(4, 1, 32, 32) * 10
        hr = torch.rand(4, 1, 128, 128) * 10
        result = eval_swinir_zeroshot(
            hr, lr, PRETRAINED_WEIGHTS, device=str(device), with_addcl=True
        )
        assert result["mass_violation"] < 1e-5

    def test_eval_finetuned(self, device):
        """eval_swinir_finetuned produces valid metrics."""
        torch.manual_seed(42)
        lr = torch.rand(4, 1, 32, 32) * 10
        hr = torch.rand(4, 1, 128, 128) * 10
        result = eval_swinir_finetuned(
            hr, lr, PRETRAINED_WEIGHTS, FINETUNED_CHECKPOINT, device=str(device)
        )
        for key in ("crps", "mae", "rmse", "mass_violation"):
            assert np.isfinite(result[key])

    def test_eval_finetuned_addcl(self, device):
        """Finetuned + AddCL has near-zero mass violation."""
        torch.manual_seed(42)
        lr = torch.rand(4, 1, 32, 32) * 10
        hr = torch.rand(4, 1, 128, 128) * 10
        result = eval_swinir_finetuned(
            hr,
            lr,
            PRETRAINED_WEIGHTS,
            FINETUNED_CHECKPOINT,
            device=str(device),
            with_addcl=True,
        )
        assert result["mass_violation"] < 1e-5
