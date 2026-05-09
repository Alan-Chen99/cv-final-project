"""Integration tests for constraint layers.

Tests verify that AddCL and SmCL enforce physical conservation laws:
- avgpool(constrained_output) == lr_orig exactly
- SmCL additionally enforces non-negativity
"""

import pytest
import torch

from downscaling.constraints.layers import apply_addcl, apply_smcl


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def lr_orig(device) -> torch.Tensor:
    """Random LR field, shape (B=4, 1, 32, 32)."""
    torch.manual_seed(42)
    return torch.rand(4, 1, 32, 32, device=device) * 10 + 1  # positive values


@pytest.fixture
def pred_hr(device) -> torch.Tensor:
    """Random unconstrained HR prediction, shape (B=4, 1, 128, 128)."""
    torch.manual_seed(99)
    return torch.randn(4, 1, 128, 128, device=device) * 5 + 3


class TestAddCL:
    """Test additive constraint layer."""

    def test_conservation_exact(self, pred_hr, lr_orig):
        """After AddCL, avgpool(output) == lr_orig exactly."""
        result = apply_addcl(pred_hr, lr_orig, upsampling_factor=4)
        pool = torch.nn.AvgPool2d(kernel_size=4)
        pooled = pool(result)
        torch.testing.assert_close(pooled, lr_orig, atol=1e-5, rtol=1e-5)

    def test_output_shape_preserved(self, pred_hr, lr_orig):
        """Output shape matches input HR shape."""
        result = apply_addcl(pred_hr, lr_orig)
        assert result.shape == pred_hr.shape

    def test_already_constrained_unchanged(self, device):
        """If input already satisfies constraint, AddCL is identity."""
        torch.manual_seed(0)
        lr = torch.ones(1, 1, 8, 8, device=device) * 5.0
        # Create HR that already satisfies constraint
        hr = torch.ones(1, 1, 32, 32, device=device) * 5.0
        result = apply_addcl(hr, lr, upsampling_factor=4)
        torch.testing.assert_close(result, hr, atol=1e-5, rtol=1e-5)

    def test_different_upsampling_factors(self, device):
        """AddCL works with different upsampling factors."""
        torch.manual_seed(0)
        for factor in [2, 4, 8]:
            lr_size = 16
            hr_size = lr_size * factor
            lr = torch.rand(2, 1, lr_size, lr_size, device=device)
            hr = torch.randn(2, 1, hr_size, hr_size, device=device)
            result = apply_addcl(hr, lr, upsampling_factor=factor)
            pool = torch.nn.AvgPool2d(kernel_size=factor)
            pooled = pool(result)
            torch.testing.assert_close(pooled, lr, atol=1e-5, rtol=1e-5)


class TestSmCL:
    """Test softmax constraint layer."""

    def test_conservation_exact(self, lr_orig, device):
        """After SmCL, avgpool(output) == lr_orig exactly."""
        torch.manual_seed(42)
        pred_log = torch.randn(4, 1, 128, 128, device=device)  # log-space input
        result = apply_smcl(pred_log, lr_orig, upsampling_factor=4)
        pool = torch.nn.AvgPool2d(kernel_size=4)
        pooled = pool(result)
        torch.testing.assert_close(pooled, lr_orig, atol=1e-4, rtol=1e-4)

    def test_non_negativity(self, lr_orig, device):
        """SmCL output is always non-negative."""
        torch.manual_seed(42)
        pred_log = torch.randn(4, 1, 128, 128, device=device) * 3  # wide range
        result = apply_smcl(pred_log, lr_orig)
        assert (result >= 0).all(), "SmCL output contains negative values"

    def test_output_shape_preserved(self, lr_orig, device):
        """Output shape matches input HR shape."""
        torch.manual_seed(42)
        pred_log = torch.randn(4, 1, 128, 128, device=device)
        result = apply_smcl(pred_log, lr_orig)
        assert result.shape == (4, 1, 128, 128)

    def test_different_upsampling_factors(self, device):
        """SmCL works with different upsampling factors."""
        torch.manual_seed(0)
        for factor in [2, 4, 8]:
            lr_size = 16
            hr_size = lr_size * factor
            lr = torch.rand(2, 1, lr_size, lr_size, device=device) * 5 + 1
            pred = torch.randn(2, 1, hr_size, hr_size, device=device)
            result = apply_smcl(pred, lr, upsampling_factor=factor)
            pool = torch.nn.AvgPool2d(kernel_size=factor)
            pooled = pool(result)
            torch.testing.assert_close(pooled, lr, atol=1e-4, rtol=1e-4)
            assert (result >= 0).all()
