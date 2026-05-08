"""Integration tests for constraint layers."""

import torch

from downscaling.constraints import apply_addcl, apply_smcl


class TestAddCL:
    def test_exact_conservation(self):
        """AddCL must enforce exact mass conservation: avgpool(out) == lr."""
        torch.manual_seed(0)
        lr = torch.rand(4, 1, 32, 32) * 50
        pred_hr = torch.rand(4, 1, 128, 128) * 50

        constrained = apply_addcl(pred_hr, lr)

        pool = torch.nn.AvgPool2d(kernel_size=4)
        pooled = pool(constrained)
        torch.testing.assert_close(pooled, lr, atol=1e-5, rtol=1e-5)

    def test_preserves_shape(self):
        lr = torch.rand(2, 1, 32, 32)
        pred = torch.rand(2, 1, 128, 128)
        out = apply_addcl(pred, lr)
        assert out.shape == pred.shape

    def test_already_conserving_is_unchanged(self):
        """If prediction already satisfies constraint, AddCL is identity."""
        lr = torch.ones(1, 1, 32, 32) * 5.0
        pred = torch.ones(1, 1, 128, 128) * 5.0
        out = apply_addcl(pred, lr)
        torch.testing.assert_close(out, pred, atol=1e-5, rtol=1e-5)


class TestSmCL:
    def test_non_negativity(self):
        """SmCL output must be non-negative."""
        torch.manual_seed(0)
        lr = torch.rand(4, 1, 32, 32) * 50 + 0.1  # positive LR
        pred_hr = torch.randn(4, 1, 128, 128)  # can be negative

        constrained = apply_smcl(pred_hr, lr)
        assert (constrained >= 0).all()

    def test_conservation(self):
        """SmCL must enforce avgpool(out) == lr."""
        torch.manual_seed(0)
        lr = torch.rand(4, 1, 32, 32) * 10 + 0.1
        pred_hr = torch.randn(4, 1, 128, 128) * 0.5  # small values to avoid overflow

        constrained = apply_smcl(pred_hr, lr)

        pool = torch.nn.AvgPool2d(kernel_size=4)
        pooled = pool(constrained)
        torch.testing.assert_close(pooled, lr, atol=1e-4, rtol=1e-4)

    def test_preserves_shape(self):
        lr = torch.rand(2, 1, 32, 32) + 0.1
        pred = torch.randn(2, 1, 128, 128)
        out = apply_smcl(pred, lr)
        assert out.shape == pred.shape
