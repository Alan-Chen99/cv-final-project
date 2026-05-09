"""Integration tests for ERA5 and NorESM data loading.

Tests verify that the data loaders correctly read and process
actual data from the pool disk.
"""

import pytest
import torch

from downscaling.data.era5 import load_era5_tcw
from downscaling.data.noresm import load_noresm_tas

DATA_DIR = "/home/chenxy/orcd/pool/datasets"


@pytest.fixture(scope="module")
def test_data():
    """Load test split once for all tests in this module."""
    return load_era5_tcw(DATA_DIR, "test")


class TestERA5Loading:
    """Test ERA5 data loading with actual pool data."""

    def test_load_returns_four_tensors(self, test_data):
        """load_era5_tcw returns a tuple of 4 tensors."""
        lr_up, residual, hr, lr_orig = test_data
        assert isinstance(lr_up, torch.Tensor)
        assert isinstance(residual, torch.Tensor)
        assert isinstance(hr, torch.Tensor)
        assert isinstance(lr_orig, torch.Tensor)

    def test_shapes_correct(self, test_data):
        """Shapes match expected dimensions for 4x SR."""
        lr_up, residual, hr, lr_orig = test_data
        n = lr_up.shape[0]
        assert lr_up.shape == (n, 1, 128, 128)
        assert residual.shape == (n, 1, 128, 128)
        assert hr.shape == (n, 1, 128, 128)
        assert lr_orig.shape == (n, 1, 32, 32)

    def test_residual_equals_hr_minus_lr_up(self, test_data):
        """residual == hr - lr_up by construction."""
        lr_up, residual, hr, _ = test_data
        torch.testing.assert_close(residual, hr - lr_up, atol=1e-5, rtol=1e-5)

    def test_lr_up_is_upsampled_lr(self, test_data):
        """lr_up is bilinear upsampling of lr_orig."""
        lr_up, _, _, lr_orig = test_data
        expected = torch.nn.functional.interpolate(
            lr_orig,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )
        torch.testing.assert_close(lr_up, expected, atol=1e-5, rtol=1e-5)

    def test_test_split_size(self, test_data):
        """Test split should have 10K samples."""
        lr_up, _, _, _ = test_data
        assert lr_up.shape[0] == 10000

    def test_values_finite(self, test_data):
        """All values should be finite (no NaN or Inf)."""
        for tensor in test_data:
            assert torch.isfinite(tensor).all()

    def test_val_split_loads(self):
        """Validation split loads correctly."""
        lr_up, _, _, _ = load_era5_tcw(DATA_DIR, "val")
        assert lr_up.shape[0] == 10000
        assert lr_up.shape[1:] == (1, 128, 128)


@pytest.fixture(scope="module")
def noresm_test_data():
    """Load NorESM test split once for all tests in this module."""
    return load_noresm_tas(DATA_DIR, "test")


class TestNorESMLoading:
    """Test NorESM data loading with actual pool data."""

    def test_load_returns_four_tensors(self, noresm_test_data):
        lr_up, residual, hr, lr_orig = noresm_test_data
        assert isinstance(lr_up, torch.Tensor)
        assert isinstance(residual, torch.Tensor)
        assert isinstance(hr, torch.Tensor)
        assert isinstance(lr_orig, torch.Tensor)

    def test_shapes_correct(self, noresm_test_data):
        """Shapes match expected dimensions for 2x SR (32->64)."""
        lr_up, residual, hr, lr_orig = noresm_test_data
        n = lr_up.shape[0]
        assert lr_up.shape == (n, 1, 64, 64)
        assert residual.shape == (n, 1, 64, 64)
        assert hr.shape == (n, 1, 64, 64)
        assert lr_orig.shape == (n, 1, 32, 32)

    def test_residual_equals_hr_minus_lr_up(self, noresm_test_data):
        lr_up, residual, hr, _ = noresm_test_data
        torch.testing.assert_close(residual, hr - lr_up, atol=1e-5, rtol=1e-5)

    def test_lr_up_is_upsampled_lr(self, noresm_test_data):
        lr_up, _, _, lr_orig = noresm_test_data
        expected = torch.nn.functional.interpolate(
            lr_orig,
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        )
        torch.testing.assert_close(lr_up, expected, atol=1e-5, rtol=1e-5)

    def test_test_split_size(self, noresm_test_data):
        """Test split should have 12384 samples."""
        lr_up, _, _, _ = noresm_test_data
        assert lr_up.shape[0] == 12384

    def test_values_finite(self, noresm_test_data):
        for tensor in noresm_test_data:
            assert torch.isfinite(tensor).all()

    def test_temperature_range(self, noresm_test_data):
        """Surface temperature should be in physically reasonable range (K)."""
        _, _, hr, lr_orig = noresm_test_data
        assert lr_orig.min() > 190  # below coldest recorded
        assert lr_orig.max() < 340  # above hottest recorded
        assert hr.min() > 190
        assert hr.max() < 340
