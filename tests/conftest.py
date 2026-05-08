"""Shared fixtures for integration tests."""

import pytest
import torch


@pytest.fixture
def device():
    """GPU device (tests assume GPU availability)."""
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    return torch.device("cuda")


@pytest.fixture
def sample_data():
    """Synthetic data matching ERA5 TCW4 shapes."""
    torch.manual_seed(42)
    n = 8
    lr_orig = torch.rand(n, 1, 32, 32) * 50  # TCW-like range [0, 50]
    hr = torch.rand(n, 1, 128, 128) * 50
    lr_up = torch.nn.functional.interpolate(
        lr_orig, size=(128, 128), mode="bilinear", align_corners=False
    )
    residual = hr - lr_up
    return lr_up, residual, hr, lr_orig
