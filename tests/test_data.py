"""Integration tests for data module."""

import torch

from downscaling.data import NormStats, compute_norm_stats, denormalize, normalize


class TestNormalization:
    def test_normalize_denormalize_roundtrip(self):
        torch.manual_seed(0)
        data = torch.randn(100, 1, 128, 128)
        mean, std = data.mean().item(), data.std().item()
        normed = normalize(data, mean, std)
        recovered = denormalize(normed, mean, std)
        torch.testing.assert_close(recovered, data, atol=1e-5, rtol=1e-5)

    def test_normalized_has_unit_stats(self):
        torch.manual_seed(0)
        data = torch.randn(1000, 1, 32, 32) * 5 + 10
        mean, std = data.mean().item(), data.std().item()
        normed = normalize(data, mean, std)
        assert abs(normed.mean().item()) < 0.01
        assert abs(normed.std().item() - 1.0) < 0.01


class TestNormStats:
    def test_save_load_roundtrip(self, tmp_path):
        stats = NormStats(res_mean=1.0, res_std=2.0, lr_mean=3.0, lr_std=4.0)
        path = tmp_path / "stats.pt"
        stats.save(path)
        loaded = NormStats.load(path)
        assert loaded.res_mean == stats.res_mean
        assert loaded.res_std == stats.res_std
        assert loaded.lr_mean == stats.lr_mean
        assert loaded.lr_std == stats.lr_std

    def test_compute_norm_stats(self):
        torch.manual_seed(0)
        lr = torch.randn(100, 1, 128, 128) * 3 + 5
        res = torch.randn(100, 1, 128, 128) * 0.5
        stats = compute_norm_stats(lr, res)
        assert abs(stats.lr_mean - 5.0) < 0.1
        assert abs(stats.lr_std - 3.0) < 0.1
        assert abs(stats.res_mean) < 0.05
        assert abs(stats.res_std - 0.5) < 0.05
