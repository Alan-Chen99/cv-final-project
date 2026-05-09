"""Integration tests for evaluation pipeline."""

import os

import numpy as np
import torch

from downscaling.constraints import apply_addcl
from downscaling.data import NormStats
from downscaling.evaluation import (
    EvalResult,
    bicubic_predict,
    bilinear_predict,
    evaluate_flow_model,
    load_flow_checkpoint,
)
from downscaling.models.unet import AttentionUNet


class TestDeterministicBaselines:
    def test_bilinear_returns_lr_up(self, sample_data):
        lr_up, _, _, lr_orig = sample_data
        result = bilinear_predict(lr_up[:1], lr_orig[:1])
        torch.testing.assert_close(result, lr_up[:1])

    def test_bicubic_shape(self, sample_data):
        lr_up, _, _, lr_orig = sample_data
        result = bicubic_predict(lr_up[:1], lr_orig[:1])
        assert result.shape == (1, 1, 128, 128)

    def test_bilinear_addcl_zero_mass_violation(self, sample_data):
        """Bilinear + AddCL should have near-zero mass violation."""
        lr_up, _, _, lr_orig = sample_data
        pred = bilinear_predict(lr_up[:1], lr_orig[:1])
        constrained = apply_addcl(pred, lr_orig[:1])
        pool = torch.nn.AvgPool2d(kernel_size=4)
        pooled = pool(constrained)
        torch.testing.assert_close(pooled, lr_orig[:1], atol=1e-5, rtol=1e-5)


class TestLoadFlowCheckpoint:
    def test_roundtrip_save_load(self, device, tmp_path):
        """Save and reload a model checkpoint, verify weights match."""
        model = AttentionUNet(in_channels=2, out_channels=1, base_channels=32, channel_mults=(1, 2))
        stats = NormStats(res_mean=0.0, res_std=1.0, lr_mean=10.0, lr_std=5.0)

        ckpt_path = str(tmp_path / "best_flow.pt")
        stats_path = str(tmp_path / "norm_stats.pt")

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": {},
                "epoch": 10,
                "val_loss": 0.5,
                "args": {
                    "base_channels": 32,
                    "channel_mults_tuple": (1, 2),
                    "attn_heads": 4,
                },
            },
            ckpt_path,
        )
        stats.save(stats_path)

        loaded_model, loaded_stats = load_flow_checkpoint(ckpt_path, stats_path, device=device)

        assert loaded_stats.lr_mean == stats.lr_mean
        assert loaded_stats.lr_std == stats.lr_std
        assert loaded_model.training is False

        # Verify weights match
        for key in model.state_dict():
            torch.testing.assert_close(
                model.state_dict()[key],
                loaded_model.state_dict()[key].cpu(),
            )


    def test_load_without_args_uses_defaults(self, tmp_path):
        """Checkpoints without 'args' key should load with default architecture."""
        model = AttentionUNet(in_channels=2, out_channels=1, base_channels=64, channel_mults=(1, 2, 4))
        stats = NormStats(res_mean=0.0, res_std=1.0, lr_mean=10.0, lr_std=5.0)

        ckpt_path = str(tmp_path / "best_flow.pt")
        stats_path = str(tmp_path / "norm_stats.pt")

        # Save without args (legacy format)
        torch.save(
            {"model": model.state_dict(), "optimizer": {}, "epoch": 5, "val_loss": 0.3},
            ckpt_path,
        )
        stats.save(stats_path)

        loaded_model, _ = load_flow_checkpoint(ckpt_path, stats_path, device="cpu")
        # Default base_channels=64, channel_mults=(1,2,4) should match
        for key in model.state_dict():
            torch.testing.assert_close(
                model.state_dict()[key],
                loaded_model.state_dict()[key],
            )


class TestEvaluateFlowModel:
    def test_flow_eval_runs(self, device):
        """Smoke test: flow evaluation produces finite metrics."""
        data_path = "external/constrained-downscaling"
        if not os.path.exists(f"{data_path}/data/era5_sr_data/test/input_test.pt"):
            import pytest

            pytest.skip("Test data not available")

        model = AttentionUNet(
            in_channels=2, out_channels=1, base_channels=32, channel_mults=(1, 2)
        ).to(device)
        model.eval()
        stats = NormStats(res_mean=0.0, res_std=1.0, lr_mean=22.5, lr_std=17.0)

        result = evaluate_flow_model(
            model,
            stats,
            basedir=data_path,
            split="test",
            n_ensemble=2,
            ode_steps=2,
            max_samples=4,
            device=device,
        )

        assert isinstance(result, EvalResult)
        assert np.isfinite(result.crps)
        assert np.isfinite(result.mae)
        assert result.n_samples == 4
        assert result.n_ensemble == 2
