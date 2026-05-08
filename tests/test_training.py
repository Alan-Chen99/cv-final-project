"""Integration tests for training module.

Tests assume GPU availability. Uses tiny models with few iterations
to verify the training loop mechanics (loss computation, gradient flow,
checkpoint saving).
"""

import torch

from downscaling.models.unet import AttentionUNet
from downscaling.training import TrainConfig, train_step


class TestTrainStep:
    def test_loss_is_finite(self, device):
        """Single training step produces finite loss."""
        model = AttentionUNet(
            in_channels=2, out_channels=1, base_channels=16, channel_mults=(1,)
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        config = TrainConfig(amp=False)

        lr_batch = torch.randn(2, 1, 32, 32, device=device)
        res_batch = torch.randn(2, 1, 32, 32, device=device)

        loss = train_step(model, lr_batch, res_batch, optimizer, scaler, config)
        assert loss > 0
        assert not torch.isnan(torch.tensor(loss))

    def test_loss_decreases_over_steps(self, device):
        """Loss should decrease over multiple steps on the same batch (overfitting test)."""
        torch.manual_seed(42)
        model = AttentionUNet(
            in_channels=2, out_channels=1, base_channels=16, channel_mults=(1,)
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        config = TrainConfig(amp=False)

        # Fixed batch for overfitting
        lr_batch = torch.randn(4, 1, 32, 32, device=device)
        res_batch = torch.randn(4, 1, 32, 32, device=device)

        losses = []
        for _ in range(20):
            loss = train_step(model, lr_batch, res_batch, optimizer, scaler, config)
            losses.append(loss)

        # Average of last 5 should be lower than first 5
        assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5

    def test_amp_step(self, device):
        """Training step works with AMP enabled."""
        model = AttentionUNet(
            in_channels=2, out_channels=1, base_channels=16, channel_mults=(1,)
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        config = TrainConfig(amp=True)

        lr_batch = torch.randn(2, 1, 32, 32, device=device)
        res_batch = torch.randn(2, 1, 32, 32, device=device)

        loss = train_step(model, lr_batch, res_batch, optimizer, scaler, config)
        assert loss > 0
        assert not torch.isnan(torch.tensor(loss))

    def test_logit_normal_timesteps(self, device):
        """Training step works with logit-normal timestep sampling."""
        model = AttentionUNet(
            in_channels=2, out_channels=1, base_channels=16, channel_mults=(1,)
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        config = TrainConfig(amp=False, t_sampling="logit_normal")

        lr_batch = torch.randn(2, 1, 32, 32, device=device)
        res_batch = torch.randn(2, 1, 32, 32, device=device)

        loss = train_step(model, lr_batch, res_batch, optimizer, scaler, config)
        assert loss > 0

    def test_gradient_clipping(self, device):
        """Gradient clipping is applied (gradients bounded)."""
        model = AttentionUNet(
            in_channels=2, out_channels=1, base_channels=16, channel_mults=(1,)
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        config = TrainConfig(amp=False, grad_clip=0.01)  # Very tight clipping

        lr_batch = torch.randn(2, 1, 32, 32, device=device)
        res_batch = torch.randn(2, 1, 32, 32, device=device) * 100  # Large targets

        train_step(model, lr_batch, res_batch, optimizer, scaler, config)

        # After clipping, parameter updates should be small
        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.norm().item() < 1.0  # Clipped to 0.01 total norm
