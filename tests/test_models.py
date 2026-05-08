"""Integration tests for model architectures.

Tests assume GPU availability and verify forward pass shapes/gradients.
"""

import torch

from downscaling.models.ddpm import DDPMSchedule, ddim_sample
from downscaling.models.dit import DiT
from downscaling.models.unet import AttentionUNet


class TestAttentionUNet:
    def test_forward_shape(self, device):
        model = AttentionUNet(
            in_channels=2, out_channels=1, base_channels=32, channel_mults=(1, 2)
        ).to(device)
        x = torch.randn(2, 1, 128, 128, device=device)
        t = torch.rand(2, device=device)
        cond = torch.randn(2, 1, 128, 128, device=device)
        out = model(x, t, cond)
        assert out.shape == (2, 1, 128, 128)

    def test_gradient_flow(self, device):
        model = AttentionUNet(
            in_channels=2, out_channels=1, base_channels=32, channel_mults=(1, 2)
        ).to(device)
        x = torch.randn(1, 1, 64, 64, device=device)
        t = torch.rand(1, device=device)
        cond = torch.randn(1, 1, 64, 64, device=device)
        out = model(x, t, cond)
        loss = out.sum()
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert all(g > 0 for g in grad_norms[:5])  # first few layers should have gradients


class TestDiT:
    def test_forward_shape(self, device):
        model = DiT(
            img_size=64,
            patch_size=8,
            in_channels=2,
            out_channels=1,
            hidden_dim=64,
            depth=2,
            num_heads=4,
        ).to(device)
        x = torch.randn(2, 1, 64, 64, device=device)
        t = torch.rand(2, device=device)
        cond = torch.randn(2, 1, 64, 64, device=device)
        out = model(x, t, cond)
        assert out.shape == (2, 1, 64, 64)

    def test_adaln_zero_init(self, device):
        """DiT blocks start as identity due to zero-init alpha gates."""
        model = DiT(
            img_size=64,
            patch_size=8,
            in_channels=2,
            out_channels=1,
            hidden_dim=64,
            depth=2,
            num_heads=4,
        ).to(device)
        # Final layer linear should be zero-initialized
        assert model.final_layer.linear.weight.abs().max().item() == 0.0


class TestDDPM:
    def test_q_sample_shape(self, device):
        schedule = DDPMSchedule(T=100, device=device)
        x_0 = torch.randn(2, 1, 64, 64, device=device)
        t = torch.randint(0, 100, (2,), device=device)
        x_t, noise = schedule.q_sample(x_0, t)
        assert x_t.shape == x_0.shape
        assert noise.shape == x_0.shape

    def test_noise_increases_with_t(self, device):
        schedule = DDPMSchedule(T=1000, device=device)
        x_0 = torch.ones(1, 1, 32, 32, device=device)
        t_early = torch.tensor([10], device=device)
        t_late = torch.tensor([900], device=device)
        x_early, _ = schedule.q_sample(x_0, t_early)
        x_late, _ = schedule.q_sample(x_0, t_late)
        # Later timesteps should have more deviation from original
        assert (x_late - x_0).abs().mean() > (x_early - x_0).abs().mean()


class TestSampling:
    def test_euler_produces_output(self, device):
        from downscaling.sampling import euler_sample

        model = AttentionUNet(
            in_channels=2, out_channels=1, base_channels=16, channel_mults=(1,)
        ).to(device)
        model.eval()
        cond = torch.randn(1, 1, 32, 32, device=device)
        out = euler_sample(model, cond, shape=(1, 1, 32, 32), steps=3)
        assert out.shape == (1, 1, 32, 32)
        assert not torch.isnan(out).any()

    def test_midpoint_produces_output(self, device):
        from downscaling.sampling import midpoint_sample

        model = AttentionUNet(
            in_channels=2, out_channels=1, base_channels=16, channel_mults=(1,)
        ).to(device)
        model.eval()
        cond = torch.randn(1, 1, 32, 32, device=device)
        out = midpoint_sample(model, cond, shape=(1, 1, 32, 32), steps=3)
        assert out.shape == (1, 1, 32, 32)
        assert not torch.isnan(out).any()

    def test_ddim_produces_output(self, device):
        model = AttentionUNet(
            in_channels=2, out_channels=1, base_channels=16, channel_mults=(1,)
        ).to(device)
        model.eval()
        schedule = DDPMSchedule(T=100, device=device)
        cond = torch.randn(1, 1, 32, 32, device=device)
        out = ddim_sample(model, cond, shape=(1, 1, 32, 32), schedule=schedule, steps=5)
        assert out.shape == (1, 1, 32, 32)
        assert not torch.isnan(out).any()
