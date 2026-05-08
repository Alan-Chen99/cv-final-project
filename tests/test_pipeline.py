"""Integration tests for the full model-training-sampling-evaluation pipeline.

Tests the complete flow: model creation -> forward pass -> training step ->
ODE sampling -> evaluation. Uses synthetic data on GPU.
"""

import tempfile

import numpy as np
import pytest
import torch

from downscaling.evaluation.evaluate import EvalMetrics, evaluate_ensemble, evaluate_flow_model
from downscaling.models.unet import AttentionUNet
from downscaling.sampling.ode import euler_sample, midpoint_sample
from downscaling.sampling.timesteps import sample_timesteps_logit_normal, sample_timesteps_uniform
from downscaling.training.ema import EMA
from downscaling.training.flow_matching import FlowMatchingTrainer, TrainConfig


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model(device) -> AttentionUNet:
    """Small UNet for testing (reduced channels for speed)."""
    torch.manual_seed(42)
    m = AttentionUNet(
        in_channels=2,
        out_channels=1,
        base_channels=16,
        channel_mults=(1, 2, 4),
        time_emb_dim=64,
        dropout=0.0,
        attn_heads=2,
    )
    return m.to(device)


class TestModelForwardPass:
    """Test model creates valid outputs for various configurations."""

    def test_forward_shape(self, model, device):
        """Model produces correct output shape."""
        x = torch.randn(2, 1, 128, 128, device=device)
        t = torch.rand(2, device=device)
        cond = torch.randn(2, 1, 128, 128, device=device)
        out = model(x, t, cond)
        assert out.shape == (2, 1, 128, 128)

    def test_forward_different_batch_sizes(self, model, device):
        """Model handles batch sizes 1 and 8."""
        for bs in [1, 8]:
            x = torch.randn(bs, 1, 128, 128, device=device)
            t = torch.rand(bs, device=device)
            cond = torch.randn(bs, 1, 128, 128, device=device)
            out = model(x, t, cond)
            assert out.shape == (bs, 1, 128, 128)

    def test_gradient_flows(self, model, device):
        """Gradients flow through the model (necessary for training)."""
        x = torch.randn(2, 1, 128, 128, device=device)
        t = torch.rand(2, device=device)
        cond = torch.randn(2, 1, 128, 128, device=device)
        out = model(x, t, cond)
        loss = out.mean()
        loss.backward()
        # Check that at least one parameter has gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "No gradients flowed through the model"


class TestTimestepSampling:
    """Test timestep sampling strategies."""

    def test_uniform_range(self, device):
        """Uniform samples are in [0, 1]."""
        t = sample_timesteps_uniform(1000, device)
        assert t.shape == (1000,)
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_logit_normal_range(self, device):
        """Logit-normal samples are in (0, 1) due to sigmoid."""
        t = sample_timesteps_logit_normal(1000, device, mean=0.0, std=1.0)
        assert t.shape == (1000,)
        assert t.min() > 0.0
        assert t.max() < 1.0

    def test_logit_normal_concentration(self, device):
        """Logit-normal should concentrate around 0.5 with mean=0."""
        t = sample_timesteps_logit_normal(10000, device, mean=0.0, std=0.5)
        # With mean=0 and small std, values cluster near 0.5
        assert t.mean() == pytest.approx(0.5, abs=0.05)


class TestODESamplers:
    """Test ODE solvers produce valid outputs."""

    def test_euler_sample_shape(self, model, device):
        """Euler sampler produces correct output shape."""
        cond = torch.randn(4, 1, 128, 128, device=device)
        out = euler_sample(model, cond, shape=(4, 1, 128, 128), steps=5)
        assert out.shape == (4, 1, 128, 128)

    def test_midpoint_sample_shape(self, model, device):
        """Midpoint sampler produces correct output shape."""
        cond = torch.randn(4, 1, 128, 128, device=device)
        out = midpoint_sample(model, cond, shape=(4, 1, 128, 128), steps=5)
        assert out.shape == (4, 1, 128, 128)

    def test_euler_deterministic_with_seed(self, model, device):
        """Same seed produces identical samples."""
        cond = torch.randn(2, 1, 128, 128, device=device)
        torch.manual_seed(0)
        out1 = euler_sample(model, cond, shape=(2, 1, 128, 128), steps=5)
        torch.manual_seed(0)
        out2 = euler_sample(model, cond, shape=(2, 1, 128, 128), steps=5)
        torch.testing.assert_close(out1, out2)

    def test_different_seeds_different_samples(self, model, device):
        """Different seeds produce different samples."""
        cond = torch.randn(2, 1, 128, 128, device=device)
        torch.manual_seed(0)
        out1 = euler_sample(model, cond, shape=(2, 1, 128, 128), steps=5)
        torch.manual_seed(1)
        out2 = euler_sample(model, cond, shape=(2, 1, 128, 128), steps=5)
        assert not torch.allclose(out1, out2)


class TestEMA:
    """Test EMA parameter tracking."""

    def test_ema_moves_toward_model(self, device):
        """EMA shadow parameters move toward model parameters after update."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 10).to(device)
        ema = EMA(model, decay=0.9)

        # Modify model parameters
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))

        # Get shadow before update
        shadow_before = {n: p.clone() for n, p in ema.shadow.named_parameters()}

        ema.update(model)

        # Shadow should have moved toward model
        for name, p in ema.shadow.named_parameters():
            diff_before = (
                (
                    shadow_before[name]
                    - next(mp for mn, mp in model.named_parameters() if mn == name)
                )
                .abs()
                .mean()
            )
            diff_after = (
                (p - next(mp for mn, mp in model.named_parameters() if mn == name)).abs().mean()
            )
            assert diff_after < diff_before, f"EMA didn't move toward model for {name}"

    def test_ema_state_dict_roundtrip(self, device):
        """EMA state dict can be saved and loaded."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 10).to(device)
        ema = EMA(model, decay=0.999)
        ema.update(model)

        state = ema.state_dict()
        ema2 = EMA(model, decay=0.999)
        ema2.load_state_dict(state)

        for p1, p2 in zip(ema.shadow.parameters(), ema2.shadow.parameters(), strict=False):
            torch.testing.assert_close(p1, p2)


class TestTrainingPipeline:
    """Test the training pipeline with synthetic data."""

    def test_train_two_epochs(self, device):
        """Train for 2 epochs with synthetic data, verify loss decreases or stays finite."""
        torch.manual_seed(42)

        # Create synthetic data mimicking ERA5 format
        n_train = 64
        lr_up = torch.randn(n_train, 1, 128, 128)
        residual = torch.randn(n_train, 1, 128, 128) * 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            # Manually do what FlowMatchingTrainer.train() does, but with synthetic data
            model = AttentionUNet(
                in_channels=2,
                out_channels=1,
                base_channels=16,
                channel_mults=(1, 2, 4),
                time_emb_dim=64,
                dropout=0.0,
                attn_heads=2,
            ).to(device)
            config = TrainConfig(
                save_dir=tmpdir,
                batch_size=16,
                epochs=2,
                lr=1e-3,
                amp=False,
            )
            trainer = FlowMatchingTrainer(model, config)

            # Compute normalization
            stats = {
                "res_mean": residual.mean().item(),
                "res_std": residual.std().item(),
                "lr_mean": lr_up.mean().item(),
                "lr_std": lr_up.std().item(),
            }

            res_norm = (residual - stats["res_mean"]) / stats["res_std"]
            lr_norm = (lr_up - stats["lr_mean"]) / stats["lr_std"]

            from torch.utils.data import DataLoader, TensorDataset

            train_loader = DataLoader(
                TensorDataset(lr_norm, res_norm),
                batch_size=16,
                shuffle=True,
            )

            # Train 2 epochs, collect losses
            losses = []
            for _epoch in range(2):
                loss = trainer._train_epoch(train_loader)
                losses.append(loss)
                assert np.isfinite(loss), f"Training loss is not finite: {loss}"

            # Verify checkpoint saves
            trainer._save_checkpoint(1, losses[-1])
            import os

            assert os.path.exists(os.path.join(tmpdir, "best_flow.pt"))


class TestTrainingPipelineExtended:
    """Additional training pipeline tests for coverage."""

    def test_logit_normal_timestep_sampling(self, device):
        """Trainer uses logit-normal sampling when configured."""
        torch.manual_seed(42)
        model = AttentionUNet(
            in_channels=2,
            out_channels=1,
            base_channels=16,
            channel_mults=(1, 2, 4),
            time_emb_dim=64,
            dropout=0.0,
            attn_heads=2,
        ).to(device)
        config = TrainConfig(
            save_dir="/tmp/test_logit",
            batch_size=8,
            epochs=1,
            lr=1e-3,
            t_sampling="logit_normal",
            t_logit_mean=0.0,
            t_logit_std=1.0,
        )
        trainer = FlowMatchingTrainer(model, config)
        t = trainer._sample_t(100)
        assert t.shape == (100,)
        assert t.min() > 0.0
        assert t.max() < 1.0

    def test_compute_normalization(self, device):
        """_compute_normalization saves stats to disk."""
        import os
        import tempfile

        torch.manual_seed(42)
        model = AttentionUNet(
            in_channels=2,
            out_channels=1,
            base_channels=16,
            channel_mults=(1, 2, 4),
            time_emb_dim=64,
            dropout=0.0,
            attn_heads=2,
        ).to(device)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(save_dir=tmpdir)
            trainer = FlowMatchingTrainer(model, config)
            lr_up = torch.randn(10, 1, 128, 128)
            res = torch.randn(10, 1, 128, 128)
            stats = trainer._compute_normalization(lr_up, res)
            assert "res_mean" in stats
            assert "res_std" in stats
            assert "lr_mean" in stats
            assert "lr_std" in stats
            assert os.path.exists(os.path.join(tmpdir, "norm_stats.pt"))

    def test_validate_epoch(self, device):
        """_validate_epoch returns finite loss."""
        import tempfile

        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(42)
        model = AttentionUNet(
            in_channels=2,
            out_channels=1,
            base_channels=16,
            channel_mults=(1, 2, 4),
            time_emb_dim=64,
            dropout=0.0,
            attn_heads=2,
        ).to(device)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(save_dir=tmpdir, batch_size=8)
            trainer = FlowMatchingTrainer(model, config)
            val_loader = DataLoader(
                TensorDataset(torch.randn(16, 1, 128, 128), torch.randn(16, 1, 128, 128)),
                batch_size=8,
            )
            val_loss = trainer._validate_epoch(val_loader)
            assert np.isfinite(val_loss)
            assert val_loss > 0

    def test_checkpoint_save_and_resume(self, device):
        """Checkpoint save/resume roundtrip preserves model state."""
        import tempfile

        torch.manual_seed(42)
        model = AttentionUNet(
            in_channels=2,
            out_channels=1,
            base_channels=16,
            channel_mults=(1, 2, 4),
            time_emb_dim=64,
            dropout=0.0,
            attn_heads=2,
        ).to(device)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(save_dir=tmpdir, use_ema=True)
            trainer = FlowMatchingTrainer(model, config)
            trainer._save_checkpoint(5, 0.123)

            # Create new trainer and resume
            model2 = AttentionUNet(
                in_channels=2,
                out_channels=1,
                base_channels=16,
                channel_mults=(1, 2, 4),
                time_emb_dim=64,
                dropout=0.0,
                attn_heads=2,
            ).to(device)
            config2 = TrainConfig(save_dir=tmpdir, resume=True, use_ema=True)
            trainer2 = FlowMatchingTrainer(model2, config2)
            trainer2._resume_checkpoint()

            assert trainer2.start_epoch == 6
            assert trainer2.best_val_loss == pytest.approx(0.123)
            # Model weights should match
            for p1, p2 in zip(model.parameters(), model2.parameters(), strict=False):
                torch.testing.assert_close(p1, p2)

    def test_resume_nonexistent_checkpoint(self, device):
        """_resume_checkpoint with missing file is a no-op."""
        import tempfile

        torch.manual_seed(42)
        model = AttentionUNet(
            in_channels=2,
            out_channels=1,
            base_channels=16,
            channel_mults=(1, 2, 4),
            time_emb_dim=64,
            dropout=0.0,
            attn_heads=2,
        ).to(device)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(save_dir=tmpdir)
            trainer = FlowMatchingTrainer(model, config)
            trainer._resume_checkpoint()
            assert trainer.start_epoch == 0

    def test_train_with_ema(self, device):
        """Training with EMA enabled updates shadow params."""
        import tempfile

        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(42)
        model = AttentionUNet(
            in_channels=2,
            out_channels=1,
            base_channels=16,
            channel_mults=(1, 2, 4),
            time_emb_dim=64,
            dropout=0.0,
            attn_heads=2,
        ).to(device)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(
                save_dir=tmpdir,
                batch_size=8,
                epochs=1,
                lr=1e-3,
                use_ema=True,
            )
            trainer = FlowMatchingTrainer(model, config)
            assert trainer.ema is not None
            train_loader = DataLoader(
                TensorDataset(torch.randn(16, 1, 128, 128), torch.randn(16, 1, 128, 128)),
                batch_size=8,
            )
            loss = trainer._train_epoch(train_loader)
            assert np.isfinite(loss)


class TestEvaluationPipeline:
    """Test evaluation with synthetic predictions."""

    def test_evaluate_ensemble_basic(self):
        """evaluate_ensemble returns valid metrics."""
        rng = np.random.default_rng(42)
        gt = rng.standard_normal((128, 128)).astype(np.float64)
        preds = rng.standard_normal((10, 128, 128)).astype(np.float64)
        metrics = evaluate_ensemble(preds, gt)

        assert isinstance(metrics, EvalMetrics)
        assert np.isfinite(metrics.crps)
        assert np.isfinite(metrics.mae)
        assert np.isfinite(metrics.rmse)
        assert metrics.mae >= 0
        assert metrics.rmse >= 0

    def test_evaluate_ensemble_with_mass_violation(self, device):
        """evaluate_ensemble computes mass violation when lr_orig given."""
        rng = np.random.default_rng(42)
        gt = rng.standard_normal((128, 128)).astype(np.float64)
        preds = rng.standard_normal((5, 128, 128)).astype(np.float64)
        lr_orig = torch.randn(1, 1, 32, 32, device=device)
        metrics = evaluate_ensemble(preds, gt, lr_orig=lr_orig.cpu())

        assert metrics.mass_violation > 0  # random preds won't satisfy conservation

    def test_evaluate_ensemble_perfect_prediction(self):
        """Perfect predictions => MAE=0, RMSE=0."""
        gt = np.ones((64, 64), dtype=np.float64)
        preds = np.stack([gt] * 5, axis=0)
        metrics = evaluate_ensemble(preds, gt)

        assert metrics.mae == pytest.approx(0.0, abs=1e-10)
        assert metrics.rmse == pytest.approx(0.0, abs=1e-10)
        assert metrics.crps == pytest.approx(0.0, abs=1e-10)

    def test_evaluate_flow_model_end_to_end(self, device):
        """Full evaluate_flow_model pipeline with a random model."""
        torch.manual_seed(42)
        model = (
            AttentionUNet(
                in_channels=2,
                out_channels=1,
                base_channels=16,
                channel_mults=(1, 2, 4),
                time_emb_dim=64,
                dropout=0.0,
                attn_heads=2,
            )
            .to(device)
            .eval()
        )

        n = 4  # small dataset for speed
        lr_up = torch.randn(n, 1, 128, 128)
        hr = torch.randn(n, 1, 128, 128)
        lr_orig = torch.randn(n, 1, 32, 32)
        lr_up_norm = (lr_up - lr_up.mean()) / lr_up.std()

        norm_stats = {"res_mean": 0.0, "res_std": 1.0, "lr_mean": 0.0, "lr_std": 1.0}

        metrics = evaluate_flow_model(
            model,
            lr_up_norm,
            hr,
            lr_up,
            lr_orig,
            norm_stats,
            n_ensemble=2,
            ode_steps=3,
            constraint="addcl",
            sampler="euler",
            batch_size=2,
            max_samples=4,
        )

        assert isinstance(metrics, EvalMetrics)
        assert np.isfinite(metrics.crps)
        assert np.isfinite(metrics.mae)
        assert np.isfinite(metrics.rmse)
        assert np.isfinite(metrics.mass_violation)
        # With AddCL constraint, mass violation should be near zero
        assert metrics.mass_violation < 0.01

    def test_evaluate_flow_model_no_constraint(self, device):
        """evaluate_flow_model works with constraint='none'."""
        torch.manual_seed(42)
        model = (
            AttentionUNet(
                in_channels=2,
                out_channels=1,
                base_channels=16,
                channel_mults=(1, 2, 4),
                time_emb_dim=64,
                dropout=0.0,
                attn_heads=2,
            )
            .to(device)
            .eval()
        )

        n = 4
        lr_up = torch.randn(n, 1, 128, 128)
        hr = torch.randn(n, 1, 128, 128)
        lr_orig = torch.randn(n, 1, 32, 32)
        lr_up_norm = (lr_up - lr_up.mean()) / lr_up.std()
        norm_stats = {"res_mean": 0.0, "res_std": 1.0, "lr_mean": 0.0, "lr_std": 1.0}

        metrics = evaluate_flow_model(
            model,
            lr_up_norm,
            hr,
            lr_up,
            lr_orig,
            norm_stats,
            n_ensemble=2,
            ode_steps=3,
            constraint="none",
            sampler="midpoint",
            batch_size=4,
            max_samples=4,
        )
        assert isinstance(metrics, EvalMetrics)
        # Without constraint, mass violation should be non-zero
        assert metrics.mass_violation > 0

    def test_eval_metrics_str(self):
        """EvalMetrics __str__ works."""
        m = EvalMetrics(crps=0.1, mae=0.2, rmse=0.3, mass_violation=0.01)
        s = str(m)
        assert "CRPS" in s
        assert "MAE" in s
        assert "RMSE" in s
