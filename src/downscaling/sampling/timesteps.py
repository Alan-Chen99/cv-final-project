"""Timestep sampling strategies for flow matching training."""

import torch


def sample_timesteps_uniform(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample timesteps uniformly from [0, 1]."""
    return torch.rand(batch_size, device=device)


def sample_timesteps_logit_normal(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Logit-normal timestep distribution (from Stable Diffusion 3).

    Concentrates sampling on intermediate timesteps where the velocity
    field is most informative. Better gradient signal than uniform sampling.
    """
    u = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(u)
