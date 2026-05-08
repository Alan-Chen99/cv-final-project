"""ODE/SDE solvers for generative sampling.

Samplers for flow matching (ODE) and DDPM (DDIM) models.
"""

import torch


@torch.no_grad()
def euler_sample(
    model: torch.nn.Module,
    condition: torch.Tensor,
    shape: tuple[int, ...],
    steps: int = 25,
) -> torch.Tensor:
    """Euler ODE integration from noise (t=0) to data (t=1).

    Args:
        model: Velocity field v(x, t, cond) for flow matching
        condition: Conditioning input (B, C, H, W)
        shape: Output shape (B, C, H, W)
        steps: Number of Euler steps (= NFE)

    Returns:
        Sampled output at t=1.
    """
    device = condition.device
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), i * dt, device=device)
        v = model(x, t, condition)
        x = x + v * dt
    return x


@torch.no_grad()
def midpoint_sample(
    model: torch.nn.Module,
    condition: torch.Tensor,
    shape: tuple[int, ...],
    steps: int = 25,
) -> torch.Tensor:
    """Midpoint (RK2) ODE integration from noise (t=0) to data (t=1).

    Uses 2 function evaluations per step (2*steps NFE total).

    Args:
        model: Velocity field v(x, t, cond) for flow matching
        condition: Conditioning input (B, C, H, W)
        shape: Output shape (B, C, H, W)
        steps: Number of midpoint steps (NFE = 2*steps)

    Returns:
        Sampled output at t=1.
    """
    device = condition.device
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), i * dt, device=device)
        t_mid = torch.full((shape[0],), (i + 0.5) * dt, device=device)
        v1 = model(x, t, condition)
        x_mid = x + v1 * (0.5 * dt)
        v2 = model(x_mid, t_mid, condition)
        x = x + v2 * dt
    return x


def sample_timesteps_uniform(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample timesteps uniformly from [0, 1)."""
    return torch.rand(batch_size, device=device)


def sample_timesteps_logit_normal(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Logit-normal timestep distribution (from SD3).

    Concentrates sampling on intermediate timesteps where the velocity
    field has the most structure.
    """
    u = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(u)
