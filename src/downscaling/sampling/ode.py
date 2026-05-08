"""ODE solvers for flow matching sampling.

Flow matching transports from noise (t=0) to data (t=1) via learned velocity fields.
These solvers integrate the ODE dx/dt = v(x, t) from t=0 to t=1.
"""

import torch
import torch.nn as nn


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    condition: torch.Tensor,
    shape: tuple[int, ...],
    steps: int = 25,
) -> torch.Tensor:
    """First-order Euler ODE integration from noise (t=0) to data (t=1).

    Args:
        model: Velocity prediction network. Called as model(x, t, condition).
        condition: Conditioning input (e.g., upsampled LR field).
        shape: Output tensor shape (B, C, H, W).
        steps: Number of integration steps (= number of function evaluations).

    Returns:
        Sampled tensor at t=1.
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
    model: nn.Module,
    condition: torch.Tensor,
    shape: tuple[int, ...],
    steps: int = 25,
) -> torch.Tensor:
    """Midpoint (2nd-order Runge-Kutta) ODE integration.

    Uses 2 function evaluations per step (2*steps NFE total).
    More accurate than Euler at the same number of steps.

    Args:
        model: Velocity prediction network. Called as model(x, t, condition).
        condition: Conditioning input.
        shape: Output tensor shape (B, C, H, W).
        steps: Number of integration steps (NFE = 2 * steps).

    Returns:
        Sampled tensor at t=1.
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
