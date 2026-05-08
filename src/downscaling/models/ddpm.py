"""DDPM (VP-SDE) noise schedule and DDIM sampling.

Implements the linear variance-preserving noise schedule from Ho et al. (2020)
and the deterministic DDIM sampler from Song et al. (2021).
"""

import torch


class DDPMSchedule:
    """Linear VP-SDE noise schedule.

    Forward process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

    Args:
        T: Number of diffusion timesteps
        beta_start: Starting noise level
        beta_end: Ending noise level
        device: Torch device
    """

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = "cpu",
    ):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bar = alpha_bar.to(device)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar).to(device)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar).to(device)

    def to(self, device: str | torch.device) -> DDPMSchedule:
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        return self

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward process: add noise at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return sqrt_ab * x_0 + sqrt_1mab * noise, noise


@torch.no_grad()
def ddim_sample(
    model: torch.nn.Module,
    condition: torch.Tensor,
    shape: tuple[int, ...],
    schedule: DDPMSchedule,
    steps: int = 20,
    eta: float = 0.0,
) -> torch.Tensor:
    """DDIM sampling. eta=0 is deterministic, eta=1 is DDPM stochastic.

    Args:
        model: Noise prediction model eps(x, t, cond)
        condition: Conditioning input (B, C, H, W)
        shape: Output shape (B, C, H, W)
        schedule: DDPMSchedule instance
        steps: Number of DDIM steps
        eta: Stochasticity parameter (0=deterministic)

    Returns:
        Sampled output at t=0.
    """
    device = condition.device
    T = schedule.T

    step_indices = torch.linspace(T - 1, 0, steps + 1, dtype=torch.long)
    timesteps = step_indices[:-1]
    timesteps_prev = step_indices[1:]

    x = torch.randn(shape, device=device)

    for i in range(steps):
        t_idx = int(timesteps[i].item())
        t_prev_idx = int(timesteps_prev[i].item())

        t_batch = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        t_continuous = t_batch.float() / T
        eps_pred = model(x, t_continuous, condition)

        alpha_bar_t = schedule.alpha_bar[t_idx]
        alpha_bar_prev = (
            schedule.alpha_bar[max(t_prev_idx, 0)]
            if t_prev_idx >= 0
            else torch.tensor(1.0, device=device)
        )

        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        sigma_t = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t))
            * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        )
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_pred

        noise = torch.randn_like(x) if (eta > 0 and i < steps - 1) else torch.zeros_like(x)
        x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma_t * noise

    return x
