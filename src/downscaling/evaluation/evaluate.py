"""Evaluation functions for downscaling models.

Supports flow matching models and arbitrary ensemble predictions.
Computes CRPS, MAE, RMSE, and mass violation metrics.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from downscaling.constraints.layers import apply_addcl, apply_smcl
from downscaling.metrics.crps import crps_energy
from downscaling.sampling.ode import euler_sample, midpoint_sample


@dataclass
class EvalMetrics:
    crps: float
    mae: float
    rmse: float
    mass_violation: float

    def __str__(self) -> str:
        return (
            f"CRPS: {self.crps:.6f}, MAE: {self.mae:.6f}, "
            f"RMSE: {self.rmse:.6f}, Mass viol: {self.mass_violation:.6f}"
        )


def evaluate_ensemble(
    ensemble_preds: np.ndarray,
    ground_truth: np.ndarray,
    lr_orig: torch.Tensor | None = None,
    upsampling_factor: int = 4,
) -> EvalMetrics:
    """Evaluate an ensemble of predictions against ground truth.

    Args:
        ensemble_preds: Predictions, shape (M, H, W) where M = ensemble size.
        ground_truth: Ground truth, shape (H, W).
        lr_orig: Optional LR field for mass violation, shape (1, 1, H/f, W/f).
        upsampling_factor: SR factor for mass violation computation.

    Returns:
        EvalMetrics with CRPS, MAE, RMSE, and mass violation.
    """
    ens_mean = ensemble_preds.mean(axis=0)

    crps = crps_energy(ground_truth, ensemble_preds)
    mae = float(np.mean(np.abs(ground_truth - ens_mean)))
    rmse = float(np.sqrt(np.mean((ground_truth - ens_mean) ** 2)))

    mass_viol = 0.0
    if lr_orig is not None:
        pool = nn.AvgPool2d(kernel_size=upsampling_factor)
        pred_t = torch.from_numpy(ens_mean).unsqueeze(0).unsqueeze(0)
        pooled = pool(pred_t).squeeze()
        mass_viol = float(torch.mean(torch.abs(pooled - lr_orig.squeeze())).item())

    return EvalMetrics(crps=crps, mae=mae, rmse=rmse, mass_violation=mass_viol)


def evaluate_flow_model(
    model: nn.Module,
    lr_up_norm: torch.Tensor,
    hr: torch.Tensor,
    lr_up: torch.Tensor,
    lr_orig: torch.Tensor,
    norm_stats: dict[str, float],
    n_ensemble: int = 10,
    ode_steps: int = 10,
    constraint: str = "addcl",
    sampler: str = "midpoint",
    batch_size: int = 32,
    max_samples: int | None = None,
    upsampling_factor: int = 4,
) -> EvalMetrics:
    """Evaluate a flow matching model on a dataset split.

    Args:
        model: Trained velocity network (in eval mode).
        lr_up_norm: Normalized upsampled LR, shape (N, 1, 128, 128).
        hr: Ground truth HR, shape (N, 1, 128, 128).
        lr_up: Unnormalized upsampled LR, shape (N, 1, 128, 128).
        lr_orig: Original LR, shape (N, 1, 32, 32).
        norm_stats: Dict with res_mean, res_std, lr_mean, lr_std.
        n_ensemble: Number of ensemble members.
        ode_steps: ODE integration steps.
        constraint: 'addcl', 'smcl', or 'none'.
        sampler: 'euler' or 'midpoint'.
        batch_size: Evaluation batch size.
        max_samples: Limit number of samples evaluated.
        upsampling_factor: SR factor for mass violation computation.

    Returns:
        EvalMetrics averaged over all samples.
    """
    device = next(model.parameters()).device
    sample_fn = midpoint_sample if sampler == "midpoint" else euler_sample
    n_samples = min(lr_up.shape[0], max_samples) if max_samples is not None else lr_up.shape[0]

    all_crps: list[float] = []
    all_mae: list[float] = []
    all_rmse_sq: list[float] = []
    all_mass_viol: list[float] = []
    pool = nn.AvgPool2d(kernel_size=upsampling_factor)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_lr = lr_up_norm[start_idx:end_idx].to(device)
        batch_hr = hr[start_idx:end_idx]
        batch_lr_up = lr_up[start_idx:end_idx]
        batch_lr_orig = lr_orig[start_idx:end_idx]
        bs = batch_lr.shape[0]

        ensemble_preds = []
        for _ in range(n_ensemble):
            with torch.no_grad():
                hr_h, hr_w = batch_hr.shape[2], batch_hr.shape[3]
                sampled = sample_fn(model, batch_lr, shape=(bs, 1, hr_h, hr_w), steps=ode_steps)
                sampled_res = sampled.cpu() * norm_stats["res_std"] + norm_stats["res_mean"]
                pred_hr = batch_lr_up + sampled_res

                if constraint == "addcl":
                    pred_hr = apply_addcl(pred_hr, batch_lr_orig, upsampling_factor)
                elif constraint == "smcl":
                    pred_hr = apply_smcl(pred_hr, batch_lr_orig, upsampling_factor)

                ensemble_preds.append(pred_hr.numpy())

        ensemble_np = np.stack(ensemble_preds, axis=1)  # (bs, M, 1, 128, 128)

        for i in range(bs):
            gt = batch_hr[i, 0, ...].numpy()
            ens = ensemble_np[i, :, 0, ...]
            ens_mean = ens.mean(axis=0)

            all_crps.append(crps_energy(gt, ens))
            all_mae.append(float(np.mean(np.abs(gt - ens_mean))))
            all_rmse_sq.append(float(np.mean((gt - ens_mean) ** 2)))

            pred_t = torch.from_numpy(ens_mean).unsqueeze(0).unsqueeze(0)
            pooled = pool(pred_t).squeeze()
            lr_i = batch_lr_orig[i, 0, ...]
            all_mass_viol.append(float(torch.mean(torch.abs(pooled - lr_i)).item()))

        if (start_idx // batch_size) % 20 == 0:
            print(f"  Evaluated {end_idx}/{n_samples}...")

    return EvalMetrics(
        crps=float(np.mean(all_crps)),
        mae=float(np.mean(all_mae)),
        rmse=float(np.sqrt(np.mean(all_rmse_sq))),
        mass_violation=float(np.mean(all_mass_viol)),
    )
