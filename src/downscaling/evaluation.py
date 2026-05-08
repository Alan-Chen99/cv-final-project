"""Unified evaluation pipeline for all downscaling methods.

Provides a single interface to evaluate any method (flow matching, DDPM,
deterministic baselines) with consistent metrics and constraint application.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from downscaling.constraints import apply_addcl, apply_smcl
from downscaling.data import NormStats, denormalize, load_era5_tcw4, normalize
from downscaling.metrics import crps_energy, crps_paper, ensemble_spread, mae, mass_violation
from downscaling.sampling import euler_sample, midpoint_sample


@dataclass
class EvalResult:
    """Evaluation results for a single method."""

    crps: float
    crps_paper: float
    mae: float
    rmse: float
    mass_violation: float
    spread: float
    n_samples: int
    n_ensemble: int
    method: str
    constraint: str


def evaluate_flow_model(
    model: torch.nn.Module,
    stats: NormStats,
    basedir: str,
    split: str = "test",
    n_ensemble: int = 10,
    ode_steps: int = 10,
    sampler: str = "euler",
    constraint: str = "none",
    batch_size: int = 32,
    max_samples: int | None = None,
    device: str | torch.device = "cuda",
    method_name: str = "flow_matching",
) -> EvalResult:
    """Evaluate a flow matching model on the test set.

    Args:
        model: Trained velocity field model (already on device, in eval mode)
        stats: Normalization statistics
        basedir: Data root directory
        split: Data split to evaluate
        n_ensemble: Number of ensemble members to generate
        ode_steps: Number of ODE solver steps
        sampler: ODE solver ("euler" or "midpoint")
        constraint: Post-hoc constraint ("none", "addcl", "smcl")
        batch_size: Evaluation batch size
        max_samples: Limit evaluation to first N samples
        device: Torch device
        method_name: Name for the result record

    Returns:
        EvalResult with all metrics.
    """
    lr_up, _residual, hr, lr_orig = load_era5_tcw4(basedir, split)

    n_samples = lr_up.shape[0]
    if max_samples:
        n_samples = min(n_samples, max_samples)

    sample_fn = midpoint_sample if sampler == "midpoint" else euler_sample

    all_crps = []
    all_crps_paper = []
    all_mae = []
    all_mse = []
    all_mass_viol = []
    all_spread = []

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_lr_norm = normalize(lr_up[start_idx:end_idx], stats.lr_mean, stats.lr_std).to(device)
        batch_hr = hr[start_idx:end_idx]
        batch_lr_up = lr_up[start_idx:end_idx]
        batch_lr_orig = lr_orig[start_idx:end_idx]
        bs = batch_lr_norm.shape[0]

        ensemble_preds = []
        for _e in range(n_ensemble):
            with torch.no_grad():
                sampled_res_norm = sample_fn(
                    model,
                    batch_lr_norm,
                    shape=(bs, 1, 128, 128),
                    steps=ode_steps,
                )
                sampled_res = denormalize(sampled_res_norm.cpu(), stats.res_mean, stats.res_std)
                pred_hr = batch_lr_up + sampled_res

                if constraint == "addcl":
                    pred_hr = apply_addcl(pred_hr, batch_lr_orig)
                elif constraint == "smcl":
                    pred_hr = apply_smcl(pred_hr, batch_lr_orig)

                ensemble_preds.append(pred_hr.numpy())

        ensemble_arr = np.stack(ensemble_preds, axis=1)  # (B, M, 1, H, W)

        for i in range(bs):
            gt = batch_hr[i, 0, ...].numpy()
            ens = ensemble_arr[i, :, 0, ...]  # (M, H, W)
            ens_mean = ens.mean(axis=0)

            all_crps.append(crps_energy(gt, ens))
            all_crps_paper.append(crps_paper(gt, ens))
            all_mae.append(mae(ens_mean, gt))
            all_mse.append(float(np.mean((ens_mean - gt) ** 2)))
            all_spread.append(ensemble_spread(ens))

            pred_t = torch.from_numpy(ens_mean).unsqueeze(0).unsqueeze(0)
            lr_i = batch_lr_orig[i : i + 1]
            all_mass_viol.append(mass_violation(pred_t, lr_i))

    return EvalResult(
        crps=float(np.mean(all_crps)),
        crps_paper=float(np.mean(all_crps_paper)),
        mae=float(np.mean(all_mae)),
        rmse=float(np.sqrt(np.mean(all_mse))),
        mass_violation=float(np.mean(all_mass_viol)),
        spread=float(np.mean(all_spread)),
        n_samples=n_samples,
        n_ensemble=n_ensemble,
        method=method_name,
        constraint=constraint,
    )


def evaluate_deterministic(
    predict_fn,
    basedir: str,
    split: str = "test",
    constraint: str = "none",
    max_samples: int | None = None,
    method_name: str = "deterministic",
) -> EvalResult:
    """Evaluate a deterministic (non-ensemble) method.

    For deterministic methods, CRPS = MAE since there is no spread term.

    Args:
        predict_fn: Callable(lr_up, lr_orig) -> pred_hr, all as tensors
        basedir: Data root directory
        split: Data split
        constraint: Post-hoc constraint
        max_samples: Limit samples
        method_name: Name for the result record

    Returns:
        EvalResult with all metrics.
    """
    lr_up, _residual, hr, lr_orig = load_era5_tcw4(basedir, split)

    n_samples = lr_up.shape[0]
    if max_samples:
        n_samples = min(n_samples, max_samples)

    all_mae = []
    all_mse = []
    all_mass_viol = []

    for i in range(n_samples):
        lr_i = lr_up[i : i + 1]
        lr_orig_i = lr_orig[i : i + 1]

        with torch.no_grad():
            pred_hr = predict_fn(lr_i, lr_orig_i)

        if constraint == "addcl":
            pred_hr = apply_addcl(pred_hr, lr_orig_i)
        elif constraint == "smcl":
            pred_hr = apply_smcl(pred_hr, lr_orig_i)

        gt = hr[i, 0, ...].numpy()
        pred = pred_hr[0, 0, ...].numpy()

        all_mae.append(mae(pred, gt))
        all_mse.append(float(np.mean((pred - gt) ** 2)))
        all_mass_viol.append(mass_violation(pred_hr, lr_orig_i))

    mean_mae = float(np.mean(all_mae))
    return EvalResult(
        crps=mean_mae,  # CRPS = MAE for deterministic
        crps_paper=mean_mae,
        mae=mean_mae,
        rmse=float(np.sqrt(np.mean(all_mse))),
        mass_violation=float(np.mean(all_mass_viol)),
        spread=0.0,
        n_samples=n_samples,
        n_ensemble=1,
        method=method_name,
        constraint=constraint,
    )


def bicubic_predict(lr_up: torch.Tensor, lr_orig: torch.Tensor) -> torch.Tensor:
    """Bicubic interpolation baseline (already upsampled via bilinear in data loading)."""
    return F.interpolate(lr_orig, size=(128, 128), mode="bicubic", align_corners=False)
