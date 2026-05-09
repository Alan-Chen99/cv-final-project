"""Proper CRPS evaluation for climate downscaling ensembles.

Computes CRPS using the standard formula:
  CRPS = (1/M) Σ|fm - y| - 1/(2M²) ΣΣ|fm - fn|

The baseline code has a bug in crps_ensemble() — first loop uses
fc.shape[-1]**2 (=128²) instead of fc.shape[0]**2 (=M²).
This script provides a correct implementation.
"""
import argparse

import numpy as np
import torch


def crps_ensemble(obs: np.ndarray, forecasts: np.ndarray) -> float:
    """Compute CRPS for a single sample.

    Args:
        obs: (H, W) ground truth
        forecasts: (M, H, W) ensemble forecasts

    Returns:
        Scalar CRPS value (lower is better)
    """
    M = forecasts.shape[0]
    # Term 1: mean absolute error of ensemble members
    mae_term = np.mean(np.abs(forecasts - obs[None, ...]), axis=0)  # (H, W)

    # Term 2: mean pairwise absolute difference
    spread_term = np.zeros_like(obs)
    for i in range(M):
        for j in range(M):
            spread_term += np.abs(forecasts[i] - forecasts[j])
    spread_term /= (2.0 * M * M)

    crps = mae_term - spread_term  # (H, W)
    return float(np.mean(crps))


def crps_ensemble_fast(obs: np.ndarray, forecasts: np.ndarray) -> float:
    """Fast CRPS using sorted ensemble members.

    Uses the identity: CRPS = (1/M)Σ|fm-y| - 1/(2M²)ΣΣ|fm-fn|
    where the spread term can be computed efficiently from sorted forecasts.
    """
    M = forecasts.shape[0]
    # Term 1
    mae_term = np.mean(np.abs(forecasts - obs[None, ...]), axis=0)

    # Term 2: efficient computation via sorted forecasts
    fc_sorted = np.sort(forecasts, axis=0)  # (M, H, W)
    spread_term = np.zeros_like(obs)
    for i in range(M):
        # Weight for sorted member i: 2i - M + 1 (Gneiting & Raftery 2007)
        weight = (2.0 * (i + 1) - M - 1.0) / (M * M)
        spread_term += weight * fc_sorted[i]

    crps = mae_term - spread_term
    return float(np.mean(crps))


def evaluate_crps(pred_path: str, target_path: str, input_path: str | None = None) -> dict:
    """Evaluate ensemble predictions.

    Args:
        pred_path: Path to ensemble predictions (N, M, 1, 1, H, W) or deterministic (N, 1, 1, H, W)
        target_path: Path to ground truth (N, 1, 1, H, W)
        input_path: Optional path to LR input for mass violation check

    Returns:
        Dictionary of metrics
    """
    pred = torch.load(pred_path, weights_only=False)
    target = torch.load(target_path, weights_only=False)

    is_ensemble = pred.dim() == 6  # (N, M, 1, 1, H, W)

    if is_ensemble:
        N, M = pred.shape[0], pred.shape[1]
        ensemble_np: np.ndarray = pred[:, :, 0, 0, :, :].numpy()  # (N, M, H, W)
        mean_pred = pred.mean(dim=1)  # (N, 1, 1, H, W)
    else:
        N = pred.shape[0]
        M = 1
        ensemble_np = np.empty(0)
        mean_pred = pred

    target_np = target[:, 0, 0, :, :].numpy()  # (N, H, W)
    mean_np = mean_pred[:, 0, 0, :, :].numpy()  # (N, H, W)

    # Compute metrics
    mse = float(np.mean((mean_np - target_np) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(mean_np - target_np)))

    # CRPS
    crps_values = []
    if is_ensemble:
        for i in range(N):
            c = crps_ensemble_fast(target_np[i], ensemble_np[i])
            crps_values.append(c)
    else:
        # For deterministic: CRPS = MAE
        for i in range(N):
            crps_values.append(float(np.mean(np.abs(mean_np[i] - target_np[i]))))
    crps = float(np.mean(crps_values))

    # Ensemble spread (std of ensemble members)
    spread = float(np.mean(np.std(ensemble_np, axis=1))) if is_ensemble else 0.0

    # Mass violation
    mass_viol = 0.0
    if input_path is not None:
        from skimage import transform as sktransform
        lr = torch.load(input_path, weights_only=False)
        lr_np = lr[:, 0, 0, :, :].numpy()
        for i in range(N):
            ds = sktransform.downscale_local_mean(mean_np[i], (4, 4))
            mass_viol += float(np.mean(np.abs(ds - lr_np[i])))
        mass_viol /= N

    results = {
        'CRPS': crps,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Ensemble_Size': M,
        'Ensemble_Spread': spread,
        'Mass_Violation': mass_viol,
        'N_Samples': N,
    }
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='Path to predictions')
    parser.add_argument('--target', required=True, help='Path to targets')
    parser.add_argument('--input', default=None, help='Path to LR inputs (for mass violation)')
    args = parser.parse_args()

    results = evaluate_crps(args.pred, args.target, args.input)
    for k, v in results.items():
        if isinstance(v, float):
            print(f'{k}: {v:.6f}')
        else:
            print(f'{k}: {v}')
