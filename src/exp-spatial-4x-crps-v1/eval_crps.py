"""Evaluate CRPS for trained models on the constrained-downscaling task.

Usage:
    python scripts/eval_crps.py --model_id <id> --model <cnn|gan> --constraints <none|softmax|...>

For deterministic models (CNN), CRPS = MAE.
For ensemble models (GAN), CRPS is computed from 10-member ensemble.
"""

import argparse
import sys
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'constrained-downscaling'))


def crps_ensemble(observation, forecasts):
    """CRPS for ensemble forecasts. From constrained-downscaling/training.py."""
    fc = forecasts.copy()
    fc.sort(axis=0)
    obs = observation
    fc_below = fc < obs[None, ...]
    crps = np.zeros_like(obs)
    for i in range(fc.shape[0]):
        below = fc_below[i, ...]
        weight = ((i + 1) ** 2 - i ** 2) / fc.shape[-1] ** 2
        crps[below] += weight * (obs[below] - fc[i, ...][below])
    for i in range(fc.shape[0] - 1, -1, -1):
        above = ~fc_below[i, ...]
        k = fc.shape[0] - 1 - i
        weight = ((k + 1) ** 2 - k ** 2) / fc.shape[0] ** 2
        crps[above] += weight * (fc[i, ...][above] - obs[above])
    return np.mean(crps)


def crps_deterministic(observation, prediction):
    """CRPS for deterministic forecast = MAE."""
    return np.mean(np.abs(prediction - observation))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--model", default="cnn")
    parser.add_argument("--dataset", default="era5_sr_data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--basedir", default="external/constrained-downscaling")
    args = parser.parse_args()

    basedir = args.basedir
    split = args.split

    # Load ground truth
    target = torch.load(
        f'{basedir}/data/{args.dataset}/{split}/target_{split}.pt', weights_only=False
    )

    is_gan = args.model == 'gan'

    if is_gan:
        pred_path = f'{basedir}/data/prediction/{args.dataset}_{args.model_id}_{split}_ensemble.pt'
        en_pred = torch.load(pred_path, weights_only=False)
        # en_pred shape: (N, num_ensemble, 1, 1, H, W)
        print(f"Ensemble prediction shape: {en_pred.shape}")
        det_pred = torch.mean(en_pred, dim=1)  # ensemble mean
    else:
        pred_path = f'{basedir}/data/prediction/{args.dataset}_{args.model_id}_{split}.pt'
        det_pred = torch.load(pred_path, weights_only=False)
        print(f"Deterministic prediction shape: {det_pred.shape}")

    target_np = target.numpy()
    det_pred_np = det_pred.numpy()

    # Compute metrics
    n = target.shape[0]
    crps_total = 0
    mae_total = 0
    rmse_total = 0

    for i in range(n):
        hr = target_np[i, 0, 0, ...]
        pr = det_pred_np[i, 0, 0, ...]

        mae_total += np.mean(np.abs(hr - pr))
        rmse_total += np.mean((hr - pr) ** 2)

        if is_gan:
            ens = en_pred[i, :, 0, 0, ...].numpy()
            crps_total += crps_ensemble(hr, ens)
        else:
            crps_total += crps_deterministic(hr, pr)

    mae = mae_total / n
    rmse = np.sqrt(rmse_total / n)
    crps = crps_total / n

    print(f"\nResults for {args.model_id}:")
    print(f"  CRPS: {crps:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")

    return crps, mae, rmse


if __name__ == "__main__":
    main()
