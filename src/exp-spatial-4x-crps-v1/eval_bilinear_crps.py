"""Compute CRPS for bilinear interpolation baseline (trivial baseline)."""
import numpy as np
import torch
import torch.nn.functional as F


def crps_deterministic(observation, prediction):
    return np.mean(np.abs(prediction - observation))


def main():
    for split in ['val', 'test']:
        inp = torch.load(f'external/constrained-downscaling/data/era5_sr_data/{split}/input_{split}.pt', weights_only=False)
        tgt = torch.load(f'external/constrained-downscaling/data/era5_sr_data/{split}/target_{split}.pt', weights_only=False)

        # Bilinear interpolation: 32x32 -> 128x128
        inp_up = F.interpolate(inp[:, 0, :, :, :], size=(128, 128), mode='bilinear', align_corners=False)

        pred_np = inp_up.numpy()
        tgt_np = tgt[:, 0, :, :, :].numpy()

        crps = 0
        mae = 0
        rmse = 0
        n = tgt.shape[0]

        for i in range(n):
            hr = tgt_np[i, 0, ...]
            pr = pred_np[i, 0, ...]
            crps += crps_deterministic(hr, pr)
            mae += np.mean(np.abs(hr - pr))
            rmse += np.mean((hr - pr) ** 2)

        crps /= n
        mae /= n
        rmse = np.sqrt(rmse / n)

        print(f"\nBilinear Interpolation ({split}):")
        print(f"  CRPS: {crps:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")


if __name__ == "__main__":
    main()
