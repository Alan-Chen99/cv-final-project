"""Compute metrics for a saved prediction file."""
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure, structural_similarity_index_measure
from skimage import transform

BASEDIR = "external/constrained-downscaling"


def crps_ensemble(observation, forecasts):
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


def compute_scores(model_id, split="test", model_type="cnn", upsampling_factor=4):
    input_data = torch.load(f'{BASEDIR}/data/era5_sr_data/{split}/input_{split}.pt', weights_only=False)
    target_data = torch.load(f'{BASEDIR}/data/era5_sr_data/{split}/target_{split}.pt', weights_only=False)

    is_gan = model_type == 'gan'

    if is_gan:
        en_pred = torch.load(f'{BASEDIR}/data/prediction/era5_sr_data_{model_id}_{split}_ensemble.pt', weights_only=False)
        pred = torch.mean(en_pred, dim=1)
        en_pred_np = en_pred.detach().cpu().numpy()
    else:
        pred = torch.load(f'{BASEDIR}/data/prediction/era5_sr_data_{model_id}_{split}.pt', weights_only=False)

    pred_np = pred.detach().cpu().numpy()
    max_val = target_data.max()
    min_val = target_data.min()

    l2_crit = nn.MSELoss()
    l1_crit = nn.L1Loss()

    n = input_data.shape[0]
    mse = mae = ssim_sum = ms_ssim_sum = corr_sum = crps_sum = mass_viol = 0
    mean_bias = mean_abs_bias = 0
    neg_num = neg_mean = 0

    for i in range(n):
        lr = input_data[i].numpy()
        hr = target_data[i]
        pr = torch.Tensor(pred_np[i])

        j = 0
        mse += l2_crit(pr[j, ...], hr[j, ...]).item()
        mae += l1_crit(pr[j, ...], hr[j, ...]).item()
        mean_bias += torch.mean(hr[j, ...] - pr[j, ...]).item()
        mean_abs_bias += torch.abs(torch.mean(hr[j, ...] - pr[j, ...])).item()

        # Pearson correlation
        x = pr[j, ...].flatten()
        y = hr[j, ...].flatten()
        xm = x - x.mean()
        ym = y - y.mean()
        corr_sum += (xm.dot(ym) / (torch.norm(xm, 2) * torch.norm(ym, 2))).item()

        ms_ssim_sum += multiscale_structural_similarity_index_measure(
            pr[j:j+1, ...].unsqueeze(0), hr[j:j+1, ...].unsqueeze(0),
            data_range=max_val - min_val, kernel_size=11,
            betas=(0.2856, 0.3001, 0.2363)
        ).item()

        ssim_sum += structural_similarity_index_measure(
            pr[j:j+1, ...].unsqueeze(0), hr[j:j+1, ...].unsqueeze(0),
            data_range=max_val - min_val, kernel_size=11
        ).item()

        neg_num += np.sum(pred_np[i, j, ...] < 0)

        if is_gan:
            crps_sum += crps_ensemble(hr[j, 0, ...].numpy(), en_pred_np[i, :, j, 0, ...])
        else:
            # CRPS for deterministic = MAE
            crps_sum += np.mean(np.abs(hr[j, ...].numpy() - pred_np[i, j, ...]))

        mass_viol += np.mean(np.abs(
            transform.downscale_local_mean(pred_np[i, j, ...], (1, upsampling_factor, upsampling_factor)) - lr[j, ...]
        ))

    n_inv = 1.0 / n
    mse *= n_inv
    mae *= n_inv
    ssim = ssim_sum * n_inv
    ms_ssim = ms_ssim_sum * n_inv
    corr = corr_sum * n_inv
    crps = crps_sum * n_inv
    mean_bias *= n_inv
    mean_abs_bias *= n_inv
    mass_viol *= n_inv
    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(max_val.item() / rmse)

    results = {
        'model_id': model_id,
        'RMSE': f'{rmse:.4f}',
        'MAE': f'{mae:.4f}',
        'CRPS': f'{crps:.6f}',
        'PSNR': f'{psnr:.2f}',
        'SSIM': f'{ssim:.4f}',
        'MS-SSIM': f'{ms_ssim:.4f}',
        'Pearson': f'{corr:.4f}',
        'Mass_viol': f'{mass_viol:.6f}',
        'Mean_bias': f'{mean_bias:.4f}',
        'Neg_pixels': neg_num,
    }

    print(f"\nResults for {model_id} ({split}):")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results


if __name__ == "__main__":
    model_id = sys.argv[1] if len(sys.argv) > 1 else "twc_cnn_none"
    model_type = sys.argv[2] if len(sys.argv) > 2 else "cnn"
    split = sys.argv[3] if len(sys.argv) > 3 else "test"
    compute_scores(model_id, split, model_type)
