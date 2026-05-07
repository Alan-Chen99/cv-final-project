"""Generate sample visualizations for flow matching model outputs.

Produces:
  - Grid of LR (upscaled), HR ground truth, predicted HR, and |error| for N samples
  - Ensemble spread visualization
  - Saves to output directory as PNG files
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Import from flow_matching_v2
sys.path.insert(0, os.path.dirname(__file__))
from flow_matching_v2 import (
    AttentionUNet, euler_sample, apply_addcl, apply_smcl,
    load_tcw4_data, crps_ensemble_correct,
)


def generate_samples(model, lr_up_norm, lr_up, lr_orig, stats, device,
                     n_ensemble=10, ode_steps=10, constraint='addcl'):
    """Generate ensemble predictions for a batch of samples."""
    bs = lr_up_norm.shape[0]
    ensemble = []
    for _ in range(n_ensemble):
        with torch.no_grad():
            res_norm = euler_sample(
                model, lr_up_norm.to(device),
                shape=(bs, 1, 128, 128), steps=ode_steps,
            )
            res = res_norm.cpu() * stats['res_std'] + stats['res_mean']
            pred = lr_up + res
            if constraint == 'addcl':
                pred = apply_addcl(pred, lr_orig)
            elif constraint == 'smcl':
                pred = apply_smcl(pred, lr_orig)
            ensemble.append(pred.numpy())
    return np.stack(ensemble, axis=1)  # (N, M, 1, 128, 128)


def plot_sample_grid(lr_up, hr, ensemble_preds, indices, save_path):
    """Plot grid: LR upscaled | HR | Ensemble Mean | |Error| | Spread."""
    n = len(indices)
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[None, :]

    vmin_global = min(hr[i, 0].numpy().min() for i in indices)
    vmax_global = max(hr[i, 0].numpy().max() for i in indices)

    for row, idx in enumerate(indices):
        lr_img = lr_up[idx, 0].numpy()
        hr_img = hr[idx, 0].numpy()
        ens = ensemble_preds[idx, :, 0]  # (M, 128, 128)
        mean_pred = ens.mean(axis=0)
        error = np.abs(hr_img - mean_pred)
        spread = ens.std(axis=0)

        im0 = axes[row, 0].imshow(lr_img, cmap='viridis', vmin=vmin_global, vmax=vmax_global)
        axes[row, 0].set_title(f'LR bilinear (sample {idx})')
        axes[row, 0].axis('off')

        im1 = axes[row, 1].imshow(hr_img, cmap='viridis', vmin=vmin_global, vmax=vmax_global)
        axes[row, 1].set_title('HR ground truth')
        axes[row, 1].axis('off')

        im2 = axes[row, 2].imshow(mean_pred, cmap='viridis', vmin=vmin_global, vmax=vmax_global)
        axes[row, 2].set_title('Ensemble mean')
        axes[row, 2].axis('off')

        im3 = axes[row, 3].imshow(error, cmap='hot')
        axes[row, 3].set_title(f'|Error| (MAE={error.mean():.4f})')
        axes[row, 3].axis('off')
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046)

        im4 = axes[row, 4].imshow(spread, cmap='YlOrRd')
        axes[row, 4].set_title(f'Spread (std={spread.mean():.4f})')
        axes[row, 4].axis('off')
        plt.colorbar(im4, ax=axes[row, 4], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_ensemble_members(lr_up, hr, ensemble_preds, sample_idx, save_path, n_show=5):
    """Show individual ensemble members for one sample."""
    ens = ensemble_preds[sample_idx, :, 0]  # (M, 128, 128)
    hr_img = hr[sample_idx, 0].numpy()
    lr_img = lr_up[sample_idx, 0].numpy()

    n_cols = min(n_show, ens.shape[0]) + 2  # +2 for LR and HR
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    vmin = hr_img.min()
    vmax = hr_img.max()

    axes[0].imshow(lr_img, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('LR bilinear')
    axes[0].axis('off')

    axes[1].imshow(hr_img, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('HR truth')
    axes[1].axis('off')

    for i in range(min(n_show, ens.shape[0])):
        crps_i = np.mean(np.abs(ens[i] - hr_img))
        axes[i + 2].imshow(ens[i], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i + 2].set_title(f'Member {i+1}\nMAE={crps_i:.4f}')
        axes[i + 2].axis('off')

    plt.suptitle(f'Sample {sample_idx}: Ensemble Members', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True, help="Model checkpoint directory")
    parser.add_argument("--output_dir", required=True, help="Where to save images")
    parser.add_argument("--basedir", default="external/constrained-downscaling")
    parser.add_argument("--n_samples", type=int, default=8, help="Number of test samples")
    parser.add_argument("--n_ensemble", type=int, default=10)
    parser.add_argument("--ode_steps", type=int, default=10)
    parser.add_argument("--constraint", default="addcl", choices=["none", "addcl", "smcl"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    stats = torch.load(os.path.join(args.save_dir, 'norm_stats.pt'),
                        weights_only=False, map_location=device)
    ckpt = torch.load(os.path.join(args.save_dir, 'best_flow.pt'),
                       weights_only=False, map_location=device)
    saved_args = ckpt['args']
    saved_mults = saved_args.get('channel_mults_tuple', (1, 2, 4))
    if isinstance(saved_mults, str):
        saved_mults = tuple(int(x) for x in saved_mults.split(','))

    model = AttentionUNet(
        in_channels=2, out_channels=1,
        base_channels=saved_args.get('base_channels', 64),
        channel_mults=saved_mults,
        time_emb_dim=256, dropout=0.0,
        attn_heads=saved_args.get('attn_heads', 4),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Model loaded: epoch {ckpt['epoch']+1}, val_loss {ckpt['val_loss']:.6f}")

    # Load data
    lr_up, residual, hr, lr_orig = load_tcw4_data(args.basedir, 'test')
    lr_up_norm = (lr_up - stats['lr_mean']) / stats['lr_std']

    # Select samples (evenly spaced + some random)
    n_test = lr_up.shape[0]
    n = min(args.n_samples, n_test)
    indices = list(range(0, n_test, n_test // n))[:n]

    # Generate predictions
    print(f"Generating {args.n_ensemble}-member ensemble for {n} samples...")
    sel_lr_norm = lr_up_norm[indices]
    sel_lr_up = lr_up[indices]
    sel_lr_orig = lr_orig[indices]
    sel_hr = hr[indices]

    preds = generate_samples(model, sel_lr_norm, sel_lr_up, sel_lr_orig, stats, device,
                              n_ensemble=args.n_ensemble, ode_steps=args.ode_steps,
                              constraint=args.constraint)

    # Compute per-sample CRPS
    for i in range(n):
        gt = sel_hr[i, 0].numpy()
        ens = preds[i, :, 0]
        crps = crps_ensemble_correct(gt, ens)
        print(f"  Sample {indices[i]}: CRPS={crps:.4f}, MAE={np.mean(np.abs(gt - ens.mean(0))):.4f}")

    # Plot grid
    plot_sample_grid(sel_lr_up, sel_hr, preds, list(range(n)),
                     os.path.join(args.output_dir, 'sample_grid.png'))

    # Plot ensemble members for first sample
    plot_ensemble_members(sel_lr_up, sel_hr, preds, 0,
                          os.path.join(args.output_dir, 'ensemble_members.png'))

    # Plot ensemble members for a high-error sample
    errors = [np.mean(np.abs(sel_hr[i, 0].numpy() - preds[i, :, 0].mean(0)))
              for i in range(n)]
    worst_idx = np.argmax(errors)
    if worst_idx != 0:
        plot_ensemble_members(sel_lr_up, sel_hr, preds, worst_idx,
                              os.path.join(args.output_dir, 'ensemble_worst.png'))

    print("Done!")


if __name__ == "__main__":
    main()
