"""
Inference ablation: test different solver, constraint, steps, TTA settings
on the best UNet v2 model (unet_uniform_amp) WITHOUT retraining.

Usage:
  python src/exp-spatial-4x-crps-v1/inference_ablation.py \
    --save_dir /path/to/model --basedir external/constrained-downscaling \
    --configs "euler_10_addcl,euler_10_smcl,midpoint_5_addcl,euler_20_addcl,euler_10_addcl_tta"
"""

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import from flow_matching_v2
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from flow_matching_v2 import (
    AttentionUNet, load_tcw4_data,
    euler_sample, midpoint_sample,
    apply_addcl, apply_smcl,
    crps_ensemble_correct,
)


def run_eval(model, lr_up_norm, lr_up, hr, lr_orig, stats, device,
             sampler, steps, constraint, n_ensemble, tta, batch_size, max_samples):
    """Run one evaluation configuration and return metrics."""
    n_samples = min(lr_up.shape[0], max_samples) if max_samples else lr_up.shape[0]
    sample_fn = midpoint_sample if sampler == 'midpoint' else euler_sample

    all_crps = []
    all_mae = []
    all_rmse = []
    all_mass_viol = []
    pool = torch.nn.AvgPool2d(kernel_size=4)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_lr = lr_up_norm[start_idx:end_idx].to(device)
        batch_hr = hr[start_idx:end_idx]
        batch_lr_up = lr_up[start_idx:end_idx]
        batch_lr_orig = lr_orig[start_idx:end_idx]
        bs = batch_lr.shape[0]

        ensemble_preds = []
        for e in range(n_ensemble):
            with torch.no_grad(), torch.amp.autocast('cuda'):
                do_flip = tta and (e % 2 == 1)
                cond = torch.flip(batch_lr, [-1]) if do_flip else batch_lr
                sampled_res_norm = sample_fn(
                    model, cond, shape=(bs, 1, 128, 128), steps=steps,
                )
                if do_flip:
                    sampled_res_norm = torch.flip(sampled_res_norm, [-1])
                sampled_res = sampled_res_norm.cpu() * stats['res_std'] + stats['res_mean']
                pred_hr = batch_lr_up + sampled_res

                if constraint == 'addcl':
                    pred_hr = apply_addcl(pred_hr, batch_lr_orig)
                elif constraint == 'smcl':
                    pred_hr = apply_smcl(pred_hr, batch_lr_orig)

                ensemble_preds.append(pred_hr.numpy())

        ensemble_preds = np.stack(ensemble_preds, axis=1)

        for i in range(bs):
            gt = batch_hr[i, 0, ...].numpy()
            ens = ensemble_preds[i, :, 0, ...]
            ens_mean = ens.mean(axis=0)

            all_crps.append(crps_ensemble_correct(gt, ens))
            all_mae.append(np.mean(np.abs(gt - ens_mean)))
            all_rmse.append(np.mean((gt - ens_mean) ** 2))

            pred_mean_t = torch.from_numpy(ens_mean).unsqueeze(0).unsqueeze(0)
            pooled = pool(pred_mean_t).squeeze()
            lr_i = batch_lr_orig[i, 0, ...]
            all_mass_viol.append(torch.mean(torch.abs(pooled - lr_i)).item())

    return {
        'CRPS': float(np.mean(all_crps)),
        'MAE': float(np.mean(all_mae)),
        'RMSE': float(np.sqrt(np.mean(all_rmse))),
        'Mass_Viol': float(np.mean(all_mass_viol)),
        'N': len(all_crps),
    }


def parse_config(config_str):
    """Parse config string like 'euler_10_addcl_tta' into dict."""
    parts = config_str.split('_')
    cfg = {'sampler': parts[0], 'steps': int(parts[1]), 'constraint': parts[2]}
    cfg['tta'] = 'tta' in parts[3:]
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--basedir", default="external/constrained-downscaling")
    parser.add_argument("--configs", required=True,
                        help="Comma-separated config strings: solver_steps_constraint[_tta]")
    parser.add_argument("--n_ensemble", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    stats = torch.load(os.path.join(args.save_dir, 'norm_stats.pt'),
                        weights_only=False, map_location='cpu')
    ckpt = torch.load(os.path.join(args.save_dir, 'best_flow.pt'),
                       weights_only=False, map_location='cpu')
    saved_args = ckpt['args']
    saved_mults = saved_args.get('channel_mults_tuple', (1, 2, 4))
    if isinstance(saved_mults, str):
        saved_mults = tuple(int(x) for x in saved_mults.split(','))

    model = AttentionUNet(
        in_channels=2, out_channels=1,
        base_channels=saved_args.get('base_channels', 64),
        channel_mults=saved_mults,
        time_emb_dim=256,
        dropout=0.0,
        attn_heads=saved_args.get('attn_heads', 4),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Model loaded: epoch {ckpt['epoch']+1}, val_loss {ckpt['val_loss']:.6f}")

    # Load data
    lr_up, residual, hr, lr_orig = load_tcw4_data(args.basedir, args.split)
    lr_up_norm = (lr_up - stats['lr_mean']) / stats['lr_std']
    print(f"Data: {lr_up.shape[0]} {args.split} samples")

    # Run configs
    configs = [c.strip() for c in args.configs.split(',')]
    results = {}

    for config_str in configs:
        cfg = parse_config(config_str)
        print(f"\n{'='*60}")
        print(f"Config: {config_str}")
        print(f"  sampler={cfg['sampler']}, steps={cfg['steps']}, "
              f"constraint={cfg['constraint']}, tta={cfg['tta']}")
        t0 = time.time()

        metrics = run_eval(
            model, lr_up_norm, lr_up, hr, lr_orig, stats, device,
            sampler=cfg['sampler'], steps=cfg['steps'],
            constraint=cfg['constraint'], n_ensemble=args.n_ensemble,
            tta=cfg['tta'], batch_size=args.eval_batch_size,
            max_samples=args.max_samples,
        )
        elapsed = time.time() - t0
        results[config_str] = metrics

        print(f"  CRPS: {metrics['CRPS']:.6f}")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  MAE:  {metrics['MAE']:.6f}")
        print(f"  Mass: {metrics['Mass_Viol']:.6f}")
        print(f"  Time: {elapsed:.0f}s ({metrics['N']} samples)")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Config':<30} {'CRPS':>8} {'RMSE':>8} {'MAE':>8} {'Mass':>10}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for config_str, m in results.items():
        print(f"{config_str:<30} {m['CRPS']:8.6f} {m['RMSE']:8.6f} {m['MAE']:8.6f} {m['Mass_Viol']:10.6f}")


if __name__ == '__main__':
    main()
