"""Run zero-shot evaluation of pretrained SR models on ERA5 TCW 4x downscaling.

Evaluates: bilinear, bicubic, SwinIR zero-shot
Uses correct energy CRPS for ensemble, MAE for deterministic.

Run on GPU node:
    srun --jobid=<JOB_ID> python experiments/pretrained-sr-downscaling/src/run_zeroshot_eval.py
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

POOL = Path("/home/chenxy/orcd/pool/datasets")
DATA_DIR = POOL / "era5_sr_data"


def load_data(split="test"):
    """Load ERA5 SR data. Returns (inputs, targets) with shapes (N,1,1,H,W)."""
    inputs = torch.load(DATA_DIR / split / f"input_{split}.pt", map_location="cpu", weights_only=True)
    targets = torch.load(DATA_DIR / split / f"target_{split}.pt", map_location="cpu", weights_only=True)
    return inputs, targets


def eval_deterministic(predictions, targets, inputs, n=None):
    """Evaluate deterministic predictions. CRPS = MAE for deterministic."""
    if n is None:
        n = predictions.shape[0]

    # predictions: (N, 1, 1, 128, 128), targets same, inputs: (N, 1, 1, 32, 32)
    pr = predictions[:n, 0, 0]  # (N, 128, 128)
    hr = targets[:n, 0, 0]  # (N, 128, 128)
    lr = inputs[:n, 0, 0]  # (N, 32, 32)

    mae = torch.mean(torch.abs(hr - pr)).item()
    mse = torch.mean((hr - pr) ** 2).item()
    rmse = mse ** 0.5

    # Mass violation: block-average prediction vs LR
    pr_ds = pr.reshape(n, 32, 4, 32, 4).mean(dim=(2, 4))  # (N, 32, 32)
    mass_viol = torch.mean(torch.abs(pr_ds - lr)).item()

    # Negative pixels
    neg_frac = (pr < 0).float().mean().item()

    return {
        'CRPS': mae,  # For deterministic, CRPS = MAE
        'MAE': mae,
        'RMSE': rmse,
        'Mass_viol': mass_viol,
        'Neg_frac': neg_frac,
        'N': n,
    }


def run_swinir_zeroshot(inputs, device='cuda', batch_size=128):
    """Run SwinIR zero-shot on ERA5 data.

    Strategy: normalize each sample to [0,1], replicate to 3 channels,
    run SwinIR, average output channels, denormalize.
    """
    from spandrel import ModelLoader

    weights_path = POOL / "research5" / "pretrained_weights" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    model_descriptor = ModelLoader().load_from_file(str(weights_path))
    model = model_descriptor.model.to(device).eval()

    N = inputs.shape[0]
    results = torch.zeros(N, 1, 1, 128, 128)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = inputs[start:end, 0, 0]  # (B, 32, 32)
            B = batch.shape[0]

            # Per-sample normalization to [0, 1]
            batch_flat = batch.reshape(B, -1)
            vmin = batch_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)  # (B, 1, 1)
            vmax = batch_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)  # (B, 1, 1)
            scale = (vmax - vmin).clamp(min=1e-8)
            batch_norm = (batch - vmin) / scale  # (B, 32, 32) in [0, 1]

            # Replicate to 3 channels
            x = batch_norm.unsqueeze(1).expand(-1, 3, -1, -1).to(device)  # (B, 3, 32, 32)

            # SwinIR inference
            out = model(x)  # (B, 3, 128, 128)

            # Average channels and denormalize
            out_1ch = out.mean(dim=1, keepdim=True)  # (B, 1, 128, 128)
            out_1ch = out_1ch.clamp(0, 1).cpu()

            # Denormalize back to physical units
            vmin_hr = vmin.unsqueeze(-1)  # (B, 1, 1, 1)
            scale_hr = scale.unsqueeze(-1)  # (B, 1, 1, 1)
            out_phys = out_1ch * scale_hr + vmin_hr  # (B, 1, 128, 128)

            results[start:end, 0, 0] = out_phys.squeeze(1)

            if start % (batch_size * 10) == 0:
                print(f"  SwinIR: {end}/{N}")

    return results


def main():
    print("=" * 60)
    print("Zero-Shot Evaluation: ERA5 TCW 4x Downscaling")
    print("=" * 60)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\nLoading test data...")
    inputs, targets = load_data('test')
    N = inputs.shape[0]
    print(f"  Test set: {N} samples, LR: {inputs.shape[-2:]}, HR: {targets.shape[-2:]}")
    print(f"  Value range: [{inputs.min():.2f}, {inputs.max():.2f}] (LR), [{targets.min():.2f}, {targets.max():.2f}] (HR)")

    results_all = {}

    # 1. Bilinear
    print("\n--- Bilinear Interpolation ---")
    t0 = time.time()
    lr = inputs[:, 0, :, :, :]  # (N, 1, 32, 32)
    bilinear_pred = F.interpolate(lr, size=(128, 128), mode='bilinear', align_corners=False)
    bilinear_pred = bilinear_pred.unsqueeze(1)  # (N, 1, 1, 128, 128)
    r = eval_deterministic(bilinear_pred, targets, inputs)
    r['time'] = time.time() - t0
    results_all['bilinear'] = r
    print(f"  MAE/CRPS: {r['MAE']:.4f}, RMSE: {r['RMSE']:.4f}, Mass viol: {r['Mass_viol']:.4f}, Time: {r['time']:.1f}s")

    # 2. Bicubic
    print("\n--- Bicubic Interpolation ---")
    t0 = time.time()
    bicubic_pred = F.interpolate(lr, size=(128, 128), mode='bicubic', align_corners=False)
    bicubic_pred = bicubic_pred.unsqueeze(1)
    r = eval_deterministic(bicubic_pred, targets, inputs)
    r['time'] = time.time() - t0
    results_all['bicubic'] = r
    print(f"  MAE/CRPS: {r['MAE']:.4f}, RMSE: {r['RMSE']:.4f}, Mass viol: {r['Mass_viol']:.4f}, Time: {r['time']:.1f}s")

    # 3. SwinIR zero-shot
    print("\n--- SwinIR (Zero-Shot, pretrained on DF2K) ---")
    t0 = time.time()
    swinir_pred = run_swinir_zeroshot(inputs, device=device, batch_size=64)
    r = eval_deterministic(swinir_pred, targets, inputs)
    r['time'] = time.time() - t0
    results_all['swinir_zeroshot'] = r
    print(f"  MAE/CRPS: {r['MAE']:.4f}, RMSE: {r['RMSE']:.4f}, Mass viol: {r['Mass_viol']:.4f}, Neg: {r['Neg_frac']:.4f}, Time: {r['time']:.1f}s")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY (Deterministic CRPS = MAE)")
    print("=" * 60)
    print(f"{'Method':<20} {'CRPS':<10} {'RMSE':<10} {'Mass Viol':<12} {'Time(s)':<10}")
    print("-" * 60)
    for name, r in results_all.items():
        print(f"{name:<20} {r['CRPS']:<10.4f} {r['RMSE']:<10.4f} {r['Mass_viol']:<12.4f} {r['time']:<10.1f}")

    # Save results
    out_dir = POOL / "research5" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results_all, out_dir / "zeroshot_results.pt")
    print(f"\nResults saved to {out_dir / 'zeroshot_results.pt'}")


if __name__ == "__main__":
    main()
