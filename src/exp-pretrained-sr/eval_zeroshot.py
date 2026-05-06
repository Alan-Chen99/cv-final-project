"""Zero-shot evaluation of pretrained image SR models on ERA5 TCW 4x downscaling.

Computes correct energy CRPS: CRPS = E|X-y| - 0.5*E|X-X'|
For deterministic models, CRPS = MAE (since X=X' always, the second term is 0).

Usage:
    python src/exp-pretrained-sr/eval_zeroshot.py --method bilinear
    python src/exp-pretrained-sr/eval_zeroshot.py --method bicubic
    python src/exp-pretrained-sr/eval_zeroshot.py --method swinir --device cuda
"""

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

POOL = Path("/home/chenxy/orcd/pool/datasets")
DATA_DIR = POOL / "era5_sr_data"


def energy_crps(samples: np.ndarray, observation: np.ndarray) -> float:
    """Correct energy CRPS: E|X-y| - 0.5*E|X-X'|.

    Args:
        samples: (M, H, W) ensemble of predictions
        observation: (H, W) ground truth
    Returns:
        Scalar CRPS value (averaged over spatial dims)
    """
    M = samples.shape[0]
    # E|X - y|: average absolute difference between each sample and obs
    term1 = np.mean(np.abs(samples - observation[None, ...]))

    # E|X - X'|: average absolute difference between all pairs
    # For efficiency, use the identity: E|X-X'| = (2/(M^2)) * sum_{i<j} |X_i - X_j|
    # But for small M (10-20), brute force is fine
    term2 = 0.0
    count = 0
    for i in range(M):
        for j in range(i + 1, M):
            term2 += np.mean(np.abs(samples[i] - samples[j]))
            count += 1
    term2 = term2 / count  # This gives E|X-X'| for distinct pairs
    # Correction: E|X-X'| over all M^2 pairs (including i==j where |X-X|=0)
    # = (M-1)/M * E|X_i - X_j| for i!=j... actually let's just compute directly
    # E|X-X'| = (1/M^2) * sum_ij |X_i - X_j| = (2*count / M^2) * avg_pair_diff
    term2 = (2.0 * count / (M * M)) * (term2 * count / count)  # simplifies to 2*sum/(M^2)
    # Actually simpler: sum_distinct_pairs / count = avg |X_i - X_j| for i<j
    # E|X-X'| = (1/M^2) * 2 * sum_{i<j} |X_i - X_j| = 2*count*avg / M^2
    # Let me just recompute cleanly:
    pair_sum = 0.0
    for i in range(M):
        for j in range(M):
            if i != j:
                pair_sum += np.mean(np.abs(samples[i] - samples[j]))
    term2 = pair_sum / (M * M)

    return term1 - 0.5 * term2


def energy_crps_fast(samples: np.ndarray, observation: np.ndarray) -> float:
    """Vectorized energy CRPS for small ensembles."""
    M = samples.shape[0]
    # Term 1: E|X - y|
    term1 = np.mean(np.abs(samples - observation[None, ...]))

    # Term 2: E|X - X'| using all M*M pairs (including self)
    # For M <= 20, this is fine
    # Reshape for broadcasting: (M, 1, H, W) - (1, M, H, W)
    diff = np.abs(samples[:, None, ...] - samples[None, :, ...])  # (M, M, H, W)
    term2 = np.mean(diff)

    return term1 - 0.5 * term2


def load_data(split="test"):
    """Load ERA5 SR data."""
    inputs = torch.load(DATA_DIR / split / f"input_{split}.pt", map_location="cpu", weights_only=True)
    targets = torch.load(DATA_DIR / split / f"target_{split}.pt", map_location="cpu", weights_only=True)
    return inputs, targets


def upsample_bilinear(lr_batch):
    """Bilinear 4x upsampling."""
    # lr_batch: (N, 1, 1, 32, 32) -> (N, 1, 1, 128, 128)
    x = lr_batch[:, 0, 0, :, :]  # (N, 32, 32)
    x = x.unsqueeze(1)  # (N, 1, 32, 32)
    x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
    return x.unsqueeze(1).unsqueeze(1)  # (N, 1, 1, 128, 128)


def upsample_bicubic(lr_batch):
    """Bicubic 4x upsampling."""
    x = lr_batch[:, 0, 0, :, :]  # (N, 32, 32)
    x = x.unsqueeze(1)  # (N, 1, 32, 32)
    x = F.interpolate(x, size=(128, 128), mode='bicubic', align_corners=False)
    return x.unsqueeze(1).unsqueeze(1)


def eval_deterministic(predictions: np.ndarray, targets: np.ndarray, n_samples=None):
    """Evaluate deterministic model. CRPS = MAE for deterministic."""
    N = predictions.shape[0]
    if n_samples is not None:
        N = min(N, n_samples)

    mae_sum = 0.0
    rmse_sum = 0.0
    mass_viol_sum = 0.0

    for i in range(N):
        hr = targets[i, 0, 0]  # (128, 128)
        pr = predictions[i, 0, 0]  # (128, 128)

        mae_sum += np.mean(np.abs(hr - pr))
        rmse_sum += np.mean((hr - pr) ** 2)

        # Mass violation: compare downscaled prediction mean to LR
        pr_ds = pr.reshape(32, 4, 32, 4).mean(axis=(1, 3))  # (32, 32)
        lr = targets[i, 0, 0].reshape(32, 4, 32, 4).mean(axis=(1, 3))  # approximate LR from HR
        mass_viol_sum += np.mean(np.abs(pr_ds - lr))

    mae = mae_sum / N
    rmse = np.sqrt(rmse_sum / N)
    crps = mae  # For deterministic, CRPS = MAE

    return {'CRPS': crps, 'MAE': mae, 'RMSE': rmse, 'Mass_viol': mass_viol_sum / N, 'N': N}


def eval_deterministic_with_lr(predictions: np.ndarray, targets: np.ndarray,
                                inputs: np.ndarray, n_samples=None):
    """Evaluate deterministic model with LR inputs for mass violation."""
    N = predictions.shape[0]
    if n_samples is not None:
        N = min(N, n_samples)

    mae_sum = 0.0
    rmse_sum = 0.0
    mass_viol_sum = 0.0

    for i in range(N):
        hr = targets[i, 0, 0]  # (128, 128)
        pr = predictions[i, 0, 0]  # (128, 128)
        lr = inputs[i, 0, 0]  # (32, 32)

        mae_sum += np.mean(np.abs(hr - pr))
        rmse_sum += np.mean((hr - pr) ** 2)

        # Mass violation: compare block-averaged prediction to LR
        pr_ds = pr.reshape(32, 4, 32, 4).mean(axis=(1, 3))  # (32, 32)
        mass_viol_sum += np.mean(np.abs(pr_ds - lr))

    mae = mae_sum / N
    rmse = np.sqrt(rmse_sum / N)
    crps = mae  # For deterministic, CRPS = MAE

    return {'CRPS': crps, 'MAE': mae, 'RMSE': rmse, 'Mass_viol': mass_viol_sum / N, 'N': N}


def load_swinir(device='cpu'):
    """Load pretrained SwinIR classical SR x4 model."""
    from spandrel import ModelLoader

    # Download SwinIR weights if not available
    weights_dir = POOL / "research5" / "pretrained_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    swinir_path = weights_dir / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"

    if not swinir_path.exists():
        print("Downloading SwinIR x4 classical SR weights...")
        import urllib.request
        url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
        urllib.request.urlretrieve(url, str(swinir_path))
        print(f"Downloaded to {swinir_path}")

    model = ModelLoader().load_from_file(str(swinir_path))
    model = model.model.to(device).eval()
    return model


def predict_swinir(model, lr_batch, device='cpu'):
    """Run SwinIR on batch. Handles 1ch→3ch→1ch conversion and normalization."""
    # lr_batch: (N, 1, 1, 32, 32)
    N = lr_batch.shape[0]
    results = []

    # SwinIR expects input in [0, 1] range with 3 channels
    # Normalize to [0, 1]
    vmin = lr_batch.min()
    vmax = lr_batch.max()

    for i in range(N):
        x = lr_batch[i, 0, 0]  # (32, 32)
        # Normalize per-sample
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min < 1e-8:
            x_norm = torch.zeros_like(x)
        else:
            x_norm = (x - x_min) / (x_max - x_min)

        # Replicate to 3 channels
        x_3ch = x_norm.unsqueeze(0).repeat(3, 1, 1)  # (3, 32, 32)
        x_3ch = x_3ch.unsqueeze(0).to(device)  # (1, 3, 32, 32)

        with torch.no_grad():
            out = model(x_3ch)  # (1, 3, 128, 128) ideally

        # Average channels back to 1
        out_1ch = out[0].mean(dim=0)  # (128, 128)

        # Denormalize back to physical range
        out_1ch = out_1ch.clamp(0, 1) * (x_max - x_min) + x_min
        results.append(out_1ch.cpu())

    # Stack results back to (N, 1, 1, 128, 128)
    results = torch.stack(results).unsqueeze(1).unsqueeze(1)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="bilinear",
                       choices=["bilinear", "bicubic", "swinir"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_samples", type=int, default=None,
                       help="Number of samples to evaluate (None = all)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading {args.split} data...")
    inputs, targets = load_data(args.split)
    print(f"  Inputs: {inputs.shape}, Targets: {targets.shape}")

    N = inputs.shape[0]
    if args.n_samples:
        N = min(N, args.n_samples)
        inputs = inputs[:N]
        targets = targets[:N]

    print(f"Evaluating {args.method} on {N} samples...")
    t0 = time.time()

    if args.method == "bilinear":
        predictions = upsample_bilinear(inputs)
    elif args.method == "bicubic":
        predictions = upsample_bicubic(inputs)
    elif args.method == "swinir":
        model = load_swinir(args.device)
        # Process in batches
        predictions = []
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            batch = inputs[start:end]
            pred = predict_swinir(model, batch, args.device)
            predictions.append(pred)
            if (start // args.batch_size) % 10 == 0:
                print(f"  Processed {end}/{N}...")
        predictions = torch.cat(predictions, dim=0)

    elapsed = time.time() - t0
    print(f"  Inference time: {elapsed:.1f}s")

    # Convert to numpy for evaluation
    predictions_np = predictions.numpy()
    targets_np = targets.numpy()
    inputs_np = inputs.numpy()

    results = eval_deterministic_with_lr(predictions_np, targets_np, inputs_np, N)

    print(f"\n{'='*50}")
    print(f"Results: {args.method} ({args.split}, N={results['N']})")
    print(f"{'='*50}")
    print(f"  CRPS (=MAE for det.): {results['CRPS']:.6f}")
    print(f"  MAE:                  {results['MAE']:.6f}")
    print(f"  RMSE:                 {results['RMSE']:.6f}")
    print(f"  Mass violation:       {results['Mass_viol']:.6f}")
    print(f"{'='*50}")

    return results


if __name__ == "__main__":
    main()
