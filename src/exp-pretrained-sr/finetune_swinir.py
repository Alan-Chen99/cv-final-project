"""Finetune pretrained SwinIR on ERA5 TCW 4x downscaling.

Modifies SwinIR (originally 3ch RGB) for 1ch climate data:
- Input conv: 3ch→1ch (avg initialization)
- Output conv: 3ch→1ch (sum initialization)
- Global normalization to [0,1] range using dataset statistics

Training:
- MSE loss (default) or L1 loss
- AdamW optimizer with cosine annealing
- 2hr wall-clock budget

Usage:
    python src/exp-pretrained-sr/finetune_swinir.py --epochs 50 --batch_size 64 --lr 2e-4
    python src/exp-pretrained-sr/finetune_swinir.py --mode eval --checkpoint best
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

POOL = Path("/home/chenxy/orcd/pool/datasets")
DATA_DIR = POOL / "era5_sr_data"
SAVE_DIR = POOL / "research5" / "models" / "swinir_ft"


def load_swinir_1ch(weights_path, freeze_backbone=False):
    """Load SwinIR and adapt from 3ch to 1ch input/output."""
    from spandrel import ModelLoader

    model_desc = ModelLoader().load_from_file(str(weights_path))
    model = model_desc.model

    # Fix mean buffer: (1,3,1,1) -> (1,1,1,1) to avoid broadcasting 1ch to 3ch
    if hasattr(model, 'mean'):
        new_mean = model.mean.mean(dim=1, keepdim=True)  # (1,1,1,1)
        model.mean = new_mean

    # Adapt input convolution: (out_ch, 3, kh, kw) -> (out_ch, 1, kh, kw)
    old_conv_first = model.conv_first
    in_ch_out = old_conv_first.out_channels
    kh, kw = old_conv_first.kernel_size
    new_conv_first = nn.Conv2d(1, in_ch_out, (kh, kw), padding=old_conv_first.padding)
    # Initialize: average over 3 input channels
    with torch.no_grad():
        new_conv_first.weight.copy_(old_conv_first.weight.mean(dim=1, keepdim=True))
        new_conv_first.bias.copy_(old_conv_first.bias)
    model.conv_first = new_conv_first

    # Adapt output convolution: last conv_last (in_ch, 3, kh, kw) -> (in_ch, 1, kh, kw)
    old_conv_last = model.conv_last
    in_ch = old_conv_last.in_channels
    kh, kw = old_conv_last.kernel_size
    new_conv_last = nn.Conv2d(in_ch, 1, (kh, kw), padding=old_conv_last.padding)
    # Initialize: sum over 3 output channels (since avg of 3 channels = sum/3)
    with torch.no_grad():
        new_conv_last.weight.copy_(old_conv_last.weight.mean(dim=0, keepdim=True))
        if old_conv_last.bias is not None:
            new_conv_last.bias.copy_(old_conv_last.bias.mean(dim=0, keepdim=True))
    model.conv_last = new_conv_last

    # Set img_range=1.0 since our data is already normalized to [0,1]
    if hasattr(model, 'img_range'):
        model.img_range = 1.0

    # Set mean to 0 since we handle normalization externally
    model.mean = torch.zeros(1, 1, 1, 1)

    if freeze_backbone:
        # Freeze everything except first/last conv
        for name, param in model.named_parameters():
            if 'conv_first' not in name and 'conv_last' not in name:
                param.requires_grad = False

    return model


def load_data(split='train'):
    """Load and normalize ERA5 data. Returns normalized tensors and stats."""
    inputs = torch.load(DATA_DIR / split / f"input_{split}.pt", map_location="cpu", weights_only=True)
    targets = torch.load(DATA_DIR / split / f"target_{split}.pt", map_location="cpu", weights_only=True)

    # Shape: (N, 1, 1, H, W) -> (N, 1, H, W)
    lr = inputs[:, 0, :, :, :]  # (N, 1, 32, 32)
    hr = targets[:, 0, :, :, :]  # (N, 1, 128, 128)

    return lr, hr


def normalize(x, vmin, vmax):
    """Normalize to [0, 1]."""
    return (x - vmin) / (vmax - vmin + 1e-8)


def denormalize(x, vmin, vmax):
    """Denormalize from [0, 1]."""
    return x * (vmax - vmin) + vmin


def energy_crps_fast(samples, observation):
    """Vectorized energy CRPS: E|X-y| - 0.5*E|X-X'|.

    samples: (M, H, W) numpy array
    observation: (H, W) numpy array
    """
    M = samples.shape[0]
    term1 = np.mean(np.abs(samples - observation[None, ...]))
    if M == 1:
        return term1  # No spread term for single sample
    diff = np.abs(samples[:, None, ...] - samples[None, :, ...])  # (M, M, H, W)
    term2 = np.mean(diff)
    return term1 - 0.5 * term2


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("Loading data...")
    lr_train, hr_train = load_data('train')
    lr_val, hr_val = load_data('val')

    # Compute global normalization stats from training data
    all_data = torch.cat([lr_train.flatten(), hr_train.flatten()])
    vmin = all_data.min().item()
    vmax = all_data.max().item()
    print(f"  Normalization range: [{vmin:.4f}, {vmax:.4f}]")

    # Normalize
    lr_train_n = normalize(lr_train, vmin, vmax)
    hr_train_n = normalize(hr_train, vmin, vmax)
    lr_val_n = normalize(lr_val, vmin, vmax)
    hr_val_n = normalize(hr_val, vmin, vmax)

    # DataLoaders
    train_ds = TensorDataset(lr_train_n, hr_train_n)
    val_ds = TensorDataset(lr_val_n, hr_val_n)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Model
    weights_path = POOL / "research5" / "pretrained_weights" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    model = load_swinir_1ch(str(weights_path), freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss
    if args.loss == 'l1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    # Save dir
    save_dir = SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['vmin'] = vmin
    config['vmax'] = vmax
    config['n_params'] = n_params
    config['n_trainable'] = n_trainable
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    start_time = time.time()
    wall_limit = args.wall_hours * 3600

    print(f"\nTraining for up to {args.epochs} epochs ({args.wall_hours}h wall limit)...")
    print(f"  Loss: {args.loss}, LR: {args.lr}, BS: {args.batch_size}")

    for epoch in range(args.epochs):
        # Check wall time
        elapsed = time.time() - start_time
        if elapsed > wall_limit:
            print(f"\nWall time limit reached ({elapsed/3600:.2f}h). Stopping.")
            break

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for lr_batch, hr_batch in train_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            pred = model(lr_batch)
            loss = criterion(pred, hr_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                pred = model(lr_batch)
                val_loss += criterion(pred, hr_batch).item()
                n_val += 1
        val_loss /= n_val
        val_losses.append(val_loss)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'optimizer': optimizer.state_dict(),
                'vmin': vmin,
                'vmax': vmax,
            }, save_dir / 'best_swinir.pt')

        elapsed = time.time() - start_time
        lr_now = scheduler.get_last_lr()[0]
        print(f"  Ep {epoch+1:3d}/{args.epochs} | train: {train_loss:.6f} | val: {val_loss:.6f} | "
              f"best: {best_val_loss:.6f} | lr: {lr_now:.2e} | {elapsed/60:.1f}min")

    # Save final model and losses
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'vmin': vmin,
        'vmax': vmax,
    }, save_dir / 'final_swinir.pt')
    torch.save({'train': train_losses, 'val': val_losses}, save_dir / 'losses.pt')

    elapsed = time.time() - start_time
    print(f"\nTraining complete. {elapsed/60:.1f} minutes, {epoch+1} epochs.")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Saved to: {save_dir}")


def evaluate(args):
    """Evaluate finetuned SwinIR."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    save_dir = SAVE_DIR
    ckpt_path = save_dir / f'{args.checkpoint}_swinir.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    vmin = ckpt['vmin']
    vmax = ckpt['vmax']
    print(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.6f})")
    print(f"  Normalization: [{vmin:.4f}, {vmax:.4f}]")

    # Load model
    weights_path = POOL / "research5" / "pretrained_weights" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    model = load_swinir_1ch(str(weights_path))
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()

    # Load test data
    print(f"Loading {args.split} data...")
    lr, hr = load_data(args.split)
    N = lr.shape[0]
    if args.n_samples:
        N = min(N, args.n_samples)
        lr = lr[:N]
        hr = hr[:N]

    # Normalize and predict
    lr_n = normalize(lr, vmin, vmax)
    predictions = []
    batch_size = args.batch_size

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = lr_n[start:end].to(device)
            pred = model(batch).cpu()
            predictions.append(pred)
            if start % (batch_size * 10) == 0:
                print(f"  {end}/{N}")

    predictions = torch.cat(predictions, dim=0)  # (N, 1, 128, 128) normalized
    predictions_phys = denormalize(predictions, vmin, vmax)

    # Compute metrics
    hr_np = hr[:N, 0].numpy()  # (N, 128, 128)
    pred_np = predictions_phys[:, 0].numpy()  # (N, 128, 128)
    lr_np = lr[:N, 0].numpy()  # (N, 32, 32)

    mae = np.mean(np.abs(hr_np - pred_np))
    rmse = np.sqrt(np.mean((hr_np - pred_np) ** 2))

    # Mass violation
    pred_ds = pred_np.reshape(N, 32, 4, 32, 4).mean(axis=(2, 4))  # (N, 32, 32)
    mass_viol = np.mean(np.abs(pred_ds - lr_np))

    # Negative fraction
    neg_frac = (predictions_phys < 0).float().mean().item()

    print(f"\n{'='*60}")
    print(f"Results: SwinIR finetuned ({args.split}, N={N})")
    print(f"{'='*60}")
    print(f"  CRPS (=MAE, det.): {mae:.6f}")
    print(f"  MAE:               {mae:.6f}")
    print(f"  RMSE:              {rmse:.6f}")
    print(f"  Mass violation:    {mass_viol:.6f}")
    print(f"  Neg fraction:      {neg_frac:.6f}")

    # Also apply AddCL constraint and re-evaluate
    pred_constrained = apply_addcl(predictions_phys[:, 0].numpy(), lr_np)
    mae_c = np.mean(np.abs(hr_np - pred_constrained))
    rmse_c = np.sqrt(np.mean((hr_np - pred_constrained) ** 2))
    pred_c_ds = pred_constrained.reshape(N, 32, 4, 32, 4).mean(axis=(2, 4))
    mass_viol_c = np.mean(np.abs(pred_c_ds - lr_np))

    print(f"\n  With AddCL constraint:")
    print(f"  CRPS (=MAE, det.): {mae_c:.6f}")
    print(f"  RMSE:              {rmse_c:.6f}")
    print(f"  Mass violation:    {mass_viol_c:.6f}")

    return {'MAE': mae, 'RMSE': rmse, 'Mass_viol': mass_viol,
            'MAE_addcl': mae_c, 'RMSE_addcl': rmse_c, 'Mass_viol_addcl': mass_viol_c}


def apply_addcl(predictions, lr):
    """Apply additive constraint layer: shift prediction so block-means match LR.

    predictions: (N, 128, 128) numpy
    lr: (N, 32, 32) numpy
    Returns: (N, 128, 128) constrained predictions
    """
    N = predictions.shape[0]
    result = predictions.copy()
    for i in range(N):
        # Block-average prediction
        pred_ds = predictions[i].reshape(32, 4, 32, 4).mean(axis=(1, 3))  # (32, 32)
        # Correction per block
        correction = lr[i] - pred_ds  # (32, 32)
        # Spread correction to HR grid
        correction_hr = np.repeat(np.repeat(correction, 4, axis=0), 4, axis=1)  # (128, 128)
        result[i] = predictions[i] + correction_hr
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", choices=["train", "eval"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss", default="l1", choices=["l1", "mse"])
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--wall_hours", type=float, default=2.0)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
