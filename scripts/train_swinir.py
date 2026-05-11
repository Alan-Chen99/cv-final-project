#!/usr/bin/env python3
"""Train (finetune) SwinIR on ERA5 or NorESM dataset.

Adapts pretrained SwinIR (DF2K, 3ch RGB) to 1ch climate data and finetunes
with global min-max normalization to [0,1].

Usage:
    # NorESM 2x (uses x2 pretrained weights)
    python scripts/train_swinir.py --dataset noresm \
        --pretrained-weights /pool/noresm-dataset/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth \
        --save-dir /pool/noresm-dataset/models/swinir_ft

    # ERA5 4x (uses x4 pretrained weights)
    python scripts/train_swinir.py --dataset era5 \
        --pretrained-weights /pool/research5/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth \
        --save-dir /pool/spatial-4x-add-v2/models/swinir_ft
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from downscaling.evaluation.swinir import load_swinir_1ch


def load_dataset(data_dir: Path, dataset: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load (lr, hr) tensors for a dataset split. Returns 4D (N, 1, H, W)."""
    if dataset == "noresm":
        base = data_dir / "noresm-dataset" / "noresm"
        inp = torch.load(base / f"input_{split}.pt", weights_only=False)
        tgt = torch.load(base / f"target_{split}.pt", weights_only=False)
        # (N, 1, 1, H, W) -> (N, 1, H, W)
        return inp[:, 0, :, :, :], tgt[:, 0, :, :, :]
    elif dataset == "era5":
        base = data_dir / "era5_sr_data" / split
        inp = torch.load(base / f"input_{split}.pt", weights_only=False)
        tgt = torch.load(base / f"target_{split}.pt", weights_only=False)
        return inp[:, 0, :, :, :], tgt[:, 0, :, :, :]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main() -> None:
    p = argparse.ArgumentParser(description="Finetune SwinIR on climate data")
    p.add_argument("--dataset", required=True, choices=["era5", "noresm"])
    p.add_argument("--data-dir", type=Path, default=Path("/home/chenxy/orcd/pool/datasets"))
    p.add_argument("--pretrained-weights", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss-fn", default="l1", choices=["l1", "mse"])
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--wall-hours", type=float, default=2.0)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"Loading {args.dataset} data...")
    lr_train, hr_train = load_dataset(args.data_dir, args.dataset, "train")
    lr_val, hr_val = load_dataset(args.data_dir, args.dataset, "val")
    print(f"  Train: {lr_train.shape} -> {hr_train.shape}")
    print(f"  Val:   {lr_val.shape} -> {hr_val.shape}")

    upsampling_factor = hr_train.shape[-1] // lr_train.shape[-1]
    print(f"  Upsampling factor: {upsampling_factor}x")

    # Global min-max normalization
    all_data = torch.cat([lr_train.flatten(), hr_train.flatten()])
    vmin = all_data.min().item()
    vmax = all_data.max().item()
    print(f"  Normalization range: [{vmin:.4f}, {vmax:.4f}]")

    def normalize(x: torch.Tensor) -> torch.Tensor:
        return (x - vmin) / (vmax - vmin + 1e-8)

    lr_train_n = normalize(lr_train)
    hr_train_n = normalize(hr_train)
    lr_val_n = normalize(lr_val)
    hr_val_n = normalize(hr_val)

    train_loader = DataLoader(
        TensorDataset(lr_train_n, hr_train_n),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(lr_val_n, hr_val_n),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model: load pretrained and adapt to 1ch
    print(f"Loading pretrained SwinIR from {args.pretrained_weights}")
    model = load_swinir_1ch(args.pretrained_weights, device=str(device))
    model.train()

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "conv_first" not in name and "conv_last" not in name:
                param.requires_grad = False

    n_params = sum(p_.numel() for p_ in model.parameters())
    n_trainable = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Verify output shape
    with torch.no_grad():
        test_in = torch.randn(1, 1, lr_train.shape[-2], lr_train.shape[-1], device=device)
        test_out = model(test_in)
        print(f"  Output shape: {test_in.shape} -> {test_out.shape}")
        expected_h = lr_train.shape[-2] * upsampling_factor
        expected_w = lr_train.shape[-1] * upsampling_factor
        assert test_out.shape == (1, 1, expected_h, expected_w), (
            f"Expected output ({1}, {1}, {expected_h}, {expected_w}), got {test_out.shape}"
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p_ for p_ in model.parameters() if p_.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion: nn.Module = nn.L1Loss() if args.loss_fn == "l1" else nn.MSELoss()
    use_amp = not args.no_amp
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Save directory
    args.save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "loss_fn": args.loss_fn,
        "freeze_backbone": args.freeze_backbone,
        "wall_hours": args.wall_hours,
        "use_amp": use_amp,
        "vmin": vmin,
        "vmax": vmax,
        "upsampling_factor": upsampling_factor,
        "n_params": n_params,
        "n_trainable": n_trainable,
        "pretrained_weights": str(args.pretrained_weights),
    }
    with open(args.save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    best_val_loss = float("inf")
    val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []
    start_time = time.time()
    wall_limit = args.wall_hours * 3600
    epochs_trained = 0

    print(f"\nTraining for up to {args.epochs} epochs ({args.wall_hours}h wall limit)...")
    print(f"  Loss: {args.loss_fn}, LR: {args.lr}, BS: {args.batch_size}, AMP: {use_amp}")
    print(f"  Freeze backbone: {args.freeze_backbone}")

    for epoch in range(args.epochs):
        elapsed = time.time() - start_time
        if elapsed > wall_limit:
            print(f"\nWall time limit reached ({elapsed / 3600:.2f}h). Stopping.")
            break

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for lr_batch, hr_batch in train_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(lr_batch)
                loss = criterion(pred, hr_batch)

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
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    pred = model(lr_batch)
                    val_loss += criterion(pred, hr_batch).item()
                n_val += 1
        val_loss /= n_val
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "vmin": vmin,
                    "vmax": vmax,
                    "config": config,
                },
                args.save_dir / "best_swinir.pt",
            )

        elapsed = time.time() - start_time
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"  Ep {epoch + 1:3d}/{args.epochs} | train: {train_loss:.6f} | val: {val_loss:.6f} | "
            f"best: {best_val_loss:.6f} | lr: {lr_now:.2e} | {elapsed / 60:.1f}min"
        )
        epochs_trained = epoch + 1

    # Save final + losses
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epochs_trained - 1,
            "val_loss": val_loss,
            "vmin": vmin,
            "vmax": vmax,
        },
        args.save_dir / "final_swinir.pt",
    )
    torch.save({"train": train_losses, "val": val_losses}, args.save_dir / "losses.pt")

    elapsed = time.time() - start_time
    print(f"\nTraining complete. {elapsed / 60:.1f} min, {epochs_trained} epochs.")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
