#!/usr/bin/env python3
"""Train Harder et al. CNN or GAN model on ERA5 or NorESM dataset.

Uses the ResNet architecture from external/constrained-downscaling/models.py
with min-max normalization (matching the original Harder et al. training).

Usage:
    # CNN without constraints on NorESM
    python scripts/train_harder.py --dataset noresm --model cnn --constraints none \
        --save-dir /pool/noresm-dataset/models/harder

    # CNN with softmax constraints
    python scripts/train_harder.py --dataset noresm --model cnn --constraints softmax \
        --save-dir /pool/noresm-dataset/models/harder --model-id harder-cnn-smcl

    # GAN with softmax constraints
    python scripts/train_harder.py --dataset noresm --model gan --constraints softmax \
        --save-dir /pool/noresm-dataset/models/harder --model-id harder-gan-smcl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Harder models live in external/
HARDER_ROOT = Path(__file__).resolve().parents[1] / "external" / "constrained-downscaling"
if str(HARDER_ROOT) not in sys.path:
    sys.path.insert(0, str(HARDER_ROOT))

import models as harder_models  # noqa: E402


def load_dataset(data_dir: Path, dataset: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load input/target tensors for a split. Returns (input, target)."""
    if dataset == "noresm":
        base = data_dir / "noresm-dataset" / "noresm"
        inp = torch.load(base / f"input_{split}.pt", weights_only=False)
        tgt = torch.load(base / f"target_{split}.pt", weights_only=False)
    elif dataset == "era5":
        base = data_dir / "era5_sr_data" / split
        inp = torch.load(base / f"input_{split}.pt", weights_only=False)
        tgt = torch.load(base / f"target_{split}.pt", weights_only=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return inp, tgt


def get_upsampling_factor(dataset: str) -> int:
    if dataset == "noresm":
        return 2
    return 4


def train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    upsampling_factor = get_upsampling_factor(args.dataset)
    is_gan = args.model == "gan"

    # Load data
    print(f"Loading {args.dataset} data...")
    inp_train, tgt_train = load_dataset(data_dir, args.dataset, "train")
    inp_val, tgt_val = load_dataset(data_dir, args.dataset, "val")
    print(f"  Train: {inp_train.shape} -> {tgt_train.shape}")
    print(f"  Val:   {inp_val.shape} -> {tgt_val.shape}")

    # Min-max normalization from training targets (matches Harder approach)
    min_val = float(tgt_train[:, 0, 0, ...].min())
    max_val = float(tgt_train[:, 0, 0, ...].max())
    val_range = max_val - min_val
    print(f"  Min-max range: [{min_val:.2f}, {max_val:.2f}]")

    def normalize(t: torch.Tensor) -> torch.Tensor:
        t = t.clone()
        t[:, 0, 0, ...] = (t[:, 0, 0, ...] - min_val) / val_range
        return t

    inp_train = normalize(inp_train)
    tgt_train = normalize(tgt_train)
    inp_val = normalize(inp_val)
    tgt_val = normalize(tgt_val)

    train_loader = DataLoader(
        TensorDataset(inp_train, tgt_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(inp_val, tgt_val),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Build model
    generator = harder_models.ResNet(
        number_channels=args.number_channels,
        number_residual_blocks=args.number_residual_blocks,
        upsampling_factor=upsampling_factor,
        noise=is_gan,
        constraints=args.constraints,
        dim=1,
    ).cuda()

    n_params = sum(p.numel() for p in generator.parameters())
    print(f"\nModel: {args.model} | constraints: {args.constraints}")
    print(f"  Params: {n_params:,} | upsampling: {upsampling_factor}x")

    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    discriminator = None
    optimizer_d = None
    criterion_d = None
    if is_gan:
        discriminator = harder_models.Discriminator().cuda()
        n_d = sum(p.numel() for p in discriminator.parameters())
        print(f"  Discriminator params: {n_d:,}")
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        criterion_d = nn.BCELoss()

    # Training loop
    best_val = float("inf")
    model_id = args.model_id
    start_time = time.time()

    for epoch in range(args.epochs):
        generator.train()
        if discriminator is not None:
            discriminator.train()

        running_loss = 0.0
        running_d_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            if is_gan:
                assert discriminator is not None
                assert optimizer_d is not None
                assert criterion_d is not None

                bs = inputs.shape[0]
                z = torch.randn(bs, 100, 1, 1, device="cuda")
                fake = generator(inputs, z)

                # Discriminator step
                optimizer_d.zero_grad()
                real_label = torch.ones(bs, 1, device="cuda")
                fake_label = torch.zeros(bs, 1, device="cuda")
                d_real = discriminator(targets)
                d_fake = discriminator(fake.detach())
                d_loss = criterion_d(d_real, real_label) + criterion_d(d_fake, fake_label)
                d_loss.backward()
                optimizer_d.step()
                running_d_loss += d_loss.item()

                # Generator step
                optimizer.zero_grad()
                d_fake_for_g = discriminator(fake)
                g_loss = criterion(fake, targets) + args.adv_factor * criterion_d(
                    d_fake_for_g, real_label
                )
                g_loss.backward()
                optimizer.step()
                running_loss += g_loss.item()
            else:
                optimizer.zero_grad()
                outputs = generator(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation
        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                if is_gan:
                    z = torch.randn(inputs.shape[0], 100, 1, 1, device="cuda")
                    outputs = generator(inputs, z)
                else:
                    outputs = generator(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)

        if is_gan:
            d_avg = running_d_loss / len(train_loader)
            print(
                f"Epoch {epoch + 1:3d}/{args.epochs} | "
                f"train: {train_loss:.6f} | d_loss: {d_avg:.6f} | val: {val_loss:.6f}"
            )
        else:
            print(
                f"Epoch {epoch + 1:3d}/{args.epochs} | "
                f"train: {train_loss:.6f} | val: {val_loss:.6f}"
            )

        # Checkpoint best
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {"model": generator, "state_dict": generator.state_dict()}
            torch.save(ckpt, save_dir / f"{model_id}.pth")
            print(f"  -> Saved best model (val={best_val:.6f})")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.1f} min")
    print(f"Best val loss: {best_val:.6f}")
    print(f"Checkpoint: {save_dir / f'{model_id}.pth'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train Harder et al. CNN/GAN model")
    p.add_argument("--dataset", required=True, choices=["era5", "noresm"])
    p.add_argument("--data-dir", default="/home/chenxy/orcd/pool/datasets")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--model", required=True, choices=["cnn", "gan"])
    p.add_argument("--model-id", required=True, help="Checkpoint filename (without .pth)")
    p.add_argument(
        "--constraints", default="none", choices=["none", "softmax", "add", "scadd", "mult"]
    )
    p.add_argument("--number-channels", type=int, default=32)
    p.add_argument("--number-residual-blocks", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-9)
    p.add_argument("--adv-factor", type=float, default=0.0001)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
