#!/usr/bin/env python3
"""Train a flow matching model on a specified dataset.

Usage:
    python scripts/train_flow.py --dataset noresm --save-dir /path/to/save
    python scripts/train_flow.py --dataset era5 --base-channels 96 --amp
"""

import argparse
import sys

import torch

from downscaling.models.unet import AttentionUNet
from downscaling.training.flow_matching import FlowMatchingTrainer, TrainConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Train flow matching model")
    p.add_argument("--dataset", required=True, choices=["era5", "noresm"])
    p.add_argument("--data-dir", default="/home/chenxy/orcd/pool/datasets")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--channel-mults", default="1,2,4")
    p.add_argument("--attn-heads", type=int, default=4)
    p.add_argument("--t-sampling", default="uniform", choices=["uniform", "logit_normal"])
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    channel_mults = tuple(int(x) for x in args.channel_mults.split(","))

    config = TrainConfig(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        base_channels=args.base_channels,
        channel_mults=channel_mults,
        attn_heads=args.attn_heads,
        t_sampling=args.t_sampling,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        amp=args.amp,
        resume=args.resume,
    )

    model = AttentionUNet(
        base_channels=config.base_channels,
        channel_mults=config.channel_mults,
        time_emb_dim=config.time_emb_dim,
        dropout=config.dropout,
        attn_heads=config.attn_heads,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Dataset: {config.dataset}")
    print(f"Model params: {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"Save dir: {config.save_dir}")
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}")

    trainer = FlowMatchingTrainer(model, config)
    best_val = trainer.train()

    print(f"\nDone. Best val loss: {best_val:.6f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
