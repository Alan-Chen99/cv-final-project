"""SwinIR finetuning for ERA5 TCW 4x downscaling.

Adapts pretrained SwinIR (DF2K, 3ch RGB -> 1ch climate) and finetunes
on ERA5 TCW data with global min-max normalization to [0,1].

Usage:
    # Train with defaults (L1 loss, full backbone, AMP, 2hr wall limit)
    python -m downscaling.training.swinir --pool-dir /path/to/pool

    # Hyperparameter sweep (short runs for tuning)
    python -m downscaling.training.swinir --pool-dir /path --epochs 5 --tag sweep-lr1e4 --lr 1e-4

    # Evaluate existing checkpoint
    python -m downscaling.training.swinir --mode eval --pool-dir /path
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from downscaling.evaluation.swinir import load_swinir_1ch


def _load_splits(
    data_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load train and val splits as (lr, hr) pairs."""
    lr_train = torch.load(
        data_dir / "era5_sr_data" / "train" / "input_train.pt",
        map_location="cpu",
        weights_only=True,
    )[:, 0, :, :, :]  # (N, 1, 32, 32)
    hr_train = torch.load(
        data_dir / "era5_sr_data" / "train" / "target_train.pt",
        map_location="cpu",
        weights_only=True,
    )[:, 0, :, :, :]  # (N, 1, 128, 128)
    lr_val = torch.load(
        data_dir / "era5_sr_data" / "val" / "input_val.pt",
        map_location="cpu",
        weights_only=True,
    )[:, 0, :, :, :]
    hr_val = torch.load(
        data_dir / "era5_sr_data" / "val" / "target_val.pt",
        map_location="cpu",
        weights_only=True,
    )[:, 0, :, :, :]
    return lr_train, hr_train, lr_val, hr_val


def train_swinir(
    pool_dir: Path,
    save_dir: Path,
    pretrained_weights: Path,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    loss_fn: str = "l1",
    freeze_backbone: bool = False,
    wall_hours: float = 2.0,
    use_amp: bool = True,
    tag: str = "",
) -> dict[str, object]:
    """Finetune SwinIR on ERA5 TCW data.

    Returns:
        Dict with training results (best_val_loss, epochs_trained, etc).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("Loading data...")
    lr_train, hr_train, lr_val, hr_val = _load_splits(pool_dir)

    # Compute global normalization from training data
    all_data = torch.cat([lr_train.flatten(), hr_train.flatten()])
    vmin = all_data.min().item()
    vmax = all_data.max().item()
    print(f"  Normalization range: [{vmin:.4f}, {vmax:.4f}]")

    # Normalize to [0, 1]
    def normalize(x: torch.Tensor) -> torch.Tensor:
        return (x - vmin) / (vmax - vmin + 1e-8)

    lr_train_n = normalize(lr_train)
    hr_train_n = normalize(hr_train)
    lr_val_n = normalize(lr_val)
    hr_val_n = normalize(hr_val)

    train_loader = DataLoader(
        TensorDataset(lr_train_n, hr_train_n),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(lr_val_n, hr_val_n),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = load_swinir_1ch(pretrained_weights, device=str(device))
    model.train()

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "conv_first" not in name and "conv_last" not in name:
                param.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion: nn.Module = nn.L1Loss() if loss_fn == "l1" else nn.MSELoss()
    # Use bfloat16 for AMP - SwinIR attention overflows in float16
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Save directory
    actual_save_dir = save_dir / tag if tag else save_dir
    actual_save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "loss_fn": loss_fn,
        "freeze_backbone": freeze_backbone,
        "wall_hours": wall_hours,
        "use_amp": use_amp,
        "tag": tag,
        "vmin": vmin,
        "vmax": vmax,
        "n_params": n_params,
        "n_trainable": n_trainable,
    }
    with open(actual_save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    best_val_loss = float("inf")
    val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []
    start_time = time.time()
    wall_limit = wall_hours * 3600
    epochs_trained = 0

    print(f"\nTraining for up to {epochs} epochs ({wall_hours}h wall limit)...")
    print(f"  Loss: {loss_fn}, LR: {lr}, BS: {batch_size}, AMP: {use_amp}")
    print(f"  Freeze backbone: {freeze_backbone}")

    for epoch in range(epochs):
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
                    "optimizer": optimizer.state_dict(),
                    "vmin": vmin,
                    "vmax": vmax,
                    "config": config,
                },
                actual_save_dir / "best_swinir.pt",
            )

        elapsed = time.time() - start_time
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"  Ep {epoch + 1:3d}/{epochs} | train: {train_loss:.6f} | val: {val_loss:.6f} | "
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
        actual_save_dir / "final_swinir.pt",
    )
    torch.save({"train": train_losses, "val": val_losses}, actual_save_dir / "losses.pt")

    elapsed = time.time() - start_time
    print(f"\nTraining complete. {elapsed / 60:.1f} min, {epochs_trained} epochs.")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Saved to: {actual_save_dir}")

    return {
        "best_val_loss": best_val_loss,
        "epochs_trained": epochs_trained,
        "elapsed_min": elapsed / 60,
        "save_dir": str(actual_save_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune SwinIR for ERA5 TCW 4x SR")
    parser.add_argument("--pool-dir", type=Path, default=Path("/home/chenxy/orcd/pool/datasets"))
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("/home/chenxy/orcd/pool/datasets/spatial-4x-add-v2/models/swinir_ft"),
    )
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=Path(
            "/home/chenxy/orcd/pool/datasets/research5/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-fn", default="l1", choices=["l1", "mse"])
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--wall-hours", type=float, default=2.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tag", default="", help="Subdirectory tag for sweep runs")
    args = parser.parse_args()

    train_swinir(
        pool_dir=args.pool_dir,
        save_dir=args.save_dir,
        pretrained_weights=args.pretrained_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_fn=args.loss_fn,
        freeze_backbone=args.freeze_backbone,
        wall_hours=args.wall_hours,
        use_amp=not args.no_amp,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
