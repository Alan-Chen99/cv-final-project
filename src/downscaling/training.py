"""Flow matching training loop for climate downscaling.

Implements OT-CFM (Optimal Transport Conditional Flow Matching):
  x_t = (1-t)*noise + t*target
  v = target - noise
  loss = MSE(model(x_t, t, cond), v)

Canonical recipe (from research3/spatial-4x-flow-matching experiments):
  - AdamW, lr=1e-4, weight_decay=1e-5
  - Cosine annealing LR schedule
  - AMP (mixed precision) for ~2x throughput
  - Gradient clipping at 1.0
  - Uniform timestep sampling (logit-normal and EMA proven harmful at <50 epochs)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from downscaling.sampling import sample_timesteps_logit_normal, sample_timesteps_uniform


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    amp: bool = True
    save_dir: str = "models"
    t_sampling: str = "uniform"  # "uniform" or "logit_normal"
    t_logit_mean: float = 0.0
    t_logit_std: float = 1.0


@dataclass
class TrainResult:
    """Training results."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_time_min: float = 0.0


def _sample_t(config: TrainConfig, batch_size: int, device: torch.device) -> torch.Tensor:
    if config.t_sampling == "logit_normal":
        return sample_timesteps_logit_normal(
            batch_size, device, config.t_logit_mean, config.t_logit_std
        )
    return sample_timesteps_uniform(batch_size, device)


def train_flow_matching(  # pragma: no cover
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: str | torch.device = "cuda",
    model_args: dict | None = None,
) -> TrainResult:
    """Train a flow matching model using OT-CFM.

    The model predicts velocity v(x_t, t, cond) where:
      x_t = (1-t)*noise + t*residual  (linear interpolation)
      v = residual - noise            (target velocity)

    DataLoaders should yield (lr_condition, residual_target) pairs,
    both z-score normalized. Use data.make_dataloaders() to create them.

    Saves best checkpoint to config.save_dir/best_flow.pt with keys:
      model, optimizer, epoch, val_loss, args (architecture params for reload).

    Args:
        model: Velocity field network (e.g. AttentionUNet)
        train_loader: Normalized (lr_cond, residual) pairs
        val_loader: Normalized (lr_cond, residual) pairs
        config: Training hyperparameters
        device: Torch device
        model_args: Architecture args dict saved in checkpoint for reload via
            load_flow_checkpoint(). Should include base_channels, channel_mults_tuple, etc.

    Returns:
        TrainResult with per-epoch losses and timing.
    """
    device = torch.device(device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    result = TrainResult()
    start_time = time.time()

    for epoch in range(config.epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for lr_batch, res_batch in train_loader:
            lr_batch = lr_batch.to(device)
            res_batch = res_batch.to(device)
            bs = lr_batch.shape[0]

            t = _sample_t(config, bs, device)
            noise = torch.randn_like(res_batch)
            t_expand = t[:, None, None, None]
            x_t = (1 - t_expand) * noise + t_expand * res_batch
            target_v = res_batch - noise

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=config.amp):
                pred_v = model(x_t, t, lr_batch)
                loss = F.mse_loss(pred_v, target_v)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        scheduler.step()

        # --- Validate (always uniform t for comparability) ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for lr_batch, res_batch in val_loader:
                lr_batch = lr_batch.to(device)
                res_batch = res_batch.to(device)
                bs = lr_batch.shape[0]

                t = sample_timesteps_uniform(bs, device)
                noise = torch.randn_like(res_batch)
                t_expand = t[:, None, None, None]
                x_t = (1 - t_expand) * noise + t_expand * res_batch
                target_v = res_batch - noise

                with torch.amp.autocast("cuda", enabled=config.amp):
                    pred_v = model(x_t, t, lr_batch)
                    val_loss_sum += F.mse_loss(pred_v, target_v).item()

        val_loss = val_loss_sum / len(val_loader)

        result.train_losses.append(train_loss)
        result.val_losses.append(val_loss)

        # Save best model
        if val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            result.best_epoch = epoch
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }
            if model_args is not None:
                ckpt["args"] = model_args
            torch.save(ckpt, save_dir / "best_flow.pt")

    result.total_time_min = (time.time() - start_time) / 60.0
    return result


def train_step(
    model: nn.Module,
    lr_batch: torch.Tensor,
    res_batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
) -> float:
    """Single OT-CFM training step. Returns loss value.

    Useful for custom training loops or integration tests.
    Inputs must already be on the correct device.
    """
    device = lr_batch.device
    bs = lr_batch.shape[0]

    t = _sample_t(config, bs, device)
    noise = torch.randn_like(res_batch)
    t_expand = t[:, None, None, None]
    x_t = (1 - t_expand) * noise + t_expand * res_batch
    target_v = res_batch - noise

    optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=config.amp):
        pred_v = model(x_t, t, lr_batch)
        loss = F.mse_loss(pred_v, target_v)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    scaler.step(optimizer)
    scaler.update()

    return loss.item()
