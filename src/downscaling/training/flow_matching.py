"""OT-CFM (Optimal Transport Conditional Flow Matching) training loop.

Trains a velocity network v(x_t, t, condition) to predict the velocity field
for transporting noise (t=0) to residual targets (t=1). Uses z-score normalization
on residuals for proper noise-to-signal ratio matching.
"""

import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from downscaling.data.era5 import load_era5_tcw
from downscaling.sampling.timesteps import sample_timesteps_logit_normal, sample_timesteps_uniform
from downscaling.training.ema import EMA


@dataclass
class TrainConfig:
    data_dir: str = ""
    save_dir: str = "models/flow"
    batch_size: int = 64
    epochs: int = 40
    lr: float = 1e-4
    base_channels: int = 64
    channel_mults: tuple[int, ...] = (1, 2, 4)
    attn_heads: int = 4
    time_emb_dim: int = 256
    dropout: float = 0.1
    t_sampling: str = "uniform"
    t_logit_mean: float = 0.0
    t_logit_std: float = 1.0
    use_ema: bool = False
    ema_decay: float = 0.9999
    amp: bool = False
    resume: bool = False


class FlowMatchingTrainer:
    """OT-CFM training with z-score normalization and optional AMP/EMA."""

    def __init__(self, model: nn.Module, config: TrainConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
        self.ema = EMA(model, decay=config.ema_decay) if config.use_ema else None
        self.best_val_loss = float("inf")
        self.start_epoch = 0

    def _sample_t(self, batch_size: int) -> torch.Tensor:
        if self.config.t_sampling == "logit_normal":
            return sample_timesteps_logit_normal(
                batch_size, self.device, self.config.t_logit_mean, self.config.t_logit_std
            )
        return sample_timesteps_uniform(batch_size, self.device)

    def _compute_normalization(
        self,
        lr_up_train: torch.Tensor,
        res_train: torch.Tensor,
    ) -> dict[str, float]:
        stats = {
            "res_mean": res_train.mean().item(),
            "res_std": res_train.std().item(),
            "lr_mean": lr_up_train.mean().item(),
            "lr_std": lr_up_train.std().item(),
        }
        os.makedirs(self.config.save_dir, exist_ok=True)
        torch.save(stats, os.path.join(self.config.save_dir, "norm_stats.pt"))
        return stats

    def train(self) -> float:
        """Run full training loop. Returns best validation loss."""
        cfg = self.config

        lr_up_train, res_train, _, _ = load_era5_tcw(cfg.data_dir, "train")
        lr_up_val, res_val, _, _ = load_era5_tcw(cfg.data_dir, "val")

        stats = self._compute_normalization(lr_up_train, res_train)

        res_train_norm = (res_train - stats["res_mean"]) / stats["res_std"]
        res_val_norm = (res_val - stats["res_mean"]) / stats["res_std"]
        lr_up_train_norm = (lr_up_train - stats["lr_mean"]) / stats["lr_std"]
        lr_up_val_norm = (lr_up_val - stats["lr_mean"]) / stats["lr_std"]

        train_loader = DataLoader(
            TensorDataset(lr_up_train_norm, res_train_norm),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            TensorDataset(lr_up_val_norm, res_val_norm),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        if cfg.resume:
            self._resume_checkpoint()

        start_time = time.time()

        for epoch in range(self.start_epoch, cfg.epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)
            self.scheduler.step()

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{cfg.epochs}, "
                f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
                f"LR: {self.scheduler.get_last_lr()[0]:.6f}, "
                f"Time: {elapsed / 60:.1f}min"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)

        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.6f}")
        print(f"Total time: {(time.time() - start_time) / 60:.1f} min")
        return self.best_val_loss

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for lr_batch, res_batch in loader:
            lr_batch = lr_batch.to(self.device)
            res_batch = res_batch.to(self.device)

            bs = lr_batch.shape[0]
            t = self._sample_t(bs)
            x_0 = torch.randn_like(res_batch)
            t_expand = t[:, None, None, None]
            x_t = (1 - t_expand) * x_0 + t_expand * res_batch
            target_v = res_batch - x_0

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=self.config.amp):
                pred_v = self.model(x_t, t, lr_batch)
                loss = F.mse_loss(pred_v, target_v)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update(self.model)

            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for lr_batch, res_batch in loader:
                lr_batch = lr_batch.to(self.device)
                res_batch = res_batch.to(self.device)
                bs = lr_batch.shape[0]
                t = self._sample_t(bs)
                x_0 = torch.randn_like(res_batch)
                t_expand = t[:, None, None, None]
                x_t = (1 - t_expand) * x_0 + t_expand * res_batch
                target_v = res_batch - x_0
                with torch.amp.autocast("cuda", enabled=self.config.amp):
                    pred_v = self.model(x_t, t, lr_batch)
                    total_loss += F.mse_loss(pred_v, target_v).item()
        return total_loss / len(loader)

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        save_dict: dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }
        if self.ema is not None:
            save_dict["ema"] = self.ema.state_dict()
        torch.save(save_dict, os.path.join(self.config.save_dir, "best_flow.pt"))

    def _resume_checkpoint(self) -> None:
        ckpt_path = os.path.join(self.config.save_dir, "best_flow.pt")
        if not os.path.exists(ckpt_path):
            return
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_loss = ckpt["val_loss"]
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        for _ in range(self.start_epoch):
            self.scheduler.step()
        if self.ema is not None and "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])
        print(f"Resumed from epoch {self.start_epoch}, best val loss: {self.best_val_loss:.6f}")
