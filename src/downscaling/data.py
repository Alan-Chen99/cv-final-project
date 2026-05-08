"""ERA5 data loading for TCW 4x super-resolution."""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class NormStats:
    """Z-score normalization statistics for residuals and LR condition."""

    res_mean: float
    res_std: float
    lr_mean: float
    lr_std: float

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "res_mean": self.res_mean,
                "res_std": self.res_std,
                "lr_mean": self.lr_mean,
                "lr_std": self.lr_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> NormStats:
        d = torch.load(path, weights_only=False, map_location=device)
        return cls(
            res_mean=d["res_mean"],
            res_std=d["res_std"],
            lr_mean=d["lr_mean"],
            lr_std=d["lr_std"],
        )


def load_era5_tcw4(  # pragma: no cover
    basedir: str | Path, split: str = "train"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load ERA5 TCW 4x SR data.

    Args:
        basedir: Root directory containing data/era5_sr_data/
        split: One of 'train', 'val', 'test'

    Returns:
        (lr_up, residual, hr, lr_orig) where:
        - lr_up: bilinear-upsampled LR (N, 1, 128, 128)
        - residual: hr - lr_up (N, 1, 128, 128)
        - hr: high-resolution target (N, 1, 128, 128)
        - lr_orig: original low-resolution (N, 1, 32, 32)
    """
    basedir = Path(basedir)
    inp = torch.load(
        basedir / "data/era5_sr_data" / split / f"input_{split}.pt", weights_only=False
    )
    tgt = torch.load(
        basedir / "data/era5_sr_data" / split / f"target_{split}.pt", weights_only=False
    )

    lr = inp[:, 0, :, :, :]  # (N, 1, 32, 32)
    hr = tgt[:, 0, :, :, :]  # (N, 1, 128, 128)
    lr_up = F.interpolate(lr, size=(128, 128), mode="bilinear", align_corners=False)
    residual = hr - lr_up

    return lr_up, residual, hr, lr


def compute_norm_stats(lr_up_train: torch.Tensor, res_train: torch.Tensor) -> NormStats:
    """Compute z-score normalization statistics from training data."""
    return NormStats(
        res_mean=res_train.mean().item(),
        res_std=res_train.std().item(),
        lr_mean=lr_up_train.mean().item(),
        lr_std=lr_up_train.std().item(),
    )


def normalize(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Z-score normalize."""
    return (data - mean) / std


def denormalize(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Inverse z-score normalize."""
    return data * std + mean


def make_dataloaders(  # pragma: no cover
    lr_up_train: torch.Tensor,
    res_train: torch.Tensor,
    lr_up_val: torch.Tensor,
    res_val: torch.Tensor,
    stats: NormStats,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Create normalized train/val dataloaders."""
    lr_train_norm = normalize(lr_up_train, stats.lr_mean, stats.lr_std)
    res_train_norm = normalize(res_train, stats.res_mean, stats.res_std)
    lr_val_norm = normalize(lr_up_val, stats.lr_mean, stats.lr_std)
    res_val_norm = normalize(res_val, stats.res_mean, stats.res_std)

    train_loader = DataLoader(
        TensorDataset(lr_train_norm, res_train_norm),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(lr_val_norm, res_val_norm),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
