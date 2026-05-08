"""ERA5 Total Column Water (TCW) data loading for 4x super-resolution.

Dataset: 32x32 -> 128x128 spatial SR, single channel (TCW).
Source: Harder et al. (2208.05424) preprocessed ERA5 data.
Splits: 40K train / 10K val / 10K test.
"""

from pathlib import Path

import torch
import torch.nn.functional as F


def load_era5_tcw(
    data_dir: str | Path,
    split: str = "train",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load ERA5 TCW 4x super-resolution data.

    Args:
        data_dir: Path to directory containing era5_sr_data/{split}/ subdirectories.
        split: One of 'train', 'val', 'test'.

    Returns:
        Tuple of (lr_up, residual, hr, lr_orig) where:
            lr_up: Bilinear-upsampled LR field, shape (N, 1, 128, 128).
            residual: HR - lr_up, shape (N, 1, 128, 128).
            hr: High-resolution ground truth, shape (N, 1, 128, 128).
            lr_orig: Original low-resolution field, shape (N, 1, 32, 32).
    """
    data_dir = Path(data_dir)
    split_dir = data_dir / "era5_sr_data" / split

    inp = torch.load(split_dir / f"input_{split}.pt", weights_only=False)
    tgt = torch.load(split_dir / f"target_{split}.pt", weights_only=False)

    lr: torch.Tensor = inp[:, 0, :, :, :]  # (N, 1, 32, 32)
    hr: torch.Tensor = tgt[:, 0, :, :, :]  # (N, 1, 128, 128)
    lr_up = F.interpolate(lr, size=(128, 128), mode="bilinear", align_corners=False)
    residual = hr - lr_up

    return lr_up, residual, hr, lr
