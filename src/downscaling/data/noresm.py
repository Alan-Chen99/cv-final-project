"""NorESM surface temperature (tas) data loading for 2x super-resolution.

Dataset: 32x32 -> 64x64 spatial SR, single channel (surface temperature).
Source: Harder et al. (2208.05424) preprocessed NorESM2 data.
Splits: ~24K train / ~12K val / ~12K test.

NorESM LR/HR come from separate simulation runs (NorESM2-LM at 2deg,
NorESM2-MM at 1deg), so the downscaling constraint avgpool(HR) ~= LR
is only approximately satisfied (violation ~1.8K). Additive constraints
applied post-hoc may degrade metrics.
"""

from pathlib import Path

import torch
import torch.nn.functional as F


def load_noresm_tas(
    data_dir: str | Path,
    split: str = "train",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load NorESM surface temperature 2x super-resolution data.

    Args:
        data_dir: Path to pool directory containing noresm-dataset/noresm/.
        split: One of 'train', 'val', 'test'.

    Returns:
        Tuple of (lr_up, residual, hr, lr_orig) where:
            lr_up: Bilinear-upsampled LR field, shape (N, 1, 64, 64).
            residual: HR - lr_up, shape (N, 1, 64, 64).
            hr: High-resolution ground truth, shape (N, 1, 64, 64).
            lr_orig: Original low-resolution field, shape (N, 1, 32, 32).
    """
    data_dir = Path(data_dir)
    noresm_dir = data_dir / "noresm-dataset" / "noresm"

    inp = torch.load(noresm_dir / f"input_{split}.pt", weights_only=False)
    tgt = torch.load(noresm_dir / f"target_{split}.pt", weights_only=False)

    lr: torch.Tensor = inp[:, 0, :, :, :]  # (N, 1, 32, 32)
    hr: torch.Tensor = tgt[:, 0, :, :, :]  # (N, 1, 64, 64)

    hr_h, hr_w = hr.shape[2], hr.shape[3]
    lr_up = F.interpolate(lr, size=(hr_h, hr_w), mode="bilinear", align_corners=False)
    residual = hr - lr_up

    return lr_up, residual, hr, lr
