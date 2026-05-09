"""Checkpoint loading utilities for trained models.

Two checkpoint patterns exist across experiments:

Pattern A (research3, research6/flow_v2_zscore):
    Full training checkpoint dict with keys: model, optimizer, epoch, val_loss, args
    Load with: state_dict = checkpoint['model']

Pattern B (research5/residual_flow, research6/flow_v2):
    Pure OrderedDict state_dict (model weights only).
    Load directly: model.load_state_dict(checkpoint)

Norm stats are stored separately in norm_stats.pt:
    Dict with keys: res_mean, res_std, lr_mean, lr_std
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pathlib import Path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Load model weights from checkpoint, handling both patterns.

    Args:
        model: Model to load weights into.
        checkpoint_path: Path to .pt/.pth file.
        device: Device to map tensors to.

    Returns:
        Dict with metadata (epoch, val_loss) if available, empty dict otherwise.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata: dict[str, object] = {}

    if isinstance(ckpt, dict) and "model" in ckpt:
        # Pattern A: full checkpoint
        model.load_state_dict(ckpt["model"])
        if "epoch" in ckpt:
            metadata["epoch"] = ckpt["epoch"]
        if "val_loss" in ckpt:
            metadata["val_loss"] = ckpt["val_loss"]
        if "args" in ckpt:
            metadata["args"] = ckpt["args"]
    else:
        # Pattern B: state_dict only
        model.load_state_dict(ckpt)

    model.eval()
    return metadata


def load_norm_stats(
    stats_path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Load normalization statistics from norm_stats.pt.

    Args:
        stats_path: Path to norm_stats.pt file.
        device: Device to map tensors to.

    Returns:
        Dict with res_mean, res_std, lr_mean, lr_std as floats.
    """
    raw = torch.load(stats_path, map_location=device, weights_only=False)
    return {
        "res_mean": float(raw["res_mean"]),
        "res_std": float(raw["res_std"]),
        "lr_mean": float(raw["lr_mean"]),
        "lr_std": float(raw["lr_std"]),
    }
