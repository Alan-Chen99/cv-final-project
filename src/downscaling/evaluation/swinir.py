"""SwinIR evaluation for ERA5 TCW 4x downscaling.

Supports zero-shot (pretrained on DF2K) and finetuned evaluation.
Zero-shot uses per-sample normalization to [0,1] with 3-channel replication.
Finetuned uses global min-max normalization from training data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from downscaling.constraints.layers import apply_addcl
from downscaling.metrics.crps import crps_energy

if TYPE_CHECKING:
    from pathlib import Path


def load_swinir_pretrained(weights_path: str | Path, device: str = "cpu") -> nn.Module:
    """Load pretrained SwinIR classical SR x4 model (3-channel RGB).

    Args:
        weights_path: Path to 001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth
        device: Target device.

    Returns:
        SwinIR model in eval mode.
    """
    from spandrel import ModelLoader

    model_desc = ModelLoader().load_from_file(str(weights_path))
    model = model_desc.model.to(device).eval()
    return model


def load_swinir_1ch(weights_path: str | Path, device: str = "cpu") -> nn.Module:
    """Load pretrained SwinIR and adapt from 3ch to 1ch input/output.

    Channel adaptation:
    - Input conv: average over 3 input channels -> 1 input channel
    - Output conv: average over 3 output channels -> 1 output channel
    - mean buffer: averaged to 1ch, then zeroed (external normalization)
    - img_range: set to 1.0

    Args:
        weights_path: Path to pretrained SwinIR weights.
        device: Target device.

    Returns:
        Adapted 1-channel SwinIR model.
    """
    from spandrel import ModelLoader

    model_desc = ModelLoader().load_from_file(str(weights_path))
    # spandrel returns a dynamic model type; cast to Any for attribute access
    model: Any = model_desc.model

    # Fix mean buffer: (1,3,1,1) -> (1,1,1,1)
    if hasattr(model, "mean"):
        model.mean = torch.zeros(1, 1, 1, 1)

    # Adapt input conv: (out_ch, 3, kh, kw) -> (out_ch, 1, kh, kw)
    old_conv_first: nn.Conv2d = model.conv_first
    kh_in, kw_in = old_conv_first.kernel_size
    pad_in: tuple[int, int] = (int(old_conv_first.padding[0]), int(old_conv_first.padding[1]))
    new_conv_first = nn.Conv2d(1, old_conv_first.out_channels, (kh_in, kw_in), padding=pad_in)
    with torch.no_grad():
        new_conv_first.weight.copy_(old_conv_first.weight.mean(dim=1, keepdim=True))
        assert new_conv_first.bias is not None
        assert old_conv_first.bias is not None
        new_conv_first.bias.copy_(old_conv_first.bias)
    model.conv_first = new_conv_first

    # Adapt output conv: (in_ch, 3, kh, kw) -> (in_ch, 1, kh, kw)
    old_conv_last: nn.Conv2d = model.conv_last
    kh_out, kw_out = old_conv_last.kernel_size
    pad_out: tuple[int, int] = (int(old_conv_last.padding[0]), int(old_conv_last.padding[1]))
    new_conv_last = nn.Conv2d(old_conv_last.in_channels, 1, (kh_out, kw_out), padding=pad_out)
    with torch.no_grad():
        new_conv_last.weight.copy_(old_conv_last.weight.mean(dim=0, keepdim=True))
        if old_conv_last.bias is not None:
            assert new_conv_last.bias is not None
            new_conv_last.bias.copy_(old_conv_last.bias.mean(dim=0, keepdim=True))
    model.conv_last = new_conv_last

    if hasattr(model, "img_range"):
        model.img_range = 1.0

    model = model.to(device).eval()
    return model


def load_swinir_finetuned(
    pretrained_weights_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[nn.Module, float, float]:
    """Load finetuned SwinIR from checkpoint.

    Args:
        pretrained_weights_path: Path to original pretrained weights (for architecture).
        checkpoint_path: Path to finetuned checkpoint (best_swinir.pt).
        device: Target device.

    Returns:
        Tuple of (model, vmin, vmax) where vmin/vmax are normalization stats.
    """
    model = load_swinir_1ch(pretrained_weights_path, device="cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model, ckpt["vmin"], ckpt["vmax"]


def predict_swinir_zeroshot(
    model: nn.Module,
    lr_orig: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Run zero-shot SwinIR (3ch pretrained) on 1ch climate data.

    Uses per-sample normalization to [0,1], replicates to 3 channels,
    runs inference, averages output channels, and denormalizes.
    Processes one sample at a time due to per-sample normalization.

    Args:
        model: Pretrained 3ch SwinIR model.
        lr_orig: LR input, shape (N, 1, 32, 32).
        device: Inference device.

    Returns:
        HR predictions, shape (N, 1, 128, 128).
    """
    N = lr_orig.shape[0]
    results = []

    with torch.no_grad():
        for i in range(N):
            x = lr_orig[i, 0]  # (32, 32)
            x_min = x.min()
            x_max = x.max()
            if x_max - x_min < 1e-8:
                x_norm = torch.zeros_like(x)
            else:
                x_norm = (x - x_min) / (x_max - x_min)

            # Replicate to 3 channels
            x_3ch = x_norm.unsqueeze(0).expand(3, -1, -1)  # (3, 32, 32)
            x_3ch = x_3ch.unsqueeze(0).to(device)  # (1, 3, 32, 32)

            out = model(x_3ch)  # (1, 3, 128, 128)
            out_1ch = out[0].mean(dim=0)  # (128, 128)

            # Denormalize back to physical range
            out_1ch = out_1ch.clamp(0, 1) * (x_max - x_min) + x_min
            results.append(out_1ch.cpu())

    return torch.stack(results).unsqueeze(1)  # (N, 1, 128, 128)


def predict_swinir_finetuned(
    model: nn.Module,
    lr_orig: torch.Tensor,
    vmin: float,
    vmax: float,
    device: str = "cpu",
    batch_size: int = 64,
) -> torch.Tensor:
    """Run finetuned 1ch SwinIR on climate data.

    Uses global min-max normalization from training data stats.

    Args:
        model: Finetuned 1ch SwinIR model.
        lr_orig: LR input, shape (N, 1, 32, 32).
        vmin: Global min from training data.
        vmax: Global max from training data.
        device: Inference device.
        batch_size: Batch size for inference.

    Returns:
        HR predictions, shape (N, 1, 128, 128) in physical units.
    """
    N = lr_orig.shape[0]
    # Normalize to [0, 1]
    lr_norm = (lr_orig - vmin) / (vmax - vmin + 1e-8)
    results = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = lr_norm[start:end].to(device)
            pred = model(batch)  # (B, 1, 128, 128) normalized
            results.append(pred.cpu())

    preds = torch.cat(results, dim=0)  # (N, 1, 128, 128) in [0,1]
    # Denormalize
    return preds * (vmax - vmin) + vmin


def eval_swinir_zeroshot(
    hr: torch.Tensor,
    lr_orig: torch.Tensor,
    weights_path: str | Path,
    device: str = "cpu",
    with_addcl: bool = False,
    upsampling_factor: int = 4,
) -> dict[str, float]:
    """Evaluate zero-shot SwinIR on test data.

    Args:
        hr: Ground truth HR, shape (N, 1, 128, 128).
        lr_orig: Original LR, shape (N, 1, 32, 32).
        weights_path: Path to pretrained SwinIR weights.
        device: Inference device.
        with_addcl: Whether to apply AddCL constraint.
        upsampling_factor: SR factor.

    Returns:
        Dict with crps, mae, rmse, mass_violation.
    """
    model = load_swinir_pretrained(weights_path, device)
    preds = predict_swinir_zeroshot(model, lr_orig, device)
    if with_addcl:
        preds = apply_addcl(preds, lr_orig, upsampling_factor)
    del model
    torch.cuda.empty_cache()
    return _eval_deterministic(preds, hr, lr_orig, upsampling_factor)


def eval_swinir_finetuned(
    hr: torch.Tensor,
    lr_orig: torch.Tensor,
    pretrained_weights_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "cpu",
    with_addcl: bool = False,
    upsampling_factor: int = 4,
) -> dict[str, float]:
    """Evaluate finetuned SwinIR on test data.

    Args:
        hr: Ground truth HR, shape (N, 1, 128, 128).
        lr_orig: Original LR, shape (N, 1, 32, 32).
        pretrained_weights_path: Path to original pretrained weights.
        checkpoint_path: Path to finetuned checkpoint.
        device: Inference device.
        with_addcl: Whether to apply AddCL constraint.
        upsampling_factor: SR factor.

    Returns:
        Dict with crps, mae, rmse, mass_violation.
    """
    model, vmin, vmax = load_swinir_finetuned(pretrained_weights_path, checkpoint_path, device)
    preds = predict_swinir_finetuned(model, lr_orig, vmin, vmax, device)
    if with_addcl:
        preds = apply_addcl(preds, lr_orig, upsampling_factor)
    del model
    torch.cuda.empty_cache()
    return _eval_deterministic(preds, hr, lr_orig, upsampling_factor)


def _eval_deterministic(
    preds: torch.Tensor,
    hr: torch.Tensor,
    lr_orig: torch.Tensor,
    upsampling_factor: int = 4,
) -> dict[str, float]:
    """Evaluate deterministic predictions. CRPS = MAE for single member."""
    pool = nn.AvgPool2d(kernel_size=upsampling_factor)
    N = preds.shape[0]

    crps_sum = 0.0
    mae_sum = 0.0
    rmse_sq_sum = 0.0
    mass_viol_sum = 0.0

    for i in range(N):
        pred = preds[i, 0].numpy()
        gt = hr[i, 0].numpy()
        crps_sum += crps_energy(gt, pred[None, ...])
        mae_sum += float(np.mean(np.abs(gt - pred)))
        rmse_sq_sum += float(np.mean((gt - pred) ** 2))
        pooled = pool(preds[i : i + 1]).squeeze()
        lr_i = lr_orig[i, 0]
        mass_viol_sum += float(torch.mean(torch.abs(pooled - lr_i)).item())

    return {
        "crps": crps_sum / N,
        "mae": mae_sum / N,
        "rmse": float(np.sqrt(rmse_sq_sum / N)),
        "mass_violation": mass_viol_sum / N,
    }
