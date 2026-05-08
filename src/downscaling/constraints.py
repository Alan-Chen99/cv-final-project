"""Hard constraint layers for climate downscaling (Harder et al., 2208.05424).

These layers enforce exact mass conservation between HR predictions and LR inputs
by post-processing model outputs. They are architecture-agnostic and differentiable.
"""

import torch


def apply_addcl(
    pred_hr: torch.Tensor, lr_orig: torch.Tensor, upsampling_factor: int = 4
) -> torch.Tensor:
    """Additive Constraint Layer (AddCL).

    Applies a uniform additive correction to each upsampling block so that
    avgpool(pred_hr) == lr_orig exactly.

    Args:
        pred_hr: HR prediction (B, C, H, W)
        lr_orig: Original LR input (B, C, H/f, W/f)
        upsampling_factor: Super-resolution factor

    Returns:
        Constrained prediction with exact mass conservation.
    """
    pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
    pooled = pool(pred_hr)
    correction = lr_orig - pooled
    correction_hr = correction.repeat_interleave(upsampling_factor, dim=-2).repeat_interleave(
        upsampling_factor, dim=-1
    )
    return pred_hr + correction_hr


def apply_smcl(
    pred_hr: torch.Tensor, lr_orig: torch.Tensor, upsampling_factor: int = 4
) -> torch.Tensor:
    """Softmax Constraint Layer (SmCL).

    Applies exp() then multiplicative renormalization. Enforces both
    non-negativity AND mass conservation.

    Warning: Can overflow on physical-space values (e.g., TCW 0-135).
    Works best when pred_hr is in a normalized/log space.

    Args:
        pred_hr: HR prediction (B, C, H, W) — ideally in log-space
        lr_orig: Original LR input (B, C, H/f, W/f)
        upsampling_factor: Super-resolution factor

    Returns:
        Non-negative, mass-conserving prediction.
    """
    pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
    y = torch.exp(pred_hr)
    pooled = pool(y)
    correction = lr_orig / (pooled + 1e-8)
    correction_hr = correction.repeat_interleave(upsampling_factor, dim=-2).repeat_interleave(
        upsampling_factor, dim=-1
    )
    return y * correction_hr
