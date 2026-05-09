"""Hard constraint layers for conservation enforcement.

From Harder et al. (2208.05424): architectural enforcement of conservation
between LR and HR fields. These are post-hoc projections applied to any
generator's output.

- AddCL: Additive correction. avgpool(output) == lr_orig exactly.
- SmCL: Softmax constraint. Non-negativity AND conservation.
         Requires model to output in log-space (not compatible with
         flow matching residual predictions).
"""

import torch
import torch.nn as nn


def apply_addcl(
    pred_hr: torch.Tensor,
    lr_orig: torch.Tensor,
    upsampling_factor: int = 4,
) -> torch.Tensor:
    """Additive constraint layer: avgpool(output) == lr_orig.

    Adds a spatially-uniform correction within each super-pixel block
    so that the block mean equals the corresponding LR pixel.

    Args:
        pred_hr: Predicted HR field, shape (B, 1, H, W).
        lr_orig: Original LR field, shape (B, 1, H/f, W/f).
        upsampling_factor: Super-resolution factor (default 4).

    Returns:
        Corrected HR field with exact conservation.
    """
    pool = nn.AvgPool2d(kernel_size=upsampling_factor)
    pooled = pool(pred_hr)
    correction = lr_orig - pooled
    correction_hr = correction.repeat_interleave(upsampling_factor, dim=-2).repeat_interleave(
        upsampling_factor, dim=-1
    )
    return pred_hr + correction_hr


def apply_smcl(
    pred_hr: torch.Tensor,
    lr_orig: torch.Tensor,
    upsampling_factor: int = 4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Softmax constraint layer: non-negativity + conservation.

    Applies exp() for non-negativity, then multiplicative renormalization
    so block means equal LR values. Only compatible with models that output
    log-space predictions; NOT compatible with flow matching residuals.

    Args:
        pred_hr: Predicted HR field in log-space, shape (B, 1, H, W).
        lr_orig: Original LR field, shape (B, 1, H/f, W/f).
        upsampling_factor: Super-resolution factor (default 4).
        eps: Small constant for numerical stability.

    Returns:
        Non-negative HR field with exact conservation.
    """
    pool = nn.AvgPool2d(kernel_size=upsampling_factor)
    y = torch.exp(pred_hr)
    pooled = pool(y)
    correction = lr_orig / (pooled + eps)
    correction_hr = correction.repeat_interleave(upsampling_factor, dim=-2).repeat_interleave(
        upsampling_factor, dim=-1
    )
    return y * correction_hr
