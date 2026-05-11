"""Evaluation of Harder et al. (2208.05424) CNN and GAN baselines.

Loads checkpoints from the constrained-downscaling codebase and evaluates
using the same metrics as our flow matching models (CRPS, MAE, RMSE, mass
violation).

Harder models use min-max normalization (computed from training targets) and
apply constraints internally in forward(). Input format: (B, 1, 1, H, W).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from downscaling.metrics.crps import crps_energy

# Harder code lives at external/constrained-downscaling/
HARDER_ROOT = Path(__file__).resolve().parents[3] / "external" / "constrained-downscaling"


def _get_harder_models_module() -> object:
    """Import Harder et al. models module from external directory."""
    harder_str = str(HARDER_ROOT)
    if harder_str not in sys.path:
        sys.path.insert(0, harder_str)
    import importlib

    return importlib.import_module("models")


def compute_minmax_stats(
    pool_dir: Path,
    dataset: str = "era5",
) -> tuple[float, float]:
    """Compute min-max stats from training targets (matches Harder's normalization).

    Args:
        pool_dir: Pool datasets directory.
        dataset: 'era5' or 'noresm'.

    Returns:
        (min_val, max_val) from training target data.
    """
    if dataset == "noresm":
        tgt_path = pool_dir / "noresm-dataset" / "noresm" / "target_train.pt"
    elif dataset == "era5":
        tgt_path = pool_dir / "era5_sr_data" / "train" / "target_train.pt"
    else:
        raise ValueError(f"Unknown dataset {dataset!r}, expected 'era5' or 'noresm'")

    train_tgt = torch.load(tgt_path, weights_only=False)
    # Harder code: max_val[i] = target_train[:,0,i,...].max() for channel i=0
    max_val = float(train_tgt[:, 0, 0, ...].max())
    min_val = float(train_tgt[:, 0, 0, ...].min())
    return min_val, max_val


def load_harder_model(
    checkpoint_path: Path,
    model_type: str,
    constraints: str,
    device: str | torch.device = "cpu",
    number_channels: int = 32,
    number_residual_blocks: int = 4,
    upsampling_factor: int = 4,
) -> nn.Module:
    """Load a Harder et al. model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        model_type: 'cnn' or 'gan'.
        constraints: 'none', 'softmax', 'add', 'scadd', 'mult'.
        device: Target device.
        number_channels: Hidden channel count (paper default: 32).
        number_residual_blocks: Residual blocks (paper default: 4).
        upsampling_factor: SR factor (4 for ERA5, 2 for NorESM).

    Returns:
        Model in eval mode on target device.
    """
    harder_models = _get_harder_models_module()

    model = harder_models.ResNet(  # type: ignore[attr-defined]
        number_channels=number_channels,
        number_residual_blocks=number_residual_blocks,
        upsampling_factor=upsampling_factor,
        noise=(model_type == "gan"),
        constraints=constraints,
        dim=1,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_harder_cnn(
    model: nn.Module,
    lr_orig: torch.Tensor,
    hr: torch.Tensor,
    min_val: float,
    max_val: float,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
    max_samples: int | None = None,
    upsampling_factor: int = 4,
) -> dict[str, float]:
    """Evaluate a deterministic Harder CNN model.

    Args:
        model: Loaded Harder CNN model (eval mode).
        lr_orig: Original LR, shape (N, 1, H_lr, W_lr).
        hr: Ground truth HR, shape (N, 1, H_hr, W_hr).
        min_val: Min-max normalization min value.
        max_val: Min-max normalization max value.
        device: Computation device.
        batch_size: Evaluation batch size.
        max_samples: Limit number of samples.
        upsampling_factor: SR factor (4 for ERA5, 2 for NorESM).

    Returns:
        Dict with crps, mae, rmse, mass_violation.
    """
    pool = nn.AvgPool2d(kernel_size=upsampling_factor)
    n = min(lr_orig.shape[0], max_samples) if max_samples else lr_orig.shape[0]
    val_range = max_val - min_val

    all_crps: list[float] = []
    all_mae: list[float] = []
    all_rmse_sq: list[float] = []
    all_mass_viol: list[float] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # Harder input format: (B, 1, 1, H, W), min-max normalized
        lr_batch = lr_orig[start:end]
        lr_norm = (lr_batch - min_val) / val_range
        lr_in = lr_norm.unsqueeze(1).to(device)

        with torch.no_grad():
            out = model(lr_in)  # (B, 1, H_hr, W_hr)
        # Denormalize
        pred = out.cpu().squeeze(1) * val_range + min_val  # (B, 1, 128, 128) or (B, 128, 128)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)  # (B, 1, 128, 128)

        hr_batch = hr[start:end]
        lr_batch_orig = lr_orig[start:end]

        for i in range(pred.shape[0]):
            gt = hr_batch[i, 0].numpy()
            p = pred[i, 0].numpy()
            all_crps.append(crps_energy(gt, p[None, ...]))
            all_mae.append(float(np.mean(np.abs(gt - p))))
            all_rmse_sq.append(float(np.mean((gt - p) ** 2)))
            pooled = pool(pred[i : i + 1]).squeeze()
            lr_i = lr_batch_orig[i, 0]
            all_mass_viol.append(float(torch.mean(torch.abs(pooled - lr_i)).item()))

        if (start // batch_size) % 20 == 0:
            print(f"  Evaluated {end}/{n}...")

    return {
        "crps": float(np.mean(all_crps)),
        "mae": float(np.mean(all_mae)),
        "rmse": float(np.sqrt(np.mean(all_rmse_sq))),
        "mass_violation": float(np.mean(all_mass_viol)),
    }


def evaluate_harder_gan(
    model: nn.Module,
    lr_orig: torch.Tensor,
    hr: torch.Tensor,
    min_val: float,
    max_val: float,
    device: str | torch.device = "cpu",
    n_ensemble: int = 10,
    batch_size: int = 256,
    max_samples: int | None = None,
    upsampling_factor: int = 4,
) -> dict[str, float]:
    """Evaluate a Harder GAN model with ensemble sampling.

    Args:
        model: Loaded Harder GAN model (eval mode).
        lr_orig: Original LR, shape (N, 1, H_lr, W_lr).
        hr: Ground truth HR, shape (N, 1, H_hr, W_hr).
        min_val: Min-max normalization min value.
        max_val: Min-max normalization max value.
        device: Computation device.
        n_ensemble: Number of noise samples per input.
        batch_size: Evaluation batch size.
        max_samples: Limit number of samples.
        upsampling_factor: SR factor (4 for ERA5, 2 for NorESM).

    Returns:
        Dict with crps, mae, rmse, mass_violation.
    """
    pool = nn.AvgPool2d(kernel_size=upsampling_factor)
    n = min(lr_orig.shape[0], max_samples) if max_samples else lr_orig.shape[0]
    val_range = max_val - min_val

    all_crps: list[float] = []
    all_mae: list[float] = []
    all_rmse_sq: list[float] = []
    all_mass_viol: list[float] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bs = end - start
        lr_batch = lr_orig[start:end]
        lr_norm = (lr_batch - min_val) / val_range
        lr_in = lr_norm.unsqueeze(1).to(device)  # (B, 1, 1, 32, 32)

        ensemble_preds = []
        for _ in range(n_ensemble):
            z = torch.randn(bs, 100, 1, 1, device=device)
            with torch.no_grad():
                out = model(lr_in, z)  # (B, 1, H_hr, W_hr)
            pred = out.cpu().squeeze(1) * val_range + min_val
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)
            ensemble_preds.append(pred.numpy())

        ens_np = np.stack(ensemble_preds, axis=1)  # (B, M, 1, 128, 128)

        hr_batch = hr[start:end]
        lr_batch_orig = lr_orig[start:end]

        for i in range(bs):
            gt = hr_batch[i, 0].numpy()
            ens = ens_np[i, :, 0, ...]  # (M, 128, 128)
            ens_mean = ens.mean(axis=0)

            all_crps.append(crps_energy(gt, ens))
            all_mae.append(float(np.mean(np.abs(gt - ens_mean))))
            all_rmse_sq.append(float(np.mean((gt - ens_mean) ** 2)))
            pred_t = torch.from_numpy(ens_mean).unsqueeze(0).unsqueeze(0)
            pooled = pool(pred_t).squeeze()
            lr_i = lr_batch_orig[i, 0]
            all_mass_viol.append(float(torch.mean(torch.abs(pooled - lr_i)).item()))

        if (start // batch_size) % 20 == 0:
            print(f"  Evaluated {end}/{n}...")

    return {
        "crps": float(np.mean(all_crps)),
        "mae": float(np.mean(all_mae)),
        "rmse": float(np.sqrt(np.mean(all_rmse_sq))),
        "mass_violation": float(np.mean(all_mass_viol)),
    }


def generate_harder_cnn_predictions(
    model: nn.Module,
    lr_orig: torch.Tensor,
    min_val: float,
    max_val: float,
    device: str | torch.device = "cpu",
    n_samples: int = 5,
) -> torch.Tensor:
    """Generate CNN predictions for visualization.

    Returns:
        Predictions, shape (n_samples, 1, H_hr, W_hr).
    """
    val_range = max_val - min_val
    lr = lr_orig[:n_samples]
    lr_norm = (lr - min_val) / val_range
    lr_in = lr_norm.unsqueeze(1).to(device)  # (n, 1, 1, 32, 32)

    with torch.no_grad():
        out = model(lr_in)  # (n, 1, 128, 128)

    pred = out.cpu() * val_range + min_val
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    return pred


def generate_harder_gan_predictions(
    model: nn.Module,
    lr_orig: torch.Tensor,
    min_val: float,
    max_val: float,
    device: str | torch.device = "cpu",
    n_samples: int = 5,
    n_ensemble: int = 10,
) -> tuple[torch.Tensor, np.ndarray]:
    """Generate GAN ensemble predictions for visualization.

    Returns:
        Tuple of (ensemble_mean, ensemble_all):
            ensemble_mean: shape (n_samples, 1, H_hr, W_hr)
            ensemble_all: shape (n_samples, n_ensemble, H_hr, W_hr)
    """
    val_range = max_val - min_val
    lr = lr_orig[:n_samples]
    lr_norm = (lr - min_val) / val_range
    lr_in = lr_norm.unsqueeze(1).to(device)  # (n, 1, 1, 32, 32)

    all_preds = []
    for _ in range(n_ensemble):
        z = torch.randn(n_samples, 100, 1, 1, device=device)
        with torch.no_grad():
            out = model(lr_in, z)
        pred = out.cpu() * val_range + min_val
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        all_preds.append(pred[:, 0].numpy())  # (n, 128, 128)

    ensemble_all = np.stack(all_preds, axis=1)  # (n, M, 128, 128)
    ensemble_mean = torch.from_numpy(ensemble_all.mean(axis=1)).unsqueeze(1)  # (n, 1, 128, 128)
    return ensemble_mean, ensemble_all
