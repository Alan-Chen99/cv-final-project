"""Generate all visualization figures for climate downscaling results.

Central model configuration via DATASETS dict. All model names match
MODEL_FAMILIES from comprehensive.py — one canonical set of 8 models
used across all metric and sample plots.

Produces:
1. Metric plots (PSD, rank histograms, calibration, per-metric bars)
   — delegates to downscaling.plotting.comprehensive
2. Per-dataset sample visualizations (GPU required)

Usage:
    # Metrics only (no GPU)
    python scripts/make_figures.py --metrics-only

    # Full figures with sample predictions (GPU needed)
    python scripts/make_figures.py

    # Specify output directory
    python scripts/make_figures.py --output-dir figures/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from downscaling.constraints.layers import apply_addcl
from downscaling.data.era5 import load_era5_tcw
from downscaling.data.noresm import load_noresm_tas
from downscaling.evaluation.checkpoints import load_checkpoint, load_norm_stats
from downscaling.evaluation.harder import (
    compute_minmax_stats,
    generate_harder_cnn_predictions,
    generate_harder_gan_predictions,
    load_harder_model,
)
from downscaling.evaluation.swinir import load_swinir_finetuned, predict_swinir_finetuned
from downscaling.models.unet import AttentionUNet
from downscaling.plotting.comprehensive import (
    ERA5_RESULTS,
    MODEL_FAMILIES,
    NORESM_RESULTS,
)
from downscaling.plotting.comprehensive import (
    generate_all_figures as generate_metric_figures,
)
from downscaling.plotting.samples import (
    plot_ensemble_spread,
    plot_error_maps,
    plot_output_grid,
    plot_sample_comparison,
)
from downscaling.sampling.ode import midpoint_sample

POOL = Path("/home/chenxy/orcd/pool/datasets")
ANVITA = Path("/orcd/pool/007/chenxy/datasets/anvita")


# ─────────────────────────────────────────────────────────────────────────────
# Central model configuration
#
# ONE place that defines every model. Names match MODEL_FAMILIES:
#   Bicubic, CNN, CNN+SmCL (train), GAN, GAN+SmCL (train),
#   Flow, SwinIR, Truth+AddCL
#
# Bicubic and Truth+AddCL need no checkpoints (computed from data).
# Harder (CNN/GAN) checkpoints: Anvita primary, pool fallback.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HarderSpec:
    """Harder et al. CNN/GAN model."""

    model_type: str  # "cnn" or "gan"
    constraints: str  # "none" or "softmax"
    primary: str  # checkpoint path relative to ANVITA
    fallback: str  # checkpoint path relative to POOL


@dataclass(frozen=True)
class FlowSpec:
    """Flow matching model."""

    model_dir: str  # relative to POOL
    base_channels: int = 96
    channel_mults: tuple[int, ...] = (1, 2, 4)


@dataclass(frozen=True)
class SwinIRSpec:
    """SwinIR finetuned model."""

    pretrained: str  # relative to POOL
    checkpoint: str  # relative to POOL


@dataclass(frozen=True)
class DatasetSpec:
    """Per-dataset model and data configuration."""

    label: str
    upsampling_factor: int
    hr_size: tuple[int, int]
    harder: dict[str, HarderSpec]
    flow: FlowSpec
    swinir: SwinIRSpec


DATASETS: dict[str, DatasetSpec] = {
    "era5": DatasetSpec(
        label="ERA5 TCW 4x",
        upsampling_factor=4,
        hr_size=(128, 128),
        harder={
            "CNN": HarderSpec(
                "cnn",
                "none",
                "twc_cnn_noconstraints.pth",
                "organize2/models/harder/twc_cnn_none.pth",
            ),
            "CNN+SmCL (train)": HarderSpec(
                "cnn",
                "softmax",
                "twc_cnn_softmax.pth",
                "organize2/models/harder/twc_cnn_softmax.pth",
            ),
            "GAN": HarderSpec(
                "gan",
                "none",
                "twc_gan_noconstraints.pth",
                "organize2/models/harder/twc_gan_none.pth",
            ),
            "GAN+SmCL (train)": HarderSpec(
                "gan",
                "softmax",
                "twc_gan_softmax.pth",
                "organize2/models/harder/twc_gan_softmax.pth",
            ),
        },
        flow=FlowSpec("research3/models/unet_wide96_amp"),
        swinir=SwinIRSpec(
            "research5/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
            "research5/models/swinir_ft/best_swinir.pt",
        ),
    ),
    "noresm": DatasetSpec(
        label="NorESM TAS 2x",
        upsampling_factor=2,
        hr_size=(64, 64),
        harder={
            "CNN": HarderSpec(
                "cnn",
                "none",
                "noresm_cnn_none.pth",
                "noresm-dataset/models/harder/twc_cnn_none.pth",
            ),
            "CNN+SmCL (train)": HarderSpec(
                "cnn",
                "softmax",
                "noresm_cnn_softmax.pth",
                "noresm-dataset/models/harder/twc_cnn_softmax.pth",
            ),
            "GAN": HarderSpec(
                "gan",
                "none",
                "noresm_gan_none.pth",
                "noresm-dataset/models/harder/twc_gan_none.pth",
            ),
            "GAN+SmCL (train)": HarderSpec(
                "gan",
                "softmax",
                "noresm_gan_softmax.pth",
                "noresm-dataset/models/harder/twc_gan_softmax.pth",
            ),
        },
        flow=FlowSpec("noresm-dataset/models/flow-wide96-amp"),
        swinir=SwinIRSpec(
            "noresm-dataset/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
            "noresm-dataset/models/swinir_ft/best_swinir.pt",
        ),
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_harder_ckpt(spec: HarderSpec, pool_dir: Path, anvita_dir: Path) -> Path | None:
    """Return first existing checkpoint, or None."""
    primary = anvita_dir / spec.primary
    if primary.exists():
        return primary
    fallback = pool_dir / spec.fallback
    if fallback.exists():
        return fallback
    return None


def _pick_random_indices(n_total: int, n_pick: int, seed: int = 42) -> list[int]:
    """Stratified random sampling: one random index per equal-sized bin."""
    if n_pick >= n_total:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    bin_size = n_total // n_pick
    return [rng.integers(i * bin_size, (i + 1) * bin_size) for i in range(n_pick)]


def _find_best_crop(hr: torch.Tensor, crop_frac: float = 0.5) -> tuple[int, int, int, int]:
    """Find the crop region with highest average gradient magnitude.

    Slides a crop_frac-sized window across the HR fields and picks the
    position that maximizes average spatial gradient across all samples.
    Returns (row_start, row_end, col_start, col_end).
    """
    hr_np = hr[:, 0].numpy() if hr.ndim == 4 else hr.numpy()
    h, w = hr_np.shape[-2], hr_np.shape[-1]
    ch, cw = int(h * crop_frac), int(w * crop_frac)

    # Gradient magnitude map averaged across samples
    # diff axis=-2 → (N, H-1, W), diff axis=-1 → (N, H, W-1)
    dy = np.abs(np.diff(hr_np, axis=-2)).mean(axis=0)  # (H-1, W)
    dx = np.abs(np.diff(hr_np, axis=-1)).mean(axis=0)  # (H, W-1)

    grad_map = np.zeros((h, w), dtype=np.float64)
    grad_map[: h - 1, :] += dy
    grad_map[:, : w - 1] += dx

    # Integral image for fast window sums
    integral = grad_map.cumsum(axis=0).cumsum(axis=1)

    def _rect_sum(r0: int, c0: int, r1: int, c1: int) -> float:
        s = integral[r1 - 1, c1 - 1]
        if r0 > 0:
            s -= integral[r0 - 1, c1 - 1]
        if c0 > 0:
            s -= integral[r1 - 1, c0 - 1]
        if r0 > 0 and c0 > 0:
            s += integral[r0 - 1, c0 - 1]
        return float(s)

    best_score = -1.0
    best_r0, best_c0 = 0, 0
    for r0 in range(h - ch + 1):
        for c0 in range(w - cw + 1):
            score = _rect_sum(r0, c0, r0 + ch, c0 + cw)
            if score > best_score:
                best_score = score
                best_r0, best_c0 = r0, c0

    return (best_r0, best_r0 + ch, best_c0, best_c0 + cw)


def _load_test_data(
    dataset_name: str, pool_dir: Path
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load test split, returning (lr_up, hr, lr_orig)."""
    if dataset_name == "era5":
        lr_up, _, hr, lr_orig = load_era5_tcw(pool_dir, "test")
    elif dataset_name == "noresm":
        lr_up, _, hr, lr_orig = load_noresm_tas(pool_dir, "test")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return lr_up, hr, lr_orig


# ─────────────────────────────────────────────────────────────────────────────
# Prediction generation — driven by DATASETS config
# ─────────────────────────────────────────────────────────────────────────────


def generate_predictions(
    dataset_name: str,
    ds: DatasetSpec,
    lr_up: torch.Tensor,
    lr_orig: torch.Tensor,
    hr: torch.Tensor,
    device: str,
    n_samples: int,
    n_ensemble: int,
    pool_dir: Path = POOL,
    anvita_dir: Path = ANVITA,
) -> tuple[dict[str, torch.Tensor | np.ndarray], np.ndarray | None]:
    """Generate predictions for all MODEL_FAMILIES models.

    Returns (predictions dict ordered by MODEL_FAMILIES, flow_ensemble or None).
    Models with missing checkpoints are skipped.
    """
    preds: dict[str, torch.Tensor | np.ndarray] = {}
    flow_ensemble: np.ndarray | None = None
    harder_minmax: tuple[float, float] | None = None

    for name in MODEL_FAMILIES:
        if name == "Bicubic":
            preds[name] = F.interpolate(
                lr_orig[:n_samples],
                scale_factor=ds.upsampling_factor,
                mode="bicubic",
                align_corners=False,
            )

        elif name in ds.harder:
            spec = ds.harder[name]
            ckpt = _resolve_harder_ckpt(spec, pool_dir, anvita_dir)
            if ckpt is None:
                print(f"  Skipping {name}: no checkpoint")
                continue
            if harder_minmax is None:
                harder_minmax = compute_minmax_stats(pool_dir, dataset=dataset_name)
            min_val, max_val = harder_minmax
            print(f"  {name} <- {ckpt.name}")
            model = load_harder_model(
                ckpt,
                spec.model_type,
                spec.constraints,
                device,
                upsampling_factor=ds.upsampling_factor,
            )
            if spec.model_type == "gan":
                mean_pred, _ = generate_harder_gan_predictions(
                    model,
                    lr_orig,
                    min_val,
                    max_val,
                    device,
                    n_samples=n_samples,
                    n_ensemble=n_ensemble,
                )
                preds[name] = mean_pred
            else:
                preds[name] = generate_harder_cnn_predictions(
                    model,
                    lr_orig,
                    min_val,
                    max_val,
                    device,
                    n_samples=n_samples,
                )
            del model
            torch.cuda.empty_cache()

        elif name == "Flow":
            flow_dir = pool_dir / ds.flow.model_dir
            if not flow_dir.exists() or device == "cpu":
                reason = "missing weights" if not flow_dir.exists() else "no GPU"
                print(f"  Skipping {name}: {reason}")
                continue
            print(f"  {name} <- {flow_dir.name}")
            model = AttentionUNet(
                in_channels=2,
                out_channels=1,
                base_channels=ds.flow.base_channels,
                channel_mults=ds.flow.channel_mults,
            ).to(device)
            ckpt_path = flow_dir / "best_flow.pt"
            if not ckpt_path.exists():
                ckpt_path = flow_dir / "flow_best.pth"
            load_checkpoint(model, ckpt_path, device)
            norm_stats = load_norm_stats(flow_dir / "norm_stats.pt")
            lr_norm = (lr_up[:n_samples] - norm_stats["lr_mean"]) / norm_stats["lr_std"]

            members: list[np.ndarray] = []
            for _ in range(n_ensemble):
                with torch.no_grad():
                    sampled = midpoint_sample(
                        model,
                        lr_norm.to(device),
                        shape=(n_samples, 1, *ds.hr_size),
                        steps=10,
                    )
                    res = sampled.cpu() * norm_stats["res_std"] + norm_stats["res_mean"]
                    members.append((lr_up[:n_samples] + res)[:, 0].numpy())

            flow_ensemble = np.stack(members, axis=1)
            preds[name] = torch.from_numpy(flow_ensemble.mean(axis=1)).unsqueeze(1)
            del model
            torch.cuda.empty_cache()
            print(f"  {name}: {n_samples} samples x {n_ensemble} ensemble")

        elif name == "SwinIR":
            pretrained = pool_dir / ds.swinir.pretrained
            ckpt = pool_dir / ds.swinir.checkpoint
            if not pretrained.exists() or not ckpt.exists() or device == "cpu":
                reason = (
                    "missing weights" if not (pretrained.exists() and ckpt.exists()) else "no GPU"
                )
                print(f"  Skipping {name}: {reason}")
                continue
            print(f"  {name} <- {ckpt.name}")
            swinir_model, vmin, vmax = load_swinir_finetuned(pretrained, ckpt, device)
            preds[name] = predict_swinir_finetuned(
                swinir_model, lr_orig, vmin, vmax, device, batch_size=64
            )[:n_samples]
            del swinir_model
            torch.cuda.empty_cache()

        elif name == "Truth+AddCL":
            preds[name] = apply_addcl(hr[:n_samples], lr_orig[:n_samples], ds.upsampling_factor)

    return preds, flow_ensemble


# ─────────────────────────────────────────────────────────────────────────────
# Sample figure generation
# ─────────────────────────────────────────────────────────────────────────────


def make_sample_figures(
    dataset_name: str,
    pool_dir: Path,
    output_dir: Path,
    device: str,
    n_vis_samples: int = 5,
    n_ensemble: int = 10,
) -> None:
    """Generate sample prediction visualizations for one dataset."""
    ds = DATASETS[dataset_name]
    print(f"\n=== {ds.label} Sample Figures ===")

    lr_up, hr, lr_orig = _load_test_data(dataset_name, pool_dir)
    print(f"  Loaded: lr_orig={lr_orig.shape}, hr={hr.shape}")

    vis_idx = _pick_random_indices(hr.shape[0], n_vis_samples)
    print(f"  Sample indices: {vis_idx}")
    lr_up = lr_up[vis_idx]
    hr = hr[vis_idx]
    lr_orig = lr_orig[vis_idx]

    print("  Generating predictions...")
    preds, flow_ensemble = generate_predictions(
        dataset_name,
        ds,
        lr_up,
        lr_orig,
        hr,
        device,
        n_vis_samples,
        n_ensemble,
        pool_dir,
    )
    print(f"  Generated {len(preds)}/{len(MODEL_FAMILIES)} models")

    # Per-sample comparisons and error maps
    for idx in range(min(n_vis_samples, 5)):
        print(f"  Plotting {dataset_name} sample {idx}...")
        plot_sample_comparison(
            lr=lr_orig,
            hr=hr,
            predictions=preds,
            sample_idx=idx,
            output_path=output_dir / f"{dataset_name}_sample_{idx}_comparison.png",
        )
        plot_error_maps(
            hr=hr,
            predictions=preds,
            sample_idx=idx,
            output_path=output_dir / f"{dataset_name}_sample_{idx}_errors.png",
        )

    # Ensemble spread (flow model only)
    if flow_ensemble is not None:
        for idx in range(min(3, n_vis_samples)):
            plot_ensemble_spread(
                hr=hr,
                ensemble_preds=flow_ensemble,
                sample_idx=idx,
                output_path=output_dir / f"{dataset_name}_sample_{idx}_ensemble.png",
            )
            print(f"  Saved {dataset_name} ensemble spread sample {idx}")

    # Output grids: full + zoomed (best-detail crop)
    grid_n = min(3, n_vis_samples)
    print(f"  Plotting {dataset_name} output grid ({len(preds)} methods x {grid_n} samples)...")
    plot_output_grid(
        lr=lr_orig,
        hr=hr,
        predictions=preds,
        n_samples=grid_n,
        output_path=output_dir / f"{dataset_name}_output_grid.png",
    )

    crop_box = _find_best_crop(hr[:grid_n])
    print(f"  Plotting {dataset_name} zoomed grid (crop {crop_box})...")
    plot_output_grid(
        lr=lr_orig,
        hr=hr,
        predictions=preds,
        n_samples=grid_n,
        crop=crop_box,
        title="Output Comparison (Zoomed)",
        output_path=output_dir / f"{dataset_name}_output_grid_zoomed.png",
    )

    plt.close("all")
    print(f"  {dataset_name} sample figures saved to {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualization figures")
    parser.add_argument("--pool-dir", type=Path, default=POOL)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--n-ensemble", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Metric figures from comprehensive evaluation results
    if NORESM_RESULTS.exists() and ERA5_RESULTS.exists():
        print("=== Metric Figures ===")
        generate_metric_figures(args.output_dir)
    else:
        print("Skipping metric figures: comprehensive results not found")
        if not NORESM_RESULTS.exists():
            print(f"  Missing: {NORESM_RESULTS}")
        if not ERA5_RESULTS.exists():
            print(f"  Missing: {ERA5_RESULTS}")

    # Sample prediction figures (GPU required)
    if not args.metrics_only:
        for name in DATASETS:
            ds_dir = args.output_dir / name
            ds_dir.mkdir(exist_ok=True)
            make_sample_figures(
                name,
                args.pool_dir,
                ds_dir,
                args.device,
                args.n_samples,
                args.n_ensemble,
            )

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
