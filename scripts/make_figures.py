"""Generate all visualization figures for ERA5 TCW 4x downscaling results.

Produces:
1. Metric comparison plots (from eval_results JSON)
2. Sample prediction comparisons (baselines: CPU; flow models: GPU)
3. Ensemble spread visualization (GPU)

Usage:
    # Metrics only (no GPU)
    python scripts/make_figures.py --metrics-only

    # Full figures with sample predictions (GPU needed for flow models)
    python scripts/make_figures.py --pool-dir /home/chenxy/orcd/pool/datasets

    # Specify output directory
    python scripts/make_figures.py --output-dir figures/
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from downscaling.constraints.layers import apply_addcl
from downscaling.data.era5 import load_era5_tcw
from downscaling.evaluation.checkpoints import load_checkpoint, load_norm_stats
from downscaling.models.unet import AttentionUNet
from downscaling.plotting.metrics import (
    load_results,
    plot_crps_comparison,
    plot_flow_vs_baseline,
    plot_metrics_panel,
)
from downscaling.plotting.samples import (
    generate_baseline_predictions,
    plot_ensemble_spread,
    plot_error_maps,
    plot_sample_comparison,
)
from downscaling.sampling.ode import midpoint_sample

POOL = Path("/home/chenxy/orcd/pool/datasets")


def generate_flow_predictions(
    model_dir: Path,
    lr_up: torch.Tensor,
    lr_orig: torch.Tensor,
    norm_stats: dict[str, float],
    device: str,
    n_samples: int = 5,
    n_ensemble: int = 10,
    ode_steps: int = 10,
    base_channels: int = 96,
    channel_mults: tuple[int, ...] = (1, 2, 4),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate flow model predictions for visualization.

    Returns:
        Tuple of (ensemble_mean, ensemble_all):
            ensemble_mean: shape (n_samples, 128, 128)
            ensemble_all: shape (n_samples, n_ensemble, 128, 128)
    """
    model = AttentionUNet(
        in_channels=2,
        out_channels=1,
        base_channels=base_channels,
        channel_mults=channel_mults,
    ).to(device)

    ckpt_path = model_dir / "best_flow.pt"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "flow_best.pth"
    load_checkpoint(model, ckpt_path, device)

    lr_sub = lr_up[:n_samples]
    lr_orig_sub = lr_orig[:n_samples]
    lr_norm = (lr_sub - norm_stats["lr_mean"]) / norm_stats["lr_std"]

    all_preds = []
    for _ in range(n_ensemble):
        with torch.no_grad():
            sampled = midpoint_sample(
                model,
                lr_norm.to(device),
                shape=(n_samples, 1, 128, 128),
                steps=ode_steps,
            )
            res = sampled.cpu() * norm_stats["res_std"] + norm_stats["res_mean"]
            pred_hr = lr_sub + res
            pred_hr = apply_addcl(pred_hr, lr_orig_sub)
            all_preds.append(pred_hr[:, 0].numpy())

    ensemble_all = np.stack(all_preds, axis=1)  # (n_samples, n_ensemble, 128, 128)
    ensemble_mean = ensemble_all.mean(axis=1)
    return ensemble_mean, ensemble_all


def make_metrics_figures(results_path: Path, output_dir: Path) -> None:
    """Generate all metrics comparison plots."""
    results = load_results(results_path)
    print(f"Loaded results for {len(results)} methods from {results_path}")

    plot_crps_comparison(results, output_dir / "crps_comparison.png")
    print(f"  Saved {output_dir / 'crps_comparison.png'}")

    plot_metrics_panel(results, output_dir / "metrics_panel.png")
    print(f"  Saved {output_dir / 'metrics_panel.png'}")

    # Only plot flow-vs-baseline if flow models are in results
    flow_methods = [m for m in results if m.startswith("flow-")]
    if flow_methods:
        plot_flow_vs_baseline(results, output_dir / "flow_vs_baseline.png")
        print(f"  Saved {output_dir / 'flow_vs_baseline.png'}")

    plt.close("all")


def make_sample_figures(
    pool_dir: Path,
    output_dir: Path,
    device: str,
    n_vis_samples: int = 5,
    n_ensemble: int = 10,
) -> None:
    """Generate sample prediction comparison plots."""
    print(f"\nLoading test data from {pool_dir}...")
    lr_up, _, hr, lr_orig = load_era5_tcw(pool_dir, "test")
    print(f"  Loaded: lr_orig={lr_orig.shape}, hr={hr.shape}")

    # Baseline predictions
    print("Generating baseline predictions...")
    baselines = generate_baseline_predictions(lr_orig, n_samples=n_vis_samples)

    # Flow model predictions (best model: wide96)
    flow_preds = None
    flow_ensemble = None
    best_model_dir = pool_dir / "research3/models/unet_wide96_amp"
    stats_path = best_model_dir / "norm_stats.pt"

    if best_model_dir.exists() and device != "cpu":
        print(f"Generating flow model predictions on {device}...")
        norm_stats = load_norm_stats(stats_path)
        flow_mean, flow_all = generate_flow_predictions(
            model_dir=best_model_dir,
            lr_up=lr_up,
            lr_orig=lr_orig,
            norm_stats=norm_stats,
            device=device,
            n_samples=n_vis_samples,
            n_ensemble=n_ensemble,
            base_channels=96,
            channel_mults=(1, 2, 4),
        )
        flow_preds = torch.from_numpy(flow_mean).unsqueeze(1)  # (N, 1, 128, 128)
        flow_ensemble = flow_all
        print(f"  Generated {n_vis_samples} samples x {n_ensemble} ensemble members")
    elif device == "cpu":
        print("  Skipping flow model predictions (no GPU)")
    else:
        print(f"  Skipping: {best_model_dir} not found")

    # Build prediction dict for comparison
    all_preds: dict[str, torch.Tensor | np.ndarray] = {}
    all_preds.update(baselines)
    if flow_preds is not None:
        all_preds["Wide96 Flow"] = flow_preds

    # Generate sample comparison figures
    for idx in range(min(n_vis_samples, 5)):
        print(f"  Plotting sample {idx}...")
        plot_sample_comparison(
            lr=lr_orig[:n_vis_samples],
            hr=hr[:n_vis_samples],
            predictions=all_preds,
            sample_idx=idx,
            output_path=output_dir / f"sample_{idx}_comparison.png",
        )

        plot_error_maps(
            hr=hr[:n_vis_samples],
            predictions=all_preds,
            sample_idx=idx,
            output_path=output_dir / f"sample_{idx}_errors.png",
        )

    # Ensemble spread (only if we have flow predictions)
    if flow_ensemble is not None:
        for idx in range(min(3, n_vis_samples)):
            plot_ensemble_spread(
                hr=hr[:n_vis_samples],
                ensemble_preds=flow_ensemble,
                sample_idx=idx,
                output_path=output_dir / f"sample_{idx}_ensemble.png",
            )
            print(f"  Saved ensemble spread for sample {idx}")

    plt.close("all")
    print(f"Sample figures saved to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualization figures")
    parser.add_argument("--pool-dir", type=Path, default=POOL)
    parser.add_argument("--results-json", type=Path, default=Path("eval_results_500.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--n-ensemble", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Generating Metrics Figures ===")
    make_metrics_figures(args.results_json, args.output_dir)

    if not args.metrics_only:
        print("\n=== Generating Sample Figures ===")
        make_sample_figures(
            pool_dir=args.pool_dir,
            output_dir=args.output_dir,
            device=args.device,
            n_vis_samples=args.n_samples,
            n_ensemble=args.n_ensemble,
        )

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
