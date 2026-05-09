"""Generate all visualization artifacts.

Produces two categories of plots:
  1. Output artifacts: sample grids and ensemble members from the best model
  2. Data plots: metrics comparison charts from evaluation JSON

Usage:
    python scripts/visualize.py                  # data plots only (no GPU needed)
    python scripts/visualize.py --samples        # include sample generation (needs GPU)
    python scripts/visualize.py --samples --n-samples 8 --n-ensemble 10
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from downscaling.constraints import apply_addcl
from downscaling.data import denormalize, load_era5_tcw4, normalize
from downscaling.evaluation import load_flow_checkpoint
from downscaling.sampling import midpoint_sample
from downscaling.visualization import (
    plot_constraint_effect,
    plot_ensemble_members,
    plot_mass_violation,
    plot_metrics_comparison,
    plot_sample_grid,
)

POOL = Path("/home/chenxy/orcd/pool/datasets")
BASEDIR = "external/constrained-downscaling"
RESULTS_JSON = Path("results/eval_200samples.json")
BEST_MODEL = {
    "checkpoint": POOL / "research3/models/unet_wide96_amp/best_flow.pt",
    "norm_stats": POOL / "research3/models/unet_wide96_amp/norm_stats.pt",
}
FIGURES_DIR = Path("figures")


def generate_ensemble_predictions(
    n_samples: int = 8,
    n_ensemble: int = 10,
    ode_steps: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Generate ensemble predictions from the best flow matching model.

    Returns:
        (lr_up, hr, ensemble_preds, indices) all as numpy arrays.
        lr_up: (N, 1, H, W), hr: (N, 1, H, W), ensemble_preds: (N, M, 1, H, W)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    print(f"Loading model from {BEST_MODEL['checkpoint']}...")
    model, stats = load_flow_checkpoint(
        str(BEST_MODEL["checkpoint"]),
        str(BEST_MODEL["norm_stats"]),
        device=str(device),
    )

    print("Loading test data...")
    lr_up, _residual, hr, lr_orig = load_era5_tcw4(BASEDIR, "test")
    n_test = lr_up.shape[0]

    # Select evenly spaced samples
    step = max(1, n_test // n_samples)
    indices = list(range(0, n_test, step))[:n_samples]

    sel_lr_up = lr_up[indices]
    sel_hr = hr[indices]
    sel_lr_orig = lr_orig[indices]
    sel_lr_norm = normalize(sel_lr_up, stats.lr_mean, stats.lr_std)

    print(f"Generating {n_ensemble}-member ensemble for {len(indices)} samples...")
    ensemble = []
    bs = sel_lr_norm.shape[0]
    for e in range(n_ensemble):
        with torch.no_grad():
            res_norm = midpoint_sample(
                model,
                sel_lr_norm.to(device),
                shape=(bs, 1, 128, 128),
                steps=ode_steps,
            )
            res = denormalize(res_norm.cpu(), stats.res_mean, stats.res_std)
            pred = sel_lr_up + res
            pred = apply_addcl(pred, sel_lr_orig)
            ensemble.append(pred.numpy())
        print(f"  member {e + 1}/{n_ensemble} done")

    ensemble_preds = np.stack(ensemble, axis=1)  # (N, M, 1, H, W)
    return sel_lr_up.numpy(), sel_hr.numpy(), ensemble_preds, indices


def make_sample_plots(
    lr_up: np.ndarray,
    hr: np.ndarray,
    ensemble_preds: np.ndarray,
    indices: list[int],
    out_dir: Path,
) -> None:
    """Generate all output artifact plots."""
    # Sample grid
    print("Plotting sample grid...")
    plot_sample_grid(
        lr_up,
        hr,
        ensemble_preds,
        list(range(len(indices))),
        out_dir / "sample_grid.png",
        method_name="OT-CFM (wide96+AddCL)",
    )
    print(f"  Saved {out_dir / 'sample_grid.png'}")

    # Ensemble members for first sample
    print("Plotting ensemble members...")
    plot_ensemble_members(lr_up, hr, ensemble_preds, 0, out_dir / "ensemble_members_best.png")
    print(f"  Saved {out_dir / 'ensemble_members_best.png'}")

    # Ensemble members for highest-error sample
    errors = [
        float(np.mean(np.abs(hr[i, 0] - ensemble_preds[i, :, 0].mean(axis=0))))
        for i in range(len(indices))
    ]
    worst_idx = int(np.argmax(errors))
    if worst_idx != 0:
        plot_ensemble_members(
            lr_up, hr, ensemble_preds, worst_idx, out_dir / "ensemble_members_worst.png"
        )
        print(f"  Saved {out_dir / 'ensemble_members_worst.png'}")


def make_data_plots(results_path: Path, out_dir: Path) -> None:
    """Generate all data comparison plots from evaluation JSON."""
    print("Plotting metrics comparison (CRPS)...")
    plot_metrics_comparison(results_path, out_dir / "metrics_crps.png", metric="crps")
    print(f"  Saved {out_dir / 'metrics_crps.png'}")

    print("Plotting metrics comparison (MAE)...")
    plot_metrics_comparison(results_path, out_dir / "metrics_mae.png", metric="mae")
    print(f"  Saved {out_dir / 'metrics_mae.png'}")

    print("Plotting constraint effect...")
    plot_constraint_effect(results_path, out_dir / "constraint_effect.png")
    print(f"  Saved {out_dir / 'constraint_effect.png'}")

    print("Plotting mass violation...")
    plot_mass_violation(results_path, out_dir / "mass_violation.png")
    print(f"  Saved {out_dir / 'mass_violation.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualization artifacts")
    parser.add_argument("--samples", action="store_true", help="Generate sample plots (needs GPU)")
    parser.add_argument("--n-samples", type=int, default=8, help="Number of test samples")
    parser.add_argument("--n-ensemble", type=int, default=10, help="Ensemble size")
    parser.add_argument("--ode-steps", type=int, default=10, help="ODE solver steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results", type=str, default=str(RESULTS_JSON), help="Results JSON path")
    parser.add_argument("--output-dir", type=str, default=str(FIGURES_DIR), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data plots (no GPU needed)
    results_path = Path(args.results)
    if results_path.exists():
        make_data_plots(results_path, out_dir)
    else:
        print(f"WARNING: Results JSON not found at {results_path}, skipping data plots")

    # Sample plots (needs GPU + model weights)
    if args.samples:
        if not BEST_MODEL["checkpoint"].exists():
            print(f"ERROR: Model checkpoint not found at {BEST_MODEL['checkpoint']}")
            return
        lr_up, hr, ensemble_preds, indices = generate_ensemble_predictions(
            n_samples=args.n_samples,
            n_ensemble=args.n_ensemble,
            ode_steps=args.ode_steps,
            seed=args.seed,
        )
        make_sample_plots(lr_up, hr, ensemble_preds, indices, out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
