"""Generate visualization figures for climate downscaling results.

Supports ERA5 TCW (4x) and NorESM TAS (2x) datasets.
Produces:
1. Per-dataset metric bar charts + flow-vs-baseline
2. Dual-dataset comparison (side-by-side CRPS, metrics panel, constraint impact)
3. Per-dataset sample predictions (GPU required for flow models)

Usage:
    # Metrics only (no GPU)
    python scripts/make_figures.py --metrics-only

    # Full figures with sample predictions (GPU needed)
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
from downscaling.plotting.metrics import (
    load_results,
    plot_constraint_impact,
    plot_crps_comparison,
    plot_dual_crps,
    plot_dual_metrics_panel,
    plot_flow_vs_baseline,
    plot_metrics_panel,
)
from downscaling.plotting.samples import (
    generate_baseline_predictions,
    plot_ensemble_spread,
    plot_error_maps,
    plot_sample_comparison,
)
from downscaling.plotting.spectral import (
    plot_extended_metrics_panel,
    plot_psd_comparison,
    plot_ralsd_comparison,
    plot_spectral_bias,
)
from downscaling.sampling.ode import midpoint_sample

POOL = Path("/home/chenxy/orcd/pool/datasets")


def generate_flow_predictions(
    model_dir: Path,
    lr_up: torch.Tensor,
    lr_orig: torch.Tensor,
    norm_stats: dict[str, float],
    device: str,
    hr_size: tuple[int, int],
    n_samples: int = 5,
    n_ensemble: int = 10,
    ode_steps: int = 10,
    base_channels: int = 96,
    channel_mults: tuple[int, ...] = (1, 2, 4),
    apply_constraint: bool = True,
    upsampling_factor: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate flow model predictions for visualization.

    Returns:
        Tuple of (ensemble_mean, ensemble_all):
            ensemble_mean: shape (n_samples, H, W)
            ensemble_all: shape (n_samples, n_ensemble, H, W)
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
                shape=(n_samples, 1, hr_size[0], hr_size[1]),
                steps=ode_steps,
            )
            res = sampled.cpu() * norm_stats["res_std"] + norm_stats["res_mean"]
            pred_hr = lr_sub + res
            if apply_constraint:
                pred_hr = apply_addcl(pred_hr, lr_orig_sub, upsampling_factor=upsampling_factor)
            all_preds.append(pred_hr[:, 0].numpy())

    ensemble_all = np.stack(all_preds, axis=1)
    ensemble_mean = ensemble_all.mean(axis=1)
    return ensemble_mean, ensemble_all


# ---------------------------------------------------------------------------
# Per-dataset metric figures
# ---------------------------------------------------------------------------


def make_metrics_figures(
    results_path: Path,
    output_dir: Path,
    dataset_label: str = "ERA5 TCW 4x Downscaling",
) -> dict[str, dict[str, float]]:
    """Generate per-dataset metrics comparison plots. Returns loaded results."""
    results = load_results(results_path)
    print(f"Loaded results for {len(results)} methods from {results_path}")

    plot_crps_comparison(
        results,
        output_dir / "crps_comparison.png",
        title=f"CRPS Comparison \u2014 {dataset_label}",
    )
    print(f"  Saved {output_dir / 'crps_comparison.png'}")

    plot_metrics_panel(
        results,
        output_dir / "metrics_panel.png",
        title=f"{dataset_label} \u2014 Method Comparison",
    )
    print(f"  Saved {output_dir / 'metrics_panel.png'}")

    flow_methods = [m for m in results if m.startswith("flow-")]
    if flow_methods:
        plot_flow_vs_baseline(results, output_dir / "flow_vs_baseline.png")
        print(f"  Saved {output_dir / 'flow_vs_baseline.png'}")

    # Extended metrics panel (7 metrics) — only if RALSD data available
    has_ralsd = any("ralsd" in r for r in results.values())
    if has_ralsd:
        plot_extended_metrics_panel(
            results,
            output_path=str(output_dir / "extended_metrics_panel.png"),
            title=f"{dataset_label} \u2014 All Metrics",
        )
        print(f"  Saved {output_dir / 'extended_metrics_panel.png'}")

        plot_ralsd_comparison(
            results,
            output_path=str(output_dir / "ralsd_comparison.png"),
            title=f"RALSD \u2014 {dataset_label}",
        )
        print(f"  Saved {output_dir / 'ralsd_comparison.png'}")

    # Spectral PSD/bias plots — only if .npz file exists
    spectral_path = results_path.with_name(results_path.stem + "_spectral.npz")
    if spectral_path.exists():
        data = np.load(spectral_path)
        freq = data["freq"]
        psd_truth = data["psd_truth"]

        # NPZ keys use sanitized names (spaces->_, parens removed).
        # Build reverse mapping from the canonical results dict.
        def _sanitize(n: str) -> str:
            return n.replace(" ", "_").replace("(", "").replace(")", "")

        sanitized_to_canonical = {_sanitize(name): name for name in results}

        method_psds: dict[str, np.ndarray] = {}
        method_biases: dict[str, np.ndarray] = {}
        for key in data.files:
            if key.startswith("psd_") and key != "psd_truth":
                sanitized_name = key[4:]  # strip "psd_"
                canonical = sanitized_to_canonical.get(sanitized_name, sanitized_name)
                method_psds[canonical] = data[key]
                bias_key = f"bias_{sanitized_name}"
                if bias_key in data.files:
                    method_biases[canonical] = data[bias_key]
        if method_psds:
            plot_psd_comparison(
                freq=freq,
                psd_truth=psd_truth,
                method_psds=method_psds,
                output_path=str(output_dir / "psd_comparison.png"),
                title=f"Power Spectral Density \u2014 {dataset_label}",
            )
            print(f"  Saved {output_dir / 'psd_comparison.png'}")

            plot_spectral_bias(
                freq=freq,
                method_biases=method_biases,
                output_path=str(output_dir / "spectral_bias.png"),
                title=f"Spectral Bias \u2014 {dataset_label}",
            )
            print(f"  Saved {output_dir / 'spectral_bias.png'}")

    plt.close("all")
    return results


def make_dual_dataset_figures(
    era5_results: dict[str, dict[str, float]],
    noresm_results: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    """Generate cross-dataset comparison figures."""
    plot_dual_crps(era5_results, noresm_results, output_path=output_dir / "dual_crps.png")
    print(f"  Saved {output_dir / 'dual_crps.png'}")

    plot_dual_metrics_panel(
        era5_results, noresm_results, output_path=output_dir / "dual_metrics_panel.png"
    )
    print(f"  Saved {output_dir / 'dual_metrics_panel.png'}")

    plot_constraint_impact(
        era5_results, noresm_results, output_path=output_dir / "constraint_impact.png"
    )
    print(f"  Saved {output_dir / 'constraint_impact.png'}")

    plt.close("all")


# ---------------------------------------------------------------------------
# Per-dataset sample figures
# ---------------------------------------------------------------------------


def _load_harder_predictions(
    pool_dir: Path,
    lr_orig: torch.Tensor,
    device: str,
    n_vis_samples: int,
    n_ensemble: int,
    harder_configs: dict[str, dict[str, str]],
    upsampling_factor: int,
    dataset: str = "era5",
) -> dict[str, torch.Tensor | np.ndarray]:
    """Load and run Harder models for sample visualization."""
    preds: dict[str, torch.Tensor | np.ndarray] = {}
    harder_min_val, harder_max_val = None, None

    for hname, hcfg in harder_configs.items():
        ckpt_path = pool_dir / hcfg["checkpoint"]
        if not ckpt_path.exists():
            print(f"  Skipping {hname}: {ckpt_path} not found")
            continue
        if harder_min_val is None:
            harder_min_val, harder_max_val = compute_minmax_stats(pool_dir, dataset=dataset)
        print(f"Generating {hname} predictions...")
        hmodel = load_harder_model(
            checkpoint_path=ckpt_path,
            model_type=hcfg["model_type"],
            constraints=hcfg["constraints"],
            device=device,
            upsampling_factor=upsampling_factor,
        )
        if hcfg["model_type"] == "gan":
            gan_mean, _ = generate_harder_gan_predictions(
                model=hmodel,
                lr_orig=lr_orig,
                min_val=harder_min_val,
                max_val=harder_max_val,
                device=device,
                n_samples=n_vis_samples,
                n_ensemble=n_ensemble,
            )
            preds[hname] = gan_mean
        else:
            preds[hname] = generate_harder_cnn_predictions(
                model=hmodel,
                lr_orig=lr_orig,
                min_val=harder_min_val,
                max_val=harder_max_val,
                device=device,
                n_samples=n_vis_samples,
            )
        del hmodel
        torch.cuda.empty_cache()
    return preds


def make_era5_sample_figures(
    pool_dir: Path,
    output_dir: Path,
    device: str,
    n_vis_samples: int = 5,
    n_ensemble: int = 10,
) -> None:
    """Generate sample prediction comparison plots for ERA5."""
    print(f"\nLoading ERA5 test data from {pool_dir}...")
    lr_up, _, hr, lr_orig = load_era5_tcw(pool_dir, "test")
    print(f"  Loaded: lr_orig={lr_orig.shape}, hr={hr.shape}")

    baselines = generate_baseline_predictions(lr_orig, n_samples=n_vis_samples, upsampling_factor=4)

    harder_configs = {
        "Harder CNN": {
            "checkpoint": "organize2/models/harder/twc_cnn_none.pth",
            "model_type": "cnn",
            "constraints": "none",
        },
        "Harder CNN+SmCL": {
            "checkpoint": "organize2/models/harder/twc_cnn_softmax.pth",
            "model_type": "cnn",
            "constraints": "softmax",
        },
        "Harder GAN+SmCL": {
            "checkpoint": "organize2/models/harder/twc_gan_softmax.pth",
            "model_type": "gan",
            "constraints": "softmax",
        },
    }
    harder_preds = _load_harder_predictions(
        pool_dir,
        lr_orig,
        device,
        n_vis_samples,
        n_ensemble,
        harder_configs,
        upsampling_factor=4,
        dataset="era5",
    )
    baselines.update(harder_preds)

    # SwinIR finetuned + AddCL
    swinir_pretrained = (
        pool_dir / "research5/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    )
    swinir_ckpt = pool_dir / "spatial-4x-add-v2/models/swinir_ft/best_swinir.pt"
    if swinir_pretrained.exists() and swinir_ckpt.exists() and device != "cpu":
        print("Generating SwinIR finetuned+AddCL predictions...")
        swinir_model, vmin, vmax = load_swinir_finetuned(swinir_pretrained, swinir_ckpt, device)
        swinir_preds = predict_swinir_finetuned(
            swinir_model, lr_orig, vmin, vmax, device, batch_size=64
        )
        swinir_preds = apply_addcl(swinir_preds[:n_vis_samples], lr_orig[:n_vis_samples])
        baselines["SwinIR FT+AddCL"] = swinir_preds
        del swinir_model
        torch.cuda.empty_cache()
    else:
        print("  Skipping SwinIR: missing weights or GPU")

    # Flow model predictions
    flow_preds = None
    flow_ensemble = None
    best_model_dir = pool_dir / "research3/models/unet_wide96_amp"
    stats_path = best_model_dir / "norm_stats.pt"

    if best_model_dir.exists() and device != "cpu":
        print(f"Generating ERA5 flow predictions on {device}...")
        norm_stats = load_norm_stats(stats_path)
        flow_mean, flow_all = generate_flow_predictions(
            model_dir=best_model_dir,
            lr_up=lr_up,
            lr_orig=lr_orig,
            norm_stats=norm_stats,
            device=device,
            hr_size=(128, 128),
            n_samples=n_vis_samples,
            n_ensemble=n_ensemble,
            base_channels=96,
            channel_mults=(1, 2, 4),
        )
        flow_preds = torch.from_numpy(flow_mean).unsqueeze(1)
        flow_ensemble = flow_all
        print(f"  Generated {n_vis_samples} samples x {n_ensemble} ensemble members")
    else:
        print("  Skipping flow model: missing weights or GPU")

    all_preds: dict[str, torch.Tensor | np.ndarray] = {}
    all_preds.update(baselines)
    if flow_preds is not None:
        all_preds["Wide96 Flow"] = flow_preds

    _save_sample_figures(lr_orig, hr, all_preds, flow_ensemble, output_dir, n_vis_samples, "era5")


def make_noresm_sample_figures(
    pool_dir: Path,
    output_dir: Path,
    device: str,
    n_vis_samples: int = 5,
    n_ensemble: int = 10,
) -> None:
    """Generate sample prediction comparison plots for NorESM."""
    print(f"\nLoading NorESM test data from {pool_dir}...")
    lr_up, _, hr, lr_orig = load_noresm_tas(pool_dir, "test")
    print(f"  Loaded: lr_orig={lr_orig.shape}, hr={hr.shape}")

    baselines = generate_baseline_predictions(lr_orig, n_samples=n_vis_samples, upsampling_factor=2)

    noresm_models = "noresm-dataset/models"
    harder_configs = {
        "Harder CNN": {
            "checkpoint": f"{noresm_models}/harder/twc_cnn_none.pth",
            "model_type": "cnn",
            "constraints": "none",
        },
        "Harder CNN+SmCL": {
            "checkpoint": f"{noresm_models}/harder/twc_cnn_softmax.pth",
            "model_type": "cnn",
            "constraints": "softmax",
        },
        "Harder GAN+SmCL": {
            "checkpoint": f"{noresm_models}/harder/twc_gan_softmax.pth",
            "model_type": "gan",
            "constraints": "softmax",
        },
    }
    harder_preds = _load_harder_predictions(
        pool_dir,
        lr_orig,
        device,
        n_vis_samples,
        n_ensemble,
        harder_configs,
        upsampling_factor=2,
        dataset="noresm",
    )
    baselines.update(harder_preds)

    # SwinIR finetuned
    swinir_x2_weights = (
        pool_dir / "noresm-dataset/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
    )
    swinir_ckpt = pool_dir / noresm_models / "swinir_ft" / "best_swinir.pt"
    if swinir_x2_weights.exists() and swinir_ckpt.exists() and device != "cpu":
        print("Generating SwinIR finetuned predictions...")
        swinir_model, vmin, vmax = load_swinir_finetuned(swinir_x2_weights, swinir_ckpt, device)
        swinir_preds = predict_swinir_finetuned(
            swinir_model, lr_orig, vmin, vmax, device, batch_size=64
        )
        baselines["SwinIR FT"] = swinir_preds[:n_vis_samples]
        del swinir_model
        torch.cuda.empty_cache()
    else:
        print("  Skipping SwinIR: missing weights or GPU")

    # Flow model
    flow_preds = None
    flow_ensemble = None
    flow_model_dir = pool_dir / noresm_models / "flow-wide96-amp"
    stats_path = flow_model_dir / "norm_stats.pt"

    if flow_model_dir.exists() and device != "cpu":
        print(f"Generating NorESM flow predictions on {device}...")
        norm_stats = load_norm_stats(stats_path)
        flow_mean, flow_all = generate_flow_predictions(
            model_dir=flow_model_dir,
            lr_up=lr_up,
            lr_orig=lr_orig,
            norm_stats=norm_stats,
            device=device,
            hr_size=(64, 64),
            n_samples=n_vis_samples,
            n_ensemble=n_ensemble,
            base_channels=96,
            channel_mults=(1, 2, 4),
            apply_constraint=False,
        )
        flow_preds = torch.from_numpy(flow_mean).unsqueeze(1)
        flow_ensemble = flow_all
        print(f"  Generated {n_vis_samples} samples x {n_ensemble} ensemble members")
    else:
        print("  Skipping flow model: missing weights or GPU")

    all_preds: dict[str, torch.Tensor | np.ndarray] = {}
    all_preds.update(baselines)
    if flow_preds is not None:
        all_preds["Wide96 Flow"] = flow_preds

    _save_sample_figures(lr_orig, hr, all_preds, flow_ensemble, output_dir, n_vis_samples, "noresm")


def _save_sample_figures(
    lr_orig: torch.Tensor,
    hr: torch.Tensor,
    all_preds: dict[str, torch.Tensor | np.ndarray],
    flow_ensemble: np.ndarray | None,
    output_dir: Path,
    n_vis_samples: int,
    prefix: str,
) -> None:
    """Save sample comparison, error, and ensemble figures."""
    for idx in range(min(n_vis_samples, 5)):
        print(f"  Plotting {prefix} sample {idx}...")
        plot_sample_comparison(
            lr=lr_orig[:n_vis_samples],
            hr=hr[:n_vis_samples],
            predictions=all_preds,
            sample_idx=idx,
            output_path=output_dir / f"{prefix}_sample_{idx}_comparison.png",
        )
        plot_error_maps(
            hr=hr[:n_vis_samples],
            predictions=all_preds,
            sample_idx=idx,
            output_path=output_dir / f"{prefix}_sample_{idx}_errors.png",
        )

    if flow_ensemble is not None:
        for idx in range(n_vis_samples):
            plot_ensemble_spread(
                hr=hr[:n_vis_samples],
                ensemble_preds=flow_ensemble,
                sample_idx=idx,
                output_path=output_dir / f"{prefix}_sample_{idx}_ensemble.png",
            )
            print(f"  Saved {prefix} ensemble spread for sample {idx}")

    plt.close("all")
    print(f"{prefix.upper()} sample figures saved to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualization figures")
    parser.add_argument("--pool-dir", type=Path, default=POOL)
    parser.add_argument("--era5-results", type=Path, default=Path("eval_results_8metrics.json"))
    parser.add_argument(
        "--noresm-results", type=Path, default=Path("noresm_eval_results_8metrics.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--n-ensemble", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-dataset metrics ---
    era5_results = None
    noresm_results = None

    if args.era5_results.exists():
        print("=== ERA5 Metrics Figures ===")
        era5_dir = args.output_dir / "era5"
        era5_dir.mkdir(exist_ok=True)
        era5_results = make_metrics_figures(args.era5_results, era5_dir, "ERA5 TCW 4x Downscaling")

    if args.noresm_results.exists():
        print("\n=== NorESM Metrics Figures ===")
        noresm_dir = args.output_dir / "noresm"
        noresm_dir.mkdir(exist_ok=True)
        noresm_results = make_metrics_figures(
            args.noresm_results, noresm_dir, "NorESM TAS 2x Downscaling"
        )

    # --- Dual-dataset comparison ---
    if era5_results and noresm_results:
        print("\n=== Dual-Dataset Comparison Figures ===")
        make_dual_dataset_figures(era5_results, noresm_results, args.output_dir)

    if not args.metrics_only:
        # --- Per-dataset sample figures ---
        if args.era5_results.exists():
            print("\n=== ERA5 Sample Figures ===")
            era5_sample_dir = args.output_dir / "era5"
            era5_sample_dir.mkdir(exist_ok=True)
            make_era5_sample_figures(
                pool_dir=args.pool_dir,
                output_dir=era5_sample_dir,
                device=args.device,
                n_vis_samples=args.n_samples,
                n_ensemble=args.n_ensemble,
            )

        if args.noresm_results.exists():
            print("\n=== NorESM Sample Figures ===")
            noresm_sample_dir = args.output_dir / "noresm"
            noresm_sample_dir.mkdir(exist_ok=True)
            make_noresm_sample_figures(
                pool_dir=args.pool_dir,
                output_dir=noresm_sample_dir,
                device=args.device,
                n_vis_samples=args.n_samples,
                n_ensemble=args.n_ensemble,
            )

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
