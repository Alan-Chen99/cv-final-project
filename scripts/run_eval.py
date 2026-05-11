"""Comprehensive evaluation of all methods on ERA5 TCW 4x downscaling.

Evaluates baselines (bicubic, bilinear) and trained flow matching models
using correct energy CRPS. Results are printed as a comparison table and
saved to a JSON file.

Usage:
    # Baselines only (no GPU needed)
    python scripts/run_eval.py --baselines-only

    # Full eval with trained models (GPU needed)
    python scripts/run_eval.py --pool-dir /home/chenxy/orcd/pool/datasets

    # Quick eval on subset
    python scripts/run_eval.py --max-samples 200
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from downscaling.constraints.layers import apply_addcl
from downscaling.data.era5 import load_era5_tcw
from downscaling.evaluation.baselines import (
    eval_bicubic,
    eval_bilinear,
    upsample_bicubic,
    upsample_bilinear,
)
from downscaling.evaluation.batch_metrics import compute_batch_metrics, compute_spectral_curves
from downscaling.evaluation.checkpoints import load_checkpoint, load_norm_stats
from downscaling.evaluation.evaluate import evaluate_flow_model
from downscaling.evaluation.harder import (
    compute_minmax_stats,
    evaluate_harder_cnn,
    evaluate_harder_gan,
    load_harder_model,
)
from downscaling.evaluation.swinir import eval_swinir_finetuned, eval_swinir_zeroshot
from downscaling.models.unet import AttentionUNet

POOL = Path("/home/chenxy/orcd/pool/datasets")

SWINIR_PRETRAINED_WEIGHTS = (
    POOL / "research5" / "pretrained_weights" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
)

# SwinIR models: name -> config
SWINIR_REGISTRY: dict[str, dict[str, object]] = {
    "swinir-zeroshot": {
        "mode": "zeroshot",
    },
    "swinir-zeroshot+addcl": {
        "mode": "zeroshot",
        "with_addcl": True,
    },
    "swinir-finetuned": {
        "mode": "finetuned",
        "checkpoint": "spatial-4x-add-v2/models/swinir_ft/best_swinir.pt",
    },
    "swinir-finetuned+addcl": {
        "mode": "finetuned",
        "checkpoint": "spatial-4x-add-v2/models/swinir_ft/best_swinir.pt",
        "with_addcl": True,
    },
}


def eval_flow_matching_model(
    name: str,
    model_dir: Path,
    lr_up_norm: torch.Tensor,
    hr: torch.Tensor,
    lr_up: torch.Tensor,
    lr_orig: torch.Tensor,
    norm_stats: dict[str, float],
    device: str,
    n_ensemble: int = 10,
    ode_steps: int = 10,
    constraint: str = "addcl",
    sampler: str = "midpoint",
    max_samples: int | None = None,
    base_channels: int = 64,
    channel_mults: tuple[int, ...] = (1, 2, 4),
) -> tuple[dict[str, float], np.ndarray]:
    """Load and evaluate a flow matching model.

    Returns:
        Tuple of (metrics_dict, predictions) where predictions has shape (N, H, W).
    """
    model = AttentionUNet(
        in_channels=2,
        out_channels=1,
        base_channels=base_channels,
        channel_mults=channel_mults,
    ).to(device)

    # Find checkpoint
    ckpt_path = None
    for fname in ["best_flow.pt", "flow_best.pth"]:
        candidate = model_dir / fname
        if candidate.exists():
            ckpt_path = candidate
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")

    metadata = load_checkpoint(model, ckpt_path, device)
    print(f"  Loaded {name} from {ckpt_path.name}", end="")
    if "epoch" in metadata:
        print(f" (epoch {metadata['epoch']})", end="")
    print()

    result, preds = evaluate_flow_model(
        model=model,
        lr_up_norm=lr_up_norm,
        hr=hr,
        lr_up=lr_up,
        lr_orig=lr_orig,
        norm_stats=norm_stats,
        n_ensemble=n_ensemble,
        ode_steps=ode_steps,
        constraint=constraint,
        sampler=sampler,
        max_samples=max_samples,
        return_predictions=True,
    )
    return {
        "crps": result.crps,
        "mae": result.mae,
        "rmse": result.rmse,
        "mass_violation": result.mass_violation,
    }, preds


# Model registry: name -> config
# All use AttentionUNet with in_channels=2, out_channels=1
MODEL_REGISTRY: dict[str, dict[str, object]] = {
    "flow-wide96-amp (28M)": {
        "model_dir": "research3/models/unet_wide96_amp",
        "norm_stats": "research3/models/unet_wide96_amp/norm_stats.pt",
        "base_channels": 96,
        "channel_mults": (1, 2, 4),
    },
    "flow-uniform-amp (13M)": {
        "model_dir": "research3/models/unet_uniform_amp",
        "norm_stats": "research3/models/unet_uniform_amp/norm_stats.pt",
        "base_channels": 64,
        "channel_mults": (1, 2, 4),
    },
    "flow-logitnorm-ema (13M)": {
        "model_dir": "research3/models/unet_ema_logitnorm",
        "norm_stats": "research3/models/unet_ema_logitnorm/norm_stats.pt",
        "base_channels": 64,
        "channel_mults": (1, 2, 4),
    },
    "flow-v2-zscore (13M)": {
        "model_dir": "research6/models/flow_v2_zscore",
        "norm_stats": "research6/models/flow_v2_zscore/norm_stats.pt",
        "base_channels": 64,
        "channel_mults": (1, 2, 4),
    },
}

# Harder et al. (2208.05424) baselines
# Checkpoints stored in pool under organize2/models/harder/ after training
HARDER_REGISTRY: dict[str, dict[str, object]] = {
    "harder-cnn": {
        "checkpoint": "organize2/models/harder/twc_cnn_none.pth",
        "model_type": "cnn",
        "constraints": "none",
    },
    "harder-cnn+smcl": {
        "checkpoint": "organize2/models/harder/twc_cnn_softmax.pth",
        "model_type": "cnn",
        "constraints": "softmax",
    },
    "harder-gan+smcl": {
        "checkpoint": "organize2/models/harder/twc_gan_softmax.pth",
        "model_type": "gan",
        "constraints": "softmax",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate all methods on ERA5 TCW 4x SR")
    parser.add_argument("--pool-dir", type=Path, default=POOL)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-ensemble", type=int, default=10)
    parser.add_argument("--ode-steps", type=int, default=10)
    parser.add_argument("--constraint", default="addcl", choices=["addcl", "smcl", "none"])
    parser.add_argument("--sampler", default="midpoint", choices=["euler", "midpoint"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    print(f"Loading {args.split} data from {args.pool_dir}...")
    t0 = time.time()
    lr_up, residual, hr, lr_orig = load_era5_tcw(args.pool_dir, args.split)
    print(f"  Loaded in {time.time() - t0:.1f}s: lr_orig={lr_orig.shape}, hr={hr.shape}")

    if args.max_samples:
        n = min(args.max_samples, hr.shape[0])
        lr_up, residual, hr, lr_orig = lr_up[:n], residual[:n], hr[:n], lr_orig[:n]
        print(f"  Using {n} samples")

    results: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}  # method -> (N, H, W) for batch metrics

    n_eval = min(hr.shape[0], args.max_samples) if args.max_samples else hr.shape[0]
    gt = hr[:n_eval, 0].numpy()  # (N, H, W) ground truth for batch metrics

    def _print_metrics(name: str) -> None:
        r = results[name]
        parts = [f"CRPS={r['crps']:.6f}", f"MAE={r['mae']:.6f}", f"RMSE={r['rmse']:.6f}"]
        if "ralsd" in r:
            parts.append(f"RALSD={r['ralsd']:.2f}dB")
        if "ssim" in r:
            parts.append(f"SSIM={r['ssim']:.4f}")
        if "emd" in r:
            parts.append(f"EMD={r['emd']:.4f}")
        print(f"  {'  '.join(parts)}")

    def _add_batch_metrics(name: str) -> None:
        """Compute RALSD/SSIM/PSNR from stored predictions and merge into results."""
        if name in predictions:
            batch = compute_batch_metrics(gt, predictions[name])
            results[name].update(batch)

    def _save_incremental() -> None:
        """Save results after each method to prevent data loss from crashes."""
        output_path = args.output or Path("eval_results.json")
        with open(output_path, "w") as f:
            json.dump(
                {
                    "split": args.split,
                    "n_samples": int(n_eval),
                    "n_ensemble": args.n_ensemble,
                    "ode_steps": args.ode_steps,
                    "constraint": args.constraint,
                    "sampler": args.sampler,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"  [incremental save: {len(results)} methods -> {output_path}]")

    # Baselines — cheap to recompute predictions for batch metrics
    print("\n=== Baselines ===")

    for bname, upsample_fn, with_addcl in [
        ("bilinear", upsample_bilinear, False),
        ("bilinear+addcl", upsample_bilinear, True),
        ("bicubic", upsample_bicubic, False),
        ("bicubic+addcl", upsample_bicubic, True),
    ]:
        print(f"Evaluating {bname}...")
        eval_fn = eval_bilinear if "bilinear" in bname else eval_bicubic
        results[bname] = eval_fn(hr[:n_eval], lr_orig[:n_eval], with_addcl=with_addcl)
        # Generate predictions for batch metrics
        pred = upsample_fn(lr_orig[:n_eval])
        if with_addcl:
            pred = apply_addcl(pred, lr_orig[:n_eval], 4)
        predictions[bname] = pred[:, 0].numpy()
        _add_batch_metrics(bname)
        _print_metrics(bname)
    _save_incremental()

    # SwinIR models
    if not args.baselines_only:
        print("\n=== SwinIR Models ===")
        for name, config in SWINIR_REGISTRY.items():
            with_addcl = bool(config.get("with_addcl", False))
            if config["mode"] == "zeroshot":
                if not SWINIR_PRETRAINED_WEIGHTS.exists():
                    print(f"  SKIP {name}: pretrained weights not found")
                    continue
                print(f"Evaluating {name}...")
                t_start = time.time()
                try:
                    r, preds = eval_swinir_zeroshot(
                        hr=hr[:n_eval],
                        lr_orig=lr_orig[:n_eval],
                        weights_path=SWINIR_PRETRAINED_WEIGHTS,
                        device=args.device,
                        with_addcl=with_addcl,
                        return_predictions=True,
                    )
                    results[name] = r
                    predictions[name] = preds
                    _add_batch_metrics(name)
                    elapsed = time.time() - t_start
                    _print_metrics(name)
                    print(f"  ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"  ERROR {name}: {e}")
            else:
                ckpt_path = args.pool_dir / str(config["checkpoint"])
                if not ckpt_path.exists():
                    print(f"  SKIP {name}: {ckpt_path} not found")
                    continue
                print(f"Evaluating {name}...")
                t_start = time.time()
                try:
                    r, preds = eval_swinir_finetuned(
                        hr=hr[:n_eval],
                        lr_orig=lr_orig[:n_eval],
                        pretrained_weights_path=SWINIR_PRETRAINED_WEIGHTS,
                        checkpoint_path=ckpt_path,
                        device=args.device,
                        with_addcl=with_addcl,
                        return_predictions=True,
                    )
                    results[name] = r
                    predictions[name] = preds
                    _add_batch_metrics(name)
                    elapsed = time.time() - t_start
                    _print_metrics(name)
                    print(f"  ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"  ERROR {name}: {e}")
        _save_incremental()

    # Harder et al. trained models
    if not args.baselines_only:
        print("\n=== Harder et al. Baselines ===")
        min_val, max_val = compute_minmax_stats(args.pool_dir)
        print(f"  Min-max normalization: min={min_val:.4f}, max={max_val:.4f}")

        for name, config in HARDER_REGISTRY.items():
            ckpt_path = args.pool_dir / str(config["checkpoint"])
            if not ckpt_path.exists():
                print(f"  SKIP {name}: {ckpt_path} not found")
                continue

            print(f"\nEvaluating {name}...")
            t_start = time.time()
            try:
                model = load_harder_model(
                    checkpoint_path=ckpt_path,
                    model_type=str(config["model_type"]),
                    constraints=str(config["constraints"]),
                    device=args.device,
                )
                if config["model_type"] == "gan":
                    r, preds = evaluate_harder_gan(
                        model=model,
                        lr_orig=lr_orig[:n_eval],
                        hr=hr[:n_eval],
                        min_val=min_val,
                        max_val=max_val,
                        device=args.device,
                        n_ensemble=args.n_ensemble,
                        max_samples=args.max_samples,
                        return_predictions=True,
                    )
                else:
                    r, preds = evaluate_harder_cnn(
                        model=model,
                        lr_orig=lr_orig[:n_eval],
                        hr=hr[:n_eval],
                        min_val=min_val,
                        max_val=max_val,
                        device=args.device,
                        max_samples=args.max_samples,
                        return_predictions=True,
                    )
                results[name] = r
                predictions[name] = preds
                _add_batch_metrics(name)
                elapsed = time.time() - t_start
                _print_metrics(name)
                print(f"  ({elapsed:.1f}s)")
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ERROR {name}: {e}")
        _save_incremental()

    if not args.baselines_only:
        print(f"\n=== Flow Matching Models (device={args.device}) ===")

        # Compute normalized LR once
        # Need norm stats for normalization — load from first available model
        first_model = next(iter(MODEL_REGISTRY.values()))
        stats_path = args.pool_dir / str(first_model["norm_stats"])
        if stats_path.exists():
            ref_stats = load_norm_stats(stats_path)
            lr_up_norm = (lr_up - ref_stats["lr_mean"]) / ref_stats["lr_std"]
        else:
            print(f"  WARNING: norm_stats not found at {stats_path}, skipping models")
            lr_up_norm = None

        if lr_up_norm is not None:
            for name, config in MODEL_REGISTRY.items():
                model_dir = args.pool_dir / str(config["model_dir"])
                norm_path = args.pool_dir / str(config["norm_stats"])

                if not model_dir.exists():
                    print(f"  SKIP {name}: {model_dir} not found")
                    continue
                if not norm_path.exists():
                    print(f"  SKIP {name}: {norm_path} not found")
                    continue

                norm_stats = load_norm_stats(norm_path)
                # Re-normalize with this model's stats
                model_lr_up_norm = (lr_up - norm_stats["lr_mean"]) / norm_stats["lr_std"]

                print(f"\nEvaluating {name}...")
                t_start = time.time()
                try:
                    r, preds = eval_flow_matching_model(
                        name=name,
                        model_dir=model_dir,
                        lr_up_norm=model_lr_up_norm,
                        hr=hr,
                        lr_up=lr_up,
                        lr_orig=lr_orig,
                        norm_stats=norm_stats,
                        device=args.device,
                        n_ensemble=args.n_ensemble,
                        ode_steps=args.ode_steps,
                        constraint=args.constraint,
                        sampler=args.sampler,
                        max_samples=args.max_samples,
                        base_channels=int(config["base_channels"]),
                        channel_mults=tuple(config["channel_mults"]),  # type: ignore[arg-type]
                    )
                    results[name] = r
                    predictions[name] = preds
                    _add_batch_metrics(name)
                    elapsed = time.time() - t_start
                    _print_metrics(name)
                    print(f"  ({elapsed:.1f}s)")
                    _save_incremental()
                except Exception as e:
                    print(f"  ERROR {name}: {e}")

    # Print comparison table
    all_metrics = ["crps", "mae", "rmse", "mass_violation", "ralsd", "ssim", "psnr", "emd"]
    has_batch = any("ralsd" in r for r in results.values())
    display_metrics = all_metrics if has_batch else ["crps", "mae", "rmse", "mass_violation"]

    header = f"{'Method':<40}" + "".join(f" {m:>10}" for m in display_metrics)
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, r in sorted(results.items(), key=lambda x: x[1]["crps"]):
        vals = []
        for m in display_metrics:
            v = r.get(m)
            if v is None:
                vals.append(f"{'N/A':>10}")
            elif m == "ralsd":
                vals.append(f"{v:>9.2f}dB")
            elif m in ("ssim", "psnr"):
                vals.append(f"{v:>10.4f}")
            elif m == "emd":
                vals.append(f"{v:>10.4f}")
            else:
                vals.append(f"{v:>10.6f}")
        print(f"{name:<40}{''.join(vals)}")
    print(sep)

    # Compute spectral curves for plotting
    spectral_data: dict[str, dict[str, np.ndarray]] = {}
    if predictions:
        print("\nComputing spectral curves for all methods...")
        for name, preds in predictions.items():
            curves = compute_spectral_curves(gt, preds)
            spectral_data[name] = curves
        print(f"  Spectral curves computed for {len(spectral_data)} methods")

    # Save results
    output_path = args.output or Path("eval_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "split": args.split,
                "n_samples": int(n_eval),
                "n_ensemble": args.n_ensemble,
                "ode_steps": args.ode_steps,
                "constraint": args.constraint,
                "sampler": args.sampler,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")

    # Save spectral curves as .npz (numpy arrays can't go in JSON)
    if spectral_data:
        spectral_path = output_path.with_name(output_path.stem + "_spectral.npz")
        flat: dict[str, np.ndarray] = {}
        # Use first method's freq (all share same bins)
        first_method = next(iter(spectral_data.values()))
        flat["freq"] = first_method["freq"]
        flat["psd_truth"] = first_method["psd_truth"]
        for name, curves in spectral_data.items():
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            flat[f"psd_{safe_name}"] = curves["psd_pred"]
            flat[f"bias_{safe_name}"] = curves["bias"]
        np.savez(spectral_path, **flat)
        print(f"Spectral data saved to {spectral_path}")


if __name__ == "__main__":
    main()
