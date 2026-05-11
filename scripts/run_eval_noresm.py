"""Evaluate all methods on NorESM TAS 2x downscaling.

Evaluates baselines (bicubic, bilinear), Harder CNN/GAN, SwinIR,
and flow matching models on the NorESM surface temperature dataset.
Mirrors run_eval.py but for 2x SR (32->64) with NorESM-specific
checkpoint paths and pretrained weights.

Computes all 8 metrics: CRPS, MAE, RMSE, mass_violation, RALSD, SSIM, PSNR, EMD.
Also saves spectral curve data (PSD, bias) for plotting.

Usage:
    # Baselines only (CPU)
    python scripts/run_eval_noresm.py --baselines-only

    # Full eval (GPU)
    python scripts/run_eval_noresm.py --max-samples 500

    # All test samples
    python scripts/run_eval_noresm.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from downscaling.constraints.layers import apply_addcl
from downscaling.data.noresm import load_noresm_tas
from downscaling.evaluation.baselines import eval_bicubic, eval_bilinear
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
NORESM_MODELS = POOL / "noresm-dataset" / "models"
UPSAMPLING_FACTOR = 2

SWINIR_X2_WEIGHTS = (
    POOL / "noresm-dataset" / "pretrained_weights" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
)

FLOW_REGISTRY: dict[str, dict[str, object]] = {
    "flow-wide96-amp (28M)": {
        "model_dir": NORESM_MODELS / "flow-wide96-amp",
        "norm_stats": NORESM_MODELS / "flow-wide96-amp" / "norm_stats.pt",
        "base_channels": 96,
        "channel_mults": (1, 2, 4),
    },
}

HARDER_REGISTRY: dict[str, dict[str, object]] = {
    "harder-cnn": {
        "checkpoint": NORESM_MODELS / "harder" / "twc_cnn_none.pth",
        "model_type": "cnn",
        "constraints": "none",
    },
    "harder-cnn+smcl": {
        "checkpoint": NORESM_MODELS / "harder" / "twc_cnn_softmax.pth",
        "model_type": "cnn",
        "constraints": "softmax",
    },
    "harder-gan+smcl": {
        "checkpoint": NORESM_MODELS / "harder" / "twc_gan_softmax.pth",
        "model_type": "gan",
        "constraints": "softmax",
    },
}

SWINIR_REGISTRY: dict[str, dict[str, object]] = {
    "swinir-zeroshot": {"mode": "zeroshot"},
    "swinir-zeroshot+addcl": {"mode": "zeroshot", "with_addcl": True},
    "swinir-finetuned": {
        "mode": "finetuned",
        "checkpoint": NORESM_MODELS / "swinir_ft" / "best_swinir.pt",
    },
    "swinir-finetuned+addcl": {
        "mode": "finetuned",
        "checkpoint": NORESM_MODELS / "swinir_ft" / "best_swinir.pt",
        "with_addcl": True,
    },
}


def _print_result(name: str, r: dict[str, float], elapsed: float | None = None) -> None:
    parts = [f"CRPS={r['crps']:.6f}", f"MAE={r['mae']:.6f}", f"RMSE={r['rmse']:.6f}"]
    if "ralsd" in r:
        parts.append(f"RALSD={r['ralsd']:.2f}dB")
    if "ssim" in r:
        parts.append(f"SSIM={r['ssim']:.4f}")
    if "emd" in r:
        parts.append(f"EMD={r['emd']:.4f}")
    msg = f"  {'  '.join(parts)}"
    if elapsed is not None:
        msg += f"  ({elapsed:.1f}s)"
    print(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all methods on NorESM TAS 2x SR")
    parser.add_argument("--pool-dir", type=Path, default=POOL)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-ensemble", type=int, default=10)
    parser.add_argument("--ode-steps", type=int, default=10)
    # NorESM LR/HR from separate simulations — addcl constraint degrades metrics
    parser.add_argument("--constraint", default="none", choices=["addcl", "smcl", "none"])
    parser.add_argument("--sampler", default="midpoint", choices=["euler", "midpoint"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("noresm_eval_results.json"))
    args = parser.parse_args()

    print(f"Loading NorESM {args.split} data from {args.pool_dir}...")
    t0 = time.time()
    lr_up, residual, hr, lr_orig = load_noresm_tas(args.pool_dir, args.split)
    print(f"  Loaded in {time.time() - t0:.1f}s: lr_orig={lr_orig.shape}, hr={hr.shape}")

    if args.max_samples:
        n = min(args.max_samples, hr.shape[0])
        lr_up, residual, hr, lr_orig = lr_up[:n], residual[:n], hr[:n], lr_orig[:n]
        print(f"  Using {n} samples")

    results: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}  # method -> (N, H, W) for batch metrics

    n_eval = min(hr.shape[0], args.max_samples) if args.max_samples else hr.shape[0]
    gt = hr[:n_eval, 0].numpy()  # (N, H, W) ground truth for batch metrics

    def _add_batch_metrics(name: str) -> None:
        """Compute RALSD/SSIM/PSNR from stored predictions and merge into results."""
        if name in predictions:
            batch = compute_batch_metrics(gt, predictions[name])
            results[name].update(batch)

    def _save_incremental() -> None:
        """Save results after each method to prevent data loss from crashes."""
        with open(args.output, "w") as f:
            json.dump(
                {
                    "dataset": "noresm",
                    "variable": "tas",
                    "upsampling_factor": UPSAMPLING_FACTOR,
                    "split": args.split,
                    "n_samples": n_eval,
                    "n_ensemble": args.n_ensemble,
                    "ode_steps": args.ode_steps,
                    "constraint": args.constraint,
                    "sampler": args.sampler,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"  [incremental save: {len(results)} methods -> {args.output}]")

    # === Baselines ===
    print("\n=== Baselines ===")
    for bname, with_addcl in [
        ("bilinear", False),
        ("bilinear+addcl", True),
        ("bicubic", False),
        ("bicubic+addcl", True),
    ]:
        print(f"Evaluating {bname}...")
        eval_fn = eval_bilinear if "bilinear" in bname else eval_bicubic
        results[bname] = eval_fn(
            hr[:n_eval],
            lr_orig[:n_eval],
            with_addcl=with_addcl,
            upsampling_factor=UPSAMPLING_FACTOR,
        )
        # Generate predictions for batch metrics
        mode = "bilinear" if "bilinear" in bname else "bicubic"
        pred = F.interpolate(
            lr_orig[:n_eval], scale_factor=UPSAMPLING_FACTOR, mode=mode, align_corners=False
        )
        if with_addcl:
            pred = apply_addcl(pred, lr_orig[:n_eval], UPSAMPLING_FACTOR)
        predictions[bname] = pred[:, 0].numpy()
        _add_batch_metrics(bname)
        _print_result(bname, results[bname])
    _save_incremental()

    if args.baselines_only:
        _save_and_print(results, predictions, gt, args, n_eval)
        return

    # === SwinIR Models ===
    print("\n=== SwinIR Models ===")
    for name, config in SWINIR_REGISTRY.items():
        with_addcl = bool(config.get("with_addcl", False))
        t_start = time.time()

        if config["mode"] == "zeroshot":
            if not SWINIR_X2_WEIGHTS.exists():
                print(f"  SKIP {name}: x2 pretrained weights not found at {SWINIR_X2_WEIGHTS}")
                continue
            print(f"Evaluating {name}...")
            try:
                r, preds = eval_swinir_zeroshot(
                    hr=hr[:n_eval],
                    lr_orig=lr_orig[:n_eval],
                    weights_path=SWINIR_X2_WEIGHTS,
                    device=args.device,
                    with_addcl=with_addcl,
                    upsampling_factor=UPSAMPLING_FACTOR,
                    return_predictions=True,
                )
                results[name] = r
                predictions[name] = preds
                _add_batch_metrics(name)
                _print_result(name, results[name], time.time() - t_start)
            except Exception as e:
                print(f"  ERROR {name}: {e}")
        else:
            ckpt_path = Path(str(config["checkpoint"]))
            if not ckpt_path.exists():
                print(f"  SKIP {name}: {ckpt_path} not found")
                continue
            if not SWINIR_X2_WEIGHTS.exists():
                print(f"  SKIP {name}: x2 pretrained weights not found")
                continue
            print(f"Evaluating {name}...")
            try:
                r, preds = eval_swinir_finetuned(
                    hr=hr[:n_eval],
                    lr_orig=lr_orig[:n_eval],
                    pretrained_weights_path=SWINIR_X2_WEIGHTS,
                    checkpoint_path=ckpt_path,
                    device=args.device,
                    with_addcl=with_addcl,
                    upsampling_factor=UPSAMPLING_FACTOR,
                    return_predictions=True,
                )
                results[name] = r
                predictions[name] = preds
                _add_batch_metrics(name)
                _print_result(name, results[name], time.time() - t_start)
            except Exception as e:
                print(f"  ERROR {name}: {e}")
    _save_incremental()

    # === Harder et al. ===
    print("\n=== Harder et al. Baselines ===")
    min_val, max_val = compute_minmax_stats(args.pool_dir, dataset="noresm")
    print(f"  Min-max normalization: min={min_val:.4f}, max={max_val:.4f}")

    for name, config in HARDER_REGISTRY.items():
        ckpt_path = Path(str(config["checkpoint"]))
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
                upsampling_factor=UPSAMPLING_FACTOR,
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
                    upsampling_factor=UPSAMPLING_FACTOR,
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
                    upsampling_factor=UPSAMPLING_FACTOR,
                    return_predictions=True,
                )
            results[name] = r
            predictions[name] = preds
            _add_batch_metrics(name)
            _print_result(name, results[name], time.time() - t_start)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR {name}: {e}")
    _save_incremental()

    # === Flow Matching ===
    print(f"\n=== Flow Matching Models (device={args.device}) ===")
    for name, config in FLOW_REGISTRY.items():
        model_dir = Path(str(config["model_dir"]))
        norm_path = Path(str(config["norm_stats"]))

        if not model_dir.exists():
            print(f"  SKIP {name}: {model_dir} not found")
            continue
        if not norm_path.exists():
            print(f"  SKIP {name}: {norm_path} not found")
            continue

        norm_stats = load_norm_stats(norm_path)
        lr_up_norm = (lr_up - norm_stats["lr_mean"]) / norm_stats["lr_std"]

        print(f"\nEvaluating {name}...")
        t_start = time.time()

        model = AttentionUNet(
            in_channels=2,
            out_channels=1,
            base_channels=int(config["base_channels"]),
            channel_mults=tuple(config["channel_mults"]),  # type: ignore[arg-type]
        ).to(args.device)

        ckpt_path = model_dir / "best_flow.pt"
        if not ckpt_path.exists():
            ckpt_path = model_dir / "flow_best.pth"
        metadata = load_checkpoint(model, ckpt_path, args.device)
        print(f"  Loaded from {ckpt_path.name}", end="")
        if "epoch" in metadata:
            print(f" (epoch {metadata['epoch']})", end="")
        print()

        eval_result, preds = evaluate_flow_model(
            model=model,
            lr_up_norm=lr_up_norm,
            hr=hr,
            lr_up=lr_up,
            lr_orig=lr_orig,
            norm_stats=norm_stats,
            n_ensemble=args.n_ensemble,
            ode_steps=args.ode_steps,
            constraint=args.constraint,
            sampler=args.sampler,
            max_samples=args.max_samples,
            upsampling_factor=UPSAMPLING_FACTOR,
            return_predictions=True,
        )
        results[name] = {
            "crps": eval_result.crps,
            "mae": eval_result.mae,
            "rmse": eval_result.rmse,
            "mass_violation": eval_result.mass_violation,
        }
        predictions[name] = preds
        _add_batch_metrics(name)
        _print_result(name, results[name], time.time() - t_start)
        _save_incremental()
        del model
        torch.cuda.empty_cache()

    _save_and_print(results, predictions, gt, args, n_eval)


def _save_and_print(
    results: dict[str, dict[str, float]],
    predictions: dict[str, np.ndarray],
    gt: np.ndarray,
    args: argparse.Namespace,
    n_samples: int,
) -> None:
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
            elif m in ("ssim", "psnr", "emd"):
                vals.append(f"{v:>10.4f}")
            else:
                vals.append(f"{v:>10.6f}")
        print(f"{name:<40}{''.join(vals)}")
    print(sep)

    # Save JSON results
    with open(args.output, "w") as f:
        json.dump(
            {
                "dataset": "noresm",
                "variable": "tas",
                "upsampling_factor": UPSAMPLING_FACTOR,
                "split": args.split,
                "n_samples": n_samples,
                "n_ensemble": args.n_ensemble,
                "ode_steps": args.ode_steps,
                "constraint": args.constraint,
                "sampler": args.sampler,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {args.output}")

    # Compute and save spectral curves
    if predictions:
        print("\nComputing spectral curves for all methods...")
        spectral_data: dict[str, dict[str, np.ndarray]] = {}
        for name, preds in predictions.items():
            curves = compute_spectral_curves(gt, preds)
            spectral_data[name] = curves
        print(f"  Spectral curves computed for {len(spectral_data)} methods")

        spectral_path = args.output.with_name(args.output.stem + "_spectral.npz")
        flat: dict[str, np.ndarray] = {}
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
