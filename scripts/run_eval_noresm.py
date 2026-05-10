"""Evaluate all methods on NorESM TAS 2x downscaling.

Evaluates baselines (bicubic, bilinear), Harder CNN/GAN, SwinIR,
and flow matching models on the NorESM surface temperature dataset.
Mirrors run_eval.py but for 2x SR (32->64) with NorESM-specific
checkpoint paths and pretrained weights.

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

import torch

from downscaling.data.noresm import load_noresm_tas
from downscaling.evaluation.baselines import eval_bicubic, eval_bilinear
from downscaling.evaluation.checkpoints import load_checkpoint, load_norm_stats
from downscaling.evaluation.evaluate import evaluate_flow_model
from downscaling.evaluation.harder import (
    _compute_minmax_stats,
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
    msg = (
        f"  CRPS={r['crps']:.6f}  MAE={r['mae']:.6f}  "
        f"RMSE={r['rmse']:.6f}  MassViol={r['mass_violation']:.6f}"
    )
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

    # === Baselines ===
    print("\n=== Baselines ===")
    for name, with_addcl in [("bilinear", False), ("bilinear+addcl", True)]:
        print(f"Evaluating {name}...")
        results[name] = eval_bilinear(
            hr, lr_orig, with_addcl=with_addcl, upsampling_factor=UPSAMPLING_FACTOR
        )
        _print_result(name, results[name])

    for name, with_addcl in [("bicubic", False), ("bicubic+addcl", True)]:
        print(f"Evaluating {name}...")
        results[name] = eval_bicubic(
            hr, lr_orig, with_addcl=with_addcl, upsampling_factor=UPSAMPLING_FACTOR
        )
        _print_result(name, results[name])

    if args.baselines_only:
        _save_and_print(results, args, hr.shape[0])
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
            results[name] = eval_swinir_zeroshot(
                hr=hr,
                lr_orig=lr_orig,
                weights_path=SWINIR_X2_WEIGHTS,
                device=args.device,
                with_addcl=with_addcl,
                upsampling_factor=UPSAMPLING_FACTOR,
            )
        else:
            ckpt_path = Path(str(config["checkpoint"]))
            if not ckpt_path.exists():
                print(f"  SKIP {name}: {ckpt_path} not found")
                continue
            if not SWINIR_X2_WEIGHTS.exists():
                print(f"  SKIP {name}: x2 pretrained weights not found")
                continue
            print(f"Evaluating {name}...")
            results[name] = eval_swinir_finetuned(
                hr=hr,
                lr_orig=lr_orig,
                pretrained_weights_path=SWINIR_X2_WEIGHTS,
                checkpoint_path=ckpt_path,
                device=args.device,
                with_addcl=with_addcl,
                upsampling_factor=UPSAMPLING_FACTOR,
            )

        _print_result(name, results[name], time.time() - t_start)

    # === Harder et al. ===
    print("\n=== Harder et al. Baselines ===")
    min_val, max_val = _compute_minmax_stats(args.pool_dir, dataset="noresm")
    print(f"  Min-max normalization: min={min_val:.4f}, max={max_val:.4f}")

    for name, config in HARDER_REGISTRY.items():
        ckpt_path = Path(str(config["checkpoint"]))
        if not ckpt_path.exists():
            print(f"  SKIP {name}: {ckpt_path} not found")
            continue

        print(f"\nEvaluating {name}...")
        t_start = time.time()
        model = load_harder_model(
            checkpoint_path=ckpt_path,
            model_type=str(config["model_type"]),
            constraints=str(config["constraints"]),
            device=args.device,
            upsampling_factor=UPSAMPLING_FACTOR,
        )
        if config["model_type"] == "gan":
            results[name] = evaluate_harder_gan(
                model=model,
                lr_orig=lr_orig,
                hr=hr,
                min_val=min_val,
                max_val=max_val,
                device=args.device,
                n_ensemble=args.n_ensemble,
                max_samples=args.max_samples,
                upsampling_factor=UPSAMPLING_FACTOR,
            )
        else:
            results[name] = evaluate_harder_cnn(
                model=model,
                lr_orig=lr_orig,
                hr=hr,
                min_val=min_val,
                max_val=max_val,
                device=args.device,
                max_samples=args.max_samples,
                upsampling_factor=UPSAMPLING_FACTOR,
            )
        _print_result(name, results[name], time.time() - t_start)
        del model
        torch.cuda.empty_cache()

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

        eval_result = evaluate_flow_model(
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
        )
        results[name] = {
            "crps": eval_result.crps,
            "mae": eval_result.mae,
            "rmse": eval_result.rmse,
            "mass_violation": eval_result.mass_violation,
        }
        _print_result(name, results[name], time.time() - t_start)
        del model
        torch.cuda.empty_cache()

    _save_and_print(results, args, hr.shape[0])


def _save_and_print(
    results: dict[str, dict[str, float]], args: argparse.Namespace, n_samples: int
) -> None:
    print(f"\n{'=' * 85}")
    print(f"{'Method':<40} {'CRPS':>10} {'MAE':>10} {'RMSE':>10} {'MassViol':>10}")
    print(f"{'=' * 85}")
    for name, r in sorted(results.items(), key=lambda x: x[1]["crps"]):
        print(
            f"{name:<40} {r['crps']:>10.6f} {r['mae']:>10.6f} {r['rmse']:>10.6f} {r['mass_violation']:>10.6f}"
        )
    print(f"{'=' * 85}")

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


if __name__ == "__main__":
    main()
