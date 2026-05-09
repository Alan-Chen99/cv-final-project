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

import torch

from downscaling.data.era5 import load_era5_tcw
from downscaling.evaluation.baselines import eval_bicubic, eval_bilinear
from downscaling.evaluation.checkpoints import load_checkpoint, load_norm_stats
from downscaling.evaluation.evaluate import evaluate_flow_model
from downscaling.models.unet import AttentionUNet

POOL = Path("/home/chenxy/orcd/pool/datasets")


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
) -> dict[str, float]:
    """Load and evaluate a flow matching model."""
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

    result = evaluate_flow_model(
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
    )
    return {
        "crps": result.crps,
        "mae": result.mae,
        "rmse": result.rmse,
        "mass_violation": result.mass_violation,
    }


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

    # Baselines
    print("\n=== Baselines ===")

    print("Evaluating bilinear...")
    results["bilinear"] = eval_bilinear(hr, lr_orig)
    print(
        f"  CRPS={results['bilinear']['crps']:.6f}  MAE={results['bilinear']['mae']:.6f}  "
        f"RMSE={results['bilinear']['rmse']:.6f}  MassViol={results['bilinear']['mass_violation']:.6f}"
    )

    print("Evaluating bilinear + AddCL...")
    results["bilinear+addcl"] = eval_bilinear(hr, lr_orig, with_addcl=True)
    print(
        f"  CRPS={results['bilinear+addcl']['crps']:.6f}  MAE={results['bilinear+addcl']['mae']:.6f}  "
        f"RMSE={results['bilinear+addcl']['rmse']:.6f}  MassViol={results['bilinear+addcl']['mass_violation']:.6f}"
    )

    print("Evaluating bicubic...")
    results["bicubic"] = eval_bicubic(hr, lr_orig)
    print(
        f"  CRPS={results['bicubic']['crps']:.6f}  MAE={results['bicubic']['mae']:.6f}  "
        f"RMSE={results['bicubic']['rmse']:.6f}  MassViol={results['bicubic']['mass_violation']:.6f}"
    )

    print("Evaluating bicubic + AddCL...")
    results["bicubic+addcl"] = eval_bicubic(hr, lr_orig, with_addcl=True)
    print(
        f"  CRPS={results['bicubic+addcl']['crps']:.6f}  MAE={results['bicubic+addcl']['mae']:.6f}  "
        f"RMSE={results['bicubic+addcl']['rmse']:.6f}  MassViol={results['bicubic+addcl']['mass_violation']:.6f}"
    )

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
                    results[name] = eval_flow_matching_model(
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
                    elapsed = time.time() - t_start
                    r = results[name]
                    print(
                        f"  CRPS={r['crps']:.6f}  MAE={r['mae']:.6f}  "
                        f"RMSE={r['rmse']:.6f}  MassViol={r['mass_violation']:.6f}  "
                        f"({elapsed:.1f}s)"
                    )
                except Exception as e:
                    print(f"  ERROR {name}: {e}")

    # Print comparison table
    print(f"\n{'=' * 85}")
    print(f"{'Method':<40} {'CRPS':>10} {'MAE':>10} {'RMSE':>10} {'MassViol':>10}")
    print(f"{'=' * 85}")
    for name, r in sorted(results.items(), key=lambda x: x[1]["crps"]):
        print(
            f"{name:<40} {r['crps']:>10.6f} {r['mae']:>10.6f} {r['rmse']:>10.6f} {r['mass_violation']:>10.6f}"
        )
    print(f"{'=' * 85}")

    # Save results
    output_path = args.output or Path("eval_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "split": args.split,
                "n_samples": hr.shape[0],
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


if __name__ == "__main__":
    main()
