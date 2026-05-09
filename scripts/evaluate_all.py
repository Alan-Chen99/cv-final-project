"""Comprehensive evaluation of all downscaling methods.

Runs deterministic baselines and all available flow matching models through
the canonical evaluation pipeline. Produces a results table and saves to JSON.

Usage:
    python scripts/evaluate_all.py                    # CPU-only baselines
    python scripts/evaluate_all.py --flow              # include flow models (needs GPU)
    python scripts/evaluate_all.py --flow --max-samples 100  # quick test run
"""

import argparse
import json
import time
from pathlib import Path

import torch

from downscaling.evaluation import (
    EvalResult,
    bicubic_predict,
    bilinear_predict,
    evaluate_deterministic,
    evaluate_flow_model,
    load_flow_checkpoint,
)

POOL = Path("/home/chenxy/orcd/pool/datasets")
BASEDIR = "external/constrained-downscaling"

# Flow matching models to evaluate, keyed by display name
FLOW_MODELS = {
    "wide96 (research3)": {
        "checkpoint": POOL / "research3/models/unet_wide96_amp/best_flow.pt",
        "norm_stats": POOL / "research3/models/unet_wide96_amp/norm_stats.pt",
    },
    "base64 zscore (research6)": {
        "checkpoint": POOL / "research6/models/flow_v2_zscore/best_flow.pt",
        "norm_stats": POOL / "research6/models/flow_v2_zscore/norm_stats.pt",
    },
    "base64 uniform (research3)": {
        "checkpoint": POOL / "research3/models/unet_uniform_amp/best_flow.pt",
        "norm_stats": POOL / "research3/models/unet_uniform_amp/norm_stats.pt",
    },
    "logit-normal (research4)": {
        "checkpoint": POOL / "research4/models/unet_logit_normal_best.pt",
        "norm_stats": POOL / "research4/models/unet_logit_normal_norm_stats.pt",
    },
    "cfg (research4)": {
        "checkpoint": POOL / "research4/models/unet_cfg_best.pt",
        "norm_stats": POOL / "research4/models/unet_cfg_norm_stats.pt",
    },
}


def format_results_table(results: list[EvalResult]) -> str:
    """Format results as a readable table."""
    header = f"{'Method':<35} {'Constraint':<10} {'CRPS':>8} {'MAE':>8} {'RMSE':>8} {'MassViol':>10} {'Spread':>8} {'N':>5}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        lines.append(
            f"{r.method:<35} {r.constraint:<10} {r.crps:>8.4f} {r.mae:>8.4f} {r.rmse:>8.4f} "
            f"{r.mass_violation:>10.6f} {r.spread:>8.4f} {r.n_samples:>5}"
        )
    return "\n".join(lines)


def results_to_dicts(results: list[EvalResult]) -> list[dict]:
    """Convert results to serializable dicts."""
    return [
        {
            "method": r.method,
            "constraint": r.constraint,
            "crps": r.crps,
            "crps_paper": r.crps_paper,
            "mae": r.mae,
            "rmse": r.rmse,
            "mass_violation": r.mass_violation,
            "spread": r.spread,
            "n_samples": r.n_samples,
            "n_ensemble": r.n_ensemble,
        }
        for r in results
    ]


def run_deterministic_baselines(
    max_samples: int | None = None,
) -> list[EvalResult]:
    """Run all deterministic baselines with and without constraints."""
    results = []
    methods = [
        ("bilinear", bilinear_predict),
        ("bicubic", bicubic_predict),
    ]
    # SmCL uses exp() which overflows on physical-space values (TCW 0-135)
    constraints = ["none", "addcl"]

    for method_name, predict_fn in methods:
        for constraint in constraints:
            print(f"  {method_name} + {constraint}...", end=" ", flush=True)
            t0 = time.time()
            r = evaluate_deterministic(
                predict_fn,
                basedir=BASEDIR,
                split="test",
                constraint=constraint,
                max_samples=max_samples,
                method_name=method_name,
            )
            dt = time.time() - t0
            print(f"CRPS={r.crps:.4f} ({dt:.1f}s)")
            results.append(r)
    return results


def run_flow_models(
    max_samples: int | None = None,
    n_ensemble: int = 10,
    ode_steps: int = 10,
    sampler: str = "midpoint",
    device: str = "cuda",
) -> list[EvalResult]:
    """Load and evaluate all available flow matching models."""
    results = []
    constraints = ["none", "addcl"]

    for name, paths in FLOW_MODELS.items():
        if not paths["checkpoint"].exists():
            print(f"  SKIP {name}: checkpoint not found at {paths['checkpoint']}")
            continue
        if not paths["norm_stats"].exists():
            print(f"  SKIP {name}: norm_stats not found at {paths['norm_stats']}")
            continue

        print(f"  Loading {name}...")
        model, stats = load_flow_checkpoint(
            str(paths["checkpoint"]),
            str(paths["norm_stats"]),
            device=device,
        )

        for constraint in constraints:
            print(f"    {name} + {constraint}...", end=" ", flush=True)
            t0 = time.time()
            r = evaluate_flow_model(
                model,
                stats,
                basedir=BASEDIR,
                split="test",
                n_ensemble=n_ensemble,
                ode_steps=ode_steps,
                sampler=sampler,
                constraint=constraint,
                max_samples=max_samples,
                device=device,
                method_name=name,
            )
            dt = time.time() - t0
            print(f"CRPS={r.crps:.4f} ({dt:.1f}s)")
            results.append(r)

        del model
        torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all downscaling methods")
    parser.add_argument(
        "--flow", action="store_true", help="Include flow matching models (needs GPU)"
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit test samples")
    parser.add_argument("--n-ensemble", type=int, default=10, help="Ensemble size for flow models")
    parser.add_argument("--ode-steps", type=int, default=10, help="ODE solver steps")
    parser.add_argument("--sampler", default="midpoint", choices=["euler", "midpoint"])
    parser.add_argument("--output", default="results/eval_results.json", help="Output JSON path")
    args = parser.parse_args()

    all_results: list[EvalResult] = []

    print("=== Deterministic Baselines ===")
    all_results.extend(run_deterministic_baselines(max_samples=args.max_samples))

    if args.flow:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n=== Flow Matching Models (device={device}) ===")
        all_results.extend(
            run_flow_models(
                max_samples=args.max_samples,
                n_ensemble=args.n_ensemble,
                ode_steps=args.ode_steps,
                sampler=args.sampler,
                device=device,
            )
        )

    print("\n" + "=" * 100)
    print(format_results_table(all_results))
    print("=" * 100)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "max_samples": args.max_samples,
                    "n_ensemble": args.n_ensemble,
                    "ode_steps": args.ode_steps,
                    "sampler": args.sampler,
                },
                "results": results_to_dicts(all_results),
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
