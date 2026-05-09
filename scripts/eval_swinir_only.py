"""Run SwinIR evaluation only and merge into existing eval results.

Usage:
    srun --jobid=<JOB_ID> python scripts/eval_swinir_only.py
"""

import json
import time
from pathlib import Path

import torch

from downscaling.data.era5 import load_era5_tcw
from downscaling.evaluation.swinir import eval_swinir_finetuned, eval_swinir_zeroshot

POOL = Path("/home/chenxy/orcd/pool/datasets")
PRETRAINED = POOL / "research5/pretrained_weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
CHECKPOINT = POOL / "spatial-4x-add-v2/models/swinir_ft/best_swinir.pt"
RESULTS_FILE = Path("eval_results_500.json")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading test data...")
    _, _, hr, lr_orig = load_era5_tcw(POOL, "test")
    print(f"  hr={hr.shape}, lr_orig={lr_orig.shape}")

    configs = {
        "swinir-zeroshot": {"mode": "zeroshot", "with_addcl": False},
        "swinir-zeroshot+addcl": {"mode": "zeroshot", "with_addcl": True},
        "swinir-finetuned": {"mode": "finetuned", "with_addcl": False},
        "swinir-finetuned+addcl": {"mode": "finetuned", "with_addcl": True},
    }

    results: dict[str, dict[str, float]] = {}

    for name, cfg in configs.items():
        print(f"\nEvaluating {name}...")
        t0 = time.time()
        if cfg["mode"] == "zeroshot":
            results[name] = eval_swinir_zeroshot(
                hr=hr, lr_orig=lr_orig, weights_path=PRETRAINED,
                device=device, with_addcl=cfg["with_addcl"],
            )
        else:
            results[name] = eval_swinir_finetuned(
                hr=hr, lr_orig=lr_orig, pretrained_weights_path=PRETRAINED,
                checkpoint_path=CHECKPOINT, device=device, with_addcl=cfg["with_addcl"],
            )
        r = results[name]
        elapsed = time.time() - t0
        print(f"  CRPS={r['crps']:.6f}  MAE={r['mae']:.6f}  "
              f"RMSE={r['rmse']:.6f}  MassViol={r['mass_violation']:.6f}  ({elapsed:.1f}s)")

    # Merge into existing results
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            existing = json.load(f)
        existing["results"].update(results)
        with open(RESULTS_FILE, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"\nMerged {len(results)} SwinIR results into {RESULTS_FILE}")
    else:
        print(f"\nWARNING: {RESULTS_FILE} not found, saving standalone")
        with open("eval_results_swinir.json", "w") as f:
            json.dump({"results": results}, f, indent=2)

    # Print comparison table
    print(f"\n{'Method':<30} {'CRPS':>10} {'MAE':>10} {'RMSE':>10} {'MassViol':>10}")
    print("=" * 80)
    for name, r in sorted(results.items(), key=lambda x: x[1]["crps"]):
        print(f"{name:<30} {r['crps']:>10.6f} {r['mae']:>10.6f} "
              f"{r['rmse']:>10.6f} {r['mass_violation']:>10.6f}")


if __name__ == "__main__":
    main()
