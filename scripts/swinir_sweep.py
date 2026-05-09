"""Quick hyperparameter sweep for SwinIR finetuning.

Runs 5-epoch short training runs with different hyperparameters to find
the best configuration before committing to a 2hr training run.

Usage:
    python scripts/swinir_sweep.py
"""

import json
import time
from pathlib import Path

from downscaling.training.swinir import train_swinir

POOL = Path("/home/chenxy/orcd/pool/datasets")
SAVE_DIR = Path("/home/chenxy/orcd/pool/datasets/spatial-4x-add-v2/models/swinir_ft")
PRETRAINED = POOL / "research5" / "pretrained_weights" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"

SWEEP_CONFIGS = [
    {"tag": "sweep-lr2e4-bs64", "lr": 2e-4, "batch_size": 64, "freeze_backbone": False},
    {"tag": "sweep-lr1e4-bs64", "lr": 1e-4, "batch_size": 64, "freeze_backbone": False},
    {"tag": "sweep-lr5e5-bs64", "lr": 5e-5, "batch_size": 64, "freeze_backbone": False},
    {"tag": "sweep-lr2e4-frozen", "lr": 2e-4, "batch_size": 64, "freeze_backbone": True},
    {"tag": "sweep-lr1e3-frozen", "lr": 1e-3, "batch_size": 64, "freeze_backbone": True},
]

SWEEP_EPOCHS = 5
SWEEP_WALL_HOURS = 0.25  # 15 min per config max


def main() -> None:
    results = []
    for cfg in SWEEP_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Sweep config: {cfg['tag']}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            result = train_swinir(
                pool_dir=POOL,
                save_dir=SAVE_DIR,
                pretrained_weights=PRETRAINED,
                epochs=SWEEP_EPOCHS,
                batch_size=cfg["batch_size"],
                lr=cfg["lr"],
                freeze_backbone=cfg["freeze_backbone"],
                wall_hours=SWEEP_WALL_HOURS,
                tag=cfg["tag"],
            )
            elapsed = time.time() - t0
            results.append({
                "config": cfg,
                "best_val_loss": result["best_val_loss"],
                "epochs_trained": result["epochs_trained"],
                "elapsed_min": elapsed / 60,
            })
            print(f"  => best_val_loss={result['best_val_loss']:.6f} in {elapsed/60:.1f}min")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"config": cfg, "error": str(e)})

    # Print summary
    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: x.get("best_val_loss", float("inf"))):
        tag = r["config"]["tag"]
        if "error" in r:
            print(f"  {tag}: ERROR - {r['error']}")
        else:
            print(f"  {tag}: val_loss={r['best_val_loss']:.6f} ({r['epochs_trained']} epochs, {r['elapsed_min']:.1f}min)")

    # Save results
    out_path = SAVE_DIR / "sweep_results.json"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
