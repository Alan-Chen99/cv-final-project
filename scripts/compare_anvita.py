"""Compare Anvita checkpoints (6h training) vs ours on matching model configs.

Evaluates CNN/GAN with none/softmax constraints on both NorESM and ERA5 test sets.
Outputs a side-by-side metrics comparison table.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from downscaling.constraints.layers import apply_addcl
from downscaling.data import load_noresm_tas
from downscaling.evaluation.comprehensive import (
    ERA5_UPSAMPLING_FACTOR,
    POOL,
    UPSAMPLING_FACTOR,
    compute_all_metrics,
    generate_cnn_predictions,
    generate_gan_predictions,
)
from downscaling.evaluation.harder import compute_minmax_stats, load_harder_model

ANVITA = Path("/orcd/pool/007/chenxy/datasets/anvita")
NORESM_MODELS = POOL / "noresm-dataset" / "models" / "harder"
ERA5_MODELS = POOL / "organize2" / "models" / "harder"

# (label, ours_path, anvita_path, model_type, constraints, dataset)
COMPARISONS = [
    # NorESM
    ("CNN (none)", NORESM_MODELS / "twc_cnn_none.pth", ANVITA / "noresm_cnn_none.pth", "cnn", "none", "noresm"),
    ("CNN (SmCL)", NORESM_MODELS / "twc_cnn_softmax.pth", ANVITA / "noresm_cnn_softmax.pth", "cnn", "softmax", "noresm"),
    ("GAN (SmCL)", NORESM_MODELS / "twc_gan_softmax.pth", ANVITA / "noresm_gan_softmax.pth", "gan", "softmax", "noresm"),
    # ERA5
    ("CNN (none)", ERA5_MODELS / "twc_cnn_none.pth", ANVITA / "twc_cnn_noconstraints.pth", "cnn", "none", "era5"),
    ("CNN (SmCL)", ERA5_MODELS / "twc_cnn_softmax.pth", ANVITA / "twc_cnn_softmax.pth", "cnn", "softmax", "era5"),
    ("GAN (none)", ERA5_MODELS / "twc_gan_none.pth", ANVITA / "twc_gan_noconstraints.pth", "gan", "none", "era5"),
    ("GAN (SmCL)", ERA5_MODELS / "twc_gan_softmax.pth", ANVITA / "twc_gan_softmax.pth", "gan", "softmax", "era5"),
]

METRICS_KEYS = ["crps", "mae", "rmse", "mass_violation", "ssim", "ralsd"]


def eval_model(
    ckpt_path: Path,
    model_type: str,
    constraints: str,
    lr_orig: torch.Tensor,
    truth: np.ndarray,
    min_val: float,
    max_val: float,
    device: str,
    upsampling_factor: int,
    n_ensemble: int = 10,
) -> dict[str, object]:
    model = load_harder_model(
        ckpt_path, model_type=model_type, constraints=constraints,
        device=device, upsampling_factor=upsampling_factor,
    )
    t0 = time.time()
    if model_type == "gan":
        preds = generate_gan_predictions(model, lr_orig, min_val, max_val, device, n_ensemble=n_ensemble)
        is_ensemble = True
    else:
        preds = generate_cnn_predictions(model, lr_orig, min_val, max_val, device)
        is_ensemble = False
    elapsed = time.time() - t0
    del model
    torch.cuda.empty_cache()
    result = compute_all_metrics(truth, preds, lr_orig, is_ensemble=is_ensemble, upsampling_factor=upsampling_factor)
    result["inference_time_s"] = elapsed
    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_samples = 2000

    # Load NorESM test data
    print("Loading NorESM test data...")
    lr_up_n, _, hr_n, lr_orig_n = load_noresm_tas(str(POOL), "test")
    n = min(hr_n.shape[0], max_samples)
    hr_n, lr_orig_n = hr_n[:n], lr_orig_n[:n]
    truth_n = hr_n[:, 0].numpy()
    min_n, max_n = compute_minmax_stats(POOL, "noresm")
    print(f"  NorESM: {n} samples")

    # Load ERA5 test data
    print("Loading ERA5 test data...")
    from downscaling.data import load_era5_tcw
    lr_up_e, _, hr_e, lr_orig_e = load_era5_tcw(str(POOL), "test")
    n_e = min(hr_e.shape[0], max_samples)
    hr_e, lr_orig_e = hr_e[:n_e], lr_orig_e[:n_e]
    truth_e = hr_e[:, 0].numpy()
    min_e, max_e = compute_minmax_stats(POOL, "era5")
    print(f"  ERA5: {n_e} samples")

    all_results: dict[str, dict[str, dict[str, object]]] = {}

    for label, ours_path, anvita_path, model_type, constraints, dataset in COMPARISONS:
        key = f"{dataset} {label}"
        if dataset == "noresm":
            truth, lr_orig, min_val, max_val, uf = truth_n, lr_orig_n, min_n, max_n, UPSAMPLING_FACTOR
        else:
            truth, lr_orig, min_val, max_val, uf = truth_e, lr_orig_e, min_e, max_e, ERA5_UPSAMPLING_FACTOR

        all_results[key] = {}
        for source, path in [("ours", ours_path), ("anvita", anvita_path)]:
            if not path.exists():
                print(f"  SKIP {source} {key}: {path} not found")
                continue
            print(f"\n--- {source} {key} ---")
            # Anvita's "noconstraints" checkpoints are identical architecture to "none"
            actual_constraints = constraints
            if "noconstraints" in path.name:
                actual_constraints = "none"
            all_results[key][source] = eval_model(
                path, model_type, actual_constraints, lr_orig, truth,
                min_val, max_val, device, uf,
            )
            r = all_results[key][source]
            print(f"  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  SSIM={r['ssim']:.4f}  RALSD={r['ralsd']:.3f}")

    # Print comparison table
    print("\n" + "=" * 120)
    print("COMPARISON: Ours vs Anvita (6h training)")
    print("=" * 120)
    header = f"{'Config':<22} {'Source':<8} {'MAE':>8} {'RMSE':>8} {'SSIM':>8} {'CRPS':>8} {'MassViol':>8} {'RALSD':>8}"
    print(header)
    print("-" * len(header))

    for key, sources in all_results.items():
        for source in ["ours", "anvita"]:
            if source not in sources:
                continue
            r = sources[source]
            print(
                f"{key:<22} {source:<8} {r['mae']:8.4f} {r['rmse']:8.4f} "
                f"{r['ssim']:8.4f} {r['crps']:8.4f} {r['mass_violation']:8.4f} {r['ralsd']:8.3f}"
            )
        # Delta row
        if "ours" in sources and "anvita" in sources:
            o, a = sources["ours"], sources["anvita"]
            deltas = []
            for mk in METRICS_KEYS:
                d = float(a[mk]) - float(o[mk])
                deltas.append(f"{d:+8.4f}")
            print(f"{'  (anvita - ours)':<22} {'delta':<8} {' '.join(deltas)}")
        print()

    # Save results
    out_path = POOL / "metrics" / "anvita_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, sources in all_results.items():
        serializable[key] = {}
        for source, r in sources.items():
            serializable[key][source] = {k: r[k] for k in METRICS_KEYS if k in r}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
