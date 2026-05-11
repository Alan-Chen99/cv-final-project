"""Comprehensive evaluation of all trained models.

Computes all metrics (CRPS, MAE, RMSE, mass violation, SSIM, KL divergence,
PSD log-ratio, RALSD, spectral coherence, rank histogram, SSR) for NorESM 2x SR
and ERA5 4x SR models. ERA5 uses cached ensemble predictions; NorESM runs inference.

Run: python -m downscaling.evaluation.comprehensive [--max-samples N] [--dataset noresm|era5|both]
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
from downscaling.evaluation.checkpoints import load_checkpoint, load_norm_stats
from downscaling.evaluation.harder import compute_minmax_stats, load_harder_model
from downscaling.evaluation.swinir import (
    load_swinir_finetuned,
    predict_swinir_finetuned,
)
from downscaling.metrics import (
    crps_energy,
    ensemble_mean_kl_divergence,
    ensemble_mean_psd,
    ensemble_mean_ssim,
    histogram_kl_divergence,
    mean_spectral_coherence,
    psd_log_ratio,
    radial_psd,
    ralsd,
    rank_histogram,
    spread_skill_ratio,
)
from downscaling.metrics import (
    ssim as ssim_metric,
)
from downscaling.models import AttentionUNet
from downscaling.sampling.ode import midpoint_sample

POOL = Path("/home/chenxy/orcd/pool/datasets")
UPSAMPLING_FACTOR = 2


# ---------------------------------------------------------------------------
# Prediction generators
# ---------------------------------------------------------------------------


def generate_flow_predictions(
    model: nn.Module,
    lr_up_norm: torch.Tensor,
    lr_up: torch.Tensor,
    lr_orig: torch.Tensor,
    norm_stats: dict[str, float],
    device: str,
    *,
    n_ensemble: int = 10,
    ode_steps: int = 10,
    batch_size: int = 64,
    with_addcl: bool = True,
    upsampling_factor: int = UPSAMPLING_FACTOR,
) -> np.ndarray:
    """Generate flow matching ensemble predictions. Returns (N, M, H, W)."""
    n = lr_up_norm.shape[0]
    hr_h, hr_w = lr_up.shape[2], lr_up.shape[3]
    preds = np.empty((n, n_ensemble, hr_h, hr_w), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bs = end - start
        batch_lr = lr_up_norm[start:end].to(device)
        batch_lr_up = lr_up[start:end]
        batch_lr_orig = lr_orig[start:end]

        for m in range(n_ensemble):
            with torch.no_grad():
                sampled = midpoint_sample(
                    model, batch_lr, shape=(bs, 1, hr_h, hr_w), steps=ode_steps
                )
                res = sampled.cpu() * norm_stats["res_std"] + norm_stats["res_mean"]
                pred_hr = batch_lr_up + res
                if with_addcl:
                    pred_hr = apply_addcl(pred_hr, batch_lr_orig, upsampling_factor)
                preds[start:end, m] = pred_hr[:, 0].numpy()

        print(f"  flow: {end}/{n}")

    return preds


def generate_cnn_predictions(
    model: nn.Module,
    lr_orig: torch.Tensor,
    min_val: float,
    max_val: float,
    device: str,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Generate Harder CNN predictions. Returns (N, H, W)."""
    n = lr_orig.shape[0]
    val_range = max_val - min_val
    parts: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        lr_norm = (lr_orig[start:end] - min_val) / val_range
        lr_in = lr_norm.unsqueeze(1).to(device)  # (B, 1, 1, H, W)

        with torch.no_grad():
            out = model(lr_in)
        pred = out.cpu().squeeze(1) * val_range + min_val
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        parts.append(pred[:, 0].numpy())

    return np.concatenate(parts, axis=0)


def generate_gan_predictions(
    model: nn.Module,
    lr_orig: torch.Tensor,
    min_val: float,
    max_val: float,
    device: str,
    *,
    n_ensemble: int = 10,
    batch_size: int = 256,
) -> np.ndarray:
    """Generate Harder GAN ensemble predictions. Returns (N, M, H, W)."""
    n = lr_orig.shape[0]
    val_range = max_val - min_val

    # Probe output shape
    probe_in = ((lr_orig[:1] - min_val) / val_range).unsqueeze(1).to(device)
    z = torch.randn(1, 100, 1, 1, device=device)
    with torch.no_grad():
        probe_out = model(probe_in, z)
    hr_h, hr_w = probe_out.shape[-2], probe_out.shape[-1]
    del probe_out

    preds = np.empty((n, n_ensemble, hr_h, hr_w), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bs = end - start
        lr_norm = (lr_orig[start:end] - min_val) / val_range
        lr_in = lr_norm.unsqueeze(1).to(device)

        for m in range(n_ensemble):
            z = torch.randn(bs, 100, 1, 1, device=device)
            with torch.no_grad():
                out = model(lr_in, z)
            pred = out.cpu().squeeze(1) * val_range + min_val
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)
            preds[start:end, m] = pred[:, 0].numpy()

    return preds


def generate_swinir_predictions(
    lr_orig: torch.Tensor,
    pretrained_path: str | Path,
    checkpoint_path: str | Path,
    device: str,
    *,
    with_addcl: bool = True,
    upsampling_factor: int = UPSAMPLING_FACTOR,
) -> np.ndarray:
    """Generate SwinIR finetuned predictions. Returns (N, H, W)."""
    model, vmin, vmax = load_swinir_finetuned(pretrained_path, checkpoint_path, device)
    preds_t = predict_swinir_finetuned(model, lr_orig, vmin, vmax, device)
    if with_addcl:
        preds_t = apply_addcl(preds_t, lr_orig, upsampling_factor)
    del model
    torch.cuda.empty_cache()
    return preds_t[:, 0].numpy()


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_all_metrics(
    truth: np.ndarray,
    preds: np.ndarray,
    lr_orig: torch.Tensor,
    is_ensemble: bool,
    *,
    upsampling_factor: int = 2,
    psd_samples: int = 500,
    coherence_samples: int = 500,
) -> dict[str, object]:
    """Compute all metrics including CRPS, MAE, RMSE, mass violation, SSIM,
    KL divergence, PSD log-ratio, RALSD, spectral coherence, rank histogram, SSR.

    truth: (N,H,W), preds: (N,H,W) or (N,M,H,W).
    """
    n = truth.shape[0]
    pool = nn.AvgPool2d(kernel_size=upsampling_factor)

    # Dataset-level dynamic range for SSIM — ensures values are comparable
    # across models. Per-pair auto-range would systematically favor models
    # with compressed outputs (smaller range → smaller C1/C2 constants).
    truth_data_range = float(truth.max()) - float(truth.min())
    if truth_data_range < 1e-30:
        truth_data_range = 1.0

    crps_v: list[float] = []
    mae_v: list[float] = []
    rmse_sq_v: list[float] = []
    mass_v: list[float] = []
    ssim_v: list[float] = []
    kl_v: list[float] = []

    for i in range(n):
        gt = truth[i]
        if is_ensemble:
            ens = preds[i]  # (M, H, W)
            ens_mean = ens.mean(axis=0)
            crps_v.append(crps_energy(gt, ens))
            ssim_v.append(ensemble_mean_ssim(gt, ens, data_range=truth_data_range))
            kl_v.append(ensemble_mean_kl_divergence(gt, ens))
        else:
            p = preds[i]
            ens_mean = p
            crps_v.append(crps_energy(gt, p[None, ...]))
            ssim_v.append(float(ssim_metric(gt, p, data_range=truth_data_range)))
            kl_v.append(float(histogram_kl_divergence(gt, p)))

        mae_v.append(float(np.mean(np.abs(gt - ens_mean))))
        rmse_sq_v.append(float(np.mean((gt - ens_mean) ** 2)))

        pred_t = torch.from_numpy(ens_mean).unsqueeze(0).unsqueeze(0).float()
        pooled = pool(pred_t).squeeze()
        mass_v.append(float(torch.mean(torch.abs(pooled - lr_orig[i, 0])).item()))

    # PSD
    n_psd = min(n, psd_samples)
    truth_p: list[np.ndarray] = []
    pred_p: list[np.ndarray] = []
    k = None
    for i in range(n_psd):
        k_t, p_t = radial_psd(truth[i])
        truth_p.append(p_t)
        if is_ensemble:
            _, p_m = ensemble_mean_psd(preds[i])
        else:
            _, p_m = radial_psd(preds[i])
        pred_p.append(p_m)
        k = k_t

    if k is None:
        raise ValueError("psd_samples must be > 0 (got 0 after min(n, psd_samples))")
    mean_truth_psd = np.mean(truth_p, axis=0)
    mean_pred_psd = np.mean(pred_p, axis=0)
    psd_ratio = psd_log_ratio(k, mean_pred_psd, k, mean_truth_psd)
    ralsd_val = ralsd(k, mean_pred_psd, k, mean_truth_psd)

    # Spectral coherence (batch metric — needs (N, H, W) pairs)
    n_coh = min(n, coherence_samples)
    if is_ensemble:
        # Use ensemble mean for coherence
        ens_means = preds[:n_coh].mean(axis=1)  # (N, H, W)
    else:
        ens_means = preds[:n_coh]
    coh = mean_spectral_coherence(ens_means, truth[:n_coh])

    result: dict[str, object] = {
        "crps": float(np.mean(crps_v)),
        "mae": float(np.mean(mae_v)),
        "rmse": float(np.sqrt(np.mean(rmse_sq_v))),
        "mass_violation": float(np.mean(mass_v)),
        "ssim": float(np.mean(ssim_v)),
        "kl_divergence": float(np.mean(kl_v)),
        "psd_log_ratio": psd_ratio,
        "ralsd": ralsd_val,
        "spectral_coherence": coh,
        "psd_k": k.tolist(),
        "psd_power": mean_pred_psd.tolist(),
        "psd_truth_power": mean_truth_psd.tolist(),
    }

    if is_ensemble:
        total_rh: np.ndarray | None = None
        ssr_v: list[float] = []
        for i in range(n):
            rh = rank_histogram(truth[i], preds[i])
            total_rh = rh.copy() if total_rh is None else total_rh + rh
            ssr_v.append(float(spread_skill_ratio(truth[i], preds[i])))
        result["ssr"] = float(np.mean(ssr_v))
        result["rank_histogram"] = total_rh.tolist() if total_rh is not None else None

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_comprehensive_eval(
    output_dir: Path,
    device: str = "cuda",
    max_samples: int = 2000,
    n_ensemble: int = 10,
    ode_steps: int = 10,
) -> dict[str, dict[str, object]]:
    """Run comprehensive evaluation on all NorESM models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    noresm_data = POOL  # load_noresm_tas appends noresm-dataset/noresm/ internally
    models_dir = POOL / "noresm-dataset" / "models"
    pretrained_dir = POOL / "noresm-dataset" / "pretrained_weights"

    print(f"Loading NorESM test data from {noresm_data}...")
    lr_up, _residual, hr, lr_orig = load_noresm_tas(str(noresm_data), "test")
    n = min(hr.shape[0], max_samples)
    lr_up, hr, lr_orig = lr_up[:n], hr[:n], lr_orig[:n]
    truth = hr[:, 0].numpy()  # (N, 64, 64)
    print(f"  {n} samples, HR shape: {hr.shape}, LR shape: {lr_orig.shape}")

    results: dict[str, dict[str, object]] = {}

    # --- Flow Matching ---
    print("\n--- Loading Flow model ---")
    model = AttentionUNet(base_channels=96).to(device)
    load_checkpoint(model, str(models_dir / "flow-wide96-amp" / "best_flow.pt"), device)
    norm_stats = load_norm_stats(str(models_dir / "flow-wide96-amp" / "norm_stats.pt"), device)
    lr_up_norm = (lr_up - norm_stats["lr_mean"]) / norm_stats["lr_std"]

    for addcl, label in [(False, "Flow"), (True, "Flow+AddCL")]:
        print(f"\n--- {label} ---")
        t0 = time.time()
        flow_preds = generate_flow_predictions(
            model, lr_up_norm, lr_up, lr_orig, norm_stats, device,
            n_ensemble=n_ensemble, ode_steps=ode_steps, with_addcl=addcl,
        )
        flow_time = time.time() - t0
        print(f"  Inference: {flow_time:.1f}s")
        print("  Computing metrics...")
        results[label] = compute_all_metrics(truth, flow_preds, lr_orig, is_ensemble=True, upsampling_factor=UPSAMPLING_FACTOR)
        results[label]["inference_time_s"] = flow_time
        results[label]["is_ensemble"] = True
        results[label]["n_ensemble"] = n_ensemble
        print(f"  CRPS={results[label]['crps']:.4f}  MAE={results[label]['mae']:.4f}  "
              f"RMSE={results[label]['rmse']:.4f}  SSIM={results[label]['ssim']:.4f}  "
              f"SSR={results[label].get('ssr', '?')}")

    del model
    torch.cuda.empty_cache()

    # --- CNN (Harder et al.) ---
    min_val, max_val = compute_minmax_stats(POOL, "noresm")
    print(f"\nHarder min/max: {min_val:.2f} / {max_val:.2f}")

    print("\n--- Loading CNN model ---")
    cnn = load_harder_model(
        models_dir / "harder" / "twc_cnn_none.pth",
        model_type="cnn", constraints="none", device=device,
        upsampling_factor=UPSAMPLING_FACTOR,
    )
    t0 = time.time()
    cnn_preds_raw = generate_cnn_predictions(cnn, lr_orig, min_val, max_val, device)
    del cnn
    torch.cuda.empty_cache()
    cnn_time = time.time() - t0
    print(f"  Inference: {cnn_time:.1f}s")

    # CNN without and with AddCL (post-hoc)
    cnn_preds_addcl = apply_addcl(
        torch.from_numpy(cnn_preds_raw).unsqueeze(1), lr_orig, UPSAMPLING_FACTOR,
    )[:, 0].numpy()

    for preds, label in [(cnn_preds_raw, "CNN"), (cnn_preds_addcl, "CNN+AddCL")]:
        print(f"\n--- {label} ---")
        print("  Computing metrics...")
        results[label] = compute_all_metrics(truth, preds, lr_orig, is_ensemble=False, upsampling_factor=UPSAMPLING_FACTOR)
        results[label]["inference_time_s"] = cnn_time
        results[label]["is_ensemble"] = False
        print(f"  CRPS={results[label]['crps']:.4f}  MAE={results[label]['mae']:.4f}  "
              f"RMSE={results[label]['rmse']:.4f}  SSIM={results[label]['ssim']:.4f}")

    # --- SwinIR finetuned ---
    print("\n--- Loading SwinIR model ---")
    pretrained_weights = pretrained_dir / "001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
    swinir_ckpt = models_dir / "swinir_ft" / "best_swinir.pt"

    for addcl, label in [(False, "SwinIR"), (True, "SwinIR+AddCL")]:
        print(f"\n--- {label} ---")
        t0 = time.time()
        swinir_preds = generate_swinir_predictions(
            lr_orig, pretrained_weights, swinir_ckpt, device, with_addcl=addcl,
        )
        swinir_time = time.time() - t0
        print(f"  Inference: {swinir_time:.1f}s")
        print("  Computing metrics...")
        results[label] = compute_all_metrics(truth, swinir_preds, lr_orig, is_ensemble=False, upsampling_factor=UPSAMPLING_FACTOR)
        results[label]["inference_time_s"] = swinir_time
        results[label]["is_ensemble"] = False
        print(f"  CRPS={results[label]['crps']:.4f}  MAE={results[label]['mae']:.4f}  "
              f"RMSE={results[label]['rmse']:.4f}  SSIM={results[label]['ssim']:.4f}")

    # --- Truth+AddCL (upper bound for AddCL constraint) ---
    print("\n--- Truth+AddCL ---")
    truth_addcl = apply_addcl(hr, lr_orig, UPSAMPLING_FACTOR)[:, 0].numpy()
    results["Truth+AddCL"] = compute_all_metrics(truth, truth_addcl, lr_orig, is_ensemble=False, upsampling_factor=UPSAMPLING_FACTOR)
    results["Truth+AddCL"]["inference_time_s"] = 0.0
    results["Truth+AddCL"]["is_ensemble"] = False
    print(f"  CRPS={results['Truth+AddCL']['crps']:.4f}  MAE={results['Truth+AddCL']['mae']:.4f}  "
          f"RMSE={results['Truth+AddCL']['rmse']:.4f}  SSIM={results['Truth+AddCL']['ssim']:.4f}")

    # --- Save results ---
    print("\n--- Saving results ---")
    results_path = output_dir / "comprehensive_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {results_path}")

    # --- Summary table ---
    print("\n" + "=" * 100)
    print("COMPREHENSIVE EVALUATION RESULTS — NorESM TAS 2x SR")
    print(f"Samples: {n}, Ensemble size: {n_ensemble}, ODE steps: {ode_steps}")
    print("=" * 100)
    header = f"{'Model':<18} {'CRPS':>8} {'MAE':>8} {'RMSE':>8} {'MassViol':>8} {'SSIM':>8} {'KL':>8} {'PSD-LR':>8} {'RALSD':>8} {'Coh':>8} {'SSR':>8}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        ssr_str = f"{r['ssr']:.3f}" if "ssr" in r else "  —"
        coh_str = f"{r['spectral_coherence']:.3f}" if "spectral_coherence" in r else "  —"
        print(
            f"{name:<18} {r['crps']:8.4f} {r['mae']:8.4f} {r['rmse']:8.4f} "
            f"{r['mass_violation']:8.4f} {r['ssim']:8.4f} {r['kl_divergence']:8.4f} "
            f"{r['psd_log_ratio']:8.4f} {r['ralsd']:8.4f} {coh_str:>8} {ssr_str:>8}"
        )
    print("=" * 100)

    return results


ERA5_UPSAMPLING_FACTOR = 4

# Cached flow predictions (pre-computed .pt files)
ERA5_FLOW_PREDICTIONS: dict[str, str] = {
    "flow_flow_none_test_ensemble.pt": "Flow",
    "flow_flow_v2_addcl_test_ensemble.pt": "Flow+AddCL",
}

# ERA5 Harder CNN checkpoint (organize2 branch)
ERA5_HARDER_DIR = POOL / "organize2" / "models" / "harder"

# ERA5 SwinIR checkpoint (research5 branch)
ERA5_SWINIR_PRETRAINED = POOL / "research5" / "pretrained_weights" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
ERA5_SWINIR_CKPT = POOL / "research5" / "models" / "swinir_ft" / "best_swinir.pt"


def run_era5_eval(
    output_dir: Path,
    device: str = "cuda",
    max_samples: int = 2000,
) -> dict[str, dict[str, object]]:
    """Run comprehensive evaluation on ERA5 4x SR models.

    Flow predictions are pre-computed .pt files. CNN and SwinIR require GPU inference.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    from downscaling.data import load_era5_tcw

    print("Loading ERA5 test data...")
    lr_up, _residual, hr, lr_orig = load_era5_tcw(str(POOL), "test")
    n = min(hr.shape[0], max_samples)
    lr_up, hr, lr_orig = lr_up[:n], hr[:n], lr_orig[:n]
    truth = hr[:, 0].numpy()  # (N, 128, 128)
    print(f"  {n} samples, HR shape: {hr.shape}, LR shape: {lr_orig.shape}")

    results: dict[str, dict[str, object]] = {}
    pred_dir = POOL / "era5_sr_data" / "prediction"

    # --- Flow (cached ensemble predictions) ---
    for filename, display_name in ERA5_FLOW_PREDICTIONS.items():
        pred_path = pred_dir / filename
        if not pred_path.exists():
            print(f"  Skipping {display_name}: {pred_path} not found")
            continue

        print(f"\n--- {display_name} ---")
        t0 = time.time()
        raw = torch.load(str(pred_path), map_location="cpu", weights_only=True)
        # Shape: (N, M, 1, 1, 128, 128) → (N, M, 128, 128)
        preds_all = raw[:n].squeeze(2).squeeze(2).numpy()
        n_ens = preds_all.shape[1]
        load_time = time.time() - t0
        print(f"  Loaded {preds_all.shape}, {n_ens} members, {load_time:.1f}s")

        print("  Computing metrics...")
        results[display_name] = compute_all_metrics(
            truth, preds_all, lr_orig, is_ensemble=True,
            upsampling_factor=ERA5_UPSAMPLING_FACTOR,
        )
        results[display_name]["is_ensemble"] = True
        results[display_name]["n_ensemble"] = n_ens
        results[display_name]["inference_time_s"] = 0.0  # pre-computed
        r = results[display_name]
        print(f"  CRPS={r['crps']:.4f}  MAE={r['mae']:.4f}  "
              f"RMSE={r['rmse']:.4f}  SSIM={r['ssim']:.4f}  "
              f"SSR={r.get('ssr', '?')}  Coh={r.get('spectral_coherence', '?')}")

    # --- CNN (Harder et al.) ---
    min_val, max_val = compute_minmax_stats(POOL, "era5")
    print(f"\nHarder min/max: {min_val:.2f} / {max_val:.2f}")

    cnn_ckpt = ERA5_HARDER_DIR / "twc_cnn_none.pth"
    if cnn_ckpt.exists():
        print("\n--- Loading CNN model ---")
        cnn = load_harder_model(
            cnn_ckpt, model_type="cnn", constraints="none", device=device,
            upsampling_factor=ERA5_UPSAMPLING_FACTOR,
        )
        t0 = time.time()
        cnn_preds_raw = generate_cnn_predictions(cnn, lr_orig, min_val, max_val, device)
        del cnn
        torch.cuda.empty_cache()
        cnn_time = time.time() - t0
        print(f"  Inference: {cnn_time:.1f}s")

        # CNN without and with AddCL (post-hoc)
        cnn_preds_addcl = apply_addcl(
            torch.from_numpy(cnn_preds_raw).unsqueeze(1), lr_orig, ERA5_UPSAMPLING_FACTOR,
        )[:, 0].numpy()

        for preds, label in [(cnn_preds_raw, "CNN"), (cnn_preds_addcl, "CNN+AddCL")]:
            print(f"\n--- {label} ---")
            print("  Computing metrics...")
            results[label] = compute_all_metrics(truth, preds, lr_orig, is_ensemble=False, upsampling_factor=ERA5_UPSAMPLING_FACTOR)
            results[label]["inference_time_s"] = cnn_time
            results[label]["is_ensemble"] = False
            print(f"  CRPS={results[label]['crps']:.4f}  MAE={results[label]['mae']:.4f}  "
                  f"RMSE={results[label]['rmse']:.4f}  SSIM={results[label]['ssim']:.4f}")
    else:
        print(f"  Skipping CNN: {cnn_ckpt} not found")

    # --- SwinIR finetuned ---
    if ERA5_SWINIR_CKPT.exists():
        print("\n--- Loading SwinIR model ---")
        for addcl, label in [(False, "SwinIR"), (True, "SwinIR+AddCL")]:
            print(f"\n--- {label} ---")
            t0 = time.time()
            swinir_preds = generate_swinir_predictions(
                lr_orig, ERA5_SWINIR_PRETRAINED, ERA5_SWINIR_CKPT, device,
                with_addcl=addcl, upsampling_factor=ERA5_UPSAMPLING_FACTOR,
            )
            swinir_time = time.time() - t0
            print(f"  Inference: {swinir_time:.1f}s")
            print("  Computing metrics...")
            results[label] = compute_all_metrics(truth, swinir_preds, lr_orig, is_ensemble=False, upsampling_factor=ERA5_UPSAMPLING_FACTOR)
            results[label]["inference_time_s"] = swinir_time
            results[label]["is_ensemble"] = False
            print(f"  CRPS={results[label]['crps']:.4f}  MAE={results[label]['mae']:.4f}  "
                  f"RMSE={results[label]['rmse']:.4f}  SSIM={results[label]['ssim']:.4f}")
    else:
        print(f"  Skipping SwinIR: {ERA5_SWINIR_CKPT} not found")

    # --- Truth+AddCL (upper bound for AddCL constraint) ---
    print("\n--- Truth+AddCL ---")
    truth_addcl = apply_addcl(hr, lr_orig, ERA5_UPSAMPLING_FACTOR)[:, 0].numpy()
    results["Truth+AddCL"] = compute_all_metrics(truth, truth_addcl, lr_orig, is_ensemble=False, upsampling_factor=ERA5_UPSAMPLING_FACTOR)
    results["Truth+AddCL"]["inference_time_s"] = 0.0
    results["Truth+AddCL"]["is_ensemble"] = False
    print(f"  CRPS={results['Truth+AddCL']['crps']:.4f}  MAE={results['Truth+AddCL']['mae']:.4f}  "
          f"RMSE={results['Truth+AddCL']['rmse']:.4f}  SSIM={results['Truth+AddCL']['ssim']:.4f}")

    # --- Save results ---
    print("\n--- Saving results ---")
    results_path = output_dir / "era5_comprehensive_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {results_path}")

    # --- Summary table ---
    print("\n" + "=" * 110)
    print("COMPREHENSIVE EVALUATION RESULTS — ERA5 TCW 4x SR")
    print(f"Samples: {n}")
    print("=" * 110)
    header = f"{'Model':<22} {'CRPS':>8} {'MAE':>8} {'RMSE':>8} {'MassViol':>8} {'SSIM':>8} {'KL':>8} {'PSD-LR':>8} {'RALSD':>8} {'Coh':>8} {'SSR':>8}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        ssr_str = f"{r['ssr']:.3f}" if "ssr" in r else "  —"
        coh_str = f"{r['spectral_coherence']:.3f}" if "spectral_coherence" in r else "  —"
        print(
            f"{name:<22} {r['crps']:8.4f} {r['mae']:8.4f} {r['rmse']:8.4f} "
            f"{r['mass_violation']:8.4f} {r['ssim']:8.4f} {r['kl_divergence']:8.4f} "
            f"{r['psd_log_ratio']:8.4f} {r['ralsd']:8.4f} {coh_str:>8} {ssr_str:>8}"
        )
    print("=" * 110)

    return results


if __name__ == "__main__":
    max_samples = 2000
    dataset = "both"

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--max-samples":
            max_samples = int(args[i + 1])
            i += 2
        elif args[i] == "--dataset":
            dataset = args[i + 1]
            i += 2
        else:
            i += 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset in ("noresm", "both"):
        noresm_output = POOL / "metrics" / "noresm"
        run_comprehensive_eval(noresm_output, device=device, max_samples=max_samples)

    if dataset in ("era5", "both"):
        era5_output = POOL / "metrics" / "era5"
        run_era5_eval(era5_output, device=device, max_samples=max_samples)
