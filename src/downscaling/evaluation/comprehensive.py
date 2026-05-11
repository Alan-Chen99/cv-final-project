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
from downscaling.evaluation.baselines import upsample_bicubic, upsample_bilinear
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
                pred_hr = apply_addcl(batch_lr_up + res, batch_lr_orig, UPSAMPLING_FACTOR)
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
) -> np.ndarray:
    """Generate SwinIR finetuned predictions. Returns (N, H, W)."""
    model, vmin, vmax = load_swinir_finetuned(pretrained_path, checkpoint_path, device)
    preds_t = predict_swinir_finetuned(model, lr_orig, vmin, vmax, device)
    if with_addcl:
        preds_t = apply_addcl(preds_t, lr_orig, UPSAMPLING_FACTOR)
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

    # --- Flow Matching (wide96-amp, AddCL) ---
    print("\n--- Flow Matching (AddCL) ---")
    t0 = time.time()
    model = AttentionUNet(base_channels=96).to(device)
    load_checkpoint(model, str(models_dir / "flow-wide96-amp" / "best_flow.pt"), device)
    norm_stats = load_norm_stats(str(models_dir / "flow-wide96-amp" / "norm_stats.pt"), device)
    lr_up_norm = (lr_up - norm_stats["lr_mean"]) / norm_stats["lr_std"]

    flow_preds = generate_flow_predictions(
        model, lr_up_norm, lr_up, lr_orig, norm_stats, device,
        n_ensemble=n_ensemble, ode_steps=ode_steps,
    )
    del model
    torch.cuda.empty_cache()
    flow_time = time.time() - t0
    print(f"  Inference: {flow_time:.1f}s")

    print("  Computing metrics...")
    results["Flow+AddCL"] = compute_all_metrics(truth, flow_preds, lr_orig, is_ensemble=True, upsampling_factor=UPSAMPLING_FACTOR)
    results["Flow+AddCL"]["inference_time_s"] = flow_time
    results["Flow+AddCL"]["is_ensemble"] = True
    results["Flow+AddCL"]["n_ensemble"] = n_ensemble
    print(f"  CRPS={results['Flow+AddCL']['crps']:.4f}  MAE={results['Flow+AddCL']['mae']:.4f}  "
          f"RMSE={results['Flow+AddCL']['rmse']:.4f}  SSIM={results['Flow+AddCL']['ssim']:.4f}  "
          f"SSR={results['Flow+AddCL'].get('ssr', '?')}")

    # --- Harder models ---
    min_val, max_val = compute_minmax_stats(POOL, "noresm")
    print(f"\nHarder min/max: {min_val:.2f} / {max_val:.2f}")

    # CNN (no constraint)
    print("\n--- Harder CNN (none) ---")
    t0 = time.time()
    cnn = load_harder_model(
        models_dir / "harder" / "twc_cnn_none.pth",
        model_type="cnn", constraints="none", device=device,
        upsampling_factor=UPSAMPLING_FACTOR,
    )
    cnn_preds = generate_cnn_predictions(cnn, lr_orig, min_val, max_val, device)
    del cnn
    torch.cuda.empty_cache()
    cnn_time = time.time() - t0
    print(f"  Inference: {cnn_time:.1f}s")
    print("  Computing metrics...")
    results["CNN(none)"] = compute_all_metrics(truth, cnn_preds, lr_orig, is_ensemble=False, upsampling_factor=UPSAMPLING_FACTOR)
    results["CNN(none)"]["inference_time_s"] = cnn_time
    results["CNN(none)"]["is_ensemble"] = False
    print(f"  CRPS={results['CNN(none)']['crps']:.4f}  MAE={results['CNN(none)']['mae']:.4f}  "
          f"RMSE={results['CNN(none)']['rmse']:.4f}  SSIM={results['CNN(none)']['ssim']:.4f}")

    # CNN (softmax)
    print("\n--- Harder CNN (softmax) ---")
    t0 = time.time()
    cnn_sm = load_harder_model(
        models_dir / "harder" / "twc_cnn_softmax.pth",
        model_type="cnn", constraints="softmax", device=device,
        upsampling_factor=UPSAMPLING_FACTOR,
    )
    cnn_sm_preds = generate_cnn_predictions(cnn_sm, lr_orig, min_val, max_val, device)
    del cnn_sm
    torch.cuda.empty_cache()
    cnn_sm_time = time.time() - t0
    print(f"  Inference: {cnn_sm_time:.1f}s")
    print("  Computing metrics...")
    results["CNN(softmax)"] = compute_all_metrics(truth, cnn_sm_preds, lr_orig, is_ensemble=False, upsampling_factor=UPSAMPLING_FACTOR)
    results["CNN(softmax)"]["inference_time_s"] = cnn_sm_time
    results["CNN(softmax)"]["is_ensemble"] = False
    print(f"  CRPS={results['CNN(softmax)']['crps']:.4f}  MAE={results['CNN(softmax)']['mae']:.4f}  "
          f"RMSE={results['CNN(softmax)']['rmse']:.4f}  SSIM={results['CNN(softmax)']['ssim']:.4f}")

    # GAN (softmax) — ensemble
    print("\n--- Harder GAN (softmax) ---")
    t0 = time.time()
    gan = load_harder_model(
        models_dir / "harder" / "twc_gan_softmax.pth",
        model_type="gan", constraints="softmax", device=device,
        upsampling_factor=UPSAMPLING_FACTOR,
    )
    gan_preds = generate_gan_predictions(
        gan, lr_orig, min_val, max_val, device, n_ensemble=n_ensemble,
    )
    del gan
    torch.cuda.empty_cache()
    gan_time = time.time() - t0
    print(f"  Inference: {gan_time:.1f}s")
    print("  Computing metrics...")
    results["GAN(softmax)"] = compute_all_metrics(truth, gan_preds, lr_orig, is_ensemble=True, upsampling_factor=UPSAMPLING_FACTOR)
    results["GAN(softmax)"]["inference_time_s"] = gan_time
    results["GAN(softmax)"]["is_ensemble"] = True
    results["GAN(softmax)"]["n_ensemble"] = n_ensemble
    print(f"  CRPS={results['GAN(softmax)']['crps']:.4f}  MAE={results['GAN(softmax)']['mae']:.4f}  "
          f"RMSE={results['GAN(softmax)']['rmse']:.4f}  SSIM={results['GAN(softmax)']['ssim']:.4f}  "
          f"SSR={results['GAN(softmax)'].get('ssr', '?')}")

    # --- SwinIR finetuned ---
    print("\n--- SwinIR finetuned (AddCL) ---")
    pretrained_weights = pretrained_dir / "001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
    swinir_ckpt = models_dir / "swinir_ft" / "best_swinir.pt"
    t0 = time.time()
    swinir_preds = generate_swinir_predictions(
        lr_orig, pretrained_weights, swinir_ckpt, device, with_addcl=True,
    )
    swinir_time = time.time() - t0
    print(f"  Inference: {swinir_time:.1f}s")
    print("  Computing metrics...")
    results["SwinIR+AddCL"] = compute_all_metrics(truth, swinir_preds, lr_orig, is_ensemble=False, upsampling_factor=UPSAMPLING_FACTOR)
    results["SwinIR+AddCL"]["inference_time_s"] = swinir_time
    results["SwinIR+AddCL"]["is_ensemble"] = False
    print(f"  CRPS={results['SwinIR+AddCL']['crps']:.4f}  MAE={results['SwinIR+AddCL']['mae']:.4f}  "
          f"RMSE={results['SwinIR+AddCL']['rmse']:.4f}  SSIM={results['SwinIR+AddCL']['ssim']:.4f}")

    # --- Baselines ---
    print("\n--- Bicubic baseline ---")
    bicubic_preds = upsample_bicubic(lr_orig, UPSAMPLING_FACTOR)[:, 0].numpy()
    results["Bicubic"] = compute_all_metrics(truth, bicubic_preds, lr_orig, is_ensemble=False, upsampling_factor=UPSAMPLING_FACTOR)
    results["Bicubic"]["is_ensemble"] = False
    results["Bicubic"]["inference_time_s"] = 0.0
    print(f"  CRPS={results['Bicubic']['crps']:.4f}  MAE={results['Bicubic']['mae']:.4f}  "
          f"RMSE={results['Bicubic']['rmse']:.4f}  SSIM={results['Bicubic']['ssim']:.4f}")

    print("\n--- Bilinear baseline ---")
    bilinear_preds = upsample_bilinear(lr_orig, UPSAMPLING_FACTOR)[:, 0].numpy()
    results["Bilinear"] = compute_all_metrics(truth, bilinear_preds, lr_orig, is_ensemble=False, upsampling_factor=UPSAMPLING_FACTOR)
    results["Bilinear"]["is_ensemble"] = False
    results["Bilinear"]["inference_time_s"] = 0.0
    print(f"  CRPS={results['Bilinear']['crps']:.4f}  MAE={results['Bilinear']['mae']:.4f}  "
          f"RMSE={results['Bilinear']['rmse']:.4f}  SSIM={results['Bilinear']['ssim']:.4f}")

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

# Mapping from prediction filename to display name
ERA5_PREDICTION_NAMES: dict[str, str] = {
    "flow_flow_none_test_ensemble.pt": "Flow(none)",
    "flow_flow_residual_none_test_ensemble.pt": "ResFlow(none)",
    "flow_flow_v2_addcl_test_ensemble.pt": "FlowV2+AddCL",
    "flow_flow_res_20step_addcl_test_ensemble.pt": "ResFlow-20s+AddCL",
    "flow_flow_res_heun_addcl_test_ensemble.pt": "ResFlow-Heun+AddCL",
}


def run_era5_eval(
    output_dir: Path,
    max_samples: int = 2000,
) -> dict[str, dict[str, object]]:
    """Run comprehensive evaluation on ERA5 4x SR cached predictions.

    All predictions are pre-computed .pt files — no GPU inference needed.
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

    # --- Cached ensemble predictions ---
    for filename, display_name in ERA5_PREDICTION_NAMES.items():
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

    # --- Baselines ---
    print("\n--- Bicubic baseline ---")
    bicubic_preds = upsample_bicubic(lr_orig, ERA5_UPSAMPLING_FACTOR)[:, 0].numpy()
    results["Bicubic"] = compute_all_metrics(
        truth, bicubic_preds, lr_orig, is_ensemble=False,
        upsampling_factor=ERA5_UPSAMPLING_FACTOR,
    )
    results["Bicubic"]["is_ensemble"] = False
    results["Bicubic"]["inference_time_s"] = 0.0
    print(f"  CRPS={results['Bicubic']['crps']:.4f}  MAE={results['Bicubic']['mae']:.4f}")

    print("\n--- Bilinear baseline ---")
    bilinear_preds = upsample_bilinear(lr_orig, ERA5_UPSAMPLING_FACTOR)[:, 0].numpy()
    results["Bilinear"] = compute_all_metrics(
        truth, bilinear_preds, lr_orig, is_ensemble=False,
        upsampling_factor=ERA5_UPSAMPLING_FACTOR,
    )
    results["Bilinear"]["is_ensemble"] = False
    results["Bilinear"]["inference_time_s"] = 0.0
    print(f"  CRPS={results['Bilinear']['crps']:.4f}  MAE={results['Bilinear']['mae']:.4f}")

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
        run_era5_eval(era5_output, max_samples=max_samples)
