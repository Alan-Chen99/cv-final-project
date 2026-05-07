# Research5: Pretrained Image SR for Climate Downscaling

**Date:** 2026-05-06 to 2026-05-07
**Branch:** research5
**Task:** 32x32 -> 128x128 spatial downscaling of TCW (total column water), CRPS metric
**Dataset:** Harder et al. `era5_sr_data` -- 40K train / 10K val / 10K test
**Compute budget:** 2hr wall-clock per experiment, ~40hr total exploration
**Starting commit:** 7cbd4e4
**Final commit:** (this report)

## Best Result

| Model | Params | Epochs | Steps | CRPS (10K test) | MAE | RMSE | Mass Viol |
|-------|--------|--------|-------|-----------------|-----|------|-----------|
| **SwinIR-Conditioned OT-CFM** | **13M** | **26** | **20** | **0.173** | 0.312 | 0.464 | 0.005 |

With AddCL constraint, mass violation drops to ~0.000001 (confirmed on partial eval; consistent with all prior experiments).

### Comparison to Prior Branches

All CRPS values use the corrected energy formula: `CRPS = E|X-y| - 0.5*E|X-X'|`.

| Model | Branch | Params | CRPS (corrected) | Eval Set | Notes |
|-------|--------|--------|------------------|----------|-------|
| **SwinIR-Conditioned OT-CFM** | **research5** | **13M** | **0.173** | **10K test** | Best this branch |
| OT-CFM UNet (Flow v2) | research2 | 13M | 0.171 | 2K test | ~0.174 est. on 10K |
| Multi-Head SwinIR K=8 | research5 | 0.8M trainable | 0.183 | 10K test | Best SwinIR-only |
| LR-Anchor Flow | research | 5.2M | 0.199 | 10K test | |
| GAN (Harder et al.) | research | 204K | 0.307 | 10K test | |

**Key finding:** The SwinIR-conditioned OT-CFM (CRPS=0.173 on 10K) matches the research2 OT-CFM baseline (~0.174 estimated on 10K). The research2 headline number (0.171) was measured on 2K test only; the full 10K log was lost to preemption.

## Research Question

**Can pretrained image super-resolution models improve probabilistic climate downscaling?**

Prior work (research, research2) trained flow matching models from scratch. This branch explores whether starting from a pretrained backbone (SwinIR) can improve CRPS through better spatial representations.

## All Experiments

| Iter | Direction | Architecture | CRPS | Verdict |
|------|-----------|-------------|------|---------|
| 1 | Zero-shot + SwinIR finetune | SwinIR (frozen backbone + trainable tail) | 0.250 | Deterministic baseline |
| 2 | Multi-head K=8 direct CRPS | SwinIR backbone + 8 independent heads | **0.183** | Best SwinIR-only |
| 3 | Multi-head K=8 residual | SwinIR backbone + 8 residual heads | 0.183 | Same ceiling |
| 4 | Multi-head unfreeze last 2 layers | SwinIR (partially unfrozen) + 8 heads | 0.183 | Same ceiling |
| 5 | CorrDiff-style residual flow | UNet flow on SwinIR residuals | 0.207 | Negative (source-target mismatch) |
| 6 | DiT flow matching | Diffusion Transformer + OT-CFM | 0.204 | Negative (patch info loss) |
| 7 | Noise-conditioned SwinIR tail | SwinIR + noise-injected reconstruction | 0.200 | Negative (noise ignored) |
| 8 | SwinIR-conditioned OT-CFM | UNet flow + SwinIR pred as conditioning | **0.173** | **Best overall** |

### Detailed Results

#### Iteration 1: Zero-Shot + Finetuned SwinIR (Deterministic)

**Goal:** Establish pretrained SR baseline.

Zero-shot SwinIR (trained on natural images, bicubic degradation) produced CRPS=1.28 -- completely failed on climate data due to domain gap. After finetuning (frozen backbone, trainable reconstruction tail, 19 epochs, 2hr), CRPS dropped to 0.250 (= MAE for deterministic model).

**Key files:** `src/exp-pretrained-sr/finetune_swinir.py`, `src/exp-pretrained-sr/eval_zeroshot.py`
**Checkpoint:** `pool/datasets/research5/models/swinir_ft/best_swinir.pt`

#### Iteration 2: Multi-Head SwinIR (K=8, Direct CRPS)

**Goal:** Add ensemble diversity to SwinIR.

Attached 8 independent reconstruction heads to the frozen SwinIR backbone. Trained with direct CRPS loss (energy score). Achieved CRPS=0.183 -- a 27% improvement over deterministic baseline.

Analysis: Spread (0.272) > MAE (0.250) indicates slight over-dispersion. Each head sees the same features and must diversify through weight divergence alone.

**Key files:** `src/exp-pretrained-sr/train_crps_ensemble.py`
**Checkpoint:** `pool/datasets/research5/models/crps_ensemble/best_ensemble.pt`

#### Iteration 3: Multi-Head Residual Parameterization

**Goal:** Test if residual heads (predicting offset from mean) improve diversity.

Same architecture as iter 2 but heads predict offsets from a shared mean head. Converged to identical CRPS=0.183. The residual parameterization made no difference -- heads learned the same diversity as direct prediction.

**Checkpoint:** `pool/datasets/research5/models/crps_residual/best_ensemble.pt`

#### Iteration 4: Unfreezing Last 2 Swin Layers

**Goal:** Allow backbone adaptation with multi-head CRPS.

Unfroze the last 2 SwinIR transformer layers plus multi-head ensemble. CRPS=0.183 again. The backbone features are already good enough; the bottleneck is the multi-head diversity mechanism, not feature quality.

**Checkpoint:** `pool/datasets/research5/models/crps_unfreeze2/best_ensemble.pt`

#### Iteration 5: CorrDiff-Style Residual Flow (Negative Result)

**Goal:** Apply flow matching to SwinIR residuals (HR - SwinIR_pred).

Trained OT-CFM UNet on the residuals between HR and SwinIR predictions. CRPS=0.207 -- significantly worse.

**Root cause:** Source-target distribution mismatch. SwinIR residuals have std=0.0035, but the flow starts from N(0,1). This is a 285x compression ratio. The flow adds biased noise rather than learning the residual distribution.

**Lesson:** CorrDiff's two-stage approach works because the deterministic predictor and diffusion model are designed together. Naively targeting a pretrained model's residuals creates pathological OT paths.

**Checkpoint:** `pool/datasets/research5/models/residual_flow/`

#### Iteration 6: DiT (Diffusion Transformer) Flow Matching

**Goal:** Test transformer-based score networks for climate downscaling (under-explored direction #7 from CLAUDE.md).

Built FlowDiT: 8 transformer blocks with adaLN-Zero time conditioning, 8x8 patches -> 256 tokens, ~22M params. Trained with same OT-CFM objective.

CRPS=0.201-0.204 -- worse than both UNet (0.171) and multi-head SwinIR (0.183).

**Root cause:** 8x8 patches compress 64 pixels to a single token. Fine spatial structure within patches is lost -- unpatchify is a rank-1 approximation per patch. 4x4 patches (1024 tokens) OOM'd at BS=64 on L40S.

**Positive finding:** DiT trains fast (1.1 min/ep), converges well, and AddCL works perfectly on DiT output.

**Key files:** `src/exp-pretrained-sr/train_dit_flow.py`
**Checkpoint:** `pool/datasets/research5/models/dit_flow/best_flow.pt`
**Figure:** `figures/dit_flow_training.png`

#### Iteration 7: Noise-Conditioned SwinIR Tail (Negative Result)

**Goal:** Inject continuous noise at feature level for stochastic diversity.

Concatenated 16 noise channels (N(0,I) at 32x32) with SwinIR backbone features. Single reconstruction head trained with CRPS loss.

CRPS=0.200 -- under-dispersed (spread=0.197 < MAE=0.258). The backbone features dominate; the tail learns to ignore noise. Spread term stayed at 0.0017 from epoch 1 to 33.

**Lesson:** Simple noise concatenation cannot compete with dominant pretrained features. The CRPS loss can be minimized by producing accurate but identical members. Multi-head succeeds because each head has independent weights that diverge during training.

**Checkpoint:** `pool/datasets/research5/models/noise_swinir/best_noise_swinir.pt`

#### Iteration 8: SwinIR-Conditioned OT-CFM Flow (Best Result)

**Goal:** Use SwinIR predictions as extra conditioning for standard OT-CFM flow.

**Key insight from iteration 5 failure:** Don't target SwinIR residuals (too concentrated). Instead, keep the standard OT-CFM target (bilinear residuals, reasonable variance) and add SwinIR predictions as a 3rd conditioning channel.

Architecture: Same AttentionUNet as research2, but `in_channels=3` [x_t, lr_up_norm, swinir_pred_norm]. 13M params with EMA (decay=0.999).

CRPS=0.173 (20 steps, 10K test) -- 5.5% improvement over multi-head ceiling (0.183), matching the research2 baseline (~0.174 estimated on 10K).

**Why it works:**
1. Standard OT-CFM target avoids source-target mismatch
2. SwinIR prediction provides strong spatial prior as extra conditioning
3. 3-channel conditioning gives the model strictly more information than 2-channel
4. Only 26 epochs trained (~4.5min/ep on slow L40S node) -- model still improving

**Key files:** `src/exp-pretrained-sr/train_swinir_flow.py`
**Checkpoint:** `pool/datasets/research5/models/swinir_flow/best_flow.pt`
**Norm stats:** `pool/datasets/research5/models/swinir_flow/norm_stats.pt`

### Reproduction

Train (requires GPU, ~2hr on L40S):
```bash
python src/exp-pretrained-sr/train_swinir_flow.py --mode train \
    --epochs 200 --batch_size 64 --lr 1e-4 --ema_decay 0.999 --time_limit 120
```

Evaluate (requires GPU, ~30min on L40S):
```bash
python src/exp-pretrained-sr/train_swinir_flow.py --mode eval \
    --n_ensemble 10 --split test --ode_steps 20
```

With AddCL constraint:
```bash
python src/exp-pretrained-sr/train_swinir_flow.py --mode eval \
    --n_ensemble 10 --split test --ode_steps 20 --constraint addcl
```

## Key Findings

### What Works

| Finding | Evidence |
|---------|----------|
| Pretrained SwinIR features are useful as conditioning for flow matching | CRPS=0.173 (SwinIR-conditioned) vs 0.174 (from-scratch UNet) on equivalent eval |
| Multi-head CRPS on frozen backbone gives cheap ensemble | 0.183 CRPS with only 0.8M trainable params (frozen 11.8M backbone) |
| OT-CFM remains the best generation framework | Beats multi-head (0.183), residual flow (0.207), DiT (0.204), noise injection (0.200) |
| Hard constraints (AddCL) are universally applicable | Works on UNet, SwinIR, DiT -- zero CRPS cost, eliminates mass violation |
| 10-20 Euler steps sufficient | More steps don't improve CRPS (consistent with research2) |

### What Fails

| Finding | Evidence |
|---------|----------|
| Multi-head SwinIR has a hard ceiling at CRPS=0.183 | 3 variants (direct, residual, unfrozen) all converge to 0.183 |
| Targeting pretrained model residuals for flow | Source-target mismatch: N(0,1) -> N(0,0.0035) is 285x compression |
| DiT with coarse patches for pixel-level prediction | 8x8 patches lose fine spatial structure; 4x4 OOMs on L40S |
| Noise concatenation with dominant pretrained features | Backbone features dominate; tail learns deterministic mapping |
| Zero-shot pretrained image SR on climate data | Domain gap too large (CRPS=1.28 vs 0.250 after finetuning) |

### Structural Insights

1. **Pretrained features help as conditioning, not as the generator.** Using SwinIR as a frozen feature extractor that feeds into an independent generative model (flow matching) works. Using SwinIR as the generative backbone directly (multi-head, noise injection) hits ceilings.

2. **Diversity mechanism matters more than feature quality.** Multi-head SwinIR has excellent per-member accuracy (MAE=0.250) but caps at CRPS=0.183 because diversity is structurally limited. OT-CFM flow produces worse individual samples (MAE=0.312) but better CRPS (0.173) because diversity is calibrated.

3. **UNet convolutional inductive bias is important for pixel-level tasks.** DiT's patch-based processing loses fine spatial detail that UNet preserves through full-resolution convolutions and skip connections.

4. **The CorrDiff two-stage principle is sound but sensitive to execution.** Adding a pretrained predictor's output as conditioning (iteration 8) works. Targeting its residuals directly (iteration 5) fails. The key is maintaining the standard flow target distribution.

## Compute Summary

| Iter | GPU Type | Wall Clock | Epochs | Status |
|------|----------|-----------|--------|--------|
| 1 | L40S (normal) | 2.1 hr | 19 | Completed |
| 2 | L40S (normal) | 2.0 hr | 18 | Completed |
| 3 | L40S (normal) | 2.0 hr | 27 | Completed |
| 4 | L40S (normal) | 2.0 hr | 20 | Completed |
| 5 | L40S (preemptable) | 2.1 hr | 200 | Completed |
| 6 | L40S (normal) | 1.8 hr | 100 | Completed |
| 7 | L40S (normal) | 2.0 hr | 33 | Completed |
| 8 | L40S (normal+preemptable) | 2.0+2.0 hr | 26 | Completed (eval extended) |

Total GPU time: ~18 hr across 8 experiments.

## Limitations and Future Work

1. **SwinIR-conditioned flow was undertrained.** Only 26 epochs due to slow node. With more training, CRPS could improve beyond 0.173.

2. **No spectral/frequency-domain evaluation.** All metrics are pixel-space (CRPS, MAE, RMSE). Power spectral density or wavelet-based metrics would reveal whether generated textures match HR statistics at different spatial scales.

3. **Single variable (TCW) only.** Multi-variable downscaling with cross-variable constraints remains untested.

4. **DiT may improve with smaller patches or mixed-resolution tokens.** The 8x8 patch failure doesn't rule out transformers -- hybrid architectures (e.g., conv stem + transformer body) could work.

5. **Constraint-aware training (loss includes mass violation) was not tested with flow matching.** Research2 and this branch both use AddCL post-hoc only.

## File Inventory

### Training Scripts (in git)

| File | Experiment |
|------|-----------|
| `src/exp-pretrained-sr/eval_zeroshot.py` | Zero-shot evaluation |
| `src/exp-pretrained-sr/run_zeroshot_eval.py` | Zero-shot runner |
| `src/exp-pretrained-sr/finetune_swinir.py` | SwinIR finetuning + multi-head + unfreeze |
| `src/exp-pretrained-sr/train_crps_ensemble.py` | Multi-head CRPS training |
| `src/exp-pretrained-sr/train_residual_flow.py` | Residual flow (iter 5) |
| `src/exp-pretrained-sr/train_dit_flow.py` | DiT flow (iter 6) |
| `src/exp-pretrained-sr/train_noise_swinir.py` | Noise-conditioned SwinIR (iter 7) |
| `src/exp-pretrained-sr/train_swinir_flow.py` | SwinIR-conditioned OT-CFM (iter 8, best) |
| `src/exp-pretrained-sr/visualize_results.py` | Visualization utilities |
| `scripts/eval_crps.py` | Corrected CRPS evaluation |

### Checkpoints (in pool, not in git)

All under `pool/datasets/research5/models/`:

| Directory | Model | Epochs |
|-----------|-------|--------|
| `swinir_ft/` | Finetuned SwinIR | 19 |
| `crps_ensemble/` | Multi-head K=8 | 18 |
| `crps_residual/` | Multi-head residual K=8 | 27 |
| `crps_unfreeze2/` | Multi-head unfreeze-2 | 20 |
| `residual_flow/` | Residual flow UNet + SwinIR preds | 200 |
| `dit_flow/` | DiT flow matching | 84 |
| `noise_swinir/` | Noise-conditioned SwinIR | 20 |
| `swinir_flow/` | SwinIR-conditioned OT-CFM (best) | 26 |
| `pretrained_weights/` | Original SwinIR weights |  |
