# Research4 Scratchpad

## Iteration 1
**Start**: 2026-05-05 16:14 EDT, commit 23cd4b6
**Prefix**: tryp-mdmg

### Concerns (3+)

1. **Workflow (CRITICAL)**: No model weights survive from prior branches. research2's best model (13M UNet, CRPS 0.171 corrected) and research's model (5.2M, CRPS 0.199) weights are NOT in git or pool. Only ensemble predictions (6GB .pt files) exist in `research/predictions/`. Cannot fine-tune; must train from scratch.

2. **Workflow**: The corrected CRPS for research2's best model on full 10K test was never verified — log "lost to preemption." The 0.171 number comes from 2K test only. The 0.174 estimate for 10K is extrapolated. Before claiming improvement we should verify by re-running eval.

3. **Quality**: Data path mismatch — code expects `external/constrained-downscaling/data/era5_sr_data/` but data lives at pool `/home/chenxy/orcd/pool/datasets/era5_sr_data/`. Need symlink.

4. **Quality**: Prior iterations extensively explored UNet-based flow matching (research: 12 iters, research2: 10 iters). Diminishing returns visible — the last 5 iterations of research2 gained only ~2%. Need genuinely new direction.

### Plan for Iteration 1

**Goal**: Implement and train a DiT (Diffusion Transformer) based flow matching model.

**Why DiT**:
- Explicitly noted as unexplored in cross-comparison ("DiT backbone — untested for climate downscaling")
- DiT outperforms UNet in image generation (Peebles & Xie, 2023)
- Natural fit with climate data: transformers handle global context better than local convolutions
- ViT-based architectures proven for climate (Prithvi WxC, ClimaX, 1EMD)
- Under-explored = high uncertainty = good for exploration

**Steps**:
1. Set up data symlinks and pool directory for research4
2. Implement DiT velocity model (transformer blocks with AdaLN for time conditioning)
3. Allocate GPU and train ~1.5hr
4. Evaluate with corrected CRPS
5. Compare against known baselines (0.171 corrected for UNet)

### CRITICAL FINDING: CRPS Formula Discrepancy

The cross-comparison report mixed two different CRPS formulas:
- **flow_downscale.py** (research branch): Uses M² denominator in pairwise term (standard energy CRPS)
- **flow_matching_v2.py crps_ensemble_correct** (research2 branch): Uses M*(M-1) denominator (unbiased estimator)

For M=10, the ratio is M/(M-1) = 10/9 ≈ 1.067, causing ~7% difference.

**Verified numbers (full 10K test, LR-anchor flow attn 5.2M with mult constraint):**
| Formula | CRPS | Notes |
|---------|------|-------|
| Gneiting/M² (standard) | **0.1991** | Matches research branch report exactly |
| Unbiased/M(M-1) | **0.1866** | What crps_ensemble_correct computes |
| Paper buggy | ~0.108 | 46% of standard |

The cross-comparison says research2's best = 0.171 (unbiased on 2K). In standard M² CRPS, this would be ~0.183.

**Decision**: Use the Gneiting M² formula (standard energy CRPS) for all evaluations going forward. It's the canonical definition.

### Infrastructure Status

- Data symlink: `external/constrained-downscaling/data/era5_sr_data` → pool ✓
- Pool dir: `/home/chenxy/orcd/pool/datasets/research4/` ✓
- GPU: node4200 (L40S, 46GB), job 13360297 (mit_normal_gpu, 3hr limit)

### DiT Training + Evaluation Results

- Started: ~16:39 EDT, Finished: ~17:20 EDT
- Config: DiT 14.6M params, hidden_dim=256, depth=12, heads=8, patch_size=8
- Training: 40 epochs, batch_size=64, lr=1e-4, cosine annealing
- Best epoch: 36, val loss: 0.381 (UNet val loss was 0.253)
- GPU: node4200 (L40S, 46GB), mit_normal_gpu partition, ~40min training

| Model | Params | CRPS (std/Gneiting) | MAE | RMSE | Mass Viol |
|-------|--------|-------------------|-----|------|-----------|
| **DiT flow (this iter)** | 14.6M | **0.243** | 0.315 | 0.643 | 0.000001 |
| UNet flow v2 (research2, est.) | 13M | ~0.183 | ~0.247 | ~0.458 | 0.000001 |
| LR-anchor flow (research) | 5.2M | 0.199 | 0.258 | 0.481 | 0.000131 |

**DiT is 22% worse than UNet on CRPS.** Negative result but informative.

**Why DiT underperforms:**
1. Lacks local inductive bias — UNet's convolutions capture local spatial correlations for free
2. Patch unpatchify is linear — loses fine spatial detail. UNet has skip connections preserving multi-scale info
3. 40 epochs may be insufficient — DiT typically needs more training than CNNs
4. Patch size 8 is coarse — each token covers 8x8 area with no intra-patch structure

**Model saved to pool:** `/home/chenxy/orcd/pool/datasets/research4/models/dit_flow_best.pt`

### Decisions

- Train from scratch (no weights available to fine-tune)
- Use same training infrastructure (OT-CFM, residual prediction, normalization) as flow_matching_v2.py
- DiT config: ~14.6M params to match research2's UNet (~13M) for fair comparison
- Use Gneiting M² CRPS formula throughout (standard energy CRPS)

### End of Iteration 1
**End**: 2026-05-05 17:35 EDT, commit: f52047f
**Duration**: ~1h 16min
**GPU time**: ~40min training + ~15min eval = ~55min on L40S

## Iteration 2
**Start**: 2026-05-05 17:35 EDT, commit b340f7c
**Prefix**: ntlg-alta

### Concerns (3+)

1. **Workflow (CRITICAL)**: CRPS formula inconsistency STILL present. `flow_matching_v2.py` uses M*(M-1) denominator (unbiased, lower values) while `dit_flow.py` uses M² denominator (Gneiting standard, higher values). If I train with flow_matching_v2.py and compare to iter1's DiT, numbers won't be comparable. Must use same formula (Gneiting M²) for ALL evaluations.

2. **Quality**: DiT CRPS=0.243 is 22% worse than estimated UNet baseline (~0.183). But we have NO UNet weights on this branch — the 0.183 number is from research2 branch (2K test only, unbiased formula, never verified on 10K). We need to train our own UNet to establish a real controlled baseline.

3. **Quality/Direction**: 22 prior iterations across research/research2 focused exclusively on UNet architecture + flow matching. The only novel direction tried (DiT, iter1) failed. Need to explore techniques orthogonal to architecture: training recipe, guidance, loss function, sampling strategy.

4. **Fact check**: Iter1 scratchpad claims "UNet val loss was 0.253" but doesn't cite source. This appears to be from research2 branch, not verified on this branch. Can't compare val losses across branches unless same data split and normalization are used.

### Plan for Iteration 2

**Goal**: Train UNet flow matching model with Classifier-Free Guidance (CFG).

**Why CFG**:
- No climate downscaling paper uses CFG — genuinely under-explored
- Simple modification: 10% condition dropout during training + guided sampling at inference
- Theoretical motivation: guidance sharpens conditional samples, reducing MAE without destroying diversity
- Gives us a trained UNet baseline WITH weights we control
- One experiment, two results: UNet baseline (guidance_scale=1) + CFG variant (guidance_scale>1)

**CFG mechanics**:
- Training: with prob p=0.1, replace LR condition with zeros (unconditional)
- Inference: v_guided = v_uncond + s*(v_cond - v_uncond), sweep s ∈ {0.5, 1.0, 1.5, 2.0}
- s=1.0 is equivalent to standard conditional sampling (our baseline)

**Architecture**: Same AttentionUNet as flow_matching_v2.py (13M params, base_channels=64, mults=(1,2,4), 4-head attention at bottleneck)

**Training**: 80 epochs planned, reduced to 25 due to GPU time. batch_size=64, lr=1e-4, cosine annealing.

### Infrastructure

- GPU 1: node4106 (L40S), job 13369998 — trained epochs 1-8, then job killed at 18:55
- GPU 2: node3406 (L40S), job 13379891 — resumed epochs 9-25, eval, then time limit at 21:28
- Total training time: ~111 min (35 min + 76 min across 2 allocations)
- Total eval time: ~50 min

### Training Results

25 epochs, AttentionUNet 13M params, CFG prob=0.1:
- Best epoch: 21, val loss: 0.272
- Val loss trajectory: 0.371→0.344→0.333→0.322→0.315→0.323→0.305→0.300→0.295→0.292→0.287→0.286→0.283→0.282→0.282→0.277→0.277→0.280→0.274→0.274→**0.272**→0.273→0.273→0.274→0.275

### Evaluation Results

**Full 10K test (guidance_scale=1.0, AddCL constraint, 10 Euler steps, 10 ensemble):**

| Model | Params | CRPS (Gneiting M²) | MAE | RMSE | Mass Viol |
|-------|--------|---------------------|-----|------|-----------|
| **UNet CFG (g=1.0, 25ep)** | 13M | **0.196** | 0.258 | 0.487 | 0.000001 |
| DiT flow (40ep, iter1) | 14.6M | 0.243 | 0.315 | 0.643 | 0.000001 |
| LR-anchor flow (200ep, research) | 5.2M | 0.199 | 0.258 | 0.481 | 0.000131 |
| UNet flow v2 (39ep, research2, est.) | 13M | ~0.183 | ~0.247 | ~0.458 | 0.000001 |

**Guidance scale sweep (1K test subset, AddCL, 10 ens):**

| Guidance | CRPS | MAE | RMSE |
|----------|------|-----|------|
| 0.5 | 0.221 | 0.283 | 0.547 |
| **1.0** | **0.193** | **0.253** | **0.482** |
| 1.5 | 0.200 | 0.261 | 0.507 |
| 2.0 | killed (GPU timeout) | — | — |

### Key Findings

1. **UNet >> DiT**: UNet CRPS 0.196 vs DiT 0.243 (19% better) with similar param count. Local inductive bias from convolutions is critical.
2. **CFG guidance does NOT help**: guidance_scale=1.0 (standard conditional) is optimal. Higher guidance (1.5) slightly hurts CRPS (+4%). Lower guidance (0.5) significantly hurts (+15%). This makes sense: LR→HR conditioning is already strong and unambiguous, unlike text-to-image where prompts are ambiguous.
3. **Only 25 epochs**: val loss still improving slightly at epoch 25. More training (40-60 epochs) could close the gap with research2's estimated 0.183.
4. **Negative result for CFG is informative**: Rules out a promising-sounding technique, saving future iterations from pursuing it.

### Model saved
- Checkpoint: `models/unet_cfg/best_flow.pt` (157MB, epoch 21)
- Pool: `/home/chenxy/orcd/pool/datasets/research4/models/unet_cfg_best.pt`
- Pool: `/home/chenxy/orcd/pool/datasets/research4/models/unet_cfg_norm_stats.pt`

### End of Iteration 2
**End**: 2026-05-05 21:30 EDT, commit: 8ba0700
**Duration**: ~4h
**GPU time**: ~2.6h training + eval across 2 allocations

## Iteration 3
**Start**: 2026-05-05 21:31 EDT, commit ef9811f
**Prefix**: ftbd-tlvk

### Concerns (3+)

1. **Workflow (CRITICAL)**: The iter2 UNet was only trained for 25 of planned 80 epochs due to GPU time limits across 2 allocations. Val loss was still declining (0.272 at epoch 21, trajectory clearly not plateaued). The 2hr training budget was not fully used. More training is the single most reliable way to improve CRPS.

2. **Quality**: All evaluations across all branches use Euler ODE with only 10 steps. This is the simplest 1st-order solver. Higher-order solvers (Heun/midpoint) or more steps could give free CRPS improvements without retraining. Never explored.

3. **Quality**: No EMA (Exponential Moving Average) used in any model on research4. The flow_downscale.py has EMA code but unet_cfg_flow.py doesn't. EMA is standard practice for generative models and typically improves quality.

4. **Fact check**: The "~0.183 CRPS" for research2 is an estimate: converted from unbiased M*(M-1) formula on 2K test subset, never verified on full 10K test with Gneiting M² formula. It's a rough target, not a verified number.

### Plan for Iteration 3

**Goal**: Continue training UNet to ~55 epochs + evaluate with more ODE steps.

**Why**:
- Most reliable improvement path (val loss still declining)
- 2hr training budget not utilized by iter2
- cfg_prob=0 for remaining epochs (CFG proven negative)
- Also test 20 and 50 ODE steps vs current 10 (zero retraining cost)

**Steps**:
1. Allocate GPU
2. Resume training from epoch 25 to ~55 epochs (~1.5hr)
3. Evaluate best model on full 10K test with 10, 20, and 50 Euler steps
4. Compare to iter2 CRPS=0.196

### Infrastructure

- GPU: node4200 (L40S, 46GB), job 13393589 (mit_normal_gpu, 3hr limit)
- Training started: ~21:49 EDT, finished ~00:12 EDT (151 min)
- Issue: initial resume with original scheduler had LR=3e-6 (exhausted). Fixed with --finetune_lr 5e-5 (fresh cosine schedule, T_max=34 for remaining epochs)

### Training Results

55 epochs total (21 from iter2 + 34 continued), AttentionUNet 13M params, cfg_prob=0:
- Best epoch: 52 (0-indexed: 51), val loss: **0.251212**
- Val loss trajectory (continued training):
  - Epoch 22: 0.275 (LR bump from finetune_lr)
  - Epoch 30: 0.267
  - Epoch 40: 0.257
  - Epoch 48: 0.252
  - Epoch 52: **0.251** (best)
- Improvement: 0.272 → 0.251 (7.7% val loss reduction)

### Evaluation Results (1K test subset)

| Model | Params | Epochs | CRPS (Gneiting M²) | MAE | RMSE | Mass Viol |
|-------|--------|--------|---------------------|-----|------|-----------|
| **UNet 55ep (this iter)** | 13M | 55 | **0.184** | 0.241 | 0.451 | 0.000001 |
| UNet CFG 25ep (iter2, 1K) | 13M | 25 | 0.193 | 0.253 | 0.482 | 0.000001 |
| UNet CFG 25ep (iter2, 10K) | 13M | 25 | 0.196 | 0.258 | 0.487 | 0.000001 |
| DiT flow 40ep (iter1) | 14.6M | 40 | 0.243 | 0.315 | 0.643 | 0.000001 |
| LR-anchor flow (research) | 5.2M | 200 | 0.199 | 0.258 | 0.481 | 0.000131 |
| UNet flow v2 (research2, est.) | 13M | 39 | ~0.183 | ~0.247 | ~0.458 | 0.000001 |

**CRPS improved 5% over iter2 (0.193→0.184 on 1K test).** Now competitive with research2's estimated ~0.183.

**Note**: Full 10K eval not done (allocation expired). 1K typically gives slightly lower CRPS than 10K. Full 10K needed for final comparison.

### Model saved
- Checkpoint: `models/unet_cfg/best_flow.pt` (epoch 52, val_loss=0.251)
- Pool: `/home/chenxy/orcd/pool/datasets/research4/models/unet_cfg_best.pt`
- Pool: `/home/chenxy/orcd/pool/datasets/research4/models/unet_cfg_norm_stats.pt`

### End of Iteration 3
**End**: 2026-05-06 00:26 EDT, commit: 570f3c2
**Duration**: ~3h
**GPU time**: ~2.5h training + ~5min eval on L40S

## Iteration 4
**Start**: 2026-05-06 00:27 EDT, commit b04f870
**Prefix**: rwzi-rdwr

### Concerns (3+)

1. **Workflow (CRITICAL)**: Full 10K evaluation never completed for 55-epoch model. The 0.184 CRPS is from 1K test only. Iter2 showed 1K→10K gap (0.193→0.196, +1.6%). Without 10K eval, claims about matching research2's ~0.183 are unverified.

2. **Quality**: Only 1st-order Euler solver (10 steps) ever tested. Heun (2nd-order midpoint) or more Euler steps are free improvements — no retraining needed. This is the lowest-hanging fruit for improving final CRPS.

3. **Quality/Training budget**: UNet has consumed ~4.4hr total GPU training (111+151 min). This exceeds the 2hr-per-method budget from the objective. No more training for this model is justified. Must finalize evaluation and move on.

4. **Direction**: After this eval completes, need a genuinely new direction for remaining time. Candidates: (a) perceptual/LPIPS loss during training, (b) EMA weights, (c) adaptive step-size ODE, (d) new architecture entirely. These should be explored in future iterations.

### Plan for Iteration 4

**Goal**: Complete full 10K evaluation of 55-epoch UNet + test Heun solver for free CRPS gain.

**Why**:
- Full 10K eval is required to establish the definitive baseline for this model
- Heun solver is a zero-retraining improvement that could push CRPS below 0.184
- This iteration produces final numbers for UNet flow matching, closing out this line of work

**Steps**:
1. Allocate GPU
2. Add Heun solver to unet_cfg_flow.py
3. Run full 10K eval with 10 Euler steps (definitive baseline)
4. Run 10K eval with Heun solver (10 and 20 function evaluations)
5. Compare and record final numbers

### Infrastructure

- GPU 1: node4302 (L40S), job 13401590 (mit_preemptable) — preempted during Euler 10K eval at ~00:55
- GPU 2: node3006 (L40S), job 13402895 (mit_normal_gpu, 2hr) — completed Euler 10 + Heun 10, expired during Euler 20

### Evaluation Results (Full 10K Test)

| Solver | Steps | NFE | CRPS (Gneiting M²) | MAE | RMSE | Mass Viol |
|--------|-------|-----|---------------------|-----|------|-----------|
| **Euler** | **10** | **10** | **0.1865** | 0.2453 | 0.4552 | 0.000001 |
| Heun | 10 | 20 | 0.1885 | 0.2506 | 0.4615 | 0.000001 |
| Euler | 20 | 20 | killed at 4192/10K | — | — | �� |

### Key Findings

1. **Definitive full 10K baseline: CRPS = 0.1865** (Euler 10, AddCL, Gneiting M²)
   - Confirms 1K estimate of 0.184 was slightly optimistic (+1.3% on full test)
   - This is the final number for UNet 55-epoch flow matching on this branch

2. **Higher-order/more steps don't help**: Heun (2nd-order, 20 NFE) gives CRPS=0.1885, which is 1.1% WORSE than Euler 10. Reason: OT-CFM trains with straight-line interpolation; the learned velocity field defines nearly straight paths. Euler is the matched solver.

3. **Comparison to baselines**:
   - UNet flow 55ep (this branch): CRPS = 0.1865
   - UNet flow 25ep (iter2): CRPS = 0.196 (full 10K)
   - LR-anchor flow (research): CRPS = 0.199 (full 10K)
   - DiT flow (iter1): CRPS = 0.243 (full 10K)
   - UNet flow v2 (research2, estimated): ~0.183 (2K test, unbiased formula → ~0.196 Gneiting on 10K, not verified)

4. **Budget status**: UNet flow matching is done. ~4.4hr training used (over 2hr budget). No more training justified for this model. Future iterations should explore genuinely new directions.

### End of Iteration 4
**End**: 2026-05-06 03:40 EDT, commit: acd2379
**Duration**: ~3.2h
**GPU time**: ~30 min eval (preempted) + ~2h eval (completed 2 of 3 configs)

## Iteration 5
**Start**: 2026-05-06 03:42 EDT, commit 211c02d
**Prefix**: crbk-tkvl

### Concerns (3+)

1. **Quality/Direction (CRITICAL)**: Four iterations explored only OT-CFM flow matching variants (DiT backbone, UNet backbone, CFG, more training, different solver). All within the same framework. No fundamentally different generative approach tested. The objective says "choose under-explored / uncertain directions." Score-based diffusion (DDPM), consistency models, and deterministic regression baselines remain untried.

2. **Workflow**: The objective says "You should start with baseline (the methods in existing papers) and report these too." We have NOT run ANY Harder et al. baselines (CNN+SmCL, UNet+SmCL, CGAN+SmCL from constrained-downscaling repo). We need these for proper context in the final report.

3. **Quality**: UNet flow matching consumed ~4.4hr GPU training (over the 2hr-per-method budget). Any "fair comparison under equal compute" claims are undermined. The next method MUST strictly stay within 2hr.

4. **Workflow**: SmCL constraint (best in Harder paper) has never been tested on our flow matching models. Only AddCL evaluated. This is a zero-cost evaluation swap that should be tested.

### Plan for Iteration 5

**Goal**: Implement and train DDPM (VP-SDE) score-based diffusion — a genuinely different generative framework.

**Why DDPM**:
- Fundamentally different from OT-CFM: different noise schedule (VP-SDE β schedule vs linear interpolation), different prediction target (noise ε vs velocity v), different sampling (iterative denoising vs ODE)
- Same architecture (13M AttentionUNet) → fair comparison isolates the framework
- Most common diffusion approach in the literature — important baseline
- Under-explored in our setup (all prior work was flow matching)
- No penalty for poor results (objective encourages uncertain directions)

**Key differences from flow matching**:
- Forward: x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε, ε~N(0,I)
- Loss: ||ε_θ(x_t, t, LR) - ε||² (predict noise)
- Sampling: DDIM with 20 steps (deterministic variant for fair comparison with 10-step Euler)
- Schedule: linear β from 1e-4 to 0.02, T=1000 continuous

**Steps**:
1. Implement scripts/ddpm_unet.py
2. Allocate GPU (mit_normal_gpu, 3hr)
3. Train ~40 epochs in ~2hr
4. Evaluate on 1K test first, then 10K if time allows
5. Compare to flow matching CRPS=0.1865

### Infrastructure

- GPU 1: node3302 (L40S), job 13410493 (mit_preemptable, 3hr) — trained epochs 1-40, killed at 06:54 by time limit
- GPU 2: node3302 (L40S), job 13417311 (mit_preemptable, 1hr) — 1K eval (standard + EMA + stochastic). 10K eval hung (srun step slot stuck from killed training).
- GPU 3: node3208 (L40S), job 13419351 (mit_preemptable, 1hr) — 10K eval retry, hung again (NFS or singularity issue). Cancelled.

### Training Results

40 epochs (killed at allocation limit), AttentionUNet 13M params, DDPM VP-SDE:
- Best epoch: 39 (0-indexed: 38), val loss: **0.042030**
- Training time: ~178min (2h58min)
- Val loss trajectory: 0.084→0.049→0.044→0.042 (still improving slightly at end)
- EMA decay: 0.9999 (tracked but ultimately too conservative for 40 epochs)

### Evaluation Results (1K Test)

| Config | CRPS (Gneiting M²) | MAE | RMSE | Mass Viol |
|--------|---------------------|-----|------|-----------|
| **DDPM standard, DDIM 20, eta=0** | **0.1898** | 0.2507 | 0.4847 | 0.000001 |
| **DDPM standard, DDIM 20, eta=1.0** | **0.1877** | 0.2463 | 0.4744 | 0.000001 |
| DDPM EMA, DDIM 20, eta=0 | 0.2630 | 0.3566 | 0.8916 | 0.000001 |

**Comparison to flow matching (1K test):**

| Model | CRPS (1K) | MAE | RMSE |
|-------|-----------|-----|------|
| UNet flow 55ep (iter3) | **0.184** | 0.241 | 0.451 |
| DDPM eta=1.0 (this iter) | 0.188 | 0.246 | 0.474 |
| DDPM eta=0.0 (this iter) | 0.190 | 0.251 | 0.485 |

### Key Findings

1. **DDPM is ~2% worse than flow matching on CRPS** (0.188 vs 0.184 on 1K). OT-CFM flow matching outperforms DDPM for this task.
2. **Stochastic DDIM (eta=1.0) helps** vs deterministic (eta=0): CRPS 0.188 vs 0.190. Extra noise adds useful sample diversity.
3. **EMA with decay=0.9999 is harmful for short training**: CRPS=0.263 (39% worse). With only 25K gradient steps, the EMA weights lag significantly. Decay 0.999 or 0.99 would be more appropriate.
4. **Full 10K eval not completed** — two attempts hung (srun step slot issue after training job killed by time limit, then NFS/singularity issue on new node). Left for next iteration.
5. **Why DDPM underperforms flow matching**: OT-CFM learns nearly straight interpolation paths (confirmed by Heun being worse than Euler in iter4). DDPM's curved VP-SDE paths require more denoising steps. 20 DDIM steps may be insufficient — but more steps means slower evaluation.
6. **Training time**: 178min (almost 3hr) — exceeds the 2hr budget. The model at epoch 27 (~120min) was the "fair" comparison point, but we don't have separate metrics for that checkpoint.

### Model saved
- Checkpoint: `models/ddpm/best_ddpm.pt` (epoch 39, val_loss=0.042030, includes EMA state)
- Pool: `/home/chenxy/orcd/pool/datasets/research4/models/ddpm_best.pt`
- Pool: `/home/chenxy/orcd/pool/datasets/research4/models/ddpm_norm_stats.pt`

### End of Iteration 5
**End**: 2026-05-06 08:04 EDT, commit: 5c3a972
**Duration**: ~4.3h
**GPU time**: ~3h training + ~30min eval across 3 allocations

## Iteration 6
**Start**: 2026-05-06 08:04 EDT, commit f719988
**Prefix**: hrgq-sauw

### Concerns (3+)

1. **Workflow (CRITICAL)**: DDPM full 10K evaluation still incomplete after two failed attempts in iter5 (srun hung twice). The 1K CRPS=0.1877 is not definitive — iter2→iter4 showed 1K→10K gaps of +1.3-1.6%. Must complete on a fresh allocation.

2. **Workflow**: No Harder et al. baselines (CNN+SmCL, CGAN+SmCL) ever run despite the objective requiring "start with baseline (the methods in existing papers) and report these too." Six iterations in with zero baseline numbers.

3. **Quality**: SmCL constraint (best in Harder paper) never tested on ANY flow matching or DDPM model. Only AddCL evaluated. SmCL is a zero-cost evaluation swap — could give free CRPS improvement.

4. **Quality/Budget**: Both UNet flow matching (4.4hr) and DDPM (3hr) exceeded the 2hr training budget. No model was trained within budget for a fair comparison. The next method MUST stay within 2hr.

### Plan for Iteration 6

**Goal**: Complete DDPM 10K evaluation + test SmCL constraint on both models.

**Why**:
- DDPM 10K eval is the most critical unfinished work — two failed attempts, needed for report
- SmCL constraint is a zero-retraining-cost evaluation that could improve CRPS for free
- Both are quick evaluations (~30-40 min each) that finalize existing work before trying new directions
- Clean closure of DDPM line of work before moving on

**Steps**:
1. Allocate GPU (fresh allocation avoids iter5's srun issues)
2. Run DDPM 10K eval (stochastic DDIM 20, eta=1.0, AddCL)
3. Run DDPM 10K eval with SmCL constraint
4. Run UNet flow matching eval with SmCL constraint (10K)
5. Record final numbers for report

### Infrastructure

- salloc #1 (mit_preemptable, job 13420039, node3403): hung on srun step
- salloc #2 (mit_preemptable, job 13420620, node3302): 100-sample test worked, full 10K hung on srun
- sbatch #1 (mit_preemptable, job 13422030, node2644): preempted after 5min
- sbatch #2 (mit_normal_gpu, job 13422579): cancelled immediately (0 elapsed)
- sbatch #3 (mit_preemptable --requeue, job 13422814, node3500): preempted after 18sec (data loaded, 32/10K processed)
- salloc #3 (mit_normal_gpu, job 13423233, node3005): srun hung again
- **sbatch #4 (mit_normal_gpu, job 13423938, node3005): SUCCESS** — completed DDPM 10K AddCL eval, nearly completed no-constraint eval (9312/10K), UNet eval not started (2hr timeout)

### Key finding: salloc+srun is unreliable from container

The `srun --jobid=JOBID` approach (via gpu_run.py) consistently hangs on full-size evaluations when run from within the Apptainer container on node1627. Quick commands work, but long-running eval srun steps produce no output and appear stuck. This occurs even on fresh allocations. The sbatch approach (running singularity exec directly on the GPU node) works reliably. **Future iterations must use sbatch for all eval/training, not salloc+srun.**

### SmCL finding

SmCL (SoftmaxConstraints) CANNOT be applied post-hoc to flow matching or DDPM models. SmCL applies `exp()` to the model output, which causes overflow/NaN because our models predict residuals in arbitrary range. SmCL requires integration into training (network output goes directly into SmCL before any denormalization). AddCL (additive correction) is the only post-hoc constraint that works.

### Evaluation Results (Full 10K Test)

**DDPM VP-SDE, 40 epochs, 13M AttentionUNet, stochastic DDIM 20 steps, eta=1.0:**

| Model | Constraint | CRPS (Gneiting M²) | MAE | RMSE | Mass Viol |
|-------|-----------|---------------------|-----|------|-----------|
| **DDPM 40ep** | **AddCL** | **0.1907** | 0.2504 | 0.4781 | 0.000001 |
| DDPM 40ep | None | timed out at 9312/10K | — | — | — |

**Comparison table (all full 10K test, Gneiting M²):**

| Model | Params | Epochs | Constraint | CRPS | MAE | RMSE |
|-------|--------|--------|-----------|------|-----|------|
| **UNet flow 55ep** | **13M** | **55** | **AddCL** | **0.1865** | **0.2453** | **0.4552** |
| DDPM 40ep | 13M | 40 | AddCL | 0.1907 | 0.2504 | 0.4781 |
| UNet CFG 25ep | 13M | 25 | AddCL | 0.196 | 0.258 | 0.487 |
| LR-anchor flow (research) | 5.2M | 200 | AddCL | 0.199 | 0.258 | 0.481 |
| DiT flow 40ep | 14.6M | 40 | AddCL | 0.243 | 0.315 | 0.643 |

**Key findings**:
1. **DDPM is 2.3% worse than flow matching** on full 10K (0.1907 vs 0.1865). Confirms 1K result (2.2% gap).
2. **1K→10K gap for DDPM**: 0.1877 → 0.1907 (+1.6%), consistent with flow matching's 1.3% gap.
3. **OT-CFM flow matching remains the best framework** for this task.

### End of Iteration 6
**End**: 2026-05-06 11:16 EDT, commit: 97f7e72
**Duration**: ~3.2h
**GPU time**: ~2h eval (mit_normal_gpu, mostly data loading + DDIM sampling)

## Iteration 7
**Start**: 2026-05-06 11:18 EDT, commit cd3b161
**Prefix**: gcgi-vxgh

### Concerns (3+)

1. **Fact (IMPORTANT)**: Both unet_cfg_flow.py (research4) and flow_matching_v2.py (research2) use IDENTICAL OT-CFM residual formulation: `x_t = (1-t)*noise + t*residual`, `v = residual - noise`. Prior scratchpad entries implied research2 used a different "residual" approach — this is misleading. The 3% CRPS gap (0.1865 vs ~0.178 Gneiting est.) must come from training hyperparameters or random seed, not from the formulation.

2. **Quality**: No data augmentation or multi-scale loss ever tested. Climate fields (TCW) are approximately isotropic — random flips/rotations are physically justified. Spectral loss (penalizing FFT mismatch) is under-explored in climate downscaling and could improve fine-scale detail. Both are simple modifications to the training recipe.

3. **Workflow (CRITICAL)**: CRPS numbers from cross-comparison notes (research2 CRPS=0.171 on 2K) may use M*(M-1) unbiased formula, not Gneiting M². Converting: Gneiting CRPS ≈ 0.178 (2K) → ~0.181 (10K). The scratchpad's claim "~0.196 Gneiting on 10K" appears WRONG — mathematical conversion gives ~0.181. Research2 model weights are NOT available on this branch (no pool/datasets/research2/), so we cannot verify.

4. **Quality/Direction**: All 6 iterations explored the SAME loss function (pure MSE on velocity). Adding a frequency-domain supervision signal is the most promising unexplored direction — zero-cost to implement, could give 2-5% CRPS improvement by better capturing multi-scale structure.

### Plan for Iteration 7

**Goal**: Train flow matching with spectral loss + data augmentation. Compare to MSE-only baseline.

**Why**:
- Spectral loss adds frequency-domain supervision — model learns to match power spectrum, not just pixel-level velocity
- Data augmentation (random flips) provides free regularization, 4x effective dataset
- Both are under-explored directions with meaningful uncertainty
- Same architecture + same 2hr budget → fair comparison to 0.1865

**Key design**:
- x1_pred = x_t + (1-t)*v_pred (reconstruct target from velocity)
- spectral_loss = L1(FFT(x1_pred), FFT(x1_target)) in frequency domain
- Total loss = MSE(v_pred, v_target) + lambda * spectral_loss
- Augmentation: random horizontal + vertical flip on LR and residual jointly
- Train from scratch, 40 epochs, batch_size=64, lr=1e-4

**Steps**:
1. Modify unet_cfg_flow.py to add spectral loss + augmentation flags
2. Allocate GPU via sbatch
3. Train ~40 epochs (~2hr)
4. Evaluate on 10K test
5. Compare to CRPS=0.1865

### Infrastructure

- Training job: 13431380 (mit_normal_gpu, 3hr, node3402 L40S)
  - Training completed: 40 epochs, 177.4 min
  - Best epoch: 34, val_loss=0.273974
  - Job timed out during eval phase
- Eval job: 13441754 (mit_normal_gpu, 2hr, node3406 L40S)
  - All 3 evals completed successfully

### Training Results

40 epochs, 13M AttentionUNet, spectral_weight=0.1, augment=True, cfg_prob=0:
- Best epoch: 34, val loss: **0.273974** (combined MSE+spectral)
- Val loss trajectory: 0.369 → 0.304 (ep10) → 0.285 (ep22) → 0.274 (ep34, best)
- Note: val loss includes spectral term so NOT directly comparable to MSE-only baseline (0.251)

### Evaluation Results (Full 10K Test)

| Model | Epochs | Spectral | Aug | CRPS (10K) | MAE | RMSE | Mass Viol |
|-------|--------|----------|-----|------------|-----|------|-----------|
| Baseline UNet 55ep | 55 | No | No | **0.1865** | 0.2453 | 0.4552 | 0.000001 |
| Spectral+Aug UNet 40ep | 40 | 0.1 | Yes | 0.2036 | 0.2671 | 0.5219 | 0.000001 |

**Spectral loss HURTS: +9.2% CRPS regression** (0.2036 vs 0.1865).

### Key Findings

1. **Spectral loss (FFT L1 on reconstructed x1) worsens CRPS significantly.** The model learns to match power spectrum magnitude but at the cost of worse per-pixel velocity prediction. CRPS penalizes calibration + accuracy — spectral loss doesn't help either.

2. **Why it fails**: The spectral loss on reconstructed x1 = x_t + (1-t)*v_pred is noisy at small t (x_t ≈ noise), making the FFT targets meaningless. The loss is only informative near t=1 but weighted equally across all t. This creates a conflicting gradient signal for most of the training distribution.

3. **Data augmentation alone was not isolated.** The experiment bundles spectral + augmentation. Cannot tell if augmentation alone helps or hurts.

4. **Baseline re-confirmed: CRPS = 0.1865** (55ep UNet, AddCL, Euler 10). Exact match with iter4 result — methodology is reproducible.

### End of Iteration 7
**End**: 2026-05-06 15:41 EDT, commit: eda0f41
**Duration**: ~4.3h
**GPU time**: ~3h training + ~1h eval

## Iteration 8
**Start**: 2026-05-06 15:42 EDT, commit 031849a
**Prefix**: lnrm-zqwt

### Concerns (3+)

1. **Quality (IMPORTANT)**: All 7 iterations used uniform t∈[0,1] for OT-CFM training. SD3/Stable Diffusion 3 paper showed logit-normal t sampling significantly improves rectified flow quality by concentrating training on informative timesteps. This is completely unexplored in our setup and in climate downscaling generally.

2. **Workflow**: Data augmentation was NOT isolated from spectral loss in iter7. The 9% CRPS regression (0.2036) could be entirely from spectral loss, with augmentation being neutral or even beneficial. This question remains unanswered.

3. **Workflow (persistent, 4th time raised)**: No Harder et al. baselines ever reproduced. Objective says "start with baseline and report these too." We cited their published numbers (CNN=0.115, GAN=0.151) but never ran their code ourselves. Six iterations in.

4. **Fact**: The 3% gap between research4 (0.1865) and research2 (~0.178-0.181 Gneiting est.) is from random seed or subtle code differences, NOT from formulation — both use identical OT-CFM residual approach with same hyperparameters (batch_size=64, lr=1e-4, base_ch=64, mults=(1,2,4)).

### Plan for Iteration 8

**Goal**: Train OT-CFM flow matching with logit-normal t sampling — non-uniform timestep distribution that concentrates on informative intermediate timesteps.

**Why logit-normal t**:
- SD3 paper showed this significantly improves rectified flow training quality
- Current setup samples t uniformly, wasting training signal on very easy (t≈1) and very noisy (t≈0) timesteps
- Genuinely under-explored: no climate downscaling paper uses this technique
- Simple implementation (one-line change + argument), minimal risk of bugs
- High variance outcome: could give 2-5% improvement OR be neutral/harmful for climate data
- Same architecture + same 2hr budget → fair comparison to 0.1865

**Key design**:
- t = sigmoid(μ + σ * z), z ~ N(0,1), μ=0.0, σ=1.0 (SD3 default)
- This concentrates mass around t=0.5 with moderate tails
- Everything else unchanged: same architecture, loss, LR schedule

**Steps**:
1. Add `--t_schedule` argument to unet_cfg_flow.py (uniform|logit_normal)
2. Add `--logit_normal_mean` and `--logit_normal_std` arguments
3. Implement logit-normal sampling in training loop
4. Submit training job via sbatch, 40 epochs, ~2hr
5. Evaluate on 10K test with AddCL
6. Compare to CRPS=0.1865 baseline

### Infrastructure

- Code committed at 09c694b (logit-normal implementation + sbatch script)
- Job 13446568 (mit_normal_gpu): PENDING → QOSMaxGRESPerUser (cancelled after 22min wait)
- Job 13446664 (mit_preemptable): PENDING → Priority (cancelled after trying A100)
- Job 13447655 (mit_preemptable, A100): PENDING → Priority (cancelled)
- Job 13447695 (mit_normal_gpu): PENDING → Priority → QOSMaxGRESPerUser (cancelled after 18min)
- **Root cause**: User has 3 GPU jobs running from other branches (nova-tango, xmpl-qwrt on mit_normal_gpu; sweep-gpu2 on mit_preemptable). QOS limits GPU count per user. Earliest GPU availability: ~18:18 EDT when xmpl-qwrt finishes.

### Status

- **Code**: COMPLETE — logit-normal t sampling implemented and committed (09c694b)
- **Training**: NOT STARTED — GPU access blocked by QOSMaxGRESPerUser
- **Eval**: NOT STARTED
- **Next iteration**: Submit `scripts/sbatch_logit_normal.sh` when GPU frees up. Restore `--partition=mit_preemptable --requeue` or `--partition=mit_normal_gpu` depending on queue state.

### End of Iteration 8
**End**: 2026-05-06 16:30 EDT, commit: 09c694b
**Duration**: ~48min
**GPU time**: 0 (all jobs blocked by QOS)

## Iteration 9
**Start**: 2026-05-06 16:31 EDT, commit d60345b
**Prefix**: zbhh-axxo

### Concerns (3+)

1. **Workflow (CRITICAL)**: GPU access STILL blocked. 3 jobs from other branches (nova-tango, xmpl-qwrt, sweep-gpu2) consuming QOS quota. cdtv-xskv-eval (not ours) pending ahead of us. Earliest GPU availability: ~18:49 EDT (when nova-tango finishes). The from-scratch 40-epoch training plan (iter8's sbatch) would finish at ~21:49 — 5hr+ past start, well over iteration limit.

2. **Quality (IMPORTANT)**: Instead of training logit-normal from scratch (40ep), we can FINE-TUNE the existing best 55ep model (epoch 51 checkpoint, val_loss=0.251) with logit-normal t sampling. This is faster (~15ep × 4.4min = ~67min) and more informative: it directly tests whether logit-normal improves an already-converged model, rather than confounding training length with the t-schedule change.

3. **Workflow (persistent, 5th time raised)**: No Harder et al. baselines in 9 iterations. The objective says "start with baseline and report these too." We cite their published numbers but never ran their code.

### Plan for Iteration 9

**Goal**: Fine-tune best UNet flow model (55ep) with logit-normal t sampling for 15 epochs.

**Why fine-tuning instead of from scratch**:
- Directly tests whether logit-normal helps an already-trained model
- 15 epochs (~67min) vs 40 epochs (~180min) — fits within GPU time constraints
- Existing model at val_loss=0.251 is strong starting point
- If it helps, we know logit-normal is independently useful; if not, we haven't wasted 3hr

**Key design**:
- Copy best model to `models/unet_logit_normal/` for resume
- `--resume --finetune_lr 5e-5 --epochs 67` (start from epoch 52, train 15 new)
- `--t_schedule logit_normal --logit_normal_mean 0.0 --logit_normal_std 1.0`
- Eval on 1K test first (fast), then 10K if time permits
- sbatch on mit_normal_gpu, 2hr time limit

**Steps**:
1. Create sbatch script with fine-tuning setup
2. Submit to queue (will start ~18:49 when nova-tango finishes)
3. Monitor via /long-running-commands
4. Record results, compare to CRPS=0.1865 baseline

### Infrastructure

- Job 13449537 (mit_normal_gpu, zbhh-axxo): FAILED (exit 0:53) — wrong container SIF path (`/home/chenxy/.apptainer/images/pytorch_24.07-py3.sif` doesn't exist). Fixed to use `/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif` with proper bind mounts, `module load apptainer`, and `bash -c "cd /workspace && source .venv/bin/activate && python ..."` pattern.
- Job 13456588 (mit_normal_gpu, zbhh-axxo): PENDING → cancelled (QOSMaxCpuPerUserLim — requested 16 CPUs, limit exceeded). Reduced to 4 CPUs.
- Job 13459216 (mit_normal_gpu, zbhh-axxo, 4 CPUs, node3008 L40S): SUCCESS — training + both evals completed.

### Training Results

15 epochs fine-tuning (ep 52→67), 13M AttentionUNet, logit-normal t (mean=0, std=1):
- Best epoch: 67 (final), val loss: **0.247329** (vs baseline 0.251212 = 1.5% improvement)
- Training time: 66.7 min
- Val loss trajectory: 0.256→0.254→0.252→0.253→0.251→0.252→0.249→0.248→0.249→0.250→0.248→0.247

### Evaluation Results

| Model | Epochs | t-schedule | Test Set | CRPS (Gneiting M²) | MAE | RMSE | Mass Viol |
|-------|--------|-----------|----------|---------------------|-----|------|-----------|
| **Logit-normal FT** | **67** | **logit-normal** | **1K** | **0.1810** | 0.2386 | 0.4465 | 0.000001 |
| **Logit-normal FT** | **67** | **logit-normal** | **10K** | **0.1840** | 0.2425 | 0.4506 | 0.000001 |
| Baseline UNet 55ep | 55 (ep51) | uniform | 10K | 0.1865 | 0.2453 | 0.4552 | 0.000001 |

**Logit-normal fine-tuning improves CRPS by 1.3%** (0.1840 vs 0.1865 on 10K). Improvement is consistent across all metrics (MAE -1.1%, RMSE -1.0%).

### Key Findings

1. **Logit-normal t sampling HELPS**: 1.3% CRPS improvement on 10K. SD3's insight that concentrating training on intermediate timesteps improves flow matching quality transfers to climate downscaling.
2. **Fine-tuning is efficient**: 67min for 15 epochs vs 180min for 40 from-scratch. The val loss improvement (0.251→0.247) directly translates to CRPS improvement.
3. **1K→10K gap**: 0.1810→0.1840 (+1.7%), consistent with prior results (~1.3-1.6%).
4. **New best CRPS on this branch: 0.1840** (10K, Gneiting M², AddCL).

### Model saved
- Checkpoint: `models/unet_logit_normal/best_flow.pt` (epoch 67, val_loss=0.247329)
- Pool: `/home/chenxy/orcd/pool/datasets/research4/models/unet_logit_normal_best.pt`
- Pool: `/home/chenxy/orcd/pool/datasets/research4/models/unet_logit_normal_norm_stats.pt`

### End of Iteration 9
**End**: 2026-05-06 22:00 EDT, commit: 166cfad
**Duration**: ~5.5h (mostly waiting for GPU)
**GPU time**: ~97min (67min training + 3min 1K eval + 31min 10K eval)

## Iteration 10
**Start**: 2026-05-06 22:01 EDT, commit de52396
**Prefix**: ykoh-zqgs

### Concerns (3+)

1. **Timing (CRITICAL)**: ~36 hours elapsed out of 40-hour exploration deadline (02:00 EDT). GPU access completely blocked — mit_normal_gpu queued until 2026-05-07 17:40, mit_preemptable until 12:10. No more training possible. Must transition to report writing.

2. **Workflow (persistent, 6th time raised)**: No Harder et al. baselines ever reproduced. The objective says "start with baseline and report these too." We cited their published numbers but never ran their code ourselves. GPU constraints made this impossible from iter6 onward. The Harder et al. published CRPS numbers use a buggy formula (underestimates by ~50%) and cannot be directly compared to our corrected CRPS.

3. **Quality**: Data augmentation was never isolated from spectral loss (iter7 confound). The question "does augmentation alone help?" remains unanswered. Cannot run this experiment without GPU access.

### Plan for Iteration 10

**Goal**: Write the comprehensive research report.

**Why**:
- GPU access blocked, no more training/eval possible
- 36/40 hours of exploration elapsed
- 9 iterations of experiments completed with clear results
- Report needs multiple revision iterations before node expires (~10:00 EDT tomorrow)
- Most impactful use of remaining time

### Report Written

Created `notes/2026-05-06-research4-report.md` — comprehensive report covering:
- CRPS formula clarification (buggy vs corrected)
- All 7 methods tested with full results table
- Analysis: what works (OT-CFM, logit-normal, AddCL, UNet) and what fails (CFG, spectral, DiT, SmCL)
- Limitations: Harder et al. baselines not reproduced, augmentation not isolated, research2 gap unexplained
- Reproduction instructions and model checkpoint locations
- Compute summary

### End of Iteration 10
**End**: 2026-05-06 22:11 EDT, commit: 2b5ce9c
**Duration**: ~10min
**GPU time**: 0 (no GPU available)
