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
