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
