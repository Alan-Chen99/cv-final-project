# Decision Journal

## DEC-001: DiT backbone for flow matching

- **Decision:** Use a Diffusion Transformer (DiT) backbone instead of UNet for OT-CFM flow matching
- **Chosen Option:** Pure DiT (patch_size=8, hidden=256, depth=12, heads=4, ~14.6M params)
- **Confidence:** 55
- **Alternatives Considered:**
  - U-ViT (transformer + UNet skip connections) — more complex, less novel
  - Hybrid (DiT bottleneck in UNet) — less novel, moderate improvement expected
  - Multi-scale attention UNet — incremental improvement, not a new direction
  - SwinIR backbone — good for SR but more complex to adapt for flow matching
- **Reasoning:**
  - DiT is the modern standard for image generation (Stable Diffusion 3, DiT paper)
  - Completely untested for climate downscaling — high novelty
  - Global self-attention at every layer (vs only bottleneck in UNet) captures long-range spatial correlations
  - Patch_size=8 gives 256 tokens — manageable for 128x128 images
  - Risk: no skip connections means model must learn to reconstruct spatial details purely from patches
  - Working on residual (HR - bilinear(LR)) mitigates this — model only needs fine detail
- **When to re-evaluate:** After seeing training convergence and CRPS results
- **Independent evaluation:** not-started
- **Framing Biases:** Attracted to novelty over incremental improvement. DiT's success in natural images may not transfer to climate data.
- **Timestamp:** 2026-05-05T20:12:00Z

## DEC-002: U-ViT (DiT + skip connections) for flow matching

- **Decision:** Add long skip connections and conv refinement to DiT backbone
- **Chosen Option:** U-ViT (patch_size=8, hidden=256, depth=12, heads=4, ~16.5M params) with concat+linear skip projections and 3x3 conv refinement after unpatchify
- **Confidence:** 65
- **Alternatives Considered:**
  - Smaller patch size (patch_size=4, 1024 tokens) — much slower attention, untested tradeoff
  - Hybrid conv-transformer — different architecture, less direct test of skip connection hypothesis
  - Return to UNet with improvements — abandons transformer exploration
- **Reasoning:**
  - DiT's main diagnosed weakness was lack of skip connections (DEC-001 results)
  - U-ViT paper showed skip connections are critical for ViT-based diffusion models
  - Direct ablation: only change is adding skip connections, everything else identical to DiT
  - Additional conv refinement layer avoids grid artifacts from patch stitching
- **Result:** CRPS 0.194 (vs DiT 0.195) — marginal improvement. Skip connections alone don't close the gap with UNet v2 (0.171). The bottleneck is the patch tokenization itself, not information flow between layers.
- **When to re-evaluate:** After trying smaller patch sizes or hybrid approaches
- **Independent evaluation:** not-started
- **Framing Biases:** Confirmation bias toward transformer architectures. The gap with UNet (13.5%) suggests convolutions have stronger inductive bias for this spatial SR task.
- **Timestamp:** 2026-05-06T01:06:00Z

## DEC-003: Logit-normal timestep sampling + EMA for UNet v2

- **Decision:** Test logit-normal timestep sampling (from SD3) and EMA on UNet v2 architecture
- **Chosen Option:** UNet v2 (13M params) with logit-normal t-sampling (mean=0, std=1) + EMA (decay=0.9999), 26 epochs
- **Confidence:** 50
- **Alternatives Considered:**
  - Uniform t + EMA only — less novel, incremental
  - CRPS-aware loss — more uncertain, harder to implement
  - Smaller DiT patch size — still in transformer space which has plateaued
  - Data augmentation — safe but incremental
- **Reasoning:**
  - Logit-normal concentrates training on harder intermediate timesteps — SD3 found this crucial
  - EMA is standard practice in diffusion/flow matching (smooths training noise)
  - Both are easy to implement (5-10 lines combined) and novel for this task
  - Risk: SD3 finding may not transfer to this scale/task
- **Result:** CRPS 0.179 (regular) / 0.228 (EMA) vs baseline 0.171 (39ep uniform).
  - Logit-normal does NOT help — val loss 5.5% higher than uniform
  - EMA (0.9999) HURTS with short training — averages over poorly-trained early weights
  - UNet at 128x128 is ~9x slower than DiT per epoch, limiting to 26 epochs in 2hr budget
- **When to re-evaluate:** If longer training (100+ epochs) with lower EMA decay is tested
- **Independent evaluation:** not-started
- **Framing Biases:** Assumed SD3 findings transfer to flow matching on climate data. They don't at this scale. Also underestimated UNet's per-epoch cost vs DiT.
- **Timestamp:** 2026-05-06T04:51:00Z
