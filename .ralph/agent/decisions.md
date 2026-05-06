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
