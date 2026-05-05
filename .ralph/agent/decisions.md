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
