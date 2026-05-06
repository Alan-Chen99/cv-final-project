# Decisions Journal

## DEC-001: Use SwinIR as pretrained SR backbone
- **Decision:** Use SwinIR (11.9M params, pretrained on DF2K for 4x classical SR) as the starting point for pretrained model evaluation
- **Chosen Option:** SwinIR-M x4 classical SR
- **Confidence:** 75
- **Alternatives Considered:**
  - HAT (Hybrid Attention Transformer) — potentially better but larger, needs more setup
  - Real-ESRGAN — designed for real-world degradation, less suitable for clean downscaling
  - EDSR/RCAN — older architectures, likely worse
- **Reasoning:** SwinIR is well-established, available via spandrel, 4x scale matches our task, and has reasonable model size (11.9M, comparable to flow matching model at 13M). The Swin Transformer attention should help with spatial patterns in climate data.
- **When to re-evaluate:** If SwinIR saturates quickly, try HAT. If model size is a concern, try SwinIR-S.
- **Independent evaluation:** not-started
- **Framing Biases:** Biased toward models with easy-to-load pretrained weights (spandrel compatibility). HAT might be strictly better.
- **Timestamp:** 2026-05-06T05:00:00Z

## DEC-002: Use global normalization to [0,1] for SwinIR
- **Decision:** Normalize data to [0,1] using global min/max from train+val set rather than per-sample normalization
- **Chosen Option:** Global min/max normalization
- **Confidence:** 70
- **Alternatives Considered:**
  - Per-sample normalization (used in zero-shot) — loses relative magnitude info between samples
  - Standardization (zero-mean, unit-variance) — SwinIR expects [0,1] range, would need architecture change
  - Log-transform + normalization — could help with skewed distribution
- **Reasoning:** SwinIR expects img_range=1.0 input. Global normalization preserves relative magnitudes across samples, important for climate variables. The range [0.04, 131] maps cleanly to [0, 1].
- **When to re-evaluate:** If extreme values (high TCW) are poorly predicted, consider per-sample or log normalization.
- **Independent evaluation:** not-started
- **Framing Biases:** Assumes SwinIR's internal handling of [0,1] inputs is appropriate for climate data.
- **Timestamp:** 2026-05-06T05:30:00Z

## DEC-003: Residual vs Direct Head Parameterization for Multi-Head Ensemble
- **Decision:** Tested residual parameterization (heads predict corrections to frozen det. mean) vs direct prediction. Both achieve identical CRPS=0.183.
- **Chosen Option:** Either — parameterization doesn't matter
- **Confidence:** 90 (empirically validated, clear negative result)
- **Alternatives Considered:**
  - Direct heads (iter 2): each head predicts full HR output from shared features
  - Residual heads (iter 3): each head predicts residual, output = det_mean + residual
  - Residual + regularization: not tested, but unlikely to help given identical equilibrium
- **Reasoning:** The CRPS loss is the dominant optimization force. Both parameterizations converge to the same t1≈0.0026, t2≈0.0024 balance in normalized space. The frozen backbone constrains what diversity is achievable — changing the head parameterization cannot overcome feature-level limitations.
- **When to re-evaluate:** If backbone is unfrozen (features change), residual mode may then matter
- **Independent evaluation:** not needed — clear empirical result
- **Framing Biases:** Only tested with frozen backbone. With unfrozen backbone, residual mode could behave differently since the gradient paths differ.
- **Timestamp:** 2026-05-06T12:58:00Z
