# Decision Journal

## DEC-001

- **Decision:** Architecture for first experiment on research4
- **Chosen Option:** DiT (Diffusion Transformer) flow matching
- **Confidence:** 65
- **Alternatives Considered:**
  - Re-train UNet flow matching v2 (same as research2) — would give known-good result but no new information
  - WassDiff-style Wasserstein regularization — requires NCSN++ architecture, bigger change
  - CorrDiff two-stage (deterministic mean + stochastic residual) — needs 2 models in 2hr budget
  - Consistency distillation — needs pre-trained teacher, which we don't have
- **Reasoning:** DiT is explicitly noted as unexplored in cross-comparison open questions. Transformer backbone has proven effective for climate data (Prithvi, ClimaX, 1EMD). Same training pipeline (OT-CFM, residual prediction) isolates the architecture change. ~14.6M params matches UNet baseline for fair comparison.
- **When to re-evaluate:** After seeing first training loss curves — if DiT fails to converge within 10 epochs, consider architecture adjustments or fallback to UNet
- **Independent evaluation:** not-started
- **Framing Biases:** Novelty bias — choosing DiT partly because it's "new" rather than because evidence strongly suggests it will be better. The DiT paper showed gains on image generation but climate data is structurally different (smooth fields, not natural images).
- **Timestamp:** 2026-05-05T20:18:00Z
