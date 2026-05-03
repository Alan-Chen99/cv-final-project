# Decisions Journal

## DEC-001
- **Decision:** Start with CNN baselines (no constraints, SmCL) at 200 epochs before running GANs
- **Chosen Option:** CNN first, GAN in next iteration
- **Confidence:** 85
- **Alternatives Considered:** Run all 4 models in parallel (not possible with 1 GPU), run GAN first (GAN needs CNN comparison for context)
- **Reasoning:** CNN is faster, establishes deterministic CRPS baseline. GAN ensemble CRPS is the more important metric but takes longer. CNN first validates the pipeline.
- **Reversibility:** High — can always run more models later
- **Timestamp:** 2026-05-03T04:30:00Z
