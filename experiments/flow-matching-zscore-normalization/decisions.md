# Decision Journal — research6

## DEC-001: Use research2's flow_matching_v2.py recipe instead of fixing flow_downscale.py

- **Decision:** Train a new model using `src/exp-spatial-4x-crps-v1/flow_matching_v2.py` directly (z-score normalization, 13M params, 2 ResBlocks/level) rather than patching flow_downscale.py
- **Chosen Option:** Use existing research2 script
- **Confidence:** 85
- **Alternatives Considered:**
  1. Patch flow_downscale.py with z-score normalization (risky — many places to change, easy to introduce bugs)
  2. Port research2's AttentionUNet into flow_downscale.py (unnecessary complexity)
  3. Just evaluate research2's weights if accessible (weights not available on this branch)
- **Reasoning:** The research2 script already has the correct recipe (z-score normalization, 13M architecture, proper hyperparameters). Using it directly minimizes risk of introducing bugs. The main finding (normalization is the root cause) is best tested by reproducing research2's result, not by incrementally patching a broken script.
- **When to re-evaluate:** If training fails or CRPS doesn't approach 0.171, re-examine other differences.
- **Independent evaluation:** not-started
- **Framing Biases:** Assumes normalization is the primary factor. Could be wrong if the gap is due to subtle training dynamics (fp32 vs AMP, dropout, grad clipping).
- **Timestamp:** 2026-05-06T18:46:00Z

## DEC-002: Correcting the "OT coupling" narrative

- **Decision:** Prior iterations' claim that "minibatch OT coupling is the key missing ingredient" is factually wrong. Research2 uses NO OT coupling. The "OT-CFM" label refers to the OT probability path (straight interpolation), not minibatch coupling. All OT coupling work (iterations 3-4) was based on this false premise.
- **Chosen Option:** Abandon OT coupling pursuit, focus on normalization fix
- **Confidence:** 99
- **Alternatives Considered:** None — this is a factual correction based on reading the code
- **Reasoning:** grep for `ot_coupling|linear_sum_assignment|sinkhorn` in flow_matching_v2.py returns zero matches. The training loop uses `x_0 = torch.randn_like(res_batch)` — plain random noise, no coupling.
- **When to re-evaluate:** N/A — this is a verified fact
- **Independent evaluation:** not-started (but trivially verifiable by reading the code)
- **Framing Biases:** None identified
- **Timestamp:** 2026-05-06T18:46:00Z
