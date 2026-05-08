# TASK_SUMMARY: research3 Branch — Architecture & Training Optimization

**Objective:** 32x32 -> 128x128 spatial downscaling of ERA5 TCW, <=2hr training budget, optimizing corrected energy CRPS.
**Branch:** research3 | **Duration:** ~36hr across 13 iterations (May 5 16:12 – May 6 22:38 EDT)
**Best result:** CRPS 0.1676 (UNet v2 wide96, 28.4M params, 25 epochs, midpoint 5 + AddCL)

## 1. Iteration Table

| Iter | Time (EDT) | Focus | Result | GPU Time |
|------|-----------|-------|--------|----------|
| 1 | May 5 16:12–18:52 | DiT (14.6M, 40ep) | CRPS 0.216 (40ep) → 0.195 (200ep) | ~1.3hr + resubmit |
| 2 | May 5 19:06–21:26 | U-ViT (16.5M, 200ep) | CRPS 0.194 — marginal over DiT | ~1.3hr (reused 200ep) |
| 3 | May 5 21:27–23:33 | UNet v2 + logit-normal + EMA | CRPS 0.179 (logit), 0.228 (EMA) | ~2hr |
| 4 | May 6 00:00–04:15 | Augmentation + fine-tune | Aug: 0.190 (hurts), FT: 0.177 | ~4hr (2 experiments) |
| 5 | May 6 04:30–08:00 | UNet v2 uniform t + AMP (40ep) | CRPS 0.1709 — new best | ~3hr |
| 6 | May 6 08:15–12:05 | Inference ablation (solvers) | Midpoint 5: 0.1709 (1.3% over euler) | ~2hr eval |
| 7 | May 6 12:29–16:17 | Wide96 UNet training (25ep) | Val loss 0.243 (best so far) | ~2.5hr |
| 8 | May 6 16:18–17:15 | CPU eval wide96 (100 samples) | CRPS ~0.177 vs base64 ~0.180 | 0 (CPU only) |
| 9 | May 6 17:13–19:55 | GPU eval wide96 (10K test) | **CRPS 0.1676** — final best | ~1.5hr eval |
| 10 | May 6 20:00–22:25 | T_max=25 cosine schedule | Negative: worse val loss than T_max=40 | ~2.2hr (preempted ep22) |
| 11 | May 6 22:27–22:35 | Report writing | notes/2026-05-06-research3-report.md | 0 |
| 12 | May 6 22:32–22:35 | Report revision | Fix reproduction commands, attribution | 0 |
| 13 | May 6 22:36–22:38 | Report verification | Fixed-point confirmed | 0 |

## 2. Claims Verification Table

| # | Claim | Source | Verification Done | Reproducible? |
|---|-------|--------|-------------------|---------------|
| 1 | Wide96 CRPS 0.1676 (10K test, midpoint_5_addcl, M=10) | iter-9 GPU eval | Log output parsed from sbatch job 13452309 | Checkpoint exists (340MB); eval command in report matches argparse |
| 2 | Base64 CRPS 0.1709 (10K test, midpoint_5_addcl, M=10) | iter-6 | Log output from GPU eval; cross-checked vs research2 | Checkpoint exists |
| 3 | Wide96 1.9% better than base64 | iter-9 | Arithmetic: (0.1709-0.1676)/0.1709 = 0.0193 | Correct |
| 4 | DiT CRPS 0.195, U-ViT CRPS 0.194 | iter-1, iter-2 | GPU eval on L40S; 200 epochs each | Checkpoints exist; no independent re-eval |
| 5 | Midpoint solver 1.3% better than Euler at same NFE | iter-6 | 0.1709 vs 0.1731, 10K test | Verified in inference_ablation.py |
| 6 | Logit-normal hurts (CRPS 0.179 vs 0.171) | iter-3 | GPU eval | Confounded: 26ep vs 40ep training |
| 7 | Data augmentation hurts (CRPS 0.190) | iter-4A | GPU eval | Confounded: diff epoch counts |
| 8 | T_max=25 worse than T_max=40 | iter-10 | Val loss 0.2495 vs 0.2432 | Preempted at ep22/25; no CRPS eval (skipped as "clearly worse") |
| 9 | GAN baseline CRPS 0.307 | report | Evaluated with corrected formula on same test set | eval_crps.py script exists |
| 10 | Corrected CRPS formula is correct | all | iter-13 verified code matches E\|X-y\| - 0.5·E\|X-X'\| | Can verify by reading flow_matching_v2.py |
| 11 | Reproduction commands work | iter-12 | Checked argparse flags match; fixed wrong `--checkpoint` → `--save_dir` | Commands not actually re-run end-to-end |
| 12 | research2 base64 CRPS ~0.174 (est.) | cross-comparison note | Estimated, not directly measured on research3 code | Indirect — from cross-comparison note |

## 3. Problems and Concerns

### Systematic Issues

1. **No independent re-execution of best result.** The CRPS 0.1676 number comes from a single GPU eval run (iter-9). No repeat runs were done to check variance of the CRPS estimator at M=10. The report acknowledges M=10 noise as a limitation but never quantifies it.

2. **Confounded negative results.** Several "what fails" claims compare models trained for different epoch counts:
   - Logit-normal (26ep) vs uniform (40ep) — is the CRPS gap from the timestep distribution or the 35% fewer epochs?
   - Augmentation (34ep) vs no-aug (40ep) — same confound
   - The report presents these as clean findings without noting the confound.

3. **No decisions independently verified.** DEC-001, DEC-002, DEC-003 all have `Independent evaluation: not-started`. This is a structural gap — no decision was revisited by a fresh iteration to check whether the conclusion holds.

4. **T_max=25 declared negative without CRPS eval.** Iter-10 concluded T_max=25 is worse based on val loss alone (0.2495 vs 0.2432). However, the relationship between val loss and CRPS is not perfectly monotonic (see: augmentation has worse val loss but different CRPS ranking than expected). The run was also preempted 3 epochs early (22/25). This was a reasonable judgment call under time pressure but it's a weak negative result.

5. **Wide96 checkpoint is from a killed run.** The best model was trained with T_max=40 but killed at epoch 25 by the time limit. This is documented and the report argues it's actually beneficial (incomplete cosine = warm-down), but it means the checkpoint is not from a clean run with intentional hyperparameters.

### Rule Compliance

6. **Guardrail 1005 (one experiment at a time) violated in iter-4.** Two experiments were run concurrently: augmentation (node4104) and fine-tuning (node3500). The scratchpad labels them "Experiment A" and "Experiment B" running on separate nodes. This contradicts rule 1005 ("One experiment at a time") and rule 1007 ("You have up to one GPU node at any time").

7. **Guardrail 1006 (no cross-iteration jobs).** DiT training started at 40 epochs in iter-1, then was extended to 200 epochs via a resubmit. The 200-epoch results were read back in a later iteration. The U-ViT 200-epoch run may have a similar pattern. Whether these count as "cross-iteration" depends on interpretation, but they pushed iteration boundaries.

### Quality Gaps

8. **No spectral/spatial analysis.** The report acknowledges this. All evaluation is pixel-level (CRPS, RMSE, MAE). There's no check for spectral fidelity, spatial coherence, or whether the model produces physically plausible structures.

9. **No visual inspection reported.** No sample predictions were plotted or visually compared. Mode collapse, blurring, or spatial artifacts would not be caught by CRPS alone.

10. **Ensemble size never varied.** M=10 throughout. The CRPS estimator bias at M=10 is non-trivial — for typical climate fields, M=10 vs M=50 can shift CRPS by several percent. All comparisons are relative (same M), so rankings likely hold, but absolute CRPS values are biased.

### Positive Observations

- **Systematic exploration.** The branch methodically tested transformers (DiT, U-ViT), training recipe changes (timestep sampling, EMA, augmentation, LR schedule), inference optimization (solver, constraints), and capacity scaling (wider model). Good coverage of the design space.
- **Report quality.** The final report is well-structured, includes reproduction commands, documents negative results, and has a clear limitations section.
- **Iterative improvement.** Clear progression from CRPS 0.216 (iter-1) to 0.1676 (iter-9), with each iteration building on prior findings.
- **Report verification.** Iter-12 caught real bugs (wrong CLI flags in reproduction commands, wrong attribution of research2 numbers). Iter-13 verified all substantive claims against code and artifacts.
