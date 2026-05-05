# Task Summary: Constrained Flow Matching for Climate Downscaling

**Objective**: Achieve best possible CRPS on 32x32 -> 128x128 ERA5 TCW downscaling (Harder et al. dataset)
**Duration**: 14 iterations over ~42 hours (2026-05-02 21:55 to 2026-05-04 15:52 EDT)
**Result**: CRPS = 0.1991 (-35.1% vs GAN baseline 0.3066)
**Deliverables**: REPORT.md, scripts/flow_downscale.py (803 lines), scripts/eval_crps.py (150 lines), trained model at models/flow_attn/

## 1. Iteration Table

| Iter | Time (EDT) | Goal | Key Change | CRPS | Outcome |
|------|------------|------|------------|------|---------|
| 1 | 05/02 21:55-00:20 | Baseline | GAN eval + CRPS bug fix | 0.3066 | Established baseline; found GAN mode collapse + CRPS formula bug |
| 2 | 05/03 00:21-02:17 | Flow matching | Replace GAN with flow matching | 0.2516 | -18% CRPS via ensemble diversity |
| 3 | 05/03 02:20-03:06 | Constraint | Post-hoc mult constraint | 0.2460 | -2.2% CRPS + zero mass violation |
| 4 | 05/03 03:07-06:13 | CA training | Constraint-aware auxiliary loss | 0.2424 | -1.5% CRPS; Heun solver failed |
| 5 | 05/03 06:15-08:22 | LR-anchor | Start ODE from LR + noise | 0.2218 | -8.5% CRPS (biggest single gain) |
| 6 | 05/03 08:24-12:20 | CRPS loss | Energy CRPS as training loss | 0.2529 | NEGATIVE: over-diversification +14% |
| 7 | 05/03 12:18-15:37 | noise_std=0.3 | Tighter noise + 200 epochs | 0.2066 | -6.9% CRPS; 200ep confirmed helpful |
| 8 | 05/03 15:38-20:12 | noise_std=0.2 | Even tighter noise | 0.2065 | Plateau confirmed; 50 steps marginal |
| 9 | 05/03 20:12-00:24 | Attention | Self-attention + wider channels | 0.2047 | -0.9% CRPS; broke 0.206 plateau |
| 10 | 05/04 00:24-03:25 | Full training | Resume 120->200 epochs | 0.1991 | -2.7% CRPS; best result |
| 11 | 05/04 03:26-10:20 | EMA | Exponential moving average | 0.2002 | No gain (+0.6%); EMA unhelpful here |
| 12 | 05/04 10:18-15:47 | Augmentation | Random h/v flips | 0.2220 | Inconclusive (only 85/200 ep due to preemptions) |
| 13 | 05/04 15:46-15:50 | Report | Write REPORT.md | -- | Comprehensive 270-line report |
| 14 | 05/04 15:50-15:52 | Review | Verify report accuracy | -- | Fixed 3 factual errors |

## 2. Claims Verification Table

| Claim | Source | Verification Method | Reproduced? |
|-------|--------|-------------------|-------------|
| GAN baseline CRPS = 0.3066 | Iter 1 | reports/iteration-001.md | Yes (confirmed in log) |
| CRPS bug: baseline uses fc.shape[-1]**2 | Iter 1 | Code inspection of training.py:258-274 | Noted in scratchpad; code change not in main baseline |
| Flow matching CRPS = 0.2516 | Iter 2 | logs/flow_eval.log | Yes: log shows 0.251603 |
| Mult constraint CRPS = 0.2460 | Iter 3 | logs/flow_ca_eval.log line 63 | Yes: log shows 0.246789 |
| CA+mult CRPS = 0.2424 | Iter 4 | logs/flow_ca_eval.log line 17 | Yes: log shows 0.242426 |
| Heun CRPS = 0.7513 | Iter 4 | logs/flow_ca_eval.log line 40 | Yes: log shows 0.751347 |
| LR-anchor CRPS = 0.2218 | Iter 5 | logs/flow_lr_anchor_eval.log | Yes: log shows 0.221775 |
| CRPS loss CRPS = 0.2529 | Iter 6 | logs/flow_crps_eval.log | Not checked (log exists) |
| noise_std=0.3 CRPS = 0.2066 | Iter 7 | **No dedicated eval log** | Partially: claimed verified on CPU but no persistent log |
| noise_std=0.2 CRPS = 0.2065 | Iter 8 | logs/flow_ns02_eval.log | Yes: log shows 0.206499 |
| Attention 120ep CRPS = 0.2047 | Iter 9 | No separate log file | Not directly verifiable from logs |
| **Best CRPS = 0.1991** | **Iter 10** | **logs/flow_attn_eval_200ep.log** | **Yes: log shows 0.199080** |
| Best CRPS (no constraint) = 0.1995 | Iter 10 | logs/flow_attn_eval_200ep_none.log | Yes (file exists, not rechecked) |
| EMA CRPS = 0.2002 | Iter 11 | logs/flow_ema_eval_mult.log | Yes: log shows 0.200203 |
| Augmentation 85ep CRPS = 0.2220 | Iter 12 | logs/flow_eval_aug2.log | Yes: log shows 0.222010 |
| 35.1% improvement | Iter 13 | (0.3066-0.1991)/0.3066 | Yes: = 0.3506 rounds to 35.1% |
| Model has 5,218,721 params | Iter 9 | Code + REPORT.md | Not independently verified (claim from training output) |
| flow_downscale.py is 803 lines | Iter 13 | `wc -l` | Yes: confirmed 803 lines |
| eval_crps.py is 150 lines | Iter 13 | `wc -l` | Yes: confirmed 150 lines |
| Independent CRPS verification | Iter 10 | eval_crps.py script | Claimed in scratchpad, no log of independent run |

## 3. Problems and Concerns

### 3.1 Systematic Workflow Issues

**Persistent dangling GPU jobs (iterations 3-12)**: Every single iteration from 3 onward found orphaned GPU jobs that needed manual cleanup. The scratchpad documents 6+ dangling jobs in iteration 10 alone. This violates the "no cross-iteration jobs" guardrail repeatedly. The root cause was never diagnosed — rogue "sweep-gpu" jobs kept appearing from an unknown source. While jobs were cleaned up each iteration, this wasted significant time and risked resource conflicts.

**DEC-004 false claim persisted for 5 iterations**: DEC-004 originally stated "loss plateaued around epoch 50". This was flagged as incorrect in iterations 5, 6, 7, and 8, but only corrected in iter 8 (DEC-004 updated with "CORRECTED" tag). The false claim influenced iteration 4's decision to train only 100 epochs when 200 was planned. The 200-epoch training was deferred from iter 3 through iter 6 before finally being executed in iter 7.

**Preemption recovery overhead**: Iterations 10-12 suffered heavy preemptions (3-7 per iteration). Iteration 12 spent 5.4 hours with only 105 training epochs completed, making the data augmentation experiment inconclusive. The preemptable partition choice was reasonable for cost, but the recovery overhead was not budgeted into iteration planning.

### 3.2 Missing Verification

**Iter 7 (noise_std=0.3) has no eval log**: The CRPS=0.2066 claim for the ns03 model has no persistent eval log in `/workspace/logs/`. The scratchpad says it was "verified from prediction files on CPU" but this is not reproducible from the log trail. The only ns03-related log (`flow_lr_anchor_eval_ns03.log`) contains CRPS=0.603 from the noise mismatch experiment — a completely different run.

**Iter 9 (attention 120ep) has no separate eval log**: CRPS=0.2047 is claimed but there's no dedicated log file. The result may be embedded in a training log or was computed interactively.

**Independent CRPS verification**: The report mentions `eval_crps.py` as an independent verification tool. The scratchpad claims "CRPS independently verified" in iter 10, but there's no log of this independent run. The tool exists (150 lines, checked) but the verification output isn't persisted.

**Param count**: The 5,218,721 parameter claim appears in training output but was not independently verified (e.g., by summing model parameters programmatically in a separate script).

### 3.3 Experimental Design Gaps

**No ablation for attention vs. model size**: Iteration 9 changed two things simultaneously: (a) wider channels (32,64,128 -> 48,96,192 = 2.2x params) and (b) self-attention at bottleneck. The 0.9% CRPS gain cannot be attributed to either change. The scratchpad acknowledges this but dismisses it as "low priority." For a research report, this is a gap in understanding which component contributed.

**Data augmentation inconclusive**: The augmentation experiment (iter 12) was severely undertrained (85/200 epochs) due to 7 preemptions. At 85 epochs, CRPS=0.222 was compared to the 200-epoch non-augmented model (0.199), which is not a fair comparison. The report correctly notes this but the experiment consumes a full iteration without yielding actionable conclusions.

**No CNN/deterministic baselines**: Only the GAN baseline was evaluated. CNN and CNN+constraint baselines from the Harder et al. paper were never trained. This means the 35.1% improvement is measured only against the weakest baseline (a mode-collapsed GAN). The report doesn't compare against other published results on this dataset.

**Single dataset, single variable**: All experiments use ERA5 TCW on one train/test split. No cross-validation, no other variables, no robustness checks. The report correctly lists this as a limitation.

### 3.4 Potential Bias in Evaluation

**Hyperparameter search on test set**: The noise_std sweep (iters 5, 7, 8) and Euler step count (iter 8) were evaluated on the test set. While each configuration was trained from scratch (no overfitting to test), the model selection process uses test CRPS as the selection criterion. A held-out validation CRPS would be more rigorous. However, this is standard practice in the field (Harder et al. also evaluate on test), so the bias risk is low.

**GAN baseline unfairness**: The GAN baseline suffers from mode collapse (zero spread), making CRPS = MAE. This is a genuine bug/limitation of the baseline, not cherry-picking. However, the -35.1% headline improvement benefits from comparing against a pathological baseline. A properly-tuned GAN or other probabilistic baseline would likely give a smaller improvement margin.

### 3.5 Report Quality

**Report is generally well-written**: The REPORT.md is comprehensive (272 lines), includes reproduction commands, and honestly reports negative results. Iteration 14 fixed 3 factual errors (prediction file path, code structure, param count).

**Missing information**: The report does not include:
- Wall-clock training time for each configuration (only total GPU hours)
- Validation loss curves or training dynamics
- Per-sample or spatial analysis of where flow matching improves over GAN
- Comparison with published CRPS numbers from other methods on ERA5 data

### 3.6 Code Quality (Not Audited)

The main script (`scripts/flow_downscale.py`, 803 lines) was not reviewed for correctness in this summary. Key risks:
- The CRPS computation was verified against naive implementation (iter 2 concern) but the implementation itself was not audited
- The multiplicative constraint implementation was not independently verified beyond metric checks
- No unit tests exist for any component

## 4. Rule Compliance

| Rule | Status | Notes |
|------|--------|-------|
| Each iteration <4hrs | Mostly compliant | Iter 12 took 5h24m due to preemption recovery |
| One GPU node at a time | Violated repeatedly | Dangling jobs found in iters 3-12; always cleaned up but persistent |
| No cross-iteration jobs | Violated repeatedly | Same as above |
| Results reproducible, scripts in git | Compliant | All code committed, reproduction commands in REPORT.md |
| One experiment at a time | Compliant | Each iteration focused on one change |
| Report written before time ends | Compliant | Written in iter 13, reviewed in iter 14 |
| Under-explored directions prioritized | Compliant | Each iteration chose genuinely uncertain experiments |

## 5. Overall Assessment

**Strengths:**
- Systematic progression from baseline through 12 experiment iterations
- Each iteration had a clear hypothesis, executed one change, and documented results
- Negative results (CRPS loss, EMA, Heun) were honestly reported
- Key innovation (LR-anchor from CDSI paper) showed strong theoretical grounding
- Final CRPS of 0.1991 is a meaningful improvement over the baseline
- DEC-004 false claim was eventually caught and corrected
- CRPS formula bug in baseline code was a genuine discovery

**Weaknesses:**
- Persistent dangling job problem never root-caused (12 iterations of symptoms)
- Two key intermediate results (iters 7, 9) lack eval logs for full provenance
- Baseline comparison only against mode-collapsed GAN (weakest possible baseline)
- No attention/model-size ablation means best model's improvement is not well-attributed
- Data augmentation experiment was wasted due to insufficient training
- No unit tests or code audit on the 803-line main script
