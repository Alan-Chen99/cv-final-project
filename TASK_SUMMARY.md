# Task Summary: NorESM Dataset Integration

**Objective:** Add NorESM TAS 2x SR dataset, train all applicable models, update visualizations, verify output.

**Branch:** `noresm-dataset` (12 iterations, commits 90eeb59..ed3a158)
**Duration:** ~2026-05-09 21:26 to 2026-05-10 02:01 UTC (~4.5 hours)

---

## Iteration Table

| # | Focus | Key Output | Commit |
|---|-------|-----------|--------|
| 1 | Data loader + planning | `src/downscaling/data/noresm.py`, 7 NorESM tests | 90eeb59 |
| 2 | Training CLI (flow) | `scripts/train_flow.py`, flow-wide96-amp trained | 982cbf1 |
| 3 | Training CLI (Harder) | `scripts/train_harder.py`, 3 Harder models trained | 152957a |
| 4 | Training CLI (SwinIR) | `scripts/train_swinir.py`, swinir-finetuned trained | 0853673 |
| 5 | Evaluation | `scripts/run_eval_noresm.py`, eval results JSON | fa63326 |
| 6 | Dual-dataset metrics figures | Extended `metrics.py`, 9 metric figures | 04e8172 |
| 7 | Sample prediction figures | Full 35 figures generated on GPU, addcl bug fix | a681142 |
| 8 | Code quality fixes | Silent normalization bug, type widening, dataset validation | 6396801 |
| 9 | Integration tests + CLAUDE.md | All 49 non-GPU tests pass, pool docs updated | bd88a28 |
| 10 | Polish | Default constraint fix, docstring generalization | 78d4473→404b4a8 |
| 11 | Latent bug fixes | apply_addcl factor bug, public API rename, zero-std guard | ea9e8fa |
| 12 | Final docstrings | evaluate.py + swinir.py shape generalization | ed3a158 |

---

## Claims Verification Table

| Claim | Stated In | Verification Method | Status |
|-------|-----------|-------------------|--------|
| NorESM data loader works (32x32→64x64) | Iter 1 | Code review: `noresm.py` uses `F.interpolate` to (hr_h, hr_w) | **Confirmed** |
| Flow model CRPS=0.649 (best) | Iter 5 | `noresm_eval_results_500.json` confirms 0.649 | **Confirmed** |
| SwinIR-FT CRPS=0.988 (2nd) | Iter 5 | JSON confirms 0.988 | **Confirmed** |
| Constraints hurt NorESM metrics | Iter 5 | JSON: swinir-ft=0.988 vs swinir-ft+addcl=1.455 | **Confirmed** |
| All 49 tests pass | Iter 9 | Test file counts: 15+17+8+10+25+29=104 total functions; "49" likely refers to non-GPU subset. **Cannot reproduce** (GPU required) | **Unverifiable** (needs GPU) |
| 35 figures generated | Iter 7 | `ls figures/` shows 16+16+3=35 PNG files | **Confirmed** |
| Lint/typecheck pass | Iter 12 | `ruff check`: All passed. `basedpyright`: 0 errors, 2 pre-existing warnings | **Confirmed** |
| Trained models saved to pool | Iter 2-4 | `ls pool/noresm-dataset/models/`: flow-wide96-amp, harder (3 ckpts), swinir_ft | **Confirmed** |
| Default constraint changed to "none" | Iter 10 | `run_eval_noresm.py:107`: `default="none"` | **Confirmed** |
| compute_minmax_stats made public | Iter 11 | `harder.py:36`: function name has no underscore prefix | **Confirmed** |
| Zero-std guard added | Iter 11 | Code review needed | **Not spot-checked** |
| Figures visually correct | Iter 7,9 | Agent claims reviewed; `dual_crps.png` is valid 2384x773 PNG | **Partially confirmed** (file exists, dimensions reasonable) |

---

## Problems and Concerns

### 1. Test count discrepancy
Scratchpad claims "all 49 tests pass" but the test files contain 104 test functions. The 49 likely refers to a filtered subset (non-GPU tests via markers), but this is never explicitly stated. The exact pytest invocation that produced 49 is not recorded.

### 2. No decisions.md created
Guardrails require non-obvious decisions to be written to `decisions.md`. Several consequential decisions were made (e.g., "constraints degrade NorESM so default to none", "use same architecture capacity for 2x SR as 4x", "skip Harder GAN without SmCL") with no formal documentation or independent verification.

### 3. No literature search performed
The task was dataset integration (not direction-setting), so this is acceptable. However, the decision to skip certain model variants (e.g., not adapting architecture for 2x SR) would benefit from checking whether 2x climate SR has specific best practices.

### 4. Visual verification is agent-reported only
The agent claims visual review in iterations 7 and 9, but there is no mechanism to verify these claims independently. The figure files exist and have reasonable dimensions, but no human reviewed them.

### 5. Systematic failure class: NorESM constraint behavior
The finding that "constraints universally hurt NorESM" is significant and well-documented, but the root cause (LR/HR from different simulations) was identified from the data documentation rather than empirically verified. The mass violation of ~1.8K between avgpool(HR) and LR is stated but not directly measured in any iteration.

### 6. Training hyperparameters not documented
Models were trained using CLI scripts but the exact hyperparameters (epochs, LR, batch size) used for each NorESM model are not recorded in the scratchpad or any log file. Reproducibility relies on script defaults being unchanged.

### 7. Pool write rule potentially violated
CLAUDE.md states agents may only write to `pool/datasets/<own-branch>/`. Models were saved to `pool/datasets/noresm-dataset/models/` which is a shared top-level path, not `pool/datasets/noresm-dataset/`. This appears intentional (shared dataset) but technically violates the stated rule.

---

## Overall Assessment

The task is **functionally complete**: NorESM dataset integrated, models trained, figures generated, code passes lint/typecheck. The implementation quality is good — proper data loading, generalized evaluation, dual-dataset visualization.

Main risk: the training runs are not reproducible without knowing exact hyperparameters, and visual verification relies entirely on agent self-report.
