# Scratchpad — Constrained Downscaling CRPS Optimization

## Iteration 1
**Start:** 2026-05-03 00:28 EDT
**Start commit:** edc0a69

### Objective
32×32 → 128×128 spatial downscaling. CRPS as eval metric. <2hr training budget per run.
Task uses constrained-downscaling codebase (Harder et al. 2208.05424).

### Plan for This Iteration
1. Download ERA5 TCW data (the ML-ready dataset from Google Drive)
2. Allocate GPU node
3. Run CNN baselines (none, softmax constraints) — quick sanity check
4. Run GAN baselines (none, softmax constraints) — ensemble for proper CRPS
5. Compute and record CRPS for all baselines
6. Commit results

### Key Understanding
- CRPS requires ensemble forecasts. Deterministic CNN → CRPS = MAE.
- GAN produces 10 ensemble members per sample.
- Constraint layers: AddCL, MultCL, SmCL (softmax — best default per paper).
- Data: ERA5 total column water, 4x upsampling (32×32 → 128×128).
- Upsampling factor = 4, dataset = era5_sr_data.

### Existing Jobs (DO NOT TOUCH)
- node1625: cpu-hold (mit_normal)
- node1620: cpu-pree (mit_preemptable)
- node3302: sweep-gp (mit_preemptable)
- node3600: sweep-gp (mit_preemptable)
- node1602: cpu-pree — THIS IS WHERE I RUN

### Concerns (First Iteration — Focus on Setup)
1. **Data availability**: RESOLVED — downloaded from Google Drive, verified shapes
2. **CRPS evaluation**: RESOLVED — for deterministic models CRPS = MAE; wrote scripts/compute_metrics.py
3. **Training time**: RESOLVED — ~24s/epoch on L40S, 200 epochs ~80 min per model. Loss plateaus by epoch 15.

### Progress
- [x] Download ERA5 TCW4 data (32x32→128x128, 40K train, 10K val/test)
- [x] Bilinear baseline: CRPS = 0.506 on test
- [x] CNN none (200 epochs): CRPS = MAE = 0.310, RMSE = 0.621 (paper: MAE=0.326, RMSE=0.657)
- [ ] CNN softmax: training in progress (started ~02:36 EDT)
- [ ] GAN baselines: deferred to next iteration

### Key Results So Far
| Model | Constraint | CRPS | RMSE | Notes |
|-------|-----------|------|------|-------|
| Bilinear (no train) | none | 0.506 | 0.949 | Trivial baseline |
| CNN (ours) | none | 0.310 | 0.621 | Slightly better than paper |
| CNN (paper) | none | 0.326 | 0.657 | Reference |
| GAN (paper) | ScAddCL | **0.151** | 0.604 | Best paper CRPS |

### Preemption Events
- First GPU job (node3615, job 13092600) preempted at ~02:31 EDT
- CNN none training completed but eval was interrupted
- Successfully recovered: prediction file was saved, computed metrics offline
- Re-allocated GPU (node4304, job 13098698) and restarted CNN softmax

### Next Iteration Plan
1. Run GAN baselines (none, SmCL, ScAddCL)
2. Train simple conditional diffusion model (scripts/simple_diffusion.py ready)
3. Evaluate all with CRPS on test set
4. Key question: can diffusion beat GAN CRPS of 0.151?

### CNN Softmax Results (61 epochs, preempted)
- CRPS = MAE = 0.2977, RMSE = 0.5978, mass_viol = 0.000001, neg_pixels = 0
- Very close to paper (MAE=0.291, RMSE=0.582)

### Created Artifacts
- scripts/eval_crps.py — CRPS evaluation for saved predictions
- scripts/eval_bilinear_crps.py — bilinear baseline
- scripts/compute_metrics.py — simplified metric computation
- scripts/simple_diffusion.py — conditional DDPM for TCW4 (ready to train)
- scripts/run_cnn_baselines.sh — baseline training script
- reports/iteration-001-baselines.md — final report

### End of Iteration 1
**End time:** 2026-05-03 03:15 EDT
**End commit:** TBD (committing now)
**Duration:** ~2h 47m
**GPU preemptions:** 2 (node3615 at 02:31, node4304 at 03:10)
