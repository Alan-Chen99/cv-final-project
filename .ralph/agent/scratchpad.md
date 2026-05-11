# Metrics V2 Scratchpad

## Iteration 1
- **Start**: 2026-05-11T19:31:45Z, commit 3921716
- **Prefix**: gamma-delta

### Current State Assessment
**Existing metrics**: CRPS (energy), MAE, RMSE, mass violation — all pixelwise/probabilistic.
**Existing figures**: Bar charts (CRPS, MAE, RMSE, mass_violation), sample comparisons, error maps, ensemble spread.
**Existing eval results**: ERA5 (15 methods, 500 samples), NorESM (12 methods, 500 samples).

### Concerns (3+)

1. **Quality: No spectral metrics** — The entire evaluation is pixelwise. No spectral power density, no RALSD. Climate downscaling papers universally use spectral analysis (CorrDiff, GenDiff, intercomparison paper, CDSI, all use PSD). This is a critical gap — flow models might produce correct pixel values but wrong spatial structure (or vice versa). The intercomparison paper (2025-12-16) explicitly states RALSD "weights errors at small scales equally to large scales, ensuring fine-scale features are adequately evaluated."

2. **Quality: No structural similarity metric** — No SSIM or perceptual quality metric. While SSIM isn't standard in climate papers, it's universal in image SR papers (SwinIR, SR3, HAT all report it). Since our baselines include SwinIR, we should include it for completeness.

3. **Quality: Missing distribution metrics** — No LHD (Logarithmic Histogram Distance), no Q-Q analysis. The intercomparison paper uses LHD to assess intensity distribution fidelity. Spread-skill ratio missing too (CDSI uses it).

### Plan for this iteration
Implement spectral metrics (PSD computation + RALSD + spectral power plots) in `src/downscaling/metrics/spectral.py`. This is the highest-value missing metric and addresses concern #1.

### Work done
- Implemented `src/downscaling/metrics/spectral.py`: radial_psd, radial_psd_batch, ralsd, spectral_bias
- Implemented `src/downscaling/metrics/structural.py`: ssim, psnr
- Updated `src/downscaling/metrics/__init__.py` to export new metrics
- Created `tests/test_spectral.py` with 16 integration tests (all pass)
- All existing tests still pass (10/10)
- Lint: pass, Format: pass, Typecheck: 0 errors

### Next iteration work
- Update `src/downscaling/evaluation/evaluate.py` to compute RALSD, SSIM, PSNR alongside CRPS/MAE/RMSE
- Add spectral power density plots to `src/downscaling/plotting/`
- Re-run full evaluation on GPU to generate new results with spectral metrics
- Start report file

### RALSD Definition (from intercomparison paper, 2025-12-16)
1. Compute 2D FFT of each field
2. Radially integrate (bin by wavenumber) → 1D power spectrum
3. RALSD(dB) = sqrt(1/N * sum_i (10*log10(F_true_i / F_pred_i))^2)
- Lower is better
- Key: weights errors at all scales equally in log space
