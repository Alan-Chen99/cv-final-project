# Foundation Models vs Task-Specific Downscaling: Two Disconnected Lineages

Date: 2026-04-27

## Citation graph

Two groups of climate papers with minimal cross-citation.

**Lineage A -- Foundation models** (deterministic ViT-based regressors, MSE/MAE trained):
- ClimaX (2301.10343) -> Aurora (2405.13063) -> Prithvi WxC (2409.13598) -> WeatherGFM (2411.05420)
- Sequential citation: Prithvi cites ClimaX + Aurora. WeatherGFM cites ClimaX + Aurora.
- Primary task: weather forecasting (1-10 day). Downscaling is a secondary fine-tuning demo.
- None are generative. Aurora mentions ensembles only as future work.

**Lineage B -- Task-specific downscaling** (UNet/CNN-based, mostly generative):
- Harder (2208.05424) -> DFNO/STVD -> CorrDiff -> GenDiff/WassDiff/CDSI/Intercomparison
- Dense cross-citation within this group.
- All use UNet backbones. All (except Harder) produce stochastic ensembles.
- Care about: physical constraints, extreme events, bias correction, uncertainty quantification.

**Bridges**: Only two weak connections:
- GenDiff cites ClimaX
- 1EMD (2506.22447) cites both Harder and CorrDiff -- the only paper bridging both lineages architecturally (ViT for downscaling) but has no constraints and is deterministic

## Why the disconnect

Not a fundamental incompatibility -- a community gap. Foundation model papers come from NWP/forecasting community; downscaling papers from climate impacts/statistical downscaling community. Different metrics, different venues.

Key structural differences:
- **Distribution shift**: Foundation models dodge it. Prithvi WxC only demonstrated perfect-model downscaling (coarsen MERRA-2 then recover). ClimaX does GCM->ERA5 but that's the exception.
- **Input schema**: Foundation models pretrained on specific variable sets (160 MERRA-2 vars for Prithvi); real GCM outputs have different variables, grids, biases.
- **Deterministic vs generative**: All foundation models produce single MSE-optimized outputs. Downscaling needs stochastic ensembles for uncertainty quantification.
- **Architecture**: Foundation models use ViT; downscaling uses UNet. DiT (diffusion transformer) hasn't reached climate downscaling.

## Research directions from bridging the gap

**A. Pretrained ViT backbone + diffusion head + constraints.** Use frozen/LoRA'd Prithvi WxC as encoder in CorrDiff-style two-stage setup. SmCL on deterministic mean, diffusion for stochastic residuals. Combines foundation model representations, generative ensembles, physical constraints.

**B. DiT for climate downscaling.** Replace UNet backbone in diffusion models with transformer. Natural fit if also using foundation model pretraining.

**C. Foundation model on real GCM->RCM shift.** Test Prithvi WxC on actual GCM input with RCM targets (not perfect-model). Measure distribution shift degradation. If severe, motivates bias-informed fine-tuning.

**D. Constrained stochastic interpolants.** CDSI outperforms CorrDiff, starts from LR field instead of noise. Adding SmCL to final output is straightforward (differentiable projection). Probably lowest-hanging fruit.

**E. Multi-variable constraints in generative framework.** 1EMD shows multi-variable ViT downscaling works but has no constraints. Multi-variable constraints paper (2308.01868) only handles Tmin/Tmean/Tmax in UNet. Combining both in a generative model is completely open.

These partially supersede original CLAUDE.md directions 1-3 by providing better baselines (CDSI over standard EDM, pretrained ViT over random UNet).
