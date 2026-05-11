# Memories

## Patterns

## Decisions

## Fixes

### mem-1778532942-27e4
> make_figures.py ensemble plot bug: line 458 had range(min(3, n_vis_samples)) limiting ensemble plots to first 3 samples. Fixed to range(n_vis_samples). Requires GPU re-run to regenerate missing samples 3-4 plots.
<!-- tags: figures, plotting | created: 2026-05-11 -->

### mem-1778532937-348a
> GPU allocation blocker: ivy-ash workflow (ralph/build.yml on pts/25) cancels all non-ivy-ash pending preemptable jobs every ~10min. Both normal GPU slots occupied by ivy-ash until time limit expires. Workaround: submit sbatch when ivy-ash is idle, or use salloc when normal slots are free.
<!-- tags: gpu, slurm, workflow | created: 2026-05-11 -->

## Context
