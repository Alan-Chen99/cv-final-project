Complete the following user request:

<request added="original (start of loop1)" from="user" commit="a14302e" time="2026-05-11 EDT">

Task: Add evaluation metrics beyond pixel-space (CRPS/MAE/RMSE/mass violation) for all trained models. **No training.** You decide what to compute and in what order — use findings from each metric to guide what to investigate next.

Start by finding and adding papers to learn about what to evaluate.

Use `src/downscaling/`. This is not a experiment: do not use `./experiments/`

MUST: No training. Inference and metric computation only.
MUST: Code in `src/downscaling/`, pass ruff/basedpyright, add integration tests.
MUST: Check all plots visually — verify they look physically reasonable.
MUST: You have up to **four** GPU node at any given time. I have a limit of 2 "normal" and 4 "preemptable"; check `squeue` to decide which to use. This is not enforced externally: you must track it yourself.
MUST: Write a report file tracked in git. Have subsequent iterations review and revise it.

</request>
