Complete the following user request:

<request added="original (start of loop1)" from="user" commit="d0bc53a" time="2026-05-06 00:45 EDT">

Task: focus on `32×32 -> 128×128` spatial case with CRPS as metric. ≤2 hr of training allowed. Do the best you can.
test dataset/task: `32×32 -> 128×128` exactly as in "constrained-downscaling"
Use CRPS as eval metric; you may use any loss function for training. Finetuning pretrained weights (from prior iterations, foundation models, pretrained backbones) is allowed — the 2hr wall-clock budget applies to the finetune run as well.

The ≤2hr training budget exists so different methods can be compared fairly under equal compute. All methods get the same budget.

Use the correct energy CRPS: `CRPS = E|X−y| − 0.5·E|X−X'|`. The `constrained-downscaling` codebase has a bug in `crps_ensemble()` — it uses `fc.shape[-1]**2` (=16384) instead of `fc.shape[0]**2` (=M²) as denominator, underestimating CRPS by ~50%. Do NOT report numbers from the buggy formula.

Most iterations should spend ~2hrs to try models, tune parameters, research papers, and then 2hrs as the final run to get result (only do if useful; likely, after the first couple iterations).

MUST: If possible, choose under-explored / uncertain directions to attempt, rather than applying techniques that will likely help but by a very tiny amount. There are no penalties if you did very poorly for a iteration.
MUST: Review loss graph, output samples, metrics, etc visually to guide research. Check key graphs and outputs into git.
MUST: Each iteration finishes in 4hrs. All jobs in that iteration is stopped and cleanedup before you emit the result.
MUST: You have up to **one** gpu node to use at any given time. I have a limit of 2 "normal" and 4 "preemptable"; Check squeue to decide to use normal or preemptable. This is not enforced externally: you must keep track yourself.

Claude Code runs on **node1627** (CPU, preemptable — may be killed without notice, with very low probability; you do not need to account for this). GPU nodes must be allocated separately via `srun`/`salloc` on a GPU partition. The orchestration node (node1627) job expires at approximately **2026-05-07 10:00 EDT** (48h limit, started ~2026-05-05 10:00 EDT). Check `squeue --me` for remaining time. Avoid killing node1627 or nodes that are not spawned by you.

When time is about to end: stop, and write a report file tracked in git. Have subsequent iterations review and revise: if report is changed, its not fix-point.

MUST: Agents should aim to finish all exploration/training by the **40-hour mark** (~8hr remaining on the allocation). After 40hr elapsed (approximately **2026-05-07 02:00 EDT**), no more new ideas or new training jobs. Instead, the remaining time must be spent on: evaluation of existing models, organizing data/artifacts, and writing and revising reports.

Starting research direction:
Start by running best general image downscaling model(s) available zero-shot, evaluate. Evaluate other zero shot methods.
Then, finetune general image models and evaluate.
Decide what to go from here.

### Prior findings

Previous iterations produced reports — treat as context (verify claims before relying on them):

- [notes/2026-05-02-flow-matching-downscaling.md](notes/2026-05-02-flow-matching-downscaling.md) — research branch: LR-anchor flow matching, 5.2M params, CRPS 0.199
- [notes/2026-05-05-spatial-4x-crps-experiment.md](notes/2026-05-05-spatial-4x-crps-experiment.md) — research2 branch: OT-CFM residual flow, 13M params, CRPS 0.171 (2K test) / ~0.094 (their own metric)
- [notes/2026-05-05-cross-comparison.md](notes/2026-05-05-cross-comparison.md) — unified cross-comparison of both branches under corrected CRPS

</request>
