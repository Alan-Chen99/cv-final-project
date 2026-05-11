Complete the following user request:

<request added="original (start of loop1)" from="user" commit="a14302e">

Task: Add and improve evaluation metrics for all trained models. Update `.figures/` and write report.
Goal: comprehensive set of visualizations, fair comparison, solid and sound results. Key results demonstrated. Plots make sense and show what they need to show well.

Requried metrics:

- spectral power
- https://arxiv.org/pdf/2604.03459 (Relative Average Log Spectral Distance )
- find more

You must also find problems and fix current `./figures` including inconsitency and other problems.

Use `src/downscaling/`. This is not a experiment: do not use `./experiments/`

MUST: Find and add papers to learn about what to evaluate.
MUST: Ground in papers: what metric matters? what others used?
MUST: Run all on gpu, not this node
MUST: Follow SWE best practices.
MUST: Check all plots visually.
MUST: You have up to **four** GPU node at any given time. I have a limit of 2 "normal" and 4 "preemptable"; check `squeue` to decide which to use. This is not enforced externally: you must track it yourself.
MUST: Write a report file tracked in git. Have subsequent iterations review and revise it.

</request>
