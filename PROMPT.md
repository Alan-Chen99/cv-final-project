Complete the following user request:

<request added="original (start of loop1)" from="user" commit="2d03959" time="2026-05-02 21:52 EDT">

Task: focus on `32×32 -> 128×128` spatial case with CRPS as metric. <2 hr of training allowed. Do the best you can.
test dataset/task: `32×32 -> 128×128` exactly as in "constrained-downscaling"
Use CRPS as eval metric; you may use any loss function for training.

Most iterations should spend ~2hrs to try models, tune parameters, research papers, and then 2hrs as the final run to get result (only do if useful; likely, after the first couple iterations).
In general, choose under-explored / uncertain directions to attempt, rather than applying techniques that will likely help but by a very tiny amount. There are no penalties if you did very poorly for a iteration.

MUST: Each iteration finishes in 4hrs. All jobs in that iteration is stopped and cleanedup before you emit the result.
MUST: You have up to **one** gpu node to use at any given time. This is not enforced externally: you must keep track yourself.

Claude Code runs on **node1602** (CPU, preemptable — may be killed without notice, with very low probability; You do not need to acount for this). GPU nodes must be allocated separately via `srun`/`salloc` on a GPU partition. The orchestration node (node1602) job expires at approximately **2026-05-04 20:52 EDT** (48h limit, started ~2026-05-02 20:52 EDT). Check `squeue --me` for remaining time. Avoid killing node1602 or nodes that are not spawned by you.

When time is about to end: stop, and write a report file tracked in git. Have subsequent iterations review and revise: if report is changed, its not fix-point.

You should start with baseline (the methods in existing papers) and report these too.

</request>

<request added="start of loop2" from="user" commit="973f5c3" time="2026-05-05 00:21 EDT">

Now: move the report to ./notes, make it self contained.
Compared to `2d03959`, it should be one note added in `./notes` and important and useful code if any. generally all else removed, EXCEPT .ralph which will still stay in git after this step. Do not delete trained weights or other intermediates.

All results in your note must document reproduction, and a valid commit up to HEAD that can be used with all needed scripts. Any metric that you cannot reproduce with the model weights here must be marked as such. Do not re-train: just ignore the results, if lost.

</request>
