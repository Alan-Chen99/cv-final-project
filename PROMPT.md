Complete the following user request:

<request added="original (start of loop1)" from="user" commit="8751577" time="2026-05-08 18:57 EDT">

6 Experiment threads has ran (see notes/ and experiments/ (3-6)) on this task:

> focus on `32×32 -> 128×128` spatial case with CRPS as metric. ≤2 hr of training allowed. Do the best you can.
> test dataset/task: `32×32 -> 128×128` exactly as in "constrained-downscaling"
> Use CRPS as eval metric; you may use any loss function for training. Finetuning pretrained weights (from prior iterations, foundation models, pretrained backbones) is allowed — the 2hr wall-clock budget applies to the finetune run as well.
> The ≤2hr training budget exists so different methods can be compared fairly under equal compute. All methods get the same budget.
> Use the correct energy CRPS: `constrained-downscaling` codebase has a bug in `crps_ensemble()`.

Your task is to organize still-useful code into one place, and run evaluations.

(1) setup a proper python project structure. install and configure `basedpyright, ruff, black, isort, coverage`
(2) create ./src to create persisent code. ./experiments is used in experiments and frozen afterwards. document this.
(3) organize all code in a good way in ./src: training and eval and plotting
(4) write tests. 100% coverage on core logic; only boilerplate can be uncovered. Integration tests only: no unit tests. tests assumes having a gpu. Test on training code should run only a few iterations.
(5) add different evaluation and methods that dont require training
(6) add proper visualization code for key results, using pre-trained weights. Always use fair comaprision. check results visually. Have both output artifact plotting and "data plotting".
(7) write a report file tracked in git. Have subsequent iterations review and revise.

MUST: Do not run significant training; do what you can with available weights.
MUST: Follow software engineering best practices
MUST: You have up to **one** gpu node to use at any given time. I have a limit of 2 "normal" and 4 "preemptable"; Check squeue to decide to use normal or preemptable. This is not enforced externally: you must keep track yourself.

</request>
