---
name: slurm-preemptable
description: Submit and monitor GPU training jobs on MIT Engaging mit_preemptable partition. TRIGGER when about to run GPU training, or when user asks to run something on the cluster / on GPUs / on preemptable.
---

# Slurm Preemptable GPU Jobs

Submit and run GPU work on `mit_preemptable` nodes. You are inside an
Apptainer container on a CPU node with Slurm binaries bind-mounted
from the host (see `scripts/container.sh`). Call `sbatch`/`salloc`/`srun`
directly — no need to exit the container.

The shared NFS filesystem means files you edit are immediately visible
on GPU nodes. The .venv and code live on this shared FS.

**GPU-side jobs still need `singularity exec`** in sbatch scripts because:
the .venv was built inside Ubuntu 24.04 (glibc 2.39) but GPU nodes run
Rocky 8 (glibc 2.28). Running .venv Python directly on the host would fail.

## Quick reference

### Partitions

| Partition          | GPUs                              | Max time | Limits             |
| ------------------ | --------------------------------- | -------- | ------------------ |
| `mit_preemptable`  | A100, L40S, H100, H200            | 48h      | 4 GPUs, 1024 cores |
| `mit_normal_gpu`   | L40S, H100, H200                   | 6h       | 2 GPUs, 32 cores   |

Preemptable jobs can be killed at any time by higher-priority jobs.
Always use `--requeue` and checkpoint.

### GPU hardware on preemptable

| GPU type    | VRAM   | CPUs/GPU | Nodes |
| ----------- | ------ | -------- | ----- |
| L40S        | 44 GB  | 16       | 58    |
| A100 80GB   | 80 GB  | varies   | many  |
| H100        | 79 GB  | 16       | few   |
| H200        | 140 GB | 15       | 11    |

### Filesystems

| Path                          | Quota | Backed up | Speed | Use for                    |
| ----------------------------- | ----- | --------- | ----- | -------------------------- |
| `/home/<user>`                | 200GB | yes       | fast  | code, important files      |
| `/home/<user>/orcd/pool`      | 1 TB  | no        | slow  | datasets, large files      |
| `/home/<user>/orcd/scratch`   | 1 TB  | no        | fast  | checkpoints, training I/O  |

Scratch files auto-deleted after 6 months inactivity.

### Key Slurm flags

```
-p mit_preemptable          # partition
-G [type:]N                 # GPUs: -G 1, -G l40s:1, -G h200:2
-c N                        # CPUs per task
--mem=XG                    # memory
-t HH:MM:SS                # wall time (max 48:00:00)
--requeue                   # auto-resubmit on preemption
--signal=USR1@120           # send SIGUSR1 120s before kill
-o /path/to/output-%j.log  # stdout (%j = job ID)
-e /path/to/error-%j.log   # stderr
-J jobname                  # job name
```

### Monitoring commands

```bash
squeue --me                                    # all my jobs
squeue --me -p mit_preemptable                 # preemptable only
sacct -j JOBID -o JobID,State,Elapsed,MaxRSS   # job stats
scancel JOBID                                  # cancel job
tail -f /path/to/output-JOBID.log              # watch output
```

## Notification integration (long-running-commands)

Slurm jobs run on remote GPU nodes — they are NOT local processes.
To get notified when a job completes or gets preempted, bridge with
a local background waiter that polls `squeue`.

### Pattern: sbatch + notification

Use `long-running-commands` procedure. The backgrounded command is a
submit-and-wait script, not the training itself.

**Step 1**: Submit and background a waiter (both in one command):

```bash
# run_in_background: true
bash -c '
  JOBID=$(sbatch --parsable scripts/train_preemptable.sh)
  echo "[slurm] Submitted job $JOBID at $(date)"
  while squeue -j $JOBID -h 2>/dev/null | grep -q .; do
    sleep 30
  done
  STATE=$(sacct -j $JOBID -n -o State -X | head -1 | xargs)
  echo "[slurm] Job $JOBID finished at $(date). State: $STATE"
  if [ "$STATE" = "PREEMPTED" ]; then
    NEWID=$(squeue --me -n $(sacct -j $JOBID -n -o JobName -X | head -1 | xargs) -h -o %i | head -1)
    echo "[slurm] Requeued as job $NEWID"
  fi
  echo "[slurm] Last 20 lines of log:"
  tail -20 /home/chenxy/orcd/scratch/logs/train-${JOBID}.log 2>/dev/null || echo "(no log file)"
'
```

This gives you a background task ID. When the Slurm job finishes
(COMPLETED, FAILED, PREEMPTED, TIMEOUT, etc.), the waiter exits and
triggers a `<task-notification>`.

**Step 2**: Read the output file on notification. It contains the
submit confirmation, final state, and log tail. If PREEMPTED with
`--requeue`, it also shows the new job ID — start another waiter for
that one.

### Pattern: salloc + notification

`salloc` blocks until resources are granted. Background it:

```bash
# run_in_background: true
bash -c '
  echo "[salloc] Requesting allocation at $(date)"
  salloc -p mit_preemptable -G 1 -c 16 --mem=64G -t 24:00:00 --no-shell 2>&1
  echo "[salloc] Allocation granted at $(date)"
  squeue --me -h -o "%i %P %T %M %l %R" | head -5
'
```

On notification, extract the job ID from the output and proceed with
`srun --jobid=JOBID` commands.

### Pattern: srun + notification within held allocation

Each `srun` command within an `salloc` can also be long-running.
Background them the same way:

```bash
# run_in_background: true
bash -c '
  echo "[srun] Started at $(date)"
  scripts/gpu_run.sh JOBID python train.py --config=A
  echo "[srun] Finished at $(date) with exit=$?"
'
```

If the allocation gets preempted mid-srun, the srun exits with a
non-zero code. The notification fires, output shows the error. You
then need a new `salloc`.

### Handling preemption notification

When the waiter notification arrives, read the output file and check
the state:

| State       | Meaning                    | Action                                          |
| ----------- | -------------------------- | ----------------------------------------------- |
| COMPLETED   | Job finished normally      | Read results, continue task                     |
| FAILED      | Job errored                | Read log, diagnose, fix, resubmit               |
| PREEMPTED   | Killed by higher-priority  | If `--requeue`: start new waiter for requeued ID |
| TIMEOUT     | Hit wall time limit        | Increase `-t` or checkpoint more often           |
| CANCELLED   | User or admin cancelled    | Check if intentional                            |
| NODE_FAIL   | Node hardware failure      | If `--requeue`: auto-resubmitted                |

## Two submission modes

### Mode 1: `sbatch` (fire-and-forget)

One script, one allocation. Script runs to completion, allocation
released. Good for single long-running training jobs. Cannot run
additional commands after submission without submitting a new job.

### Mode 2: `salloc` + `srun` (hold allocation, run multiple commands)

Reserve a GPU node, then dispatch multiple commands to it. The
allocation stays until you release it, the wall time expires, or you
get preempted. **Preferred when you need to run multiple things**
(train several configs, run training then evaluation, iterate).

#### Step 1: Request allocation

```bash
salloc -p mit_preemptable -G 1 -c 16 --mem=64G -t 24:00:00 --no-shell
```

Blocks until resources are available (seconds to hours). Use
`long-running-commands` skill to background it. Record the job ID
from the output (e.g. `salloc: Granted job allocation 12345678`).

#### Step 2: Run commands on the allocated node

GPU nodes run Rocky 8, but the .venv targets Ubuntu 24.04. Use
`scripts/gpu_run.sh` to wrap `srun` + `singularity exec`:

```bash
scripts/gpu_run.sh JOBID python train.py --config=A
scripts/gpu_run.sh JOBID python train.py --config=B
scripts/gpu_run.sh JOBID python evaluate.py
```

Each call dispatches to the GPU node via srun. Run sequentially,
check results between runs.

#### Step 3: Release the allocation

```bash
scancel JOBID
```

#### Preemption during salloc

All running `srun` commands are killed. Unlike `sbatch --requeue`,
`salloc` does NOT auto-requeue. Request a new allocation.

#### When to use which mode

| Scenario                                    | Mode                |
| ------------------------------------------- | ------------------- |
| Single long training run (hours)            | `sbatch --requeue`  |
| Multiple sequential runs, check between     | `salloc` + `srun`   |
| Parameter sweep (independent jobs)          | `sbatch` job arrays |
| Quick one-off GPU command (nvidia-smi, test) | `salloc` + `srun`  |

## Container setup

**Local (CPU node):** You are inside an Apptainer instance started by
`scripts/container.sh`. Slurm binaries are bind-mounted from the host.
Call `sbatch`/`salloc`/`srun` directly.

**Remote (GPU nodes):** sbatch scripts and `scripts/gpu_run.sh` use
`singularity exec --nv` to run commands inside the same container image
on GPU nodes. This is necessary because .venv targets Ubuntu 24.04 but
GPU nodes run Rocky 8.

SIF: `/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif`

## Procedure

### Step 1: Write the training script with checkpoint/resume

The training script MUST support:
1. Saving checkpoints periodically (every N epochs)
2. Resuming from a checkpoint on restart
3. Handling SIGUSR1 for graceful preemption exit

PyTorch signal handling pattern:

```python
import signal
import sys

_PREEMPT = False

def _sigusr1_handler(signum, frame):
    global _PREEMPT
    _PREEMPT = True

signal.signal(signal.SIGUSR1, _sigusr1_handler)

# In training loop:
for epoch in range(start_epoch, total_epochs):
    train_one_epoch(...)
    save_checkpoint(...)  # every epoch or every N epochs

    if _PREEMPT:
        print(f"Preemption signal received at epoch {epoch}. Saving and exiting.")
        save_checkpoint(...)
        sys.exit(0)  # clean exit; --requeue handles resubmission
```

Checkpoint must include: model state_dict, optimizer state_dict,
epoch number, best metric, RNG states, any scheduler state.

### Step 2: Write the sbatch script

Template — save as e.g. `scripts/train_preemptable.sh`:

```bash
#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -J train-downscale
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH -o /home/chenxy/orcd/scratch/logs/train-%j.log
#SBATCH -e /home/chenxy/orcd/scratch/logs/train-%j.log

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SIF='/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif'

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

module load apptainer

singularity exec --nv \
  --cleanenv \
  --mount 'type=bind,source=/orcd,destination=/orcd' \
  --mount 'type=bind,source=/home/chenxy/nix_store,destination=/nix,ro' \
  --mount "type=bind,source=$PROJECT_DIR,destination=/workspace" \
  --env PREPEND_PATH=/nix/state/profile/bin:/nix/nix_path/bin \
  --env NIX_REMOTE=daemon \
  --env BASH_ENV="$HOME/.bashrc" \
  "$SIF" \
  bash -c 'cd /workspace && source .venv/bin/activate && python <your_script.py> --args'

echo "Job $SLURM_JOB_ID finished at $(date)"
```

Adjust `-G`, `-c`, `--mem`, `-t` based on the workload. Rules of thumb:
- 16 CPUs per GPU (L40S/H100), 15 per GPU (H200)
- 64 GB memory per GPU is a safe start; check with `sacct` after
- Request specific GPU type when needed: `-G h200:1`

### Step 3: Submit and create log directory

```bash
mkdir -p /home/chenxy/orcd/scratch/logs
sbatch scripts/train_preemptable.sh
```

Record the returned job ID.

### Step 4: Monitor

Poll job status and tail logs from the CPU node:

```bash
squeue --me -p mit_preemptable        # is it running or pending?
tail -n 50 /home/chenxy/orcd/scratch/logs/train-JOBID.log   # recent output
sacct -j JOBID -o JobID,State,Elapsed,MaxRSS --units=G      # resource usage
```

If the job was preempted, `sacct` shows State=PREEMPTED followed by a
new JOBID for the requeued run. Check that it resumed from checkpoint.

### Step 5: After completion

```bash
sacct -j JOBID -o JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize,ReqTRES --units=G
```

Check that State=COMPLETED. Retrieve results from checkpoint/output paths.

## Preemption behavior

When a job is preempted:
1. Slurm sends SIGUSR1 (if `--signal=USR1@120` is set) 120s before kill
2. After grace period, sends SIGTERM then SIGKILL
3. If `--requeue` is set, job re-enters the queue automatically
4. The requeued job gets a NEW job ID but same script
5. The script must detect and load the latest checkpoint

If you don't catch the signal, the last periodic checkpoint is used.
The 120s grace period is enough to save a PyTorch checkpoint.

## Multi-GPU jobs

For 2+ GPUs on a single node:

```bash
#SBATCH -G 2
#SBATCH -c 32
#SBATCH --mem=128G

# Inside container:
torchrun --nnodes=1 --nproc_per_node=2 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_endpoint="localhost:1234" \
  train.py --args
```

For multi-node (rare on preemptable due to preemption risk):

```bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
master_node=${nodes[0]}
master_ip=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address)

srun torchrun --nnodes=$SLURM_NNODES \
  --nproc-per-node=2 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$master_ip:1234 \
  train.py --args
```

## Job arrays (parameter sweeps)

```bash
#SBATCH -p mit_preemptable
#SBATCH -a 0-7
#SBATCH -G 1
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH -o /home/chenxy/orcd/scratch/logs/sweep-%A-%a.log

# Use SLURM_ARRAY_TASK_ID to select hyperparameters
python train.py --config configs/sweep_$SLURM_ARRAY_TASK_ID.yaml
```

## Debugging failed jobs

```bash
# Check why a job failed
sacct -j JOBID -o JobID,State,ExitCode,Elapsed
# Read the log
cat /home/chenxy/orcd/scratch/logs/train-JOBID.log
# Common issues:
#   OOM → reduce batch size or request more memory
#   TIMEOUT → increase -t or checkpoint more frequently
#   PREEMPTED without requeue → add --requeue flag
#   NODE_FAIL → transient, requeue handles it
```

## Environment modules (host only, used in sbatch scripts)

```bash
module load apptainer          # needed in sbatch scripts for singularity exec
module load miniforge          # if not using container's .venv
```

These are NOT available inside the container. sbatch scripts run on
GPU node hosts, so they need `module load apptainer` before
`singularity exec`.
