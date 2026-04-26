---
name: slurm-preemptable
description: Submit and monitor GPU training jobs on MIT Engaging mit_preemptable partition.
---

# Slurm Preemptable GPU Jobs

> **New, lightly tested.** See `CLAUDE.md` in this directory for reference
> docs and architecture details. Official ORCD docs at
> https://orcd-docs.mit.edu/running-jobs/overview/ are authoritative.

**Always load the `long-running-commands` skill alongside this one.**
Slurm operations (queue waits, salloc, training runs) have unpredictable
runtimes. Use the long-running-commands protocol for all Bash calls that
submit, wait on, or monitor jobs.

You are inside an Apptainer container on a CPU node with Slurm
bind-mounted from the host (`scripts/container.sh`). Call
`sbatch`/`salloc`/`srun` directly.

Shared NFS means edits are immediately visible on GPU nodes.

**GPU-side jobs need `singularity exec`** because .venv targets
Ubuntu 24.04 (glibc 2.39) but GPU nodes run Rocky 8 (glibc 2.28).

## Quick reference

| Partition         | Max time | GPU limit | Notes         |
| ----------------- | -------- | --------- | ------------- |
| `mit_preemptable` | 48h      | 4         | Can be killed |
| `mit_normal_gpu`  | 6h       | 2         | Guaranteed    |

### Key Slurm flags

```
-p mit_preemptable          # partition
-G [type:]N                 # GPUs: -G 1, -G l40s:1, -G h200:2
-c N                        # CPUs per task (16 per GPU default)
--mem=XG                    # memory (64G per GPU default)
-t HH:MM:SS                # wall time (max 48:00:00)
--requeue                   # auto-resubmit on preemption
--signal=USR1@120           # send SIGUSR1 120s before kill
-o /path/to/output-%j.log  # stdout (%j = job ID)
-J jobname                  # job name
```

### Monitoring

```bash
sinfo -p mit_preemptable -O Partition:20,Nodes:10,CPUsState:15,FreeMem:15,StateLong:20,Gres:40,GresUsed:40,Reason:60,Reservation:40  # GPU availability
squeue --me -o "%i %P %j %T %M %l %R"         # all my jobs (untruncated)
sacct -j JOBID -o JobID,State,Elapsed,MaxRSS   # job stats
scancel JOBID                                  # cancel job
tail -f /path/to/output-JOBID.log              # watch output
```

## Two submission modes

### Mode 1: `sbatch` (fire-and-forget)

One script, one allocation. Use for long unattended training.

### Mode 2: `salloc` + `srun` (hold allocation, multiple commands)

Hold a GPU, dispatch commands to it. Preferred for iterative work.

```bash
salloc -p mit_preemptable -G 1 -c 16 --mem=64G -t 24:00:00 --no-shell
```

Then run commands via `scripts/gpu_run.py` (or `gpu_run.sh` which delegates to it):

```bash
python3 scripts/gpu_run.py JOBID python train.py --config=A
python3 scripts/gpu_run.py JOBID python evaluate.py
```

Release with `scancel JOBID`. salloc does NOT auto-requeue on preemption.

| Scenario                         | Mode                |
| -------------------------------- | ------------------- |
| Single long training run         | `sbatch --requeue`  |
| Multiple runs, check between     | `salloc` + `srun`   |
| Parameter sweep (independent)    | `sbatch` job arrays |
| Quick one-off (nvidia-smi, test) | `salloc` + `srun`   |

## sbatch template

Save as e.g. `scripts/train_preemptable.sh`:

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

# Hardcoded: sbatch copies scripts to spool, so $0 won't resolve to the original path
PROJECT_DIR='/home/chenxy/repos/workspace/main'
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

Submit: `mkdir -p /home/chenxy/orcd/scratch/logs && sbatch scripts/train_preemptable.sh`

## Preemption

Training scripts MUST checkpoint periodically and resume from latest
checkpoint on restart. Use `--signal=USR1@120` + a SIGUSR1 handler
that triggers an immediate save-and-exit.

When preempted with `--requeue`: job gets a NEW job ID, same script
re-runs. `SLURM_RESTART_COUNT` is incremented.

## Notification integration (long-running-commands)

Slurm jobs run on remote GPU nodes — not local processes. Bridge to
the `long-running-commands` notification system with a local waiter:

### sbatch + notification

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
  tail -20 /home/chenxy/orcd/scratch/logs/train-${JOBID}.log 2>/dev/null || true
'
```

### salloc + notification

```bash
# run_in_background: true
bash -c '
  echo "[salloc] Requesting allocation at $(date)"
  salloc -p mit_preemptable -G 1 -c 16 --mem=64G -t 24:00:00 --no-shell 2>&1
  echo "[salloc] Allocation granted at $(date)"
  squeue --me -h -o "%i %P %T %M %l %R" | head -5
'
```

### srun + notification

```bash
# run_in_background: true
bash -c '
  echo "[srun] Started at $(date)"
  python3 scripts/gpu_run.py JOBID python train.py --config=A
  echo "[srun] Finished at $(date) with exit=$?"
'
```

### State actions

| State     | Action                                      |
| --------- | ------------------------------------------- |
| COMPLETED | Read results                                |
| FAILED    | Read log, diagnose, fix, resubmit           |
| PREEMPTED | If `--requeue`: start waiter for new job ID |
| TIMEOUT   | Increase `-t` or checkpoint more often      |
| NODE_FAIL | Auto-resubmitted if `--requeue`             |
