# Slurm Preemptable — Reference

Sources: [ORCD docs](https://orcd-docs.mit.edu), host diagnostics (2026-04-26),
[openclaw-engaging](https://github.com/qsimeon/openclaw-engaging).

## MIT Engaging Cluster

### GPU hardware

| GPU type         | VRAM   | CPUs/GPU | RAM/node   | Nodes | Partition          |
| ---------------- | ------ | -------- | ---------- | ----- | ------------------ |
| L40S             | 44 GB  | 16       | 256 GB     | 58    | preemptable, gpu   |
| A100 80GB        | 80 GB  | varies   | varies     | many  | preemptable, gpu   |
| H100             | 79 GB  | 16       | 512 GB     | few   | preemptable, gpu   |
| H200             | 140 GB | 15       | 1.5 TB     | 11    | preemptable, gpu   |
| RTX PRO 6000     | 96 GB  | —        | —          | some  | preemptable        |

Request specific type: `-G l40s:1`, `-G h200:2`.

### Partitions

| Partition         | Max time   | GPU limit | CPU limit | Notes                           |
| ----------------- | ---------- | --------- | --------- | ------------------------------- |
| `mit_preemptable` | 48h        | 4         | 1024      | Can be killed anytime           |
| `mit_normal_gpu`  | 6h         | 2         | 32        | Guaranteed, shorter             |
| `mit_normal`      | 12h        | —         | —         | CPU only                        |
| `mit_quicktest`   | 15m        | —         | —         | Quick validation                |

### Filesystems

| Path                        | Quota | Backed up | Speed  | Deletion policy       | Use for                   |
| --------------------------- | ----- | --------- | ------ | --------------------- | ------------------------- |
| `/home/<user>`              | 200GB | yes       | fast   | never                 | code, config              |
| `/home/<user>/orcd/pool`    | 1 TB  | no        | slow   | never                 | datasets, SIF images      |
| `/home/<user>/orcd/scratch` | 1 TB  | no        | fast   | 6 months inactivity   | checkpoints, training I/O |

All three are NFS-mounted and visible from every node (login, CPU, GPU).

### Environment modules

```bash
module load apptainer          # Apptainer 1.4.2 (system has 1.3.2)
module load apptainer/1.4.2    # explicit version
module load miniforge/24.3.0-0 # conda/mamba (if not using container)
module avail                   # list all
module list                    # loaded modules
```

Modules are host-only. Not available inside Apptainer containers.
sbatch scripts running on GPU nodes need `module load apptainer`
before `singularity exec`.

### Requesting resources

**CPUs**: `-c N` (cores per task). Match to GPU count × CPUs/GPU from hardware table.

**Memory**: `--mem=XG` (per node). 64 GB per GPU is safe default. Check actual
usage after with `sacct -j JOBID -o MaxRSS --units=G`.

**GPUs**: `-G [type:]N`. For multi-node: `--gpus-per-node=N` instead of `-G`.

**Time**: `-t HH:MM:SS`. Max 48h on preemptable, 6h on normal_gpu.

### Application analysis

After a job completes, check resource efficiency:

```bash
# Detailed resource usage
sacct -j JOBID -o JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize,ReqTRES --units=G

# GPU utilization (if job is still running)
srun --jobid=JOBID nvidia-smi

# Memory high-water mark
sacct -j JOBID -o MaxRSS --units=G
```

Overrequesting wastes shared resources. Underrequesting causes OOM kills.

## Multi-GPU jobs

### Single node (2-4 GPUs)

```bash
#SBATCH -G 2
#SBATCH -c 32
#SBATCH --mem=128G

# Inside container, use torchrun:
torchrun --nnodes=1 --nproc_per_node=2 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_endpoint="localhost:1234" \
  train.py --args
```

### Multi-node (rare on preemptable — preemption kills all nodes)

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

Intra-node communication uses NVLink/PCIe. Inter-node uses InfiniBand/Ethernet.

## Job arrays (parameter sweeps)

```bash
#SBATCH -p mit_preemptable
#SBATCH -a 0-7
#SBATCH -G 1
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH -o /home/chenxy/orcd/scratch/logs/sweep-%A-%a.log

# %A = array master job ID, %a = task index
# SLURM_ARRAY_TASK_ID is the task index (0-7)
python train.py --config configs/sweep_$SLURM_ARRAY_TASK_ID.yaml
```

Each array task is an independent job with its own allocation.
`--requeue` applies per-task. Checkpointing must be per-task
(include `SLURM_ARRAY_TASK_ID` in checkpoint path).

Limit concurrent tasks: `#SBATCH -a 0-7%4` (max 4 at once).

## Debugging failed jobs

```bash
# Check exit state and code
sacct -j JOBID -o JobID,State,ExitCode,Elapsed

# Read the log
cat /home/chenxy/orcd/scratch/logs/train-JOBID.log

# Common failure patterns:
#   State=OUT_OF_MEMORY  → reduce batch size or --mem
#   State=TIMEOUT        → increase -t or checkpoint more
#   State=PREEMPTED      → normal if --requeue is set
#   State=FAILED, Exit=1 → Python exception, read log
#   State=FAILED, Exit=9 → SIGKILL (OOM killer)
#   State=NODE_FAIL      → transient hardware, requeue handles it
```

## Preemption signal chain

1. Slurm sends signal specified by `--signal=USR1@120` (SIGUSR1, 120s before kill)
2. After grace period: SIGCONT → SIGTERM → (30s) → SIGKILL
3. If `--requeue`: job re-enters queue with NEW job ID, same script
4. Requeued job sees `SLURM_RESTART_COUNT` incremented
5. Training script must detect and load latest checkpoint on startup

Signal propagation through Apptainer: SIGUSR1 passes through
`singularity exec` to the child process. Verified to work with the
bind-mount setup in `scripts/container.sh`.

The 120s grace period is enough to save a PyTorch checkpoint (typically <10s).
If your checkpoint is very large (multi-GB), increase the grace period:
`--signal=USR1@300`.

## Container bind-mount architecture

`scripts/container.sh` mounts Slurm client tools from the Rocky 8 host
into the Ubuntu 24.04 container. This works because:

**glibc compatibility**: Rocky 8 has glibc 2.28, container has 2.39. Newer
glibc runs older binaries (backwards compatible). Standard libs
(libc, libm, libpthread, libdl, librt, libresolv) resolve via the
container's `/lib/x86_64-linux-gnu/` through its ldconfig cache.

**Mounts required** (15 total):

| Category   | Source (host)                     | Destination (container)                     | Why                                        |
| ---------- | --------------------------------- | ------------------------------------------- | ------------------------------------------ |
| Binaries   | `/usr/bin/{sbatch,srun,...}`      | same path                                   | 8 Slurm client commands                    |
| Libs       | `/usr/lib64/slurm/`              | same path                                   | libslurmfull.so (sbatch RPATH)             |
| Libs       | `/usr/lib64/libmunge.so.2.0.1`   | `/usr/lib64/libmunge.so.2`                  | auth_munge.so RPATH=/usr/lib64             |
| Libs       | `/lib64/libreadline.so.7.0`      | `/lib/x86_64-linux-gnu/libreadline.so.7`    | scontrol needs readline 7, container has 8 |
| Libs       | `/lib64/libhistory.so.7.0`       | `/lib/x86_64-linux-gnu/libhistory.so.7`     | scontrol dependency                        |
| Config     | `/etc/slurm/`                    | same path                                   | slurm.conf (symlinks to /home/systems/)    |
| Config     | `/home/systems/`                 | same path                                   | symlink target for slurm.conf includes     |
| Auth       | `/run/munge/`                    | same path                                   | munge daemon socket for auth               |
| Users      | merged passwd/group              | `/etc/passwd`, `/etc/group`                 | SlurmUser=slurm must resolve               |

The merged passwd/group files combine the container's default users
(including Apptainer-injected current user) with the host's `slurm:x:450:450`
entry. Stored at `$HOME/.cache/container-slurm/{passwd,group}`.

**Library dependency chain**:
- `sbatch` → `libslurmfull.so` (RPATH `/usr/lib64/slurm`)
- `libslurmfull.so` → standard glibc only
- `auth_munge.so` (plugin, loaded at runtime) → `libmunge.so.2` (RPATH `/usr/lib64`)
- `scontrol` → `libreadline.so.7` + `libhistory.so.7`
- All others (srun, salloc, squeue, sacct, scancel, sinfo) → libslurmfull.so + standard glibc

**slurm.conf indirection**:
`/etc/slurm/slurm.conf` → symlink → `/home/systems/slurm/etc/slurm/slurm.conf`
(real path: `/orcd/home/002/systems/slurm/etc/slurm/slurm.conf`).
slurm.conf then uses `include /home/systems/slurm/etc/slurm/nodes8.conf` etc.
Both `/etc/slurm/` and `/home/systems/` must be mounted for the full chain to resolve.

## OpenClaw comparison

[openclaw-engaging](https://github.com/qsimeon/openclaw-engaging) by qsimeon
solves the same Slurm-in-container problem with `OPENCLAW_SLURM_BINDS=1`.
Their implementation (copy-pasted across 4 scripts):

```bash
if [ "${OPENCLAW_SLURM_BINDS:-}" = "1" ]; then
  for cmd in sbatch squeue scancel sinfo srun sacct; do
    [ -f "/usr/bin/$cmd" ] && BIND_FLAGS="$BIND_FLAGS -B /usr/bin/$cmd"
  done
  [ -d /etc/slurm ] && BIND_FLAGS="$BIND_FLAGS -B /etc/slurm"
  [ -d /usr/lib64/slurm ] && BIND_FLAGS="$BIND_FLAGS -B /usr/lib64/slurm"
  [ -d /run/munge ] && BIND_FLAGS="$BIND_FLAGS -B /run/munge"
fi
```

Missing from theirs vs ours: no libmunge.so.2 mount, no passwd/group merge
for SlurmUser, no libreadline for scontrol, no /home/systems for slurm.conf
symlink targets, no salloc/scontrol binaries. Uses `-B` (same src/dest) instead
of `--mount` with explicit paths (needed for library renames).
