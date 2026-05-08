#!/usr/bin/env python3
"""Run a command on an allocated GPU node via srun + singularity exec.

Usage: gpu_run.py JOBID command [args...]
"""

import base64
import shlex
import subprocess
import sys
from pathlib import Path

SIF = "/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif"


def get_project_dir() -> str:
    """Resolve real NFS host path for /workspace mount."""
    result = subprocess.run(
        ["findmnt", "-n", "-o", "SOURCE", "/workspace"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and ":" in result.stdout.strip():
        # NFS source like "nfs001.lb:/home/chenxy/repos/workspace/main"
        return result.stdout.strip().split(":", 1)[1]
    # Fallback: assume script is in <project>/scripts/
    return str(Path(__file__).resolve().parent.parent)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} JOBID command [args...]", file=sys.stderr)
        sys.exit(1)

    jobid = sys.argv[1]
    # Single arg: treat as shell command (may contain pipes, redirects).
    # Multiple args: re-quote to preserve args with spaces/special chars.
    args = sys.argv[2:]
    user_cmd = args[0] if len(args) == 1 else shlex.join(args)
    project_dir = get_project_dir()

    inner = f"cd /workspace && source .venv/bin/activate && {user_cmd}"
    # base64-encode the inner command to avoid nested quoting through
    # srun -> bash -c -> singularity exec -> bash -c
    inner_b64 = base64.b64encode(inner.encode()).decode()

    # srun dispatches to the GPU node, bash loads apptainer module,
    # singularity exec runs inside the container
    srun_cmd = [
        "srun",
        f"--jobid={jobid}",
        "--overlap",
        "bash",
        "-c",
        # This string runs on the GPU node's host OS
        f"""module load apptainer
singularity exec --nv \
    --cleanenv \
    --mount type=bind,source=/orcd,destination=/orcd \
    --mount type=bind,source=/home/chenxy/nix_store,destination=/nix,ro \
    --mount type=bind,source={project_dir},destination=/workspace \
    --env PREPEND_PATH=/nix/state/profile/bin:/nix/nix_path/bin \
    --env NIX_REMOTE=daemon \
    --env BASH_ENV=$HOME/.bashrc \
    --env PYTHONUNBUFFERED=1 \
    {SIF} \
    bash -c 'eval "$(echo {inner_b64} | base64 -d)"' """,
    ]

    sys.exit(subprocess.call(srun_cmd))


if __name__ == "__main__":
    main()
