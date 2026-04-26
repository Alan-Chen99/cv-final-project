#!/usr/bin/bash
# Run a command on an allocated GPU node via srun + singularity exec.
# Usage: gpu_run.sh JOBID python train.py --config=A
set -euo pipefail

if [[ $# -lt 2 ]]; then
	echo "Usage: $0 JOBID command [args...]" >&2
	exit 1
fi

JOBID=$1
shift

SIF='/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif'
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

srun --jobid="$JOBID" singularity exec --nv \
	--cleanenv \
	--mount 'type=bind,source=/orcd,destination=/orcd' \
	--mount 'type=bind,source=/home/chenxy/nix_store,destination=/nix,ro' \
	--mount "type=bind,source=$PROJECT_DIR,destination=/workspace" \
	--env PREPEND_PATH=/nix/state/profile/bin:/nix/nix_path/bin \
	--env NIX_REMOTE=daemon \
	--env BASH_ENV="$HOME/.bashrc" \
	"$SIF" \
	bash -c "cd /workspace && source .venv/bin/activate && $*"
