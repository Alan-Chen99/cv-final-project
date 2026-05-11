#!/bin/bash
# Run eval on GPU node via srun (interactive, no singularity).
# Requires: active salloc with GPU
#
# Usage:
#   srun --jobid=<JOBID> --overlap scripts/eval_srun.sh [era5|noresm|both] [--max-samples N]
#
# The editable install .pth references /workspace/src which doesn't exist on GPU nodes.
# PYTHONPATH workaround makes it importable from the worktree path.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

cd "$PROJECT_DIR"
source .venv/bin/activate

MODE="${1:-both}"
shift || true
EXTRA_ARGS="$*"

echo "Project: $PROJECT_DIR"
echo "Python: $(which python)"
echo "Mode: $MODE"
echo "Started at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

case "$MODE" in
  era5)
    python scripts/run_eval.py --max-samples 500 --output eval_results_8metrics.json $EXTRA_ARGS
    ;;
  noresm)
    python scripts/run_eval_noresm.py --max-samples 500 --output noresm_eval_results_8metrics.json $EXTRA_ARGS
    ;;
  both)
    python scripts/run_eval.py --max-samples 500 --output eval_results_8metrics.json $EXTRA_ARGS
    echo "ERA5 done at $(date)"
    python scripts/run_eval_noresm.py --max-samples 500 --output noresm_eval_results_8metrics.json $EXTRA_ARGS
    echo "NorESM done at $(date)"
    ;;
  *)
    echo "Usage: $0 [era5|noresm|both] [extra args...]"
    exit 1
    ;;
esac

echo "Finished at $(date)"
