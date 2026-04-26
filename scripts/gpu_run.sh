#!/usr/bin/bash
# Wrapper: delegates to gpu_run.py
exec python3 "$(dirname "$0")/gpu_run.py" "$@"
