#!/usr/bin/bash

set -eux
cd "$(dirname "$0")"

# https://orcd-docs.mit.edu/software/apptainer/
module load apptainer

# # Host's exported `which` function hardcodes /usr/bin/which with GNU flags;
# # unset it so nix's GNU which (on PREPEND_PATH) resolves via PATH instead
# unset -f which 2>/dev/null
# unset which_declare

export APPTAINER_SHELL=/bin/bash

exec singularity shell \
	--pid \
	--cleanenv \
	--env BASH_ENV="$HOME/.bashrc" \
	--mount 'type=bind,source=/orcd,destination=/orcd' \
	--mount 'type=bind,source=/nix,destination=/nix,ro' \
	--mount 'type=bind,source=./,destination=/workspace' \
	--env PREPEND_PATH=/nix/state/profile/bin:/nix/nix_path/bin \
	--env NIX_REMOTE=daemon \
	/home/chenxy/orcd/pool/ubuntu.sif
