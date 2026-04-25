#!/usr/bin/bash

set -eux
cd "$(dirname "$0")"
cd ..

# https://orcd-docs.mit.edu/software/apptainer/
module load apptainer

# # Host's exported `which` function hardcodes /usr/bin/which with GNU flags;
# # unset it so nix's GNU which (on PREPEND_PATH) resolves via PATH instead
# unset -f which 2>/dev/null
# unset which_declare

export APPTAINER_SHELL=/bin/bash

x11_args=()
if [[ -n ${DISPLAY:-} ]]; then
	x11_args+=(--env "DISPLAY=$DISPLAY")
	if [[ -n ${XAUTHORITY:-} && -f $XAUTHORITY ]]; then
		x11_args+=(--env "XAUTHORITY=$XAUTHORITY")
		x11_args+=(--mount "type=bind,source=$XAUTHORITY,destination=$XAUTHORITY,ro")
	fi
fi

exec singularity shell \
	--nv \
	--pid \
	--cleanenv \
	--env BASH_ENV="$HOME/.bashrc" \
	"${x11_args[@]}" \
	--mount 'type=bind,source=/orcd,destination=/orcd' \
	--mount 'type=bind,source=/nix,destination=/nix,ro' \
	--mount 'type=bind,source=./,destination=/workspace' \
	--env PREPEND_PATH=/nix/state/profile/bin:/nix/nix_path/bin \
	--env NIX_REMOTE=daemon \
	'/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif'
