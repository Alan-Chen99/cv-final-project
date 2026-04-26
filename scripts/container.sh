#!/usr/bin/bash

set -eux
cd "$(dirname "$0")"
cd ..

# https://orcd-docs.mit.edu/software/apptainer/
module load apptainer

export APPTAINER_SHELL=/bin/bash

INSTANCE_NAME=workspace
SIF='/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif'

x11_args=()
if [[ -n ${DISPLAY:-} ]]; then
	x11_args+=(--env "DISPLAY=$DISPLAY")
	if [[ -n ${XAUTHORITY:-} && -f $XAUTHORITY ]]; then
		x11_args+=(--env "XAUTHORITY=$XAUTHORITY")
		x11_args+=(--mount "type=bind,source=$XAUTHORITY,destination=$XAUTHORITY,ro")
	fi
fi

# Flags for instance start (namespaces, GPU, mounts)
instance_args=(
	--nv
	--cleanenv
	--env BASH_ENV="$HOME/.bashrc"
	"${x11_args[@]}"
	--mount 'type=bind,source=/orcd,destination=/orcd'
	# --mount 'type=bind,source=/nix,destination=/nix,ro'
	--mount 'type=bind,source=/home/chenxy/nix_store,destination=/nix,ro'
	--mount 'type=bind,source=./,destination=/workspace'
	--env PREPEND_PATH=/nix/state/profile/bin:/nix/nix_path/bin
	--env NIX_REMOTE=daemon
)

# Env-only args shared between instance start and shell attach
x11_env_args=()
if [[ -n ${DISPLAY:-} ]]; then
	x11_env_args+=(--env "DISPLAY=$DISPLAY")
	if [[ -n ${XAUTHORITY:-} && -f $XAUTHORITY ]]; then
		x11_env_args+=(--env "XAUTHORITY=$XAUTHORITY")
	fi
fi

# Flags for shell into running instance (env only; namespaces/mounts inherited)
shell_args=(
	--cleanenv
	--env BASH_ENV="$HOME/.bashrc"
	"${x11_env_args[@]}"
	--env PREPEND_PATH=/nix/state/profile/bin:/nix/nix_path/bin
	--env NIX_REMOTE=daemon
)

start_instance() {
	if singularity instance list | grep -q "$INSTANCE_NAME"; then
		return 0
	fi
	singularity instance start "${instance_args[@]}" "$SIF" "$INSTANCE_NAME"
}

case "${1:-}" in
stop)
	singularity instance stop "$INSTANCE_NAME"
	exit
	;;
start)
	start_instance
	exit
	;;
esac

# Default: start instance if needed, then shell into it
start_instance
exec singularity shell "${shell_args[@]}" "instance://$INSTANCE_NAME"
