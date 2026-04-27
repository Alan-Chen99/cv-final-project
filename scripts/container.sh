#!/usr/bin/bash

set -eux
cd "$(dirname "$0")"
cd ..

# https://orcd-docs.mit.edu/software/apptainer/
module load apptainer

export APPTAINER_SHELL=/bin/bash

# Each worktree gets its own instance, named after its directory.
# e.g. /home/chenxy/repos/workspace/research -> workspace-research
WORKTREE_DIR="$(basename "$(pwd)")"
INSTANCE_NAME="workspace-${WORKTREE_DIR}"
SIF='/home/chenxy/orcd/pool/cuda:13.0.2-cudnn-devel-ubuntu24.04.sif'

x11_args=()
if [[ -n ${DISPLAY:-} ]]; then
	x11_args+=(--env "DISPLAY=$DISPLAY")
	if [[ -n ${XAUTHORITY:-} && -f $XAUTHORITY ]]; then
		x11_args+=(--env "XAUTHORITY=$XAUTHORITY")
		x11_args+=(--mount "type=bind,source=$XAUTHORITY,destination=$XAUTHORITY,ro")
	fi
fi

# Slurm client mounts — Rocky 8 binaries run fine on Ubuntu 24.04 (glibc 2.39 > 2.28).
# sbatch RPATH=/usr/lib64/slurm (libslurmfull.so), auth_munge.so RPATH=/usr/lib64 (libmunge.so.2).
# Standard glibc libs resolve via container's /lib/x86_64-linux-gnu/.
# slurm.conf requires SlurmUser=slurm to resolve, so we merge it into the container's passwd.
slurm_args=()
if command -v sbatch &>/dev/null; then
	for cmd in sbatch srun salloc squeue sacct scancel sinfo scontrol; do
		slurm_args+=(--mount "type=bind,source=/usr/bin/$cmd,destination=/usr/bin/$cmd,ro")
	done

	# Merge container's passwd/group with slurm system user.
	# Apptainer injects the current user into the container's /etc/passwd,
	# but slurm.conf's SlurmUser=slurm requires that user to exist too.
	# Files persist in .cache/ for the lifetime of the instance.
	slurm_passwd="$HOME/.cache/container-slurm/$INSTANCE_NAME/passwd"
	slurm_group="$HOME/.cache/container-slurm/$INSTANCE_NAME/group"
	mkdir -p "$(dirname "$slurm_passwd")"
	singularity exec --cleanenv "$SIF" cat /etc/passwd > "$slurm_passwd"
	grep '^slurm:' /etc/passwd >> "$slurm_passwd" 2>/dev/null || true
	singularity exec --cleanenv "$SIF" cat /etc/group > "$slurm_group"
	grep '^slurm:' /etc/group >> "$slurm_group" 2>/dev/null || true

	slurm_args+=(
		--mount "type=bind,source=$slurm_passwd,destination=/etc/passwd,ro"
		--mount "type=bind,source=$slurm_group,destination=/etc/group,ro"
		--mount 'type=bind,source=/usr/lib64/slurm,destination=/usr/lib64/slurm,ro'
		--mount 'type=bind,source=/usr/lib64/libmunge.so.2.0.1,destination=/usr/lib64/libmunge.so.2,ro'
		# scontrol links against readline 7; container has 8
		--mount 'type=bind,source=/lib64/libreadline.so.7.0,destination=/lib/x86_64-linux-gnu/libreadline.so.7,ro'
		--mount 'type=bind,source=/lib64/libhistory.so.7.0,destination=/lib/x86_64-linux-gnu/libhistory.so.7,ro'
		--mount 'type=bind,source=/etc/slurm,destination=/etc/slurm,ro'
		# slurm.conf includes use absolute /home/systems/... paths
		--mount 'type=bind,source=/home/systems,destination=/home/systems,ro'
		--mount 'type=bind,source=/run/munge,destination=/run/munge'
	)
fi

# Flags for instance start (namespaces, GPU, mounts)
instance_args=(
	--nv
	--cleanenv
	--env BASH_ENV="$HOME/.bashrc"
	"${x11_args[@]}"
	"${slurm_args[@]}"
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
