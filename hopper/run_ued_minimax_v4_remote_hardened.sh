#!/usr/bin/env bash
# Hermetic outer runner for the immutable d602 v4 engineering scripts.
set -euo pipefail
umask 027
export PATH=/usr/bin:/bin
unset HOME PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONOPTIMIZE
unset LD_LIBRARY_PATH LD_PRELOAD ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

[[ $# == 3 ]] || {
  echo "usage: $0 RUNG HARDENING_SBATCH --ued-input-envelope=ABSOLUTE_PATH" >&2
  exit 2
}
readonly RUNG=$1
readonly HARDENING_SBATCH=$2
case "$RUNG" in import|one_update|terminal) ;; *) echo "invalid rung" >&2; exit 2;; esac
[[ "$3" == --ued-input-envelope=* ]] || { echo "missing exact input-envelope argument" >&2; exit 2; }
readonly INPUT_ENVELOPE=${3#--ued-input-envelope=}
[[ "$INPUT_ENVELOPE" == /* && "$INPUT_ENVELOPE" != *'/../'* \
      && -f "$INPUT_ENVELOPE" && ! -L "$INPUT_ENVELOPE" \
      && "$(readlink -f -- "$INPUT_ENVELOPE")" == "$INPUT_ENVELOPE" ]] || {
  echo "unsafe input envelope" >&2
  exit 2
}
[[ "$HARDENING_SBATCH" == /* && -f "$HARDENING_SBATCH" \
      && ! -L "$HARDENING_SBATCH" \
      && "$(readlink -f -- "$HARDENING_SBATCH")" == "$HARDENING_SBATCH" ]] || {
  echo "unsafe hardening sbatch" >&2
  exit 2
}
[[ "${SLURM_EXPORT_ENV:-}" == NIL ]] || {
  echo "Slurm NIL export mode required" >&2
  exit 2
}
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo "arrays forbidden" >&2
  exit 2
}
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || {
  echo "restarted jobs forbidden" >&2
  exit 2
}
while IFS= read -r name; do
  [[ "$name" != UED_* ]] || { echo "ambient UED variable forbidden: $name" >&2; exit 2; }
done < <(compgen -e)

declare -A INPUTS=()
while IFS= read -r -d '' record; do
  [[ "$record" == UED_*=* ]] || { echo "malformed UED input record" >&2; exit 2; }
  key=${record%%=*}
  value=${record#*=}
  [[ "$key" =~ ^UED_[A-Z0-9_]+$ && -n "$value" \
       && -z "${INPUTS[$key]:-}" ]] || { echo "duplicate/empty UED input" >&2; exit 2; }
  [[ "$value" != *$'\n'* && "$value" != *$'\r'* && "$value" != *$'\t'* ]] || {
    echo "unsafe UED input value" >&2; exit 2;
  }
  INPUTS["$key"]=$value
done < "$INPUT_ENVELOPE"
[[ ${#INPUTS[@]} -gt 0 ]] || { echo "empty UED input envelope" >&2; exit 2; }
for key in "${!INPUTS[@]}"; do export "$key=${INPUTS[$key]}"; done

readonly BUNDLE_DIR=${INPUTS[UED_BUNDLE_DIR]:-}
[[ -n "$BUNDLE_DIR" && "$HARDENING_SBATCH" == "$BUNDLE_DIR/hopper/sbatch/"* ]] || {
  echo "hardening sbatch/bundle mismatch" >&2
  exit 2
}
readonly JOB_GUARD="$BUNDLE_DIR/ued_benchmark/hopper_v4_remote_hardening/job_guard.py"
[[ -f "$JOB_GUARD" && ! -L "$JOB_GUARD" ]] || { echo "missing job guard" >&2; exit 2; }
[[ "$(sha256sum "$JOB_GUARD" | awk '{print $1}')" == "${INPUTS[UED_JOB_GUARD_SHA256]:-}" ]] || {
  echo "job-guard binding drift" >&2
  exit 2
}
/usr/bin/python3 -I -B "$JOB_GUARD" preflight \
  --rung "$RUNG" --input-envelope "$INPUT_ENVELOPE"

case "$RUNG" in
  import)
    legacy_names=(
      UED_BUNDLE_DIR UED_BUNDLE_MANIFEST_SHA256 UED_UPSTREAM_COMMIT
      UED_UPSTREAM_TREE UED_UPSTREAM_BUNDLE_SHA256 UED_OVERLAY_MANIFEST_SHA256
      UED_SBATCH_SHA256 UED_ENV_DIR UED_ENV_LOCK_SHA256 UED_ENV_FREEZE_SHA256
      UED_ENV_MANIFEST_SHA256
    )
    legacy_rel=hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch
    ;;
  one_update)
    legacy_names=(
      UED_BUNDLE_DIR UED_BUNDLE_MANIFEST_SHA256 UED_UPSTREAM_COMMIT
      UED_UPSTREAM_TREE UED_UPSTREAM_BUNDLE_SHA256 UED_OVERLAY_MANIFEST_SHA256
      UED_SBATCH_SHA256 UED_ENV_DIR UED_ENV_LOCK_SHA256 UED_ENV_FREEZE_SHA256
      UED_ENV_MANIFEST_SHA256 UED_CONFIG_SHA256 UED_CONTRACT_SHA256
      UED_IMPORT_SMOKE_RESULT_DIR UED_IMPORT_SMOKE_MANIFEST_SHA256
    )
    legacy_rel=hopper/sbatch/ued_minimax_v4_one_update_smoke.sbatch
    ;;
  terminal)
    legacy_names=(
      UED_BUNDLE_DIR UED_BUNDLE_MANIFEST_SHA256 UED_UPSTREAM_COMMIT
      UED_UPSTREAM_TREE UED_UPSTREAM_BUNDLE_SHA256 UED_OVERLAY_MANIFEST_SHA256
      UED_SBATCH_SHA256 UED_ENV_DIR UED_ENV_LOCK_SHA256 UED_ENV_FREEZE_SHA256
      UED_ENV_MANIFEST_SHA256 UED_IMPORT_SMOKE_RESULT_DIR
      UED_IMPORT_SMOKE_MANIFEST_SHA256 UED_ONE_UPDATE_RESULT_DIR
      UED_ONE_UPDATE_MANIFEST_SHA256 UED_ARM UED_CONFIG_SHA256
      UED_CONTRACT_SHA256 UED_PROTOCOL_SHA256 UED_PHASE_A_DRIVER_SHA256
      UED_TRAINING_DRIVER_SHA256 UED_EVALUATION_DRIVER_SHA256
      UED_ASSEMBLER_SHA256 UED_FINALIZER_SHA256
    )
    legacy_rel=hopper/sbatch/ued_minimax_v4_terminal_chain_smoke.sbatch
    ;;
esac
readonly LEGACY="$BUNDLE_DIR/$legacy_rel"
[[ -f "$LEGACY" && ! -L "$LEGACY" \
   && "$(sha256sum "$LEGACY" | awk '{print $1}')" == "${INPUTS[UED_LEGACY_SBATCH_SHA256]}" ]] || {
  echo "legacy sbatch drift" >&2
  exit 2
}

child_env=(
  PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
  SLURM_EXPORT_ENV=NIL
)
for name in \
  SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_PARTITION SLURM_JOB_QOS \
  SLURM_RESTART_COUNT SLURM_CPUS_PER_TASK SLURM_JOB_NUM_NODES SLURM_NTASKS \
  SLURM_MEM_PER_NODE SLURM_GPUS_ON_NODE SLURM_TRES_PER_NODE SLURM_JOB_GPUS \
  SLURM_NODELIST SLURM_SUBMIT_DIR SLURM_SUBMIT_HOST SLURMD_NODENAME \
  CUDA_VISIBLE_DEVICES; do
  [[ -n "${!name:-}" ]] && child_env+=("$name=${!name}")
done
for name in "${legacy_names[@]}"; do
  [[ -n "${INPUTS[$name]:-}" ]] || { echo "missing legacy input: $name" >&2; exit 2; }
  child_env+=("$name=${INPUTS[$name]}")
done
/usr/bin/env -i "${child_env[@]}" /usr/bin/bash "$LEGACY"

/usr/bin/python3 -I -B "$JOB_GUARD" postflight \
  --rung "$RUNG" --input-envelope "$INPUT_ENVELOPE"
