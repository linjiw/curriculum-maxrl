#!/usr/bin/env bash
# Local-only renderer for exact v4 Slurm submissions. It validates the complete
# export closure but deliberately never calls sbatch in this unaudited candidate.
set -euo pipefail
umask 027

usage() {
  echo "usage: $0 render SBATCH_PATH KEY=VALUE..." >&2
  echo "       $0 submit ...  # permanently refused in this candidate" >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
readonly ACTION=$1
shift
if [[ "$ACTION" == submit ]]; then
  echo "v4 remote submission is HOLD pending an independent frozen-byte audit" >&2
  exit 2
fi
[[ "$ACTION" == render && $# -ge 2 ]] || usage
while IFS= read -r ambient_name; do
  if [[ "$ambient_name" == SBATCH_* ]]; then
    echo "ambient Slurm client override forbidden: $ambient_name" >&2
    exit 2
  fi
done < <(compgen -e)
readonly SCRIPT=$1
shift
[[ -f "$SCRIPT" && ! -L "$SCRIPT" ]] || {
  echo "unsafe or missing sbatch path: $SCRIPT" >&2
  exit 2
}
readonly BASENAME="$(basename "$SCRIPT")"

common=(
  UED_BUNDLE_DIR UED_BUNDLE_MANIFEST_SHA256 UED_UPSTREAM_COMMIT
  UED_UPSTREAM_TREE UED_UPSTREAM_BUNDLE_SHA256 UED_OVERLAY_MANIFEST_SHA256
  UED_ENV_DIR UED_ENV_LOCK_SHA256 UED_ENV_FREEZE_SHA256
  UED_ENV_MANIFEST_SHA256 UED_SBATCH_SHA256
)
case "$BASENAME" in
  ued_minimax_v4_gpu_smoke.sbatch)
    required=("${common[@]}")
    ;;
  ued_minimax_v4_one_update_smoke.sbatch)
    required=("${common[@]}" UED_IMPORT_SMOKE_RESULT_DIR
      UED_IMPORT_SMOKE_MANIFEST_SHA256 UED_CONFIG_SHA256
      UED_CONTRACT_SHA256)
    ;;
  ued_minimax_v4_terminal_chain_smoke.sbatch)
    required=("${common[@]}" UED_IMPORT_SMOKE_RESULT_DIR
      UED_IMPORT_SMOKE_MANIFEST_SHA256 UED_ONE_UPDATE_RESULT_DIR
      UED_ONE_UPDATE_MANIFEST_SHA256 UED_ARM UED_CONFIG_SHA256
      UED_CONTRACT_SHA256 UED_PROTOCOL_SHA256 UED_PHASE_A_DRIVER_SHA256
      UED_TRAINING_DRIVER_SHA256
      UED_EVALUATION_DRIVER_SHA256 UED_ASSEMBLER_SHA256
      UED_FINALIZER_SHA256)
    ;;
  *)
    echo "unrecognized v4 sbatch basename: $BASENAME" >&2
    exit 2
    ;;
esac

declare -A allowed=() seen=() values=()
for key in "${required[@]}"; do
  allowed["$key"]=1
done
for assignment in "$@"; do
  [[ "$assignment" == *=* ]] || { echo "export lacks '=': $assignment" >&2; exit 2; }
  key=${assignment%%=*}
  value=${assignment#*=}
  [[ "$key" =~ ^[A-Z][A-Z0-9_]*$ ]] || { echo "unsafe export key: $key" >&2; exit 2; }
  [[ -n "$value" ]] || { echo "empty export value: $key" >&2; exit 2; }
  [[ -n "${allowed[$key]:-}" ]] || { echo "extra export forbidden: $key" >&2; exit 2; }
  [[ -z "${seen[$key]:-}" ]] || { echo "duplicate export: $key" >&2; exit 2; }
  [[ "$value" != *','* && "$value" != *$'\n'* && "$value" != *$'\r'* \
        && "$value" != *$'\t'* && "$value" != *' '* ]] || {
    echo "unsafe export value: $key" >&2
    exit 2
  }
  seen["$key"]=1
  values["$key"]=$value
done
for key in "${required[@]}"; do
  [[ -n "${seen[$key]:-}" ]] || { echo "missing export: $key" >&2; exit 2; }
done
[[ ${#seen[@]} -eq ${#required[@]} ]] || { echo "export cardinality drift" >&2; exit 2; }

for key in "${required[@]}"; do
  case "$key" in
    *_SHA256)
      [[ "${values[$key]}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "malformed SHA-256: $key" >&2
        exit 2
      }
      ;;
  esac
done
[[ "${values[UED_UPSTREAM_COMMIT]}" == d053054c5290a04c1c4cd8b55704d999cad73e30 ]]
[[ "${values[UED_UPSTREAM_TREE]}" == b0cace1fc54984e21a842f12d15d0b899e33d270 ]]
[[ "${values[UED_BUNDLE_DIR]}" =~ ^/scratch/[A-Za-z0-9._-]+/maxrl/bundles/ued_minimax_v4_engineering/[0-9a-f]{20}$ ]]
[[ "$(basename -- "${values[UED_BUNDLE_DIR]}")" \
    == "${values[UED_BUNDLE_MANIFEST_SHA256]:0:20}" ]]
bundle_user=${values[UED_BUNDLE_DIR]#/scratch/}
bundle_user=${bundle_user%%/*}
[[ "${values[UED_ENV_DIR]}" =~ ^/scratch/[A-Za-z0-9._-]+/envs/ued-minimax-v2-[0-9a-f]{16}-[0-9a-f]{16}$ ]]
[[ "${values[UED_ENV_DIR]}" == "/scratch/$bundle_user/"* ]]
[[ "${values[UED_BUNDLE_DIR]}" != *06ffeeeb6998e8ddb1ce* ]]
[[ "${values[UED_SBATCH_SHA256]}" == "$(sha256sum "$SCRIPT" | awk '{print $1}')" ]] || {
  echo "sbatch self hash mismatch" >&2
  exit 2
}
if [[ -n "${values[UED_ARM]:-}" ]]; then
  [[ "${values[UED_ARM]}" == frontier || "${values[UED_ARM]}" == maxmc ]]
  if [[ "${values[UED_ARM]}" == frontier ]]; then
    [[ "${values[UED_CONFIG_SHA256]}" == 0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2 ]]
  else
    [[ "${values[UED_CONFIG_SHA256]}" == a3cc3ddf387a3bb7cf3d9759c3f13e6a74f2c7e32de311ac344d54ef5e703ec6 ]]
  fi
  [[ "${values[UED_PROTOCOL_SHA256]}" == 1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269 ]]
fi
if [[ -n "${values[UED_CONTRACT_SHA256]:-}" ]]; then
  [[ "${values[UED_CONTRACT_SHA256]}" == 3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b ]]
fi
if [[ -n "${values[UED_IMPORT_SMOKE_RESULT_DIR]:-}" ]]; then
  [[ "$(dirname -- "${values[UED_IMPORT_SMOKE_RESULT_DIR]}")" \
      == "/scratch/$bundle_user/maxrl/tests/ued-minimax-v4-import" ]]
  [[ "$(basename -- "${values[UED_IMPORT_SMOKE_RESULT_DIR]}")" =~ ^[0-9]+$ ]]
fi
if [[ -n "${values[UED_ONE_UPDATE_RESULT_DIR]:-}" ]]; then
  [[ "$(dirname -- "${values[UED_ONE_UPDATE_RESULT_DIR]}")" \
      == "/scratch/$bundle_user/maxrl/tests/ued-minimax-v4-one-update" ]]
  [[ "$(basename -- "${values[UED_ONE_UPDATE_RESULT_DIR]}")" =~ ^[0-9]+$ ]]
fi
if [[ "$BASENAME" == ued_minimax_v4_one_update_smoke.sbatch ]]; then
  [[ "${values[UED_CONFIG_SHA256]}" == 0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2 ]]
fi

exports=()
for key in "${required[@]}"; do
  exports+=("$key=${values[$key]}")
done
joined=$(IFS=,; printf '%s' "${exports[*]}")
# Slurm's NONE sentinel cannot be combined with explicit assignments.  A
# nonempty assignment-only list already means "export exactly these values"
# (plus scheduler-owned SLURM_/SPANK variables), without inheriting ambient
# user variables.
printf 'sbatch --parsable --export=%q %q\n' "$joined" "$SCRIPT"
