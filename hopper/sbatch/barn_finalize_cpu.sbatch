#!/usr/bin/env bash
#SBATCH --job-name=barn-finalize-safe-log
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/maxrl/barn/logs/%x_%j.out

# CPU-only, metadata-only postprocessing for one exact four-cell BARN campaign.
set -euo pipefail
umask 077
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=""

required=(
  BARN_CAMPAIGN_ID
  BARN_SOURCE_BUNDLE_DIR
  BARN_SOURCE_SHA256
  BARN_FINALIZED_LEDGER_SHA256
  BARN_FINALIZE_WRAPPER_SHA256
  BARN_FINALIZE_SBATCH_SHA256
  BARN_SCRATCH_ROOT
  SLURM_JOB_ID
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || {
    printf 'missing required variable: %s\n' "$name" >&2
    exit 2
  }
done
[[ "$BARN_CAMPAIGN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || {
  echo "unsafe campaign ID" >&2; exit 2;
}
for name in BARN_SOURCE_SHA256 BARN_FINALIZED_LEDGER_SHA256 \
            BARN_FINALIZE_WRAPPER_SHA256 BARN_FINALIZE_SBATCH_SHA256; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'invalid SHA-256 variable: %s\n' "$name" >&2
    exit 2
  }
done
[[ "$BARN_SCRATCH_ROOT" =~ ^/scratch/[A-Za-z0-9._-]+$ ]] || {
  echo "unsafe BARN_SCRATCH_ROOT" >&2; exit 2;
}
readonly EXPECTED_SOURCE="$BARN_SCRATCH_ROOT/maxrl/bundles/barn_source/${BARN_SOURCE_SHA256:0:20}"
[[ "$BARN_SOURCE_BUNDLE_DIR" == "$EXPECTED_SOURCE" ]] || {
  echo "source bundle path is not content addressed" >&2; exit 2;
}
readonly BUNDLED_SBATCH="$BARN_SOURCE_BUNDLE_DIR/hopper/sbatch/barn_finalize_cpu.sbatch"
readonly BUNDLED_WRAPPER="$BARN_SOURCE_BUNDLE_DIR/hopper/finalize_barn_campaign.sh"
for path in "$BUNDLED_SBATCH" "$BUNDLED_WRAPPER"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "bundled postprocessing input missing or symbolic" >&2; exit 2;
  }
done
[[ "$(sha256sum -- "$0" | awk '{print $1}')" == \
   "$BARN_FINALIZE_SBATCH_SHA256" ]] || {
  echo "executing postprocess sbatch differs from submitted binding" >&2
  exit 2
}
[[ "$(sha256sum -- "$BUNDLED_SBATCH" | awk '{print $1}')" == \
   "$BARN_FINALIZE_SBATCH_SHA256" ]] || {
  echo "executing postprocess sbatch differs from source bundle" >&2
  exit 2
}
[[ "$(sha256sum -- "$BUNDLED_WRAPPER" | awk '{print $1}')" == \
   "$BARN_FINALIZE_WRAPPER_SHA256" ]] || {
  echo "campaign wrapper differs from source binding" >&2
  exit 2
}
readonly APPTAINER_BIN="/opt/sw/other/apps/apptainer/1.4.1/bin/apptainer"
[[ -x "$APPTAINER_BIN" && ! -L "$APPTAINER_BIN" ]] || {
  echo "pinned Apptainer executable unavailable" >&2; exit 2;
}

# The bundled wrapper captures all selector/merger/analyzer output and emits
# only the sealed package path/hash/count sentinels consumed by the local side.
bash "$BUNDLED_WRAPPER" --remote \
  "$BARN_CAMPAIGN_ID" "$BARN_SOURCE_BUNDLE_DIR" "$BARN_SOURCE_SHA256" \
  "$BARN_FINALIZED_LEDGER_SHA256" "$BARN_FINALIZE_WRAPPER_SHA256" \
  "$BARN_SCRATCH_ROOT" "$APPTAINER_BIN"
printf 'BARN_FINALIZE_JOB_COMPLETE\tjob_id=%s\tcampaign=%s\tcpu_only=true\n' \
  "$SLURM_JOB_ID" "$BARN_CAMPAIGN_ID"
