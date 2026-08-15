#!/usr/bin/env bash
#SBATCH --job-name=maze-score-v2
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=0-12:00:00
#SBATCH --output=/scratch/%u/maxrl/maze_score/logs/%x_%A_%a.out
#SBATCH --array=20-49%5

# Evidence-bearing MAZE-SCORE v2 campaign. This file is deliberately unusable
# without a clean evidence bundle, exact environment lock, and explicit
# campaign/attempt IDs supplied through sbatch --export.
set -euo pipefail
umask 027
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

required=(
  MAZE_BUNDLE_DIR
  MAZE_SOURCE_MANIFEST_SHA256
  MAZE_ENV_DIR
  MAZE_ENV_LOCK_SHA256
  MAZE_ENV_FREEZE_SHA256
  MAZE_ENV_JSON_SHA256
  MAZE_CAMPAIGN_ID
  MAZE_ATTEMPT_ID
  SLURM_ARRAY_TASK_ID
  SLURM_ARRAY_JOB_ID
  SLURM_JOB_ID
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing required variable: $name" >&2; exit 2; }
done
[[ "$MAZE_CAMPAIGN_ID" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "unsafe MAZE_CAMPAIGN_ID" >&2; exit 2; }
[[ "$MAZE_ATTEMPT_ID" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "unsafe MAZE_ATTEMPT_ID" >&2; exit 2; }
[[ "$MAZE_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "invalid source manifest hash" >&2; exit 2; }
[[ "$MAZE_ENV_LOCK_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "invalid environment lock hash" >&2; exit 2; }
[[ "$MAZE_ENV_FREEZE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "invalid environment freeze hash" >&2; exit 2; }
[[ "$MAZE_ENV_JSON_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "invalid environment JSON hash" >&2; exit 2; }

RUN_USER=$(id -un)
SCRATCH_ROOT="/scratch/$RUN_USER"
case "$MAZE_BUNDLE_DIR" in "$SCRATCH_ROOT"/*) ;; *) echo "bundle outside scratch" >&2; exit 2;; esac
case "$MAZE_ENV_DIR" in "$SCRATCH_ROOT"/*) ;; *) echo "environment outside scratch" >&2; exit 2;; esac

SOURCE_MANIFEST="$MAZE_BUNDLE_DIR/SHA256SUMS"
SOURCE_STATE="$MAZE_BUNDLE_DIR/SOURCE_STATE.json"
BUNDLE_ENV_LOCK="$MAZE_BUNDLE_DIR/hopper/requirements-maze-hopper.lock"
PREREG="$MAZE_BUNDLE_DIR/hopper/MAZE_SCORE_PREREG.md"
PY="$MAZE_ENV_DIR/bin/python"
TRAIN="$MAZE_BUNDLE_DIR/curriculum_maxrl/maze_gpu/train.py"
ANALYZER_ROOT="$MAZE_BUNDLE_DIR"
[[ -f "$SOURCE_MANIFEST" && -f "$SOURCE_STATE" && -f "$BUNDLE_ENV_LOCK" \
   && -f "$PREREG" && -x "$PY" && -f "$TRAIN" ]] \
  || { echo "bundle/environment preflight missing files" >&2; exit 2; }
actual_manifest=$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')
[[ "$actual_manifest" == "$MAZE_SOURCE_MANIFEST_SHA256" ]] \
  || { echo "source manifest hash mismatch" >&2; exit 2; }
grep -q '"mode": "evidence"' "$SOURCE_STATE"
grep -q '"worktree_dirty": false' "$SOURCE_STATE"
grep -q '^\*\*Status:\*\* FROZEN' "$PREREG"
(cd "$MAZE_BUNDLE_DIR" && sha256sum -c --strict SHA256SUMS >/dev/null)

actual_lock=$(awk 'NR == 1 {print $1}' "$MAZE_ENV_DIR/LOCK_SHA256")
[[ "$actual_lock" == "$MAZE_ENV_LOCK_SHA256" ]] \
  || { echo "environment lock hash mismatch" >&2; exit 2; }
actual_bundle_lock=$(sha256sum "$BUNDLE_ENV_LOCK" | awk '{print $1}')
[[ "$actual_bundle_lock" == "$MAZE_ENV_LOCK_SHA256" ]] \
  || { echo "source/environment lock mismatch" >&2; exit 2; }
actual_freeze=$(sha256sum "$MAZE_ENV_DIR/ENVIRONMENT.freeze" | awk '{print $1}')
[[ "$actual_freeze" == "$MAZE_ENV_FREEZE_SHA256" ]] \
  || { echo "environment freeze hash mismatch" >&2; exit 2; }
actual_env_json=$(sha256sum "$MAZE_ENV_DIR/ENVIRONMENT.json" | awk '{print $1}')
[[ "$actual_env_json" == "$MAZE_ENV_JSON_SHA256" ]] \
  || { echo "environment JSON hash mismatch" >&2; exit 2; }
"$PY" -m pip check >/dev/null

SEED=$SLURM_ARRAY_TASK_ID
[[ "$SEED" =~ ^(2[0-9]|3[0-9]|4[0-9])$ ]] \
  || { echo "evidence seed must be in 20..49" >&2; exit 2; }
CAMPAIGN_ROOT="$SCRATCH_ROOT/maxrl/maze_score/campaigns/$MAZE_CAMPAIGN_ID"
BLOCK="$CAMPAIGN_ROOT/attempts/$MAZE_ATTEMPT_ID/seed-$SEED"
if [[ -e "$BLOCK" || -L "$BLOCK" ]]; then
  echo "refusing to overwrite attempt block: $BLOCK" >&2
  exit 2
fi
mkdir -p "$BLOCK"/{results,telemetry,warmstarts,meta,tmp}

cp "$SOURCE_MANIFEST" "$BLOCK/meta/SOURCE_SHA256SUMS"
cp "$SOURCE_STATE" "$BLOCK/meta/SOURCE_STATE.json"
cp "$MAZE_ENV_DIR/LOCK_SHA256" "$BLOCK/meta/ENV_LOCK_SHA256"
cp "$MAZE_ENV_DIR/ENVIRONMENT.json" "$BLOCK/meta/ENVIRONMENT.json"
cp "$MAZE_ENV_DIR/ENVIRONMENT.freeze" "$BLOCK/meta/ENVIRONMENT.freeze"
cp "$MAZE_BUNDLE_DIR/hopper/MAZE_SCORE_PREREG.md" "$BLOCK/meta/MAZE_SCORE_PREREG.md"

{
  printf 'key\tvalue\n'
  printf 'campaign\t%s\n' "$MAZE_CAMPAIGN_ID"
  printf 'attempt\t%s\n' "$MAZE_ATTEMPT_ID"
  printf 'seed\t%s\n' "$SEED"
  printf 'job_id\t%s\n' "$SLURM_JOB_ID"
  printf 'array_job_id\t%s\n' "$SLURM_ARRAY_JOB_ID"
  printf 'hostname\t%s\n' "$(hostname)"
  printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'source_manifest_sha256\t%s\n' "$MAZE_SOURCE_MANIFEST_SHA256"
  printf 'environment_lock_sha256\t%s\n' "$MAZE_ENV_LOCK_SHA256"
  printf 'environment_freeze_sha256\t%s\n' "$MAZE_ENV_FREEZE_SHA256"
  printf 'environment_json_sha256\t%s\n' "$MAZE_ENV_JSON_SHA256"
  printf 'cuda_visible_devices\t%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'gpu\t%s\n' "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
} > "$BLOCK/meta/job.tsv"

SFT="$BLOCK/warmstarts/seed-${SEED}-sft.pt"
"$PY" "$TRAIN" \
  --protocol maze_score_v2 --prepare-sft-only \
  --seed "$SEED" --sft-seed "$SEED" --sft-ckpt "$SFT" \
  --sft-steps 600 --d-model 128 --n-layers 6
SFT_SHA=$(sha256sum "$SFT" | awk '{print $1}')
printf '%s  %s\n' "$SFT_SHA" "$(basename "$SFT")" \
  > "$BLOCK/warmstarts/SHA256SUMS"

# Counterbalance process order after warmstart preparation. This removes the
# old systematic "un always first" position effect while preserving pairing.
case $((SEED % 3)) in
  0) arms=(un learn unif) ;;
  1) arms=(learn unif un) ;;
  2) arms=(unif un learn) ;;
esac
printf '%s\n' "${arms[*]}" > "$BLOCK/meta/arm_order.txt"

for arm in "${arms[@]}"; do
  case "$arm" in
    un) teacher=frontier_un ;;
    learn) teacher=learnability ;;
    unif) teacher=uniform ;;
  esac
  final="$BLOCK/results/mazescore_${arm}_s${SEED}.jsonl"
  mkdir "$BLOCK/tmp/$arm"
  partial="$BLOCK/tmp/$arm/mazescore_${arm}_s${SEED}.jsonl"
  telemetry_final="$BLOCK/telemetry/mazescore_${arm}_s${SEED}.telemetry.jsonl"
  telemetry_partial="$BLOCK/tmp/$arm/mazescore_${arm}_s${SEED}.telemetry.jsonl"

  "$PY" "$TRAIN" \
    --protocol maze_score_v2 --campaign "$MAZE_CAMPAIGN_ID" \
    --source-manifest "$MAZE_SOURCE_MANIFEST_SHA256" --arm "$arm" \
    --teacher "$teacher" --estimator maxrl \
    --steps 250 --eval-every 25 --seed "$SEED" \
    --sft-seed "$SEED" --rl-seed "$SEED" \
    --eval-seed "$((202608130 + SEED))" \
    --eval-sample-seed "$((302608130 + SEED))" \
    --lr 1e-4 --rollouts 32 --tasks-per-step 8 --sft-steps 600 \
    --d-model 128 --n-layers 6 --teacher-power 1 \
    --eval-tasks-per-level 32 --eval-samples 8 \
    --sft-ckpt "$SFT" --out "$partial" --telemetry-out "$telemetry_partial"

  PYTHONPATH="$ANALYZER_ROOT" "$PY" - "$partial" <<'PY'
import sys
from curriculum_maxrl.maze_score.analyze_maze_score import load_run
load_run(sys.argv[1])
PY
  mv "$partial" "$final"
  mv "$telemetry_partial" "$telemetry_final"
done

printf 'finished_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "$BLOCK/meta/job.tsv"
(
  cd "$BLOCK"
  find . -type f ! -name SHA256SUMS ! -name COMPLETE -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c --strict SHA256SUMS >/dev/null
)
printf 'complete\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BLOCK/COMPLETE"
echo "MAZE_SCORE_BLOCK_COMPLETE campaign=$MAZE_CAMPAIGN_ID attempt=$MAZE_ATTEMPT_ID seed=$SEED"
