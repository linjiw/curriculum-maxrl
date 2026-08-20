#!/usr/bin/env bash
#SBATCH --job-name=group-law-flip-v1
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=0-12:00:00
#SBATCH --output=/scratch/%u/maxrl/group_law_flip/logs/%x_%A_%a.out
#SBATCH --array=3001-3024%5

# Evidence-bearing GROUP-LAW-FLIP v1 campaign.  Submit two chunks because the
# Hopper account permits at most 40 queued jobs:
#   --array=3001-3024%5
#   --array=3025-3048%5
set -euo pipefail
umask 027
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

required=(
  GLF_BUNDLE_DIR
  GLF_SOURCE_MANIFEST_SHA256
  GLF_ENV_DIR
  GLF_ENV_LOCK_SHA256
  GLF_ENV_FREEZE_SHA256
  GLF_ENV_JSON_SHA256
  GLF_CAMPAIGN_ID
  GLF_ATTEMPT_ID
  SLURM_ARRAY_TASK_ID
  SLURM_ARRAY_JOB_ID
  SLURM_JOB_ID
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing required variable: $name" >&2; exit 2; }
done
[[ "$GLF_CAMPAIGN_ID" == "group-law-flip-v1-20260820-001" ]] \
  || { echo "wrong frozen campaign ID" >&2; exit 2; }
[[ "$GLF_ATTEMPT_ID" == "attempt-001" ]] \
  || { echo "wrong frozen attempt ID" >&2; exit 2; }
for value in "$GLF_SOURCE_MANIFEST_SHA256" "$GLF_ENV_LOCK_SHA256" \
             "$GLF_ENV_FREEZE_SHA256" "$GLF_ENV_JSON_SHA256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid SHA-256 value" >&2; exit 2; }
done

RUN_USER=$(id -un)
SCRATCH_ROOT="/scratch/$RUN_USER"
case "$GLF_BUNDLE_DIR" in "$SCRATCH_ROOT"/*) ;; *) echo "bundle outside scratch" >&2; exit 2;; esac
case "$GLF_ENV_DIR" in "$SCRATCH_ROOT"/*) ;; *) echo "environment outside scratch" >&2; exit 2;; esac

SOURCE_MANIFEST="$GLF_BUNDLE_DIR/SHA256SUMS"
SOURCE_STATE="$GLF_BUNDLE_DIR/SOURCE_STATE.json"
BUNDLE_ENV_LOCK="$GLF_BUNDLE_DIR/hopper/requirements-maze-hopper.lock"
PREREG="$GLF_BUNDLE_DIR/granularity_flip/GROUP_LAW_FLIP_PREREG.md"
PY="$GLF_ENV_DIR/bin/python"
TRAIN="$GLF_BUNDLE_DIR/curriculum_maxrl/maze_gpu/train.py"
[[ -f "$SOURCE_MANIFEST" && -f "$SOURCE_STATE" && -f "$BUNDLE_ENV_LOCK" \
   && -f "$PREREG" && -x "$PY" && -f "$TRAIN" ]] \
  || { echo "bundle/environment preflight missing files" >&2; exit 2; }
[[ "$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')" == "$GLF_SOURCE_MANIFEST_SHA256" ]] \
  || { echo "source manifest hash mismatch" >&2; exit 2; }
grep -q '"mode": "evidence"' "$SOURCE_STATE"
grep -q '"worktree_dirty": false' "$SOURCE_STATE"
grep -q '^\*\*Status:\*\* FROZEN' "$PREREG"
(cd "$GLF_BUNDLE_DIR" && sha256sum -c --strict SHA256SUMS >/dev/null)

[[ "$(awk 'NR == 1 {print $1}' "$GLF_ENV_DIR/LOCK_SHA256")" == "$GLF_ENV_LOCK_SHA256" ]] \
  || { echo "environment lock hash mismatch" >&2; exit 2; }
[[ "$(sha256sum "$BUNDLE_ENV_LOCK" | awk '{print $1}')" == "$GLF_ENV_LOCK_SHA256" ]] \
  || { echo "source/environment lock mismatch" >&2; exit 2; }
[[ "$(sha256sum "$GLF_ENV_DIR/ENVIRONMENT.freeze" | awk '{print $1}')" == "$GLF_ENV_FREEZE_SHA256" ]] \
  || { echo "environment freeze hash mismatch" >&2; exit 2; }
[[ "$(sha256sum "$GLF_ENV_DIR/ENVIRONMENT.json" | awk '{print $1}')" == "$GLF_ENV_JSON_SHA256" ]] \
  || { echo "environment JSON hash mismatch" >&2; exit 2; }
"$PY" -m pip check >/dev/null

SEED=$SLURM_ARRAY_TASK_ID
[[ "$SEED" =~ ^30(0[1-9]|[1-3][0-9]|4[0-8])$ ]] \
  || { echo "evidence seed must be in 3001..3048" >&2; exit 2; }
CAMPAIGN_ROOT="$SCRATCH_ROOT/maxrl/group_law_flip/campaigns/$GLF_CAMPAIGN_ID"
ATTEMPT_ROOT="$CAMPAIGN_ROOT/attempts/$GLF_ATTEMPT_ID"
FINAL_BLOCK="$ATTEMPT_ROOT/seed-$SEED"
INCOMPLETE_ROOT="$CAMPAIGN_ROOT/incomplete/$GLF_ATTEMPT_ID"
BLOCK="$INCOMPLETE_ROOT/seed-${SEED}.job-${SLURM_JOB_ID}"
if [[ -e "$FINAL_BLOCK" || -L "$FINAL_BLOCK" ]]; then
  echo "refusing to overwrite completed attempt block: $FINAL_BLOCK" >&2
  exit 2
fi
if [[ -e "$BLOCK" || -L "$BLOCK" ]]; then
  echo "refusing to overwrite incomplete work block: $BLOCK" >&2
  exit 2
fi
mkdir -p "$BLOCK"/{results,telemetry,warmstarts,checkpoints,meta,tmp}

cp "$SOURCE_MANIFEST" "$BLOCK/meta/SOURCE_SHA256SUMS"
cp "$SOURCE_STATE" "$BLOCK/meta/SOURCE_STATE.json"
cp "$GLF_ENV_DIR/LOCK_SHA256" "$BLOCK/meta/ENV_LOCK_SHA256"
cp "$GLF_ENV_DIR/ENVIRONMENT.json" "$BLOCK/meta/ENVIRONMENT.json"
cp "$GLF_ENV_DIR/ENVIRONMENT.freeze" "$BLOCK/meta/ENVIRONMENT.freeze"
cp "$PREREG" "$BLOCK/meta/GROUP_LAW_FLIP_PREREG.md"

{
  printf 'key\tvalue\n'
  printf 'campaign\t%s\n' "$GLF_CAMPAIGN_ID"
  printf 'attempt\t%s\n' "$GLF_ATTEMPT_ID"
  printf 'seed\t%s\n' "$SEED"
  printf 'job_id\t%s\n' "$SLURM_JOB_ID"
  printf 'array_job_id\t%s\n' "$SLURM_ARRAY_JOB_ID"
  printf 'hostname\t%s\n' "$(hostname)"
  printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'source_manifest_sha256\t%s\n' "$GLF_SOURCE_MANIFEST_SHA256"
  printf 'environment_lock_sha256\t%s\n' "$GLF_ENV_LOCK_SHA256"
  printf 'environment_freeze_sha256\t%s\n' "$GLF_ENV_FREEZE_SHA256"
  printf 'environment_json_sha256\t%s\n' "$GLF_ENV_JSON_SHA256"
  printf 'cuda_visible_devices\t%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'gpu\t%s\n' "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
} > "$BLOCK/meta/job.tsv"

SFT="$BLOCK/warmstarts/seed-${SEED}-sft.pt"
"$PY" "$TRAIN" \
  --protocol group_law_flip_v1 --prepare-sft-only \
  --seed "$SEED" --sft-seed "$SEED" --sft-ckpt "$SFT" \
  --sft-steps 600 --d-model 128 --n-layers 6
SFT_SHA=$(sha256sum "$SFT" | awk '{print $1}')

# Counterbalance process order exactly 24/24 across the 48 seed blocks.
if (( SEED % 2 == 0 )); then
  arms=(plugin grouplaw)
else
  arms=(grouplaw plugin)
fi
printf '%s\n' "${arms[*]}" > "$BLOCK/meta/arm_order.txt"

for arm in "${arms[@]}"; do
  case "$arm" in
    plugin) teacher=group_law_plugin ;;
    grouplaw) teacher=group_law_activity ;;
    *) echo "unknown arm $arm" >&2; exit 2 ;;
  esac
  temp_dir="$BLOCK/tmp/$arm"
  mkdir "$temp_dir"
  result_partial="$temp_dir/groupflip_${arm}_s${SEED}.jsonl"
  telemetry_partial="$temp_dir/groupflip_${arm}_s${SEED}.telemetry.jsonl"
  checkpoint_partial="$temp_dir/groupflip_${arm}_s${SEED}.pt"
  result_final="$BLOCK/results/groupflip_${arm}_s${SEED}.jsonl"
  telemetry_final="$BLOCK/telemetry/groupflip_${arm}_s${SEED}.telemetry.jsonl"
  checkpoint_final="$BLOCK/checkpoints/groupflip_${arm}_s${SEED}.pt"

  "$PY" "$TRAIN" \
    --protocol group_law_flip_v1 --campaign "$GLF_CAMPAIGN_ID" \
    --source-manifest "$GLF_SOURCE_MANIFEST_SHA256" --arm "$arm" \
    --teacher "$teacher" --estimator maxrl \
    --steps 250 --eval-every 25 --seed "$SEED" \
    --sft-seed "$SEED" --rl-seed "$SEED" \
    --eval-seed "$((202608130 + SEED))" \
    --eval-sample-seed "$((302608130 + SEED))" \
    --lr 1e-4 --rollouts 32 --tasks-per-step 8 --sft-steps 600 \
    --d-model 128 --n-layers 6 --teacher-power 1 \
    --eval-tasks-per-level 32 --eval-samples 8 \
    --sft-ckpt "$SFT" --out "$result_partial" \
    --telemetry-out "$telemetry_partial" --save-ckpt "$checkpoint_partial"

  PYTHONPATH="$GLF_BUNDLE_DIR" "$PY" - "$result_partial" "$telemetry_partial" <<'PY'
import sys
from curriculum_maxrl.group_law_flip.analyze_group_law_flip import load_run, load_telemetry
load_run(sys.argv[1])
load_telemetry(sys.argv[2])
PY
  [[ -s "$checkpoint_partial" ]] || { echo "missing final checkpoint" >&2; exit 2; }
  mv "$result_partial" "$result_final"
  mv "$telemetry_partial" "$telemetry_final"
  mv "$checkpoint_partial" "$checkpoint_final"
  rmdir "$temp_dir"

  RESULT_SHA=$(sha256sum "$result_final" | awk '{print $1}')
  TELEMETRY_SHA=$(sha256sum "$telemetry_final" | awk '{print $1}')
  CHECKPOINT_SHA=$(sha256sum "$checkpoint_final" | awk '{print $1}')
  PYTHONPATH="$GLF_BUNDLE_DIR" "$PY" - \
    "$BLOCK/meta/${arm}.DONE.json" "$arm" "$SEED" "$GLF_CAMPAIGN_ID" \
    "$GLF_ATTEMPT_ID" "$GLF_SOURCE_MANIFEST_SHA256" "$SFT_SHA" \
    "$RESULT_SHA" "$TELEMETRY_SHA" "$CHECKPOINT_SHA" <<'PY'
import json
import sys
from pathlib import Path

(output, arm, seed, campaign, attempt, source, warm,
 result_sha, telemetry_sha, checkpoint_sha) = sys.argv[1:]
seed = int(seed)
payload = {
    "schema": "curriculum-maxrl/group-law-flip/arm-receipt/v1",
    "protocol": "group_law_flip_v1",
    "campaign": campaign,
    "attempt": attempt,
    "arm": arm,
    "seed": seed,
    "completed_updates": 250,
    "source_manifest_sha256": source,
    "sft_checkpoint_sha256": warm,
    "paths": {
        "result": f"results/groupflip_{arm}_s{seed}.jsonl",
        "telemetry": f"telemetry/groupflip_{arm}_s{seed}.telemetry.jsonl",
        "checkpoint": f"checkpoints/groupflip_{arm}_s{seed}.pt",
    },
    "sha256": {
        "result": result_sha,
        "telemetry": telemetry_sha,
        "checkpoint": checkpoint_sha,
    },
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
done

rmdir "$BLOCK/tmp"
printf 'finished_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$BLOCK/meta/job.tsv"
(
  cd "$BLOCK"
  find . -type f ! -name SHA256SUMS ! -name COMPLETE -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c --strict SHA256SUMS >/dev/null
)
"$PY" - "$BLOCK/COMPLETE" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "schema": "curriculum-maxrl/group-law-flip/block-complete/v1",
    "protocol": "group_law_flip_v1",
    "campaign": "group-law-flip-v1-20260820-001",
    "attempt": "attempt-001",
    "seed": int(sys.argv[2]),
    "completed_arms": ["plugin", "grouplaw"],
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
mkdir -p "$ATTEMPT_ROOT"
mv -T "$BLOCK" "$FINAL_BLOCK"
echo "GROUP_LAW_FLIP_BLOCK_COMPLETE campaign=$GLF_CAMPAIGN_ID attempt=$GLF_ATTEMPT_ID seed=$SEED"
