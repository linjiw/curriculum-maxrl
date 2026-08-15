#!/usr/bin/env bash
#SBATCH --job-name=maze-full-arm-smoke
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=0-06:00:00
#SBATCH --output=/scratch/%u/maxrl/tests/logs/%x_%j.out

# Full-cost, one-arm engineering profile on non-evidence seed 99. The result is
# never passed to the confirmatory analyzer or paper registry.
set -euo pipefail
umask 027
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

required=(MAZE_BUNDLE_DIR MAZE_SOURCE_MANIFEST_SHA256 MAZE_ENV_DIR MAZE_ENV_LOCK_SHA256)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing required variable: $name" >&2; exit 2; }
done
[[ "$MAZE_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$MAZE_ENV_LOCK_SHA256" =~ ^[0-9a-f]{64}$ ]]
RUN_USER=$(id -un)
SCRATCH_ROOT="/scratch/$RUN_USER"
case "$MAZE_BUNDLE_DIR" in "$SCRATCH_ROOT"/*) ;; *) exit 2;; esac
case "$MAZE_ENV_DIR" in "$SCRATCH_ROOT"/*) ;; *) exit 2;; esac
manifest="$MAZE_BUNDLE_DIR/SHA256SUMS"
[[ "$(sha256sum "$manifest" | awk '{print $1}')" == "$MAZE_SOURCE_MANIFEST_SHA256" ]]
(cd "$MAZE_BUNDLE_DIR" && sha256sum -c --strict SHA256SUMS >/dev/null)
bundle_lock="$MAZE_BUNDLE_DIR/hopper/requirements-maze-hopper.lock"
[[ "$(sha256sum "$bundle_lock" | awk '{print $1}')" == "$MAZE_ENV_LOCK_SHA256" ]]
[[ "$(awk 'NR == 1 {print $1}' "$MAZE_ENV_DIR/LOCK_SHA256")" == "$MAZE_ENV_LOCK_SHA256" ]]

SEED=99
PY="$MAZE_ENV_DIR/bin/python"
TRAIN="$MAZE_BUNDLE_DIR/curriculum_maxrl/maze_gpu/train.py"
result="$SCRATCH_ROOT/maxrl/tests/maze-full-arm/$SLURM_JOB_ID"
[[ ! -e "$result" && ! -L "$result" ]] || { echo "result exists: $result" >&2; exit 2; }
mkdir -p "$result"/{result,telemetry,warmstart,meta}
SFT="$result/warmstart/seed-${SEED}-sft.pt"

started=$(date +%s)
"$PY" "$TRAIN" --protocol maze_score_v2 --prepare-sft-only \
  --seed "$SEED" --sft-seed "$SEED" --sft-ckpt "$SFT" \
  --sft-steps 600 --d-model 128 --n-layers 6

out="$result/result/mazescore_un_s${SEED}.jsonl"
telemetry="$result/telemetry/mazescore_un_s${SEED}.telemetry.jsonl"
"$PY" "$TRAIN" \
  --protocol maze_score_v2 --campaign "engineering-full-arm-$SLURM_JOB_ID" \
  --source-manifest "$MAZE_SOURCE_MANIFEST_SHA256" --arm un \
  --teacher frontier_un --estimator maxrl \
  --steps 250 --eval-every 25 --seed "$SEED" \
  --sft-seed "$SEED" --rl-seed "$SEED" \
  --eval-seed "$((202608130 + SEED))" \
  --eval-sample-seed "$((302608130 + SEED))" \
  --lr 1e-4 --rollouts 32 --tasks-per-step 8 --sft-steps 600 \
  --d-model 128 --n-layers 6 --teacher-power 1 \
  --eval-tasks-per-level 32 --eval-samples 8 \
  --sft-ckpt "$SFT" --out "$out" --telemetry-out "$telemetry"
finished=$(date +%s)

# Schema-only validation deliberately avoids calculating or printing endpoints.
MAZE_SMOKE_JSONL="$out" "$PY" - <<'PY'
import json
import os

rows = [json.loads(line) for line in open(os.environ["MAZE_SMOKE_JSONL"], encoding="utf-8")]
configs = [row for row in rows if row.get("record_type") == "config"]
evals = [row for row in rows if row.get("record_type") == "evaluation"]
assert len(configs) == 1
assert configs[0]["seed"] == 99 and configs[0]["effective_exponent"] == 32
assert [row["completed_updates"] for row in evals] == [0, *range(25, 251, 25)]
assert evals[-1].get("final") is True
PY

{
  printf 'key\tvalue\n'
  printf 'job_id\t%s\n' "$SLURM_JOB_ID"
  printf 'seed\t%s\n' "$SEED"
  printf 'arm\tun\n'
  printf 'elapsed_seconds\t%s\n' "$((finished - started))"
  printf 'source_manifest_sha256\t%s\n' "$MAZE_SOURCE_MANIFEST_SHA256"
  printf 'environment_lock_sha256\t%s\n' "$MAZE_ENV_LOCK_SHA256"
  printf 'sft_checkpoint_sha256\t%s\n' "$(sha256sum "$SFT" | awk '{print $1}')"
  printf 'gpu\t%s\n' "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
} > "$result/meta/profile.tsv"
(
  cd "$result"
  find . -type f ! -name SHA256SUMS ! -name COMPLETE -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c --strict SHA256SUMS >/dev/null
)
printf 'complete\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$result/COMPLETE"
echo "MAZE_FULL_ARM_SMOKE_PASS $result"
