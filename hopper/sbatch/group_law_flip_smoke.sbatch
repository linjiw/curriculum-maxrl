#!/usr/bin/env bash
#SBATCH --job-name=group-law-flip-smoke
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=/scratch/%u/maxrl/tests/logs/%x_%j.out

# Engineering-only GPU smoke.  It deliberately uses legacy_v1, two SFT steps,
# and one RL update, so its outputs can never satisfy the evidence analyzer.
set -euo pipefail
umask 027
required=(GLF_BUNDLE_DIR GLF_SOURCE_MANIFEST_SHA256 GLF_ENV_DIR)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing required variable: $name" >&2; exit 2; }
done
[[ "$(sha256sum "$GLF_BUNDLE_DIR/SHA256SUMS" | awk '{print $1}')" == "$GLF_SOURCE_MANIFEST_SHA256" ]]
(cd "$GLF_BUNDLE_DIR" && sha256sum -c --strict SHA256SUMS >/dev/null)
PY="$GLF_ENV_DIR/bin/python"
SMOKE=$(mktemp -d "/scratch/$(id -un)/maxrl/tests/group-law-flip-smoke.XXXXXX")
cleanup() {
  if [[ -n "${SMOKE:-}" && "$SMOKE" == /scratch/*/maxrl/tests/group-law-flip-smoke.* ]]; then
    rm -rf -- "$SMOKE"
  fi
}
trap cleanup EXIT
cp -a "$GLF_BUNDLE_DIR" "$SMOKE/source"
TRAIN="$SMOKE/source/curriculum_maxrl/maze_gpu/train.py"
for teacher in group_law_plugin group_law_activity; do
  "$PY" "$TRAIN" --protocol legacy_v1 --teacher "$teacher" --estimator maxrl \
    --steps 1 --eval-every 1 --seed 991 --lr 1e-4 --rollouts 32 \
    --tasks-per-step 2 --sft-steps 2 --d-model 32 --n-layers 1 \
    --eval-tasks-per-level 1 --eval-samples 1 \
    --sft-ckpt smoke-sft.pt --out "$SMOKE/${teacher}.jsonl" \
    --telemetry-out "$SMOKE/${teacher}.telemetry.jsonl" \
    --save-ckpt "$SMOKE/${teacher}.pt"
  [[ -s "$SMOKE/${teacher}.jsonl" && -s "$SMOKE/${teacher}.telemetry.jsonl" \
     && -s "$SMOKE/${teacher}.pt" ]]
done
echo "GROUP_LAW_FLIP_ENGINEERING_SMOKE_COMPLETE"
