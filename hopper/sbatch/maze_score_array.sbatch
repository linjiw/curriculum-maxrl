#!/bin/bash
#SBATCH --job-name=maze-score
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=0-10:00:00
#SBATCH --output=/scratch/%u/maze_score/logs/mazescore_s%a_%j.out
#SBATCH --array=20-29

# MAZE-SCORE: neural-scale deployed-N score test (hopper/MAZE_SCORE_PREREG.md).
# One array task = one independent seed block, running its three arms
# sequentially so the block's SFT warmstart is created once, without races.
set -euo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

SEED="${SLURM_ARRAY_TASK_ID}"
WORK="/scratch/$USER/maze_score"
PY="/scratch/$USER/envs/maxrl/bin/python"
SRC="/scratch/$USER/curriculum-maxrl/curriculum_maxrl/maze_gpu"
mkdir -p "$WORK/logs"
cd "$WORK"
cp -n "$SRC"/*.py . 2>/dev/null || true

echo "MAZE-SCORE block $SEED on $(hostname); $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Order matters only for warmstart creation; 'un' first creates seed<S>_sft_warmstart.pt.
for pair in "un:frontier_un" "learn:learnability" "unif:uniform"; do
  arm="${pair%%:*}"; teacher="${pair##*:}"
  out="mazescore_${arm}_s${SEED}.jsonl"
  if [ -f "$out" ] && grep -q '"final"' "$out"; then
    echo "skip $out (complete)"; continue
  fi
  echo "=== block $SEED arm $arm (teacher=$teacher) $(date -u +%H:%M:%SZ) ==="
  "$PY" train.py --teacher "$teacher" --estimator maxrl \
    --steps 250 --eval-every 25 --seed "$SEED" --lr 1e-4 \
    --rollouts 32 --tasks-per-step 8 --sft-steps 600 \
    --d-model 128 --n-layers 6 --out "$out"
done

echo "MAZE-SCORE block $SEED DONE"
