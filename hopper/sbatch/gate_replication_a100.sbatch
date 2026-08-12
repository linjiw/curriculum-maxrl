#!/bin/bash
#SBATCH --job-name=gate-repl
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=122G
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/%u/curriculum-maxrl-runtime/logs/gate_repl_s%a_%j.out
#SBATCH --array=1-3

# Corrected-decay saturation-gate replication: one Countdown training run per
# array index (seed = array index), moderate gate operating point, fixed code.
#
# GOVERNANCE — do not submit until BOTH are true:
#   1. A NEW preregistration for this study is frozen and committed (it must
#      name Hopper + this script's SHA-256 as the environment; the E2c prereg
#      does NOT cover it and E2c itself never runs here).
#   2. The runtime + clean-SFT checkpoint have been rsync'd to
#      /scratch/$USER/curriculum-maxrl-runtime and their SHA-256s verified
#      against E2C_CODE_MANIFEST.json.
set -euo pipefail
module load gnu10 2>/dev/null || true
module load miniconda3 2>/dev/null || module load miniconda 2>/dev/null || true
source activate maxrl

RUNTIME="/scratch/$USER/curriculum-maxrl-runtime"
SEED="${SLURM_ARRAY_TASK_ID}"

echo "gate replication seed $SEED on $(hostname); $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# The launcher is the lab's verl-based trainer with the gate flag enabled and
# the corrected decay implementation. Mirror the local launcher invocation,
# only substituting Hopper paths. Placeholder until the new prereg pins it:
cd "$RUNTIME/maxrl"
python -m verl.trainer.main_ppo \
  --config-path "$RUNTIME/maxrl/configs" \
  --config-name countdown_gate_moderate \
  trainer.seed="$SEED" \
  trainer.total_training_steps=60 \
  trainer.default_local_dir="$RUNTIME/checkpoints/gate_repl_s$SEED" \
  2>&1 | tee "$RUNTIME/logs/gate_repl_s${SEED}_train.log"

echo "DONE seed $SEED"
