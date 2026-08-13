#!/bin/bash
#SBATCH --job-name=gate-dr
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=122G
#SBATCH --time=0-03:00:00
#SBATCH --output=/scratch/%u/curriculum-maxrl-runtime/logs/gatedr_%a_%j.out
#SBATCH --array=0-11

# GATE-DR preregistered dose-response (hopper/GATE_DR_PREREG.md).
# 12 tasks: arm index = ID/3 in {b1h,g0,g085,g070}, seed = ID%3 + 1.
# Do NOT submit before the prereg-freeze commit exists.
set -euo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES  # Slurm exports these alongside CUDA_VISIBLE_DEVICES; verl rejects the combo

ARMS=(b1h g0 g085 g070)
ARM="${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="$((SLURM_ARRAY_TASK_ID % 3 + 1))"
RUN_ID="gatedr_${ARM}_s${SEED}"

case "$ARM" in
  b1h)  HS=false; UG=false; GMP=0.5  ;;
  g0)   HS=true;  UG=false; GMP=0.5  ;;
  g085) HS=true;  UG=true;  GMP=0.85 ;;
  g070) HS=true;  UG=true;  GMP=0.70 ;;
esac

export RUNTIME_ROOT="/scratch/$USER/curriculum-maxrl-runtime"
export PYTHON_BIN="/scratch/$USER/envs/maxrl-train/bin/python"
export HF_HOME="/scratch/$USER/.hf"
export RAY_DIR="/scratch/$USER/ray/gd${SLURM_ARRAY_TASK_ID}_$$"
export RUN_ID
export OUTPUT_DIR="$RUNTIME_ROOT/checkpoints/$RUN_ID"

# Frozen B-stage settings (see prereg; deviations: VAL_ON_LAST_STEP, VAL_N, SAVE)
export STEPS=60 TRAIN_BATCH=8 N_ROLLOUTS=16 SEED LR=1e-5
export ROLLOUT_MICRO_BATCH=128 LOGPROB_MICRO_BATCH=8 PPO_MICRO_BATCH=4
export MAX_RESPONSE_LENGTH=128 FILTER_OVERLONG=false REWARD_MANAGER=dapo
export GRADIENT_CHECKPOINTING=false
export VAL_BEFORE_TRAIN=false TEST_FREQ=-1 VAL_ON_LAST_STEP=true VAL_N=16
export SAVE_FREQ=60 SAVE_CONTENTS="['model','optimizer','extra','hf_model']"
export HINDSIGHT_ENABLE="$HS" HINDSIGHT_MAX_GROUPS=8 HINDSIGHT_ONE_TARGET=false
export HINDSIGHT_UTILITY_GATE="$UG" HINDSIGHT_GATE_MAX_P="$GMP"
export HINDSIGHT_AUDIT_PATH="$OUTPUT_DIR/dose_accounting.jsonl"
export LIVE_REPLAY_ENABLE=false REPLAY_RESERVOIR_COLLECT=false
export ATTN_IMPLEMENTATION=sdpa  # flash-attn wheel needs GLIBC>=2.32; Hopper nodes are older

echo "GATE-DR task $SLURM_ARRAY_TASK_ID: arm=$ARM seed=$SEED gate_max_p=$GMP on $(hostname)"
bash "/scratch/$USER/curriculum-maxrl/hopper/countdown_hopper_gate.sh"
echo "GATE-DR $RUN_ID DONE"
