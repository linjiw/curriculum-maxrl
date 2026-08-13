#!/bin/bash
#SBATCH --job-name=cd-train-smoke
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=122G
#SBATCH --time=0-01:00:00
#SBATCH --output=/scratch/%u/curriculum-maxrl-runtime/logs/cd_train_smoke_%j.out

# ENGINEERING VALIDATION — NOT EVIDENCE. One MaxRL training step of the real
# Countdown pipeline (verl fork, HF rollout, MaxRL advantage, hindsight off)
# on a Hopper A100, via the unchanged lab launcher with environment overrides.
# Pass criteria: step completes, validation runs, no NaN, checkpoint logic OK.
set -euo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES  # Slurm exports these alongside CUDA_VISIBLE_DEVICES; verl rejects the combo

export RUNTIME_ROOT="/scratch/$USER/curriculum-maxrl-runtime"
export PYTHON_BIN="/scratch/$USER/envs/maxrl-train/bin/python"
export RAY_DIR="/scratch/$USER/ray/smk$$"
export HF_HOME="/scratch/$USER/.hf"
export RUN_ID="hopper_train_smoke"
export STEPS=1 TRAIN_BATCH=8 VAL_N=4 SEED=1
export HINDSIGHT_ENABLE=false LIVE_REPLAY_ENABLE=false
export ATTN_IMPLEMENTATION=sdpa  # flash-attn wheel needs GLIBC>=2.32; Hopper nodes are older

bash "/scratch/$USER/curriculum-maxrl/hopper/countdown_hopper_gate.sh"
echo "TRAIN SMOKE PASS"
