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

export RUNTIME_ROOT="/scratch/$USER/curriculum-maxrl-runtime"
export PYTHON_BIN="/scratch/$USER/envs/maxrl-train/bin/python"
export RAY_DIR="/scratch/$USER/ray/smk$$"
export HF_HOME="/scratch/$USER/.hf"
export RUN_ID="hopper_train_smoke"
export STEPS=1 TRAIN_BATCH=8 VAL_N=4 SEED=1
export HINDSIGHT_ENABLE=false LIVE_REPLAY_ENABLE=false

bash "/scratch/$USER/curriculum-maxrl/verl_integration/countdown_rtx5090.sh"
echo "TRAIN SMOKE PASS"
