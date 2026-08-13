#!/bin/bash
#SBATCH --job-name=gate-dr-eval
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=122G
#SBATCH --time=0-02:30:00
#SBATCH --output=/scratch/%u/curriculum-maxrl-runtime/logs/gate_dr_eval_%j.out

# Sealed GATE-DR endpoint evaluation (prereg amendment 2026-08-13b).
set -euo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export HF_HOME="/scratch/$USER/.hf"
/scratch/$USER/envs/maxrl/bin/python /scratch/$USER/sbatch/gate_dr_eval.py
