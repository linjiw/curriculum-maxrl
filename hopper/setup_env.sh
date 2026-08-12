#!/bin/bash
# One-time environment setup on hopper.orc.gmu.edu (run on the login node).
# Creates the conda env for the Countdown/MaxRL GPU stack and scratch layout.
# Safe to re-run; every step is idempotent.
set -euo pipefail

echo "== Hopper env setup for curriculum-maxrl =="

SCRATCH="/scratch/$USER"
mkdir -p "$SCRATCH/curriculum-maxrl-runtime"/{models,data,checkpoints,logs}

module load gnu10 2>/dev/null || true
if ! command -v conda >/dev/null 2>&1; then
  # ORC provides miniconda via modules on Hopper
  ml spider miniconda 2>/dev/null | head -5 || true
  module load miniconda3 2>/dev/null || module load miniconda 2>/dev/null || {
    echo "No conda module found — install miniforge to scratch:"
    echo "  wget -O $SCRATCH/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    echo "  bash $SCRATCH/miniforge.sh -b -p $SCRATCH/miniforge && $SCRATCH/miniforge/bin/conda init bash"
    exit 1
  }
fi

ENV_NAME=maxrl
if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -y -n $ENV_NAME python=3.10
fi
# shellcheck disable=SC1091
source activate $ENV_NAME 2>/dev/null || conda activate $ENV_NAME

# Torch first (CUDA 12.x wheels), then the inference/training stack.
# Versions chosen to match the lab runtime; adjust only with a matching
# fingerprint note in whatever prereg governs the run.
pip install --no-cache-dir "torch==2.6.*" --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir "vllm>=0.8,<0.9" ray transformers accelerate datasets safetensors

echo
echo "== Done. Next steps =="
echo "1. rsync the MaxRL runtime + verl_integration patches from the lab machine:"
echo "   rsync -av /data/robotixx/curriculum-maxrl-runtime/maxrl $USER@hopper.orc.gmu.edu:$SCRATCH/curriculum-maxrl-runtime/"
echo "   rsync -av /data/robotixx/curriculum-maxrl-runtime/models $USER@hopper.orc.gmu.edu:$SCRATCH/curriculum-maxrl-runtime/"
echo "2. Submit hopper/sbatch/mig_smoke.sbatch and check the log before any real job."
echo "NOTE: E2c itself never runs here — frozen local protocol. See HOPPER_SETUP.md."
