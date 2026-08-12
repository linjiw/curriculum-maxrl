#!/bin/bash
# One-time environment setup on hopper.orc.gmu.edu (run on the login node).
# Matches the deployment performed 2026-08-12 for user lwang44.
# Key facts discovered at deploy time:
#   - personal miniconda exists at ~/miniconda3 (no module needed)
#   - home quota is ~95% full -> env, package caches, and HF cache all live
#     under /scratch/$USER (scratch: no quota, purged >90-day files monthly,
#     NOT backed up — treat it as rebuildable)
#   - partitions (live sinfo): gpuq 5-00:00, normal/contrib 7-00:00,
#     contrib-H100 and contrib-B200 are separate dedicated partitions
set -euo pipefail

SCRATCH="/scratch/$USER"
export CONDA_PKGS_DIRS="$SCRATCH/.conda_pkgs"
export PIP_CACHE_DIR="$SCRATCH/.pipcache"

mkdir -p "$SCRATCH"/{envs,sbatch,.conda_pkgs,.pipcache,.hf} \
         "$SCRATCH/curriculum-maxrl-runtime"/{models,data,checkpoints,logs}

CONDA="$HOME/miniconda3/bin/conda"
ENVP="$SCRATCH/envs/maxrl"

[ -x "$ENVP/bin/python" ] || "$CONDA" create -y -p "$ENVP" python=3.10

"$ENVP/bin/pip" install --no-cache-dir "torch==2.6.*" \
  --index-url https://download.pytorch.org/whl/cu124
"$ENVP/bin/pip" install --no-cache-dir transformers accelerate safetensors

# vLLM + ray are only needed for the gate-replication study; install lazily:
#   "$ENVP/bin/pip" install --no-cache-dir "vllm>=0.8,<0.9" ray

echo "Env ready: $ENVP"
echo "Submit the validation job:  sbatch $SCRATCH/sbatch/mig_smoke.sbatch"
echo "NOTE: E2c itself never runs on Hopper — frozen local protocol."
