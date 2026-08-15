#!/usr/bin/env bash
# Real local both-arm Phase-A -> receipt-gated Phase-B E2E.
set -euo pipefail
umask 077
readonly HERE="$(cd "$(dirname "$0")" && pwd)"
readonly ROOT="$(cd "$HERE/.." && pwd)"
readonly PY="${UED_CPU_PYTHON:-/data/robotixx/ued_bench/envs/minimax-jax0431-cpu/bin/python}"
env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u LD_PRELOAD \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu \
  JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
  "$PY" -I -B "$ROOT/ued_benchmark/hopper_v4/test_local_terminal_chain_v4.py"
