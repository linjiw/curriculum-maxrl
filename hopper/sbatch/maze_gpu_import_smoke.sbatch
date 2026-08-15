#!/usr/bin/env bash
#SBATCH --job-name=maze-gpu-import
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/%u/maxrl/tests/logs/%x_%j.out

# Import-only, non-evidence GPU smoke. No model/data download and no endpoint.
set -euo pipefail
umask 027
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

required=(MAZE_BUNDLE_DIR MAZE_SOURCE_MANIFEST_SHA256 MAZE_ENV_DIR MAZE_ENV_LOCK_SHA256)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing required variable: $name" >&2; exit 2; }
done
[[ "$MAZE_SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$MAZE_ENV_LOCK_SHA256" =~ ^[0-9a-f]{64}$ ]]
RUN_USER=$(id -un)
SCRATCH_ROOT="/scratch/$RUN_USER"
case "$MAZE_BUNDLE_DIR" in "$SCRATCH_ROOT"/*) ;; *) exit 2;; esac
case "$MAZE_ENV_DIR" in "$SCRATCH_ROOT"/*) ;; *) exit 2;; esac

manifest="$MAZE_BUNDLE_DIR/SHA256SUMS"
[[ "$(sha256sum "$manifest" | awk '{print $1}')" == "$MAZE_SOURCE_MANIFEST_SHA256" ]]
(cd "$MAZE_BUNDLE_DIR" && sha256sum -c --strict SHA256SUMS >/dev/null)
bundle_lock="$MAZE_BUNDLE_DIR/hopper/requirements-maze-hopper.lock"
[[ "$(sha256sum "$bundle_lock" | awk '{print $1}')" == "$MAZE_ENV_LOCK_SHA256" ]]
[[ "$(awk 'NR == 1 {print $1}' "$MAZE_ENV_DIR/LOCK_SHA256")" == "$MAZE_ENV_LOCK_SHA256" ]]

result="$SCRATCH_ROOT/maxrl/tests/gpu-import/$SLURM_JOB_ID"
[[ ! -e "$result" && ! -L "$result" ]] || { echo "result exists: $result" >&2; exit 2; }
mkdir -p "$result"
PY="$MAZE_ENV_DIR/bin/python"

PYTHONPATH="$MAZE_BUNDLE_DIR/curriculum_maxrl" \
MAZE_GPU_SMOKE_RESULT="$result/runtime.json" \
MAZE_GPU_SMOKE_SOURCE="$MAZE_SOURCE_MANIFEST_SHA256" \
MAZE_GPU_SMOKE_ENV="$MAZE_ENV_LOCK_SHA256" \
"$PY" - <<'PY'
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
from estimators import coefficient_activity, legacy_frontier_activity

assert torch.cuda.is_available(), "CUDA is not visible inside Slurm allocation"
assert np.allclose(coefficient_activity(np.array([0.2]), 2), np.array([0.16]))
assert np.allclose(legacy_frontier_activity(np.array([0.2]), 32),
                   coefficient_activity(np.array([0.2]), 33))
device = torch.cuda.get_device_properties(0)
record = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "hostname": platform.node(),
    "python": platform.python_version(),
    "numpy": np.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": True,
    "gpu_name": device.name,
    "gpu_memory_bytes": device.total_memory,
    "source_manifest_sha256": os.environ["MAZE_GPU_SMOKE_SOURCE"],
    "environment_lock_sha256": os.environ["MAZE_GPU_SMOKE_ENV"],
}
Path(os.environ["MAZE_GPU_SMOKE_RESULT"]).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

"$PY" -m py_compile \
  "$MAZE_BUNDLE_DIR/curriculum_maxrl/estimators.py" \
  "$MAZE_BUNDLE_DIR/curriculum_maxrl/maze_gpu/model.py" \
  "$MAZE_BUNDLE_DIR/curriculum_maxrl/maze_gpu/train.py" \
  "$MAZE_BUNDLE_DIR/curriculum_maxrl/maze_score/analyze_maze_score.py"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  > "$result/nvidia-smi.csv"
(
  cd "$result"
  sha256sum runtime.json nvidia-smi.csv > SHA256SUMS
  sha256sum -c --strict SHA256SUMS >/dev/null
)
printf 'complete\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$result/COMPLETE"
echo "MAZE_GPU_IMPORT_SMOKE_PASS $result"
