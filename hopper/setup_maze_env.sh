#!/bin/bash
# Create an immutable, lock-hash-addressed Hopper environment for MAZE-SCORE.
# Run on a Hopper login node from a staged repository checkout.
set -euo pipefail
umask 027

HERE=$(cd "$(dirname "$0")" && pwd)
LOCK="$HERE/requirements-maze-hopper.lock"
[[ -f "$LOCK" ]] || { echo "missing $LOCK" >&2; exit 1; }

SCRATCH_ROOT=${HOPPER_SCRATCH:-/scratch/$USER}
if [[ ! "$SCRATCH_ROOT" =~ ^/scratch/[A-Za-z0-9._-]+$ ]]; then
  echo "unsafe scratch root: $SCRATCH_ROOT" >&2
  exit 2
fi

LOCK_SHA=$(sha256sum "$LOCK" | awk '{print $1}')
ENV_DIR="$SCRATCH_ROOT/envs/maze-score-${LOCK_SHA:0:16}"
CONDA_BIN=${CONDA_BIN:-$HOME/miniconda3/bin/conda}
[[ -x "$CONDA_BIN" ]] || { echo "missing conda: $CONDA_BIN" >&2; exit 1; }

EXPECTED_NUMPY=$(awk -F '==' '$1 == "numpy" {print $2}' "$LOCK")
EXPECTED_TORCH=$(awk -F '==' '$1 == "torch" {print $2}' "$LOCK")
[[ -n "$EXPECTED_NUMPY" && -n "$EXPECTED_TORCH" ]] \
  || { echo "lock must pin numpy and torch" >&2; exit 1; }

export CONDA_PKGS_DIRS="$SCRATCH_ROOT/.conda_pkgs"
export PIP_CACHE_DIR="$SCRATCH_ROOT/.pipcache"
mkdir -p "$SCRATCH_ROOT/envs" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

verify_env() {
  local candidate=$1 actual_lock
  [[ -d "$candidate" && ! -L "$candidate" && -x "$candidate/bin/python" \
     && -f "$candidate/LOCK_SHA256" \
     && -f "$candidate/ENVIRONMENT.freeze" \
     && -f "$candidate/ENVIRONMENT.json" ]] \
    || { echo "incomplete lock-addressed environment: $candidate" >&2; return 1; }
  actual_lock=$(awk 'NR == 1 {print $1}' "$candidate/LOCK_SHA256")
  [[ "$actual_lock" == "$LOCK_SHA" ]] \
    || { echo "environment lock mismatch: $candidate" >&2; return 1; }
  "$candidate/bin/python" -m pip check
  MAZE_EXPECTED_NUMPY="$EXPECTED_NUMPY" \
  MAZE_EXPECTED_TORCH="$EXPECTED_TORCH" \
  "$candidate/bin/python" - <<'PY'
import os
import numpy
import torch

assert numpy.__version__ == os.environ["MAZE_EXPECTED_NUMPY"]
assert torch.__version__ == os.environ["MAZE_EXPECTED_TORCH"]
PY
}

if [[ -e "$ENV_DIR" || -L "$ENV_DIR" ]]; then
  verify_env "$ENV_DIR"
  printf 'MAZE_ENV=%s\n' "$ENV_DIR"
  printf 'LOCK_SHA256=%s\n' "$LOCK_SHA"
  sha256sum "$ENV_DIR/ENVIRONMENT.freeze" "$ENV_DIR/ENVIRONMENT.json"
  exit 0
fi

BUILD_DIR="$SCRATCH_ROOT/envs/.maze-score-${LOCK_SHA:0:16}.build-$$"
[[ ! -e "$BUILD_DIR" && ! -L "$BUILD_DIR" ]] \
  || { echo "build path exists: $BUILD_DIR" >&2; exit 1; }
"$CONDA_BIN" create -y -p "$BUILD_DIR" python=3.10.20 pip

"$BUILD_DIR/bin/pip" install --disable-pip-version-check "numpy==$EXPECTED_NUMPY"
"$BUILD_DIR/bin/pip" install --disable-pip-version-check \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple \
  "torch==$EXPECTED_TORCH"
"$BUILD_DIR/bin/pip" check

FREEZE_TMP="$BUILD_DIR/ENVIRONMENT.freeze.tmp"
"$BUILD_DIR/bin/pip" freeze | LC_ALL=C sort > "$FREEZE_TMP"
mv "$FREEZE_TMP" "$BUILD_DIR/ENVIRONMENT.freeze"
printf '%s  %s\n' "$LOCK_SHA" requirements-maze-hopper.lock \
  > "$BUILD_DIR/LOCK_SHA256"

MAZE_ENV_RECORD_TMP="$BUILD_DIR/ENVIRONMENT.json.tmp" "$BUILD_DIR/bin/python" - <<'PY'
import json
import os
import platform
from pathlib import Path

import numpy
import torch

record = {
    "python": platform.python_version(),
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}
Path(os.environ["MAZE_ENV_RECORD_TMP"]).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
mv "$BUILD_DIR/ENVIRONMENT.json.tmp" "$BUILD_DIR/ENVIRONMENT.json"

verify_env "$BUILD_DIR"
# GNU mv -T refuses to merge into a concurrently created final directory.
mv -T "$BUILD_DIR" "$ENV_DIR"

printf 'MAZE_ENV=%s\n' "$ENV_DIR"
printf 'LOCK_SHA256=%s\n' "$LOCK_SHA"
sha256sum "$ENV_DIR/ENVIRONMENT.freeze" "$ENV_DIR/ENVIRONMENT.json"
