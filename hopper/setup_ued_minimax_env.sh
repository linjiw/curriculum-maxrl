#!/usr/bin/env bash
# Create a verified, lock- and setup-addressed Python/JAX CUDA 12 environment.
# Run on a Hopper login node from the content-addressed UED bundle.
set -euo pipefail
umask 027

readonly HERE="$(cd "$(dirname "$0")" && pwd)"
readonly LOCK="$HERE/requirements-ued-minimax-hopper.lock"
readonly SETUP_SCRIPT="$HERE/setup_ued_minimax_env.sh"
[[ -f "$LOCK" && ! -L "$LOCK" ]] || {
  echo "missing or symbolic environment lock: $LOCK" >&2
  exit 1
}
[[ -f "$SETUP_SCRIPT" && ! -L "$SETUP_SCRIPT" ]] || {
  echo "missing or symbolic environment setup script: $SETUP_SCRIPT" >&2
  exit 1
}

readonly SCRATCH_ROOT="${HOPPER_SCRATCH:-/scratch/$USER}"
[[ "$SCRATCH_ROOT" =~ ^/scratch/[A-Za-z0-9._-]+$ ]] || {
  echo "unsafe scratch root: $SCRATCH_ROOT" >&2
  exit 2
}

readonly LOCK_SHA256="$(sha256sum "$LOCK" | awk '{print $1}')"
readonly SETUP_SHA256="$(sha256sum "$SETUP_SCRIPT" | awk '{print $1}')"
readonly ENV_SCHEMA=2
readonly ENV_DIR="$SCRATCH_ROOT/envs/ued-minimax-v${ENV_SCHEMA}-${LOCK_SHA256:0:16}-${SETUP_SHA256:0:16}"
readonly CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
[[ -x "$CONDA_BIN" ]] || { echo "missing conda: $CONDA_BIN" >&2; exit 1; }

readonly EXPECTED_PYTHON=3.10.20
readonly EXPECTED_PYTHON_BUILD=h741d88c_0
readonly EXPECTED_GIT=2.45.2
readonly EXPECTED_GIT_BUILD=pl5340h9abc3c3_0
readonly EXPECTED_NUMPY=1.25.2
readonly EXPECTED_JAX=0.4.31
readonly EXPECTED_JAXLIB=0.4.31
readonly EXPECTED_JAX_PLUGIN=0.4.31
readonly EXPECTED_JAX_PJRT=0.4.31

export CONDA_PKGS_DIRS="$SCRATCH_ROOT/.conda_pkgs"
export PIP_CACHE_DIR="$SCRATCH_ROOT/.pipcache"
mkdir -p "$SCRATCH_ROOT/envs" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

verify_env() {
  local candidate=$1 require_complete=${2:-true}
  local actual_lock actual_setup freeze_sha manifest_sha
  local complete_schema complete_manifest complete_setup
  local recorded_freeze current_freeze recorded_conda current_conda
  [[ -d "$candidate" && ! -L "$candidate" && -x "$candidate/bin/python" \
     && -f "$candidate/LOCK_SHA256" && ! -L "$candidate/LOCK_SHA256" \
     && -f "$candidate/SETUP_SHA256" && ! -L "$candidate/SETUP_SHA256" \
     && -f "$candidate/CONDA_EXPLICIT.txt" && ! -L "$candidate/CONDA_EXPLICIT.txt" \
     && -f "$candidate/ENVIRONMENT.freeze" && ! -L "$candidate/ENVIRONMENT.freeze" \
     && -f "$candidate/ENVIRONMENT.json" && ! -L "$candidate/ENVIRONMENT.json" \
     && -f "$candidate/ENVIRONMENT_SHA256SUMS" \
     && ! -L "$candidate/ENVIRONMENT_SHA256SUMS" \
     && -x "$candidate/bin/git" ]] || {
    echo "incomplete lock-addressed environment: $candidate" >&2
    return 1
  }
  if [[ "$require_complete" == true ]]; then
    [[ -f "$candidate/ENVIRONMENT_COMPLETE" \
       && ! -L "$candidate/ENVIRONMENT_COMPLETE" ]] || {
      echo "environment has no completion marker: $candidate" >&2
      return 1
    }
  elif [[ "$require_complete" != false ]]; then
    echo "invalid verify_env completion mode: $require_complete" >&2
    return 1
  fi
  actual_lock=$(awk 'NR == 1 {print $1}' "$candidate/LOCK_SHA256")
  [[ "$actual_lock" == "$LOCK_SHA256" ]] || {
    echo "environment lock mismatch: $candidate" >&2
    return 1
  }
  actual_setup=$(awk 'NR == 1 {print $1}' "$candidate/SETUP_SHA256")
  [[ "$actual_setup" == "$SETUP_SHA256" ]] || {
    echo "environment setup-script mismatch: $candidate" >&2
    return 1
  }
  (cd "$candidate" && sha256sum -c --strict ENVIRONMENT_SHA256SUMS >/dev/null)
  "$candidate/bin/python" -m pip check
  recorded_freeze=$(<"$candidate/ENVIRONMENT.freeze")
  current_freeze=$("$candidate/bin/python" -m pip freeze --all | LC_ALL=C sort)
  if [[ "$current_freeze" != "$recorded_freeze" ]]; then
    echo "installed packages differ from recorded environment freeze: $candidate" >&2
    return 1
  fi
  recorded_conda=$(<"$candidate/CONDA_EXPLICIT.txt")
  current_conda=$("$CONDA_BIN" list --explicit -p "$candidate")
  if [[ "$current_conda" != "$recorded_conda" ]]; then
    echo "installed Conda packages differ from recorded explicit spec: $candidate" >&2
    return 1
  fi
  grep -Eq "/python-${EXPECTED_PYTHON}-${EXPECTED_PYTHON_BUILD}\\.(conda|tar\\.bz2)(#|$)" \
    "$candidate/CONDA_EXPLICIT.txt" || {
    echo "expected Python build is absent from Conda explicit spec: $candidate" >&2
    return 1
  }
  grep -Eq "/git-${EXPECTED_GIT}-${EXPECTED_GIT_BUILD}\\.(conda|tar\\.bz2)(#|$)" \
    "$candidate/CONDA_EXPLICIT.txt" || {
    echo "expected Git build is absent from Conda explicit spec: $candidate" >&2
    return 1
  }
  [[ "$("$candidate/bin/git" --version)" == "git version $EXPECTED_GIT" ]] || {
    echo "unexpected Git runtime in environment: $candidate" >&2
    return 1
  }
  UED_EXPECTED_PYTHON="$EXPECTED_PYTHON" \
  UED_EXPECTED_NUMPY="$EXPECTED_NUMPY" \
  UED_EXPECTED_JAX="$EXPECTED_JAX" \
  UED_EXPECTED_JAXLIB="$EXPECTED_JAXLIB" \
  UED_EXPECTED_JAX_PLUGIN="$EXPECTED_JAX_PLUGIN" \
  UED_EXPECTED_JAX_PJRT="$EXPECTED_JAX_PJRT" \
  UED_EXPECTED_ENV_SCHEMA="$ENV_SCHEMA" \
  UED_EXPECTED_SETUP_SHA256="$SETUP_SHA256" \
  UED_EXPECTED_GIT="$EXPECTED_GIT" \
  UED_EXPECTED_GIT_BUILD="$EXPECTED_GIT_BUILD" \
  UED_ENV_RECORD="$candidate/ENVIRONMENT.json" \
  "$candidate/bin/python" - <<'PY'
import importlib.metadata as metadata
import json
import os
import platform

assert platform.python_version() == os.environ["UED_EXPECTED_PYTHON"]
expected = {
    "numpy": os.environ["UED_EXPECTED_NUMPY"],
    "jax": os.environ["UED_EXPECTED_JAX"],
    "jaxlib": os.environ["UED_EXPECTED_JAXLIB"],
    "jax-cuda12-plugin": os.environ["UED_EXPECTED_JAX_PLUGIN"],
    "jax-cuda12-pjrt": os.environ["UED_EXPECTED_JAX_PJRT"],
}
actual = {name: metadata.version(name) for name in expected}
assert actual == expected, (actual, expected)
with open(os.environ["UED_ENV_RECORD"], encoding="utf-8") as stream:
    record = json.load(stream)
assert record["environment_schema"] == int(os.environ["UED_EXPECTED_ENV_SCHEMA"])
assert record["setup_script_sha256"] == os.environ["UED_EXPECTED_SETUP_SHA256"]
assert record["git"] == os.environ["UED_EXPECTED_GIT"]
assert record["git_conda_build"] == os.environ["UED_EXPECTED_GIT_BUILD"]
PY
  freeze_sha=$(sha256sum "$candidate/ENVIRONMENT.freeze" | awk '{print $1}')
  manifest_sha=$(sha256sum "$candidate/ENVIRONMENT_SHA256SUMS" | awk '{print $1}')
  if [[ "$require_complete" == true ]]; then
    complete_schema=$(awk -F '\t' '$1 == "environment_schema" {print $2}' \
      "$candidate/ENVIRONMENT_COMPLETE")
    complete_manifest=$(awk -F '\t' '$1 == "manifest_sha256" {print $2}' \
      "$candidate/ENVIRONMENT_COMPLETE")
    complete_setup=$(awk -F '\t' '$1 == "setup_script_sha256" {print $2}' \
      "$candidate/ENVIRONMENT_COMPLETE")
    [[ "$complete_schema" == "$ENV_SCHEMA" \
       && "$complete_manifest" == "$manifest_sha" \
       && "$complete_setup" == "$SETUP_SHA256" ]] || {
      echo "environment completion marker binding mismatch: $candidate" >&2
      return 1
    }
  fi
  printf 'UED_ENV_DIR=%s\n' "$candidate"
  printf 'UED_ENV_LOCK_SHA256=%s\n' "$LOCK_SHA256"
  printf 'UED_ENV_SETUP_SHA256=%s\n' "$SETUP_SHA256"
  printf 'UED_ENV_FREEZE_SHA256=%s\n' "$freeze_sha"
  printf 'UED_ENV_MANIFEST_SHA256=%s\n' "$manifest_sha"
}

if [[ -e "$ENV_DIR" || -L "$ENV_DIR" ]]; then
  verify_env "$ENV_DIR"
  exit 0
fi

readonly SETUP_LOCK_DIR="$SCRATCH_ROOT/envs/.ued-minimax-v${ENV_SCHEMA}-${LOCK_SHA256:0:16}-${SETUP_SHA256:0:16}.lock"
if ! mkdir "$SETUP_LOCK_DIR"; then
  echo "environment setup lock is held: $SETUP_LOCK_DIR" >&2
  exit 1
fi
LOCK_HELD=true
BUILD_OWNED=false

cleanup() {
  if [[ "${BUILD_OWNED:-false}" == true \
        && "$ENV_DIR" == "$SCRATCH_ROOT/envs/ued-minimax-v${ENV_SCHEMA}-${LOCK_SHA256:0:16}-${SETUP_SHA256:0:16}" \
        && (-e "$ENV_DIR" || -L "$ENV_DIR") ]]; then
    rm -rf -- "$ENV_DIR"
  fi
  if [[ "${LOCK_HELD:-false}" == true && -d "$SETUP_LOCK_DIR" ]]; then
    rmdir -- "$SETUP_LOCK_DIR" || true
  fi
}
trap cleanup EXIT

if [[ -e "$ENV_DIR" || -L "$ENV_DIR" ]]; then
  verify_env "$ENV_DIR"
  exit 0
fi
BUILD_OWNED=true

# Build directly at the final Conda prefix: Conda environments contain
# prefix-bound shebangs and metadata and therefore must never be relocated.
"$CONDA_BIN" create -y -p "$ENV_DIR" \
  "python=$EXPECTED_PYTHON=$EXPECTED_PYTHON_BUILD" \
  "git=$EXPECTED_GIT=$EXPECTED_GIT_BUILD" pip
"$ENV_DIR/bin/python" -m pip install --disable-pip-version-check \
  --requirement "$LOCK"
"$ENV_DIR/bin/python" -m pip check

"$ENV_DIR/bin/python" -m pip freeze --all \
  | LC_ALL=C sort > "$ENV_DIR/ENVIRONMENT.freeze.tmp"
mv "$ENV_DIR/ENVIRONMENT.freeze.tmp" "$ENV_DIR/ENVIRONMENT.freeze"
if grep -Eq '(^-e | @ file:|/tmp/)' "$ENV_DIR/ENVIRONMENT.freeze"; then
  echo "environment freeze contains a local or editable dependency" >&2
  exit 1
fi
printf '%s  %s\n' "$LOCK_SHA256" requirements-ued-minimax-hopper.lock \
  > "$ENV_DIR/LOCK_SHA256"
printf '%s  %s\n' "$SETUP_SHA256" setup_ued_minimax_env.sh \
  > "$ENV_DIR/SETUP_SHA256"
"$CONDA_BIN" list --explicit -p "$ENV_DIR" \
  > "$ENV_DIR/CONDA_EXPLICIT.txt.tmp"
mv "$ENV_DIR/CONDA_EXPLICIT.txt.tmp" "$ENV_DIR/CONDA_EXPLICIT.txt"

UED_ENV_RECORD_TMP="$ENV_DIR/ENVIRONMENT.json.tmp" \
UED_ENV_LOCK_SHA256="$LOCK_SHA256" \
UED_ENV_PYTHON_BUILD="$EXPECTED_PYTHON_BUILD" \
UED_ENV_GIT="$EXPECTED_GIT" \
UED_ENV_GIT_BUILD="$EXPECTED_GIT_BUILD" \
UED_ENV_SCHEMA="$ENV_SCHEMA" \
UED_ENV_SETUP_SHA256="$SETUP_SHA256" \
"$ENV_DIR/bin/python" - <<'PY'
import importlib.metadata as metadata
import json
import os
import platform
from pathlib import Path

packages = {
    dist.metadata["Name"]: dist.version
    for dist in metadata.distributions()
    if dist.metadata.get("Name")
}
record = {
    "environment_schema": int(os.environ["UED_ENV_SCHEMA"]),
    "purpose": "bounded UED minimax/AMaze engineering smoke",
    "python": platform.python_version(),
    "python_conda_build": os.environ["UED_ENV_PYTHON_BUILD"],
    "git": os.environ["UED_ENV_GIT"],
    "git_conda_build": os.environ["UED_ENV_GIT_BUILD"],
    "platform": platform.platform(),
    "environment_lock_sha256": os.environ["UED_ENV_LOCK_SHA256"],
    "setup_script_sha256": os.environ["UED_ENV_SETUP_SHA256"],
    "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
}
Path(os.environ["UED_ENV_RECORD_TMP"]).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
mv "$ENV_DIR/ENVIRONMENT.json.tmp" "$ENV_DIR/ENVIRONMENT.json"
(
  cd "$ENV_DIR"
  sha256sum CONDA_EXPLICIT.txt ENVIRONMENT.freeze ENVIRONMENT.json LOCK_SHA256 \
    SETUP_SHA256 \
    > ENVIRONMENT_SHA256SUMS
  sha256sum -c --strict ENVIRONMENT_SHA256SUMS >/dev/null
)

verify_env "$ENV_DIR" false >/dev/null
ENV_MANIFEST_SHA256=$(sha256sum "$ENV_DIR/ENVIRONMENT_SHA256SUMS" | awk '{print $1}')
{
  printf 'environment_schema\t%s\n' "$ENV_SCHEMA"
  printf 'manifest_sha256\t%s\n' "$ENV_MANIFEST_SHA256"
  printf 'setup_script_sha256\t%s\n' "$SETUP_SHA256"
} > "$ENV_DIR/ENVIRONMENT_COMPLETE.tmp"
mv "$ENV_DIR/ENVIRONMENT_COMPLETE.tmp" "$ENV_DIR/ENVIRONMENT_COMPLETE"
verify_env "$ENV_DIR"
BUILD_OWNED=false
rmdir -- "$SETUP_LOCK_DIR"
LOCK_HELD=false
trap - EXIT
