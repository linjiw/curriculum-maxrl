#!/usr/bin/env bash
# Build/verify the frozen v4 environment, then close every installed byte.
# This script contains no SSH, scheduler, endpoint, production, or cost100 path.
set -euo pipefail
umask 027
export PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONOPTIMIZE
unset LD_LIBRARY_PATH LD_PRELOAD

readonly HERE="$(cd "$(dirname "$0")" && pwd -P)"
readonly BUNDLE="$(cd "$HERE/.." && pwd -P)"
[[ "$HERE" == "$BUNDLE/hopper" && -f "$BUNDLE/SHA256SUMS" ]] || {
  echo "setup must execute from a canonical staged bundle" >&2
  exit 2
}
readonly USER_NAME="$(id -un)"
[[ "$USER_NAME" =~ ^[A-Za-z0-9._-]+$ && "$USER_NAME" != . && "$USER_NAME" != .. ]] || {
  echo "unsafe runtime user" >&2; exit 2;
}
readonly SCRATCH_ROOT="/scratch/$USER_NAME"
readonly CONDA_BIN="/home/$USER_NAME/miniconda3/bin/conda"
[[ -x "$CONDA_BIN" && ! -L "$CONDA_BIN" \
   && "$(readlink -f -- "$CONDA_BIN")" == "$CONDA_BIN" ]] || {
  echo "fixed canonical Conda executable missing: $CONDA_BIN" >&2
  exit 1
}
readonly LEGACY_SETUP="$HERE/setup_ued_minimax_env.sh"
readonly LOCK="$HERE/requirements-ued-minimax-hopper.lock"
readonly TREE_TOOL="$BUNDLE/ued_benchmark/hopper_v4_remote_hardening/environment_tree.py"
for path in "$LEGACY_SETUP" "$LOCK" "$TREE_TOOL"; do
  [[ -f "$path" && ! -L "$path" ]] || { echo "missing setup input: $path" >&2; exit 1; }
done
readonly LOCK_SHA="$(sha256sum "$LOCK" | awk '{print $1}')"
readonly LEGACY_SETUP_SHA="$(sha256sum "$LEGACY_SETUP" | awk '{print $1}')"
readonly TREE_TOOL_SHA="$(sha256sum "$TREE_TOOL" | awk '{print $1}')"
readonly ENV_DIR="$SCRATCH_ROOT/envs/ued-minimax-v2-${LOCK_SHA:0:16}-${LEGACY_SETUP_SHA:0:16}"

/usr/bin/env -i \
  PATH=/usr/bin:/bin USER="$USER_NAME" HOME="/home/$USER_NAME" \
  HOPPER_SCRATCH="$SCRATCH_ROOT" CONDA_BIN="$CONDA_BIN" \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /usr/bin/bash "$LEGACY_SETUP"
[[ -d "$ENV_DIR" && ! -L "$ENV_DIR" \
   && "$(readlink -f -- "$ENV_DIR")" == "$ENV_DIR" ]] || {
  echo "legacy setup did not produce the fixed environment" >&2
  exit 1
}

readonly CLOSURE_PARENT="$SCRATCH_ROOT/maxrl/provenance/ued-minimax-v4h-environments"
mkdir -p "$CLOSURE_PARENT"
[[ "$(readlink -f -- "$CLOSURE_PARENT")" == "$CLOSURE_PARENT" && ! -L "$CLOSURE_PARENT" ]] || {
  echo "unsafe environment-closure parent" >&2; exit 2;
}
readonly CLOSURE="$CLOSURE_PARENT/v1-${LOCK_SHA:0:16}-${LEGACY_SETUP_SHA:0:16}-${TREE_TOOL_SHA:0:16}"
if [[ ! -e "$CLOSURE" && ! -L "$CLOSURE" ]]; then
  /usr/bin/python3 -I -B "$TREE_TOOL" create \
    --environment "$ENV_DIR" --conda "$CONDA_BIN" --closure "$CLOSURE" \
    --expected-tool-sha256 "$TREE_TOOL_SHA" --expected-python-version 3.10.20
fi
[[ -d "$CLOSURE" && ! -L "$CLOSURE" \
   && "$(readlink -f -- "$CLOSURE")" == "$CLOSURE" ]] || {
  echo "unsafe environment closure" >&2; exit 2;
}
readonly CLOSURE_MANIFEST_SHA="$(sha256sum "$CLOSURE/SHA256SUMS" | awk '{print $1}')"
readonly CLOSURE_RECEIPT_SHA="$(sha256sum "$CLOSURE/receipt.json" | awk '{print $1}')"
/usr/bin/python3 -I -B "$TREE_TOOL" verify \
  --environment "$ENV_DIR" --conda "$CONDA_BIN" --closure "$CLOSURE" \
  --expected-tool-sha256 "$TREE_TOOL_SHA" --expected-python-version 3.10.20 \
  --expected-manifest-sha256 "$CLOSURE_MANIFEST_SHA" \
  --expected-receipt-sha256 "$CLOSURE_RECEIPT_SHA"

printf 'environment_dir\t%s\n' "$ENV_DIR"
printf 'environment_tree_dir\t%s\n' "$CLOSURE"
printf 'environment_tree_manifest_sha256\t%s\n' "$CLOSURE_MANIFEST_SHA"
printf 'environment_tree_receipt_sha256\t%s\n' "$CLOSURE_RECEIPT_SHA"
printf 'environment_tree_tool_sha256\t%s\n' "$TREE_TOOL_SHA"
printf 'conda_path\t%s\n' "$CONDA_BIN"
printf 'conda_sha256\t%s\n' "$(sha256sum "$CONDA_BIN" | awk '{print $1}')"
