#!/usr/bin/env bash
# Local, network-free checks for the bounded UED minimax Hopper path.
set -euo pipefail
umask 077

readonly HERE="$(cd "$(dirname "$0")" && pwd)"
readonly ROOT="$(cd "$HERE/.." && pwd)"
readonly PINNED_COMMIT=d053054c5290a04c1c4cd8b55704d999cad73e30
readonly SOURCE_DIR="${MINIMAX_SOURCE_DIR:-/tmp/root-minimax-260814}"
readonly CPU_PYTHON="${UED_CPU_PYTHON:-/home/robotixx/miniconda3/pkgs/python-3.10.20-h741d88c_0/bin/python}"
readonly CPU_SITE_PACKAGES="${UED_CPU_SITE_PACKAGES:-/data/robotixx/ued_bench/envs/minimax-jax0431-cpu/lib/python3.10/site-packages}"
readonly CPU_LIBRARY_PATH="${UED_CPU_LIBRARY_PATH:-/home/robotixx/miniconda3/pkgs/openssl-3.5.5-h1b28b03_0/lib}"
[[ -d "$SOURCE_DIR/.git" || -f "$SOURCE_DIR/.git" ]] || {
  echo "set MINIMAX_SOURCE_DIR to a clone at $PINNED_COMMIT" >&2
  exit 1
}
[[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == "$PINNED_COMMIT" ]]
[[ -x "$CPU_PYTHON" ]] || {
  echo "set UED_CPU_PYTHON to the pinned Python 3.10.20 interpreter" >&2
  exit 1
}
[[ -d "$CPU_SITE_PACKAGES" ]] || {
  echo "set UED_CPU_SITE_PACKAGES to the JAX 0.4.31 CPU validation packages" >&2
  exit 1
}
[[ -d "$CPU_LIBRARY_PATH" ]] || {
  echo "set UED_CPU_LIBRARY_PATH to libraries for the pinned interpreter" >&2
  exit 1
}

files=(
  "$HERE/setup_ued_minimax_env.sh"
  "$HERE/stage_ued_minimax.sh"
  "$HERE/sbatch/ued_minimax_gpu_smoke.sbatch"
  "$HERE/sbatch/ued_minimax_one_update_smoke.sbatch"
  "$HERE/test_ued_minimax_local.sh"
  "$HERE/test_ued_minimax_one_update_local.sh"
)
for file in "${files[@]}"; do
  bash -n "$file"
done

readonly LOCK="$HERE/requirements-ued-minimax-hopper.lock"
grep -Fxq 'jax==0.4.31' "$LOCK"
grep -Fxq 'jaxlib==0.4.31' "$LOCK"
grep -Fxq 'jax-cuda12-plugin==0.4.31' "$LOCK"
grep -Fxq 'jax-cuda12-pjrt==0.4.31' "$LOCK"
if awk '!/^($|#)/ && $0 !~ /==/' "$LOCK" | grep -q .; then
  echo "all UED lock entries must use exact == pins" >&2
  exit 1
fi
grep -Fq 'readonly EXPECTED_PYTHON=3.10.20' "$HERE/setup_ued_minimax_env.sh"
grep -Fq 'readonly EXPECTED_PYTHON_BUILD=h741d88c_0' "$HERE/setup_ued_minimax_env.sh"
grep -Fq 'readonly EXPECTED_GIT=2.45.2' "$HERE/setup_ued_minimax_env.sh"
grep -Fq 'readonly EXPECTED_GIT_BUILD=pl5340h9abc3c3_0' "$HERE/setup_ued_minimax_env.sh"
grep -Fq 'CONDA_EXPLICIT.txt' "$HERE/setup_ued_minimax_env.sh"
grep -Fq 'UED_ENV_MANIFEST_SHA256' "$HERE/setup_ued_minimax_env.sh"
grep -Fq 'readonly ENV_SCHEMA=2' "$HERE/setup_ued_minimax_env.sh"
grep -Fq 'ENVIRONMENT_COMPLETE' "$HERE/setup_ued_minimax_env.sh"
grep -Fq 'SETUP_SHA256' "$HERE/setup_ued_minimax_env.sh"
grep -Fq 'create -y -p "$ENV_DIR"' "$HERE/setup_ued_minimax_env.sh"
! grep -Fq 'mv -T' "$HERE/setup_ued_minimax_env.sh"

readonly SBATCH="$HERE/sbatch/ued_minimax_gpu_smoke.sbatch"
grep -Fxq '#SBATCH --gres=gpu:1g.10gb:1' "$SBATCH"
grep -Fxq '#SBATCH --time=00:10:00' "$SBATCH"
grep -Fxq '#SBATCH --cpus-per-task=2' "$SBATCH"
grep -Fxq '#SBATCH --mem=15G' "$SBATCH"
! grep -Eq 'python[^#\n]*-m[[:space:]]+minimax\.train|python[^#\n]*minimax/train\.py' "$SBATCH"
grep -Fq '"training_endpoint": False' "$SBATCH"
grep -Fq 'SHA256SUMS' "$SBATCH"
grep -Fq 'COMPLETE' "$SBATCH"
grep -Fq 'UED_SBATCH_SHA256' "$SBATCH"
grep -Fq 'UED_ENV_MANIFEST_SHA256' "$SBATCH"
grep -Fq 'CONDA_EXPLICIT.txt' "$SBATCH"
grep -Fq 'ENVIRONMENT_COMPLETE' "$SBATCH"
grep -Fq 'ENV_SETUP_SHA256' "$SBATCH"
grep -Fq 'readonly GIT="$UED_ENV_DIR/bin/git"' "$SBATCH"
grep -Fq '"$GIT" clone --quiet' "$SBATCH"

readonly TMP="$(mktemp -d /tmp/ued-minimax-local-test.XXXXXX)"
cleanup() {
  if [[ -n "${TMP:-}" && "$TMP" == /tmp/ued-minimax-local-test.* && -d "$TMP" ]]; then
    rm -rf -- "$TMP"
  fi
}
trap cleanup EXIT

MINIMAX_SOURCE_DIR="$SOURCE_DIR" \
  bash "$HERE/stage_ued_minimax.sh" local "$TMP/bundle" \
  > "$TMP/stage.out"
(
  cd "$TMP/bundle"
  sha256sum -c --strict SHA256SUMS >/dev/null
  cd ued_benchmark
  sha256sum -c --strict OVERLAY_SHA256SUMS >/dev/null
)

UED_TEST_STATE="$TMP/bundle/BUNDLE_STATE.json" \
UED_TEST_LOCK_SHA="$(sha256sum "$LOCK" | awk '{print $1}')" \
python3 - <<'PY'
import json
import os

with open(os.environ["UED_TEST_STATE"], encoding="utf-8") as stream:
    state = json.load(stream)
assert state["bundle_schema"] == 4
assert state["purpose"] == "bounded UED minimax/AMaze engineering smokes only"
assert state["paper_evidence"] is False
assert state["workspace_tree_exclusions"] == [
    "ued_benchmark/blackwell_probe/**",
    "ued_benchmark/blackwell_training_probe/**",
]
assert state["allowed_engineering_endpoints"] == [
    "frontier_exact_grouped_one_update",
    "frontier_terminal_chain_components",
    "gpu_import_formula_jit",
]
assert state["max_student_updates"] == 1
assert state["terminal_chain_contract"] == {
    "analyzer_eligible": False,
    "actual_external_evaluation": True,
    "paper_evidence": False,
    "phase_a": "slurm_closed_training_and_actual_external_evaluation_components",
    "phase_a_submission_export": "explicit_ued_allowlist_no_all",
    "phase_a_python_flags": "-I -B",
    "phase_b": "post_terminal_local_atomic_engineering_assembly",
    "phase_b_python": "isolated_clean_python_3.10.20_venv",
    "phase_b_python_flags": "-I -B",
    "finalizer_self_bound": True,
    "post_terminal_fetch_receipt_schema": 2,
    "production_analyzer_invoked": False,
    "submission_receipt_required": True,
    "terminal_receipt_schema": 2,
    "terminal_sacct_phase": "post_completion_local_finalize",
}
assert state["resource_accounting_contract"] == {
    "in_process": "python_resource_getrusage_self_and_monotonic_ns",
    "external_authority": "terminal_slurm_sacct",
    "host_gnu_time_required": False,
}
assert state["upstream_commit"] == "d053054c5290a04c1c4cd8b55704d999cad73e30"
assert state["environment_lock_sha256"] == os.environ["UED_TEST_LOCK_SHA"]
assert len(state["upstream_git_bundle_sha256"]) == 64
assert len(state["overlay_manifest_sha256"]) == 64
PY

source_bundle="$TMP/bundle/upstream/minimax-${PINNED_COMMIT}.bundle"
git clone --quiet "$source_bundle" "$TMP/source-faithful"
git clone --quiet "$source_bundle" "$TMP/patched"
[[ "$(git -C "$TMP/source-faithful" rev-parse HEAD)" == "$PINNED_COMMIT" ]]
[[ -z "$(git -C "$TMP/source-faithful" status --porcelain --untracked-files=all)" ]]

apply="$TMP/bundle/ued_benchmark/scripts/apply_minimax_overlay.py"
PYTHONDONTWRITEBYTECODE=1 python3 "$apply" --target "$TMP/patched" --check \
  > "$TMP/check.json"
PYTHONDONTWRITEBYTECODE=1 python3 "$apply" --target "$TMP/patched" --apply \
  > "$TMP/apply.json"
PYTHONDONTWRITEBYTECODE=1 python3 "$apply" --target "$TMP/patched" --check \
  > "$TMP/postcheck.json"
grep -Fq '"status": "applicable"' "$TMP/check.json"
grep -Fq '"status": "applied"' "$TMP/apply.json"
grep -Fq '"status": "already_applied"' "$TMP/postcheck.json"
[[ -f "$TMP/patched/src/minimax/util/rl/frontier_activity.py" ]]
git -C "$TMP/patched" diff --check
[[ -z "$(git -C "$TMP/source-faithful" status --porcelain --untracked-files=all)" ]]

# Runtime-equivalent check of the exact formula and one-JIT AMaze body uses the
# target CPython build and pinned JAX/NumPy on CPU. CUDA plugin/device checks
# necessarily remain inside the bounded Slurm smoke.
JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu \
LD_LIBRARY_PATH="$CPU_LIBRARY_PATH" \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TMP/patched/src:$CPU_SITE_PACKAGES" \
"$CPU_PYTHON" - <<'PY'
import importlib.metadata as metadata
import math
import platform

import jax
import jax.numpy as jnp
import minimax
from minimax.envs.maze.maze import Maze
from minimax.util.rl.frontier_activity import (
    beta_posterior_mean,
    coefficient_activity_score,
)

assert metadata.version("jax") == "0.4.31"
assert metadata.version("jaxlib") == "0.4.31"
assert metadata.version("numpy") == "1.25.2"
assert platform.python_version() == "3.10.20"
assert jax.default_backend() == "cpu"

p = float(beta_posterior_mean(2, 3, 1.0, 1.0))
expected_score = float(coefficient_activity_score(
    2, 3, 4, 1.0, 1.0, posterior_mode="expected_activity"))
mean_plugin_score = float(coefficient_activity_score(
    2, 3, 4, 1.0, 1.0, posterior_mode="mean_plugin"))
expected_mean_plugin = 1.0 - (1.0 - 0.6) ** 4 - 0.6
expected_posterior_activity = 1.0 - (2.0 * 3.0 * 4.0 * 5.0) / (
    5.0 * 6.0 * 7.0 * 8.0) - 3.0 / 5.0
assert math.isclose(p, 0.6, rel_tol=0.0, abs_tol=1e-6)
assert math.isclose(
    expected_score, expected_posterior_activity, rel_tol=0.0, abs_tol=1e-6)
assert math.isclose(
    mean_plugin_score, expected_mean_plugin, rel_tol=0.0, abs_tol=1e-6)
assert expected_score < mean_plugin_score

env = Maze(height=7, width=7, n_walls=5, max_episode_steps=16)

def amaze_formula_smoke(key):
    reset_key, step_key = jax.random.split(key)
    obs, state = env.reset_env(reset_key)
    obs2, state2, reward, done, _ = env.step_env(
        step_key, state, jnp.asarray(0, dtype=jnp.int32))
    activity = coefficient_activity_score(
        jnp.asarray(2), jnp.asarray(3), 4, 1.0, 1.0,
        posterior_mode="expected_activity")
    leaves = jax.tree_util.tree_leaves((obs, obs2))
    checksum = sum(
        (jnp.sum(jnp.asarray(leaf, dtype=jnp.float32)) for leaf in leaves),
        jnp.asarray(0.0, dtype=jnp.float32))
    return activity, checksum, reward, done, state2.time

compiled = jax.jit(amaze_formula_smoke)
activity, checksum, reward, done, next_time = compiled(jax.random.PRNGKey(0))
activity.block_until_ready()
assert math.isclose(
    float(activity), expected_posterior_activity, rel_tol=0.0, abs_tol=1e-6)
assert math.isfinite(float(checksum))
assert math.isfinite(float(reward))
assert int(next_time) == 1
PY

if command -v shellcheck >/dev/null 2>&1; then
  shellcheck -x "${files[@]}"
fi
printf 'UED_MINIMAX_LOCAL_CHECK_PASS\n'
