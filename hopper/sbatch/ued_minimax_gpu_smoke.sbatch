#!/usr/bin/env bash
#SBATCH --job-name=ued-minimax-gpu-smoke
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/%u/maxrl/tests/logs/%x_%j.out

# Bounded engineering smoke only: source-faithful import, overlay formula, and
# one AMaze JIT. This script has no training invocation or paper endpoint.
set -euo pipefail
umask 027
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1

required=(
  UED_BUNDLE_DIR
  UED_BUNDLE_MANIFEST_SHA256
  UED_UPSTREAM_COMMIT
  UED_UPSTREAM_TREE
  UED_UPSTREAM_BUNDLE_SHA256
  UED_OVERLAY_MANIFEST_SHA256
  UED_SBATCH_SHA256
  UED_ENV_DIR
  UED_ENV_LOCK_SHA256
  UED_ENV_FREEZE_SHA256
  UED_ENV_MANIFEST_SHA256
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing required variable: $name" >&2; exit 2; }
done
for name in UED_BUNDLE_MANIFEST_SHA256 UED_UPSTREAM_BUNDLE_SHA256 \
            UED_OVERLAY_MANIFEST_SHA256 UED_SBATCH_SHA256 UED_ENV_LOCK_SHA256 \
            UED_ENV_FREEZE_SHA256 UED_ENV_MANIFEST_SHA256; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "invalid SHA-256 in $name" >&2
    exit 2
  }
done
[[ "$UED_UPSTREAM_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$UED_UPSTREAM_TREE" =~ ^[0-9a-f]{40}$ ]]
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || {
  echo "invalid or missing SLURM_JOB_ID" >&2
  exit 2
}

readonly RUN_USER="$(id -un)"
[[ "$RUN_USER" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe runtime user" >&2; exit 2; }
readonly SCRATCH_ROOT="/scratch/$RUN_USER"
[[ "$UED_BUNDLE_DIR" =~ ^/scratch/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]] \
  || { echo "unsafe bundle path syntax" >&2; exit 2; }
[[ "$UED_ENV_DIR" =~ ^/scratch/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]] \
  || { echo "unsafe environment path syntax" >&2; exit 2; }
case "$UED_BUNDLE_DIR" in "$SCRATCH_ROOT"/*) ;; *) echo "unsafe bundle path" >&2; exit 2;; esac
case "$UED_ENV_DIR" in "$SCRATCH_ROOT"/*) ;; *) echo "unsafe environment path" >&2; exit 2;; esac
[[ -d "$UED_BUNDLE_DIR" && ! -L "$UED_BUNDLE_DIR" ]]
[[ -d "$UED_ENV_DIR" && ! -L "$UED_ENV_DIR" ]]

readonly BUNDLE_MANIFEST="$UED_BUNDLE_DIR/SHA256SUMS"
[[ -f "$BUNDLE_MANIFEST" && ! -L "$BUNDLE_MANIFEST" ]]
[[ "$(sha256sum "$BUNDLE_MANIFEST" | awk '{print $1}')" == "$UED_BUNDLE_MANIFEST_SHA256" ]]
(cd "$UED_BUNDLE_DIR" && sha256sum -c --strict SHA256SUMS >/dev/null)

readonly STATE="$UED_BUNDLE_DIR/BUNDLE_STATE.json"
UED_STATE="$STATE" \
UED_EXPECTED_COMMIT="$UED_UPSTREAM_COMMIT" \
UED_EXPECTED_TREE="$UED_UPSTREAM_TREE" \
UED_EXPECTED_SOURCE="$UED_UPSTREAM_BUNDLE_SHA256" \
UED_EXPECTED_OVERLAY="$UED_OVERLAY_MANIFEST_SHA256" \
UED_EXPECTED_LOCK="$UED_ENV_LOCK_SHA256" \
"$UED_ENV_DIR/bin/python" - <<'PY'
import json
import os

with open(os.environ["UED_STATE"], encoding="utf-8") as stream:
    state = json.load(stream)
assert state["bundle_schema"] == 4
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
assert state["upstream_commit"] == os.environ["UED_EXPECTED_COMMIT"]
assert state["upstream_tree_git_sha1"] == os.environ["UED_EXPECTED_TREE"]
assert state["upstream_git_bundle_sha256"] == os.environ["UED_EXPECTED_SOURCE"]
assert state["overlay_manifest_sha256"] == os.environ["UED_EXPECTED_OVERLAY"]
assert state["environment_lock_sha256"] == os.environ["UED_EXPECTED_LOCK"]
PY

readonly UPSTREAM_BUNDLE_REL="upstream/minimax-${UED_UPSTREAM_COMMIT}.bundle"
readonly UPSTREAM_BUNDLE="$UED_BUNDLE_DIR/$UPSTREAM_BUNDLE_REL"
[[ -f "$UPSTREAM_BUNDLE" && ! -L "$UPSTREAM_BUNDLE" ]]
[[ "$(sha256sum "$UPSTREAM_BUNDLE" | awk '{print $1}')" == "$UED_UPSTREAM_BUNDLE_SHA256" ]]
readonly OVERLAY_MANIFEST="$UED_BUNDLE_DIR/ued_benchmark/OVERLAY_SHA256SUMS"
[[ -f "$OVERLAY_MANIFEST" && ! -L "$OVERLAY_MANIFEST" ]]
[[ "$(sha256sum "$OVERLAY_MANIFEST" | awk '{print $1}')" == "$UED_OVERLAY_MANIFEST_SHA256" ]]
(cd "$UED_BUNDLE_DIR/ued_benchmark" && \
  sha256sum -c --strict OVERLAY_SHA256SUMS >/dev/null)

readonly BUNDLED_SBATCH="$UED_BUNDLE_DIR/hopper/sbatch/ued_minimax_gpu_smoke.sbatch"
[[ -f "$BUNDLED_SBATCH" && ! -L "$BUNDLED_SBATCH" ]]
[[ "$(sha256sum "$BUNDLED_SBATCH" | awk '{print $1}')" == "$UED_SBATCH_SHA256" ]]
[[ "$(sha256sum "$0" | awk '{print $1}')" == "$UED_SBATCH_SHA256" ]]
readonly BUNDLED_SETUP="$UED_BUNDLE_DIR/hopper/setup_ued_minimax_env.sh"
[[ -f "$BUNDLED_SETUP" && ! -L "$BUNDLED_SETUP" ]]
[[ -f "$UED_ENV_DIR/SETUP_SHA256" && ! -L "$UED_ENV_DIR/SETUP_SHA256" ]]
[[ -f "$UED_ENV_DIR/ENVIRONMENT_COMPLETE" \
   && ! -L "$UED_ENV_DIR/ENVIRONMENT_COMPLETE" ]]
readonly ENV_SETUP_SHA256="$(awk 'NR == 1 {print $1}' "$UED_ENV_DIR/SETUP_SHA256")"
[[ "$ENV_SETUP_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$(sha256sum "$BUNDLED_SETUP" | awk '{print $1}')" == "$ENV_SETUP_SHA256" ]]

readonly BUNDLED_LOCK="$UED_BUNDLE_DIR/hopper/requirements-ued-minimax-hopper.lock"
[[ "$(sha256sum "$BUNDLED_LOCK" | awk '{print $1}')" == "$UED_ENV_LOCK_SHA256" ]]
[[ "$(awk 'NR == 1 {print $1}' "$UED_ENV_DIR/LOCK_SHA256")" == "$UED_ENV_LOCK_SHA256" ]]
[[ "$(sha256sum "$UED_ENV_DIR/ENVIRONMENT.freeze" | awk '{print $1}')" == "$UED_ENV_FREEZE_SHA256" ]]
[[ "$(sha256sum "$UED_ENV_DIR/ENVIRONMENT_SHA256SUMS" | awk '{print $1}')" == "$UED_ENV_MANIFEST_SHA256" ]]
readonly COMPLETE_SCHEMA="$(awk -F '\t' '$1 == "environment_schema" {print $2}' "$UED_ENV_DIR/ENVIRONMENT_COMPLETE")"
readonly COMPLETE_MANIFEST="$(awk -F '\t' '$1 == "manifest_sha256" {print $2}' "$UED_ENV_DIR/ENVIRONMENT_COMPLETE")"
readonly COMPLETE_SETUP="$(awk -F '\t' '$1 == "setup_script_sha256" {print $2}' "$UED_ENV_DIR/ENVIRONMENT_COMPLETE")"
[[ "$COMPLETE_SCHEMA" == 2 \
   && "$COMPLETE_MANIFEST" == "$UED_ENV_MANIFEST_SHA256" \
   && "$COMPLETE_SETUP" == "$ENV_SETUP_SHA256" ]]
(cd "$UED_ENV_DIR" && sha256sum -c --strict ENVIRONMENT_SHA256SUMS >/dev/null)
"$UED_ENV_DIR/bin/python" -m pip check
RECORDED_FREEZE=$(<"$UED_ENV_DIR/ENVIRONMENT.freeze")
CURRENT_FREEZE=$("$UED_ENV_DIR/bin/python" -m pip freeze --all | LC_ALL=C sort)
[[ "$CURRENT_FREEZE" == "$RECORDED_FREEZE" ]] || {
  echo "installed packages differ from the submitted environment freeze" >&2
  exit 1
}
unset RECORDED_FREEZE CURRENT_FREEZE
readonly CONDA_BIN="${UED_CONDA_BIN:-$HOME/miniconda3/bin/conda}"
[[ -x "$CONDA_BIN" ]] || { echo "missing Conda executable: $CONDA_BIN" >&2; exit 1; }
RECORDED_CONDA=$(<"$UED_ENV_DIR/CONDA_EXPLICIT.txt")
CURRENT_CONDA=$("$CONDA_BIN" list --explicit -p "$UED_ENV_DIR")
[[ "$CURRENT_CONDA" == "$RECORDED_CONDA" ]] || {
  echo "installed Conda packages differ from submitted explicit spec" >&2
  exit 1
}
unset RECORDED_CONDA CURRENT_CONDA
readonly CONDA_EXPLICIT_SHA256="$(sha256sum "$UED_ENV_DIR/CONDA_EXPLICIT.txt" | awk '{print $1}')"
readonly ENVIRONMENT_JSON_SHA256="$(sha256sum "$UED_ENV_DIR/ENVIRONMENT.json" | awk '{print $1}')"
readonly GIT="$UED_ENV_DIR/bin/git"
[[ -x "$GIT" ]]
readonly GIT_VERSION="$("$GIT" --version)"
[[ "$GIT_VERSION" == "git version 2.45.2" ]]
export PATH="$UED_ENV_DIR/bin:/usr/bin:/bin"
[[ "$(command -v git)" == "$GIT" ]]

readonly RESULT_DIR="$SCRATCH_ROOT/maxrl/tests/ued-minimax-gpu-smoke/$SLURM_JOB_ID"
[[ ! -e "$RESULT_DIR" && ! -L "$RESULT_DIR" ]] || {
  echo "result exists: $RESULT_DIR" >&2
  exit 2
}
mkdir -p "$RESULT_DIR"
nvidia-smi > "$RESULT_DIR/nvidia-smi.txt"
cp -- "$STATE" "$RESULT_DIR/bundle-state.json"
mkdir "$RESULT_DIR/environment"
cp -- "$UED_ENV_DIR/CONDA_EXPLICIT.txt" \
      "$UED_ENV_DIR/ENVIRONMENT.freeze" \
      "$UED_ENV_DIR/ENVIRONMENT.json" \
      "$UED_ENV_DIR/ENVIRONMENT_SHA256SUMS" \
      "$UED_ENV_DIR/ENVIRONMENT_COMPLETE" \
      "$UED_ENV_DIR/LOCK_SHA256" \
      "$UED_ENV_DIR/SETUP_SHA256" \
      "$RESULT_DIR/environment/"
(cd "$RESULT_DIR/environment" && \
  sha256sum -c --strict ENVIRONMENT_SHA256SUMS >/dev/null)

readonly WORK_PARENT="$SCRATCH_ROOT/maxrl/tests/work"
mkdir -p "$WORK_PARENT"
readonly WORK_DIR="$WORK_PARENT/ued-minimax-$SLURM_JOB_ID"
[[ ! -e "$WORK_DIR" && ! -L "$WORK_DIR" ]] || {
  echo "work path exists: $WORK_DIR" >&2
  exit 2
}
mkdir "$WORK_DIR"
cleanup() {
  if [[ -n "${WORK_DIR:-}" \
        && "$WORK_DIR" == "$WORK_PARENT/ued-minimax-$SLURM_JOB_ID" \
        && -d "$WORK_DIR" ]]; then
    rm -rf -- "$WORK_DIR"
  fi
}
trap cleanup EXIT

"$GIT" clone --quiet "$UPSTREAM_BUNDLE" "$WORK_DIR/source-faithful"
"$GIT" clone --quiet "$UPSTREAM_BUNDLE" "$WORK_DIR/patched"
for clone in "$WORK_DIR/source-faithful" "$WORK_DIR/patched"; do
  [[ "$("$GIT" -C "$clone" rev-parse HEAD)" == "$UED_UPSTREAM_COMMIT" ]]
  [[ "$("$GIT" -C "$clone" rev-parse 'HEAD^{tree}')" == "$UED_UPSTREAM_TREE" ]]
  [[ -z "$("$GIT" -C "$clone" status --porcelain --untracked-files=all)" ]]
done

readonly PY="$UED_ENV_DIR/bin/python"
UED_SOURCE_IMPORT_JSON="$RESULT_DIR/source-import.json" \
UED_SOURCE_DIR="$WORK_DIR/source-faithful" \
PYTHONPATH="$WORK_DIR/source-faithful/src" \
"$PY" - <<'PY'
import importlib.metadata as metadata
import json
import os
from pathlib import Path

import jax
import minimax
from minimax.envs.maze.maze import Maze

source = Path(os.environ["UED_SOURCE_DIR"]).resolve()
module_path = Path(minimax.__file__).resolve()
assert module_path.is_relative_to(source), (module_path, source)
devices = jax.devices("gpu")
assert len(devices) == 1, devices
assert Maze(height=7, width=7, n_walls=5).name == "Maze"
record = {
    "source_faithful_import": True,
    "minimax_module": str(module_path),
    "jax": metadata.version("jax"),
    "jaxlib": metadata.version("jaxlib"),
    "gpu_devices": [str(device) for device in devices],
}
Path(os.environ["UED_SOURCE_IMPORT_JSON"]).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
"$GIT" -C "$WORK_DIR/source-faithful" status --porcelain --untracked-files=all \
  > "$RESULT_DIR/source-faithful.status"
[[ ! -s "$RESULT_DIR/source-faithful.status" ]]
"$GIT" -C "$WORK_DIR/source-faithful" diff --quiet

readonly APPLY="$UED_BUNDLE_DIR/ued_benchmark/scripts/apply_minimax_overlay.py"
"$PY" "$APPLY" --target "$WORK_DIR/patched" --check \
  > "$RESULT_DIR/overlay-check.json"
"$PY" "$APPLY" --target "$WORK_DIR/patched" --apply \
  > "$RESULT_DIR/overlay-apply.json"
"$PY" "$APPLY" --target "$WORK_DIR/patched" --check \
  > "$RESULT_DIR/overlay-postcheck.json"
"$GIT" -C "$WORK_DIR/patched" diff --check
readonly APPLIED_MANIFEST="$WORK_DIR/patched/.frontierrl_overlay.json"
[[ -f "$APPLIED_MANIFEST" && ! -L "$APPLIED_MANIFEST" ]]
readonly APPLIED_MANIFEST_SHA256="$(sha256sum "$APPLIED_MANIFEST" | awk '{print $1}')"
cp -- "$APPLIED_MANIFEST" "$RESULT_DIR/applied-overlay-manifest.json"
[[ "$(sha256sum "$RESULT_DIR/applied-overlay-manifest.json" | awk '{print $1}')" \
    == "$APPLIED_MANIFEST_SHA256" ]]

UED_RUNTIME_JSON="$RESULT_DIR/runtime.json" \
UED_SOURCE_IMPORT_JSON="$RESULT_DIR/source-import.json" \
UED_PATCHED_DIR="$WORK_DIR/patched" \
UED_BUNDLE_MANIFEST_SHA256="$UED_BUNDLE_MANIFEST_SHA256" \
UED_UPSTREAM_COMMIT="$UED_UPSTREAM_COMMIT" \
UED_UPSTREAM_TREE="$UED_UPSTREAM_TREE" \
UED_UPSTREAM_BUNDLE_SHA256="$UED_UPSTREAM_BUNDLE_SHA256" \
UED_OVERLAY_MANIFEST_SHA256="$UED_OVERLAY_MANIFEST_SHA256" \
UED_APPLIED_MANIFEST_SHA256="$APPLIED_MANIFEST_SHA256" \
UED_SBATCH_SHA256="$UED_SBATCH_SHA256" \
UED_ENV_LOCK_SHA256="$UED_ENV_LOCK_SHA256" \
UED_ENV_FREEZE_SHA256="$UED_ENV_FREEZE_SHA256" \
UED_ENV_MANIFEST_SHA256="$UED_ENV_MANIFEST_SHA256" \
UED_ENV_SETUP_SHA256="$ENV_SETUP_SHA256" \
UED_CONDA_EXPLICIT_SHA256="$CONDA_EXPLICIT_SHA256" \
UED_ENVIRONMENT_JSON_SHA256="$ENVIRONMENT_JSON_SHA256" \
UED_GIT_VERSION="$GIT_VERSION" \
PYTHONPATH="$WORK_DIR/patched/src" \
"$PY" - <<'PY'
import importlib.metadata as metadata
import json
import math
import os
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import minimax
from minimax.envs.maze.maze import Maze
from minimax.util.rl.frontier_activity import (
    beta_posterior_mean,
    coefficient_activity_score,
)

patched = Path(os.environ["UED_PATCHED_DIR"]).resolve()
module_path = Path(minimax.__file__).resolve()
assert module_path.is_relative_to(patched), (module_path, patched)
devices = jax.devices("gpu")
assert len(devices) == 1, devices

# Exact non-JIT formula checks. The posterior is Beta(3,2), so E[p]=0.6,
# u_4(E[p])=0.3744, and E[u_4(p)]=1-(2)_4/(5)_4-3/5.
p = float(beta_posterior_mean(2, 3, 1.0, 1.0))
expected_score = float(coefficient_activity_score(
    2, 3, 4, 1.0, 1.0, posterior_mode="expected_activity"))
mean_plugin_score = float(coefficient_activity_score(
    2, 3, 4, 1.0, 1.0, posterior_mode="mean_plugin"))
expected_mean_plugin = 1.0 - (1.0 - 0.6) ** 4 - 0.6
expected_posterior_activity = 1.0 - (2.0 * 3.0 * 4.0 * 5.0) / (
    5.0 * 6.0 * 7.0 * 8.0) - 3.0 / 5.0
assert math.isclose(p, 0.6, rel_tol=0.0, abs_tol=1e-6), p
assert math.isclose(
    mean_plugin_score, expected_mean_plugin, rel_tol=0.0, abs_tol=1e-6), (
        mean_plugin_score, expected_mean_plugin)
assert math.isclose(
    expected_score, expected_posterior_activity, rel_tol=0.0, abs_tol=1e-6), (
        expected_score, expected_posterior_activity)
assert expected_score < mean_plugin_score

# Exactly one explicit smoke JIT combines an AMaze reset/step and the Frontier
# formula. It is deliberately tiny and performs no PPO update or training.
env = Maze(height=7, width=7, n_walls=5, max_episode_steps=16)

def amaze_formula_smoke(key):
    reset_key, step_key = jax.random.split(key)
    obs, state = env.reset_env(reset_key)
    obs2, state2, reward, done, _ = env.step_env(
        step_key, state, jnp.asarray(0, dtype=jnp.int32))
    activity = coefficient_activity_score(
        jnp.asarray(2), jnp.asarray(3), 4, 1.0, 1.0,
        posterior_mode="expected_activity")
    # AMaze observations are OrderedDict pytrees (image + agent_dir), not a
    # single ndarray. Summing every leaf catches shape/dtype/JIT regressions.
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
assert int(next_time) == 1

with open(os.environ["UED_SOURCE_IMPORT_JSON"], encoding="utf-8") as stream:
    source_import = json.load(stream)
record = {
    "runtime_schema": 1,
    "purpose": "bounded UED minimax/AMaze engineering smoke",
    "training_endpoint": False,
    "job_id": os.environ["SLURM_JOB_ID"],
    "hostname": platform.node(),
    "python": platform.python_version(),
    "jax": metadata.version("jax"),
    "jaxlib": metadata.version("jaxlib"),
    "jax_cuda12_plugin": metadata.version("jax-cuda12-plugin"),
    "jax_cuda12_pjrt": metadata.version("jax-cuda12-pjrt"),
    "git": os.environ["UED_GIT_VERSION"],
    "backend": jax.default_backend(),
    "devices": [
        {"id": device.id, "platform": device.platform,
         "device_kind": device.device_kind}
        for device in devices
    ],
    "source_faithful_import": source_import,
    "patched_minimax_module": str(module_path),
    "formula": {
        "successes": 2,
        "trials": 3,
        "prior_alpha": 1.0,
        "prior_beta": 1.0,
        "frontier_n": 4,
        "posterior_p": p,
        "default_posterior_mode": "expected_activity",
        "expected_activity_score": expected_score,
        "expected_activity_analytic": expected_posterior_activity,
        "mean_plugin_score": mean_plugin_score,
        "mean_plugin_analytic": expected_mean_plugin,
    },
    "one_jit": {
        "passed": True,
        "amaze_height": 7,
        "amaze_width": 7,
        "amaze_n_walls": 5,
        "observation_shape": list(env.obs_shape),
        "checksum": float(checksum),
        "reward": float(reward),
        "done": bool(done),
        "next_time": int(next_time),
    },
    "hashes": {
        "bundle_manifest_sha256": os.environ["UED_BUNDLE_MANIFEST_SHA256"],
        "upstream_commit": os.environ["UED_UPSTREAM_COMMIT"],
        "upstream_tree_git_sha1": os.environ["UED_UPSTREAM_TREE"],
        "upstream_git_bundle_sha256": os.environ["UED_UPSTREAM_BUNDLE_SHA256"],
        "overlay_manifest_sha256": os.environ["UED_OVERLAY_MANIFEST_SHA256"],
        "applied_overlay_manifest_sha256": os.environ["UED_APPLIED_MANIFEST_SHA256"],
        "sbatch_sha256": os.environ["UED_SBATCH_SHA256"],
        "environment_lock_sha256": os.environ["UED_ENV_LOCK_SHA256"],
        "environment_freeze_sha256": os.environ["UED_ENV_FREEZE_SHA256"],
        "environment_manifest_sha256": os.environ["UED_ENV_MANIFEST_SHA256"],
        "environment_setup_script_sha256": os.environ["UED_ENV_SETUP_SHA256"],
        "conda_explicit_sha256": os.environ["UED_CONDA_EXPLICIT_SHA256"],
        "environment_json_sha256": os.environ["UED_ENVIRONMENT_JSON_SHA256"],
    },
}
Path(os.environ["UED_RUNTIME_JSON"]).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

{
  printf 'field\tvalue\n'
  printf 'job_id\t%s\n' "$SLURM_JOB_ID"
  printf 'utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host\t%s\n' "$(hostname)"
  printf 'result_dir\t%s\n' "$RESULT_DIR"
  printf 'bundle_manifest_sha256\t%s\n' "$UED_BUNDLE_MANIFEST_SHA256"
  printf 'upstream_commit\t%s\n' "$UED_UPSTREAM_COMMIT"
  printf 'upstream_tree_git_sha1\t%s\n' "$UED_UPSTREAM_TREE"
  printf 'upstream_git_bundle_sha256\t%s\n' "$UED_UPSTREAM_BUNDLE_SHA256"
  printf 'overlay_manifest_sha256\t%s\n' "$UED_OVERLAY_MANIFEST_SHA256"
  printf 'applied_overlay_manifest_sha256\t%s\n' "$APPLIED_MANIFEST_SHA256"
  printf 'sbatch_sha256\t%s\n' "$UED_SBATCH_SHA256"
  printf 'environment_lock_sha256\t%s\n' "$UED_ENV_LOCK_SHA256"
  printf 'environment_freeze_sha256\t%s\n' "$UED_ENV_FREEZE_SHA256"
  printf 'environment_manifest_sha256\t%s\n' "$UED_ENV_MANIFEST_SHA256"
  printf 'environment_setup_script_sha256\t%s\n' "$ENV_SETUP_SHA256"
  printf 'conda_explicit_sha256\t%s\n' "$CONDA_EXPLICIT_SHA256"
  printf 'environment_json_sha256\t%s\n' "$ENVIRONMENT_JSON_SHA256"
  printf 'git\t%s\n' "$GIT_VERSION"
  printf 'training_endpoint\tfalse\n'
} > "$RESULT_DIR/receipt.tsv"
(
  cd "$RESULT_DIR"
  find . -type f ! -name SHA256SUMS ! -name COMPLETE -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c --strict SHA256SUMS >/dev/null
)
printf 'complete\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$RESULT_DIR/COMPLETE"
echo "UED_MINIMAX_GPU_SMOKE_PASS $RESULT_DIR"
