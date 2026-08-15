#!/usr/bin/env bash
#SBATCH --job-name=ued-v4-import
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/%u/maxrl/tests/logs/%x_%j.out

# Bounded engineering smoke only: source-faithful import, overlay formula, and
# one AMaze JIT. This script has no training invocation or paper endpoint.
set -euo pipefail
umask 027
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES LD_LIBRARY_PATH
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONOPTIMIZE LD_PRELOAD
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
declare -A expected_ued=()
for name in "${required[@]}"; do expected_ued["$name"]=1; done
while IFS= read -r name; do
  if [[ "$name" == UED_* && -z "${expected_ued[$name]:-}" ]]; then
    echo "unexpected UED variable: $name" >&2
    exit 2
  fi
done < <(compgen -e)
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
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo "arrays forbidden" >&2
  exit 2
}
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || {
  echo "restarted jobs forbidden" >&2
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
[[ "$(dirname -- "$UED_BUNDLE_DIR")" \
    == "$SCRATCH_ROOT/maxrl/bundles/ued_minimax_v4_engineering" ]]
[[ "$(basename -- "$UED_BUNDLE_DIR")" \
    == "${UED_BUNDLE_MANIFEST_SHA256:0:20}" ]]
readonly PY="$UED_ENV_DIR/bin/python"
[[ -x "$PY" ]] || { echo "missing environment Python" >&2; exit 1; }

# Reject unsafe manifest paths and, for source/result trees, any unlisted
# regular file or symlink before importing code from the tree.
verify_manifest() {
  local root=$1 name=$2 expected=$3 closure_mode=$4
  local manifest="$root/$name"
  [[ -f "$manifest" && ! -L "$manifest" ]]
  [[ "$(sha256sum "$manifest" | awk '{print $1}')" == "$expected" ]]
  UED_VERIFY_ROOT="$root" UED_VERIFY_MANIFEST="$manifest" \
  UED_VERIFY_CLOSURE="$closure_mode" "$PY" -I -B - <<'PY'
import os, re
from pathlib import Path, PurePosixPath
root = Path(os.environ["UED_VERIFY_ROOT"])
manifest = Path(os.environ["UED_VERIFY_MANIFEST"])
mode = os.environ["UED_VERIFY_CLOSURE"]
assert mode in {"exact", "result", "metadata"}
assert root.resolve() == root and not root.is_symlink()
seen = set()
for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\\]+)", raw)
    assert match, (number, raw)
    text = match.group(2); rel = PurePosixPath(text)
    assert text not in ("", ".") and not rel.is_absolute()
    assert all(part not in ("", ".", "..") for part in rel.parts)
    canonical = rel.as_posix()
    assert canonical not in seen
    seen.add(canonical)
    target = root.joinpath(*rel.parts)
    assert target.is_file() and not target.is_symlink(), target
    assert target.resolve().is_relative_to(root), target
assert seen
if mode != "metadata":
    actual = set()
    for target in root.rglob("*"):
        assert not target.is_symlink(), target
        if target.is_file():
            actual.add(target.relative_to(root).as_posix())
        else:
            assert target.is_dir(), target
    allowed_unlisted = {manifest.relative_to(root).as_posix()}
    if mode == "result":
        allowed_unlisted.add("COMPLETE")
    assert actual == seen | allowed_unlisted, (
        sorted(actual - seen - allowed_unlisted), sorted(seen - actual)
    )
PY
  (cd "$root" && sha256sum -c --strict "$name" >/dev/null)
}

readonly BUNDLE_MANIFEST="$UED_BUNDLE_DIR/SHA256SUMS"
verify_manifest "$UED_BUNDLE_DIR" SHA256SUMS "$UED_BUNDLE_MANIFEST_SHA256" exact

readonly STATE="$UED_BUNDLE_DIR/BUNDLE_STATE.json"
UED_STATE="$STATE" \
UED_EXPECTED_COMMIT="$UED_UPSTREAM_COMMIT" \
UED_EXPECTED_TREE="$UED_UPSTREAM_TREE" \
UED_EXPECTED_SOURCE="$UED_UPSTREAM_BUNDLE_SHA256" \
UED_EXPECTED_OVERLAY="$UED_OVERLAY_MANIFEST_SHA256" \
UED_EXPECTED_LOCK="$UED_ENV_LOCK_SHA256" \
"$PY" -I -B - <<'PY'
import json
import os

with open(os.environ["UED_STATE"], encoding="utf-8") as stream:
    state = json.load(stream)
assert state["bundle_schema"] == 5
assert state["paper_evidence"] is False
assert state["analyzer_eligible"] is False
assert state["endpoint_access_authorized"] is False
assert state["production_authorized"] is False
assert state["remote_stage_authorized"] is False
assert state["workspace_tree_exclusions"] == [
    "ued_benchmark/blackwell_probe/**",
    "ued_benchmark/blackwell_training_probe/**",
]
assert state["allowed_engineering_endpoints"] == [
    "v4_gpu_import_tie_formula_jit",
    "v4_frontier_grouped_one_update",
    "v4_frontier_terminal_chain_components",
    "v4_maxmc_terminal_chain_components",
]
assert state["max_student_updates"] == 1
terminal = state["terminal_chain_contract"]
assert terminal["analyzer_eligible"] is False
assert terminal["paper_evidence"] is False
assert terminal["production_analyzer_invoked"] is False
assert terminal["phase_a_submission_export"] == "explicit_ued_assignments_no_all_or_none"
assert terminal["phase_a_python_flags"] == terminal["phase_b_python_flags"] == "-I -B"
assert terminal["common_replay_snapshot"] == "plr-replay-snapshot.json for both arms"
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
assert state["protected_v3_remote_job"]["job_id"] == "9367063"
assert state["protected_v3_remote_job"]["reusable_as_v4_prerequisite"] is False
assert state["core_bindings"]["contract_sha256"] == (
    "3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b")
assert state["core_bindings"]["expected_applied_manifest_sha256"] == (
    "9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae")
PY

readonly UPSTREAM_BUNDLE_REL="upstream/minimax-${UED_UPSTREAM_COMMIT}.bundle"
readonly UPSTREAM_BUNDLE="$UED_BUNDLE_DIR/$UPSTREAM_BUNDLE_REL"
[[ -f "$UPSTREAM_BUNDLE" && ! -L "$UPSTREAM_BUNDLE" ]]
[[ "$(sha256sum "$UPSTREAM_BUNDLE" | awk '{print $1}')" == "$UED_UPSTREAM_BUNDLE_SHA256" ]]
readonly OVERLAY_MANIFEST="$UED_BUNDLE_DIR/ued_benchmark/OVERLAY_SHA256SUMS"
verify_manifest "$UED_BUNDLE_DIR/ued_benchmark" OVERLAY_SHA256SUMS \
  "$UED_OVERLAY_MANIFEST_SHA256" exact

readonly BUNDLED_SBATCH="$UED_BUNDLE_DIR/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch"
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
verify_manifest "$UED_ENV_DIR" ENVIRONMENT_SHA256SUMS \
  "$UED_ENV_MANIFEST_SHA256" metadata
"$PY" -I -B -m pip check
RECORDED_FREEZE=$(<"$UED_ENV_DIR/ENVIRONMENT.freeze")
CURRENT_FREEZE=$("$PY" -I -B -m pip freeze --all | LC_ALL=C sort)
[[ "$CURRENT_FREEZE" == "$RECORDED_FREEZE" ]] || {
  echo "installed packages differ from the submitted environment freeze" >&2
  exit 1
}
unset RECORDED_FREEZE CURRENT_FREEZE
readonly CONDA_BIN="/home/$RUN_USER/miniconda3/bin/conda"
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

readonly RESULT_DIR="$SCRATCH_ROOT/maxrl/tests/ued-minimax-v4-import/$SLURM_JOB_ID"
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
readonly WORK_DIR="$WORK_PARENT/ued-minimax-v4-import-$SLURM_JOB_ID"
[[ ! -e "$WORK_DIR" && ! -L "$WORK_DIR" ]] || {
  echo "work path exists: $WORK_DIR" >&2
  exit 2
}
mkdir "$WORK_DIR"
cleanup() {
  if [[ -n "${WORK_DIR:-}" \
        && "$WORK_DIR" == "$WORK_PARENT/ued-minimax-v4-import-$SLURM_JOB_ID" \
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
readonly PRISTINE_PLR_REL=src/minimax/config/configs/maze/plr.json
readonly PRISTINE_ACCEL_REL=src/minimax/config/configs/maze/accel.json
for clone in "$WORK_DIR/source-faithful" "$WORK_DIR/patched"; do
  [[ "$(sha256sum "$clone/$PRISTINE_PLR_REL" | awk '{print $1}')" \
      == a5b8b87799bce31564959b3e8b55cfdaba658b31b8135fa35e3b96704d65185b ]]
  [[ "$(sha256sum "$clone/$PRISTINE_ACCEL_REL" | awk '{print $1}')" \
      == d7f100195159baff4f7490b755903e7586c895918ade4a248f5ae42b1827aee5 ]]
done

UED_SOURCE_IMPORT_JSON="$RESULT_DIR/source-import.json" \
UED_SOURCE_DIR="$WORK_DIR/source-faithful" \
"$PY" -I -B - "$WORK_DIR/source-faithful/src" <<'PY'
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])

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

readonly APPLY="$UED_BUNDLE_DIR/ued_benchmark/scripts/apply_minimax_overlay_v4.py"
"$PY" -I -B "$APPLY" --target "$WORK_DIR/patched" --check \
  > "$RESULT_DIR/overlay-check.json"
"$PY" -I -B "$APPLY" --target "$WORK_DIR/patched" --apply \
  > "$RESULT_DIR/overlay-apply.json"
"$PY" -I -B "$APPLY" --target "$WORK_DIR/patched" --check \
  > "$RESULT_DIR/overlay-postcheck.json"
"$GIT" -C "$WORK_DIR/patched" diff --check
readonly APPLIED_MANIFEST="$WORK_DIR/patched/.frontierrl_overlay.json"
[[ -f "$APPLIED_MANIFEST" && ! -L "$APPLIED_MANIFEST" ]]
readonly APPLIED_MANIFEST_SHA256="$(sha256sum "$APPLIED_MANIFEST" | awk '{print $1}')"
[[ "$APPLIED_MANIFEST_SHA256" == 9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae ]]
[[ "$(sha256sum "$WORK_DIR/patched/$PRISTINE_PLR_REL" | awk '{print $1}')" \
    == a5b8b87799bce31564959b3e8b55cfdaba658b31b8135fa35e3b96704d65185b ]]
[[ "$(sha256sum "$WORK_DIR/patched/$PRISTINE_ACCEL_REL" | awk '{print $1}')" \
    == d7f100195159baff4f7490b755903e7586c895918ade4a248f5ae42b1827aee5 ]]
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
"$PY" -I -B - "$WORK_DIR/patched/src" \
  "$UED_BUNDLE_DIR/ued_benchmark/configs/maze_frontier_exact_grouped_n8_tie_aware_v4.json" \
  "$UED_BUNDLE_DIR/ued_benchmark/configs/maze_maxmc_group_matched_4x8_b500_tie_aware_v4.json" <<'PY'
import importlib.metadata as metadata
import json
import math
import os
import platform
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])

import jax
import jax.numpy as jnp
import minimax
from minimax.envs.maze.maze import Maze
from minimax.util.rl.frontier_activity import (
    beta_posterior_mean,
    coefficient_activity_score,
)
from minimax.util.rl.tie_aware_rank import (
    effective_support,
    exact_score_tie_diagnostics,
    tie_aware_rank_mass,
)

patched = Path(os.environ["UED_PATCHED_DIR"]).resolve()
module_path = Path(minimax.__file__).resolve()
assert module_path.is_relative_to(patched), (module_path, patched)
devices = jax.devices("gpu")
assert len(devices) == 1, devices

frontier_config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["args"]
maxmc_config = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))["args"]
for config, score in ((frontier_config, "coefficient_activity"), (maxmc_config, "max_mc")):
    temperature = float(config["plr_temp"][0])
    assert math.isfinite(temperature) and temperature > 0.0
    assert config["ued_score"] == [score]
    assert config["plr_tie_aware_score_ranks"] == [True]
    assert config["n_parallel"] == [4] and config["n_eval"] == [8]

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
    tie_scores = jnp.asarray([1.0, 1.0, 0.0, 99.0], dtype=jnp.float32)
    tie_filled = jnp.asarray([True, True, True, False])
    tie_mass = tie_aware_rank_mass(tie_scores, tie_filled, 0.3)
    tie_diagnostics = exact_score_tie_diagnostics(tie_scores, tie_filled)
    tie_support = effective_support(tie_mass/tie_mass.sum())
    # AMaze observations are OrderedDict pytrees (image + agent_dir), not a
    # single ndarray. Summing every leaf catches shape/dtype/JIT regressions.
    leaves = jax.tree_util.tree_leaves((obs, obs2))
    checksum = sum(
        (jnp.sum(jnp.asarray(leaf, dtype=jnp.float32)) for leaf in leaves),
        jnp.asarray(0.0, dtype=jnp.float32))
    return activity, checksum, reward, done, state2.time, tie_mass, tie_support, tie_diagnostics

compiled = jax.jit(amaze_formula_smoke)
activity, checksum, reward, done, next_time, tie_mass, tie_support, tie_diagnostics = compiled(jax.random.PRNGKey(0))
activity.block_until_ready()
assert math.isclose(
    float(activity), expected_posterior_activity, rel_tol=0.0, abs_tol=1e-6)
assert math.isfinite(float(checksum))
assert int(next_time) == 1
assert bool(jnp.isfinite(tie_mass).all())
assert float(tie_mass[0]) == float(tie_mass[1])
assert float(tie_mass[3]) == 0.0
assert float(tie_support) > 1.0
assert int(tie_diagnostics["distinct_filled_score_count"]) == 2
assert int(tie_diagnostics["score_max_tie_block_size"]) == 2

with open(os.environ["UED_SOURCE_IMPORT_JSON"], encoding="utf-8") as stream:
    source_import = json.load(stream)
record = {
    "runtime_schema": 1,
    "purpose": "bounded tie-aware v4 UED import/formula/JIT engineering smoke",
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
    "official_pristine_configs": {
        "plr_sha256": "a5b8b87799bce31564959b3e8b55cfdaba658b31b8135fa35e3b96704d65185b",
        "accel_sha256": "d7f100195159baff4f7490b755903e7586c895918ade4a248f5ae42b1827aee5",
        "unchanged_in_patched_clone": True,
        "execution_in_v4_rungs": False,
    },
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
        "tie_aware_rank_passed": True,
        "tie_effective_support": float(tie_support),
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
readonly RESULT_MANIFEST_SHA256="$(sha256sum "$RESULT_DIR/SHA256SUMS" | awk '{print $1}')"
printf 'complete\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$RESULT_DIR/COMPLETE"
verify_manifest "$RESULT_DIR" SHA256SUMS "$RESULT_MANIFEST_SHA256" result
echo "UED_MINIMAX_V4_GPU_SMOKE_PASS $RESULT_DIR"
