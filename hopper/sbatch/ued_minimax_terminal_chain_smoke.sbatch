#!/bin/bash
#SBATCH --job-name=ued-minimax-terminal-chain
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/%u/maxrl/tests/logs/%x_%j.out
#SBATCH --error=/scratch/%u/maxrl/tests/logs/%x_%j.err

# Phase A of a permanently non-evidence engineering smoke.  It runs exactly
# one Frontier PPO update and one actual external 30-episode evaluation, then
# atomically publishes closed source components.  It deliberately does not run
# the assembler: authoritative terminal sacct does not exist until this job
# exits, so assembly is a separate post-terminal local phase.
set -euo pipefail
umask 027
export PATH=/usr/bin:/bin
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES LD_LIBRARY_PATH LD_PRELOAD
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONOPTIMIZE
unset JAX_DISABLE_JIT JAX_ENABLE_X64 JAX_DEFAULT_MATMUL_PRECISION
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS NVIDIA_TF32_OVERRIDE
unset XLA_PYTHON_CLIENT_MEM_FRACTION XLA_PYTHON_CLIENT_ALLOCATOR
unset CUDA_LAUNCH_BLOCKING TF_ENABLE_ONEDNN_OPTS
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

readonly EXPECTED_COMMIT=d053054c5290a04c1c4cd8b55704d999cad73e30
readonly EXPECTED_TREE=b0cace1fc54984e21a842f12d15d0b899e33d270
readonly EXPECTED_CONTRACT=5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000
readonly ENGINEERING_SEED=101
readonly ARM=frontier

required=(
  UED_BUNDLE_DIR UED_BUNDLE_MANIFEST_SHA256
  UED_UPSTREAM_COMMIT UED_UPSTREAM_TREE UED_UPSTREAM_BUNDLE_SHA256
  UED_OVERLAY_MANIFEST_SHA256 UED_TERMINAL_CHAIN_SBATCH_SHA256
  UED_FRONTIER_CONFIG_SHA256 UED_CONTRACT_SHA256
  UED_PROTOCOL_SHA256 UED_ANALYZER_SHA256 UED_TRAINING_DRIVER_SHA256
  UED_EVALUATION_DRIVER_SHA256 UED_ASSEMBLER_SHA256
  UED_ENV_DIR UED_ENV_LOCK_SHA256 UED_ENV_FREEZE_SHA256 UED_ENV_MANIFEST_SHA256
  UED_IMPORT_SMOKE_RESULT_DIR UED_IMPORT_SMOKE_MANIFEST_SHA256
  UED_ONE_UPDATE_RESULT_DIR UED_ONE_UPDATE_MANIFEST_SHA256
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing required variable: $name" >&2; exit 2; }
done
for name in UED_BUNDLE_MANIFEST_SHA256 UED_UPSTREAM_BUNDLE_SHA256 \
            UED_OVERLAY_MANIFEST_SHA256 UED_TERMINAL_CHAIN_SBATCH_SHA256 \
            UED_FRONTIER_CONFIG_SHA256 UED_CONTRACT_SHA256 UED_PROTOCOL_SHA256 \
            UED_ANALYZER_SHA256 UED_TRAINING_DRIVER_SHA256 \
            UED_EVALUATION_DRIVER_SHA256 UED_ASSEMBLER_SHA256 \
            UED_ENV_LOCK_SHA256 UED_ENV_FREEZE_SHA256 UED_ENV_MANIFEST_SHA256 \
            UED_IMPORT_SMOKE_MANIFEST_SHA256 UED_ONE_UPDATE_MANIFEST_SHA256; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "invalid SHA-256 in $name" >&2
    exit 2
  }
done
[[ "$UED_UPSTREAM_COMMIT" == "$EXPECTED_COMMIT" ]]
[[ "$UED_UPSTREAM_TREE" == "$EXPECTED_TREE" ]]
[[ "$UED_CONTRACT_SHA256" == "$EXPECTED_CONTRACT" ]]
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo "invalid SLURM_JOB_ID" >&2; exit 2; }
[[ "${SLURM_CPUS_PER_TASK:-}" == 2 && "${SLURM_JOB_NUM_NODES:-}" == 1 ]]
[[ "${SLURM_JOB_PARTITION:-}" == gpuq ]]
[[ "${SLURM_NTASKS:-}" == 1 ]]
[[ "${SLURM_MEM_PER_NODE:-}" == 15360 ]]
[[ "${SLURM_RESTART_COUNT:-}" == 0 ]]

readonly RUN_USER="$(id -un)"
[[ "$RUN_USER" =~ ^[A-Za-z0-9._-]+$ && "$RUN_USER" != . && "$RUN_USER" != .. ]]
readonly SCRATCH_ROOT="/scratch/$RUN_USER"
require_canonical_dir() {
  local path=$1 prefix=$2
  [[ "$path" == /* && -d "$path" && ! -L "$path" ]] || return 1
  [[ "$(readlink -f -- "$path")" == "$path" ]] || return 1
  case "$path" in "$prefix"/*) ;; *) return 1 ;; esac
}
require_canonical_dir "$UED_BUNDLE_DIR" "$SCRATCH_ROOT"
require_canonical_dir "$UED_ENV_DIR" "$SCRATCH_ROOT"
require_canonical_dir "$UED_IMPORT_SMOKE_RESULT_DIR" \
  "$SCRATCH_ROOT/maxrl/tests/ued-minimax-gpu-smoke"
require_canonical_dir "$UED_ONE_UPDATE_RESULT_DIR" \
  "$SCRATCH_ROOT/maxrl/tests/ued-minimax-one-update"

readonly PY="$UED_ENV_DIR/bin/python"
readonly GIT="$UED_ENV_DIR/bin/git"
[[ -x "$PY" && -x "$GIT" ]]
readonly GIT_VERSION="$("$GIT" --version)"
[[ "$GIT_VERSION" == "git version 2.45.2" ]]
export PATH="$UED_ENV_DIR/bin:/usr/bin:/bin"
[[ "$(command -v git)" == "$GIT" ]]
UED_EXPECTED_PY="$PY" "$PY" -I -B - <<'PY'
import os
import platform
from pathlib import Path
import site
import sys

assert Path(sys.executable).resolve() == Path(os.environ["UED_EXPECTED_PY"]).resolve()
assert platform.python_version() == "3.10.20"
assert site.ENABLE_USER_SITE is False
assert sys.flags.optimize == 0
assert sys.flags.isolated == 1
assert sys.flags.ignore_environment == 1
assert sys.flags.no_user_site == 1
assert sys.dont_write_bytecode is True
for name in (
    "PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP",
    "PYTHONOPTIMIZE", "LD_LIBRARY_PATH", "LD_PRELOAD", "JAX_DISABLE_JIT",
    "JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION",
    "JAX_COMPILATION_CACHE_DIR", "XLA_FLAGS", "NVIDIA_TF32_OVERRIDE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR",
    "CUDA_LAUNCH_BLOCKING", "TF_ENABLE_ONEDNN_OPTS",
):
    assert name not in os.environ
PY

verify_manifest() {
  local root=$1 name=$2 expected=$3 allowed_unlisted=${4:-} exact_tree=${5:-1}
  local manifest="$root/$name"
  [[ -f "$manifest" && ! -L "$manifest" ]]
  [[ "$(sha256sum "$manifest" | awk '{print $1}')" == "$expected" ]]
  UED_VERIFY_ROOT="$root" UED_VERIFY_MANIFEST="$manifest" \
  UED_VERIFY_ALLOWED="$allowed_unlisted" UED_VERIFY_EXACT="$exact_tree" \
    "$PY" -I -B - <<'PY'
import os
from pathlib import Path, PurePosixPath
import re

root = Path(os.environ["UED_VERIFY_ROOT"])
manifest = Path(os.environ["UED_VERIFY_MANIFEST"])
assert root.resolve() == root
seen = set()
for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\\]+)", raw)
    assert match, (number, raw)
    rel = PurePosixPath(match.group(2))
    assert not rel.is_absolute() and all(part not in ("", ".", "..") for part in rel.parts)
    normalized = rel.as_posix()
    assert normalized not in seen
    seen.add(normalized)
    target = root.joinpath(*rel.parts)
    assert target.is_file() and not target.is_symlink()
    assert target.resolve().is_relative_to(root)
assert seen
allowed = {manifest.name}
allowed.update(item for item in os.environ["UED_VERIFY_ALLOWED"].split(",") if item)
if os.environ["UED_VERIFY_EXACT"] == "1":
    actual = set()
    for path in root.rglob("*"):
        assert not path.is_symlink()
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        else:
            assert path.is_dir()
    assert actual == seen | allowed, (sorted(actual - seen - allowed), sorted(seen - actual))
else:
    for relative in allowed:
        target = root / relative
        assert target.is_file() and not target.is_symlink()
PY
  (cd "$root" && sha256sum -c --strict "$name" >/dev/null)
}

verify_manifest "$UED_BUNDLE_DIR" SHA256SUMS "$UED_BUNDLE_MANIFEST_SHA256"
verify_manifest "$UED_IMPORT_SMOKE_RESULT_DIR" SHA256SUMS \
  "$UED_IMPORT_SMOKE_MANIFEST_SHA256" COMPLETE
verify_manifest "$UED_ONE_UPDATE_RESULT_DIR" SHA256SUMS \
  "$UED_ONE_UPDATE_MANIFEST_SHA256" COMPLETE
for root in "$UED_BUNDLE_DIR" "$UED_IMPORT_SMOKE_RESULT_DIR" "$UED_ONE_UPDATE_RESULT_DIR"; do
  if find "$root" -type l -print -quit | grep -q .; then
    echo "symbolic link in immutable input: $root" >&2
    exit 1
  fi
done

readonly STATE="$UED_BUNDLE_DIR/BUNDLE_STATE.json"
readonly CONFIG="$UED_BUNDLE_DIR/ued_benchmark/configs/maze_frontier_exact_grouped_n8.json"
readonly CONTRACT="$UED_BUNDLE_DIR/ued_benchmark/OVERLAY_CONTRACT.json"
readonly PROTOCOL="$UED_BUNDLE_DIR/ued_benchmark/analysis/development_protocol_v1.json"
readonly ANALYZER="$UED_BUNDLE_DIR/ued_benchmark/analysis/preregistered_dev_analysis.py"
readonly TRAIN_DRIVER="$UED_BUNDLE_DIR/ued_benchmark/scripts/run_matched_terminal.py"
readonly EVAL_DRIVER="$UED_BUNDLE_DIR/ued_benchmark/scripts/evaluate_matched_terminal.py"
readonly ASSEMBLER="$UED_BUNDLE_DIR/ued_benchmark/scripts/assemble_matched_run.py"
readonly FINALIZER="$UED_BUNDLE_DIR/hopper/finalize_ued_minimax_terminal_chain.py"
readonly HOPPER_WRAPPER="$UED_BUNDLE_DIR/hopper/hopper.sh"
readonly BUNDLED_SBATCH="$UED_BUNDLE_DIR/hopper/sbatch/ued_minimax_terminal_chain_smoke.sbatch"
for spec in \
  "$CONFIG:$UED_FRONTIER_CONFIG_SHA256" "$CONTRACT:$UED_CONTRACT_SHA256" \
  "$PROTOCOL:$UED_PROTOCOL_SHA256" "$ANALYZER:$UED_ANALYZER_SHA256" \
  "$TRAIN_DRIVER:$UED_TRAINING_DRIVER_SHA256" \
  "$EVAL_DRIVER:$UED_EVALUATION_DRIVER_SHA256" \
  "$ASSEMBLER:$UED_ASSEMBLER_SHA256" "$BUNDLED_SBATCH:$UED_TERMINAL_CHAIN_SBATCH_SHA256"; do
  path=${spec%%:*}; expected=${spec##*:}
  [[ -f "$path" && ! -L "$path" && "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]]
done
readonly FINALIZER_SHA256="$(sha256sum "$FINALIZER" | awk '{print $1}')"
readonly HOPPER_WRAPPER_SHA256="$(sha256sum "$HOPPER_WRAPPER" | awk '{print $1}')"
[[ "$(sha256sum "$0" | awk '{print $1}')" == "$UED_TERMINAL_CHAIN_SBATCH_SHA256" ]]

UED_STATE="$STATE" UED_COMMIT="$UED_UPSTREAM_COMMIT" UED_TREE="$UED_UPSTREAM_TREE" \
UED_SOURCE="$UED_UPSTREAM_BUNDLE_SHA256" UED_OVERLAY="$UED_OVERLAY_MANIFEST_SHA256" \
UED_LOCK="$UED_ENV_LOCK_SHA256" "$PY" -I -B - <<'PY'
import json
import os
from pathlib import Path

state = json.loads(Path(os.environ["UED_STATE"]).read_text(encoding="utf-8"))
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
assert state["upstream_commit"] == os.environ["UED_COMMIT"]
assert state["upstream_tree_git_sha1"] == os.environ["UED_TREE"]
assert state["upstream_git_bundle_sha256"] == os.environ["UED_SOURCE"]
assert state["overlay_manifest_sha256"] == os.environ["UED_OVERLAY"]
assert state["environment_lock_sha256"] == os.environ["UED_LOCK"]
PY

# Verify the setup-addressed environment, including realized pip and Conda state.
readonly BUNDLED_SETUP="$UED_BUNDLE_DIR/hopper/setup_ued_minimax_env.sh"
readonly BUNDLED_LOCK="$UED_BUNDLE_DIR/hopper/requirements-ued-minimax-hopper.lock"
readonly ENV_SETUP_SHA256="$(awk 'NR == 1 {print $1}' "$UED_ENV_DIR/SETUP_SHA256")"
[[ -f "$UED_ENV_DIR/ENVIRONMENT_COMPLETE" && ! -L "$UED_ENV_DIR/ENVIRONMENT_COMPLETE" ]]
[[ "$(sha256sum "$BUNDLED_SETUP" | awk '{print $1}')" == "$ENV_SETUP_SHA256" ]]
[[ "$(sha256sum "$BUNDLED_LOCK" | awk '{print $1}')" == "$UED_ENV_LOCK_SHA256" ]]
[[ "$(awk 'NR == 1 {print $1}' "$UED_ENV_DIR/LOCK_SHA256")" == "$UED_ENV_LOCK_SHA256" ]]
[[ "$(sha256sum "$UED_ENV_DIR/ENVIRONMENT.freeze" | awk '{print $1}')" == "$UED_ENV_FREEZE_SHA256" ]]
[[ "$(sha256sum "$UED_ENV_DIR/ENVIRONMENT_SHA256SUMS" | awk '{print $1}')" == "$UED_ENV_MANIFEST_SHA256" ]]
verify_manifest "$UED_ENV_DIR" ENVIRONMENT_SHA256SUMS "$UED_ENV_MANIFEST_SHA256" \
  ENVIRONMENT_COMPLETE 0
readonly CONDA_EXPLICIT_SHA256="$(sha256sum "$UED_ENV_DIR/CONDA_EXPLICIT.txt" | awk '{print $1}')"
readonly ENVIRONMENT_JSON_SHA256="$(sha256sum "$UED_ENV_DIR/ENVIRONMENT.json" | awk '{print $1}')"
"$PY" -I -B -m pip check
RECORDED_FREEZE=$(<"$UED_ENV_DIR/ENVIRONMENT.freeze")
CURRENT_FREEZE=$("$PY" -I -B -m pip freeze --all | LC_ALL=C sort)
[[ "$CURRENT_FREEZE" == "$RECORDED_FREEZE" ]]
unset RECORDED_FREEZE CURRENT_FREEZE
readonly CONDA_BIN="${UED_CONDA_BIN:-/home/$RUN_USER/miniconda3/bin/conda}"
[[ -x "$CONDA_BIN" ]]
RECORDED_CONDA=$(<"$UED_ENV_DIR/CONDA_EXPLICIT.txt")
CURRENT_CONDA=$("$CONDA_BIN" list --explicit -p "$UED_ENV_DIR")
[[ "$CURRENT_CONDA" == "$RECORDED_CONDA" ]]
unset RECORDED_CONDA CURRENT_CONDA

# Both prerequisite rungs must come from this exact changed bundle and env.
UED_IMPORT="$UED_IMPORT_SMOKE_RESULT_DIR" UED_ONE="$UED_ONE_UPDATE_RESULT_DIR" \
UED_BUNDLE="$UED_BUNDLE_MANIFEST_SHA256" UED_ENV="$UED_ENV_MANIFEST_SHA256" \
UED_IMPORT_SHA="$UED_IMPORT_SMOKE_MANIFEST_SHA256" \
UED_ONE_SHA="$UED_ONE_UPDATE_MANIFEST_SHA256" "$PY" -I -B - <<'PY'
import json
import os
from pathlib import Path

import_root = Path(os.environ["UED_IMPORT"])
one_root = Path(os.environ["UED_ONE"])
assert (import_root / "COMPLETE").is_file()
runtime = json.loads((import_root / "runtime.json").read_text(encoding="utf-8"))
assert runtime["training_endpoint"] is False and runtime["backend"] == "gpu"
assert runtime["hashes"]["bundle_manifest_sha256"] == os.environ["UED_BUNDLE"]
assert runtime["hashes"]["environment_manifest_sha256"] == os.environ["UED_ENV"]
assert runtime["one_jit"]["passed"] is True

complete = json.loads((one_root / "COMPLETE").read_text(encoding="utf-8"))
provenance = json.loads((one_root / "provenance.json").read_text(encoding="utf-8"))
result = json.loads((one_root / "run-result.json").read_text(encoding="utf-8"))
assert complete["complete_schema"] == 2 and complete["paper_evidence"] is False
assert complete["sha256sums_sha256"] == os.environ["UED_ONE_SHA"]
assert complete["actual_ppo_updates"] == complete["n_grad_updates"] == 1
assert complete["optimizer_step_applications"] == 5
assert complete["terminal_sacct_included"] is False
assert provenance["hashes"]["bundle_manifest_sha256"] == os.environ["UED_BUNDLE"]
assert provenance["hashes"]["environment_manifest_sha256"] == os.environ["UED_ENV"]
assert provenance["hashes"]["import_smoke_manifest_sha256"] == os.environ["UED_IMPORT_SHA"]
assert result["actual_student_updates"] == 1 and result["ood_evaluation"] is False
final = result["final_state"]
assert final["n_updates"] == final["n_grad_updates"] == 1
assert final["optimizer_step_applications"] == 5
assert final["frontier_incomplete_group_count"] == 0
assert final["frontier_duplicate_new_group_count"] == 0
assert result["checkpoint"]["counter_and_buffer_continuity"] is True
assert result["checkpoint"]["train_state_exact_leaf_continuity"] is True
PY

# Freeze every endpoint input before choosing a content-addressed result path.
# The job ID identifies the attempt below that immutable input closure.
readonly CLOSURE_TMP="$(mktemp /tmp/ued-terminal-chain-closure.XXXXXX)"
cleanup_closure() { rm -f -- "$CLOSURE_TMP"; }
trap cleanup_closure EXIT
UED_CLOSURE="$CLOSURE_TMP" UED_GIT_VERSION="$GIT_VERSION" \
UED_BUNDLE="$UED_BUNDLE_MANIFEST_SHA256" UED_COMMIT="$UED_UPSTREAM_COMMIT" \
UED_TREE="$UED_UPSTREAM_TREE" UED_SOURCE="$UED_UPSTREAM_BUNDLE_SHA256" \
UED_OVERLAY="$UED_OVERLAY_MANIFEST_SHA256" UED_SBATCH="$UED_TERMINAL_CHAIN_SBATCH_SHA256" \
UED_CONFIG="$UED_FRONTIER_CONFIG_SHA256" UED_CONTRACT="$UED_CONTRACT_SHA256" \
UED_PROTOCOL="$UED_PROTOCOL_SHA256" UED_ANALYZER="$UED_ANALYZER_SHA256" \
UED_TRAIN="$UED_TRAINING_DRIVER_SHA256" UED_EVAL="$UED_EVALUATION_DRIVER_SHA256" \
UED_ASSEMBLER="$UED_ASSEMBLER_SHA256" UED_FINALIZER="$FINALIZER_SHA256" \
UED_HOPPER="$HOPPER_WRAPPER_SHA256" UED_LOCK="$UED_ENV_LOCK_SHA256" \
UED_FREEZE="$UED_ENV_FREEZE_SHA256" UED_ENV="$UED_ENV_MANIFEST_SHA256" \
UED_SETUP="$ENV_SETUP_SHA256" UED_CONDA="$CONDA_EXPLICIT_SHA256" \
UED_ENV_JSON="$ENVIRONMENT_JSON_SHA256" \
UED_IMPORT="$UED_IMPORT_SMOKE_MANIFEST_SHA256" \
UED_ONE="$UED_ONE_UPDATE_MANIFEST_SHA256" "$PY" -I -B - <<'PY'
import json
import os
from pathlib import Path

record = {
    "input_closure_schema": 1,
    "purpose": "bounded Frontier terminal-chain Slurm engineering smoke",
    "paper_evidence": False,
    "analyzer_eligible": False,
    "endpoint_class": "bounded_engineering_terminal_chain_components",
    "git": os.environ["UED_GIT_VERSION"],
    "two_phase": {
        "phase_a": "slurm_components_only",
        "phase_b": "post_terminal_local_engineering_assembly",
        "production_analyzer_invoked": False,
        "terminal_sacct_required": True,
    },
    "resources": {
        "partition": "gpuq", "qos": "gpu", "gres": "gpu:1g.10gb:1",
        "nodes": 1, "ntasks": 1, "cpus_per_task": 2,
        "memory": "15G", "walltime": "00:30:00", "requeue": False,
    },
    "schedule": {
        "arm": "frontier", "training_seed": 101, "evaluation_seed": 100101,
        "n_updates": 1, "n_grad_updates": 1,
        "optimizer_step_applications": 1, "outer_cycles": 2,
        "student_training_transitions": 128,
        "actual_external_evaluation": True,
        "evaluation_environments": 3, "episodes_per_environment": 10,
        "max_episode_horizon": 450, "primary_evaluation_transitions": 13500,
        "independent_verification_transitions": 0,
    },
    "hashes": {
        "bundle_manifest_sha256": os.environ["UED_BUNDLE"],
        "upstream_commit": os.environ["UED_COMMIT"],
        "upstream_tree_git_sha1": os.environ["UED_TREE"],
        "upstream_git_bundle_sha256": os.environ["UED_SOURCE"],
        "overlay_manifest_sha256": os.environ["UED_OVERLAY"],
        "terminal_chain_sbatch_sha256": os.environ["UED_SBATCH"],
        "frontier_config_sha256": os.environ["UED_CONFIG"],
        "overlay_contract_sha256": os.environ["UED_CONTRACT"],
        "protocol_sha256": os.environ["UED_PROTOCOL"],
        "analyzer_sha256": os.environ["UED_ANALYZER"],
        "training_driver_sha256": os.environ["UED_TRAIN"],
        "evaluation_driver_sha256": os.environ["UED_EVAL"],
        "assembler_sha256": os.environ["UED_ASSEMBLER"],
        "terminal_finalizer_sha256": os.environ["UED_FINALIZER"],
        "hopper_wrapper_sha256": os.environ["UED_HOPPER"],
        "environment_lock_sha256": os.environ["UED_LOCK"],
        "environment_freeze_sha256": os.environ["UED_FREEZE"],
        "environment_manifest_sha256": os.environ["UED_ENV"],
        "environment_setup_script_sha256": os.environ["UED_SETUP"],
        "conda_explicit_sha256": os.environ["UED_CONDA"],
        "environment_json_sha256": os.environ["UED_ENV_JSON"],
        "import_smoke_manifest_sha256": os.environ["UED_IMPORT"],
        "one_update_manifest_sha256": os.environ["UED_ONE"],
    },
}
Path(os.environ["UED_CLOSURE"]).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
readonly INPUT_CLOSURE_SHA256="$(sha256sum "$CLOSURE_TMP" | awk '{print $1}')"
readonly RESULT_PARENT="$SCRATCH_ROOT/maxrl/tests/ued-minimax-terminal-chain/${INPUT_CLOSURE_SHA256:0:20}"
mkdir -p "$RESULT_PARENT"
readonly FINAL_DIR="$RESULT_PARENT/job-$SLURM_JOB_ID"
STAGE_DIR="$RESULT_PARENT/.stage-job-$SLURM_JOB_ID-$$"
[[ ! -e "$FINAL_DIR" && ! -L "$FINAL_DIR" ]]
[[ ! -e "$STAGE_DIR" && ! -L "$STAGE_DIR" ]]
mkdir "$STAGE_DIR"
mv -- "$CLOSURE_TMP" "$STAGE_DIR/INPUT_CLOSURE.json"
trap - EXIT

# Clone only the pinned Git bundle and apply the manifest-bound overlay.
readonly UPSTREAM_BUNDLE="$UED_BUNDLE_DIR/upstream/minimax-${UED_UPSTREAM_COMMIT}.bundle"
[[ "$(sha256sum "$UPSTREAM_BUNDLE" | awk '{print $1}')" == "$UED_UPSTREAM_BUNDLE_SHA256" ]]
readonly WORK_PARENT="$SCRATCH_ROOT/maxrl/tests/work"
mkdir -p "$WORK_PARENT"
readonly WORK_DIR="$WORK_PARENT/ued-minimax-terminal-chain-$SLURM_JOB_ID"
[[ ! -e "$WORK_DIR" && ! -L "$WORK_DIR" ]]
mkdir "$WORK_DIR"
cleanup() {
  if [[ -n "${WORK_DIR:-}" && "$WORK_DIR" == "$WORK_PARENT/ued-minimax-terminal-chain-$SLURM_JOB_ID" \
        && -d "$WORK_DIR" ]]; then rm -rf -- "$WORK_DIR"; fi
  if [[ -n "${STAGE_DIR:-}" && "$STAGE_DIR" == "$RESULT_PARENT/.stage-job-$SLURM_JOB_ID-$$" \
        && -d "$STAGE_DIR" ]]; then rm -rf -- "$STAGE_DIR"; fi
}
trap cleanup EXIT

"$GIT" clone --quiet "$UPSTREAM_BUNDLE" "$WORK_DIR/patched"
[[ "$("$GIT" -C "$WORK_DIR/patched" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
[[ "$("$GIT" -C "$WORK_DIR/patched" rev-parse 'HEAD^{tree}')" == "$EXPECTED_TREE" ]]
readonly APPLY="$UED_BUNDLE_DIR/ued_benchmark/scripts/apply_minimax_overlay.py"
"$PY" -I -B "$APPLY" --target "$WORK_DIR/patched" --check > "$STAGE_DIR/overlay-check.json"
"$PY" -I -B "$APPLY" --target "$WORK_DIR/patched" --apply > "$STAGE_DIR/overlay-apply.json"
"$PY" -I -B "$APPLY" --target "$WORK_DIR/patched" --check > "$STAGE_DIR/overlay-postcheck.json"
grep -Fq '"status": "already_applied"' "$STAGE_DIR/overlay-postcheck.json"
"$GIT" -C "$WORK_DIR/patched" diff --check
readonly APPLIED_MANIFEST="$WORK_DIR/patched/.frontierrl_overlay.json"
readonly APPLIED_SHA256="$(sha256sum "$APPLIED_MANIFEST" | awk '{print $1}')"
cp -- "$APPLIED_MANIFEST" "$STAGE_DIR/applied-overlay-manifest.json"

nvidia-smi --query-gpu=uuid,name,memory.total,driver_version \
  --format=csv,noheader > "$STAGE_DIR/nvidia-smi-before.csv"
readonly GPU_MODEL="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
[[ -n "$GPU_MODEL" && "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
readonly RUN_ID="engineering-slurm-${SLURM_JOB_ID}-${ARM}-s${ENGINEERING_SEED}"

# Freeze the one-submission engineering campaign and run context before the
# training endpoint exists.  Both are bound to exact source/env/driver hashes.
UED_CAMPAIGN="$STAGE_DIR/campaign-manifest.json" UED_CONTEXT="$STAGE_DIR/run-context.json" \
UED_JOB="$SLURM_JOB_ID" UED_RUN_ID="$RUN_ID" UED_GPU_MODEL="$GPU_MODEL" \
UED_PROTOCOL="$UED_PROTOCOL_SHA256" UED_ANALYZER="$UED_ANALYZER_SHA256" \
UED_BASE_COMMIT="$UED_UPSTREAM_COMMIT" UED_BASE_TREE="$UED_UPSTREAM_TREE" \
UED_CONTRACT="$UED_CONTRACT_SHA256" UED_BUNDLE="$UED_BUNDLE_MANIFEST_SHA256" \
UED_OVERLAY="$UED_OVERLAY_MANIFEST_SHA256" UED_APPLIED="$APPLIED_SHA256" \
UED_ENV="$UED_ENV_MANIFEST_SHA256" UED_TRAIN="$UED_TRAINING_DRIVER_SHA256" \
UED_EVAL="$UED_EVALUATION_DRIVER_SHA256" UED_SBATCH="$UED_TERMINAL_CHAIN_SBATCH_SHA256" \
UED_ASSEMBLER="$UED_ASSEMBLER_SHA256" "$PY" -I -B - <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

provenance = {
    "base_commit": os.environ["UED_BASE_COMMIT"],
    "base_tree": os.environ["UED_BASE_TREE"],
    "overlay_contract_sha256": os.environ["UED_CONTRACT"],
    "bundle_manifest_sha256": os.environ["UED_BUNDLE"],
    "overlay_manifest_sha256": os.environ["UED_OVERLAY"],
    "applied_overlay_manifest_sha256": os.environ["UED_APPLIED"],
    "environment_manifest_sha256": os.environ["UED_ENV"],
    "training_driver_sha256": os.environ["UED_TRAIN"],
    "evaluation_driver_sha256": os.environ["UED_EVAL"],
    "sbatch_sha256": os.environ["UED_SBATCH"],
    "assembler_driver_sha256": os.environ["UED_ASSEMBLER"],
}
campaign = {
    "schema": 1,
    "protocol_id": "ued-dev-frontier-vs-maxmc-4x8-b500-v1",
    "purpose": "engineering_development_only_not_paper_evidence",
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "frozen_before_endpoint_access": True,
    "protocol_sha256": os.environ["UED_PROTOCOL"],
    "analyzer_sha256": os.environ["UED_ANALYZER"],
    "provenance": provenance,
    "hardware": {
        "partition": "gpuq", "gpu_model": os.environ["UED_GPU_MODEL"],
        "gpu_profile": "1g.10gb", "gpu_count": 1, "n_devices": 1,
    },
    "submissions": [{
        "arm": "frontier", "training_seed": 101, "evaluation_seed": 100101,
        "run_id": os.environ["UED_RUN_ID"], "job_id": os.environ["UED_JOB"],
        "attempt": 1,
    }],
}
campaign_path = Path(os.environ["UED_CAMPAIGN"])
campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8")
campaign_sha = hashlib.sha256(campaign_path.read_bytes()).hexdigest()
context = {
    "schema": 1,
    "protocol_id": campaign["protocol_id"], "purpose": campaign["purpose"],
    "run_id": os.environ["UED_RUN_ID"], "arm": "frontier", "training_seed": 101,
    "job_id": os.environ["UED_JOB"], "campaign_manifest_sha256": campaign_sha,
    "provenance": {key: value for key, value in provenance.items()
                   if key != "assembler_driver_sha256"},
}
Path(os.environ["UED_CONTEXT"]).write_text(
    json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
readonly CAMPAIGN_SHA256="$(sha256sum "$STAGE_DIR/campaign-manifest.json" | awk '{print $1}')"
readonly CONTEXT_SHA256="$(sha256sum "$STAGE_DIR/run-context.json" | awk '{print $1}')"

mkdir "$STAGE_DIR/training-output-parent" "$STAGE_DIR/training-sidecar-parent" \
      "$STAGE_DIR/evaluation-package-parent"
readonly TRAIN_OUTPUT="$STAGE_DIR/training-output-parent/$RUN_ID"
readonly TRAIN_SIDECAR="$STAGE_DIR/training-sidecar-parent/${RUN_ID}-sidecar"
readonly EVAL_OUTPUT="$STAGE_DIR/evaluation-package-parent/${RUN_ID}-evaluation"

# A single Python parent launches both bounded commands so UTC, monotonic time,
# and RUSAGE_CHILDREN are meaningful without a host /usr/bin/time dependency.
UED_PY="$PY" UED_GIT="$GIT" UED_TRAIN_DRIVER="$TRAIN_DRIVER" \
UED_EVAL_DRIVER="$EVAL_DRIVER" UED_CONFIG="$CONFIG" UED_PROTOCOL_FILE="$PROTOCOL" \
UED_CONTEXT_FILE="$STAGE_DIR/run-context.json" UED_CONTEXT_SHA="$CONTEXT_SHA256" \
UED_CAMPAIGN_FILE="$STAGE_DIR/campaign-manifest.json" UED_CAMPAIGN_SHA="$CAMPAIGN_SHA256" \
UED_TRAIN_SHA="$UED_TRAINING_DRIVER_SHA256" UED_EVAL_SHA="$UED_EVALUATION_DRIVER_SHA256" \
UED_SOURCE_DIR="$WORK_DIR/patched" UED_TRAIN_OUTPUT="$TRAIN_OUTPUT" \
UED_TRAIN_SIDECAR="$TRAIN_SIDECAR" UED_EVAL_OUTPUT="$EVAL_OUTPUT" \
UED_STAGE="$STAGE_DIR" UED_JOB="$SLURM_JOB_ID" "$PY" -I -B - <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import time

stage = Path(os.environ["UED_STAGE"])
train_output = Path(os.environ["UED_TRAIN_OUTPUT"])
train_sidecar = Path(os.environ["UED_TRAIN_SIDECAR"])
eval_output = Path(os.environ["UED_EVAL_OUTPUT"])
common = [
    "--protocol", os.environ["UED_PROTOCOL_FILE"],
    "--campaign-manifest", os.environ["UED_CAMPAIGN_FILE"],
    "--expected-campaign-manifest-sha256", os.environ["UED_CAMPAIGN_SHA"],
    "--run-context", os.environ["UED_CONTEXT_FILE"],
    "--expected-run-context-sha256", os.environ["UED_CONTEXT_SHA"],
    "--patched-source-dir", os.environ["UED_SOURCE_DIR"],
    "--git-executable", os.environ["UED_GIT"],
]
train = [
    os.environ["UED_PY"], "-I", "-B", os.environ["UED_TRAIN_DRIVER"],
    "--arm", "frontier", "--config", os.environ["UED_CONFIG"],
    *common, "--expected-driver-sha256", os.environ["UED_TRAIN_SHA"],
    "--output-dir", str(train_output), "--sidecar-dir", str(train_sidecar),
    "--slurm-engineering-test-mode",
]
overrides = [
    "n_total_updates=1", "test_interval=0", "log_interval=1",
    "train_runner_args.buffer_size=8", "train_runner_args.replay_prob=1.0",
    "train_runner_args.min_fill_ratio=0.5", "train_runner_args.n_rollout_steps=2",
    "train_runner_args.n_unroll_rollout=1", "env_args.max_episode_steps=2",
    "student_rl_args.n_unroll_update=1", "student_rl_args.n_epochs=1",
    "student_model_args.hidden_dim=16", "student_model_args.recurrent_hidden_dim=16",
    "student_model_args.n_conv_filters=4", "driver.max_outer_cycles=4",
]
for override in overrides:
    train.extend(["--engineering-override", override])
evaluation_launcher = (
    "import runpy,sys; p=sys.argv.pop(1); "
    "sys.path.insert(0, str(__import__('pathlib').Path(p).resolve().parent)); "
    "runpy.run_path(p, run_name='__main__')"
)
evaluation = [
    os.environ["UED_PY"], "-I", "-B", "-c", evaluation_launcher,
    os.environ["UED_EVAL_DRIVER"], "--arm", "frontier",
    *common, "--expected-driver-sha256", os.environ["UED_EVAL_SHA"],
    "--checkpoint", str(train_output / "checkpoint.pkl"),
    "--endpoint", str(train_output / "endpoint.json"),
    "--training-receipt", str(train_sidecar / "training-receipt.json"),
    "--meta", str(train_output / "meta.json"), "--output-dir", str(eval_output),
    "--slurm-engineering-test-mode",
]
(stage / "command.txt").write_text(
    "training: " + shlex.join(train) + "\nevaluation: " + shlex.join(evaluation) + "\n",
    encoding="utf-8",
)
started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
started_ns = time.monotonic_ns()
before = resource.getrusage(resource.RUSAGE_CHILDREN)
for label, command in (("training", train), ("evaluation", evaluation)):
    with (stage / f"{label}.stdout").open("x", encoding="utf-8") as stdout, \
         (stage / f"{label}.stderr").open("x", encoding="utf-8") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, text=True)
    if completed.returncode != 0:
        tail = (stage / f"{label}.stderr").read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"{label} command failed with {completed.returncode}:\n{tail}")
after = resource.getrusage(resource.RUSAGE_CHILDREN)
ended_ns = time.monotonic_ns()
ended_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
receipt = {
    "resource_schema": 1,
    "scope": "phase-A bounded training plus actual external evaluation child processes",
    "job_id": os.environ["UED_JOB"],
    "run_start_utc": started_utc, "run_end_utc": ended_utc,
    "monotonic_elapsed_seconds": (ended_ns - started_ns) / 1e9,
    "child_user_seconds": after.ru_utime - before.ru_utime,
    "child_system_seconds": after.ru_stime - before.ru_stime,
    "child_max_rss_kib": int(after.ru_maxrss),
    "requested": {
        "partition": "gpuq", "qos": "gpu", "gres": "gpu:1g.10gb:1",
        "cpus_per_task": 2, "memory": "15G", "walltime": "00:30:00",
    },
    "allocation": {
        key: os.environ.get(key) for key in (
            "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK",
            "SLURM_MEM_PER_NODE", "SLURM_JOB_PARTITION", "SLURM_NTASKS",
            "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES", "SLURM_RESTART_COUNT")
    },
    "peak_gpu_memory_bytes_observed": 0,
    "peak_gpu_memory_observation": "terminal sacct TRESUsageInMax preferred; zero means unavailable",
    "external_accounting_authority": "terminal_slurm_sacct",
    "terminal_sacct_included": False,
}
receipt["allocation"]["SLURM_RESTART_COUNT"] = os.environ.get("SLURM_RESTART_COUNT", "0")
(stage / "resource-accounting.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

nvidia-smi --query-gpu=uuid,name,memory.total,driver_version \
  --format=csv,noheader > "$STAGE_DIR/nvidia-smi-after.csv"

# Flatten the three driver-created packages into fixed component paths and
# validate the exact one-update / actual-evaluation contract before publish.
mv -- "$TRAIN_OUTPUT" "$STAGE_DIR/training-output"
mv -- "$TRAIN_SIDECAR" "$STAGE_DIR/training-sidecar"
mv -- "$EVAL_OUTPUT" "$STAGE_DIR/evaluation-package"
rmdir "$STAGE_DIR/training-output-parent" "$STAGE_DIR/training-sidecar-parent" \
      "$STAGE_DIR/evaluation-package-parent"
verify_manifest "$STAGE_DIR/training-sidecar" SHA256SUMS \
  "$(sha256sum "$STAGE_DIR/training-sidecar/SHA256SUMS" | awk '{print $1}')" COMPLETE
verify_manifest "$STAGE_DIR/evaluation-package" SHA256SUMS \
  "$(sha256sum "$STAGE_DIR/evaluation-package/SHA256SUMS" | awk '{print $1}')" COMPLETE

UED_STAGE="$STAGE_DIR" UED_RUN_ID="$RUN_ID" UED_JOB="$SLURM_JOB_ID" \
UED_BUNDLE="$UED_BUNDLE_MANIFEST_SHA256" UED_GPU_MODEL="$GPU_MODEL" \
  "$PY" -I -B - <<'PY'
import csv
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["UED_STAGE"])
endpoint = json.loads((root / "training-output/endpoint.json").read_text(encoding="utf-8"))
training = json.loads((root / "training-sidecar/training-receipt.json").read_text(encoding="utf-8"))
evaluation = json.loads((root / "evaluation-package/evaluation-receipt.json").read_text(encoding="utf-8"))
context = json.loads((root / "run-context.json").read_text(encoding="utf-8"))
assert endpoint["status"] == "completed" and endpoint["run_id"] == os.environ["UED_RUN_ID"]
assert endpoint["arm"] == "frontier" and endpoint["training_seed"] == 101
assert endpoint["n_updates"] == endpoint["n_grad_updates"] == 1
assert endpoint["optimizer_step_applications"] == 1 and endpoint["outer_cycles"] == 2
assert endpoint["student_training_transitions"] == 128 and endpoint["resumed"] is False
integrity = endpoint["frontier_integrity"]
assert integrity["n_rollouts"] == integrity["n_eval"] == 8
assert integrity["group_size_match"] is True
assert integrity["incomplete_group_count"] == integrity["duplicate_new_group_count"] == 0
assert training["n_updates"] == training["upstream_n_grad_updates"] == 1
assert training["optimizer_step_applications"] == 1
assert training["terminal_checkpoint"]["sha256"] == endpoint["checkpoint_sha256"]
assert training["engineering_test"]["enabled"] is True
assert training["engineering_test"]["execution_mode"] == "slurm"
assert training["provenance"]["backend"] == "gpu"
assert len(training["provenance"]["devices"]) == 1
assert training["provenance"]["devices"][0]["platform"] == "gpu"
expected_device_kind = os.environ["UED_GPU_MODEL"] + " MIG 1g.10gb"
assert training["provenance"]["devices"][0]["device_kind"] == expected_device_kind
source = training["provenance"]["source"]
assert source["git_version"] == "git version 2.45.2"
assert source["git_executable"].endswith("/bin/git")
assert evaluation["run_id"] == os.environ["UED_RUN_ID"]
assert evaluation["training_seed"] == 101 and evaluation["evaluation_seed"] == 100101
assert evaluation["synthetic_test_mode"] is False
assert evaluation["raw_results"]["record_count"] == 30
accounting = evaluation["evaluation_transition_accounting"]
assert accounting["effective_primary_transitions"] == 13500
assert accounting["budgeted_primary_max_transitions"] == 13500
assert accounting["primary_runner_scans_full_horizon"] is True
assert accounting["engineering_independent_verification_transitions"] == 0
assert accounting["total_runtime_transitions"] == 13500
assert accounting["excluded_from_student_training_transitions"] is True
runtime = evaluation["provenance"]["runtime"]
assert runtime["backend"] == "gpu" and runtime["device_count"] == 1
assert len(runtime["devices"]) == 1
assert runtime["devices"][0]["platform"] == "gpu"
assert runtime["devices"][0]["device_kind"] == expected_device_kind
assert evaluation["provenance"]["source"] == source
assert runtime["raw_vs_independent_evalrunner"]["checked"] is False
raw = (root / "evaluation-package/evaluation-episodes.jsonl").read_text(encoding="utf-8").splitlines()
assert len(raw) == 30 and all(json.loads(line) for line in raw)
with (root / "evaluation-package/evaluation.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 1 and None not in rows[0]
assert context["job_id"] == os.environ["UED_JOB"]
assert context["provenance"]["bundle_manifest_sha256"] == os.environ["UED_BUNDLE"]
with (root / "nvidia-smi-before.csv").open(encoding="utf-8", newline="") as stream:
    before = [tuple(field.strip() for field in row) for row in csv.reader(stream)]
with (root / "nvidia-smi-after.csv").open(encoding="utf-8", newline="") as stream:
    after = [tuple(field.strip() for field in row) for row in csv.reader(stream)]
assert before == after and len(before) == 1 and len(before[0]) == 4
assert before[0][1] == os.environ["UED_GPU_MODEL"]
PY

# Copy the exact prerequisite closures and environment receipt.  This result is
# self-auditing but remains phase-A/non-evidence and lacks terminal sacct.
mkdir "$STAGE_DIR/environment" "$STAGE_DIR/import-smoke" "$STAGE_DIR/one-update"
cp -- "$UED_ENV_DIR/CONDA_EXPLICIT.txt" "$UED_ENV_DIR/ENVIRONMENT.freeze" \
      "$UED_ENV_DIR/ENVIRONMENT.json" "$UED_ENV_DIR/ENVIRONMENT_SHA256SUMS" \
      "$UED_ENV_DIR/ENVIRONMENT_COMPLETE" "$UED_ENV_DIR/LOCK_SHA256" \
      "$UED_ENV_DIR/SETUP_SHA256" "$STAGE_DIR/environment/"
cp -a -- "$UED_IMPORT_SMOKE_RESULT_DIR/." "$STAGE_DIR/import-smoke/"
cp -a -- "$UED_ONE_UPDATE_RESULT_DIR/." "$STAGE_DIR/one-update/"
cp -- "$STATE" "$STAGE_DIR/bundle-state.json"
verify_manifest "$STAGE_DIR/environment" ENVIRONMENT_SHA256SUMS "$UED_ENV_MANIFEST_SHA256" \
  ENVIRONMENT_COMPLETE
verify_manifest "$STAGE_DIR/import-smoke" SHA256SUMS "$UED_IMPORT_SMOKE_MANIFEST_SHA256" COMPLETE
verify_manifest "$STAGE_DIR/one-update" SHA256SUMS "$UED_ONE_UPDATE_MANIFEST_SHA256" COMPLETE

readonly TRAIN_SIDECAR_SHA="$(sha256sum "$STAGE_DIR/training-sidecar/SHA256SUMS" | awk '{print $1}')"
readonly EVAL_PACKAGE_SHA="$(sha256sum "$STAGE_DIR/evaluation-package/SHA256SUMS" | awk '{print $1}')"
(
  cd "$STAGE_DIR"
  find . -type f ! -path ./SHA256SUMS ! -path ./COMPONENTS_COMPLETE.json \
    ! -path ./.SHA256SUMS.tmp -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > .SHA256SUMS.tmp
  mv -T .SHA256SUMS.tmp SHA256SUMS
  sha256sum -c --strict SHA256SUMS >/dev/null
)
readonly COMPONENTS_MANIFEST_SHA256="$(sha256sum "$STAGE_DIR/SHA256SUMS" | awk '{print $1}')"
readonly FILE_COUNT="$(wc -l < "$STAGE_DIR/SHA256SUMS")"
UED_COMPLETE="$STAGE_DIR/COMPONENTS_COMPLETE.json" UED_JOB="$SLURM_JOB_ID" \
UED_RUN_ID="$RUN_ID" UED_MANIFEST="$COMPONENTS_MANIFEST_SHA256" UED_COUNT="$FILE_COUNT" \
UED_BUNDLE="$UED_BUNDLE_MANIFEST_SHA256" UED_CAMPAIGN="$CAMPAIGN_SHA256" \
UED_CONTEXT="$CONTEXT_SHA256" UED_TRAIN_SIDE="$TRAIN_SIDECAR_SHA" \
UED_EVAL_PACKAGE="$EVAL_PACKAGE_SHA" UED_INPUT_CLOSURE="$INPUT_CLOSURE_SHA256" \
UED_RESULT_DIR="$FINAL_DIR" "$PY" -I -B - <<'PY'
import json
import os
from pathlib import Path

record = {
    "schema": 1, "status": "complete", "paper_evidence": False,
    "analyzer_eligible": False,
    "endpoint_class": "bounded_engineering_terminal_chain_components",
    "job_id": os.environ["UED_JOB"], "run_id": os.environ["UED_RUN_ID"],
    "arm": "frontier", "sha256sums_sha256": os.environ["UED_MANIFEST"],
    "file_count": int(os.environ["UED_COUNT"]),
    "bundle_manifest_sha256": os.environ["UED_BUNDLE"],
    "campaign_manifest_sha256": os.environ["UED_CAMPAIGN"],
    "run_context_sha256": os.environ["UED_CONTEXT"],
    "training_sidecar_manifest_sha256": os.environ["UED_TRAIN_SIDE"],
    "evaluation_package_manifest_sha256": os.environ["UED_EVAL_PACKAGE"],
    "actual_student_updates": 1, "actual_external_evaluation": True,
    "raw_evaluation_records": 30, "terminal_sacct_included": False,
    "phase_b_required": True,
    "input_closure_sha256": os.environ["UED_INPUT_CLOSURE"],
    "result_dir": os.environ["UED_RESULT_DIR"],
}
path = Path(os.environ["UED_COMPLETE"])
tmp = path.with_name(".COMPONENTS_COMPLETE.json.tmp")
tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY

[[ ! -e "$FINAL_DIR" && ! -L "$FINAL_DIR" ]]
mv -T -- "$STAGE_DIR" "$FINAL_DIR"
STAGE_DIR=""
printf 'UED_TERMINAL_COMPONENTS_COMPLETE job=%s manifest=%s result=%s analyzer_eligible=false\n' \
  "$SLURM_JOB_ID" "$COMPONENTS_MANIFEST_SHA256" "$FINAL_DIR"
