#!/usr/bin/env bash
#SBATCH --job-name=ued-v4-one-update
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

# Engineering-only validation of the exact Frontier 4x8/N=8 lane.  Two outer
# cycles are permitted: a robust-PLR warmup cycle and one forced-replay PPO
# update.  This is not an OOD evaluation or a paper endpoint.
set -euo pipefail
umask 027
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES LD_LIBRARY_PATH
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONOPTIMIZE LD_PRELOAD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1

readonly EXPECTED_COMMIT=d053054c5290a04c1c4cd8b55704d999cad73e30
readonly EXPECTED_CONTRACT=3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b
readonly EXPECTED_CONFIG=0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2
readonly EXPECTED_XPID=eng1-ca-ovv4ch3d5f3827_N8ne8a1.0b1.0th0.0eastrict-rt-4p-b8-rp1-mf0.5-seed1

required=(
  UED_BUNDLE_DIR
  UED_BUNDLE_MANIFEST_SHA256
  UED_UPSTREAM_COMMIT
  UED_UPSTREAM_TREE
  UED_UPSTREAM_BUNDLE_SHA256
  UED_OVERLAY_MANIFEST_SHA256
  UED_SBATCH_SHA256
  UED_CONFIG_SHA256
  UED_CONTRACT_SHA256
  UED_ENV_DIR
  UED_ENV_LOCK_SHA256
  UED_ENV_FREEZE_SHA256
  UED_ENV_MANIFEST_SHA256
  UED_IMPORT_SMOKE_RESULT_DIR
  UED_IMPORT_SMOKE_MANIFEST_SHA256
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
            UED_OVERLAY_MANIFEST_SHA256 UED_SBATCH_SHA256 \
            UED_CONFIG_SHA256 UED_CONTRACT_SHA256 UED_ENV_LOCK_SHA256 \
            UED_ENV_FREEZE_SHA256 UED_ENV_MANIFEST_SHA256 \
            UED_IMPORT_SMOKE_MANIFEST_SHA256; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "invalid SHA-256 in $name" >&2
    exit 2
  }
done
[[ "$UED_UPSTREAM_COMMIT" == "$EXPECTED_COMMIT" ]]
[[ "$UED_UPSTREAM_TREE" =~ ^[0-9a-f]{40}$ ]]
[[ "$UED_CONFIG_SHA256" == "$EXPECTED_CONFIG" ]]
[[ "$UED_CONTRACT_SHA256" == "$EXPECTED_CONTRACT" ]]
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
[[ "${SLURM_CPUS_PER_TASK:-}" == 2 ]]
[[ "${SLURM_JOB_NUM_NODES:-1}" == 1 ]]

readonly RUN_USER="$(id -un)"
[[ "$RUN_USER" =~ ^[A-Za-z0-9._-]+$ && "$RUN_USER" != . && "$RUN_USER" != .. ]] \
  || { echo "unsafe runtime user" >&2; exit 2; }
readonly SCRATCH_ROOT="/scratch/$RUN_USER"

require_canonical_dir() {
  local path=$1
  local prefix=$2
  [[ "$path" == /* && -d "$path" && ! -L "$path" ]] || return 1
  [[ "$(readlink -f -- "$path")" == "$path" ]] || return 1
  case "$path" in "$prefix"/*) ;; *) return 1;; esac
}
require_canonical_dir "$UED_BUNDLE_DIR" \
  "$SCRATCH_ROOT/maxrl/bundles/ued_minimax_v4_engineering" || {
  echo "unsafe or non-canonical bundle path" >&2
  exit 2
}
[[ "$(basename -- "$UED_BUNDLE_DIR")" \
    == "${UED_BUNDLE_MANIFEST_SHA256:0:20}" ]]
require_canonical_dir "$UED_ENV_DIR" "$SCRATCH_ROOT" || {
  echo "unsafe or non-canonical environment path" >&2
  exit 2
}
require_canonical_dir "$UED_IMPORT_SMOKE_RESULT_DIR" \
  "$SCRATCH_ROOT/maxrl/tests/ued-minimax-v4-import" || {
  echo "unsafe or non-canonical import-smoke path" >&2
  exit 2
}
[[ "$(basename -- "$UED_IMPORT_SMOKE_RESULT_DIR")" =~ ^[0-9]+$ ]]
readonly PY="$UED_ENV_DIR/bin/python"
[[ -x "$PY" ]] || { echo "missing environment Python" >&2; exit 1; }

# Reject malicious or ambiguous GNU SHA256SUMS paths before invoking sha256sum.
verify_manifest() {
  local root=$1
  local manifest_name=$2
  local expected_sha=$3
  local closure_mode=$4
  local manifest="$root/$manifest_name"
  [[ -f "$manifest" && ! -L "$manifest" ]]
  [[ "$(sha256sum "$manifest" | awk '{print $1}')" == "$expected_sha" ]]
  UED_VERIFY_ROOT="$root" UED_VERIFY_MANIFEST="$manifest" \
  UED_VERIFY_CLOSURE="$closure_mode" "$PY" -I -B - <<'PY'
import os
from pathlib import Path, PurePosixPath
import re

root = Path(os.environ["UED_VERIFY_ROOT"])
manifest = Path(os.environ["UED_VERIFY_MANIFEST"])
mode = os.environ["UED_VERIFY_CLOSURE"]
assert mode in {"exact", "result", "metadata"}
assert root.resolve() == root and not root.is_symlink()
seen = set()
for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\\]+)", raw)
    assert match, (number, raw)
    rel_text = match.group(2)
    rel = PurePosixPath(rel_text)
    assert not rel.is_absolute() and rel_text not in ("", ".")
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
        sorted(actual - seen - allowed_unlisted),
        sorted(seen - actual),
    )
PY
  (cd "$root" && sha256sum -c --strict "$manifest_name" >/dev/null)
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
assert state["purpose"] == "bounded tie-aware v4 UED engineering rungs 0-3 only"
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
    "v4_gpu_import_tie_formula_jit", "v4_frontier_grouped_one_update",
    "v4_frontier_terminal_chain_components",
    "v4_maxmc_terminal_chain_components"]
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
PY

readonly UPSTREAM_BUNDLE_REL="upstream/minimax-${UED_UPSTREAM_COMMIT}.bundle"
readonly UPSTREAM_BUNDLE="$UED_BUNDLE_DIR/$UPSTREAM_BUNDLE_REL"
[[ -f "$UPSTREAM_BUNDLE" && ! -L "$UPSTREAM_BUNDLE" ]]
[[ "$(sha256sum "$UPSTREAM_BUNDLE" | awk '{print $1}')" == "$UED_UPSTREAM_BUNDLE_SHA256" ]]
readonly OVERLAY_MANIFEST="$UED_BUNDLE_DIR/ued_benchmark/OVERLAY_SHA256SUMS"
[[ "$(sha256sum "$OVERLAY_MANIFEST" | awk '{print $1}')" == "$UED_OVERLAY_MANIFEST_SHA256" ]]
verify_manifest "$UED_BUNDLE_DIR/ued_benchmark" OVERLAY_SHA256SUMS \
  "$UED_OVERLAY_MANIFEST_SHA256" exact

readonly BUNDLED_SBATCH="$UED_BUNDLE_DIR/hopper/sbatch/ued_minimax_v4_one_update_smoke.sbatch"
readonly IMPORT_SBATCH="$UED_BUNDLE_DIR/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch"
[[ -f "$BUNDLED_SBATCH" && ! -L "$BUNDLED_SBATCH" ]]
[[ -f "$IMPORT_SBATCH" && ! -L "$IMPORT_SBATCH" ]]
[[ "$(sha256sum "$BUNDLED_SBATCH" | awk '{print $1}')" \
    == "$UED_SBATCH_SHA256" ]]
[[ "$(sha256sum "$0" | awk '{print $1}')" == "$UED_SBATCH_SHA256" ]]
readonly IMPORT_SBATCH_SHA256="$(sha256sum "$IMPORT_SBATCH" | awk '{print $1}')"
readonly CONFIG="$UED_BUNDLE_DIR/ued_benchmark/configs/maze_frontier_exact_grouped_n8_tie_aware_v4.json"
readonly CONTRACT="$UED_BUNDLE_DIR/ued_benchmark/OVERLAY_CONTRACT_V4.json"
readonly DRIVER="$UED_BUNDLE_DIR/ued_benchmark/scripts/run_grouped_one_update_v4.py"
for input in "$CONFIG" "$CONTRACT" "$DRIVER"; do
  [[ -f "$input" && ! -L "$input" ]]
done
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == "$EXPECTED_CONFIG" ]]
[[ "$(sha256sum "$CONTRACT" | awk '{print $1}')" == "$EXPECTED_CONTRACT" ]]

# Verify the complete environment receipt and the currently installed state.
readonly BUNDLED_SETUP="$UED_BUNDLE_DIR/hopper/setup_ued_minimax_env.sh"
readonly BUNDLED_LOCK="$UED_BUNDLE_DIR/hopper/requirements-ued-minimax-hopper.lock"
[[ -f "$UED_ENV_DIR/ENVIRONMENT_COMPLETE" && ! -L "$UED_ENV_DIR/ENVIRONMENT_COMPLETE" ]]
readonly ENV_SETUP_SHA256="$(awk 'NR == 1 {print $1}' "$UED_ENV_DIR/SETUP_SHA256")"
[[ "$ENV_SETUP_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$(sha256sum "$BUNDLED_SETUP" | awk '{print $1}')" == "$ENV_SETUP_SHA256" ]]
[[ "$(sha256sum "$BUNDLED_LOCK" | awk '{print $1}')" == "$UED_ENV_LOCK_SHA256" ]]
[[ "$(awk 'NR == 1 {print $1}' "$UED_ENV_DIR/LOCK_SHA256")" == "$UED_ENV_LOCK_SHA256" ]]
[[ "$(sha256sum "$UED_ENV_DIR/ENVIRONMENT.freeze" | awk '{print $1}')" \
    == "$UED_ENV_FREEZE_SHA256" ]]
[[ "$(sha256sum "$UED_ENV_DIR/ENVIRONMENT_SHA256SUMS" | awk '{print $1}')" \
    == "$UED_ENV_MANIFEST_SHA256" ]]
readonly COMPLETE_SCHEMA="$(awk -F '\t' '$1 == "environment_schema" {print $2}' "$UED_ENV_DIR/ENVIRONMENT_COMPLETE")"
readonly COMPLETE_MANIFEST="$(awk -F '\t' '$1 == "manifest_sha256" {print $2}' "$UED_ENV_DIR/ENVIRONMENT_COMPLETE")"
readonly COMPLETE_SETUP="$(awk -F '\t' '$1 == "setup_script_sha256" {print $2}' "$UED_ENV_DIR/ENVIRONMENT_COMPLETE")"
[[ "$COMPLETE_SCHEMA" == 2 \
   && "$COMPLETE_MANIFEST" == "$UED_ENV_MANIFEST_SHA256" \
   && "$COMPLETE_SETUP" == "$ENV_SETUP_SHA256" ]]
verify_manifest "$UED_ENV_DIR" ENVIRONMENT_SHA256SUMS "$UED_ENV_MANIFEST_SHA256" metadata
"$PY" -I -B -m pip check
RECORDED_FREEZE=$(<"$UED_ENV_DIR/ENVIRONMENT.freeze")
CURRENT_FREEZE=$("$PY" -I -B -m pip freeze --all | LC_ALL=C sort)
[[ "$CURRENT_FREEZE" == "$RECORDED_FREEZE" ]] || {
  echo "installed packages differ from the submitted environment freeze" >&2
  exit 1
}
unset RECORDED_FREEZE CURRENT_FREEZE
readonly CONDA_BIN="/home/$RUN_USER/miniconda3/bin/conda"
[[ -x "$CONDA_BIN" ]]
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
[[ -x "$GIT" ]] || { echo "missing environment Git" >&2; exit 1; }
readonly GIT_VERSION="$("$GIT" --version)"
[[ "$GIT_VERSION" == "git version 2.45.2" ]]
export PATH="$UED_ENV_DIR/bin:/usr/bin:/bin"
[[ "$(command -v git)" == "$GIT" ]]

# The prerequisite must be a complete import/formula/JIT smoke from this exact
# bundle and environment, not merely a scientifically similar earlier bundle.
[[ -f "$UED_IMPORT_SMOKE_RESULT_DIR/COMPLETE" \
   && ! -L "$UED_IMPORT_SMOKE_RESULT_DIR/COMPLETE" ]]
if find "$UED_IMPORT_SMOKE_RESULT_DIR" -type l -print -quit | grep -q .; then
  echo "import-smoke result contains a symbolic link" >&2
  exit 1
fi
verify_manifest "$UED_IMPORT_SMOKE_RESULT_DIR" SHA256SUMS \
  "$UED_IMPORT_SMOKE_MANIFEST_SHA256" result
for rel in runtime.json receipt.tsv source-import.json \
           applied-overlay-manifest.json bundle-state.json; do
  [[ -f "$UED_IMPORT_SMOKE_RESULT_DIR/$rel" ]]
done
UED_IMPORT_DIR="$UED_IMPORT_SMOKE_RESULT_DIR" \
UED_EXPECTED_BUNDLE="$UED_BUNDLE_MANIFEST_SHA256" \
UED_EXPECTED_COMMIT="$UED_UPSTREAM_COMMIT" \
UED_EXPECTED_TREE="$UED_UPSTREAM_TREE" \
UED_EXPECTED_SOURCE="$UED_UPSTREAM_BUNDLE_SHA256" \
UED_EXPECTED_OVERLAY="$UED_OVERLAY_MANIFEST_SHA256" \
UED_EXPECTED_LOCK="$UED_ENV_LOCK_SHA256" \
UED_EXPECTED_FREEZE="$UED_ENV_FREEZE_SHA256" \
UED_EXPECTED_ENV_MANIFEST="$UED_ENV_MANIFEST_SHA256" \
UED_EXPECTED_SETUP="$ENV_SETUP_SHA256" \
UED_EXPECTED_CONDA="$CONDA_EXPLICIT_SHA256" \
UED_EXPECTED_ENV_JSON="$ENVIRONMENT_JSON_SHA256" \
UED_EXPECTED_IMPORT_SBATCH="$IMPORT_SBATCH_SHA256" \
"$PY" -I -B - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["UED_IMPORT_DIR"])
runtime = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
source_import = json.loads((root / "source-import.json").read_text(encoding="utf-8"))
applied = json.loads((root / "applied-overlay-manifest.json").read_text(encoding="utf-8"))
receipt = {}
for line in (root / "receipt.tsv").read_text(encoding="utf-8").splitlines()[1:]:
    key, value = line.split("\t", 1)
    assert key not in receipt
    receipt[key] = value

expected = {
    "bundle_manifest_sha256": os.environ["UED_EXPECTED_BUNDLE"],
    "upstream_commit": os.environ["UED_EXPECTED_COMMIT"],
    "upstream_tree_git_sha1": os.environ["UED_EXPECTED_TREE"],
    "upstream_git_bundle_sha256": os.environ["UED_EXPECTED_SOURCE"],
    "overlay_manifest_sha256": os.environ["UED_EXPECTED_OVERLAY"],
    "environment_lock_sha256": os.environ["UED_EXPECTED_LOCK"],
    "environment_freeze_sha256": os.environ["UED_EXPECTED_FREEZE"],
    "environment_manifest_sha256": os.environ["UED_EXPECTED_ENV_MANIFEST"],
    "environment_setup_script_sha256": os.environ["UED_EXPECTED_SETUP"],
    "conda_explicit_sha256": os.environ["UED_EXPECTED_CONDA"],
    "environment_json_sha256": os.environ["UED_EXPECTED_ENV_JSON"],
}
assert runtime["runtime_schema"] == 1
assert runtime["training_endpoint"] is False
assert runtime["backend"] == "gpu"
assert len(runtime["devices"]) == 1
assert runtime["one_jit"]["passed"] is True
assert runtime["source_faithful_import"]["source_faithful_import"] is True
assert source_import["source_faithful_import"] is True
assert len(source_import["gpu_devices"]) == 1
assert runtime["git"] == receipt["git"] == "git version 2.45.2"
assert runtime["hashes"]["sbatch_sha256"] == os.environ["UED_EXPECTED_IMPORT_SBATCH"]
assert receipt["sbatch_sha256"] == os.environ["UED_EXPECTED_IMPORT_SBATCH"]
for package in ("jax", "jaxlib", "jax_cuda12_plugin", "jax_cuda12_pjrt"):
    assert runtime[package] == "0.4.31", (package, runtime[package])
for field, value in expected.items():
    assert runtime["hashes"][field] == value, field
    assert receipt[field] == value, field
assert receipt["training_endpoint"] == "false"
assert receipt["job_id"] == root.name == runtime["job_id"]
assert Path(receipt["result_dir"]).resolve() == root
applied_bytes = (root / "applied-overlay-manifest.json").read_bytes()
applied_sha = hashlib.sha256(applied_bytes).hexdigest()
assert runtime["hashes"]["applied_overlay_manifest_sha256"] == applied_sha
assert receipt["applied_overlay_manifest_sha256"] == applied_sha
assert applied["base_commit"] == os.environ["UED_EXPECTED_COMMIT"]
assert applied["overlay"] == "frontier-activity-tie-aware-v4"
assert applied["overlay_contract_sha256"] == (
    "3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b")
PY
readonly IMPORT_BUNDLE_SHA256="$UED_BUNDLE_MANIFEST_SHA256"

# Freeze every job input into a canonical closure before selecting the result
# path.  The final attempt is nested under the closure hash.
readonly CLOSURE_TMP="$(mktemp /tmp/ued-one-update-closure.XXXXXX)"
cleanup_closure() { rm -f -- "$CLOSURE_TMP"; }
trap cleanup_closure EXIT
UED_CLOSURE_PATH="$CLOSURE_TMP" \
UED_XPID="$EXPECTED_XPID" \
UED_GIT_VERSION="$GIT_VERSION" \
UED_BUNDLE="$UED_BUNDLE_MANIFEST_SHA256" \
UED_COMMIT="$UED_UPSTREAM_COMMIT" \
UED_TREE="$UED_UPSTREAM_TREE" \
UED_SOURCE="$UED_UPSTREAM_BUNDLE_SHA256" \
UED_OVERLAY="$UED_OVERLAY_MANIFEST_SHA256" \
UED_SBATCH="$UED_SBATCH_SHA256" \
UED_CONFIG="$EXPECTED_CONFIG" UED_CONTRACT="$EXPECTED_CONTRACT" \
UED_LOCK="$UED_ENV_LOCK_SHA256" UED_FREEZE="$UED_ENV_FREEZE_SHA256" \
UED_ENV_MANIFEST="$UED_ENV_MANIFEST_SHA256" UED_SETUP="$ENV_SETUP_SHA256" \
UED_CONDA="$CONDA_EXPLICIT_SHA256" UED_ENV_JSON="$ENVIRONMENT_JSON_SHA256" \
UED_IMPORT="$UED_IMPORT_SMOKE_MANIFEST_SHA256" \
UED_IMPORT_SBATCH="$IMPORT_SBATCH_SHA256" \
"$PY" -I -B - <<'PY'
import json
import os
from pathlib import Path

record = {
    "input_closure_schema": 1,
    "purpose": "bounded Frontier grouped one-update engineering validation",
    "paper_evidence": False,
    "xpid": os.environ["UED_XPID"],
    "git": os.environ["UED_GIT_VERSION"],
    "schedule": {
        "outer_cycles": 2, "max_student_updates": 1,
        "actual_ppo_updates": 1, "n_grad_updates": 1,
        "ppo_epochs": 5, "ppo_minibatches": 1,
        "optimizer_step_applications": 5,
        "n_parallel": 4, "n_eval": 8, "frontier_n_rollouts": 8,
        "buffer_size": 8, "minimum_fill_ratio": 0.5,
        "replay_probability": 1.0, "rollout_steps": 256,
    },
    "hashes": {
        "bundle_manifest_sha256": os.environ["UED_BUNDLE"],
        "upstream_commit": os.environ["UED_COMMIT"],
        "upstream_tree_git_sha1": os.environ["UED_TREE"],
        "upstream_git_bundle_sha256": os.environ["UED_SOURCE"],
        "overlay_manifest_sha256": os.environ["UED_OVERLAY"],
        "one_update_sbatch_sha256": os.environ["UED_SBATCH"],
        "config_sha256": os.environ["UED_CONFIG"],
        "overlay_contract_sha256": os.environ["UED_CONTRACT"],
        "environment_lock_sha256": os.environ["UED_LOCK"],
        "environment_freeze_sha256": os.environ["UED_FREEZE"],
        "environment_manifest_sha256": os.environ["UED_ENV_MANIFEST"],
        "environment_setup_script_sha256": os.environ["UED_SETUP"],
        "conda_explicit_sha256": os.environ["UED_CONDA"],
        "environment_json_sha256": os.environ["UED_ENV_JSON"],
        "import_smoke_manifest_sha256": os.environ["UED_IMPORT"],
        "import_smoke_bundle_manifest_sha256": os.environ["UED_BUNDLE"],
        "import_smoke_sbatch_sha256": os.environ["UED_IMPORT_SBATCH"],
    },
}
Path(os.environ["UED_CLOSURE_PATH"]).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
readonly INPUT_CLOSURE_SHA256="$(sha256sum "$CLOSURE_TMP" | awk '{print $1}')"
readonly RESULT_PARENT="$SCRATCH_ROOT/maxrl/tests/ued-minimax-v4-one-update/${INPUT_CLOSURE_SHA256:0:20}"
mkdir -p "$RESULT_PARENT"
readonly FINAL_DIR="$RESULT_PARENT/job-$SLURM_JOB_ID"
STAGE_DIR="$RESULT_PARENT/.stage-job-$SLURM_JOB_ID-$$"
[[ ! -e "$FINAL_DIR" && ! -L "$FINAL_DIR" ]]
[[ ! -e "$STAGE_DIR" && ! -L "$STAGE_DIR" ]]
mkdir "$STAGE_DIR"
mv -- "$CLOSURE_TMP" "$STAGE_DIR/INPUT_CLOSURE.json"
trap - EXIT

readonly WORK_PARENT="$SCRATCH_ROOT/maxrl/tests/work"
mkdir -p "$WORK_PARENT"
readonly WORK_DIR="$WORK_PARENT/ued-minimax-v4-one-update-$SLURM_JOB_ID"
[[ ! -e "$WORK_DIR" && ! -L "$WORK_DIR" ]]
mkdir "$WORK_DIR"
cleanup() {
  if [[ -n "${WORK_DIR:-}" \
        && "$WORK_DIR" == "$WORK_PARENT/ued-minimax-v4-one-update-$SLURM_JOB_ID" \
        && -d "$WORK_DIR" ]]; then
    rm -rf -- "$WORK_DIR"
  fi
  if [[ -n "${STAGE_DIR:-}" \
        && "$STAGE_DIR" == "$RESULT_PARENT/.stage-job-$SLURM_JOB_ID-$$" \
        && -d "$STAGE_DIR" ]]; then
    rm -rf -- "$STAGE_DIR"
  fi
}
trap cleanup EXIT

cp -- "$STATE" "$STAGE_DIR/bundle-state.json"
mkdir "$STAGE_DIR/environment" "$STAGE_DIR/import-smoke"
cp -- "$UED_ENV_DIR/CONDA_EXPLICIT.txt" \
      "$UED_ENV_DIR/ENVIRONMENT.freeze" \
      "$UED_ENV_DIR/ENVIRONMENT.json" \
      "$UED_ENV_DIR/ENVIRONMENT_SHA256SUMS" \
      "$UED_ENV_DIR/ENVIRONMENT_COMPLETE" \
      "$UED_ENV_DIR/LOCK_SHA256" \
      "$UED_ENV_DIR/SETUP_SHA256" \
      "$STAGE_DIR/environment/"
cp -a -- "$UED_IMPORT_SMOKE_RESULT_DIR/." "$STAGE_DIR/import-smoke/"
verify_manifest "$STAGE_DIR/environment" ENVIRONMENT_SHA256SUMS \
  "$UED_ENV_MANIFEST_SHA256" metadata
verify_manifest "$STAGE_DIR/import-smoke" SHA256SUMS \
  "$UED_IMPORT_SMOKE_MANIFEST_SHA256" result

nvidia-smi --query-gpu=uuid,name,memory.total,driver_version \
  --format=csv,noheader > "$STAGE_DIR/nvidia-smi-before.csv"

"$GIT" clone --quiet "$UPSTREAM_BUNDLE" "$WORK_DIR/patched"
[[ "$("$GIT" -C "$WORK_DIR/patched" rev-parse HEAD)" == "$UED_UPSTREAM_COMMIT" ]]
[[ "$("$GIT" -C "$WORK_DIR/patched" rev-parse 'HEAD^{tree}')" == "$UED_UPSTREAM_TREE" ]]
[[ -z "$("$GIT" -C "$WORK_DIR/patched" status --porcelain --untracked-files=all)" ]]
readonly APPLY="$UED_BUNDLE_DIR/ued_benchmark/scripts/apply_minimax_overlay_v4.py"
"$PY" -I -B "$APPLY" --target "$WORK_DIR/patched" --check \
  > "$STAGE_DIR/overlay-check.json"
"$PY" -I -B "$APPLY" --target "$WORK_DIR/patched" --apply \
  > "$STAGE_DIR/overlay-apply.json"
"$PY" -I -B "$APPLY" --target "$WORK_DIR/patched" --check \
  > "$STAGE_DIR/overlay-postcheck.json"
"$GIT" -C "$WORK_DIR/patched" diff --check
readonly APPLIED_MANIFEST="$WORK_DIR/patched/.frontierrl_overlay.json"
readonly APPLIED_MANIFEST_SHA256="$(sha256sum "$APPLIED_MANIFEST" | awk '{print $1}')"
[[ "$APPLIED_MANIFEST_SHA256" == 9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae ]]
cp -- "$APPLIED_MANIFEST" "$STAGE_DIR/applied-overlay-manifest.json"
[[ "$APPLIED_MANIFEST_SHA256" \
    == "$(sha256sum "$UED_IMPORT_SMOKE_RESULT_DIR/applied-overlay-manifest.json" | awk '{print $1}')" ]]

UED_PROVENANCE_PATH="$STAGE_DIR/provenance.json" \
UED_RESULT_DIR="$FINAL_DIR" UED_JOB_ID="$SLURM_JOB_ID" UED_XPID="$EXPECTED_XPID" \
UED_GIT_VERSION="$GIT_VERSION" \
UED_BUNDLE="$UED_BUNDLE_MANIFEST_SHA256" UED_COMMIT="$UED_UPSTREAM_COMMIT" \
UED_TREE="$UED_UPSTREAM_TREE" UED_SOURCE="$UED_UPSTREAM_BUNDLE_SHA256" \
UED_OVERLAY="$UED_OVERLAY_MANIFEST_SHA256" UED_APPLIED="$APPLIED_MANIFEST_SHA256" \
UED_SBATCH="$UED_SBATCH_SHA256" UED_CONFIG="$EXPECTED_CONFIG" \
UED_CONTRACT="$EXPECTED_CONTRACT" UED_LOCK="$UED_ENV_LOCK_SHA256" \
UED_FREEZE="$UED_ENV_FREEZE_SHA256" UED_ENV_MANIFEST="$UED_ENV_MANIFEST_SHA256" \
UED_SETUP="$ENV_SETUP_SHA256" UED_CONDA="$CONDA_EXPLICIT_SHA256" \
UED_ENV_JSON="$ENVIRONMENT_JSON_SHA256" UED_IMPORT="$UED_IMPORT_SMOKE_MANIFEST_SHA256" \
UED_IMPORT_SBATCH="$IMPORT_SBATCH_SHA256" \
UED_INPUT_CLOSURE="$INPUT_CLOSURE_SHA256" \
"$PY" -I -B - <<'PY'
import json
import os
from pathlib import Path

record = {
    "provenance_schema": 1,
    "purpose": "bounded Frontier grouped one-update engineering validation",
    "paper_evidence": False,
    "endpoint_class": "bounded_engineering_one_update",
    "max_student_updates": 1,
    "git": os.environ["UED_GIT_VERSION"],
    "job_id": os.environ["UED_JOB_ID"],
    "result_dir": os.environ["UED_RESULT_DIR"],
    "xpid": os.environ["UED_XPID"],
    "input_closure_sha256": os.environ["UED_INPUT_CLOSURE"],
    "resources": {
        "partition": "gpuq", "qos": "gpu", "gres": "gpu:1g.10gb:1",
        "cpus_per_task": 2, "memory": "15G", "walltime": "00:30:00",
    },
    "hashes": {
        "bundle_manifest_sha256": os.environ["UED_BUNDLE"],
        "upstream_commit": os.environ["UED_COMMIT"],
        "upstream_tree_git_sha1": os.environ["UED_TREE"],
        "upstream_git_bundle_sha256": os.environ["UED_SOURCE"],
        "overlay_manifest_sha256": os.environ["UED_OVERLAY"],
        "applied_overlay_manifest_sha256": os.environ["UED_APPLIED"],
        "sbatch_sha256": os.environ["UED_SBATCH"],
        "config_sha256": os.environ["UED_CONFIG"],
        "overlay_contract_sha256": os.environ["UED_CONTRACT"],
        "environment_lock_sha256": os.environ["UED_LOCK"],
        "environment_freeze_sha256": os.environ["UED_FREEZE"],
        "environment_manifest_sha256": os.environ["UED_ENV_MANIFEST"],
        "environment_setup_script_sha256": os.environ["UED_SETUP"],
        "conda_explicit_sha256": os.environ["UED_CONDA"],
        "environment_json_sha256": os.environ["UED_ENV_JSON"],
        "import_smoke_manifest_sha256": os.environ["UED_IMPORT"],
        "import_smoke_bundle_manifest_sha256": os.environ["UED_BUNDLE"],
        "import_smoke_sbatch_sha256": os.environ["UED_IMPORT_SBATCH"],
    },
}
path = Path(os.environ["UED_PROVENANCE_PATH"])
tmp = path.with_name(".provenance.json.tmp")
tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY

UED_COMMAND_PATH="$STAGE_DIR/command.json" \
UED_XPID="$EXPECTED_XPID" UED_CONFIG="$EXPECTED_CONFIG" \
UED_CONTRACT="$EXPECTED_CONTRACT" "$PY" -I -B - <<'PY'
import json
import os
from pathlib import Path

record = {
    "command_schema": 1,
    "driver": "ued_benchmark/scripts/run_grouped_one_update_v4.py",
    "xpid": os.environ["UED_XPID"],
    "authored_config_sha256": os.environ["UED_CONFIG"],
    "overlay_contract_sha256": os.environ["UED_CONTRACT"],
    "local_test_mode": False,
    "allowed_overrides": {
        "n_total_updates": 1,
        "test_interval": 0,
        "plr_buffer_size": 8,
        "plr_min_fill_ratio": 0.5,
        "plr_replay_prob": 1.0,
        "test_env_names": None,
    },
}
Path(os.environ["UED_COMMAND_PATH"]).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

readonly ISOLATED_LAUNCHER='import runpy,sys; from pathlib import Path; source=sys.argv.pop(1); bundle=sys.argv.pop(1); script=sys.argv.pop(1); sys.path[:0]=[source,bundle,str(Path(script).parent)]; sys.argv[0]=script; runpy.run_path(script,run_name="__main__")'
"$PY" -I -B -c "$ISOLATED_LAUNCHER" \
  "$WORK_DIR/patched/src" "$UED_BUNDLE_DIR" "$DRIVER" \
  --config "$CONFIG" \
  --contract "$CONTRACT" \
  --provenance "$STAGE_DIR/provenance.json" \
  --patched-source-dir "$WORK_DIR/patched" \
  --output-dir "$STAGE_DIR" \
  > "$STAGE_DIR/driver.stdout" 2> "$STAGE_DIR/driver.stderr"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version \
  --format=csv,noheader > "$STAGE_DIR/nvidia-smi-after.csv"

RESOURCE_FIELDS="$(
  UED_RESULT="$STAGE_DIR/run-result.json" \
  UED_CHECKPOINT="$STAGE_DIR/checkpoint.pkl" \
  UED_RESOURCE_JSON="$STAGE_DIR/resource-accounting.json" \
  UED_JOB_ID="$SLURM_JOB_ID" UED_INPUT_CLOSURE="$INPUT_CLOSURE_SHA256" \
  "$PY" -I -B - <<'PY'
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re

result = json.loads(Path(os.environ["UED_RESULT"]).read_text(encoding="utf-8"))
checkpoint = Path(os.environ["UED_CHECKPOINT"])
checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert result["status"] == "passed"
assert result["paper_evidence"] is False
assert "training_endpoint" not in result
assert result["endpoint_class"] == "bounded_engineering_one_update"
assert result["max_student_updates"] == result["actual_student_updates"] == 1
assert result["ood_evaluation"] is False
assert result["job_id"] == os.environ["UED_JOB_ID"]
assert result["xpid"] == (
    "eng1-ca-ovv4ch3d5f3827_N8ne8a1.0b1.0th0.0eastrict-rt-4p-b8-rp1-mf0.5-seed1")
schedule = result["engineering_schedule"]
assert schedule["outer_cycles"] == 2
assert schedule["actual_ppo_updates"] == 1
assert schedule["ppo_epochs"] == 5
assert schedule["ppo_minibatches"] == 1
assert schedule["expected_optimizer_step_applications"] == 5
assert schedule["optimizer_step_applications"] == 5
assert schedule["n_parallel"] == schedule["n_eval"] // 2 == 4
assert schedule["frontier_n_rollouts"] == schedule["n_eval"] == 8
assert schedule["rollout_steps"] == 256
assert schedule["total_transitions"] == 16384
final = result["final_state"]
assert final["n_iters"] == 2 and final["n_updates"] == 1
assert final["n_grad_updates"] == 1
assert final["optimizer_step_applications"] == 5
assert final["buffer_filled_count"] == 4
assert final["frontier_total_trials"] == 64
assert final["frontier_incomplete_group_count"] == 0
assert final["frontier_duplicate_new_group_count"] == 0
assert final["frontier_group_size_match"] is True
assert final["tie_aware_score_ranks"] is True
assert final["nonfinite_filled_score_count"] == 0
assert final["nonfinite_score_rejection_count"] == 0
assert final["replay_group_draw_count"] == 4
assert final["last_replay_group_count"] == 4
assert final["replay_group_draw_count"] == (
    final["replay_distinct_group_count"] + final["replay_duplicate_group_count"])
assert final["last_replay_group_count"] == (
    final["last_replay_distinct_group_count"]
    + final["last_replay_duplicate_group_count"])
assert 1 <= final["last_replay_distinct_group_count"] <= 4
cycle2 = result["cycles"][1]["selected_stats"]
for key in ("plr/score_distribution_effective_support",
            "plr/replay_distribution_effective_support"):
    assert math.isfinite(float(cycle2[key])) and 1.0 <= float(cycle2[key]) <= 4.00001
assert cycle2["plr/distinct_filled_score_count"] >= 1
assert cycle2["plr/score_max_tie_block_size"] >= 1
assert result["checkpoint"]["sha256"] == checkpoint_sha
assert result["checkpoint"]["counter_and_buffer_continuity"] is True
assert result["checkpoint"]["train_state_exact_leaf_continuity"] is True
assert result["checkpoint"]["post_resume_update_executed"] is False

driver_accounting = result["resource_accounting"]
assert set(driver_accounting) == {
    "resource_schema", "scope", "accounting_source", "run_start_utc",
    "run_end_utc", "monotonic_elapsed_seconds", "process_user_seconds",
    "process_system_seconds", "process_max_rss_kib",
    "process_minor_page_faults", "process_major_page_faults",
    "process_block_input_operations", "process_block_output_operations",
    "process_voluntary_context_switches",
    "process_involuntary_context_switches", "transitions",
    "transitions_per_wall_second", "slurm_request",
    "external_accounting_authority", "terminal_sacct_included",
}
assert driver_accounting["resource_schema"] == 2
assert driver_accounting["scope"] == "in-process engineering diagnostics"
assert driver_accounting["accounting_source"] == (
    "python_resource_getrusage_self_and_monotonic_ns")
utc_pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")
run_start = driver_accounting["run_start_utc"]
run_end = driver_accounting["run_end_utc"]
assert utc_pattern.fullmatch(run_start)
assert utc_pattern.fullmatch(run_end)
assert datetime.strptime(run_end, "%Y-%m-%dT%H:%M:%S.%fZ") >= datetime.strptime(
    run_start, "%Y-%m-%dT%H:%M:%S.%fZ")
elapsed = float(driver_accounting["monotonic_elapsed_seconds"])
user_seconds = float(driver_accounting["process_user_seconds"])
system_seconds = float(driver_accounting["process_system_seconds"])
max_rss_kib = int(driver_accounting["process_max_rss_kib"])
assert math.isfinite(elapsed) and elapsed > 0.0
assert math.isfinite(user_seconds) and user_seconds >= 0.0
assert math.isfinite(system_seconds) and system_seconds >= 0.0
assert max_rss_kib > 0
for field in (
        "process_minor_page_faults", "process_major_page_faults",
        "process_block_input_operations", "process_block_output_operations",
        "process_voluntary_context_switches",
        "process_involuntary_context_switches"):
    assert isinstance(driver_accounting[field], int)
    assert driver_accounting[field] >= 0
assert driver_accounting["transitions"] == 16384
assert math.isfinite(driver_accounting["transitions_per_wall_second"])
assert driver_accounting["transitions_per_wall_second"] > 0.0
assert driver_accounting["slurm_request"] == {
    "partition": "gpuq", "qos": "gpu", "gres": "gpu:1g.10gb:1",
    "cpus_per_task": 2, "memory": "15G", "walltime": "00:30:00",
}
assert driver_accounting["external_accounting_authority"] == (
    "terminal_slurm_sacct")
assert driver_accounting["terminal_sacct_included"] is False
accounting = {
    "resource_schema": 2,
    "scope": (
        "in-process diagnostics; terminal Slurm sacct is authoritative external "
        "accounting and must be retrieved after job completion"),
    "job_id": os.environ["UED_JOB_ID"],
    "input_closure_sha256": os.environ["UED_INPUT_CLOSURE"],
    "driver_accounting": driver_accounting,
    "requested": {
        "partition": os.environ.get("SLURM_JOB_PARTITION", "gpuq"),
        "cpus_per_task": int(os.environ["SLURM_CPUS_PER_TASK"]),
        "memory": "15G", "gres": "gpu:1g.10gb:1", "walltime": "00:30:00",
    },
    "allocation": {
        key: os.environ.get(key)
        for key in (
            "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK",
            "SLURM_MEM_PER_NODE", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES")
    },
    "external_accounting_authority": "terminal_slurm_sacct",
    "terminal_sacct_included": False,
}
path = Path(os.environ["UED_RESOURCE_JSON"])
tmp = path.with_name(".resource-accounting.json.tmp")
tmp.write_text(json.dumps(accounting, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, path)
print("\t".join((
    run_start, run_end, repr(elapsed), repr(user_seconds), repr(system_seconds),
    str(max_rss_kib))))
PY
)"
IFS=$'\t' read -r RUN_START_UTC RUN_END_UTC MONOTONIC_ELAPSED_SECONDS \
  PROCESS_USER_SECONDS PROCESS_SYSTEM_SECONDS PROCESS_MAX_RSS_KIB \
  <<< "$RESOURCE_FIELDS"
readonly RUN_START_UTC RUN_END_UTC MONOTONIC_ELAPSED_SECONDS \
  PROCESS_USER_SECONDS PROCESS_SYSTEM_SECONDS PROCESS_MAX_RSS_KIB
readonly RESOURCE_ACCOUNTING_SHA256="$(sha256sum "$STAGE_DIR/resource-accounting.json" | awk '{print $1}')"
readonly RUN_RESULT_SHA256="$(sha256sum "$STAGE_DIR/run-result.json" | awk '{print $1}')"
readonly CHECKPOINT_SHA256="$(sha256sum "$STAGE_DIR/checkpoint.pkl" | awk '{print $1}')"

{
  printf 'field\tvalue\n'
  printf 'job_id\t%s\n' "$SLURM_JOB_ID"
  printf 'utc\t%s\n' "$RUN_END_UTC"
  printf 'run_start_utc\t%s\n' "$RUN_START_UTC"
  printf 'run_end_utc\t%s\n' "$RUN_END_UTC"
  printf 'monotonic_elapsed_seconds\t%s\n' "$MONOTONIC_ELAPSED_SECONDS"
  printf 'process_user_seconds\t%s\n' "$PROCESS_USER_SECONDS"
  printf 'process_system_seconds\t%s\n' "$PROCESS_SYSTEM_SECONDS"
  printf 'process_max_rss_kib\t%s\n' "$PROCESS_MAX_RSS_KIB"
  printf 'resource_accounting_source\tpython_resource_getrusage_self_and_monotonic_ns\n'
  printf 'external_accounting_authority\tterminal_slurm_sacct\n'
  printf 'resource_accounting_sha256\t%s\n' "$RESOURCE_ACCOUNTING_SHA256"
  printf 'run_result_sha256\t%s\n' "$RUN_RESULT_SHA256"
  printf 'checkpoint_sha256\t%s\n' "$CHECKPOINT_SHA256"
  printf 'host\t%s\n' "$(hostname)"
  printf 'result_dir\t%s\n' "$FINAL_DIR"
  printf 'input_closure_sha256\t%s\n' "$INPUT_CLOSURE_SHA256"
  printf 'bundle_manifest_sha256\t%s\n' "$UED_BUNDLE_MANIFEST_SHA256"
  printf 'upstream_commit\t%s\n' "$UED_UPSTREAM_COMMIT"
  printf 'upstream_tree_git_sha1\t%s\n' "$UED_UPSTREAM_TREE"
  printf 'upstream_git_bundle_sha256\t%s\n' "$UED_UPSTREAM_BUNDLE_SHA256"
  printf 'overlay_manifest_sha256\t%s\n' "$UED_OVERLAY_MANIFEST_SHA256"
  printf 'applied_overlay_manifest_sha256\t%s\n' "$APPLIED_MANIFEST_SHA256"
  printf 'one_update_sbatch_sha256\t%s\n' "$UED_SBATCH_SHA256"
  printf 'config_sha256\t%s\n' "$EXPECTED_CONFIG"
  printf 'overlay_contract_sha256\t%s\n' "$EXPECTED_CONTRACT"
  printf 'environment_lock_sha256\t%s\n' "$UED_ENV_LOCK_SHA256"
  printf 'environment_freeze_sha256\t%s\n' "$UED_ENV_FREEZE_SHA256"
  printf 'environment_manifest_sha256\t%s\n' "$UED_ENV_MANIFEST_SHA256"
  printf 'environment_setup_script_sha256\t%s\n' "$ENV_SETUP_SHA256"
  printf 'conda_explicit_sha256\t%s\n' "$CONDA_EXPLICIT_SHA256"
  printf 'environment_json_sha256\t%s\n' "$ENVIRONMENT_JSON_SHA256"
  printf 'git\t%s\n' "$GIT_VERSION"
  printf 'import_smoke_manifest_sha256\t%s\n' "$UED_IMPORT_SMOKE_MANIFEST_SHA256"
  printf 'import_smoke_sbatch_sha256\t%s\n' "$IMPORT_SBATCH_SHA256"
  printf 'xpid\t%s\n' "$EXPECTED_XPID"
  printf 'n_parallel\t4\n'
  printf 'n_eval\t8\n'
  printf 'frontier_n_rollouts\t8\n'
  printf 'outer_cycles\t2\n'
  printf 'actual_ppo_updates\t1\n'
  printf 'n_grad_updates\t1\n'
  printf 'ppo_epochs\t5\n'
  printf 'ppo_minibatches\t1\n'
  printf 'optimizer_step_applications\t5\n'
  printf 'endpoint_class\tbounded_engineering_one_update\n'
  printf 'max_student_updates\t1\n'
  printf 'actual_student_updates\t1\n'
  printf 'total_transitions\t16384\n'
  printf 'frontier_incomplete_group_count\t0\n'
  printf 'frontier_duplicate_new_group_count\t0\n'
  printf 'paper_evidence\tfalse\n'
  printf 'terminal_sacct_included\tfalse\n'
} > "$STAGE_DIR/receipt.tsv"

if find "$STAGE_DIR" -type l -print -quit | grep -q .; then
  echo "result stage contains a symbolic link" >&2
  exit 1
fi
(
  cd "$STAGE_DIR"
  # Exclude only this result root's markers. Nested prerequisite manifests and
  # completion markers are payloads and must remain in the outer closure.
  find . -type f ! -path ./SHA256SUMS ! -path ./.SHA256SUMS.tmp \
    ! -path ./COMPLETE ! -path ./.COMPLETE.tmp -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > .SHA256SUMS.tmp
  mv -T .SHA256SUMS.tmp SHA256SUMS
  sha256sum -c --strict SHA256SUMS >/dev/null
)
readonly RESULT_MANIFEST_SHA256="$(sha256sum "$STAGE_DIR/SHA256SUMS" | awk '{print $1}')"
UED_COMPLETE_TMP="$STAGE_DIR/.COMPLETE.tmp" UED_COMPLETE="$STAGE_DIR/COMPLETE" \
UED_JOB_ID="$SLURM_JOB_ID" UED_INPUT_CLOSURE="$INPUT_CLOSURE_SHA256" \
UED_RESULT_MANIFEST="$RESULT_MANIFEST_SHA256" "$PY" -I -B - <<'PY'
import json
import os
from pathlib import Path

record = {
    "complete_schema": 2,
    "artifact_type": "frontier_exact_grouped_one_update_engineering",
    "job_id": os.environ["UED_JOB_ID"],
    "paper_evidence": False,
    "actual_ppo_updates": 1,
    "n_grad_updates": 1,
    "ppo_epochs": 5,
    "ppo_minibatches": 1,
    "optimizer_step_applications": 5,
    "resource_accounting_source": (
        "python_resource_getrusage_self_and_monotonic_ns"),
    "external_accounting_authority": "terminal_slurm_sacct",
    "terminal_sacct_included": False,
    "input_closure_sha256": os.environ["UED_INPUT_CLOSURE"],
    "sha256sums_sha256": os.environ["UED_RESULT_MANIFEST"],
}
tmp = Path(os.environ["UED_COMPLETE_TMP"])
tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, Path(os.environ["UED_COMPLETE"]))
PY
[[ "$(sha256sum "$STAGE_DIR/SHA256SUMS" | awk '{print $1}')" \
    == "$RESULT_MANIFEST_SHA256" ]]
(cd "$STAGE_DIR" && sha256sum -c --strict SHA256SUMS >/dev/null)
verify_manifest "$STAGE_DIR" SHA256SUMS "$RESULT_MANIFEST_SHA256" result
mv -T -- "$STAGE_DIR" "$FINAL_DIR"
STAGE_DIR=""
echo "UED_MINIMAX_V4_ONE_UPDATE_PASS $FINAL_DIR"
