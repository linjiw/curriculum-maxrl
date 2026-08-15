#!/usr/bin/env bash
#SBATCH --job-name=ued-v4-terminal
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/%u/maxrl/tests/logs/%x_%j.out

# One-arm Phase A for the permanently non-evidence tie-aware v4 engineering
# rung.  Frontier and MaxMC are separate, ordinary Slurm submissions selected
# by UED_ARM.  This job never queries sacct, fetches results, assembles a final
# package, invokes an analyzer, or accesses a paper endpoint.
set -euo pipefail
umask 027
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES LD_LIBRARY_PATH
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONOPTIMIZE LD_PRELOAD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1

readonly EXPECTED_COMMIT=d053054c5290a04c1c4cd8b55704d999cad73e30
readonly EXPECTED_TREE=b0cace1fc54984e21a842f12d15d0b899e33d270
readonly EXPECTED_CONTRACT=3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b
readonly EXPECTED_PROTOCOL=1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269
readonly EXPECTED_APPLIED=9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae
readonly FRONTIER_CONFIG_SHA=0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2
readonly MAXMC_CONFIG_SHA=a3cc3ddf387a3bb7cf3d9759c3f13e6a74f2c7e32de311ac344d54ef5e703ec6

required=(
  UED_BUNDLE_DIR UED_BUNDLE_MANIFEST_SHA256 UED_UPSTREAM_COMMIT
  UED_UPSTREAM_TREE UED_UPSTREAM_BUNDLE_SHA256 UED_OVERLAY_MANIFEST_SHA256
  UED_ENV_DIR UED_ENV_LOCK_SHA256 UED_ENV_FREEZE_SHA256
  UED_ENV_MANIFEST_SHA256 UED_SBATCH_SHA256 UED_IMPORT_SMOKE_RESULT_DIR
  UED_IMPORT_SMOKE_MANIFEST_SHA256 UED_ONE_UPDATE_RESULT_DIR
  UED_ONE_UPDATE_MANIFEST_SHA256 UED_ARM UED_CONFIG_SHA256
  UED_CONTRACT_SHA256 UED_PROTOCOL_SHA256 UED_PHASE_A_DRIVER_SHA256
  UED_TRAINING_DRIVER_SHA256 UED_EVALUATION_DRIVER_SHA256
  UED_ASSEMBLER_SHA256 UED_FINALIZER_SHA256
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
            UED_OVERLAY_MANIFEST_SHA256 UED_ENV_LOCK_SHA256 \
            UED_ENV_FREEZE_SHA256 UED_ENV_MANIFEST_SHA256 UED_SBATCH_SHA256 \
            UED_IMPORT_SMOKE_MANIFEST_SHA256 UED_ONE_UPDATE_MANIFEST_SHA256 \
            UED_CONFIG_SHA256 UED_CONTRACT_SHA256 UED_PROTOCOL_SHA256 \
            UED_PHASE_A_DRIVER_SHA256 UED_TRAINING_DRIVER_SHA256 \
            UED_EVALUATION_DRIVER_SHA256 UED_ASSEMBLER_SHA256 \
            UED_FINALIZER_SHA256; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || { echo "bad SHA-256: $name" >&2; exit 2; }
done
[[ "$UED_UPSTREAM_COMMIT" == "$EXPECTED_COMMIT" ]]
[[ "$UED_UPSTREAM_TREE" == "$EXPECTED_TREE" ]]
[[ "$UED_CONTRACT_SHA256" == "$EXPECTED_CONTRACT" ]]
[[ "$UED_PROTOCOL_SHA256" == "$EXPECTED_PROTOCOL" ]]
[[ "$UED_ARM" == frontier || "$UED_ARM" == maxmc ]] || { echo "bad arm" >&2; exit 2; }
case "$UED_ARM" in
  frontier)
    readonly CONFIG_REL=ued_benchmark/configs/maze_frontier_exact_grouped_n8_tie_aware_v4.json
    readonly EXPECTED_CONFIG_SHA="$FRONTIER_CONFIG_SHA"
    ;;
  maxmc)
    readonly CONFIG_REL=ued_benchmark/configs/maze_maxmc_group_matched_4x8_b500_tie_aware_v4.json
    readonly EXPECTED_CONFIG_SHA="$MAXMC_CONFIG_SHA"
    ;;
esac
[[ "$UED_CONFIG_SHA256" == "$EXPECTED_CONFIG_SHA" ]] || { echo "arm/config swap" >&2; exit 2; }
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo "missing numeric Slurm job" >&2; exit 2; }
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "arrays forbidden" >&2; exit 2; }
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || { echo "restarted jobs forbidden" >&2; exit 2; }
[[ "${SLURM_CPUS_PER_TASK:-}" == 2 && "${SLURM_JOB_NUM_NODES:-1}" == 1 ]]

readonly RUN_USER="$(id -un)"
[[ "$RUN_USER" =~ ^[A-Za-z0-9._-]+$ && "$RUN_USER" != . && "$RUN_USER" != .. ]]
readonly SCRATCH_ROOT="/scratch/$RUN_USER"
require_dir() {
  local path=$1 prefix=$2
  [[ "$path" == /* && -d "$path" && ! -L "$path" ]] || return 1
  [[ "$(readlink -f -- "$path")" == "$path" ]] || return 1
  case "$path" in "$prefix"/*) ;; *) return 1;; esac
}
require_dir "$UED_BUNDLE_DIR" "$SCRATCH_ROOT/maxrl/bundles/ued_minimax_v4_engineering" \
  || { echo "unsafe v4 bundle" >&2; exit 2; }
[[ "$(basename -- "$UED_BUNDLE_DIR")" =~ ^[0-9a-f]{20}$ ]]
[[ "$(basename -- "$UED_BUNDLE_DIR")" \
    == "${UED_BUNDLE_MANIFEST_SHA256:0:20}" ]]
[[ "$(basename -- "$UED_BUNDLE_DIR")" != 06ffeeeb6998e8ddb1ce ]]
require_dir "$UED_ENV_DIR" "$SCRATCH_ROOT" || { echo "unsafe environment" >&2; exit 2; }
require_dir "$UED_IMPORT_SMOKE_RESULT_DIR" "$SCRATCH_ROOT/maxrl/tests/ued-minimax-v4-import" \
  || { echo "unsafe import prerequisite" >&2; exit 2; }
require_dir "$UED_ONE_UPDATE_RESULT_DIR" "$SCRATCH_ROOT/maxrl/tests/ued-minimax-v4-one-update" \
  || { echo "unsafe one-update prerequisite" >&2; exit 2; }
[[ "$(basename -- "$UED_IMPORT_SMOKE_RESULT_DIR")" =~ ^[0-9]+$ ]]
[[ "$(basename -- "$UED_ONE_UPDATE_RESULT_DIR")" =~ ^[0-9]+$ ]]

readonly PY="$UED_ENV_DIR/bin/python"
readonly GIT="$UED_ENV_DIR/bin/git"
[[ -x "$PY" && -x "$GIT" ]]
export PATH="$UED_ENV_DIR/bin:/usr/bin:/bin"
[[ "$(command -v git)" == "$GIT" ]]

verify_manifest() {
  local root=$1 name=$2 expected=$3 closure_mode=$4
  [[ -f "$root/$name" && ! -L "$root/$name" ]]
  [[ "$(sha256sum "$root/$name" | awk '{print $1}')" == "$expected" ]]
  UED_VERIFY_ROOT="$root" UED_VERIFY_MANIFEST="$root/$name" \
  UED_VERIFY_CLOSURE="$closure_mode" "$PY" -I -B - <<'PY'
import os, re
from pathlib import Path, PurePosixPath
root = Path(os.environ["UED_VERIFY_ROOT"])
manifest = Path(os.environ["UED_VERIFY_MANIFEST"])
mode = os.environ["UED_VERIFY_CLOSURE"]
assert mode in {"exact", "result", "metadata"}
assert root.resolve() == root and not root.is_symlink()
seen = set()
for line in manifest.read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\\]+)", line)
    assert match
    text = match.group(2); rel = PurePosixPath(text)
    assert text not in ("", ".") and not rel.is_absolute()
    assert all(part not in ("", ".", "..") for part in rel.parts)
    canonical = rel.as_posix()
    assert canonical not in seen
    seen.add(canonical)
    target = root.joinpath(*rel.parts)
    assert target.is_file() and not target.is_symlink()
    assert target.resolve().is_relative_to(root)
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

verify_manifest "$UED_BUNDLE_DIR" SHA256SUMS "$UED_BUNDLE_MANIFEST_SHA256" exact
verify_manifest "$UED_BUNDLE_DIR/ued_benchmark" OVERLAY_SHA256SUMS "$UED_OVERLAY_MANIFEST_SHA256" exact
verify_manifest "$UED_ENV_DIR" ENVIRONMENT_SHA256SUMS "$UED_ENV_MANIFEST_SHA256" metadata
[[ "$(sha256sum "$UED_ENV_DIR/ENVIRONMENT.freeze" | awk '{print $1}')" == "$UED_ENV_FREEZE_SHA256" ]]
[[ "$(awk 'NR == 1 {print $1}' "$UED_ENV_DIR/LOCK_SHA256")" == "$UED_ENV_LOCK_SHA256" ]]
"$PY" -I -B -m pip check
RECORDED_FREEZE=$(<"$UED_ENV_DIR/ENVIRONMENT.freeze")
CURRENT_FREEZE=$("$PY" -I -B -m pip freeze --all | LC_ALL=C sort)
[[ "$CURRENT_FREEZE" == "$RECORDED_FREEZE" ]] || { echo "installed package freeze drift" >&2; exit 1; }
unset RECORDED_FREEZE CURRENT_FREEZE
readonly CONDA_BIN="/home/$RUN_USER/miniconda3/bin/conda"
[[ -x "$CONDA_BIN" ]]
RECORDED_CONDA=$(<"$UED_ENV_DIR/CONDA_EXPLICIT.txt")
CURRENT_CONDA=$("$CONDA_BIN" list --explicit -p "$UED_ENV_DIR")
[[ "$CURRENT_CONDA" == "$RECORDED_CONDA" ]] || { echo "installed Conda state drift" >&2; exit 1; }
unset RECORDED_CONDA CURRENT_CONDA
readonly CONFIG="$UED_BUNDLE_DIR/$CONFIG_REL"
readonly PROTOCOL="$UED_BUNDLE_DIR/ued_benchmark/analysis/development_protocol_v2_tie_aware_draft.json"
readonly CONTRACT="$UED_BUNDLE_DIR/ued_benchmark/OVERLAY_CONTRACT_V4.json"
readonly PHASE_A="$UED_BUNDLE_DIR/ued_benchmark/hopper_v4/run_terminal_phase_a_v4.py"
readonly TRAINING="$UED_BUNDLE_DIR/ued_benchmark/hopper_v4/run_matched_terminal_v4.py"
readonly EVALUATION="$UED_BUNDLE_DIR/ued_benchmark/hopper_v4/evaluate_matched_terminal_v4.py"
readonly ASSEMBLER="$UED_BUNDLE_DIR/ued_benchmark/hopper_v4/assemble_matched_run_v4.py"
readonly FINALIZER="$UED_BUNDLE_DIR/hopper/finalize_ued_minimax_v4_terminal_chain.py"
readonly BUNDLED_SBATCH="$UED_BUNDLE_DIR/hopper/sbatch/ued_minimax_v4_terminal_chain_smoke.sbatch"
for binding in \
  "$CONFIG:$UED_CONFIG_SHA256" "$PROTOCOL:$UED_PROTOCOL_SHA256" \
  "$CONTRACT:$UED_CONTRACT_SHA256" "$PHASE_A:$UED_PHASE_A_DRIVER_SHA256" \
  "$TRAINING:$UED_TRAINING_DRIVER_SHA256" "$EVALUATION:$UED_EVALUATION_DRIVER_SHA256" \
  "$ASSEMBLER:$UED_ASSEMBLER_SHA256" "$FINALIZER:$UED_FINALIZER_SHA256" \
  "$BUNDLED_SBATCH:$UED_SBATCH_SHA256"; do
  path=${binding%%:*}; expected=${binding#*:}
  [[ -f "$path" && ! -L "$path" && "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]]
done
[[ "$(sha256sum "$0" | awk '{print $1}')" == "$UED_SBATCH_SHA256" ]]

UED_STATE="$UED_BUNDLE_DIR/BUNDLE_STATE.json" \
UED_BUNDLE="$UED_BUNDLE_MANIFEST_SHA256" UED_CONFIG="$UED_CONFIG_SHA256" \
UED_PHASE_A="$UED_PHASE_A_DRIVER_SHA256" UED_TRAIN="$UED_TRAINING_DRIVER_SHA256" \
UED_EVAL="$UED_EVALUATION_DRIVER_SHA256" UED_ASSEMBLER="$UED_ASSEMBLER_SHA256" \
UED_FINALIZER="$UED_FINALIZER_SHA256" UED_SBATCH="$UED_SBATCH_SHA256" \
"$PY" -I -B - <<'PY'
import json, os
state = json.load(open(os.environ["UED_STATE"], encoding="utf-8"))
assert state["bundle_schema"] == 5 and state["max_student_updates"] == 1
for key in ("paper_evidence", "analyzer_eligible", "endpoint_access_authorized",
            "production_authorized", "remote_stage_authorized"):
    assert state[key] is False
assert state["protected_v3_remote_job"] == {
    "bundle_id": "06ffeeeb6998e8ddb1ce",
    "bundle_manifest_sha256": "06ffeeeb6998e8ddb1ce516c8982ef8e78627f7cc876ea0b712dab466aa1e8ff",
    "job_id": "9367063", "mutation_forbidden": True,
    "reusable_as_v4_prerequisite": False}
assert state["core_bindings"]["contract_sha256"] == "3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b"
assert state["core_bindings"]["protocol_sha256"] == "1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269"
assert state["core_bindings"]["frontier_config_sha256"] == "0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2"
assert state["core_bindings"]["maxmc_config_sha256"] == "a3cc3ddf387a3bb7cf3d9759c3f13e6a74f2c7e32de311ac344d54ef5e703ec6"
assert state["official_pristine_source_configs"]["plr"]["sha256"] == "a5b8b87799bce31564959b3e8b55cfdaba658b31b8135fa35e3b96704d65185b"
assert state["official_pristine_source_configs"]["accel"]["sha256"] == "d7f100195159baff4f7490b755903e7586c895918ade4a248f5ae42b1827aee5"
expected = {
 "hopper/sbatch/ued_minimax_v4_terminal_chain_smoke.sbatch": os.environ["UED_SBATCH"],
 "hopper/finalize_ued_minimax_v4_terminal_chain.py": os.environ["UED_FINALIZER"],
 "ued_benchmark/hopper_v4/run_terminal_phase_a_v4.py": os.environ["UED_PHASE_A"],
 "ued_benchmark/hopper_v4/run_matched_terminal_v4.py": os.environ["UED_TRAIN"],
 "ued_benchmark/hopper_v4/evaluate_matched_terminal_v4.py": os.environ["UED_EVAL"],
 "ued_benchmark/hopper_v4/assemble_matched_run_v4.py": os.environ["UED_ASSEMBLER"],
}
for path, digest in expected.items(): assert state["engineering_file_hashes"][path] == digest
PY

# Both prerequisites must be from this exact v4 bundle.  The import rung has
# zero updates; the one-update rung is Frontier-only and has exactly one.
verify_manifest "$UED_IMPORT_SMOKE_RESULT_DIR" SHA256SUMS "$UED_IMPORT_SMOKE_MANIFEST_SHA256" result
verify_manifest "$UED_ONE_UPDATE_RESULT_DIR" SHA256SUMS "$UED_ONE_UPDATE_MANIFEST_SHA256" result
readonly IMPORT_COMPLETE_SHA="$(sha256sum "$UED_IMPORT_SMOKE_RESULT_DIR/COMPLETE" | awk '{print $1}')"
readonly ONE_COMPLETE_SHA="$(sha256sum "$UED_ONE_UPDATE_RESULT_DIR/COMPLETE" | awk '{print $1}')"
UED_IMPORT="$UED_IMPORT_SMOKE_RESULT_DIR" UED_ONE="$UED_ONE_UPDATE_RESULT_DIR" \
UED_BUNDLE="$UED_BUNDLE_MANIFEST_SHA256" "$PY" -I -B - <<'PY'
import json, os
from pathlib import Path
def tsv(path):
    rows = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        key, value = line.split("\t", 1)
        if number == 0:
            assert (key, value) == ("field", "value"); continue
        assert key not in rows; rows[key] = value
    return rows
imp = Path(os.environ["UED_IMPORT"]); one = Path(os.environ["UED_ONE"])
ir = tsv(imp / "receipt.tsv"); ore = tsv(one / "receipt.tsv")
assert ir["bundle_manifest_sha256"] == os.environ["UED_BUNDLE"]
assert ir["training_endpoint"] == "false"
assert ir["applied_overlay_manifest_sha256"] == "9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae"
assert ore["bundle_manifest_sha256"] == os.environ["UED_BUNDLE"]
assert ore["endpoint_class"] == "bounded_engineering_one_update"
assert ore["actual_student_updates"] == "1" and ore["paper_evidence"] == "false"
assert ore["config_sha256"] == "0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2"
assert ore["applied_overlay_manifest_sha256"] == "9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae"
complete = json.load(open(one / "COMPLETE", encoding="utf-8"))
assert complete["actual_ppo_updates"] == complete["n_grad_updates"] == 1
assert complete["paper_evidence"] is False and complete["terminal_sacct_included"] is False
PY

readonly UPSTREAM_BUNDLE="$UED_BUNDLE_DIR/upstream/minimax-${UED_UPSTREAM_COMMIT}.bundle"
[[ "$(sha256sum "$UPSTREAM_BUNDLE" | awk '{print $1}')" == "$UED_UPSTREAM_BUNDLE_SHA256" ]]
readonly WORK_PARENT="$SCRATCH_ROOT/maxrl/tests/work"
mkdir -p "$WORK_PARENT"
readonly WORK_DIR="$WORK_PARENT/ued-minimax-v4-terminal-$SLURM_JOB_ID-$UED_ARM"
[[ ! -e "$WORK_DIR" && ! -L "$WORK_DIR" ]]
mkdir "$WORK_DIR"
cleanup() {
  if [[ -n "${WORK_DIR:-}" && "$WORK_DIR" == "$WORK_PARENT/ued-minimax-v4-terminal-$SLURM_JOB_ID-$UED_ARM" && -d "$WORK_DIR" ]]; then
    rm -rf -- "$WORK_DIR"
  fi
}
trap cleanup EXIT
"$GIT" clone --quiet "$UPSTREAM_BUNDLE" "$WORK_DIR/patched"
[[ "$("$GIT" -C "$WORK_DIR/patched" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
[[ "$("$GIT" -C "$WORK_DIR/patched" rev-parse 'HEAD^{tree}')" == "$EXPECTED_TREE" ]]
readonly APPLY="$UED_BUNDLE_DIR/ued_benchmark/scripts/apply_minimax_overlay_v4.py"
"$PY" -I -B "$APPLY" --target "$WORK_DIR/patched" --check >/dev/null
"$PY" -I -B "$APPLY" --target "$WORK_DIR/patched" --apply >/dev/null
"$PY" -I -B "$APPLY" --target "$WORK_DIR/patched" --check >/dev/null
[[ "$(sha256sum "$WORK_DIR/patched/.frontierrl_overlay.json" | awk '{print $1}')" == "$EXPECTED_APPLIED" ]]
"$GIT" -C "$WORK_DIR/patched" diff --check

readonly PUBLISH_ROOT="$SCRATCH_ROOT/maxrl/tests/ued-minimax-v4-terminal-components"
mkdir -p "$PUBLISH_ROOT"
readonly FINAL_DIR="$PUBLISH_ROOT/$SLURM_JOB_ID-$UED_ARM"
[[ ! -e "$FINAL_DIR" && ! -L "$FINAL_DIR" ]] || { echo "result exists" >&2; exit 2; }
readonly INPUT_CLOSURE="$WORK_DIR/INPUT_CLOSURE.json"
UED_INPUT="$INPUT_CLOSURE" UED_ARM="$UED_ARM" UED_JOB_ID="$SLURM_JOB_ID" \
UED_CONFIG_REL="$CONFIG_REL" UED_CONFIG="$UED_CONFIG_SHA256" \
UED_BUNDLE="$UED_BUNDLE_MANIFEST_SHA256" UED_OVERLAY="$UED_OVERLAY_MANIFEST_SHA256" \
UED_ENV="$UED_ENV_MANIFEST_SHA256" UED_SBATCH="$UED_SBATCH_SHA256" \
UED_PHASE_A="$UED_PHASE_A_DRIVER_SHA256" UED_TRAIN="$UED_TRAINING_DRIVER_SHA256" \
UED_EVAL="$UED_EVALUATION_DRIVER_SHA256" UED_ASSEMBLER="$UED_ASSEMBLER_SHA256" \
UED_FINALIZER="$UED_FINALIZER_SHA256" UED_IMPORT_DIR="$UED_IMPORT_SMOKE_RESULT_DIR" \
UED_IMPORT_MANIFEST="$UED_IMPORT_SMOKE_MANIFEST_SHA256" UED_IMPORT_COMPLETE="$IMPORT_COMPLETE_SHA" \
UED_ONE_DIR="$UED_ONE_UPDATE_RESULT_DIR" UED_ONE_MANIFEST="$UED_ONE_UPDATE_MANIFEST_SHA256" \
UED_ONE_COMPLETE="$ONE_COMPLETE_SHA" "$PY" -I -B - <<'PY'
import hashlib, json, os
from pathlib import Path

def archive_provenance(root_text):
    root = Path(root_text)
    files = {}
    for name in ("receipt.tsv", "SHA256SUMS", "COMPLETE"):
        path = root / name
        assert path.is_file() and not path.is_symlink()
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        assert text.encode("utf-8") == raw
        files[name] = {
            "encoding": "utf-8",
            "text": text,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return {"schema": 1, "files": files}

record = {
 "schema": 1, "status": "frozen_before_phase_a",
 "purpose": "draft_engineering_development_only_no_endpoint_authorization_not_paper_evidence",
 "endpoint_class": "bounded_engineering_terminal_chain_components_v4",
 "paper_evidence": False, "analyzer_eligible": False,
 "endpoint_access_authorized": False, "production_authorized": False,
 "cost100_implemented": False, "max_student_updates": 1,
 "arm": os.environ["UED_ARM"], "training_seed": 101,
 "job_id": os.environ["UED_JOB_ID"], "attempt": 1,
 "from_last_checkpoint": False, "archive_interval": 0,
 "periodic_checkpoint_used": False, "no_requeue": True,
 "source_commit": "d053054c5290a04c1c4cd8b55704d999cad73e30",
 "source_tree": "b0cace1fc54984e21a842f12d15d0b899e33d270",
 "bundle_manifest_sha256": os.environ["UED_BUNDLE"],
 "overlay_manifest_sha256": os.environ["UED_OVERLAY"],
 "applied_overlay_manifest_sha256": "9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae",
 "environment_manifest_sha256": os.environ["UED_ENV"],
 "sbatch_sha256": os.environ["UED_SBATCH"],
 "phase_a_driver_sha256": os.environ["UED_PHASE_A"],
 "training_driver_sha256": os.environ["UED_TRAIN"],
 "evaluation_driver_sha256": os.environ["UED_EVAL"],
 "assembler_sha256": os.environ["UED_ASSEMBLER"],
 "finalizer_sha256": os.environ["UED_FINALIZER"],
 "protocol_sha256": "1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269",
 "config_path": os.environ["UED_CONFIG_REL"],
 "config_sha256": os.environ["UED_CONFIG"],
 "prerequisites": {
  "import": {"result_dir": os.environ["UED_IMPORT_DIR"],
   "manifest_sha256": os.environ["UED_IMPORT_MANIFEST"],
   "complete_sha256": os.environ["UED_IMPORT_COMPLETE"],
   "bundle_manifest_sha256": os.environ["UED_BUNDLE"],
   "actual_student_updates": 0, "paper_evidence": False, "analyzer_eligible": False},
  "one_update": {"result_dir": os.environ["UED_ONE_DIR"],
   "manifest_sha256": os.environ["UED_ONE_MANIFEST"],
   "complete_sha256": os.environ["UED_ONE_COMPLETE"],
   "bundle_manifest_sha256": os.environ["UED_BUNDLE"],
   "actual_student_updates": 1, "paper_evidence": False, "analyzer_eligible": False},
 }}
record["prerequisites"]["import"]["archived_provenance"] = archive_provenance(
    os.environ["UED_IMPORT_DIR"])
record["prerequisites"]["one_update"]["archived_provenance"] = archive_provenance(
    os.environ["UED_ONE_DIR"])
Path(os.environ["UED_INPUT"]).write_text(json.dumps(record, indent=2, sort_keys=True)+"\n", encoding="utf-8")
PY
readonly INPUT_CLOSURE_SHA="$(sha256sum "$INPUT_CLOSURE" | awk '{print $1}')"
readonly ISOLATED='import runpy,sys; from pathlib import Path; source=sys.argv.pop(1); helper=sys.argv.pop(1); script=sys.argv.pop(1); sys.path[:0]=[source,helper]; sys.argv[0]=script; runpy.run_path(script,run_name="__main__")'
"$PY" -I -B -c "$ISOLATED" "$WORK_DIR/patched/src" \
  "$UED_BUNDLE_DIR/ued_benchmark/hopper_v4" "$PHASE_A" \
  --arm "$UED_ARM" --job-id "$SLURM_JOB_ID" --protocol "$PROTOCOL" \
  --config "$CONFIG" --patched-source-dir "$WORK_DIR/patched" \
  --git-executable "$GIT" --python "$PY" --input-closure "$INPUT_CLOSURE" \
  --expected-input-closure-sha256 "$INPUT_CLOSURE_SHA" \
  --bundle-manifest-sha256 "$UED_BUNDLE_MANIFEST_SHA256" \
  --overlay-manifest-sha256 "$UED_OVERLAY_MANIFEST_SHA256" \
  --applied-overlay-manifest-sha256 "$EXPECTED_APPLIED" \
  --environment-manifest-sha256 "$UED_ENV_MANIFEST_SHA256" \
  --sbatch-sha256 "$UED_SBATCH_SHA256" \
  --expected-phase-a-driver-sha256 "$UED_PHASE_A_DRIVER_SHA256" \
  --expected-training-driver-sha256 "$UED_TRAINING_DRIVER_SHA256" \
  --expected-evaluation-driver-sha256 "$UED_EVALUATION_DRIVER_SHA256" \
  --expected-assembler-sha256 "$UED_ASSEMBLER_SHA256" \
  --expected-finalizer-sha256 "$UED_FINALIZER_SHA256" \
  --output-dir "$WORK_DIR/components" --slurm-engineering-test-mode \
  > "$WORK_DIR/phase-a.stdout" 2> "$WORK_DIR/phase-a.stderr"
[[ -d "$WORK_DIR/components" && ! -L "$WORK_DIR/components" ]]
[[ -f "$WORK_DIR/components/COMPONENTS_COMPLETE.json" ]]
mv -T -- "$WORK_DIR/components" "$FINAL_DIR"
echo "UED_MINIMAX_V4_TERMINAL_PHASE_A_PASS arm=$UED_ARM result=$FINAL_DIR phase_b_required=true"
