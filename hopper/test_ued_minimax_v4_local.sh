#!/usr/bin/env bash
# Local-only Rung-0 bundle, export-closure, and deterministic-build checks.
set -euo pipefail
umask 077
readonly HERE="$(cd "$(dirname "$0")" && pwd)"
readonly ROOT="$(cd "$HERE/.." && pwd)"
readonly SOURCE="${MINIMAX_SOURCE_DIR:-/tmp/root-minimax-260814}"
[[ -d "$SOURCE/.git" || -f "$SOURCE/.git" ]]
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == d053054c5290a04c1c4cd8b55704d999cad73e30 ]]
readonly TMP="$(mktemp -d /tmp/ued-v4-r0-local.XXXXXX)"
cleanup() { [[ "$TMP" == /tmp/ued-v4-r0-local.* && -d "$TMP" ]] && rm -rf -- "$TMP"; }
trap cleanup EXIT

# Frozen history/core stays byte-identical while only sibling files evolve.
checks=(
  'ued_benchmark/OVERLAY_CONTRACT_V4.json 3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b'
  'ued_benchmark/analysis/development_protocol_v2_tie_aware_draft.json 1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269'
  'ued_benchmark/configs/maze_frontier_exact_grouped_n8_tie_aware_v4.json 0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2'
  'ued_benchmark/configs/maze_maxmc_group_matched_4x8_b500_tie_aware_v4.json a3cc3ddf387a3bb7cf3d9759c3f13e6a74f2c7e32de311ac344d54ef5e703ec6'
  'ued_benchmark/configs/maze_maxmc_upstream_official_reference_32x1_b4000.json a5b8b87799bce31564959b3e8b55cfdaba658b31b8135fa35e3b96704d65185b'
  'hopper/stage_ued_minimax.sh 73ad318fe21f6f99c92fe09ac6ec76c5dd6fe4b0d7fec8a5ca5939c59483ba55'
  'hopper/sbatch/ued_minimax_terminal_chain_smoke.sbatch 5ed5186e010decdcc6bf97ff7dc820e0f4cf13e580e9b20c996a8dc561b13a14'
  'hopper/finalize_ued_minimax_terminal_chain.py 57eb4394cedf30cc1a5bfeca4734199652cbfcfd5fbcaaca08035e8001a2c5ec'
)
for binding in "${checks[@]}"; do
  path=${binding% *}; expected=${binding##* }
  [[ "$(sha256sum "$ROOT/$path" | awk '{print $1}')" == "$expected" ]]
done

bash -n "$ROOT/hopper/stage_ued_minimax_v4.sh" "$ROOT/hopper/hopper_v4.sh" \
  "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" \
  "$ROOT/hopper/sbatch/ued_minimax_v4_one_update_smoke.sbatch" \
  "$ROOT/hopper/sbatch/ued_minimax_v4_terminal_chain_smoke.sbatch"
for script in \
  "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" \
  "$ROOT/hopper/sbatch/ued_minimax_v4_one_update_smoke.sbatch" \
  "$ROOT/hopper/sbatch/ued_minimax_v4_terminal_chain_smoke.sbatch"; do
  [[ "$(grep -Fc '#SBATCH --no-requeue' "$script")" == 1 ]]
  grep -Fq 'SLURM_ARRAY_JOB_ID' "$script"
  grep -Fq 'SLURM_RESTART_COUNT' "$script"
  ! grep -Eq '(\$\{?HOME|UED_CONDA_BIN)' "$script"
done
UED_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 python3 -I -B - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["UED_ROOT"])
for relative in (
    "ued_benchmark/hopper_v4/run_matched_terminal_v4.py",
    "ued_benchmark/hopper_v4/evaluate_matched_terminal_v4.py",
    "ued_benchmark/hopper_v4/run_terminal_phase_a_v4.py",
    "ued_benchmark/hopper_v4/assemble_matched_run_v4.py",
    "hopper/finalize_ued_minimax_v4_terminal_chain.py",
):
    path = root / relative
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
PY

fake=$(printf 'a%.0s' {1..64})
common=(
  UED_BUNDLE_DIR=/scratch/test/maxrl/bundles/ued_minimax_v4_engineering/aaaaaaaaaaaaaaaaaaaa
  UED_BUNDLE_MANIFEST_SHA256="$fake"
  UED_UPSTREAM_COMMIT=d053054c5290a04c1c4cd8b55704d999cad73e30
  UED_UPSTREAM_TREE=b0cace1fc54984e21a842f12d15d0b899e33d270
  UED_UPSTREAM_BUNDLE_SHA256="$fake"
  UED_OVERLAY_MANIFEST_SHA256="$fake"
  UED_ENV_DIR=/scratch/test/envs/ued-minimax-v2-aaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb
  UED_ENV_LOCK_SHA256="$fake" UED_ENV_FREEZE_SHA256="$fake"
  UED_ENV_MANIFEST_SHA256="$fake"
)
render_import=("${common[@]}" UED_SBATCH_SHA256="$(sha256sum "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" | awk '{print $1}')")
bash "$ROOT/hopper/hopper_v4.sh" render "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" "${render_import[@]}" > "$TMP/import.render"
grep -Fq 'sbatch --parsable --export=' "$TMP/import.render"
! grep -Eq -- '--export=(ALL|NONE)(,|[[:space:]])' "$TMP/import.render"
! bash "$ROOT/hopper/hopper_v4.sh" submit "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" "${render_import[@]}" >/dev/null 2>&1
! bash "$ROOT/hopper/hopper_v4.sh" render "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" "${render_import[@]:1}" >/dev/null 2>&1
! bash "$ROOT/hopper/hopper_v4.sh" render "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" "${render_import[@]}" UED_ARM=frontier >/dev/null 2>&1
! bash "$ROOT/hopper/hopper_v4.sh" render "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" "${render_import[@]}" "${render_import[0]}" >/dev/null 2>&1
! bash "$ROOT/hopper/hopper_v4.sh" render "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" "${render_import[@]}" SBATCH_JOB_NAME=x >/dev/null 2>&1
! bash "$ROOT/hopper/hopper_v4.sh" render "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" "${render_import[@]}" ALL=true >/dev/null 2>&1
bad_empty=("${render_import[@]}")
bad_empty[0]=UED_BUNDLE_DIR=
! bash "$ROOT/hopper/hopper_v4.sh" render "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" "${bad_empty[@]}" >/dev/null 2>&1
bad_comma=("${render_import[@]}")
bad_comma[0]=UED_BUNDLE_DIR=/scratch/test/maxrl/bundles/ued_minimax_v4_engineering/aaaaaaaaaaaaaaaaaaaa,ALL
! bash "$ROOT/hopper/hopper_v4.sh" render "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" "${bad_comma[@]}" >/dev/null 2>&1
bad_alias=("${render_import[@]}")
bad_alias[0]=UED_BUNDLE_DIR=/scratch/test/maxrl/bundles/ued_minimax_v4_engineering/bbbbbbbbbbbbbbbbbbbb
! bash "$ROOT/hopper/hopper_v4.sh" render "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" "${bad_alias[@]}" >/dev/null 2>&1
bad_self=("${render_import[@]}")
bad_self[$((${#bad_self[@]} - 1))]=UED_SBATCH_SHA256="$fake"
! bash "$ROOT/hopper/hopper_v4.sh" render "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" "${bad_self[@]}" >/dev/null 2>&1
PYTHONPATH=/tmp/ambient LD_LIBRARY_PATH=/tmp/ambient \
  bash "$ROOT/hopper/hopper_v4.sh" render "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" \
  "${render_import[@]}" > "$TMP/import-ambient.render"
cmp "$TMP/import.render" "$TMP/import-ambient.render"
! SBATCH_JOB_NAME=ambient bash "$ROOT/hopper/hopper_v4.sh" render \
  "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" \
  "${render_import[@]}" >/dev/null 2>&1
! SBATCH_ARRAY_INX=0 bash "$ROOT/hopper/hopper_v4.sh" render \
  "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" \
  "${render_import[@]}" >/dev/null 2>&1
env -i PATH=/usr/bin:/bin \
  bash "$ROOT/hopper/hopper_v4.sh" render \
  "$ROOT/hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch" \
  "${render_import[@]}" > "$TMP/import-empty-environment.render"
cmp "$TMP/import.render" "$TMP/import-empty-environment.render"

one=("${common[@]}" UED_SBATCH_SHA256="$(sha256sum "$ROOT/hopper/sbatch/ued_minimax_v4_one_update_smoke.sbatch" | awk '{print $1}')"
  UED_IMPORT_SMOKE_RESULT_DIR=/scratch/test/maxrl/tests/ued-minimax-v4-import/1
  UED_IMPORT_SMOKE_MANIFEST_SHA256="$fake"
  UED_CONFIG_SHA256=0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2
  UED_CONTRACT_SHA256=3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b)
bash "$ROOT/hopper/hopper_v4.sh" render "$ROOT/hopper/sbatch/ued_minimax_v4_one_update_smoke.sbatch" "${one[@]}" > "$TMP/one.render"

terminal_base=("${common[@]}" UED_SBATCH_SHA256="$(sha256sum "$ROOT/hopper/sbatch/ued_minimax_v4_terminal_chain_smoke.sbatch" | awk '{print $1}')"
  UED_IMPORT_SMOKE_RESULT_DIR=/scratch/test/maxrl/tests/ued-minimax-v4-import/1
  UED_IMPORT_SMOKE_MANIFEST_SHA256="$fake"
  UED_ONE_UPDATE_RESULT_DIR=/scratch/test/maxrl/tests/ued-minimax-v4-one-update/2
  UED_ONE_UPDATE_MANIFEST_SHA256="$fake"
  UED_CONTRACT_SHA256=3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b
  UED_PROTOCOL_SHA256=1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269
  UED_PHASE_A_DRIVER_SHA256="$(sha256sum "$ROOT/ued_benchmark/hopper_v4/run_terminal_phase_a_v4.py" | awk '{print $1}')"
  UED_TRAINING_DRIVER_SHA256="$(sha256sum "$ROOT/ued_benchmark/hopper_v4/run_matched_terminal_v4.py" | awk '{print $1}')"
  UED_EVALUATION_DRIVER_SHA256="$(sha256sum "$ROOT/ued_benchmark/hopper_v4/evaluate_matched_terminal_v4.py" | awk '{print $1}')"
  UED_ASSEMBLER_SHA256="$(sha256sum "$ROOT/ued_benchmark/hopper_v4/assemble_matched_run_v4.py" | awk '{print $1}')"
  UED_FINALIZER_SHA256="$(sha256sum "$ROOT/hopper/finalize_ued_minimax_v4_terminal_chain.py" | awk '{print $1}')")
for arm in frontier maxmc; do
  if [[ "$arm" == frontier ]]; then config=0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2; else config=a3cc3ddf387a3bb7cf3d9759c3f13e6a74f2c7e32de311ac344d54ef5e703ec6; fi
  bash "$ROOT/hopper/hopper_v4.sh" render "$ROOT/hopper/sbatch/ued_minimax_v4_terminal_chain_smoke.sbatch" \
    "${terminal_base[@]}" UED_ARM="$arm" UED_CONFIG_SHA256="$config" > "$TMP/$arm.render"
done
! bash "$ROOT/hopper/hopper_v4.sh" render "$ROOT/hopper/sbatch/ued_minimax_v4_terminal_chain_smoke.sbatch" \
  "${terminal_base[@]}" UED_ARM=frontier UED_CONFIG_SHA256=a3cc3ddf387a3bb7cf3d9759c3f13e6a74f2c7e32de311ac344d54ef5e703ec6 >/dev/null 2>&1

# Two fresh local builds must be byte-identical and content-addressed.
MINIMAX_SOURCE_DIR="$SOURCE" bash "$ROOT/hopper/stage_ued_minimax_v4.sh" local "$TMP/bundle-a" > "$TMP/a.receipt"
MINIMAX_SOURCE_DIR="$SOURCE" bash "$ROOT/hopper/stage_ued_minimax_v4.sh" local "$TMP/bundle-b" > "$TMP/b.receipt"
grep -v '^UED_LOCAL_BUNDLE=' "$TMP/a.receipt" > "$TMP/a.normalized"
grep -v '^UED_LOCAL_BUNDLE=' "$TMP/b.receipt" > "$TMP/b.normalized"
cmp "$TMP/a.normalized" "$TMP/b.normalized"
cmp "$TMP/bundle-a/SHA256SUMS" "$TMP/bundle-b/SHA256SUMS"
diff -qr "$TMP/bundle-a" "$TMP/bundle-b" >/dev/null
(cd "$TMP/bundle-a" && sha256sum -c --strict SHA256SUMS >/dev/null)
# Exact-tree verification must reject a regular file that is not named by the
# otherwise valid root manifest.  This mirrors the Slurm-side closure mode.
UED_TEST_BUNDLE="$TMP/bundle-a" python3 -I -B - <<'PY'
import os, re
from pathlib import Path, PurePosixPath

root = Path(os.environ["UED_TEST_BUNDLE"])
manifest = root / "SHA256SUMS"
def verify_exact():
    seen = set()
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\\]+)", raw)
        assert match
        rel = PurePosixPath(match.group(2)); canonical = rel.as_posix()
        assert canonical not in seen
        seen.add(canonical)
    actual = set()
    for target in root.rglob("*"):
        assert not target.is_symlink()
        if target.is_file(): actual.add(target.relative_to(root).as_posix())
        else: assert target.is_dir()
    assert actual == seen | {"SHA256SUMS"}

verify_exact()
injected = root / "unlisted-injection.py"
injected.write_text("raise RuntimeError('must never import')\n", encoding="utf-8")
try:
    verify_exact()
except AssertionError:
    pass
else:
    raise AssertionError("exact manifest closure accepted an unlisted file")
injected.unlink()
verify_exact()
PY
UED_STATE="$TMP/bundle-a/BUNDLE_STATE.json" python3 -I -B - <<'PY'
import json, os
state = json.load(open(os.environ["UED_STATE"], encoding="utf-8"))
assert state["bundle_schema"] == 5 and state["max_student_updates"] == 1
for key in ("paper_evidence", "analyzer_eligible", "endpoint_access_authorized",
            "production_authorized", "remote_stage_authorized"):
    assert state[key] is False
assert state["protected_v3_remote_job"]["job_id"] == "9367063"
assert state["protected_v3_remote_job"]["mutation_forbidden"] is True
assert state["official_pristine_source_configs"]["plr"]["sha256"] == "a5b8b87799bce31564959b3e8b55cfdaba658b31b8135fa35e3b96704d65185b"
assert state["official_pristine_source_configs"]["accel"]["sha256"] == "d7f100195159baff4f7490b755903e7586c895918ade4a248f5ae42b1827aee5"
assert state["calibration_telemetry_advisory"]["activated"] is False
assert state["remote_launch_hold"] == {
    "authorized": False,
    "hopper_behavior_validated": False,
    "resolution_required": "independent audit plus a separately authorized Hopper validation of sanitized export semantics",
    "slurm_assignment_export_may_invoke_get_user_env": True,
    "submission_command_implemented": False,
}
assert state["environment_integrity_scope"]["installed_file_byte_closure"] is False
assert state["paper_lane_outstanding"]["cost100_implemented"] is False
assert state["paper_lane_outstanding"]["five_seed_two_sided_sign_flip_is_descriptive_only"] is True
terminal = state["terminal_chain_contract"]
assert terminal["phase_b_host_scope"] == "post_fetch_operator_workstation"
assert terminal["phase_b_required_python_version"] == "3.10.20"
assert terminal["phase_b_environment_manifest_bound"] is False
assert terminal["phase_b_non_test_general_release_ready"] is False
PY
echo "UED_MINIMAX_V4_R0_LOCAL_PASS"
