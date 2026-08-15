#!/usr/bin/env bash
# Deterministically layer remote-hardening siblings over the frozen d602 bundle.
# Usage: bash ... local ABS_D602_BUNDLE ABS_NEW_OUTPUT_PARENT
# There is deliberately no remote, SSH, rsync, scp, scheduler, or submit mode.
set -euo pipefail
umask 027
export PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

[[ "${1:-}" == local && $# == 3 ]] || {
  echo "local-only usage: $0 local ABS_D602_BUNDLE ABS_NEW_OUTPUT_PARENT" >&2
  exit 2
}
readonly BASE=$2
readonly OUTPUT_PARENT=$3
readonly BASE_SHA=d602ce7854f8f3e99352025b97eed2fde32733c0dd23297d5c28b1051e7aeaf0
readonly HERE="$(cd "$(dirname "$0")" && pwd -P)"
readonly ROOT="$(cd "$HERE/.." && pwd -P)"
for path in "$BASE" "$OUTPUT_PARENT"; do
  [[ "$path" == /* && "$path" != *'/../'* && -d "$path" && ! -L "$path" \
     && "$(readlink -f -- "$path")" == "$path" ]] || {
    echo "unsafe staging path: $path" >&2; exit 2;
  }
done
[[ "$BASE" != "$OUTPUT_PARENT" ]] || { echo "base/output overlap" >&2; exit 2; }
case "$OUTPUT_PARENT/" in "$BASE/"*) echo "base/output overlap" >&2; exit 2;; esac
case "$BASE/" in "$OUTPUT_PARENT/"*) echo "base/output overlap" >&2; exit 2;; esac
[[ "$(sha256sum "$BASE/SHA256SUMS" | awk '{print $1}')" == "$BASE_SHA" ]] || {
  echo "historical d602 base manifest drift" >&2; exit 1;
}

UED_STAGE_ROOT="$BASE" /usr/bin/python3 -I -B - <<'PY'
import hashlib, os, re
from pathlib import Path, PurePosixPath
root=Path(os.environ['UED_STAGE_ROOT'])
assert root.resolve()==root and not root.is_symlink()
listed={}
for raw in (root/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
    m=re.fullmatch(r'([0-9a-f]{64})  (.+)',raw); assert m,raw
    text=m.group(2).removeprefix('./'); rel=PurePosixPath(text)
    assert text and not rel.is_absolute() and all(p not in ('','.','..') for p in rel.parts)
    assert rel.as_posix() not in listed
    target=root.joinpath(*rel.parts); assert target.is_file() and not target.is_symlink()
    assert hashlib.sha256(target.read_bytes()).hexdigest()==m.group(1)
    listed[rel.as_posix()]=m.group(1)
actual=set()
for target in root.rglob('*'):
    assert not target.is_symlink(),target
    if target.is_file():
        rel=target.relative_to(root).as_posix()
        if rel!='SHA256SUMS': actual.add(rel)
    else: assert target.is_dir(),target
assert actual==set(listed),(sorted(actual-set(listed)),sorted(set(listed)-actual))
PY

readonly TMP="$(mktemp -d /tmp/ued-v4h-stage.XXXXXX)"
cleanup() {
  if [[ -n "${TMP:-}" && "$TMP" == /tmp/ued-v4h-stage.* && -d "$TMP" ]]; then
    rm -rf -- "$TMP"
  fi
}
trap cleanup EXIT
mkdir "$TMP/source"
cp -a -- "$BASE/." "$TMP/source/"

readonly CALIBRATION_PROTOCOL=ued_benchmark/analysis/frontier_calibration_telemetry_v1_draft.json
readonly CALIBRATION_ANALYZER=ued_benchmark/analysis/frontier_calibration_telemetry.py
readonly CALIBRATION_TEST=ued_benchmark/tests/test_frontier_calibration_telemetry.py
[[ "$(sha256sum "$ROOT/$CALIBRATION_PROTOCOL" | awk '{print $1}')" == 4053c52052ade233224903b0c989d9f39b1a626762209da93c4432428c430004 ]]
[[ "$(sha256sum "$ROOT/$CALIBRATION_ANALYZER" | awk '{print $1}')" == 19b07d2f88f46221c53b1d607ca6198857cf378249f61c6599bc5867adcc9816 ]]
[[ "$(sha256sum "$ROOT/$CALIBRATION_TEST" | awk '{print $1}')" == c0597e3ce863f2a34bb45996483cdfa2b89a9018ea425f395f6c1e6dd0e8a621 ]]
for rel in "$CALIBRATION_PROTOCOL" "$CALIBRATION_ANALYZER" "$CALIBRATION_TEST"; do
  cp -- "$ROOT/$rel" "$TMP/source/$rel"
done

if find "$ROOT/ued_benchmark/hopper_v4_remote_hardening" -type l -print -quit | grep -q .; then
  echo "remote-hardening source contains a symlink" >&2; exit 1
fi
while IFS= read -r -d '' source; do
  rel=${source#"$ROOT/"}
  mkdir -p "$TMP/source/$(dirname "$rel")"
  cp -- "$source" "$TMP/source/$rel"
done < <(find "$ROOT/ued_benchmark/hopper_v4_remote_hardening" -type f \
  ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 | LC_ALL=C sort -z)

HOPPER_FILES=(
  hopper/hopper_v4_remote_hardened.py
  hopper/hopper_v4_remote_hardened.sh
  hopper/run_ued_minimax_v4_remote_hardened.sh
  hopper/setup_ued_minimax_env_v4_remote_hardened.sh
  hopper/stage_ued_minimax_v4_remote_hardened.sh
  hopper/finalize_ued_minimax_v4_remote_hardened.py
  hopper/sbatch/ued_minimax_v4_remote_hardened_gpu_smoke.sbatch
  hopper/sbatch/ued_minimax_v4_remote_hardened_one_update_smoke.sbatch
  hopper/sbatch/ued_minimax_v4_remote_hardened_terminal_chain_smoke.sbatch
)
for rel in "${HOPPER_FILES[@]}"; do
  [[ -f "$ROOT/$rel" && ! -L "$ROOT/$rel" ]] || { echo "missing hardening file: $rel" >&2; exit 1; }
  mkdir -p "$TMP/source/$(dirname "$rel")"
  cp -- "$ROOT/$rel" "$TMP/source/$rel"
done

(
  cd "$TMP/source/ued_benchmark"
  find . -type f ! -name OVERLAY_SHA256SUMS -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > OVERLAY_SHA256SUMS
  sha256sum -c --strict OVERLAY_SHA256SUMS >/dev/null
)
readonly OVERLAY_SHA="$(sha256sum "$TMP/source/ued_benchmark/OVERLAY_SHA256SUMS" | awk '{print $1}')"

UED_STATE="$TMP/source/REMOTE_HARDENING_STATE.json" \
UED_SOURCE="$TMP/source" UED_OVERLAY_SHA="$OVERLAY_SHA" \
/usr/bin/python3 -I -B - <<'PY'
import hashlib,json,os
from pathlib import Path
root=Path(os.environ['UED_SOURCE'])
def digest(path): return hashlib.sha256((root/path).read_bytes()).hexdigest()
engineering=[
 'hopper/hopper_v4_remote_hardened.py','hopper/hopper_v4_remote_hardened.sh',
 'hopper/run_ued_minimax_v4_remote_hardened.sh',
 'hopper/setup_ued_minimax_env_v4_remote_hardened.sh',
 'hopper/stage_ued_minimax_v4_remote_hardened.sh',
 'hopper/finalize_ued_minimax_v4_remote_hardened.py',
 'hopper/sbatch/ued_minimax_v4_remote_hardened_gpu_smoke.sbatch',
 'hopper/sbatch/ued_minimax_v4_remote_hardened_one_update_smoke.sbatch',
 'hopper/sbatch/ued_minimax_v4_remote_hardened_terminal_chain_smoke.sbatch',
]
engineering += [
 p.relative_to(root).as_posix()
 for p in sorted((root/'ued_benchmark/hopper_v4_remote_hardening').glob('*.py'))
]
state={
 'schema':1,'status':'local_candidate_remote_submission_hold',
 'purpose':'sibling-only_v4_remote_hardening_rungs_0_3_not_paper_evidence',
 'paper_evidence':False,'analyzer_eligible':False,
 'production_authorized':False,'endpoint_access_authorized':False,
 'cost100_implemented':False,'remote_stage_authorized':False,
 'remote_submission_authorized':False,'max_student_updates':1,
 'historical_base_bundle_manifest_sha256':'d602ce7854f8f3e99352025b97eed2fde32733c0dd23297d5c28b1051e7aeaf0',
 'historical_base_reusable_for_submission':False,
 'slurm_input_contract':{
   'client_environment':'/usr/bin/env -i','export_mode':'NIL',
   'get_user_env':False,'explicit_export_assignments':False,
   'runtime_input':'exact_NUL_envelope_as_batch_script_argument',
   'export_file_combined_with_NIL':False,'submit_command_implemented':False,
   'ambient_SBATCH_or_UED_controls_forbidden':True,
 },
 'resource_contract':{
   'partition':'gpuq','qos':'gpu','gres':'gpu:1g.10gb:1','gpu_count':1,
   'cpus_per_task':2,'memory':'15G','nodes':1,'tasks':1,
   'terminal_time_minutes':45,'arrays':False,'requeue':False,'restarts':0,
   'post_terminal_steps':['batch','extern'],
 },
 'environment_contract':{
   'python':'3.10.20','conda_path':'/home/<runtime-user>/miniconda3/bin/conda',
   'installed_file_byte_closure':True,'closure_external_to_environment':True,
   'phase_b_same_closed_environment_required':True,
 },
 'phase_boundary':{
   'phase_a_inside_slurm':True,'phase_b_post_terminal_only':True,
   'terminal_receipt_required':True,'fetch_after_terminal_required':True,
   'performance_values_inspected':False,'production_analyzer_invoked':False,
 },
 'pair_contract':{
   'separate_non_array_jobs':True,'training_seed':101,
   'arms':['frontier','maxmc'],'common_r1_r2_pair_plan_required':True,
 },
 'protected_v3_job':{
   'job_id':'9367063','bundle_id':'06ffeeeb6998e8ddb1ce',
   'mutation_forbidden':True,'cancel_requeue_relabel_forbidden':True,
   'reusable_as_v4_prerequisite':False,
 },
 'v4_core':{
   'contract_sha256':'3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b',
   'protocol_sha256':'1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269',
   'frontier_config_sha256':'0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2',
   'maxmc_config_sha256':'a3cc3ddf387a3bb7cf3d9759c3f13e6a74f2c7e32de311ac344d54ef5e703ec6',
   'official_maxmc_config_sha256':'a5b8b87799bce31564959b3e8b55cfdaba658b31b8135fa35e3b96704d65185b',
   'applied_overlay_manifest_sha256':'9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae',
 },
 'official_pristine_source_configs':{
   'overlay_execution_forbidden':True,
   'plr_sha256':'a5b8b87799bce31564959b3e8b55cfdaba658b31b8135fa35e3b96704d65185b',
   'accel_sha256':'d7f100195159baff4f7490b755903e7586c895918ade4a248f5ae42b1827aee5',
 },
 'calibration_advisory':{
   'activated':False,'required_for_rungs_0_3':False,
   'protocol_sha256':'4053c52052ade233224903b0c989d9f39b1a626762209da93c4432428c430004',
   'analyzer_sha256':'19b07d2f88f46221c53b1d607ca6198857cf378249f61c6599bc5867adcc9816',
   'tests_sha256':'c0597e3ce863f2a34bb45996483cdfa2b89a9018ea425f395f6c1e6dd0e8a621',
 },
 'overlay_manifest_sha256':os.environ['UED_OVERLAY_SHA'],
 'engineering_file_hashes':{path:digest(path) for path in engineering},
}
Path(os.environ['UED_STATE']).write_text(
 json.dumps(state,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
PY

(
  cd "$TMP/source"
  find . -type f ! -path ./SHA256SUMS -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c --strict SHA256SUMS >/dev/null
)
readonly MANIFEST_SHA="$(sha256sum "$TMP/source/SHA256SUMS" | awk '{print $1}')"
readonly BUNDLE_ID=${MANIFEST_SHA:0:20}
readonly OUTPUT="$OUTPUT_PARENT/$BUNDLE_ID"
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || { echo "output bundle exists: $OUTPUT" >&2; exit 2; }
mv -T -- "$TMP/source" "$OUTPUT"
UED_STAGE_ROOT="$OUTPUT" /usr/bin/python3 -I -B - <<'PY'
import hashlib,os,re
from pathlib import Path,PurePosixPath
root=Path(os.environ['UED_STAGE_ROOT']); listed={}
for raw in (root/'SHA256SUMS').read_text().splitlines():
 m=re.fullmatch(r'([0-9a-f]{64})  (.+)',raw); assert m
 rel=PurePosixPath(m.group(2).removeprefix('./')); assert all(p not in ('','.','..') for p in rel.parts)
 target=root.joinpath(*rel.parts); assert target.is_file() and not target.is_symlink()
 assert hashlib.sha256(target.read_bytes()).hexdigest()==m.group(1); listed[rel.as_posix()]=m.group(1)
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix()!='SHA256SUMS'}
assert not any(p.is_symlink() for p in root.rglob('*')); assert actual==set(listed)
PY
trap - EXIT
rmdir "$TMP"
printf 'V4H_BUNDLE_COMPLETE\t%s\t%s\t%s\n' "$BUNDLE_ID" "$MANIFEST_SHA" "$OUTPUT"
