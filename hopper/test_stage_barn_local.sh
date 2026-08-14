#!/bin/bash
# Local, network-free contract test for stage_barn_campaign.sh.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
STAGER="$HERE/stage_barn_campaign.sh"
ARCHIVE=${BARN_DATASET_ZIP:-/home/robotixx/datasets/barn/BARN_dataset.zip}
EXPECTED_ARCHIVE_SHA256=5ad443412f6f2f38b6d0e1d330c9a820ab48e566553197459005e751711fe320

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -f "$STAGER" ]] || fail "missing stager: $STAGER"
[[ -f "$ARCHIVE" && ! -L "$ARCHIVE" ]] \
  || fail "set BARN_DATASET_ZIP to the official BARN_dataset.zip"
[[ "$(sha256sum -- "$ARCHIVE" | awk '{print $1}')" == \
   "$EXPECTED_ARCHIVE_SHA256" ]] \
  || fail "local test requires the official hash-verified BARN archive"

TMP=$(mktemp -d)
cleanup() {
  if [[ -n "${TMP:-}" && "$TMP" == /tmp/* && -d "$TMP" ]]; then
    rm -rf -- "$TMP"
  fi
}
trap cleanup EXIT

FIXTURE="$TMP/repo"
MOCK_BIN="$TMP/mock-bin"
FAKE_REMOTE_ROOT="$TMP/remote"
SSH_LOG="$TMP/ssh.log"
RSYNC_LOG="$TMP/rsync.log"
mkdir -p "$FIXTURE/hopper" "$MOCK_BIN" "$FAKE_REMOTE_ROOT"
: > "$SSH_LOG"
: > "$RSYNC_LOG"

# Keep this list in lockstep with the stager.  The test checks that the source
# bundle is the explicit runtime/provenance closure and nothing recursive.
FILES=(
  frontier_rl/__init__.py
  frontier_rl/interfaces.py
  frontier_rl/estimators.py
  frontier_rl/teacher.py
  frontier_rl/trainer.py
  frontier_rl/evaluation.py
  frontier_rl/adapters/__init__.py
  frontier_rl/adapters/barn_gazebo.py
  icra2027/__init__.py
  icra2027/freeze_pool_split.py
  icra2027/barn_campaign.py
  icra2027/analyze_campaign.py
  icra2027/merge_barn_campaign.py
  icra2027/select_barn_attempts.py
  icra2027/prereg_icra.md
  icra2027/barn_protocol.json
  icra2027/barn_split.json
  icra2027/barn_manifest.jsonl
  icra2027/assets/barn_diff_drive.sdf
  icra2027/barn_exact_step_plugin.cpp
  icra2027/gazebo_stepper.cpp
  icra2027/build_gazebo_stepper.sh
  icra2027/build_barn_manifest.py
  icra2027/verify_barn_smoke_package.py
  icra2027/receipts/barn_dataset_acquisition.json
  icra2027/receipts/barn_hopper_environment.json
  icra2027/receipts/barn_hopper_dataset_prepare.json
  icra2027/receipts/barn_hopper_training_smoke.json
  icra2027/receipts/barn_hopper_feasibility_projection.json
  icra2027/receipts/barn_hopper_directory_publish_probe.json
  icra2027/results/barn_backend_throughput_2026-08-14.json
  hopper/ros2-gazebo-classic.def
  hopper/stage_barn_campaign.sh
  hopper/submit_barn_campaign.sh
  hopper/finalize_barn_ledger.sh
  hopper/finalize_barn_campaign.sh
  hopper/sbatch/barn_seed_cpu.sbatch
  hopper/sbatch/barn_finalize_cpu.sbatch
  hopper/sbatch/barn_dataset_prepare.sbatch
  hopper/sbatch/barn_training_smoke.sbatch
)

cp -p -- "$STAGER" "$FIXTURE/hopper/stage_barn_campaign.sh"
for rel in "${FILES[@]}"; do
  mkdir -p "$FIXTURE/$(dirname "$rel")"
  case "$rel" in
    hopper/stage_barn_campaign.sh)
      # Keep the real stager copied above so the source closure binds itself.
      ;;
    icra2027/prereg_icra.md)
      printf '# Fixture preregistration\n\n**Status:** FROZEN\n' \
        > "$FIXTURE/$rel"
      ;;
    *.json)
      printf '{"fixture": "%s"}\n' "$rel" > "$FIXTURE/$rel"
      ;;
    *.jsonl)
      printf '{"fixture": "%s"}\n' "$rel" > "$FIXTURE/$rel"
      ;;
    *)
      printf 'fixture content for %s\n' "$rel" > "$FIXTURE/$rel"
      ;;
  esac
done
chmod 755 "$FIXTURE/icra2027/build_gazebo_stepper.sh"

git -C "$FIXTURE" init -q
git -C "$FIXTURE" config user.name 'BARN staging test'
git -C "$FIXTURE" config user.email 'barn-staging-test@example.invalid'
git -C "$FIXTURE" add -- .
git -C "$FIXTURE" commit -q -m 'frozen fixture'

# The fake ssh maps the only allowed remote scratch prefix into the temporary
# directory.  It rejects any unexpected host, so this test cannot reach Hopper.
cat > "$MOCK_BIN/ssh" <<'MOCK_SSH'
#!/bin/bash
set -euo pipefail
while (( $# )); do
  case "$1" in
    -o) shift 2 ;;
    --) shift ;;
    *) remote_host=$1; shift; break ;;
  esac
done
[[ "${remote_host:-}" == mock@hopper.invalid ]] || {
  echo "mock ssh rejected host: ${remote_host:-missing}" >&2
  exit 90
}
(( $# == 1 )) || {
  echo "mock ssh expected one remote command" >&2
  exit 91
}
remote_command=$1
printf '%s\t%q\n' "$remote_host" "$remote_command" >> "$SSH_LOG"
mapped_command=${remote_command//\/scratch\/lwang44/$FAKE_REMOTE_ROOT\/scratch\/lwang44}
/bin/bash -c "$mapped_command"
MOCK_SSH

# The fake rsync implements only the two transfer shapes used by the stager.
cat > "$MOCK_BIN/rsync" <<'MOCK_RSYNC'
#!/bin/bash
set -euo pipefail
printf '%q ' "$@" >> "$RSYNC_LOG"
printf '\n' >> "$RSYNC_LOG"
(( $# >= 2 )) || exit 92
source_path=${@: -2:1}
remote_spec=${@: -1}
[[ "$remote_spec" == mock@hopper.invalid:/scratch/lwang44/* ]] || {
  echo "mock rsync rejected destination: $remote_spec" >&2
  exit 93
}
remote_path=${remote_spec#*:}
local_path="$FAKE_REMOTE_ROOT$remote_path"
if [[ "$source_path" == */ ]]; then
  mkdir -p "$local_path"
  cp -a -- "${source_path%/}/." "$local_path/"
else
  mkdir -p "$(dirname "$local_path")"
  cp -a -- "$source_path" "$local_path"
fi
MOCK_RSYNC
chmod 755 "$MOCK_BIN/ssh" "$MOCK_BIN/rsync"

export FAKE_REMOTE_ROOT SSH_LOG RSYNC_LOG

run_stage() {
  local mode=$1
  local archive=${2:-$ARCHIVE}
  (
    cd "$FIXTURE"
    PATH="$MOCK_BIN:/usr/bin:/bin" \
    HOPPER_HOST=mock@hopper.invalid \
    HOPPER_SCRATCH=/scratch/lwang44 \
    BARN_DATASET_ZIP="$archive" \
      /bin/bash hopper/stage_barn_campaign.sh "$mode"
  )
}

key_from() {
  local output=$1
  local key=$2
  printf '%s\n' "$output" \
    | awk -F= -v wanted="$key" '$1 == wanted {print substr($0, length($1) + 2)}'
}

ssh_count() {
  wc -l < "$SSH_LOG" | tr -d ' '
}

# Unrelated dirt, including .codex, is permitted in evidence mode and must not
# perturb the source address or leak a path into the bundle.
mkdir -p "$FIXTURE/.codex"
printf 'must never be bundled\n' > "$FIXTURE/.codex/secret.txt"
printf 'unrelated dirty file\n' > "$FIXTURE/UNRELATED_NOTES.txt"

first_output=$(run_stage evidence)
source_bundle=$(key_from "$first_output" BARN_SOURCE_BUNDLE_DIR)
source_sha=$(key_from "$first_output" BARN_SOURCE_SHA256)
dataset_archive=$(key_from "$first_output" BARN_DATASET_ARCHIVE)
dataset_sha=$(key_from "$first_output" BARN_DATASET_ARCHIVE_SHA256)
[[ -n "$source_bundle" && -n "$source_sha" ]] \
  || fail "missing shell-readable source keys"
[[ "$dataset_sha" == "$EXPECTED_ARCHIVE_SHA256" ]] \
  || fail "wrong dataset SHA output"
[[ "$source_bundle" == "/scratch/lwang44/maxrl/bundles/barn_source/${source_sha:0:20}" ]] \
  || fail "source bundle is not addressed by manifest prefix"
[[ "$dataset_archive" == "/scratch/lwang44/maxrl/bundles/barn_dataset/${dataset_sha:0:20}/BARN_dataset.zip" ]] \
  || fail "dataset archive is not content-addressed"
[[ "$(key_from "$first_output" SOURCE_REUSED)" == false ]] \
  || fail "first source publish unexpectedly reused"
[[ "$(key_from "$first_output" DATASET_REUSED)" == false ]] \
  || fail "first dataset publish unexpectedly reused"

local_source_bundle="$FAKE_REMOTE_ROOT$source_bundle"
local_dataset_archive="$FAKE_REMOTE_ROOT$dataset_archive"
[[ -d "$local_source_bundle" && -f "$local_dataset_archive" ]] \
  || fail "mock remote artifacts are missing"
[[ "$(sha256sum "$local_source_bundle/SHA256SUMS" | awk '{print $1}')" == \
   "$source_sha" ]] || fail "source manifest output does not verify"
(
  cd "$local_source_bundle"
  sha256sum -c SHA256SUMS >/dev/null
)
[[ "$(sha256sum "$local_dataset_archive" | awk '{print $1}')" == \
   "$EXPECTED_ARCHIVE_SHA256" ]] || fail "remote dataset archive mismatch"
find "$local_source_bundle" -path '*/.codex*' -print -quit \
  | grep -q . && fail ".codex leaked into source bundle"
for rel in "${FILES[@]}"; do
  [[ -f "$local_source_bundle/$rel" ]] || fail "bundle omitted $rel"
done

python3 - "$local_source_bundle/SOURCE_STATE.json" "${#FILES[@]}" <<'PY'
import json
import pathlib
import sys

state = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert state["mode"] == "evidence"
assert state["worktree_dirty"] is False
assert state["relevant_paths_match_head"] is True
assert state["dirty_relevant_paths"] == []
assert len(state["files"]) == int(sys.argv[2])
assert all(row["git_status"] == "clean" for row in state["files"])
assert all(row["porcelain"] == "" for row in state["files"])
assert all(row["head_mode"] in {"100644", "100755"}
           for row in state["files"])
assert all(row["worktree_mode"] for row in state["files"])
assert all(len(row["worktree_sha256"]) == 64 for row in state["files"])
assert all(len(row["head_blob"]) == 40 for row in state["files"])
assert all(".codex" not in row["path"] for row in state["files"])
PY

# An identical stage reuses both immutable objects only after strict checking.
second_output=$(run_stage evidence)
[[ "$(key_from "$second_output" BARN_SOURCE_SHA256)" == "$source_sha" ]] \
  || fail "identical evidence stage changed source address"
[[ "$(key_from "$second_output" SOURCE_REUSED)" == true ]] \
  || fail "verified source object was not reused"
[[ "$(key_from "$second_output" DATASET_REUSED)" == true ]] \
  || fail "verified dataset object was not reused"

# Existing dataset corruption is a collision, never an overwrite.
cp -p -- "$local_dataset_archive" "$TMP/archive.backup"
printf 'corrupt archive\n' > "$local_dataset_archive"
if run_stage evidence > "$TMP/dataset-collision.out" 2> "$TMP/dataset-collision.err"; then
  fail "corrupt existing dataset was accepted"
fi
grep -q 'dataset id collision or corruption' "$TMP/dataset-collision.err" \
  || fail "dataset collision did not report its cause"
cp -p -- "$TMP/archive.backup" "$local_dataset_archive"

# Relevant dirt is rejected before any remote call in evidence mode, while an
# engineering bundle records the exact dirty path and per-file status.
cp -p -- "$FIXTURE/icra2027/barn_campaign.py" "$TMP/barn_campaign.clean"
printf 'dirty engineering edit\n' >> "$FIXTURE/icra2027/barn_campaign.py"
calls_before=$(ssh_count)
if run_stage evidence > "$TMP/relevant-dirty.out" 2> "$TMP/relevant-dirty.err"; then
  fail "evidence mode accepted a modified bundled path"
fi
[[ "$(ssh_count)" == "$calls_before" ]] \
  || fail "evidence rejection contacted the mock remote"
grep -q 'icra2027/barn_campaign.py' "$TMP/relevant-dirty.err" \
  || fail "evidence rejection did not name the dirty path"

engineering_output=$(run_stage engineering)
engineering_bundle=$(key_from "$engineering_output" BARN_SOURCE_BUNDLE_DIR)
engineering_sha=$(key_from "$engineering_output" BARN_SOURCE_SHA256)
[[ "$engineering_sha" != "$source_sha" ]] \
  || fail "dirty engineering bundle reused evidence identity"
python3 - "$FAKE_REMOTE_ROOT$engineering_bundle/SOURCE_STATE.json" <<'PY'
import json
import pathlib
import sys

state = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert state["mode"] == "engineering"
assert state["worktree_dirty"] is True
assert state["relevant_paths_match_head"] is False
assert state["dirty_relevant_paths"] == ["icra2027/barn_campaign.py"]
by_path = {row["path"]: row for row in state["files"]}
assert by_path["icra2027/barn_campaign.py"]["git_status"] == "dirty"
assert by_path["icra2027/barn_campaign.py"]["porcelain"] == " M"
PY

# Existing source corruption is likewise rejected rather than repaired.
printf 'corrupt source\n' > \
  "$FAKE_REMOTE_ROOT$engineering_bundle/icra2027/barn_campaign.py"
if run_stage engineering > "$TMP/source-collision.out" 2> "$TMP/source-collision.err"; then
  fail "corrupt existing source bundle was accepted"
fi
grep -q 'source bundle id collision or corruption' "$TMP/source-collision.err" \
  || fail "source collision did not report its cause"

# Exactly one literal FROZEN status line is mandatory.
cp -p -- "$FIXTURE/icra2027/prereg_icra.md" "$TMP/prereg.clean"
sed 's/\*\*Status:\*\* FROZEN/\*\*Status:\*\* FROZEN, test fixture./' \
  "$TMP/prereg.clean" > "$FIXTURE/icra2027/prereg_icra.md"
calls_before=$(ssh_count)
if run_stage evidence > "$TMP/punctuated.out" 2> "$TMP/punctuated.err"; then
  fail "evidence mode accepted a punctuated FROZEN status"
fi
[[ "$(ssh_count)" == "$calls_before" ]] \
  || fail "punctuated FROZEN rejection contacted the mock remote"
grep -q 'literal Status: FROZEN' "$TMP/punctuated.err" \
  || fail "punctuated FROZEN rejection did not report the freeze requirement"

sed 's/\*\*Status:\*\* FROZEN/\*\*Status:\*\* DRAFT/' \
  "$TMP/prereg.clean" > "$FIXTURE/icra2027/prereg_icra.md"
calls_before=$(ssh_count)
if run_stage evidence > "$TMP/draft.out" 2> "$TMP/draft.err"; then
  fail "evidence mode accepted a DRAFT preregistration"
fi
[[ "$(ssh_count)" == "$calls_before" ]] \
  || fail "DRAFT rejection contacted the mock remote"
grep -q 'literal Status: FROZEN' "$TMP/draft.err" \
  || fail "DRAFT rejection did not report the freeze requirement"
cp -p -- "$TMP/prereg.clean" "$FIXTURE/icra2027/prereg_icra.md"
cp -p -- "$TMP/barn_campaign.clean" "$FIXTURE/icra2027/barn_campaign.py"

# A bad local archive fails before ssh/rsync, and staging never invokes unzip.
printf 'not the official archive\n' > "$TMP/not-barn.zip"
calls_before=$(ssh_count)
if run_stage engineering "$TMP/not-barn.zip" \
     > "$TMP/bad-archive.out" 2> "$TMP/bad-archive.err"; then
  fail "bad local archive was accepted"
fi
[[ "$(ssh_count)" == "$calls_before" ]] \
  || fail "bad archive rejection contacted the mock remote"
grep -q 'archive SHA-256 mismatch' "$TMP/bad-archive.err" \
  || fail "bad archive rejection did not report the digest mismatch"
if grep -Eq '(^|[[:space:]])unzip([[:space:]]|$)' "$SSH_LOG" "$RSYNC_LOG"; then
  fail "login-node staging attempted to unzip the dataset"
fi
if grep -v '^mock@hopper.invalid' "$SSH_LOG" | grep -q .; then
  fail "an unexpected ssh host appeared in the mock log"
fi

# Hopper compute nodes do not provide host /usr/bin/time. Every BARN runtime
# job must use Bash's built-in timer and keep command stderr separate.
for runtime_job in \
  sbatch/barn_dataset_prepare.sbatch \
  sbatch/barn_training_smoke.sbatch \
  sbatch/barn_seed_cpu.sbatch; do
  if grep -Fq '/usr/bin/time' "$HERE/$runtime_job"; then
    fail "$runtime_job retained the unavailable host /usr/bin/time dependency"
  fi
  grep -Fq 'TIMEFORMAT=' "$HERE/$runtime_job" \
    || fail "$runtime_job lacks Bash built-in resource timing"
  grep -Fq 'runtime-stderr.txt' "$HERE/$runtime_job" \
    || fail "$runtime_job does not separate runtime stderr from timing"
done
dataset_job="$HERE/sbatch/barn_dataset_prepare.sbatch"
if grep -Fq 'ctypes.CDLL' "$dataset_job"; then
  fail "dataset preparation retained the unsupported directory rename syscall"
fi
grep -Fq 'mkdir_claim_plus_complete_last_hardlink' "$dataset_job" \
  || fail "dataset preparation lacks the completion-last publication receipt"
grep -Fq 'os.link(temporary, destination)' "$dataset_job" \
  || fail "dataset preparation lacks atomic exclusive COMPLETE publication"
seed_job="$HERE/sbatch/barn_seed_cpu.sbatch"
if grep -Fq 'ctypes.CDLL' "$seed_job"; then
  fail "evidence seed publication retained unsupported syscall publisher"
fi
grep -Fq 'os.link(complete, claim, follow_symlinks=False)' "$seed_job" \
  || fail "evidence seed publication lacks its exclusive hard-link claim"
grep -Fq 'os.rename(source, destination)' "$seed_job" \
  || fail "evidence seed publication lacks atomic same-parent rename"
for ros_job in \
  "$HERE/sbatch/barn_training_smoke.sbatch" \
  "$HERE/sbatch/barn_seed_cpu.sbatch"; do
  grep -Fq '${PYTHONPATH:+:$PYTHONPATH}' "$ros_job" \
    || fail "$(basename "$ros_job") overwrites the ROS Python path"
done
grep -Fq '"training_sim_steps": training_sim_steps' \
  "$HERE/sbatch/barn_training_smoke.sbatch" \
  || fail "training smoke omits simulator-step throughput needed for freeze"
grep -Fq '#SBATCH --time=1-12:00:00' \
  "$HERE/sbatch/barn_seed_cpu.sbatch" \
  || fail "evidence job does not request the frozen 36-hour scheduler limit"

echo "PASS: BARN staging is content-addressed, scoped, collision-safe, and network-free under mocks"
