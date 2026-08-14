#!/usr/bin/env bash
# Small, network-free transaction test for finalize_barn_campaign.sh.
set -euo pipefail
umask 077

readonly HERE="$(cd "$(dirname "$0")" && pwd)"
readonly SOURCE_FINALIZER="$HERE/finalize_barn_campaign.sh"
readonly SOURCE_SBATCH="$HERE/sbatch/barn_finalize_cpu.sbatch"
readonly TMP="$(mktemp -d /tmp/barn-campaign-finalize-test.XXXXXX)"
cleanup() {
  if [[ -n "${TMP:-}" && "$TMP" == /tmp/barn-campaign-finalize-test.* && -d "$TMP" ]]; then
    chmod -R u+w "$TMP" 2>/dev/null || true
    rm -rf -- "$TMP"
  fi
}
trap cleanup EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

readonly LOCAL_HOPPER="$TMP/local-hopper"
readonly FINALIZER="$LOCAL_HOPPER/finalize_barn_campaign.sh"
mkdir -p "$LOCAL_HOPPER/sbatch"
cp -p -- "$SOURCE_FINALIZER" "$FINALIZER"
cp -p -- "$SOURCE_SBATCH" "$LOCAL_HOPPER/sbatch/barn_finalize_cpu.sbatch"
bash -n "$FINALIZER" "$LOCAL_HOPPER/sbatch/barn_finalize_cpu.sbatch"

# Exercise the exact embedded remote directory publisher independently of the
# larger mocked Slurm transaction so collision and crash-window behavior are
# deterministic and do not require opening any package endpoint.
readonly SEALED_PUBLISHER="$TMP/sealed-publish.py"
cat > "$SEALED_PUBLISHER" <<'PY'
import os
from pathlib import Path
import stat
import sys


class Refusal(RuntimeError):
    pass


def refuse(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def require_canonical_directory(path: Path, *, label: str) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError as error:
        raise Refusal(f"{label} is missing") from error
    refuse(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
           f"{label} is non-directory or symbolic")
    refuse(path.resolve(strict=True) == path,
           f"{label} has a symbolic or non-canonical ancestor")


def require_regular(path: Path, *, label: str) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError as error:
        raise Refusal(f"{label} is missing") from error
    refuse(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
           f"{label} is non-regular or symbolic")
    refuse(path.resolve(strict=True) == path,
           f"{label} has a symbolic or non-canonical ancestor")
PY
awk '
  /^# BARN_SEALED_PUBLISH_PY_BEGIN$/ { capture = 1; next }
  /^# BARN_SEALED_PUBLISH_PY_END$/ { capture = 0; found = 1; next }
  capture { print }
  END { if (!found) exit 1 }
' "$FINALIZER" >> "$SEALED_PUBLISHER" \
  || fail "could not extract embedded sealed publisher"
cat >> "$SEALED_PUBLISHER" <<'PY'


rename_mode = os.environ.get("BARN_TEST_SEALED_RENAME_MODE")
if rename_mode:
    original_rename = os.rename

    def injected_rename(source, destination):
        if rename_mode == "fail_absent":
            raise OSError("synthetic handled rename failure")
        if rename_mode == "nonempty_collision":
            target = Path(destination)
            target.mkdir()
            (target / "original.txt").write_text("original\n")
            return original_rename(source, destination)
        raise RuntimeError("unknown synthetic rename mode")

    os.rename = injected_rename
atomic_publish_directory(Path(sys.argv[1]), Path(sys.argv[2]))
PY
python3 -m py_compile "$SEALED_PUBLISHER"
! grep -Fq 'ctypes.CDLL' "$SEALED_PUBLISHER" \
  || fail "remote sealed publisher retained unsupported rename syscall"
grep -Fq 'os.link(complete, claim, follow_symlinks=False)' "$SEALED_PUBLISHER" \
  || fail "remote sealed publisher lacks exclusive COMPLETE claim"
grep -Fq 'os.rename(source, destination)' "$SEALED_PUBLISHER" \
  || fail "remote sealed publisher lacks ordinary same-parent rename"

make_publish_stage() {
  local stage=$1 payload=$2
  mkdir -p -- "$stage/results"
  printf 'artifact_type\tbarn_four_cell_campaign_complete\n' \
    > "$stage/COMPLETE"
  printf '%s\n' "$payload" > "$stage/results/payload.txt"
}

run_sealed_publish() {
  python3 "$SEALED_PUBLISHER" "$1" "$2"
}

expect_sealed_publish_failure() {
  local source=$1 destination=$2 label=$3
  if run_sealed_publish "$source" "$destination" \
      > "$TMP/sealed-publish-failure.out" 2>&1; then
    fail "$label unexpectedly published"
  fi
}

assert_publish_claim() {
  local claim=$1 complete=$2
  [[ -f "$claim" && ! -L "$claim" ]] \
    || fail "sealed publication claim is not regular"
  [[ "$(stat -c '%d:%i' -- "$claim")" == \
     "$(stat -c '%d:%i' -- "$complete")" ]] \
    || fail "sealed publication claim is not hard-linked to COMPLETE"
}

# Happy path and the post-rename crash window: the retained claim identifies
# the canonical COMPLETE inode, and an idempotent retry cannot replace it.
publish_root="$TMP/publish-happy"
mkdir -p -- "$publish_root"
publish_source="$publish_root/.campaign-stage-winner"
publish_destination="$publish_root/campaign-happy"
publish_claim="$publish_root/.campaign-happy.publish-claim"
make_publish_stage "$publish_source" winner
run_sealed_publish "$publish_source" "$publish_destination"
[[ ! -e "$publish_source" && -d "$publish_destination" ]] \
  || fail "sealed happy-path rename was not atomic"
[[ "$(< "$publish_destination/results/payload.txt")" == winner ]] \
  || fail "sealed happy-path payload changed"
assert_publish_claim "$publish_claim" "$publish_destination/COMPLETE"
publish_retry="$publish_root/.campaign-stage-retry"
make_publish_stage "$publish_retry" retry-must-not-win
expect_sealed_publish_failure "$publish_retry" "$publish_destination" \
  "post-rename crash retry"
[[ -d "$publish_retry" \
   && "$(< "$publish_destination/results/payload.txt")" == winner ]] \
  || fail "post-rename retry overwrote the sealed destination"

# Existing destination collisions of every filesystem kind fail before the
# exclusive claim.  In particular, a nonempty canonical-looking package is
# preserved byte-for-byte.
for kind in nonempty_dir empty_dir regular_file symlink; do
  publish_root="$TMP/publish-existing-$kind"
  mkdir -p -- "$publish_root"
  publish_source="$publish_root/.campaign-stage-contender"
  publish_destination="$publish_root/campaign-collision"
  publish_claim="$publish_root/.campaign-collision.publish-claim"
  make_publish_stage "$publish_source" contender
  case "$kind" in
    nonempty_dir)
      mkdir -- "$publish_destination"
      printf 'original\n' > "$publish_destination/original.txt"
      ;;
    empty_dir) mkdir -- "$publish_destination" ;;
    regular_file) printf 'original\n' > "$publish_destination" ;;
    symlink) ln -s -- nowhere "$publish_destination" ;;
  esac
  expect_sealed_publish_failure "$publish_source" "$publish_destination" \
    "existing $kind"
  [[ -d "$publish_source" && ! -e "$publish_claim" \
     && ! -L "$publish_claim" ]] \
    || fail "existing $kind did not fail before sealed claim"
  if [[ "$kind" == nonempty_dir ]]; then
    [[ "$(< "$publish_destination/original.txt")" == original ]] \
      || fail "nonempty sealed destination was overwritten"
  fi
done

# A retained pre-rename claim models a crash after the exclusive hard link.
# Retry fails closed, leaves the sealed stage hidden, and exposes no endpoint.
publish_root="$TMP/publish-crash-before-rename"
mkdir -p -- "$publish_root"
publish_source="$publish_root/.campaign-stage-crashed"
publish_destination="$publish_root/campaign-crashed"
publish_claim="$publish_root/.campaign-crashed.publish-claim"
make_publish_stage "$publish_source" hidden-after-crash
ln -- "$publish_source/COMPLETE" "$publish_claim"
expect_sealed_publish_failure "$publish_source" "$publish_destination" \
  "stale pre-rename claim"
[[ -d "$publish_source" && ! -e "$publish_destination" \
   && ! -L "$publish_destination" ]] \
  || fail "pre-rename crash exposed a sealed endpoint"
assert_publish_claim "$publish_claim" "$publish_source/COMPLETE"

# A handled rename failure with the destination still absent removes only the
# inode-identical claim it just created.  The stage stays hidden here; the real
# finalizer's surrounding finally block then removes it.
publish_root="$TMP/publish-handled-rename-failure"
mkdir -p -- "$publish_root"
publish_source="$publish_root/.campaign-stage-handled-failure"
publish_destination="$publish_root/campaign-handled-failure"
publish_claim="$publish_root/.campaign-handled-failure.publish-claim"
make_publish_stage "$publish_source" handled-failure
if BARN_TEST_SEALED_RENAME_MODE=fail_absent \
    run_sealed_publish "$publish_source" "$publish_destination" \
    > "$TMP/handled-rename-failure.out" 2>&1; then
  fail "synthetic handled rename failure unexpectedly published"
fi
[[ -d "$publish_source" && ! -e "$publish_destination" \
   && ! -L "$publish_destination" && ! -e "$publish_claim" \
   && ! -L "$publish_claim" ]] \
  || fail "handled rename failure stranded its publication claim"

# If a nonempty destination appears after the claim but before rename, the
# ordinary rename fails, preserves the collision, and retains the claim as an
# ambiguity fence instead of risking overwrite or automatic reclamation.
publish_root="$TMP/publish-late-collision"
mkdir -p -- "$publish_root"
publish_source="$publish_root/.campaign-stage-late-collision"
publish_destination="$publish_root/campaign-late-collision"
publish_claim="$publish_root/.campaign-late-collision.publish-claim"
make_publish_stage "$publish_source" late-contender
if BARN_TEST_SEALED_RENAME_MODE=nonempty_collision \
    run_sealed_publish "$publish_source" "$publish_destination" \
    > "$TMP/late-collision.out" 2>&1; then
  fail "late nonempty collision unexpectedly published"
fi
[[ -d "$publish_source" && -d "$publish_destination" \
   && "$(< "$publish_destination/original.txt")" == original ]] \
  || fail "late nonempty collision did not preserve the destination"
assert_publish_claim "$publish_claim" "$publish_source/COMPLETE"

# Concurrent compliant publishers race only on the exclusive hard-link
# claim. Exactly one complete directory wins and the loser remains hidden.
for iteration in $(seq 1 16); do
  publish_root="$TMP/publish-concurrent-$iteration"
  mkdir -p -- "$publish_root"
  publish_source_a="$publish_root/.campaign-stage-a"
  publish_source_b="$publish_root/.campaign-stage-b"
  publish_destination="$publish_root/campaign-concurrent"
  publish_claim="$publish_root/.campaign-concurrent.publish-claim"
  make_publish_stage "$publish_source_a" A
  make_publish_stage "$publish_source_b" B

  set +e
  run_sealed_publish "$publish_source_a" "$publish_destination" \
    > "$publish_root/a.out" 2>&1 & publish_pid_a=$!
  run_sealed_publish "$publish_source_b" "$publish_destination" \
    > "$publish_root/b.out" 2>&1 & publish_pid_b=$!
  wait "$publish_pid_a"; publish_status_a=$?
  wait "$publish_pid_b"; publish_status_b=$?
  set -e

  if (( (publish_status_a == 0) + (publish_status_b == 0) != 1 )); then
    fail "sealed concurrency iteration $iteration lacked exactly one winner"
  fi
  [[ -d "$publish_destination" \
     && -f "$publish_destination/COMPLETE" \
     && -f "$publish_destination/results/payload.txt" ]] \
    || fail "sealed concurrent winner exposed an incomplete package"
  publish_winner=$(< "$publish_destination/results/payload.txt")
  case "$publish_winner" in
    A) [[ ! -e "$publish_source_a" && -d "$publish_source_b" ]] \
         || fail "sealed concurrent A winner state is inconsistent" ;;
    B) [[ ! -e "$publish_source_b" && -d "$publish_source_a" ]] \
         || fail "sealed concurrent B winner state is inconsistent" ;;
    *) fail "sealed concurrent winner payload is invalid" ;;
  esac
  assert_publish_claim "$publish_claim" "$publish_destination/COMPLETE"
done

readonly MOCK_BIN="$TMP/mock-bin"
readonly FAKE_REMOTE_ROOT="$TMP/remote"
readonly LOCAL_PACKAGES="$TMP/local-packages"
readonly TOOL_TRACE="$TMP/tool-trace.tsv"
readonly MOCK_JOB_LOG="$TMP/job-424242.out"
readonly SLURM_TRACE="$TMP/slurm-trace.tsv"
mkdir -p "$MOCK_BIN" "$FAKE_REMOTE_ROOT" "$LOCAL_PACKAGES"
: > "$TOOL_TRACE"
: > "$SLURM_TRACE"

cat > "$MOCK_BIN/ssh" <<'MOCK_SSH'
#!/usr/bin/env bash
set -euo pipefail
while [[ ${1:-} == -o ]]; do shift 2; done
[[ ${1:-} == mock@hopper.invalid ]] || exit 90
shift
mapped=()
for argument in "$@"; do
  case "$argument" in
    /scratch/mock)
      mapped+=("$FAKE_REMOTE_ROOT") ;;
    /scratch/mock/*)
      mapped+=("$FAKE_REMOTE_ROOT/${argument#/scratch/mock/}") ;;
    /opt/sw/other/apps/apptainer/1.4.1/bin/apptainer)
      mapped+=("$MOCK_BIN/apptainer") ;;
    *) mapped+=("$argument") ;;
  esac
done
set +e
output=$("${mapped[@]}")
status=$?
set -e
(( status == 0 )) || exit "$status"
output=${output//$FAKE_REMOTE_ROOT/\/scratch\/mock}
printf '%s\n' "$output"
MOCK_SSH

cat > "$MOCK_BIN/rsync" <<'MOCK_RSYNC'
#!/usr/bin/env bash
set -euo pipefail
arguments=("$@")
count=${#arguments[@]}
source_spec=${arguments[count-2]}
destination=${arguments[count-1]}
[[ "$source_spec" == mock@hopper.invalid:/scratch/mock/*/ ]] || exit 91
source_path=${source_spec#*:}
source_path="$FAKE_REMOTE_ROOT/${source_path#/scratch/mock/}"
mkdir -p "$destination"
cp -a -- "${source_path%/}/." "$destination/"
MOCK_RSYNC

cat > "$MOCK_BIN/apptainer" <<'MOCK_APPTAINER'
#!/usr/bin/env bash
set -euo pipefail
while (( $# )) && [[ "$1" != env ]]; do shift; done
[[ ${1:-} == env ]] || exit 92
shift
printf '%q ' "$@" >> "$TOOL_TRACE"
printf '\n' >> "$TOOL_TRACE"
exec /usr/bin/env "$@"
MOCK_APPTAINER
cat > "$MOCK_BIN/sbatch" <<'MOCK_SBATCH'
#!/usr/bin/env bash
set -euo pipefail
export_spec=""
script=""
for argument in "$@"; do
  case "$argument" in
    --parsable) ;;
    --export=*) export_spec=${argument#--export=} ;;
    --gres=*|--gpus=*|--gpus-per-node=*) exit 93 ;;
    --*) exit 94 ;;
    *) script=$argument ;;
  esac
done
[[ -n "$export_spec" && -f "$script" ]] || exit 95
IFS=, read -r -a bindings <<< "$export_spec"
for binding in "${bindings[@]}"; do
  [[ "$binding" == ALL ]] && continue
  name=${binding%%=*}
  value=${binding#*=}
  export "$name=$value"
done
[[ "$(sha256sum "$script" | awk '{print $1}')" == "$BARN_FINALIZE_SBATCH_SHA256" ]]
[[ "$(sha256sum "$BARN_SOURCE_BUNDLE_DIR/hopper/finalize_barn_campaign.sh" | awk '{print $1}')" == "$BARN_FINALIZE_WRAPPER_SHA256" ]]
printf 'submit\t%s\n' "$script" >> "$SLURM_TRACE"
mkdir -p "$BARN_SCRATCH_ROOT/maxrl/barn/logs"
bash "$BARN_SOURCE_BUNDLE_DIR/hopper/finalize_barn_campaign.sh" --remote \
  "$BARN_CAMPAIGN_ID" "$BARN_SOURCE_BUNDLE_DIR" "$BARN_SOURCE_SHA256" \
  "$BARN_FINALIZED_LEDGER_SHA256" "$BARN_FINALIZE_WRAPPER_SHA256" \
  "$BARN_SCRATCH_ROOT" "$MOCK_BIN/apptainer" \
  > "$BARN_SCRATCH_ROOT/maxrl/barn/logs/barn-finalize-safe-log_424242.out"
printf '424242\n'
MOCK_SBATCH

cat > "$MOCK_BIN/sacct" <<'MOCK_SACCT'
#!/usr/bin/env bash
set -euo pipefail
job=""
while (( $# )); do
  if [[ "$1" == -j ]]; then job=$2; shift 2; else shift; fi
done
[[ "$job" == 424242 ]] || exit 96
printf '424242|COMPLETED|0:0|\n'
MOCK_SACCT
chmod 0755 "$MOCK_BIN/ssh" "$MOCK_BIN/rsync" "$MOCK_BIN/apptainer" \
  "$MOCK_BIN/sbatch" "$MOCK_BIN/sacct"
export FAKE_REMOTE_ROOT MOCK_BIN TOOL_TRACE MOCK_JOB_LOG SLURM_TRACE

SOURCE_TEMP="$TMP/source-build"
mkdir -p "$SOURCE_TEMP/icra2027" "$SOURCE_TEMP/hopper/sbatch"
cp -p -- "$FINALIZER" "$SOURCE_TEMP/hopper/finalize_barn_campaign.sh"
cp -p -- "$LOCAL_HOPPER/sbatch/barn_finalize_cpu.sbatch" \
  "$SOURCE_TEMP/hopper/sbatch/barn_finalize_cpu.sbatch"

cat > "$SOURCE_TEMP/icra2027/select_barn_attempts.py" <<'PY'
import argparse
import hashlib
import json
from pathlib import Path

fields = ("manifest_sha256", "split_sha256", "prereg_sha256",
          "analyzer_sha256", "protocol_sha256", "container_sha256",
          "source_sha256")
parser = argparse.ArgumentParser()
parser.add_argument("artifacts", nargs="+")
parser.add_argument("--ledger", required=True)
parser.add_argument("--campaign-id", required=True)
parser.add_argument("--campaign-cell", required=True)
parser.add_argument("--protocol", required=True)
parser.add_argument("--expected-seeds", required=True)
parser.add_argument("--output", required=True)
for field in fields:
    parser.add_argument("--expected-" + field.replace("_", "-"), required=True)
args = parser.parse_args()
ledger_path = Path(args.ledger)
ledger = json.loads(ledger_path.read_text())
hashes = {field: getattr(args, "expected_" + field) for field in fields}
rows = [row for row in ledger["submissions"]
        if row["campaign_id"] == args.campaign_id
        and row["campaign_cell"] == args.campaign_cell]
complete = [row for row in rows if row["artifact_complete"]]
if set(map(Path, args.artifacts)) != {Path(row["artifact_path"]) for row in complete}:
    raise SystemExit("selector input closure mismatch")
selected = []
for seed in range(1, 6):
    candidates = sorted(
        (row for row in complete if row["seed"] == seed),
        key=lambda row: (row["submitted_utc"], row["attempt_id"]))
    if not candidates:
        raise SystemExit("missing complete seed")
    row = candidates[0]
    artifact = json.loads(Path(row["artifact_path"]).read_text())
    execution = artifact["execution"]
    selected.append({
        "seed": seed,
        "attempt_id": row["attempt_id"],
        "submitted_utc": execution["submitted_utc"],
        "slurm_array_job_id": execution["slurm_array_job_id"],
        "slurm_array_task_id": execution["slurm_array_task_id"],
        "slurm_job_id": execution["slurm_job_id"],
        "artifact_path": row["artifact_path"],
        "artifact_sha256": row["artifact_sha256"],
    })
receipt = {
    "schema_version": 1,
    "selection_rule": "earliest_submitted_complete_hash_valid_attempt_per_seed",
    "outcome_blind": True,
    "campaign_id": args.campaign_id,
    "campaign_cell": args.campaign_cell,
    "expected_seed_list": [1, 2, 3, 4, 5],
    "expected_hashes": hashes,
    "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
    "selected": selected,
    "excluded": [],
}
Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n")
print(f"wrote {args.output}: selected=5 excluded=0")
PY

cat > "$SOURCE_TEMP/icra2027/merge_barn_campaign.py" <<'PY'
import argparse
import hashlib
import json
from pathlib import Path

fields = ("manifest_sha256", "split_sha256", "prereg_sha256",
          "analyzer_sha256", "protocol_sha256", "container_sha256",
          "source_sha256")
parser = argparse.ArgumentParser()
parser.add_argument("artifacts", nargs="+")
parser.add_argument("--selection-receipt", required=True)
parser.add_argument("--protocol", required=True)
parser.add_argument("--campaign-cell", required=True)
parser.add_argument("--expected-seeds", required=True)
parser.add_argument("--output", required=True)
for field in fields:
    parser.add_argument("--expected-" + field.replace("_", "-"), required=True)
args = parser.parse_args()
receipt_path = Path(args.selection_receipt)
receipt = json.loads(receipt_path.read_text())
selected_paths = [row["artifact_path"] for row in receipt["selected"]]
if args.artifacts != selected_paths or len(args.artifacts) != 5:
    raise SystemExit("merge inputs differ from selector receipt")
for path, row in zip(args.artifacts, receipt["selected"]):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != row["artifact_sha256"]:
        raise SystemExit("selected artifact hash mismatch")
selection = {
    "schema_version": 1,
    "selection_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    "ledger_sha256": receipt["ledger_sha256"],
    "rule": receipt["selection_rule"],
    "campaign_id": receipt["campaign_id"],
    "campaign_cell": args.campaign_cell,
    "selected": [{
        "seed": row["seed"], "attempt_id": row["attempt_id"],
        "artifact_sha256": row["artifact_sha256"],
        "execution": {
            key: row[key] for key in (
                "attempt_id", "submitted_utc", "slurm_job_id",
                "slurm_array_job_id", "slurm_array_task_id")
        } | {"campaign_id": receipt["campaign_id"]},
    } for row in receipt["selected"]],
}
merged = {
    "schema_version": 1,
    "domain": "barn_gazebo_cpu_navigation",
    "config": {"campaign_cell": args.campaign_cell},
    "merge": {"schema_version": 1, "outcome_blind": True,
              "selection": selection},
    "results": {},
}
Path(args.output).write_text(json.dumps(merged, indent=2) + "\n")
print(f"wrote {args.output}: cell={args.campaign_cell} seeds=5")
PY

cat > "$SOURCE_TEMP/icra2027/analyze_campaign.py" <<'PY'
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("artifact")
parser.add_argument("--ablation-artifact", action="append", default=[])
parser.add_argument("--output", required=True)
args = parser.parse_args()
if "campaign-analysis-fail" in args.output and args.ablation_artifact:
    raise SystemExit("synthetic analyzer failure")
here = Path(__file__).resolve().parent
analyzer_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
protocol_sha = hashlib.sha256((here / "barn_protocol.json").read_bytes()).hexdigest()
if args.ablation_artifact:
    paths = [Path(args.artifact), *(Path(path) for path in args.ablation_artifact)]
    by_n = {}
    for path in paths:
        cell = json.loads(path.read_text())["config"]["campaign_cell"]
        n = 8 if cell == "primary" else int(cell.removeprefix("ablation_n"))
        by_n[n] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "analysis_schema_version": 2,
        "analysis_kind": "barn_n_ablation",
        "analyzer_sha256": analyzer_sha,
        "protocol_sha256": protocol_sha,
        "cells": [{"n_rollouts": n, "input_artifact": {"sha256": by_n[n]}}
                  for n in (2, 4, 8, 16)],
    }
else:
    artifact = Path(args.artifact)
    report = {
        "analysis_schema_version": 2,
        "input_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "analyzer_sha256": analyzer_sha,
        "protocol_sha256": protocol_sha,
    }
Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
print(f"wrote {args.output}: analysis complete")
PY

printf '{"status":"FROZEN","fixture":true}\n' \
  > "$SOURCE_TEMP/icra2027/barn_protocol.json"
python3 - "$SOURCE_TEMP" <<'PY'
import hashlib
import json
from pathlib import Path
import stat
import sys
root = Path(sys.argv[1])
files = []
for path in sorted(candidate for candidate in root.rglob("*")
                   if candidate.is_file()):
    files.append({
        "path": path.relative_to(root).as_posix(),
        "worktree_mode": format(stat.S_IMODE(path.stat().st_mode), "o"),
        "worktree_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
state = {
    "mode": "evidence", "worktree_dirty": False,
    "relevant_paths_match_head": True, "files": files,
}
(root / "SOURCE_STATE.json").write_text(json.dumps(state) + "\n")
PY
chmod 0755 "$SOURCE_TEMP/hopper/finalize_barn_campaign.sh"
(
  cd "$SOURCE_TEMP"
  find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)
SOURCE_SHA=$(sha256sum "$SOURCE_TEMP/SHA256SUMS" | awk '{print $1}')
SOURCE_DIR="$FAKE_REMOTE_ROOT/maxrl/bundles/barn_source/${SOURCE_SHA:0:20}"
mkdir -p "$(dirname "$SOURCE_DIR")"
mv -- "$SOURCE_TEMP" "$SOURCE_DIR"
readonly SOURCE_SHA SOURCE_DIR
readonly LOGICAL_SOURCE="/scratch/mock/maxrl/bundles/barn_source/${SOURCE_SHA:0:20}"

printf 'synthetic pinned CPU container\n' > "$FAKE_REMOTE_ROOT/ros2-gazebo-classic.sif"
CONTAINER_SHA=$(sha256sum "$FAKE_REMOTE_ROOT/ros2-gazebo-classic.sif" | awk '{print $1}')
ANALYZER_SHA=$(sha256sum "$SOURCE_DIR/icra2027/analyze_campaign.py" | awk '{print $1}')
PROTOCOL_SHA=$(sha256sum "$SOURCE_DIR/icra2027/barn_protocol.json" | awk '{print $1}')
readonly CONTAINER_SHA ANALYZER_SHA PROTOCOL_SHA

make_campaign() {
  local campaign=$1 mode=${2:-complete}
  python3 - "$FAKE_REMOTE_ROOT" "$campaign" "$mode" "$SOURCE_SHA" \
    "$CONTAINER_SHA" "$ANALYZER_SHA" "$PROTOCOL_SHA" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

scratch = Path(sys.argv[1])
campaign, mode, source_sha, container_sha, analyzer_sha, protocol_sha = sys.argv[2:]
root = scratch / "maxrl" / "barn" / "campaigns" / campaign
cells = ["primary", "ablation_n2", "ablation_n4", "ablation_n16"]
if mode == "subset":
    cells = ["primary"]
hashes = {
    "manifest_sha256": "1" * 64,
    "split_sha256": "2" * 64,
    "prereg_sha256": "3" * 64,
    "analyzer_sha256": analyzer_sha,
    "protocol_sha256": protocol_sha,
    "container_sha256": container_sha,
    "source_sha256": source_sha,
}
rows = []
for cell_index, cell in enumerate(cells):
    for seed in range(1, 6):
        attempt = "attempt-001"
        artifact = (root / "cells" / cell / "attempts" / attempt
                    / f"seed-{seed}" / "results" / f"seed-{seed}.json")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "execution": {
                "campaign_id": campaign,
                "attempt_id": attempt,
                "submitted_utc": "2026-08-14T01:02:03+00:00",
                "slurm_job_id": str(10000 + cell_index * 10 + seed),
                "slurm_array_job_id": str(9900 + cell_index * 10),
                "slurm_array_task_id": seed,
            },
            "config": {"campaign_cell": cell, "campaign_seed": seed},
        }
        payload = json.dumps(document, sort_keys=True) + "\n"
        artifact.write_text(payload)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        rows.append({
            "campaign_id": campaign,
            "campaign_cell": cell,
            "attempt_id": attempt,
            "seed": seed,
            "submitted_utc": "2026-08-14T01:02:03Z",
            "slurm_job_id": str(10000 + cell_index * 10 + seed),
            "slurm_array_job_id": str(9900 + cell_index * 10),
            "slurm_array_task_id": seed,
            "artifact_path": str(artifact),
            "artifact_complete": True,
            "artifact_sha256": digest,
            "expected_hashes": hashes,
        })
ledger = {"schema_version": 1, "submissions": rows}
payload = (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode()
digest = hashlib.sha256(payload).hexdigest()
directory = root / "finalized_ledgers"
directory.mkdir(parents=True, exist_ok=True)
(directory / f"SUBMISSION_LEDGER.finalized-{digest}.json").write_bytes(payload)
print(digest)
PY
}

run_finalize_with() {
  local script=$1 campaign=$2 ledger_sha=$3
  PATH="$MOCK_BIN:/usr/bin:/bin" \
  HOPPER_HOST=mock@hopper.invalid HOPPER_SCRATCH=/scratch/mock \
  FINALIZE_BARN_POLL_INTERVAL=1 FINALIZE_BARN_WAIT_SECONDS=2 \
  FINALIZE_BARN_LOCAL_PACKAGE_DIR="$LOCAL_PACKAGES" \
    /bin/bash "$script" "$campaign" "$LOGICAL_SOURCE" \
      "$SOURCE_SHA" "$ledger_sha"
}
run_finalize() {
  run_finalize_with "$FINALIZER" "$@"
}

ledger_sha=$(make_campaign campaign-complete)

# A post-stage edit to either executing local file must fail before sbatch.
readonly TAMPERED_HOPPER="$TMP/tampered-hopper"
mkdir -p "$TAMPERED_HOPPER/sbatch"
cp -p -- "$FINALIZER" "$TAMPERED_HOPPER/finalize_barn_campaign.sh"
cp -p -- "$LOCAL_HOPPER/sbatch/barn_finalize_cpu.sbatch" \
  "$TAMPERED_HOPPER/sbatch/barn_finalize_cpu.sbatch"
printf '\n# post-stage wrapper tamper\n' \
  >> "$TAMPERED_HOPPER/finalize_barn_campaign.sh"
submits_before=$(wc -l < "$SLURM_TRACE")
if run_finalize_with "$TAMPERED_HOPPER/finalize_barn_campaign.sh" \
    campaign-complete "$ledger_sha" > "$TMP/wrapper-tamper.out" \
    2> "$TMP/wrapper-tamper.err"; then
  fail "post-stage campaign-wrapper edit unexpectedly submitted"
fi
grep -Fq 'source-bound CPU finalizer preflight/submission failed closed' \
  "$TMP/wrapper-tamper.err" || fail "wrapper tamper lacked fail-closed reason"
[[ $(wc -l < "$SLURM_TRACE") -eq "$submits_before" ]] \
  || fail "wrapper tamper reached sbatch"

cp -p -- "$FINALIZER" "$TAMPERED_HOPPER/finalize_barn_campaign.sh"
printf '\n# post-stage sbatch tamper\n' \
  >> "$TAMPERED_HOPPER/sbatch/barn_finalize_cpu.sbatch"
if run_finalize_with "$TAMPERED_HOPPER/finalize_barn_campaign.sh" \
    campaign-complete "$ledger_sha" > "$TMP/sbatch-tamper.out" \
    2> "$TMP/sbatch-tamper.err"; then
  fail "post-stage finalizer-sbatch edit unexpectedly submitted"
fi
[[ $(wc -l < "$SLURM_TRACE") -eq "$submits_before" ]] \
  || fail "sbatch tamper reached sbatch"

if ! run_finalize campaign-complete "$ledger_sha" \
    > "$TMP/complete.out" 2> "$TMP/complete.err"; then
  sed -n '1,120p' "$TMP/complete.err" >&2
  fail "complete campaign transaction failed"
fi
grep -Fq $'BARN_CAMPAIGN_SEALED\tcampaign=campaign-complete\tcells=4\tseeds_per_cell=5\tfiles=11' \
  "$TMP/complete.out" || fail "campaign-level completion metadata missing"
! grep -Eiq 'mean_success|auc|episode_return|success_rate|directional_bar' \
  "$TMP/complete.out" "$TMP/complete.err" \
  || fail "result endpoint label leaked to wrapper output"
package=$(find "$LOCAL_PACKAGES" -maxdepth 1 -type d \
  -name 'campaign-complete.sealed-*' -print)
[[ -n "$package" && $(printf '%s\n' "$package" | wc -l) -eq 1 ]] \
  || fail "expected exactly one fetched sealed campaign package"
[[ -f "$package/COMPLETE" && -f "$package/SHA256SUMS" \
   && -f "$package/PACKAGE_METADATA.json" ]] || fail "package controls missing"
for cell in primary ablation_n2 ablation_n4 ablation_n16; do
  [[ -f "$package/selection/$cell.json" && -f "$package/merged/$cell.json" ]] \
    || fail "package omitted selector/merge for $cell"
done
[[ -f "$package/reports/primary_gate.json" \
   && -f "$package/reports/n_ablation.json" ]] || fail "analysis reports missing"
(
  cd "$package"
  sha256sum -c --strict --quiet SHA256SUMS
)
[[ $(grep -c 'select_barn_attempts.py' "$TOOL_TRACE") -eq 4 ]] \
  || fail "selector did not run exactly four times"
[[ $(grep -c 'merge_barn_campaign.py' "$TOOL_TRACE") -eq 4 ]] \
  || fail "merger did not run exactly four times"
[[ $(grep -c 'analyze_campaign.py' "$TOOL_TRACE") -eq 2 ]] \
  || fail "analyzer did not run exactly twice"
[[ $(grep -c -- '--selection-receipt' "$TOOL_TRACE") -eq 4 ]] \
  || fail "merger calls were not selection-receipt bound"

package_sha=$(sha256sum "$package/SHA256SUMS" | awk '{print $1}')
remote_package="$FAKE_REMOTE_ROOT/maxrl/barn/campaigns/campaign-complete/sealed_campaigns/campaign-$package_sha"
remote_claim="$(dirname "$remote_package")/.$(basename "$remote_package").publish-claim"
[[ -d "$remote_package" ]] || fail "remote sealed campaign package is missing"
assert_publish_claim "$remote_claim" "$remote_package/COMPLETE"
if run_finalize campaign-complete "$ledger_sha" > "$TMP/repeat.out" 2> "$TMP/repeat.err"; then
  fail "repeat fetch overwrote a sealed local campaign"
fi
grep -Fq 'refusing to overwrite prior sealed campaign' "$TMP/repeat.err" \
  || fail "repeat fetch lacked no-clobber refusal"
[[ "$(sha256sum "$package/SHA256SUMS" | awk '{print $1}')" == "$package_sha" ]] \
  || fail "repeat invocation changed the prior package"
assert_publish_claim "$remote_claim" "$remote_package/COMPLETE"

subset_sha=$(make_campaign campaign-subset subset)
if run_finalize campaign-subset "$subset_sha" > "$TMP/subset.out" 2> "$TMP/subset.err"; then
  fail "subset campaign unexpectedly sealed"
fi
if find "$FAKE_REMOTE_ROOT/maxrl/barn/campaigns/campaign-subset/sealed_campaigns" \
    -mindepth 1 -maxdepth 1 -type d -name 'campaign-*' -print -quit \
    2>/dev/null | grep -q .; then
  fail "subset campaign published a package"
fi

failure_sha=$(make_campaign campaign-analysis-fail)
if run_finalize campaign-analysis-fail "$failure_sha" \
    > "$TMP/analysis-fail.out" 2> "$TMP/analysis-fail.err"; then
  fail "campaign with failed second analysis unexpectedly sealed"
fi
if find "$FAKE_REMOTE_ROOT/maxrl/barn/campaigns/campaign-analysis-fail/sealed_campaigns" \
    -mindepth 1 -maxdepth 1 -type d -name 'campaign-*' -print -quit \
    2>/dev/null | grep -q .; then
  fail "failed all-cell analysis transaction published a partial package"
fi

if command -v shellcheck >/dev/null 2>&1; then
  shellcheck -x "$FINALIZER" "$0"
fi
printf 'BARN_CAMPAIGN_FINALIZER_LOCAL_CHECK_PASS\n'
