#!/usr/bin/env bash
# Network-free contract test for finalize_barn_ledger.sh. ssh, rsync, sacct,
# and the remote scratch tree are all small local mocks.
set -euo pipefail
umask 077

readonly HERE="$(cd "$(dirname "$0")" && pwd)"
readonly FINALIZER="$HERE/finalize_barn_ledger.sh"
readonly TMP="$(mktemp -d /tmp/barn-ledger-finalize-test.XXXXXX)"
cleanup() {
  if [[ -n "${TMP:-}" && "$TMP" == /tmp/barn-ledger-finalize-test.* && -d "$TMP" ]]; then
    rm -rf -- "$TMP"
  fi
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -f "$FINALIZER" ]] || fail "missing finalizer"
bash -n "$FINALIZER"

readonly MOCK_BIN="$TMP/mock-bin"
readonly FAKE_REMOTE_ROOT="$TMP/remote-scratch"
readonly LOCAL_LEDGERS="$TMP/local-ledgers"
mkdir -p -- "$MOCK_BIN" "$FAKE_REMOTE_ROOT" "$LOCAL_LEDGERS"

cat > "$MOCK_BIN/ssh" <<'MOCK_SSH'
#!/usr/bin/env bash
set -euo pipefail
while [[ ${1:-} == -o ]]; do
  shift 2
done
[[ ${1:-} == mock@hopper.invalid ]] || exit 90
shift
[[ ${1:-} == bash && ${2:-} == -s && ${3:-} == -- ]] || exit 91
shift 3
mapped=()
for argument in "$@"; do
  if [[ "$argument" == /scratch/mock/* ]]; then
    mapped+=("$FAKE_REMOTE_ROOT/${argument#/scratch/mock/}")
  else
    mapped+=("$argument")
  fi
done
set +e
output=$(/bin/bash -s -- "${mapped[@]}")
status=$?
set -e
if (( status != 0 )); then
  exit "$status"
fi
output=${output//$FAKE_REMOTE_ROOT/\/scratch\/mock}
printf '%s\n' "$output"
MOCK_SSH

cat > "$MOCK_BIN/rsync" <<'MOCK_RSYNC'
#!/usr/bin/env bash
set -euo pipefail
arguments=("$@")
count=${#arguments[@]}
(( count >= 2 )) || exit 92
source_spec=${arguments[count-2]}
destination=${arguments[count-1]}
[[ "$source_spec" == mock@hopper.invalid:/scratch/mock/* ]] || exit 93
source_path=${source_spec#*:}
source_path="$FAKE_REMOTE_ROOT/${source_path#/scratch/mock/}"
[[ -f "$source_path" && ! -L "$source_path" ]] || exit 94
cp -- "$source_path" "$destination"
MOCK_RSYNC

cat > "$MOCK_BIN/sacct" <<'MOCK_SACCT'
#!/usr/bin/env bash
set -euo pipefail
array=""
while (( $# )); do
  if [[ "$1" == -j ]]; then
    array=$2
    shift 2
  else
    shift
  fi
done
[[ "$array" =~ ^[0-9]+$ ]] || exit 95
for seed in 1 2 3 4 5; do
  state=COMPLETED
  exit_code=0:0
  if [[ "$array" == 9100 && "$seed" == 5 ]]; then
    state=FAILED
    exit_code=1:0
  fi
  printf '%s_%s|%s|%s|%s|\n' "$array" "$seed" "$((array + seed))" \
    "$state" "$exit_code"
done
MOCK_SACCT
chmod 0755 "$MOCK_BIN/ssh" "$MOCK_BIN/rsync" "$MOCK_BIN/sacct"
export FAKE_REMOTE_ROOT

# Minimal immutable evidence source binding containing the executing finalizer.
SOURCE_BUILD="$TMP/source-build"
mkdir -p "$SOURCE_BUILD/hopper"
cp -p -- "$FINALIZER" "$SOURCE_BUILD/hopper/finalize_barn_ledger.sh"
python3 - "$SOURCE_BUILD" <<'PY'
import hashlib
import json
from pathlib import Path
import stat
import sys
root = Path(sys.argv[1])
path = root / "hopper" / "finalize_barn_ledger.sh"
state = {
    "mode": "evidence",
    "worktree_dirty": False,
    "relevant_paths_match_head": True,
    "files": [{
        "path": "hopper/finalize_barn_ledger.sh",
        "worktree_mode": format(stat.S_IMODE(path.stat().st_mode), "o"),
        "worktree_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }],
}
(root / "SOURCE_STATE.json").write_text(json.dumps(state) + "\n")
PY
(
  cd "$SOURCE_BUILD"
  find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)
SOURCE_SHA=$(sha256sum "$SOURCE_BUILD/SHA256SUMS" | awk '{print $1}')
SOURCE_DIR="$FAKE_REMOTE_ROOT/maxrl/bundles/barn_source/${SOURCE_SHA:0:20}"
mkdir -p "$(dirname "$SOURCE_DIR")"
mv -- "$SOURCE_BUILD" "$SOURCE_DIR"
readonly SOURCE_SHA SOURCE_DIR
readonly LOGICAL_SOURCE="/scratch/mock/maxrl/bundles/barn_source/${SOURCE_SHA:0:20}"

make_fixture() {
  local campaign=$1 array_job=$2 mode=$3
  python3 - "$LOCAL_LEDGERS" "$FAKE_REMOTE_ROOT" "$campaign" \
    "$array_job" "$mode" "$SOURCE_SHA" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

local_root = Path(sys.argv[1])
scratch = Path(sys.argv[2])
campaign = sys.argv[3]
array_job = int(sys.argv[4])
mode = sys.argv[5]
source_sha = sys.argv[6]
campaign_root = scratch / "maxrl" / "barn" / "campaigns" / campaign
attempt = "attempt-001"
submitted = "2026-08-14T01:02:03Z"
hash_fields = (
    "manifest_sha256", "split_sha256", "prereg_sha256",
    "analyzer_sha256", "protocol_sha256", "container_sha256",
    "source_sha256",
)
expected_hashes = {
    field: f"{index + 1:x}" * 64 for index, field in enumerate(hash_fields)
}
expected_hashes["source_sha256"] = source_sha
cells = (
    ("primary", 8, ["ours_uN", "uniform", "learnability", "staged"],
     "full_barn_campaign"),
    ("ablation_n2", 2, ["ours_uN", "learnability"],
     "full_barn_n_ablation"),
    ("ablation_n4", 4, ["ours_uN", "learnability"],
     "full_barn_n_ablation"),
    ("ablation_n16", 16, ["ours_uN", "learnability"],
     "full_barn_n_ablation"),
)
rows = []
for cell_index, (cell, n_rollouts, arms, evidence_status) in enumerate(cells):
    cell_array_job = array_job + cell_index * 10
    seed_count = 4 if mode == "omitted" and cell == "primary" else 5
    for seed in range(1, seed_count + 1):
        block = (campaign_root / "cells" / cell / "attempts" / attempt
                 / f"seed-{seed}")
        artifact_path = block / "results" / f"seed-{seed}.json"
        row_path = artifact_path
        if mode == "path" and cell == "primary" and seed == 1:
            row_path = block / "results" / "wrong-name.json"
        row_hashes = dict(expected_hashes)
        if mode == "ledger_hash_drift" and cell == "ablation_n2":
            row_hashes["source_sha256"] = "e" * 64
        rows.append({
            "campaign_id": campaign,
            "campaign_cell": cell,
            "attempt_id": attempt,
            "seed": seed,
            "submitted_utc": submitted,
            "slurm_job_id": None,
            "slurm_array_job_id": str(cell_array_job),
            "slurm_array_task_id": seed,
            "artifact_path": str(row_path),
            "artifact_complete": False,
            "artifact_sha256": None,
            "expected_hashes": row_hashes,
        })
        if mode == "mixed" and cell == "primary" and seed == 5:
            hidden = block.parent / f".seed-{seed}.stage-{cell_array_job + seed}"
            hidden.mkdir(parents=True)
            (hidden / "retained-partial.txt").write_text(
                "non-canonical failed work remains sealed from selectors\n")
            continue
        if mode == "canonical_partial" and cell == "primary" and seed == 5:
            block.mkdir(parents=True)
            continue
        block.mkdir(parents=True, exist_ok=True)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        provenance = dict(expected_hashes)
        provenance.update({
            "asset_hashes_verified": True,
            "split_bound_manifest_sha256": expected_hashes["manifest_sha256"],
        })
        if mode == "provenance" and cell == "primary" and seed == 1:
            provenance["source_sha256"] = "f" * 64
        artifact_attempt = (
            "wrong-attempt"
            if mode == "identity" and cell == "primary" and seed == 1
            else attempt)
        artifact = {
            "schema_version": 1,
            "domain": "barn_gazebo_cpu_navigation",
            "evidence_status": evidence_status,
            "execution": {
                "campaign_id": campaign,
                "attempt_id": artifact_attempt,
                "submitted_utc": datetime.fromisoformat(
                    submitted.replace("Z", "+00:00")).astimezone(
                        timezone.utc).isoformat(),
                "slurm_job_id": str(cell_array_job + seed),
                "slurm_array_job_id": str(cell_array_job),
                "slurm_array_task_id": seed,
            },
            "provenance": provenance,
            "config": {
                "campaign_cell": cell,
                "seed_list": [seed],
                "seeds": 1,
                "seed_start": seed,
                "campaign_seed": seed,
                "arms": arms,
                "execution_order": arms,
                "n_rollouts": n_rollouts,
                "steps": 200,
                "max_training_updates": 200,
                "tasks_per_step": 2,
                "eval_every": 10,
                "eval_episodes": 1,
                "training_sim_step_budget": 1_000_000,
                "eval_sim_step_interval": 200_000,
                "n_strata": 10,
                "n_train_courses": 240,
                "n_heldout_courses": 60,
            },
            # Empty per-arm containers exercise key-shape validation without
            # any scientific endpoint values in the fixture.
            "results": {arm: [] for arm in arms},
        }
        payload = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
        artifact_path.write_text(payload, encoding="utf-8")
        artifact_sha = hashlib.sha256(payload.encode()).hexdigest()
        manifest_payload = f"{artifact_sha}  ./results/seed-{seed}.json\n"
        (block / "SHA256SUMS").write_text(manifest_payload, encoding="utf-8")
        manifest_sha = hashlib.sha256(manifest_payload.encode()).hexdigest()
        complete = {
            "artifact_type": "barn_evidence_seed_complete",
            "campaign_id": campaign,
            "campaign_cell": cell,
            "attempt_id": attempt,
            "seed": str(seed),
            "completed_utc": "2026-08-14T02:03:04Z",
            "sha256sums_sha256": manifest_sha,
        }
        order = (
            "artifact_type", "campaign_id", "campaign_cell", "attempt_id",
            "seed", "completed_utc", "sha256sums_sha256")
        (block / "COMPLETE").write_text(
            "".join(f"{key}\t{complete[key]}\n" for key in order),
            encoding="utf-8")

if mode == "unknown":
    (campaign_root / "cells" / "primary" / "attempts" / "unlisted-attempt"
     / "seed-1").mkdir(parents=True)

ledger = {"schema_version": 1, "submissions": rows}
payload = json.dumps(ledger, indent=2, sort_keys=True) + "\n"
local_root.mkdir(parents=True, exist_ok=True)
(local_root / f"{campaign}.json").write_text(payload, encoding="utf-8")
campaign_root.mkdir(parents=True, exist_ok=True)
(campaign_root / "SUBMISSION_LEDGER.json").write_text(payload, encoding="utf-8")
PY
}

run_finalize_with() {
  local script=$1 campaign=$2
  PATH="$MOCK_BIN:/usr/bin:/bin" \
  HOPPER_HOST=mock@hopper.invalid \
  HOPPER_SCRATCH=/scratch/mock \
  FINALIZE_BARN_LOCAL_LEDGER_DIR="$LOCAL_LEDGERS" \
    /bin/bash "$script" "$campaign" \
      "$LOGICAL_SOURCE" "$SOURCE_SHA"
}
run_finalize() { run_finalize_with "$FINALIZER" "$1"; }

make_fixture campaign-complete 9000 valid
input_sha=$(sha256sum "$LOCAL_LEDGERS/campaign-complete.json" | awk '{print $1}')
TAMPERED_FINALIZER="$TMP/finalize_barn_ledger.tampered.sh"
cp -p -- "$FINALIZER" "$TAMPERED_FINALIZER"
printf '\n# post-stage finalizer tamper\n' >> "$TAMPERED_FINALIZER"
if run_finalize_with "$TAMPERED_FINALIZER" campaign-complete \
    > "$TMP/tamper.out" 2> "$TMP/tamper.err"; then
  fail "tampered ledger finalizer unexpectedly executed"
fi
grep -Fq 'source-bound ledger finalizer preflight failed closed' \
  "$TMP/tamper.err" || fail "tampered finalizer lacked source-binding refusal"
if find "$FAKE_REMOTE_ROOT/maxrl/barn/campaigns/campaign-complete" -type f \
    -name 'SUBMISSION_LEDGER.finalized-*.json' -print -quit | grep -q .; then
  fail "tampered finalizer published a ledger"
fi
run_finalize campaign-complete > "$TMP/complete.out" 2> "$TMP/complete.err"
grep -Fq $'BARN_LEDGER_FINALIZED\tcampaign=campaign-complete\tsubmissions=20\tcomplete=20\tincomplete=0' \
  "$TMP/complete.out" || fail "complete summary is missing"
! grep -Eiq 'mean_success|auc|episode_return|success_rate' \
  "$TMP/complete.out" "$TMP/complete.err" \
  || fail "scientific endpoint label leaked to finalizer output"
[[ "$(sha256sum "$LOCAL_LEDGERS/campaign-complete.json" | awk '{print $1}')" == "$input_sha" ]] \
  || fail "immutable input ledger changed"
complete_copy=$(find "$LOCAL_LEDGERS" -maxdepth 1 -type f \
  -name 'campaign-complete.finalized-*.json' -print)
[[ -n "$complete_copy" && $(printf '%s\n' "$complete_copy" | wc -l) -eq 1 ]] \
  || fail "expected one local complete finalized copy"
complete_hash=$(sha256sum "$complete_copy" | awk '{print $1}')
[[ "$(basename "$complete_copy")" == "campaign-complete.finalized-$complete_hash.json" ]] \
  || fail "local finalized ledger is not content addressed"
remote_copy="$FAKE_REMOTE_ROOT/maxrl/barn/campaigns/campaign-complete/finalized_ledgers/SUBMISSION_LEDGER.finalized-$complete_hash.json"
[[ -f "$remote_copy" && "$(sha256sum "$remote_copy" | awk '{print $1}')" == "$complete_hash" ]] \
  || fail "remote finalized ledger is missing or checksum-invalid"
python3 - "$complete_copy" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

ledger = json.loads(Path(sys.argv[1]).read_text())
assert set(ledger) == {"schema_version", "submissions"}
assert ledger["schema_version"] == 1 and len(ledger["submissions"]) == 20
for row in ledger["submissions"]:
    assert row["artifact_complete"] is True
    offsets = {"primary": 0, "ablation_n2": 10,
               "ablation_n4": 20, "ablation_n16": 30}
    assert row["slurm_job_id"] == str(9000 + offsets[row["campaign_cell"]]
                                      + row["seed"])
    artifact = Path(row["artifact_path"])
    assert row["artifact_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
PY

if run_finalize campaign-complete > "$TMP/repeat.out" 2> "$TMP/repeat.err"; then
  fail "repeat finalization overwrote a prior local copy"
fi
grep -Fq 'refusing to overwrite prior finalized ledger' "$TMP/repeat.err" \
  || fail "repeat finalization did not fail with no-clobber reason"
[[ "$(sha256sum "$complete_copy" | awk '{print $1}')" == "$complete_hash" ]] \
  || fail "repeat finalization changed the prior copy"

make_fixture campaign-mixed 9100 mixed
run_finalize campaign-mixed > "$TMP/mixed.out" 2> "$TMP/mixed.err"
grep -Fq $'submissions=20\tcomplete=19\tincomplete=1' "$TMP/mixed.out" \
  || fail "terminal failed task was not retained as incomplete"
mixed_copy=$(find "$LOCAL_LEDGERS" -maxdepth 1 -type f \
  -name 'campaign-mixed.finalized-*.json' -print)
python3 - "$mixed_copy" <<'PY'
import json
from pathlib import Path
import sys
ledger = json.loads(Path(sys.argv[1]).read_text())
failed = next(row for row in ledger["submissions"]
              if row["campaign_cell"] == "primary" and row["seed"] == 5)
assert failed["slurm_job_id"] == "9105"
assert failed["artifact_complete"] is False
assert failed["artifact_sha256"] is None
PY

expect_refusal() {
  local campaign=$1 array_job=$2 mode=$3 message=$4
  make_fixture "$campaign" "$array_job" "$mode"
  if run_finalize "$campaign" > "$TMP/$campaign.out" 2> "$TMP/$campaign.err"; then
    fail "$mode fixture unexpectedly finalized"
  fi
  grep -Fq "$message" "$TMP/$campaign.err" \
    || fail "$mode fixture did not report its fail-closed reason"
  if find "$FAKE_REMOTE_ROOT/maxrl/barn/campaigns/$campaign" -type f \
      -name 'SUBMISSION_LEDGER.finalized-*.json' -print -quit | grep -q .; then
    fail "$mode fixture published a finalized ledger"
  fi
}

expect_refusal campaign-identity 9200 identity \
  'artifact campaign/attempt identity mismatch'
expect_refusal campaign-provenance 9300 provenance \
  'artifact provenance source_sha256 mismatch'
expect_refusal campaign-unknown 9400 unknown \
  'remote campaign contains an unknown/omitted attempt'
expect_refusal campaign-omitted 9500 omitted \
  'a recorded campaign attempt omits or adds a seed'
expect_refusal campaign-path 9600 path \
  'artifact path does not match its identity'
expect_refusal campaign-hash-drift 9650 ledger_hash_drift \
  'expected hashes drift across campaign submissions'
expect_refusal campaign-canonical-partial 9100 canonical_partial \
  'failed Slurm task exposes a canonical partial/final seed block'

# Even a perfectly complete primary cell must not become a separately
# finalized/unblindable campaign ledger.
make_fixture campaign-subset 9700 valid
python3 - "$LOCAL_LEDGERS/campaign-subset.json" \
  "$FAKE_REMOTE_ROOT/maxrl/barn/campaigns/campaign-subset/SUBMISSION_LEDGER.json" <<'PY'
import json
from pathlib import Path
import sys
for name in sys.argv[1:]:
    path = Path(name)
    ledger = json.loads(path.read_text())
    ledger["submissions"] = [
        row for row in ledger["submissions"]
        if row["campaign_cell"] == "primary"]
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
PY
if run_finalize campaign-subset > "$TMP/subset.out" 2> "$TMP/subset.err"; then
  fail "primary-only campaign ledger unexpectedly finalized"
fi
grep -Fq 'campaign ledger must cover exactly the four preregistered cells' \
  "$TMP/subset.err" || fail "primary-only campaign did not fail exact-cell closure"

make_fixture campaign-symlink 9800 valid
symlink_cell="$FAKE_REMOTE_ROOT/maxrl/barn/campaigns/campaign-symlink/cells/ablation_n2"
mv -- "$symlink_cell" "$symlink_cell.real"
ln -s -- "$symlink_cell.real" "$symlink_cell"
if run_finalize campaign-symlink > "$TMP/symlink.out" 2> "$TMP/symlink.err"; then
  fail "symlinked campaign-cell ancestor unexpectedly finalized"
fi
grep -Fq 'symbolic or non-canonical ancestor' "$TMP/symlink.err" \
  || fail "symlink ancestor did not produce canonicality refusal"

if command -v shellcheck >/dev/null 2>&1; then
  shellcheck -x "$FINALIZER" "$0"
fi
printf 'BARN_LEDGER_FINALIZER_LOCAL_CHECK_PASS\n'
