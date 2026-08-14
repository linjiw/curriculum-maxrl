#!/usr/bin/env bash
# Network-free held-submission transaction test.
set -euo pipefail
umask 077

readonly HERE="$(cd "$(dirname "$0")" && pwd)"
readonly SOURCE_SUBMITTER="$HERE/submit_barn_campaign.sh"
readonly SOURCE_SBATCH="$HERE/sbatch/barn_seed_cpu.sbatch"
readonly TMP="$(mktemp -d /tmp/barn-submit-test.XXXXXX)"
cleanup() {
  if [[ ${BARN_SUBMIT_TEST_KEEP:-0} == 1 ]]; then
    echo "kept test directory: $TMP" >&2
    return
  fi
  [[ -d "$TMP" ]] && rm -rf -- "$TMP"
}
trap cleanup EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

readonly REPO="$TMP/repo"
readonly MOCK_BIN="$TMP/mock-bin"
readonly REMOTE="$TMP/remote"
readonly EVENTS="$TMP/events"
readonly LEDGERS="$TMP/ledgers"
mkdir -p "$REPO/hopper/sbatch" "$REPO/icra2027" "$MOCK_BIN" \
  "$REMOTE" "$LEDGERS"
cp -p -- "$SOURCE_SUBMITTER" "$REPO/hopper/submit_barn_campaign.sh"
cp -p -- "$SOURCE_SBATCH" "$REPO/hopper/sbatch/barn_seed_cpu.sbatch"
printf '**Status:** FROZEN\n' > "$REPO/icra2027/prereg_icra.md"
printf '{"status":"FROZEN","ablation":{"fresh_cell_names":["ablation_n2","ablation_n4","ablation_n16"]}}\n' \
  > "$REPO/icra2027/barn_protocol.json"
printf '{"fixture":"manifest"}\n' > "$REPO/icra2027/barn_manifest.jsonl"
printf '{"fixture":"split"}\n' > "$REPO/icra2027/barn_split.json"
printf '# fixture analyzer\n' > "$REPO/icra2027/analyze_campaign.py"
: > "$EVENTS"

SOURCE_BUILD="$TMP/source-build"
mkdir -p "$SOURCE_BUILD/hopper/sbatch" "$SOURCE_BUILD/icra2027"
for rel in hopper/submit_barn_campaign.sh hopper/sbatch/barn_seed_cpu.sbatch \
  icra2027/prereg_icra.md icra2027/barn_protocol.json \
  icra2027/barn_manifest.jsonl icra2027/barn_split.json \
  icra2027/analyze_campaign.py; do
  mkdir -p "$SOURCE_BUILD/$(dirname "$rel")"
  cp -p -- "$REPO/$rel" "$SOURCE_BUILD/$rel"
done
python3 - "$SOURCE_BUILD" <<'PY'
import hashlib
import json
from pathlib import Path
import stat
import sys
root = Path(sys.argv[1])
files = []
for path in sorted(p for p in root.rglob("*") if p.is_file()):
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
(
  cd "$SOURCE_BUILD"
  find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)
SOURCE_SHA=$(sha256sum "$SOURCE_BUILD/SHA256SUMS" | awk '{print $1}')
SOURCE_DIR="$REMOTE/maxrl/bundles/barn_source/${SOURCE_SHA:0:20}"
mkdir -p "$(dirname "$SOURCE_DIR")"
mv -- "$SOURCE_BUILD" "$SOURCE_DIR"
readonly SOURCE_SHA SOURCE_DIR
readonly LOGICAL_SOURCE="/scratch/mock/maxrl/bundles/barn_source/${SOURCE_SHA:0:20}"

cat > "$MOCK_BIN/ssh" <<'MOCK_SSH'
#!/usr/bin/env bash
set -euo pipefail
while [[ ${1:-} == -o ]]; do shift 2; done
[[ ${1:-} == mock@hopper.invalid ]] || exit 90
shift
mapped=()
for argument in "$@"; do
  case "$argument" in
    /scratch/mock) mapped+=("$REMOTE") ;;
    /scratch/mock/*) mapped+=("$REMOTE/${argument#/scratch/mock/}") ;;
    *) mapped+=("$argument") ;;
  esac
done
set +e
output=$("${mapped[@]}")
status=$?
set -e
(( status == 0 )) || exit "$status"
if [[ ${mapped[0]:-} == bash && ${mapped[1]:-} == -s ]]; then
  joined=" ${mapped[*]} "
  if [[ "$joined" == *".SUBMISSION_LEDGER.stage-"* \
     && "$joined" == *"SUBMISSION_LEDGER.json"* ]]; then
    printf 'ledger_verified\n' >> "$EVENTS"
  fi
fi
output=${output//$REMOTE/\/scratch\/mock}
printf '%s\n' "$output"
MOCK_SSH

cat > "$MOCK_BIN/scp" <<'MOCK_SCP'
#!/usr/bin/env bash
set -euo pipefail
args=("$@")
positional=()
index=0
while (( index < ${#args[@]} )); do
  case "${args[$index]}" in
    -o) index=$((index + 2)) ;;
    -p|--) index=$((index + 1)) ;;
    *) positional+=("${args[$index]}"); index=$((index + 1)) ;;
  esac
done
(( ${#positional[@]} == 2 )) || exit 91
source=${positional[0]}
spec=${positional[1]}
[[ "$spec" == mock@hopper.invalid:/scratch/mock/* ]] || exit 92
target="$REMOTE/${spec#mock@hopper.invalid:/scratch/mock/}"
if [[ "$target" == *".SUBMISSION_LEDGER.stage-"* ]]; then
  printf 'ledger_scp\n' >> "$EVENTS"
  [[ ${MOCK_FAIL_LEDGER_UPLOAD:-0} != 1 ]] || exit 93
else
  printf 'sbatch_scp\n' >> "$EVENTS"
fi
mkdir -p "$(dirname "$target")"
cp -p -- "$source" "$target"
MOCK_SCP

cat > "$MOCK_BIN/sbatch" <<'MOCK_SBATCH'
#!/usr/bin/env bash
set -euo pipefail
parsable=false
held=false
for argument in "$@"; do
  [[ "$argument" == --parsable ]] && parsable=true
  [[ "$argument" == --hold ]] && held=true
done
[[ "$parsable" == true && "$held" == true ]] || exit 94
printf 'submit_held\n' >> "$EVENTS"
printf '7001\n'
MOCK_SBATCH

cat > "$MOCK_BIN/scontrol" <<'MOCK_SCONTROL'
#!/usr/bin/env bash
set -euo pipefail
[[ ${1:-} == release && ${2:-} == 7001 ]] || exit 95
grep -q '^ledger_verified$' "$EVENTS" || exit 96
printf 'release\n' >> "$EVENTS"
MOCK_SCONTROL
cat > "$MOCK_BIN/squeue" <<'MOCK_SQUEUE'
#!/usr/bin/env bash
set -euo pipefail
job=""
format=""
while (( $# )); do
  case "$1" in
    --jobs=*) job=${1#--jobs=} ;;
    --format=*) format=${1#--format=} ;;
  esac
  shift
done
[[ "$job" == 7001 && "$format" == '%F|%T|%r' ]] || exit 97
printf '7001|PENDING|JobHeldUser\n'
MOCK_SQUEUE
chmod 0755 "$MOCK_BIN/ssh" "$MOCK_BIN/scp" "$MOCK_BIN/sbatch" \
  "$MOCK_BIN/scontrol" "$MOCK_BIN/squeue"
export REMOTE EVENTS

run_submit() {
  local script=$1 campaign=$2 attempt=$3
  PATH="$MOCK_BIN:/usr/bin:/bin" HOPPER_HOST=mock@hopper.invalid \
  HOPPER_SCRATCH=/scratch/mock BARN_SUBMISSION_LEDGER_DIR="$LEDGERS" \
    /bin/bash "$script" "$campaign" primary "$attempt" \
      "$LOGICAL_SOURCE" "$SOURCE_SHA"
}

if ! run_submit "$REPO/hopper/submit_barn_campaign.sh" \
    campaign-ok attempt-001 > "$TMP/ok.out" 2> "$TMP/ok.err"; then
  sed -n '1,120p' "$TMP/ok.err" >&2
  sed -n '1,120p' "$EVENTS" >&2
  fail "valid held submission transaction failed"
fi
grep -Fq $'BARN_ARRAY_RELEASED\tjob_id=7001\tcampaign=campaign-ok' \
  "$TMP/ok.out" || fail "release metadata missing"
python3 - "$EVENTS" <<'PY'
from pathlib import Path
events = Path(__import__('sys').argv[1]).read_text().splitlines()
assert events.index("submit_held") < events.index("ledger_scp")
assert events.index("ledger_scp") < events.index("ledger_verified")
assert events.index("ledger_verified") < events.index("release")
PY
cmp -- "$LEDGERS/campaign-ok.json" \
  "$REMOTE/maxrl/barn/campaigns/campaign-ok/SUBMISSION_LEDGER.json" \
  || fail "installed remote ledger differs from durable local ledger"

mkdir "$LEDGERS/.campaign-locked.submit.lock"
: > "$EVENTS"
if run_submit "$REPO/hopper/submit_barn_campaign.sh" \
    campaign-locked attempt-001 > "$TMP/locked.out" 2> "$TMP/locked.err"; then
  fail "concurrent local campaign submission unexpectedly proceeded"
fi
[[ ! -s "$EVENTS" ]] || fail "local campaign lock was checked after transport"
grep -Fq 'holds the local campaign-ledger lock' "$TMP/locked.err" \
  || fail "local campaign lock refusal was not explicit"
rmdir "$LEDGERS/.campaign-locked.submit.lock"

: > "$EVENTS"
if MOCK_FAIL_LEDGER_UPLOAD=1 run_submit \
    "$REPO/hopper/submit_barn_campaign.sh" campaign-upload-fail attempt-001 \
    > "$TMP/fail.out" 2> "$TMP/fail.err"; then
  fail "ledger upload failure unexpectedly released held job"
fi
grep -q '^submit_held$' "$EVENTS" || fail "upload failure never held a job"
! grep -q '^release$' "$EVENTS" || fail "upload failure released held job"
grep -Fq 'remains held' "$TMP/fail.err" \
  || fail "upload failure omitted held-job warning"
events_before=$(wc -l < "$EVENTS")
if run_submit "$REPO/hopper/submit_barn_campaign.sh" \
    campaign-upload-fail attempt-002 > "$TMP/other-attempt.out" \
    2> "$TMP/other-attempt.err"; then
  fail "new attempt bypassed an unresolved held-array transaction"
fi
[[ $(wc -l < "$EVENTS") -eq "$events_before" ]] \
  || fail "conflicting new attempt reached remote transport"
if ! run_submit "$REPO/hopper/submit_barn_campaign.sh" \
    campaign-upload-fail attempt-001 > "$TMP/resume.out" 2> "$TMP/resume.err"; then
  sed -n '1,120p' "$TMP/resume.err" >&2
  fail "same-attempt resume did not reconcile and release the held array"
fi
grep -Fq $'BARN_ARRAY_RESUMED\tjob_id=7001\tcampaign=campaign-upload-fail' \
  "$TMP/resume.out" || fail "resume metadata missing"
[[ $(grep -c '^submit_held$' "$EVENTS") -eq 1 ]] \
  || fail "resume allocated a second array"
[[ $(grep -c '^release$' "$EVENTS") -eq 1 ]] \
  || fail "resume did not release exactly once"
[[ ! -e "$LEDGERS/.campaign-upload-fail.pending-submission.json" ]] \
  || fail "successful resume retained its pending marker"

mkdir -p "$REMOTE/maxrl/barn/campaigns/campaign-sealed/sealed_campaigns/campaign-deadbeef"
printf 'sealed metadata control\n' \
  > "$REMOTE/maxrl/barn/campaigns/campaign-sealed/sealed_campaigns/campaign-deadbeef/COMPLETE"
: > "$EVENTS"
if run_submit "$REPO/hopper/submit_barn_campaign.sh" \
    campaign-sealed attempt-001 > "$TMP/sealed.out" 2> "$TMP/sealed.err"; then
  fail "post-seal campaign unexpectedly accepted another attempt"
fi
[[ ! -s "$EVENTS" ]] || fail "post-seal refusal reached submission transport"
grep -Fq 'source-bound evidence submission preflight failed closed' \
  "$TMP/sealed.err" || fail "post-seal refusal was not fail-closed"

TAMPERED="$TMP/tampered"
cp -a -- "$REPO" "$TAMPERED"
printf '\n# post-stage tamper\n' >> "$TAMPERED/hopper/submit_barn_campaign.sh"
: > "$EVENTS"
if run_submit "$TAMPERED/hopper/submit_barn_campaign.sh" \
    campaign-tamper attempt-001 > "$TMP/tamper.out" 2> "$TMP/tamper.err"; then
  fail "tampered submitter unexpectedly submitted"
fi
[[ ! -s "$EVENTS" ]] || fail "tampered submitter reached remote sbatch"
grep -Fq 'source-bound evidence submission preflight failed closed' \
  "$TMP/tamper.err" || fail "tampered submitter lacked binding refusal"

cp -p -- "$REPO/icra2027/prereg_icra.md" "$TMP/prereg.clean"
printf '**Status:** FROZEN, test fixture.\n' > "$REPO/icra2027/prereg_icra.md"
: > "$EVENTS"
if run_submit "$REPO/hopper/submit_barn_campaign.sh" \
    campaign-punctuation attempt-001 > "$TMP/punct.out" 2> "$TMP/punct.err"; then
  fail "punctuated FROZEN status unexpectedly submitted"
fi
[[ ! -s "$EVENTS" ]] || fail "punctuated FROZEN reached remote transport"
cp -p -- "$TMP/prereg.clean" "$REPO/icra2027/prereg_icra.md"

! rg -q 'hopper\.sh' "$REPO/hopper/submit_barn_campaign.sh" \
  || fail "submitter retained generic hopper.sh dependency"
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck -x "$REPO/hopper/submit_barn_campaign.sh" "$0"
fi
printf 'BARN_SUBMIT_LOCAL_CHECK_PASS\n'
