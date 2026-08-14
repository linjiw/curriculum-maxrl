#!/usr/bin/env bash
# Submit one held five-seed BARN evidence array, durably record its normalized
# outcome-blind ledger, atomically install and verify that ledger on Hopper,
# and only then release the array.
#
# Usage:
#   submit_barn_campaign.sh CAMPAIGN_ID CELL ATTEMPT_ID SOURCE_BUNDLE SOURCE_SHA
set -euo pipefail
umask 077

if (( $# != 5 )); then
  echo "usage: $0 CAMPAIGN_ID CELL ATTEMPT_ID SOURCE_BUNDLE SOURCE_SHA" >&2
  exit 2
fi
readonly CAMPAIGN_ID=$1
readonly CAMPAIGN_CELL=$2
readonly ATTEMPT_ID=$3
readonly SOURCE_BUNDLE=$4
readonly SOURCE_SHA=$5
readonly HERE="$(cd "$(dirname "$0")" && pwd)"
readonly ROOT="$(cd "$HERE/.." && pwd)"
readonly HOST="${HOPPER_HOST:-lwang44@hopper.orc.gmu.edu}"
readonly SCRATCH="${HOPPER_SCRATCH:-/scratch/lwang44}"
readonly PREREG="$ROOT/icra2027/prereg_icra.md"
readonly PROTOCOL="$ROOT/icra2027/barn_protocol.json"
readonly LOCAL_SBATCH="$HERE/sbatch/barn_seed_cpu.sbatch"
readonly LEDGER_DIR="${BARN_SUBMISSION_LEDGER_DIR:-$ROOT/autoresearch/iterate-260814-0047/submission_ledgers}"
readonly LEDGER="$LEDGER_DIR/$CAMPAIGN_ID.json"
readonly REMOTE_CAMPAIGN_ROOT="$SCRATCH/maxrl/barn/campaigns/$CAMPAIGN_ID"
readonly REMOTE_LEDGER="$REMOTE_CAMPAIGN_ROOT/SUBMISSION_LEDGER.json"
readonly REMOTE_SCRIPT_PARENT="$SCRATCH/maxrl/barn/submission_scripts"

[[ "$CAMPAIGN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || {
  echo "unsafe CAMPAIGN_ID" >&2; exit 2;
}
[[ "$ATTEMPT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || {
  echo "unsafe ATTEMPT_ID" >&2; exit 2;
}
case "$CAMPAIGN_CELL" in
  primary|ablation_n2|ablation_n4|ablation_n16) ;;
  *) echo "invalid campaign cell" >&2; exit 2 ;;
esac
[[ "$HOST" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+$ ]] || {
  echo "unsafe HOPPER_HOST" >&2; exit 2;
}
[[ "$SCRATCH" =~ ^/scratch/[A-Za-z0-9._-]+$ ]] || {
  echo "HOPPER_SCRATCH must be canonical below /scratch" >&2; exit 2;
}
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{64}$ ]] || {
  echo "invalid source SHA-256" >&2; exit 2;
}
[[ "$SOURCE_BUNDLE" == \
   "$SCRATCH/maxrl/bundles/barn_source/${SOURCE_SHA:0:20}" ]] || {
  echo "source bundle path is not content addressed" >&2; exit 2;
}
for path in "$0" "$LOCAL_SBATCH" "$PREREG" "$PROTOCOL" \
            "$ROOT/icra2027/barn_manifest.jsonl" \
            "$ROOT/icra2027/barn_split.json" \
            "$ROOT/icra2027/analyze_campaign.py"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    printf 'missing or symbolic local evidence input: %s\n' "$path" >&2
    exit 2
  }
done
grep -Eq '^\*\*Status:\*\* FROZEN[[:blank:]]*$' "$PREREG" || {
  echo "refusing evidence submission: preregistration is not exactly FROZEN" >&2
  exit 1
}
python3 - "$PROTOCOL" "$CAMPAIGN_CELL" <<'PY'
import json
import sys
protocol = json.load(open(sys.argv[1], encoding="utf-8"))
cell = sys.argv[2]
if protocol.get("status") != "FROZEN":
    raise SystemExit("refusing evidence submission: protocol is not FROZEN")
if cell != "primary" and cell not in protocol["ablation"]["fresh_cell_names"]:
    raise SystemExit("campaign cell is not declared as a fresh protocol cell")
PY

readonly SUBMITTER_SHA="$(sha256sum -- "$0" | awk '{print $1}')"
readonly SBATCH_SHA="$(sha256sum -- "$LOCAL_SBATCH" | awk '{print $1}')"
readonly MANIFEST_SHA="$(sha256sum -- "$ROOT/icra2027/barn_manifest.jsonl" | awk '{print $1}')"
readonly SPLIT_SHA="$(sha256sum -- "$ROOT/icra2027/barn_split.json" | awk '{print $1}')"
readonly PREREG_SHA="$(sha256sum -- "$PREREG" | awk '{print $1}')"
readonly ANALYZER_SHA="$(sha256sum -- "$ROOT/icra2027/analyze_campaign.py" | awk '{print $1}')"
readonly PROTOCOL_SHA="$(sha256sum -- "$PROTOCOL" | awk '{print $1}')"
readonly ARCHIVE_SHA="5ad443412f6f2f38b6d0e1d330c9a820ab48e566553197459005e751711fe320"
readonly CONTAINER_SHA="cd6620e33c0822f7d6a03c6de6ea9dd4304f0927e8d7997c003560f5b4781be0"
SUBMITTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p -- "$LEDGER_DIR"
[[ -d "$LEDGER_DIR" && ! -L "$LEDGER_DIR" ]] || {
  echo "local submission ledger directory is symbolic or non-directory" >&2
  exit 2
}
readonly LOCAL_LOCK="$LEDGER_DIR/.$CAMPAIGN_ID.submit.lock"
if ! mkdir -- "$LOCAL_LOCK"; then
  echo "another submission transaction holds the local campaign-ledger lock" >&2
  exit 2
fi
lock_held=true
released=false
cleanup_submission() {
  status=$?
  if [[ "${lock_held:-false}" == true ]]; then
    rmdir -- "$LOCAL_LOCK" 2>/dev/null || true
  fi
  if (( status != 0 )) && [[ -n "${ARRAY_JOB_ID:-}" \
     && "${released:-false}" != true ]]; then
    printf 'array %s remains held; ledger transaction did not complete\n' \
      "$ARRAY_JOB_ID" >&2
  fi
  return "$status"
}
trap cleanup_submission EXIT
readonly PENDING_TX="$LEDGER_DIR/.$CAMPAIGN_ID.pending-submission.json"
RESUME=false
PROPOSED_LEDGER=
if [[ -e "$PENDING_TX" || -L "$PENDING_TX" ]]; then
  [[ -f "$PENDING_TX" && ! -L "$PENDING_TX" ]] || {
    echo "pending submission marker is symbolic or non-regular" >&2; exit 2;
  }
  if ! pending_output=$(python3 - "$PENDING_TX" "$LEDGER_DIR" \
      "$CAMPAIGN_ID" "$CAMPAIGN_CELL" "$ATTEMPT_ID" "$SOURCE_SHA" \
      "$MANIFEST_SHA" "$SPLIT_SHA" "$PREREG_SHA" "$ANALYZER_SHA" \
      "$PROTOCOL_SHA" "$CONTAINER_SHA" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys
marker_path, ledger_dir = Path(sys.argv[1]), Path(sys.argv[2])
campaign, cell, attempt, source_sha, *hash_values = sys.argv[3:]
hash_names = (
    "manifest_sha256", "split_sha256", "prereg_sha256",
    "analyzer_sha256", "protocol_sha256", "container_sha256",
    "source_sha256",
)
expected_hashes = dict(zip(hash_names, [*hash_values, source_sha], strict=True))
marker = json.loads(marker_path.read_text(encoding="utf-8"))
fields = {
    "schema_version", "campaign_id", "campaign_cell", "attempt_id",
    "submitted_utc", "slurm_array_job_id", "previous_ledger_sha256",
    "proposed_ledger_sha256", "proposed_ledger_name", "source_sha256",
}
if set(marker) != fields or marker.get("schema_version") != 1:
    raise SystemExit("pending submission marker schema mismatch")
if (marker["campaign_id"], marker["campaign_cell"], marker["attempt_id"],
        marker["source_sha256"]) != (campaign, cell, attempt, source_sha):
    raise SystemExit("another unresolved campaign submission must be resumed first")
if not re.fullmatch(r"[0-9]+", marker["slurm_array_job_id"]):
    raise SystemExit("pending submission job identity is invalid")
for field in ("proposed_ledger_sha256",):
    if not re.fullmatch(r"[0-9a-f]{64}", marker[field]):
        raise SystemExit("pending submission digest is invalid")
previous = marker["previous_ledger_sha256"]
if previous != "ABSENT" and not re.fullmatch(r"[0-9a-f]{64}", previous):
    raise SystemExit("pending previous-ledger digest is invalid")
proposed = ledger_dir / marker["proposed_ledger_name"]
if (proposed.parent != ledger_dir or not proposed.is_file()
        or proposed.is_symlink()
        or hashlib.sha256(proposed.read_bytes()).hexdigest()
           != marker["proposed_ledger_sha256"]):
    raise SystemExit("pending proposed ledger is missing or invalid")
ledger = json.loads(proposed.read_text(encoding="utf-8"))
if (set(ledger) != {"schema_version", "submissions"}
        or ledger.get("schema_version") != 1
        or not isinstance(ledger.get("submissions"), list)):
    raise SystemExit("pending proposed ledger schema mismatch")
rows = [row for row in ledger.get("submissions", [])
        if row.get("campaign_id") == campaign
        and row.get("campaign_cell") == cell
        and row.get("attempt_id") == attempt]
if (len(rows) != 5 or {row.get("seed") for row in rows} != set(range(1, 6))
        or any(row.get("slurm_array_job_id")
               != marker["slurm_array_job_id"] for row in rows)
        or any(row.get("submitted_utc") != marker["submitted_utc"]
               for row in rows)
        or any(row.get("artifact_complete") is not False
               or row.get("artifact_sha256") is not None
               or row.get("slurm_job_id") is not None
               or row.get("expected_hashes") != expected_hashes
               for row in rows)):
    raise SystemExit("pending ledger lacks the exact five incomplete attempt rows")
print(marker["submitted_utc"])
print(marker["slurm_array_job_id"])
print(previous)
print(marker["proposed_ledger_sha256"])
print(proposed)
PY
  ); then
    echo "pending submission resume validation failed closed" >&2
    exit 2
  fi
  mapfile -t pending_fields <<< "$pending_output"
  (( ${#pending_fields[@]} == 5 )) || {
    echo "pending submission marker output mismatch" >&2; exit 2;
  }
  SUBMITTED_UTC=${pending_fields[0]}
  ARRAY_JOB_ID=${pending_fields[1]}
  PREVIOUS_LEDGER_SHA=${pending_fields[2]}
  LEDGER_SHA=${pending_fields[3]}
  PROPOSED_LEDGER=${pending_fields[4]}
  python3 - "$LEDGER" "$PROPOSED_LEDGER" "$PREVIOUS_LEDGER_SHA" \
    "$LEDGER_SHA" <<'PY'
import hashlib
import os
from pathlib import Path
import sys
ledger, proposed = Path(sys.argv[1]), Path(sys.argv[2])
previous, expected = sys.argv[3:]
current = (hashlib.sha256(ledger.read_bytes()).hexdigest()
           if ledger.is_file() and not ledger.is_symlink() else "ABSENT")
if current == expected:
    raise SystemExit(0)
if current != previous:
    raise SystemExit("local ledger conflicts with pending transaction")
payload = proposed.read_bytes()
temporary = ledger.with_name(f".{ledger.name}.resume-{os.getpid()}")
with temporary.open("xb") as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, ledger)
PY
  RESUME=true
else
  if [[ -e "$LEDGER" || -L "$LEDGER" ]]; then
    [[ -f "$LEDGER" && ! -L "$LEDGER" ]] || {
      echo "local submission ledger is symbolic or non-regular" >&2; exit 2;
    }
    PREVIOUS_LEDGER_SHA="$(sha256sum -- "$LEDGER" | awk '{print $1}')"
  else
    PREVIOUS_LEDGER_SHA=ABSENT
  fi

  # Reject malformed state and duplicate attempts before allocating a held job.
  python3 - "$LEDGER" "$CAMPAIGN_ID" "$CAMPAIGN_CELL" "$ATTEMPT_ID" \
  "$SCRATCH" "$MANIFEST_SHA" "$SPLIT_SHA" "$PREREG_SHA" \
  "$ANALYZER_SHA" "$PROTOCOL_SHA" "$CONTAINER_SHA" "$SOURCE_SHA" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
campaign, cell, attempt, scratch = sys.argv[2:6]
hash_names = (
    "manifest_sha256", "split_sha256", "prereg_sha256",
    "analyzer_sha256", "protocol_sha256", "container_sha256",
    "source_sha256",
)
expected_hashes = dict(zip(hash_names, sys.argv[6:], strict=True))
if not path.exists():
    raise SystemExit(0)
ledger = json.loads(path.read_text(encoding="utf-8"))
row_fields = {
    "campaign_id", "campaign_cell", "attempt_id", "seed", "submitted_utc",
    "slurm_job_id", "slurm_array_job_id", "slurm_array_task_id",
    "artifact_path", "artifact_complete", "artifact_sha256",
    "expected_hashes",
}
if (set(ledger) != {"schema_version", "submissions"}
        or ledger.get("schema_version") != 1
        or not isinstance(ledger.get("submissions"), list)):
    raise SystemExit("existing BARN ledger has unsupported schema")
groups = {}
for index, row in enumerate(ledger["submissions"]):
    if not isinstance(row, dict) or set(row) != row_fields:
        raise SystemExit(f"existing BARN ledger row {index} fields are not exact")
    if row["campaign_id"] != campaign:
        raise SystemExit("existing BARN ledger campaign identity mismatch")
    if row["expected_hashes"] != expected_hashes:
        raise SystemExit("existing BARN ledger expected hashes differ")
    seed = row["seed"]
    if (not isinstance(seed, int) or isinstance(seed, bool)
            or seed not in range(1, 6)
            or row["slurm_array_task_id"] != seed):
        raise SystemExit("existing BARN ledger seed/task mismatch")
    expected_path = (
        f"{scratch}/maxrl/barn/campaigns/{campaign}/cells/"
        f"{row['campaign_cell']}/attempts/{row['attempt_id']}/seed-{seed}/"
        f"results/seed-{seed}.json")
    if row["artifact_path"] != expected_path:
        raise SystemExit("existing BARN ledger artifact path mismatch")
    key = (row["campaign_cell"], row["attempt_id"],
           row["slurm_array_job_id"])
    groups.setdefault(key, set()).add(seed)
    if row["campaign_cell"] == cell and row["attempt_id"] == attempt:
        raise SystemExit("ledger already contains this campaign/cell/attempt")
if any(seeds != set(range(1, 6)) for seeds in groups.values()):
    raise SystemExit("existing BARN ledger contains a partial attempt")
PY
fi

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 \
          -o ServerAliveCountMax=3)

# Verify the frozen remote closure and bind every local launch input to its
# bundled counterpart. No result artifact is read by this preflight.
if ! ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- \
  "$SOURCE_BUNDLE" "$SOURCE_SHA" "$SCRATCH" "$CAMPAIGN_ID" "$SUBMITTER_SHA" \
  "$SBATCH_SHA" "$MANIFEST_SHA" "$SPLIT_SHA" "$PREREG_SHA" \
  "$ANALYZER_SHA" "$PROTOCOL_SHA" <<'REMOTE'
set -euo pipefail
bundle=$1
source_sha=$2
scratch=$3
campaign=$4
shift 4
expected=("$@")
[[ "$bundle" == "$scratch/maxrl/bundles/barn_source/${source_sha:0:20}" \
   && -d "$bundle" && ! -L "$bundle" \
   && "$(readlink -f -- "$bundle")" == "$bundle" ]] || exit 2
[[ -f "$bundle/SHA256SUMS" && ! -L "$bundle/SHA256SUMS" \
   && "$(sha256sum -- "$bundle/SHA256SUMS" | awk '{print $1}')" \
      == "$source_sha" ]] || exit 2
(cd "$bundle" && sha256sum -c --strict --quiet SHA256SUMS)
test -z "$(find "$bundle" -type l -print -quit)"
python3 - "$bundle/SOURCE_STATE.json" "$bundle" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
state_path, root = Path(sys.argv[1]), Path(sys.argv[2])
state = json.loads(state_path.read_text(encoding="utf-8"))
if (state.get("mode") != "evidence"
        or state.get("worktree_dirty") is not False
        or state.get("relevant_paths_match_head") is not True):
    raise SystemExit("source bundle is not a clean evidence closure")
for row in state.get("files", []):
    path = root / row["path"]
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise SystemExit("source state contains a non-regular file")
    mode = format(stat.S_IMODE(info.st_mode), "o")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if mode != row["worktree_mode"] or digest != row["worktree_sha256"]:
        raise SystemExit("source state file evidence mismatch")
PY
paths=(
  "$bundle/hopper/submit_barn_campaign.sh"
  "$bundle/hopper/sbatch/barn_seed_cpu.sbatch"
  "$bundle/icra2027/barn_manifest.jsonl"
  "$bundle/icra2027/barn_split.json"
  "$bundle/icra2027/prereg_icra.md"
  "$bundle/icra2027/analyze_campaign.py"
  "$bundle/icra2027/barn_protocol.json"
)
for index in "${!paths[@]}"; do
  [[ -f "${paths[$index]}" && ! -L "${paths[$index]}" \
     && "$(sha256sum -- "${paths[$index]}" | awk '{print $1}')" \
        == "${expected[$index]}" ]] || exit 2
done
sealed_root="$scratch/maxrl/barn/campaigns/$campaign/sealed_campaigns"
if [[ -e "$sealed_root" || -L "$sealed_root" ]]; then
  [[ -d "$sealed_root" && ! -L "$sealed_root" \
     && "$(readlink -f -- "$sealed_root")" == "$sealed_root" ]] || exit 2
  test -z "$(find "$sealed_root" -mindepth 1 -maxdepth 1 \
    \( -type l -o ! -type d \) -print -quit)"
  test -z "$(find "$sealed_root" -mindepth 1 -maxdepth 1 \
    -type d -name 'campaign-*' -print -quit)"
fi
REMOTE
then
  echo "source-bound evidence submission preflight failed closed" >&2
  exit 2
fi

if [[ "$RESUME" != true ]]; then
  # Stage the already-bound local sbatch through direct transport, then publish
  # a content-addressed remote copy without replacement and verify it again.
  readonly REMOTE_SBATCH="$REMOTE_SCRIPT_PARENT/barn_seed_cpu-$SBATCH_SHA.sbatch"
  readonly REMOTE_SBATCH_STAGE="$REMOTE_SCRIPT_PARENT/.barn_seed_cpu-$SBATCH_SHA.stage-$$-${RANDOM:-0}"
ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- \
  "$REMOTE_SCRIPT_PARENT" "$REMOTE_SBATCH_STAGE" <<'REMOTE'
set -euo pipefail
mkdir -p -- "$1"
[[ -d "$1" && ! -L "$1" && ! -e "$2" && ! -L "$2" ]] || exit 2
REMOTE
scp "${SSH_OPTS[@]}" -p -- "$LOCAL_SBATCH" "$HOST:$REMOTE_SBATCH_STAGE"
if ! ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- \
  "$REMOTE_SBATCH_STAGE" "$REMOTE_SBATCH" "$SBATCH_SHA" <<'REMOTE'
set -euo pipefail
stage=$1
target=$2
expected=$3
[[ -f "$stage" && ! -L "$stage" \
   && "$(sha256sum -- "$stage" | awk '{print $1}')" == "$expected" ]] || exit 2
if [[ -e "$target" || -L "$target" ]]; then
  [[ -f "$target" && ! -L "$target" \
     && "$(sha256sum -- "$target" | awk '{print $1}')" == "$expected" ]] \
    || exit 2
  rm -f -- "$stage"
else
  ln -- "$stage" "$target"
  rm -f -- "$stage"
fi
[[ "$(sha256sum -- "$target" | awk '{print $1}')" == "$expected" ]] || exit 2
REMOTE
then
  echo "remote evidence sbatch staging failed closed" >&2
  exit 2
fi

if ! submit_output=$(ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- \
  "$REMOTE_SBATCH" "$SBATCH_SHA" "$SOURCE_BUNDLE" "$SOURCE_SHA" \
  "$ARCHIVE_SHA" "$CONTAINER_SHA" "$MANIFEST_SHA" "$SPLIT_SHA" \
  "$PREREG_SHA" "$ANALYZER_SHA" "$PROTOCOL_SHA" "$CAMPAIGN_ID" \
  "$CAMPAIGN_CELL" "$ATTEMPT_ID" "$SUBMITTED_UTC" <<'REMOTE'
set -euo pipefail
script=$1
script_sha=$2
shift 2
[[ -f "$script" && ! -L "$script" \
   && "$(sha256sum -- "$script" | awk '{print $1}')" == "$script_sha" ]] \
  || exit 2
exports="ALL,BARN_SOURCE_BUNDLE_DIR=$1,BARN_SOURCE_SHA256=$2,BARN_DATASET_ARCHIVE_SHA256=$3,BARN_CONTAINER_SHA256=$4,BARN_MANIFEST_SHA256=$5,BARN_SPLIT_SHA256=$6,BARN_PREREG_SHA256=$7,BARN_ANALYZER_SHA256=$8,BARN_PROTOCOL_SHA256=$9"
shift 9
exports="$exports,BARN_CAMPAIGN_ID=$1,BARN_CAMPAIGN_CELL=$2,BARN_ATTEMPT_ID=$3,BARN_SUBMITTED_UTC=$4"
record=$(sbatch --parsable --hold --export="$exports" "$script")
job_id=${record%%;*}
[[ "$job_id" =~ ^[0-9]+$ ]] || exit 2
printf 'ARRAY_JOB_ID=%s\n' "$job_id"
REMOTE
); then
  echo "held evidence array submission failed closed" >&2
  exit 2
fi
  ARRAY_JOB_ID=$(printf '%s\n' "$submit_output" \
  | awk -F= '$1 == "ARRAY_JOB_ID" {print $2}')
  [[ "$ARRAY_JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "submission succeeded but array job ID could not be parsed; job remains held" >&2
    exit 2
  }
  printf 'BARN_ARRAY_HELD\tjob_id=%s\tcampaign=%s\tcell=%s\n' \
    "$ARRAY_JOB_ID" "$CAMPAIGN_ID" "$CAMPAIGN_CELL"
else
  if ! held_state=$(ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- \
      "$ARRAY_JOB_ID" <<'REMOTE'
set -euo pipefail
squeue --noheader --jobs="$1" --format='%F|%T|%r'
REMOTE
  ); then
    echo "could not verify pending held array for resume" >&2
    exit 2
  fi
  [[ "$held_state" == "$ARRAY_JOB_ID|PENDING|JobHeldUser" ]] || {
    echo "pending transaction job is not an exact user-held array" >&2
    exit 2
  }
  printf 'BARN_ARRAY_RESUMED\tjob_id=%s\tcampaign=%s\tcell=%s\n' \
    "$ARRAY_JOB_ID" "$CAMPAIGN_ID" "$CAMPAIGN_CELL"
fi
readonly ARRAY_JOB_ID SUBMITTED_UTC PREVIOUS_LEDGER_SHA

if [[ "$RESUME" != true ]]; then
  export BARN_LEDGER_PATH="$LEDGER"
  export BARN_LEDGER_CAMPAIGN_ID="$CAMPAIGN_ID"
  export BARN_LEDGER_CELL="$CAMPAIGN_CELL"
  export BARN_LEDGER_ATTEMPT_ID="$ATTEMPT_ID"
  export BARN_LEDGER_SUBMITTED_UTC="$SUBMITTED_UTC"
  export BARN_LEDGER_ARRAY_JOB_ID="$ARRAY_JOB_ID"
  export BARN_LEDGER_SCRATCH="$SCRATCH"
  export BARN_LEDGER_MANIFEST_SHA="$MANIFEST_SHA"
  export BARN_LEDGER_SPLIT_SHA="$SPLIT_SHA"
  export BARN_LEDGER_PREREG_SHA="$PREREG_SHA"
  export BARN_LEDGER_ANALYZER_SHA="$ANALYZER_SHA"
  export BARN_LEDGER_PROTOCOL_SHA="$PROTOCOL_SHA"
  export BARN_LEDGER_CONTAINER_SHA="$CONTAINER_SHA"
  export BARN_LEDGER_SOURCE_SHA="$SOURCE_SHA"
  if ! proposal_output=$(python3 - "$PENDING_TX" "$PREVIOUS_LEDGER_SHA" <<'PY'
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

path = Path(os.environ["BARN_LEDGER_PATH"])
marker_path = Path(sys.argv[1])
previous_sha = sys.argv[2]
campaign = os.environ["BARN_LEDGER_CAMPAIGN_ID"]
cell = os.environ["BARN_LEDGER_CELL"]
attempt = os.environ["BARN_LEDGER_ATTEMPT_ID"]
submitted = os.environ["BARN_LEDGER_SUBMITTED_UTC"]
array_job = os.environ["BARN_LEDGER_ARRAY_JOB_ID"]
scratch = os.environ["BARN_LEDGER_SCRATCH"]
datetime.fromisoformat(submitted.replace("Z", "+00:00"))
if path.exists():
    ledger = json.loads(path.read_text(encoding="utf-8"))
else:
    ledger = {"schema_version": 1, "submissions": []}
expected_hashes = {
    "manifest_sha256": os.environ["BARN_LEDGER_MANIFEST_SHA"],
    "split_sha256": os.environ["BARN_LEDGER_SPLIT_SHA"],
    "prereg_sha256": os.environ["BARN_LEDGER_PREREG_SHA"],
    "analyzer_sha256": os.environ["BARN_LEDGER_ANALYZER_SHA"],
    "protocol_sha256": os.environ["BARN_LEDGER_PROTOCOL_SHA"],
    "container_sha256": os.environ["BARN_LEDGER_CONTAINER_SHA"],
    "source_sha256": os.environ["BARN_LEDGER_SOURCE_SHA"],
}
for seed in range(1, 6):
    ledger["submissions"].append({
        "campaign_id": campaign,
        "campaign_cell": cell,
        "attempt_id": attempt,
        "seed": seed,
        "submitted_utc": submitted,
        "slurm_job_id": None,
        "slurm_array_job_id": array_job,
        "slurm_array_task_id": seed,
        "artifact_path": (
            f"{scratch}/maxrl/barn/campaigns/{campaign}/cells/{cell}/"
            f"attempts/{attempt}/seed-{seed}/results/seed-{seed}.json"),
        "artifact_complete": False,
        "artifact_sha256": None,
        "expected_hashes": expected_hashes,
    })
payload = (json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False)
           + "\n").encode()
digest = hashlib.sha256(payload).hexdigest()
proposed = path.parent / f".{campaign}.pending-ledger-{digest}.json"
with proposed.open("xb") as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
marker = {
    "schema_version": 1,
    "campaign_id": campaign,
    "campaign_cell": cell,
    "attempt_id": attempt,
    "submitted_utc": submitted,
    "slurm_array_job_id": array_job,
    "previous_ledger_sha256": previous_sha,
    "proposed_ledger_sha256": digest,
    "proposed_ledger_name": proposed.name,
    "source_sha256": os.environ["BARN_LEDGER_SOURCE_SHA"],
}
with marker_path.open("x", encoding="utf-8") as handle:
    json.dump(marker, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
with temporary.open("xb") as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
directory = os.open(path.parent, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
print(digest)
print(proposed)
PY
  ); then
    echo "could not durably record pending held-array transaction" >&2
    exit 2
  fi
  mapfile -t proposal_fields <<< "$proposal_output"
  (( ${#proposal_fields[@]} == 2 )) || {
    echo "pending ledger proposal metadata mismatch" >&2; exit 2;
  }
  LEDGER_SHA=${proposal_fields[0]}
  PROPOSED_LEDGER=${proposal_fields[1]}
fi
readonly LEDGER_SHA PROPOSED_LEDGER
readonly REMOTE_LEDGER_STAGE="$REMOTE_CAMPAIGN_ROOT/.SUBMISSION_LEDGER.stage-$ARRAY_JOB_ID-$LEDGER_SHA"
if ! ledger_stage_state=$(ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- \
  "$REMOTE_CAMPAIGN_ROOT" "$REMOTE_LEDGER_STAGE" "$LEDGER_SHA" <<'REMOTE'
set -euo pipefail
mkdir -p -- "$1"
[[ -d "$1" && ! -L "$1" && ! -L "$2" ]] || exit 2
if [[ -e "$2" ]]; then
  [[ -f "$2" ]] || exit 2
  if [[ "$(sha256sum -- "$2" | awk '{print $1}')" == "$3" ]]; then
    printf 'PRESENT\n'
  else
    # A transport interruption may leave a partial upload.  This path is a
    # transaction-owned staging file, never the canonical ledger; discard the
    # partial bytes and retry from the durable local proposal.
    rm -f -- "$2"
    printf 'MISSING\n'
  fi
else
  printf 'MISSING\n'
fi
REMOTE
); then
  echo "remote ledger staging preflight failed; array remains held" >&2
  exit 2
fi
case "$ledger_stage_state" in
  MISSING)
    scp "${SSH_OPTS[@]}" -p -- "$PROPOSED_LEDGER" \
      "$HOST:$REMOTE_LEDGER_STAGE"
    ;;
  PRESENT) ;;
  *)
    echo "remote ledger staging state mismatch; array remains held" >&2
    exit 2
    ;;
esac

if ! install_output=$(ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- \
  "$REMOTE_LEDGER_STAGE" "$REMOTE_LEDGER" "$PREVIOUS_LEDGER_SHA" \
  "$LEDGER_SHA" <<'REMOTE'
set -euo pipefail
stage=$1
target=$2
previous=$3
expected=$4
lock="$target.lock"
[[ -f "$stage" && ! -L "$stage" \
   && "$(sha256sum -- "$stage" | awk '{print $1}')" == "$expected" ]] \
  || exit 2
mkdir "$lock"
release_lock() { rmdir "$lock"; }
trap release_lock EXIT HUP INT TERM
if [[ -f "$target" && ! -L "$target" \
   && "$(sha256sum -- "$target" | awk '{print $1}')" == "$expected" ]]; then
  rm -f -- "$stage"
else
  if [[ "$previous" == ABSENT ]]; then
    [[ ! -e "$target" && ! -L "$target" ]] || exit 2
  else
    [[ -f "$target" && ! -L "$target" \
       && "$(sha256sum -- "$target" | awk '{print $1}')" == "$previous" ]] \
      || exit 2
  fi
  mv -f -- "$stage" "$target"
fi
[[ -f "$target" && ! -L "$target" \
   && "$(sha256sum -- "$target" | awk '{print $1}')" == "$expected" ]] \
  || exit 2
printf 'LEDGER_INSTALLED_SHA256=%s\n' "$expected"
REMOTE
); then
  echo "ledger upload/install verification failed; array remains held" >&2
  exit 2
fi
[[ "$install_output" == "LEDGER_INSTALLED_SHA256=$LEDGER_SHA" ]] || {
  echo "ledger install acknowledgement mismatch; array remains held" >&2
  exit 2
}
[[ -f "$LEDGER" && ! -L "$LEDGER" \
   && "$(sha256sum -- "$LEDGER" | awk '{print $1}')" == "$LEDGER_SHA" ]] || {
  echo "durable local ledger differs before held-array release" >&2
  exit 2
}

ssh "${SSH_OPTS[@]}" "$HOST" scontrol release "$ARRAY_JOB_ID"
released=true
rm -f -- "$PENDING_TX" "$PROPOSED_LEDGER"
if ! rmdir -- "$LOCAL_LOCK"; then
  echo "released array but failed to remove local campaign-ledger lock" >&2
  exit 2
fi
lock_held=false
trap - EXIT
printf 'BARN_ARRAY_RELEASED\tjob_id=%s\tcampaign=%s\tcell=%s\tattempt=%s\tledger_sha256=%s\n' \
  "$ARRAY_JOB_ID" "$CAMPAIGN_ID" "$CAMPAIGN_CELL" "$ATTEMPT_ID" \
  "$LEDGER_SHA"
