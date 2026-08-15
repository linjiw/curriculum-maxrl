#!/usr/bin/env bash
# Retrieve a completed Acrobot U64 campaign from Hopper and verify it.
#
# Fetches only result JSONs, refuses an existing destination, refuses any
# .partial file, and reports the arm x seed matrix so an incomplete campaign is
# obvious before the analyzer is ever run.
#
# usage: hopper/fetch_acrobot_u64.sh <array_job_id> [dest_dir]
set -euo pipefail

REMOTE=${HOPPER_REMOTE:-lwang44@hopper.orc.gmu.edu}
SCRATCH=${HOPPER_SCRATCH:-/scratch/lwang44}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

# A campaign may span several array submissions against the IDENTICAL bundle
# digest and lock (e.g. a one-task validation followed by the remainder). Each
# submission writes under its own array job id, so accept several and merge
# them into one matrix. The analyzer independently requires a single shared
# lock digest, so a merge across mismatched submissions still fails closed.
DEST_OVERRIDE=""
JOBS=()
for arg in "$@"; do
  if [[ "$arg" =~ ^[0-9]+$ ]]; then JOBS+=("$arg"); else DEST_OVERRIDE=$arg; fi
done
(( ${#JOBS[@]} >= 1 )) || { echo "usage: fetch_acrobot_u64.sh <job_id...> [dest]" >&2; exit 1; }
DEST=${DEST_OVERRIDE:-$ROOT/acrobot_u64/results/confirmatory-${JOBS[0]}}

if [[ -e "$DEST" ]]; then
  echo "destination already exists, refusing to overwrite: $DEST" >&2
  exit 1
fi

mkdir -p "$DEST"
REMOTE_N=0
for JOB in "${JOBS[@]}"; do
  SRC="$SCRATCH/acrobot_u64/results/$JOB"
  n=$(ssh "$REMOTE" "ls '$SRC'/*.json 2>/dev/null | wc -l" | tr -d '[:space:]')
  p=$(ssh "$REMOTE" "ls '$SRC'/*.partial 2>/dev/null | wc -l" | tr -d '[:space:]')
  echo "remote $JOB : $n json, $p partial"
  if [[ "$p" != "0" ]]; then
    echo "partial files present in $SRC; that submission is still writing" >&2
    exit 1
  fi
  (( n > 0 )) || { echo "no results under $SRC" >&2; exit 1; }
  scp -q "$REMOTE:$SRC/*.json" "$DEST/"
  REMOTE_N=$(( REMOTE_N + n ))
done
echo "remote total   : $REMOTE_N"

LOCAL_N=$(ls "$DEST"/*.json 2>/dev/null | wc -l | tr -d '[:space:]')
echo "fetched        : $LOCAL_N"
if [[ "$LOCAL_N" != "$REMOTE_N" ]]; then
  echo "fetch is short: $LOCAL_N of $REMOTE_N" >&2
  exit 1
fi

DIGEST=$(cd "$DEST" && find . -name '*.json' -print0 | LC_ALL=C sort -z \
  | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
echo "tree digest    : $DIGEST"

echo
echo "arm x seed matrix:"
python3 - "$DEST" <<'PY'
import json, pathlib, sys
from collections import defaultdict
d = pathlib.Path(sys.argv[1])
seen = defaultdict(set)
for p in sorted(d.glob("*.json")):
    r = json.loads(p.read_text())
    seen[r["arm"]].add(int(r["logical_seed"]))
arms = ["uniform_shared_h64", "p1mp_shared_h64", "u16_shared_h64", "u64_shared_h64"]
want = set(range(20000, 20020))
ok = True
for a in arms:
    got = seen.get(a, set())
    missing = sorted(want - got)
    print(f"  {a:22s} {len(got):2d}/20" + (f"  MISSING {missing}" if missing else ""))
    ok &= not missing
extra = set(seen) - set(arms)
if extra:
    print(f"  UNEXPECTED ARMS: {sorted(extra)}")
    ok = False
print("\nCOMPLETE" if ok else "\nINCOMPLETE -- do not analyze")
sys.exit(0 if ok else 1)
PY

echo
echo "next: uv run --python 3.12.13 --with numpy==2.5.1 python \\"
echo "        acrobot_u64/analyze_u64_tournament.py $DEST \\"
echo "        --output acrobot_u64/ACROBOT_U64_ANALYSIS.json"
