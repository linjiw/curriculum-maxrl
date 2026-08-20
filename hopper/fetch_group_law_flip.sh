#!/usr/bin/env bash
# Retrieve the complete blind campaign into its canonical immutable local path.
# This performs structural/hash validation only; it never computes an endpoint.
set -euo pipefail

HOST=${HOPPER_HOST:-lwang44@hopper.orc.gmu.edu}
REMOTE=${GLF_REMOTE_ATTEMPT:-/scratch/lwang44/maxrl/group_law_flip/campaigns/group-law-flip-v1-20260820-001/attempts/attempt-001}
LOCAL=${GLF_LOCAL_ATTEMPT:-/data/robotixx/group_law_flip/group-law-flip-v1-20260820-001/attempt-001}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${GLF_LOCAL_PYTHON:-/home/robotixx/miniconda3/envs/agenticrl/bin/python}
EXPECTED_ANALYZER_SHA=9d88d6d5e63110eb1f1fa292f88a4d6c8b33d4878ef6fbf803c3bf5272186603

[[ "$REMOTE" == /scratch/lwang44/maxrl/group_law_flip/* ]] \
  || { echo "unsafe remote attempt path" >&2; exit 2; }
[[ "$LOCAL" == /data/robotixx/group_law_flip/* ]] \
  || { echo "unsafe local attempt path" >&2; exit 2; }
[[ ! -e "$LOCAL" && ! -L "$LOCAL" ]] \
  || { echo "refusing to overwrite local campaign: $LOCAL" >&2; exit 2; }
[[ "$(sha256sum "$ROOT/curriculum_maxrl/group_law_flip/analyze_group_law_flip.py" | awk '{print $1}')" == "$EXPECTED_ANALYZER_SHA" ]] \
  || { echo "local analyzer differs from frozen hash" >&2; exit 2; }

ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "bash -s" -- "$REMOTE" <<'REMOTE_CHECK'
set -euo pipefail
remote=$1
[[ -d "$remote" ]]
count=$(find "$remote" -mindepth 1 -maxdepth 1 -type d -name 'seed-*' | wc -l)
[[ "$count" -eq 48 ]] || { echo "remote matrix incomplete: $count/48" >&2; exit 2; }
for seed in $(seq 3001 3048); do
  block="$remote/seed-$seed"
  [[ -f "$block/COMPLETE" && -f "$block/SHA256SUMS" ]]
  (cd "$block" && sha256sum -c --strict SHA256SUMS >/dev/null)
done
echo REMOTE_48_BLOCK_HASH_PREFLIGHT_PASS
REMOTE_CHECK

mkdir -p "$(dirname "$LOCAL")"
STAGE=$(mktemp -d "$(dirname "$LOCAL")/.attempt-001.stage.XXXXXX")
printf 'retrieval_stage=%s\n' "$STAGE"
rsync -a --protect-args \
  -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
  "$HOST:$REMOTE/" "$STAGE/"

PYTHONPATH="$ROOT" "$PY" - "$STAGE" <<'PY'
import sys
from curriculum_maxrl.group_law_flip.analyze_group_law_flip import validate_campaign

validated = validate_campaign(sys.argv[1])
assert len(validated.runs) == 96
assert len(validated.telemetry) == 96
print("LOCAL_48_BLOCK_STRUCTURAL_VALIDATION_PASS")
PY
mv -T "$STAGE" "$LOCAL"
find "$LOCAL" -mindepth 1 -maxdepth 1 -type d -name 'seed-*' \
  -exec chmod -R a-w {} +
printf 'canonical_local_campaign=%s\n' "$LOCAL"
