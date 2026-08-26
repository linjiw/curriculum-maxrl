#!/usr/bin/env bash
# Outcome-blind terminal gate and single-use closure for GROUP-LAW-FLIP v1.
#
# This post-freeze operator script is not an evidence source. It may only call
# the frozen retrieval validator and analyzer after Slurm and marker-only
# preflights prove that all 48 evidence allocations completed successfully.
set -euo pipefail
umask 077

MODE=close
if (( $# > 1 )); then
  printf 'usage: %s [--preflight-only]\n' "$0" >&2
  exit 2
elif (( $# == 1 )); then
  [[ "$1" == --preflight-only ]] || {
    printf 'usage: %s [--preflight-only]\n' "$0" >&2
    exit 2
  }
  MODE=preflight
fi

HOST=${HOPPER_HOST:-lwang44@hopper.orc.gmu.edu}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${GLF_LOCAL_PYTHON:-/home/robotixx/miniconda3/envs/agenticrl/bin/python}
LOCAL_ATTEMPT=${GLF_LOCAL_ATTEMPT:-/data/robotixx/group_law_flip/group-law-flip-v1-20260820-001/attempt-001}
OUTPUT=${GLF_ANALYSIS_OUTPUT:-$ROOT/curriculum_maxrl/group_law_flip/GROUP_LAW_FLIP_ANALYSIS.json}
ANALYZER=$ROOT/curriculum_maxrl/group_law_flip/analyze_group_law_flip.py
EXPECTED_ANALYZER_SHA=9d88d6d5e63110eb1f1fa292f88a4d6c8b33d4878ef6fbf803c3bf5272186603

die() {
  printf 'group-law-flip closure: %s\n' "$*" >&2
  exit 2
}

[[ "$HOST" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+$ ]] \
  || die "unsafe HOPPER_HOST: $HOST"
[[ -x "$PY" ]] || die "local Python is not executable: $PY"
[[ -f "$ANALYZER" && ! -L "$ANALYZER" ]] || die "frozen analyzer missing or symlinked"
[[ "$(sha256sum "$ANALYZER" | awk '{print $1}')" == "$EXPECTED_ANALYZER_SHA" ]] \
  || die "local analyzer differs from the frozen hash"
[[ ! -e "$LOCAL_ATTEMPT" && ! -L "$LOCAL_ATTEMPT" ]] \
  || die "canonical local campaign already exists: $LOCAL_ATTEMPT"
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] \
  || die "single-use analysis output already exists: $OUTPUT"

terminal_preflight=$(ssh -o BatchMode=yes -o ConnectTimeout=15 \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 "$HOST" bash -s <<'REMOTE'
set -euo pipefail
attempt=/scratch/lwang44/maxrl/group_law_flip/campaigns/group-law-flip-v1-20260820-001/attempts/attempt-001
incomplete=/scratch/lwang44/maxrl/group_law_flip/campaigns/group-law-flip-v1-20260820-001/incomplete/attempt-001

live=$(squeue -h -n group-law-flip-v1 -o '%i|%T' | sed '/^$/d' | wc -l)
mapfile -t allocations < <(
  sacct -S 2026-08-20 -X -n -P -o JobName,State,ExitCode,AllocCPUS \
    | awk -F'|' '$1 == "group-law-flip-v1" && $4 == 8 {print}'
)
completed=0
failed=0
nonterminal=0
for row in "${allocations[@]}"; do
  IFS='|' read -r name state exit_code cpus extra <<< "$row"
  [[ -z "${extra:-}" && "$name" == group-law-flip-v1 && "$cpus" == 8 ]]
  if [[ "$state" == COMPLETED && "$exit_code" == 0:0 ]]; then
    ((completed += 1))
  elif [[ "$state" =~ ^(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT)$ ]]; then
    ((failed += 1))
  else
    ((nonterminal += 1))
  fi
done

helper=$(sacct -j 9420095 -X -n -P -o State,ExitCode \
  | sed -e 's/|$//' -e '/^$/d')
final=0
complete_markers=0
manifests=0
arm_receipts=0
invalid=0
quarantines=0
if [[ -d "$attempt" && ! -L "$attempt" ]]; then
  final=$(find "$attempt" -mindepth 1 -maxdepth 1 -type d -name 'seed-*' | wc -l)
  complete_markers=$(find "$attempt" -mindepth 2 -maxdepth 2 -type f -name COMPLETE | wc -l)
  manifests=$(find "$attempt" -mindepth 2 -maxdepth 2 -type f -name SHA256SUMS | wc -l)
  arm_receipts=$(find "$attempt" -mindepth 3 -maxdepth 3 -type f \
    -path '*/meta/*.DONE.json' | wc -l)
  invalid=$(find "$attempt" -mindepth 1 -maxdepth 1 -type d -name 'seed-*' \
    \( ! -exec test -f '{}/COMPLETE' \; -o ! -exec test -f '{}/SHA256SUMS' \; \) \
    | wc -l)
fi
if [[ -d "$incomplete" && ! -L "$incomplete" ]]; then
  quarantines=$(find "$incomplete" -mindepth 1 -maxdepth 1 -type d \
    -name 'seed-*.job-*' | wc -l)
fi

printf 'live=%s completed_allocations=%s failed_allocations=%s nonterminal_allocations=%s helper=%s final=%s complete=%s manifests=%s arm_receipts=%s invalid=%s quarantines=%s\n' \
  "$live" "$completed" "$failed" "$nonterminal" "$helper" "$final" \
  "$complete_markers" "$manifests" "$arm_receipts" "$invalid" "$quarantines"

(( live == 0 && completed == 48 && failed == 0 && nonterminal == 0 ))
[[ "$helper" == 'COMPLETED|0:0' ]]
(( final == 48 && complete_markers == 48 && manifests == 48 \
   && arm_receipts == 96 && invalid == 0 && quarantines == 0 ))
printf 'REMOTE_TERMINAL_AND_MARKER_GATE_PASS\n'
REMOTE
) || {
  printf '%s\n' "$terminal_preflight" >&2
  die "campaign is not ready; retrieval and analysis were not started"
}
printf '%s\n' "$terminal_preflight"

if [[ "$MODE" == preflight ]]; then
  printf 'GROUP_LAW_FLIP_PREFLIGHT_ONLY_PASS\n'
  exit 0
fi

bash "$ROOT/hopper/fetch_group_law_flip.sh"

[[ -d "$LOCAL_ATTEMPT" && ! -L "$LOCAL_ATTEMPT" ]] \
  || die "frozen retrieval did not publish the canonical local campaign"
PYTHONPATH="$ROOT" "$PY" -m \
  curriculum_maxrl.group_law_flip.analyze_group_law_flip \
  "$LOCAL_ATTEMPT" --output "$OUTPUT"

"$PY" - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
assert payload["schema"] == "curriculum-maxrl/group-law-flip/analysis/v1"
assert payload["protocol"] == "group_law_flip_v1"
assert payload["campaign"] == "group-law-flip-v1-20260820-001"
assert payload["complete"] is True
assert len(payload["seed_blocks"]) == 48
decision = payload["primary_grouplaw_minus_plugin"]["decision"]
assert decision in {
    "supported",
    "practically_ruled_out",
    "inconclusive",
    "treatment_not_delivered",
}
print(f"GROUP_LAW_FLIP_SINGLE_USE_CLOSURE_PASS decision={decision} output={path}")
PY
