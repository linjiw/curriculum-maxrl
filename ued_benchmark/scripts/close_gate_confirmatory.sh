#!/usr/bin/env bash
# Outcome-blind closure for the terminal AMaze gate confirmatory rerun.
#
# The preflight reads only completion receipts, declared configuration, file
# presence, and checkpoint n_updates. Evaluation values remain sealed until
# every conjunct passes and the pinned frozen analyzer is invoked once.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CAMPAIGN=/data/robotixx/ued_bench/gate-confirmatory-20260819
OUTPUT="$ROOT/ued_benchmark/AMAZE_GATE_ANALYSIS.json"
PREFLIGHT_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    --campaign=*) CAMPAIGN=${arg#*=} ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

MINIMAX_PY=${MINIMAX_PY:-/data/robotixx/ued_bench/envs/jax062-cuda129-probe/bin/python}
MINIMAX_SRC=${MINIMAX_SRC:-/data/robotixx/ued_bench/src/minimax-frontier-v6-gated-d053054}
HOST_PY=${HOST_PY:-/home/robotixx/miniconda3/envs/agenticrl/bin/python}
ANALYZER="$ROOT/ued_benchmark/scripts/analyze_gate_confirmatory.py"
VERIFIER="$ROOT/ued_benchmark/scripts/verify_checkpoint_budget.py"
PREREG="$ROOT/ued_benchmark/AMAZE_GATE_PREREG.md"

EXPECTED_ANALYZER_SHA256=aaf54f229795987c96310f3b48362be986f9c4966045806ada9c5b414151d755
EXPECTED_VERIFIER_SHA256=9eab521e1eda3fdd17e4492ecee5e1bfdc535a471572d046ceec9ce898b6308c
EXPECTED_PREREG_SHA256=50967735878967c6e8ce223762b90dbff6a9e19fcd3e3b039571a8d21063960f

[[ -d "$CAMPAIGN" && ! -L "$CAMPAIGN" ]] || {
  echo "invalid campaign directory: $CAMPAIGN" >&2; exit 1; }
[[ -x "$MINIMAX_PY" && -x "$HOST_PY" ]] || {
  echo "required Python interpreter is unavailable" >&2; exit 1; }
[[ -d "$MINIMAX_SRC/src/minimax" && ! -L "$MINIMAX_SRC" ]] || {
  echo "frozen minimax source tree is unavailable: $MINIMAX_SRC" >&2; exit 1; }
[[ ! -e "$OUTPUT" ]] || {
  echo "refusing single-use analysis: $OUTPUT already exists" >&2; exit 1; }
[[ ! -e "$CAMPAIGN/ckpt_budget.json" ]] || {
  echo "refusing closure: canonical ckpt_budget.json already exists" >&2; exit 1; }

check_hash() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  [[ "$actual" == "$expected" ]] || {
    echo "hash mismatch: $path" >&2; exit 1; }
}
check_hash "$ANALYZER" "$EXPECTED_ANALYZER_SHA256"
check_hash "$VERIFIER" "$EXPECTED_VERIFIER_SHA256"
check_hash "$PREREG" "$EXPECTED_PREREG_SHA256"

# Validate only the outcome-blind campaign surface. In particular, do not open
# any eval CSV here; the frozen analyzer owns the first value-bearing read.
"$HOST_PY" - "$CAMPAIGN" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
expected = {
    f"arm-{arm}-s{seed}-u30000"
    for arm in ("plrMM", "plrGate")
    for seed in range(2001, 2011)
}

jobs = root / "jobs.tsv"
driver = root / "driver.log"
if not jobs.is_file() or not driver.is_file():
    raise SystemExit("missing jobs.tsv or driver.log")

expected_jobs = []
gate = (
    "ued_score=coefficient_activity plr_frontier_mode=gate "
    "plr_frontier_n_rollouts=8 "
    "plr_frontier_require_n_eval_match=False"
)
for seed in range(2001, 2011):
    expected_jobs.append(f"plr\tplrMM\t{seed}\t")
    expected_jobs.append(f"plr\tplrGate\t{seed}\t{gate}")
if jobs.read_text().splitlines() != expected_jobs:
    raise SystemExit("jobs.tsv does not match the frozen 2x10 matrix")

ok = set()
eval_ok = set()
failed = []
for line in driver.read_text().splitlines():
    m = re.fullmatch(r"OK\s+(arm-(?:plrMM|plrGate)-s\d+-u30000)", line)
    if m:
        ok.add(m.group(1))
    m = re.fullmatch(r"EVAL-OK\s+(arm-(?:plrMM|plrGate)-s\d+-u30000)", line)
    if m:
        eval_ok.add(m.group(1))
    if re.match(r"(?:FAILED|EVAL-FAILED)\s+", line):
        failed.append(line)
if failed:
    raise SystemExit(f"failure receipts present: {failed}")
if ok != expected or eval_ok != expected:
    raise SystemExit(
        f"receipt mismatch: train={len(ok)}/20 eval={len(eval_ok)}/20")

for xpid in sorted(expected):
    cell = root / xpid
    for name in ("checkpoint.pkl", "meta.json", "logs.csv"):
        path = cell / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise SystemExit(f"missing/invalid {path}")
    evaluation = root / "eval" / f"{xpid}.csv"
    if (not evaluation.is_file() or evaluation.is_symlink()
            or evaluation.stat().st_size == 0):
        raise SystemExit(f"missing/invalid evaluation artifact: {evaluation}")
    if (cell / "DONE").exists():
        raise SystemExit(f"unexpected pre-closure DONE marker: {cell / 'DONE'}")
print("campaign surface: 20/20 train receipts, 20/20 eval receipts, 0 failures")
print("artifact surface: 20 checkpoints, 20 meta files, 20 logs, 20 sealed eval CSVs")
PY

TMP_BUDGET=$(mktemp "$CAMPAIGN/.ckpt_budget.json.tmp.XXXXXX")
cleanup() {
  [[ -n "${TMP_BUDGET:-}" && -e "$TMP_BUDGET" ]] && rm -f -- "$TMP_BUDGET"
}
trap cleanup EXIT

export PYTHONPATH="$MINIMAX_SRC/src${PYTHONPATH:+:$PYTHONPATH}"
"$MINIMAX_PY" "$VERIFIER" "$CAMPAIGN" --output "$TMP_BUDGET"
"$HOST_PY" - "$TMP_BUDGET" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
budget = json.loads(path.read_text())
expected = {
    f"arm-{arm}-s{seed}-u30000"
    for arm in ("plrMM", "plrGate")
    for seed in range(2001, 2011)
}
if set(budget) != expected:
    raise SystemExit(f"checkpoint-budget matrix is {len(budget)}/20")
values = {key: int(value) for key, value in budget.items()}
bad = {key: value for key, value in values.items()
       if value < 29_900 or value > 30_000}
if bad:
    raise SystemExit(f"checkpoint-budget gate failed: {bad}")
print(f"checkpoint-budget gate: 20/20 in [29900,30000]; "
      f"range [{min(values.values())},{max(values.values())}]")
PY

if (( PREFLIGHT_ONLY )); then
  echo "PREFLIGHT PASSED: outcomes remain sealed; no canonical files or markers written"
  exit 0
fi

chmod 0444 "$TMP_BUDGET"
mv "$TMP_BUDGET" "$CAMPAIGN/ckpt_budget.json"
TMP_BUDGET=

# Backfill explicit completion markers only after the full checkpoint-budget
# matrix passes. Future driver restarts now skip these terminal cells safely.
for arm in plrMM plrGate; do
  for seed in $(seq 2001 2010); do
    touch "$CAMPAIGN/arm-${arm}-s${seed}-u30000/DONE"
  done
done

"$MINIMAX_PY" "$ANALYZER" "$CAMPAIGN" --output "$OUTPUT"

"$HOST_PY" - "$OUTPUT" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
result = json.loads(path.read_text())
if result.get("schema") != "curriculum-maxrl/amaze-gate-confirmatory-analysis/v1":
    raise SystemExit("unexpected analysis schema")
if result.get("verdict") not in {
    "gate_beats_upstream", "gate_does_not_beat_upstream", "inconclusive_at_n10"
}:
    raise SystemExit("unexpected frozen verdict")
primary = result.get("primary_mean_solved_rate", {})
if primary.get("n") != 10 or len(primary.get("paired_differences", [])) != 10:
    raise SystemExit("analysis primary is not the frozen ten-pair estimand")
print(f"analysis schema/decision assertions passed: {result['verdict']}")
PY

[[ $(find "$CAMPAIGN" -mindepth 2 -maxdepth 2 -type f -name DONE | wc -l) -eq 20 ]] || {
  echo "post-analysis marker count is not 20" >&2; exit 1; }
sha256sum "$CAMPAIGN/ckpt_budget.json" "$OUTPUT"
