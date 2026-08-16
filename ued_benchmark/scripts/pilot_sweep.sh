#!/usr/bin/env bash
# DEVELOPMENT sweep for the coefficient-activity PLR score on minimax AMaze.
#
# NOT EVIDENCE. Development seeds 1001-1003, reduced update budget. Its only
# purpose is to select one configuration to carry into a separately
# preregistered confirmatory campaign. No number produced here may appear in
# the paper as a result.
#
# Design question it answers: at a FIXED total evaluation budget of 32 streams
# per update, does scoring levels by coefficient activity beat MaxMC regret,
# and which score exponent works best?
#
# Baselines use the upstream shipped `maze/plr` config verbatim. The
# group-matched baseline trades level diversity for per-level fidelity
# (4 levels x 8 eval streams) exactly as the frontier arms do, so the
# comparison holds total rollout budget constant.
#
# The exponent sweep is motivated by the 2026-08-15 Acrobot result: peak
# LOCATION did not matter and harder-peaked scores kept winning up to N~64,
# so N is treated here as a difficulty-target knob rather than something that
# must equal the deployed rollout count.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-/data/robotixx/ued_bench/pilot-$(date +%Y%m%d-%H%M)}
UPDATES=${2:-2500}
CONC=${3:-3}
mkdir -p "$OUT"

MATCHED="n_parallel=4 n_eval=8 plr_buffer_size=500"
FRONT="ued_score=coefficient_activity"

emit () {  # name, extra args...
  local name=$1; shift
  for seed in 1001 1002 1003; do
    printf '%s\t%s\t%s\n' "$name" "$seed" "$*"
  done
}

{
  emit maxmc32x1 "test_interval=500"
  emit maxmc4x8  "$MATCHED test_interval=500"
  emit frontN8   "$MATCHED $FRONT plr_frontier_n_rollouts=8 test_interval=500"
  emit frontN16  "$MATCHED $FRONT plr_frontier_n_rollouts=16 plr_frontier_require_n_eval_match=False test_interval=500"
  emit frontN32  "$MATCHED $FRONT plr_frontier_n_rollouts=32 plr_frontier_require_n_eval_match=False test_interval=500"
  emit frontN64  "$MATCHED $FRONT plr_frontier_n_rollouts=64 plr_frontier_require_n_eval_match=False test_interval=500"
} > "$OUT/jobs.tsv"

echo "pilot dir : $OUT"
echo "runs      : $(wc -l < "$OUT/jobs.tsv")  x ${UPDATES} updates, ${CONC} concurrent"

launch () {
  local name=$1 seed=$2 extra=$3
  # shellcheck disable=SC2086  # $extra is a deliberate word-split arg list
  if bash "$ROOT/ued_benchmark/scripts/run_arm.sh" plr "$name" "$OUT" "$UPDATES" \
       "$seed" $extra >> "$OUT/driver.log" 2>&1; then
    echo "OK     $name seed=$seed" >> "$OUT/driver.log"
  else
    echo "FAILED $name seed=$seed" >> "$OUT/driver.log"
  fi
}

# Bounded parallelism with a real tab IFS. The earlier xargs -I{} form split
# fields on the literal string \t and silently mangled every arm name.
running=0
while IFS=$'\t' read -r name seed extra; do
  [[ -n "$name" ]] || continue
  launch "$name" "$seed" "$extra" &
  running=$((running + 1))
  if (( running >= CONC )); then wait -n; running=$((running - 1)); fi
done < "$OUT/jobs.tsv"
wait

echo "done: $(date -Is)"
grep -c '^OK' "$OUT/driver.log" 2>/dev/null | sed 's/^/completed: /' || true
grep '^FAILED' "$OUT/driver.log" 2>/dev/null || echo "no failures"
