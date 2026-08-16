#!/usr/bin/env bash
# DEVELOPMENT sweep: coefficient activity incorporated into each replay-based
# minimax method, against that method's own upstream baseline.
#
# NOT EVIDENCE. Development seeds 1001-1005, reduced update budget. Its only
# purpose is to choose which method x configuration to carry into a separately
# preregistered confirmatory campaign.
#
# Scope note. The idea is hosted in the replay-based family (PLR, robust PLR,
# ACCEL) because that is where minimax's strongest methods live and where a
# level score is the natural intervention point. PAIRED is EXCLUDED: upstream
# PAIRED sizes its teacher rollout to n_parallel while the student batch is
# n_parallel x n_eval, so it fails closed for any n_eval > 1 -- verified with
# the UNMODIFIED relative_regret baseline, so this is an upstream structural
# limit, not our integration. With n_eval forced to 1 a success-rate score sees
# a single Bernoulli per designed level, which degenerates to the existing
# minimax-adversary objective and would not be a new method.
#
# Every baseline uses the upstream shipped config verbatim; our arms change the
# score only. The 4x8 arms hold TOTAL rollout budget at 32 streams per update
# and trade level diversity for per-level fidelity, so score families are
# compared at equal evaluation cost.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-/data/robotixx/ued_bench/methods-$(date +%Y%m%d-%H%M)}
UPDATES=${2:-5000}
CONC=${3:-3}
SEEDS=${4:-"1001 1002 1003 1004 1005"}
mkdir -p "$OUT"

export MINIMAX_SRC=${MINIMAX_SRC:-/data/robotixx/ued_bench/src/minimax-frontier-v4-allmethods-d053054}

MATCHED="n_parallel=4 n_eval=8 plr_buffer_size=500"
CA="ued_score=coefficient_activity plr_frontier_n_rollouts=8"
COMMON="test_interval=1000"

emit () {  # base_config arm_name extra...
  local base=$1 name=$2; shift 2
  for seed in $SEEDS; do
    printf '%s\t%s\t%s\t%s\n' "$base" "$name" "$seed" "$* $COMMON"
  done
}

{
  emit dr    drControl   ""
  emit plr   plrMM32     ""
  emit plr   plrMM4x8    "$MATCHED"
  emit plr   plrCA8      "$MATCHED $CA"
  emit accel accelMM32   ""
  emit accel accelMM4x8  "$MATCHED"
  emit accel accelCA8    "$MATCHED $CA"
} > "$OUT/jobs.tsv"

echo "sweep dir : $OUT"
echo "clone     : $MINIMAX_SRC"
echo "runs      : $(wc -l < "$OUT/jobs.tsv") x ${UPDATES} updates, ${CONC} concurrent"

launch () {
  local base=$1 name=$2 seed=$3 extra=$4
  # shellcheck disable=SC2086
  if bash "$ROOT/ued_benchmark/scripts/run_arm.sh" "$base" "$name" "$OUT" "$UPDATES" \
       "$seed" $extra >> "$OUT/driver.log" 2>&1; then
    echo "OK     $name seed=$seed" >> "$OUT/driver.log"
  else
    echo "FAILED $name seed=$seed" >> "$OUT/driver.log"
  fi
}

running=0
while IFS=$'\t' read -r base name seed extra; do
  [[ -n "$name" ]] || continue
  launch "$base" "$name" "$seed" "$extra" &
  running=$((running + 1))
  if (( running >= CONC )); then wait -n; running=$((running - 1)); fi
done < "$OUT/jobs.tsv"
wait

echo "done: $(date -Is)"
echo "completed: $(grep -c '^OK' "$OUT/driver.log" 2>/dev/null || echo 0)"
grep '^FAILED' "$OUT/driver.log" 2>/dev/null || echo "no failures"
