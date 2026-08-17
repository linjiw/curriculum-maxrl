#!/usr/bin/env bash
# Run a prebuilt jobs.tsv (base<TAB>arm<TAB>seed<TAB>extra) with bounded
# parallelism. Development use only.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:?jobs dir containing jobs.tsv}
UPDATES=${2:-5000}
CONC=${3:-3}
export MINIMAX_SRC=${MINIMAX_SRC:-/data/robotixx/ued_bench/src/minimax-frontier-v4-allmethods-d053054}

echo "dir    : $OUT"
echo "clone  : $MINIMAX_SRC"
echo "runs   : $(wc -l < "$OUT/jobs.tsv") x ${UPDATES} updates, ${CONC} concurrent"

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
