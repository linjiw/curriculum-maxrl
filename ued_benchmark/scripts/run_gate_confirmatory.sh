#!/usr/bin/env bash
# Confirmatory campaign for AMAZE_GATE_PREREG.md.
#   train:  2 arms x seeds 2001-2010 at the full shipped 30,000 updates
#   eval:   shipped minimax.evaluate on every final checkpoint, 100 eps/maze
# The analyzer is run separately, exactly once.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-/data/robotixx/ued_bench/gate-confirmatory-$(date +%Y%m%d)}
CONC=${2:-3}
export MINIMAX_SRC=${MINIMAX_SRC:-/data/robotixx/ued_bench/src/minimax-frontier-v6-gated-d053054}
PY=${MINIMAX_PY:-/data/robotixx/ued_bench/envs/jax062-cuda129-probe/bin/python}
mkdir -p "$OUT/eval"

GATE="ued_score=coefficient_activity plr_frontier_mode=gate plr_frontier_n_rollouts=8 plr_frontier_require_n_eval_match=False"

if [[ ! -f "$OUT/jobs.tsv" ]]; then
  for seed in $(seq 2001 2010); do
    printf 'plr\tplrMM\t%s\t\n' "$seed"
    printf 'plr\tplrGate\t%s\t%s\n' "$seed" "$GATE"
  done > "$OUT/jobs.tsv"
fi
echo "campaign : $OUT"
echo "runs     : $(wc -l < "$OUT/jobs.tsv") x 30000 updates, $CONC concurrent"

# ---- train (idempotent: skip only explicitly completed xpids) ---------------
launch () {
  local base=$1 name=$2 seed=$3 extra=$4
  local xpid="arm-${name}-s${seed}-u30000"
  if [[ -f "$OUT/$xpid/DONE" ]]; then
    echo "SKIP   $xpid (completed)" >> "$OUT/driver.log"; return
  fi
  # shellcheck disable=SC2086
  if bash "$ROOT/ued_benchmark/scripts/run_arm.sh" "$base" "$name" "$OUT" 30000 \
       "$seed" $extra >> "$OUT/driver.log" 2>&1; then
    touch "$OUT/$xpid/DONE"
    echo "OK     $xpid" >> "$OUT/driver.log"
  else
    echo "FAILED $xpid" >> "$OUT/driver.log"
  fi
}
running=0
while IFS=$'\t' read -r base name seed extra; do
  [[ -n "$name" ]] || continue
  launch "$base" "$name" "$seed" "$extra" &
  running=$((running+1))
  if (( running >= CONC )); then wait -n; running=$((running-1)); fi
done < "$OUT/jobs.tsv"
wait
echo "train done: $(date -Is)"; grep -c '^OK\|^SKIP' "$OUT/driver.log" || true
grep '^FAILED' "$OUT/driver.log" || echo "no training failures"

# ---- evaluate every final checkpoint with the shipped evaluator -------------
cd "$MINIMAX_SRC"
export PYTHONPATH="$MINIMAX_SRC/src" XLA_PYTHON_CLIENT_PREALLOCATE=false
for d in "$OUT"/arm-*-u30000/; do
  xpid=$(basename "$d")
  [[ -f "$d/checkpoint.pkl" ]] || { echo "EVAL-SKIP $xpid no checkpoint" >> "$OUT/driver.log"; continue; }
  [[ -f "$OUT/eval/$xpid.csv" ]] && continue
  "$PY" -m minimax.evaluate --seed 1 --log_dir "$OUT" --xpid "$xpid" \
      --env_names Maze-SixteenRooms,Maze-Labyrinth,Maze-StandardMaze \
      --n_episodes 100 --results_path "$OUT/eval" --results_fname "$xpid" \
      > "$OUT/eval/$xpid.log" 2>&1 \
    && echo "EVAL-OK $xpid" >> "$OUT/driver.log" \
    || echo "EVAL-FAILED $xpid" >> "$OUT/driver.log"
done
echo "eval done: $(date -Is)"; ls "$OUT/eval"/*.csv 2>/dev/null | wc -l
echo "next: python ued_benchmark/scripts/analyze_gate_confirmatory.py $OUT --output ued_benchmark/AMAZE_GATE_ANALYSIS.json"
