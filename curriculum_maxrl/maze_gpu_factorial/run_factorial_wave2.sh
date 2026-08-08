#!/bin/bash
# Confirmation factorial, wave 2 — pre-registered 2026-08-05 BEFORE any
# run, on FRESH seed blocks 6-11.
#
# Motivation: wave 1 (run_factorial.sh) registered the ENDPOINT delta
# cov8 and FAILED P-F1 (3/6, 4/6) -> claim dropped per the committed
# branch. Two patterns survived wave 1 as EXPLORATORY findings only:
#   (a) time-integrated coverage: MaxRL - GRPO cov_auc_delta positive
#       in 12/12 paired blocks across both samplers;
#   (b) easy-band (L1-3) endpoint asymmetry, 9/12.
# Exploratory findings cannot be claimed from the data that generated
# them. This wave tests (a) as the REGISTERED primary on new blocks.
#
#   Design: {maxrl, grpo} x {uniform, frontier_un} x seeds 6-11.
#   Identical protocol to wave 1: shared per-block SFT warmstart,
#   250 fixed steps, eval every 25, lr 1e-4, 16-maze/level held-out
#   set, Chen-unbiased pass@8.
#
#   PRIMARY (P-F2, registered): paired (same block, same sampler)
#   MaxRL - GRPO on cov_auc_delta (mean in-training cov8 over evals
#   steps 25..250 minus post-SFT init; computed by fact_analyze.py,
#   same code path as wave 1) positive in >= 5/6 blocks under BOTH
#   samplers. 6/6 across a sampler: exact two-sided sign p = 0.031.
#   Falsification branch (committed): <= 4/6 under either sampler ->
#   the time-integrated coverage ordering is ALSO dropped; the paper's
#   estimator-coverage story at neural scale reduces to the easy-band
#   decomposition as descriptive statistics, with no cross-estimator
#   coverage claim of any kind.
#
#   SECONDARY (P-F3, registered, one test): paired easy-band (L1-3)
#   endpoint delta: GRPO loses more easy-band cov8 than MaxRL in a
#   majority of the 12 new pairs (sign test at 9/12: p = 0.15;
#   >= 10/12: p = 0.039). Falsification: < 7/12 -> easy-band
#   asymmetry demoted to wave-1-only description.
#
#   Everything else is exploratory and will be labeled so.
#
# Runs at nice -n 10 with concurrency 1 (verl reviewer arms own the
# GPU; maze cells are ~1-2GB and tolerate sharing; step budgets make
# contention harmless to the contrasts).
cd "$(dirname "$0")"
exec 9>/tmp/maze_factorial_w2.lock
flock -n 9 || { echo "wave-2 driver already running"; exit 0; }

STEPS=250
EVERY=25
LR=1e-4

run_cell() { # teacher estimator seed
  local out="fact250_${1}_${2}_s${3}.jsonl"
  if [ -f "$out" ] && grep -q '"final"' "$out"; then
    echo "skip $out (complete)"; return
  fi
  echo "=== w2 fact250 $1 $2 seed $3 ($(date -u +%H:%M:%SZ)) ==="
  nice -n 10 python3 train.py --teacher "$1" --estimator "$2" \
    --steps $STEPS --eval-every $EVERY --seed "$3" --lr $LR \
    --out "$out"
}

# NOTE: train.py auto-generates seed{s}_sft_warmstart.pt when missing;
# concurrency is 1, so the first cell of each block creates it with no
# race and the remaining three cells load it.

for s in 6 7 8 9 10 11; do
  for est in maxrl grpo; do
    for t in uniform frontier_un; do
      run_cell "$t" "$est" "$s"
    done
  done
done

echo "WAVE2 FACTORIAL DONE $(date -u)"
echo "run: python3 fact_analyze.py --prefix fact250 --seeds 6:12  (P-F2/P-F3 per header)"
