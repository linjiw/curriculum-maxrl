#!/bin/bash
# Balanced maze factorial — the decisive experiment the 2026-08-04 draft
# review (P0-3) and every prior review asked for.
#
# Pre-registered 2026-08-05 BEFORE any run:
#
#   Design: {maxrl, grpo} x {uniform, frontier_un} x seeds 0-5.
#   Six independent seed blocks; identical SFT warmstart per block
#   (seed{s}_sft_warmstart.pt, shared across all 4 cells of a block);
#   MATCHED STEP BUDGET (250 steps = matched generation AND optimization
#   budget: 8 groups x 32 rollouts x 250 steps for every cell). The old
#   2400s wall-clock protocol is abandoned deliberately: the GPU is now
#   shared with unrelated jobs, so wall-clock matching would randomize
#   step counts across arms; fixed steps also removes the
#   more-optimization-steps confound the reviews flagged (R:P0-3.4).
#
#   Primary endpoint (one, prespecified): delta mean pass@8 over the 13
#   levels (final step 250 minus post-SFT step -1), fixed 16-maze/level
#   held-out set, Chen-unbiased pass@8 from 8 samples.
#   Primary contrast P-F1: paired (same seed, same sampler) MaxRL - GRPO
#   delta-cov is positive in >= 5/6 blocks under BOTH samplers.
#   6/6 across a sampler gives exact two-sided sign-test p = 0.031.
#   Falsification branch (committed): if <= 4/6 under either sampler,
#   the abstract's estimator-conditioned coverage claim is DROPPED, not
#   softened.
#   Secondary (exploratory, stated in advance): interaction = the
#   teacher amplifies GRPO's coverage loss ((grpo,teacher)-(grpo,unif)
#   delta-cov < 0 in a majority of blocks); easy-band L1-3 resolution.
#
#   Extra arms (existing prereg P-G0a / P-G0c, sweep_grpo_own.sh
#   2026-08-04, protocol migrated to the same fixed-step budget):
#     grpo_mass x grpo  seeds 0-5 (GRPO scheduled by ITS OWN mass)
#     uniform  x grpo_nostd seeds 0-5 (Dr.GRPO-style no-std ablation)
#
# Concurrency 2 (measured: ~15% per-run slowdown for 2x throughput).
# Resumable: cells with an existing complete log (FINAL line) are skipped.
cd "$(dirname "$0")"
exec 9>/tmp/maze_factorial.lock
flock -n 9 || { echo "another factorial driver is running"; exit 0; }

STEPS=250
EVERY=25
LR=1e-4
run_cell() { # teacher estimator seed
  local out="fact250_${1}_${2}_s${3}.jsonl"
  if [ -f "$out" ] && grep -q '"final"' "$out"; then
    echo "skip $out (complete)"; return
  fi
  echo "=== fact250 $1 $2 seed $3 ($(date -u +%H:%M:%SZ)) ==="
  nice -n 5 python3 train.py --teacher "$1" --estimator "$2" \
    --steps $STEPS --eval-every $EVERY --seed "$3" --lr $LR \
    --out "$out" 2>&1 | grep -E "^step|^FINAL|^model|^loaded|^saved" | tail -3
}

# warmstarts for new seed blocks, generated serially (no ckpt race)
for s in 3 4 5; do
  [ -f "seed${s}_sft_warmstart.pt" ] || \
    nice -n 5 python3 train.py --steps 0 --seed "$s" --out /tmp/warm_s${s}.jsonl \
      2>&1 | grep -E "saved SFT"
done

# factorial: per seed block, pairwise-concurrent (same-sampler pairs)
for s in 0 1 2 3 4 5; do
  run_cell uniform maxrl "$s" & run_cell uniform grpo "$s" & wait
  run_cell frontier_un maxrl "$s" & run_cell frontier_un grpo "$s" & wait
done

# extra arms (P-G0a, P-G0c), same budget
for s in 0 1 2 3 4 5; do
  run_cell grpo_mass grpo "$s" & run_cell uniform grpo_nostd "$s" & wait
done

echo "FACTORIAL DONE $(date -u)"
