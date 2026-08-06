#!/bin/bash
# Review-round-4 GPU arms (Countdown v2 pool), queued LAST.
#
# Pre-registered (2026-08-04, before any of these cells):
#
# ARM A — corrected-gate B3 at the DESIGNED operating point, seeds 1-3
#   (R2-Q7, R3-Q6, R4-Q4). Current verl/utils/hindsight.py carries the
#   corrected decay (both-sides F5 fix) and task-keyed destinations, so
#   GATE=true now IS the designed gate.
#   P-R1: designed-gate B3 lands on the dose-response line between the
#   under-gated 3-seed point (~60% mean kept, coverage restored) and
#   the single-seed full-strength point (gain erased, coverage above
#   baseline) — i.e., mean gain kept in [0%, 60%], tier-1 coverage in
#   [baseline, baseline+.03]. If it lands OFF the line (e.g. coverage
#   lost), the dial claim is refuted and Fig 7a must be redrawn as a
#   scatter, not a frontier.
#
# ARM B — dose-matched replay at LLM scale (R2-Q3, R4-Q5), seeds 1-3.
#   Reasoning fixed at design time: at matched GENERATION budget,
#   relabels are free (they reuse already-paid rollouts), and MaxRL
#   already "skips" dead groups (zero weights). So the correct dose
#   control is the CPU battery's replay placebo transplanted to LLM
#   scale: extra optimizer passes on live groups = ppo_epochs=2,
#   hindsight OFF. Same generation budget as B1/B2, roughly B2's extra
#   update dose, none of B2's off-target direction.
#   P-R2: replay captures >= half of B2's tier-1 mean@16 gain with NO
#   pass@16 loss (it adds no off-target updates). If it captures ~all,
#   recycling's LLM-scale case reduces to the direction term and 6.8
#   must say so (committed).
set -e
cd "$(dirname "$0")/.."
exec 9>/tmp/run_reviewer_arms.lock
flock -n 9 || exit 0
export HF_HOME=~/hf-cache

wait_gpu() {
  while true; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$free_mb" -gt 18000 ] && break
    sleep 300
  done
}

# ARM A: designed-gate B3, 3 seeds
for seed in 1 2 3; do
  marker=~/ckpt/countdown_a10g/.done_b3fix_s${seed}
  [ -f "$marker" ] && continue
  wait_gpu
  echo "=== ARM-A designed-gate B3 seed $seed ($(date)) ==="
  log=/tmp/armA_b3fix_s${seed}.log
  TRAIN_SEED=$seed POOL_TAG="b3fix_s${seed}" ESTIMATOR=maxrl CURRICULUM=false \
    HINDSIGHT=true GATE=true bash smollm/countdown_a10g.sh > "$log" 2>&1
  grep -E "step:|hindsight/|Error" "$log" | tail -30
  # marker only on completion (global step 60 = STEPS default)
  if grep -q "global_step:60" "$log"; then touch "$marker"; fi
done

# ARM B: dose-matched replay (ppo_epochs=2, hindsight off), 3 seeds
for seed in 1 2 3; do
  marker=~/ckpt/countdown_a10g/.done_replay_s${seed}
  [ -f "$marker" ] && continue
  wait_gpu
  echo "=== ARM-B replay (ppo_epochs=2) seed $seed ($(date)) ==="
  TRAIN_SEED=$seed POOL_TAG="replay_s${seed}" ESTIMATOR=maxrl \
    CURRICULUM=false HINDSIGHT=false \
    EXTRA_ARGS="actor_rollout_ref.actor.ppo_epochs=2" \
    bash smollm/countdown_a10g.sh 2>&1 \
    | grep -E "step:|val|Error|Traceback" | tail -60
  touch "$marker"
done
echo "REVIEWER ARMS DONE $(date)"
