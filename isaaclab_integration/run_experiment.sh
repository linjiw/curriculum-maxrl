#!/usr/bin/env bash
# Ladder step 1: frontier teacher vs {control, greedy, scripted, uniform} on
# Isaac-Velocity-Rough-Anymal-C-v0 — the pre-registered 5-arm experiment
# (ISAACLAB_DESIGN.md D9 + P-A/P-B/P-C; INTEGRATION.md §4).
#
# Runs from the HOST; all GPU work docker-execs into isaac-lab-base.
# Unattended-safe: per-arm logs + state file in _pilot/, arms skip if already
# completed (idempotent resume), one arm failing does not abort the rest.
#
# Usage:   ./run_experiment.sh [pilot|full]
#   pilot: 1 seed × 5 arms × 600 iters,  1024 envs  (~5-6 h on one A10G)
#   full:  5 seeds × 5 arms × 1500 iters, 4096 envs  (schedule on a bigger box)

set -uo pipefail
MODE="${1:-pilot}"
C=isaac-lab-base
TASK=Isaac-Velocity-Rough-Anymal-C-v0
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/_pilot"
mkdir -p "$OUT"
STATE="$OUT/completed_${MODE}.txt"
touch "$STATE"

if [[ "$MODE" == "full" ]]; then
  SEEDS=(42 43 44 45 46); ITERS=1500; ENVS=4096
else
  SEEDS=(42);             ITERS=600;  ENVS=1024
fi
# scripted ramp spans the whole run: iters * num_steps_per_env(24)
RAMP=$((ITERS * 24))

# Override with e.g. ARMS="teacher_g4 hybrid" ./run_experiment.sh pilot
ARMS=(${ARMS:-control greedy scripted uniform teacher})
FAILED=()

# GPU gate: Isaac Sim's kit python exits 0 even on CUDA-context crashes, and a
# co-tenant (e.g. a verl run) can spike to the full card between its quiet
# phases (measured: 13 GB steady, 22.5 GB for ~4 min every ~24-min step —
# co-running kills whichever process allocates during the spike).
#
# Queue-chaining protocol (agreed in REVIEW_ADVICE.md §1 with the verl-queue
# owner): (1) wait on the co-tenant's PROCESS NAMES, not instantaneous memory;
# (2) claim files close the check-then-launch race in both directions —
# we hold /tmp/gpu_claim_isaaclab while an arm trains, they hold
# /tmp/gpu_claim_verl while a cell trains.
# NOTE: wait on their TRAINER + claim file only — their queue-driver scripts
# (run_2x2_resume.sh) idle-wait for OUR process names and yield to us (FIFO by
# remaining cost, agreed 7/23); including them here would deadlock both queues.
GATE_HOURS="${GATE_HOURS:-96}"
VERL_QUEUE_PATTERN="verl.trainer.main_ppo"
CLAIM_OURS=/tmp/gpu_claim_isaaclab
CLAIM_THEIRS=/tmp/gpu_claim_verl
trap 'rm -f "$CLAIM_OURS"' EXIT

wait_gpu_free() {
  local deadline=$((SECONDS + GATE_HOURS * 3600)) ok=0
  while [ $SECONDS -lt $deadline ]; do
    # chain on the co-tenant queue: name-based, cannot fire mid-queue
    if pgrep -f "$VERL_QUEUE_PATTERN" >/dev/null || [ -e "$CLAIM_THEIRS" ]; then
      ok=0
      if [ $((SECONDS % 1800)) -lt 60 ]; then
        echo "    [gate $(date +%H:%M)] verl queue active — chained-waiting..."
      fi
      sleep 300
      continue
    fi
    local procs used
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$procs" -eq 0 ] && [ "$used" -lt 2000 ]; then
      ok=$((ok + 1)); [ $ok -ge 2 ] && return 0   # stable across 2 checks
    else
      ok=0
      if [ $((SECONDS % 600)) -lt 60 ]; then
        echo "    [gate $(date +%H:%M)] GPU busy (${procs} procs, ${used} MiB) — waiting..."
      fi
    fi
    sleep 60
  done
  return 1
}

# Success = the training artifact, NEVER the exit code (kit exits 0 on crash).
run_ok() { grep -q "Training time:" "$1"; }

for SEED in "${SEEDS[@]}"; do
  for ARM in "${ARMS[@]}"; do
    KEY="${ARM}_s${SEED}"
    if grep -qx "$KEY" "$STATE"; then
      echo "=== $KEY already completed, skipping ==="
      continue
    fi
    LOG="$OUT/${KEY}_${MODE}.log"
    echo "=== arm=$ARM seed=$SEED ($ITERS iters, $ENVS envs) → $LOG ==="
    if ! wait_gpu_free; then
      FAILED+=("$KEY"); echo "    FAILED: GPU never freed within gate window"; continue
    fi
    touch "$CLAIM_OURS"
    START=$(date +%s)
    docker exec "$C" bash -c "cd /workspace/isaaclab && /isaac-sim/python.sh \
        scripts/curriculum-maxrl/isaaclab_integration/train_frontier.py \
        --task $TASK --headless --num_envs $ENVS --max_iterations $ITERS \
        --seed $SEED --arm $ARM --success_fn tile \
        --scripted_total_steps $RAMP \
        agent.run_name=s${SEED}" > "$LOG" 2>&1
    rm -f "$CLAIM_OURS"
    if run_ok "$LOG"; then
      echo "$KEY" >> "$STATE"
      echo "    OK in $((($(date +%s) - START) / 60)) min"
    else
      FAILED+=("$KEY")
      echo "    FAILED — no 'Training time:' artifact in $LOG (continuing)"
      grep -iE "error|out of memory" "$LOG" | head -3 | sed 's/^/    | /'
    fi
  done
done

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "DONE WITH FAILURES: ${FAILED[*]}"
  exit 1
fi
echo "All arms done. Logs: logs/rsl_rl/anymal_c_rough/<ts>_<arm>_s<seed>/"
echo "Analyze: docker exec $C /isaac-sim/python.sh \
/workspace/isaaclab/scripts/curriculum-maxrl/isaaclab_integration/analyze_arms.py"
