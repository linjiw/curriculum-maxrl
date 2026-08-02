#!/bin/bash
# Feasibility queue for E-LLM-3. Ordering constraints:
#  - needs pool_v1.jsonl (generator running)
#  - must NOT overlap sweep_un_form.sh (matched WALL-CLOCK protocol —
#    GPU contention would corrupt it), so wait for its DONE marker
#  - H6 verl job: safe to follow (its metrics are step-indexed)
cd "$(dirname "$0")"
exec 9>/tmp/jugs_feasibility.lock
flock -n 9 || exit 0

echo "=== queue start $(date) ==="
# 1. wait for the pool
while [ ! -s pool_v1.jsonl ]; do sleep 120; done
echo "pool ready ($(wc -l < pool_v1.jsonl) tasks, $(date))"

# 2. wait for the maze sweep to finish (marker in its log), if it started
MAZE_LOG=/home/ec2-user/work/curriculumrl/maxrl/curriculum_maxrl/maze_gpu/sweep_un_form.log
while true; do
  if grep -q "UN-FORM SWEEP DONE" "$MAZE_LOG" 2>/dev/null; then break; fi
  sleep 600
done
echo "maze sweep done ($(date))"

# 3. run feasibility per model (script itself waits for free GPU)
export HF_HOME=~/hf-cache
PY=/home/ec2-user/venvs/maxrl311/bin/python3
for model in HuggingFaceTB/SmolLM2-360M-Instruct Qwen/Qwen2.5-0.5B-Instruct; do
  echo "=== feasibility $model ($(date)) ==="
  $PY eval_feasibility.py --model "$model" --per-tier 40 --n 16 --two-shot \
    2>&1 | tail -30
done
echo "FEASIBILITY QUEUE DONE $(date)"
