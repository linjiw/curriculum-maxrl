#!/bin/bash
# E-LLM-3 stage 2: SFT warmstart -> zero-shot feasibility on the SFT
# checkpoint. The prereg numbers bind to THIS feasibility (the RL cells
# run zero-shot from the SFT checkpoint, so its tier landscape is the
# one that matters).
# Waits for the two-shot feasibility queue (SmolLM2 done, Qwen running)
# to finish first.
cd "$(dirname "$0")"
exec 9>/tmp/jugs_sft_queue.lock
flock -n 9 || exit 0
export HF_HOME=~/hf-cache
PY=/home/ec2-user/venvs/maxrl311/bin/python3

while ! grep -q "FEASIBILITY QUEUE DONE" feasibility_queue.log 2>/dev/null; do
  sleep 300
done
echo "=== two-shot feasibility done, starting SFT ($(date)) ==="

while true; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free_mb" -gt 16000 ] && break
  sleep 300
done

cd /home/ec2-user/work/curriculumrl/maxrl
$PY curriculum_maxrl/jugs/sft_warmstart.py 2>&1 | tail -15
echo "=== SFT done, running post-SFT zero-shot feasibility ($(date)) ==="

cd /home/ec2-user/work/curriculumrl/curriculum-maxrl/jugs_llm
$PY eval_feasibility.py --model "$HOME/ckpt/jugs_sft_v1" --per-tier 40 --n 16 \
  2>&1 | tail -20
echo "SFT+FEASIBILITY DONE $(date)"
