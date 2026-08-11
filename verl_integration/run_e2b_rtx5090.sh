#!/usr/bin/env bash
# E2b recent-buffer follow-up driver. See E2B_PREREG.md.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RESEARCH_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
RUNTIME_ROOT=${RUNTIME_ROOT:-/data/robotixx/curriculum-maxrl-runtime}
PYTHON_BIN=${PYTHON_BIN:-$RUNTIME_ROOT/venv/bin/python}
ITERATION_DIR=${ITERATION_DIR:-$RESEARCH_ROOT/autoresearch/iterate-260809-1533}
LOG_DIR=${LOG_DIR:-$ITERATION_DIR/e2_logs}
RESULT_DIR=${RESULT_DIR:-$ITERATION_DIR/e2_results}
TEST_DATA=${TEST_DATA:-$RUNTIME_ROOT/data/countdown_v2_rebuilt/test.parquet}
SEEDS=${SEEDS:-"1 2 3"}
STEPS=${STEPS:-60}

mkdir -p "$LOG_DIR" "$RESULT_DIR"

run_arm() {
  local arm=$1
  local seed=$2
  local run_id="e2_clean_${arm}_s${seed}_260809"
  local output_dir="$RUNTIME_ROOT/checkpoints/$run_id"
  local log_path="$LOG_DIR/${run_id}.log"
  local marker="$output_dir/.complete"
  if [[ -f "$marker" ]]; then
    echo "complete, skipping: $run_id"
    return
  fi
  if [[ -e "$output_dir" || -e "$log_path" ]]; then
    echo "refusing to append to incomplete run: $run_id" >&2
    exit 1
  fi

  local hindsight=false
  if [[ "$arm" == b2 ]]; then
    hindsight=true
  fi
  echo "starting $run_id"
  RUN_ID="$run_id" OUTPUT_DIR="$output_dir" \
    STEPS="$STEPS" TRAIN_BATCH=8 N_ROLLOUTS=16 \
    MAX_RESPONSE_LENGTH=128 VAL_BEFORE_TRAIN=false TEST_FREQ=-1 \
    VAL_ON_LAST_STEP=false SAVE_FREQ="$STEPS" \
    SAVE_CONTENTS="['model','optimizer','extra','hf_model']" \
    SEED="$seed" LR=1e-5 HINDSIGHT_ENABLE="$hindsight" \
    HINDSIGHT_MAX_GROUPS=8 HINDSIGHT_ONE_TARGET=false \
    HINDSIGHT_UTILITY_GATE=false LIVE_REPLAY_ENABLE=false \
    bash "$SCRIPT_DIR/countdown_rtx5090.sh" 2>&1 | tee "$log_path"

  verify_checkpoint "$run_id" "$output_dir" "$log_path"
  if [[ "$arm" == b2 ]] && \
      [[ $(wc -l < "$output_dir/dose_accounting.jsonl") -ne "$STEPS" ]]; then
    echo "B2 audit line count failed: $run_id" >&2
    exit 1
  fi
  touch "$marker"
}

run_buffer_replay() {
  local seed=$1
  local run_id="e2b_buffer_replay_s${seed}_260809"
  local output_dir="$RUNTIME_ROOT/checkpoints/$run_id"
  local log_path="$LOG_DIR/${run_id}.log"
  local marker="$output_dir/.complete"
  local b2_audit="$RUNTIME_ROOT/checkpoints/e2_clean_b2_s${seed}_260809/dose_accounting.jsonl"
  if [[ -f "$marker" ]]; then
    echo "complete, skipping: $run_id"
    return
  fi
  if [[ -e "$output_dir" || -e "$log_path" ]]; then
    echo "refusing to append to incomplete run: $run_id" >&2
    exit 1
  fi
  if [[ $(wc -l < "$b2_audit") -ne "$STEPS" ]]; then
    echo "B2 audit is incomplete for seed $seed" >&2
    exit 1
  fi

  echo "starting $run_id"
  RUN_ID="$run_id" OUTPUT_DIR="$output_dir" \
    STEPS="$STEPS" TRAIN_BATCH=8 N_ROLLOUTS=16 \
    MAX_RESPONSE_LENGTH=128 VAL_BEFORE_TRAIN=false TEST_FREQ=-1 \
    VAL_ON_LAST_STEP=false SAVE_FREQ="$STEPS" \
    SAVE_CONTENTS="['model','optimizer','extra','hf_model']" \
    SEED="$seed" LR=1e-5 HINDSIGHT_ENABLE=false \
    LIVE_REPLAY_ENABLE=true LIVE_REPLAY_SCHEDULE="$b2_audit" \
    LIVE_REPLAY_STRICT=true LIVE_REPLAY_MAX_TOKEN_MISMATCH=0.05 \
    LIVE_REPLAY_BUFFER_GROUPS=64 LIVE_REPLAY_MAX_BUFFER_AGE=8 \
    bash "$SCRIPT_DIR/countdown_rtx5090.sh" 2>&1 | tee "$log_path"

  verify_checkpoint "$run_id" "$output_dir" "$log_path"
  if [[ $(wc -l < "$output_dir/replay_accounting.jsonl") -ne "$STEPS" ]]; then
    echo "Rb audit line count failed: $run_id" >&2
    exit 1
  fi
  touch "$marker"
}

verify_checkpoint() {
  local run_id=$1
  local output_dir=$2
  local log_path=$3
  if ! grep -q "training/global_step:${STEPS}\.000" "$log_path"; then
    echo "missing final training step: $run_id" >&2
    exit 1
  fi
  if [[ ! -f "$output_dir/global_step_${STEPS}/actor/huggingface/config.json" ]]; then
    echo "missing final Hugging Face checkpoint: $run_id" >&2
    exit 1
  fi
}

evaluate_arm() {
  local arm=$1
  local seed=$2
  local run_id
  if [[ "$arm" == rb ]]; then
    run_id="e2b_buffer_replay_s${seed}_260809"
  else
    run_id="e2_clean_${arm}_s${seed}_260809"
  fi
  local model="$RUNTIME_ROOT/checkpoints/$run_id/global_step_${STEPS}/actor/huggingface"
  local output="$RESULT_DIR/${run_id}_eval.json"
  if [[ -f "$output" ]]; then
    echo "evaluation exists, skipping: $output"
    return
  fi
  PYTHONPATH="$RESEARCH_ROOT" HF_HOME="$RUNTIME_ROOT/hf" \
    "$PYTHON_BIN" -m curriculum_maxrl.countdown.eval_countdown \
      --model "$model" --data "$TEST_DATA" --output "$output" \
      --k 16 --batch-size 8 --max-new-tokens 128 \
      --temperature 1.0 --top-p 1.0 --seed "$((10000 + seed))"
}

for seed in $SEEDS; do
  run_arm b1 "$seed"
  run_arm b2 "$seed"
  run_buffer_replay "$seed"
  evaluate_arm b1 "$seed"
  evaluate_arm b2 "$seed"
  evaluate_arm rb "$seed"
done

echo "E2b execution complete"
