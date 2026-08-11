#!/usr/bin/env bash
# Frozen E2 execution driver. See the preregistration under autoresearch/.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RESEARCH_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
RUNTIME_ROOT=${RUNTIME_ROOT:-/data/robotixx/curriculum-maxrl-runtime}
PYTHON_BIN=${PYTHON_BIN:-$RUNTIME_ROOT/venv/bin/python}
DATA_PATH=${DATA_PATH:-$RUNTIME_ROOT/data/countdown_v2_rebuilt/test.parquet}
ITERATION_DIR=${ITERATION_DIR:-$RESEARCH_ROOT/autoresearch/iterate-260809-1533}
LOG_DIR=${LOG_DIR:-$ITERATION_DIR/e2_logs}
RESULT_DIR=${RESULT_DIR:-$ITERATION_DIR/e2_results}
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
  local b2_audit="$RUNTIME_ROOT/checkpoints/e2_clean_b2_s${seed}_260809/dose_accounting.jsonl"

  if [[ -f "$marker" ]]; then
    echo "complete, skipping: $run_id"
    return
  fi
  if [[ -e "$output_dir" || -e "$log_path" ]]; then
    echo "refusing to append to an incomplete scientific run: $run_id" >&2
    exit 1
  fi

  local hindsight=false
  local replay=false
  if [[ "$arm" == b2 ]]; then
    hindsight=true
  elif [[ "$arm" == replay ]]; then
    replay=true
    if [[ $(wc -l < "$b2_audit") -ne "$STEPS" ]]; then
      echo "B2 audit is incomplete for seed $seed: $b2_audit" >&2
      exit 1
    fi
  fi

  echo "starting $run_id"
  RUN_ID="$run_id" OUTPUT_DIR="$output_dir" \
    STEPS="$STEPS" TRAIN_BATCH=8 N_ROLLOUTS=16 \
    MAX_RESPONSE_LENGTH=128 VAL_BEFORE_TRAIN=false TEST_FREQ=-1 \
    VAL_ON_LAST_STEP=false SAVE_FREQ="$STEPS" \
    SAVE_CONTENTS="['model','optimizer','extra','hf_model']" \
    SEED="$seed" LR=1e-5 \
    HINDSIGHT_ENABLE="$hindsight" HINDSIGHT_MAX_GROUPS=8 \
    HINDSIGHT_ONE_TARGET=false HINDSIGHT_UTILITY_GATE=false \
    LIVE_REPLAY_ENABLE="$replay" LIVE_REPLAY_SCHEDULE="$b2_audit" \
    LIVE_REPLAY_STRICT=true LIVE_REPLAY_MAX_TOKEN_MISMATCH=0.05 \
    bash "$SCRIPT_DIR/countdown_rtx5090.sh" 2>&1 | tee "$log_path"

  if ! grep -q "training/global_step:${STEPS}\.000" "$log_path"; then
    echo "missing final training step in $log_path" >&2
    exit 1
  fi
  local hf_dir="$output_dir/global_step_${STEPS}/actor/huggingface"
  if [[ ! -f "$hf_dir/config.json" ]]; then
    echo "missing final Hugging Face checkpoint: $hf_dir" >&2
    exit 1
  fi
  if [[ "$arm" == b2 ]] && [[ $(wc -l < "$output_dir/dose_accounting.jsonl") -ne "$STEPS" ]]; then
    echo "B2 audit line count failed: $run_id" >&2
    exit 1
  fi
  if [[ "$arm" == replay ]] && [[ $(wc -l < "$output_dir/replay_accounting.jsonl") -ne "$STEPS" ]]; then
    echo "replay audit line count failed: $run_id" >&2
    exit 1
  fi

  touch "$marker"
}

evaluate_arm() {
  local arm=$1
  local seed=$2
  local run_id="e2_clean_${arm}_s${seed}_260809"
  local hf_dir="$RUNTIME_ROOT/checkpoints/$run_id/global_step_${STEPS}/actor/huggingface"
  local output="$RESULT_DIR/${run_id}_eval.json"
  if [[ -f "$output" ]]; then
    echo "evaluation exists, skipping: $output"
    return
  fi
  PYTHONPATH="$RESEARCH_ROOT" HF_HOME="$RUNTIME_ROOT/hf" \
    "$PYTHON_BIN" -m curriculum_maxrl.countdown.eval_countdown \
      --model "$hf_dir" --data "$DATA_PATH" --output "$output" \
      --k 16 --batch-size 8 --max-new-tokens 128 \
      --temperature 1.0 --top-p 1.0 --seed "$((10000 + seed))"
}

for seed in $SEEDS; do
  run_arm b1 "$seed"
  run_arm b2 "$seed"
  run_arm replay "$seed"
  evaluate_arm b1 "$seed"
  evaluate_arm b2 "$seed"
  evaluate_arm replay "$seed"
done

echo "E2 execution complete"
