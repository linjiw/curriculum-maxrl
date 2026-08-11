#!/usr/bin/env bash
# E2c immutable-reservoir replay driver. See E2C_PREREG.md.

set -euo pipefail

READINESS_ONLY=false
if [[ ${1:-} == "--readiness-only" ]]; then
  READINESS_ONLY=true
  shift
fi
if (( $# != 0 )); then
  echo "usage: $0 [--readiness-only]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RESEARCH_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
RUNTIME_ROOT=${RUNTIME_ROOT:-/data/robotixx/curriculum-maxrl-runtime}
readonly PYTHON_BIN="$RUNTIME_ROOT/venv/bin/python"
readonly ITERATION_DIR="$RESEARCH_ROOT/autoresearch/iterate-260810-2240"
readonly LOG_DIR="$ITERATION_DIR/e2c_logs"
readonly RESULT_DIR="$ITERATION_DIR/e2c_results"
readonly TRAIN_DATA="$RUNTIME_ROOT/data/countdown_v2_rebuilt/train.parquet"
readonly TEST_DATA="$RUNTIME_ROOT/data/countdown_v2_rebuilt/test.parquet"
readonly MODEL_PATH="$RUNTIME_ROOT/models/countdown_sft_clean_v1"
readonly SEEDS="1 2 3"
readonly STEPS=60
readonly GPU_MEMORY_CEILING_MIB=4096
readonly MAXRL_COMMIT=7197bbb46a2ecd866da52f6b401ff20a34fe9390
readonly CODE_MANIFEST="$ITERATION_DIR/E2C_CODE_MANIFEST.json"
readonly CODE_MANIFEST_SHA256=0e46b89fcc01300b52bc6fc4e0c8a0ee5f2aa72d357b233e85c357972e8d3828
readonly RESERVOIR_RUN_ID=e2c_reservoir_collect_260810
readonly RESERVOIR_DIR="$RUNTIME_ROOT/checkpoints/$RESERVOIR_RUN_ID"
readonly RESERVOIR_PATH="$RESERVOIR_DIR/replay_reservoir.pt"
readonly RESERVOIR_MANIFEST="$RESERVOIR_PATH.manifest.json"

validate_runtime() {
  [[ -x "$PYTHON_BIN" ]] || {
    echo "missing E2c Python: $PYTHON_BIN" >&2
    exit 1
  }
  [[ -f "$TRAIN_DATA" && -f "$TEST_DATA" ]] || {
    echo "missing frozen Countdown data" >&2
    exit 1
  }
  [[ -f "$MODEL_PATH/model.safetensors" ]] || {
    echo "missing frozen clean-SFT model" >&2
    exit 1
  }
  local actual_commit
  actual_commit=$(git -C "$RUNTIME_ROOT/maxrl" rev-parse HEAD)
  [[ "$actual_commit" == "$MAXRL_COMMIT" ]] || {
    echo "MaxRL commit $actual_commit != frozen $MAXRL_COMMIT" >&2
    exit 1
  }
  cmp -s "$RESEARCH_ROOT/verl_integration/vendored/hindsight.py" \
    "$RUNTIME_ROOT/maxrl/verl/utils/hindsight.py" || {
    echo "runtime hindsight implementation differs from vendored source" >&2
    exit 1
  }
  local manifest_digest
  manifest_digest=$(sha256sum "$CODE_MANIFEST" | cut -d' ' -f1)
  [[ "$manifest_digest" == "$CODE_MANIFEST_SHA256" ]] || {
    echo "E2c code manifest $manifest_digest != frozen $CODE_MANIFEST_SHA256" >&2
    exit 1
  }
}

validate_runtime
run_readiness_audit() {
  PYTHONPATH="$RESEARCH_ROOT" "$PYTHON_BIN" \
    -m curriculum_maxrl.countdown.audit_e2c_readiness \
    --runtime-root "$RUNTIME_ROOT" --research-root "$RESEARCH_ROOT" \
    --iteration-dir "$ITERATION_DIR" \
    --output "$ITERATION_DIR/E2C_LAUNCH_READINESS.json"
}
run_readiness_audit
if [[ "$READINESS_ONLY" == true ]]; then
  exit 0
fi
mkdir -p "$LOG_DIR" "$RESULT_DIR"
exec 9>"$RUNTIME_ROOT/e2c_gpu.lock"
if ! flock -n 9; then
  echo "another E2c driver owns $RUNTIME_ROOT/e2c_gpu.lock" >&2
  exit 1
fi

require_gpu_clear() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if (( used > GPU_MEMORY_CEILING_MIB )); then
    echo "shared GPU is occupied: ${used} MiB > ${GPU_MEMORY_CEILING_MIB} MiB" >&2
    exit 1
  fi
}

verify_checkpoint() {
  local run_id=$1
  local output_dir=$2
  local log_path=$3
  if [[ ! -f "$log_path" ]] || \
      ! grep -q "training/global_step:${STEPS}\.000" "$log_path"; then
    echo "missing final training step: $run_id" >&2
    exit 1
  fi
  if [[ ! -f "$output_dir/global_step_${STEPS}/actor/huggingface/config.json" ]]; then
    echo "missing final Hugging Face checkpoint: $run_id" >&2
    exit 1
  fi
  if [[ ! -f "$output_dir/global_step_${STEPS}/actor/huggingface/model.safetensors" ]]; then
    echo "missing final Hugging Face model weights: $run_id" >&2
    exit 1
  fi
}

verify_b2_schedule() {
  local run_id=$1
  local audit_path=$2
  if [[ ! -f "$audit_path" ]] || \
      [[ $(wc -l < "$audit_path") -ne "$STEPS" ]]; then
    echo "B2 audit is not a complete ${STEPS}-row schedule: $run_id" >&2
    exit 1
  fi
}

run_comparator() {
  local arm=$1
  local seed=$2
  local run_id="e2_clean_${arm}_s${seed}_260809"
  local output_dir="$RUNTIME_ROOT/checkpoints/$run_id"
  local log_path="$LOG_DIR/${run_id}.log"
  local historical_log="$RESEARCH_ROOT/autoresearch/iterate-260809-1533/e2_logs/${run_id}.log"
  local marker="$output_dir/.complete"
  if [[ -f "$marker" ]]; then
    local completed_log=$log_path
    if [[ ! -f "$completed_log" ]]; then
      completed_log=$historical_log
    fi
    verify_checkpoint "$run_id" "$output_dir" "$completed_log"
    if [[ "$arm" == b2 ]]; then
      verify_b2_schedule "$run_id" "$output_dir/dose_accounting.jsonl"
    fi
    echo "complete, skipping comparator: $run_id"
    return
  fi
  if [[ -e "$output_dir" || -e "$log_path" ]]; then
    echo "refusing to append to incomplete comparator: $run_id" >&2
    exit 1
  fi
  local hindsight=false
  if [[ "$arm" == b2 ]]; then
    hindsight=true
  fi
  require_gpu_clear
  echo "starting comparator $run_id"
  RUNTIME_ROOT="$RUNTIME_ROOT" MAXRL_ROOT="$RUNTIME_ROOT/maxrl" \
    PYTHON_BIN="$PYTHON_BIN" MODEL_PATH="$MODEL_PATH" \
    HF_HOME="$RUNTIME_ROOT/hf" RAY_DIR="/data/robotixx/cmrl-ray/${run_id:0:16}" \
    TRAIN_DATA="$TRAIN_DATA" VAL_DATA="$TEST_DATA" \
    REWARD_PATH="$RESEARCH_ROOT/curriculum_maxrl/countdown/countdown_reward.py" \
    RUN_ID="$run_id" OUTPUT_DIR="$output_dir" \
    STEPS="$STEPS" TRAIN_BATCH=8 N_ROLLOUTS=16 \
    ROLLOUT_MICRO_BATCH=128 LOGPROB_MICRO_BATCH=8 PPO_MICRO_BATCH=4 \
    MAX_RESPONSE_LENGTH=128 VAL_BEFORE_TRAIN=false TEST_FREQ=-1 \
    VAL_ON_LAST_STEP=false SAVE_FREQ="$STEPS" \
    SAVE_CONTENTS="['model','optimizer','extra','hf_model']" \
    FILTER_OVERLONG=false REWARD_MANAGER=dapo GRADIENT_CHECKPOINTING=false \
    SEED="$seed" LR=1e-5 HINDSIGHT_ENABLE="$hindsight" \
    HINDSIGHT_MAX_GROUPS=8 HINDSIGHT_ONE_TARGET=false \
    HINDSIGHT_UTILITY_GATE=false \
    HINDSIGHT_AUDIT_PATH="$output_dir/dose_accounting.jsonl" \
    LIVE_REPLAY_ENABLE=false REPLAY_RESERVOIR_COLLECT=false \
    bash "$SCRIPT_DIR/countdown_rtx5090.sh" 2>&1 | tee "$log_path"
  verify_checkpoint "$run_id" "$output_dir" "$log_path"
  if [[ "$arm" == b2 ]]; then
    verify_b2_schedule "$run_id" "$output_dir/dose_accounting.jsonl"
  fi
  touch "$marker"
  # Keep the active-iteration log self-contained without altering historical
  # records. Existing completed arms retain their original log location.
  if [[ -f "$historical_log" ]]; then
    echo "historical log also exists: $historical_log"
  fi
}

collect_reservoir() {
  local log_path="$LOG_DIR/${RESERVOIR_RUN_ID}.log"
  local marker="$RESERVOIR_DIR/.complete"
  if [[ -f "$marker" ]]; then
    [[ -f "$RESERVOIR_PATH" && -f "$RESERVOIR_MANIFEST" && \
       -f "$log_path" ]] || {
      echo "reservoir marker exists without complete artifacts" >&2
      exit 1
    }
    grep -q 'training/global_step:60\.000' "$log_path" || {
      echo "reservoir marker exists without a complete training log" >&2
      exit 1
    }
    echo "complete, skipping reservoir: $RESERVOIR_PATH"
    return
  fi
  if [[ -e "$RESERVOIR_DIR" || -e "$log_path" ]]; then
    echo "refusing to append to incomplete reservoir collection" >&2
    exit 1
  fi
  require_gpu_clear
  echo "starting frozen-SFT reservoir collection"
  RUNTIME_ROOT="$RUNTIME_ROOT" MAXRL_ROOT="$RUNTIME_ROOT/maxrl" \
    PYTHON_BIN="$PYTHON_BIN" MODEL_PATH="$MODEL_PATH" \
    HF_HOME="$RUNTIME_ROOT/hf" \
    RAY_DIR="/data/robotixx/cmrl-ray/${RESERVOIR_RUN_ID:0:16}" \
    TRAIN_DATA="$TRAIN_DATA" VAL_DATA="$TEST_DATA" \
    REWARD_PATH="$RESEARCH_ROOT/curriculum_maxrl/countdown/countdown_reward.py" \
    RUN_ID="$RESERVOIR_RUN_ID" OUTPUT_DIR="$RESERVOIR_DIR" \
    STEPS=60 TRAIN_BATCH=8 N_ROLLOUTS=16 \
    ROLLOUT_MICRO_BATCH=128 LOGPROB_MICRO_BATCH=8 PPO_MICRO_BATCH=4 \
    MAX_RESPONSE_LENGTH=128 VAL_BEFORE_TRAIN=false TEST_FREQ=-1 \
    VAL_ON_LAST_STEP=false SAVE_FREQ=-1 SEED=424242 LR=0 \
    FILTER_OVERLONG=false REWARD_MANAGER=dapo GRADIENT_CHECKPOINTING=false \
    HINDSIGHT_ENABLE=false LIVE_REPLAY_ENABLE=false \
    REPLAY_RESERVOIR_COLLECT=true \
    REPLAY_RESERVOIR_OUTPUT="$RESERVOIR_PATH" \
    REPLAY_RESERVOIR_MAX_GROUPS=256 REPLAY_RESERVOIR_GROUP_SIZE=16 \
    REPLAY_RESERVOIR_SEED=424242 \
    bash "$SCRIPT_DIR/countdown_rtx5090.sh" 2>&1 | tee "$log_path"
  if ! grep -q 'training/global_step:60\.000' "$log_path"; then
    echo "reservoir collection did not complete 60 steps" >&2
    exit 1
  fi
  [[ -f "$RESERVOIR_PATH" && -f "$RESERVOIR_MANIFEST" ]] || {
    echo "reservoir artifacts are missing" >&2
    exit 1
  }
  touch "$marker"
}

run_preflight() {
  local output="$ITERATION_DIR/E2C_PREFLIGHT.json"
  local args=()
  for seed in $SEEDS; do
    args+=(--schedule "$RUNTIME_ROOT/checkpoints/e2_clean_b2_s${seed}_260809/dose_accounting.jsonl")
    args+=(--seed "$seed")
  done
  PYTHONPATH="$RESEARCH_ROOT" "$PYTHON_BIN" \
    -m curriculum_maxrl.countdown.preflight_e2c \
    --reservoir "$RESERVOIR_PATH" \
    --train-data "$TRAIN_DATA" --test-data "$TEST_DATA" \
    --model "$MODEL_PATH" \
    --maxrl-commit "$MAXRL_COMMIT" \
    --output "$output" "${args[@]}"
}

reservoir_sha256() {
  "$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["sha256"])' \
    "$RESERVOIR_MANIFEST"
}

run_e2c() {
  local seed=$1
  local digest=$2
  local run_id="e2c_reservoir_replay_s${seed}_260810"
  local output_dir="$RUNTIME_ROOT/checkpoints/$run_id"
  local log_path="$LOG_DIR/${run_id}.log"
  local marker="$output_dir/.complete"
  local b2_audit="$RUNTIME_ROOT/checkpoints/e2_clean_b2_s${seed}_260809/dose_accounting.jsonl"
  if [[ -f "$marker" ]]; then
    verify_checkpoint "$run_id" "$output_dir" "$log_path"
    if [[ ! -f "$output_dir/replay_accounting.jsonl" ]] || \
        [[ $(wc -l < "$output_dir/replay_accounting.jsonl") -ne "$STEPS" ]]; then
      echo "E2c marker exists without a complete delivery audit: $run_id" >&2
      exit 1
    fi
    echo "complete, skipping E2c: $run_id"
    return
  fi
  if [[ -e "$output_dir" || -e "$log_path" ]]; then
    echo "refusing to append to incomplete E2c run: $run_id" >&2
    exit 1
  fi
  [[ $(wc -l < "$b2_audit") -eq "$STEPS" ]] || {
    echo "B2 schedule is incomplete for seed $seed" >&2
    exit 1
  }
  require_gpu_clear
  echo "starting $run_id"
  RUNTIME_ROOT="$RUNTIME_ROOT" MAXRL_ROOT="$RUNTIME_ROOT/maxrl" \
    PYTHON_BIN="$PYTHON_BIN" MODEL_PATH="$MODEL_PATH" \
    HF_HOME="$RUNTIME_ROOT/hf" RAY_DIR="/data/robotixx/cmrl-ray/${run_id:0:16}" \
    TRAIN_DATA="$TRAIN_DATA" VAL_DATA="$TEST_DATA" \
    REWARD_PATH="$RESEARCH_ROOT/curriculum_maxrl/countdown/countdown_reward.py" \
    RUN_ID="$run_id" OUTPUT_DIR="$output_dir" \
    STEPS="$STEPS" TRAIN_BATCH=8 N_ROLLOUTS=16 \
    ROLLOUT_MICRO_BATCH=128 LOGPROB_MICRO_BATCH=8 PPO_MICRO_BATCH=4 \
    MAX_RESPONSE_LENGTH=128 VAL_BEFORE_TRAIN=false TEST_FREQ=-1 \
    VAL_ON_LAST_STEP=false SAVE_FREQ="$STEPS" \
    SAVE_CONTENTS="['model','optimizer','extra','hf_model']" \
    FILTER_OVERLONG=false REWARD_MANAGER=dapo GRADIENT_CHECKPOINTING=false \
    SEED="$seed" LR=1e-5 HINDSIGHT_ENABLE=false \
    HINDSIGHT_MAX_GROUPS=8 HINDSIGHT_ONE_TARGET=false \
    HINDSIGHT_UTILITY_GATE=false \
    LIVE_REPLAY_ENABLE=true LIVE_REPLAY_SCHEDULE="$b2_audit" \
    LIVE_REPLAY_STRICT=true LIVE_REPLAY_MAX_TOKEN_MISMATCH=0.05 \
    LIVE_REPLAY_BUFFER_GROUPS=0 LIVE_REPLAY_MAX_BUFFER_AGE=8 \
    LIVE_REPLAY_RESERVOIR="$RESERVOIR_PATH" \
    LIVE_REPLAY_RESERVOIR_SHA256="$digest" \
    LIVE_REPLAY_AUDIT_PATH="$output_dir/replay_accounting.jsonl" \
    REPLAY_RESERVOIR_COLLECT=false \
    bash "$SCRIPT_DIR/countdown_rtx5090.sh" 2>&1 | tee "$log_path"
  verify_checkpoint "$run_id" "$output_dir" "$log_path"
  if [[ $(wc -l < "$output_dir/replay_accounting.jsonl") -ne "$STEPS" ]]; then
    echo "E2c audit line count failed: $run_id" >&2
    exit 1
  fi
  touch "$marker"
}

validate_delivery() {
  local output="$ITERATION_DIR/E2C_DELIVERY.json"
  local args=()
  for seed in $SEEDS; do
    args+=(--audit "$RUNTIME_ROOT/checkpoints/e2c_reservoir_replay_s${seed}_260810/replay_accounting.jsonl")
    args+=(--schedule "$RUNTIME_ROOT/checkpoints/e2_clean_b2_s${seed}_260809/dose_accounting.jsonl")
    args+=(--seed "$seed")
  done
  PYTHONPATH="$RESEARCH_ROOT" "$PYTHON_BIN" \
    -m curriculum_maxrl.countdown.analyze_e2c \
    --output "$output" "${args[@]}"
}

evaluate_arm() {
  local arm=$1
  local seed=$2
  local run_id
  if [[ "$arm" == e2c ]]; then
    run_id="e2c_reservoir_replay_s${seed}_260810"
  else
    run_id="e2_clean_${arm}_s${seed}_260809"
  fi
  local model="$RUNTIME_ROOT/checkpoints/$run_id/global_step_${STEPS}/actor/huggingface"
  local output="$RESULT_DIR/${run_id}_eval.json"
  local raw_output="${output%.json}.jsonl"
  local eval_log="$RESULT_DIR/${run_id}_eval.log"
  if [[ -f "$output" ]]; then
    [[ -f "$raw_output" ]] || {
      echo "evaluation summary exists without raw task outcomes: $output" >&2
      exit 1
    }
    echo "evaluation exists, skipping: $output"
    return
  fi
  require_gpu_clear
  if ! PYTHONPATH="$RESEARCH_ROOT" HF_HOME="$RUNTIME_ROOT/hf" \
    "$PYTHON_BIN" -m curriculum_maxrl.countdown.eval_countdown \
      --model "$model" --data "$TEST_DATA" --output "$output" \
      --k 16 --batch-size 8 --max-new-tokens 128 \
      --temperature 1.0 --top-p 1.0 --seed "$((10000 + seed))" \
      >"$eval_log" 2>&1; then
    echo "evaluation failed before matrix completion: $run_id" >&2
    tail -80 "$eval_log" >&2
    exit 1
  fi
  echo "evaluation sealed pending complete nine-arm analysis: $run_id"
}

for seed in $SEEDS; do
  run_comparator b1 "$seed"
  run_comparator b2 "$seed"
done
# Deeply revalidate every reusable comparator, including the newly completed
# seed-3 pair, before generating the shared treatment source.
run_readiness_audit
collect_reservoir
run_preflight
# Freeze a post-preflight receipt before any E2c optimizer update.
run_readiness_audit
digest=$(reservoir_sha256)
for seed in $SEEDS; do
  run_e2c "$seed" "$digest"
done
validate_delivery
# Revalidate all training/delivery artifacts at the no-outcome barrier.
run_readiness_audit
"$PYTHON_BIN" -c '
import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
if receipt.get("integrity_status") != "pass":
    raise SystemExit("E2c integrity receipt does not pass")
if receipt.get("delivery_gate", {}).get("status") != "pass":
    raise SystemExit("E2c delivery receipt does not pass")
if receipt.get("next_stage") not in {
        "generate_paired_heldout_endpoints",
        "analyze_and_propagate_e2c_verdict"}:
    raise SystemExit("E2c receipt does not authorize endpoint work")
' "$ITERATION_DIR/E2C_LAUNCH_READINESS.json"
# Held-out generation is deliberately last and cannot run unless all three
# delivery audits pass the validator above.
for seed in $SEEDS; do
  evaluate_arm b1 "$seed"
  evaluate_arm b2 "$seed"
  evaluate_arm e2c "$seed"
done

analysis_args=()
for seed in $SEEDS; do
  analysis_args+=(--result "$seed,b1,$RESULT_DIR/e2_clean_b1_s${seed}_260809_eval.json")
  analysis_args+=(--result "$seed,b2,$RESULT_DIR/e2_clean_b2_s${seed}_260809_eval.json")
  analysis_args+=(--result "$seed,e2c,$RESULT_DIR/e2c_reservoir_replay_s${seed}_260810_eval.json")
done
PYTHONPATH="$RESEARCH_ROOT" "$PYTHON_BIN" \
  -m curriculum_maxrl.countdown.analyze_e2c_endpoints \
  --delivery "$ITERATION_DIR/E2C_DELIVERY.json" \
  --readiness "$ITERATION_DIR/E2C_LAUNCH_READINESS.json" \
  --output "$ITERATION_DIR/E2C_ENDPOINTS.json" "${analysis_args[@]}"

echo "E2c execution and paired endpoint generation complete"
