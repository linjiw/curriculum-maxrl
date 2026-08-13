#!/usr/bin/env bash
# GATE-DR launcher: byte-faithful derivative of verl_integration/countdown_rtx5090.sh
# (frozen in E2C_CODE_MANIFEST — do not edit the original) adding one additive
# parameter: HINDSIGHT_GATE_MAX_P -> +algorithm.hindsight.gate_max_p.
#
# Required runtime assets are kept off the full root filesystem under /data.
# Override any upper-case variable from the environment. Example smoke run:
#   STEPS=1 TRAIN_BATCH=8 VAL_N=4 bash verl_integration/countdown_rtx5090.sh

set -euo pipefail

RUNTIME_ROOT=${RUNTIME_ROOT:-/data/robotixx/curriculum-maxrl-runtime}
MAXRL_ROOT=${MAXRL_ROOT:-$RUNTIME_ROOT/maxrl}
PYTHON_BIN=${PYTHON_BIN:-$RUNTIME_ROOT/venv/bin/python}
MODEL_PATH=${MODEL_PATH:-$RUNTIME_ROOT/models/countdown_sft_clean_v1}
TRAIN_DATA=${TRAIN_DATA:-$RUNTIME_ROOT/data/countdown_v2_rebuilt/train.parquet}
VAL_DATA=${VAL_DATA:-$RUNTIME_ROOT/data/countdown_v2_rebuilt/test.parquet}
REWARD_PATH=${REWARD_PATH:-$(cd "$(dirname "$0")/.." && pwd)/curriculum_maxrl/countdown/countdown_reward.py}
RUN_ID=${RUN_ID:-maxrl_baseline_smoke}
OUTPUT_DIR=${OUTPUT_DIR:-$RUNTIME_ROOT/checkpoints/$RUN_ID}
# Ray's Unix-domain socket path is capped at 107 bytes; keep this deliberately short.
RAY_DIR=${RAY_DIR:-/data/robotixx/cmrl-ray/${RUN_ID:0:16}}
TRAIN_BATCH=${TRAIN_BATCH:-8}
N_ROLLOUTS=${N_ROLLOUTS:-16}
VAL_N=${VAL_N:-4}
STEPS=${STEPS:-1}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
ROLLOUT_MICRO_BATCH=${ROLLOUT_MICRO_BATCH:-128}
LOGPROB_MICRO_BATCH=${LOGPROB_MICRO_BATCH:-8}
PPO_MICRO_BATCH=${PPO_MICRO_BATCH:-4}
FILTER_OVERLONG=${FILTER_OVERLONG:-false}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
TEST_FREQ=${TEST_FREQ:-1}
VAL_ON_LAST_STEP=${VAL_ON_LAST_STEP:-true}
SAVE_FREQ=${SAVE_FREQ:--1}
SAVE_CONTENTS=${SAVE_CONTENTS:-"['model','optimizer','extra']"}
REWARD_MANAGER=${REWARD_MANAGER:-dapo}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
HINDSIGHT_ENABLE=${HINDSIGHT_ENABLE:-false}
HINDSIGHT_MAX_GROUPS=${HINDSIGHT_MAX_GROUPS:-8}
HINDSIGHT_ONE_TARGET=${HINDSIGHT_ONE_TARGET:-false}
HINDSIGHT_UTILITY_GATE=${HINDSIGHT_UTILITY_GATE:-false}
HINDSIGHT_GATE_MAX_P=${HINDSIGHT_GATE_MAX_P:-0.5}
HINDSIGHT_AUDIT_PATH=${HINDSIGHT_AUDIT_PATH:-$OUTPUT_DIR/dose_accounting.jsonl}
LIVE_REPLAY_ENABLE=${LIVE_REPLAY_ENABLE:-false}
LIVE_REPLAY_SCHEDULE=${LIVE_REPLAY_SCHEDULE:-}
LIVE_REPLAY_AUDIT_PATH=${LIVE_REPLAY_AUDIT_PATH:-$OUTPUT_DIR/replay_accounting.jsonl}
LIVE_REPLAY_MAX_TOKEN_MISMATCH=${LIVE_REPLAY_MAX_TOKEN_MISMATCH:-0.05}
LIVE_REPLAY_STRICT=${LIVE_REPLAY_STRICT:-true}
LIVE_REPLAY_BUFFER_GROUPS=${LIVE_REPLAY_BUFFER_GROUPS:-0}
LIVE_REPLAY_MAX_BUFFER_AGE=${LIVE_REPLAY_MAX_BUFFER_AGE:-8}
LIVE_REPLAY_RESERVOIR=${LIVE_REPLAY_RESERVOIR:-}
LIVE_REPLAY_RESERVOIR_SHA256=${LIVE_REPLAY_RESERVOIR_SHA256:-}
REPLAY_RESERVOIR_COLLECT=${REPLAY_RESERVOIR_COLLECT:-false}
REPLAY_RESERVOIR_OUTPUT=${REPLAY_RESERVOIR_OUTPUT:-$OUTPUT_DIR/replay_reservoir.pt}
REPLAY_RESERVOIR_MAX_GROUPS=${REPLAY_RESERVOIR_MAX_GROUPS:-256}
REPLAY_RESERVOIR_GROUP_SIZE=${REPLAY_RESERVOIR_GROUP_SIZE:-$N_ROLLOUTS}
REPLAY_RESERVOIR_SEED=${REPLAY_RESERVOIR_SEED:-424242}
LR=${LR:-1e-5}
SEED=${SEED:-1}

export HF_HOME=${HF_HOME:-$RUNTIME_ROOT/hf}
export PYTHONPATH=$(cd "$(dirname "$0")/.." && pwd):$MAXRL_ROOT${PYTHONPATH:+:$PYTHONPATH}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$OUTPUT_DIR" "$RAY_DIR"
cd "$MAXRL_ROOT"

exec "$PYTHON_BIN" -m verl.trainer.main_ppo \
  ray_init.ray_dir="$RAY_DIR" \
  ray_init.num_cpus=32 \
  +ray_init.num_gpus=1 \
  algorithm.adv_estimator=maxrl \
  algorithm.use_kl_in_reward=false \
  algorithm.pass_k="$N_ROLLOUTS" \
  algorithm.truncate_order="$N_ROLLOUTS" \
  algorithm.kl_ctrl.kl_coef=0.0 \
  +algorithm.hindsight.enable="$HINDSIGHT_ENABLE" \
  +algorithm.hindsight.scale=1.0 \
  +algorithm.hindsight.max_groups_per_step="$HINDSIGHT_MAX_GROUPS" \
  +algorithm.hindsight.one_target_per_group="$HINDSIGHT_ONE_TARGET" \
  +algorithm.hindsight.utility_gate="$HINDSIGHT_UTILITY_GATE" \
  +algorithm.hindsight.gate_max_p="$HINDSIGHT_GATE_MAX_P" \
  +algorithm.hindsight.audit_path="$HINDSIGHT_AUDIT_PATH" \
  +algorithm.live_replay.enable="$LIVE_REPLAY_ENABLE" \
  +algorithm.live_replay.schedule_path="$LIVE_REPLAY_SCHEDULE" \
  +algorithm.live_replay.seed="$SEED" \
  +algorithm.live_replay.max_cumulative_token_mismatch_fraction="$LIVE_REPLAY_MAX_TOKEN_MISMATCH" \
  +algorithm.live_replay.strict="$LIVE_REPLAY_STRICT" \
  +algorithm.live_replay.buffer_capacity_groups="$LIVE_REPLAY_BUFFER_GROUPS" \
  +algorithm.live_replay.max_buffer_age_steps="$LIVE_REPLAY_MAX_BUFFER_AGE" \
  +algorithm.live_replay.reservoir_path="$LIVE_REPLAY_RESERVOIR" \
  +algorithm.live_replay.reservoir_sha256="$LIVE_REPLAY_RESERVOIR_SHA256" \
  +algorithm.live_replay.audit_path="$LIVE_REPLAY_AUDIT_PATH" \
  +algorithm.replay_reservoir.enable="$REPLAY_RESERVOIR_COLLECT" \
  +algorithm.replay_reservoir.output_path="$REPLAY_RESERVOIR_OUTPUT" \
  +algorithm.replay_reservoir.seed="$REPLAY_RESERVOIR_SEED" \
  +algorithm.replay_reservoir.max_groups="$REPLAY_RESERVOIR_MAX_GROUPS" \
  +algorithm.replay_reservoir.expected_group_size="$REPLAY_RESERVOIR_GROUP_SIZE" \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size="$TRAIN_BATCH" \
  +data.seed="$SEED" \
  data.filter_overlong_prompts="$FILTER_OVERLONG" \
  data.max_prompt_length=256 \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  +data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  +actor_rollout_ref.seed="$SEED" \
  actor_rollout_ref.model.attn_implementation=flash_attention_2 \
  actor_rollout_ref.model.enable_gradient_checkpointing="$GRADIENT_CHECKPOINTING" \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH" \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.checkpoint.save_contents="$SAVE_CONTENTS" \
  actor_rollout_ref.rollout.name=hf \
  +actor_rollout_ref.rollout.micro_batch_size="$ROLLOUT_MICRO_BATCH" \
  actor_rollout_ref.rollout.n="$N_ROLLOUTS" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=true \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=0 \
  reward_model.reward_manager="$REWARD_MANAGER" \
  custom_reward_function.path="$REWARD_PATH" \
  custom_reward_function.name=compute_score \
  trainer.project_name=CurriculumMaxRL_Countdown_RTX5090 \
  trainer.experiment_name="$RUN_ID" \
  trainer.logger="['console']" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.val_on_last_step="$VAL_ON_LAST_STEP" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=200 \
  trainer.total_training_steps="$STEPS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.default_local_dir="$OUTPUT_DIR"
