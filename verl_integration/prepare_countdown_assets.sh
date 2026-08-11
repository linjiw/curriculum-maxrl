#!/usr/bin/env bash
# Download the frozen base model, build the leakage-free Countdown v2 splits,
# and train the completion-only SFT warmstart used by countdown_rtx5090.sh.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RESEARCH_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
RUNTIME_ROOT=${RUNTIME_ROOT:-/data/robotixx/curriculum-maxrl-runtime}
PYTHON_BIN=${PYTHON_BIN:-$RUNTIME_ROOT/venv/bin/python}
MODEL_REPO=${MODEL_REPO:-HuggingFaceTB/SmolLM2-360M-Instruct}
MODEL_REVISION=${MODEL_REVISION:-a10cc1512eabd3dde888204e902eca88bddb4951}
DATA_DIR=${DATA_DIR:-$RUNTIME_ROOT/data/countdown_v2_rebuilt}
SFT_DIR=${SFT_DIR:-$RUNTIME_ROOT/models/countdown_sft_clean_v1}
FORCE=${FORCE:-false}

export HF_HOME=${HF_HOME:-$RUNTIME_ROOT/hf}
export PYTHONPATH="$RESEARCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"

BASE_MODEL=$("$PYTHON_BIN" -c \
  'from huggingface_hub import snapshot_download; import sys; print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2]))' \
  "$MODEL_REPO" "$MODEL_REVISION")
echo "Frozen base model: $BASE_MODEL"

if [[ "$FORCE" == true || ! -f "$DATA_DIR/manifest.json" ]]; then
  "$PYTHON_BIN" -m curriculum_maxrl.countdown.prep_countdown \
    --out-dir "$DATA_DIR" \
    --seed 7
else
  echo "Dataset already exists: $DATA_DIR"
fi

if [[ "$FORCE" == true || ! -f "$SFT_DIR/training_metrics.json" ]]; then
  "$PYTHON_BIN" -m curriculum_maxrl.countdown.sft_countdown \
    --model "$BASE_MODEL" \
    --data "$DATA_DIR/sft_train.jsonl" \
    --output-dir "$SFT_DIR" \
    --max-length 384 \
    --batch-size 16 \
    --gradient-accumulation 2 \
    --epochs 1 \
    --learning-rate 5e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.10 \
    --seed 2026
else
  echo "SFT checkpoint already exists: $SFT_DIR"
fi

sha256sum "$DATA_DIR/test.parquet" "$DATA_DIR/sft_train.jsonl"
echo "Assets ready. Run: STEPS=1 bash $SCRIPT_DIR/countdown_rtx5090.sh"
