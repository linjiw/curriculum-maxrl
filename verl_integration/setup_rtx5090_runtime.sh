#!/usr/bin/env bash
# Rebuild the exact single-RTX-5090 MaxRL/verl runtime used by the local
# Countdown recovery. Model/data preparation is handled separately by
# prepare_countdown_assets.sh.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RESEARCH_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
RUNTIME_ROOT=${RUNTIME_ROOT:-/data/robotixx/curriculum-maxrl-runtime}
MAXRL_ROOT=${MAXRL_ROOT:-$RUNTIME_ROOT/maxrl}
VENV_ROOT=${VENV_ROOT:-$RUNTIME_ROOT/venv}
BASE_PYTHON=${BASE_PYTHON:-/home/robotixx/miniconda3/envs/env_isaaclab/bin/python}
MAXRL_URL=${MAXRL_URL:-https://github.com/tajwarfahim/maxrl.git}
MAXRL_COMMIT=${MAXRL_COMMIT:-7197bbb46a2ecd866da52f6b401ff20a34fe9390}
FLASH_WHEEL=${FLASH_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl}

mkdir -p "$RUNTIME_ROOT"
if [[ ! -d "$MAXRL_ROOT/.git" ]]; then
  git clone "$MAXRL_URL" "$MAXRL_ROOT"
  git -C "$MAXRL_ROOT" checkout --detach "$MAXRL_COMMIT"
fi

actual_commit=$(git -C "$MAXRL_ROOT" rev-parse HEAD)
if [[ "$actual_commit" != "$MAXRL_COMMIT" ]]; then
  echo "Refusing to patch $actual_commit; expected $MAXRL_COMMIT" >&2
  exit 1
fi

if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV_ROOT"
fi
PYTHON_BIN="$VENV_ROOT/bin/python"

"$PYTHON_BIN" -m pip install \
  codetiming==1.4.0 \
  torchdata==0.11.0 \
  peft==0.17.1 \
  math-verify==0.8.0 \
  pylatexenc==2.10 \
  py-spy==0.4.2
if ! "$PYTHON_BIN" -c 'import flash_attn; assert flash_attn.__version__ == "2.8.3.post1"' >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install --no-deps "$FLASH_WHEEL"
fi
"$PYTHON_BIN" -m pip install --no-deps -e "$MAXRL_ROOT"

apply_once() {
  local patch_path=$1
  if git -C "$MAXRL_ROOT" apply --reverse --check "$patch_path" >/dev/null 2>&1; then
    echo "already applied: $(basename "$patch_path")"
  elif git -C "$MAXRL_ROOT" apply --check "$patch_path"; then
    git -C "$MAXRL_ROOT" apply "$patch_path"
    echo "applied: $(basename "$patch_path")"
  else
    echo "Patch is neither applicable nor already applied: $patch_path" >&2
    exit 1
  fi
}

patchset_present() {
  grep -Fq 'curriculum_teacher = FrontierTeacher(' "$MAXRL_ROOT/verl/trainer/main_ppo.py" &&
    grep -Fq 'ray_init_kwargs["num_gpus"] = config.ray_init.num_gpus' "$MAXRL_ROOT/verl/trainer/main_ppo.py" &&
    grep -Fq 'Curriculum teacher state loaded from' "$MAXRL_ROOT/verl/trainer/ppo/ray_trainer.py" &&
    grep -Fq 'self.hindsight = CountdownHindsight(' "$MAXRL_ROOT/verl/trainer/ppo/ray_trainer.py" &&
    grep -Fq 'self.live_replay = DoseMatchedLiveReplay(' "$MAXRL_ROOT/verl/trainer/ppo/ray_trainer.py" &&
    grep -Fq 'self.replay_reservoir_collector = ReplayReservoirCollector(' "$MAXRL_ROOT/verl/trainer/ppo/ray_trainer.py" &&
    grep -Fq 'backend = "gloo" if world_size == 1' "$MAXRL_ROOT/verl/workers/fsdp_workers.py" &&
    grep -Fq 'if torch.distributed.get_world_size() == 1:' "$MAXRL_ROOT/verl/workers/fsdp_workers.py" &&
    grep -Fq 'worker_seed = int(self.config.get("seed", 1))' "$MAXRL_ROOT/verl/workers/fsdp_workers.py" &&
    grep -Fq 'dist.get_world_size() == 1' "$MAXRL_ROOT/verl/utils/debug/performance.py" &&
    grep -Fq 'self.overlong_buffer_cfg is not None' "$MAXRL_ROOT/verl/workers/reward_manager/dapo.py" &&
    ! grep -Fq '[HFRollout] Starting generate' "$MAXRL_ROOT/verl/workers/rollout/hf_rollout.py"
}

PATCHES=(
  main_ppo.patch
  ray_trainer.patch
  ray_num_gpus.patch
  fsdp_single_gpu.patch
  hf_rollout_no_param_scan.patch
  single_gpu_reduce_timing.patch
  dapo_optional_overlong.patch
  hindsight_trainer.patch
  dose_matched_replay.patch
  e2c_reservoir.patch
  deterministic_worker_seed.patch
)
if patchset_present; then
  echo "full MaxRL patch set already present"
else
  for patch_name in "${PATCHES[@]}"; do
    apply_once "$SCRIPT_DIR/$patch_name"
  done
  if ! patchset_present; then
    echo "Patch application finished but semantic validation failed" >&2
    exit 1
  fi
fi

install_link() {
  local source_path=$1
  local target_path=$2
  if [[ -e "$target_path" && ! -L "$target_path" ]]; then
    echo "Refusing to replace non-symlink: $target_path" >&2
    exit 1
  fi
  ln -sfn "$source_path" "$target_path"
}

install_link "$RESEARCH_ROOT/curriculum_maxrl/verl_curriculum.py" "$MAXRL_ROOT/verl/utils/curriculum.py"
install_link "$RESEARCH_ROOT/verl_integration/vendored/hindsight.py" "$MAXRL_ROOT/verl/utils/hindsight.py"

PYTHONPATH="$RESEARCH_ROOT:$MAXRL_ROOT" "$PYTHON_BIN" - <<'PY'
import flash_attn
import torch
import verl.trainer.main_ppo
from verl.utils.hindsight import CountdownHindsight

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.get_device_capability(0) == (12, 0), torch.cuda.get_device_capability(0)
print({
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "flash_attn": flash_attn.__version__,
    "torch": torch.__version__,
})
PY

echo "Runtime ready: $MAXRL_ROOT"
echo "Next: bash $SCRIPT_DIR/prepare_countdown_assets.sh"
