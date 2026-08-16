#!/usr/bin/env bash
# Launch one minimax maze arm from an upstream config with explicit overrides.
#
# Baselines are the upstream shipped configs, used verbatim except for the
# overrides passed here, so any comparison is against the authors' own tuned
# settings rather than a reimplementation.
#
# usage:
#   run_arm.sh <base_config> <xpid_suffix> <log_dir> <n_updates> <seed> [k=v ...]
# example:
#   run_arm.sh plr maxmc32x1 /tmp/pilot 3000 1001
#   run_arm.sh plr frontN16 /tmp/pilot 3000 1001 \
#       n_parallel=4 n_eval=8 plr_buffer_size=500 \
#       ued_score=coefficient_activity frontier_n_rollouts=16 \
#       frontier_require_n_eval_match=False
set -euo pipefail

PY=${MINIMAX_PY:-/data/robotixx/ued_bench/envs/jax062-cuda129-probe/bin/python}
SRC=${MINIMAX_SRC:-/data/robotixx/ued_bench/src/minimax-frontier-blackwell-training-jax062-5868d346-d053054}

BASE=${1:?base config, e.g. plr}
SUFFIX=${2:?xpid suffix}
LOGDIR=${3:?log dir}
UPDATES=${4:?n_total_updates}
SEED=${5:?seed}
shift 5

mkdir -p "$LOGDIR"
cd "$SRC"
export PYTHONPATH="$SRC/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CMD=$("$PY" -m minimax.config.make_cmd --config "maze/$BASE" 2>/dev/null | tr -d '\\\n')

# Fixed overrides for every pilot/campaign arm.
CMD=$(printf '%s' "$CMD" \
  | sed "s#--log_dir=[^ ]*#--log_dir=$LOGDIR#" \
  | sed "s#--n_total_updates=[0-9]*#--n_total_updates=$UPDATES#" \
  | sed "s#--seed=[0-9]*#--seed=$SEED#" \
  | sed "s#--checkpoint_interval=[0-9]*#--checkpoint_interval=$UPDATES#" \
  | sed "s#--log_interval=[0-9]*#--log_interval=50#")

# Caller overrides: replace the flag if present, otherwise append it.
for kv in "$@"; do
  k=${kv%%=*}; v=${kv#*=}
  if printf '%s' "$CMD" | grep -q -- "--$k="; then
    CMD=$(printf '%s' "$CMD" | sed "s#--$k=[^ ]*#--$k=$v#")
  else
    CMD="$CMD --$k=$v"
  fi
done

# One unambiguous xpid so arms never collide in the log directory.
XPID="arm-${SUFFIX}-s${SEED}-u${UPDATES}"
CMD=$(printf '%s' "$CMD" | sed "s#--xpid=[^ ]*#--xpid=$XPID#")

printf '%s\n' "$CMD" > "$LOGDIR/${XPID}.cmd"
echo "[$(date +%H:%M:%S)] $XPID"
t0=$SECONDS
eval "${CMD/#python /\"$PY\" }" > "$LOGDIR/${XPID}.out" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] $XPID rc=$rc elapsed=$((SECONDS-t0))s"
exit $rc
