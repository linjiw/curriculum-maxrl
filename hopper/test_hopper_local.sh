#!/usr/bin/env bash
# Local behavior test for hopper.sh. All network and Slurm commands are mocked.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
test_root=$(mktemp -d /tmp/maxrl-hopper-test.XXXXXX)
trap 'rm -rf -- "$test_root"' EXIT

export MOCK_REMOTE="$test_root/remote"
export MOCK_TRACE="$test_root/trace"
mkdir -p -- "$MOCK_REMOTE"

map_remote_path() {
  local value=$1
  value=${value//\/scratch\/mock/$MOCK_REMOTE}
  printf '%s\n' "$value"
}

ssh() {
  while [[ ${1:-} == -o ]]; do shift 2; done
  [[ $# -ge 2 ]] || return 90
  shift # host
  if [[ $# == 1 ]]; then
    if [[ $1 == *'__HOPPER_USER__'* ]]; then
      printf '__HOPPER_USER__mock\n'
      return
    fi
    local command=${1//\/scratch\/mock/$MOCK_REMOTE}
    bash -c "$command"
    return
  fi
  [[ $1 == bash && $2 == -s && $3 == -- ]] || return 91
  shift 3
  local mapped=() arg
  for arg in "$@"; do mapped+=("$(map_remote_path "$arg")"); done
  bash -s -- "${mapped[@]}"
}

scp() {
  local args=("$@") count=${#args[@]} source destination
  source=${args[count-2]}
  destination=${args[count-1]}
  if [[ $source == *:* ]]; then
    source=${source#*:}
    source=$(map_remote_path "$source")
  fi
  if [[ $destination == *:* ]]; then
    destination=${destination#*:}
    destination=$(map_remote_path "$destination")
  fi
  cp -a -- "$source" "$destination"
}

rsync() {
  local args=("$@") count=${#args[@]} source destination
  source=${args[count-2]}
  destination=${args[count-1]}
  if [[ $source == *:* ]]; then
    source=${source#*:}
    source=$(map_remote_path "$source")
  fi
  if [[ $destination == *:* ]]; then
    destination=${destination#*:}
    destination=$(map_remote_path "$destination")
  fi
  if [[ $source == */ ]]; then
    cp -a -- "$source." "$destination"
  else
    cp -a -- "$source" "$destination"
  fi
}

sbatch() {
  printf '%s\n' "$*" > "$MOCK_TRACE.sbatch"
  printf '424242\n'
}

scontrol() {
  local name=${MOCK_SCONTROL_NAME:-maxrl-io-smoke}
  local stdout=${MOCK_SCONTROL_STDOUT:-/scratch/mock/maxrl/tests/logs/maxrl-io-smoke_424242.out}
  printf 'JobId=424242 JobName=%s StdOut=%s\n' "$name" "$stdout"
}

sacct() {
  if [[ " $* " == *' JobIDRaw,JobName,Partition,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,NodeList,Submit,Start,End,AllocTRES,QOS,TimelimitRaw,Restarts,WorkDir,StdOut,StdErr,SubmitLine '* ]]; then
    printf '%s\n' "${MOCK_TERMINAL_ROW:-424242|ued-minimax-terminal-chain|gpuq|COMPLETED|0:0|102|2|15G|gpu021|2026-08-14T00:00:00|2026-08-14T00:01:00|2026-08-14T00:02:42|billing=20,cpu=2,gres/gpu:1g.10gb=1,gres/gpu=1,mem=15G,node=1|gpu|30|0|/scratch/mock/maxrl|/scratch/%u/maxrl/tests/logs/%x_%j.out||sbatch --parsable --export=ALL,X=Y /scratch/mock/maxrl/sbatch/terminal.sbatch}"
    if [[ -n "${MOCK_SECOND_TERMINAL_ROW:-}" ]]; then
      printf '%s\n' "$MOCK_SECOND_TERMINAL_ROW"
    fi
  elif [[ " $* " == *' JobIDRaw,MaxRSS,TRESUsageInMax '* ]]; then
    printf '%s\n' "${MOCK_RESOURCE_ROWS:-424242||
424242.batch|123456K|cpu=00:01:30,gres/gpumem=2048M}"
  elif [[ " $* " == *' JobName,StdOut '* ]]; then
    printf '%s|%s|\n' "${MOCK_JOB_NAME:-maze-score}" \
      "${MOCK_SACCT_STDOUT:-/scratch/%u/maxrl/tests/logs/maze-score_424242.out}"
  elif [[ " $* " == *' StdOut '* ]]; then
    printf '%s|\n' \
      "${MOCK_SACCT_STDOUT:-/scratch/%u/maxrl/tests/logs/maze-score_424242.out}"
  else
    printf 'COMPLETED|\n'
  fi
}

sinfo() {
  printf 'partition=normal availability=up timelimit=1-00:00:00 nodes=1 state=idle\n'
}

squeue() {
  printf 'JOBID NAME PARTITION STATE\n'
}

export -f map_remote_path ssh scp rsync sbatch scontrol sacct sinfo squeue
export HOPPER_HOST=mock@hopper.example
export HOPPER_SCRATCH=/scratch/mock
export HOPPER_REGISTRY="$test_root/job_registry.tsv"

"$here/hopper.sh" submit "$here/sbatch/workflow_io_smoke.sbatch" \
  > "$test_root/submit.out"
grep -q '^--parsable ' "$MOCK_TRACE.sbatch"
[[ $(awk -F '\t' 'NR == 1 {print NF}' "$HOPPER_REGISTRY") == 10 ]]
remote_script=$(awk -F '\t' 'NR == 1 {print $6}' "$HOPPER_REGISTRY")
remote_script=$(map_remote_path "$remote_script")
[[ -f "$remote_script" ]]
[[ -d "$MOCK_REMOTE/maxrl/tests/logs" ]]
compgen -G "$MOCK_REMOTE/maxrl/receipts/job-424242-*.tsv" >/dev/null

if "$here/hopper.sh" submit "$here/sbatch/workflow_io_smoke.sbatch" --mem=1G \
    >/dev/null 2>&1; then
  printf 'sbatch resource override was not rejected\n' >&2
  exit 1
fi

if "$here/hopper.sh" submit "$here/sbatch/ued_minimax_terminal_chain_smoke.sbatch" \
    --export=ALL,UED_BUNDLE_DIR=/scratch/mock/bundle >/dev/null 2>&1; then
  printf 'terminal-chain ambient export was not rejected\n' >&2
  exit 1
fi
if "$here/hopper.sh" submit "$here/sbatch/ued_minimax_terminal_chain_smoke.sbatch" \
    --export=UED_BUNDLE_DIR=/scratch/mock/bundle >/dev/null 2>&1; then
  printf 'incomplete terminal-chain export allowlist was not rejected\n' >&2
  exit 1
fi

result="$MOCK_REMOTE/maxrl/tests/results/424242"
mkdir -p -- "$result"
printf 'key\tvalue\njob_id\t424242\n' > "$result/receipt.tsv"
printf 'payload\n' > "$result/payload.txt"
printf 'complete\n' > "$result/COMPLETE"
(
  cd "$result"
  sha256sum receipt.tsv payload.txt > SHA256SUMS
  sha256sum -c --strict SHA256SUMS >/dev/null
)

"$here/hopper.sh" fetch /scratch/mock/maxrl/tests/results/424242 \
  "$test_root/fetched" "$test_root/fetch.tsv" > "$test_root/fetch.out"
[[ -f "$test_root/fetched/COMPLETE" ]]
grep -Fxq $'fetch_receipt_schema\t2' "$test_root/fetch.tsv"
grep -Fxq $'terminal_receipt_sha256\t-' "$test_root/fetch.tsv"
(
  cd "$test_root/fetched"
  sha256sum -c --strict SHA256SUMS >/dev/null
)

if "$here/hopper.sh" fetch /scratch/mock/maxrl/tests/results/424242 \
  "$test_root/fetched" >/dev/null 2>&1; then
  printf 'existing fetch destination was not rejected\n' >&2
  exit 1
fi

export MOCK_SCONTROL_NAME=maze-score
export MOCK_SCONTROL_STDOUT=/scratch/mock/maxrl/tests/logs/maze-score_424242.out
if "$here/hopper.sh" logs 424242 > /dev/null 2> "$test_root/logs.err"; then
  printf 'maze-score logs were not protected\n' >&2
  exit 1
fi
grep -q -- '--allow-endpoints' "$test_root/logs.err"

export MOCK_JOB_NAME=maze-full-arm-smoke
export MOCK_SCONTROL_NAME=maze-full-arm-smoke
if "$here/hopper.sh" logs 424242 > /dev/null 2> "$test_root/full-arm-logs.err"; then
  printf 'maze full-arm logs were not protected\n' >&2
  exit 1
fi
grep -q -- '--allow-endpoints' "$test_root/full-arm-logs.err"
unset MOCK_JOB_NAME

# sacct may preserve literal Slurm tokens even though scontrol reports the
# resolved path. Logs must use the latter and never look for `/scratch/%u/...`.
export MOCK_SCONTROL_NAME=ued-minimax-gpu-smoke
export MOCK_SCONTROL_STDOUT=/scratch/mock/maxrl/tests/logs/ued-minimax-gpu-smoke_424242.out
export MOCK_SACCT_STDOUT=/scratch/%u/maxrl/tests/logs/ued-minimax-gpu-smoke_424242.out
mkdir -p -- "$MOCK_REMOTE/maxrl/tests/logs"
printf 'resolved-log-line\n' \
  > "$MOCK_REMOTE/maxrl/tests/logs/ued-minimax-gpu-smoke_424242.out"
"$here/hopper.sh" logs 424242 1 > "$test_root/resolved-logs.out"
grep -Fxq 'resolved-log-line' "$test_root/resolved-logs.out"
unset MOCK_SCONTROL_NAME MOCK_SCONTROL_STDOUT MOCK_SACCT_STDOUT

if "$here/hopper.sh" push "$here/hopper.sh" ../escape >/dev/null 2>&1; then
  printf 'unsafe remote path was not rejected\n' >&2
  exit 1
fi

"$here/hopper.sh" health > "$test_root/health.out"
grep -q $'scratch\tOK\t' "$test_root/health.out"

campaign="$MOCK_REMOTE/maxrl/campaigns/example/attempts/attempt-001"
incomplete="$MOCK_REMOTE/maxrl/campaigns/example/incomplete/attempt-001"
mkdir -p -- "$campaign/seed-1/meta" "$campaign/seed-2/meta" \
  "$incomplete/seed-3.job-777"
printf 'complete\n' > "$campaign/seed-1/COMPLETE"
printf 'manifest\n' > "$campaign/seed-1/SHA256SUMS"
printf '{}\n' > "$campaign/seed-1/meta/plugin.DONE.json"
printf '{}\n' > "$campaign/seed-1/meta/grouplaw.DONE.json"
printf 'complete\n' > "$campaign/seed-2/COMPLETE"
printf 'manifest\n' > "$campaign/seed-2/SHA256SUMS"
"$here/hopper.sh" campaign-status \
  /scratch/mock/maxrl/campaigns/example/attempts/attempt-001 3 \
  /scratch/mock/maxrl/campaigns/example/incomplete/attempt-001 \
  > "$test_root/campaign-status.out"
grep -Fxq $'final_blocks\t2' "$test_root/campaign-status.out"
grep -Fxq $'complete_markers\t2' "$test_root/campaign-status.out"
grep -Fxq $'sha256_manifests\t2' "$test_root/campaign-status.out"
grep -Fxq $'arm_done_receipts\t2' "$test_root/campaign-status.out"
grep -Fxq $'incomplete_quarantines\t1' "$test_root/campaign-status.out"
grep -Fxq $'structural_state\tIN_PROGRESS' "$test_root/campaign-status.out"
if "$here/hopper.sh" campaign-status /tmp/outside 3 >/dev/null 2>&1; then
  printf 'campaign status escaped Hopper scratch\n' >&2
  exit 1
fi

export MOCK_SCONTROL_NAME=group-law-flip-v1
export MOCK_SCONTROL_STDOUT=/scratch/mock/maxrl/group_law_flip/logs/group-law-flip-v1_424242.out
if "$here/hopper.sh" logs 424242 --allow-endpoints \
    >/dev/null 2> "$test_root/group-law-logs.err"; then
  printf 'group-law-flip logs were not sealed\n' >&2
  exit 1
fi
grep -q 'logs are sealed' "$test_root/group-law-logs.err"
unset MOCK_SCONTROL_NAME MOCK_SCONTROL_STDOUT

"$here/hopper.sh" watch 424242 5 5 > "$test_root/watch.out"
grep -q 'job=424242 state=COMPLETED' "$test_root/watch.out"

export HOPPER_LOCAL_RESULTS_ROOT="$test_root"
"$here/hopper.sh" terminal-receipt 424242 "$test_root/terminal-sacct.tsv" \
  > "$test_root/terminal-receipt.out"
[[ -f "$test_root/terminal-sacct.tsv" ]]
[[ "$(stat -c '%a' "$test_root/terminal-sacct.tsv")" == 600 ]]
grep -Fxq $'terminal_receipt_schema\t2' "$test_root/terminal-sacct.tsv"
grep -Eq $'^retrieved_epoch\t[0-9]+$' "$test_root/terminal-sacct.tsv"
grep -Eq $'^terminal_end_epoch\t[0-9]+$' "$test_root/terminal-sacct.tsv"
grep -Fq $'terminal_row\t424242|ued-minimax-terminal-chain|gpuq|COMPLETED|0:0|102|' \
  "$test_root/terminal-sacct.tsv"
[[ "$(grep -c $'^resource_row\t' "$test_root/terminal-sacct.tsv")" == 2 ]]

"$here/hopper.sh" fetch /scratch/mock/maxrl/tests/results/424242 \
  "$test_root/fetched-terminal" "$test_root/fetch-terminal.tsv" \
  "$test_root/terminal-sacct.tsv" > "$test_root/fetch-terminal.out"
terminal_sha=$(sha256sum "$test_root/terminal-sacct.tsv" | awk '{print $1}')
grep -Fxq $'fetch_receipt_schema\t2' "$test_root/fetch-terminal.tsv"
grep -Fxq $'terminal_receipt_sha256\t'"$terminal_sha" "$test_root/fetch-terminal.tsv"
awk -F '\t' '
  $1 == "fetch_started_epoch" { start=$2 }
  $1 == "terminal_end_epoch" { end=$2 }
  END { exit !(start ~ /^[0-9]+$/ && end ~ /^[0-9]+$/ && start + 0 >= end + 0) }
' "$test_root/fetch-terminal.tsv"

if "$here/hopper.sh" terminal-receipt 424242 "$test_root/terminal-sacct.tsv" \
    >/dev/null 2>&1; then
  printf 'existing terminal receipt destination was not rejected\n' >&2
  exit 1
fi
if "$here/hopper.sh" terminal-receipt 42_array "$test_root/bad-job.tsv" \
    >/dev/null 2>&1; then
  printf 'nonnumeric terminal receipt job ID was not rejected\n' >&2
  exit 1
fi
mkdir -p "$test_root/confined"
export HOPPER_LOCAL_RESULTS_ROOT="$test_root/confined"
if "$here/hopper.sh" terminal-receipt 424242 "$test_root/outside.tsv" \
    >/dev/null 2>&1; then
  printf 'terminal receipt escaped its configured local root\n' >&2
  exit 1
fi
export HOPPER_LOCAL_RESULTS_ROOT="$test_root"
export MOCK_TERMINAL_ROW='424242|ued-minimax-terminal-chain|gpuq|RUNNING|0:0|1|2|15G|gpu021|2026-08-14T00:00:00|2026-08-14T00:01:00|Unknown|billing=20,cpu=2,gres/gpu:1g.10gb=1,gres/gpu=1,mem=15G,node=1|gpu|30|0|/scratch/mock/maxrl|/scratch/%u/maxrl/tests/logs/%x_%j.out|/scratch/%u/maxrl/tests/logs/%x_%j.err|sbatch --parsable /scratch/mock/terminal.sbatch|'
if "$here/hopper.sh" terminal-receipt 424242 "$test_root/nonterminal.tsv" \
    >/dev/null 2>&1; then
  printf 'nonterminal sacct state was not rejected\n' >&2
  exit 1
fi
unset MOCK_TERMINAL_ROW
export MOCK_SECOND_TERMINAL_ROW='424242|duplicate|gpuq|COMPLETED|0:0|102|2|15G|gpu021|2026-08-14T00:00:00|2026-08-14T00:01:00|2026-08-14T00:02:42|billing=20,cpu=2,mem=15G,node=1|gpu|30|0|/scratch/mock/maxrl|/scratch/%u/out|/scratch/%u/err|sbatch --parsable /scratch/mock/duplicate.sbatch|'
if "$here/hopper.sh" terminal-receipt 424242 "$test_root/duplicate-terminal.tsv" \
    >/dev/null 2>&1; then
  printf 'duplicate terminal allocation rows were not rejected\n' >&2
  exit 1
fi
unset MOCK_SECOND_TERMINAL_ROW
printf 'hopper local mock tests: PASS\n'
