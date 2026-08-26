#!/usr/bin/env bash
# Reusable Hopper workflow: health -> stage/submit -> track/watch -> fetch.
#
#   hopper.sh health                                  read-only SSH/Slurm/scratch checks
#   hopper.sh submit <sbatch-file> [sbatch-args...]   stage + submit + write receipts
#   hopper.sh status [jobid]                          queue/accounting status
#   hopper.sh campaign-status <attempt> <expected> [incomplete]
#                                                     marker-only campaign progress
#   hopper.sh watch <jobid> [poll_s] [max_wait_s]     wait for a terminal state
#   hopper.sh terminal-receipt <jobid> <new-local-file>
#                                                     capture terminal sacct only
#   hopper.sh fetch <remote-path> <new-local-path> [new-receipt] [terminal-receipt]
#                                                     verified result retrieval
#   hopper.sh logs <jobid> [n] [--allow-endpoints]    tail the recorded stdout path
#   hopper.sh push <local-path> <remote-rel-path>     stage code/data below scratch
#   hopper.sh registry                                show local submission receipts
#
# HOPPER_HOST and HOPPER_SCRATCH may override the defaults. Remote paths are
# deliberately confined below HOPPER_SCRATCH; fetch destinations must not exist.
set -euo pipefail
umask 077

HOST=${HOPPER_HOST:-lwang44@hopper.orc.gmu.edu}
SCRATCH=${HOPPER_SCRATCH:-/scratch/lwang44}
HERE="$(cd "$(dirname "$0")" && pwd)"
REG=${HOPPER_REGISTRY:-$HERE/.job_registry}
REMOTE_ROOT="$SCRATCH/maxrl"
REMOTE_STAGE="$REMOTE_ROOT/sbatch"
REMOTE_RECEIPTS="$REMOTE_ROOT/receipts"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
)
SCP_OPTS=("${SSH_OPTS[@]}")
RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3"

tmp_receipt=""
cleanup() {
  if [[ -n "$tmp_receipt" && -f "$tmp_receipt" ]]; then
    rm -f -- "$tmp_receipt"
  fi
}
trap cleanup EXIT

die() {
  printf 'hopper: %s\n' "$*" >&2
  exit 2
}

usage() {
  sed -n '2,19p' "$0" >&2
}

ssh_hopper() {
  # Keep stderr visible: authentication, quota, and Slurm failures are evidence.
  ssh "${SSH_OPTS[@]}" "$HOST" "$@"
}

validate_components() {
  local path=$1 part rest
  rest=${path#/}
  [[ -n "$rest" && "$rest" != */ && "$rest" != *//* ]] || return 1
  IFS='/' read -r -a parts <<< "$rest"
  for part in "${parts[@]}"; do
    [[ -n "$part" && "$part" != . && "$part" != .. ]] || return 1
  done
}

validate_safe_rel() {
  local path=$1
  [[ -n "$path" && "$path" != /* ]] || return 1
  [[ "$path" =~ ^[A-Za-z0-9._+%@=-]+(/[A-Za-z0-9._+%@=-]+)*$ ]] || return 1
  validate_components "$path"
}

validate_safe_abs() {
  local path=$1
  [[ "$path" == /* ]] || return 1
  [[ "$path" =~ ^/[A-Za-z0-9._+%@=-]+(/[A-Za-z0-9._+%@=-]+)*$ ]] || return 1
  validate_components "$path"
}

validate_jobid() {
  [[ "$1" =~ ^[0-9]+([_.+][A-Za-z0-9]+)*$ ]]
}

validate_manifest_paths() {
  local manifest=$1
  awk '
    NF == 0 { next }
    {
      hash=substr($0, 1, 64)
      separator=substr($0, 65, 2)
      if (length($0) < 67 || hash !~ /^[0-9A-Fa-f]+$/ || separator !~ /^ [ *]$/) {
        bad=1; next
      }
      f=substr($0, 67)
      if (f == "" || f ~ /^\// || f ~ /(^|\/)\.\.($|\/)/) bad=1
    }
    END { exit bad ? 1 : 0 }
  ' "$manifest"
}

local_tree_digest() {
  local root=$1
  (
    cd "$root"
    LC_ALL=C find . \( -type f -o -type l \) -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 -r sha256sum \
      | sha256sum \
      | awk '{print $1}'
  )
}

extract_output_spec() {
  local script=$1 value
  value=$(awk '
    $1 == "#SBATCH" {
      for (i=2; i<=NF; i++) {
        if ($i ~ /^--output=/) {
          sub(/^--output=/, "", $i); print $i; exit
        }
        if ($i == "--output" || $i == "-o") {
          if (i < NF) print $(i+1); exit
        }
        if ($i ~ /^-o./) { print substr($i, 3); exit }
      }
    }
  ' "$script")
  value=${value#\"}; value=${value%\"}
  value=${value#\'}; value=${value%\'}
  printf '%s\n' "$value"
}

[[ "$HOST" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+$ ]] \
  || die "unsafe HOPPER_HOST: $HOST"
validate_safe_abs "$SCRATCH" || die "unsafe HOPPER_SCRATCH: $SCRATCH"
[[ "$SCRATCH" == /scratch/* ]] || die "HOPPER_SCRATCH must be below /scratch"
[[ -n "$REG" && "$REG" != *$'\n'* && "$REG" != *$'\t'* ]] \
  || die "unsafe HOPPER_REGISTRY path"

command=${1:-}
[[ -n "$command" ]] || { usage; exit 1; }
shift

case "$command" in
  health)
    (( $# == 0 )) || die "usage: hopper.sh health"
    ssh_hopper bash -s -- "$SCRATCH" <<'REMOTE'
set -euo pipefail
scratch=$1
printf 'host\t%s\nuser\t%s\nutc\t%s\n' "$(hostname)" "$(id -un)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
for command in sbatch squeue sacct sinfo sha256sum; do
  if command -v "$command" >/dev/null 2>&1; then
    printf '%s\tOK\n' "$command"
  else
    printf '%s\tMISSING\n' "$command"
    exit 1
  fi
done
[[ -d "$scratch" ]] || { printf 'scratch\tMISSING\t%s\n' "$scratch"; exit 1; }
[[ -w "$scratch" ]] || { printf 'scratch\tNOT_WRITABLE\t%s\n' "$scratch"; exit 1; }
printf 'scratch\tOK\t%s\n' "$scratch"
sinfo -h -p normal,gpuq -o 'partition=%P availability=%a timelimit=%l nodes=%D state=%t' | sort -u
REMOTE
    ;;

  submit)
    (( $# >= 1 )) || die "usage: hopper.sh submit <sbatch-file> [sbatch-args...]"
    script=$1
    shift
    [[ -f "$script" && -r "$script" ]] || die "not a readable sbatch file: $script"
    [[ "$script" != *$'\n'* && "$script" != *$'\t'* ]] \
      || die "script path may not contain tabs or newlines"

    sbatch_args=("$@")
    for arg in "${sbatch_args[@]}"; do
      [[ -n "$arg" && "$arg" =~ ^[A-Za-z0-9_./%+,=@:-]+$ ]] \
        || die "unsafe sbatch argument: $arg"
      case "$arg" in
        --output|--output=*|-o|-o*|--error|--error=*|-e|-e*|--chdir|--chdir=*|\
        --partition|--partition=*|-p|-p*|--qos|--qos=*|--gres|--gres=*|\
        --gpus|--gpus=*|--gpus-per-node|--gpus-per-node=*|\
        --gpus-per-task|--gpus-per-task=*|--mem-per-gpu|--mem-per-gpu=*|\
        --nodes|--nodes=*|-N|-N*|--ntasks|--ntasks=*|-n|-n*|\
        --ntasks-per-node|--ntasks-per-node=*|--cpus-per-task|--cpus-per-task=*|\
        -c|-c*|--mem|--mem=*|--mem-per-cpu|--mem-per-cpu=*|\
        --time|--time=*|-t|-t*|--requeue|--no-requeue|\
        --job-name|--job-name=*|-J|-J*)
          die "put identity, resource, output, and chdir settings in the sbatch file so they can be audited"
          ;;
      esac
    done
    if [[ "$(basename -- "$script")" == ued_minimax_terminal_chain_smoke.sbatch ]]; then
      (( ${#sbatch_args[@]} == 1 )) \
        || die "terminal-chain submission requires one explicit UED export allowlist"
      [[ "${sbatch_args[0]}" == --export=UED_* ]] \
        || die "terminal-chain submission forbids --export=ALL and ambient exports"
      terminal_export_text=${sbatch_args[0]#--export=}
      IFS=',' read -r -a terminal_exports <<< "$terminal_export_text"
      terminal_required=(
        UED_BUNDLE_DIR UED_BUNDLE_MANIFEST_SHA256
        UED_UPSTREAM_COMMIT UED_UPSTREAM_TREE UED_UPSTREAM_BUNDLE_SHA256
        UED_OVERLAY_MANIFEST_SHA256 UED_TERMINAL_CHAIN_SBATCH_SHA256
        UED_FRONTIER_CONFIG_SHA256 UED_CONTRACT_SHA256 UED_PROTOCOL_SHA256
        UED_ANALYZER_SHA256 UED_TRAINING_DRIVER_SHA256
        UED_EVALUATION_DRIVER_SHA256 UED_ASSEMBLER_SHA256
        UED_ENV_DIR UED_ENV_LOCK_SHA256 UED_ENV_FREEZE_SHA256
        UED_ENV_MANIFEST_SHA256 UED_IMPORT_SMOKE_RESULT_DIR
        UED_IMPORT_SMOKE_MANIFEST_SHA256 UED_ONE_UPDATE_RESULT_DIR
        UED_ONE_UPDATE_MANIFEST_SHA256
      )
      declare -A terminal_seen=()
      for terminal_export in "${terminal_exports[@]}"; do
        terminal_key=${terminal_export%%=*}
        terminal_value=${terminal_export#*=}
        [[ "$terminal_export" == *=* && "$terminal_key" =~ ^UED_[A-Z0-9_]+$ \
              && -n "$terminal_value" && -z "${terminal_seen[$terminal_key]:-}" ]] \
          || die "unsafe, empty, or duplicate terminal-chain export: $terminal_key"
        terminal_seen[$terminal_key]=1
      done
      (( ${#terminal_seen[@]} == ${#terminal_required[@]} )) \
        || die "terminal-chain export allowlist key count drift"
      for terminal_key in "${terminal_required[@]}"; do
        [[ -n "${terminal_seen[$terminal_key]:-}" ]] \
          || die "terminal-chain export allowlist is missing $terminal_key"
      done
      unset terminal_export_text terminal_exports terminal_required terminal_seen \
        terminal_export terminal_key terminal_value
    fi
    if (( ${#sbatch_args[@]} == 0 )); then
      sbatch_args_text="-"
    else
      sbatch_args_text=$(IFS=,; printf '%s' "${sbatch_args[*]}")
    fi

    local_script=$(realpath "$script")
    local_sha=$(sha256sum "$local_script" | awk '{print $1}')
    base=$(basename "$script")
    safe_base=$(printf '%s' "$base" | tr -c 'A-Za-z0-9._-' '_')
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    remote_name="${safe_base%.sbatch}-${local_sha:0:16}-${stamp}-$$.sbatch"
    remote_script="$REMOTE_STAGE/$remote_name"

    output_spec=$(extract_output_spec "$script")
    [[ -n "$output_spec" ]] || output_spec="slurm-%j.out"
    [[ "$output_spec" =~ ^/?[A-Za-z0-9._+/%=-]+$ ]] \
      || die "unsupported #SBATCH output path: $output_spec"

    remote_user_raw=$(ssh_hopper 'printf "__HOPPER_USER__%s\n" "$(id -un)"')
    remote_user=$(printf '%s\n' "$remote_user_raw" | awk -F'__HOPPER_USER__' '/^__HOPPER_USER__/ {print $2; exit}')
    [[ "$remote_user" =~ ^[A-Za-z0-9._-]+$ ]] || die "could not determine remote user"
    resolved_output=${output_spec//%u/$remote_user}
    if [[ "$resolved_output" != /* ]]; then
      resolved_output="$REMOTE_ROOT/$resolved_output"
    fi
    validate_safe_abs "$resolved_output" || die "unsafe resolved output path: $resolved_output"
    case "$resolved_output" in
      "$SCRATCH"/*) ;;
      *) die "job output must remain below $SCRATCH: $resolved_output" ;;
    esac
    output_parent=$(dirname "$resolved_output")
    [[ "$output_parent" != *%* ]] \
      || die "Slurm replacement tokens are not supported in the output directory: $output_parent"

    ssh_hopper bash -s -- "$REMOTE_STAGE" "$REMOTE_RECEIPTS" "$output_parent" <<'REMOTE'
set -euo pipefail
umask 077
mkdir -p -- "$1" "$2" "$3"
REMOTE
    scp "${SCP_OPTS[@]}" -- "$local_script" "$HOST:$remote_script"

    remote_sha_raw=$(ssh_hopper bash -s -- "$remote_script" <<'REMOTE'
set -euo pipefail
printf '__HOPPER_SHA__%s\n' "$(sha256sum "$1" | awk '{print $1}')"
REMOTE
    )
    remote_sha=$(printf '%s\n' "$remote_sha_raw" | awk -F'__HOPPER_SHA__' '/^__HOPPER_SHA__/ {print $2; exit}')
    [[ "$remote_sha" == "$local_sha" ]] \
      || die "staged script checksum mismatch (local=$local_sha remote=${remote_sha:-missing})"

    submit_raw=$(ssh_hopper bash -s -- "$REMOTE_ROOT" "$remote_script" "${sbatch_args[@]}" <<'REMOTE'
set -euo pipefail
root=$1
script=$2
shift 2
job=$(cd "$root" && sbatch --parsable "$@" "$script")
printf '__HOPPER_JOB__%s\n' "$job"
job_id=${job%%;*}
record=$(scontrol show job -o "$job_id" || true)
stdout=$(printf '%s\n' "$record" | sed -n 's/.* StdOut=\([^ ]*\).*/\1/p')
printf '__HOPPER_STDOUT__%s\n' "$stdout"
REMOTE
    )
    job_raw=$(printf '%s\n' "$submit_raw" | awk -F'__HOPPER_JOB__' '/^__HOPPER_JOB__/ {print $2; exit}')
    jid=${job_raw%%;*}
    validate_jobid "$jid" || die "sbatch returned an invalid job id: ${job_raw:-missing}"

    slurm_stdout=$(printf '%s\n' "$submit_raw" | awk -F'__HOPPER_STDOUT__' '/^__HOPPER_STDOUT__/ {print $2; exit}')
    receipt_output=$resolved_output
    if [[ -n "$slurm_stdout" ]] && validate_safe_abs "$slurm_stdout"; then
      case "$slurm_stdout" in
        "$SCRATCH"/*) receipt_output=$slurm_stdout ;;
      esac
    fi

    utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    receipt_remote="$REMOTE_RECEIPTS/job-${jid}-${stamp}.tsv"
    receipt_line=$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
      "$jid" "$utc" "$HOST" "$local_script" "$local_sha" \
      "$remote_script" "$remote_sha" "$receipt_output" "$receipt_remote" \
      "$sbatch_args_text")
    printf '%s\n' "$receipt_line" >> "$REG"

    tmp_receipt=$(mktemp /tmp/hopper-receipt.XXXXXX)
    printf 'job_id\tutc\thost\tlocal_script\tlocal_sha256\tremote_script\tremote_sha256\toutput_path\tremote_receipt\tsbatch_args\n%s\n' \
      "$receipt_line" > "$tmp_receipt"
    printf 'submitted %s (%s)\noutput: %s\nlocal receipt: %s\n' \
      "$jid" "$base" "$receipt_output" "$REG"
    if ! scp "${SCP_OPTS[@]}" -- "$tmp_receipt" "$HOST:$receipt_remote"; then
      die "job $jid was submitted and locally recorded, but remote receipt upload failed"
    fi
    printf 'remote receipt: %s\n' "$receipt_remote"
    ;;

  status)
    (( $# <= 1 )) || die "usage: hopper.sh status [jobid]"
    if (( $# == 1 )); then
      jid=$1
      validate_jobid "$jid" || die "invalid job id: $jid"
      ssh_hopper bash -s -- "$jid" <<'REMOTE'
set -euo pipefail
jid=$1
printf '%s\n' '=== live queue ==='
squeue -j "$jid" -o '%.18i %.28j %.10P %.12T %.10M %.10l %.6C %.12m %.24R' || true
printf '%s\n' '=== accounting ==='
sacct -j "$jid" -X -P \
  --format=JobIDRaw,JobName,Partition,State,ExitCode,Elapsed,Timelimit,AllocCPUS,ReqMem,NodeList,Submit,Start,End
REMOTE
    else
      ssh_hopper 'squeue -u "$USER" -o "%.18i %.28j %.10P %.12T %.10M %.10l %.6C %.12m %.24R"'
    fi
    ;;

  campaign-status)
    (( $# >= 2 && $# <= 3 )) \
      || die "usage: hopper.sh campaign-status <remote-attempt-path> <expected-blocks> [remote-incomplete-path]"
    remote_attempt=$1
    expected_blocks=$2
    remote_incomplete=${3:--}
    validate_safe_abs "$remote_attempt" \
      || die "campaign attempt must be a safe absolute path"
    case "$remote_attempt" in
      "$SCRATCH"/*) ;;
      *) die "campaign attempt must remain below $SCRATCH" ;;
    esac
    [[ "$expected_blocks" =~ ^[0-9]+$ ]] \
      && (( expected_blocks >= 1 && expected_blocks <= 100000 )) \
      || die "expected-blocks must be an integer from 1 to 100000"
    if [[ "$remote_incomplete" != - ]]; then
      validate_safe_abs "$remote_incomplete" \
        || die "campaign incomplete path must be a safe absolute path"
      case "$remote_incomplete" in
        "$SCRATCH"/*) ;;
        *) die "campaign incomplete path must remain below $SCRATCH" ;;
      esac
    fi

    # Read only directory names and completion-file presence. Never open a
    # result, telemetry, checkpoint, or log payload from a blinded campaign.
    ssh_hopper bash -s -- \
      "$remote_attempt" "$expected_blocks" "$remote_incomplete" <<'REMOTE'
set -euo pipefail
attempt=$1
expected=$2
incomplete=$3
final_blocks=0
complete_markers=0
sha256_manifests=0
arm_receipts=0
invalid_final_blocks=0
quarantines=0

if [[ -d "$attempt" && ! -L "$attempt" ]]; then
  while IFS= read -r -d '' block; do
    ((final_blocks += 1))
    valid=1
    if [[ -f "$block/COMPLETE" && ! -L "$block/COMPLETE" ]]; then
      ((complete_markers += 1))
    else
      valid=0
    fi
    if [[ -f "$block/SHA256SUMS" && ! -L "$block/SHA256SUMS" ]]; then
      ((sha256_manifests += 1))
    else
      valid=0
    fi
    if [[ -d "$block/meta" && ! -L "$block/meta" ]]; then
      while IFS= read -r -d '' receipt; do
        ((arm_receipts += 1))
      done < <(find "$block/meta" -mindepth 1 -maxdepth 1 -type f \
        -name '*.DONE.json' -print0)
    fi
    (( valid == 1 )) || ((invalid_final_blocks += 1))
  done < <(find "$attempt" -mindepth 1 -maxdepth 1 -type d \
    -name 'seed-*' -print0)
fi

if [[ "$incomplete" != - && -d "$incomplete" && ! -L "$incomplete" ]]; then
  while IFS= read -r -d '' quarantine; do
    ((quarantines += 1))
  done < <(find "$incomplete" -mindepth 1 -maxdepth 1 -type d \
    -name 'seed-*.job-*' -print0)
fi

state=IN_PROGRESS
if (( final_blocks == expected && complete_markers == expected \
      && sha256_manifests == expected && invalid_final_blocks == 0 )); then
  # Hashes and schemas still require the experiment-specific frozen retrieval
  # validator. This state is intentionally not called "verified".
  state=ALL_COMPLETION_MARKERS_PRESENT
elif (( final_blocks > expected )); then
  state=OVERCOMPLETE_FAIL_CLOSED
elif (( invalid_final_blocks > 0 )); then
  state=INVALID_FINAL_BLOCKS
fi

printf 'remote_attempt\t%s\n' "$attempt"
printf 'expected_blocks\t%s\n' "$expected"
printf 'final_blocks\t%s\n' "$final_blocks"
printf 'complete_markers\t%s\n' "$complete_markers"
printf 'sha256_manifests\t%s\n' "$sha256_manifests"
printf 'arm_done_receipts\t%s\n' "$arm_receipts"
printf 'invalid_final_blocks\t%s\n' "$invalid_final_blocks"
printf 'incomplete_quarantines\t%s\n' "$quarantines"
printf 'structural_state\t%s\n' "$state"
REMOTE
    ;;

  watch)
    (( $# >= 1 && $# <= 3 )) \
      || die "usage: hopper.sh watch <jobid> [poll_s] [max_wait_s]"
    jid=$1
    poll=${2:-120}
    max_wait=${3:-604800}
    validate_jobid "$jid" || die "invalid job id: $jid"
    [[ "$poll" =~ ^[0-9]+$ ]] && (( poll >= 5 && poll <= 3600 )) \
      || die "poll_s must be an integer from 5 to 3600"
    [[ "$max_wait" =~ ^[0-9]+$ ]] && (( max_wait >= poll && max_wait <= 2592000 )) \
      || die "max_wait_s must be between poll_s and 2592000"
    watch_start=$SECONDS
    while :; do
      state_raw=$(ssh_hopper bash -s -- "$jid" <<'REMOTE'
set -euo pipefail
jid=$1
states=$(sacct -j "$jid" -X -n -P -o State \
  | sed -e 's/|$//' -e '/^[[:space:]]*$/d' || true)
if [[ -z "$states" ]]; then
  states=$(squeue -h -j "$jid" -o '%T' || true)
fi
printf '__HOPPER_STATES__%s\n' "$(printf '%s\n' "$states" | sort -u | paste -sd, -)"
REMOTE
      )
      states=$(printf '%s\n' "$state_raw" | awk -F'__HOPPER_STATES__' '/^__HOPPER_STATES__/ {print $2; exit}')
      [[ -n "$states" ]] || die "job $jid was not found in squeue or sacct"
      printf '%s job=%s state=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$jid" "$states"
      case "$states" in
        *PENDING*|*RUNNING*|*CONFIGURING*|*COMPLETING*|*REQUEUED*|*RESIZING*|*SUSPENDED*)
          (( SECONDS - watch_start < max_wait )) \
            || die "timed out waiting for job $jid after ${max_wait}s"
          sleep "$poll"
          ;;
        *)
          ssh_hopper "sacct -j $jid -X -P --format=JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End"
          if [[ "$states" == COMPLETED ]]; then
            break
          fi
          printf 'hopper: job %s ended unsuccessfully: %s\n' "$jid" "$states" >&2
          exit 1
          ;;
      esac
    done
    ;;

  terminal-receipt)
    (( $# == 2 )) \
      || die "usage: hopper.sh terminal-receipt <numeric-jobid> <new-local-file>"
    jid=$1
    local_dest=$2
    [[ "$jid" =~ ^[0-9]+$ ]] || die "terminal receipt requires a numeric job id"
    validate_safe_abs "$local_dest" \
      || die "terminal receipt destination must be a safe absolute path"
    [[ ! -e "$local_dest" && ! -L "$local_dest" ]] \
      || die "terminal receipt destination already exists: $local_dest"

    local_root=${HOPPER_LOCAL_RESULTS_ROOT:-$HERE}
    validate_safe_abs "$local_root" \
      || die "unsafe HOPPER_LOCAL_RESULTS_ROOT: $local_root"
    [[ -d "$local_root" && ! -L "$local_root" ]] \
      || die "HOPPER_LOCAL_RESULTS_ROOT must be an existing real directory"
    local_root=$(readlink -f -- "$local_root")
    [[ "$local_root" != / ]] || die "HOPPER_LOCAL_RESULTS_ROOT may not be /"
    local_parent=$(dirname -- "$local_dest")
    [[ -d "$local_parent" && ! -L "$local_parent" ]] \
      || die "terminal receipt parent must be an existing real directory"
    local_parent=$(readlink -f -- "$local_parent")
    [[ "$local_dest" == "$local_parent/$(basename -- "$local_dest")" ]] \
      || die "terminal receipt destination is not canonical"
    case "$local_dest" in
      "$local_root"/*) ;;
      *) die "terminal receipt destination must remain below $local_root" ;;
    esac

    terminal_raw=$(ssh_hopper bash -s -- "$jid" <<'REMOTE'
set -euo pipefail
umask 077
export TZ=America/New_York
jid=$1
[[ "$jid" =~ ^[0-9]+$ ]]
    terminal_header='JobIDRaw|JobName|Partition|State|ExitCode|ElapsedRaw|AllocCPUS|ReqMem|NodeList|Submit|Start|End|AllocTRES|QOS|TimelimitRaw|Restarts|WorkDir|StdOut|StdErr|SubmitLine'
resource_header='JobIDRaw|MaxRSS|TRESUsageInMax'
mapfile -t terminal_rows < <(
  sacct -j "$jid" -X -n -P \
    -o JobIDRaw,JobName,Partition,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,NodeList,Submit,Start,End,AllocTRES,QOS,TimelimitRaw,Restarts,WorkDir,StdOut,StdErr,SubmitLine \
    | sed -e '/^[[:space:]]*$/d'
)
(( ${#terminal_rows[@]} == 1 )) || {
  printf 'expected exactly one terminal allocation row for %s; got %s\n' \
    "$jid" "${#terminal_rows[@]}" >&2
  exit 3
}
terminal_row=${terminal_rows[0]}
IFS='|' read -r row_id job_name partition state exit_code elapsed_raw \
  alloc_cpus req_mem node_list submit start end alloc_tres qos timelimit_raw \
  restarts work_dir stdout_path stderr_path submit_line extra <<< "$terminal_row"
[[ -z "${extra:-}" && "$row_id" == "$jid" ]]
[[ "$state" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT)$ ]] || {
  printf 'job %s is not terminal: %s\n' "$jid" "$state" >&2
  exit 4
}
[[ "$exit_code" =~ ^[0-9]+:[0-9]+$ && "$elapsed_raw" =~ ^[0-9]+$ ]]
[[ "$alloc_cpus" =~ ^[0-9]+$ && -n "$partition" && -n "$node_list" ]]
[[ "$submit" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]
[[ "$start" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]
[[ "$end" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]
[[ -n "$qos" && "$timelimit_raw" =~ ^[0-9]+$ && "$restarts" =~ ^[0-9]+$ ]]
# Hopper may leave StdErr empty when an sbatch file specifies only --output;
# Slurm then retains a valid terminal allocation row with no separate error
# path. StdOut and the immutable SubmitLine remain required.
[[ -n "$work_dir" && -n "$stdout_path" && -n "$submit_line" ]]

mapfile -t resource_rows < <(
  sacct -j "$jid" -n -P -o JobIDRaw,MaxRSS,TRESUsageInMax \
    | sed -e '/^[[:space:]]*$/d'
)
(( ${#resource_rows[@]} >= 1 )) || {
  printf 'job %s has no accounting resource rows\n' "$jid" >&2
  exit 5
}
declare -A seen=()
for row in "${resource_rows[@]}"; do
  IFS='|' read -r resource_id max_rss tres_max resource_extra <<< "$row"
  [[ -z "${resource_extra:-}" ]]
  [[ "$resource_id" == "$jid" || "$resource_id" == "$jid".* ]] || {
    printf 'foreign accounting row for %s: %s\n' "$jid" "$resource_id" >&2
    exit 6
  }
  [[ -z "${seen[$resource_id]:-}" ]] || {
    printf 'duplicate accounting row for %s: %s\n' "$jid" "$resource_id" >&2
    exit 7
  }
  seen[$resource_id]=1
done

printf '%s\n' '__HOPPER_TERMINAL_RECEIPT_BEGIN__'
submit_epoch=$(date -d "$submit" +%s)
start_epoch=$(date -d "$start" +%s)
terminal_end_epoch=$(date -d "$end" +%s)
(( submit_epoch <= start_epoch && start_epoch <= terminal_end_epoch \
     && terminal_end_epoch - start_epoch == elapsed_raw )) || {
  printf 'terminal receipt scheduler timestamps are unordered for %s\n' "$jid" >&2
  exit 8
}
retrieved_epoch=$(date -u +%s)
(( retrieved_epoch >= terminal_end_epoch )) || {
  printf 'terminal receipt timestamp predates job end for %s\n' "$jid" >&2
  exit 9
}
retrieved_utc=$(date -u -d "@$retrieved_epoch" +%Y-%m-%dT%H:%M:%SZ)
printf 'terminal_receipt_schema\t2\n'
printf 'retrieved_utc\t%s\n' "$retrieved_utc"
printf 'retrieved_epoch\t%s\n' "$retrieved_epoch"
printf 'terminal_end_epoch\t%s\n' "$terminal_end_epoch"
printf 'terminal_header\t%s\n' "$terminal_header"
printf 'terminal_row\t%s\n' "$terminal_row"
printf 'resource_header\t%s\n' "$resource_header"
for row in "${resource_rows[@]}"; do
  printf 'resource_row\t%s\n' "$row"
done
printf '%s\n' '__HOPPER_TERMINAL_RECEIPT_END__'
REMOTE
    )

    tmp_receipt=$(mktemp "$local_parent/.terminal-receipt.XXXXXX")
    printf '%s\n' "$terminal_raw" | awk '
      /^__HOPPER_TERMINAL_RECEIPT_BEGIN__$/ { inside=1; next }
      /^__HOPPER_TERMINAL_RECEIPT_END__$/ { inside=0; ended=1; next }
      inside { print }
      END { if (!ended || inside) exit 1 }
    ' > "$tmp_receipt" || die "malformed terminal receipt markers"
    awk -F '\t' -v jid="$jid" '
      BEGIN {
        terminal_header="JobIDRaw|JobName|Partition|State|ExitCode|ElapsedRaw|AllocCPUS|ReqMem|NodeList|Submit|Start|End|AllocTRES|QOS|TimelimitRaw|Restarts|WorkDir|StdOut|StdErr|SubmitLine"
        resource_header="JobIDRaw|MaxRSS|TRESUsageInMax"
      }
      $1 == "terminal_receipt_schema" { schema++; if ($2 != "2" || NF != 2) bad=1; next }
      $1 == "retrieved_utc" {
        utc++; if ($2 !~ /^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z$/ || NF != 2) bad=1; next
      }
      $1 == "retrieved_epoch" {
        retrieved_epoch++; retrieved=$2
        if ($2 !~ /^[0-9]+$/ || NF != 2) bad=1
        next
      }
      $1 == "terminal_end_epoch" {
        terminal_end_epoch++; terminal_end=$2
        if ($2 !~ /^[0-9]+$/ || NF != 2) bad=1
        next
      }
      $1 == "terminal_header" { th++; if ($2 != terminal_header || NF != 2) bad=1; next }
      $1 == "terminal_row" {
        tr++; if (NF != 2) { bad=1; next }
        count=split($2, value, "|")
        if (count != 20 || value[1] != jid || value[4] !~ /^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT)$/ || value[5] !~ /^[0-9]+:[0-9]+$/ || value[6] !~ /^[0-9]+$/ || value[15] !~ /^[0-9]+$/ || value[16] !~ /^[0-9]+$/) bad=1
        next
      }
      $1 == "resource_header" { rh++; if ($2 != resource_header || NF != 2) bad=1; next }
      $1 == "resource_row" {
        rr++; if (NF != 2) { bad=1; next }
        count=split($2, value, "|")
        if (count != 3 || (value[1] != jid && value[1] !~ ("^" jid "\\."))) bad=1
        if (seen[value[1]]++) bad=1
        next
      }
      { bad=1 }
      END {
        if (schema != 1 || utc != 1 || retrieved_epoch != 1 || terminal_end_epoch != 1 ||
            retrieved + 0 < terminal_end + 0 || th != 1 || tr != 1 || rh != 1 || rr < 1 || bad) exit 1
      }
    ' "$tmp_receipt" || die "terminal receipt failed local validation"
    chmod 600 "$tmp_receipt"
    ln -- "$tmp_receipt" "$local_dest" \
      || die "terminal receipt destination appeared during capture"
    rm -f -- "$tmp_receipt"
    tmp_receipt=""
    printf 'terminal accounting receipt: %s\nsha256: %s\n' \
      "$local_dest" "$(sha256sum "$local_dest" | awk '{print $1}')"
    ;;

  fetch)
    (( $# >= 2 && $# <= 4 )) \
      || die "usage: hopper.sh fetch <remote-path> <new-local-path> [new-receipt] [terminal-receipt]"
    remote_arg=$1
    local_dest=$2
    fetch_receipt=${3:-}
    terminal_gate=${4:-}
    [[ -z "$terminal_gate" || -n "$fetch_receipt" ]] \
      || die "a terminal-gated fetch requires a new fetch receipt"
    if [[ "$remote_arg" == /* ]]; then
      remote_abs=$remote_arg
      validate_safe_abs "$remote_abs" || die "unsafe remote path: $remote_arg"
    else
      validate_safe_rel "$remote_arg" || die "unsafe remote relative path: $remote_arg"
      remote_abs="$SCRATCH/$remote_arg"
    fi
    case "$remote_abs" in
      "$SCRATCH"/*) ;;
      *) die "remote fetch path must remain below $SCRATCH" ;;
    esac
    [[ ! -e "$local_dest" && ! -L "$local_dest" ]] \
      || die "fetch destination already exists: $local_dest"
    [[ "$local_dest" != *$'\n'* && "$local_dest" != *$'\t'* ]] \
      || die "local destination may not contain tabs or newlines"
    if [[ -n "$fetch_receipt" ]]; then
      validate_safe_abs "$fetch_receipt" || die "fetch receipt must be a safe absolute path"
      [[ ! -e "$fetch_receipt" && ! -L "$fetch_receipt" ]] \
        || die "fetch receipt destination already exists: $fetch_receipt"
    fi

    terminal_end_epoch=0
    terminal_receipt_sha256=-
    terminal_retrieved_epoch=0
    if [[ -n "$terminal_gate" ]]; then
      validate_safe_abs "$terminal_gate" \
        || die "terminal gate must be a safe absolute receipt path"
      [[ -f "$terminal_gate" && ! -L "$terminal_gate" ]] \
        || die "terminal gate must be an existing regular receipt"
      [[ "$(readlink -f -- "$terminal_gate")" == "$terminal_gate" ]] \
        || die "terminal gate path is not canonical"
      gate_values=$(awk -F '\t' '
        $1 == "terminal_receipt_schema" { schema++; schema_value=$2; next }
        $1 == "retrieved_epoch" { retrieved++; retrieved_value=$2; next }
        $1 == "terminal_end_epoch" { ended++; end_value=$2; next }
        $1 == "terminal_row" {
          row++; n=split($2, value, "|")
          if (n != 20 || value[4] != "COMPLETED" || value[5] != "0:0") bad=1
          next
        }
        END {
          if (schema != 1 || schema_value != "2" || retrieved != 1 || ended != 1 ||
              row != 1 || retrieved_value !~ /^[0-9]+$/ || end_value !~ /^[0-9]+$/ ||
              retrieved_value + 0 < end_value + 0 || bad) exit 1
          printf "%s %s\n", end_value, retrieved_value
        }
      ' "$terminal_gate") || die "terminal gate is not an exact clean-completion receipt"
      read -r terminal_end_epoch terminal_retrieved_epoch extra <<< "$gate_values"
      [[ -z "${extra:-}" ]]
      terminal_receipt_sha256=$(sha256sum "$terminal_gate" | awk '{print $1}')
    fi

    # This timestamp precedes the first remote probe.  A terminal-chain
    # finalizer accepts only receipts whose fetch began after Slurm End.
    fetch_started_epoch=$(date -u +%s)
    fetch_started_utc=$(date -u -d "@$fetch_started_epoch" +%Y-%m-%dT%H:%M:%SZ)
    if [[ -n "$terminal_gate" ]]; then
      (( fetch_started_epoch >= terminal_end_epoch \
         && fetch_started_epoch >= terminal_retrieved_epoch )) \
        || die "fetch started before the terminal receipt was complete"
    fi

    probe=$(ssh_hopper bash -s -- "$remote_abs" <<'REMOTE'
set -euo pipefail
p=$1
[[ -e "$p" || -L "$p" ]] || { printf 'remote source not found: %s\n' "$p" >&2; exit 3; }
check_manifest() {
  local manifest=$1
  awk '
    NF == 0 { next }
    {
      hash=substr($0, 1, 64)
      separator=substr($0, 65, 2)
      if (length($0) < 67 || hash !~ /^[0-9A-Fa-f]+$/ || separator !~ /^ [ *]$/) {
        bad=1; next
      }
      f=substr($0, 67)
      if (f == "" || f ~ /^\// || f ~ /(^|\/)\.\.($|\/)/) bad=1
    }
    END { exit bad ? 1 : 0 }
  ' "$manifest" || { printf 'unsafe paths in %s\n' "$manifest" >&2; exit 4; }
}
if [[ -d "$p" ]]; then
  manifest=0
  if [[ -f "$p/SHA256SUMS" ]]; then
    check_manifest "$p/SHA256SUMS"
    (cd "$p" && sha256sum -c --strict SHA256SUMS)
    manifest=1
  fi
  digest=$(cd "$p" && LC_ALL=C find . \( -type f -o -type l \) -print0 \
    | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')
  printf '__HOPPER_FETCH_TYPE__dir\n__HOPPER_MANIFEST__%s\n__HOPPER_DIGEST__%s\n' "$manifest" "$digest"
elif [[ -f "$p" ]]; then
  manifest=0
  parent=$(dirname "$p")
  if [[ -f "$parent/SHA256SUMS" ]]; then
    check_manifest "$parent/SHA256SUMS"
    (cd "$parent" && sha256sum -c --strict SHA256SUMS)
    manifest=1
  fi
  digest=$(sha256sum "$p" | awk '{print $1}')
  printf '__HOPPER_FETCH_TYPE__file\n__HOPPER_MANIFEST__%s\n__HOPPER_DIGEST__%s\n' "$manifest" "$digest"
else
  printf 'remote source is not a regular file or directory: %s\n' "$p" >&2
  exit 5
fi
REMOTE
    )
    printf '%s\n' "$probe"
    remote_type=$(printf '%s\n' "$probe" | awk -F'__HOPPER_FETCH_TYPE__' '/^__HOPPER_FETCH_TYPE__/ {print $2; exit}')
    has_manifest=$(printf '%s\n' "$probe" | awk -F'__HOPPER_MANIFEST__' '/^__HOPPER_MANIFEST__/ {print $2; exit}')
    remote_digest=$(printf '%s\n' "$probe" | awk -F'__HOPPER_DIGEST__' '/^__HOPPER_DIGEST__/ {print $2; exit}')
    [[ "$remote_type" == dir || "$remote_type" == file ]] || die "remote probe returned no source type"
    [[ "$remote_digest" =~ ^[0-9a-f]{64}$ ]] || die "remote probe returned no checksum"

    local_parent=$(dirname "$local_dest")
    local_base=$(basename "$local_dest")
    mkdir -p -- "$local_parent"
    local_partial="$local_parent/.${local_base}.partial-$$"
    [[ ! -e "$local_partial" && ! -L "$local_partial" ]] \
      || die "temporary fetch path already exists: $local_partial"
    if [[ "$remote_type" == dir ]]; then
      mkdir -- "$local_partial"
      rsync -az --safe-links --protect-args -e "$RSYNC_RSH" \
        "$HOST:$remote_abs/" "$local_partial/"
      if [[ "$has_manifest" == 1 ]]; then
        [[ -f "$local_partial/SHA256SUMS" ]] || die "remote manifest was not transferred"
        validate_manifest_paths "$local_partial/SHA256SUMS" \
          || die "unsafe paths in local SHA256SUMS"
        (cd "$local_partial" && sha256sum -c --strict SHA256SUMS)
      fi
      local_digest=$(local_tree_digest "$local_partial")
    else
      rsync -az --safe-links --protect-args -e "$RSYNC_RSH" \
        "$HOST:$remote_abs" "$local_partial"
      local_digest=$(sha256sum "$local_partial" | awk '{print $1}')
    fi
    [[ "$local_digest" == "$remote_digest" ]] \
      || die "post-transfer checksum mismatch (remote=$remote_digest local=$local_digest)"
    mv -- "$local_partial" "$local_dest"
    if [[ -n "$fetch_receipt" ]]; then
      fetch_receipt_parent=$(dirname -- "$fetch_receipt")
      [[ -d "$fetch_receipt_parent" && ! -L "$fetch_receipt_parent" ]] \
        || die "fetch receipt parent must be an existing real directory"
      fetch_receipt_parent=$(readlink -f -- "$fetch_receipt_parent")
      [[ "$fetch_receipt" == "$fetch_receipt_parent/$(basename -- "$fetch_receipt")" ]] \
        || die "fetch receipt destination is not canonical"
      local_dest_abs=$(readlink -f -- "$local_dest")
      retrieved_epoch=$(date -u +%s)
      retrieved_utc=$(date -u -d "@$retrieved_epoch" +%Y-%m-%dT%H:%M:%SZ)
      tmp_receipt=$(mktemp "$fetch_receipt_parent/.fetch-receipt.XXXXXX")
      {
        printf 'fetch_receipt_schema\t2\n'
        printf 'fetch_started_utc\t%s\n' "$fetch_started_utc"
        printf 'fetch_started_epoch\t%s\n' "$fetch_started_epoch"
        printf 'retrieved_utc\t%s\n' "$retrieved_utc"
        printf 'retrieved_epoch\t%s\n' "$retrieved_epoch"
        printf 'terminal_end_epoch\t%s\n' "$terminal_end_epoch"
        printf 'terminal_receipt_sha256\t%s\n' "$terminal_receipt_sha256"
        printf 'remote_path\t%s\n' "$remote_abs"
        printf 'remote_type\t%s\n' "$remote_type"
        printf 'remote_digest\t%s\n' "$remote_digest"
        printf 'manifest_verified\t%s\n' "$has_manifest"
        printf 'local_path\t%s\n' "$local_dest_abs"
        printf 'local_digest\t%s\n' "$local_digest"
      } > "$tmp_receipt"
      chmod 600 "$tmp_receipt"
      ln -- "$tmp_receipt" "$fetch_receipt" \
        || die "fetch receipt destination appeared during capture"
      rm -f -- "$tmp_receipt"
      tmp_receipt=""
    fi
    printf 'fetched and verified: %s -> %s\nsha256/tree digest: %s\n' \
      "$remote_abs" "$local_dest" "$local_digest"
    ;;

  push)
    (( $# == 2 )) || die "usage: hopper.sh push <local-path> <remote-rel-path>"
    local_path=$1
    remote_rel=$2
    [[ -e "$local_path" || -L "$local_path" ]] || die "local path not found: $local_path"
    validate_safe_rel "$remote_rel" || die "unsafe remote relative path: $remote_rel"
    remote_abs="$SCRATCH/$remote_rel"
    remote_parent=$(dirname "$remote_abs")
    ssh_hopper "umask 077; mkdir -p -- $remote_parent"
    rsync -az --safe-links --protect-args --exclude '__pycache__' -e "$RSYNC_RSH" \
      "$local_path" "$HOST:$remote_abs"
    printf 'pushed -> %s\n' "$remote_abs"
    ;;

  logs)
    (( $# >= 1 )) || die "usage: hopper.sh logs <jobid> [n] [--allow-endpoints]"
    jid=$1
    shift
    validate_jobid "$jid" || die "invalid job id: $jid"
    lines=40
    allow_endpoints=0
    saw_lines=0
    for arg in "$@"; do
      case "$arg" in
        --allow-endpoints) allow_endpoints=1 ;;
        *[!0-9]*|'') die "unknown logs argument: $arg" ;;
        *)
          (( saw_lines == 0 )) || die "line count supplied more than once"
          lines=$arg
          saw_lines=1
          ;;
      esac
    done
    (( lines >= 1 && lines <= 10000 )) || die "log line count must be from 1 to 10000"

    meta=$(ssh_hopper bash -s -- "$jid" <<'REMOTE'
set -euo pipefail
jid=$1
name=""
stdout=""
# scontrol reports Slurm's resolved StdOut path while sacct may retain the
# literal #SBATCH template (for example /scratch/%u/...); prefer scontrol.
record=$(scontrol show job -o "$jid" || true)
name=$(printf '%s\n' "$record" | sed -n 's/.* JobName=\([^ ]*\).*/\1/p')
stdout=$(printf '%s\n' "$record" | sed -n 's/.* StdOut=\([^ ]*\).*/\1/p')
if [[ -z "$name" || -z "$stdout" ]]; then
  accounting=$(sacct -j "$jid" -X -n -P -o JobName,StdOut \
    | awk -F'|' 'NF && $1 != "" {print; exit}' || true)
  accounting_name=${accounting%%|*}
  remainder=${accounting#*|}
  accounting_stdout=${remainder%%|*}
  if [[ -z "$name" ]]; then
    name=$accounting_name
  fi
  if [[ -z "$stdout" ]]; then
    stdout=$accounting_stdout
  fi
fi
# Resolve the documented tokens if accounting was the only retained source.
remote_user=$(id -un)
array_job=${jid%%_*}
stdout=${stdout//%u/$remote_user}
stdout=${stdout//%j/$jid}
stdout=${stdout//%A/$array_job}
stdout=${stdout//%x/$name}
if [[ "$jid" == *_* ]]; then
  array_task=${jid#*_}
  array_task=${array_task%%.*}
  stdout=${stdout//%a/$array_task}
fi
printf '__HOPPER_JOB_NAME__%s\n__HOPPER_STDOUT__%s\n' "$name" "$stdout"
REMOTE
    )
    job_name=$(printf '%s\n' "$meta" | awk -F'__HOPPER_JOB_NAME__' '/^__HOPPER_JOB_NAME__/ {print $2; exit}')
    stdout_path=$(printf '%s\n' "$meta" | awk -F'__HOPPER_STDOUT__' '/^__HOPPER_STDOUT__/ {print $2; exit}')
    [[ -n "$job_name" ]] \
      || die "could not identify job $jid; refusing blind log discovery"
    case "${job_name,,}" in
      *group-law-flip*|*group_law_flip*)
        # The frozen P0 path is the single-use campaign validator/analyzer.
        # Direct log inspection is never an authorized unblinding route.
        die "group-law-flip logs are sealed; use campaign-status, the frozen retrieval validator, and the single-use analyzer"
        ;;
      *barn-evidence-safe-log*)
        # The audited BARN evidence sbatch redirects the runner and build logs
        # into the sealed attempt tree.  Slurm stdout contains provenance and
        # a completion marker only, so scheduler monitoring remains blind.
        ;;
      *barn-evidence*|*barn_seed*|*barn-seed*)
        (( allow_endpoints == 1 )) \
          || die "BARN evidence logs may expose experiment endpoints; rerun with --allow-endpoints"
        ;;
      *maze-score*|*maze_score*|*mazescore*|*maze-full-arm*|*maze_full_arm*)
        (( allow_endpoints == 1 )) \
          || die "maze-score logs may expose experiment endpoints; rerun with --allow-endpoints"
        ;;
    esac
    [[ -n "$stdout_path" ]] || die "Slurm has no stdout path for job $jid"
    [[ "$stdout_path" != *%* ]] \
      || die "unresolved Slurm token in stdout path for job $jid: $stdout_path"
    validate_safe_abs "$stdout_path" \
      || die "unsafe or unresolved Slurm stdout path for job $jid: $stdout_path"
    case "$stdout_path" in
      "$SCRATCH"/*) ;;
      *) die "job stdout must remain below $SCRATCH: $stdout_path" ;;
    esac
    ssh_hopper bash -s -- "$jid" "$lines" "$stdout_path" <<'REMOTE'
set -euo pipefail
jid=$1
lines=$2
stdout=$3
[[ -n "$stdout" && -f "$stdout" ]] || { printf 'stdout not found for job %s: %s\n' "$jid" "$stdout" >&2; exit 3; }
tail -n "$lines" -- "$stdout"
REMOTE
    ;;

  registry)
    (( $# == 0 )) || die "usage: hopper.sh registry"
    if [[ ! -s "$REG" ]]; then
      printf '(no jobs tracked yet)\n'
      exit 0
    fi
    registry_view=$(awk -F '\t' -v OFS='\t' '
      BEGIN {
        print "JOB_ID","UTC","HOST","LOCAL_SCRIPT","LOCAL_SHA256","REMOTE_SCRIPT","REMOTE_SHA256","OUTPUT_PATH","REMOTE_RECEIPT","SBATCH_ARGS"
      }
      NF == 3 { print $1,$3,"(legacy)",$2,"-","-","-","-","-","-"; next }
      { print }
    ' "$REG")
    if command -v column >/dev/null 2>&1; then
      printf '%s\n' "$registry_view" | column -t -s $'\t'
    else
      printf '%s\n' "$registry_view"
    fi
    ;;

  *)
    usage
    exit 1
    ;;
esac
