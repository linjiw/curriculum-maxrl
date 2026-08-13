#!/bin/bash
# Reusable Hopper workflow: submit -> track -> watch -> fetch.
#
#   hopper.sh submit <sbatch-file> [sbatch-args...]   stage + submit, record in registry
#   hopper.sh status [jobid]                          live queue + sacct for tracked jobs
#   hopper.sh watch <jobid> [poll_s]                  block until terminal, print states
#   hopper.sh fetch <remote-path> <local-path>        rsync results back
#   hopper.sh logs <jobid> [n]                        tail the job's stdout
#   hopper.sh push <local-path> <remote-rel-path>     stage code/data to scratch
#   hopper.sh registry                                show tracked jobs
#
# Registry lives at hopper/.job_registry (tracked in git-ignored form).
set -euo pipefail

HOST=${HOPPER_HOST:-lwang44@hopper.orc.gmu.edu}
SCRATCH=${HOPPER_SCRATCH:-/scratch/lwang44}
HERE="$(cd "$(dirname "$0")" && pwd)"
REG="$HERE/.job_registry"

ssh_q() { ssh -o BatchMode=yes "$HOST" "$@" 2>/dev/null | grep -v '^|' | grep -v '^\*'; }

case "${1:-}" in
  submit)
    shift
    script="$1"; shift
    scp -q "$script" "$HOST:$SCRATCH/sbatch/$(basename "$script")"
    out=$(ssh_q "cd $SCRATCH && sbatch $* sbatch/$(basename "$script")")
    jid=$(echo "$out" | grep -o '[0-9]\+' | head -1)
    printf '%s\t%s\t%s\n' "$jid" "$(basename "$script")" "$(date -Is)" >> "$REG"
    echo "submitted $jid ($(basename "$script"))"
    ;;
  status)
    if [ -n "${2:-}" ]; then
      ssh_q "squeue -j $2 -o '%.10i %.20j %.8T %.10M %.16R' 2>/dev/null; sacct -j $2 --format=JobID%14,State,Elapsed -X -n"
    else
      echo "=== live queue ==="; ssh_q "squeue -u \$USER -o '%.10i %.20j %.8T %.10M %.16R'"
    fi
    ;;
  watch)
    jid="$2"; poll="${3:-120}"
    while :; do
      st=$(ssh_q "sacct -j $jid --format=State -n -X" | tr -d ' ' | sort -u | tr '\n' ',')
      case "$st" in
        ""|*PENDING*|*RUNNING*) sleep "$poll" ;;
        *) echo "job $jid terminal: $st"
           ssh_q "sacct -j $jid --format=JobID%14,State,Elapsed -X -n"; break ;;
      esac
    done
    ;;
  fetch)
    mkdir -p "$(dirname "$3")"
    rsync -az "$HOST:$2" "$3" && echo "fetched -> $3"
    ;;
  push)
    ssh_q "mkdir -p $SCRATCH/$(dirname "$3")"
    rsync -az --exclude '__pycache__' "$2" "$HOST:$SCRATCH/$3" && echo "pushed -> $SCRATCH/$3"
    ;;
  logs)
    ssh_q "tail -${3:-40} \$(ls -t $SCRATCH/curriculum-maxrl-runtime/logs/*${2}* $SCRATCH/*/logs/*${2}* 2>/dev/null | head -1)"
    ;;
  registry)
    [ -f "$REG" ] && column -t "$REG" || echo "(no jobs tracked yet)"
    ;;
  *)
    sed -n '2,18p' "$0"; exit 1 ;;
esac
