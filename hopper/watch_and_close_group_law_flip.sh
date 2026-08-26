#!/usr/bin/env bash
# Persistently monitor GROUP-LAW-FLIP v1 and run its single-use closure once.
# Only the marker/scheduler preflight in close_group_law_flip.sh is polled.
set -euo pipefail
umask 077

HERE=$(cd "$(dirname "$0")" && pwd)
CLOSE=$HERE/close_group_law_flip.sh
POLL_SECONDS=${GLF_POLL_SECONDS:-300}
MAX_POLLS=${GLF_MAX_POLLS:-0}

[[ "$POLL_SECONDS" =~ ^[0-9]+$ && "$POLL_SECONDS" -ge 60 \
   && "$POLL_SECONDS" -le 3600 ]] || {
  printf 'watch-and-close: GLF_POLL_SECONDS must be 60..3600\n' >&2
  exit 2
}
[[ "$MAX_POLLS" =~ ^[0-9]+$ ]] || {
  printf 'watch-and-close: GLF_MAX_POLLS must be a nonnegative integer\n' >&2
  exit 2
}
[[ -x "$CLOSE" ]] || {
  printf 'watch-and-close: closure script is not executable: %s\n' "$CLOSE" >&2
  exit 2
}

polls=0
while :; do
  ((polls += 1))
  set +e
  closure_output=$("$CLOSE" 2>&1)
  closure_rc=$?
  set -e
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  if (( closure_rc == 0 )); then
    printf '%s\n' "$closure_output"
    printf '%s WATCH_AND_CLOSE_COMPLETE polls=%s\n' "$now" "$polls"
    exit 0
  fi

  if grep -Fq \
      'campaign is not ready; retrieval and analysis were not started' \
      <<< "$closure_output"; then
    summary=$(grep '^live=' <<< "$closure_output" | tail -n 1)
    [[ -n "$summary" ]] || {
      printf '%s\n' "$closure_output" >&2
      printf 'watch-and-close: readiness refusal lacked a structural summary\n' >&2
      exit 2
    }
    printf '%s poll=%s %s\n' "$now" "$polls" "$summary"
  else
    printf '%s\n' "$closure_output" >&2
    printf 'watch-and-close: closure failed for a reason other than readiness\n' >&2
    exit "$closure_rc"
  fi

  if (( MAX_POLLS > 0 && polls >= MAX_POLLS )); then
    printf '%s WATCH_AND_CLOSE_MAX_POLLS polls=%s\n' "$now" "$polls"
    exit 3
  fi
  sleep "$POLL_SECONDS"
done
