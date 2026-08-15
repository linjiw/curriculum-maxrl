#!/usr/bin/env bash
# Exact local renderer for the sibling-only v4 remote-hardening candidate.
# The Python implementation permanently refuses `submit` in this candidate.
set -euo pipefail
umask 027
readonly HERE="$(cd "$(dirname "$0")" && pwd -P)"
exec /usr/bin/python3 -I -B "$HERE/hopper_v4_remote_hardened.py" "$@"
