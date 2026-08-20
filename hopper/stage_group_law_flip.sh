#!/usr/bin/env bash
# Build and stage a content-addressed GROUP-LAW-FLIP source bundle.
# Usage: bash hopper/stage_group_law_flip.sh engineering|evidence
set -euo pipefail

MODE=${1:-}
case "$MODE" in
  engineering|evidence) ;;
  *) echo "usage: $0 engineering|evidence" >&2; exit 2 ;;
esac

HOST=${HOPPER_HOST:-lwang44@hopper.orc.gmu.edu}
SCRATCH=${HOPPER_SCRATCH:-/scratch/lwang44}
[[ "$SCRATCH" =~ ^/scratch/[A-Za-z0-9._-]+$ ]] \
  || { echo "unsafe HOPPER_SCRATCH: $SCRATCH" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
COMMIT=$(git -C "$ROOT" rev-parse HEAD)
TREE_STATE=$(git -C "$ROOT" status --porcelain --untracked-files=all)
PREREG="$ROOT/granularity_flip/GROUP_LAW_FLIP_PREREG.md"
if [[ "$MODE" == evidence ]] && ! grep -q '^\*\*Status:\*\* FROZEN' "$PREREG"; then
  echo "evidence staging requires a frozen preregistration" >&2
  exit 1
fi
if [[ "$MODE" == evidence && -n "$TREE_STATE" ]]; then
  echo "evidence staging requires a clean committed worktree" >&2
  git -C "$ROOT" status --short >&2
  exit 1
fi

FILES=(
  curriculum_maxrl/__init__.py
  curriculum_maxrl/count_law_stats.py
  curriculum_maxrl/estimators.py
  curriculum_maxrl/maze_gpu/maze_env.py
  curriculum_maxrl/maze_gpu/model.py
  curriculum_maxrl/maze_gpu/train.py
  curriculum_maxrl/group_law_flip/__init__.py
  curriculum_maxrl/group_law_flip/analyze_group_law_flip.py
  granularity_flip/GROUP_LAW_FLIP_PREREG.md
  granularity_flip/GROUP_LAW_FLIP_POWER_MEMO_2026-08-20.md
  granularity_flip/GROUP_LAW_FLIP_DELIVERY_REPLAY_2026-08-20.json
  hopper/requirements-maze-hopper.lock
  hopper/sbatch/group_law_flip_array.sbatch
  hopper/sbatch/group_law_flip_smoke.sbatch
)
TMP=$(mktemp -d)
cleanup() {
  if [[ -n "${TMP:-}" && "$TMP" == /tmp/* && -d "$TMP" ]]; then
    rm -rf -- "$TMP"
  fi
}
trap cleanup EXIT
mkdir -p "$TMP/source"
for rel in "${FILES[@]}"; do
  if [[ ! -f "$ROOT/$rel" || -L "$ROOT/$rel" ]]; then
    echo "missing or symlinked bundle input: $rel" >&2
    exit 1
  fi
  mkdir -p "$TMP/source/$(dirname "$rel")"
  cp -- "$ROOT/$rel" "$TMP/source/$rel"
done

DIRTY=false
[[ -n "$TREE_STATE" ]] && DIRTY=true
printf '{\n  "bundle_schema": 1,\n  "mode": "%s",\n  "git_commit": "%s",\n  "worktree_dirty": %s\n}\n' \
  "$MODE" "$COMMIT" "$DIRTY" > "$TMP/source/SOURCE_STATE.json"
(
  cd "$TMP/source"
  find . -type f ! -name SHA256SUMS -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS
)
MANIFEST_SHA=$(sha256sum "$TMP/source/SHA256SUMS" | awk '{print $1}')
BUNDLE_ID=${MANIFEST_SHA:0:20}
REMOTE_PARENT="$SCRATCH/maxrl/bundles/group_law_flip"
REMOTE_BUNDLE="$REMOTE_PARENT/$BUNDLE_ID"
REMOTE_STAGE="$REMOTE_PARENT/.stage-${BUNDLE_ID}-$$"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 \
          -o ServerAliveCountMax=3)
ssh "${SSH_OPTS[@]}" "$HOST" \
  "mkdir -p '$REMOTE_PARENT' '$SCRATCH/maxrl/tests/logs' '$SCRATCH/maxrl/group_law_flip/logs'"
if ssh "${SSH_OPTS[@]}" "$HOST" "test -d '$REMOTE_BUNDLE'"; then
  REMOTE_SHA=$(ssh "${SSH_OPTS[@]}" "$HOST" \
    "sha256sum '$REMOTE_BUNDLE/SHA256SUMS'" | awk '{print $1}')
  [[ "$REMOTE_SHA" == "$MANIFEST_SHA" ]] \
    || { echo "remote bundle id collision: $REMOTE_BUNDLE" >&2; exit 1; }
  REUSED=true
else
  ssh "${SSH_OPTS[@]}" "$HOST" "mkdir '$REMOTE_STAGE'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
    "$TMP/source/" "$HOST:$REMOTE_STAGE/"
  ssh "${SSH_OPTS[@]}" "$HOST" \
    "cd '$REMOTE_STAGE' && sha256sum -c --strict SHA256SUMS >/dev/null && mv '$REMOTE_STAGE' '$REMOTE_BUNDLE'"
  REUSED=false
fi
printf 'BUNDLE_ID=%s\n' "$BUNDLE_ID"
printf 'MANIFEST_SHA256=%s\n' "$MANIFEST_SHA"
printf 'REMOTE_BUNDLE=%s\n' "$REMOTE_BUNDLE"
printf 'MODE=%s\n' "$MODE"
printf 'REUSED=%s\n' "$REUSED"
