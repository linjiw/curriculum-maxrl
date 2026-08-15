#!/bin/bash
# Build and stage a content-addressed MAZE-SCORE source bundle.
#
# Usage:
#   bash hopper/stage_maze_score.sh engineering
#   bash hopper/stage_maze_score.sh evidence
#
# Evidence mode refuses any tracked or untracked worktree change. Engineering
# mode is allowed to bundle the current worktree, but records that fact and may
# never be used for a paper endpoint.
set -euo pipefail

MODE=${1:-}
case "$MODE" in
  engineering|evidence) ;;
  *) echo "usage: $0 engineering|evidence" >&2; exit 2 ;;
esac

HOST=${HOPPER_HOST:-lwang44@hopper.orc.gmu.edu}
SCRATCH=${HOPPER_SCRATCH:-/scratch/lwang44}
if [[ ! "$SCRATCH" =~ ^/scratch/[A-Za-z0-9._-]+$ ]]; then
  echo "unsafe HOPPER_SCRATCH: $SCRATCH" >&2
  exit 2
fi

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
COMMIT=$(git -C "$ROOT" rev-parse HEAD)
TREE_STATE=$(git -C "$ROOT" status --porcelain --untracked-files=all)
PREREG="$ROOT/hopper/MAZE_SCORE_PREREG.md"
if [[ "$MODE" == evidence ]] \
   && ! grep -q '^\*\*Status:\*\* FROZEN' "$PREREG"; then
  echo "evidence staging requires a preregistration with Status: FROZEN" >&2
  exit 1
fi
if [[ "$MODE" == evidence && -n "$TREE_STATE" ]]; then
  echo "evidence staging requires a clean committed worktree" >&2
  git -C "$ROOT" status --short >&2
  exit 1
fi

FILES=(
  curriculum_maxrl/estimators.py
  curriculum_maxrl/maze_gpu/maze_env.py
  curriculum_maxrl/maze_gpu/model.py
  curriculum_maxrl/maze_gpu/train.py
  curriculum_maxrl/maze_score/__init__.py
  curriculum_maxrl/maze_score/analyze_maze_score.py
  hopper/MAZE_SCORE_PREREG.md
  hopper/requirements-maze-hopper.lock
  hopper/sbatch/maze_score_array.sbatch
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
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)
MANIFEST_SHA=$(sha256sum "$TMP/source/SHA256SUMS" | awk '{print $1}')
BUNDLE_ID=${MANIFEST_SHA:0:20}
REMOTE_PARENT="$SCRATCH/maxrl/bundles/maze_score"
REMOTE_BUNDLE="$REMOTE_PARENT/$BUNDLE_ID"
REMOTE_STAGE="$REMOTE_PARENT/.stage-${BUNDLE_ID}-$$"
REMOTE_TEST_LOGS="$SCRATCH/maxrl/tests/logs"
REMOTE_EVIDENCE_LOGS="$SCRATCH/maxrl/maze_score/logs"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 \
          -o ServerAliveCountMax=3)
# Slurm opens #SBATCH --output before executing the job body, so these parent
# directories must exist before either smoke or evidence submission.
ssh "${SSH_OPTS[@]}" "$HOST" \
  "mkdir -p '$REMOTE_PARENT' '$REMOTE_TEST_LOGS' '$REMOTE_EVIDENCE_LOGS'"

if ssh "${SSH_OPTS[@]}" "$HOST" "test -d '$REMOTE_BUNDLE'"; then
  REMOTE_SHA=$(ssh "${SSH_OPTS[@]}" "$HOST" \
    "sha256sum '$REMOTE_BUNDLE/SHA256SUMS'" | awk '{print $1}')
  if [[ "$REMOTE_SHA" != "$MANIFEST_SHA" ]]; then
    echo "remote bundle id collision: $REMOTE_BUNDLE" >&2
    exit 1
  fi
  REUSED=true
else
  ssh "${SSH_OPTS[@]}" "$HOST" "mkdir '$REMOTE_STAGE'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
    "$TMP/source/" "$HOST:$REMOTE_STAGE/"
  ssh "${SSH_OPTS[@]}" "$HOST" \
    "cd '$REMOTE_STAGE' && sha256sum -c SHA256SUMS >/dev/null && mv '$REMOTE_STAGE' '$REMOTE_BUNDLE'"
  REUSED=false
fi

printf 'BUNDLE_ID=%s\n' "$BUNDLE_ID"
printf 'MANIFEST_SHA256=%s\n' "$MANIFEST_SHA"
printf 'REMOTE_BUNDLE=%s\n' "$REMOTE_BUNDLE"
printf 'MODE=%s\n' "$MODE"
printf 'REUSED=%s\n' "$REUSED"
