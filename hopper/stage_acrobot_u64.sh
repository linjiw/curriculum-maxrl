#!/usr/bin/env bash
# Stage the frozen Acrobot U64 campaign to Hopper as a content-addressed bundle.
#
# The bundle is identified by the SHA-256 of its own sorted manifest, so the
# sbatch can assert the exact bytes it is about to execute.  Staging is
# idempotent: an existing bundle with the same id is verified, never rewritten.
#
# usage: hopper/stage_acrobot_u64.sh
set -euo pipefail

REMOTE=${HOPPER_REMOTE:-lwang44@hopper.orc.gmu.edu}
SCRATCH=${HOPPER_SCRATCH:-/scratch/lwang44}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/acrobot_u64"
REMOTE_ROOT="$SCRATCH/acrobot_u64"

[[ -d "$SRC" ]] || { echo "missing $SRC" >&2; exit 1; }

# Refuse to stage a dirty campaign: the lock pins source hashes, and a dirty
# tree means the staged bytes are not the frozen bytes.
if ! git -C "$ROOT" diff --quiet -- acrobot_u64 || \
   [[ -n "$(git -C "$ROOT" ls-files --others --exclude-standard -- acrobot_u64)" ]]; then
  echo "acrobot_u64 has uncommitted changes; freeze before staging" >&2
  git -C "$ROOT" status --short -- acrobot_u64 >&2
  exit 1
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
STAGE="$WORK/acrobot_u64"
mkdir -p "$STAGE"

# Exactly the files needed to execute and to audit; never results.
tar -C "$SRC" -cf - \
  --exclude='results' --exclude='__pycache__' --exclude='*.pyc' \
  run_u64_tournament.py analyze_u64_tournament.py verify_vendor_lock.py \
  check_arms_differ.py ACROBOT_U64_LOCK.json ACROBOT_U64_PREREG.md \
  AMENDMENT_2026-08-15_runtime_scope.md vendor \
  | tar -C "$STAGE" -xf -

( cd "$STAGE" && find . -type f -print0 | LC_ALL=C sort -z \
    | xargs -0 sha256sum > "$WORK/MANIFEST" )
BUNDLE_SHA=$(sha256sum "$WORK/MANIFEST" | cut -d' ' -f1)
BUNDLE_ID=${BUNDLE_SHA:0:20}
cp "$WORK/MANIFEST" "$STAGE/MANIFEST.sha256"

echo "bundle id     : $BUNDLE_ID"
echo "manifest sha  : $BUNDLE_SHA"
echo "files         : $(wc -l < "$WORK/MANIFEST")"

DEST="$REMOTE_ROOT/bundles/$BUNDLE_ID"
ssh "$REMOTE" "mkdir -p '$DEST' '$REMOTE_ROOT/results' '$REMOTE_ROOT/logs'"

if ssh "$REMOTE" "[ -f '$DEST/MANIFEST.sha256' ]"; then
  echo "bundle already staged; verifying"
else
  tar -C "$WORK" -czf "$WORK/bundle.tgz" acrobot_u64
  scp -q "$WORK/bundle.tgz" "$REMOTE:$DEST/bundle.tgz"
  ssh "$REMOTE" "set -e; cd '$DEST'; tar --strip-components=1 -xzf bundle.tgz; rm -f bundle.tgz"
fi

REMOTE_SHA=$(ssh "$REMOTE" "set -e; cd '$DEST'; \
  find . -type f -not -name MANIFEST.sha256 -print0 | LC_ALL=C sort -z \
  | xargs -0 sha256sum | sha256sum | cut -d' ' -f1")

if [[ "$REMOTE_SHA" != "$BUNDLE_SHA" ]]; then
  echo "REMOTE BUNDLE MISMATCH" >&2
  echo "  local  $BUNDLE_SHA" >&2
  echo "  remote $REMOTE_SHA" >&2
  exit 1
fi

echo "remote verify : OK ($REMOTE_SHA)"
echo "bundle dir    : $DEST"
echo
echo "export ACROBOT_U64_BUNDLE_DIR=$DEST"
echo "export ACROBOT_U64_BUNDLE_SHA256=$BUNDLE_SHA"
