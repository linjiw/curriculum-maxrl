#!/bin/bash
# Build and stage immutable, content-addressed BARN source and dataset bundles.
#
# Usage:
#   bash hopper/stage_barn_campaign.sh engineering [BARN_dataset.zip]
#   bash hopper/stage_barn_campaign.sh evidence [BARN_dataset.zip]
#
# BARN_DATASET_ZIP may be used instead of the optional archive argument.
# Evidence mode requires a literal FROZEN preregistration and requires every
# bundled repository path to match HEAD.  Unrelated worktree changes are
# deliberately ignored; no path below .codex is ever inspected or bundled.
set -euo pipefail

MODE=${1:-}
case "$MODE" in
  engineering|evidence) ;;
  *) echo "usage: $0 engineering|evidence [BARN_dataset.zip]" >&2; exit 2 ;;
esac
if (( $# > 2 )); then
  echo "usage: $0 engineering|evidence [BARN_dataset.zip]" >&2
  exit 2
fi

HOST=${HOPPER_HOST:-lwang44@hopper.orc.gmu.edu}
SCRATCH=${HOPPER_SCRATCH:-/scratch/lwang44}
if [[ ! "$SCRATCH" =~ ^/scratch/[A-Za-z0-9._-]+$ ]]; then
  echo "unsafe HOPPER_SCRATCH: $SCRATCH" >&2
  exit 2
fi

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
COMMIT=$(git -C "$ROOT" rev-parse --verify HEAD)
LOCAL_ARCHIVE=${2:-${BARN_DATASET_ZIP:-/home/robotixx/datasets/barn/BARN_dataset.zip}}
OFFICIAL_ARCHIVE_SHA256=5ad443412f6f2f38b6d0e1d330c9a820ab48e566553197459005e751711fe320

# This is an explicit runtime and provenance closure, not a recursive copy.
# In particular it excludes generated binaries, results, tests, caches, the
# rest of the worktree, and .codex under every mode.
FILES=(
  frontier_rl/__init__.py
  frontier_rl/interfaces.py
  frontier_rl/estimators.py
  frontier_rl/teacher.py
  frontier_rl/trainer.py
  frontier_rl/evaluation.py
  frontier_rl/adapters/__init__.py
  frontier_rl/adapters/barn_gazebo.py
  icra2027/__init__.py
  icra2027/freeze_pool_split.py
  icra2027/barn_campaign.py
  icra2027/analyze_campaign.py
  icra2027/merge_barn_campaign.py
  icra2027/select_barn_attempts.py
  icra2027/prereg_icra.md
  icra2027/barn_protocol.json
  icra2027/barn_split.json
  icra2027/barn_manifest.jsonl
  icra2027/assets/barn_diff_drive.sdf
  icra2027/barn_exact_step_plugin.cpp
  icra2027/gazebo_stepper.cpp
  icra2027/build_gazebo_stepper.sh
  icra2027/build_barn_manifest.py
  icra2027/verify_barn_smoke_package.py
  icra2027/receipts/barn_dataset_acquisition.json
  icra2027/receipts/barn_hopper_environment.json
  icra2027/receipts/barn_hopper_dataset_prepare.json
  icra2027/receipts/barn_hopper_training_smoke.json
  icra2027/receipts/barn_hopper_feasibility_projection.json
  icra2027/results/barn_backend_throughput_2026-08-14.json
  hopper/ros2-gazebo-classic.def
  hopper/stage_barn_campaign.sh
  hopper/submit_barn_campaign.sh
  hopper/finalize_barn_ledger.sh
  hopper/finalize_barn_campaign.sh
  hopper/sbatch/barn_seed_cpu.sbatch
  hopper/sbatch/barn_finalize_cpu.sbatch
  hopper/sbatch/barn_dataset_prepare.sbatch
  hopper/sbatch/barn_training_smoke.sbatch
)

DIRTY_RELEVANT=()
FILE_GIT_STATUS=()
FILE_PORCELAIN=()
FILE_WORKTREE_MODE=()
FILE_HEAD_MODE=()
FILE_WORKTREE_SHA256=()
FILE_HEAD_BLOB=()
for rel in "${FILES[@]}"; do
  if [[ "$rel" == .codex || "$rel" == .codex/* || "$rel" == */.codex/* ]]; then
    echo "internal error: .codex may not enter a BARN bundle: $rel" >&2
    exit 2
  fi
  if [[ ! "$rel" =~ ^[A-Za-z0-9._/-]+$ ]]; then
    echo "unsafe bundle path: $rel" >&2
    exit 2
  fi
  if [[ ! -f "$ROOT/$rel" || -L "$ROOT/$rel" ]]; then
    echo "missing or symlinked bundle input: $rel" >&2
    exit 1
  fi

  # A path is relevant-dirty if it is absent from HEAD, differs from HEAD in
  # content/mode, or has index/worktree state recorded by Git.  Checking each
  # explicit path makes unrelated dirty files harmless to evidence staging.
  porcelain=$(git -C "$ROOT" status --porcelain=v1 \
    --untracked-files=all -- "$rel")
  path_dirty=false
  if git -C "$ROOT" cat-file -e "HEAD:$rel" 2>/dev/null; then
    head_mode=$(git -C "$ROOT" ls-tree HEAD -- "$rel" | awk '{print $1}')
    head_blob=$(git -C "$ROOT" rev-parse "HEAD:$rel")
    git_status=clean
    if ! git -C "$ROOT" diff --quiet HEAD -- "$rel" \
       || [[ -n "$porcelain" ]]; then
      path_dirty=true
      git_status=dirty
    fi
  else
    head_mode=
    head_blob=
    git_status=untracked
    path_dirty=true
  fi
  if [[ -n "$porcelain" ]]; then
    porcelain=${porcelain:0:2}
  fi
  FILE_GIT_STATUS+=("$git_status")
  FILE_PORCELAIN+=("$porcelain")
  FILE_WORKTREE_MODE+=("$(stat -c '%a' -- "$ROOT/$rel")")
  FILE_HEAD_MODE+=("$head_mode")
  FILE_WORKTREE_SHA256+=("$(sha256sum -- "$ROOT/$rel" | awk '{print $1}')")
  FILE_HEAD_BLOB+=("$head_blob")
  if [[ "$path_dirty" == true ]]; then
    DIRTY_RELEVANT+=("$rel")
  fi
done

PREREG="$ROOT/icra2027/prereg_icra.md"
if [[ "$MODE" == evidence ]] \
   && ! grep -Eq '^\*\*Status:\*\* FROZEN[[:blank:]]*$' \
        "$PREREG"; then
  echo "evidence staging requires a preregistration with literal Status: FROZEN" >&2
  exit 1
fi
if [[ "$MODE" == evidence && ${#DIRTY_RELEVANT[@]} -ne 0 ]]; then
  echo "evidence staging requires every bundled path to match HEAD" >&2
  printf '  %s\n' "${DIRTY_RELEVANT[@]}" >&2
  exit 1
fi

if [[ ! -f "$LOCAL_ARCHIVE" || -L "$LOCAL_ARCHIVE" ]]; then
  echo "missing or symlinked official BARN archive: $LOCAL_ARCHIVE" >&2
  exit 1
fi
LOCAL_ARCHIVE_SHA256=$(sha256sum -- "$LOCAL_ARCHIVE" | awk '{print $1}')
if [[ "$LOCAL_ARCHIVE_SHA256" != "$OFFICIAL_ARCHIVE_SHA256" ]]; then
  echo "official BARN archive SHA-256 mismatch" >&2
  echo "  expected: $OFFICIAL_ARCHIVE_SHA256" >&2
  echo "  actual:   $LOCAL_ARCHIVE_SHA256" >&2
  exit 1
fi

TMP=$(mktemp -d)
cleanup() {
  if [[ -n "${TMP:-}" && "$TMP" == /tmp/* && -d "$TMP" ]]; then
    rm -rf -- "$TMP"
  fi
}
trap cleanup EXIT
mkdir -p "$TMP/source"

for index in "${!FILES[@]}"; do
  rel=${FILES[$index]}
  mkdir -p "$TMP/source/$(dirname "$rel")"
  cp -p -- "$ROOT/$rel" "$TMP/source/$rel"
  copied_sha=$(sha256sum -- "$TMP/source/$rel" | awk '{print $1}')
  copied_mode=$(stat -c '%a' -- "$TMP/source/$rel")
  if [[ "$copied_sha" != "${FILE_WORKTREE_SHA256[$index]}" \
     || "$copied_mode" != "${FILE_WORKTREE_MODE[$index]}" ]]; then
    echo "bundle input changed while staging: $rel" >&2
    exit 1
  fi
done

RELEVANT_DIRTY=false
[[ ${#DIRTY_RELEVANT[@]} -ne 0 ]] && RELEVANT_DIRTY=true
{
  printf '{\n'
  printf '  "bundle_schema": 1,\n'
  printf '  "mode": "%s",\n' "$MODE"
  printf '  "git_commit": "%s",\n' "$COMMIT"
  printf '  "worktree_dirty": %s,\n' "$RELEVANT_DIRTY"
  printf '  "dirty_scope": "bundled_paths_only",\n'
  printf '  "relevant_paths_match_head": %s,\n' \
    "$([[ "$RELEVANT_DIRTY" == false ]] && printf true || printf false)"
  printf '  "dirty_relevant_paths": ['
  for index in "${!DIRTY_RELEVANT[@]}"; do
    (( index == 0 )) || printf ', '
    printf '"%s"' "${DIRTY_RELEVANT[$index]}"
  done
  printf '],\n'
  printf '  "files": [\n'
  for index in "${!FILES[@]}"; do
    printf '    {"path": "%s", "git_status": "%s", ' \
      "${FILES[$index]}" "${FILE_GIT_STATUS[$index]}"
    printf '"porcelain": "%s", "worktree_mode": "%s", ' \
      "${FILE_PORCELAIN[$index]}" "${FILE_WORKTREE_MODE[$index]}"
    if [[ -n "${FILE_HEAD_MODE[$index]}" ]]; then
      printf '"head_mode": "%s", ' "${FILE_HEAD_MODE[$index]}"
    else
      printf '"head_mode": null, '
    fi
    printf '"worktree_sha256": "%s", ' \
      "${FILE_WORKTREE_SHA256[$index]}"
    if [[ -n "${FILE_HEAD_BLOB[$index]}" ]]; then
      printf '"head_blob": "%s"}' "${FILE_HEAD_BLOB[$index]}"
    else
      printf '"head_blob": null}'
    fi
    if (( index + 1 < ${#FILES[@]} )); then
      printf ','
    fi
    printf '\n'
  done
  printf '  ]\n}\n'
} > "$TMP/source/SOURCE_STATE.json"

(
  cd "$TMP/source"
  find . -type f ! -name SHA256SUMS -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)
SOURCE_SHA256=$(sha256sum "$TMP/source/SHA256SUMS" | awk '{print $1}')
SOURCE_ID=${SOURCE_SHA256:0:20}
DATASET_ID=${OFFICIAL_ARCHIVE_SHA256:0:20}

SOURCE_PARENT="$SCRATCH/maxrl/bundles/barn_source"
SOURCE_BUNDLE="$SOURCE_PARENT/$SOURCE_ID"
SOURCE_STAGE="$SOURCE_PARENT/.stage-$SOURCE_ID-$$-${RANDOM:-0}"
DATASET_PARENT="$SCRATCH/maxrl/bundles/barn_dataset"
DATASET_DIR="$DATASET_PARENT/$DATASET_ID"
DATASET_STAGE="$DATASET_PARENT/.stage-$DATASET_ID-$$-${RANDOM:-0}"
DATASET_ARCHIVE="$DATASET_DIR/BARN_dataset.zip"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 \
          -o ServerAliveCountMax=3)
ssh "${SSH_OPTS[@]}" "$HOST" \
  "mkdir -p '$SOURCE_PARENT' '$DATASET_PARENT'"

remote_kind() {
  local path=$1
  ssh "${SSH_OPTS[@]}" "$HOST" \
    "if test -L '$path'; then printf symlink; elif test -d '$path'; then printf directory; elif test -e '$path'; then printf other; else printf absent; fi"
}

verify_remote_source() {
  ssh "${SSH_OPTS[@]}" "$HOST" "
    set -eu
    test -d '$SOURCE_BUNDLE'
    test ! -L '$SOURCE_BUNDLE'
    cd '$SOURCE_BUNDLE'
    test -f SHA256SUMS
    test ! -L SHA256SUMS
    test \"\$(sha256sum SHA256SUMS | cut -d ' ' -f 1)\" = '$SOURCE_SHA256'
    sha256sum -c SHA256SUMS >/dev/null
    test -z \"\$(find . -type l -print -quit)\"
    test -z \"\$(find . -type d -empty -print -quit)\"
    listed=\$(wc -l < SHA256SUMS)
    present=\$(find . -type f | wc -l)
    test \"\$present\" -eq \"\$((listed + 1))\"
    python3 -c 'import hashlib, json, os, stat
state = json.load(open(\"SOURCE_STATE.json\", encoding=\"utf-8\"))
for row in state[\"files\"]:
    info = os.lstat(row[\"path\"])
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise SystemExit(\"non-regular state file \" + row[\"path\"])
    actual = format(stat.S_IMODE(info.st_mode), \"o\")
    if actual != row[\"worktree_mode\"]:
        raise SystemExit(\"mode mismatch for \" + row[\"path\"])
    digest = hashlib.sha256(open(row[\"path\"], \"rb\").read()).hexdigest()
    if digest != row[\"worktree_sha256\"]:
        raise SystemExit(\"hash mismatch for \" + row[\"path\"])'
  "
}

verify_remote_dataset() {
  ssh "${SSH_OPTS[@]}" "$HOST" "
    set -eu
    test -d '$DATASET_DIR'
    test ! -L '$DATASET_DIR'
    test -f '$DATASET_ARCHIVE'
    test ! -L '$DATASET_ARCHIVE'
    test \"\$(sha256sum '$DATASET_ARCHIVE' | cut -d ' ' -f 1)\" = '$OFFICIAL_ARCHIVE_SHA256'
    test -z \"\$(find '$DATASET_DIR' -type l -print -quit)\"
    entries=\$(find '$DATASET_DIR' -mindepth 1 -maxdepth 1 | wc -l)
    test \"\$entries\" -eq 1
  "
}

# The per-target mkdir is an atomic lock.  A concurrent winner is accepted
# only after the caller verifies the complete content at the final path.
promote_remote_dir() {
  local stage=$1
  local target=$2
  local lock="$target.lock"
  ssh "${SSH_OPTS[@]}" "$HOST" "
    set -eu
    mkdir '$lock'
    release_lock() { rmdir '$lock'; }
    trap release_lock EXIT HUP INT TERM
    if test -e '$target' || test -L '$target'; then
      exit 17
    fi
    mv -- '$stage' '$target'
  "
}

SOURCE_KIND=$(remote_kind "$SOURCE_BUNDLE")
case "$SOURCE_KIND" in
  directory)
    if ! verify_remote_source; then
      echo "remote source bundle id collision or corruption: $SOURCE_BUNDLE" >&2
      exit 1
    fi
    SOURCE_REUSED=true
    ;;
  absent)
    ssh "${SSH_OPTS[@]}" "$HOST" "mkdir '$SOURCE_STAGE'"
    rsync -a \
      -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
      "$TMP/source/" "$HOST:$SOURCE_STAGE/"
    # Verify the staging directory with the same strict checks used on reuse.
    SOURCE_BUNDLE_SAVED=$SOURCE_BUNDLE
    SOURCE_BUNDLE=$SOURCE_STAGE
    if ! verify_remote_source; then
      echo "transferred BARN source bundle failed verification" >&2
      exit 1
    fi
    SOURCE_BUNDLE=$SOURCE_BUNDLE_SAVED
    if promote_remote_dir "$SOURCE_STAGE" "$SOURCE_BUNDLE"; then
      SOURCE_REUSED=false
    elif [[ "$(remote_kind "$SOURCE_BUNDLE")" == directory ]] \
         && verify_remote_source; then
      SOURCE_REUSED=true
    else
      echo "failed to publish source bundle atomically: $SOURCE_BUNDLE" >&2
      exit 1
    fi
    ;;
  *)
    echo "remote source bundle id collision: $SOURCE_BUNDLE ($SOURCE_KIND)" >&2
    exit 1
    ;;
esac

DATASET_KIND=$(remote_kind "$DATASET_DIR")
case "$DATASET_KIND" in
  directory)
    if ! verify_remote_dataset; then
      echo "remote BARN dataset id collision or corruption: $DATASET_DIR" >&2
      exit 1
    fi
    DATASET_REUSED=true
    ;;
  absent)
    ssh "${SSH_OPTS[@]}" "$HOST" "mkdir '$DATASET_STAGE'"
    rsync -a \
      -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
      "$LOCAL_ARCHIVE" "$HOST:$DATASET_STAGE/BARN_dataset.zip"
    DATASET_DIR_SAVED=$DATASET_DIR
    DATASET_ARCHIVE_SAVED=$DATASET_ARCHIVE
    DATASET_DIR=$DATASET_STAGE
    DATASET_ARCHIVE="$DATASET_STAGE/BARN_dataset.zip"
    if ! verify_remote_dataset; then
      echo "transferred official BARN archive failed verification" >&2
      exit 1
    fi
    DATASET_DIR=$DATASET_DIR_SAVED
    DATASET_ARCHIVE=$DATASET_ARCHIVE_SAVED
    if promote_remote_dir "$DATASET_STAGE" "$DATASET_DIR"; then
      DATASET_REUSED=false
    elif [[ "$(remote_kind "$DATASET_DIR")" == directory ]] \
         && verify_remote_dataset; then
      DATASET_REUSED=true
    else
      echo "failed to publish BARN dataset atomically: $DATASET_DIR" >&2
      exit 1
    fi
    ;;
  *)
    echo "remote BARN dataset id collision: $DATASET_DIR ($DATASET_KIND)" >&2
    exit 1
    ;;
esac

printf 'BARN_SOURCE_BUNDLE_DIR=%s\n' "$SOURCE_BUNDLE"
printf 'BARN_SOURCE_SHA256=%s\n' "$SOURCE_SHA256"
printf 'BARN_DATASET_ARCHIVE=%s\n' "$DATASET_ARCHIVE"
printf 'BARN_DATASET_ARCHIVE_SHA256=%s\n' "$OFFICIAL_ARCHIVE_SHA256"
printf 'MODE=%s\n' "$MODE"
printf 'SOURCE_REUSED=%s\n' "$SOURCE_REUSED"
printf 'DATASET_REUSED=%s\n' "$DATASET_REUSED"
