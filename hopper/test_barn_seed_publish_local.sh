#!/bin/bash
# Network-free contract test for the terminal seed-block publisher embedded in
# sbatch/barn_seed_cpu.sbatch.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SBATCH="$HERE/sbatch/barn_seed_cpu.sbatch"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -f "$SBATCH" ]] || fail "missing seed sbatch: $SBATCH"

TMP=$(mktemp -d)
cleanup() {
  if [[ -n "${TMP:-}" && "$TMP" == /tmp/* && -d "$TMP" ]]; then
    rm -rf -- "$TMP"
  fi
}
trap cleanup EXIT

PUBLISHER="$TMP/publish.py"
awk '
  /^# BARN_SEED_PUBLISH_PY_BEGIN$/ { capture = 1; next }
  /^# BARN_SEED_PUBLISH_PY_END$/ { capture = 0; found = 1; next }
  capture { print }
  END { if (!found) exit 1 }
' "$SBATCH" > "$PUBLISHER" \
  || fail "could not extract embedded publisher"
python3 -m py_compile "$PUBLISHER"

make_block() {
  local block=$1 payload=$2
  mkdir -p -- "$block/results"
  printf 'artifact_type\tbarn_evidence_seed_complete\n' > "$block/COMPLETE"
  printf '%s\n' "$payload" > "$block/results/payload.txt"
}

run_publish() {
  python3 "$PUBLISHER" "$1" "$2"
}

expect_publish_failure() {
  local source=$1 destination=$2 label=$3
  if run_publish "$source" "$destination" > "$TMP/failure.out" 2>&1; then
    fail "$label unexpectedly published"
  fi
}

assert_claim_matches_complete() {
  local claim=$1 complete=$2
  [[ -f "$claim" && ! -L "$claim" ]] || fail "publication claim is not regular"
  [[ "$(stat -c '%d:%i' -- "$claim")" == \
     "$(stat -c '%d:%i' -- "$complete")" ]] \
    || fail "publication claim is not hard-linked to COMPLETE"
}

# Happy path: only the complete hidden directory becomes canonical, and the
# retained regular-file claim remains linked to its COMPLETE marker.
case_root="$TMP/happy"
mkdir -p -- "$case_root"
source="$case_root/.seed-1.stage-100"
destination="$case_root/seed-1"
claim="$case_root/.seed-1.publish-claim"
make_block "$source" winner
run_publish "$source" "$destination"
[[ ! -e "$source" && -d "$destination" ]] || fail "happy publish was not atomic"
[[ "$(< "$destination/results/payload.txt")" == winner ]] \
  || fail "happy publish payload changed"
assert_claim_matches_complete "$claim" "$destination/COMPLETE"

# A retry after a crash immediately after rename sees both the canonical block
# and retained claim, and must leave the original nonempty destination intact.
retry="$case_root/.seed-1.stage-101"
make_block "$retry" retry-must-not-win
expect_publish_failure "$retry" "$destination" "post-rename crash retry"
[[ -d "$retry" && "$(< "$destination/results/payload.txt")" == winner ]] \
  || fail "post-rename retry overwrote the canonical block"

# Every pre-existing destination kind fails closed before a claim is created.
for kind in nonempty_dir empty_dir regular_file symlink; do
  case_root="$TMP/existing-$kind"
  mkdir -p -- "$case_root"
  source="$case_root/.seed-2.stage-200"
  destination="$case_root/seed-2"
  claim="$case_root/.seed-2.publish-claim"
  make_block "$source" contender
  case "$kind" in
    nonempty_dir)
      mkdir -- "$destination"
      printf 'original\n' > "$destination/original.txt"
      ;;
    empty_dir) mkdir -- "$destination" ;;
    regular_file) printf 'original\n' > "$destination" ;;
    symlink) ln -s -- nowhere "$destination" ;;
  esac
  expect_publish_failure "$source" "$destination" "existing $kind"
  [[ -d "$source" && ! -e "$claim" && ! -L "$claim" ]] \
    || fail "existing $kind did not fail before claiming"
  if [[ "$kind" == nonempty_dir ]]; then
    [[ "$(< "$destination/original.txt")" == original ]] \
      || fail "nonempty destination was overwritten"
  fi
done

# A stale hard-link claim models a process crash after exclusive claim but
# before rename.  A retry cannot publish, and no canonical partial appears.
case_root="$TMP/crash-before-rename"
mkdir -p -- "$case_root"
source="$case_root/.seed-3.stage-300"
destination="$case_root/seed-3"
claim="$case_root/.seed-3.publish-claim"
make_block "$source" hidden-after-crash
ln -- "$source/COMPLETE" "$claim"
expect_publish_failure "$source" "$destination" "stale pre-rename claim"
[[ -d "$source" && ! -e "$destination" && ! -L "$destination" ]] \
  || fail "pre-rename crash exposed a canonical endpoint"
assert_claim_matches_complete "$claim" "$source/COMPLETE"

# Concurrent compliant contenders for one canonical seed are serialized by
# the sibling claim.  Exactly one complete block wins; the loser stays hidden.
for iteration in $(seq 1 24); do
  case_root="$TMP/concurrent-$iteration"
  mkdir -p -- "$case_root"
  source_a="$case_root/.seed-4.stage-a"
  source_b="$case_root/.seed-4.stage-b"
  destination="$case_root/seed-4"
  claim="$case_root/.seed-4.publish-claim"
  make_block "$source_a" A
  make_block "$source_b" B

  set +e
  run_publish "$source_a" "$destination" \
    > "$case_root/a.out" 2>&1 & pid_a=$!
  run_publish "$source_b" "$destination" \
    > "$case_root/b.out" 2>&1 & pid_b=$!
  wait "$pid_a"; status_a=$?
  wait "$pid_b"; status_b=$?
  set -e

  if (( (status_a == 0) + (status_b == 0) != 1 )); then
    fail "concurrent iteration $iteration did not produce exactly one winner"
  fi
  [[ -d "$destination" && -f "$destination/COMPLETE" \
     && -f "$destination/results/payload.txt" ]] \
    || fail "concurrent winner exposed an incomplete endpoint"
  winner=$(< "$destination/results/payload.txt")
  case "$winner" in
    A) [[ ! -e "$source_a" && -d "$source_b" ]] \
         || fail "concurrent A winner/loser state is inconsistent" ;;
    B) [[ ! -e "$source_b" && -d "$source_a" ]] \
         || fail "concurrent B winner/loser state is inconsistent" ;;
    *) fail "concurrent winner payload is invalid" ;;
  esac
  assert_claim_matches_complete "$claim" "$destination/COMPLETE"
done

printf 'BARN_SEED_PUBLISH_LOCAL_PASS\n'
