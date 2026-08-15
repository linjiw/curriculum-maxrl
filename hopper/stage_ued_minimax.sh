#!/usr/bin/env bash
# Build a source-faithful, content-addressed minimax + Frontier overlay bundle.
#
# Usage:
#   bash hopper/stage_ued_minimax.sh local NEW_OUTPUT_DIR
#   bash hopper/stage_ued_minimax.sh stage
#
# MINIMAX_SOURCE_DIR may name an existing clean or dirty clone. Only its pinned
# Git commit object is bundled; worktree modifications are never copied. If it
# is unset, the script obtains a temporary read-only clone from the official
# upstream repository.
set -euo pipefail
umask 027

readonly PINNED_COMMIT=d053054c5290a04c1c4cd8b55704d999cad73e30
readonly UPSTREAM_URL=https://github.com/facebookresearch/minimax.git
readonly MODE="${1:-}"
case "$MODE" in
  local)
    (( $# == 2 )) || { echo "usage: $0 local NEW_OUTPUT_DIR" >&2; exit 2; }
    ;;
  stage)
    (( $# == 1 )) || { echo "usage: $0 stage" >&2; exit 2; }
    ;;
  *)
    echo "usage: $0 local NEW_OUTPUT_DIR | stage" >&2
    exit 2
    ;;
esac

readonly HERE="$(cd "$(dirname "$0")" && pwd)"
readonly ROOT="$(cd "$HERE/.." && pwd)"
readonly OVERLAY_ROOT="$ROOT/ued_benchmark"
readonly LOCK="$HERE/requirements-ued-minimax-hopper.lock"
readonly REQUIRED_OVERLAY_SCRIPT=ued_benchmark/scripts/apply_minimax_overlay.py
readonly REQUIRED_OVERLAY_MODULE=ued_benchmark/overlay/minimax/util/rl/frontier_activity.py

for path in "$LOCK" "$ROOT/$REQUIRED_OVERLAY_SCRIPT" "$ROOT/$REQUIRED_OVERLAY_MODULE"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "missing or symbolic required UED input: ${path#"$ROOT/"}" >&2
    exit 1
  }
done
# The local JAX 0.6.2/Blackwell probes are intentionally disjoint from this
# source-faithful JAX 0.4.31 Hopper lane. Their contents must neither enter nor
# perturb the content address of this bundle.
if find "$OVERLAY_ROOT" \
    \( -path "$OVERLAY_ROOT/blackwell_probe" \
       -o -path "$OVERLAY_ROOT/blackwell_training_probe" \) -prune \
    -o -type l -print -quit | grep -q .; then
  echo "ued_benchmark must not contain symbolic links" >&2
  exit 1
fi

readonly TMP="$(mktemp -d /tmp/ued-minimax-stage.XXXXXX)"
REMOTE_STAGE=""
cleanup() {
  if [[ -n "${TMP:-}" && "$TMP" == /tmp/ued-minimax-stage.* && -d "$TMP" ]]; then
    rm -rf -- "$TMP"
  fi
  if [[ -n "${REMOTE_STAGE:-}" && -n "${REMOTE_PARENT:-}" \
        && "${REMOTE_STAGE:-}" == "${REMOTE_PARENT:-}"/.stage-* \
        && -n "${HOST:-}" && -n "${SSH_OPTS[*]:-}" ]]; then
    ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- "$REMOTE_STAGE" "$REMOTE_PARENT" <<'REMOTE' || true
set -euo pipefail
stage=$1
parent=$2
case "$stage" in
  "$parent"/.stage-*) rm -rf -- "$stage" ;;
  *) echo "refusing unsafe remote stage cleanup: $stage" >&2; exit 2 ;;
esac
REMOTE
  fi
}
trap cleanup EXIT
mkdir -p "$TMP/source/upstream" "$TMP/source/ued_benchmark" "$TMP/source/hopper/sbatch"

SOURCE_DIR="${MINIMAX_SOURCE_DIR:-}"
if [[ -z "$SOURCE_DIR" ]]; then
  SOURCE_DIR="$TMP/minimax-clone"
  git clone --quiet --no-checkout "$UPSTREAM_URL" "$SOURCE_DIR"
  git -C "$SOURCE_DIR" checkout --quiet --detach "$PINNED_COMMIT"
fi
[[ -d "$SOURCE_DIR/.git" || -f "$SOURCE_DIR/.git" ]] || {
  echo "MINIMAX_SOURCE_DIR is not a Git checkout: $SOURCE_DIR" >&2
  exit 1
}
readonly SOURCE_HEAD="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
[[ "$SOURCE_HEAD" == "$PINNED_COMMIT" ]] || {
  echo "minimax HEAD must be $PINNED_COMMIT; got $SOURCE_HEAD" >&2
  exit 1
}
readonly UPSTREAM_TREE="$(git -C "$SOURCE_DIR" rev-parse 'HEAD^{tree}')"
readonly UPSTREAM_BUNDLE="upstream/minimax-${PINNED_COMMIT}.bundle"
git -C "$SOURCE_DIR" bundle create "$TMP/source/$UPSTREAM_BUNDLE" HEAD
git bundle verify "$TMP/source/$UPSTREAM_BUNDLE" >/dev/null
readonly UPSTREAM_BUNDLE_SHA256="$(sha256sum "$TMP/source/$UPSTREAM_BUNDLE" | awk '{print $1}')"
printf '%s  %s\n' "$UPSTREAM_BUNDLE_SHA256" "$(basename "$UPSTREAM_BUNDLE")" \
  > "$TMP/source/upstream/SHA256SUMS"

# Copy the complete authored UED tree while excluding interpreter caches. New
# overlay config/tests/docs are picked up automatically on every staging run.
while IFS= read -r -d '' src; do
  rel=${src#"$OVERLAY_ROOT/"}
  mkdir -p "$TMP/source/ued_benchmark/$(dirname "$rel")"
  cp -- "$src" "$TMP/source/ued_benchmark/$rel"
done < <(find "$OVERLAY_ROOT" \
  \( -path "$OVERLAY_ROOT/blackwell_probe" \
     -o -path "$OVERLAY_ROOT/blackwell_training_probe" \) -prune \
  -o -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
  | LC_ALL=C sort -z)

for required in "${REQUIRED_OVERLAY_SCRIPT#ued_benchmark/}" \
                "${REQUIRED_OVERLAY_MODULE#ued_benchmark/}"; do
  [[ -f "$TMP/source/ued_benchmark/$required" ]] || {
    echo "required overlay file was not staged: ued_benchmark/$required" >&2
    exit 1
  }
done
(
  cd "$TMP/source/ued_benchmark"
  find . -type f ! -name OVERLAY_SHA256SUMS -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > OVERLAY_SHA256SUMS
  sha256sum -c --strict OVERLAY_SHA256SUMS >/dev/null
)
readonly OVERLAY_MANIFEST_SHA256="$(sha256sum "$TMP/source/ued_benchmark/OVERLAY_SHA256SUMS" | awk '{print $1}')"

HOPPER_FILES=(
  hopper/hopper.sh
  hopper/requirements-ued-minimax-hopper.lock
  hopper/setup_ued_minimax_env.sh
  hopper/stage_ued_minimax.sh
  hopper/finalize_ued_minimax_terminal_chain.py
  hopper/sbatch/ued_minimax_gpu_smoke.sbatch
  hopper/sbatch/ued_minimax_one_update_smoke.sbatch
  hopper/sbatch/ued_minimax_terminal_chain_smoke.sbatch
  hopper/test_hopper_local.sh
  hopper/test_ued_minimax_local.sh
  hopper/test_ued_minimax_one_update_local.sh
  hopper/test_ued_minimax_terminal_chain_local.sh
)
for rel in "${HOPPER_FILES[@]}"; do
  [[ -f "$ROOT/$rel" && ! -L "$ROOT/$rel" ]] || {
    echo "missing or symbolic bundle input: $rel" >&2
    exit 1
  }
  mkdir -p "$TMP/source/$(dirname "$rel")"
  cp -- "$ROOT/$rel" "$TMP/source/$rel"
done

readonly ENV_LOCK_SHA256="$(sha256sum "$LOCK" | awk '{print $1}')"
readonly PROJECT_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
PROJECT_DIRTY=false
[[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] && PROJECT_DIRTY=true
UED_BUNDLE_STATE_PATH="$TMP/source/BUNDLE_STATE.json" \
UED_PROJECT_COMMIT="$PROJECT_COMMIT" \
UED_PROJECT_DIRTY="$PROJECT_DIRTY" \
UED_UPSTREAM_URL="$UPSTREAM_URL" \
UED_UPSTREAM_COMMIT="$PINNED_COMMIT" \
UED_UPSTREAM_TREE="$UPSTREAM_TREE" \
UED_UPSTREAM_BUNDLE="$UPSTREAM_BUNDLE" \
UED_UPSTREAM_BUNDLE_SHA256="$UPSTREAM_BUNDLE_SHA256" \
UED_OVERLAY_MANIFEST_SHA256="$OVERLAY_MANIFEST_SHA256" \
UED_ENV_LOCK_SHA256="$ENV_LOCK_SHA256" \
python3 -I -B - <<'PY'
import json
import os
from pathlib import Path

record = {
    "bundle_schema": 4,
    "purpose": "bounded UED minimax/AMaze engineering smokes only",
    "paper_evidence": False,
    "workspace_tree_exclusions": [
        "ued_benchmark/blackwell_probe/**",
        "ued_benchmark/blackwell_training_probe/**",
    ],
    "allowed_engineering_endpoints": [
        "frontier_exact_grouped_one_update",
        "frontier_terminal_chain_components",
        "gpu_import_formula_jit",
    ],
    "max_student_updates": 1,
    "terminal_chain_contract": {
        "analyzer_eligible": False,
        "actual_external_evaluation": True,
        "paper_evidence": False,
        "phase_a": "slurm_closed_training_and_actual_external_evaluation_components",
        "phase_a_submission_export": "explicit_ued_allowlist_no_all",
        "phase_a_python_flags": "-I -B",
        "phase_b": "post_terminal_local_atomic_engineering_assembly",
        "phase_b_python": "isolated_clean_python_3.10.20_venv",
        "phase_b_python_flags": "-I -B",
        "finalizer_self_bound": True,
        "post_terminal_fetch_receipt_schema": 2,
        "production_analyzer_invoked": False,
        "submission_receipt_required": True,
        "terminal_receipt_schema": 2,
        "terminal_sacct_phase": "post_completion_local_finalize",
    },
    "resource_accounting_contract": {
        "in_process": "python_resource_getrusage_self_and_monotonic_ns",
        "external_authority": "terminal_slurm_sacct",
        "host_gnu_time_required": False,
    },
    "project_git_commit": os.environ["UED_PROJECT_COMMIT"],
    "project_worktree_dirty": os.environ["UED_PROJECT_DIRTY"] == "true",
    "upstream_repository": os.environ["UED_UPSTREAM_URL"],
    "upstream_commit": os.environ["UED_UPSTREAM_COMMIT"],
    "upstream_tree_git_sha1": os.environ["UED_UPSTREAM_TREE"],
    "upstream_git_bundle": os.environ["UED_UPSTREAM_BUNDLE"],
    "upstream_git_bundle_sha256": os.environ["UED_UPSTREAM_BUNDLE_SHA256"],
    "overlay_manifest_sha256": os.environ["UED_OVERLAY_MANIFEST_SHA256"],
    "environment_lock_sha256": os.environ["UED_ENV_LOCK_SHA256"],
}
Path(os.environ["UED_BUNDLE_STATE_PATH"]).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
(
  cd "$TMP/source"
  # Exclude only the root manifest being generated; nested manifests are
  # themselves inputs and therefore remain bound by this outer manifest.
  find . -type f ! -path ./SHA256SUMS -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c --strict SHA256SUMS >/dev/null
)
readonly MANIFEST_SHA256="$(sha256sum "$TMP/source/SHA256SUMS" | awk '{print $1}')"
readonly BUNDLE_ID="${MANIFEST_SHA256:0:20}"

emit_receipt() {
  printf 'UED_BUNDLE_ID=%s\n' "$BUNDLE_ID"
  printf 'UED_BUNDLE_MANIFEST_SHA256=%s\n' "$MANIFEST_SHA256"
  printf 'UED_UPSTREAM_COMMIT=%s\n' "$PINNED_COMMIT"
  printf 'UED_UPSTREAM_TREE=%s\n' "$UPSTREAM_TREE"
  printf 'UED_UPSTREAM_BUNDLE_SHA256=%s\n' "$UPSTREAM_BUNDLE_SHA256"
  printf 'UED_OVERLAY_MANIFEST_SHA256=%s\n' "$OVERLAY_MANIFEST_SHA256"
  printf 'UED_ENV_LOCK_SHA256=%s\n' "$ENV_LOCK_SHA256"
}

if [[ "$MODE" == local ]]; then
  OUTPUT_DIR=$2
  [[ "$OUTPUT_DIR" == /* ]] || OUTPUT_DIR="$PWD/$OUTPUT_DIR"
  [[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || {
    echo "local output must not exist: $OUTPUT_DIR" >&2
    exit 1
  }
  mkdir -p "$(dirname "$OUTPUT_DIR")"
  mv "$TMP/source" "$OUTPUT_DIR"
  printf 'UED_LOCAL_BUNDLE=%s\n' "$OUTPUT_DIR"
  emit_receipt
  exit 0
fi

readonly HOST="${HOPPER_HOST:-lwang44@hopper.orc.gmu.edu}"
readonly SCRATCH="${HOPPER_SCRATCH:-/scratch/lwang44}"
[[ "$HOST" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+$ ]] || {
  echo "unsafe HOPPER_HOST: $HOST" >&2
  exit 2
}
[[ "$SCRATCH" =~ ^/scratch/[A-Za-z0-9._-]+$ ]] || {
  echo "unsafe HOPPER_SCRATCH: $SCRATCH" >&2
  exit 2
}
readonly REMOTE_PARENT="$SCRATCH/maxrl/bundles/ued_minimax"
readonly REMOTE_BUNDLE="$REMOTE_PARENT/$BUNDLE_ID"
REMOTE_STAGE="$REMOTE_PARENT/.stage-${BUNDLE_ID}-$$"
readonly REMOTE_LOGS="$SCRATCH/maxrl/tests/logs"
readonly REMOTE_RESULTS="$SCRATCH/maxrl/tests/ued-minimax-gpu-smoke"
readonly REMOTE_WORK="$SCRATCH/maxrl/tests/work"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 \
          -o ServerAliveCountMax=3)

ssh "${SSH_OPTS[@]}" "$HOST" \
  "mkdir -p '$REMOTE_PARENT' '$REMOTE_LOGS' '$REMOTE_RESULTS' '$REMOTE_WORK'"
if ssh "${SSH_OPTS[@]}" "$HOST" "test -d '$REMOTE_BUNDLE'"; then
  REMOTE_CHECK=$(ssh "${SSH_OPTS[@]}" "$HOST" bash -s -- "$REMOTE_BUNDLE" <<'REMOTE'
set -euo pipefail
bundle=$1
cd "$bundle"
sha256sum -c --strict SHA256SUMS >/dev/null
printf '__UED_MANIFEST__%s\n' "$(sha256sum SHA256SUMS | awk '{print $1}')"
REMOTE
  )
  REMOTE_SHA=$(printf '%s\n' "$REMOTE_CHECK" \
    | awk -F'__UED_MANIFEST__' '/^__UED_MANIFEST__/ {print $2; exit}')
  [[ "$REMOTE_SHA" == "$MANIFEST_SHA256" ]] || {
    echo "remote bundle id collision: $REMOTE_BUNDLE" >&2
    exit 1
  }
  REUSED=true
else
  ssh "${SSH_OPTS[@]}" "$HOST" "mkdir '$REMOTE_STAGE'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3" \
    "$TMP/source/" "$HOST:$REMOTE_STAGE/"
  ssh "${SSH_OPTS[@]}" "$HOST" \
    "cd '$REMOTE_STAGE' && sha256sum -c --strict SHA256SUMS >/dev/null && mv -T '$REMOTE_STAGE' '$REMOTE_BUNDLE'"
  REUSED=false
fi
REMOTE_STAGE=""

printf 'UED_REMOTE_BUNDLE=%s\n' "$REMOTE_BUNDLE"
printf 'UED_REMOTE_REUSED=%s\n' "$REUSED"
emit_receipt
