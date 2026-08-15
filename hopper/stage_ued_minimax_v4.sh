#!/usr/bin/env bash
# Build a source-faithful, content-addressed minimax + Frontier overlay bundle.
#
# Usage:
#   bash hopper/stage_ued_minimax_v4.sh local NEW_OUTPUT_DIR
# This candidate is deliberately local-only. A future audited commit must add
# remote staging; this file cannot SSH or mutate Hopper.
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
  *)
    echo "local-only candidate; usage: $0 local NEW_OUTPUT_DIR" >&2
    exit 2
    ;;
esac

readonly HERE="$(cd "$(dirname "$0")" && pwd)"
readonly ROOT="$(cd "$HERE/.." && pwd)"
readonly OVERLAY_ROOT="$ROOT/ued_benchmark"
readonly LOCK="$HERE/requirements-ued-minimax-hopper.lock"
readonly REQUIRED_OVERLAY_SCRIPT=ued_benchmark/scripts/apply_minimax_overlay_v4.py
readonly REQUIRED_OVERLAY_MODULE=ued_benchmark/overlay/minimax/util/rl/frontier_activity.py
readonly REQUIRED_TIE_MODULE=ued_benchmark/overlay/minimax/util/rl/tie_aware_rank.py

for path in "$LOCK" "$ROOT/$REQUIRED_OVERLAY_SCRIPT" "$ROOT/$REQUIRED_OVERLAY_MODULE" "$ROOT/$REQUIRED_TIE_MODULE"; do
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

readonly TMP="$(mktemp -d /tmp/ued-minimax-v4-stage.XXXXXX)"
cleanup() {
  if [[ -n "${TMP:-}" && "$TMP" == /tmp/ued-minimax-v4-stage.* && -d "$TMP" ]]; then
    rm -rf -- "$TMP"
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
readonly UPSTREAM_PLR_CONFIG_SHA256="$(git -C "$SOURCE_DIR" show "$PINNED_COMMIT:src/minimax/config/configs/maze/plr.json" | sha256sum | awk '{print $1}')"
readonly UPSTREAM_ACCEL_CONFIG_SHA256="$(git -C "$SOURCE_DIR" show "$PINNED_COMMIT:src/minimax/config/configs/maze/accel.json" | sha256sum | awk '{print $1}')"
[[ "$UPSTREAM_PLR_CONFIG_SHA256" == a5b8b87799bce31564959b3e8b55cfdaba658b31b8135fa35e3b96704d65185b ]]
[[ "$UPSTREAM_ACCEL_CONFIG_SHA256" == d7f100195159baff4f7490b755903e7586c895918ade4a248f5ae42b1827aee5 ]]
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
                "${REQUIRED_OVERLAY_MODULE#ued_benchmark/}" \
                "${REQUIRED_TIE_MODULE#ued_benchmark/}"; do
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
  hopper/hopper_v4.sh
  hopper/requirements-ued-minimax-hopper.lock
  hopper/setup_ued_minimax_env.sh
  hopper/stage_ued_minimax_v4.sh
  hopper/finalize_ued_minimax_v4_terminal_chain.py
  hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch
  hopper/sbatch/ued_minimax_v4_one_update_smoke.sbatch
  hopper/sbatch/ued_minimax_v4_terminal_chain_smoke.sbatch
  hopper/test_ued_minimax_v4_local.sh
  hopper/test_ued_minimax_v4_one_update_local.sh
  hopper/test_ued_minimax_v4_terminal_chain_local.sh
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

readonly ENV_SETUP_SHA256="$(sha256sum "$HERE/setup_ued_minimax_env.sh" | awk '{print $1}')"

verify_frozen() {
  local path=$1 expected=$2 actual
  [[ -f "$ROOT/$path" && ! -L "$ROOT/$path" ]] || {
    echo "missing protected input: $path" >&2
    exit 1
  }
  actual=$(sha256sum "$ROOT/$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "protected input drift: $path ($actual)" >&2
    exit 1
  }
}
verify_frozen ued_benchmark/OVERLAY_CONTRACT_V4.json 3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b
verify_frozen ued_benchmark/OVERLAY_LINEAGE.json 784e2fd1f545d49c8d10c3f3aeda37aae51fa00127e2c14578702e275bfb6971
verify_frozen ued_benchmark/scripts/apply_minimax_overlay_v4.py c2e5eb3dac02b86723ece485cd348832f1636198c781bae82c1d99df0167590b
verify_frozen ued_benchmark/overlay/minimax/util/rl/frontier_activity.py 63726251813bd9fafc2722409c4a2942c6ae2728327870797df47d01504738ca
verify_frozen ued_benchmark/overlay/minimax/util/rl/tie_aware_rank.py 1b9db20d05edd3212346e84d14606af91ae443c0665945a7b679ade161560244
verify_frozen ued_benchmark/analysis/development_protocol_v2_tie_aware_draft.json 1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269
verify_frozen ued_benchmark/configs/maze_frontier_exact_grouped_n8_tie_aware_v4.json 0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2
verify_frozen ued_benchmark/configs/maze_maxmc_group_matched_4x8_b500_tie_aware_v4.json a3cc3ddf387a3bb7cf3d9759c3f13e6a74f2c7e32de311ac344d54ef5e703ec6
verify_frozen ued_benchmark/configs/maze_maxmc_upstream_official_reference_32x1_b4000.json a5b8b87799bce31564959b3e8b55cfdaba658b31b8135fa35e3b96704d65185b
verify_frozen ued_benchmark/analysis/frontier_calibration_telemetry_v1_draft.json 8f786b4b66fe1f255b3bf00c05ad0d7378f614a88a26fe4d0440bc6076202511
verify_frozen ued_benchmark/analysis/frontier_calibration_telemetry.py 59abc1fb54c0c32dd709349462a26597a29d6b1981771a5f3226679a74c7818f
verify_frozen hopper/stage_ued_minimax.sh 73ad318fe21f6f99c92fe09ac6ec76c5dd6fe4b0d7fec8a5ca5939c59483ba55
verify_frozen hopper/sbatch/ued_minimax_terminal_chain_smoke.sbatch 5ed5186e010decdcc6bf97ff7dc820e0f4cf13e580e9b20c996a8dc561b13a14
verify_frozen hopper/finalize_ued_minimax_terminal_chain.py 57eb4394cedf30cc1a5bfeca4734199652cbfcfd5fbcaaca08035e8001a2c5ec
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
UED_UPSTREAM_PLR_CONFIG_SHA256="$UPSTREAM_PLR_CONFIG_SHA256" \
UED_UPSTREAM_ACCEL_CONFIG_SHA256="$UPSTREAM_ACCEL_CONFIG_SHA256" \
UED_OVERLAY_MANIFEST_SHA256="$OVERLAY_MANIFEST_SHA256" \
UED_ENV_LOCK_SHA256="$ENV_LOCK_SHA256" \
UED_ENV_SETUP_SHA256="$ENV_SETUP_SHA256" \
UED_SOURCE_ROOT="$TMP/source" \
python3 -I -B - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["UED_SOURCE_ROOT"])
def digest(relative):
    return hashlib.sha256((root / relative).read_bytes()).hexdigest()

record = {
    "bundle_schema": 5,
    "purpose": "bounded tie-aware v4 UED engineering rungs 0-3 only",
    "paper_evidence": False,
    "analyzer_eligible": False,
    "endpoint_access_authorized": False,
    "production_authorized": False,
    "remote_stage_authorized": False,
    "remote_launch_hold": {
        "authorized": False,
        "hopper_behavior_validated": False,
        "slurm_assignment_export_may_invoke_get_user_env": True,
        "submission_command_implemented": False,
        "resolution_required": (
            "independent audit plus a separately authorized Hopper validation "
            "of sanitized export semantics"
        ),
    },
    "environment_integrity_scope": {
        "claim": "locked package-state parity only",
        "installed_file_byte_closure": False,
        "paper_lane_requirement": (
            "immutable container digest or independently verified installed-file closure"
        ),
    },
    "workspace_tree_exclusions": [
        "ued_benchmark/blackwell_probe/**",
        "ued_benchmark/blackwell_training_probe/**",
    ],
    "allowed_engineering_endpoints": [
        "v4_gpu_import_tie_formula_jit",
        "v4_frontier_grouped_one_update",
        "v4_frontier_terminal_chain_components",
        "v4_maxmc_terminal_chain_components",
    ],
    "max_student_updates": 1,
    "paper_lane_outstanding": {
        "cost100_implemented": False,
        "confirmatory_seed_count_authorized": 0,
        "five_seed_two_sided_sign_flip_is_descriptive_only": True,
        "source_faithful_plr_accel_and_official_maxmc_baselines_required": True,
    },
    "terminal_chain_contract": {
        "analyzer_eligible": False,
        "actual_external_evaluation": True,
        "paper_evidence": False,
        "phase_a": "slurm_closed_training_and_actual_external_evaluation_components",
        "phase_a_submission_export": "explicit_ued_assignments_no_all_or_none",
        "phase_a_python_flags": "-I -B",
        "phase_b": "post_terminal_local_atomic_engineering_assembly",
        "phase_b_python": "isolated_clean_python_3.10.20_venv",
        "phase_b_host_scope": "post_fetch_operator_workstation",
        "phase_b_required_python_version": "3.10.20",
        "phase_b_executable_gate": "resolved CLI executable equals running sys.executable",
        "phase_b_environment_manifest_bound": False,
        "phase_b_non_test_general_release_ready": False,
        "phase_b_python_flags": "-I -B",
        "finalizer_self_bound": True,
        "post_terminal_fetch_receipt_schema": 2,
        "production_analyzer_invoked": False,
        "common_replay_snapshot": "plr-replay-snapshot.json for both arms",
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
    "environment_setup_script_sha256": os.environ["UED_ENV_SETUP_SHA256"],
    "core_bindings": {
        "contract_sha256": "3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b",
        "lineage_sha256": "784e2fd1f545d49c8d10c3f3aeda37aae51fa00127e2c14578702e275bfb6971",
        "applicator_sha256": "c2e5eb3dac02b86723ece485cd348832f1636198c781bae82c1d99df0167590b",
        "frontier_module_sha256": "63726251813bd9fafc2722409c4a2942c6ae2728327870797df47d01504738ca",
        "tie_rank_module_sha256": "1b9db20d05edd3212346e84d14606af91ae443c0665945a7b679ade161560244",
        "expected_applied_manifest_sha256": "9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae",
        "protocol_sha256": "1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269",
        "frontier_config_sha256": "0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2",
        "maxmc_config_sha256": "a3cc3ddf387a3bb7cf3d9759c3f13e6a74f2c7e32de311ac344d54ef5e703ec6",
        "official_upstream_reference_config_sha256": "a5b8b87799bce31564959b3e8b55cfdaba658b31b8135fa35e3b96704d65185b",
    },
    "official_pristine_source_configs": {
        "source_binding": "objects inside the pinned pristine upstream Git bundle",
        "overlay_execution_forbidden": True,
        "plr": {
            "path": "src/minimax/config/configs/maze/plr.json",
            "sha256": os.environ["UED_UPSTREAM_PLR_CONFIG_SHA256"],
        },
        "accel": {
            "path": "src/minimax/config/configs/maze/accel.json",
            "sha256": os.environ["UED_UPSTREAM_ACCEL_CONFIG_SHA256"],
        },
    },
    "engineering_file_hashes": {
        path: digest(path) for path in [
            "hopper/hopper_v4.sh",
            "hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch",
            "hopper/sbatch/ued_minimax_v4_one_update_smoke.sbatch",
            "hopper/sbatch/ued_minimax_v4_terminal_chain_smoke.sbatch",
            "hopper/finalize_ued_minimax_v4_terminal_chain.py",
            "ued_benchmark/hopper_v4/run_matched_terminal_v4.py",
            "ued_benchmark/hopper_v4/evaluate_matched_terminal_v4.py",
            "ued_benchmark/hopper_v4/assemble_matched_run_v4.py",
            "ued_benchmark/hopper_v4/run_terminal_phase_a_v4.py",
        ]
    },
    "calibration_telemetry_advisory": {
        "activated": False,
        "required_for_rungs_0_3": False,
        "protocol_sha256": "8f786b4b66fe1f255b3bf00c05ad0d7378f614a88a26fe4d0440bc6076202511",
        "analyzer_sha256": "59abc1fb54c0c32dd709349462a26597a29d6b1981771a5f3226679a74c7818f",
        "separate_overlay_writer_required_before_activation": True,
    },
    "protected_v3_remote_job": {
        "job_id": "9367063",
        "bundle_id": "06ffeeeb6998e8ddb1ce",
        "bundle_manifest_sha256": "06ffeeeb6998e8ddb1ce516c8982ef8e78627f7cc876ea0b712dab466aa1e8ff",
        "reusable_as_v4_prerequisite": False,
        "mutation_forbidden": True,
    },
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
