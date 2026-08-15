#!/usr/bin/env bash
# Network-free, CPU end-to-end checks for the grouped one-update Hopper rung.
set -euo pipefail
umask 077

readonly HERE="$(cd "$(dirname "$0")" && pwd)"
readonly ROOT="$(cd "$HERE/.." && pwd)"
readonly PINNED_COMMIT=d053054c5290a04c1c4cd8b55704d999cad73e30
readonly CONTRACT=5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000
readonly CONFIG_SHA=b49168142a9d5a5d8edce88634975ac52d7615dca8bad9fff1cbcaf29ec43508
readonly XPID=eng1-ca-ovv3ch5868d346_N8ne8a1.0b1.0th0.0eastrict-4p-b8-rp1-mf0.5-seed1
readonly SOURCE_DIR="${MINIMAX_SOURCE_DIR:-/tmp/root-minimax-260814}"
readonly CPU_PYTHON="${UED_CPU_PYTHON:-/home/robotixx/miniconda3/pkgs/python-3.10.20-h741d88c_0/bin/python}"
readonly CPU_SITE_PACKAGES="${UED_CPU_SITE_PACKAGES:-/data/robotixx/ued_bench/envs/minimax-jax0431-cpu/lib/python3.10/site-packages}"
readonly CPU_LIBRARY_PATH="${UED_CPU_LIBRARY_PATH:-/home/robotixx/miniconda3/pkgs/openssl-3.5.5-h1b28b03_0/lib}"
[[ -d "$SOURCE_DIR/.git" || -f "$SOURCE_DIR/.git" ]] || {
  echo "set MINIMAX_SOURCE_DIR to a clone at $PINNED_COMMIT" >&2
  exit 1
}
[[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == "$PINNED_COMMIT" ]]
[[ -x "$CPU_PYTHON" && -d "$CPU_SITE_PACKAGES" && -d "$CPU_LIBRARY_PATH" ]]

readonly SBATCH="$HERE/sbatch/ued_minimax_one_update_smoke.sbatch"
readonly LOCAL_TEST="$HERE/test_ued_minimax_one_update_local.sh"
readonly DRIVER="$ROOT/ued_benchmark/scripts/run_grouped_one_update.py"
for file in "$SBATCH" "$LOCAL_TEST"; do
  bash -n "$file"
done
UED_TEST_DRIVER="$DRIVER" "$CPU_PYTHON" - <<'PY'
import os
from pathlib import Path

driver = Path(os.environ["UED_TEST_DRIVER"])
compile(driver.read_text(encoding="utf-8"), str(driver), "exec")
PY

grep -Fxq '#SBATCH --gres=gpu:1g.10gb:1' "$SBATCH"
grep -Fxq '#SBATCH --cpus-per-task=2' "$SBATCH"
grep -Fxq '#SBATCH --mem=15G' "$SBATCH"
grep -Fxq '#SBATCH --time=00:30:00' "$SBATCH"
grep -Fq 'frontier_exact_grouped_one_update' "$SBATCH"
grep -Fq 'max_student_updates' "$SBATCH"
grep -Fq 'UED_IMPORT_SMOKE_RESULT_DIR' "$SBATCH"
grep -Fq 'UED_IMPORT_SMOKE_MANIFEST_SHA256' "$SBATCH"
grep -Fq 'UED_ONE_UPDATE_SBATCH_SHA256' "$SBATCH"
grep -Fq 'readonly GIT="$UED_ENV_DIR/bin/git"' "$SBATCH"
grep -Fq 'git version 2.45.2' "$SBATCH"
grep -Fq 'export PATH="$UED_ENV_DIR/bin:/usr/bin:/bin"' "$SBATCH"
grep -Fq '[[ "$(command -v git)" == "$GIT" ]]' "$SBATCH"
! grep -Eq '^[[:space:]]*git[[:space:]]' "$SBATCH"
grep -Fq 'verify_manifest' "$SBATCH"
grep -Fq 'INPUT_CLOSURE.json' "$SBATCH"
grep -Fq 'resource-accounting.json' "$SBATCH"
grep -Fq 'python_resource_getrusage_self_and_monotonic_ns' "$SBATCH"
grep -Fq '"optimizer_step_applications": 5,' "$SBATCH"
grep -Fq '"complete_schema": 2,' "$SBATCH"
grep -Fq 'terminal Slurm sacct is authoritative external' "$SBATCH"
grep -Fq '"external_accounting_authority": "terminal_slurm_sacct"' "$SBATCH"
! grep -Fq '/usr/bin/time' "$SBATCH"
! grep -Fq '"gnu_time":' "$SBATCH"
! grep -Fq 'resource-usage.txt' "$SBATCH"
! grep -Fq '/usr/bin/time' "$DRIVER"
grep -Fq 'mv -T .SHA256SUMS.tmp SHA256SUMS' "$SBATCH"
grep -Fq 'os.replace(tmp, Path(os.environ["UED_COMPLETE"]))' "$SBATCH"
grep -Fq 'mv -T -- "$STAGE_DIR" "$FINAL_DIR"' "$SBATCH"
! grep -Eq 'python[^#\n]*-m[[:space:]]+minimax\.train|python[^#\n]*minimax/train\.py' "$SBATCH"
! grep -Eq '(^|[[:space:]])(sbatch|srun|ssh)([[:space:]]|$)' "$LOCAL_TEST"

grep -Fq "$CONTRACT" "$DRIVER"
grep -Fq "$CONFIG_SHA" "$DRIVER"
grep -Fq "$XPID" "$DRIVER"
grep -Fq 'args.train_runner_args.buffer_size = 8' "$DRIVER"
grep -Fq 'args.train_runner_args.replay_prob = 1.0' "$DRIVER"
grep -Fq 'post_resume_update_executed' "$DRIVER"
grep -Fq 'resource.getrusage(resource.RUSAGE_SELF)' "$DRIVER"
grep -Fq 'time.monotonic_ns()' "$DRIVER"
grep -Fq 'optimizer_step_applications' "$DRIVER"

readonly TMP="$(mktemp -d /tmp/ued-minimax-one-update-test.XXXXXX)"
cleanup() {
  local status=$?
  if (( status != 0 )) && [[ -n "${TMP:-}" && -d "$TMP" ]]; then
    for diagnostic in driver.stderr negative.stderr slurm-negative.stderr; do
      if [[ -s "$TMP/$diagnostic" ]]; then
        printf '%s\n' "--- $diagnostic ---" >&2
        sed -n '1,160p' "$TMP/$diagnostic" >&2
      fi
    done
  fi
  if [[ -n "${TMP:-}" && "$TMP" == /tmp/ued-minimax-one-update-test.* \
        && -d "$TMP" ]]; then
    rm -rf -- "$TMP"
  fi
  trap - EXIT
  exit "$status"
}
trap cleanup EXIT

# Staging the same clean source twice must yield the same content address, and
# all three new one-update inputs must be covered by the outer manifest.
for suffix in a b; do
  MINIMAX_SOURCE_DIR="$SOURCE_DIR" \
    bash "$HERE/stage_ued_minimax.sh" local "$TMP/bundle-$suffix" \
    > "$TMP/stage-$suffix.out"
  (cd "$TMP/bundle-$suffix" && sha256sum -c --strict SHA256SUMS >/dev/null)
done
readonly BUNDLE_SHA="$(sha256sum "$TMP/bundle-a/SHA256SUMS" | awk '{print $1}')"
[[ "$(sha256sum "$TMP/bundle-b/SHA256SUMS" | awk '{print $1}')" == "$BUNDLE_SHA" ]]
cmp "$TMP/bundle-a/SHA256SUMS" "$TMP/bundle-b/SHA256SUMS"
for rel in hopper/sbatch/ued_minimax_one_update_smoke.sbatch \
           hopper/test_ued_minimax_one_update_local.sh \
           ued_benchmark/scripts/run_grouped_one_update.py; do
  grep -Fq "  ./$rel" "$TMP/bundle-a/SHA256SUMS"
done

# The local Blackwell/JAX 0.6.2 lanes must not enter or perturb the pinned
# JAX 0.4.31 Hopper bundle. Canonical benchmark inputs remain content-bound.
for excluded in blackwell_probe blackwell_training_probe; do
  [[ ! -e "$TMP/bundle-a/ued_benchmark/$excluded" \
     && ! -L "$TMP/bundle-a/ued_benchmark/$excluded" ]]
  ! grep -Fq "./ued_benchmark/$excluded/" "$TMP/bundle-a/SHA256SUMS"
  ! grep -Fq "./$excluded/" \
    "$TMP/bundle-a/ued_benchmark/OVERLAY_SHA256SUMS"
done
canonical_inputs=(
  ued_benchmark/OVERLAY_CONTRACT.json
  ued_benchmark/configs/maze_frontier_exact_grouped_n8.json
  ued_benchmark/configs/maze_frontier_posterior_bridge_n8_neval1.json
  ued_benchmark/configs/maze_maxmc_group_matched_4x8_b500.json
  ued_benchmark/overlay/minimax/util/rl/frontier_activity.py
  ued_benchmark/scripts/apply_minimax_overlay.py
  ued_benchmark/scripts/run_grouped_one_update.py
  ued_benchmark/tests/test_frontier_activity.py
  ued_benchmark/tests/test_grouped_rng_contract.py
  ued_benchmark/tests/test_grouped_runner_smoke.py
)
for rel in "${canonical_inputs[@]}"; do
  [[ -f "$TMP/bundle-a/$rel" && ! -L "$TMP/bundle-a/$rel" ]]
  grep -Fq "  ./$rel" "$TMP/bundle-a/SHA256SUMS"
done

UED_TEST_STATE="$TMP/bundle-a/BUNDLE_STATE.json" "$CPU_PYTHON" - <<'PY'
import json
import os

with open(os.environ["UED_TEST_STATE"], encoding="utf-8") as stream:
    state = json.load(stream)
assert state["bundle_schema"] == 4
assert state["purpose"] == "bounded UED minimax/AMaze engineering smokes only"
assert state["paper_evidence"] is False
assert state["workspace_tree_exclusions"] == [
    "ued_benchmark/blackwell_probe/**",
    "ued_benchmark/blackwell_training_probe/**",
]
assert state["allowed_engineering_endpoints"] == [
    "frontier_exact_grouped_one_update", "frontier_terminal_chain_components",
    "gpu_import_formula_jit"]
assert state["max_student_updates"] == 1
assert state["terminal_chain_contract"] == {
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
}
assert state["resource_accounting_contract"] == {
    "in_process": "python_resource_getrusage_self_and_monotonic_ns",
    "external_authority": "terminal_slurm_sacct",
    "host_gnu_time_required": False,
}
PY

readonly SOURCE_BUNDLE="$TMP/bundle-a/upstream/minimax-${PINNED_COMMIT}.bundle"
git clone --quiet "$SOURCE_BUNDLE" "$TMP/patched"
readonly APPLY="$TMP/bundle-a/ued_benchmark/scripts/apply_minimax_overlay.py"
PYTHONDONTWRITEBYTECODE=1 "$CPU_PYTHON" "$APPLY" --target "$TMP/patched" --check \
  > "$TMP/check.json"
PYTHONDONTWRITEBYTECODE=1 "$CPU_PYTHON" "$APPLY" --target "$TMP/patched" --apply \
  > "$TMP/apply.json"
PYTHONDONTWRITEBYTECODE=1 "$CPU_PYTHON" "$APPLY" --target "$TMP/patched" --check \
  > "$TMP/postcheck.json"
grep -Fq '"status": "applicable"' "$TMP/check.json"
grep -Fq '"status": "applied"' "$TMP/apply.json"
grep -Fq '"status": "already_applied"' "$TMP/postcheck.json"
git -C "$TMP/patched" diff --check

mkdir "$TMP/output"
mkdir "$TMP/no-host-time-path"
readonly TEST_GIT="$(command -v git)"
[[ "$TEST_GIT" == /* && -x "$TEST_GIT" ]]
readonly TREE="$(git -C "$TMP/patched" rev-parse 'HEAD^{tree}')"
readonly SOURCE_SHA="$(sha256sum "$SOURCE_BUNDLE" | awk '{print $1}')"
readonly OVERLAY_SHA="$(sha256sum "$TMP/bundle-a/ued_benchmark/OVERLAY_SHA256SUMS" | awk '{print $1}')"
readonly APPLIED_SHA="$(sha256sum "$TMP/patched/.frontierrl_overlay.json" | awk '{print $1}')"
readonly SBATCH_SHA="$(sha256sum "$TMP/bundle-a/hopper/sbatch/ued_minimax_one_update_smoke.sbatch" | awk '{print $1}')"
readonly IMPORT_SBATCH_SHA="$(sha256sum "$TMP/bundle-a/hopper/sbatch/ued_minimax_gpu_smoke.sbatch" | awk '{print $1}')"

UED_PROVENANCE="$TMP/provenance.json" UED_BUNDLE="$BUNDLE_SHA" \
UED_TREE="$TREE" UED_SOURCE="$SOURCE_SHA" UED_OVERLAY="$OVERLAY_SHA" \
UED_APPLIED="$APPLIED_SHA" UED_SBATCH="$SBATCH_SHA" \
UED_IMPORT_SBATCH="$IMPORT_SBATCH_SHA" \
"$CPU_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

fake = "a" * 64
record = {
    "provenance_schema": 1,
    "purpose": "bounded Frontier grouped one-update engineering validation",
    "paper_evidence": False,
    "endpoint_class": "bounded_engineering_one_update",
    "max_student_updates": 1,
    "git": "git version 2.45.2",
    "job_id": "local-test",
    "xpid": "eng1-ca-ovv3ch5868d346_N8ne8a1.0b1.0th0.0eastrict-4p-b8-rp1-mf0.5-seed1",
    "resources": {
        "partition": "gpuq", "qos": "gpu", "gres": "gpu:1g.10gb:1",
        "cpus_per_task": 2, "memory": "15G", "walltime": "00:30:00",
    },
    "hashes": {
        "bundle_manifest_sha256": os.environ["UED_BUNDLE"],
        "upstream_commit": "d053054c5290a04c1c4cd8b55704d999cad73e30",
        "upstream_tree_git_sha1": os.environ["UED_TREE"],
        "upstream_git_bundle_sha256": os.environ["UED_SOURCE"],
        "overlay_manifest_sha256": os.environ["UED_OVERLAY"],
        "applied_overlay_manifest_sha256": os.environ["UED_APPLIED"],
        "sbatch_sha256": os.environ["UED_SBATCH"],
        "config_sha256": "b49168142a9d5a5d8edce88634975ac52d7615dca8bad9fff1cbcaf29ec43508",
        "overlay_contract_sha256": "5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000",
        "environment_lock_sha256": fake,
        "environment_freeze_sha256": fake,
        "environment_manifest_sha256": fake,
        "environment_setup_script_sha256": fake,
        "conda_explicit_sha256": fake,
        "environment_json_sha256": fake,
        "import_smoke_manifest_sha256": fake,
        "import_smoke_bundle_manifest_sha256": os.environ["UED_BUNDLE"],
        "import_smoke_sbatch_sha256": os.environ["UED_IMPORT_SBATCH"],
    },
}
Path(os.environ["UED_PROVENANCE"]).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

# Real CPU runner execution: exact grouped 4x8/N=8 geometry, four inserts,
# 32 trials on warmup, 64 after forced replay, and exactly one PPO update.
# An empty PATH plus an explicit Git executable proves the driver has no
# GNU-time or other PATH-discovered host-command dependency. Production binds
# that one required executable to the exact Git inside UED_ENV_DIR.
PATH="$TMP/no-host-time-path" GIT_PYTHON_GIT_EXECUTABLE="$TEST_GIT" \
JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu \
LD_LIBRARY_PATH="$CPU_LIBRARY_PATH" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$TMP/patched/src:$TMP/bundle-a:$CPU_SITE_PACKAGES" \
"$CPU_PYTHON" "$TMP/bundle-a/ued_benchmark/scripts/run_grouped_one_update.py" \
  --config "$TMP/bundle-a/ued_benchmark/configs/maze_frontier_exact_grouped_n8.json" \
  --contract "$TMP/bundle-a/ued_benchmark/OVERLAY_CONTRACT.json" \
  --provenance "$TMP/provenance.json" \
  --patched-source-dir "$TMP/patched" \
  --output-dir "$TMP/output" \
  --local-test-mode \
  > "$TMP/driver.stdout" 2> "$TMP/driver.stderr"
grep -Fq 'GROUPED_ONE_UPDATE_PASS updates=1 trials=64' "$TMP/driver.stdout"

UED_TEST_RESULT="$TMP/output/run-result.json" \
UED_TEST_CHECKPOINT="$TMP/output/checkpoint.pkl" "$CPU_PYTHON" - <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path
import re

result = json.loads(Path(os.environ["UED_TEST_RESULT"]).read_text(encoding="utf-8"))
checkpoint_sha = hashlib.sha256(
    Path(os.environ["UED_TEST_CHECKPOINT"]).read_bytes()).hexdigest()
assert result["status"] == "passed"
assert result["paper_evidence"] is False and "training_endpoint" not in result
assert result["endpoint_class"] == "bounded_engineering_one_update"
assert result["max_student_updates"] == result["actual_student_updates"] == 1
assert result["runtime"]["local_test_mode"] is True
schedule = result["engineering_schedule"]
assert schedule["n_parallel"] == 4
assert schedule["n_eval"] == schedule["frontier_n_rollouts"] == 8
assert schedule["outer_cycles"] == 2 and schedule["actual_ppo_updates"] == 1
assert schedule["ppo_epochs"] == 1 and schedule["ppo_minibatches"] == 1
assert schedule["expected_optimizer_step_applications"] == 1
assert schedule["optimizer_step_applications"] == 1
assert schedule["rollout_steps"] == 2 and schedule["total_transitions"] == 128
assert [cycle["state"]["n_updates"] for cycle in result["cycles"]] == [0, 1]
assert [cycle["state"]["n_grad_updates"] for cycle in result["cycles"]] == [0, 1]
assert [cycle["state"]["optimizer_step_applications"]
        for cycle in result["cycles"]] == [0, 1]
assert [cycle["state"]["frontier_total_trials"] for cycle in result["cycles"]] == [32, 64]
final = result["final_state"]
assert final["n_iters"] == 2 and final["n_updates"] == 1
assert final["n_grad_updates"] == 1
assert final["optimizer_step_applications"] == 1
assert final["buffer_filled_count"] == 4
assert final["frontier_incomplete_group_count"] == 0
assert final["frontier_duplicate_new_group_count"] == 0
assert result["checkpoint"]["sha256"] == checkpoint_sha
assert result["checkpoint"]["counter_and_buffer_continuity"] is True
assert result["checkpoint"]["train_state_exact_leaf_continuity"] is True
assert result["checkpoint"]["post_resume_update_executed"] is False
accounting = result["resource_accounting"]
assert accounting["resource_schema"] == 2
assert accounting["scope"] == "in-process engineering diagnostics"
assert accounting["accounting_source"] == (
    "python_resource_getrusage_self_and_monotonic_ns")
utc_re = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")
assert utc_re.fullmatch(accounting["run_start_utc"])
assert utc_re.fullmatch(accounting["run_end_utc"])
assert math.isfinite(accounting["monotonic_elapsed_seconds"])
assert accounting["monotonic_elapsed_seconds"] > 0.0
assert accounting["process_user_seconds"] >= 0.0
assert accounting["process_system_seconds"] >= 0.0
assert accounting["process_max_rss_kib"] > 0
assert accounting["transitions"] == 128
assert accounting["transitions_per_wall_second"] > 0.0
assert accounting["external_accounting_authority"] == "terminal_slurm_sacct"
assert accounting["terminal_sacct_included"] is False
PY

# Fail closed before runner creation when the prerequisite came from another
# bundle, and reject local-test shrink flags in a Slurm context.
mkdir "$TMP/negative-output" "$TMP/slurm-negative-output"
UED_BAD_PROVENANCE="$TMP/bad-provenance.json" UED_GOOD_PROVENANCE="$TMP/provenance.json" \
"$CPU_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

record = json.loads(Path(os.environ["UED_GOOD_PROVENANCE"]).read_text(encoding="utf-8"))
record["hashes"]["import_smoke_bundle_manifest_sha256"] = "f" * 64
Path(os.environ["UED_BAD_PROVENANCE"]).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
if JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu LD_LIBRARY_PATH="$CPU_LIBRARY_PATH" \
  PYTHONPATH="$TMP/patched/src:$TMP/bundle-a:$CPU_SITE_PACKAGES" \
  "$CPU_PYTHON" "$TMP/bundle-a/ued_benchmark/scripts/run_grouped_one_update.py" \
    --config "$TMP/bundle-a/ued_benchmark/configs/maze_frontier_exact_grouped_n8.json" \
    --contract "$TMP/bundle-a/ued_benchmark/OVERLAY_CONTRACT.json" \
    --provenance "$TMP/bad-provenance.json" --patched-source-dir "$TMP/patched" \
    --output-dir "$TMP/negative-output" --local-test-mode \
    > "$TMP/negative.stdout" 2> "$TMP/negative.stderr"; then
  echo "cross-bundle import prerequisite unexpectedly passed" >&2
  exit 1
fi
grep -Fq 'import/JIT gate did not validate this exact bundle' "$TMP/negative.stderr"
[[ -z "$(find "$TMP/negative-output" -mindepth 1 -print -quit)" ]]

if SLURM_JOB_ID=999 JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu \
  LD_LIBRARY_PATH="$CPU_LIBRARY_PATH" \
  PYTHONPATH="$TMP/patched/src:$TMP/bundle-a:$CPU_SITE_PACKAGES" \
  "$CPU_PYTHON" "$TMP/bundle-a/ued_benchmark/scripts/run_grouped_one_update.py" \
    --config "$TMP/bundle-a/ued_benchmark/configs/maze_frontier_exact_grouped_n8.json" \
    --contract "$TMP/bundle-a/ued_benchmark/OVERLAY_CONTRACT.json" \
    --provenance "$TMP/provenance.json" --patched-source-dir "$TMP/patched" \
    --output-dir "$TMP/slurm-negative-output" --local-test-mode \
    > "$TMP/slurm-negative.stdout" 2> "$TMP/slurm-negative.stderr"; then
  echo "Slurm local-test mode unexpectedly passed" >&2
  exit 1
fi
grep -Fq 'local-test mode forbidden under Slurm' "$TMP/slurm-negative.stderr"
[[ -z "$(find "$TMP/slurm-negative-output" -mindepth 1 -print -quit)" ]]

if command -v shellcheck >/dev/null 2>&1; then
  shellcheck -x "$SBATCH" "$LOCAL_TEST"
fi
printf 'UED_MINIMAX_ONE_UPDATE_LOCAL_CHECK_PASS\n'
