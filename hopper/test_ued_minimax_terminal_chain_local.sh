#!/usr/bin/env bash
# Network-free staged E2E for the two-phase terminal-chain engineering smoke.
set -euo pipefail
umask 077

readonly HERE="$(cd "$(dirname "$0")" && pwd)"
readonly ROOT="$(cd "$HERE/.." && pwd)"
readonly PINNED_COMMIT=d053054c5290a04c1c4cd8b55704d999cad73e30
readonly SOURCE_DIR="${MINIMAX_SOURCE_DIR:-/tmp/root-minimax-260814}"
readonly CPU_PYTHON="${UED_CPU_PYTHON:-/data/robotixx/ued_bench/envs/minimax-jax0431-cpu/bin/python}"
readonly PHASE_B_BASE_PYTHON="${UED_PHASE_B_BASE_PYTHON:-/home/robotixx/miniconda3/envs/agenticrl/bin/python3.10}"
readonly SBATCH="$HERE/sbatch/ued_minimax_terminal_chain_smoke.sbatch"
readonly FINALIZER="$HERE/finalize_ued_minimax_terminal_chain.py"
readonly LOCAL_TEST="$HERE/test_ued_minimax_terminal_chain_local.sh"

[[ -d "$SOURCE_DIR/.git" || -f "$SOURCE_DIR/.git" ]] || {
  echo "set MINIMAX_SOURCE_DIR to a clone at $PINNED_COMMIT" >&2
  exit 1
}
[[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == "$PINNED_COMMIT" ]]
[[ -x "$CPU_PYTHON" && -x "$PHASE_B_BASE_PYTHON" ]]
bash -n "$SBATCH"
bash -n "$LOCAL_TEST"
"$CPU_PYTHON" -B -c \
  'from pathlib import Path; import sys; p=Path(sys.argv[1]); compile(p.read_bytes(), str(p), "exec")' \
  "$FINALIZER"
grep -Fxq '#SBATCH --gres=gpu:1g.10gb:1' "$SBATCH"
grep -Fxq '#!/bin/bash' "$SBATCH"
grep -Fxq '#SBATCH --time=00:30:00' "$SBATCH"
grep -Fxq '#SBATCH --no-requeue' "$SBATCH"
grep -Fq 'readonly GIT="$UED_ENV_DIR/bin/git"' "$SBATCH"
grep -Fq 'unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONOPTIMIZE' "$SBATCH"
grep -Fq 'unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES LD_LIBRARY_PATH LD_PRELOAD' "$SBATCH"
grep -Fq '"$PY" -I -B' "$SBATCH"
grep -Fq 'os.environ["UED_PY"], "-I", "-B"' "$SBATCH"
grep -Fq 'assert sys.flags.optimize == 0' "$SBATCH"
grep -Fq 'assert sys.dont_write_bytecode is True' "$SBATCH"
grep -Fq '[[ "${SLURM_RESTART_COUNT:-}" == 0 ]]' "$SBATCH"
grep -Fq 'ENVIRONMENT_COMPLETE 0' "$SBATCH"
grep -Fq 'if os.environ["UED_VERIFY_EXACT"] == "1":' "$SBATCH"
[[ "$(grep -Fc -- '--slurm-engineering-test-mode' "$SBATCH")" -ge 2 ]]
grep -Fq -- '--campaign-manifest' "$SBATCH"
! grep -Fq -- '--synthetic-test-mode' "$SBATCH"
! grep -Fq '/usr/bin/time' "$SBATCH"
! grep -Eq '(^|[[:space:]])(sbatch|srun|ssh)([[:space:]]|$)' "$LOCAL_TEST"
grep -Fq 'COMPONENTS_COMPLETE.json' "$SBATCH"
grep -Fq 'phase_b_required' "$SBATCH"
grep -Fq 'terminal_sacct_included' "$SBATCH"
grep -Fq 'INPUT_CLOSURE.json' "$SBATCH"
grep -Fq -- '--validate-only' "$FINALIZER"
grep -Fq 'production_analyzer_invoked' "$FINALIZER"
grep -Fq -- '--expected-python-sha256' "$FINALIZER"

readonly TMP="$(mktemp -d /tmp/ued-terminal-chain-test.XXXXXX)"
cleanup() {
  local status=$?
  if (( status != 0 )); then
    for diagnostic in train.stderr eval.stderr finalize.stderr negative.stderr; do
      if [[ -s "$TMP/$diagnostic" ]]; then
        printf '%s\n' "--- $diagnostic ---" >&2
        sed -n '1,200p' "$TMP/$diagnostic" >&2
      fi
    done
  fi
  if [[ "$TMP" == /tmp/ued-terminal-chain-test.* && -d "$TMP" ]]; then
    rm -rf -- "$TMP"
  fi
  trap - EXIT
  exit "$status"
}
trap cleanup EXIT

# Regression for the real Conda-prefix shape: live environments contain many
# unlisted files and symlinks.  The sbatch must exact-check only its recorded
# metadata payloads there; exact-tree mode remains mandatory for copied gates.
mkdir -p "$TMP/live-env/bin"
printf 'metadata\n' > "$TMP/live-env/ENVIRONMENT.json"
printf '%s  ENVIRONMENT.json\n' \
  "$(sha256sum "$TMP/live-env/ENVIRONMENT.json" | awk '{print $1}')" \
  > "$TMP/live-env/ENVIRONMENT_SHA256SUMS"
printf 'complete\n' > "$TMP/live-env/ENVIRONMENT_COMPLETE"
ln -s /bin/sh "$TMP/live-env/bin/python"
UED_SBATCH_SOURCE="$SBATCH" UED_LIVE_ENV="$TMP/live-env" "$CPU_PYTHON" - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["UED_SBATCH_SOURCE"]).read_text(encoding="utf-8")
branch = source.split('if os.environ["UED_VERIFY_EXACT"] == "1":', 1)[1].split("\nPY", 1)[0]
exact, nonexact = branch.split("\nelse:", 1)
assert 'root.rglob("*")' in exact
assert 'root.rglob("*")' not in nonexact
root = Path(os.environ["UED_LIVE_ENV"])
assert (root / "bin/python").is_symlink()
for name in ("ENVIRONMENT_SHA256SUMS", "ENVIRONMENT_COMPLETE", "ENVIRONMENT.json"):
    assert (root / name).is_file() and not (root / name).is_symlink()
PY

# Phase B is stdlib-only. Build a fresh, isolated, network-free Python 3.10.20
# venv so pip-check and the exact realized freeze are clean and independently
# bound instead of inheriting packages from the CPU/JAX fixture or agenticrl.
mkdir "$TMP/phase-b-home"
env -i HOME="$TMP/phase-b-home" PATH=/usr/bin:/bin LC_ALL=C PYTHONNOUSERSITE=1 \
  "$PHASE_B_BASE_PYTHON" -I -B -m venv "$TMP/phase-b-venv"
readonly PHASE_B_PYTHON="$TMP/phase-b-venv/bin/python"
[[ -x "$PHASE_B_PYTHON" ]]
readonly -a PHASE_B_ENV=(
  env -i HOME="$TMP/phase-b-home" PATH="$TMP/phase-b-venv/bin:/usr/bin:/bin"
  LC_ALL=C PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
  PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
)

# The same source tree must stage byte-identically twice, include the complete
# two-phase implementation, and continue excluding both Blackwell lanes.
for suffix in a b; do
  MINIMAX_SOURCE_DIR="$SOURCE_DIR" bash "$HERE/stage_ued_minimax.sh" local \
    "$TMP/bundle-$suffix" > "$TMP/stage-$suffix.out"
  (cd "$TMP/bundle-$suffix" && sha256sum -c --strict SHA256SUMS >/dev/null)
done
readonly BUNDLE="$TMP/bundle-a"
readonly BUNDLE_SHA="$(sha256sum "$BUNDLE/SHA256SUMS" | awk '{print $1}')"
[[ "$(sha256sum "$TMP/bundle-b/SHA256SUMS" | awk '{print $1}')" == "$BUNDLE_SHA" ]]
cmp "$BUNDLE/SHA256SUMS" "$TMP/bundle-b/SHA256SUMS"
for rel in \
  hopper/hopper.sh \
  hopper/finalize_ued_minimax_terminal_chain.py \
  hopper/sbatch/ued_minimax_terminal_chain_smoke.sbatch \
  hopper/test_ued_minimax_terminal_chain_local.sh \
  ued_benchmark/scripts/run_matched_terminal.py \
  ued_benchmark/scripts/evaluate_matched_terminal.py \
  ued_benchmark/scripts/assemble_matched_run.py \
  ued_benchmark/analysis/preregistered_dev_analysis.py \
  ued_benchmark/analysis/development_protocol_v1.json; do
  [[ -f "$BUNDLE/$rel" && ! -L "$BUNDLE/$rel" ]]
  grep -Fq "  ./$rel" "$BUNDLE/SHA256SUMS"
done
for excluded in blackwell_probe blackwell_training_probe; do
  [[ ! -e "$BUNDLE/ued_benchmark/$excluded" && ! -L "$BUNDLE/ued_benchmark/$excluded" ]]
  ! grep -Fq "./ued_benchmark/$excluded/" "$BUNDLE/SHA256SUMS"
done
UED_STATE="$BUNDLE/BUNDLE_STATE.json" "$CPU_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

state = json.loads(Path(os.environ["UED_STATE"]).read_text(encoding="utf-8"))
assert state["bundle_schema"] == 4 and state["paper_evidence"] is False
assert state["allowed_engineering_endpoints"] == [
    "frontier_exact_grouped_one_update",
    "frontier_terminal_chain_components",
    "gpu_import_formula_jit",
]
assert state["max_student_updates"] == 1
assert state["terminal_chain_contract"]["actual_external_evaluation"] is True
assert state["terminal_chain_contract"]["analyzer_eligible"] is False
assert state["terminal_chain_contract"]["production_analyzer_invoked"] is False
assert state["terminal_chain_contract"]["phase_a_submission_export"] == (
    "explicit_ued_allowlist_no_all")
assert state["terminal_chain_contract"]["phase_a_python_flags"] == "-I -B"
assert state["terminal_chain_contract"]["phase_b_python"] == (
    "isolated_clean_python_3.10.20_venv")
assert state["terminal_chain_contract"]["phase_b_python_flags"] == "-I -B"
assert state["terminal_chain_contract"]["finalizer_self_bound"] is True
assert state["terminal_chain_contract"]["post_terminal_fetch_receipt_schema"] == 2
assert state["terminal_chain_contract"]["submission_receipt_required"] is True
assert state["terminal_chain_contract"]["terminal_receipt_schema"] == 2
assert state["terminal_chain_contract"]["terminal_sacct_phase"] == (
    "post_completion_local_finalize")
PY

readonly SOURCE_BUNDLE="$BUNDLE/upstream/minimax-${PINNED_COMMIT}.bundle"
git clone --quiet "$SOURCE_BUNDLE" "$TMP/patched"
readonly APPLY="$BUNDLE/ued_benchmark/scripts/apply_minimax_overlay.py"
PYTHONDONTWRITEBYTECODE=1 "$CPU_PYTHON" "$APPLY" --target "$TMP/patched" --check \
  > "$TMP/overlay-check.json"
PYTHONDONTWRITEBYTECODE=1 "$CPU_PYTHON" "$APPLY" --target "$TMP/patched" --apply \
  > "$TMP/overlay-apply.json"
PYTHONDONTWRITEBYTECODE=1 "$CPU_PYTHON" "$APPLY" --target "$TMP/patched" --check \
  > "$TMP/overlay-postcheck.json"
grep -Fq '"status": "already_applied"' "$TMP/overlay-postcheck.json"

readonly TRAIN_DRIVER="$BUNDLE/ued_benchmark/scripts/run_matched_terminal.py"
readonly EVAL_DRIVER="$BUNDLE/ued_benchmark/scripts/evaluate_matched_terminal.py"
readonly ASSEMBLER="$BUNDLE/ued_benchmark/scripts/assemble_matched_run.py"
readonly ANALYZER="$BUNDLE/ued_benchmark/analysis/preregistered_dev_analysis.py"
readonly PROTOCOL="$BUNDLE/ued_benchmark/analysis/development_protocol_v1.json"
readonly CONFIG="$BUNDLE/ued_benchmark/configs/maze_frontier_exact_grouped_n8.json"
readonly TRAIN_SHA="$(sha256sum "$TRAIN_DRIVER" | awk '{print $1}')"
readonly EVAL_SHA="$(sha256sum "$EVAL_DRIVER" | awk '{print $1}')"
readonly ASSEMBLER_SHA="$(sha256sum "$ASSEMBLER" | awk '{print $1}')"
readonly ANALYZER_SHA="$(sha256sum "$ANALYZER" | awk '{print $1}')"
readonly PROTOCOL_SHA="$(sha256sum "$PROTOCOL" | awk '{print $1}')"
readonly OVERLAY_SHA="$(sha256sum "$BUNDLE/ued_benchmark/OVERLAY_SHA256SUMS" | awk '{print $1}')"
readonly APPLIED_SHA="$(sha256sum "$TMP/patched/.frontierrl_overlay.json" | awk '{print $1}')"
readonly SBATCH_SHA="$(sha256sum "$BUNDLE/hopper/sbatch/ued_minimax_terminal_chain_smoke.sbatch" | awk '{print $1}')"
readonly FINALIZER_SHA="$(sha256sum "$BUNDLE/hopper/finalize_ued_minimax_terminal_chain.py" | awk '{print $1}')"
readonly HOPPER_SHA="$(sha256sum "$BUNDLE/hopper/hopper.sh" | awk '{print $1}')"

mkdir "$TMP/run"
readonly LOCAL_RUN_ID=engineering-frontier-s101
UED_CONTEXT="$TMP/run/local-context.json" UED_RUN_ID="$LOCAL_RUN_ID" \
UED_BUNDLE="$BUNDLE_SHA" UED_OVERLAY="$OVERLAY_SHA" UED_APPLIED="$APPLIED_SHA" \
UED_TRAIN="$TRAIN_SHA" UED_EVAL="$EVAL_SHA" UED_SBATCH="$SBATCH_SHA" \
"$CPU_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

context = {
    "schema": 1,
    "protocol_id": "ued-dev-frontier-vs-maxmc-4x8-b500-v1",
    "purpose": "engineering_development_only_not_paper_evidence",
    "run_id": os.environ["UED_RUN_ID"], "arm": "frontier", "training_seed": 101,
    "job_id": "local-test", "campaign_manifest_sha256": "c" * 64,
    "provenance": {
        "base_commit": "d053054c5290a04c1c4cd8b55704d999cad73e30",
        "base_tree": "b0cace1fc54984e21a842f12d15d0b899e33d270",
        "overlay_contract_sha256": "5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000",
        "bundle_manifest_sha256": os.environ["UED_BUNDLE"],
        "overlay_manifest_sha256": os.environ["UED_OVERLAY"],
        "applied_overlay_manifest_sha256": os.environ["UED_APPLIED"],
        "environment_manifest_sha256": "e" * 64,
        "training_driver_sha256": os.environ["UED_TRAIN"],
        "evaluation_driver_sha256": os.environ["UED_EVAL"],
        "sbatch_sha256": os.environ["UED_SBATCH"],
    },
}
Path(os.environ["UED_CONTEXT"]).write_text(
    json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
readonly LOCAL_CONTEXT_SHA="$(sha256sum "$TMP/run/local-context.json" | awk '{print $1}')"
mkdir "$TMP/run/train-parent" "$TMP/run/sidecar-parent" "$TMP/run/eval-parent"
readonly TRAIN_OUTPUT="$TMP/run/train-parent/$LOCAL_RUN_ID"
readonly TRAIN_SIDECAR="$TMP/run/sidecar-parent/${LOCAL_RUN_ID}-sidecar"
readonly EVAL_OUTPUT="$TMP/run/eval-parent/${LOCAL_RUN_ID}-evaluation"

runtime_env=(
  JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1
  PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
)
train_command=(
  "$CPU_PYTHON" -I -B "$TRAIN_DRIVER" --arm frontier --config "$CONFIG"
  --protocol "$PROTOCOL" --run-context "$TMP/run/local-context.json"
  --expected-run-context-sha256 "$LOCAL_CONTEXT_SHA"
  --expected-driver-sha256 "$TRAIN_SHA" --patched-source-dir "$TMP/patched"
  --output-dir "$TRAIN_OUTPUT" --sidecar-dir "$TRAIN_SIDECAR"
  --engineering-test-mode
)
for override in \
  n_total_updates=1 test_interval=0 log_interval=1 \
  train_runner_args.buffer_size=8 train_runner_args.replay_prob=1.0 \
  train_runner_args.min_fill_ratio=0.5 train_runner_args.n_rollout_steps=2 \
  train_runner_args.n_unroll_rollout=1 env_args.max_episode_steps=2 \
  student_rl_args.n_unroll_update=1 student_rl_args.n_epochs=1 \
  student_model_args.hidden_dim=16 student_model_args.recurrent_hidden_dim=16 \
  student_model_args.n_conv_filters=4 driver.max_outer_cycles=4; do
  train_command+=(--engineering-override "$override")
done
env "${runtime_env[@]}" "${train_command[@]}" \
  > "$TMP/train.stdout" 2> "$TMP/train.stderr"
grep -Fq 'MATCHED_TERMINAL_COMPLETE' "$TMP/train.stdout"

readonly EVALUATION_LAUNCHER="import runpy,sys; p=sys.argv.pop(1); sys.path.insert(0, str(__import__('pathlib').Path(p).resolve().parent)); runpy.run_path(p, run_name='__main__')"
env "${runtime_env[@]}" "$CPU_PYTHON" -I -B -c "$EVALUATION_LAUNCHER" \
  "$EVAL_DRIVER" --arm frontier \
  --protocol "$PROTOCOL" --run-context "$TMP/run/local-context.json" \
  --expected-run-context-sha256 "$LOCAL_CONTEXT_SHA" \
  --expected-driver-sha256 "$EVAL_SHA" --patched-source-dir "$TMP/patched" \
  --checkpoint "$TRAIN_OUTPUT/checkpoint.pkl" --endpoint "$TRAIN_OUTPUT/endpoint.json" \
  --training-receipt "$TRAIN_SIDECAR/training-receipt.json" \
  --meta "$TRAIN_OUTPUT/meta.json" --output-dir "$EVAL_OUTPUT" \
  --engineering-test-mode > "$TMP/eval.stdout" 2> "$TMP/eval.stderr"
grep -Fq 'MATCHED_EVALUATION_COMPLETE' "$TMP/eval.stdout"

# Relabel the local runtime artifacts into one numeric-job structural fixture.
# The first finalization attempt intentionally retains the genuine local/CPU
# receipts and must be rejected.  Only afterward is a separate, structurally
# faithful mock-Slurm receipt built for the positive Phase-B assembly test.
mkdir "$TMP/components"
mv "$TRAIN_OUTPUT" "$TMP/components/training-output"
mv "$TRAIN_SIDECAR" "$TMP/components/training-sidecar"
mv "$EVAL_OUTPUT" "$TMP/components/evaluation-package"
cp -- "$TMP/overlay-check.json" "$TMP/components/overlay-check.json"
cp -- "$TMP/overlay-apply.json" "$TMP/components/overlay-apply.json"
cp -- "$TMP/overlay-postcheck.json" "$TMP/components/overlay-postcheck.json"
cp -- "$TMP/patched/.frontierrl_overlay.json" \
  "$TMP/components/applied-overlay-manifest.json"
cp -- "$TMP/train.stdout" "$TMP/components/training.stdout"
cp -- "$TMP/train.stderr" "$TMP/components/training.stderr"
cp -- "$TMP/eval.stdout" "$TMP/components/evaluation.stdout"
cp -- "$TMP/eval.stderr" "$TMP/components/evaluation.stderr"
cp -- "$BUNDLE/BUNDLE_STATE.json" "$TMP/components/bundle-state.json"
UED_COMPONENTS="$TMP/components" UED_JOB=8123456 \
UED_RUN_ID=engineering-slurm-8123456-frontier-s101 UED_PROTOCOL="$PROTOCOL_SHA" \
UED_ANALYZER="$ANALYZER_SHA" UED_ASSEMBLER="$ASSEMBLER_SHA" \
UED_BUNDLE="$BUNDLE_SHA" UED_OVERLAY="$OVERLAY_SHA" UED_APPLIED="$APPLIED_SHA" \
UED_TRAIN="$TRAIN_SHA" UED_EVAL="$EVAL_SHA" UED_SBATCH="$SBATCH_SHA" \
UED_FINALIZER="$FINALIZER_SHA" UED_HOPPER="$HOPPER_SHA" \
UED_CONFIG="$(sha256sum "$CONFIG" | awk '{print $1}')" \
UED_SOURCE="$(sha256sum "$SOURCE_BUNDLE" | awk '{print $1}')" \
UED_LOCK="$(sha256sum "$BUNDLE/hopper/requirements-ued-minimax-hopper.lock" | awk '{print $1}')" \
UED_SETUP="$(sha256sum "$BUNDLE/hopper/setup_ued_minimax_env.sh" | awk '{print $1}')" \
"$CPU_PYTHON" - <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["UED_COMPONENTS"])
run_id = os.environ["UED_RUN_ID"]
job_id = os.environ["UED_JOB"]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
write = lambda path, value: path.write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
canonical = lambda value: hashlib.sha256(json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False).encode("utf-8")).hexdigest()
provenance = {
    "base_commit": "d053054c5290a04c1c4cd8b55704d999cad73e30",
    "base_tree": "b0cace1fc54984e21a842f12d15d0b899e33d270",
    "overlay_contract_sha256": "5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000",
    "bundle_manifest_sha256": os.environ["UED_BUNDLE"],
    "overlay_manifest_sha256": os.environ["UED_OVERLAY"],
    "applied_overlay_manifest_sha256": os.environ["UED_APPLIED"],
    "environment_manifest_sha256": None,
    "training_driver_sha256": os.environ["UED_TRAIN"],
    "evaluation_driver_sha256": os.environ["UED_EVAL"],
    "sbatch_sha256": os.environ["UED_SBATCH"],
    "assembler_driver_sha256": os.environ["UED_ASSEMBLER"],
}

# Minimal immutable prerequisite closures for the local Phase-B structural
# test.  Their contents are fake engineering fixtures, but every byte is bound
# exactly by the component and input-closure manifests.
environment = root / "environment"
import_gate = root / "import-smoke"
one_gate = root / "one-update"
environment.mkdir(); import_gate.mkdir(); one_gate.mkdir()
(environment / "CONDA_EXPLICIT.txt").write_text("@EXPLICIT\nmock-conda\n", encoding="utf-8")
(environment / "ENVIRONMENT.freeze").write_text("mock==1\n", encoding="utf-8")
write(environment / "ENVIRONMENT.json", {"schema": 1, "fixture": True})
(environment / "LOCK_SHA256").write_text(os.environ["UED_LOCK"] + "\n", encoding="utf-8")
(environment / "SETUP_SHA256").write_text(os.environ["UED_SETUP"] + "\n", encoding="utf-8")
env_payloads = ["CONDA_EXPLICIT.txt", "ENVIRONMENT.freeze", "ENVIRONMENT.json",
                "LOCK_SHA256", "SETUP_SHA256"]
(environment / "ENVIRONMENT_SHA256SUMS").write_text("".join(
    f"{sha(environment / name)}  {name}\n" for name in sorted(env_payloads)),
    encoding="utf-8")
write(environment / "ENVIRONMENT_COMPLETE", {"schema": 1, "status": "complete"})
write(import_gate / "runtime.json", {"schema": 1, "fixture": True})
(import_gate / "SHA256SUMS").write_text(
    f"{sha(import_gate / 'runtime.json')}  runtime.json\n", encoding="utf-8")
write(import_gate / "COMPLETE", {"schema": 1, "status": "complete"})
write(one_gate / "run-result.json", {"schema": 1, "fixture": True})
(one_gate / "SHA256SUMS").write_text(
    f"{sha(one_gate / 'run-result.json')}  run-result.json\n", encoding="utf-8")
write(one_gate / "COMPLETE", {"schema": 1, "status": "complete"})
environment_manifest_sha = sha(environment / "ENVIRONMENT_SHA256SUMS")
provenance["environment_manifest_sha256"] = environment_manifest_sha
campaign = {
    "schema": 1, "protocol_id": "ued-dev-frontier-vs-maxmc-4x8-b500-v1",
    "purpose": "engineering_development_only_not_paper_evidence",
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "frozen_before_endpoint_access": True,
    "protocol_sha256": os.environ["UED_PROTOCOL"],
    "analyzer_sha256": os.environ["UED_ANALYZER"], "provenance": provenance,
    "hardware": {"partition": "gpuq", "gpu_model": "NVIDIA A100-SXM4-80GB",
                 "gpu_profile": "1g.10gb", "gpu_count": 1, "n_devices": 1},
    "submissions": [{"arm": "frontier", "training_seed": 101,
                     "evaluation_seed": 100101, "run_id": run_id,
                     "job_id": job_id, "attempt": 1}],
}
campaign_path = root / "campaign-manifest.json"
write(campaign_path, campaign)
campaign_sha = sha(campaign_path)
context = {
    "schema": 1, "protocol_id": campaign["protocol_id"], "purpose": campaign["purpose"],
    "run_id": run_id, "arm": "frontier", "training_seed": 101, "job_id": job_id,
    "campaign_manifest_sha256": campaign_sha,
    "provenance": {key: value for key, value in provenance.items()
                   if key != "assembler_driver_sha256"},
}
context_path = root / "run-context.json"
write(context_path, context)
context_sha = sha(context_path)

training_root = root / "training-output"
sidecar = root / "training-sidecar"
evaluation_root = root / "evaluation-package"
endpoint_path = training_root / "endpoint.json"
endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
endpoint["run_id"] = run_id
write(endpoint_path, endpoint)
meta_path = training_root / "meta.json"
meta = json.loads(meta_path.read_text(encoding="utf-8"))
meta["xpid"] = run_id
meta["slurm"]["job_id"] = job_id
meta["config"]["xpid"] = run_id
write(meta_path, meta)
snapshot_path = sidecar / "frontier-buffer-snapshot.json"
snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
snapshot["run_id"] = run_id
write(snapshot_path, snapshot)
training_path = sidecar / "training-receipt.json"
training = json.loads(training_path.read_text(encoding="utf-8"))
training["run_id"] = run_id
training["job_id"] = job_id
training["provenance"]["run_context"] = context
training["provenance"]["run_context_sha256"] = context_sha
training["config"]["resolved"] = meta["config"]
training["config"]["resolved_canonical_sha256"] = canonical(meta["config"])
training["config"]["meta_sha256"] = sha(meta_path)
training["config"]["logs_sha256"] = sha(training_root / "logs.csv")
training["endpoint"] = {"path": "endpoint.json", "sha256": sha(endpoint_path)}
training["frontier_snapshot"] = {
    "path": "frontier-buffer-snapshot.json", "sha256": sha(snapshot_path)}
write(training_path, training)
training_manifest = sidecar / "SHA256SUMS"
training_manifest.write_text(
    f"{sha(snapshot_path)}  frontier-buffer-snapshot.json\n"
    f"{sha(training_path)}  training-receipt.json\n", encoding="utf-8")
write(sidecar / "COMPLETE", {
    "schema": 1, "status": "complete", "run_id": run_id, "arm": "frontier",
    "sha256sums_sha256": sha(training_manifest), "file_count": 2})

eval_path = evaluation_root / "evaluation-receipt.json"
evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
assert evaluation["synthetic_test_mode"] is False
evaluation["run_id"] = run_id
evaluation["training_receipt_sha256"] = sha(training_path)
evaluation["meta_sha256"] = sha(meta_path)
evaluation["provenance"]["run_context"] = context
evaluation["provenance"]["run_context_sha256"] = context_sha
evaluation["raw_results"]["sha256"] = sha(evaluation_root / "evaluation-episodes.jsonl")
evaluation["aggregate_results"]["sha256"] = sha(evaluation_root / "evaluation.csv")
write(eval_path, evaluation)
eval_manifest = evaluation_root / "SHA256SUMS"
payloads = ["evaluation-episodes.jsonl", "evaluation-receipt.json", "evaluation.csv"]
eval_manifest.write_text("".join(
    f"{sha(evaluation_root / name)}  {name}\n" for name in sorted(payloads)),
    encoding="utf-8")
write(evaluation_root / "COMPLETE", {
    "schema": 1, "status": "complete", "run_id": run_id,
    "sha256sums_sha256": sha(eval_manifest), "file_count": 3})

(root / "command.txt").write_text(
    "local staged actual-evaluation structural finalizer fixture\n", encoding="utf-8")
write(root / "resource-accounting.json", {
    "resource_schema": 1,
    "scope": "phase-A bounded training plus actual external evaluation child processes",
    "job_id": job_id, "run_start_utc": "2026-08-14T19:58:01.000000Z",
    "run_end_utc": "2026-08-14T20:00:00.000000Z",
    "monotonic_elapsed_seconds": 119.0, "child_user_seconds": 100.0,
    "child_system_seconds": 5.0, "child_max_rss_kib": 2097152,
    "requested": {"partition": "gpuq", "qos": "gpu", "gres": "gpu:1g.10gb:1",
                  "cpus_per_task": 2, "memory": "15G", "walltime": "00:30:00"},
    "allocation": {"SLURM_JOB_ID": job_id, "SLURM_JOB_NODELIST": "gpu021",
                   "SLURM_CPUS_PER_TASK": "2", "SLURM_MEM_PER_NODE": "15360",
                   "SLURM_JOB_PARTITION": "gpuq", "SLURM_NTASKS": "1",
                   "SLURM_JOB_GPUS": "MIG-mock", "CUDA_VISIBLE_DEVICES": "MIG-mock",
                   "SLURM_RESTART_COUNT": "0"},
    "peak_gpu_memory_bytes_observed": 0,
    "peak_gpu_memory_observation": "terminal sacct TRESUsageInMax preferred; zero means unavailable",
    "external_accounting_authority": "terminal_slurm_sacct",
    "terminal_sacct_included": False})
(root / "nvidia-smi-before.csv").write_text(
    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, NVIDIA A100-SXM4-80GB, 81920 MiB, 595.71.05\n",
    encoding="utf-8")
(root / "nvidia-smi-after.csv").write_text(
    (root / "nvidia-smi-before.csv").read_text(encoding="utf-8"), encoding="utf-8")
closure = {
    "input_closure_schema": 1,
    "purpose": "bounded Frontier terminal-chain Slurm engineering smoke",
    "paper_evidence": False, "analyzer_eligible": False,
    "endpoint_class": "bounded_engineering_terminal_chain_components",
    "git": "git version 2.45.2",
    "two_phase": {"phase_a": "slurm_components_only",
                  "phase_b": "post_terminal_local_engineering_assembly",
                  "production_analyzer_invoked": False,
                  "terminal_sacct_required": True},
    "resources": {"partition": "gpuq", "qos": "gpu", "gres": "gpu:1g.10gb:1",
                  "nodes": 1, "ntasks": 1, "cpus_per_task": 2,
                  "memory": "15G", "walltime": "00:30:00", "requeue": False},
    "schedule": {"arm": "frontier", "training_seed": 101,
                 "evaluation_seed": 100101, "n_updates": 1, "n_grad_updates": 1,
                 "optimizer_step_applications": 1, "outer_cycles": 2,
                 "student_training_transitions": 128,
                 "actual_external_evaluation": True, "evaluation_environments": 3,
                 "episodes_per_environment": 10, "max_episode_horizon": 450,
                 "primary_evaluation_transitions": 13500,
                 "independent_verification_transitions": 0},
    "hashes": {
        "bundle_manifest_sha256": os.environ["UED_BUNDLE"],
        "upstream_commit": provenance["base_commit"],
        "upstream_tree_git_sha1": provenance["base_tree"],
        "upstream_git_bundle_sha256": os.environ["UED_SOURCE"],
        "overlay_manifest_sha256": os.environ["UED_OVERLAY"],
        "terminal_chain_sbatch_sha256": os.environ["UED_SBATCH"],
        "frontier_config_sha256": os.environ["UED_CONFIG"],
        "overlay_contract_sha256": provenance["overlay_contract_sha256"],
        "protocol_sha256": os.environ["UED_PROTOCOL"],
        "analyzer_sha256": os.environ["UED_ANALYZER"],
        "training_driver_sha256": os.environ["UED_TRAIN"],
        "evaluation_driver_sha256": os.environ["UED_EVAL"],
        "assembler_sha256": os.environ["UED_ASSEMBLER"],
        "terminal_finalizer_sha256": os.environ["UED_FINALIZER"],
        "hopper_wrapper_sha256": os.environ["UED_HOPPER"],
        "environment_lock_sha256": os.environ["UED_LOCK"],
        "environment_freeze_sha256": sha(environment / "ENVIRONMENT.freeze"),
        "environment_manifest_sha256": environment_manifest_sha,
        "environment_setup_script_sha256": os.environ["UED_SETUP"],
        "conda_explicit_sha256": sha(environment / "CONDA_EXPLICIT.txt"),
        "environment_json_sha256": sha(environment / "ENVIRONMENT.json"),
        "import_smoke_manifest_sha256": sha(import_gate / "SHA256SUMS"),
        "one_update_manifest_sha256": sha(one_gate / "SHA256SUMS"),
    },
}
write(root / "INPUT_CLOSURE.json", closure)
PY

build_component_closure() {
  rm -f -- "$TMP/components/SHA256SUMS" "$TMP/components/COMPONENTS_COMPLETE.json"
  (
  cd "$TMP/components"
  find . -type f ! -path ./SHA256SUMS ! -path ./COMPONENTS_COMPLETE.json \
    ! -path ./.SHA256SUMS.tmp -print0 | LC_ALL=C sort -z \
    | xargs -0 sha256sum > .SHA256SUMS.tmp
  mv -T .SHA256SUMS.tmp SHA256SUMS
  sha256sum -c --strict SHA256SUMS >/dev/null
  )
  COMPONENTS_SHA="$(sha256sum "$TMP/components/SHA256SUMS" | awk '{print $1}')"
  COMPONENT_COUNT="$(wc -l < "$TMP/components/SHA256SUMS")"
  INPUT_CLOSURE_SHA="$(sha256sum "$TMP/components/INPUT_CLOSURE.json" | awk '{print $1}')"
UED_COMPLETE="$TMP/components/COMPONENTS_COMPLETE.json" UED_MANIFEST="$COMPONENTS_SHA" \
UED_COUNT="$COMPONENT_COUNT" UED_BUNDLE="$BUNDLE_SHA" \
UED_INPUT="$INPUT_CLOSURE_SHA" "$CPU_PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["UED_COMPLETE"]).parent
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
context = root / "run-context.json"
campaign = root / "campaign-manifest.json"
record = {
    "schema": 1, "status": "complete", "paper_evidence": False,
    "analyzer_eligible": False,
    "endpoint_class": "bounded_engineering_terminal_chain_components",
    "job_id": "8123456", "run_id": "engineering-slurm-8123456-frontier-s101",
    "arm": "frontier", "sha256sums_sha256": os.environ["UED_MANIFEST"],
    "file_count": int(os.environ["UED_COUNT"]),
    "bundle_manifest_sha256": os.environ["UED_BUNDLE"],
    "campaign_manifest_sha256": sha(campaign), "run_context_sha256": sha(context),
    "training_sidecar_manifest_sha256": sha(root / "training-sidecar/SHA256SUMS"),
    "evaluation_package_manifest_sha256": sha(root / "evaluation-package/SHA256SUMS"),
    "actual_student_updates": 1, "actual_external_evaluation": True,
    "raw_evaluation_records": 30, "terminal_sacct_included": False,
    "phase_b_required": True, "input_closure_sha256": os.environ["UED_INPUT"],
    "result_dir": ("/scratch/mock/maxrl/tests/ued-minimax-terminal-chain/"
                   + os.environ["UED_INPUT"][:20] + "/job-8123456"),
}
Path(os.environ["UED_COMPLETE"]).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  printf 'phase-A prelude\nUED_TERMINAL_COMPONENTS_COMPLETE job=8123456 manifest=%s result=/scratch/mock/maxrl/tests/ued-minimax-terminal-chain/%s/job-8123456 analyzer_eligible=false\n' \
    "$COMPONENTS_SHA" "${INPUT_CLOSURE_SHA:0:20}" > "$TMP/slurm.stdout"
  printf 'terminal Slurm stderr\n' > "$TMP/slurm.stderr"
}
build_component_closure

readonly REMOTE_BUNDLE="/scratch/mock/maxrl/bundles/ued_minimax/${BUNDLE_SHA:0:20}"
readonly REMOTE_ENV=/scratch/mock/envs/ued-minimax-pinned
readonly REMOTE_IMPORT=/scratch/mock/maxrl/tests/ued-minimax-gpu-smoke/8000001
readonly REMOTE_ONE=/scratch/mock/maxrl/tests/ued-minimax-one-update/8000002
readonly REMOTE_SCRIPT="/scratch/mock/maxrl/sbatch/ued_minimax_terminal_chain_smoke-${SBATCH_SHA:0:16}-20260814T155700Z-123.sbatch"
readonly REMOTE_SUBMISSION_RECEIPT=/scratch/mock/maxrl/receipts/job-8123456-20260814T155700Z.tsv
readonly REMOTE_STDOUT=/scratch/mock/maxrl/tests/logs/ued-minimax-terminal-chain_8123456.out
readonly REMOTE_STDERR=/scratch/mock/maxrl/tests/logs/ued-minimax-terminal-chain_8123456.err
readonly ENV_FREEZE_SHA="$(sha256sum "$TMP/components/environment/ENVIRONMENT.freeze" | awk '{print $1}')"
readonly ENV_MANIFEST_SHA="$(sha256sum "$TMP/components/environment/ENVIRONMENT_SHA256SUMS" | awk '{print $1}')"
readonly IMPORT_MANIFEST_SHA="$(sha256sum "$TMP/components/import-smoke/SHA256SUMS" | awk '{print $1}')"
readonly ONE_MANIFEST_SHA="$(sha256sum "$TMP/components/one-update/SHA256SUMS" | awk '{print $1}')"
readonly SOURCE_SHA="$(sha256sum "$SOURCE_BUNDLE" | awk '{print $1}')"
readonly CONFIG_SHA="$(sha256sum "$CONFIG" | awk '{print $1}')"
readonly LOCK_SHA="$(sha256sum "$BUNDLE/hopper/requirements-ued-minimax-hopper.lock" | awk '{print $1}')"
readonly SBATCH_ARGS="--export=UED_BUNDLE_DIR=$REMOTE_BUNDLE,UED_BUNDLE_MANIFEST_SHA256=$BUNDLE_SHA,UED_UPSTREAM_COMMIT=$PINNED_COMMIT,UED_UPSTREAM_TREE=b0cace1fc54984e21a842f12d15d0b899e33d270,UED_UPSTREAM_BUNDLE_SHA256=$SOURCE_SHA,UED_OVERLAY_MANIFEST_SHA256=$OVERLAY_SHA,UED_TERMINAL_CHAIN_SBATCH_SHA256=$SBATCH_SHA,UED_FRONTIER_CONFIG_SHA256=$CONFIG_SHA,UED_CONTRACT_SHA256=5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000,UED_PROTOCOL_SHA256=$PROTOCOL_SHA,UED_ANALYZER_SHA256=$ANALYZER_SHA,UED_TRAINING_DRIVER_SHA256=$TRAIN_SHA,UED_EVALUATION_DRIVER_SHA256=$EVAL_SHA,UED_ASSEMBLER_SHA256=$ASSEMBLER_SHA,UED_ENV_DIR=$REMOTE_ENV,UED_ENV_LOCK_SHA256=$LOCK_SHA,UED_ENV_FREEZE_SHA256=$ENV_FREEZE_SHA,UED_ENV_MANIFEST_SHA256=$ENV_MANIFEST_SHA,UED_IMPORT_SMOKE_RESULT_DIR=$REMOTE_IMPORT,UED_IMPORT_SMOKE_MANIFEST_SHA256=$IMPORT_MANIFEST_SHA,UED_ONE_UPDATE_RESULT_DIR=$REMOTE_ONE,UED_ONE_UPDATE_MANIFEST_SHA256=$ONE_MANIFEST_SHA"

printf 'job_id\tutc\thost\tlocal_script\tlocal_sha256\tremote_script\tremote_sha256\toutput_path\tremote_receipt\tsbatch_args\n' \
  > "$TMP/submission-receipt.tsv"
printf '8123456\t2026-08-14T19:57:00Z\tmock@hopper.orc.gmu.edu\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$BUNDLE/hopper/sbatch/ued_minimax_terminal_chain_smoke.sbatch" "$SBATCH_SHA" \
  "$REMOTE_SCRIPT" "$SBATCH_SHA" "$REMOTE_STDOUT" "$REMOTE_SUBMISSION_RECEIPT" \
  "$SBATCH_ARGS" >> "$TMP/submission-receipt.tsv"

printf '%s\n' \
  $'terminal_receipt_schema\t2' \
  $'retrieved_utc\t2026-08-14T20:00:02Z' \
  $'retrieved_epoch\t1786737602' \
  $'terminal_end_epoch\t1786737601' \
  $'terminal_header\tJobIDRaw|JobName|Partition|State|ExitCode|ElapsedRaw|AllocCPUS|ReqMem|NodeList|Submit|Start|End|AllocTRES|QOS|TimelimitRaw|Restarts|WorkDir|StdOut|StdErr|SubmitLine' \
  "terminal_row"$'\t'"8123456|ued-minimax-terminal-chain|gpuq|COMPLETED|0:0|120|2|15G|gpu021|2026-08-14T15:58:00|2026-08-14T15:58:01|2026-08-14T16:00:01|billing=20,cpu=2,gres/gpu:1g.10gb=1,gres/gpu=1,mem=15G,node=1|gpu|30|0|/scratch/mock/maxrl|/scratch/%u/maxrl/tests/logs/%x_%j.out|/scratch/%u/maxrl/tests/logs/%x_%j.err|sbatch --parsable $SBATCH_ARGS $REMOTE_SCRIPT" \
  $'resource_header\tJobIDRaw|MaxRSS|TRESUsageInMax' \
  $'resource_row\t8123456||' \
  $'resource_row\t8123456.batch|2048M|cpu=00:01:59,gres/gpumem=3072M' \
  > "$TMP/terminal-sacct.tsv"

write_fetch_receipt() {
  local output=$1 remote=$2 type=$3 digest=$4 manifest=$5 local_path=$6
  local terminal_sha
  terminal_sha=$(sha256sum "$TMP/terminal-sacct.tsv" | awk '{print $1}')
  {
    printf 'fetch_receipt_schema\t2\n'
    printf 'fetch_started_utc\t2026-08-14T20:00:03Z\n'
    printf 'fetch_started_epoch\t1786737603\n'
    printf 'retrieved_utc\t2026-08-14T20:00:04Z\n'
    printf 'retrieved_epoch\t1786737604\n'
    printf 'terminal_end_epoch\t1786737601\n'
    printf 'terminal_receipt_sha256\t%s\n' "$terminal_sha"
    printf 'remote_path\t%s\n' "$remote"
    printf 'remote_type\t%s\n' "$type"
    printf 'remote_digest\t%s\n' "$digest"
    printf 'manifest_verified\t%s\n' "$manifest"
    printf 'local_path\t%s\n' "$local_path"
    printf 'local_digest\t%s\n' "$digest"
  } > "$output"
}

build_fetch_receipts() {
  local component_tree stdout_sha stderr_sha submission_sha remote_components
  component_tree=$(cd "$TMP/components" && LC_ALL=C find . \( -type f -o -type l \) \
    -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}')
  stdout_sha=$(sha256sum "$TMP/slurm.stdout" | awk '{print $1}')
  stderr_sha=$(sha256sum "$TMP/slurm.stderr" | awk '{print $1}')
  submission_sha=$(sha256sum "$TMP/submission-receipt.tsv" | awk '{print $1}')
  remote_components="/scratch/mock/maxrl/tests/ued-minimax-terminal-chain/${INPUT_CLOSURE_SHA:0:20}/job-8123456"
  write_fetch_receipt "$TMP/fetch-components.tsv" "$remote_components" dir \
    "$component_tree" 1 "$TMP/components"
  write_fetch_receipt "$TMP/fetch-stdout.tsv" "$REMOTE_STDOUT" file \
    "$stdout_sha" 0 "$TMP/slurm.stdout"
  write_fetch_receipt "$TMP/fetch-stderr.tsv" "$REMOTE_STDERR" file \
    "$stderr_sha" 0 "$TMP/slurm.stderr"
  write_fetch_receipt "$TMP/fetch-submission.tsv" "$REMOTE_SUBMISSION_RECEIPT" file \
    "$submission_sha" 0 "$TMP/submission-receipt.tsv"
}
build_fetch_receipts

readonly PYTHON_REAL="$(readlink -f -- "$PHASE_B_PYTHON")"
readonly PYTHON_SHA="$(sha256sum "$PYTHON_REAL" | awk '{print $1}')"
readonly PYTHON_VERSION="$("${PHASE_B_ENV[@]}" "$PHASE_B_PYTHON" -I -B \
  -c 'import platform; print(platform.python_version())')"
[[ "$PYTHON_VERSION" == 3.10.20 ]]
readonly PYTHON_FREEZE="$TMP/phase-b-python-freeze.txt"
"${PHASE_B_ENV[@]}" "$PHASE_B_PYTHON" -I -B -m pip check >/dev/null
"${PHASE_B_ENV[@]}" "$PHASE_B_PYTHON" -I -B -m pip freeze --all \
  | LC_ALL=C sort > "$PYTHON_FREEZE"
readonly PYTHON_FREEZE_SHA="$(sha256sum "$PYTHON_FREEZE" | awk '{print $1}')"
readonly PYTHON_VENV_CONFIG_SHA="$(sha256sum "$TMP/phase-b-venv/pyvenv.cfg" | awk '{print $1}')"

finalize_command=(
  "${PHASE_B_ENV[@]}" "$PHASE_B_PYTHON" -I -B
  "$BUNDLE/hopper/finalize_ued_minimax_terminal_chain.py"
  --job-id 8123456 --bundle-dir "$BUNDLE"
  --expected-bundle-manifest-sha256 "$BUNDLE_SHA"
  --components-dir "$TMP/components"
  --expected-components-manifest-sha256 "$COMPONENTS_SHA"
  --expected-input-closure-sha256 "$INPUT_CLOSURE_SHA"
  --expected-sbatch-sha256 "$SBATCH_SHA"
  --terminal-receipt "$TMP/terminal-sacct.tsv"
  --submission-receipt "$TMP/submission-receipt.tsv"
  --submission-fetch-receipt "$TMP/fetch-submission.tsv"
  --components-fetch-receipt "$TMP/fetch-components.tsv"
  --slurm-stdout "$TMP/slurm.stdout" --slurm-stderr "$TMP/slurm.stderr"
  --stdout-fetch-receipt "$TMP/fetch-stdout.tsv"
  --stderr-fetch-receipt "$TMP/fetch-stderr.tsv"
  --expected-assembler-sha256 "$ASSEMBLER_SHA"
  --expected-finalizer-sha256 "$FINALIZER_SHA" --python "$PHASE_B_PYTHON"
  --expected-python-sha256 "$PYTHON_SHA"
  --expected-python-version "$PYTHON_VERSION"
  --expected-python-freeze-sha256 "$PYTHON_FREEZE_SHA"
  --expected-python-venv-config-sha256 "$PYTHON_VENV_CONFIG_SHA"
  --output-dir "$TMP/finalized"
)

# P0 regression: a relabeled local CPU run paired with a numeric terminal Slurm
# receipt must never be accepted as a Phase-A GPU allocation.
if "${finalize_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "relabeled local component package was accepted" >&2
  exit 1
fi
grep -Fq 'did not execute in Slurm engineering mode' "$TMP/negative.stderr"
[[ ! -e "$TMP/finalized" && ! -L "$TMP/finalized" ]]

# Convert only the test fixture receipts into a structurally exact mock of the
# receipts that the real --slurm-engineering-test-mode drivers produce.
UED_COMPONENTS="$TMP/components" "$CPU_PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["UED_COMPONENTS"])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
write = lambda path, value: path.write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
device = {"id": 0, "platform": "gpu",
          "device_kind": "NVIDIA A100-SXM4-80GB MIG 1g.10gb"}
source = None
training_path = root / "training-sidecar/training-receipt.json"
training = json.loads(training_path.read_text(encoding="utf-8"))
training["engineering_test"]["enabled"] = True
training["engineering_test"]["execution_mode"] = "slurm"
training["provenance"]["backend"] = "gpu"
training["provenance"]["devices"] = [device]
source = dict(training["provenance"]["source"])
source["git_executable"] = "/scratch/mock/envs/ued-minimax-pinned/bin/git"
source["git_executable_sha256"] = "f" * 64
source["git_version"] = "git version 2.45.2"
training["provenance"]["source"] = source
write(training_path, training)
snapshot_path = root / "training-sidecar/frontier-buffer-snapshot.json"
training_manifest = root / "training-sidecar/SHA256SUMS"
training_manifest.write_text(
    f"{sha(snapshot_path)}  frontier-buffer-snapshot.json\n"
    f"{sha(training_path)}  training-receipt.json\n", encoding="utf-8")
training_complete = root / "training-sidecar/COMPLETE"
write(training_complete, {"schema": 1, "status": "complete",
      "run_id": training["run_id"], "arm": "frontier",
      "sha256sums_sha256": sha(training_manifest), "file_count": 2})

evaluation_path = root / "evaluation-package/evaluation-receipt.json"
evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
evaluation["training_receipt_sha256"] = sha(training_path)
evaluation["provenance"]["source"] = source
runtime = evaluation["provenance"]["runtime"]
runtime["backend"] = "gpu"
runtime["device_count"] = 1
runtime["devices"] = [device]
write(evaluation_path, evaluation)
evaluation_root = evaluation_path.parent
evaluation_manifest = evaluation_root / "SHA256SUMS"
names = ["evaluation-episodes.jsonl", "evaluation-receipt.json", "evaluation.csv"]
evaluation_manifest.write_text("".join(
    f"{sha(evaluation_root / name)}  {name}\n" for name in sorted(names)),
    encoding="utf-8")
write(evaluation_root / "COMPLETE", {"schema": 1, "status": "complete",
      "run_id": evaluation["run_id"], "sha256sums_sha256": sha(evaluation_manifest),
      "file_count": 3})
PY
build_component_closure
build_fetch_receipts
finalize_command=(
  "${PHASE_B_ENV[@]}" "$PHASE_B_PYTHON" -I -B
  "$BUNDLE/hopper/finalize_ued_minimax_terminal_chain.py"
  --job-id 8123456 --bundle-dir "$BUNDLE"
  --expected-bundle-manifest-sha256 "$BUNDLE_SHA"
  --components-dir "$TMP/components"
  --expected-components-manifest-sha256 "$COMPONENTS_SHA"
  --expected-input-closure-sha256 "$INPUT_CLOSURE_SHA"
  --expected-sbatch-sha256 "$SBATCH_SHA"
  --terminal-receipt "$TMP/terminal-sacct.tsv"
  --submission-receipt "$TMP/submission-receipt.tsv"
  --submission-fetch-receipt "$TMP/fetch-submission.tsv"
  --components-fetch-receipt "$TMP/fetch-components.tsv"
  --slurm-stdout "$TMP/slurm.stdout" --slurm-stderr "$TMP/slurm.stderr"
  --stdout-fetch-receipt "$TMP/fetch-stdout.tsv"
  --stderr-fetch-receipt "$TMP/fetch-stderr.tsv"
  --expected-assembler-sha256 "$ASSEMBLER_SHA"
  --expected-finalizer-sha256 "$FINALIZER_SHA" --python "$PHASE_B_PYTHON"
  --expected-python-sha256 "$PYTHON_SHA" --expected-python-version "$PYTHON_VERSION"
  --expected-python-freeze-sha256 "$PYTHON_FREEZE_SHA"
  --expected-python-venv-config-sha256 "$PYTHON_VENV_CONFIG_SHA"
  --output-dir "$TMP/finalized"
)

# Fail closed before assembly on nonterminal/incorrect-resource accounting, a
# wrong completion marker, an unmanifested component, and a wrong interpreter
# digest. No refusal may leave a partial final directory.
sed 's/|COMPLETED|0:0|/|RUNNING|0:0|/' "$TMP/terminal-sacct.tsv" \
  > "$TMP/nonterminal-sacct.tsv"
negative_command=("${finalize_command[@]}")
for index in "${!negative_command[@]}"; do
  [[ "${negative_command[index]}" == "$TMP/terminal-sacct.tsv" ]] \
    && negative_command[index]="$TMP/nonterminal-sacct.tsv"
  [[ "${negative_command[index]}" == "$TMP/components" ]] \
    && negative_command[index]="$TMP/forbidden-components-must-not-be-opened"
done
if "${negative_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "nonterminal receipt was accepted" >&2
  exit 1
fi
grep -Fq 'job did not complete cleanly' "$TMP/negative.stderr"
! grep -Fq 'missing components directory' "$TMP/negative.stderr"
[[ ! -e "$TMP/finalized" && ! -L "$TMP/finalized" ]]

# Scheduler local timestamps are authoritative America/New_York times. End
# must bind the recorded epoch, the three times must be ordered, and their
# exact Start-to-End delta must equal ElapsedRaw.
sed 's/2026-08-14T16:00:01|billing/2026-08-14T16:00:02|billing/' \
  "$TMP/terminal-sacct.tsv" > "$TMP/wrong-end-epoch-sacct.tsv"
negative_command=("${finalize_command[@]}")
for index in "${!negative_command[@]}"; do
  [[ "${negative_command[index]}" == "$TMP/terminal-sacct.tsv" ]] \
    && negative_command[index]="$TMP/wrong-end-epoch-sacct.tsv"
done
if "${negative_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "unbound scheduler End epoch was accepted" >&2
  exit 1
fi
grep -Fq 'terminal End timestamp/epoch binding drift' "$TMP/negative.stderr"

sed 's/2026-08-14T15:58:01|2026-08-14T16:00:01/2026-08-14T16:00:02|2026-08-14T16:00:01/' \
  "$TMP/terminal-sacct.tsv" > "$TMP/unordered-time-sacct.tsv"
negative_command=("${finalize_command[@]}")
for index in "${!negative_command[@]}"; do
  [[ "${negative_command[index]}" == "$TMP/terminal-sacct.tsv" ]] \
    && negative_command[index]="$TMP/unordered-time-sacct.tsv"
done
if "${negative_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "unordered scheduler timestamps were accepted" >&2
  exit 1
fi
grep -Fq 'authoritative Slurm timestamps are unordered' "$TMP/negative.stderr"

sed 's/|COMPLETED|0:0|120|2|/|COMPLETED|0:0|119|2|/' \
  "$TMP/terminal-sacct.tsv" > "$TMP/wrong-elapsed-sacct.tsv"
negative_command=("${finalize_command[@]}")
for index in "${!negative_command[@]}"; do
  [[ "${negative_command[index]}" == "$TMP/terminal-sacct.tsv" ]] \
    && negative_command[index]="$TMP/wrong-elapsed-sacct.tsv"
done
if "${negative_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "scheduler elapsed/time delta drift was accepted" >&2
  exit 1
fi
grep -Fq 'terminal ElapsedRaw/Start/End binding drift' "$TMP/negative.stderr"

sed 's/gres\/gpu:1g.10gb=1/gres\/gpu:1g.10gb=2/' "$TMP/terminal-sacct.tsv" \
  > "$TMP/wrong-resource-sacct.tsv"
negative_command=("${finalize_command[@]}")
for index in "${!negative_command[@]}"; do
  [[ "${negative_command[index]}" == "$TMP/terminal-sacct.tsv" ]] \
    && negative_command[index]="$TMP/wrong-resource-sacct.tsv"
done
if "${negative_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "incorrect MIG allocation was accepted" >&2
  exit 1
fi
[[ ! -e "$TMP/finalized" && ! -L "$TMP/finalized" ]]
cp -- "$TMP/slurm.stdout" "$TMP/wrong-marker.stdout"
sed -i 's/job=8123456/job=8123457/' "$TMP/wrong-marker.stdout"
wrong_marker_sha=$(sha256sum "$TMP/wrong-marker.stdout" | awk '{print $1}')
write_fetch_receipt "$TMP/fetch-wrong-marker.tsv" "$REMOTE_STDOUT" file \
  "$wrong_marker_sha" 0 "$TMP/wrong-marker.stdout"
negative_command=("${finalize_command[@]}")
for index in "${!negative_command[@]}"; do
  [[ "${negative_command[index]}" == "$TMP/slurm.stdout" ]] \
    && negative_command[index]="$TMP/wrong-marker.stdout"
  [[ "${negative_command[index]}" == "$TMP/fetch-stdout.tsv" ]] \
    && negative_command[index]="$TMP/fetch-wrong-marker.tsv"
done
if "${negative_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "wrong-job completion marker was accepted" >&2
  exit 1
fi
grep -Fq 'completion marker job drift' "$TMP/negative.stderr"
[[ ! -e "$TMP/finalized" && ! -L "$TMP/finalized" ]]
printf 'unmanifested\n' > "$TMP/components/extra.txt"
build_fetch_receipts
if "${finalize_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "unmanifested component was accepted" >&2
  exit 1
fi
grep -Fq 'closure contains missing or unmanifested files' "$TMP/negative.stderr"
rm -f -- "$TMP/components/extra.txt"
build_fetch_receipts
[[ ! -e "$TMP/finalized" && ! -L "$TMP/finalized" ]]

# A receipt whose transfer began after Slurm End but before the terminal
# receipt itself existed must be rejected.
sed -e 's/2026-08-14T20:00:03Z/2026-08-14T20:00:01Z/' \
    -e 's/fetch_started_epoch\t1786737603/fetch_started_epoch\t1786737601/' \
    "$TMP/fetch-components.tsv" > "$TMP/fetch-components-too-early.tsv"
negative_command=("${finalize_command[@]}")
for index in "${!negative_command[@]}"; do
  [[ "${negative_command[index]}" == "$TMP/fetch-components.tsv" ]] \
    && negative_command[index]="$TMP/fetch-components-too-early.tsv"
done
if "${negative_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "pre-terminal-receipt fetch was accepted" >&2
  exit 1
fi
grep -Fq 'fetch did not begin after authoritative job end' "$TMP/negative.stderr"

# Exact terminal policy fields and the scheduler-executed SubmitLine are not
# advisory.  Each mutation must fail before any package is published.
sed 's/|gpu|30|0|/|gpu|29|0|/' "$TMP/terminal-sacct.tsv" \
  > "$TMP/wrong-time-sacct.tsv"
negative_command=("${finalize_command[@]}")
for index in "${!negative_command[@]}"; do
  [[ "${negative_command[index]}" == "$TMP/terminal-sacct.tsv" ]] \
    && negative_command[index]="$TMP/wrong-time-sacct.tsv"
done
if "${negative_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "wrong Slurm time limit was accepted" >&2
  exit 1
fi
sed 's/|gpu|30|0|/|gpu|30|1|/' "$TMP/terminal-sacct.tsv" \
  > "$TMP/restarted-sacct.tsv"
negative_command=("${finalize_command[@]}")
for index in "${!negative_command[@]}"; do
  [[ "${negative_command[index]}" == "$TMP/terminal-sacct.tsv" ]] \
    && negative_command[index]="$TMP/restarted-sacct.tsv"
done
if "${negative_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "restarted Slurm allocation was accepted" >&2
  exit 1
fi
sed 's/sbatch --parsable --export=/sbatch --parsable --comment=drift --export=/' \
  "$TMP/terminal-sacct.tsv" > "$TMP/wrong-submitline-sacct.tsv"
negative_command=("${finalize_command[@]}")
for index in "${!negative_command[@]}"; do
  [[ "${negative_command[index]}" == "$TMP/terminal-sacct.tsv" ]] \
    && negative_command[index]="$TMP/wrong-submitline-sacct.tsv"
done
if "${negative_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "wrong scheduler SubmitLine was accepted" >&2
  exit 1
fi

# COMPONENTS_COMPLETE is intentionally outside its payload manifest, so its
# duplicate-key parser and copy-time binding get direct adversarial coverage.
cp -- "$TMP/components/COMPONENTS_COMPLETE.json" "$TMP/components-complete.good"
printf '{"schema":1,"schema":1}\n' > "$TMP/components/COMPONENTS_COMPLETE.json"
build_fetch_receipts
if "${finalize_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "duplicate-key component COMPLETE was accepted" >&2
  exit 1
fi
grep -Fq 'duplicate JSON key in component COMPLETE' "$TMP/negative.stderr"
cp -- "$TMP/components-complete.good" "$TMP/components/COMPONENTS_COMPLETE.json"
build_fetch_receipts

# Extra files are rejected even if a malicious producer adds them to the outer
# component manifest and updates COMPLETE/marker consistently.
readonly VALID_COMPONENTS_SHA="$COMPONENTS_SHA"
printf 'manifested-extra\n' > "$TMP/components/extra.txt"
build_component_closure
build_fetch_receipts
negative_command=("${finalize_command[@]}")
for index in "${!negative_command[@]}"; do
  [[ "${negative_command[index]}" == "$VALID_COMPONENTS_SHA" ]] \
    && negative_command[index]="$COMPONENTS_SHA"
done
if "${negative_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "extra manifested component was accepted" >&2
  exit 1
fi
grep -Fq 'component payload closure contains missing or extra files' "$TMP/negative.stderr"
rm -f -- "$TMP/components/extra.txt"
build_component_closure
[[ "$COMPONENTS_SHA" == "$VALID_COMPONENTS_SHA" ]]
build_fetch_receipts

# The final output cannot be nested within a source closure or the pinned venv.
negative_command=("${finalize_command[@]}")
for index in "${!negative_command[@]}"; do
  [[ "${negative_command[index]}" == "$TMP/finalized" ]] \
    && negative_command[index]="$TMP/components/nested-finalization"
done
if "${negative_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "output nested in components was accepted" >&2
  exit 1
fi
grep -Fq 'output directory overlaps an immutable input closure' "$TMP/negative.stderr"

negative_command=("${finalize_command[@]}")
for index in "${!negative_command[@]}"; do
  [[ "${negative_command[index]}" == "$FINALIZER_SHA" ]] \
    && negative_command[index]="$(printf '1%.0s' {1..64})"
done
if "${negative_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "wrong bundled finalizer digest was accepted" >&2
  exit 1
fi

negative_command=("${finalize_command[@]}")
for index in "${!negative_command[@]}"; do
  [[ "${negative_command[index]}" == "$PYTHON_SHA" ]] \
    && negative_command[index]="$(printf '0%.0s' {1..64})"
done
if "${negative_command[@]}" > /dev/null 2> "$TMP/negative.stderr"; then
  echo "wrong Phase-B Python digest was accepted" >&2
  exit 1
fi
[[ ! -e "$TMP/finalized" && ! -L "$TMP/finalized" ]]

"${finalize_command[@]}" > "$TMP/finalize.stdout" 2> "$TMP/finalize.stderr"
grep -Fq 'UED_TERMINAL_FINALIZATION_COMPLETE' "$TMP/finalize.stdout"
(cd "$TMP/finalized" && sha256sum -c --strict SHA256SUMS >/dev/null)
[[ "$(sha256sum "$BUNDLE/SHA256SUMS" | awk '{print $1}')" == "$BUNDLE_SHA" ]]
(cd "$BUNDLE" && sha256sum -c --strict SHA256SUMS >/dev/null)
! find "$BUNDLE" -type l -print -quit | grep -q .
! find "$BUNDLE" -type d -name __pycache__ -print -quit | grep -q .
(cd "$BUNDLE" && find . -type f -print | LC_ALL=C sort) > "$TMP/bundle-actual-files"
{ awk '{print $2}' "$BUNDLE/SHA256SUMS"; printf './SHA256SUMS\n'; } \
  | LC_ALL=C sort > "$TMP/bundle-expected-files"
cmp "$TMP/bundle-expected-files" "$TMP/bundle-actual-files"
UED_FINAL="$TMP/finalized" UED_PYTHON_VERSION="$PYTHON_VERSION" \
UED_SBATCH="$SBATCH_SHA" "$CPU_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["UED_FINAL"])
complete = json.loads((root / "COMPLETE").read_text(encoding="utf-8"))
finalization = json.loads((root / "FINALIZATION.json").read_text(encoding="utf-8"))
assert complete["status"] == "complete" and complete["paper_evidence"] is False
assert complete["analyzer_eligible"] is False
assert finalization["production_analyzer_invoked"] is False
assert finalization["assembler_validate_only_passed"] is True
assert finalization["terminal_scheduler"]["state"] == "COMPLETED"
assert finalization["terminal_scheduler"]["exit_code"] == "0:0"
assert finalization["input_closure_sha256"] == complete["input_closure_sha256"]
assert finalization["phase_b_python"]["version"] == os.environ["UED_PYTHON_VERSION"]
assert finalization["terminal_scheduler"]["gpu_model"] == "NVIDIA A100-SXM4-80GB"
assert finalization["terminal_chain_sbatch_sha256"] == os.environ["UED_SBATCH"]
assert len(finalization["post_terminal_fetch_receipts"]) == 4
for label in ("components", "stdout", "stderr", "submission"):
    receipt = finalization["post_terminal_fetch_receipts"][label]
    assert receipt["fetch_started_epoch"] >= 1786737602
    assert receipt["retrieved_epoch"] >= receipt["fetch_started_epoch"]
for name in (
    "submission-receipt.tsv", "fetch-components.tsv", "fetch-slurm-stdout.tsv",
    "fetch-slurm-stderr.tsv", "fetch-submission-receipt.tsv",
    "PHASE_B_PYTHON_FREEZE.txt", "PHASE_B_PYVENV.cfg",
):
    assert (root / name).is_file()
package = root / "package/engineering-slurm-8123456-frontier-s101"
manifest = json.loads((package / "run-manifest.json").read_text(encoding="utf-8"))
assert manifest["paper_evidence"] is False
assert manifest["analyzer_eligible"] is False
assert manifest["endpoint_class"] == "bounded_engineering_test"
assert (package / "COMPLETE").is_file()
PY

printf 'UED terminal-chain local staged E2E: PASS\n'
