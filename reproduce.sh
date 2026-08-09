#!/bin/bash
# One-command artifact check (draft-review 2026-08-04: "make one clean
# artifact command reproduce the paper").
#
#   bash reproduce.sh            # verify: tests + figure regeneration +
#                                # endpoint derivations + manifest checksums
#   bash reproduce.sh --build    # additionally rebuild both PDF wrappers
#
# Requires: python3 (numpy and pytest), matplotlib for figure regeneration, and pdflatex
# for --build.  The Acrobot release lane also uses uv to provision the frozen
# Python 3.12.13 / NumPy 2.5.1 / Gymnasium 1.3.0 test runtime.  Capped-HORA
# verification uses CPython 3.9.6 / NumPy 1.26.4 via HORA_PYTHON, an already
# compatible /usr/bin/python3, or an isolated uv environment. Set PYTHON_MPL to
# a matplotlib-capable interpreter when it is different from PYTHON.
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
HORA_RUNTIME_CHECK='import platform, numpy; assert platform.python_implementation() == "CPython"; assert platform.python_version() == "3.9.6"; assert numpy.__version__ == "1.26.4"'
if [ -n "${HORA_PYTHON:-}" ]; then
  HORA_RUN=("$HORA_PYTHON")
elif /usr/bin/python3 -c "$HORA_RUNTIME_CHECK" >/dev/null 2>&1; then
  HORA_RUN=(/usr/bin/python3)
elif command -v uv >/dev/null 2>&1; then
  HORA_RUN=(uv run --python 3.9.6 --with numpy==1.26.4 python)
else
  echo "The capped-HORA artifact requires CPython 3.9.6 with NumPy 1.26.4." >&2
  echo "Install uv or set HORA_PYTHON=/path/to/a matching interpreter." >&2
  exit 2
fi
if ! "${HORA_RUN[@]}" -c "$HORA_RUNTIME_CHECK" >/dev/null 2>&1; then
  echo "The capped-HORA artifact requires CPython 3.9.6 with NumPy 1.26.4." >&2
  echo "HORA_PYTHON, when set, must point to that exact runtime." >&2
  exit 2
fi
if [ -n "${PYTHON_MPL:-}" ]; then
  if ! "$PYTHON_MPL" -c 'import matplotlib, numpy' >/dev/null 2>&1; then
    echo "PYTHON_MPL=$PYTHON_MPL cannot import numpy and matplotlib" >&2
    exit 2
  fi
else
  ARTIFACT_PYTHON_CANDIDATES=("$PYTHON" "$HOME/venvs/maxrl311/bin/python")
  if command -v conda >/dev/null 2>&1; then
    ARTIFACT_CONDA_BASE=$(conda info --base 2>/dev/null || true)
    [ -n "$ARTIFACT_CONDA_BASE" ] && \
      ARTIFACT_PYTHON_CANDIDATES+=("$ARTIFACT_CONDA_BASE/bin/python")
  fi
  for ARTIFACT_PYTHON_CANDIDATE in "${ARTIFACT_PYTHON_CANDIDATES[@]}"; do
    if command -v "$ARTIFACT_PYTHON_CANDIDATE" >/dev/null 2>&1 && \
       "$ARTIFACT_PYTHON_CANDIDATE" -c \
         'import matplotlib, numpy' >/dev/null 2>&1; then
      PYTHON_MPL=$ARTIFACT_PYTHON_CANDIDATE
      break
    fi
  done
  if [ -z "${PYTHON_MPL:-}" ]; then
    echo "No Python interpreter with numpy and matplotlib was found." >&2
    echo "Install them or set PYTHON_MPL=/path/to/python, then retry." >&2
    exit 2
  fi
fi
FAIL=0
step() { echo; echo "== $* =="; }

step "1/5 proposition + integration tests"
$PYTHON -m curriculum_maxrl.test_mass_formulas
$PYTHON -m unittest curriculum_maxrl.test_fixed_budget_n_sweep
$PYTHON -m unittest curriculum_maxrl.test_postguidance_hora_factorial
$PYTHON -m unittest curriculum_maxrl.test_correlated_rollout_stress
"${HORA_RUN[@]}" -m unittest curriculum_maxrl.test_capped_hora_robustness
$PYTHON -m unittest curriculum_maxrl.test_audit_countdown_sft_overlap
$PYTHON -m unittest curriculum_maxrl.analysis.test_maze_wave2_auc_multiverse
$PYTHON -m pytest -q verl_integration/test_response_goal_rewrite.py
$PYTHON curriculum_maxrl/build_run_registry.py --check
$PYTHON curriculum_maxrl/test_verl_curriculum.py
$PYTHON frontier_rl/test_framework.py
$PYTHON frontier_rl/examples/test_analyze_acrobot_v3_mechanism.py
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required for the pinned Acrobot/Gymnasium release tests" >&2
  exit 2
fi
uv run --python 3.12.13 \
  --with numpy==2.5.1 \
  --with 'gymnasium[classic-control]==1.3.0' \
  --with pytest==8.4.2 \
  python -m pytest -q \
    frontier_rl/test_gymnasium.py \
    frontier_rl/examples/test_acrobot_neural.py \
    frontier_rl/examples/test_run_acrobot_neural.py \
    frontier_rl/examples/test_run_acrobot_curriculum_tournament.py \
    frontier_rl/examples/test_analyze_acrobot_curriculum_tournament.py \
    frontier_rl/examples/test_verify_acrobot_curriculum_tournament_portable.py
PROCURL_TEST_RUNTIME=(
  uv run --python 3.12.13
  --with numpy==2.5.1
  --with 'gymnasium[classic-control]==1.3.0'
  --with pytest==8.4.2
  python
)
"${PROCURL_TEST_RUNTIME[@]}" -m pytest -q \
  frontier_rl/examples/test_run_acrobot_procurl_selection.py \
  frontier_rl/examples/test_analyze_acrobot_procurl_selection.py \
  frontier_rl/examples/test_build_acrobot_procurl_selection_lock.py \
  frontier_rl/examples/test_verify_acrobot_procurl_selection_portable.py
"${PROCURL_TEST_RUNTIME[@]}" -m pytest -q \
  frontier_rl/examples/test_build_acrobot_procurl_external_manifest.py \
  frontier_rl/examples/test_extract_acrobot_procurl_selection_diagnostics.py
"${PROCURL_TEST_RUNTIME[@]}" \
  -m frontier_rl.examples.build_acrobot_procurl_external_manifest check \
  --manifest \
    frontier_rl/examples/ACROBOT_PROCURL_SELECTION_EXTERNAL_RAW_MANIFEST.json \
  --artifact-root . --compact >/dev/null
PROCURL_SELECTION_RAW=frontier_rl/examples/acrobot_procurl_selection_confirmatory.json
if [ -f "$PROCURL_SELECTION_RAW" ]; then
  "${PROCURL_TEST_RUNTIME[@]}" \
    -m frontier_rl.examples.build_acrobot_procurl_external_manifest check \
    --manifest \
      frontier_rl/examples/ACROBOT_PROCURL_SELECTION_EXTERNAL_RAW_MANIFEST.json \
    --artifact-root . --full --raw "$PROCURL_SELECTION_RAW" >/dev/null
  "${PROCURL_TEST_RUNTIME[@]}" \
    -m frontier_rl.examples.extract_acrobot_procurl_selection_diagnostics check \
    --report frontier_rl/examples/acrobot_procurl_selection_diagnostics.json \
    --raw "$PROCURL_SELECTION_RAW" \
    --lock frontier_rl/examples/ACROBOT_PROCURL_SELECTION_LOCK.json \
    --development-gate \
      frontier_rl/examples/acrobot_procurl_selection_development_gates.json \
    --raw-logical-path \
      frontier_rl/examples/acrobot_procurl_selection_confirmatory.json >/dev/null
else
  echo "  ProCuRL external raw absent: compact receipt verified; full replay skipped"
fi
uv run --project curriculum_maxrl/digits_factorial --locked \
  python -m pytest -q curriculum_maxrl/digits_factorial/tests
uv run --project curriculum_maxrl/digits_factorial --locked \
  python -m curriculum_maxrl.digits_factorial.verify_portable \
    --source-only --skip-runtime-check >/dev/null
$PYTHON -m curriculum_maxrl.build_digits_external_bundle_manifest --check
$PYTHON -m curriculum_maxrl.analysis.anonymous_release --audit \
  --check-compact-receipt anonymous_release_receipt.json

step "2/5 figure-endpoint derivations (fail on mismatch)"
(cd paper/figures && $PYTHON verify_fig2a_from_artifacts.py)
(cd paper/figures && $PYTHON verify_fig2c_from_logs.py \
  --logs ../../curriculum_maxrl/maze_gpu)
$PYTHON paper/figures/fig_maze_block_analysis.py --check --no-figure
$PYTHON curriculum_maxrl/countdown_reviewer_arms/extract_arm_results.py --check
$PYTHON frontier_rl/examples/analyze_acrobot_v3_mechanism.py \
  --check frontier_rl/examples/acrobot_v3_mechanism_audit.json
"${HORA_RUN[@]}" - <<'EOF'
import json
from pathlib import Path
from curriculum_maxrl.analyze_capped_hora_robustness import (
    _canonical_json_bytes,
    build_analysis,
)

raw_path = Path("curriculum_maxrl/results_capped_hora_robustness.json").resolve()
stored_path = Path("curriculum_maxrl/results_capped_hora_robustness_analysis.json")
raw = json.loads(raw_path.read_text(encoding="utf-8"))
regenerated = build_analysis(raw, raw_path)
stored = json.loads(stored_path.read_text(encoding="utf-8"))
if Path(stored["raw_artifact"]).name != raw_path.name:
    raise SystemExit("capped-HORA stored raw-artifact basename does not match")
regenerated.pop("raw_artifact")
stored.pop("raw_artifact")
if _canonical_json_bytes(regenerated) != _canonical_json_bytes(stored):
    raise SystemExit("capped-HORA independent analysis does not match")
print("Capped-HORA 800-run analysis matches frozen artifact (path normalized)")
EOF
$PYTHON - <<'EOF'
import hashlib
import json
from pathlib import Path

root = Path("curriculum_maxrl/digits_factorial")

def sha(relative):
    return hashlib.sha256((root / relative).read_bytes()).hexdigest()

expected = {
    "SOURCE_LOCK.json": "d72b93a29a2e6a096a6acb0611f69fe6df9dcda80256000aed6de5208ef4eb36",
    "digits_split_manifest.json": "13dbc30cc5143edb043d76d76aac18bcc3a456b174a18ba488498fb99eab5e3f",
    "engineering/reseal_v3/engineering_audit.json": "d448a22793dc5a52ae7809dfc7572e3d669bebc5f8ef8d00a9865a0bb16150d1",
    "engineering/reseal_v3/independent_preseal_review.json": "0c387ddd1b2bb49d2be6e1eacff96c9281cfa17eea85f01b425733c9aabe24ff",
    "authorizations/development_authorization.json": "91ec2c5dbea8e4d424abcbb8176df9bfc8e809f46248b8345d41ea530fbbff61",
    "analyses/development_registered_v1/lr_selection.json": "dfc9d69faec78cff95e63ed7cd99a0e23c883dad5eba1a2fe378366730e06795",
    "analyses/development_registered_v1/independent_preconfirmation_review.json": "6ad8ede4ebdc6517e7d44395ae14dc26cc705029c1efab454eabb366d729803a",
    "authorizations/confirmation_tuned_authorization.json": "4d8967d9cb9a499c9cb3f439385d155e3df65fae7c7aed94260b81d1501f71ca",
    "authorizations/confirmation_common_authorization.json": "8e09bf3f43f5f6ede5f1b623dfd2d89b7488a7171a87aa9818da53f1f4ec13ab",
    "analyses/confirmation_registered_v1/confirmation_analysis.json": "346e46414d82155f2064ee2a448b89cf976bdc6897e0ff8ced06a389056799d6",
    "analyses/confirmation_registered_v1/common_identity_robustness.json": "01b2e07de7eaef8e66da938285df90cb9d871b77ffaa9195cfba1fc7ca14b85e",
    "analyses/confirmation_registered_v1/confirmation_bundle_receipt.json": "002eba15698ddc91e7360c8f3795cd1dc9562595338658c13bcdcc315b9fb3d6",
}
for relative, want in expected.items():
    got = sha(relative)
    if got != want:
        raise SystemExit(f"Digits receipt input changed: {relative}: {got} != {want}")

selection = json.loads((root / "analyses/development_registered_v1/lr_selection.json").read_text())
analysis = json.loads((root / "analyses/confirmation_registered_v1/confirmation_analysis.json").read_text())
identity = json.loads((root / "analyses/confirmation_registered_v1/common_identity_robustness.json").read_text())
receipt = json.loads((root / "analyses/confirmation_registered_v1/confirmation_bundle_receipt.json").read_text())
development_auth = json.loads((root / "authorizations/development_authorization.json").read_text())
tuned_auth = json.loads((root / "authorizations/confirmation_tuned_authorization.json").read_text())
common_auth = json.loads((root / "authorizations/confirmation_common_authorization.json").read_text())

assert selection["all_development_gates_passed"] is True
assert selection["selected_learning_rates_by_estimator"] == {"practical_maxrl": 0.1, "rloo": 0.1}
assert selection["selected_common_learning_rate"] == 0.1
assert selection["development_authorization"]["sha256"] == expected["authorizations/development_authorization.json"]
assert development_auth["source_lock_sha256"] == expected["SOURCE_LOCK.json"]
assert development_auth["zero_lr_engineering_audit"]["sha256"] == expected["engineering/reseal_v3/engineering_audit.json"]
for authorization, phase in ((tuned_auth, "confirmation_tuned"), (common_auth, "confirmation_common")):
    assert authorization["authorized_phase"] == phase
    assert authorization["source_lock_sha256"] == expected["SOURCE_LOCK.json"]
    assert authorization["lr_selection"]["sha256"] == expected["analyses/development_registered_v1/lr_selection.json"]
    assert authorization["independent_preseal_review"]["review_sha256"] == expected["analyses/development_registered_v1/independent_preconfirmation_review.json"]
assert analysis["lr_selection_sha256"] == expected["analyses/development_registered_v1/lr_selection.json"]
assert analysis["source_lock_sha256"] == expected["SOURCE_LOCK.json"]
assert analysis["tuned"]["n_complete_blocks"] == 24
assert analysis["tuned"]["cell_failures"] == []
assert analysis["tuned"]["treatment_delivery"]["passed"] is True
assert analysis["tuned"]["primary_supported"] is False
assert analysis["tuned"]["contrasts"]["interaction"]["exact_two_sided_sign_flip_p"] == 0.34955739974975586
assert identity["result"]["all_ledgers_and_five_recovery_checkpoints_byte_identical"] is True
assert identity["result"]["optimizer_sensitivity_variation_present"] is False
assert identity["paired_run_count"] == 144
assert identity["paired_binary_file_count"] == 864
assert receipt["source_lock_sha256"] == expected["SOURCE_LOCK.json"]
assert receipt["learning_rate_selection_sha256"] == expected["analyses/development_registered_v1/lr_selection.json"]
assert receipt["preconfirmation_review_sha256"] == expected["analyses/development_registered_v1/independent_preconfirmation_review.json"]
assert receipt["confirmation_analysis_sha256"] == expected["analyses/confirmation_registered_v1/confirmation_analysis.json"]
assert receipt["common_identity_robustness_sha256"] == expected["analyses/confirmation_registered_v1/common_identity_robustness.json"]
assert receipt["tuned_schedule"]["authorization_sha256"] == expected["authorizations/confirmation_tuned_authorization.json"]
assert receipt["common_schedule"]["authorization_sha256"] == expected["authorizations/confirmation_common_authorization.json"]
for schedule in ("tuned_schedule", "common_schedule"):
    assert receipt[schedule]["run_count"] == 144
    assert receipt[schedule]["complete_blocks"] == 24
    assert receipt[schedule]["failures"] == 0
    assert receipt[schedule]["paid_actions"] == 37748736
print("Digits locked tests/source and receipt chain pass without rerunning evidentiary training")
EOF
uv run --python 3.12.13 \
  --with numpy==2.5.1 \
  --with 'gymnasium[classic-control]==1.3.0' \
  python -m frontier_rl.examples.verify_acrobot_curriculum_tournament_portable \
    frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json \
    frontier_rl/examples/acrobot_curriculum_tournament_confirmatory.json \
    frontier_rl/examples/acrobot_curriculum_tournament_analysis.json \
    --source-root . >/dev/null

step "3/5 manifest checksums"
$PYTHON - <<'EOF'
import hashlib, json, os, sys
m = json.load(open('paper/results/manifest.json'))
bad = []
checked = 0
for section in ('figures', 'results'):
  for name, e in m.get(section, {}).items():
    for path, want in e.get('checksums', {}).items():
        checked += 1
        if not os.path.exists(path):
            bad.append(f"{section}.{name}: missing input {path}"); continue
        got = hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]
        if got != want:
            bad.append(f"{section}.{name}: {path} checksum {got} != {want}")
timing = m.get('timing', {})
if timing.get('artifact') and timing.get('checksum'):
    checked += 1
    path, want = timing['artifact'], timing['checksum']
    if not os.path.exists(path):
        bad.append(f"timing: missing input {path}")
    else:
        got = hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]
        if got != want:
            bad.append(f"timing: {path} checksum {got} != {want}")
for b in bad: print("  MISMATCH:", b)
print(f"  {checked} inputs checked, {len(bad)} mismatches")
sys.exit(1 if bad else 0)
EOF

step "4/5 regenerate all figures from frozen inputs"
for f in paper/figures/fig*.py; do
  echo "  $f"; (cd paper/figures && $PYTHON_MPL "$(basename "$f")" >/dev/null)
done

step "5/5 stored protocol/verdict spot checks (quoted numbers vs artifacts)"
$PYTHON - <<'EOF'
import json, numpy as np
checks = []
def close(a, b, tol=2e-3): return abs(a - b) <= tol
# gate variants (6.9): truep .879 / ungated .881 / freq .798
gv = json.load(open('curriculum_maxrl/results_gate_variants.json'))
for arm, q in [('truep', .879), ('ungated', .881), ('freq', .798)]:
    checks.append((f"gate-variants {arm}", close(np.mean([r['auc'] for r in gv['arms'][arm]]), q)))
# probe budget (6.9): recovery ~.98, 10/10
pb = json.load(open('curriculum_maxrl/results_gate_probe_budget.json'))
checks.append(("probe-budget P-PB1 10/10", pb['P-PB1 estp16 gap recovery']['n_ge_0.8'] == 10))
# row-vs-group (Remark 3): .952/.881/.749/.705
rg = json.load(open('curriculum_maxrl/results_row_vs_group.json'))
for arm, q in [('row', .952), ('group', .881), ('row_shared', .749), ('none', .705)]:
    checks.append((f"row-vs-group {arm}", close(np.mean([r['auc'] for r in rg['arms'][arm]]), q)))
# lr-matched (6.3b): maxrl premium .050; grpo x2/x4 in .007-.013
lr = json.load(open('curriculum_maxrl/results_lr_matched.json'))
p = lr['premium_secondary']
checks.append(("lr-matched maxrl premium .050", close(p['maxrl']['premium8'], .050)))
checks.append(("lr-matched grpo premium in [.007,.013]",
               .006 <= p['grpo_x4.0']['premium8'] <= p['grpo_x2.0']['premium8'] <= .0135))
# factorial waves
w1 = json.load(open('curriculum_maxrl/maze_gpu_factorial/results_factorial_wave1.json'))
w2 = json.load(open('curriculum_maxrl/maze_gpu_factorial/results_factorial_wave2.json'))
pf1 = [v for k, v in w1['contrasts'].items() if 'P-F1 uniform' in k][0]
checks.append(("wave1 P-F1 uniform 3/6 (failed)", pf1['n_pos'] == 3 and pf1['n'] == 6))
pf2 = [v for k, v in w2['contrasts'].items() if 'expl-AUC uniform' in k][0]
checks.append(("wave2 P-F2 uniform 6/6 (stored rule met)", pf2['n_pos'] == 6))
# reviewer arms
ra = json.load(open('curriculum_maxrl/countdown_reviewer_arms/reviewer_arms_verdicts.json'))
checks.append(("ARM-A P-R1 refuted", ra['P_R1']['verdict'] == 'REFUTED'))
checks.append(("ARM-B P-R2 stored rule met 3/3", 'CONFIRMED' in ra['P_R2']['verdict']
               and len(ra['P_R2']['t1_mean16']) == 3))
bad = [n for n, ok in checks if not ok]
for n, ok in checks: print(f"  {'ok ' if ok else 'FAIL'} {n}")
import sys; sys.exit(1 if bad else 0)
EOF

if [ "$1" = "--build" ]; then
  step "build: both PDF wrappers"
  (cd paper && pdflatex -interaction=nonstopmode main_iclr.tex >/dev/null \
    && pdflatex -interaction=nonstopmode main_iclr.tex >/dev/null \
    && pdflatex -interaction=nonstopmode main.tex >/dev/null \
    && pdflatex -interaction=nonstopmode main.tex >/dev/null)
  cp paper/main.pdf docs/paper-draft.pdf
  cp paper/main_iclr.pdf docs/paper-iclr.pdf
  echo "  built both PDF wrappers and refreshed both docs/ copies"
fi

echo
[ "$FAIL" = 0 ] && echo "REPRODUCE: ALL CHECKS PASSED" || { echo "REPRODUCE: FAILURES ABOVE"; exit 1; }
