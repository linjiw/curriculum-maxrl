#!/bin/bash
# One-command artifact check (draft-review 2026-08-04: "make one clean
# artifact command reproduce the paper").
#
#   bash reproduce.sh            # verify: tests + figure regeneration +
#                                # endpoint derivations + manifest checksums
#   bash reproduce.sh --build    # additionally rebuild both PDF wrappers
#
# Requires: python3 (numpy), matplotlib for figure regeneration
# (PYTHON_MPL below points at a venv that has it), pdflatex for --build.
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
PYTHON_MPL=${PYTHON_MPL:-$HOME/venvs/maxrl311/bin/python}
[ -x "$PYTHON_MPL" ] || PYTHON_MPL=$PYTHON
FAIL=0
step() { echo; echo "== $* =="; }

step "1/5 proposition + integration tests"
$PYTHON -m curriculum_maxrl.test_mass_formulas
$PYTHON curriculum_maxrl/test_verl_curriculum.py
$PYTHON frontier_rl/test_framework.py
$PYTHON -m unittest curriculum_maxrl.test_audit_countdown_sft_overlap

step "2/5 figure-endpoint derivations (fail on mismatch)"
(cd paper/figures && $PYTHON verify_fig2a_from_artifacts.py)
(cd paper/figures && $PYTHON verify_fig2c_from_logs.py) || {
  echo "  (fig2c needs the execution fork's raw maze logs; skipping is a"
  echo "   MISS only if ../maxrl is present)"; [ -d ../maxrl ] && FAIL=1; }
$PYTHON curriculum_maxrl/maze_gpu_factorial/block_reanalysis.py >/dev/null

step "3/5 manifest checksums"
$PYTHON - <<'EOF'
import hashlib, json, os, sys
m = json.load(open('paper/results/manifest.json'))
bad = []
sections = [m['figures'], m.get('audits', {})]
for section in sections:
  for name, e in section.items():
    for path, want in e.get('checksums', {}).items():
        if not os.path.exists(path):
            bad.append(f"{name}: missing input {path}"); continue
        got = hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]
        if got != want:
            bad.append(f"{name}: {path} checksum {got} != {want}")
for b in bad: print("  MISMATCH:", b)
print(f"  {sum(len(e.get('checksums',{})) for section in sections for e in section.values())} inputs checked,"
      f" {len(bad)} mismatches")
sys.exit(1 if bad else 0)
EOF

step "4/5 regenerate all figures from frozen inputs"
for f in paper/figures/fig*.py; do
  echo "  $f"; (cd paper/figures && $PYTHON_MPL "$(basename "$f")" >/dev/null)
done

step "5/5 prereg-verdict spot checks (quoted numbers vs artifacts)"
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
checks.append(("wave2 P-F2 uniform 6/6 (confirmed)", pf2['n_pos'] == 6))
# reviewer arms
ra = json.load(open('curriculum_maxrl/countdown_reviewer_arms/reviewer_arms_verdicts.json'))
checks.append(("ARM-A P-R1 refuted", ra['P_R1']['verdict'] == 'REFUTED'))
checks.append(("ARM-B P-R2 confirmed 3/3", 'CONFIRMED' in ra['P_R2']['verdict']
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
  echo "  built paper/main.pdf + paper/main_iclr.pdf"
fi

echo
[ "$FAIL" = 0 ] && echo "REPRODUCE: ALL CHECKS PASSED" || { echo "REPRODUCE: FAILURES ABOVE"; exit 1; }
