#!/bin/bash
# One-command artifact check (draft-review 2026-08-04: "make one clean
# artifact command reproduce the paper").
#
#   bash reproduce.sh            # verify: tests + figure regeneration +
#                                # endpoint derivations + manifest checksums
#   bash reproduce.sh --build    # additionally rebuild the canonical compact
#                                # ICLR-2027 PDF and the extended-research PDF
#
# Requires: python3 (numpy), matplotlib for figure regeneration
# (PAPER_FIGURE_PYTHON may point at a separate venv). Portable mode validates
# declared inputs and fresh nonempty renders. Set REPRO_MODE=exact for strict
# byte comparison under the pinned figure and Tectonic toolchains. For
# --build, set TECTONIC_BIN or place tectonic on PATH; exact mode requires
# TECTONIC_BIN plus the pinned populated XDG_CACHE_HOME. SOURCE_DATE_EPOCH is
# pinned by default.
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
PAPER_FIGURE_PYTHON=${PAPER_FIGURE_PYTHON:-}
REPRO_MODE=${REPRO_MODE:-portable}
case "$REPRO_MODE" in portable|exact) ;; *)
  echo "REPRO_MODE must be portable or exact" >&2; exit 2;;
esac
FAIL=0
step() { echo; echo "== $* =="; }

step "1/5 proposition + integration tests"
$PYTHON -m curriculum_maxrl.test_mass_formulas
$PYTHON curriculum_maxrl/test_verl_curriculum.py
$PYTHON frontier_rl/test_framework.py
$PYTHON -m unittest curriculum_maxrl.test_audit_countdown_sft_overlap

step "2/5 compact-paper derivations (no outside-repository reads)"
$PYTHON control_port/verify_note_claims.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PYTHON -m pytest -q \
  curriculum_maxrl/test_count_law_stats.py \
  curriculum_maxrl/test_relabel_degeneracy.py \
  curriculum_maxrl/group_law_flip/test_analyze_group_law_flip.py

step "3/5 manifest, canonical-paper, and compact-registry checks"
$PYTHON - <<'EOF'
import collections, hashlib, json, os, re, sys
m = json.load(open('paper/results/manifest.json'))
bad = []
manuscript = m.get('manuscript', {})
expected_manuscript = {
  'body': 'paper/body_iclr.tex',
  'wrappers': ['paper/main_iclr2027.tex'],
  'extended_body': 'paper/body.tex',
  'extended_wrapper': 'paper/main.tex',
}
for key, want in expected_manuscript.items():
  if manuscript.get(key) != want:
    bad.append(f"manuscript manifest {key}: {manuscript.get(key)!r} != {want!r}")
for path in [
    expected_manuscript['body'], *expected_manuscript['wrappers'],
    expected_manuscript['extended_body'], expected_manuscript['extended_wrapper'],
]:
  if not os.path.isfile(path):
    bad.append(f"missing manuscript source {path}")
sections = [m['figures'], m.get('audits', {})]
for section in sections:
  for name, e in section.items():
    for path, want in e.get('checksums', {}).items():
        if not os.path.exists(path):
            bad.append(f"{name}: missing input {path}"); continue
        got = hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]
        if got != want:
            bad.append(f"{name}: {path} checksum {got} != {want}")

body = open(expected_manuscript['body'], encoding='utf-8').read()
included = {
    'paper/' + match + ('' if os.path.splitext(match)[1] else '.pdf')
    for match in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', body)
}
declared = {
    output
    for entry in m['figures'].values()
    for output in entry.get('outputs', [])
    if output.endswith('.pdf')
}
if included != declared:
    bad.append(
        f"compact figure perimeter mismatch: included-only={sorted(included-declared)!r}, "
        f"manifest-only={sorted(declared-included)!r}")
for name, entry in m['figures'].items():
    script = entry.get('script')
    if not script or not os.path.isfile(script):
        bad.append(f"{name}: missing declared script {script!r}")
    for output in entry.get('outputs', []):
        if not os.path.isfile(output) or os.path.getsize(output) == 0:
            bad.append(f"{name}: missing/nonempty tracked output {output}")

deposit = json.load(open('paper/PROVENANCE_DEPOSIT.json'))
if deposit.get('status') != 'not_deposited' or deposit.get('doi') is not None:
  bad.append("provenance deposit must remain explicitly unminted until PI publication")
for entry in deposit.get('payload', []):
  path = entry.get('path')
  if not path or not os.path.isfile(path):
    bad.append(f"provenance deposit: missing payload {path!r}")
    continue
  data = open(path, 'rb').read()
  if len(data) != entry.get('bytes'):
    bad.append(f"provenance deposit: {path} bytes {len(data)} != {entry.get('bytes')}")
  got = hashlib.sha256(data).hexdigest()
  if got != entry.get('sha256'):
    bad.append(f"provenance deposit: {path} sha256 {got} != {entry.get('sha256')}")

registry = json.load(open('curriculum_maxrl/run_registry.json'))
rows = registry.get('rows')
expected_suites = {
  'maze': 35, 'countdown': 11, 'gsm8k': 7,
  'group_law_flip': 1, 'amaze_gate': 1,
}
if not isinstance(rows, list):
  bad.append("compact registry rows is not a list")
  suite_counts = {}
else:
  suite_counts = dict(collections.Counter(row.get('suite') for row in rows))
  if registry.get('n_rows') != len(rows):
    bad.append(f"compact registry n_rows {registry.get('n_rows')!r} != {len(rows)} rows")
if registry.get('n_rows') != 55:
  bad.append(f"compact registry manuscript count {registry.get('n_rows')!r} != 55")
if suite_counts != expected_suites:
  bad.append(f"compact registry suite counts {suite_counts!r} != {expected_suites!r}")
print(f"  {sum(len(e.get('checksums',{})) for section in sections for e in section.values())} manifest inputs checked")
print(f"  compact registry: n_rows={registry.get('n_rows')}, suites={suite_counts}")
for b in bad: print("  MISMATCH:", b)
sys.exit(1 if bad else 0)
EOF

step "4/5 regenerate the compact figure perimeter in isolation ($REPRO_MODE)"
FIGURE_REPRO_DIR=$(mktemp -d paper/.figures-repro.XXXXXX)
figure_repro_cleanup() { rm -rf "$FIGURE_REPRO_DIR"; }
trap figure_repro_cleanup EXIT
FIGURE_PYTHON=${PAPER_FIGURE_PYTHON:-$PYTHON}
case "$FIGURE_PYTHON" in */*) ;; *) FIGURE_PYTHON=$(command -v "$FIGURE_PYTHON") ;; esac
[ -x "$FIGURE_PYTHON" ] || {
  echo "figure interpreter is not executable: $FIGURE_PYTHON" >&2; exit 1; }
FIGURE_PYTHON_SHA256=4627a60ce761a303bb866244833a914aabab9880b8082fbb0fe8cf35c91ea3ed
FIGURE_ENVIRONMENT_SHA256=9a05c8bfbd95df8f94aa413e6304b2dbcd709fabb5e006a198c96688004cf614
mkdir "$FIGURE_REPRO_DIR/.mplconfig"
if [ "$REPRO_MODE" = exact ]; then
  [ -n "$PAPER_FIGURE_PYTHON" ] || {
    echo "exact mode requires PAPER_FIGURE_PYTHON" >&2; exit 1; }
  [ "$(sha256sum "$FIGURE_PYTHON" | cut -d' ' -f1)" = "$FIGURE_PYTHON_SHA256" ] || {
    echo "PAPER_FIGURE_PYTHON executable does not match" >&2; exit 1; }
  FIGURE_ENVIRONMENT_ACTUAL=$(MPLCONFIGDIR="$FIGURE_REPRO_DIR/.mplconfig" \
    "$FIGURE_PYTHON" -I -B - <<'PY'
from hashlib import sha256
import json
import sys

import contourpy
import cycler
import dateutil
import fontTools
import kiwisolver
import matplotlib
from matplotlib import font_manager
import numpy
import packaging
import PIL
import pyparsing
import six

record = {
    "packages": {
        "python": sys.version.split()[0],
        "contourpy": contourpy.__version__,
        "cycler": cycler.__version__,
        "fonttools": fontTools.__version__,
        "kiwisolver": kiwisolver.__version__,
        "matplotlib": matplotlib.__version__,
        "numpy": numpy.__version__,
        "packaging": packaging.__version__,
        "pillow": PIL.__version__,
        "pyparsing": pyparsing.__version__,
        "python-dateutil": dateutil.__version__,
        "six": six.__version__,
    },
    "fonts": {},
}
for family in ("DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono"):
    path = font_manager.findfont(family)
    with open(path, "rb") as stream:
        record["fonts"][family] = sha256(stream.read()).hexdigest()
payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
print(sha256(payload).hexdigest())
PY
  )
  [ "$FIGURE_ENVIRONMENT_ACTUAL" = "$FIGURE_ENVIRONMENT_SHA256" ] || {
    echo "Pinned figure package/font environment does not match" >&2; exit 1; }
else
  MPLCONFIGDIR="$FIGURE_REPRO_DIR/.mplconfig" "$FIGURE_PYTHON" - <<'PY'
import matplotlib, numpy
print(f"  portable figure environment: matplotlib={matplotlib.__version__}, numpy={numpy.__version__}")
PY
fi
cp -a paper/figures/. "$FIGURE_REPRO_DIR/"
COMPACT_FIGURES=$($PYTHON - <<'PY'
import json
m = json.load(open('paper/results/manifest.json'))
for entry in m['figures'].values():
    print(entry['script'].rsplit('/', 1)[-1][:-3])
PY
)
for stem in $COMPACT_FIGURES; do
  rm -f "$FIGURE_REPRO_DIR/$stem.pdf" "$FIGURE_REPRO_DIR/$stem.png"
  echo "  paper/figures/$stem.py"
  (cd "$FIGURE_REPRO_DIR" && \
    SOURCE_DATE_EPOCH=1786718220 FORCE_SOURCE_DATE=1 \
    MPLCONFIGDIR="$FIGURE_REPRO_DIR/.mplconfig" \
    "$FIGURE_PYTHON" -I -B "$stem.py" >/dev/null)
done
for stem in $COMPACT_FIGURES; do
  for ext in pdf png; do
    tracked="paper/figures/$stem.$ext"
    regenerated="$FIGURE_REPRO_DIR/$stem.$ext"
    [ -s "$tracked" ] && [ -s "$regenerated" ] || {
      echo "  MISSING regenerated figure: $stem.$ext" >&2; exit 1; }
    if [ "$REPRO_MODE" = exact ]; then
      cmp -s "$tracked" "$regenerated" || {
        echo "  MISMATCH regenerated figure: $stem.$ext" >&2; exit 1; }
    fi
  done
done
if [ "$REPRO_MODE" = portable ]; then
  "$FIGURE_PYTHON" - "$FIGURE_REPRO_DIR" $COMPACT_FIGURES <<'PY'
from pathlib import Path
import sys
from PIL import Image

root = Path(sys.argv[1])
for stem in sys.argv[2:]:
    pdf = root / f"{stem}.pdf"
    png = root / f"{stem}.png"
    if not pdf.read_bytes().startswith(b"%PDF-") or pdf.stat().st_size < 1000:
        raise SystemExit(f"invalid regenerated PDF: {pdf}")
    with Image.open(png) as image:
        image.verify()
    with Image.open(png) as image:
        if min(image.size) < 200:
            raise SystemExit(f"implausibly small regenerated PNG: {png} {image.size}")
print(f"  portable render validation passed for {len(sys.argv)-2} compact figures")
PY
else
  echo "  exact byte comparison passed for all compact figures"
fi
figure_repro_cleanup
trap - EXIT

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

if [ "${1:-}" = "--build" ]; then
  step "build: canonical compact ICLR-2027 + extended-research PDFs"
  PAPER_SOURCE_DATE_EPOCH=${PAPER_SOURCE_DATE_EPOCH:-1786718220}
  export SOURCE_DATE_EPOCH=$PAPER_SOURCE_DATE_EPOCH FORCE_SOURCE_DATE=1
  PAPER_BUILD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/curriculum-maxrl-paper.XXXXXX")
  PAPER_STAGED_COMPACT=
  PAPER_STAGED_WEB=
  PAPER_STAGED_EXTENDED=
  PAPER_PUBLISH_STARTED=0
  PAPER_PUBLISH_COMPLETE=0
  paper_build_cleanup() {
    rc=$1
    set +e
    if [ "$PAPER_PUBLISH_STARTED" = 1 ] && [ "$PAPER_PUBLISH_COMPLETE" != 1 ]; then
      install -m 0664 "$PAPER_BUILD_DIR/previous-main_iclr.pdf" paper/main_iclr.pdf
      install -m 0664 "$PAPER_BUILD_DIR/previous-paper-iclr.pdf" docs/paper-iclr.pdf
      install -m 0664 "$PAPER_BUILD_DIR/previous-main.pdf" paper/main.pdf
    fi
    [ -z "$PAPER_STAGED_COMPACT" ] || rm -f "$PAPER_STAGED_COMPACT"
    [ -z "$PAPER_STAGED_WEB" ] || rm -f "$PAPER_STAGED_WEB"
    [ -z "$PAPER_STAGED_EXTENDED" ] || rm -f "$PAPER_STAGED_EXTENDED"
    rm -rf "$PAPER_BUILD_DIR"
    trap - EXIT
    exit "$rc"
  }
  trap 'paper_build_cleanup $?' EXIT

  if [ -z "${TECTONIC_BIN:-}" ]; then
    TECTONIC_BIN=$(command -v tectonic || true)
  fi
  [ -n "$TECTONIC_BIN" ] || {
    echo "--build requires TECTONIC_BIN or tectonic on PATH" >&2; exit 1; }
  [ -x "$TECTONIC_BIN" ] || {
    echo "TECTONIC_BIN is not executable: $TECTONIC_BIN" >&2; exit 1; }
  TECTONIC_SHA256=397efac4cabf7dfa02f238fe23681215b535ea665e99ba27d123b8bc655b88cb
  TECTONIC_BUNDLE_ID=6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c
  TECTONIC_INDEX_SHA256=0fb434b0fa5fdebea7f767ed9c31939c99a780d6f95cd3f540aae55910bb5697
  TECTONIC_MAPPING_SHA256=1f94cb6e6893fb09037585fdde65d436f90e2d726175e06363723529f52c880e
  TECTONIC_FORMAT_SHA256=a86ffcac335474fb9fae47cd9986b929719dc3ddf29bfb31123ecc1790ef6bbb
  TECTONIC_BUNDLE_TREE_SHA256=c14bb81785d0b14fc2ae638a90d8b6d96bbf8180570e4b06ec3ca50aec09db17
  EXPECTED_COMPACT_SHA256=e3d566c40ce211867cd7be4658d4886c4326825083598bc25a7c30b12b38bff6
  EXPECTED_EXTENDED_SHA256=f9f387b4e29f1fbb0d4108820f6d6d380c4fdb3e7ea73fa86075bff8607d313c
  EXPECTED_COMPACT_LOG_SHA256=cb6210b31f0695a20e0fed913468c3b28fc3f1cd9008b776bce12a177844afa4
  EXPECTED_EXTENDED_LOG_SHA256=d13067aaa00bc88e153e8ce6d43143cfd9c615e14bf86b1786fd7c86410d77c6
  if [ "$REPRO_MODE" = exact ]; then
    [ "$(sha256sum "$TECTONIC_BIN" | cut -d' ' -f1)" = "$TECTONIC_SHA256" ] || {
      echo "TECTONIC_BIN does not match the pinned 0.16.9 executable" >&2; exit 1; }
    [ -n "${XDG_CACHE_HOME:-}" ] || {
      echo "Pinned Tectonic build requires explicit XDG_CACHE_HOME" >&2; exit 1; }
    TECTONIC_INDEX="$XDG_CACHE_HOME/Tectonic/bundles/data/$TECTONIC_BUNDLE_ID.index"
    TECTONIC_MAPPING="$XDG_CACHE_HOME/Tectonic/bundles/hashes/https,58,,47,,47,relay.fullyjustified.net,47,default_bundle_v33.tar"
    TECTONIC_BUNDLE_DIR="$XDG_CACHE_HOME/Tectonic/bundles/data/$TECTONIC_BUNDLE_ID"
    TECTONIC_FORMAT="$XDG_CACHE_HOME/Tectonic/formats/$TECTONIC_BUNDLE_ID-latex-33.fmt"
    [ -f "$TECTONIC_INDEX" ] \
      && [ ! -L "$TECTONIC_INDEX" ] \
      && [ "$(sha256sum "$TECTONIC_INDEX" | cut -d' ' -f1)" = "$TECTONIC_INDEX_SHA256" ] || {
        echo "Tectonic bundle index is missing or does not match" >&2; exit 1; }
    [ -f "$TECTONIC_FORMAT" ] \
      && [ ! -L "$TECTONIC_FORMAT" ] \
      && [ "$(sha256sum "$TECTONIC_FORMAT" | cut -d' ' -f1)" = "$TECTONIC_FORMAT_SHA256" ] || {
        echo "Tectonic LaTeX format is missing or does not match" >&2; exit 1; }
    [ -f "$TECTONIC_MAPPING" ] \
      && [ ! -L "$TECTONIC_MAPPING" ] \
      && [ "$(sha256sum "$TECTONIC_MAPPING" | cut -d' ' -f1)" = "$TECTONIC_MAPPING_SHA256" ] || {
        echo "Tectonic URL-to-bundle mapping is missing or does not match" >&2; exit 1; }
    TECTONIC_BUNDLE_TREE_ACTUAL=$($PYTHON - "$TECTONIC_BUNDLE_DIR" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

root = Path(sys.argv[1])
if not root.is_dir() or root.is_symlink():
    raise SystemExit("invalid Tectonic bundle directory")
digest = sha256()
count = 0
for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"invalid Tectonic bundle member: {path}")
    relative = path.relative_to(root).as_posix().encode("utf-8")
    digest.update(relative)
    digest.update(b"\0")
    digest.update(sha256(path.read_bytes()).digest())
    count += 1
if count != 386:
    raise SystemExit(f"unexpected Tectonic bundle member count: {count}")
print(digest.hexdigest())
PY
    )
    [ "$TECTONIC_BUNDLE_TREE_ACTUAL" = "$TECTONIC_BUNDLE_TREE_SHA256" ] || {
      echo "Tectonic bundle members do not match" >&2; exit 1; }
    echo "  exact Tectonic executable/cache contract passed"
  else
    echo "  portable Tectonic: $($TECTONIC_BIN --version | head -n 1)"
  fi

  (cd paper && \
    "$TECTONIC_BIN" -C --keep-logs -o "$PAPER_BUILD_DIR" main_iclr2027.tex \
    && "$TECTONIC_BIN" -C --keep-logs -o "$PAPER_BUILD_DIR" main.tex)

  require_build_hash() {
    path=$1
    expected=$2
    [ -f "$path" ] && [ ! -L "$path" ] \
      && [ "$(sha256sum "$path" | cut -d' ' -f1)" = "$expected" ] || {
        echo "Built artifact is missing or does not match: $path" >&2; return 1; }
  }
  require_build_file() {
    path=$1
    [ -s "$path" ] && [ ! -L "$path" ] || {
      echo "Built artifact is missing, empty, or a symlink: $path" >&2; return 1; }
  }
  for built in \
      "$PAPER_BUILD_DIR/main_iclr2027.pdf" "$PAPER_BUILD_DIR/main.pdf" \
      "$PAPER_BUILD_DIR/main_iclr2027.log" "$PAPER_BUILD_DIR/main.log"; do
    require_build_file "$built"
  done
  if [ "$REPRO_MODE" = exact ]; then
    require_build_hash "$PAPER_BUILD_DIR/main_iclr2027.pdf" "$EXPECTED_COMPACT_SHA256"
    require_build_hash "$PAPER_BUILD_DIR/main.pdf" "$EXPECTED_EXTENDED_SHA256"
    require_build_hash "$PAPER_BUILD_DIR/main_iclr2027.log" "$EXPECTED_COMPACT_LOG_SHA256"
    require_build_hash "$PAPER_BUILD_DIR/main.log" "$EXPECTED_EXTENDED_LOG_SHA256"
  else
    head -c 5 "$PAPER_BUILD_DIR/main_iclr2027.pdf" | grep -q '^%PDF-' || {
      echo "Compact build is not a PDF" >&2; exit 1; }
    head -c 5 "$PAPER_BUILD_DIR/main.pdf" | grep -q '^%PDF-' || {
      echo "Extended build is not a PDF" >&2; exit 1; }
  fi
  if grep -Eiq 'undefined references|undefined citations|overfull|emergency stop|fatal error' \
      "$PAPER_BUILD_DIR/main_iclr2027.log" "$PAPER_BUILD_DIR/main.log"; then
    echo "Paper build logs contain a forbidden diagnostic" >&2
    exit 1
  fi

  for target in paper/main_iclr.pdf docs/paper-iclr.pdf paper/main.pdf; do
    [ -f "$target" ] && [ ! -L "$target" ] || {
      echo "Refusing to replace missing or non-regular PDF target: $target" >&2; exit 1; }
  done
  install -m 0664 paper/main_iclr.pdf "$PAPER_BUILD_DIR/previous-main_iclr.pdf"
  install -m 0664 docs/paper-iclr.pdf "$PAPER_BUILD_DIR/previous-paper-iclr.pdf"
  install -m 0664 paper/main.pdf "$PAPER_BUILD_DIR/previous-main.pdf"
  PAPER_STAGED_COMPACT=$(mktemp paper/.main_iclr.pdf.new.XXXXXX)
  PAPER_STAGED_WEB=$(mktemp docs/.paper-iclr.pdf.new.XXXXXX)
  PAPER_STAGED_EXTENDED=$(mktemp paper/.main.pdf.new.XXXXXX)
  install -m 0664 "$PAPER_BUILD_DIR/main_iclr2027.pdf" "$PAPER_STAGED_COMPACT"
  install -m 0664 "$PAPER_BUILD_DIR/main_iclr2027.pdf" "$PAPER_STAGED_WEB"
  install -m 0664 "$PAPER_BUILD_DIR/main.pdf" "$PAPER_STAGED_EXTENDED"
  cmp -s "$PAPER_BUILD_DIR/main_iclr2027.pdf" "$PAPER_STAGED_COMPACT"
  cmp -s "$PAPER_BUILD_DIR/main_iclr2027.pdf" "$PAPER_STAGED_WEB"
  cmp -s "$PAPER_BUILD_DIR/main.pdf" "$PAPER_STAGED_EXTENDED"
  PAPER_PUBLISH_STARTED=1
  mv -f "$PAPER_STAGED_COMPACT" paper/main_iclr.pdf
  PAPER_STAGED_COMPACT=
  mv -f "$PAPER_STAGED_WEB" docs/paper-iclr.pdf
  PAPER_STAGED_WEB=
  mv -f "$PAPER_STAGED_EXTENDED" paper/main.pdf
  PAPER_STAGED_EXTENDED=
  cmp -s "$PAPER_BUILD_DIR/main_iclr2027.pdf" paper/main_iclr.pdf
  cmp -s "$PAPER_BUILD_DIR/main_iclr2027.pdf" docs/paper-iclr.pdf
  cmp -s "$PAPER_BUILD_DIR/main.pdf" paper/main.pdf
  if [ "$REPRO_MODE" = exact ]; then
    require_build_hash paper/main_iclr.pdf "$EXPECTED_COMPACT_SHA256"
    require_build_hash docs/paper-iclr.pdf "$EXPECTED_COMPACT_SHA256"
    require_build_hash paper/main.pdf "$EXPECTED_EXTENDED_SHA256"
  fi
  PAPER_PUBLISH_COMPLETE=1
  echo "  built paper/main_iclr.pdf + docs/paper-iclr.pdf + paper/main.pdf"
  sha256sum paper/main_iclr.pdf docs/paper-iclr.pdf paper/main.pdf
fi

echo
[ "$FAIL" = 0 ] && echo "REPRODUCE: ALL CHECKS PASSED" || { echo "REPRODUCE: FAILURES ABOVE"; exit 1; }
