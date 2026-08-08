#!/usr/bin/env python3
"""Verify fig2 panel (a)'s transcribed skill-chain endpoints against
the committed per-seed artifacts (draft-review 2026-08-04, artifact
self-containment item — companion to verify_fig2c_from_logs.py).

Panel (a) bars: uniform, teacher, oracle, full stack, oracle+rec.
Sources (per-seed AUC arrays, 5 seeds each):
  uniform      v7_oracle_result.json      "uniform"
  teacher      v7_oracle_result.json      "teacher_thompson"
  oracle       hindsight_controls.json    "oracle_g4"           (no-floor,
               gamma-matched true-pass-rate oracle, post-retraction —
               the v7 battery run; v7_oracle_result's "oracle_gamma4"
               is an earlier with-floor variant at .8836, NOT the bar)
  full stack   v7_oracle_result.json      "full_stack_gamma4_hs"
  oracle+rec.  hindsight_controls.json    "oracle_g4_hs"

Tolerance 0.002 on the mean (table stores 3–4 significant digits).
Exit nonzero on mismatch.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "..", "..", "frontier_rl", "examples")

SOURCES = {
    "uniform":     ("v7_oracle_result.json", "uniform"),
    "teacher":     ("v7_oracle_result.json", "teacher_thompson"),
    "oracle":      ("hindsight_controls.json", "oracle_g4"),
    "full stack":  ("v7_oracle_result.json", "full_stack_gamma4_hs"),
    "oracle+rec.": ("hindsight_controls.json", "oracle_g4_hs"),
}


PANEL_B_SOURCES = {
    # frontier-heavy regime arms in results_baselines_regimes.json
    "uniform":    "uniform+maxrl",
    "DAPO":       "dapo+maxrl",
    "teacher":    "teacher+maxrl",
    "+recycling": "uniform+maxrl+hindsight",   # the bar quotes the
    # uniform+recycling arm (.931); teacher+recycling ties at .928
}


def main():
    data = json.load(open(os.path.join(HERE, "data",
                                       "fig2_ladder_data.json")))
    ok = True

    table = data["panel_a"]
    for i, label in enumerate(table["labels"]):
        fname, key = SOURCES[label]
        arm = json.load(open(os.path.join(EXAMPLES, fname)))[key]
        m = float(np.mean(arm["auc_per_seed"]))
        tm = table["auc"][i]
        line_ok = abs(m - tm) <= 0.002
        ok &= line_ok
        print(f"(a) {label:>12}: derived {m:.4f} (n={len(arm['auc_per_seed'])}) "
              f"vs table {tm} {'OK' if line_ok else 'MISMATCH'}")

    tb = data["panel_b"]
    regimes = json.load(open(os.path.join(
        HERE, "..", "..", "curriculum_maxrl",
        "results_baselines_regimes.json")))["frontier-heavy"]
    for i, label in enumerate(tb["labels"]):
        arm = regimes[PANEL_B_SOURCES[label]]
        m = float(np.mean(arm["auc_per_seed"]))
        tm = tb["auc"][i]
        line_ok = abs(m - tm) <= 0.005   # bar stores 2 decimals (.93)
        ok &= line_ok
        print(f"(b) {label:>12}: derived {m:.4f} (n={len(arm['auc_per_seed'])}) "
              f"vs table {tm} {'OK' if line_ok else 'MISMATCH'}")

    if not ok:
        sys.exit(1)
    print("fig2 panels (a)+(b) endpoints verified against per-seed artifacts")


if __name__ == "__main__":
    main()
