#!/usr/bin/env python3
"""Verify fig2 panel (c)'s transcribed maze endpoints against raw seed
logs (draft-review 2026-08-04, artifact self-containment: "fig2/fig3
endpoint tables are transcribed rather than derived").

Derivation: AUC = mean of the mean-eval curve over training (matched
wall-clock protocol, steps >= 0); final = last eval. Three seeds per
arm. Tolerance 0.002 absolute on every mean/sd (transcription used
3-decimal rounding). SD convention: population SD (ddof=0) over the
three seeds — the convention the paper's transcribed tables used;
"SD across training seeds" here means the SD of the seeds actually
run, not an n-1 estimate of a seed population.

Raw logs live in the execution fork (maxrl/curriculum_maxrl/maze_gpu);
pass --logs to point elsewhere. Exit nonzero on mismatch.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOGS = os.path.join(HERE, "..", "..", "..", "maxrl",
                            "curriculum_maxrl", "maze_gpu")

ARMS = {
    "unif.": ["matched_uniform_maxrl_s0.jsonl",
              "matched_uniform_maxrl_s1.jsonl",
              "matched_uniform_maxrl_s2.jsonl"],
    "champion": ["matched_falp_maxrl_hsdense_s0.jsonl",
                 "matched_falp_maxrl_hsdense_s1.jsonl",
                 "matched_falp_maxrl_hsdense_s2.jsonl"],
}


def auc(path):
    rows = [json.loads(l) for l in open(path)]
    vals = [np.mean(list(r["eval"].values()))
            for r in rows if "eval" in r and r["step"] >= 0]
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=DEFAULT_LOGS)
    ap.add_argument("--tol", type=float, default=0.002)
    args = ap.parse_args()

    table = json.load(open(os.path.join(HERE, "data",
                                        "fig2_ladder_data.json")))["panel_c"]
    ok = True
    for i, label in enumerate(table["labels"]):
        files = ARMS[label]
        aucs = [auc(os.path.join(args.logs, f)) for f in files]
        m, s = float(np.mean(aucs)), float(np.std(aucs, ddof=0))
        tm, ts = table["auc"][i], table["sd"][i]
        line_ok = abs(m - tm) <= args.tol and abs(s - ts) <= args.tol
        ok &= line_ok
        print(f"{label:>10}: derived {m:.4f}±{s:.4f} vs table {tm}±{ts} "
              f"{'OK' if line_ok else 'MISMATCH'}")
    if not ok:
        sys.exit(1)
    print("fig2 panel (c) endpoints verified against raw seed logs")


if __name__ == "__main__":
    main()
