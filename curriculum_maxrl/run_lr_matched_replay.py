"""Learning-rate sensitivity of the estimator coverage ordering
(reviewer Q4, measured: 'why compare mass across estimators without
matching update norm or learning rate?').

The paper's caveat (Prop 2 interp) says absolute cross-estimator mass
ratios are implementation facts under a common learning rate, and that
the empirical claims rest on tails and signs. This experiment tests
whether the schedule-matched coverage ordering (MaxRL over GRPO,
10/10) is itself an lr-calibration artifact: replay the same frozen
schedules under GRPO at a sweep of learning-rate multipliers spanning
the norm-matching range, and ask whether ANY global lr rescaling closes
the coverage gap to MaxRL.

Design (paired by seed; same frozen schedules as
run_schedule_matched.py):
  reference    maxrl @ lr=0.5 (the deployed setting)
  grpo-sweep   grpo @ lr in 0.5 * {1/4, 1/2, 1, 2, 4}
  rloo control rloo @ lr=0.5 (unnormalized profile, for placement)

Pre-registered 2026-08-06 before any run:
  P-LR1: no GRPO lr multiplier reaches MaxRL's delta-cov@8 on the
     teacher schedule in a majority of seeds — i.e., for every
     multiplier m, maxrl beats grpo(m) in >= 3/5 seeds — the coverage
     ordering is not an lr artifact.
  Falsification (committed): if some multiplier closes the gap
     (grpo(m) >= maxrl in >= 3/5 seeds), the paper's Q4 caveat is
     insufficient — the ordering must be restated as
     'at matched lr' and the normalization caveat upgraded from
     scope-note to limitation.

Usage: python3 run_lr_matched_replay.py [--seeds 5] [--groups 2000]
Writes results_lr_matched.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from run_schedule_matched import gen_schedule, replay

HERE = os.path.dirname(os.path.abspath(__file__))
MULTS = [0.25, 0.5, 1.0, 2.0, 4.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--groups", type=int, default=2000)
    ap.add_argument("--rollouts", type=int, default=16)
    args = ap.parse_args()

    out = {"groups": args.groups, "seeds": args.seeds, "mults": MULTS,
           "cells": {}}
    for seed in range(args.seeds):
        sched, _uniform = gen_schedule(seed, args.groups, args.rollouts)
        ref = replay(sched, seed, "maxrl", args.rollouts, lr=0.5)
        out["cells"][f"maxrl/s{seed}"] = ref
        out["cells"][f"rloo/s{seed}"] = replay(
            sched, seed, "rloo", args.rollouts, lr=0.5)
        for m in MULTS:
            out["cells"][f"grpo_x{m}/s{seed}"] = replay(
                sched, seed, "grpo", args.rollouts, lr=0.5 * m)

    print(f"{'arm':>12} {'d cov8':>16} {'d pass1':>16}")
    arms = ["maxrl", "rloo"] + [f"grpo_x{m}" for m in MULTS]
    for arm in arms:
        rs = [out["cells"][f"{arm}/s{s}"] for s in range(args.seeds)]
        print(f"{arm:>12} "
              f"{np.mean([r['delta_cov8'] for r in rs]):+.4f}±{np.std([r['delta_cov8'] for r in rs], ddof=1):.4f} "
              f"{np.mean([r['delta_pass1'] for r in rs]):+.4f}±{np.std([r['delta_pass1'] for r in rs], ddof=1):.4f}")

    # P-LR1: does any multiplier close the coverage gap?
    # (gap closed = grpo(m) >= maxrl in a strict majority of seeds)
    verdict = {}
    any_closed = False
    for m in MULTS:
        wins = sum(
            out["cells"][f"maxrl/s{s}"]["delta_cov8"]
            > out["cells"][f"grpo_x{m}/s{s}"]["delta_cov8"]
            for s in range(args.seeds))
        closed = wins <= args.seeds // 2
        any_closed |= closed
        verdict[f"x{m}"] = {"maxrl_wins": int(wins), "n": args.seeds,
                            "gap_closed": bool(closed)}
        print(f"P-LR1 grpo x{m}: maxrl wins {wins}/{args.seeds}"
              f"{'  << GAP CLOSED' if closed else ''}")
    out["P-LR1"] = {"per_mult": verdict, "any_mult_closes_gap": any_closed,
                    "verdict": "FALSIFIED (restate at matched lr)"
                    if any_closed else
                    "CONFIRMED (ordering is not an lr artifact)"}
    print("\nP-LR1:", out["P-LR1"]["verdict"])

    # Secondary (stated in advance, exploratory): the chain pool is
    # learnable-everywhere, so a global lr raise buys more of BOTH
    # currencies; the tradeoff read is the final coverage-reliability
    # premium (cov8 - pass1) — does any multiplier reach maxrl's
    # premium at comparable or better pass1?
    prem = {}
    for arm in arms:
        rs = [out["cells"][f"{arm}/s{s}"]["final"] for s in range(args.seeds)]
        prem[arm] = {"premium8": float(np.mean([r["premium8"] for r in rs])),
                     "pass1": float(np.mean([r["pass1"] for r in rs]))}
        print(f"premium {arm:>12}: {prem[arm]['premium8']:+.4f} "
              f"at pass1 {prem[arm]['pass1']:.4f}")
    out["premium_secondary"] = prem

    path = os.path.join(HERE, "results_lr_matched.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
