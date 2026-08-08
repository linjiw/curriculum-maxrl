"""Decisive Experiment 2 (guidance P1.2): schedule-matched estimator study.

The guidance's causal-identification complaint: the maze compares estimators
under teachers that ADAPT, so realized schedules differ and the estimator
effect is confounded with the teacher-feedback loop. The clean design fixes
the realized prompt sequence and varies only the estimator.

This script runs that design at the exact-gradient rung, where the coverage
metric itself is exact: for any task, pass@k = 1 - (1-p)^k with p read off
the policy, so the estimator-conditioned coverage claim is tested with zero
evaluation noise.

Design (paired by seed):
  1. SCHEDULE GENERATION: run the Thompson AdvMass teacher once per seed
     with the practical MaxRL estimator; record the realized task sequence.
     Also build a frozen uniform schedule per seed.
  2. REPLAY: for each frozen schedule, train fresh policies from the same
     init under {practical MaxRL, GRPO (sample SD, deployed), RLOO},
     feeding tasks in the recorded order. No teacher feedback anywhere.
  3. ENDPOINTS (exact, from env): mean pass@1; summed pass@8 coverage over
     the pool; coverage premium (pass@8 - pass@1 mean); all as final and
     delta-from-init.

The estimand is the guidance's estimator main effect under the SAME
realized data schedule -- the thing the maze suite could not identify.

Usage: python3 run_schedule_matched.py [--seeds 5] [--groups 2000]
Writes results_schedule_matched.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from testbed import SkillChainEnv
from estimators import (weights_maxrl, weights_grpo, weights_rloo,
                        weights_grpo_nostd)
from teachers import AdvMassTeacher

HERE = os.path.dirname(os.path.abspath(__file__))

ESTS = {"maxrl": weights_maxrl, "grpo": weights_grpo, "rloo": weights_rloo,
        "grpo_nostd": weights_grpo_nostd}


def coverage(env: SkillChainEnv, k: int) -> float:
    """Exact summed pass@k over the pool (guidance: coverage currency)."""
    p = env.true_pass_rates()
    return float((1.0 - (1.0 - p) ** k).mean())


def gen_schedule(seed: int, groups: int, n_rollouts: int):
    """Realized task sequence from a Thompson teacher driving practical
    MaxRL (the deployed loop), plus a uniform schedule of equal length."""
    env = SkillChainEnv(seed=seed)
    teacher = AdvMassTeacher(env.n_tasks, seed=seed + 1000,
                             n_rollouts=n_rollouts)
    sched = []
    for _ in range(groups):
        t = int(teacher.sample_tasks(1)[0])
        sched.append(t)
        actions, rewards = env.rollout(t, n_rollouts)
        teacher.observe(t, rewards)
        w = weights_maxrl(rewards)
        if np.any(w != 0):
            env.apply_gradient(t, actions, w, 0.5)
    rng = np.random.default_rng(seed + 77)
    uniform = rng.integers(0, env.n_tasks, size=groups).tolist()
    return sched, uniform


def replay(schedule, seed: int, est_name: str, n_rollouts: int,
           lr: float = 0.5, eval_every: int = 200):
    env = SkillChainEnv(seed=seed)  # same init as schedule generation
    wfun = ESTS[est_name]
    init = {"pass1": float(env.true_pass_rates().mean()),
            "cov8": coverage(env, 8)}
    hist = []
    for i, t in enumerate(schedule, 1):
        actions, rewards = env.rollout(int(t), n_rollouts)
        w = wfun(rewards)
        if np.any(w != 0):
            env.apply_gradient(int(t), actions, w, lr)
        if i % eval_every == 0:
            hist.append({"groups": i,
                         "pass1": float(env.true_pass_rates().mean()),
                         "cov8": coverage(env, 8)})
    final = {"pass1": float(env.true_pass_rates().mean()),
             "cov8": coverage(env, 8),
             "premium8": coverage(env, 8)
             - float(env.true_pass_rates().mean())}
    return {"init": init, "final": final,
            "delta_pass1": final["pass1"] - init["pass1"],
            "delta_cov8": final["cov8"] - init["cov8"],
            "curve": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--groups", type=int, default=2000)
    ap.add_argument("--rollouts", type=int, default=16)
    args = ap.parse_args()

    out = {"seeds": args.seeds, "groups": args.groups, "N": args.rollouts,
           "cells": {}}
    for sched_name in ("teacher", "uniform"):
        for est in ESTS:
            out["cells"][f"{sched_name}/{est}"] = []

    for seed in range(args.seeds):
        teach_sched, unif_sched = gen_schedule(seed, args.groups,
                                               args.rollouts)
        for sched_name, sched in (("teacher", teach_sched),
                                  ("uniform", unif_sched)):
            for est in ESTS:
                r = replay(sched, seed, est, args.rollouts)
                out["cells"][f"{sched_name}/{est}"].append(r)

    print(f"{'cell':>18}  {'d pass@1':>12}  {'d cov@8':>12}  (per-seed deltas)")
    for cell, runs in out["cells"].items():
        d1 = np.array([r["delta_pass1"] for r in runs])
        d8 = np.array([r["delta_cov8"] for r in runs])
        print(f"{cell:>18}  {d1.mean():+.3f}±{d1.std(ddof=1):.3f}  "
              f"{d8.mean():+.3f}±{d8.std(ddof=1):.3f}  "
              f"cov8: {[f'{x:+.3f}' for x in d8]}")

    # paired estimator contrasts under the SAME schedule (the estimand)
    out["paired_contrasts"] = {}
    for sched_name in ("teacher", "uniform"):
        m = [r["delta_cov8"] for r in out["cells"][f"{sched_name}/maxrl"]]
        g = [r["delta_cov8"] for r in out["cells"][f"{sched_name}/grpo"]]
        diffs = [a - b for a, b in zip(m, g)]
        out["paired_contrasts"][f"{sched_name}: maxrl-grpo cov8"] = {
            "per_seed": diffs,
            "mean": float(np.mean(diffs)),
            "all_positive": bool(all(d > 0 for d in diffs)),
        }
        print(f"paired maxrl-grpo Δcov@8 under {sched_name} schedule: "
              f"{[f'{d:+.3f}' for d in diffs]} "
              f"(all positive: {all(d > 0 for d in diffs)})")

    path = os.path.join(HERE, "results_schedule_matched.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
