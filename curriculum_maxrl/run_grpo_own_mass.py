"""GRPO scheduled by its OWN mass functional, exact rung (reviewer Q6).

The maze factorial runs this control at neural scale (grpo_mass arm,
P-G0a/b). This is the exact-gradient version, where coverage is exact:
if GRPO under its own scheduler matches GRPO under the u_N teacher,
teacher-estimator mismatch does not explain the estimator gap; if it
closes the gap to MaxRL, the paper's framing must shift to "the
teacher-estimator PAIR decides".

Arms (live teachers, 5 seeds, 2000 groups, N=16):
  maxrl x u_N-teacher      (reference)
  grpo  x u_N-teacher      (the alleged mismatch cell)
  grpo  x grpo-mass teacher (its own functional:
                             u_G(p) = (1/N) E sqrt(K(N-K)), K~Bin(N,p),
                             Thompson-sampled, same floor)
  grpo  x uniform          (no teacher)

Pre-registered 2026-08-05: P-Q6: grpo x grpo-mass lands within seed
noise of grpo x u_N (mismatch is NOT the mechanism) and both stay far
below maxrl x u_N on delta-cov@8. Falsification: if grpo-mass closes
>= half the gap to maxrl, the estimator-conditioned framing must be
rescoped to the pair (committed).

Usage: python3 run_grpo_own_mass.py [--seeds 5] [--groups 2000]
Writes results_grpo_own_mass.json.
"""

from __future__ import annotations

import argparse
import json
import os
from math import comb

import numpy as np

from testbed import SkillChainEnv
from estimators import weights_maxrl, weights_grpo
from teachers import AdvMassTeacher, UniformTeacher, Teacher

HERE = os.path.dirname(os.path.abspath(__file__))


class GRPOMassTeacher(Teacher):
    """Thompson teacher over GRPO's exact finite-N expected mass
    (sample-SD convention drops the constant sqrt((N-1)/N), which does
    not affect the sampling distribution)."""

    def __init__(self, n_tasks, seed=0, n_rollouts=16, explore_frac=0.1):
        super().__init__(n_tasks, seed)
        self.n_rollouts = n_rollouts
        self.explore_frac = explore_frac

    def _mass(self, p: float) -> float:
        n = self.n_rollouts
        return sum(comb(n, k) * p ** k * (1 - p) ** (n - k)
                   * np.sqrt(k * (n - k)) / n for k in range(1, n))

    def distribution(self):
        w = np.zeros(self.n_tasks)
        for i, st in enumerate(self.stats):
            a, b = st.alpha_beta
            w[i] = self._mass(float(self.rng.beta(a, b)))
        if w.sum() <= 1e-12:
            w[:] = 1.0
        probs = w / w.sum()
        uniform = np.full(self.n_tasks, 1.0 / self.n_tasks)
        return (1 - self.explore_frac) * probs + self.explore_frac * uniform


def coverage8(env):
    p = env.true_pass_rates()
    return float((1.0 - (1.0 - p) ** 8).mean())


def run(seed, groups, est_name, teacher_name, n_rollouts=16, lr=0.5):
    env = SkillChainEnv(seed=seed)
    teachers = {
        "un": lambda: AdvMassTeacher(env.n_tasks, seed=seed + 1000,
                                     n_rollouts=n_rollouts),
        "grpomass": lambda: GRPOMassTeacher(env.n_tasks, seed=seed + 1000,
                                            n_rollouts=n_rollouts),
        "uniform": lambda: UniformTeacher(env.n_tasks, seed=seed + 1000),
    }
    teacher = teachers[teacher_name]()
    wfun = {"maxrl": weights_maxrl, "grpo": weights_grpo}[est_name]
    cov0 = coverage8(env)
    p1_acc = []
    for _ in range(groups):
        t = int(teacher.sample_tasks(1)[0])
        actions, rewards = env.rollout(t, n_rollouts)
        teacher.observe(t, rewards)
        w = wfun(rewards)
        if np.any(w != 0):
            env.apply_gradient(t, actions, w, lr)
        p1_acc.append(env.true_pass_rates().mean())
    return {"final_pass1": float(env.true_pass_rates().mean()),
            "auc": float(np.mean(p1_acc)),
            "delta_cov8": coverage8(env) - cov0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--groups", type=int, default=2000)
    args = ap.parse_args()

    arms = [("maxrl", "un"), ("grpo", "un"), ("grpo", "grpomass"),
            ("grpo", "uniform")]
    out = {"seeds": args.seeds, "groups": args.groups,
           "cells": {f"{e}/{t}": [] for e, t in arms}}
    for seed in range(args.seeds):
        for e, t in arms:
            out["cells"][f"{e}/{t}"].append(run(seed, args.groups, e, t))

    print(f"{'cell':>16} {'AUC':>16} {'d cov8':>16}")
    for cell, rs in out["cells"].items():
        line = " ".join(
            f"{np.mean([r[k] for r in rs]):+.4f}±{np.std([r[k] for r in rs], ddof=1):.4f}"
            for k in ("auc", "delta_cov8"))
        print(f"{cell:>16} {line}")

    # P-Q6 contrasts
    for name, a, b in [("grpomass-un (grpo)", "grpo/grpomass", "grpo/un"),
                       ("maxrl_un-grpo_grpomass", "maxrl/un",
                        "grpo/grpomass")]:
        dif = [x["delta_cov8"] - y["delta_cov8"]
               for x, y in zip(out["cells"][a], out["cells"][b])]
        out[name] = {"per_seed": dif,
                     "n_pos": int(sum(d > 0 for d in dif)),
                     "mean": float(np.mean(dif))}
        print(f"{name} d_cov8: {sum(d>0 for d in dif)}/{len(dif)} positive, "
              f"mean {np.mean(dif):+.5f}")

    path = os.path.join(HERE, "results_grpo_own_mass.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
