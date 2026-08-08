"""Decisive Experiment 1 (guidance doc P1.1): full-CV MaxRL vs practical.

The guidance's sharpest challenge to "recycling is the only channel into
the dead zone": the baseline's own Eq.(10) full control-variate estimator
emits a nonzero update on all-fail groups, E[g | K=0] = grad p / (1-p),
so a K=0 group is *not* structurally silent under that variant. Whether
that update can actually ignite an operationally dead pool -- or is too
noisy/unstable to matter -- is the empirical question this script answers
on the exact-gradient skill chains.

Arms (all MaxRL-family, uniform or teacher sampling as marked):
  practical            uniform sampling, drop-K=0 estimator (paper default)
  fullcv               uniform sampling, control variate kept at K=0
  practical+hindsight  paper default + relabel-to-achieved-prefix
  fullcv+hindsight     both channels
  teacher+practical    AdvMass teacher, drop-K=0
  teacher+fullcv       AdvMass teacher, full CV

Suites:
  balanced       levels 1..12 (the standard chain)
  frontier-heavy levels 5..12 (pool pass rate <= 1e-5: operationally dead)

Matched generation budget (total groups); metric: mean true pass rate on
the pool (exact, from env), plus time-to-first-live-group and the fraction
of update norm contributed by all-fail groups.

Usage: python3 run_fullcv_baseline.py [--seeds 5] [--groups 3200]
Writes results_fullcv_baseline.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from testbed import SkillChainEnv
from estimators import weights_maxrl
from teachers import AdvMassTeacher
from run_hindsight import correct_prefix_len

HERE = os.path.dirname(os.path.abspath(__file__))


def weights_maxrl_fullcv(r: np.ndarray) -> np.ndarray:
    """Eq.(10) with the control variate retained on all-fail groups:
    w_i = r_i/K - 1/N for K>=1;  w_i = -1/N when K=0."""
    n = len(r)
    k = r.sum()
    if k == 0:
        return np.full(n, -1.0 / n)
    return r / k - 1.0 / n


def run(arm: str, seed: int, total_groups: int, n_rollouts: int = 16,
        lr: float = 0.5, level_range=(1, 12), eval_every: int = 200):
    env = SkillChainEnv(seed=seed)
    levels = np.array(env.task_level)
    lo, hi = level_range
    pool = np.array([t for t in range(env.n_tasks) if lo <= levels[t] <= hi])
    chain_len = env.n_levels

    use_teacher = arm.startswith("teacher")
    use_fullcv = "fullcv" in arm
    use_hindsight = "hindsight" in arm
    wfun = weights_maxrl_fullcv if use_fullcv else weights_maxrl

    teacher = (AdvMassTeacher(len(pool), seed=seed + 1000,
                              n_rollouts=n_rollouts) if use_teacher else None)
    rng = np.random.default_rng(seed + 5)

    hist = []
    first_live = None
    norm_allfail = 0.0
    norm_total = 0.0
    for used in range(1, total_groups + 1):
        if use_teacher:
            i = int(teacher.sample_tasks(1)[0])
            t = int(pool[i])
        else:
            t = int(pool[rng.integers(len(pool))])
        actions, rewards = env.rollout(t, n_rollouts)
        if use_teacher:
            teacher.observe(i, rewards)
        k = rewards.sum()
        if first_live is None and 0 < k < n_rollouts:
            first_live = used
        w = wfun(rewards)
        if np.any(w != 0):
            # track how much update norm the all-fail channel contributes
            gnorm = float(np.abs(w).sum())
            norm_total += gnorm
            if k == 0:
                norm_allfail += gnorm
            env.apply_gradient(t, actions, w, lr)
        if k == 0 and use_hindsight:
            prefixes = np.array([correct_prefix_len(a) for a in actions])
            j = int(prefixes.max())
            if j >= 1:
                target = (t // chain_len) * chain_len + (j - 1)
                r2 = (prefixes >= j).astype(float)
                w2 = weights_maxrl(r2)
                if np.any(w2 != 0):
                    env.apply_gradient(target, actions[:, :j], w2, lr)
        if used % eval_every == 0:
            hist.append(env.true_pass_rates()[pool].mean())

    hist = np.array(hist)
    return {
        "final": float(hist[-1]),
        "auc": float(hist.mean()),
        "curve": hist.round(5).tolist(),
        "first_live_group": first_live,
        "allfail_norm_fraction":
            float(norm_allfail / norm_total) if norm_total else 0.0,
    }


ARMS = ["practical", "fullcv", "practical+hindsight", "fullcv+hindsight",
        "teacher+practical", "teacher+fullcv"]
SUITES = {"balanced": (1, 12), "frontier-heavy": (5, 12)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--groups", type=int, default=3200)
    args = ap.parse_args()

    results = {}
    for suite, level_range in SUITES.items():
        results[suite] = {}
        for arm in ARMS:
            per_seed = [run(arm, s, args.groups, level_range=level_range)
                        for s in range(args.seeds)]
            aucs = np.array([r["auc"] for r in per_seed])
            finals = np.array([r["final"] for r in per_seed])
            results[suite][arm] = {
                "auc_mean": float(aucs.mean()),
                "auc_sd": float(aucs.std(ddof=1)),
                "final_mean": float(finals.mean()),
                "final_sd": float(finals.std(ddof=1)),
                "per_seed": per_seed,
            }
            r0 = results[suite][arm]
            print(f"{suite:>14} {arm:<22} AUC {r0['auc_mean']:.3f}"
                  f"±{r0['auc_sd']:.3f}  final {r0['final_mean']:.3f}"
                  f"±{r0['final_sd']:.3f}  allfail-norm-frac "
                  f"{np.mean([p['allfail_norm_fraction'] for p in per_seed]):.3f}")

    out = os.path.join(HERE, "results_fullcv_baseline.json")
    with open(out, "w") as f:
        json.dump({"seeds": args.seeds, "groups": args.groups,
                   "results": results}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
