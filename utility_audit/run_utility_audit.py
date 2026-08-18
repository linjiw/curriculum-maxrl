"""Branch-and-continue utility audit. See UTILITY_AUDIT_PREREG.md.

Measures true continuation utility U_H(x) = J(Train_H(theta; x)) - J(theta) by
branching a deep copy of the exact-gradient skill-chain env and training H
steps on task x alone with the deployed practical MaxRL estimator. Compares how
well each candidate score, evaluated at the pre-branch state, RANKS U_H.

Deployed conventions match curriculum_maxrl/run_validation.py exactly:
weights_maxrl (drop K=0), N=16, lr=0.5.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "curriculum_maxrl"))
from testbed import SkillChainEnv                          # noqa: E402
from estimators import (weights_maxrl, weights_rloo,       # noqa: E402
                        coefficient_activity)

N = 16
LR = 0.5
H = 8
DEPTHS = (400, 800, 1600)
SEEDS = tuple(range(3001, 3011))
PASS_K = 8


# ------------------------------------------------------------------ pools
def make_structured(seed: int) -> SkillChainEnv:
    return SkillChainEnv(n_chains=3, n_levels=12, seed=seed)


def make_flat(seed: int) -> SkillChainEnv:
    """Same 36 tasks and same per-task difficulty distribution as the chain
    pool (task at chain level l needs l skills), but every task owns a
    private, disjoint skill block: no training step on x moves any other
    task's pass rate. Compounding C(x) == 1 by construction."""
    env = SkillChainEnv(n_chains=3, n_levels=12, seed=seed)
    total = sum(env.task_level)                # 3 * (1+2+...+12) = 234 skills
    env.n_skills = total
    env.theta = np.zeros((total, env.n_actions), dtype=np.float64)
    env.theta[:, 0] = env.init_logit_correct
    tasks, off = [], 0
    for lvl in env.task_level:
        tasks.append(np.arange(off, off + lvl)); off += lvl
    env.tasks = tasks
    return env


def compounding_count(env: SkillChainEnv, flat: bool) -> np.ndarray:
    """C(x): number of OTHER tasks whose pass rate rises when x's skills improve
    (structural, not learned). On the chain pool that is the harder tasks in
    x's chain; on the flat pool it is 0 for every task, so we use 1+count so
    A*C == A there (see prereg section 2)."""
    if flat:
        return np.ones(env.n_tasks)
    lv = np.array(env.task_level)
    return 1.0 + (env.n_levels - lv)          # 1 + (# harder tasks sharing skills)


# ------------------------------------------------------------------ objective
def J(env: SkillChainEnv) -> float:
    p = env.true_pass_rates()
    return float(np.mean(1.0 - (1.0 - p) ** PASS_K))


def one_step(env: SkillChainEnv, task: int, weight_fn) -> None:
    actions, rewards = env.rollout(task, N)
    w = weight_fn(rewards)
    if np.any(w != 0):
        env.apply_gradient(task, actions, w, LR)


def warm(env: SkillChainEnv, steps: int, weight_fn) -> None:
    """Shipped uniform sampler for `steps` groups."""
    for _ in range(steps):
        t = int(env.rng.integers(env.n_tasks))
        one_step(env, t, weight_fn)


def continuation_utility(env: SkillChainEnv, task: int, weight_fn,
                         branch_seed: int) -> float:
    b = copy.deepcopy(env)
    b.rng = np.random.default_rng(branch_seed)   # private stream per branch
    j0 = J(b)
    for _ in range(H):
        one_step(b, task, weight_fn)
    return J(b) - j0


# ------------------------------------------------------------------ stats
def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


# ------------------------------------------------------------------ audit
def audit(pool: str, seed: int, estimator: str) -> dict:
    flat = pool == "flat"
    wfn = {"maxrl": weights_maxrl, "rloo": weights_rloo}[estimator]
    make = make_flat if flat else make_structured
    per_depth = []
    for depth in DEPTHS:
        env = make(seed)
        warm(env, depth, wfn)
        p = env.true_pass_rates()
        C = compounding_count(env, flat)
        preds = {
            "p1mp": p * (1 - p),
            "uN":   coefficient_activity(p, N),
            "u64":  coefficient_activity(p, 64),
            "AC":   coefficient_activity(p, N) * C,
        }
        U = np.array([continuation_utility(env, x, wfn, seed * 1_000_003 + depth * 1_009 + x)
                      for x in range(env.n_tasks)])
        rho = {k: spearman(v, U) for k, v in preds.items()}
        per_depth.append({
            "depth": depth,
            "rho": rho,
            "U_argmax_p": float(p[int(np.argmax(U))]),
            "U_mean": float(U.mean()),
            "p_range": [float(p.min()), float(p.max())],
        })
    mean_rho = {k: float(np.mean([d["rho"][k] for d in per_depth])) for k in per_depth[0]["rho"]}
    return {"pool": pool, "seed": seed, "estimator": estimator,
            "per_depth": per_depth, "mean_rho": mean_rho}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--estimator", default="maxrl", choices=["maxrl", "rloo"])
    args = ap.parse_args(argv)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    for pool in ("structured", "flat"):
        for seed in SEEDS:
            f = out / f"{pool}-{args.estimator}-s{seed}.json"
            if f.exists():
                continue
            rec = audit(pool, seed, args.estimator)
            f.write_text(json.dumps(rec, indent=1))
            print(f"{pool:<10} seed={seed}  " +
                  "  ".join(f"{k}={v:+.3f}" for k, v in rec["mean_rho"].items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
