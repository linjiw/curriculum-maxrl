"""Exact transfer matrix: estimator activity vs cross-task transfer.

Guidance doc, 'What estimator activity cannot decide': the factorization

    E[g_i] = nu_N(p_i) (mu_i+ - mu_i-)          (activity x contrast)
    dJ_rho(i) ~ eta * grad J_rho . E[g_i]        (transfer/alignment)

says a curriculum needs BOTH a live estimator (nu > 0) and transfer.
On the skill-chain testbed every term is exact, so we can compute the
full level-by-level transfer matrix rather than argue by anecdote:

    T[i, j] = change in mean pass rate of level-j tasks after one exact
              expected practical-MaxRL update on a level-i task
            = eta * grad p_j . [ ((1-q_i^{N-1})/p_i) grad p_i ]

Because tasks are nested chains over shared skills, grad p_i overlaps
grad p_j exactly on the shared prefix; across chains the overlap is zero.
The script reports, per training snapshot:
  - the exact activity profile nu_N(p_i) by level;
  - the exact transfer matrix T (level x level, within-chain);
  - their product: which level actually buys the most held-out improvement,
    and how far the activity-only ranking is from the true ranking.

Usage: python3 run_transfer_matrix.py [--snapshots 0 400 1600]
Writes results_transfer_matrix.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from testbed import SkillChainEnv
from estimators import weights_maxrl
from teachers import AdvMassTeacher

HERE = os.path.dirname(os.path.abspath(__file__))


def nu(p, N):
    q = 1.0 - p
    return q - q**N


def exact_expected_update(env: SkillChainEnv, task_id: int, N: int):
    """E[g_prac] over the FULL theta, = ((1-q^{N-1})/p) * grad p (exact).

    grad p over theta[s,a] for s in the task's required skills:
      dp/dtheta[s,a] = p * (1{a=0} - probs[s,a]).
    """
    req = env.tasks[task_id]
    probs = env.skill_probs()[req]
    p = float(env.true_pass_rates()[task_id])
    if p <= 0.0 or p >= 1.0:
        return None, p
    onehot0 = np.zeros_like(probs)
    onehot0[:, 0] = 1.0
    grad_p = p * (onehot0 - probs)                     # (L, A)
    w = (1.0 - (1.0 - p) ** (N - 1)) / p               # T=N-1 weight
    g = np.zeros_like(env.theta)
    g[req] = w * grad_p
    return g, p


def transfer_snapshot(env: SkillChainEnv, N: int, eta: float):
    """Exact activity, transfer matrix, and value ranking at this policy."""
    levels = np.array(env.task_level)
    n_lev = env.n_levels
    p_all = env.true_pass_rates()

    # restrict to chain 0 for the matrix (chains are symmetric at init and
    # near-symmetric later; cross-chain transfer is structurally zero)
    chain0 = [t for t in range(env.n_tasks) if t // n_lev == 0]

    theta0 = env.theta.copy()
    T = np.zeros((n_lev, n_lev))
    activity = np.zeros(n_lev)
    total_value = np.zeros(n_lev)     # mean delta p over the whole pool
    for t in chain0:
        i = levels[t] - 1
        g, p_i = exact_expected_update(env, t, N)
        activity[i] = nu(p_i, N)
        if g is None:
            continue
        env.theta = theta0 + eta * g
        dp = env.true_pass_rates() - p_all
        for j in range(n_lev):
            lev_tasks = [u for u in chain0 if levels[u] - 1 == j]
            T[i, j] = float(np.mean(dp[lev_tasks]))
        # E[g] is the unconditional expectation (the closed form already
        # accounts for dead groups), so dp.mean() IS the exact one-step
        # expected pool improvement of allocating one group to this task
        total_value[i] = float(dp.mean())
        env.theta = theta0.copy()

    return {
        "p_by_level": [float(np.mean(p_all[levels == l + 1]))
                       for l in range(n_lev)],
        "activity_nu": activity.round(6).tolist(),
        "transfer_matrix": T.round(8).tolist(),
        "pool_value_one_group": total_value.round(8).tolist(),
        "activity_rank": np.argsort(-activity).tolist(),
        "value_rank": np.argsort(-total_value).tolist(),
        "spearman_activity_vs_value": _spearman(activity, total_value),
    }


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ca = ra - ra.mean()
    cb = rb - rb.mean()
    d = np.sqrt((ca**2).sum() * (cb**2).sum())
    return float((ca * cb).sum() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", type=int, nargs="+",
                    default=[0, 400, 1600])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--eta", type=float, default=0.5)
    args = ap.parse_args()

    env = SkillChainEnv(seed=args.seed)
    teacher = AdvMassTeacher(env.n_tasks, seed=args.seed + 1000,
                             n_rollouts=args.rollouts)
    out = {"seed": args.seed, "N": args.rollouts, "eta": args.eta,
           "snapshots": {}}

    trained = 0
    for snap in sorted(args.snapshots):
        while trained < snap:
            t = int(teacher.sample_tasks(1)[0])
            actions, rewards = env.rollout(t, args.rollouts)
            teacher.observe(t, rewards)
            w = weights_maxrl(rewards)
            if np.any(w != 0):
                env.apply_gradient(t, actions, w, 0.5)
            trained += 1
        snap_res = transfer_snapshot(env, args.rollouts, args.eta)
        out["snapshots"][str(snap)] = snap_res
        act = np.array(snap_res["activity_nu"])
        val = np.array(snap_res["pool_value_one_group"])
        print(f"groups={snap:5d}  p by level: "
              f"{[round(x, 3) for x in snap_res['p_by_level'][:8]]}")
        print(f"             nu by level: {[round(x, 3) for x in act[:8]]}")
        print(f"   pool value per update: {[round(x, 5) for x in val[:8]]}")
        print(f"   spearman(activity, pool value) = "
              f"{snap_res['spearman_activity_vs_value']:.3f}")

    path = os.path.join(HERE, "results_transfer_matrix.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
