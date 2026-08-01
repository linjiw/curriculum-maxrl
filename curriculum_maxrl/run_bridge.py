"""Bridge experiment: what does the advantage-mass utility miss?

Opus5 review M5 established that advantage mass u_N(p) is not learning
progress: Thompson collects 99.6% of the oracle's mass yet loses 0.15 AUC,
and a variance-tilted utility (1-p)^2 u_N beats mass 10/10 seeds. This
script measures the gap and derives what fills it.

On the skill-chain testbed everything is exact:

  p_i = prod_{s in req_i} q_s          (q_s = softmax(theta_s)[0])
  d p_i / d theta_{s,a} = p_i (1{a=0} - q_{s,a})   for s in req_i
  <grad p_i, grad p_m> = p_i p_m sum_{s in req_i cap req_m} c_s,
      c_s = (1-q_{s0})^2 + sum_{a!=0} q_{sa}^2     ("unmastered-ness")

The practical MaxRL estimator's expected per-group gradient on task i is
g(p_i) grad p_i with g(p) = sum_{k=1}^{N-1} (1-p)^{k-1} = (1-(1-p)^{N-1})/p
(T = N-1; PROOFS.md Prop 1 correction). So the exact FIRST-ORDER expected
eval improvement from one group on task i is

  LP(i) = lr * g(p_i) * <grad p_i, grad J>,   J = mean_m p_m
        = lr * pass@{N-1}(p_i) * (1/M) sum_m p_m C_im,
          C_im = sum_{s shared} c_s.

Part A: at snapshots along a uniform-training trajectory, compare the
orderings of u_N(p), (1-p)^2 u_N(p), closed-form LP, against Monte-Carlo
ground truth E[delta eval | task] (many independent group draws, apply
update, exact eval delta, restore).

Part B: oracle teachers sampling prop-to each utility^gamma at matched
group budget, multi-seed. Isolates the utility question from posterior
estimation (all use true p; LP additionally uses true overlap/c — it is a
theory probe, not a practical method).

Artifact: results_bridge.json.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats as sstats

from testbed import SkillChainEnv
from estimators import weights_maxrl

N_ROLLOUTS = 16
LR = 0.5


# ---------------------------------------------------------------- utilities
def mass_u(p: np.ndarray, n: int = N_ROLLOUTS) -> np.ndarray:
    return (1.0 - (1.0 - p) ** n) - p


def var_tilted(p: np.ndarray, n: int = N_ROLLOUTS) -> np.ndarray:
    return (1.0 - p) ** 2 * mass_u(p, n)


def g_weight(p: np.ndarray, n: int = N_ROLLOUTS) -> np.ndarray:
    """Expected-gradient prefactor of the practical (T=N-1) estimator."""
    out = np.full_like(p, float(n - 1))  # limit as p -> 0
    nz = p > 1e-12
    out[nz] = (1.0 - (1.0 - p[nz]) ** (n - 1)) / p[nz]
    return out


def closed_form_lp(env: SkillChainEnv) -> np.ndarray:
    """Exact first-order E[delta J] per group for every task (up to lr)."""
    probs = env.skill_probs()
    q0 = probs[:, 0]
    c = (1 - q0) ** 2 + (probs[:, 1:] ** 2).sum(axis=1)
    p = env.true_pass_rates()
    M = env.n_tasks
    # prefix sums of c per chain make C_im = cpre[min(l_i, l_m)] on the same
    # chain, 0 across chains
    lp = np.zeros(M)
    lvl = np.array(env.task_level)
    chain = np.array([env.tasks[i][0] // env.n_levels for i in range(M)])
    for ch in np.unique(chain):
        idx = np.where(chain == ch)[0]
        base = ch * env.n_levels
        cpre = np.concatenate([[0.0], np.cumsum(c[base:base + env.n_levels])])
        li = lvl[idx]
        # C matrix within the chain: cpre[min(l_i, l_m)]
        Cmat = cpre[np.minimum.outer(li, li)]
        lp[idx] = g_weight(p[idx]) * p[idx] * (Cmat * p[idx][None, :]).sum(axis=1) / M
    return LR * lp


# ------------------------------------------------------- ground-truth probe
def mc_ground_truth(env: SkillChainEnv, task_id: int, n_draws: int,
                    rng: np.random.Generator) -> tuple[float, float]:
    """E[delta eval] from one MaxRL group on task_id (mean, sem)."""
    theta0 = env.theta.copy()
    base = env.true_pass_rates().mean()
    deltas = np.empty(n_draws)
    for d in range(n_draws):
        actions, rewards = env.rollout(task_id, N_ROLLOUTS)
        w = weights_maxrl(rewards)
        if np.any(w != 0):
            env.apply_gradient(task_id, actions, w, LR)
            deltas[d] = env.true_pass_rates().mean() - base
            env.theta[:] = theta0
        else:
            deltas[d] = 0.0
    return float(deltas.mean()), float(deltas.std(ddof=1) / np.sqrt(n_draws))


def snapshot_states(seed: int, checkpoints: list[int]) -> list[np.ndarray]:
    """theta snapshots along a uniform-sampling training run."""
    env = SkillChainEnv(seed=seed)
    rng = np.random.default_rng(seed + 5)
    out, used = [], 0
    for target in checkpoints:
        while used < target:
            t = int(rng.integers(env.n_tasks))
            actions, rewards = env.rollout(t, N_ROLLOUTS)
            used += 1
            w = weights_maxrl(rewards)
            if np.any(w != 0):
                env.apply_gradient(t, actions, w, LR)
        out.append(env.theta.copy())
    return out


def part_a(seed: int = 0, n_draws: int = 400) -> dict:
    checkpoints = [0, 200, 400, 800, 1600, 3200]
    thetas = snapshot_states(seed, checkpoints)
    results = []
    for ck, theta in zip(checkpoints, thetas):
        env = SkillChainEnv(seed=seed)
        env.theta = theta.copy()
        env.rng = np.random.default_rng(seed + 1000 + ck)
        p = env.true_pass_rates()
        cand = {
            "mass": mass_u(p),
            "var_tilted": var_tilted(p),
            "lp_closed": closed_form_lp(env),
        }
        rng = np.random.default_rng(seed + 2000 + ck)
        gt = np.empty(env.n_tasks)
        gt_sem = np.empty(env.n_tasks)
        for i in range(env.n_tasks):
            gt[i], gt_sem[i] = mc_ground_truth(env, i, n_draws, rng)
        row = {"checkpoint": ck,
               "eval": float(p.mean()),
               "gt_delta_eval": gt.tolist(),
               "gt_sem": gt_sem.tolist(),
               "pass_rates": p.tolist()}
        for name, u in cand.items():
            rho, _ = sstats.spearmanr(u, gt)
            # correlation restricted to tasks with any signal (u or gt > 0)
            live = (gt > 2 * gt_sem) | (u > 1e-9)
            rho_live, _ = (sstats.spearmanr(u[live], gt[live])
                           if live.sum() > 2 else (float("nan"), None))
            row[name] = {"spearman_all": float(rho),
                         "spearman_live": float(rho_live),
                         "utility": u.tolist(),
                         "n_live": int(live.sum())}
        # closed-form LP vs ground truth *magnitude* (first-order check):
        lp = cand["lp_closed"]
        nz = gt_sem > 0
        row["lp_vs_gt_slope"] = float(
            np.polyfit(lp[nz], gt[nz], 1)[0]) if nz.sum() > 2 else None
        results.append(row)
        print(f"[A] ck={ck:5d} eval={p.mean():.3f} "
              + " ".join(f"{k}:rho={row[k]['spearman_all']:.3f}"
                         for k in cand), flush=True)
    return {"seed": seed, "n_draws": n_draws, "snapshots": results}


# ------------------------------------------------------------ part B: teachers
def run_oracle_teacher(utility: str, seed: int, gamma: float,
                       total_groups: int = 3200,
                       eval_every: int = 100) -> np.ndarray:
    env = SkillChainEnv(seed=seed)
    rng = np.random.default_rng(seed + 7)
    hist, used = [], 0
    while used < total_groups:
        p = env.true_pass_rates()
        if utility == "uniform":
            w = np.ones(env.n_tasks)
        elif utility == "mass":
            w = mass_u(p)
        elif utility == "var_tilted":
            w = var_tilted(p)
        elif utility == "lp_closed":
            w = closed_form_lp(env)
        else:
            raise ValueError(utility)
        w = np.maximum(w, 0.0) ** gamma
        if w.sum() <= 1e-12:
            w[:] = 1.0
        probs = 0.9 * w / w.sum() + 0.1 / env.n_tasks  # same floor as deployed
        t = int(rng.choice(env.n_tasks, p=probs))
        actions, rewards = env.rollout(t, N_ROLLOUTS)
        used += 1
        wts = weights_maxrl(rewards)
        if np.any(wts != 0):
            env.apply_gradient(t, actions, wts, LR)
        while len(hist) < used // eval_every:
            hist.append(env.true_pass_rates().mean())
    hist.append(env.true_pass_rates().mean())
    return np.array(hist)


def part_b(n_seeds: int = 10) -> dict:
    arms = [("uniform", 1.0), ("mass", 1.0), ("mass", 4.0),
            ("var_tilted", 1.0), ("var_tilted", 4.0),
            ("lp_closed", 1.0), ("lp_closed", 4.0)]
    out = {}
    for utility, gamma in arms:
        key = f"{utility}_g{gamma:g}"
        aucs, finals, curves = [], [], []
        for seed in range(n_seeds):
            h = run_oracle_teacher(utility, seed, gamma)
            aucs.append(float(h.mean()))
            finals.append(float(h[-1]))
            curves.append(h.tolist())
        out[key] = {"auc_mean": float(np.mean(aucs)),
                    "auc_std": float(np.std(aucs)),
                    "final_mean": float(np.mean(finals)),
                    "auc_per_seed": aucs}
        print(f"[B] {key:18s} AUC={np.mean(aucs):.4f}(±{np.std(aucs):.4f}) "
              f"final={np.mean(finals):.4f}", flush=True)
    # paired contrasts vs mass at same gamma
    for gamma in ("1", "4"):
        base = np.array(out[f"mass_g{gamma}"]["auc_per_seed"])
        for utility in ("var_tilted", "lp_closed"):
            arm = np.array(out[f"{utility}_g{gamma}"]["auc_per_seed"])
            d = arm - base
            t, pv = sstats.ttest_rel(arm, base)
            out[f"contrast_{utility}_vs_mass_g{gamma}"] = {
                "delta_mean": float(d.mean()), "wins": int((d > 0).sum()),
                "n": len(d), "t": float(t), "p": float(pv)}
            print(f"[B] {utility} vs mass (g={gamma}): d={d.mean():+.4f} "
                  f"wins={(d>0).sum()}/{len(d)} p={pv:.2g}", flush=True)
    return out


def main():
    out = {"protocol": {"n_rollouts": N_ROLLOUTS, "lr": LR,
                        "estimator": "weights_maxrl (T=N-1)",
                        "eval": "mean true pass rate over all 36 tasks"},
           "part_a": part_a(),
           "part_b": part_b()}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results_bridge.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
