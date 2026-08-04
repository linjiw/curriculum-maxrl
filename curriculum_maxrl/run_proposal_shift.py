"""Destination-law shift diagnostic + cross-fitted relabel selection.

Operationalizes the guidance doc's relabeled-group factorization theorem
(hindsight = a destination semi-gradient under the source-induced proposal
law Q^{x->g}, equal to the fresh destination update only under law
matching) on the exact-gradient skill chains, where every quantity the
theorem names is computable:

  p_Pi     true fresh pass rate of the destination task (exact, from env)
  p_Q      success rate of the relabeled group (empirical proposal rate)
  cos      cosine between the applied relabel update and the exact fresh
           destination gradient direction grad p (the expected practical
           update is a positive scalar times grad p, so direction alignment
           against grad p is the right alignment probe)

Two structural gaps the guidance predicts, measured:
  1. Same-group destination selection (j* = deepest prefix achieved by the
     group being updated) guarantees K' >= 1, so p_Q is biased upward
     relative to fresh sampling at p_Pi (a fresh group is all-fail with
     probability (1-p_Pi)^N > 0).
  2. Conditioning on the source group failing (K_x = 0) tilts the prefix
     coordinates away from the unconditional law.

Cross-fitting (guidance: split selection from update) removes the adaptive
reuse: j* is chosen on one half of the group, the update applied with the
other half's rollouts relabeled at that fixed destination. We run both and
compare end performance and the bias stats.

Usage: python3 run_proposal_shift.py [--seeds 5] [--steps 400]
Writes results_proposal_shift.json.
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


def grad_p_exact(env: SkillChainEnv, task_id: int) -> np.ndarray:
    """Exact gradient of the destination pass rate w.r.t. its skill logits,
    flattened over (required skills, actions):
    dp/dtheta[s,a] = p * (1{a=0} - probs[s,a])."""
    req = env.tasks[task_id]
    probs = env.skill_probs()[req]                # (L, A)
    p = env.true_pass_rates()[task_id]
    onehot0 = np.zeros_like(probs)
    onehot0[:, 0] = 1.0
    return (p * (onehot0 - probs)).ravel()


def run(seed: int, steps: int, cross_fit: bool, n_rollouts: int = 16,
        tasks_per_batch: int = 8, lr: float = 0.5):
    env = SkillChainEnv(seed=seed)
    teacher = AdvMassTeacher(env.n_tasks, seed=seed + 1000,
                             n_rollouts=n_rollouts)
    level_of = env.task_level
    chain_of = [t // env.n_levels for t in range(env.n_tasks)]
    task_of = {(chain_of[t], level_of[t]): t for t in range(env.n_tasks)}

    diag = []          # one row per relabel event
    history = []
    relabeled = 0
    for step in range(steps):
        for t in teacher.sample_tasks(tasks_per_batch):
            t = int(t)
            actions, rewards = env.rollout(t, n_rollouts)
            teacher.observe(t, rewards)
            w = weights_maxrl(rewards)
            if np.any(w != 0):
                env.apply_gradient(t, actions, w, lr)
                continue
            # ---- dead group: relabel ----
            if cross_fit:
                half = n_rollouts // 2
                sel, upd = actions[:half], actions[half:]
                jstar = int(max(correct_prefix_len(a) for a in sel))
                if jstar < 1:
                    continue
                prefixes_upd = np.array([correct_prefix_len(a) for a in upd])
                r2 = (prefixes_upd >= jstar).astype(float)
                a2 = upd[:, :jstar]
            else:
                prefixes = np.array([correct_prefix_len(a) for a in actions])
                jstar = int(prefixes.max())
                if jstar < 1:
                    continue
                r2 = (prefixes >= jstar).astype(float)
                a2 = actions[:, :jstar]
            target = task_of[(chain_of[t], jstar)]
            w2 = weights_maxrl(r2)
            if not np.any(w2 != 0):
                # cross-fitting allows all-fail AND all-pass relabel groups;
                # record the silent event (the practical estimator drops it)
                diag.append({"step": step, "dest_level": jstar,
                             "p_Q": float(r2.mean()),
                             "p_Pi": float(env.true_pass_rates()[target]),
                             "K_prime": int(r2.sum()), "cos": None,
                             "silent": True})
                continue
            # exact fresh-destination gradient direction, before the update
            g_fresh = grad_p_exact(env, target)
            # applied relabel gradient over the same coordinates
            req = env.tasks[target]
            probs = env.skill_probs()[req]
            n2 = a2.shape[0]
            onehot = np.zeros((n2, len(req), env.n_actions))
            onehot[np.arange(n2)[:, None], np.arange(len(req))[None, :],
                   a2] = 1.0
            score = onehot - probs[None]
            g_rel = np.einsum("j,jla->la", w2, score).ravel()
            denom = np.linalg.norm(g_rel) * np.linalg.norm(g_fresh)
            cos = float(g_rel @ g_fresh / denom) if denom > 0 else 0.0
            diag.append({"step": step, "dest_level": jstar,
                         "p_Q": float(r2.mean()),
                         "p_Pi": float(env.true_pass_rates()[target]),
                         "K_prime": int(r2.sum()), "cos": cos,
                         "silent": False})
            env.apply_gradient(target, a2, w2, lr)
            relabeled += 1
        if step % 10 == 0 or step == steps - 1:
            history.append({"step": step,
                            "mean_pass": float(env.true_pass_rates().mean())})

    mp = np.array([h["mean_pass"] for h in history])
    st = np.array([h["step"] for h in history])
    live = [d for d in diag if not d["silent"]]
    return {
        "final": float(mp[-1]),
        "auc": float(np.trapezoid(mp, st) / (st[-1] - st[0])),
        "relabeled_groups": relabeled,
        "silent_relabel_groups": sum(d["silent"] for d in diag),
        "mean_p_Q": float(np.mean([d["p_Q"] for d in live])) if live else None,
        "mean_p_Pi": float(np.mean([d["p_Pi"] for d in live])) if live else None,
        "mean_gap_pQ_minus_pPi":
            float(np.mean([d["p_Q"] - d["p_Pi"] for d in live])) if live else None,
        "mean_cos": float(np.mean([d["cos"] for d in live])) if live else None,
        "frac_cos_below_0.5":
            float(np.mean([d["cos"] < 0.5 for d in live])) if live else None,
        "diag_sample": live[:: max(1, len(live) // 200)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    out = {"steps": args.steps, "seeds": args.seeds, "arms": {}}
    for cross_fit in (False, True):
        name = "cross_fitted" if cross_fit else "same_group"
        per_seed = [run(s, args.steps, cross_fit) for s in range(args.seeds)]
        aucs = np.array([r["auc"] for r in per_seed])
        out["arms"][name] = {
            "auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std(ddof=1)),
            "final_mean": float(np.mean([r["final"] for r in per_seed])),
            "per_seed": per_seed,
        }
        r = out["arms"][name]
        print(f"{name:>13}: AUC {r['auc_mean']:.3f}±{r['auc_sd']:.3f}  "
              f"final {r['final_mean']:.3f}  "
              f"p_Q-p_Pi {np.mean([s['mean_gap_pQ_minus_pPi'] for s in per_seed]):+.3f}  "
              f"cos {np.mean([s['mean_cos'] for s in per_seed]):.3f}  "
              f"silent/seed {np.mean([s['silent_relabel_groups'] for s in per_seed]):.0f}")

    path = os.path.join(HERE, "results_proposal_shift.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
