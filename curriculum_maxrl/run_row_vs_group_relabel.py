"""Per-row vs one-target-per-group relabeling (draft-review 2026-08-04
P0-2 / reviewer Q2, measured where measurement is exact).

The deployed LLM recycler relabels each parseable failure to its OWN
achieved goal inside one advantage group (mixed-target; characterized
in the paper as a weighted-SFT update). The paper's Remark licenses the
single-destination form. This experiment measures the gap on the
skill-chain testbed, where both are implementable exactly and the
learning/coverage currencies are exact.

Arms (paired by seed; identical teacher schedule via shared env/teacher
seeds; recycling on dead groups only):
  A group: one destination per dead group = the group's DEEPEST achieved
     prefix task j*; rollouts reaching j* are successes, others failures
     (K-of-N contrast of a single task — the existing run_hindsight.py
     pattern and the CPU-reference semantics).
  B row: each rollout with prefix >= min_depth relabels to its own
     achieved prefix task, trained as its own K=1 group at weight
     (1 - 1/N) — the deployed per-row semantics transplanted (each row
     a verified success of its own task; no cross-task baseline
     coupling because each row forms its own group).
  C row-shared: per-row destinations but ONE shared group contrast over
     the N rollouts (w = r2/K - 1/N with r2 = "achieved anything deep
     enough"), the closest CPU analogue of the deployed shared-K
     coupling — the update the paper now calls weighted-SFT.

Pre-registered 2026-08-05 (before any run):
  P-RG1: A >= C on final AUC in a majority of 10 seeds (the coupled
     shared-K variant pays for coupling unrelated destinations).
  P-RG2: B sits between A and C or above both (per-row without
     coupling keeps exactness per row and adds dose; its risk is
     distribution shift toward shallow prefixes, which chains punish
     less than LLM pools might).
  Falsification: if C >= A consistently, the paper's "mixed groups are
     the deviation to fix" framing loses its empirical support at this
     rung and must rest on the LLM ablation alone (committed).

Usage: python3 run_row_vs_group_relabel.py [--seeds 10] [--steps 400]
Writes results_row_vs_group.json.
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
MIN_DEPTH = 1


def correct_prefix_len(actions_row: np.ndarray) -> int:
    wrong = np.nonzero(actions_row != 0)[0]
    return int(wrong[0]) if len(wrong) else len(actions_row)


def coverage8(env) -> float:
    p = env.true_pass_rates()
    return float((1.0 - (1.0 - p) ** 8).mean())


def run(seed: int, steps: int, mode: str, tasks_per_batch=8,
        n_rollouts=16, lr=0.5):
    env = SkillChainEnv(seed=seed)
    teacher = AdvMassTeacher(env.n_tasks, seed=seed + 1000,
                             n_rollouts=n_rollouts)
    level_of = env.task_level
    chain_of = [t // env.n_levels for t in range(env.n_tasks)]
    task_of = {(chain_of[t], level_of[t]): t for t in range(env.n_tasks)}

    auc_acc, cov0 = [], coverage8(env)
    for step in range(steps):
        for t in teacher.sample_tasks(tasks_per_batch):
            t = int(t)
            actions, rewards = env.rollout(t, n_rollouts)
            teacher.observe(t, rewards)
            w = weights_maxrl(rewards)
            if np.any(w != 0):
                env.apply_gradient(t, actions, w, lr)
                continue
            if mode == "none":
                continue
            prefixes = np.array([correct_prefix_len(a) for a in actions])
            chain = chain_of[t]
            if mode == "group":
                jstar = int(prefixes.max())
                if jstar < MIN_DEPTH:
                    continue
                target = task_of[(chain, jstar)]
                r2 = (prefixes >= jstar).astype(float)
                w2 = weights_maxrl(r2)
                if np.any(w2 != 0):
                    env.apply_gradient(target, actions[:, :jstar], w2, lr)
            elif mode == "row":
                # each deep-enough rollout: its own K=1 group at 1 - 1/N
                w_row = 1.0 - 1.0 / n_rollouts
                for j in range(n_rollouts):
                    d = int(prefixes[j])
                    if d < MIN_DEPTH:
                        continue
                    target = task_of[(chain, d)]
                    env.apply_gradient(target, actions[j:j + 1, :d],
                                       np.array([w_row]), lr)
            elif mode == "row_shared":
                # per-row destinations, ONE shared group contrast
                # (deployed shared-K semantics): r2_j = achieved anything
                r2 = (prefixes >= MIN_DEPTH).astype(float)
                w2 = weights_maxrl(r2)
                if not np.any(w2 != 0):
                    continue
                for j in range(n_rollouts):
                    d = int(prefixes[j])
                    if w2[j] == 0.0:
                        continue
                    if d < MIN_DEPTH:
                        # failure row of the shared group: apply its
                        # negative weight on the requested task's prefix
                        # conditioning (closest analogue of the -1/N
                        # push-down on malformed outputs)
                        env.apply_gradient(t, actions[j:j + 1],
                                           np.array([w2[j]]), lr)
                    else:
                        target = task_of[(chain, d)]
                        env.apply_gradient(target, actions[j:j + 1, :d],
                                           np.array([w2[j]]), lr)
        auc_acc.append(env.true_pass_rates().mean())
    return {"final": float(env.true_pass_rates().mean()),
            "auc": float(np.mean(auc_acc)),
            "delta_cov8": coverage8(env) - cov0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    arms = ["none", "group", "row", "row_shared"]
    out = {"steps": args.steps, "seeds": args.seeds,
           "arms": {a: [] for a in arms}}
    for seed in range(args.seeds):
        for arm in arms:
            out["arms"][arm].append(run(seed, args.steps, arm))

    print(f"{'arm':>11} {'AUC':>16} {'final':>16} {'d cov8':>16}")
    for arm in arms:
        rs = out["arms"][arm]
        line = " ".join(
            f"{np.mean([r[k] for r in rs]):+.4f}±{np.std([r[k] for r in rs], ddof=1):.4f}"
            for k in ("auc", "final", "delta_cov8"))
        print(f"{arm:>11} {line}")

    for name, a, b in [("P-RG1 group-rowshared", "group", "row_shared"),
                       ("row-rowshared", "row", "row_shared"),
                       ("group-row", "group", "row")]:
        dif = [x["auc"] - y["auc"] for x, y in
               zip(out["arms"][a], out["arms"][b])]
        out[name] = {"per_seed": dif,
                     "n_pos": int(sum(d > 0 for d in dif)),
                     "n": len(dif), "mean": float(np.mean(dif))}
        print(f"{name} AUC: {sum(d>0 for d in dif)}/{len(dif)} positive, "
              f"mean {np.mean(dif):+.5f}")

    path = os.path.join(HERE, "results_row_vs_group.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
