"""Gate-statistic comparison (draft-review 2026-08-04, P0-1 open item;
reviewer questions 1 and 8).

The deployed relabel gate thresholds a decayed achieved-destination
FREQUENCY over the recycler's own relabel stream. The derived object is
a threshold on the destination task's fresh-rollout PASS RATE p(g').
The review asks: what random variable does the heuristic track, and
does the derived gate beat it? On the skill-chain testbed both are
measurable exactly — true p(g') is free — so this runs the comparison
the LLM suites cannot.

Pre-registered 2026-08-05 before any run:
  Arms (paired by seed, identical env/teacher/schedule seeds):
    A  no recycling
    B  ungated recycling
    C  frequency-gated recycling      (deployed statistic: decayed
       Beta-recency over relabel destinations, reject p_hat > 0.5,
       decay 0.9 — the verl implementation transplanted)
    D  true-p gated recycling         (oracle: reject if the
       destination task's TRUE pass rate > 0.5 — the derived object)
  Endpoints:
    E1 final mean true pass rate + AUC (learning currency)
    E2 summed pass@8 coverage delta (coverage currency)
    E3 per-admission-decision agreement between C's and D's verdicts,
       and the correlation between C's p_hat and true p(g') at
       decision time (the "what does the statistic track" answer)
  Predictions:
    P-GV1: C and D agree on a majority of admission decisions late in
       training (both saturate on mastered destinations), but D admits
       MORE early (fresh destinations start at the Beta(1,1) prior 0.5
       under C only if never seen; under D anything below 0.5 true-p
       is admitted immediately).
    P-GV2: D >= C on E1 AUC in a majority of seeds (the derived object
       should not lose to its own proxy). If C > D consistently, the
       frequency statistic captures something true-p does not
       (recency of self-reinforcement), and the paper's "derived
       object" framing must be revised — committed falsification.

Usage: python3 run_gate_variants.py [--seeds 10] [--steps 400]
Writes results_gate_variants.json.
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


def correct_prefix_len(actions_row: np.ndarray) -> int:
    wrong = np.nonzero(actions_row != 0)[0]
    return int(wrong[0]) if len(wrong) else len(actions_row)


def coverage8(env: SkillChainEnv) -> float:
    p = env.true_pass_rates()
    return float((1.0 - (1.0 - p) ** 8).mean())


class FreqGate:
    """The deployed statistic: decayed Beta-recency over the relabel
    stream (verl hindsight.py transplanted; b-side never increments)."""

    def __init__(self, max_p=0.5, decay=0.9):
        self.max_p, self.decay = max_p, decay
        self.hits: dict = {}
        self.batch_hits: set = set()

    def p_hat(self, key):
        a, b = self.hits.get(key, (0.0, 0.0))
        return (a + 1.0) / (a + b + 2.0)

    def decide(self, key):
        """Record the observation, return admit?"""
        p = self.p_hat(key)
        a, b = self.hits.get(key, (0.0, 0.0))
        self.hits[key] = (a * self.decay + 1.0, b * self.decay)
        self.batch_hits.add(key)
        return p <= self.max_p

    def end_batch(self):
        for key in list(self.hits):
            if key in self.batch_hits:
                continue
            a, b = self.hits[key]
            self.hits[key] = (a * self.decay, b * self.decay)
            if a + b < 0.05:
                del self.hits[key]
        self.batch_hits.clear()


def run(seed: int, steps: int, gate: str, tasks_per_batch=8,
        n_rollouts=16, lr=0.5, record_decisions=False):
    env = SkillChainEnv(seed=seed)
    teacher = AdvMassTeacher(env.n_tasks, seed=seed + 1000,
                             n_rollouts=n_rollouts)
    level_of = env.task_level
    chain_of = [t // env.n_levels for t in range(env.n_tasks)]
    task_of = {(chain_of[t], level_of[t]): t for t in range(env.n_tasks)}

    fgate = FreqGate()
    decisions = []  # (step, dest, f_p_hat, true_p, freq_admit, truep_admit)
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
            if gate == "none":
                continue
            prefixes = np.array([correct_prefix_len(a) for a in actions])
            jstar = int(prefixes.max())
            if jstar < 1:
                continue
            target = task_of[(chain_of[t], jstar)]
            true_p = float(env.true_pass_rates()[target])
            f_p = fgate.p_hat(target)
            freq_admit = fgate.decide(target)   # records the observation
            truep_admit = true_p <= 0.5
            if record_decisions:
                decisions.append((step, target, round(f_p, 4),
                                  round(true_p, 4), freq_admit, truep_admit))
            admit = {"ungated": True, "freq": freq_admit,
                     "truep": truep_admit}[gate]
            if not admit:
                continue
            r2 = (prefixes >= jstar).astype(float)
            w2 = weights_maxrl(r2)
            if np.any(w2 != 0):
                env.apply_gradient(target, actions[:, :jstar], w2, lr)
        fgate.end_batch()
        auc_acc.append(env.true_pass_rates().mean())
    return {"final": float(env.true_pass_rates().mean()),
            "auc": float(np.mean(auc_acc)),
            "delta_cov8": coverage8(env) - cov0,
            "decisions": decisions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    arms = ["none", "ungated", "freq", "truep"]
    out = {"steps": args.steps, "seeds": args.seeds,
           "arms": {a: [] for a in arms}, "agreement": []}
    for seed in range(args.seeds):
        for arm in arms:
            r = run(seed, args.steps, arm,
                    record_decisions=(arm == "freq"))
            if arm == "freq" and r["decisions"]:
                d = r["decisions"]
                agree = np.mean([x[4] == x[5] for x in d])
                halves = np.array_split(np.array(d, dtype=object), 2)
                out["agreement"].append({
                    "seed": seed, "n_decisions": len(d),
                    "overall_agree": float(agree),
                    "early_agree": float(np.mean([x[4] == x[5]
                                                  for x in halves[0]])),
                    "late_agree": float(np.mean([x[4] == x[5]
                                                 for x in halves[1]])),
                    "freq_admit_rate": float(np.mean([x[4] for x in d])),
                    "truep_admit_rate": float(np.mean([x[5] for x in d])),
                    "corr_fhat_truep": float(np.corrcoef(
                        [x[2] for x in d], [x[3] for x in d])[0, 1])
                    if len(d) > 2 else None,
                })
            r.pop("decisions")
            out["arms"][arm].append(r)

    print(f"{'arm':>8} {'AUC':>16} {'final':>16} {'d cov8':>16}")
    for arm in arms:
        rs = out["arms"][arm]
        for k in ("auc", "final", "delta_cov8"):
            vals = np.array([r[k] for r in rs])
        line = " ".join(
            f"{np.mean([r[k] for r in rs]):+.4f}±{np.std([r[k] for r in rs], ddof=1):.4f}"
            for k in ("auc", "final", "delta_cov8"))
        print(f"{arm:>8} {line}")

    # P-GV2: paired truep - freq on AUC
    dif = [t["auc"] - f["auc"] for t, f in
           zip(out["arms"]["truep"], out["arms"]["freq"])]
    out["P-GV2 truep-freq auc"] = {
        "per_seed": dif, "n_pos": int(sum(d > 0 for d in dif)),
        "n": len(dif), "mean": float(np.mean(dif))}
    print(f"\nP-GV2 truep-freq AUC: {sum(d>0 for d in dif)}/{len(dif)} "
          f"positive, mean {np.mean(dif):+.5f}")
    ag = out["agreement"]
    if ag:
        print(f"P-GV1 admission agreement: overall "
              f"{np.mean([a['overall_agree'] for a in ag]):.3f}, early "
              f"{np.mean([a['early_agree'] for a in ag]):.3f}, late "
              f"{np.mean([a['late_agree'] for a in ag]):.3f}; "
              f"corr(f_hat, true p) "
              f"{np.mean([a['corr_fhat_truep'] for a in ag if a['corr_fhat_truep'] is not None]):.3f}")

    path = os.path.join(HERE, "results_gate_variants.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
