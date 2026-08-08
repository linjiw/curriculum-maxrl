"""Practical destination-pass-rate gate: probe-budget dose response
(follow-up to run_gate_variants.py; makes 6.8's "indicated upgrade"
concrete).

run_gate_variants.py showed the ORACLE true-p gate preserves ~all of
ungated recycling's value while the deployed frequency heuristic pays
(.879 vs .798 AUC, 10/10). But an LLM loop has no oracle p(g'); the
practical upgrade must ESTIMATE destination pass rates, and fresh
probe rollouts of relabel destinations cost generation budget. This
experiment measures the dose response: how many probe rollouts per
step does an estimated-p gate need to recover the oracle gate?

Arms (paired by seed; teacher/env streams shared):
  ungated       upper reference
  freq          deployed heuristic (lower reference)
  truep         oracle gate (upper bound for any estimator)
  estp-B        gate on a decayed Beta posterior over destination pass
                rates, updated ONLY from probe rollouts: each step,
                the B most-recently-proposed distinct destinations get
                one fresh probe rollout each (result feeds the
                destination posterior, not training). B in {1, 4, 16}.
                Admission: posterior mean <= 0.5 (same threshold as
                truep/freq — no new tuning).
  PROBE ACCOUNTING: probes are charged against the same generation
  budget — an estp-B run at S steps performs S*B extra rollouts, so we
  compare at MATCHED TOTAL ROLLOUTS by shortening estp-B's training
  steps: steps_estp = S * (T*N) / (T*N + B) with T=8 groups/step,
  N=16. (B=16 -> ~11% fewer steps.)

Pre-registered 2026-08-06 before any run:
  P-PB1: estp-16 recovers >= 80% of the oracle-vs-freq AUC gap
     (i.e., auc(estp16) >= auc(freq) + 0.8*(auc(truep)-auc(freq)))
     in >= 7/10 seeds — a practical gate needs only a small probe
     budget, and the paper may say so.
  P-PB2 (dose response): auc(estp1) <= auc(estp4) <= auc(estp16) in
     a majority of seeds (monotone in probe budget).
  Falsification (committed): if estp-16 recovers < half the gap, the
     "indicated upgrade" framing in 6.8 is weakened to "the oracle
     gate is better but no practical estimator we tested recovers it"
     — stated in the paper either way.

Usage: python3 run_gate_probe_budget.py [--seeds 10] [--steps 400]
Writes results_gate_probe_budget.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from testbed import SkillChainEnv
from estimators import weights_maxrl
from teachers import AdvMassTeacher
from run_gate_variants import FreqGate, correct_prefix_len, coverage8

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS_PER_BATCH = 8
N_ROLLOUTS = 16


class ProbeGate:
    """Decayed Beta posterior over destination pass rates, fed only by
    fresh probe rollouts (one per probed destination per step)."""

    def __init__(self, max_p=0.5, decay=0.95):
        self.max_p, self.decay = max_p, decay
        self.post: dict = {}          # dest -> [a, b] decayed counts
        self.recent: list = []        # recently proposed destinations (LRU)

    def propose(self, dest):
        if dest in self.recent:
            self.recent.remove(dest)
        self.recent.append(dest)

    def admit(self, dest) -> bool:
        a, b = self.post.get(dest, (0.0, 0.0))
        return (a + 1.0) / (a + b + 2.0) <= self.max_p

    def probe_step(self, env, budget: int):
        """Probe the `budget` most recent distinct destinations."""
        for dest in list(reversed(self.recent))[:budget]:
            _, r = env.rollout(dest, 1)
            a, b = self.post.get(dest, (0.0, 0.0))
            self.post[dest] = (a * self.decay + float(r[0]),
                               b * self.decay + float(1 - r[0]))


def run(seed: int, steps: int, arm: str, lr=0.5):
    env = SkillChainEnv(seed=seed)
    teacher = AdvMassTeacher(env.n_tasks, seed=seed + 1000,
                             n_rollouts=N_ROLLOUTS)
    chain_of = [t // env.n_levels for t in range(env.n_tasks)]
    task_of = {(chain_of[t], env.task_level[t]): t
               for t in range(env.n_tasks)}

    probe_budget = int(arm[4:]) if arm.startswith("estp") else 0
    # matched total rollouts: shorten training for probe arms
    train_per_step = TASKS_PER_BATCH * N_ROLLOUTS
    eff_steps = int(round(steps * train_per_step /
                          (train_per_step + probe_budget)))
    fgate, pgate = FreqGate(), ProbeGate()
    auc_acc, cov0 = [], coverage8(env)
    for step in range(eff_steps):
        for t in teacher.sample_tasks(TASKS_PER_BATCH):
            t = int(t)
            actions, rewards = env.rollout(t, N_ROLLOUTS)
            teacher.observe(t, rewards)
            w = weights_maxrl(rewards)
            if np.any(w != 0):
                env.apply_gradient(t, actions, w, lr)
                continue
            prefixes = np.array([correct_prefix_len(a) for a in actions])
            jstar = int(prefixes.max())
            if jstar < 1:
                continue
            target = task_of[(chain_of[t], jstar)]
            if arm == "ungated":
                admit = True
            elif arm == "freq":
                admit = fgate.decide(target)
            elif arm == "truep":
                admit = float(env.true_pass_rates()[target]) <= 0.5
            else:                     # estp-B
                pgate.propose(target)
                admit = pgate.admit(target)
            if not admit:
                continue
            r2 = (prefixes >= jstar).astype(float)
            w2 = weights_maxrl(r2)
            if np.any(w2 != 0):
                env.apply_gradient(target, actions[:, :jstar], w2, lr)
        fgate.end_batch()
        if probe_budget:
            pgate.probe_step(env, probe_budget)
        auc_acc.append(env.true_pass_rates().mean())
    # pad AUC to the reference horizon so shorter probe arms are not
    # advantaged by averaging over an easier (early) prefix
    while len(auc_acc) < steps:
        auc_acc.append(auc_acc[-1])
    return {"final": float(env.true_pass_rates().mean()),
            "auc": float(np.mean(auc_acc)),
            "delta_cov8": coverage8(env) - cov0,
            "eff_steps": eff_steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    arms = ["ungated", "freq", "truep", "estp1", "estp4", "estp16"]
    out = {"steps": args.steps, "seeds": args.seeds,
           "arms": {a: [] for a in arms}}
    for seed in range(args.seeds):
        for arm in arms:
            out["arms"][arm].append(run(seed, args.steps, arm))

    print(f"{'arm':>8} {'AUC':>16} {'final':>16} {'eff_steps':>9}")
    for arm in arms:
        rs = out["arms"][arm]
        print(f"{arm:>8} "
              f"{np.mean([r['auc'] for r in rs]):+.4f}±{np.std([r['auc'] for r in rs], ddof=1):.4f} "
              f"{np.mean([r['final'] for r in rs]):+.4f}±{np.std([r['final'] for r in rs], ddof=1):.4f} "
              f"{rs[0]['eff_steps']:9d}")

    # P-PB1: gap recovery by estp16
    rec = []
    for s in range(args.seeds):
        f = out["arms"]["freq"][s]["auc"]
        t = out["arms"]["truep"][s]["auc"]
        e = out["arms"]["estp16"][s]["auc"]
        rec.append((e - f) / (t - f) if t > f else float("nan"))
    n_pass = sum(r >= 0.8 for r in rec if not np.isnan(r))
    out["P-PB1 estp16 gap recovery"] = {
        "per_seed": rec, "n_ge_0.8": int(n_pass), "n": len(rec),
        "mean": float(np.nanmean(rec))}
    print(f"\nP-PB1 estp16 recovers >=80% of oracle-freq gap: "
          f"{n_pass}/{len(rec)} seeds (mean recovery {np.nanmean(rec):.2f})")
    # P-PB2 monotonicity
    mono = sum(out["arms"]["estp1"][s]["auc"]
               <= out["arms"]["estp4"][s]["auc"]
               <= out["arms"]["estp16"][s]["auc"]
               for s in range(args.seeds))
    out["P-PB2 monotone"] = {"n_monotone": int(mono), "n": args.seeds}
    print(f"P-PB2 monotone in probe budget: {mono}/{args.seeds}")

    path = os.path.join(HERE, "results_gate_probe_budget.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
