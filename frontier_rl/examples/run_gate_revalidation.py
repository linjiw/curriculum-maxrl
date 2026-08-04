"""Gate re-validation on the CPU rungs (the user's question: does the
method modification change — or improve — the validated results?).

Arms per testbed, 5 seeds each, with BOTH currencies (mean pass = the
original metric; coverage = fraction of tasks with pass rate > 0.5,
the CPU analogue of pass@k the original runs never measured):

  skill chain (fixed pool, the +0.22 hindsight regime):
    full stack (gamma=4 + HS)      — the validated champion
    full stack + GATE              — does the gate cost the fixed-pool gain?
  gridworld reach (goal-conditioned, the contract-2 testbed):
    teacher + HS                   — validated 0.703
    teacher + HS + GATE
Writes frontier_rl/examples/gate_revalidation.json. ~12 min CPU.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np

from frontier_rl import FrontierTrainer, TrainerConfig
from frontier_rl.adapters.skill_chain import SkillChainSpace
from frontier_rl.adapters.grid_reach import GridReachSpace


def run_chain(gate, seed, steps=400):
    env = SkillChainSpace(seed=seed)
    cfg = TrainerConfig(seed=seed, hindsight=True, teacher_gamma=4.0,
                        hindsight_gate=gate)
    tr = FrontierTrainer(env, env, cfg)
    curve, cov = [], []
    def ev(i):
        p = env.true_pass_rates()
        curve.append(p.mean())
        cov.append((p > 0.5).mean())
    tr.train(steps, on_eval=ev, eval_every=10)
    return float(np.mean(curve)), float(curve[-1]), float(cov[-1])


def run_grid(gate, seed, steps=300):
    env = GridReachSpace(radius=6, seed=seed)
    cfg = TrainerConfig(seed=seed, hindsight=True, hindsight_gate=gate)
    tr = FrontierTrainer(env, env, cfg)
    curve, cov = [], []
    def ev(i):
        p = env.eval_pass_rates(n=32)
        curve.append(p.mean())
        cov.append((p > 0.5).mean())
    tr.train(steps, on_eval=ev, eval_every=15)
    return float(np.mean(curve)), float(curve[-1]), float(cov[-1])


if __name__ == "__main__":
    out = {}
    for name, fn in [("skill_chain", run_chain), ("grid_reach", run_grid)]:
        for gate in (False, True):
            key = f"{name}{'_gated' if gate else ''}"
            res = [fn(gate, s) for s in range(5)]
            aucs, finals, covs = zip(*res)
            out[key] = {"auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                        "final": float(np.mean(finals)),
                        "coverage_final": float(np.mean(covs)),
                        "coverage_std": float(np.std(covs))}
            print(f"{key:22s} AUC {np.mean(aucs):.4f}±{np.std(aucs):.4f} "
                  f"final {np.mean(finals):.3f} coverage {np.mean(covs):.3f}", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "gate_revalidation.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
