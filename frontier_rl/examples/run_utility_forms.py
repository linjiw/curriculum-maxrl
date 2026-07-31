"""Utility-form ablation (review Q5): does the exact u_N matter?

Same skill-chain protocol as run_skill_chain.py (teacher-only, no
hindsight, 5 seeds), three utility forms on the same Thompson posterior:
  exact   u(p) = (1-(1-p)^N) - p        (derived advantage mass, P1)
  legacy  u(p) = (1-(1-p)^N)(1-p)       (heuristic frontier form; maze used this)
  lp      u(p) = p(1-p)                 (learnability / N=2 slice)
Analytic TV between exact and legacy normalized sampling dists is 0.013
at N=8 — prediction: exact ~= legacy, lp shifts the band easier.
Writes utility_forms.json.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np

from frontier_rl import FrontierTrainer, TrainerConfig, FrontierTeacher
from frontier_rl.adapters.skill_chain import SkillChainSpace

FORMS = {
    "exact": lambda p, N: np.maximum((1.0 - (1.0 - p) ** N) - p, 0.0),
    "legacy": lambda p, N: (1.0 - (1.0 - p) ** N) * (1.0 - p),
    "lp": lambda p, N: p * (1.0 - p),
}


def run(form, seed, steps=400):
    env = SkillChainSpace(seed=seed)
    cfg = TrainerConfig(seed=seed, hindsight=False)
    teacher = FrontierTeacher(env.n_tasks, cfg.n_rollouts, seed=seed + 1000,
                              decay=cfg.teacher_decay, floor=cfg.teacher_floor,
                              gamma=cfg.teacher_gamma)
    fn = FORMS[form]
    teacher.utility = lambda p: fn(np.asarray(p, float), teacher.n_rollouts)
    trainer = FrontierTrainer(env, env, cfg, teacher=teacher)
    curve = []
    trainer.train(steps, on_eval=lambda i: curve.append(env.true_pass_rates().mean()),
                  eval_every=10)
    h = np.array(curve)
    return float(h.mean()), float(h[-1])


if __name__ == "__main__":
    out = {}
    for form in FORMS:
        res = [run(form, s) for s in range(5)]
        aucs = [r[0] for r in res]
        finals = [r[1] for r in res]
        out[form] = {"auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
                     "final_mean": float(np.mean(finals)), "final_sd": float(np.std(finals)),
                     "seeds": res}
        print(f"{form:8s} AUC={np.mean(aucs):.4f}(±{np.std(aucs):.4f}) "
              f"final={np.mean(finals):.3f}", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utility_forms.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
