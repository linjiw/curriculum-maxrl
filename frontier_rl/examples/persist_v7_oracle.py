"""Persist the V7 headline (full stack beats the oracle sampler) to JSON.

The audit found the most-quoted number in the repo (0.890 > 0.851, 5 seeds)
had no primary artifact — run_skill_chain.py prints to stdout. This runs the
four arms (uniform / Thompson teacher / ORACLE true-p sampler / full stack)
on the anchor and writes frontier_rl/examples/v7_oracle_result.json.
"""
from __future__ import annotations

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np

from frontier_rl import FrontierTrainer, TrainerConfig, FrontierTeacher
from frontier_rl.adapters.skill_chain import SkillChainSpace


def run(over, mode, seed, steps=400):
    env = SkillChainSpace(seed=seed)
    cfg = TrainerConfig(seed=seed, **over)
    teacher = FrontierTeacher(env.n_tasks, cfg.n_rollouts, seed=seed + 1000,
                              decay=cfg.teacher_decay, floor=cfg.teacher_floor,
                              gamma=cfg.teacher_gamma)
    if mode == "uniform":
        teacher.distribution = lambda: np.full(env.n_tasks, 1.0 / env.n_tasks)
    elif mode == "oracle":
        # perfect information: sample proportional to u(true p) each step
        def dist():
            p = env.true_pass_rates()
            u = (1.0 - (1.0 - p) ** cfg.n_rollouts) - p
            u = np.maximum(u, 0.0) ** cfg.teacher_gamma
            if u.sum() <= 0:
                return np.full(env.n_tasks, 1.0 / env.n_tasks)
            d = u / u.sum()
            return (1 - cfg.teacher_floor) * d + cfg.teacher_floor / env.n_tasks
        teacher.distribution = dist
    trainer = FrontierTrainer(env, env, cfg, teacher=teacher)
    curve = []
    trainer.train(steps, on_eval=lambda i: curve.append(env.true_pass_rates().mean()),
                  eval_every=10)
    return np.array(curve)


ARMS = [
    ("uniform", dict(hindsight=False), "uniform"),
    ("teacher_thompson", dict(hindsight=False), "teacher"),
    ("oracle_gamma1", dict(hindsight=False), "oracle"),
    ("oracle_gamma4", dict(hindsight=False, teacher_gamma=4.0), "oracle"),
    ("full_stack_gamma4_hs", dict(hindsight=True, teacher_gamma=4.0), "teacher"),
]

if __name__ == "__main__":
    out = {}
    for name, over, mode in ARMS:
        aucs, finals = [], []
        for seed in range(5):
            h = run(over, mode, seed)
            aucs.append(float(h.mean())); finals.append(float(h[-1]))
        out[name] = {"auc_per_seed": aucs, "auc_mean": float(np.mean(aucs)),
                     "auc_std": float(np.std(aucs)),
                     "final_mean": float(np.mean(finals))}
        print(f"{name:22s} AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f}", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "v7_oracle_result.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
