"""Gym performance study: does the curriculum improve THE TASK, not just
sampling mass? (User question, 2026-07-29.)

Currency = target-task performance:
  MountainCar: pass rate on the FLAG bin (x >= 0.5) — the actual gym task
  CartPole:    pass rate on the LONGEST survival bin (500 steps) — ditto
plus mean-across-bins and a coverage metric (# bins with p > 0.5).

Arms (matched total episodes; 5 seeds):
  target_only   train ONLY on the target task (the standard RL setup)
  uniform       uniform over curriculum bins
  teacher       frontier teacher (gamma=1)
  teacher_hs    teacher + hindsight (ungated)
  full_gated    teacher + hindsight + utility gate  <- the current method

Writes frontier_rl/examples/gym_performance.json. ~25 min CPU.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np

from frontier_rl import FrontierTrainer, TrainerConfig, FrontierTeacher
from frontier_rl.adapters.gym_classic import MountainCarSpace, CartPoleSurviveSpace

import os as _os
_MULT = float(_os.environ.get("BUDGET_MULT", "1"))
STEPS = {"mc": int(220 * _MULT), "cp": int(160 * _MULT)}


def make_teacher(env, cfg, mode, seed):
    t = FrontierTeacher(env.n_tasks, cfg.n_rollouts, seed=seed + 500,
                        decay=cfg.teacher_decay, floor=cfg.teacher_floor,
                        gamma=cfg.teacher_gamma)
    if mode == "uniform":
        t.distribution = lambda: np.full(env.n_tasks, 1.0 / env.n_tasks)
    elif mode == "target_only":
        d = np.zeros(env.n_tasks); d[-1] = 1.0
        t.distribution = lambda: d
    return t


def run(env_name, arm, seed):
    if env_name == "mc":
        env = MountainCarSpace(seed=seed)
    else:
        env = CartPoleSurviveSpace(seed=seed)
    hs = arm in ("teacher_hs", "full_gated")
    cfg = TrainerConfig(seed=seed, n_rollouts=10, tasks_per_step=6,
                        hindsight=hs, hindsight_gate=(arm == "full_gated"))
    mode = arm if arm in ("uniform", "target_only") else "teacher"
    teacher = make_teacher(env, cfg, mode, seed)
    tr = FrontierTrainer(env, env, cfg, teacher=teacher)
    tr.train(STEPS[env_name])
    p = env.eval_pass_rates(n=32)
    return float(p[-1]), float(p.mean()), float((p > 0.5).mean())


ARMS = ["target_only", "uniform", "teacher", "teacher_hs", "full_gated"]

if __name__ == "__main__":
    out = {}
    for env_name in ("mc", "cp"):
        for arm in ARMS:
            res = [run(env_name, arm, s) for s in range(5)]
            tgt, mean, cov = zip(*res)
            key = f"{env_name}_{arm}"
            out[key] = {"target_task": float(np.mean(tgt)), "target_std": float(np.std(tgt)),
                        "mean_bins": float(np.mean(mean)), "coverage": float(np.mean(cov))}
            print(f"{key:18s} TARGET {np.mean(tgt):.3f}±{np.std(tgt):.3f} "
                  f"mean {np.mean(mean):.3f} cov {np.mean(cov):.2f}", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gym_performance.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
