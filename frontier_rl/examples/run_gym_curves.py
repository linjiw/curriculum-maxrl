"""Training-progress curves for the gym performance figure (fig6).

Three decisive arms (target_only / uniform / full_gated), 3 seeds,
eval every 20 steps: mean-across-bins pass rate + hardest-bin pass rate
+ coverage. Writes frontier_rl/examples/gym_curves.json (plot-ready).
Budget = 3x demo (matches gym_performance_3x endpoints).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("BUDGET_MULT", "3")

import numpy as np

from frontier_rl import FrontierTrainer, TrainerConfig, FrontierTeacher
from frontier_rl.adapters.gym_classic import MountainCarSpace, CartPoleSurviveSpace

STEPS = {"mc": 660, "cp": 480}
EVAL_EVERY = 30


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
    env = MountainCarSpace(seed=seed) if env_name == "mc" else CartPoleSurviveSpace(seed=seed)
    cfg = TrainerConfig(seed=seed, n_rollouts=10, tasks_per_step=6,
                        hindsight=(arm == "full_gated"),
                        hindsight_gate=(arm == "full_gated"))
    mode = arm if arm in ("uniform", "target_only") else "teacher"
    teacher = make_teacher(env, cfg, mode, seed)
    tr = FrontierTrainer(env, env, cfg, teacher=teacher)
    xs, mean_c, hard_c, cov_c = [], [], [], []
    def ev(i):
        p = env.eval_pass_rates(n=16)
        xs.append(i)
        mean_c.append(float(p.mean()))
        hard_c.append(float(p[-1]))
        cov_c.append(float((p > 0.5).mean()))
    tr.train(STEPS[env_name], on_eval=ev, eval_every=EVAL_EVERY)
    return xs, mean_c, hard_c, cov_c


if __name__ == "__main__":
    out = {}
    for env_name in ("mc", "cp"):
        for arm in ("target_only", "uniform", "full_gated"):
            runs = [run(env_name, arm, s) for s in range(3)]
            xs = runs[0][0]
            out[f"{env_name}_{arm}"] = {
                "steps": xs,
                "mean": [[r[1][i] for r in runs] for i in range(len(xs))],
                "hard": [[r[2][i] for r in runs] for i in range(len(xs))],
                "cov":  [[r[3][i] for r in runs] for i in range(len(xs))],
            }
            m = out[f"{env_name}_{arm}"]["mean"][-1]
            h = out[f"{env_name}_{arm}"]["hard"][-1]
            print(f"{env_name}_{arm}: final mean {np.mean(m):.3f} hard-bin {np.mean(h):.3f}", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gym_curves.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
