"""Uniform-curriculum arm for the to-convergence gym study (review Q/8c).

The convergence study compared target_only vs full_gated only, so it
does not isolate the teacher from mere task spread. This adds:
  uniform          — uniform sampling over bins, no hindsight
  uniform_hs_gated — uniform sampling + gated hindsight (creation
                     channel without the allocation channel)
Same plateau rule and caps as run_gym_convergence.py. Merges results
into gym_convergence.json (existing arms preserved).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np

from frontier_rl import FrontierTrainer, TrainerConfig, FrontierTeacher
from frontier_rl.adapters.gym_classic import MountainCarSpace, CartPoleSurviveSpace

CAP = {"mc": 2400, "cp": 1600}
EVAL_EVERY = 40
PATIENCE, MIN_DELTA = 6, 0.01


def run(env_name, arm, seed):
    env = MountainCarSpace(seed=seed) if env_name == "mc" else CartPoleSurviveSpace(seed=seed)
    hs = arm == "uniform_hs_gated"
    cfg = TrainerConfig(seed=seed, n_rollouts=10, tasks_per_step=6,
                        hindsight=hs, hindsight_gate=hs)
    teacher = FrontierTeacher(env.n_tasks, cfg.n_rollouts, seed=seed + 500)
    teacher.distribution = lambda: np.full(env.n_tasks, 1.0 / env.n_tasks)
    tr = FrontierTrainer(env, env, cfg, teacher=teacher)
    xs, hard, mean = [], [], []
    best, stale = -1.0, 0
    step = 0
    while step < CAP[env_name]:
        tr.train(EVAL_EVERY)
        step += EVAL_EVERY
        p = env.eval_pass_rates(n=24)
        xs.append(step); hard.append(float(p[-1])); mean.append(float(p.mean()))
        if hard[-1] > best + MIN_DELTA:
            best, stale = hard[-1], 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return {"steps": xs, "hard": hard, "mean": mean, "converged_at": step}


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gym_convergence.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for env_name in ("mc", "cp"):
        for arm in ("uniform", "uniform_hs_gated"):
            runs = [run(env_name, arm, s) for s in range(3)]
            key = f"{env_name}_{arm}"
            out[key] = runs
            fh = [r["hard"][-1] for r in runs]
            print(f"{key:22s} final hard-bin {np.mean(fh):.3f}±{np.std(fh):.3f} "
                  f"(converged ~step {int(np.mean([r['converged_at'] for r in runs]))})", flush=True)
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
