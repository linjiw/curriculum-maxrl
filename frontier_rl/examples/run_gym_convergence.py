"""To-convergence gym study (user question: do the curves converge / does
the official task get solved?).

Arms: target_only vs full_gated. Train until the eval curve plateaus
(no improvement > 0.01 over 6 consecutive evals) or the step cap.
Report in OFFICIAL currency too: MountainCar = flag-bin pass rate
(position >= 0.5); CartPole = P(survive >= 400) and mean survival steps.
3 seeds. Writes gym_convergence.json with full curves.
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
    cfg = TrainerConfig(seed=seed, n_rollouts=10, tasks_per_step=6,
                        hindsight=(arm == "full_gated"),
                        hindsight_gate=(arm == "full_gated"))
    teacher = FrontierTeacher(env.n_tasks, cfg.n_rollouts, seed=seed + 500,
                              decay=cfg.teacher_decay, floor=cfg.teacher_floor)
    if arm == "target_only":
        d = np.zeros(env.n_tasks); d[-1] = 1.0
        teacher.distribution = lambda: d
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
    out = {}
    for env_name in ("mc", "cp"):
        for arm in ("target_only", "full_gated"):
            runs = [run(env_name, arm, s) for s in range(3)]
            key = f"{env_name}_{arm}"
            out[key] = runs
            fh = [r["hard"][-1] for r in runs]
            print(f"{key:18s} final hard-bin {np.mean(fh):.3f}±{np.std(fh):.3f} "
                  f"(converged ~step {int(np.mean([r['converged_at'] for r in runs]))})", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gym_convergence.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
