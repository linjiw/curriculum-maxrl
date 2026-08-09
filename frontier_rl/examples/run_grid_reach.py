"""Goal-conditioned gridworld reach — the robotics-style demo.

Compares uniform / teacher / teacher+hindsight at equal group budgets.
Run: python3 -m frontier_rl.examples.run_grid_reach
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np


def _trapezoid(y, x):
    integrate = getattr(np, "trapezoid", None)
    return (np.trapz if integrate is None else integrate)(y, x)

from frontier_rl import FrontierTrainer, TrainerConfig, FrontierTeacher
from frontier_rl.adapters.grid_reach import GridReachSpace


def run(hindsight: bool, uniform: bool, seed: int, steps: int = 150):
    env = GridReachSpace(radius=8, seed=seed)
    cfg = TrainerConfig(n_rollouts=16, tasks_per_step=4, hindsight=hindsight,
                        teacher_gamma=4.0, seed=seed)
    teacher = FrontierTeacher(env.n_tasks, cfg.n_rollouts, seed=seed + 1000,
                              gamma=cfg.teacher_gamma)
    if uniform:
        teacher.distribution = lambda: np.full(env.n_tasks, 1.0 / env.n_tasks)
    trainer = FrontierTrainer(env, env, cfg, teacher=teacher)
    xs = [0]
    curve = [float(env.eval_pass_rates(n=32, seed=100_000 + seed).mean())]

    def evaluate(i):
        xs.append(i + 1)
        curve.append(float(
            env.eval_pass_rates(n=32, seed=100_000 + seed).mean()
        ))

    trainer.train(
        steps,
        on_eval=evaluate,
        eval_every=15,
    )
    return np.asarray(xs), np.asarray(curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "grid_reach_validation.json"),
    )
    args = ap.parse_args()
    source_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(source_dir, "..", ".."))
    source_files = [
        os.path.abspath(__file__),
        os.path.join(repo_dir, "frontier_rl", "adapters", "grid_reach.py"),
        os.path.join(repo_dir, "frontier_rl", "trainer.py"),
        os.path.join(repo_dir, "frontier_rl", "teacher.py"),
        os.path.join(repo_dir, "frontier_rl", "estimators.py"),
        os.path.join(repo_dir, "frontier_rl", "interfaces.py"),
    ]
    source_sha256 = {}
    for path in source_files:
        with open(path, "rb") as f:
            source_sha256[os.path.relpath(path, repo_dir)] = hashlib.sha256(
                f.read()
            ).hexdigest()

    result = {
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True,
                text=True, check=False,
            ).stdout.strip() or None,
            "git_worktree_dirty": bool(subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True,
                text=True, check=False,
            ).stdout.strip()),
            "source_sha256": source_sha256,
        },
        "protocol": {
            "seeds": args.seeds,
            "steps": 150,
            "n_rollouts": 16,
            "tasks_per_step": 4,
            "teacher_gamma": 4.0,
            "evaluation_episodes_per_ring": 32,
            "evaluation": "fixed per-seed RNG; training RNG restored",
            "gradient": "one frozen-policy batch update per rollout group",
            "relabel_stopping": (
                "successful relabeled traces stop at first hit of the concrete rewritten goal"
            ),
        },
        "cases": {},
    }
    for label, hs, uni in [("uniform + maxrl", False, True),
                           ("teacher + maxrl", False, False),
                           ("teacher + maxrl + hindsight", True, False)]:
        aucs, finals, runs = [], [], []
        for seed in range(args.seeds):
            x, h = run(hs, uni, seed)
            auc = float(_trapezoid(h, x) / x[-1])
            aucs.append(auc)
            finals.append(h[-1])
            runs.append({"seed": seed, "x_steps": x.tolist(),
                         "mean_pass_curve": h.tolist(), "auc": auc,
                         "final": float(h[-1])})
        result["cases"][label] = {
            "config": {"hindsight": hs, "uniform": uni},
            "auc_mean": float(np.mean(aucs)),
            "auc_sample_std": float(np.std(aucs, ddof=1)),
            "final_mean": float(np.mean(finals)),
            "final_sample_std": float(np.std(finals, ddof=1)),
            "runs": runs,
        }
        print(f"{label:30s} AUC={np.mean(aucs):.3f}"
              f"(±{np.std(aucs, ddof=1):.3f}) "
              f"final={np.mean(finals):.3f}", flush=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
