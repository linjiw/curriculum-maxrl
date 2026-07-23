"""Regression anchor: reproduce equally spaced checkpoint-mean scores on the
skill-chain testbed (uniform+maxrl ~0.65; full stack ~0.89).

Run: python3 -m frontier_rl.examples.run_skill_chain
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np

from frontier_rl import FrontierTrainer, TrainerConfig, FrontierTeacher
from frontier_rl.adapters.skill_chain import SkillChainSpace


def run(config: TrainerConfig, mode: str, seed: int, steps: int = 400):
    env = SkillChainSpace(seed=seed)
    teacher = FrontierTeacher(env.n_tasks, config.n_rollouts, seed=seed + 1000,
                              decay=config.teacher_decay, floor=config.teacher_floor,
                              gamma=config.teacher_gamma)
    if mode == "uniform":
        teacher.distribution = lambda: np.full(env.n_tasks, 1.0 / env.n_tasks)
    elif mode == "true_p":
        # Oracle comparison inside the same trainer: exact current pass rates,
        # the same proportional coefficient-mass rule, and the same floor.
        def true_p_distribution():
            utility = teacher.utility(env.true_pass_rates()) ** teacher.gamma
            if utility.sum() <= 1e-12:
                utility = np.ones(env.n_tasks)
            priority = utility / utility.sum()
            uniform = np.full(env.n_tasks, 1.0 / env.n_tasks)
            return (1.0 - teacher.floor) * priority + teacher.floor * uniform
        teacher.distribution = true_p_distribution
    elif mode != "teacher":
        raise ValueError(f"unknown teacher mode: {mode}")
    trainer = FrontierTrainer(env, env, config, teacher=teacher)
    curve = []
    def on_eval(i):
        curve.append(env.true_pass_rates().mean())
    trainer.train(steps, on_eval=on_eval, eval_every=10)
    return np.array(curve)


def main():
    cases = [
        ("uniform + maxrl (no hindsight)", dict(hindsight=False), "uniform"),
        ("teacher gamma=1 (no hindsight)", dict(hindsight=False), "teacher"),
        ("teacher gamma=4 (no hindsight)",
         dict(hindsight=False, teacher_gamma=4.0), "teacher"),
        ("true-p proportional priority", dict(hindsight=False), "true_p"),
        ("teacher gamma=1 + hindsight", dict(hindsight=True), "teacher"),
        ("teacher gamma=4 + hindsight (full stack)",
         dict(hindsight=True, teacher_gamma=4.0), "teacher"),
    ]
    for label, over, mode in cases:
        aucs, finals = [], []
        for seed in range(5):
            cfg = TrainerConfig(seed=seed, **over)
            h = run(cfg, mode, seed)
            aucs.append(h.mean()); finals.append(h[-1])
        print(f"{label:44s} checkpoint_mean={np.mean(aucs):.3f}(±{np.std(aucs):.3f}) "
              f"final={np.mean(finals):.3f}", flush=True)
    print("\nexpected: 0.650 uniform | 0.728 gamma=1 | 0.782 gamma=4 | "
          "0.851 true-p | 0.880 gamma=1+hindsight | 0.890 full stack")


if __name__ == "__main__":
    main()
