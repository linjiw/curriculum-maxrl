"""The Opus5-review control battery for the hindsight claim (M8 + B3).

Arms (uniform teacher unless stated, 3+ seeds, 400 steps, skill chain):
  none            baseline
  lr2             pure step-size doubling (no new information)
  replay          dead groups -> re-apply the batch's LIVE gradients again
                  (extra gradient, zero relabel information)
  randgoal        relabel dead groups to a RANDOM prefix level (direction
                  destroyed, update count matched)
  hindsight       the full method
  oracle_g4       true-p sampler, gamma=4, no floor handicap, NO hindsight
  oracle_g4_hs    the same oracle WITH hindsight (the review's missing arm)
  thompson_g4_hs  the paper's 'full stack' for reference

Writes frontier_rl/examples/hindsight_controls.json. CPU-only, ~15 min.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np

from frontier_rl import FrontierTrainer, TrainerConfig, FrontierTeacher
from frontier_rl.adapters.skill_chain import SkillChainSpace
from frontier_rl.estimators import maxrl_weights


class ControlTrainer(FrontierTrainer):
    """FrontierTrainer with placebo modes for dead groups."""

    def __init__(self, *args, mode: str = "hindsight", **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self._last_live = []

    def step(self):
        if self.mode in ("none", "lr2", "hindsight"):
            return super().step()
        # custom dead-group handling
        stats_live = 0
        rewards_seen = []
        live_updates = []
        dead_groups = []
        for task_id in self.teacher.sample_tasks(self.cfg.tasks_per_step):
            task_id = int(task_id)
            group = self.env.rollout_group(task_id, self.cfg.n_rollouts)
            r = np.asarray(group.rewards, dtype=float)
            self.teacher.observe(task_id, r)
            rewards_seen.append(r.mean())
            w = maxrl_weights(r)
            if np.any(w != 0):
                self.policy.update(task_id, group.trajectories, w)
                live_updates.append((task_id, group.trajectories, w))
            else:
                dead_groups.append(group)
        for group in dead_groups:
            if self.mode == "replay" and live_updates:
                # re-apply a random live gradient: extra update, no new info
                t, trajs, w = live_updates[np.random.randint(len(live_updates))]
                self.policy.update(t, trajs, w)
            elif self.mode == "randgoal":
                rel = self.env.relabel(group)
                if rel is None:
                    continue
                # destroy direction: relabel to a RANDOM level (at most the
                # achieved level, so trajectory slicing stays valid), with
                # the same reward pattern the real relabel would grant
                real_task, new_r = rel[0], rel[1]
                real_level = real_task % self.env.n_levels  # 0-based level-1
                chain0 = (group.task_id // self.env.n_levels) * self.env.n_levels
                rand_level = np.random.randint(0, real_level + 1)
                rand_task = chain0 + rand_level
                # slice each trajectory to the random task's prefix length
                L = rand_level + 1
                trajs = [t[:L] for t in group.trajectories]
                w2 = maxrl_weights(np.asarray(new_r, dtype=float))
                if np.any(w2 != 0):
                    self.policy.update(int(rand_task), trajs, w2)
        from frontier_rl.trainer import StepStats
        s = StepStats()
        s.mean_reward = float(np.mean(rewards_seen)) if rewards_seen else 0.0
        return s


def run(mode, seed, steps=400):
    env = SkillChainSpace(seed=seed)
    lr_mult = 2.0 if mode == "lr2" else 1.0
    env.lr = env.lr * lr_mult
    gamma = 4.0 if "g4" in mode else 1.0
    cfg = TrainerConfig(seed=seed, hindsight=(mode in ("hindsight", "oracle_g4_hs",
                                                       "thompson_g4_hs")),
                        teacher_gamma=gamma,
                        teacher_floor=0.0 if mode.startswith("oracle") else 0.1)
    teacher = FrontierTeacher(env.n_tasks, cfg.n_rollouts, seed=seed + 1000,
                              decay=cfg.teacher_decay, floor=cfg.teacher_floor,
                              gamma=cfg.teacher_gamma)
    if mode.startswith("oracle"):
        def dist():
            p = env.true_pass_rates()
            u = np.maximum((1 - (1 - p) ** cfg.n_rollouts) - p, 0.0) ** gamma
            if u.sum() <= 0:
                return np.full(env.n_tasks, 1.0 / env.n_tasks)
            return u / u.sum()
        teacher.distribution = dist
    elif mode not in ("thompson_g4_hs",):
        teacher.distribution = lambda: np.full(env.n_tasks, 1.0 / env.n_tasks)
    tr = ControlTrainer(env, env, cfg, teacher=teacher,
                        mode=mode if mode in ("replay", "randgoal") else
                        ("hindsight" if cfg.hindsight else "none"))
    curve = []
    tr.train(steps, on_eval=lambda i: curve.append(env.true_pass_rates().mean()),
             eval_every=10)
    return float(np.mean(curve)), float(curve[-1])


ARMS = ["none", "lr2", "replay", "randgoal", "hindsight",
        "oracle_g4", "oracle_g4_hs", "thompson_g4_hs"]

if __name__ == "__main__":
    out = {}
    for mode in ARMS:
        aucs, finals = zip(*(run(mode, s) for s in range(5)))
        out[mode] = {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                     "final_mean": float(np.mean(finals)),
                     "auc_per_seed": list(aucs)}
        print(f"{mode:16s} AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f}", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "hindsight_controls.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
