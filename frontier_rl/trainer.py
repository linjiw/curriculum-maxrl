"""FrontierTrainer: the validated training schedule, environment-agnostic.

Per step:
  1. teacher samples `tasks_per_step` task ids
  2. env rolls a group of `n_rollouts` per task
  3. teacher.observe(requested task, rewards)          [never relabeled ones]
  4. live groups -> MaxRL weights -> policy.update
  5. dead groups -> env.relabel -> K-style weights on the relabeled task,
     scaled by hindsight_scale -> policy.update(relabeled_task, ...)

This module has no torch/gym dependency; numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from frontier_rl.estimators import maxrl_success_weights, maxrl_weights
from frontier_rl.interfaces import GroupResult, Policy, TaskSpace
from frontier_rl.teacher import FrontierTeacher


@dataclass
class TrainerConfig:
    n_rollouts: int = 16
    tasks_per_step: int = 8
    hindsight: bool = True          # dense relabeling of dead groups (F3)
    hindsight_scale: float = 1.0    # natural K=1 group weight; tune down if
                                    # self-imitation entrenches errors
    hindsight_estimator: str = "maxrl"  # "maxrl" (centered) or "success_only"
    teacher_gamma: float = 1.0      # V6: ~4 on chained pools
    teacher_decay: float = 0.7
    teacher_floor: float = 0.1
    seed: int = 0


@dataclass
class StepStats:
    live_groups: int = 0
    dead_groups: int = 0
    all_pass_groups: int = 0
    relabeled_groups: int = 0
    mean_reward: float = 0.0
    env_steps: int = 0


class FrontierTrainer:
    def __init__(self, env: TaskSpace, policy: Policy, config: TrainerConfig = None,
                 teacher: Optional[FrontierTeacher] = None):
        self.env = env
        self.policy = policy
        self.cfg = config or TrainerConfig()
        self.teacher = teacher or FrontierTeacher(
            env.n_tasks, self.cfg.n_rollouts,
            decay=self.cfg.teacher_decay, floor=self.cfg.teacher_floor,
            gamma=self.cfg.teacher_gamma, seed=self.cfg.seed)

    def step(self) -> StepStats:
        stats = StepStats()
        rewards_seen = []
        for task_id in self.teacher.sample_tasks(self.cfg.tasks_per_step):
            task_id = int(task_id)
            group = self.env.rollout_group(task_id, self.cfg.n_rollouts)
            r = np.asarray(group.rewards, dtype=float)
            self.teacher.observe(task_id, r)   # requested-task evidence only
            rewards_seen.append(r.mean())
            # Sequence trajectories naturally expose episode length.  Mapping
            # trajectories (for example the grid adapter's structured record)
            # do not: counting their keys would be a bogus transition count.
            if all(not isinstance(traj, dict) for traj in group.trajectories):
                try:
                    stats.env_steps += sum(len(traj)
                                           for traj in group.trajectories)
                except TypeError:
                    pass
            elif group.infos and all(
                    isinstance(info, dict) and "n_steps" in info
                    for info in group.infos):
                stats.env_steps += sum(int(info["n_steps"])
                                       for info in group.infos)

            k = float(r.sum())
            if 0.0 < k < len(r):
                w = maxrl_weights(r)
                stats.live_groups += 1
                self.policy.update(task_id, group.trajectories, w)
                continue

            # The practical centered estimator is zero for both extremes, but
            # only K=0 is a failed group eligible for hindsight.  Treating
            # K=N as dead silently relabels mastered-task successes.
            if k >= len(r):
                stats.all_pass_groups += 1
                continue

            stats.dead_groups += 1
            if not self.cfg.hindsight:
                continue
            relabel = self.env.relabel(group)
            if relabel is None:
                continue
            if len(relabel) == 3:           # env rewrote goal-conditioning
                new_task, new_rewards, new_trajs = relabel
            else:
                new_task, new_rewards = relabel
                new_trajs = group.trajectories
            r2 = np.asarray(new_rewards, dtype=float)
            if self.cfg.hindsight_estimator == "maxrl":
                w2 = maxrl_weights(r2)
            elif self.cfg.hindsight_estimator == "success_only":
                # The raw success average is the estimator directly justified
                # by the ML conditional-expectation identity.  Relabeling can
                # still shift the trajectory law, so this is proof-aligned,
                # not automatically unbiased for an arbitrary relabeler.
                w2 = maxrl_success_weights(r2)
            else:
                raise ValueError(
                    "hindsight_estimator must be 'maxrl' or 'success_only'"
                )
            w2 = w2 * self.cfg.hindsight_scale
            if np.any(w2 != 0):
                stats.relabeled_groups += 1
                self.policy.update(int(new_task), new_trajs, w2)

        stats.mean_reward = float(np.mean(rewards_seen)) if rewards_seen else 0.0
        return stats

    def train(self, steps: int, on_eval: Optional[Callable[[int], None]] = None,
              eval_every: int = 25) -> list[StepStats]:
        history = []
        for i in range(steps):
            history.append(self.step())
            if on_eval is not None and (i % eval_every == 0 or i == steps - 1):
                on_eval(i)
        return history
