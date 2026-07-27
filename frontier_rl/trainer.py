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

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from frontier_rl.estimators import grpo_weights, maxrl_weights, rloo_weights
from frontier_rl.interfaces import GroupResult, Policy, TaskSpace
from frontier_rl.teacher import FrontierTeacher


@dataclass
class TrainerConfig:
    n_rollouts: int = 16
    tasks_per_step: int = 8
    hindsight: bool = True          # dense relabeling of dead groups (F3)
    hindsight_scale: float = 1.0    # natural K=1 group weight; tune down if
                                    # self-imitation entrenches errors
    positive_weights: bool = False  # weighted-RFT: success weights only, for
                                    # policies without per-sample log-probs
                                    # (flow heads / weighted SFT — COSMOS3 Q1);
                                    # E[Σw⁺] = u(p) exactly, so the teacher's
                                    # algebra is unchanged
    estimator: str = "maxrl"        # BASELINE ARMS ONLY: "grpo" (the H6
                                    # collapse ablation — curricula amplify
                                    # GRPO's coverage collapse) or "rloo".
                                    # positive_weights applies to maxrl only.
    dapo_max_redraws: int = 0       # DAPO-style dynamic sampling baseline:
                                    # on a dead group, redraw a fresh task up
                                    # to this many times, PAYING for every
                                    # draw (V5's matched-generation protocol).
                                    # 0 = off. Mutually exclusive in spirit
                                    # with hindsight (DAPO discards failures;
                                    # hindsight recycles them).
    teacher_gamma: float = 1.0      # V6: ~4 on chained pools
    teacher_decay: float = 0.7
    teacher_floor: float = 0.1
    seed: int = 0


@dataclass
class StepStats:
    live_groups: int = 0
    dead_groups: int = 0
    relabeled_groups: int = 0
    mean_reward: float = 0.0


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

    def _weights(self, rewards: np.ndarray) -> np.ndarray:
        if self.cfg.estimator == "maxrl":
            return maxrl_weights(rewards,
                                 positive_part=self.cfg.positive_weights)
        if self.cfg.estimator == "grpo":
            return grpo_weights(rewards)
        if self.cfg.estimator == "rloo":
            return rloo_weights(rewards)
        raise ValueError(f"unknown estimator {self.cfg.estimator!r}")

    def step(self) -> StepStats:
        stats = StepStats()
        rewards_seen = []
        for task_id in self.teacher.sample_tasks(self.cfg.tasks_per_step):
            task_id = int(task_id)
            group = self.env.rollout_group(task_id, self.cfg.n_rollouts)
            r = np.asarray(group.rewards, dtype=float)
            self.teacher.observe(task_id, r)   # requested-task evidence only
            rewards_seen.append(r.mean())

            # DAPO baseline: redraw fresh tasks until the group is live
            # (0 < K < N), paying for every discarded draw — the trainer
            # counts them as dead groups so budget accounting stays honest
            redraws = self.cfg.dapo_max_redraws
            while redraws > 0 and (r.sum() == 0 or r.sum() == len(r)):
                stats.dead_groups += 1
                redraws -= 1
                task_id = int(self.teacher.sample_tasks(1)[0])
                group = self.env.rollout_group(task_id, self.cfg.n_rollouts)
                r = np.asarray(group.rewards, dtype=float)
                self.teacher.observe(task_id, r)
                rewards_seen.append(r.mean())

            w = self._weights(r)
            if np.any(w != 0):
                stats.live_groups += 1
                self.policy.update(task_id, group.trajectories, w)
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
            w2 = self._weights(r2) * self.cfg.hindsight_scale
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
