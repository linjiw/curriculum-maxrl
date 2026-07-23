"""Gymnasium classic-control adapters with real difficulty curricula.

Two tasks, chosen to test different claims:

MountainCarSpace — THE hard-exploration classic. Sparse binary success
  ("reach x >= 0.5 within budget") is difficult for this weak policy at the
  tested budget.  It is the external-dynamics analogue of our V5
  frontier-heavy regime.  Curriculum
  axis: target position x* from just-right-of-valley out to the flag at 0.5
  (task j = "reach x >= x*_j").  Hindsight: a failed rollout's max x reached
  IS a success for the bin containing it — exact under the env's own
  dynamics (contract 1).  The validated transfer configuration deliberately
  shares one task-agnostic policy across thresholds, so no goal input needs
  rewriting.  The optional disjoint-table control is task-indexed; there the
  adapter recomputes tiles under the achieved bin (contract 2).

CartPoleSurviveSpace — control (non-goal-conditioned): task j = "survive
  >= T_j steps".  Relabel = longest survival bin actually reached.  Tests
  that the schedule behaves on a task where difficulty is a scalar
  threshold rather than a spatial goal.

Both use REINFORCE with a tiny tile-coded linear softmax policy — weak on
purpose: the *schedule* is what's under test, and a policy strong enough to
solve the task instantly would hide scheduling differences.
"""

from __future__ import annotations

import copy

import numpy as np

from frontier_rl.interfaces import GroupResult

try:
    import gymnasium as gym
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install gymnasium") from e


# --------------------------------------------------------------------------
# small tile-coded linear policy, shared by both adapters
# --------------------------------------------------------------------------
class TilePolicy:
    """Softmax over actions with optional parameter sharing across task bins.

    ``share_across_tasks=False`` gives each curriculum bin a disjoint tile
    table (a useful negative control).  ``True`` uses one policy for every
    nested success threshold, allowing competence learned on easy thresholds
    to transfer to harder ones.
    """

    def __init__(self, n_bins_per_dim, obs_low, obs_high, n_tasks, n_actions,
                 lr=0.15, seed=0, share_across_tasks=False):
        self.bins = np.asarray(n_bins_per_dim)
        self.low = np.asarray(obs_low, dtype=float)
        self.high = np.asarray(obs_high, dtype=float)
        self.n_tasks = n_tasks
        self.share_across_tasks = share_across_tasks
        self.n_task_slots = 1 if share_across_tasks else n_tasks
        self.n_actions = n_actions
        self.lr = lr
        n_tiles = int(np.prod(self.bins)) * self.n_task_slots
        self.theta = np.zeros((n_tiles, n_actions))
        self.rng = np.random.default_rng(seed)

    def tile(self, obs, task_id):
        frac = np.clip((np.asarray(obs) - self.low) / (self.high - self.low),
                       0, 1 - 1e-9)
        idx = (frac * self.bins).astype(int)
        flat = 0
        for i, b in zip(idx, self.bins):
            flat = flat * b + i
        task_slot = 0 if self.share_across_tasks else task_id
        return int(flat) * self.n_task_slots + task_slot

    def probs(self, tile):
        z = self.theta[tile] - self.theta[tile].max()
        e = np.exp(z)
        return e / e.sum()

    def act(self, obs, task_id):
        t = self.tile(obs, task_id)
        return int(self.rng.choice(self.n_actions, p=self.probs(t))), t

    # Policy protocol: trajectories are lists of (tile, action).  In the
    # disjoint-table control task-conditioning is inside the tile, so
    # relabeled trajectories arrive already rewritten by the adapter.  In the
    # shared configuration the task slot is constant and this rewrite is a
    # no-op by construction.
    def update(self, task_id, trajectories, weights):
        # A rollout group is one Monte Carlo gradient estimate.  Evaluate every
        # score term at the same (pre-update) policy, accumulate, then apply one
        # update.  Mutating theta inside this loop would make the estimator
        # trajectory-order dependent.
        grad = np.zeros_like(self.theta)
        for traj, w in zip(trajectories, np.asarray(weights)):
            if w == 0.0:
                continue
            for tile, a in traj:
                p = self.probs(tile)
                g = -p
                g[a] += 1.0
                grad[tile] += w * g
        self.theta += self.lr * grad


# --------------------------------------------------------------------------
# MountainCar: positional curriculum
# --------------------------------------------------------------------------
class MountainCarSpace:
    """Task j = "reach x >= targets[j]"; targets walk from valley to flag."""

    def __init__(self, n_tasks: int = 10, max_steps: int = 200, seed: int = 0,
                 lr: float = 0.15, share_policy_across_tasks: bool = False):
        self.env = gym.make("MountainCar-v0")
        self.env.reset(seed=seed)
        self._n_tasks = n_tasks
        self.max_steps = max_steps
        # valley bottom ~ -0.5; flag at 0.5. Space targets in between.
        self.targets = np.linspace(-0.35, 0.5, n_tasks)
        self.policy = TilePolicy(n_bins_per_dim=[12, 12],
                                 obs_low=[-1.2, -0.07], obs_high=[0.6, 0.07],
                                 n_tasks=n_tasks, n_actions=3, lr=lr, seed=seed,
                                 share_across_tasks=share_policy_across_tasks)
        self.rng = np.random.default_rng(seed + 1)

    @property
    def n_tasks(self):
        return self._n_tasks

    def _episode(self, task_id):
        obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 30)))
        steps, x_after, max_x = [], [], -1.2
        target = self.targets[task_id]
        for _ in range(self.max_steps):
            a, tile = self.policy.act(obs, task_id)
            steps.append((obs.copy(), a))
            obs, _, term, trunc, _ = self.env.step(a)
            x_after.append(float(obs[0]))
            max_x = max(max_x, float(obs[0]))
            if obs[0] >= target:
                return steps, x_after, max_x, True
            if term or trunc:
                break
        return steps, x_after, max_x, False

    def rollout_group(self, task_id, n_rollouts):
        trajs, rewards, infos = [], [], []
        for _ in range(n_rollouts):
            steps, x_after, max_x, ok = self._episode(task_id)
            # store raw (obs, action); tiles are recomputed per credited task
            trajs.append(steps)
            rewards.append(float(ok))
            infos.append({"max_x": max_x, "x_after": x_after,
                          "n_steps": len(steps)})
        return GroupResult(task_id, np.array(rewards), trajs, infos)

    def _bin_of_x(self, x):
        """Largest task bin whose target is <= x (i.e. truly achieved)."""
        reached = np.nonzero(self.targets <= x)[0]
        return int(reached[-1]) if len(reached) else -1

    def relabel(self, group):
        bins = [self._bin_of_x(i["max_x"]) for i in group.infos]
        best = max(bins)
        if best < 0:
            return None
        new_rewards = np.array([1.0 if b == best else 0.0 for b in bins])
        # Match fresh lower-threshold episodes: a successful relabeled rollout
        # terminates at its first crossing, while a failure keeps its full trace.
        target = self.targets[best]
        new_trajs = []
        for traj, info, reward in zip(group.trajectories, group.infos,
                                      new_rewards):
            credited = traj
            if reward == 1.0:
                hit = next(i + 1 for i, x in enumerate(info["x_after"])
                           if x >= target)
                credited = traj[:hit]
            new_trajs.append([(self.policy.tile(obs, best), a)
                              for obs, a in credited])
        return best, new_rewards, new_trajs

    # Policy protocol shim: convert raw (obs, action) to tiles for the
    # *credited* task, then delegate (live groups come through here).
    def update(self, task_id, trajectories, weights):
        tiled = []
        for traj in trajectories:
            if traj and isinstance(traj[0][0], (int, np.integer)):
                tiled.append(traj)          # already tiled (relabeled path)
            else:
                tiled.append([(self.policy.tile(obs, task_id), a)
                              for obs, a in traj])
        self.policy.update(task_id, tiled, weights)

    def eval_pass_rates(self, n=24, seed=None):
        # Evaluation must not change the subsequent training trajectory.
        env_rng = copy.deepcopy(self.rng.bit_generator.state)
        policy_rng = copy.deepcopy(self.policy.rng.bit_generator.state)
        try:
            if seed is not None:
                self.rng.bit_generator.state = copy.deepcopy(
                    np.random.default_rng(seed).bit_generator.state
                )
                self.policy.rng.bit_generator.state = copy.deepcopy(
                    np.random.default_rng(seed + 1).bit_generator.state
                )
            return np.array([np.mean(self.rollout_group(t, n).rewards)
                             for t in range(self._n_tasks)])
        finally:
            self.rng.bit_generator.state = env_rng
            self.policy.rng.bit_generator.state = policy_rng

    def close(self):
        self.env.close()


# --------------------------------------------------------------------------
# CartPole: survival-duration curriculum (non-goal-conditioned control)
# --------------------------------------------------------------------------
class CartPoleSurviveSpace:
    """Task j = "survive >= durations[j] steps"."""

    def __init__(self, n_tasks: int = 8, seed: int = 0, lr: float = 0.1,
                 share_policy_across_tasks: bool = False):
        self.env = gym.make("CartPole-v1")
        self.env.reset(seed=seed)
        self._n_tasks = n_tasks
        self.durations = np.unique(np.geomspace(20, 400, n_tasks).astype(int))
        self._n_tasks = len(self.durations)
        self.policy = TilePolicy(n_bins_per_dim=[6, 6, 8, 8],
                                 obs_low=[-2.4, -3.0, -0.21, -3.0],
                                 obs_high=[2.4, 3.0, 0.21, 3.0],
                                 n_tasks=self._n_tasks, n_actions=2,
                                 lr=lr, seed=seed,
                                 share_across_tasks=share_policy_across_tasks)
        self.rng = np.random.default_rng(seed + 1)

    @property
    def n_tasks(self):
        return self._n_tasks

    def rollout_group(self, task_id, n_rollouts):
        need = int(self.durations[task_id])
        trajs, rewards, infos = [], [], []
        for _ in range(n_rollouts):
            obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 30)))
            steps, alive = [], 0
            for _ in range(need):
                a, tile = self.policy.act(obs, task_id)
                steps.append((obs.copy(), a))
                obs, _, term, trunc, _ = self.env.step(a)
                alive += 1
                if term or trunc:
                    break
            trajs.append(steps)
            rewards.append(1.0 if alive >= need else 0.0)
            infos.append({"alive": alive, "n_steps": len(steps)})
        return GroupResult(task_id, np.array(rewards), trajs, infos)

    def relabel(self, group):
        alive = np.array([i["alive"] for i in group.infos])
        reached = [np.nonzero(self.durations <= a)[0] for a in alive]
        bins = [int(r[-1]) if len(r) else -1 for r in reached]
        best = max(bins)
        if best < 0:
            return None
        new_rewards = np.array([1.0 if b >= best else 0.0 for b in bins])
        # Fresh task-best successes stop after exactly the target duration.
        # Failures terminate earlier and keep their full trace.
        need = int(self.durations[best])
        new_trajs = []
        for traj, reward in zip(group.trajectories, new_rewards):
            credited = traj[:need] if reward == 1.0 else traj
            new_trajs.append([(self.policy.tile(obs, best), a)
                              for obs, a in credited])
        return best, new_rewards, new_trajs

    def update(self, task_id, trajectories, weights):
        tiled = []
        for traj in trajectories:
            if traj and isinstance(traj[0][0], (int, np.integer)):
                tiled.append(traj)
            else:
                tiled.append([(self.policy.tile(obs, task_id), a)
                              for obs, a in traj])
        self.policy.update(task_id, tiled, weights)

    def eval_pass_rates(self, n=16, seed=None):
        env_rng = copy.deepcopy(self.rng.bit_generator.state)
        policy_rng = copy.deepcopy(self.policy.rng.bit_generator.state)
        try:
            if seed is not None:
                self.rng.bit_generator.state = copy.deepcopy(
                    np.random.default_rng(seed).bit_generator.state
                )
                self.policy.rng.bit_generator.state = copy.deepcopy(
                    np.random.default_rng(seed + 1).bit_generator.state
                )
            return np.array([np.mean(self.rollout_group(t, n).rewards)
                             for t in range(self._n_tasks)])
        finally:
            self.rng.bit_generator.state = env_rng
            self.policy.rng.bit_generator.state = policy_rng

    def close(self):
        self.env.close()
