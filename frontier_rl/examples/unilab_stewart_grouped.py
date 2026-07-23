#!/usr/bin/env python3
"""Exact grouped Curriculum-MaxRL pilot on UniLab's StewartBalance task.

This runner is intentionally separate from UniLab PPO.  For every update it
freezes one Gaussian actor, collects ``N`` complete episodes from one frozen
task distribution, forms the practical weights

    w_i = 1{K>0} (r_i / K - 1 / N),

and applies them to the complete on-policy trajectory score.  There is no PPO
clip, rollout reuse, critic, dense-reward actor term, or hindsight.  Therefore
the pre-optimizer actor gradient is the practical order-``N-1`` estimator; an
adaptive sampler still changes the mixture of tasks being optimized.

Run from the sibling UniLab checkout so its locked environment and Motrix
extra are active::

    uv run --extra motrix python \
      ../curriculum-maxrl/frontier_rl/examples/unilab_stewart_grouped.py \
      --output /tmp/unilab_stewart_grouped.json

The default run is a development mechanism pilot, not confirmatory evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontier_rl.estimators import maxrl_weights  # noqa: E402
from frontier_rl.teacher import FrontierTeacher  # noqa: E402
from unilab.envs.manipulation.stewart.balance import (  # noqa: E402
    StewartBalanceCfg,
    StewartBalanceDRProvider,
    StewartBalanceEnv,
)


Arm = Literal["uniform", "learnability", "advmass"]
DEFAULT_RATIOS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


class GaussianActor(nn.Module):
    """Small task-agnostic actor; task difficulty is visible through state."""

    def __init__(
        self,
        obs_dim: int = 15,
        action_dim: int = 2,
        hidden_dim: int = 64,
        *,
        output_tanh: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        ]
        # RSL-RL's Gaussian actor uses the last linear output directly. The
        # original standalone development actor bounded its mean with tanh.
        if output_tanh:
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(0.30)))
        self.register_buffer("obs_mean", torch.zeros((1, obs_dim)))
        self.register_buffer("obs_std", torch.ones((1, obs_dim)))
        self.register_buffer("obs_eps", torch.tensor(0.0))

    def distribution_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (obs - self.obs_mean) / (self.obs_std + self.obs_eps)
        mean = self.net(normalized)
        std = self.log_std.clamp(-4.0, 1.0).exp().expand_as(mean)
        return mean, std

    def sample(
        self,
        obs: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.distribution_params(obs)
        noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype)
        raw_action = mean + std * noise
        # REINFORCE needs the score of the sampled action, not a pathwise
        # derivative through the sampler.  Detaching here is essential:
        # without it ``raw_action - mean`` algebraically cancels the actor's
        # mean gradient and the update is not the trajectory score.
        scored_action = raw_action.detach()
        log_prob = -0.5 * (
            ((scored_action - mean) / std).square()
            + 2.0 * torch.log(std)
            + math.log(2.0 * math.pi)
        )
        return raw_action, log_prob.sum(dim=-1)


def score_function_regression_check() -> float:
    """Fail if sampled actions accidentally retain the pathwise mean gradient.

    With a reparameterized sample left attached, ``raw_action - mean`` cancels
    the Gaussian location-score gradient. A mixed binary group must instead
    produce a nonzero gradient in the actor's mean network. This deterministic
    check protects the exact-estimator claim without consuming run RNG state.
    """
    with torch.random.fork_rng():
        torch.manual_seed(31_415)
        actor = GaussianActor()
        observations = torch.linspace(-1.0, 1.0, steps=4 * 15).reshape(4, 15)
        generator = torch.Generator(device="cpu").manual_seed(27_182)
        _, log_prob = actor.sample(observations, generator)
        weights = torch.as_tensor(
            maxrl_weights(np.asarray([1.0, 0.0, 0.0, 0.0])),
            dtype=torch.float32,
        )
        loss = -(weights * log_prob).sum()
        loss.backward()
        mean_gradient_sq = sum(
            float(parameter.grad.detach().square().sum())
            for parameter in actor.net.parameters()
            if parameter.grad is not None
        )
    mean_gradient_norm = math.sqrt(mean_gradient_sq)
    if not math.isfinite(mean_gradient_norm) or mean_gradient_norm <= 1.0e-6:
        raise RuntimeError(
            "Gaussian score-function regression: actor-mean gradient vanished"
        )
    return mean_gradient_norm


def load_rsl_actor(path: Path) -> tuple[GaussianActor, dict[str, int | str]]:
    """Load the exact stochastic actor represented by a local RSL-RL checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("actor_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain actor_state_dict")
    required = (
        "obs_normalizer._mean",
        "obs_normalizer._std",
        "mlp.0.weight",
        "mlp.0.bias",
        "mlp.2.weight",
        "mlp.2.bias",
        "mlp.4.weight",
        "mlp.4.bias",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"unsupported RSL-RL actor checkpoint; missing {missing}")
    hidden_dim = int(state["mlp.0.weight"].shape[0])
    actor = GaussianActor(hidden_dim=hidden_dim, output_tanh=False)
    with torch.no_grad():
        for target_key, source_key in (
            ("0.weight", "mlp.0.weight"),
            ("0.bias", "mlp.0.bias"),
            ("2.weight", "mlp.2.weight"),
            ("2.bias", "mlp.2.bias"),
            ("4.weight", "mlp.4.weight"),
            ("4.bias", "mlp.4.bias"),
        ):
            actor.net.state_dict()[target_key].copy_(state[source_key])
        actor.obs_mean.copy_(state["obs_normalizer._mean"])
        actor.obs_std.copy_(state["obs_normalizer._std"])
        actor.obs_eps.fill_(0.01)
        if "distribution.std_param" in state:
            std = state["distribution.std_param"]
        elif "distribution.log_std_param" in state:
            std = state["distribution.log_std_param"].exp()
        else:
            raise ValueError("unsupported RSL-RL actor checkpoint: no Gaussian std")
        actor.log_std.copy_(std.clamp_min(1.0e-8).log())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return actor, {
        "checkpoint_file": path.name,
        "sha256": digest,
        "iteration": int(payload.get("iter", -1)),
    }


def make_actor(args: argparse.Namespace) -> GaussianActor:
    if args.rsl_warm_start is not None:
        actor, _ = load_rsl_actor(args.rsl_warm_start)
        return actor
    return GaussianActor()


class TaskSampler:
    """One evidence tracker with three registered selection shapes."""

    def __init__(self, arm: Arm, n_tasks: int, group_size: int, seed: int) -> None:
        self.arm = arm
        self.tracker = FrontierTeacher(
            n_tasks,
            n_rollouts=group_size,
            decay=0.7,
            floor=0.1,
            gamma=1.0,
            seed=seed,
        )

    def sample(self, group_index: int) -> int:
        # Exact one-pass coverage prevents a prior draw from hiding a task.
        if group_index < self.tracker.n_tasks:
            return group_index
        if self.arm == "uniform":
            return int(self.tracker.rng.integers(self.tracker.n_tasks))
        if self.arm == "advmass":
            return int(self.tracker.sample_tasks(1)[0])

        p = self.tracker.rng.beta(self.tracker.alpha, self.tracker.beta)
        utility = p * (1.0 - p)
        if float(utility.sum()) <= 1.0e-12:
            utility = np.ones_like(utility)
        probability = utility / utility.sum()
        probability = (
            (1.0 - self.tracker.floor) * probability
            + self.tracker.floor / self.tracker.n_tasks
        )
        return int(self.tracker.rng.choice(self.tracker.n_tasks, p=probability))

    def observe(self, task_id: int, rewards: np.ndarray) -> None:
        self.tracker.observe(task_id, rewards)

    def mean_based_distribution(self) -> np.ndarray:
        """Non-random diagnostic; does not consume the training sampler RNG."""
        if self.arm == "uniform":
            return np.full(self.tracker.n_tasks, 1.0 / self.tracker.n_tasks)
        p = self.tracker.pass_rate_estimates()
        utility = p * (1.0 - p) if self.arm == "learnability" else self.tracker.utility(p)
        if float(utility.sum()) <= 1.0e-12:
            utility = np.ones_like(utility)
        probability = utility / utility.sum()
        return (
            (1.0 - self.tracker.floor) * probability
            + self.tracker.floor / self.tracker.n_tasks
        )


class FixedRadiusResetProvider(StewartBalanceDRProvider):
    """Reset the ball at exactly the registered task radius and random angle."""

    def build_reset_plan(self, env, env_ids):  # noqa: ANN001, ANN201
        plan = super().build_reset_plan(env, env_ids)
        n = int(env_ids.shape[0])
        radius = env._cfg.platform_radius * env._cfg.init_ball_radius_ratio
        theta = np.random.uniform(0.0, 2.0 * np.pi, size=n)
        plan.qpos[:, env._ball_pos_qpos_idx[0]] = radius * np.cos(theta)
        plan.qpos[:, env._ball_pos_qpos_idx[1]] = radius * np.sin(theta)
        return plan


class DelayedSuccessStewartEnv(StewartBalanceEnv):
    """Prevent reset geometry from satisfying the stillness verifier."""

    def __init__(self, *args, success_delay_steps: int = 0, **kwargs) -> None:  # noqa: ANN002, ANN003
        self._success_delay_steps = int(success_delay_steps)
        super().__init__(*args, **kwargs)

    def _update_stillness(self, cfg, info, rel_xy, vel_xy):  # noqa: ANN001, ANN201
        steps = super()._update_stillness(cfg, info, rel_xy, vel_xy)
        ineligible = np.asarray(info["steps"] < self._success_delay_steps)
        if np.any(ineligible):
            info["still_window_active"][ineligible] = False
            info["still_steps"][ineligible] = 0
            steps = info["still_steps"]
        return steps


@dataclass
class GroupBatch:
    trajectory_scores: torch.Tensor
    rewards: np.ndarray
    episode_lengths: np.ndarray
    backend_env_steps: int
    active_episode_steps: int


def _make_env(
    ratio: float,
    n_envs: int,
    horizon_seconds: float,
    *,
    reset_mode: str,
    success_delay_seconds: float,
    still_steps_needed: int,
) -> StewartBalanceEnv:
    cfg = StewartBalanceCfg(
        max_episode_seconds=horizon_seconds,
        init_ball_radius_ratio=ratio,
        still_steps_needed=still_steps_needed,
    )
    provider = FixedRadiusResetProvider() if reset_mode == "fixed_radius" else None
    env = DelayedSuccessStewartEnv(
        cfg,
        num_envs=n_envs,
        backend_type="motrix",
        dr_provider=provider,
        success_delay_steps=int(round(success_delay_seconds / cfg.ctrl_dt)),
    )
    env.set_autoreset(False)
    env.init_state()
    assert env.state is not None
    env.state.info["steps"].fill(0)
    return env


def collect_group(
    actor: GaussianActor,
    *,
    ratio: float,
    group_size: int,
    horizon_seconds: float,
    env_seed: int,
    action_generator: torch.Generator,
    reset_mode: str,
    success_delay_seconds: float,
    still_steps_needed: int,
) -> GroupBatch:
    """Collect one frozen-policy wave and retain its true trajectory score."""
    np.random.seed(env_seed)
    env = _make_env(
        ratio,
        group_size,
        horizon_seconds,
        reset_mode=reset_mode,
        success_delay_seconds=success_delay_seconds,
        still_steps_needed=still_steps_needed,
    )
    max_steps = int(round(horizon_seconds / env.cfg.ctrl_dt))
    finished = np.zeros(group_size, dtype=bool)
    rewards = np.zeros(group_size, dtype=np.float64)
    lengths = np.zeros(group_size, dtype=np.int32)
    trajectory_scores = torch.zeros(group_size, dtype=torch.float32)
    try:
        for step in range(1, max_steps + 1):
            assert env.state is not None
            obs = torch.as_tensor(env.state.obs["obs"], dtype=torch.float32)
            raw_action, log_prob = actor.sample(obs, action_generator)
            active = torch.as_tensor(~finished, dtype=torch.float32)
            trajectory_scores = trajectory_scores + active * log_prob
            actions = raw_action.detach().numpy()
            actions[finished] = 0.0
            state = env.step(actions)
            done = np.asarray(state.terminated | state.truncated, dtype=bool)
            newly_done = done & ~finished
            if np.any(newly_done):
                # Stewart's task contract: stillness success terminates with a
                # positive reward; a fall terminates with a negative penalty.
                rewards[newly_done] = (
                    state.terminated[newly_done] & (state.reward[newly_done] > 0)
                ).astype(np.float64)
                lengths[newly_done] = step
                finished[newly_done] = True
    finally:
        env.close()

    if not np.all(finished):
        raise RuntimeError("not every grouped Stewart episode completed by the frozen horizon")
    return GroupBatch(
        trajectory_scores=trajectory_scores,
        rewards=rewards,
        episode_lengths=lengths,
        # Run the full allocated wave even if every trajectory finishes early.
        # This keeps simulator-transition budgets identical across samplers.
        backend_env_steps=group_size * max_steps,
        active_episode_steps=int(lengths.sum()),
    )


@torch.no_grad()
def evaluate_actor(
    actor: GaussianActor,
    *,
    ratios: tuple[float, ...],
    num_episodes: int,
    horizon_seconds: float,
    seed: int,
    reset_mode: str,
    success_delay_seconds: float,
    still_steps_needed: int,
) -> list[float]:
    """Estimate stochastic-policy pass rates with fixed common random numbers."""
    pass_rates = []
    for task_id, ratio in enumerate(ratios):
        np.random.seed(seed + 1000 + task_id)
        policy_rng = torch.Generator(device="cpu").manual_seed(seed + 10_000 + task_id)
        env = _make_env(
            ratio,
            num_episodes,
            horizon_seconds,
            reset_mode=reset_mode,
            success_delay_seconds=success_delay_seconds,
            still_steps_needed=still_steps_needed,
        )
        max_steps = int(round(horizon_seconds / env.cfg.ctrl_dt))
        finished = np.zeros(num_episodes, dtype=bool)
        successes = np.zeros(num_episodes, dtype=bool)
        try:
            for _ in range(max_steps):
                assert env.state is not None
                obs = torch.as_tensor(env.state.obs["obs"], dtype=torch.float32)
                actions, _ = actor.sample(obs, policy_rng)
                action_np = actions.numpy()
                action_np[finished] = 0.0
                state = env.step(action_np)
                done = np.asarray(state.terminated | state.truncated, dtype=bool)
                newly_done = done & ~finished
                successes[newly_done] = state.terminated[newly_done] & (
                    state.reward[newly_done] > 0
                )
                finished[newly_done] = True
                if np.all(finished):
                    break
        finally:
            env.close()
        if not np.all(finished):
            raise RuntimeError("evaluation episode did not complete")
        pass_rates.append(float(successes.mean()))
    return pass_rates


def _normalized_auc(checkpoints: list[dict]) -> float:
    if len(checkpoints) < 2:
        return float(checkpoints[0]["mean_pass_rate"])
    x = np.asarray([point["backend_env_steps"] for point in checkpoints], dtype=float)
    y = np.asarray([point["mean_pass_rate"] for point in checkpoints], dtype=float)
    area = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))
    return area / float(x[-1] - x[0])


def run_arm(args: argparse.Namespace, arm: Arm) -> dict:
    torch.manual_seed(args.seed)
    actor = make_actor(args)
    initial_state = copy.deepcopy(actor.state_dict())
    # The explicit copy documents pairing; every arm is reinitialized from the
    # same seed/state even when several arms run in one process.
    actor.load_state_dict(initial_state)
    optimizer = torch.optim.SGD(actor.parameters(), lr=args.learning_rate)
    action_rng = torch.Generator(device="cpu").manual_seed(args.seed + 20_000)
    sampler = TaskSampler(arm, len(args.ratios), args.group_size, args.seed + 30_000)

    checkpoints = []
    records = []
    backend_env_steps = 0
    active_episode_steps = 0
    policy_updates = 0
    wall_start = time.perf_counter()

    def record_evaluation(group: int) -> None:
        rates = evaluate_actor(
            actor,
            ratios=args.ratios,
            num_episodes=args.eval_episodes,
            horizon_seconds=args.horizon_seconds,
            seed=args.eval_seed,
            reset_mode=args.reset_mode,
            success_delay_seconds=args.success_delay_seconds,
            still_steps_needed=args.still_steps_needed,
        )
        checkpoints.append(
            {
                "group": group,
                "backend_env_steps": backend_env_steps,
                "active_episode_steps": active_episode_steps,
                "task_pass_rates": rates,
                "mean_pass_rate": float(np.mean(rates)),
                "tracker_pass_rates": sampler.tracker.pass_rate_estimates().tolist(),
                "mean_based_sampling_distribution": sampler.mean_based_distribution().tolist(),
            }
        )

    record_evaluation(0)
    for group_index in range(args.groups):
        task_id = sampler.sample(group_index)
        batch = collect_group(
            actor,
            ratio=args.ratios[task_id],
            group_size=args.group_size,
            horizon_seconds=args.horizon_seconds,
            env_seed=args.seed * 1_000_000 + group_index,
            action_generator=action_rng,
            reset_mode=args.reset_mode,
            success_delay_seconds=args.success_delay_seconds,
            still_steps_needed=args.still_steps_needed,
        )
        backend_env_steps += batch.backend_env_steps
        active_episode_steps += batch.active_episode_steps
        sampler.observe(task_id, batch.rewards)
        weights_np = maxrl_weights(batch.rewards)
        coefficient_mass = float(np.abs(weights_np).sum())
        k = int(batch.rewards.sum())
        grad_norm = 0.0
        loss_value = 0.0
        if coefficient_mass > 0.0:
            optimizer.zero_grad(set_to_none=True)
            weights = torch.as_tensor(weights_np, dtype=torch.float32)
            loss = -(weights * batch.trajectory_scores).sum()
            loss.backward()
            grad_sq = sum(
                float(parameter.grad.detach().square().sum())
                for parameter in actor.parameters()
                if parameter.grad is not None
            )
            grad_norm = math.sqrt(grad_sq)
            if not math.isfinite(grad_norm):
                raise FloatingPointError("non-finite exact grouped actor gradient")
            optimizer.step()
            policy_updates += 1
            loss_value = float(loss.detach())

        records.append(
            {
                "group": group_index + 1,
                "task_id": task_id,
                "ratio": args.ratios[task_id],
                "successes": k,
                "outcome": "all_fail" if k == 0 else "all_pass" if k == args.group_size else "mixed",
                "coefficient_mass": coefficient_mass,
                "gradient_norm": grad_norm,
                "loss": loss_value,
                "backend_env_steps": backend_env_steps,
                "active_episode_steps": active_episode_steps,
            }
        )
        if (group_index + 1) % args.eval_every == 0 or group_index + 1 == args.groups:
            record_evaluation(group_index + 1)

    outcomes = [record["outcome"] for record in records]
    parameter_delta_sq = sum(
        float((value.detach() - initial_state[name]).square().sum())
        for name, value in actor.state_dict().items()
    )
    return {
        "arm": arm,
        "checkpoints": checkpoints,
        "groups": records,
        "summary": {
            "normalized_transition_auc": _normalized_auc(checkpoints),
            "final_mean_pass_rate": checkpoints[-1]["mean_pass_rate"],
            "policy_updates": policy_updates,
            "all_fail_groups": outcomes.count("all_fail"),
            "mixed_groups": outcomes.count("mixed"),
            "all_pass_groups": outcomes.count("all_pass"),
            "realized_coefficient_mass": float(
                sum(record["coefficient_mass"] for record in records)
            ),
            "backend_env_steps": backend_env_steps,
            "active_episode_steps": active_episode_steps,
            "parameter_l2_displacement": math.sqrt(parameter_delta_sq),
            "wall_seconds": time.perf_counter() - wall_start,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("uniform", "learnability", "advmass"),
        default=["uniform", "learnability", "advmass"],
    )
    parser.add_argument("--ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS))
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--groups", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--horizon-seconds", type=float, default=0.2)
    parser.add_argument(
        "--reset-mode",
        choices=("disk", "fixed_radius"),
        default="disk",
        help="stock uniform disk or a task-identifiable exact radius with random angle",
    )
    parser.add_argument("--success-delay-seconds", type=float, default=0.0)
    parser.add_argument("--still-steps-needed", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=91_000)
    parser.add_argument(
        "--rsl-warm-start",
        type=Path,
        help="optional RSL-RL Stewart actor checkpoint used identically by every arm",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.ratios = tuple(float(value) for value in args.ratios)
    if args.group_size < 2 or args.groups < 0 or args.eval_every <= 0:
        parser.error("group-size must be >=2; groups must be nonnegative; eval-every positive")
    if args.eval_episodes <= 0 or args.horizon_seconds <= 0 or args.learning_rate <= 0:
        parser.error("evaluation count, horizon, and learning rate must be positive")
    if args.success_delay_seconds < 0 or args.still_steps_needed <= 0:
        parser.error("success delay must be nonnegative and still steps must be positive")
    if args.success_delay_seconds >= args.horizon_seconds:
        parser.error("success delay must be shorter than the episode horizon")
    if any(not 0.0 <= ratio <= 1.0 for ratio in args.ratios):
        parser.error("ratios must be in [0, 1]")
    if args.rsl_warm_start is not None and not args.rsl_warm_start.is_file():
        parser.error("rsl-warm-start must be an existing checkpoint")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    score_check_norm = score_function_regression_check()
    warm_start = None
    if args.rsl_warm_start is not None:
        _, warm_start = load_rsl_actor(args.rsl_warm_start)
    result = {
        "status": "development exact-grouped mechanism pilot; not confirmatory",
        "task": "UniLab StewartBalance",
        "backend": "motrix CPU",
        "estimator": "practical D_N; exact order N-1 true trajectory score",
        "hindsight": False,
        "dense_reward_actor_term": False,
        "action_score": "detached pre-clip Gaussian latent action",
        "score_function_regression_net_gradient_norm": score_check_norm,
        "warm_start": warm_start,
        "adaptive_sampling_changes_task_mixture": True,
        "config": {
            "arms": args.arms,
            "ratios": args.ratios,
            "group_size": args.group_size,
            "groups": args.groups,
            "eval_every": args.eval_every,
            "eval_episodes_per_task": args.eval_episodes,
            "horizon_seconds": args.horizon_seconds,
            "reset_mode": args.reset_mode,
            "success_delay_seconds": args.success_delay_seconds,
            "still_steps_needed": args.still_steps_needed,
            "learning_rate": args.learning_rate,
            "optimizer": "SGD",
            "seed": args.seed,
            "eval_seed": args.eval_seed,
        },
        "runs": [run_arm(args, arm) for arm in args.arms],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
