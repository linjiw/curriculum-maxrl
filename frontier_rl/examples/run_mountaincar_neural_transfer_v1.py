"""Development runner for the independent MountainCar neural transfer V1 study.

The default action is schedule inspection only.  Reserved development seeds
can run only against an exact development-only source/runtime/protocol lock.
Confirmatory seeds have no execution path in this version.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import gymnasium
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontier_rl.examples.mountaincar_neural_transfer_v1_core import (  # noqa: E402
    MAX_EPISODE_STEPS,
    N_ACTIONS,
    N_ROLLOUTS,
    N_TASKS,
    THRESHOLDS,
    EVALUATION_ACTION_SEED_OFFSET,
    TRAINING_ACTION_SEED_OFFSET,
    TRAINING_EPISODE_SEED_OFFSET,
    MountainCarNeuralActor,
    MountainCarSparseGoalSpace,
    _issue_registered_seed_authorization,
    practical_maxrl_weights,
)


SCHEMA = "mountaincar-neural-transfer-v1r2-development"
LOCK_SCHEMA = "mountaincar-neural-transfer-v1r2-development-lock"
STUDY = "mountaincar_neural_transfer_v1"
DEVELOPMENT_SEEDS = (17_000, 17_001, 17_002)
CONFIRMATORY_SEEDS = tuple(range(18_000, 18_020))
SMOKE_SEED = 9_053
TRANSITION_BUDGET = 500_000
EVAL_INTERVAL_TRANSITIONS = 100_000
EVAL_EPISODES_PER_TASK = 32
LEARNING_RATE = 3e-4
TEACHER_DECAY = 0.7
TEACHER_FLOOR = 0.1
TEACHER_GAMMA = 4.0
MAX_COMPLETE_GROUP_OVERSHOOT = N_ROLLOUTS * MAX_EPISODE_STEPS
MAX_GROUPS_FOR_BUDGET = math.ceil(TRANSITION_BUDGET / N_ROLLOUTS)
TEACHER_SEED_BASE = 4_000_000
EVALUATION_SEED_BASE = 6_000_000
EVALUATION_ACTION_SEED_BASE = EVALUATION_SEED_BASE + EVALUATION_ACTION_SEED_OFFSET


@dataclass(frozen=True)
class Condition:
    name: str
    sampling: str
    architecture: str


CONDITIONS = (
    Condition("frontier_shared_h64", "frontier_u16", MountainCarNeuralActor.SHARED),
    Condition("uniform_shared_h64", "uniform", MountainCarNeuralActor.SHARED),
    Condition("hardest_shared_h64", "hardest_only", MountainCarNeuralActor.SHARED),
    Condition(
        "uniform_disjoint_total_h8x8",
        "uniform",
        MountainCarNeuralActor.DISJOINT_TOTAL,
    ),
    Condition(
        "uniform_disjoint_active_h64x8",
        "uniform",
        MountainCarNeuralActor.DISJOINT_ACTIVE,
    ),
)

PAIRED_UNIFORM_CONDITIONS = (
    "uniform_shared_h64",
    "uniform_disjoint_total_h8x8",
    "uniform_disjoint_active_h64x8",
)

SOURCE_RELATIVE_PATHS = (
    "frontier_rl/examples/mountaincar_neural_transfer_v1_core.py",
    "frontier_rl/examples/run_mountaincar_neural_transfer_v1.py",
    "frontier_rl/examples/analyze_mountaincar_neural_transfer_v1.py",
    "frontier_rl/examples/MOUNTAINCAR_NEURAL_TRANSFER_PROTOCOL_V1.md",
    "frontier_rl/examples/test_mountaincar_neural_transfer_v1.py",
    "frontier_rl/examples/test_analyze_mountaincar_neural_transfer_v1.py",
    "frontier_rl/__init__.py",
    "frontier_rl/interfaces.py",
    "frontier_rl/teacher.py",
    "frontier_rl/estimators.py",
    "frontier_rl/trainer.py",
)

CAPACITY_CONTRACT = {
    MountainCarNeuralActor.SHARED: {
        "hidden_size": 64,
        "slots": 1,
        "total_parameters": 384,
        "active_parameters_per_task": 384,
    },
    MountainCarNeuralActor.DISJOINT_TOTAL: {
        "hidden_size": 8,
        "slots": 8,
        "total_parameters": 384,
        "active_parameters_per_task": 48,
    },
    MountainCarNeuralActor.DISJOINT_ACTIVE: {
        "hidden_size": 64,
        "slots": 8,
        "total_parameters": 3072,
        "active_parameters_per_task": 384,
    },
}

# Explicitly registered/executed training blocks already present in this
# repository.  Bootstrap, action, evaluation, and teacher RNG namespaces are
# separate and are not interpreted as training-seed collisions.
PRIOR_TRAINING_SEED_BLOCKS = {
    "legacy_mountaincar": tuple(range(0, 10)),
    "acrobot_v1_core": tuple(range(0, 20)),
    "acrobot_v1_scale": tuple(range(100, 110)),
    "acrobot_v1_pilot": tuple(range(10_000, 10_003)),
    "acrobot_v2_development": tuple(range(11_000, 11_003)),
    "acrobot_v3_confirmatory": tuple(range(12_000, 12_020)),
    "acrobot_hindsight_v4a": tuple(range(13_000, 13_003)),
    "acrobot_hindsight_v4b": tuple(range(14_000, 14_010)),
    "acrobot_hindsight_v5a": tuple(range(15_000, 15_003)),
    "acrobot_hindsight_v5b": tuple(range(16_000, 16_020)),
}


def _canonical_hash(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths() -> tuple[Path, ...]:
    if len(SOURCE_RELATIVE_PATHS) != len(set(SOURCE_RELATIVE_PATHS)):
        raise RuntimeError("V1 source manifest contains duplicate paths")
    root = PROJECT_ROOT.resolve()
    paths = tuple((root / relative).resolve() for relative in SOURCE_RELATIVE_PATHS)
    for path in paths:
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"V1 source path escapes project root: {path}"
            ) from error
    return paths


def _source_hashes() -> dict[str, str]:
    hashes = {}
    for path in _source_paths():
        if not path.is_file():
            raise RuntimeError(f"required V1 source is missing: {path}")
        hashes[str(path.relative_to(PROJECT_ROOT.resolve()))] = _sha256(path)
    return hashes


def _runtime() -> dict:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "gymnasium_version": gymnasium.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def seed_collision_audit() -> dict:
    development = set(DEVELOPMENT_SEEDS)
    confirmatory = set(CONFIRMATORY_SEEDS)
    prior = set().union(
        *(set(values) for values in PRIOR_TRAINING_SEED_BLOCKS.values())
    )
    new_training = development | confirmatory
    rng_blocks = {
        "actor_action": sorted(
            seed + TRAINING_ACTION_SEED_OFFSET for seed in new_training
        ),
        "environment_episode": sorted(
            seed + TRAINING_EPISODE_SEED_OFFSET for seed in new_training
        ),
        "teacher": sorted(seed + TEACHER_SEED_BASE for seed in new_training),
        "evaluation_episode": sorted(
            seed + EVALUATION_SEED_BASE for seed in new_training
        ),
        "evaluation_action": sorted(
            seed + EVALUATION_ACTION_SEED_BASE for seed in new_training
        ),
    }
    collisions = {
        "development_vs_confirmatory": sorted(development & confirmatory),
        "development_vs_prior": sorted(development & prior),
        "confirmatory_vs_prior": sorted(confirmatory & prior),
        "rng_roots_vs_training": {
            key: sorted(set(values) & (prior | new_training))
            for key, values in rng_blocks.items()
        },
        "between_rng_root_blocks": {
            f"{left}_vs_{right}": sorted(set(rng_blocks[left]) & set(rng_blocks[right]))
            for index, left in enumerate(rng_blocks)
            for right in tuple(rng_blocks)[index + 1 :]
        },
    }
    no_collisions = (
        not collisions["development_vs_confirmatory"]
        and not collisions["development_vs_prior"]
        and not collisions["confirmatory_vs_prior"]
        and not any(collisions["rng_roots_vs_training"].values())
        and not any(collisions["between_rng_root_blocks"].values())
    )
    passed = (
        len(development) == len(DEVELOPMENT_SEEDS)
        and len(confirmatory) == len(CONFIRMATORY_SEEDS)
        and no_collisions
    )
    result = {
        "passed": passed,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmatory_seeds_reserved_untouched": list(CONFIRMATORY_SEEDS),
        "prior_training_seed_blocks": {
            key: list(values) for key, values in PRIOR_TRAINING_SEED_BLOCKS.items()
        },
        "derived_rng_root_blocks": rng_blocks,
        "collisions": collisions,
        "rng_namespaces": {
            "training_seed": "reserved block value",
            "actor_action_seed": f"training seed + {TRAINING_ACTION_SEED_OFFSET}",
            "environment_episode_seed_stream": (
                f"training seed + {TRAINING_EPISODE_SEED_OFFSET}"
            ),
            "teacher_seed": f"training seed + {TEACHER_SEED_BASE}",
            "evaluation_episode_seed_root": (f"training seed + {EVALUATION_SEED_BASE}"),
            "evaluation_action_seed_root": (
                f"training seed + {EVALUATION_ACTION_SEED_BASE}"
            ),
        },
    }
    if not passed:
        raise RuntimeError(f"MountainCar V1 seed collision: {collisions}")
    return result


class FrontierU16Sampler:
    """Independent implementation of the frozen frontier-u16 sampler."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(int(seed))
        self.alpha = np.ones(N_TASKS, dtype=np.float64)
        self.beta = np.ones(N_TASKS, dtype=np.float64)
        self.visits = np.zeros(N_TASKS, dtype=np.int64)

    def draw(self) -> tuple[int, np.ndarray]:
        sampled_pass = self.rng.beta(self.alpha, self.beta)
        utility = (
            np.maximum(
                (1.0 - (1.0 - sampled_pass) ** N_ROLLOUTS) - sampled_pass,
                0.0,
            )
            ** TEACHER_GAMMA
        )
        if float(utility.sum()) <= 1e-12:
            utility = np.ones(N_TASKS, dtype=np.float64)
        focused = utility / utility.sum()
        probabilities = (1.0 - TEACHER_FLOOR) * focused + TEACHER_FLOOR / N_TASKS
        task = int(self.rng.choice(N_TASKS, p=probabilities))
        return task, probabilities

    def observe(self, task_id: int, rewards: Sequence[float]) -> None:
        values = np.asarray(rewards, dtype=np.float64)
        successes = float(values.sum())
        self.alpha[task_id] = (
            1.0 + (self.alpha[task_id] - 1.0) * TEACHER_DECAY + successes
        )
        self.beta[task_id] = (
            1.0 + (self.beta[task_id] - 1.0) * TEACHER_DECAY + len(values) - successes
        )
        self.visits[task_id] += 1


class FixedSampler:
    def __init__(self, sampling: str, seed: int):
        if sampling not in ("uniform", "hardest_only"):
            raise ValueError(f"unknown fixed sampler {sampling!r}")
        self.sampling = sampling
        self.rng = np.random.default_rng(int(seed))

    def draw(self) -> tuple[int, np.ndarray]:
        if self.sampling == "hardest_only":
            probabilities = np.zeros(N_TASKS, dtype=np.float64)
            probabilities[-1] = 1.0
            return N_TASKS - 1, probabilities
        probabilities = np.full(N_TASKS, 1.0 / N_TASKS, dtype=np.float64)
        return int(self.rng.choice(N_TASKS, p=probabilities)), probabilities

    def observe(self, task_id: int, rewards: Sequence[float]) -> None:
        del task_id, rewards


def _sampler(condition: Condition, seed: int):
    sampler_seed = int(seed) + TEACHER_SEED_BASE
    if condition.sampling == "frontier_u16":
        return FrontierU16Sampler(sampler_seed)
    return FixedSampler(condition.sampling, sampler_seed)


@lru_cache(maxsize=None)
def _uniform_task_sequence(
    seed: int, length: int = MAX_GROUPS_FOR_BUDGET
) -> tuple[int, ...]:
    """Outcome-independent uniform schedule indexed only by seed and group."""

    if int(length) <= 0:
        raise ValueError("uniform schedule length must be positive")
    sampler = FixedSampler("uniform", int(seed) + TEACHER_SEED_BASE)
    return tuple(sampler.draw()[0] for _ in range(int(length)))


@lru_cache(maxsize=None)
def _uniform_task_schedule_sha256(
    seed: int, length: int = MAX_GROUPS_FOR_BUDGET
) -> str:
    return _canonical_hash(list(_uniform_task_sequence(seed, length)))


def fixed_budget_auc(y: Sequence[float], x: Sequence[int], budget: int) -> float:
    """Normalized piecewise-linear AUC on the exact interval ``[0, budget]``."""

    values = np.asarray(y, dtype=np.float64)
    coordinates = np.asarray(x, dtype=np.float64)
    if (
        values.ndim != 1
        or coordinates.ndim != 1
        or len(values) != len(coordinates)
        or len(values) < 2
        or coordinates[0] != 0.0
        or np.any(np.diff(coordinates) <= 0.0)
        or coordinates[-1] < budget
        or budget <= 0
        or not np.isfinite(values).all()
    ):
        raise ValueError("invalid curve for fixed-budget transition AUC")
    boundary_index = int(np.searchsorted(coordinates, budget, side="left"))
    if coordinates[boundary_index] == budget:
        clipped_x = coordinates[: boundary_index + 1]
        clipped_y = values[: boundary_index + 1]
    else:
        left = boundary_index - 1
        fraction = (budget - coordinates[left]) / (
            coordinates[boundary_index] - coordinates[left]
        )
        boundary_value = values[left] + fraction * (
            values[boundary_index] - values[left]
        )
        clipped_x = np.concatenate((coordinates[:boundary_index], [float(budget)]))
        clipped_y = np.concatenate((values[:boundary_index], [boundary_value]))
    area = float(np.sum(np.diff(clipped_x) * (clipped_y[1:] + clipped_y[:-1]) / 2.0))
    return area / float(budget)


def fixed_budget_value(y: Sequence[float], x: Sequence[int], budget: int) -> float:
    values = np.asarray(y, dtype=np.float64)
    coordinates = np.asarray(x, dtype=np.float64)
    if len(values) != len(coordinates) or coordinates[-1] < budget:
        raise ValueError("curve does not bracket the fixed transition budget")
    return float(np.interp(float(budget), coordinates, values))


def _safe_evaluate(space: MountainCarSparseGoalSpace, *, n: int, seed: int) -> dict:
    result = space.evaluate(n=n, seed=seed)
    if not all(
        result.get(key) is True
        for key in (
            "training_episode_rng_preserved",
            "training_action_rng_preserved",
            "training_parameters_preserved",
        )
    ):
        raise RuntimeError("evaluation changed MountainCar training state")
    return result


def run_condition(
    condition: Condition,
    seed: int,
    *,
    transition_budget: int = TRANSITION_BUDGET,
    eval_interval_transitions: int = EVAL_INTERVAL_TRANSITIONS,
    eval_n: int = EVAL_EPISODES_PER_TASK,
    learning_rate: float = LEARNING_RATE,
    development_lock: Path | None = None,
) -> dict:
    if type(seed) is not int:
        raise TypeError("V1 seed must be a primitive Python int")
    normalized_seed = seed
    registered_seed_authorization = None
    if normalized_seed in CONFIRMATORY_SEEDS:
        raise RuntimeError(
            "confirmatory seeds are reserved and have no V1 execution path"
        )
    if normalized_seed in DEVELOPMENT_SEEDS:
        if condition not in CONDITIONS:
            raise RuntimeError(
                "registered development seed requires a locked V1 condition"
            )
        if (
            transition_budget != TRANSITION_BUDGET
            or eval_interval_transitions != EVAL_INTERVAL_TRANSITIONS
            or eval_n != EVAL_EPISODES_PER_TASK
            or learning_rate != LEARNING_RATE
        ):
            raise RuntimeError(
                "registered development seed requires the locked V1 config"
            )
        if development_lock is None:
            raise RuntimeError("reserved development seed requires an exact V1 lock")
        source_lock = _load_development_lock(development_lock)
        registered_seed_authorization = _issue_registered_seed_authorization(
            seed=normalized_seed,
            development_lock_sha256=source_lock["file_sha256"],
        )
    if (
        transition_budget <= 0
        or eval_interval_transitions <= MAX_COMPLETE_GROUP_OVERSHOOT
    ):
        raise ValueError(
            "budget must be positive and eval interval must exceed one group"
        )
    if transition_budget % eval_interval_transitions != 0:
        raise ValueError("transition budget must be an exact multiple of eval interval")

    actor = MountainCarNeuralActor(
        mode=condition.architecture,
        learning_rate=learning_rate,
        parameter_seed=normalized_seed,
        action_seed=normalized_seed + TRAINING_ACTION_SEED_OFFSET,
    )
    space = MountainCarSparseGoalSpace(
        actor=actor,
        seed=normalized_seed,
        registered_seed_authorization=registered_seed_authorization,
    )
    sampler = _sampler(condition, normalized_seed)
    evaluation_seed = EVALUATION_SEED_BASE + normalized_seed
    started = time.perf_counter()

    transitions = 0
    sampled_groups = 0
    optimizer_updates = 0
    live_applied_updates = 0
    zero_gradient_update_attempts = 0
    dead_groups = 0
    mixed_groups = 0
    all_pass_groups = 0
    task_groups = np.zeros(N_TASKS, dtype=np.int64)
    task_rollouts = np.zeros(N_TASKS, dtype=np.int64)
    task_successes = np.zeros(N_TASKS, dtype=np.int64)
    task_transitions = np.zeros(N_TASKS, dtype=np.int64)
    group_diagnostics: list[dict] = []
    update_diagnostics: list[dict] = []
    zero_gradient_diagnostics: list[dict] = []
    x_transitions: list[int] = [0]
    pass_rate_curve: list[list[float]] = []
    mean_pass_curve: list[float] = []
    hardest_pass_curve: list[float] = []
    evaluation_rng_preserved: list[bool] = []
    shared_nested_evaluations: list[bool | None] = []
    evaluation_trigger_transitions: list[int] = [0]
    evaluation_policy_sources: list[str] = ["initial"]
    evaluation_policy_parameter_sha256: list[str] = []
    evaluation_max_position_samples: list[list[list[float]]] = []
    evaluation_episode_seeds: list[int] = []
    evaluation_action_seeds: list[int] = []
    wall_seconds_at_evaluations: list[float] = [0.0]
    transition_at_budget_crossing: int | None = None
    budget_crossing_group: int | None = None

    try:
        initial = _safe_evaluate(space, n=eval_n, seed=evaluation_seed)
        pass_rate_curve.append(initial["pass_rates"])
        mean_pass_curve.append(float(initial["mean_pass"]))
        hardest_pass_curve.append(float(initial["hardest_pass"]))
        evaluation_rng_preserved.append(True)
        shared_nested_evaluations.append(initial["shared_nested_pass_rates"])
        evaluation_policy_parameter_sha256.append(initial["evaluated_parameter_sha256"])
        evaluation_max_position_samples.append(initial["max_position_samples"])
        evaluation_episode_seeds = initial["evaluation_episode_seeds"]
        evaluation_action_seeds = initial["evaluation_action_seeds"]
        next_evaluation = eval_interval_transitions

        while transitions < transition_budget:
            task_id, probabilities = sampler.draw()
            transition_start = transitions
            parameter_state_before_group = actor.parameter_state()
            parameter_sha256_before_group = actor.parameter_sha256()
            group = space.rollout_group(task_id, N_ROLLOUTS)
            group_transitions = group.transitions
            if not N_ROLLOUTS <= group_transitions <= MAX_COMPLETE_GROUP_OVERSHOOT:
                raise RuntimeError("complete MountainCar group transition bound failed")
            transitions += group_transitions
            sampled_groups += 1
            if (
                transition_at_budget_crossing is None
                and transitions >= transition_budget
            ):
                transition_at_budget_crossing = transitions
                budget_crossing_group = sampled_groups
            successes = int(group.rewards.sum())
            if not 0 <= successes <= N_ROLLOUTS:
                raise RuntimeError("invalid binary success count")
            sampler.observe(task_id, group.rewards)

            task_groups[task_id] += 1
            task_rollouts[task_id] += N_ROLLOUTS
            task_successes[task_id] += successes
            task_transitions[task_id] += group_transitions
            regime = (
                "dead"
                if successes == 0
                else "all_pass" if successes == N_ROLLOUTS else "mixed"
            )
            update_source = None
            if regime == "dead":
                dead_groups += 1
            elif regime == "all_pass":
                all_pass_groups += 1
            else:
                mixed_groups += 1
                weights = practical_maxrl_weights(group.rewards)
                diagnostics = actor.update(task_id, group.trajectories, weights)
                record = {
                    "after_group": sampled_groups,
                    "transitions": transitions,
                    "source": "requested_live",
                    "requested_task": task_id,
                    "credited_task": task_id,
                    **diagnostics,
                }
                if diagnostics["applied"]:
                    optimizer_updates += 1
                    live_applied_updates += 1
                    update_source = "requested_live"
                    record["optimizer_update"] = optimizer_updates
                    update_diagnostics.append(record)
                else:
                    zero_gradient_update_attempts += 1
                    zero_gradient_diagnostics.append(record)

            parameter_sha256_after_group = actor.parameter_sha256()

            group_diagnostics.append(
                {
                    "group": sampled_groups,
                    "transition_start": transition_start,
                    "transition_end": transitions,
                    "n_transitions": group_transitions,
                    "task_id": task_id,
                    "success_count": successes,
                    "regime": regime,
                    "sampling_probabilities": probabilities.tolist(),
                    "sampled_task_probability": float(probabilities[task_id]),
                    "optimizer_updates_after_group": optimizer_updates,
                    "update_source": update_source,
                    "parameter_sha256_before_group": parameter_sha256_before_group,
                    "parameter_sha256_after_group": parameter_sha256_after_group,
                    "rollouts": [dict(info) for info in group.infos],
                }
            )

            if next_evaluation <= transition_budget and transitions >= next_evaluation:
                if transitions >= next_evaluation + eval_interval_transitions:
                    raise RuntimeError(
                        "one complete group crossed multiple checkpoints"
                    )
                checkpoint = next_evaluation
                use_pre_group_policy = transitions > checkpoint
                if use_pre_group_policy:
                    parameter_state_after_group = actor.parameter_state()
                    actor.load_parameter_state(parameter_state_before_group)
                    try:
                        evaluated = _safe_evaluate(
                            space, n=eval_n, seed=evaluation_seed
                        )
                    finally:
                        actor.load_parameter_state(parameter_state_after_group)
                    if actor.parameter_sha256() != parameter_sha256_after_group:
                        raise RuntimeError(
                            "checkpoint evaluation did not restore live policy"
                        )
                    policy_source = "pre_crossing_group"
                    expected_evaluation_hash = parameter_sha256_before_group
                else:
                    evaluated = _safe_evaluate(space, n=eval_n, seed=evaluation_seed)
                    policy_source = "post_exact_group"
                    expected_evaluation_hash = parameter_sha256_after_group
                if evaluated["evaluated_parameter_sha256"] != expected_evaluation_hash:
                    raise RuntimeError(
                        "checkpoint evaluation used the wrong policy snapshot"
                    )
                x_transitions.append(checkpoint)
                evaluation_trigger_transitions.append(transitions)
                evaluation_policy_sources.append(policy_source)
                evaluation_policy_parameter_sha256.append(expected_evaluation_hash)
                if evaluated["evaluation_episode_seeds"] != evaluation_episode_seeds:
                    raise RuntimeError(
                        "evaluation episode CRNs changed across checkpoints"
                    )
                if evaluated["evaluation_action_seeds"] != evaluation_action_seeds:
                    raise RuntimeError(
                        "evaluation action CRNs changed across checkpoints"
                    )
                evaluation_max_position_samples.append(
                    evaluated["max_position_samples"]
                )
                pass_rate_curve.append(evaluated["pass_rates"])
                mean_pass_curve.append(float(evaluated["mean_pass"]))
                hardest_pass_curve.append(float(evaluated["hardest_pass"]))
                evaluation_rng_preserved.append(True)
                shared_nested_evaluations.append(evaluated["shared_nested_pass_rates"])
                wall_seconds_at_evaluations.append(time.perf_counter() - started)
                next_evaluation += eval_interval_transitions
    finally:
        space.close()

    if transition_at_budget_crossing is None or budget_crossing_group is None:
        raise RuntimeError("run ended before the transition budget")

    wall_seconds = time.perf_counter() - started
    mean_auc = fixed_budget_auc(mean_pass_curve, x_transitions, transition_budget)
    hardest_auc = fixed_budget_auc(hardest_pass_curve, x_transitions, transition_budget)
    final_mean = fixed_budget_value(mean_pass_curve, x_transitions, transition_budget)
    final_hardest = fixed_budget_value(
        hardest_pass_curve, x_transitions, transition_budget
    )
    trace_projection = [
        {
            "group": record["group"],
            "transition_end": record["transition_end"],
            "task_id": record["task_id"],
            "success_count": record["success_count"],
            "optimizer_updates_after_group": record["optimizer_updates_after_group"],
        }
        for record in group_diagnostics
    ]
    actor_diagnostics = actor.diagnostics()
    numeric_values = np.asarray(
        [
            *mean_pass_curve,
            *hardest_pass_curve,
            mean_auc,
            hardest_auc,
            final_mean,
            final_hardest,
            actor_diagnostics["parameter_norm"],
            wall_seconds,
        ],
        dtype=np.float64,
    )
    return {
        "seed": normalized_seed,
        "condition": condition.name,
        "sampling": condition.sampling,
        "architecture": condition.architecture,
        "hindsight": False,
        "relabel_candidates": 0,
        "relabeled_groups": 0,
        "transition_budget": int(transition_budget),
        "transitions": transitions,
        "transition_at_budget_crossing": transition_at_budget_crossing,
        "budget_crossing_group": budget_crossing_group,
        "complete_group_overshoot": transition_at_budget_crossing - transition_budget,
        "post_budget_alignment_groups": sampled_groups - budget_crossing_group,
        "post_budget_alignment_transitions": transitions
        - transition_at_budget_crossing,
        "reached_transition_budget": transitions >= transition_budget,
        "sampled_groups": sampled_groups,
        "optimizer_updates": optimizer_updates,
        "live_applied_updates": live_applied_updates,
        "zero_gradient_update_attempts": zero_gradient_update_attempts,
        "dead_groups": dead_groups,
        "mixed_groups": mixed_groups,
        "all_pass_groups": all_pass_groups,
        "task_groups": task_groups.astype(int).tolist(),
        "task_rollouts": task_rollouts.astype(int).tolist(),
        "task_successes": task_successes.astype(int).tolist(),
        "task_transitions": task_transitions.astype(int).tolist(),
        "group_diagnostics": group_diagnostics,
        "task_sequence": [record["task_id"] for record in group_diagnostics],
        "registered_uniform_task_schedule_length": (
            math.ceil(transition_budget / N_ROLLOUTS)
            if condition.sampling == "uniform"
            else None
        ),
        "registered_uniform_task_schedule_sha256": (
            _uniform_task_schedule_sha256(
                normalized_seed, math.ceil(transition_budget / N_ROLLOUTS)
            )
            if condition.sampling == "uniform"
            else None
        ),
        "update_diagnostics": update_diagnostics,
        "zero_gradient_diagnostics": zero_gradient_diagnostics,
        "x_transitions": x_transitions,
        "pass_rate_curve": pass_rate_curve,
        "mean_pass_curve": mean_pass_curve,
        "hardest_pass_curve": hardest_pass_curve,
        "evaluation_rng_preserved": evaluation_rng_preserved,
        "shared_nested_evaluations": shared_nested_evaluations,
        "evaluation_trigger_transitions": evaluation_trigger_transitions,
        "evaluation_policy_sources": evaluation_policy_sources,
        "evaluation_policy_parameter_sha256": evaluation_policy_parameter_sha256,
        "evaluation_max_position_samples": evaluation_max_position_samples,
        "evaluation_episode_seeds": evaluation_episode_seeds,
        "evaluation_action_seeds": evaluation_action_seeds,
        "wall_seconds_at_evaluations": wall_seconds_at_evaluations,
        "auc_mean_pass_fixed_transitions": mean_auc,
        "auc_hardest_pass_fixed_transitions": hardest_auc,
        "final_mean_pass_at_budget": final_mean,
        "final_hardest_pass_at_budget": final_hardest,
        "initial_pass_rates": pass_rate_curve[0],
        "actor": actor_diagnostics,
        "training_group_trace_sha256": _canonical_hash(trace_projection),
        "wall_seconds": wall_seconds,
        "numeric_valid": bool(np.isfinite(numeric_values).all()),
    }


def summarize_runs(runs: Sequence[dict]) -> dict:
    metrics = (
        "auc_mean_pass_fixed_transitions",
        "auc_hardest_pass_fixed_transitions",
        "final_mean_pass_at_budget",
        "final_hardest_pass_at_budget",
        "transitions",
        "optimizer_updates",
        "wall_seconds",
    )
    summary = {"n_runs": len(runs)}
    for metric in metrics:
        values = np.asarray([run[metric] for run in runs], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()) if len(values) else None,
            "sample_std": (
                float(values.std(ddof=1))
                if len(values) > 1
                else 0.0 if len(values) else None
            ),
            "per_seed": values.tolist(),
        }
    return summary


DEVELOPMENT_CONTRASTS = {
    "frontier_minus_uniform_shared": (
        "frontier_shared_h64",
        "uniform_shared_h64",
    ),
    "frontier_minus_hardest_shared": (
        "frontier_shared_h64",
        "hardest_shared_h64",
    ),
    "shared_minus_disjoint_total_under_uniform": (
        "uniform_shared_h64",
        "uniform_disjoint_total_h8x8",
    ),
    "shared_minus_disjoint_active_under_uniform": (
        "uniform_shared_h64",
        "uniform_disjoint_active_h64x8",
    ),
}


def _development_descriptives(cases: dict) -> dict:
    output = {}
    for name, (left_name, right_name) in DEVELOPMENT_CONTRASTS.items():
        left, right = cases[left_name]["runs"], cases[right_name]["runs"]
        if [run["seed"] for run in left] != [run["seed"] for run in right]:
            raise RuntimeError(f"unpaired development contrast {name}")
        primary_differences = [
            float(a["auc_hardest_pass_fixed_transitions"])
            - float(b["auc_hardest_pass_fixed_transitions"])
            for a, b in zip(left, right)
        ]
        supporting_differences = [
            float(a["auc_mean_pass_fixed_transitions"])
            - float(b["auc_mean_pass_fixed_transitions"])
            for a, b in zip(left, right)
        ]
        output[name] = {
            "primary_metric": "auc_hardest_pass_fixed_transitions",
            "supporting_metric": "auc_mean_pass_fixed_transitions",
            "estimand": f"{left_name} - {right_name}",
            "primary_per_seed_delta": primary_differences,
            "primary_mean_delta": float(np.mean(primary_differences)),
            "supporting_per_seed_delta": supporting_differences,
            "supporting_mean_delta": float(np.mean(supporting_differences)),
            "inference": "descriptive development result; not confirmatory",
        }
    return output


def _development_gates(cases: dict) -> dict:
    all_runs = [run for case in cases.values() for run in case["runs"]]
    complete = len(all_runs) == len(CONDITIONS) * len(DEVELOPMENT_SEEDS)
    valid = complete and all(
        run["numeric_valid"]
        and run["reached_transition_budget"]
        and 0 <= run["complete_group_overshoot"] <= MAX_COMPLETE_GROUP_OVERSHOOT
        and all(run["evaluation_rng_preserved"])
        and run["hindsight"] is False
        and run["relabel_candidates"] == run["relabeled_groups"] == 0
        for run in all_runs
    )
    capacity = complete and all(
        run["actor"]["parameter_count"]
        == CAPACITY_CONTRACT[run["architecture"]]["total_parameters"]
        and run["actor"]["active_parameter_count"]
        == CAPACITY_CONTRACT[run["architecture"]]["active_parameters_per_task"]
        for run in all_runs
    )
    coverage = complete and all(
        (
            set(
                task
                for run in cases[condition.name]["runs"]
                for task, count in enumerate(run["task_groups"])
                if count > 0
            )
            == (
                {N_TASKS - 1}
                if condition.sampling == "hardest_only"
                else set(range(N_TASKS))
            )
        )
        for condition in CONDITIONS
    )
    update_signal = complete and all(
        sum(run["optimizer_updates"] for run in cases[condition.name]["runs"]) > 0
        for condition in CONDITIONS
        if condition.sampling != "hardest_only"
    )
    regimes = {
        key: sum(run[key] for run in all_runs)
        for key in ("dead_groups", "mixed_groups", "all_pass_groups")
    }
    natural_regimes = complete and all(value > 0 for value in regimes.values())
    initial_crn = complete and all(
        all(
            cases[condition.name]["runs"][index]["initial_pass_rates"]
            == cases[CONDITIONS[0].name]["runs"][index]["initial_pass_rates"]
            for condition in CONDITIONS[1:]
        )
        for index in range(len(DEVELOPMENT_SEEDS))
    )
    paired_uniform_schedule = complete
    if paired_uniform_schedule:
        for index, seed in enumerate(DEVELOPMENT_SEEDS):
            uniform_runs = [
                cases[name]["runs"][index] for name in PAIRED_UNIFORM_CONDITIONS
            ]
            sequences = [run["task_sequence"] for run in uniform_runs]
            common_length = min(map(len, sequences))
            expected_hash = _uniform_task_schedule_sha256(seed)
            paired_uniform_schedule = (
                paired_uniform_schedule
                and all(
                    sequence[:common_length] == sequences[0][:common_length]
                    for sequence in sequences[1:]
                )
                and all(
                    run["registered_uniform_task_schedule_length"]
                    == MAX_GROUPS_FOR_BUDGET
                    and run["registered_uniform_task_schedule_sha256"] == expected_hash
                    for run in uniform_runs
                )
            )
    projected_confirmatory_hours = (
        sum(run["wall_seconds"] for run in all_runs)
        / max(len(all_runs), 1)
        * len(CONDITIONS)
        * len(CONFIRMATORY_SEEDS)
        / 3600.0
    )
    runtime = (
        complete
        and math.isfinite(projected_confirmatory_hours)
        and projected_confirmatory_hours <= 18.0
    )
    gates = {
        "complete_exact_matrix": complete,
        "technical_validity": valid,
        "exact_capacity_controls": capacity,
        "task_coverage": coverage,
        "nonzero_update_signal": update_signal,
        "natural_dead_mixed_all_pass_regimes": natural_regimes,
        "initial_common_random_numbers": initial_crn,
        "exact_paired_uniform_schedule_and_realized_common_prefix": (
            paired_uniform_schedule
        ),
        "projected_confirmatory_runtime_at_most_18h": runtime,
        "projected_confirmatory_hours": projected_confirmatory_hours,
        "pooled_regime_counts": regimes,
        "learning_outcomes_authorize_claims": False,
    }
    gates["development_feasible"] = all(
        gates[key]
        for key in (
            "complete_exact_matrix",
            "technical_validity",
            "exact_capacity_controls",
            "task_coverage",
            "nonzero_update_signal",
            "natural_dead_mixed_all_pass_regimes",
            "initial_common_random_numbers",
            "exact_paired_uniform_schedule_and_realized_common_prefix",
            "projected_confirmatory_runtime_at_most_18h",
        )
    )
    return gates


def _protocol_config(
    *, transition_budget: int, eval_interval: int, eval_n: int, learning_rate: float
) -> dict:
    return {
        "gymnasium_environment": "MountainCar-v0",
        "task_definition": "post-transition max position >= threshold",
        "thresholds": list(THRESHOLDS),
        "n_tasks": N_TASKS,
        "n_actions": N_ACTIONS,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "n_rollouts": N_ROLLOUTS,
        "estimator": "practical MaxRL: 1{K>0}(r_i/K - 1/16)",
        "hindsight": False,
        "learning_rate": learning_rate,
        "optimizer": "plain SGD ascent; summed frozen-group score gradient",
        "transition_budget": transition_budget,
        "eval_interval_transitions": eval_interval,
        "eval_episodes_per_task": eval_n,
        "primary_metric": "hardest-goal pass AUC over exact [0, transition_budget]",
        "primary_estimand": (
            "paired mean method difference in hardest-goal pass AUC for each of "
            "the four registered contrasts"
        ),
        "supporting_metric": (
            "target-uniform mean-pass AUC over exact [0, transition_budget]"
        ),
        "claim_rule": (
            "only the four hardest-goal AUC contrasts are confirmatory; mean-pass AUC "
            "is supporting and cannot rescue a failed primary contrast"
        ),
        "paired_uniform_schedule_rule": (
            "all uniform cells use one seed-indexed outcome-independent sequence and "
            "their realized task lists must equal its exact prefix"
        ),
        "uniform_task_schedule": {
            "algorithm": "repeated numpy Generator.choice over 8 tasks with p=1/8",
            "seed_rule": f"training seed + {TEACHER_SEED_BASE}",
            "registered_length": math.ceil(transition_budget / N_ROLLOUTS),
            "sha256_by_development_seed": {
                str(seed): _uniform_task_schedule_sha256(
                    seed, math.ceil(transition_budget / N_ROLLOUTS)
                )
                for seed in DEVELOPMENT_SEEDS
            },
        },
        "checkpoint_policy_rule": (
            "post-group if end == checkpoint; otherwise saved pre-crossing-group parameters"
        ),
        "audit_schema": {
            "training_rollout_fields": [
                "episode_seed",
                "n_steps",
                "max_position",
                "max_position_before_final",
                "pre_final_position",
                "final_position",
                "native_terminated",
                "native_truncated",
                "native_reward_sum",
                "success",
            ],
            "evaluation_fields": [
                "evaluation_episode_seeds",
                "evaluation_action_seeds",
                "evaluation_max_position_samples",
            ],
            "first_hit_rule": (
                "success iff max_position_before_final < requested threshold "
                "<= final_position"
            ),
        },
        "teacher": {
            "name": "frontier_u16",
            "utility": "max(1-(1-p)^16-p,0)^4",
            "decay": TEACHER_DECAY,
            "uniform_floor": TEACHER_FLOOR,
            "gamma": TEACHER_GAMMA,
            "evidence": "requested-task outcomes only",
        },
        "capacity_contract": copy.deepcopy(CAPACITY_CONTRACT),
    }


def _development_lock_payload() -> dict:
    """Return the deterministic lock authorized only for V1 development seeds."""

    return {
        "schema": LOCK_SCHEMA,
        "study": STUDY,
        "scope": "development_only",
        "authorized_development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmatory_seeds_reserved_untouched": list(CONFIRMATORY_SEEDS),
        "confirmatory_execution_available": False,
        "runtime": _runtime(),
        "source_sha256": _source_hashes(),
        "seed_collision_audit": seed_collision_audit(),
        "protocol": _protocol_config(
            transition_budget=TRANSITION_BUDGET,
            eval_interval=EVAL_INTERVAL_TRANSITIONS,
            eval_n=EVAL_EPISODES_PER_TASK,
            learning_rate=LEARNING_RATE,
        ),
        "conditions": [asdict(condition) for condition in CONDITIONS],
    }


def _load_development_lock(path: Path) -> dict:
    lock_path = Path(path)
    if not lock_path.is_file():
        raise RuntimeError(f"exact V1 development lock is missing: {lock_path}")
    try:
        with lock_path.open(encoding="utf-8") as handle:
            observed = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("V1 development lock is unreadable") from error
    expected = _development_lock_payload()
    if observed != expected:
        raise RuntimeError(
            "V1 development lock differs from current source/runtime/protocol"
        )
    return {
        "schema": LOCK_SCHEMA,
        "scope": "development_only",
        "file_sha256": _sha256(lock_path),
        "canonical_sha256": _canonical_hash(observed),
        "payload": observed,
    }


def _new_artifact(
    *,
    transition_budget: int,
    eval_interval: int,
    eval_n: int,
    learning_rate: float,
    source_lock: dict,
) -> dict:
    return {
        "schema": SCHEMA,
        "study": STUDY,
        "stage": "development_feasibility",
        "registration_status": "sealed_development_protocol",
        "confirmatory_execution_available": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "runtime": _runtime(),
            "source_sha256": _source_hashes(),
            "source_lock": copy.deepcopy(source_lock),
        },
        "seed_collision_audit": seed_collision_audit(),
        "protocol": _protocol_config(
            transition_budget=transition_budget,
            eval_interval=eval_interval,
            eval_n=eval_n,
            learning_rate=learning_rate,
        ),
        "paired_development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmatory_seeds_reserved_untouched": list(CONFIRMATORY_SEEDS),
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "cases": {
            condition.name: {
                "config": asdict(condition),
                "runs": [],
                "summary": summarize_runs([]),
            }
            for condition in CONDITIONS
        },
        "development_contrasts": {},
        "development_gates": {"development_feasible": False},
        "artifact_state": "incomplete",
    }


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _exclusive_write(path: Path, value: dict) -> None:
    """Atomically publish a new JSON file without ever replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite existing lock: {path}"
            ) from error
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def seal_development(lock_path: Path) -> dict:
    """Write the deterministic development-only lock without creating an env."""

    path = Path(lock_path)
    _exclusive_write(path, _development_lock_payload())
    return _load_development_lock(path)


def _resume_artifact(path: Path, expected: dict) -> dict:
    if not path.exists():
        return expected
    with path.open(encoding="utf-8") as handle:
        observed = json.load(handle)
    for key in (
        "schema",
        "study",
        "stage",
        "registration_status",
        "protocol",
        "paired_development_seeds",
        "confirmatory_seeds_reserved_untouched",
        "conditions",
        "seed_collision_audit",
    ):
        if observed.get(key) != expected.get(key):
            raise RuntimeError(f"resume artifact differs at {key}")
    if observed.get("provenance") != expected.get("provenance"):
        raise RuntimeError("resume provenance differs from current source/runtime")
    for condition in CONDITIONS:
        runs = observed.get("cases", {}).get(condition.name, {}).get("runs")
        if not isinstance(runs, list) or [run.get("seed") for run in runs] != list(
            DEVELOPMENT_SEEDS[: len(runs)]
        ):
            raise RuntimeError(f"resume seed prefix invalid for {condition.name}")
    return observed


def run_development(
    output: Path,
    *,
    development_lock: Path,
    resume: bool = False,
) -> dict:
    source_lock = _load_development_lock(development_lock)
    expected = _new_artifact(
        transition_budget=TRANSITION_BUDGET,
        eval_interval=EVAL_INTERVAL_TRANSITIONS,
        eval_n=EVAL_EPISODES_PER_TASK,
        learning_rate=LEARNING_RATE,
        source_lock=source_lock,
    )
    output_existed = output.exists()
    if output_existed and not resume:
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    artifact = _resume_artifact(output, expected) if resume else expected
    if not output_existed:
        _atomic_write(output, artifact)
    for condition in CONDITIONS:
        runs = artifact["cases"][condition.name]["runs"]
        for seed in DEVELOPMENT_SEEDS[len(runs) :]:
            if _load_development_lock(development_lock) != source_lock:
                raise RuntimeError("V1 development lock changed during execution")
            run = run_condition(condition, seed, development_lock=development_lock)
            if _load_development_lock(development_lock) != source_lock:
                raise RuntimeError("V1 development lock changed during execution")
            runs.append(run)
            artifact["cases"][condition.name]["summary"] = summarize_runs(runs)
            _atomic_write(output, artifact)
    artifact["development_contrasts"] = _development_descriptives(artifact["cases"])
    artifact["development_gates"] = _development_gates(artifact["cases"])
    artifact["artifact_state"] = "complete"
    _atomic_write(output, artifact)
    return artifact


def run_smoke(output: Path | None) -> dict:
    smoke_budget = 8_000
    smoke_interval = 4_000
    condition = CONDITIONS[0]
    run = run_condition(
        condition,
        SMOKE_SEED,
        transition_budget=smoke_budget,
        eval_interval_transitions=smoke_interval,
        eval_n=2,
    )
    artifact = {
        "schema": f"{SCHEMA}-excluded-smoke",
        "study": STUDY,
        "stage": "excluded_nonregistered_smoke",
        "seed": SMOKE_SEED,
        "reserved_seed_touched": False,
        "run": run,
    }
    if output is not None:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing smoke: {output}")
        _atomic_write(output, artifact)
    return artifact


def schedule() -> dict:
    return {
        "schema": SCHEMA,
        "registration_status": (
            "registration-ready but unsealed; schedule inspection executes no environment"
        ),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmatory_seeds_reserved_untouched": list(CONFIRMATORY_SEEDS),
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "protocol": _protocol_config(
            transition_budget=TRANSITION_BUDGET,
            eval_interval=EVAL_INTERVAL_TRANSITIONS,
            eval_n=EVAL_EPISODES_PER_TASK,
            learning_rate=LEARNING_RATE,
        ),
        "seed_collision_audit": seed_collision_audit(),
        "confirmatory_execution_available": False,
        "development_lock_schema": LOCK_SCHEMA,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("schedule", "smoke", "seal-development", "development"),
        default="schedule",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "schedule":
        print(json.dumps(schedule(), indent=2, allow_nan=False))
        return 0
    if args.mode == "smoke":
        result = run_smoke(args.output)
        print(
            json.dumps(
                result if args.output is None else {"wrote": str(args.output)}, indent=2
            )
        )
        return 0
    if args.mode == "seal-development":
        if args.lock is None:
            parser.error("seal-development requires --lock")
        record = seal_development(args.lock)
        print(
            json.dumps(
                {"wrote": str(args.lock), "sha256": record["file_sha256"]},
                indent=2,
            )
        )
        return 0
    if args.lock is None:
        parser.error("development requires --lock")
    if args.output is None:
        parser.error("development requires --output")
    result = run_development(
        args.output,
        development_lock=args.lock,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "wrote": str(args.output),
                "development_feasible": result["development_gates"][
                    "development_feasible"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
