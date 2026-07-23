"""Independent fail-closed verifier for MountainCar neural transfer V1r2.

The verifier deliberately imports neither the experiment runner nor its core.
It reconstructs the registered lock, uniform schedules, adaptive sampler draws,
training episode-seed stream, rollout outcomes, evaluation common random
numbers, curves, fixed-transition AUCs, summaries, contrasts, and gates from
raw JSON.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import gymnasium
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "mountaincar-neural-transfer-v1r2-development"
REPORT_SCHEMA = "mountaincar-neural-transfer-v1r2-development-verification"
LOCK_SCHEMA = "mountaincar-neural-transfer-v1r2-development-lock"
STUDY = "mountaincar_neural_transfer_v1"
DEVELOPMENT_SEEDS = (17_000, 17_001, 17_002)
CONFIRMATORY_SEEDS = tuple(range(18_000, 18_020))
THRESHOLDS = (-0.375, -0.250, -0.125, 0.0, 0.125, 0.250, 0.375, 0.500)
N_TASKS = 8
N_ACTIONS = 3
N_ROLLOUTS = 16
MAX_EPISODE_STEPS = 200
MAX_COMPLETE_GROUP_OVERSHOOT = N_ROLLOUTS * MAX_EPISODE_STEPS
TRANSITION_BUDGET = 500_000
MAX_GROUPS_FOR_BUDGET = math.ceil(TRANSITION_BUDGET / N_ROLLOUTS)
EVAL_INTERVAL_TRANSITIONS = 100_000
EVAL_EPISODES_PER_TASK = 32
LEARNING_RATE = 3e-4
TEACHER_DECAY = 0.7
TEACHER_FLOOR = 0.1
TEACHER_GAMMA = 4.0
TRAINING_ACTION_SEED_OFFSET = 1_000_000
TRAINING_EPISODE_SEED_OFFSET = 2_000_000
TEACHER_SEED_BASE = 4_000_000
EVALUATION_SEED_BASE = 6_000_000
EVALUATION_ACTION_SEED_BASE = 7_000_000

CONDITIONS = (
    {
        "name": "frontier_shared_h64",
        "sampling": "frontier_u16",
        "architecture": "shared_h64",
    },
    {
        "name": "uniform_shared_h64",
        "sampling": "uniform",
        "architecture": "shared_h64",
    },
    {
        "name": "hardest_shared_h64",
        "sampling": "hardest_only",
        "architecture": "shared_h64",
    },
    {
        "name": "uniform_disjoint_total_h8x8",
        "sampling": "uniform",
        "architecture": "disjoint_total_h8x8",
    },
    {
        "name": "uniform_disjoint_active_h64x8",
        "sampling": "uniform",
        "architecture": "disjoint_active_h64x8",
    },
)
PAIRED_UNIFORM_CONDITIONS = (
    "uniform_shared_h64",
    "uniform_disjoint_total_h8x8",
    "uniform_disjoint_active_h64x8",
)
CAPACITY = {
    "shared_h64": (64, 1, 384, 384),
    "disjoint_total_h8x8": (8, 8, 384, 48),
    "disjoint_active_h64x8": (64, 8, 3072, 384),
}
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
CONTRASTS = {
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
REQUIRED_SOURCE_FILES = (
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
ROLLOUT_AUDIT_KEYS = {
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
}


def _canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime() -> dict:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "gymnasium_version": gymnasium.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _source_hashes() -> dict[str, str]:
    if len(REQUIRED_SOURCE_FILES) != 11 or len(set(REQUIRED_SOURCE_FILES)) != 11:
        raise RuntimeError("V1r2 source manifest must contain exactly 11 unique paths")
    hashes: dict[str, str] = {}
    root = PROJECT_ROOT.resolve()
    for relative in REQUIRED_SOURCE_FILES:
        path = (PROJECT_ROOT / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"source path escapes project root: {relative}"
            ) from error
        if not path.is_file():
            raise RuntimeError(
                f"required V1r2 verification source is missing: {relative}"
            )
        hashes[relative] = _sha256(path)
    return hashes


def _expected_seed_audit() -> dict:
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
            name: sorted(set(values) & (prior | new_training))
            for name, values in rng_blocks.items()
        },
        "between_rng_root_blocks": {
            f"{left}_vs_{right}": sorted(set(rng_blocks[left]) & set(rng_blocks[right]))
            for index, left in enumerate(rng_blocks)
            for right in tuple(rng_blocks)[index + 1 :]
        },
    }
    passed = (
        len(development) == len(DEVELOPMENT_SEEDS)
        and len(confirmatory) == len(CONFIRMATORY_SEEDS)
        and not collisions["development_vs_confirmatory"]
        and not collisions["development_vs_prior"]
        and not collisions["confirmatory_vs_prior"]
        and not any(collisions["rng_roots_vs_training"].values())
        and not any(collisions["between_rng_root_blocks"].values())
    )
    return {
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
            "actor_action_seed": "training seed + 1000000",
            "environment_episode_seed_stream": "training seed + 2000000",
            "teacher_seed": "training seed + 4000000",
            "evaluation_episode_seed_root": "training seed + 6000000",
            "evaluation_action_seed_root": "training seed + 7000000",
        },
    }


def _assert_close(observed, expected, label: str, tolerance: float = 1e-12) -> None:
    if (
        isinstance(observed, bool)
        or not math.isfinite(float(observed))
        or not math.isclose(
            float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance
        )
    ):
        raise ValueError(f"{label} mismatch: {observed!r} != {expected!r}")


def fixed_budget_auc(y: Sequence[float], x: Sequence[int], budget: int) -> float:
    """Independent normalized piecewise-linear AUC on ``[0, budget]``."""

    values = [float(value) for value in y]
    coordinates = [int(value) for value in x]
    if (
        len(values) != len(coordinates)
        or len(values) < 2
        or coordinates[0] != 0
        or any(right <= left for left, right in zip(coordinates, coordinates[1:]))
        or coordinates[-1] < budget
        or budget <= 0
        or not all(math.isfinite(value) for value in values)
    ):
        raise ValueError("invalid independent fixed-budget AUC curve")
    clipped_x = [coordinates[0]]
    clipped_y = [values[0]]
    for left_x, right_x, left_y, right_y in zip(
        coordinates, coordinates[1:], values, values[1:]
    ):
        if left_x >= budget:
            break
        if right_x <= budget:
            clipped_x.append(right_x)
            clipped_y.append(right_y)
            continue
        fraction = (budget - left_x) / (right_x - left_x)
        clipped_x.append(budget)
        clipped_y.append(left_y + fraction * (right_y - left_y))
        break
    return float(
        sum(
            (right_x - left_x) * (left_y + right_y) / 2.0
            for left_x, right_x, left_y, right_y in zip(
                clipped_x, clipped_x[1:], clipped_y, clipped_y[1:]
            )
        )
        / budget
    )


def fixed_budget_value(y: Sequence[float], x: Sequence[int], budget: int) -> float:
    values = [float(value) for value in y]
    coordinates = [int(value) for value in x]
    if len(values) != len(coordinates):
        raise ValueError("curve coordinate/value lengths differ")
    for left_x, right_x, left_y, right_y in zip(
        coordinates, coordinates[1:], values, values[1:]
    ):
        if left_x <= budget <= right_x:
            fraction = (budget - left_x) / (right_x - left_x)
            return float(left_y + fraction * (right_y - left_y))
    raise ValueError("curve does not bracket fixed budget")


class _FrontierReplay:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(int(seed) + TEACHER_SEED_BASE)
        self.alpha = np.ones(N_TASKS, dtype=np.float64)
        self.beta = np.ones(N_TASKS, dtype=np.float64)

    def draw(self) -> tuple[int, np.ndarray]:
        sampled = self.rng.beta(self.alpha, self.beta)
        utility = (
            np.maximum(1.0 - (1.0 - sampled) ** N_ROLLOUTS - sampled, 0.0)
            ** TEACHER_GAMMA
        )
        if float(utility.sum()) <= 1e-12:
            utility = np.ones(N_TASKS, dtype=np.float64)
        probabilities = (
            1.0 - TEACHER_FLOOR
        ) * utility / utility.sum() + TEACHER_FLOOR / N_TASKS
        return int(self.rng.choice(N_TASKS, p=probabilities)), probabilities

    def observe(self, task_id: int, successes: int) -> None:
        self.alpha[task_id] = (
            1.0 + (self.alpha[task_id] - 1.0) * TEACHER_DECAY + successes
        )
        self.beta[task_id] = (
            1.0 + (self.beta[task_id] - 1.0) * TEACHER_DECAY + N_ROLLOUTS - successes
        )


class _UniformReplay:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(int(seed) + TEACHER_SEED_BASE)

    def draw(self) -> tuple[int, np.ndarray]:
        probabilities = np.full(N_TASKS, 1.0 / N_TASKS, dtype=np.float64)
        return int(self.rng.choice(N_TASKS, p=probabilities)), probabilities

    def observe(self, task_id: int, successes: int) -> None:
        del task_id, successes


class _HardestReplay:
    def draw(self) -> tuple[int, np.ndarray]:
        probabilities = np.zeros(N_TASKS, dtype=np.float64)
        probabilities[-1] = 1.0
        return N_TASKS - 1, probabilities

    def observe(self, task_id: int, successes: int) -> None:
        del task_id, successes


def _sampler(sampling: str, seed: int):
    if sampling == "frontier_u16":
        return _FrontierReplay(seed)
    if sampling == "uniform":
        return _UniformReplay(seed)
    if sampling == "hardest_only":
        return _HardestReplay()
    raise ValueError(f"unknown sampling rule {sampling!r}")


@functools.lru_cache(maxsize=None)
def _uniform_task_sequence(
    seed: int, length: int = MAX_GROUPS_FOR_BUDGET
) -> tuple[int, ...]:
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise ValueError("uniform schedule length must be a positive integer")
    rng = np.random.default_rng(int(seed) + TEACHER_SEED_BASE)
    probabilities = np.full(N_TASKS, 1.0 / N_TASKS, dtype=np.float64)
    return tuple(int(rng.choice(N_TASKS, p=probabilities)) for _ in range(length))


def _uniform_task_schedule_sha256(
    seed: int, length: int = MAX_GROUPS_FOR_BUDGET
) -> str:
    return _canonical_hash(list(_uniform_task_sequence(int(seed), int(length))))


def _expected_protocol() -> dict:
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
        "learning_rate": LEARNING_RATE,
        "optimizer": "plain SGD ascent; summed frozen-group score gradient",
        "transition_budget": TRANSITION_BUDGET,
        "eval_interval_transitions": EVAL_INTERVAL_TRANSITIONS,
        "eval_episodes_per_task": EVAL_EPISODES_PER_TASK,
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
            "seed_rule": "training seed + 4000000",
            "registered_length": MAX_GROUPS_FOR_BUDGET,
            "sha256_by_development_seed": {
                str(seed): _uniform_task_schedule_sha256(seed)
                for seed in DEVELOPMENT_SEEDS
            },
        },
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
        "checkpoint_policy_rule": (
            "post-group if end == checkpoint; otherwise saved pre-crossing-group parameters"
        ),
        "teacher": {
            "name": "frontier_u16",
            "utility": "max(1-(1-p)^16-p,0)^4",
            "decay": TEACHER_DECAY,
            "uniform_floor": TEACHER_FLOOR,
            "gamma": TEACHER_GAMMA,
            "evidence": "requested-task outcomes only",
        },
        "capacity_contract": {
            architecture: {
                "hidden_size": values[0],
                "slots": values[1],
                "total_parameters": values[2],
                "active_parameters_per_task": values[3],
            }
            for architecture, values in CAPACITY.items()
        },
    }


def _expected_development_lock_payload() -> dict:
    return {
        "schema": LOCK_SCHEMA,
        "study": STUDY,
        "scope": "development_only",
        "authorized_development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmatory_seeds_reserved_untouched": list(CONFIRMATORY_SEEDS),
        "confirmatory_execution_available": False,
        "runtime": _runtime(),
        "source_sha256": _source_hashes(),
        "seed_collision_audit": _expected_seed_audit(),
        "protocol": _expected_protocol(),
        "conditions": [dict(condition) for condition in CONDITIONS],
    }


def _load_json(path: Path, label: str) -> dict:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _validate_rollout_audit(
    record: dict,
    *,
    expected_episode_seed: int,
    threshold: float,
) -> tuple[int, bool]:
    if not isinstance(record, dict) or set(record) != ROLLOUT_AUDIT_KEYS:
        raise ValueError("rollout audit record has a missing or extra field")
    n_steps = record.get("n_steps")
    if (
        record.get("episode_seed") != expected_episode_seed
        or not isinstance(n_steps, int)
        or isinstance(n_steps, bool)
        or not 1 <= n_steps <= MAX_EPISODE_STEPS
        or type(record.get("native_terminated")) is not bool
        or type(record.get("native_truncated")) is not bool
        or type(record.get("success")) is not bool
    ):
        raise ValueError("rollout seed/count/boolean audit failed")
    positions = {}
    for key in (
        "max_position",
        "max_position_before_final",
        "pre_final_position",
        "final_position",
    ):
        value = record.get(key)
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("rollout contains a non-finite position")
        positions[key] = float(value)
        if not -1.2 - 1e-12 <= positions[key] <= 0.6 + 1e-12:
            raise ValueError(
                "rollout position is outside MountainCar observation bounds"
            )
    _assert_close(
        positions["max_position"],
        max(positions["max_position_before_final"], positions["final_position"]),
        "rollout maximum position",
    )
    if positions["max_position_before_final"] + 1e-12 < positions["pre_final_position"]:
        raise ValueError("pre-final position exceeds the recorded earlier maximum")
    if n_steps == 1:
        _assert_close(
            positions["max_position_before_final"],
            positions["pre_final_position"],
            "single-step pre-final maximum",
        )

    native_reward_sum = record.get("native_reward_sum")
    _assert_close(native_reward_sum, -n_steps, "MountainCar native reward sum")
    terminated = record["native_terminated"]
    truncated = record["native_truncated"]
    success = record["success"]
    if terminated != (positions["final_position"] >= THRESHOLDS[-1]):
        raise ValueError("native termination disagrees with MountainCar goal position")
    if truncated != (n_steps == MAX_EPISODE_STEPS):
        raise ValueError("native truncation disagrees with the 200-step horizon")
    derived_success = positions["max_position"] >= threshold
    if success != derived_success:
        raise ValueError("rollout success disagrees with maximum position")
    if success:
        if not (
            positions["max_position_before_final"] < threshold
            and positions["final_position"] >= threshold
        ):
            raise ValueError("successful rollout is not a first-hit termination")
    elif not (
        positions["max_position"] < threshold
        and not terminated
        and truncated
        and n_steps == MAX_EPISODE_STEPS
    ):
        raise ValueError("failed rollout did not exhaust the native horizon")
    return n_steps, success


def _validate_groups_and_updates(run: dict, condition: dict) -> None:
    groups = run.get("group_diagnostics")
    updates = run.get("update_diagnostics")
    zero_records = run.get("zero_gradient_diagnostics")
    if not all(isinstance(value, list) for value in (groups, updates, zero_records)):
        raise ValueError("raw group/update diagnostics are missing")
    if (
        len(groups) != run.get("sampled_groups")
        or not 1 <= len(groups) <= MAX_GROUPS_FOR_BUDGET
    ):
        raise ValueError("sampled group count mismatch")
    if len(updates) != run.get("optimizer_updates"):
        raise ValueError("optimizer update count mismatch")
    if len(zero_records) != run.get("zero_gradient_update_attempts"):
        raise ValueError("zero-gradient count mismatch")
    update_by_group = {int(record.get("after_group", -1)): record for record in updates}
    zero_by_group = {
        int(record.get("after_group", -1)): record for record in zero_records
    }
    if (
        len(update_by_group) != len(updates)
        or len(zero_by_group) != len(zero_records)
        or set(update_by_group) & set(zero_by_group)
        or any(group_id < 1 or group_id > len(groups) for group_id in update_by_group)
        or any(group_id < 1 or group_id > len(groups) for group_id in zero_by_group)
    ):
        raise ValueError("group update diagnostics are not a disjoint one-to-one map")

    seed = int(run["seed"])
    replay = _sampler(condition["sampling"], seed)
    uniform_schedule = (
        _uniform_task_sequence(seed) if condition["sampling"] == "uniform" else None
    )
    episode_rng = np.random.default_rng(seed + TRAINING_EPISODE_SEED_OFFSET)
    previous_end = 0
    previous_updates = 0
    previous_parameter_after = None
    regime_counts = {"dead": 0, "mixed": 0, "all_pass": 0}
    task_groups = [0] * N_TASKS
    task_successes = [0] * N_TASKS
    task_transitions = [0] * N_TASKS
    derived_sequence: list[int] = []
    crossing_group = None
    crossing_transition = None

    for index, group in enumerate(groups, start=1):
        start = group.get("transition_start")
        end = group.get("transition_end")
        count = group.get("n_transitions")
        task = group.get("task_id")
        saved_successes = group.get("success_count")
        update_count = group.get("optimizer_updates_after_group")
        if (
            group.get("group") != index
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (start, end, count, task, saved_successes, update_count)
            )
            or start != previous_end
            or end - start != count
            or not N_ROLLOUTS <= count <= MAX_COMPLETE_GROUP_OVERSHOOT
            or start >= TRANSITION_BUDGET
            or not 0 <= task < N_TASKS
            or not 0 <= saved_successes <= N_ROLLOUTS
        ):
            raise ValueError("raw complete-group accounting failed")
        if crossing_group is None and end >= TRANSITION_BUDGET:
            crossing_group, crossing_transition = index, end
        if crossing_group is not None and index > crossing_group:
            raise ValueError("run contains post-budget alignment groups")

        expected_task, expected_probabilities = replay.draw()
        observed_probabilities = np.asarray(
            group.get("sampling_probabilities"), dtype=np.float64
        )
        if (
            observed_probabilities.shape != (N_TASKS,)
            or not np.isfinite(observed_probabilities).all()
            or not np.allclose(
                observed_probabilities, expected_probabilities, rtol=0.0, atol=1e-14
            )
            or task != expected_task
            or (uniform_schedule is not None and task != uniform_schedule[index - 1])
        ):
            raise ValueError("sampler trace does not independently replay")
        _assert_close(
            group.get("sampled_task_probability"),
            expected_probabilities[task],
            "sampled task probability",
            1e-14,
        )

        rollouts = group.get("rollouts")
        if not isinstance(rollouts, list) or len(rollouts) != N_ROLLOUTS:
            raise ValueError("group lacks exactly 16 rollout audit records")
        derived_transitions = 0
        derived_successes = 0
        for rollout in rollouts:
            expected_episode_seed = int(episode_rng.integers(0, 2**31 - 1))
            n_steps, success = _validate_rollout_audit(
                rollout,
                expected_episode_seed=expected_episode_seed,
                threshold=THRESHOLDS[task],
            )
            derived_transitions += n_steps
            derived_successes += int(success)
        if count != derived_transitions or saved_successes != derived_successes:
            raise ValueError("group totals do not reproduce from rollout audits")
        regime = (
            "dead"
            if derived_successes == 0
            else "all_pass" if derived_successes == N_ROLLOUTS else "mixed"
        )
        if group.get("regime") != regime:
            raise ValueError("saved group regime differs from rollout outcomes")
        replay.observe(task, derived_successes)

        parameter_before = group.get("parameter_sha256_before_group")
        parameter_after = group.get("parameter_sha256_after_group")
        if (
            not isinstance(parameter_before, str)
            or len(parameter_before) != 64
            or not isinstance(parameter_after, str)
            or len(parameter_after) != 64
            or (
                previous_parameter_after is not None
                and parameter_before != previous_parameter_after
            )
        ):
            raise ValueError("group parameter snapshot chain is invalid")
        diagnostic = update_by_group.get(index)
        zero = zero_by_group.get(index)
        expected_source = "requested_live" if diagnostic is not None else None
        if regime != "mixed" and (diagnostic is not None or zero is not None):
            raise ValueError("dead/all-pass group attempted an update")
        if regime == "mixed" and (diagnostic is None) == (zero is None):
            raise ValueError("mixed group lacks exactly one update attempt")
        increment = int(diagnostic is not None)
        if (
            update_count != previous_updates + increment
            or group.get("update_source") != expected_source
        ):
            raise ValueError("group optimizer counter/source mismatch")
        if (diagnostic is None and parameter_before != parameter_after) or (
            diagnostic is not None and parameter_before == parameter_after
        ):
            raise ValueError("parameter hash disagrees with applied-update status")

        previous_end = end
        previous_updates = update_count
        previous_parameter_after = parameter_after
        regime_counts[regime] += 1
        task_groups[task] += 1
        task_successes[task] += derived_successes
        task_transitions[task] += derived_transitions
        derived_sequence.append(task)

    if crossing_group is None or crossing_transition is None:
        raise ValueError("run never crossed the transition budget")
    if previous_end != run.get("transitions") or previous_updates != run.get(
        "optimizer_updates"
    ):
        raise ValueError("terminal group counters mismatch")
    if (
        run.get("transition_at_budget_crossing") != crossing_transition
        or run.get("budget_crossing_group") != crossing_group
        or run.get("complete_group_overshoot")
        != crossing_transition - TRANSITION_BUDGET
        or not 0
        <= run.get("complete_group_overshoot", -1)
        <= MAX_COMPLETE_GROUP_OVERSHOOT
        or run.get("post_budget_alignment_groups") != 0
        or run.get("post_budget_alignment_transitions") != 0
        or len(groups) != crossing_group
        or run.get("transitions") != crossing_transition
        or run.get("reached_transition_budget") is not True
    ):
        raise ValueError("budget-crossing and no-alignment accounting failed")
    for key, expected in (
        ("dead_groups", regime_counts["dead"]),
        ("mixed_groups", regime_counts["mixed"]),
        ("all_pass_groups", regime_counts["all_pass"]),
        ("task_groups", task_groups),
        ("task_rollouts", [count * N_ROLLOUTS for count in task_groups]),
        ("task_successes", task_successes),
        ("task_transitions", task_transitions),
        ("task_sequence", derived_sequence),
    ):
        if run.get(key) != expected:
            raise ValueError(f"raw accounting mismatch for {key}")

    if condition["sampling"] == "uniform":
        expected_hash = _uniform_task_schedule_sha256(seed)
        if (
            run.get("registered_uniform_task_schedule_length") != MAX_GROUPS_FOR_BUDGET
            or run.get("registered_uniform_task_schedule_sha256") != expected_hash
            or derived_sequence != list(_uniform_task_sequence(seed)[: len(groups)])
        ):
            raise ValueError(
                "uniform run differs from its registered full schedule prefix"
            )
    elif (
        run.get("registered_uniform_task_schedule_length") is not None
        or run.get("registered_uniform_task_schedule_sha256") is not None
    ):
        raise ValueError("non-uniform run claims a registered uniform schedule")

    expected_slot = lambda task: (
        0 if condition["architecture"] == "shared_h64" else task
    )
    for optimizer_index, record in enumerate(updates, start=1):
        group = groups[int(record.get("after_group", 0)) - 1]
        expected_weight_l1 = 2.0 * (1.0 - group["success_count"] / N_ROLLOUTS)
        if (
            record.get("optimizer_update") != optimizer_index
            or record.get("source") != "requested_live"
            or record.get("requested_task") != group["task_id"]
            or record.get("credited_task") != group["task_id"]
            or record.get("task_id") != group["task_id"]
            or record.get("transitions") != group["transition_end"]
            or record.get("slot") != expected_slot(group["task_id"])
            or record.get("applied") is not True
            or record.get("frozen_group_parameters") is not True
            or record.get("n_trajectories") != N_ROLLOUTS
            or record.get("n_score_terms") != group["n_transitions"]
            or record.get("n_weighted_score_terms") != group["n_transitions"]
            or not math.isfinite(float(record.get("gradient_norm", math.nan)))
            or float(record["gradient_norm"]) <= 0.0
            or not math.isfinite(float(record.get("update_norm", math.nan)))
            or float(record["update_norm"]) <= 0.0
        ):
            raise ValueError("applied update diagnostic is invalid")
        _assert_close(
            record["update_norm"],
            LEARNING_RATE * record["gradient_norm"],
            "plain-SGD update norm",
            1e-10,
        )
        _assert_close(record.get("weight_l1"), expected_weight_l1, "MaxRL weight L1")
        entropy = float(record.get("mean_policy_entropy", math.nan))
        if not math.isfinite(entropy) or not 0.0 <= entropy <= math.log(3.0) + 1e-12:
            raise ValueError("policy entropy diagnostic is invalid")
    for record in zero_records:
        group = groups[int(record.get("after_group", 0)) - 1]
        expected_weight_l1 = 2.0 * (1.0 - group["success_count"] / N_ROLLOUTS)
        if (
            record.get("source") != "requested_live"
            or record.get("requested_task") != group["task_id"]
            or record.get("credited_task") != group["task_id"]
            or record.get("task_id") != group["task_id"]
            or record.get("transitions") != group["transition_end"]
            or record.get("slot") != expected_slot(group["task_id"])
            or record.get("applied") is not False
            or "optimizer_update" in record
            or record.get("frozen_group_parameters") is not True
            or record.get("n_trajectories") != N_ROLLOUTS
            or record.get("n_score_terms") != group["n_transitions"]
            or record.get("n_weighted_score_terms") != group["n_transitions"]
            or float(record.get("gradient_norm", math.nan)) != 0.0
            or float(record.get("update_norm", math.nan)) != 0.0
        ):
            raise ValueError("zero-gradient update diagnostic is invalid")
        _assert_close(record.get("weight_l1"), expected_weight_l1, "MaxRL weight L1")
        entropy = float(record.get("mean_policy_entropy", math.nan))
        if not math.isfinite(entropy) or not 0.0 <= entropy <= math.log(3.0) + 1e-12:
            raise ValueError("zero-gradient policy entropy diagnostic is invalid")
    if run.get("live_applied_updates") != len(updates):
        raise ValueError("live applied-update count mismatch")


def _validate_capacity(run: dict, condition: dict) -> None:
    actor = run.get("actor")
    expected = CAPACITY[condition["architecture"]]
    if not isinstance(actor, dict) or (
        actor.get("mode"),
        actor.get("hidden_size"),
        actor.get("n_slots"),
        actor.get("parameter_count"),
        actor.get("active_parameter_count"),
    ) != (condition["architecture"], *expected):
        raise ValueError("exact actor capacity contract failed")
    slot_calls = actor.get("slot_update_calls")
    if (
        actor.get("update_calls") != run.get("mixed_groups")
        or actor.get("applied_updates") != run.get("optimizer_updates")
        or not isinstance(slot_calls, list)
        or len(slot_calls) != expected[1]
        or sum(slot_calls) != run.get("mixed_groups")
    ):
        raise ValueError("actor update counters do not reproduce")
    expected_slot_calls = [0] * expected[1]
    for group in run["group_diagnostics"]:
        if group["regime"] == "mixed":
            slot = 0 if condition["architecture"] == "shared_h64" else group["task_id"]
            expected_slot_calls[slot] += 1
    if slot_calls != expected_slot_calls:
        raise ValueError("actor per-slot update counters do not reproduce")
    digest = actor.get("parameter_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != run["group_diagnostics"][-1]["parameter_sha256_after_group"]
    ):
        raise ValueError("final actor digest is malformed")
    if not math.isfinite(float(actor.get("parameter_norm", math.nan))):
        raise ValueError("final actor norm is non-finite")


def _validate_curves(run: dict, condition: dict) -> None:
    x = run.get("x_transitions")
    pass_curve = run.get("pass_rate_curve")
    mean_curve = run.get("mean_pass_curve")
    hardest_curve = run.get("hardest_pass_curve")
    samples = run.get("evaluation_max_position_samples")
    expected_count = TRANSITION_BUDGET // EVAL_INTERVAL_TRANSITIONS + 1
    if not all(
        isinstance(value, list)
        for value in (x, pass_curve, mean_curve, hardest_curve, samples)
    ) or not (
        len(x)
        == len(pass_curve)
        == len(mean_curve)
        == len(hardest_curve)
        == len(samples)
        == expected_count
    ):
        raise ValueError("evaluation raw data/cadence length mismatch")
    expected_x = list(range(0, TRANSITION_BUDGET + 1, EVAL_INTERVAL_TRANSITIONS))
    if x != expected_x:
        raise ValueError("evaluation transition coordinates invalid")

    seed = int(run["seed"])
    expected_episode_seeds = (
        np.random.default_rng(seed + EVALUATION_SEED_BASE)
        .integers(0, 2**31 - 1, size=EVAL_EPISODES_PER_TASK, dtype=np.int64)
        .astype(int)
        .tolist()
    )
    expected_action_seeds = (
        np.random.default_rng(seed + EVALUATION_ACTION_SEED_BASE)
        .integers(0, 2**31 - 1, size=EVAL_EPISODES_PER_TASK, dtype=np.int64)
        .astype(int)
        .tolist()
    )
    if run.get("evaluation_episode_seeds") != expected_episode_seeds:
        raise ValueError("evaluation episode CRNs do not reproduce from seed + 6000000")
    if run.get("evaluation_action_seeds") != expected_action_seeds:
        raise ValueError("evaluation action CRNs do not reproduce from seed + 7000000")

    reconstructed_rates: list[list[float]] = []
    for checkpoint_samples in samples:
        if (
            not isinstance(checkpoint_samples, list)
            or len(checkpoint_samples) != N_TASKS
        ):
            raise ValueError("evaluation sample tensor lacks eight task rows")
        if condition["architecture"] == "shared_h64" and any(
            row != checkpoint_samples[0] for row in checkpoint_samples[1:]
        ):
            raise ValueError(
                "shared evaluation did not reuse an identical max-position row"
            )
        rates = []
        for task, row in enumerate(checkpoint_samples):
            if not isinstance(row, list) or len(row) != EVAL_EPISODES_PER_TASK:
                raise ValueError("evaluation task row lacks exactly 32 samples")
            values = []
            for position in row:
                if isinstance(position, bool) or not math.isfinite(float(position)):
                    raise ValueError("evaluation sample contains a non-finite position")
                numeric = float(position)
                if not -1.2 - 1e-12 <= numeric <= 0.6 + 1e-12:
                    raise ValueError("evaluation sample is outside MountainCar bounds")
                values.append(numeric)
            rates.append(
                sum(position >= THRESHOLDS[task] for position in values)
                / EVAL_EPISODES_PER_TASK
            )
        reconstructed_rates.append(rates)
    if pass_curve != reconstructed_rates:
        raise ValueError(
            "pass-rate curve does not reproduce from raw evaluation samples"
        )
    reconstructed_mean = [float(np.mean(row)) for row in reconstructed_rates]
    reconstructed_hardest = [float(row[-1]) for row in reconstructed_rates]
    if mean_curve != reconstructed_mean:
        raise ValueError("mean pass curve does not reproduce from evaluation samples")
    if hardest_curve != reconstructed_hardest:
        raise ValueError(
            "hardest pass curve does not reproduce from evaluation samples"
        )

    triggers = run.get("evaluation_trigger_transitions")
    policy_sources = run.get("evaluation_policy_sources")
    policy_hashes = run.get("evaluation_policy_parameter_sha256")
    if (
        not isinstance(triggers, list)
        or not isinstance(policy_sources, list)
        or not isinstance(policy_hashes, list)
        or not len(triggers)
        == len(policy_sources)
        == len(policy_hashes)
        == expected_count
        or triggers[0] != 0
        or policy_sources[0] != "initial"
        or not isinstance(policy_hashes[0], str)
        or len(policy_hashes[0]) != 64
        or policy_hashes[0]
        != run["group_diagnostics"][0]["parameter_sha256_before_group"]
        or triggers[-1] != run.get("transition_at_budget_crossing")
    ):
        raise ValueError("checkpoint policy snapshot trace is invalid")
    groups_by_end = {
        group["transition_end"]: group for group in run["group_diagnostics"]
    }
    for checkpoint, trigger, source, policy_hash in zip(
        x[1:], triggers[1:], policy_sources[1:], policy_hashes[1:]
    ):
        group = groups_by_end.get(trigger)
        expected_source = (
            "post_exact_group" if trigger == checkpoint else "pre_crossing_group"
        )
        expected_hash = (
            group.get(
                "parameter_sha256_after_group"
                if trigger == checkpoint
                else "parameter_sha256_before_group"
            )
            if group is not None
            else None
        )
        if (
            group is None
            or not group["transition_start"] < checkpoint <= trigger
            or trigger > checkpoint + MAX_COMPLETE_GROUP_OVERSHOOT
            or source != expected_source
            or policy_hash != expected_hash
        ):
            raise ValueError("evaluation does not bracket registered checkpoint")

    if run.get("evaluation_rng_preserved") != [True] * expected_count:
        raise ValueError("evaluation training-state preservation failed")
    nested = run.get("shared_nested_evaluations")
    matrix = np.asarray(reconstructed_rates, dtype=np.float64)
    if condition["architecture"] == "shared_h64":
        if nested != [True] * expected_count or np.any(matrix[:, :-1] < matrix[:, 1:]):
            raise ValueError("shared task-agnostic evaluation is not nested")
    elif nested != [None] * expected_count:
        raise ValueError("disjoint nestedness field must be non-applicable")
    wall_trace = run.get("wall_seconds_at_evaluations")
    if (
        not isinstance(wall_trace, list)
        or len(wall_trace) != expected_count
        or wall_trace[0] != 0.0
        or not all(math.isfinite(float(value)) and value >= 0.0 for value in wall_trace)
        or any(right < left for left, right in zip(wall_trace, wall_trace[1:]))
        or wall_trace[-1] > run.get("wall_seconds", -1.0)
    ):
        raise ValueError("evaluation wall-time trace is invalid")

    _assert_close(
        run.get("auc_hardest_pass_fixed_transitions"),
        fixed_budget_auc(reconstructed_hardest, x, TRANSITION_BUDGET),
        "hardest fixed-transition AUC",
    )
    _assert_close(
        run.get("auc_mean_pass_fixed_transitions"),
        fixed_budget_auc(reconstructed_mean, x, TRANSITION_BUDGET),
        "mean-pass fixed-transition AUC",
    )
    _assert_close(
        run.get("final_hardest_pass_at_budget"),
        fixed_budget_value(reconstructed_hardest, x, TRANSITION_BUDGET),
        "hardest pass at exact budget",
    )
    _assert_close(
        run.get("final_mean_pass_at_budget"),
        fixed_budget_value(reconstructed_mean, x, TRANSITION_BUDGET),
        "mean pass at exact budget",
    )
    if run.get("initial_pass_rates") != reconstructed_rates[0]:
        raise ValueError("initial pass-rate copy mismatch")


def _validate_run(run: dict, condition: dict, expected_seed: int) -> None:
    if (
        run.get("seed") != expected_seed
        or run.get("condition") != condition["name"]
        or run.get("sampling") != condition["sampling"]
        or run.get("architecture") != condition["architecture"]
        or run.get("transition_budget") != TRANSITION_BUDGET
        or run.get("numeric_valid") is not True
    ):
        raise ValueError("run identity/config/numeric contract failed")
    if (
        run.get("hindsight") is not False
        or run.get("relabel_candidates") != 0
        or run.get("relabeled_groups") != 0
    ):
        raise ValueError("no-hindsight study contains relabel activity")
    _validate_groups_and_updates(run, condition)
    _validate_capacity(run, condition)
    _validate_curves(run, condition)
    projection = [
        {
            "group": record["group"],
            "transition_end": record["transition_end"],
            "task_id": record["task_id"],
            "success_count": record["success_count"],
            "optimizer_updates_after_group": record["optimizer_updates_after_group"],
        }
        for record in run["group_diagnostics"]
    ]
    if run.get("training_group_trace_sha256") != _canonical_hash(projection):
        raise ValueError("training group trace digest mismatch")
    if (
        not math.isfinite(float(run.get("wall_seconds", math.nan)))
        or run["wall_seconds"] <= 0
    ):
        raise ValueError("run wall time is invalid")


def _summary(runs: Sequence[dict]) -> dict:
    metrics = (
        "auc_mean_pass_fixed_transitions",
        "auc_hardest_pass_fixed_transitions",
        "final_mean_pass_at_budget",
        "final_hardest_pass_at_budget",
        "transitions",
        "optimizer_updates",
        "wall_seconds",
    )
    output = {"n_runs": len(runs)}
    for metric in metrics:
        values = np.asarray([run[metric] for run in runs], dtype=np.float64)
        output[metric] = {
            "mean": float(values.mean()) if len(values) else None,
            "sample_std": (
                float(values.std(ddof=1))
                if len(values) > 1
                else 0.0 if len(values) else None
            ),
            "per_seed": values.tolist(),
        }
    return output


def _descriptives(cases: dict) -> dict:
    output = {}
    for name, (left_name, right_name) in CONTRASTS.items():
        left, right = cases[left_name]["runs"], cases[right_name]["runs"]
        if [run["seed"] for run in left] != [run["seed"] for run in right]:
            raise ValueError(f"unpaired development contrast {name}")
        primary = [
            float(a["auc_hardest_pass_fixed_transitions"])
            - float(b["auc_hardest_pass_fixed_transitions"])
            for a, b in zip(left, right)
        ]
        supporting = [
            float(a["auc_mean_pass_fixed_transitions"])
            - float(b["auc_mean_pass_fixed_transitions"])
            for a, b in zip(left, right)
        ]
        output[name] = {
            "primary_metric": "auc_hardest_pass_fixed_transitions",
            "supporting_metric": "auc_mean_pass_fixed_transitions",
            "estimand": f"{left_name} - {right_name}",
            "primary_per_seed_delta": primary,
            "primary_mean_delta": float(np.mean(primary)),
            "supporting_per_seed_delta": supporting,
            "supporting_mean_delta": float(np.mean(supporting)),
            "inference": "descriptive development result; not confirmatory",
        }
    return output


def _paired_uniform_schedules_valid(cases: dict) -> bool:
    try:
        for index, seed in enumerate(DEVELOPMENT_SEEDS):
            runs = [cases[name]["runs"][index] for name in PAIRED_UNIFORM_CONDITIONS]
            sequences = [run["task_sequence"] for run in runs]
            common_length = min(len(sequence) for sequence in sequences)
            expected = _uniform_task_sequence(seed)
            expected_hash = _uniform_task_schedule_sha256(seed)
            if not all(
                sequence == list(expected[: len(sequence)]) for sequence in sequences
            ):
                return False
            if not all(
                sequence[:common_length] == sequences[0][:common_length]
                for sequence in sequences[1:]
            ):
                return False
            if not all(
                run["registered_uniform_task_schedule_length"] == MAX_GROUPS_FOR_BUDGET
                and run["registered_uniform_task_schedule_sha256"] == expected_hash
                for run in runs
            ):
                return False
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    return True


def _gates(cases: dict) -> dict:
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
        run["actor"]["parameter_count"] == CAPACITY[run["architecture"]][2]
        and run["actor"]["active_parameter_count"] == CAPACITY[run["architecture"]][3]
        for run in all_runs
    )
    coverage = complete and all(
        set(
            task
            for run in cases[condition["name"]]["runs"]
            for task, count in enumerate(run["task_groups"])
            if count > 0
        )
        == (
            {N_TASKS - 1}
            if condition["sampling"] == "hardest_only"
            else set(range(N_TASKS))
        )
        for condition in CONDITIONS
    )
    update_signal = complete and all(
        sum(run["optimizer_updates"] for run in cases[condition["name"]]["runs"]) > 0
        for condition in CONDITIONS
        if condition["sampling"] != "hardest_only"
    )
    regimes = {
        key: sum(run[key] for run in all_runs)
        for key in ("dead_groups", "mixed_groups", "all_pass_groups")
    }
    natural_regimes = complete and all(value > 0 for value in regimes.values())
    initial_crn = complete and all(
        all(
            cases[condition["name"]]["runs"][index]["initial_pass_rates"]
            == cases[CONDITIONS[0]["name"]]["runs"][index]["initial_pass_rates"]
            for condition in CONDITIONS[1:]
        )
        for index in range(len(DEVELOPMENT_SEEDS))
    )
    paired_uniform_schedule = complete and _paired_uniform_schedules_valid(cases)
    projected_hours = (
        sum(run["wall_seconds"] for run in all_runs)
        / max(len(all_runs), 1)
        * len(CONDITIONS)
        * len(CONFIRMATORY_SEEDS)
        / 3600.0
    )
    runtime = complete and math.isfinite(projected_hours) and projected_hours <= 18.0
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
        "projected_confirmatory_hours": projected_hours,
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


def verify(artifact_path: Path, lock_path: Path) -> dict:
    artifact_path = Path(artifact_path)
    lock_path = Path(lock_path)
    lock = _load_json(lock_path, "development lock")
    expected_lock = _expected_development_lock_payload()
    if lock != expected_lock:
        raise ValueError(
            "external development lock differs from exact live V1r2 payload"
        )
    lock_file_sha256 = _sha256(lock_path)
    expected_lock_record = {
        "schema": LOCK_SCHEMA,
        "scope": "development_only",
        "file_sha256": lock_file_sha256,
        "canonical_sha256": _canonical_hash(lock),
        "payload": lock,
    }

    artifact = _load_json(artifact_path, "development artifact")
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("study") != STUDY
        or artifact.get("stage") != "development_feasibility"
        or artifact.get("registration_status") != "sealed_development_protocol"
        or artifact.get("confirmatory_execution_available") is not False
        or artifact.get("artifact_state") != "complete"
    ):
        raise ValueError("artifact identity/stage/sealed status is invalid")
    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("artifact provenance is missing")
    if provenance.get("runtime") != _runtime():
        raise ValueError("artifact runtime differs from independent live runtime")
    live_sources = _source_hashes()
    if set(live_sources) != set(REQUIRED_SOURCE_FILES):
        raise ValueError(
            "independent live source manifest is not the exact 11-path set"
        )
    if provenance.get("source_sha256") != live_sources:
        raise ValueError("artifact source hashes differ from independent live hashes")
    if provenance.get("source_lock") != expected_lock_record:
        raise ValueError("artifact does not embed the exact supplied development lock")
    if lock.get("source_sha256") != live_sources:
        raise ValueError("development lock source manifest differs from live sources")
    if artifact.get("paired_development_seeds") != list(DEVELOPMENT_SEEDS):
        raise ValueError("development seed block mismatch")
    if artifact.get("confirmatory_seeds_reserved_untouched") != list(
        CONFIRMATORY_SEEDS
    ):
        raise ValueError("confirmatory reservation mismatch")
    if artifact.get("seed_collision_audit") != _expected_seed_audit():
        raise ValueError("seed collision audit/ledger mismatch")
    if artifact.get("conditions") != [dict(condition) for condition in CONDITIONS]:
        raise ValueError("condition matrix/order mismatch")
    if artifact.get("protocol") != _expected_protocol():
        raise ValueError("registered development protocol mismatch")

    cases = artifact.get("cases")
    if not isinstance(cases, dict) or tuple(cases) != tuple(
        condition["name"] for condition in CONDITIONS
    ):
        raise ValueError("case matrix/order mismatch")
    for condition in CONDITIONS:
        case = cases[condition["name"]]
        if case.get("config") != condition:
            raise ValueError("case configuration mismatch")
        runs = case.get("runs")
        if not isinstance(runs, list) or [run.get("seed") for run in runs] != list(
            DEVELOPMENT_SEEDS
        ):
            raise ValueError("case seed order/completeness mismatch")
        for run, seed in zip(runs, DEVELOPMENT_SEEDS):
            _validate_run(run, condition, seed)
        if case.get("summary") != _summary(runs):
            raise ValueError("runner case summary differs from independent summary")

    for index in range(len(DEVELOPMENT_SEEDS)):
        reference_run = cases[CONDITIONS[0]["name"]]["runs"][index]
        for condition in CONDITIONS[1:]:
            compared = cases[condition["name"]]["runs"][index]
            if compared["initial_pass_rates"] != reference_run["initial_pass_rates"]:
                raise ValueError("initial paired common-random-number rates differ")
            if (
                compared["evaluation_max_position_samples"][0]
                != reference_run["evaluation_max_position_samples"][0]
            ):
                raise ValueError("initial paired common-random-number samples differ")
    if not _paired_uniform_schedules_valid(cases):
        raise ValueError("uniform cells do not share the registered schedule prefix")

    descriptives = _descriptives(cases)
    gates = _gates(cases)
    if artifact.get("development_contrasts") != descriptives:
        raise ValueError(
            "runner development descriptives differ from independent values"
        )
    if artifact.get("development_gates") != gates:
        raise ValueError("runner development gates differ from independent values")
    return {
        "schema": REPORT_SCHEMA,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": _sha256(artifact_path),
        "development_lock": str(lock_path.resolve()),
        "development_lock_sha256": lock_file_sha256,
        "development_lock_canonical_sha256": _canonical_hash(lock),
        "checked_source_files": list(REQUIRED_SOURCE_FILES),
        "all_checks_passed": True,
        "sealed_development_protocol_verified": True,
        "source_hashes_verified": True,
        "sampler_traces_independently_replayed": True,
        "uniform_schedules_independently_regenerated": True,
        "training_rollout_audits_independently_recomputed": True,
        "evaluation_samples_and_crns_independently_recomputed": True,
        "transition_auc_independently_recomputed": True,
        "capacity_contract_independently_verified": True,
        "no_hindsight_verified": True,
        "primary_metric": "auc_hardest_pass_fixed_transitions",
        "supporting_metric": "auc_mean_pass_fixed_transitions",
        "development_gates": gates,
        "development_contrasts": descriptives,
        "confirmatory_claim_authorized": False,
        "confirmatory_seeds_verified_untouched_by_artifact": True,
        "interpretation": (
            "sealed development feasibility only; three-seed effects are descriptive "
            "and cannot support a confirmatory performance claim"
        ),
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = verify(args.artifact, args.lock)
    output = args.output or args.artifact.with_name(
        f"{args.artifact.stem}_verification.json"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite verification report: {output}")
    _atomic_write(output, report)
    print(json.dumps({"wrote": str(output), "all_checks_passed": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
