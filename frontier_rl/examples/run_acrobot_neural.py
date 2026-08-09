"""Preregistered neural Gymnasium Acrobot validation for Curriculum MaxRL.

This runner implements three deliberately separate stages:

``pilot``
    Exploratory learning-rate calibration on seeds 100+.
``core``
    Transition-matched curriculum-by-sharing causal matrix, without hindsight.
``scale``
    Nonzero-optimizer-update-matched hindsight-scale by learning-rate study.

The frozen design and interpretation rules live in
``ACROBOT_NEURAL_PROTOCOL.md``.  Existing artifacts are never overwritten
unless ``--overwrite`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium

from frontier_rl.adapters.acrobot_neural import (
    AcrobotNeuralSpace,
    TanhCategoricalActor,
)
from frontier_rl.estimators import maxrl_success_weights, maxrl_weights
from frontier_rl.teacher import FrontierTeacher


THRESHOLDS = (-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.7, 1.0)
N_ROLLOUTS = 16
TEACHER_DECAY = 0.7
TEACHER_FLOOR = 0.1
TEACHER_GAMMA = 1.0
PILOT_LRS = (1e-4, 3e-4, 1e-3, 3e-3)
SCALE_LR_MULTIPLIERS = (0.5, 1.0, 2.0)
HINDSIGHT_SCALES = (0.0, 1.0, 2.0)
CORE_ARCHITECTURE_SPECS = (
    ("shared_h64", "shared", 64),
    ("disjoint_total_h8", "disjoint_total_budget", 8),
    ("disjoint_active_h64", "disjoint_active_capacity", 64),
)
CORE_ARCHITECTURES = tuple(spec[1] for spec in CORE_ARCHITECTURE_SPECS)
PILOT_WARMUP_TRANSITIONS = 200_000
PILOT_PROGRESS_TRANSITIONS = 1_000_000
PROTOCOL_V1_PATH = Path(__file__).resolve().with_name(
    "ACROBOT_NEURAL_PROTOCOL.md"
)
PROTOCOL_PATH = Path(__file__).resolve().with_name(
    "ACROBOT_NEURAL_PROTOCOL_V2.md"
)
PROTOCOL_V3_PATH = Path(__file__).resolve().with_name(
    "ACROBOT_NEURAL_PROTOCOL_V3.md"
)
V3_VERIFIER_PATH = Path(__file__).resolve().with_name(
    "analyze_acrobot_v3_confirmatory.py"
)


@dataclass(frozen=True)
class Condition:
    """One fully specified experimental condition."""

    name: str
    stage: str
    sampling: str
    architecture: str
    hidden_size: int
    learning_rate: float
    hindsight_scale: float = 0.0
    hindsight_estimator: str = "maxrl"
    gamma: float = TEACHER_GAMMA
    lr_multiplier: float | None = None

    @property
    def hindsight(self) -> bool:
        return self.hindsight_scale > 0.0


@dataclass(frozen=True)
class RunBudget:
    """Exactly one stopping coordinate is primary for a stage."""

    transition_budget: int | None = None
    optimizer_update_budget: int | None = None
    transition_safety_cap: int | None = None

    def validate(self) -> None:
        transition_matched = self.transition_budget is not None
        update_matched = self.optimizer_update_budget is not None
        if transition_matched == update_matched:
            raise ValueError(
                "set exactly one of transition_budget or optimizer_update_budget"
            )
        if transition_matched and self.transition_budget < 1:
            raise ValueError("transition_budget must be positive")
        if update_matched:
            if self.optimizer_update_budget < 1:
                raise ValueError("optimizer_update_budget must be positive")
            if self.transition_safety_cap is None or self.transition_safety_cap < 1:
                raise ValueError("update-matched runs require a positive safety cap")


def _float_label(value: float) -> str:
    """Filesystem/JSON-safe deterministic decimal label."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def core_conditions(
    base_lr: float,
    architectures: Sequence[str] | None = None,
) -> tuple[Condition, ...]:
    """Return the registered core cells, optionally for chosen architectures."""
    selected = CORE_ARCHITECTURES if architectures is None else tuple(architectures)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("core architectures must be a non-empty unique subset")
    unknown = set(selected) - set(CORE_ARCHITECTURES)
    if unknown:
        raise ValueError(f"unknown core architectures: {sorted(unknown)}")
    return tuple(
        Condition(
            name=f"{sampling}_{label}",
            stage="core",
            sampling=sampling,
            architecture=architecture,
            hidden_size=hidden,
            learning_rate=base_lr,
        )
        for label, architecture, hidden in CORE_ARCHITECTURE_SPECS
        if architecture in selected
        for sampling in ("uniform", "teacher")
    )


def pilot_conditions(lrs: Sequence[float] = PILOT_LRS) -> tuple[Condition, ...]:
    return tuple(
        Condition(
            name=f"lr_{_float_label(float(lr))}_{sampling}_shared_h64",
            stage="pilot",
            sampling=sampling,
            architecture="shared",
            hidden_size=64,
            learning_rate=float(lr),
        )
        for lr in lrs
        for sampling in ("uniform", "teacher")
    )


def scale_conditions(base_lr: float) -> tuple[Condition, ...]:
    return tuple(
        Condition(
            name=(
                f"lr_mult_{_float_label(multiplier)}_"
                f"hs_{_float_label(scale)}"
            ),
            stage="scale",
            sampling="teacher",
            architecture="shared",
            hidden_size=64,
            learning_rate=base_lr * multiplier,
            hindsight_scale=scale,
            lr_multiplier=multiplier,
        )
        for multiplier in SCALE_LR_MULTIPLIERS
        for scale in HINDSIGHT_SCALES
    )


def normalized_trapezoid(y: Sequence[float], x: Sequence[int]) -> float:
    """Normalized trapezoid area on an explicit monotone resource axis."""
    if len(y) != len(x) or not y:
        raise ValueError("x and y must be non-empty and have equal length")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.any(np.diff(x_arr) < 0):
        raise ValueError("resource coordinates must be nondecreasing")
    if not (np.isfinite(x_arr).all() and np.isfinite(y_arr).all()):
        raise ValueError("AUC inputs must be finite")
    if x_arr[-1] <= x_arr[0]:
        return float(y_arr[-1])
    area = float(np.trapezoid(y_arr, x_arr))
    return area / float(x_arr[-1] - x_arr[0])


def bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, n_boot: int = 20_000
) -> list[float]:
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.ndim != 1 or len(values_arr) < 1:
        raise ValueError("bootstrap input must be a non-empty vector")
    if not np.isfinite(values_arr).all():
        raise ValueError("bootstrap input must be finite")
    if len(values_arr) == 1:
        return [float(values_arr[0]), float(values_arr[0])]
    rng = np.random.default_rng(seed)
    draws = values_arr[
        rng.integers(0, len(values_arr), size=(n_boot, len(values_arr)))
    ].mean(axis=1)
    return [float(x) for x in np.quantile(draws, (0.025, 0.975))]


def exact_sign_flip_p(values: Sequence[float]) -> float:
    """Exact two-sided paired sign-flip randomization p-value."""
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.ndim != 1 or not 1 <= len(values_arr) <= 20:
        raise ValueError("exact sign-flip supports between 1 and 20 pairs")
    if not np.isfinite(values_arr).all():
        raise ValueError("sign-flip input must be finite")
    observed = abs(float(values_arr.mean()))
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values_arr)):
        null_stat = abs(float(np.dot(signs, values_arr) / len(values_arr)))
        extreme += null_stat >= observed - 1e-15
    return float(extreme / (2 ** len(values_arr)))


def holm_adjust(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm step-down correction with monotone adjusted p-values."""
    if not p_values:
        return {}
    ordered = sorted((float(p), name) for name, p in p_values.items())
    m = len(ordered)
    running = 0.0
    still_rejecting = True
    out = {}
    for rank, (p_value, name) in enumerate(ordered, start=1):
        multiplier = m - rank + 1
        running = max(running, multiplier * p_value)
        reject = still_rejecting and p_value <= alpha / multiplier
        if not reject:
            still_rejecting = False
        out[name] = {
            "raw_p": p_value,
            "holm_adjusted_p": float(min(running, 1.0)),
            "reject_familywise_0.05": bool(reject),
        }
    return out


def _teacher_for(condition: Condition, seed: int) -> FrontierTeacher:
    teacher = FrontierTeacher(
        len(THRESHOLDS),
        N_ROLLOUTS,
        decay=TEACHER_DECAY,
        floor=TEACHER_FLOOR,
        gamma=condition.gamma,
        seed=seed + 10_000,
    )
    if condition.sampling == "uniform":
        uniform = np.full(len(THRESHOLDS), 1.0 / len(THRESHOLDS))
        teacher.distribution = lambda: uniform.copy()
    elif condition.sampling != "teacher":
        raise ValueError(f"unknown sampling rule {condition.sampling!r}")
    return teacher


def _weights(rewards: np.ndarray, estimator: str) -> np.ndarray:
    """Route to the canonical project estimators; do not duplicate formulas."""
    if estimator == "maxrl":
        return maxrl_weights(rewards)
    if estimator == "success_only":
        return maxrl_success_weights(rewards)
    raise ValueError(f"unknown estimator {estimator!r}")


def _group_transitions(group) -> int:
    if len(group.infos) != len(group.trajectories):
        raise RuntimeError("adapter returned misaligned infos and trajectories")
    if all(isinstance(info, dict) and "n_steps" in info for info in group.infos):
        count = sum(int(info["n_steps"]) for info in group.infos)
    else:
        count = sum(len(trajectory) for trajectory in group.trajectories)
    if count < 1:
        raise RuntimeError("a rollout group must contain at least one transition")
    return count


def _parameter_counts(actor) -> tuple[int, int]:
    total = getattr(actor, "parameter_count", None)
    active = getattr(actor, "active_parameter_count", None)
    total = total() if callable(total) else total
    active = active() if callable(active) else active
    if total is None:
        total = sum(
            int(value.size)
            for value in vars(actor).values()
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)
        )
    if active is None:
        active = total
    return int(total), int(active)


def _actor_is_finite(actor) -> bool:
    arrays = [
        value
        for value in vars(actor).values()
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)
    ]
    return bool(arrays) and all(np.isfinite(value).all() for value in arrays)


def _update(actor, task_id: int, trajectories: list, weights: np.ndarray) -> dict:
    """Apply exactly one group update and normalize adapter diagnostics."""
    if not np.any(np.asarray(weights, dtype=np.float64) != 0.0):
        raise ValueError("_update must only be called for a nonzero update")
    returned = actor.update(task_id, trajectories, weights)
    if returned is None:
        returned = getattr(actor, "last_update_stats", {})
    if callable(returned):
        returned = returned()
    if returned is None:
        returned = {}
    if not isinstance(returned, dict):
        raise TypeError("actor.update diagnostics must be a dict or None")

    aliases = {
        "gradient_norm": ("gradient_norm", "grad_norm"),
        "update_norm": ("update_norm", "step_norm"),
        "mean_policy_entropy": (
            "mean_policy_entropy",
            "policy_entropy",
            "entropy",
        ),
    }
    diagnostics = {}
    for canonical, candidates in aliases.items():
        value = next((returned[key] for key in candidates if key in returned), None)
        if value is not None:
            value = float(value)
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite {canonical}")
            diagnostics[canonical] = value
    applied = returned.get("applied")
    if applied is None:
        applied = diagnostics.get("update_norm", 0.0) != 0.0
    diagnostics["applied"] = bool(applied)
    if not _actor_is_finite(actor):
        raise FloatingPointError("actor contains a non-finite parameter")
    return diagnostics


def _nonmutating_gradient_diagnostics(
    actor, task_id: int, trajectories: list, weights: np.ndarray
) -> dict:
    """Compute scale-zero auxiliary diagnostics without an optimizer update."""
    if not hasattr(actor, "gradient_diagnostics"):
        raise AttributeError(
            "scale-zero audit requires actor.gradient_diagnostics(...)"
        )
    before = _training_state_fingerprint(None, actor)
    returned = actor.gradient_diagnostics(task_id, trajectories, weights)
    after = _training_state_fingerprint(None, actor)
    if before != after:
        raise RuntimeError("gradient_diagnostics mutated actor training state")
    if not isinstance(returned, dict):
        raise TypeError("gradient_diagnostics must return a dict")
    out = {}
    for key, value in returned.items():
        if isinstance(value, (bool, str)) or value is None:
            out[key] = value
        elif isinstance(value, (int, np.integer)):
            out[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise FloatingPointError(
                    f"non-finite auxiliary gradient diagnostic {key}"
                )
            out[key] = numeric
    if returned.get("mutated") not in (None, False):
        raise RuntimeError("gradient_diagnostics reported parameter mutation")
    if not _actor_is_finite(actor):
        raise FloatingPointError("gradient diagnostics left invalid parameters")
    return out


def _training_state_fingerprint(env, actor) -> str:
    """Hash parameters, counters, and RNGs relevant to later training.

    Gym's physical state is intentionally absent: every training episode begins
    with an explicit seed drawn from the adapter RNG. Evaluation uses a fresh
    Gym environment, so the adapter/actor state below is sufficient for exact
    mechanical cadence invariance.
    """
    records = {}
    for owner_name, owner in (("space", env), ("actor", actor)):
        if owner is None:
            continue
        for attribute, value in vars(owner).items():
            if isinstance(value, np.random.Generator):
                captured = ("generator", copy.deepcopy(value.bit_generator.state))
            elif isinstance(value, np.ndarray):
                captured = (
                    "ndarray",
                    value.dtype.str,
                    value.shape,
                    value.tobytes(order="C"),
                )
            elif isinstance(value, (str, int, float, bool, type(None), tuple)):
                captured = ("scalar", copy.deepcopy(value))
            elif isinstance(value, dict):
                captured = ("dict", copy.deepcopy(value))
            else:
                # Exclude aliases to actor/policy and the mutable Gym object.
                continue
            records[f"{owner_name}.{attribute}"] = captured
    encoded = pickle.dumps(records, protocol=5)
    return hashlib.sha256(encoded).hexdigest()


def _evaluate(env, *, n: int, seed: int) -> dict:
    """Normalize the adapter's RNG-preserving evaluation record."""
    if not hasattr(env, "evaluate"):
        raise AttributeError("AcrobotNeuralSpace must expose evaluate(n, seed)")
    actor = getattr(env, "actor", None)
    state_before = _training_state_fingerprint(env, actor)
    raw = env.evaluate(n=n, seed=seed)
    state_after = _training_state_fingerprint(env, actor)
    if state_before != state_after:
        raise RuntimeError(
            "evaluation changed actor/adapter training state (cadence violation)"
        )
    if not isinstance(raw, dict) or "pass_rates" not in raw:
        raise TypeError("evaluate must return a dict containing pass_rates")
    rates = np.asarray(raw["pass_rates"], dtype=np.float64)
    if rates.shape != (len(THRESHOLDS),):
        raise ValueError(
            f"expected {len(THRESHOLDS)} pass rates, got shape {rates.shape}"
        )
    if not np.isfinite(rates).all() or np.any((rates < 0.0) | (rates > 1.0)):
        raise FloatingPointError("evaluation pass rates are invalid")

    aliases = {
        "native_success_rate": ("native_success_rate", "hardest_success_rate"),
        "mean_native_return": ("mean_native_return", "mean_return"),
        "mean_censored_time_to_goal": (
            "mean_censored_time_to_goal",
            "mean_time_to_goal",
        ),
        "mean_policy_entropy": ("mean_policy_entropy", "policy_entropy", "entropy"),
    }
    out = {"pass_rates": rates.tolist(), "training_rng_preserved": True}
    defaults = {
        "native_success_rate": float(rates[-1]),
        "mean_native_return": None,
        "mean_censored_time_to_goal": None,
        "mean_policy_entropy": None,
    }
    for canonical, candidates in aliases.items():
        value = next((raw[key] for key in candidates if key in raw), defaults[canonical])
        if value is not None:
            value = float(value)
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite evaluation {canonical}")
        out[canonical] = value
    return out


def _append_evaluation(curves: dict, evaluation: dict, transitions: int,
                       optimizer_updates: int) -> None:
    curves["x_transitions"].append(int(transitions))
    curves["x_optimizer_updates"].append(int(optimizer_updates))
    rates = evaluation["pass_rates"]
    curves["pass_rate_curve"].append(rates)
    curves["mean_pass_curve"].append(float(np.mean(rates)))
    curves["hardest_pass_curve"].append(float(rates[-1]))
    curves["evaluation_rng_preserved"].append(
        bool(evaluation["training_rng_preserved"])
    )
    for key in (
        "native_success_rate",
        "mean_native_return",
        "mean_censored_time_to_goal",
        "mean_policy_entropy",
    ):
        curves[f"{key}_curve"].append(evaluation[key])


def _new_curves() -> dict:
    return {
        "x_transitions": [],
        "x_optimizer_updates": [],
        "pass_rate_curve": [],
        "mean_pass_curve": [],
        "hardest_pass_curve": [],
        "evaluation_rng_preserved": [],
        "native_success_rate_curve": [],
        "mean_native_return_curve": [],
        "mean_censored_time_to_goal_curve": [],
        "mean_policy_entropy_curve": [],
    }


def _continue_training(budget: RunBudget, transitions: int,
                       optimizer_updates: int) -> bool:
    budget.validate()
    if budget.transition_budget is not None:
        return transitions < budget.transition_budget
    return (
        optimizer_updates < budget.optimizer_update_budget
        and transitions < budget.transition_safety_cap
    )


def run_condition(
    condition: Condition,
    seed: int,
    *,
    budget: RunBudget,
    eval_interval_transitions: int = 100_000,
    eval_interval_updates: int = 10,
    eval_n: int = 32,
    eval_seed_base: int = 1_000_000,
) -> dict:
    """Run one condition with complete-group stopping and raw diagnostics."""
    budget.validate()
    if condition.stage == "scale" and condition.architecture != "shared":
        raise ValueError("hindsight scale study is restricted to the shared actor")
    if condition.hindsight and condition.architecture != "shared":
        raise ValueError("hindsight is invalid for disjoint actors")
    if eval_interval_transitions < 1 or eval_interval_updates < 1 or eval_n < 1:
        raise ValueError("evaluation intervals and eval_n must be positive")

    actor = TanhCategoricalActor(
        n_tasks=len(THRESHOLDS),
        hidden_size=condition.hidden_size,
        learning_rate=condition.learning_rate,
        seed=seed,
        mode=condition.architecture,
    )
    env = AcrobotNeuralSpace(
        actor=actor,
        thresholds=THRESHOLDS,
        seed=seed + 1_000,
    )
    teacher = _teacher_for(condition, seed)
    total_parameters, active_parameters = _parameter_counts(actor)
    expected_counts = {
        "shared": (640, 640),
        "disjoint_total_budget": (640, 80),
        "disjoint_active_capacity": (5_120, 640),
    }
    if (total_parameters, active_parameters) != expected_counts[condition.architecture]:
        raise RuntimeError(
            "actor parameter-count contract changed: "
            f"{condition.architecture} has {(total_parameters, active_parameters)}, "
            f"expected {expected_counts[condition.architecture]}"
        )

    transitions = 0
    optimizer_updates = 0
    sampled_groups = 0
    live_groups = 0
    live_applied_updates = 0
    dead_groups = 0
    all_pass_groups = 0
    relabeled_groups = 0
    relabel_candidates = 0
    unscaled_aux_gradient_previews = 0
    zero_gradient_update_attempts = 0
    task_groups = np.zeros(len(THRESHOLDS), dtype=np.int64)
    task_rollouts = np.zeros(len(THRESHOLDS), dtype=np.int64)
    task_successes = np.zeros(len(THRESHOLDS), dtype=np.int64)
    task_transitions = np.zeros(len(THRESHOLDS), dtype=np.int64)
    update_diagnostics = []
    zero_gradient_diagnostics = []
    auxiliary_gradient_diagnostics = []
    group_diagnostics = []
    curves = _new_curves()
    eval_seed = eval_seed_base + seed
    next_transition_eval = eval_interval_transitions
    next_update_eval = eval_interval_updates
    wall_start = time.perf_counter()

    try:
        initial = _evaluate(env, n=eval_n, seed=eval_seed)
        _append_evaluation(curves, initial, transitions, optimizer_updates)

        while _continue_training(budget, transitions, optimizer_updates):
            sampled_probs = np.asarray(teacher.distribution(), dtype=np.float64)
            if sampled_probs.shape != (len(THRESHOLDS),):
                raise RuntimeError("teacher returned the wrong distribution shape")
            if (
                not np.isfinite(sampled_probs).all()
                or np.any(sampled_probs < 0.0)
                or not np.isclose(sampled_probs.sum(), 1.0, atol=1e-12)
            ):
                raise FloatingPointError("teacher returned invalid probabilities")
            teacher_tv_from_uniform = float(
                0.5
                * np.abs(sampled_probs - 1.0 / len(THRESHOLDS)).sum()
            )
            task_id = int(teacher.rng.choice(len(THRESHOLDS), p=sampled_probs))
            transition_start = transitions
            group = env.rollout_group(task_id, N_ROLLOUTS)
            rewards = np.asarray(group.rewards, dtype=np.float64)
            if rewards.shape != (N_ROLLOUTS,) or not np.all(
                (rewards == 0.0) | (rewards == 1.0)
            ):
                raise RuntimeError("adapter rewards must be an N-vector of binary flags")

            group_transitions = _group_transitions(group)
            transitions += group_transitions
            sampled_groups += 1
            task_groups[task_id] += 1
            task_rollouts[task_id] += N_ROLLOUTS
            task_successes[task_id] += int(rewards.sum())
            task_transitions[task_id] += group_transitions
            teacher.observe(task_id, rewards)

            k = int(rewards.sum())
            update_source = None
            credited_task = task_id
            diagnostics = {}
            if 0 < k < N_ROLLOUTS:
                live_groups += 1
                weights = _weights(rewards, "maxrl")
                diagnostics = _update(actor, task_id, group.trajectories, weights)
                if diagnostics.pop("applied"):
                    optimizer_updates += 1
                    live_applied_updates += 1
                    update_source = "requested_live"
                else:
                    zero_gradient_update_attempts += 1
                    zero_gradient_diagnostics.append(
                        {
                            "after_group": sampled_groups,
                            "transitions": transitions,
                            "source": "requested_live",
                            "requested_task": task_id,
                            "credited_task": task_id,
                            **diagnostics,
                        }
                    )
            elif k == N_ROLLOUTS:
                all_pass_groups += 1
            else:
                dead_groups += 1
                # Every scale cell, including scale zero, constructs and checks
                # the identical relabel. Scale zero previews its unscaled
                # auxiliary gradient but must not mutate/count an optimizer step.
                if condition.stage == "scale" or condition.hindsight:
                    relabel = env.relabel(group)
                    if relabel is not None:
                        if len(relabel) == 3:
                            credited_task, relabeled_rewards, relabeled_trajs = relabel
                        elif len(relabel) == 2:
                            credited_task, relabeled_rewards = relabel
                            relabeled_trajs = group.trajectories
                        else:
                            raise RuntimeError("relabel must return 2 or 3 values")
                        relabeled_rewards = np.asarray(
                            relabeled_rewards, dtype=np.float64
                        )
                        relabeled_k = int(relabeled_rewards.sum())
                        if (
                            relabeled_rewards.shape != (N_ROLLOUTS,)
                            or not np.all(
                                (relabeled_rewards == 0.0)
                                | (relabeled_rewards == 1.0)
                            )
                        ):
                            raise RuntimeError("relabel returned invalid binary rewards")
                        if not (0 <= int(credited_task) < task_id):
                            raise RuntimeError(
                                "hindsight must credit a strictly lower threshold"
                            )
                        if not (0 < relabeled_k < N_ROLLOUTS):
                            raise RuntimeError(
                                "hindsight must choose a separating mixed lower task"
                            )
                        relabel_candidates += 1
                        unscaled_weights = _weights(
                            relabeled_rewards, condition.hindsight_estimator
                        )
                        if condition.stage == "scale" and condition.hindsight_scale == 0.0:
                            preview = _nonmutating_gradient_diagnostics(
                                actor,
                                int(credited_task),
                                relabeled_trajs,
                                unscaled_weights,
                            )
                            unscaled_aux_gradient_previews += 1
                            auxiliary_gradient_diagnostics.append(
                                {
                                    "after_group": sampled_groups,
                                    "transitions": transitions,
                                    "requested_task": task_id,
                                    "credited_task": int(credited_task),
                                    "applied": False,
                                    **preview,
                                }
                            )
                        weights = unscaled_weights * condition.hindsight_scale
                        if np.any(weights != 0.0):
                            diagnostics = _update(
                                actor,
                                int(credited_task),
                                relabeled_trajs,
                                weights,
                            )
                            if diagnostics.pop("applied"):
                                optimizer_updates += 1
                                relabeled_groups += 1
                                update_source = "hindsight_relabel"
                            else:
                                zero_gradient_update_attempts += 1
                                zero_gradient_diagnostics.append(
                                    {
                                        "after_group": sampled_groups,
                                        "transitions": transitions,
                                        "source": "hindsight_relabel",
                                        "requested_task": task_id,
                                        "credited_task": int(credited_task),
                                        **diagnostics,
                                    }
                                )

            regime = "dead" if k == 0 else "all_pass" if k == N_ROLLOUTS else "mixed"
            group_diagnostics.append(
                {
                    "group": sampled_groups,
                    "transition_start": transition_start,
                    "transition_end": transitions,
                    "n_transitions": group_transitions,
                    "task_id": task_id,
                    "success_count": k,
                    "regime": regime,
                    "teacher_tv_from_uniform": teacher_tv_from_uniform,
                    "sampled_task_probability": float(sampled_probs[task_id]),
                    "optimizer_updates_after_group": optimizer_updates,
                    "update_source": update_source,
                }
            )

            if update_source is not None:
                update_diagnostics.append(
                    {
                        "optimizer_update": optimizer_updates,
                        "after_group": sampled_groups,
                        "transitions": transitions,
                        "source": update_source,
                        "requested_task": task_id,
                        "credited_task": int(credited_task),
                        **diagnostics,
                    }
                )

            due = False
            if budget.transition_budget is not None and transitions >= next_transition_eval:
                due = True
                while next_transition_eval <= transitions:
                    next_transition_eval += eval_interval_transitions
            if (
                budget.optimizer_update_budget is not None
                and optimizer_updates >= next_update_eval
            ):
                due = True
                while next_update_eval <= optimizer_updates:
                    next_update_eval += eval_interval_updates
            if not _continue_training(budget, transitions, optimizer_updates):
                due = True
            if due:
                evaluation = _evaluate(env, n=eval_n, seed=eval_seed)
                _append_evaluation(
                    curves, evaluation, transitions, optimizer_updates
                )
    finally:
        env.close()
    wall_seconds = time.perf_counter() - wall_start

    if sampled_groups != live_groups + dead_groups + all_pass_groups:
        raise RuntimeError("group accounting mismatch")
    if int(task_groups.sum()) != sampled_groups:
        raise RuntimeError("per-task group accounting mismatch")
    if int(task_rollouts.sum()) != sampled_groups * N_ROLLOUTS:
        raise RuntimeError("rollout-attempt accounting mismatch")
    if int(task_transitions.sum()) != transitions:
        raise RuntimeError("per-task transition accounting mismatch")
    if sum(record["n_transitions"] for record in group_diagnostics) != transitions:
        raise RuntimeError("raw group transition accounting mismatch")
    if optimizer_updates != live_applied_updates + relabeled_groups:
        raise RuntimeError("nonzero optimizer-update accounting mismatch")
    if len(update_diagnostics) != optimizer_updates:
        raise RuntimeError("update diagnostic accounting mismatch")
    if unscaled_aux_gradient_previews > relabel_candidates:
        raise RuntimeError("auxiliary preview accounting mismatch")
    if not all(curves["evaluation_rng_preserved"]):
        raise RuntimeError("evaluation cadence changed a training RNG")
    if len(curves["x_transitions"]) < 2:
        raise RuntimeError("run ended without a final evaluation")

    transition_auc = normalized_trapezoid(
        curves["mean_pass_curve"], curves["x_transitions"]
    )
    update_auc = normalized_trapezoid(
        curves["mean_pass_curve"], curves["x_optimizer_updates"]
    )
    final_index = -1
    reached_update_budget = (
        budget.optimizer_update_budget is None
        or optimizer_updates >= budget.optimizer_update_budget
    )
    return {
        "seed": seed,
        "numeric_valid": True,
        "transitions": transitions,
        "sampled_groups": sampled_groups,
        "rollout_attempts": sampled_groups * N_ROLLOUTS,
        "optimizer_updates": optimizer_updates,
        "reached_optimizer_update_budget": reached_update_budget,
        "transition_cap_censored": bool(
            budget.optimizer_update_budget is not None and not reached_update_budget
        ),
        "live_groups": live_groups,
        "live_applied_updates": live_applied_updates,
        "dead_groups": dead_groups,
        "all_pass_groups": all_pass_groups,
        "relabeled_groups": relabeled_groups,
        "relabel_candidates": relabel_candidates,
        "unscaled_aux_gradient_previews": unscaled_aux_gradient_previews,
        "zero_gradient_update_attempts": zero_gradient_update_attempts,
        "task_groups": task_groups.tolist(),
        "task_rollouts": task_rollouts.tolist(),
        "task_successes": task_successes.tolist(),
        "task_transitions": task_transitions.tolist(),
        "total_parameters": total_parameters,
        "active_parameters_per_task": active_parameters,
        "accounting_valid": True,
        "verifier_relabel_checks_valid": True,
        "evaluation_cadence_invariant": True,
        "wall_seconds": wall_seconds,
        "training_transitions_per_wall_second": (
            float(transitions / wall_seconds) if wall_seconds > 0.0 else None
        ),
        **curves,
        "auc_mean_pass_by_transitions": transition_auc,
        "auc_mean_pass_by_optimizer_updates": update_auc,
        "initial_mean_pass": curves["mean_pass_curve"][0],
        "final_mean_pass": curves["mean_pass_curve"][final_index],
        "final_hardest_pass": curves["hardest_pass_curve"][final_index],
        "final_native_success_rate": curves["native_success_rate_curve"][final_index],
        "final_mean_native_return": curves["mean_native_return_curve"][final_index],
        "final_mean_censored_time_to_goal": curves[
            "mean_censored_time_to_goal_curve"
        ][final_index],
        "update_diagnostics": update_diagnostics,
        "zero_gradient_diagnostics": zero_gradient_diagnostics,
        "auxiliary_gradient_diagnostics": auxiliary_gradient_diagnostics,
        "group_diagnostics": group_diagnostics,
    }


SUMMARY_METRICS = (
    "auc_mean_pass_by_transitions",
    "auc_mean_pass_by_optimizer_updates",
    "final_mean_pass",
    "final_hardest_pass",
    "final_native_success_rate",
    "final_mean_native_return",
    "final_mean_censored_time_to_goal",
    "transitions",
    "optimizer_updates",
    "live_groups",
    "live_applied_updates",
    "dead_groups",
    "all_pass_groups",
    "relabeled_groups",
    "relabel_candidates",
    "unscaled_aux_gradient_previews",
    "zero_gradient_update_attempts",
    "wall_seconds",
    "training_transitions_per_wall_second",
)


def summarize_runs(runs: list[dict], *, bootstrap_seed: int) -> dict:
    valid_runs = [run for run in runs if run.get("numeric_valid", False)]
    out = {
        "n_seeds": len(valid_runs),
        "n_requested_or_attempted": len(runs),
        "n_valid": len(valid_runs),
        "n_failed": len(runs) - len(valid_runs),
        "failed_seeds": [
            run.get("seed") for run in runs if not run.get("numeric_valid", False)
        ],
    }
    for index, key in enumerate(SUMMARY_METRICS):
        values = [run.get(key) for run in valid_runs]
        if not values:
            out[key] = {"available": False, "per_seed": []}
            continue
        if any(value is None for value in values):
            out[key] = {"available": False, "per_seed": values}
            continue
        values_arr = np.asarray(values, dtype=np.float64)
        out[key] = {
            "available": True,
            "mean": float(values_arr.mean()),
            "sample_std": (
                float(values_arr.std(ddof=1)) if len(values_arr) > 1 else 0.0
            ),
            "mean_ci95_paired_seed_bootstrap": bootstrap_mean_ci(
                values_arr, seed=bootstrap_seed + index
            ),
            "per_seed": values_arr.tolist(),
        }
    return out


def _contrast_values(cases: dict, coefficients: dict[str, float],
                     metric: str) -> np.ndarray:
    seeds = None
    result = None
    for case_name, coefficient in coefficients.items():
        runs = cases[case_name]["runs"]
        case_seeds = [run["seed"] for run in runs]
        if seeds is None:
            seeds = case_seeds
            result = np.zeros(len(runs), dtype=np.float64)
        elif case_seeds != seeds:
            raise ValueError(f"paired seed mismatch in contrast for {case_name}")
        result += coefficient * np.asarray(
            [run[metric] for run in runs], dtype=np.float64
        )
    return result


def _analyze_contrast(values: np.ndarray, *, description: str,
                      coefficients: dict[str, float], seed: int) -> dict:
    return {
        "description": description,
        "coefficients": coefficients,
        "mean_contrast": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "mean_ci95_paired_seed_bootstrap": bootstrap_mean_ci(values, seed=seed),
        "exact_paired_sign_flip_p_two_sided": exact_sign_flip_p(values),
        "per_seed_contrast": values.tolist(),
    }


def attach_core_analysis(result: dict) -> None:
    metric = "auc_mean_pass_by_transitions"
    specs = {
        "curriculum_efficacy_shared": {
            "teacher_shared_h64": 1.0,
            "uniform_shared_h64": -1.0,
        },
        "sharing_interaction_total_matched": {
            "teacher_shared_h64": 1.0,
            "uniform_shared_h64": -1.0,
            "teacher_disjoint_total_h8": -1.0,
            "uniform_disjoint_total_h8": 1.0,
        },
        "sharing_interaction_active_matched": {
            "teacher_shared_h64": 1.0,
            "uniform_shared_h64": -1.0,
            "teacher_disjoint_active_h64": -1.0,
            "uniform_disjoint_active_h64": 1.0,
        },
        "curriculum_shared_minus_disjoint_total": {
            "teacher_shared_h64": 1.0,
            "teacher_disjoint_total_h8": -1.0,
        },
        "curriculum_shared_minus_disjoint_active": {
            "teacher_shared_h64": 1.0,
            "teacher_disjoint_active_h64": -1.0,
        },
    }
    analyses = {}
    for index, (name, coefficients) in enumerate(specs.items()):
        values = _contrast_values(result["cases"], coefficients, metric)
        analyses[name] = _analyze_contrast(
            values,
            description=name.replace("_", " "),
            coefficients=coefficients,
            seed=20_000 + index,
        )
        analyses[name]["metric"] = metric

    adjusted = holm_adjust(
        {
            name: analyses[name]["exact_paired_sign_flip_p_two_sided"]
            for name in specs
        }
    )
    for name, correction in adjusted.items():
        analyses[name].update(correction)
    result["paired_core_contrasts"] = analyses
    result["primary_multiplicity"] = {
        "family": list(specs),
        "metric": metric,
        "method": "Holm step-down",
        "familywise_alpha": 0.05,
        "test": "exact two-sided paired sign-flip randomization",
    }
    efficacy = analyses["curriculum_efficacy_shared"]
    efficacy_supported = bool(
        efficacy["mean_contrast"] >= 0.03
        and efficacy["reject_familywise_0.05"]
    )
    control_claims = {
        "total_parameter_matched": (
            "sharing_interaction_total_matched",
            "curriculum_shared_minus_disjoint_total",
        ),
        "active_capacity_matched": (
            "sharing_interaction_active_matched",
            "curriculum_shared_minus_disjoint_active",
        ),
    }
    supported_controls = []
    for control_name, claim_names in control_claims.items():
        if all(
            analyses[name]["mean_contrast"] > 0.0
            and analyses[name]["reject_familywise_0.05"]
            for name in claim_names
        ):
            supported_controls.append(control_name)
    result["predeclared_core_decision"] = {
        "efficacy_requires": (
            "CS-US >= +0.03 and positive/Holm-significant in the five-test family"
        ),
        "efficacy_supported": efficacy_supported,
        "supported_transfer_capacity_controls": supported_controls,
        "strong_transfer_supported": efficacy_supported
        and len(supported_controls) == len(control_claims),
        "capacity_qualified_support": efficacy_supported
        and len(supported_controls) == 1,
    }


def attach_shared_core_analysis(result: dict) -> None:
    """Attach the V3 one-test shared-policy efficacy analysis.

    This deliberately evaluates no transfer or capacity claim. The only
    registered estimand is teacher-shared minus uniform-shared transition AUC.
    """

    expected_cases = {"uniform_shared_h64", "teacher_shared_h64"}
    if set(result["cases"]) != expected_cases:
        raise ValueError(
            "shared efficacy analysis requires exactly the two shared core cells"
        )
    metric = "auc_mean_pass_by_transitions"
    coefficients = {
        "teacher_shared_h64": 1.0,
        "uniform_shared_h64": -1.0,
    }
    values = _contrast_values(result["cases"], coefficients, metric)
    analysis = _analyze_contrast(
        values,
        description="exact curriculum teacher minus uniform, shared H64",
        coefficients=coefficients,
        seed=25_000,
    )
    analysis["metric"] = metric
    p_value = analysis["exact_paired_sign_flip_p_two_sided"]
    result["paired_core_contrasts"] = {
        "curriculum_efficacy_shared": analysis,
    }
    result["primary_multiplicity"] = {
        "family": ["curriculum_efficacy_shared"],
        "metric": metric,
        "method": "one predeclared test; no multiplicity adjustment",
        "familywise_alpha": 0.05,
        "test": "exact two-sided paired sign-flip randomization",
    }
    result["predeclared_core_decision"] = {
        "efficacy_requires": "CS-US >= +0.03 and exact two-sided p <= 0.05",
        "efficacy_supported": bool(
            analysis["mean_contrast"] >= 0.03 and p_value <= 0.05
        ),
        "transfer_claim_evaluated": False,
        "transfer_claim_supported": None,
    }


def select_pilot_learning_rate(result: dict, tie_tolerance: float = 0.01) -> dict:
    """Apply the frozen pooled pilot selection and launch gates."""
    by_lr: dict[float, dict] = {}
    for case_record in result["cases"].values():
        condition = case_record["config"]
        lr = float(condition["learning_rate"])
        sampling = condition["sampling"]
        by_lr.setdefault(lr, {})[sampling] = case_record["runs"]

    candidate_records = {}
    valid_scores = {}
    for lr, sampling_runs in sorted(by_lr.items()):
        if set(sampling_runs) != {"uniform", "teacher"}:
            raise ValueError(f"pilot LR {lr} lacks both sampling conditions")
        seeds_uniform = [run["seed"] for run in sampling_runs["uniform"]]
        seeds_teacher = [run["seed"] for run in sampling_runs["teacher"]]
        if seeds_uniform != seeds_teacher:
            raise ValueError(f"pilot LR {lr} has unpaired seeds")
        all_runs = sampling_runs["uniform"] + sampling_runs["teacher"]
        numeric_valid = all(run.get("numeric_valid", False) for run in all_runs)
        pooled_improvement = float(
            np.mean(
                [run["final_mean_pass"] - run["initial_mean_pass"] for run in all_runs]
            )
        )
        candidate_records[str(lr)] = {
            "numeric_valid": numeric_valid,
            "pooled_final_minus_initial_mean_pass": pooled_improvement,
        }
        if numeric_valid:
            valid_scores[lr] = pooled_improvement

    if not valid_scores:
        return {
            "selected_learning_rate": None,
            "candidates": candidate_records,
            "gates": {"all_pass": False, "reason": "no numerically valid rate"},
        }
    best = max(valid_scores.values())
    selected = min(
        lr for lr, score in valid_scores.items() if score >= best - tie_tolerance
    )
    selected_runs = by_lr[selected]
    all_selected_runs = [
        run
        for sampling in ("uniform", "teacher")
        for run in selected_runs[sampling]
    ]
    updates_nonzero = all(run["optimizer_updates"] > 0 for run in all_selected_runs)
    implementation_invariants = all(
        run["numeric_valid"]
        and run["accounting_valid"]
        and run["verifier_relabel_checks_valid"]
        and run["evaluation_cadence_invariant"]
        for run in all_selected_runs
    )

    post_warmup_groups = [
        group
        for run in all_selected_runs
        for group in run["group_diagnostics"]
        if group["transition_start"] >= PILOT_WARMUP_TRANSITIONS
    ]
    regimes = {group["regime"] for group in post_warmup_groups}
    mixed_fraction = (
        float(
            np.mean([group["regime"] == "mixed" for group in post_warmup_groups])
        )
        if post_warmup_groups
        else 0.0
    )
    teacher_post_warmup_tvs = [
        group["teacher_tv_from_uniform"]
        for run in selected_runs["teacher"]
        for group in run["group_diagnostics"]
        if group["transition_start"] >= PILOT_WARMUP_TRANSITIONS
    ]
    mean_teacher_tv = (
        float(np.mean(teacher_post_warmup_tvs))
        if teacher_post_warmup_tvs
        else 0.0
    )

    def mean_pass_at_or_after(run: dict, coordinate: int) -> float | None:
        for x, value in zip(run["x_transitions"], run["mean_pass_curve"]):
            if x >= coordinate:
                return float(value)
        return None

    progress_records = []
    for run in selected_runs["teacher"]:
        by_one_million = mean_pass_at_or_after(run, PILOT_PROGRESS_TRANSITIONS)
        qualifies = bool(
            by_one_million is not None
            and by_one_million - run["initial_mean_pass"] >= 0.03
            and by_one_million < 0.95
        )
        progress_records.append(
            {
                "seed": run["seed"],
                "mean_pass_at_or_after_1m": by_one_million,
                "improvement_from_step_zero": (
                    None
                    if by_one_million is None
                    else by_one_million - run["initial_mean_pass"]
                ),
                "below_0p95_saturation": (
                    None if by_one_million is None else by_one_million < 0.95
                ),
                "qualifies": qualifies,
            }
        )
    progress_gate = sum(record["qualifies"] for record in progress_records) >= 2

    uniform_by_seed = {run["seed"]: run for run in selected_runs["uniform"]}
    teacher_by_seed = {run["seed"]: run for run in selected_runs["teacher"]}
    if uniform_by_seed.keys() != teacher_by_seed.keys():
        raise ValueError("selected pilot conditions are not paired")
    auc_deltas = [
        teacher_by_seed[seed]["auc_mean_pass_by_transitions"]
        - uniform_by_seed[seed]["auc_mean_pass_by_transitions"]
        for seed in sorted(uniform_by_seed)
    ]
    positive_direction = float(np.mean(auc_deltas)) > 0.0
    serial_seconds_per_transition = sum(
        run["wall_seconds"] for run in all_selected_runs
    ) / sum(run["transitions"] for run in all_selected_runs)
    projected_core_serial_hours = (
        serial_seconds_per_transition * 2_000_000 * 6 * 12 / 3600.0
    )
    runtime_feasible = projected_core_serial_hours <= 24.0

    # Conservative scale feasibility projection from the slowest selected-rate
    # no-hindsight update density. This freezes 400, one fallback to 250, or stop.
    if updates_nonzero:
        projected_transitions_per_update = max(
            run["transitions"] / run["optimizer_updates"]
            for run in all_selected_runs
        )
        projected_for_400 = projected_transitions_per_update * 400
        projected_for_250 = projected_transitions_per_update * 250
    else:
        projected_for_400 = None
        projected_for_250 = None
    if projected_for_400 is not None and projected_for_400 <= 4_000_000:
        scale_update_budget = 400
    elif projected_for_250 is not None and projected_for_250 <= 4_000_000:
        scale_update_budget = 250
    else:
        scale_update_budget = None

    gates = {
        "implementation_accounting_cadence_finite_verifier": implementation_invariants,
        "nonzero_update_every_shared_run": updates_nonzero,
        "selected_rate_numerically_valid": True,
        "all_k_regimes_after_warmup": regimes == {"dead", "mixed", "all_pass"},
        "mixed_group_fraction_at_least_0p10_after_warmup": mixed_fraction >= 0.10,
        "teacher_mean_tv_from_uniform_above_0p05_after_warmup": mean_teacher_tv > 0.05,
        "two_of_three_teacher_runs_improve_0p03_by_1m_without_saturation": progress_gate,
        "mean_teacher_minus_uniform_auc_positive": positive_direction,
        "projected_core_serial_runtime_at_most_24h": runtime_feasible,
    }
    gates["all_pass"] = all(gates.values())
    gates["interpretation"] = (
        "exploratory resource-allocation gate; not confirmatory evidence"
    )
    return {
        "selected_learning_rate": selected,
        "tie_tolerance": tie_tolerance,
        "candidates": candidate_records,
        "selected_teacher_minus_uniform_auc_per_seed": auc_deltas,
        "warmup_diagnostics": {
            "warmup_transitions": PILOT_WARMUP_TRANSITIONS,
            "observed_regimes": sorted(regimes),
            "pooled_mixed_group_fraction": mixed_fraction,
            "teacher_mean_tv_from_uniform": mean_teacher_tv,
        },
        "one_million_transition_progress": progress_records,
        "runtime_projection": {
            "serial_seconds_per_training_transition": serial_seconds_per_transition,
            "projected_core_serial_hours": projected_core_serial_hours,
            "feasibility_limit_hours": 24.0,
        },
        "scale_budget_freeze": {
            "projected_transitions_for_400_updates": projected_for_400,
            "projected_transitions_for_250_updates": projected_for_250,
            "transition_cap": 4_000_000,
            "selected_nonzero_update_budget": scale_update_budget,
            "rule": "400 if feasible; otherwise one fallback to 250; otherwise stop",
        },
        "frozen_core_schedule": {
            "transition_budget": 2_000_000,
            "eval_interval_transitions": result["protocol"][
                "eval_interval_transitions"
            ],
            "eval_n_per_task": result["protocol"]["eval_n_per_task"],
        },
        "gates": gates,
    }


def attach_scale_analysis(result: dict) -> None:
    metric = "auc_mean_pass_by_optimizer_updates"
    case_name = lambda multiplier, scale: (
        f"lr_mult_{_float_label(multiplier)}_hs_{_float_label(scale)}"
    )
    specs = {
        "scale1_minus_scale0_base_lr": {
            case_name(1.0, 1.0): 1.0,
            case_name(1.0, 0.0): -1.0,
        },
        "scale2_minus_scale1_base_lr": {
            case_name(1.0, 2.0): 1.0,
            case_name(1.0, 1.0): -1.0,
        },
        "iso_auxiliary_step_interaction": {
            case_name(0.5, 2.0): 1.0,
            case_name(0.5, 0.0): -1.0,
            case_name(1.0, 1.0): -1.0,
            case_name(1.0, 0.0): 1.0,
        },
    }
    analyses = {}
    p_values = {}
    descriptions = {
        "scale1_minus_scale0_base_lr": "scale 1 minus scale 0 at base LR",
        "scale2_minus_scale1_base_lr": "scale 2 minus scale 1 at base LR",
        "iso_auxiliary_step_interaction": (
            "[(scale 2, half LR)-(scale 0, half LR)] minus "
            "[(scale 1, base LR)-(scale 0, base LR)]"
        ),
    }
    for name, coefficients in specs.items():
        values = _contrast_values(result["cases"], coefficients, metric)
        analyses[name] = _analyze_contrast(
            values,
            description=descriptions[name],
            coefficients=coefficients,
            seed=30_000 + len(analyses),
        )
        analyses[name]["metric"] = metric
        p_values[name] = analyses[name]["exact_paired_sign_flip_p_two_sided"]

    adjusted = holm_adjust(p_values)
    for name, correction in adjusted.items():
        analyses[name].update(correction)
    result["paired_scale_contrasts"] = analyses
    result["scale_multiplicity"] = {
        "family": list(p_values),
        "metric": metric,
        "method": "Holm step-down",
        "familywise_alpha": 0.05,
        "test": "exact two-sided paired sign-flip randomization",
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _resolve_protocol_document(path: Path) -> Path:
    """Resolve a protocol file and confine it to this project checkout."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    project_root = PROJECT_ROOT.resolve()
    try:
        candidate.relative_to(project_root)
    except ValueError as error:
        raise ValueError(
            "--protocol-document must be a file inside the project"
        ) from error
    if not candidate.is_file():
        raise ValueError(
            f"--protocol-document does not name an existing file: {candidate}"
        )
    return candidate


def provenance(protocol_document: Path = PROTOCOL_PATH) -> dict:
    protocol_document = _resolve_protocol_document(protocol_document)
    relevant = [
        Path(__file__).resolve(),
        protocol_document,
        PROJECT_ROOT / "frontier_rl" / "adapters" / "acrobot_neural.py",
        PROJECT_ROOT / "frontier_rl" / "teacher.py",
        PROJECT_ROOT / "frontier_rl" / "estimators.py",
        PROJECT_ROOT / "frontier_rl" / "interfaces.py",
    ]
    if protocol_document == PROTOCOL_V3_PATH:
        relevant.append(V3_VERIFIER_PATH)
    status = _run_git("status", "--porcelain")
    diff = _run_git("diff", "--no-ext-diff", "--binary")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "git_commit": _run_git("rev-parse", "HEAD") or None,
        "git_worktree_dirty": bool(status),
        "git_status_porcelain": status.splitlines(),
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "git_diff_stat": _run_git("diff", "--stat"),
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in relevant
        },
    }


def _artifact_protocol(
    stage: str,
    args,
    seeds: list[int],
    budget: RunBudget,
    protocol_document: Path = PROTOCOL_PATH,
) -> dict:
    protocol_document = _resolve_protocol_document(protocol_document)
    explicitly_exploratory = bool(getattr(args, "exploratory", False))
    exploratory = stage == "pilot" or bool(args.quick) or explicitly_exploratory
    return {
        "stage": stage,
        "status": "exploratory" if exploratory else "confirmatory",
        "exploratory": exploratory,
        "explicit_exploratory": explicitly_exploratory,
        "protocol_document": str(protocol_document.relative_to(PROJECT_ROOT.resolve())),
        "thresholds": list(THRESHOLDS),
        "verifier": "strict post-transition Acrobot tip height > threshold",
        "gymnasium_environment": "Acrobot-v1",
        "max_episode_steps": 500,
        "paired_seeds": seeds,
        "n_rollouts": N_ROLLOUTS,
        "teacher_utility": "1-(1-p)^N-p",
        "teacher_gamma": TEACHER_GAMMA,
        "teacher_decay": TEACHER_DECAY,
        "teacher_floor": TEACHER_FLOOR,
        "optimizer": "plain SGD ascent",
        "score_reduction": "sum over trajectory transitions at frozen group parameters",
        "budget": asdict(budget),
        "eval_interval_transitions": args.eval_interval_transitions,
        "eval_interval_updates": args.eval_interval_updates,
        "eval_n_per_task": args.eval_n,
        "evaluation": "fixed per-seed common random numbers; training RNG restored",
        "primary_core_metric": (
            "normalized trapezoid AUC of target-uniform mean pass rate over actual "
            "environment transitions, including step zero"
        ),
        "complete_groups": (
            "groups are not cut at a budget; final transition coordinate may overshoot"
        ),
        "pilot_is_exploratory": stage == "pilot",
        "quick_is_exploratory": bool(args.quick),
    }


def _write_json_exclusive(path: Path, result: dict, overwrite: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing artifact {path}; pass --overwrite explicitly"
        )
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parse_float_list(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive finite values")
    return values


def _parse_core_architectures(text: str) -> tuple[str, ...]:
    """Parse and validate an explicit ordered subset of core architectures."""
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated non-empty unique architecture subset"
        )
    unknown = set(values) - set(CORE_ARCHITECTURES)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown core architectures: {sorted(unknown)}"
        )
    return values


def _default_output(stage: str, quick: bool) -> Path:
    suffix = "_quick" if quick else ""
    return Path(__file__).resolve().with_name(f"acrobot_neural_{stage}{suffix}.json")


def _stage_defaults(args) -> tuple[list[int], RunBudget]:
    if args.stage == "pilot":
        count = args.seeds if args.seeds is not None else 3
        start = args.seed_start if args.seed_start is not None else 10_000
        transition_budget = (
            args.transition_budget if args.transition_budget is not None else 1_000_000
        )
        budget = RunBudget(transition_budget=transition_budget)
    elif args.stage == "core":
        count = args.seeds if args.seeds is not None else 20
        start = args.seed_start if args.seed_start is not None else 0
        transition_budget = (
            args.transition_budget if args.transition_budget is not None else 2_000_000
        )
        budget = RunBudget(transition_budget=transition_budget)
    else:
        count = args.seeds if args.seeds is not None else 10
        start = args.seed_start if args.seed_start is not None else 100
        update_budget = args.update_budget if args.update_budget is not None else 400
        transition_cap = (
            args.transition_cap if args.transition_cap is not None else 4_000_000
        )
        budget = RunBudget(
            optimizer_update_budget=update_budget,
            transition_safety_cap=transition_cap,
        )
    if args.quick:
        count = min(count, 1)
        if budget.transition_budget is not None:
            budget = replace(budget, transition_budget=min(budget.transition_budget, 40_000))
        else:
            budget = replace(
                budget,
                optimizer_update_budget=min(budget.optimizer_update_budget, 3),
                transition_safety_cap=min(budget.transition_safety_cap, 80_000),
            )
    if not 1 <= count <= 20:
        raise ValueError("seed count must lie in [1, 20] for exact sign-flip inference")
    return list(range(start, start + count)), budget


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Preregistered neural Acrobot Curriculum-MaxRL validation"
    )
    parser.add_argument("stage", choices=("pilot", "core", "scale"))
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument(
        "--pilot-lrs", type=_parse_float_list, default=PILOT_LRS,
        help="comma-separated development rates (pilot stage only)",
    )
    parser.add_argument(
        "--core-architectures",
        type=_parse_core_architectures,
        default=None,
        help=(
            "comma-separated core subset; use 'shared' for the V3 two-cell "
            "efficacy study"
        ),
    )
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--transition-budget", type=int, default=None)
    parser.add_argument("--update-budget", type=int, default=None)
    parser.add_argument("--transition-cap", type=int, default=None)
    parser.add_argument("--eval-interval-transitions", type=int, default=100_000)
    parser.add_argument("--eval-interval-updates", type=int, default=50)
    parser.add_argument("--eval-n", type=int, default=32)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="mark the artifact exploratory without changing seeds or budgets",
    )
    parser.add_argument(
        "--protocol-document",
        type=Path,
        default=PROTOCOL_PATH,
        help="frozen protocol file to hash and identify in the artifact",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if not math.isfinite(args.base_lr) or args.base_lr <= 0.0:
        parser.error("--base-lr must be positive and finite")
    if args.eval_interval_transitions < 1 or args.eval_interval_updates < 1:
        parser.error("evaluation intervals must be positive")
    if args.eval_n < 1:
        parser.error("--eval-n must be positive")
    if args.stage != "core" and args.core_architectures is not None:
        parser.error("--core-architectures is valid only for the core stage")
    try:
        protocol_document = _resolve_protocol_document(args.protocol_document)
        seeds, budget = _stage_defaults(args)
        budget.validate()
    except ValueError as error:
        parser.error(str(error))

    if args.quick:
        args.eval_n = min(args.eval_n, 4)
        args.eval_interval_transitions = min(args.eval_interval_transitions, 20_000)
        args.eval_interval_updates = min(args.eval_interval_updates, 1)

    if args.stage == "pilot":
        conditions = pilot_conditions(args.pilot_lrs)
    elif args.stage == "core":
        selected_core_architectures = (
            CORE_ARCHITECTURES
            if args.core_architectures is None
            else args.core_architectures
        )
        conditions = core_conditions(args.base_lr, selected_core_architectures)
    else:
        conditions = scale_conditions(args.base_lr)

    condition_names = [condition.name for condition in conditions]
    core_analysis_mode = None
    if args.stage == "core":
        selected_set = set(selected_core_architectures)
        if (
            len(selected_core_architectures) == len(CORE_ARCHITECTURES)
            and selected_set == set(CORE_ARCHITECTURES)
        ):
            core_analysis_mode = "six_cell_transfer_and_capacity"
        elif selected_core_architectures == ("shared",):
            core_analysis_mode = "two_cell_shared_efficacy_only"
        else:
            core_analysis_mode = "exploratory_subset_no_registered_contrast"
            if not (args.exploratory or args.quick):
                parser.error(
                    "a partial core subset other than exactly 'shared' must be "
                    "marked --exploratory"
                )
        confirmatory_requested = not (args.exploratory or args.quick)
        if (
            confirmatory_requested
            and core_analysis_mode == "six_cell_transfer_and_capacity"
        ):
            parser.error(
                "the V2 six-cell development gate failed; the full core may only "
                "be rerun with --exploratory"
            )
        if (
            confirmatory_requested
            and core_analysis_mode == "two_cell_shared_efficacy_only"
        ):
            v3_errors = []
            if protocol_document != PROTOCOL_V3_PATH:
                v3_errors.append("--protocol-document must be the frozen V3 protocol")
            if args.base_lr != 3e-4:
                v3_errors.append("--base-lr must equal 3e-4")
            if seeds != list(range(12_000, 12_020)):
                v3_errors.append("paired seeds must be exactly 12000..12019")
            if budget != RunBudget(transition_budget=2_000_000):
                v3_errors.append("transition budget must equal 2,000,000")
            if args.eval_interval_transitions != 100_000:
                v3_errors.append("transition evaluation interval must equal 100,000")
            if args.eval_n != 32:
                v3_errors.append("evaluation episodes per task must equal 32")
            if v3_errors:
                parser.error("invalid V3 confirmation: " + "; ".join(v3_errors))

    output = args.output or _default_output(args.stage, args.quick)
    if output.exists() and not args.overwrite:
        parser.error(
            f"refusing to overwrite {output}; choose --output or pass --overwrite"
        )

    result = {
        "provenance": provenance(protocol_document),
        "protocol": _artifact_protocol(
            args.stage, args, seeds, budget, protocol_document
        ),
        "artifact_state": "in_progress",
        "run_failures": [],
        "cases": {},
    }
    result["protocol"]["condition_names"] = condition_names
    result["protocol"]["condition_count"] = len(condition_names)
    if args.stage == "core":
        result["protocol"].update(
            {
                "core_architectures": list(selected_core_architectures),
                "core_analysis_mode": core_analysis_mode,
                "transfer_claim_evaluated": (
                    core_analysis_mode == "six_cell_transfer_and_capacity"
                ),
            }
        )
    if args.stage == "pilot":
        result["protocol"]["candidate_learning_rates"] = list(args.pilot_lrs)
        result["protocol"]["selection_rule"] = (
            "reject non-finite candidates; maximize pooled final-minus-initial mean "
            "pass over uniform and teacher; within 0.01 choose smaller LR"
        )
    if args.stage == "scale":
        result["protocol"].update(
            {
                "base_learning_rate": args.base_lr,
                "learning_rate_multipliers": list(SCALE_LR_MULTIPLIERS),
                "hindsight_scales": list(HINDSIGHT_SCALES),
                "matching_axis": "nonzero optimizer updates",
                "hindsight_eligibility": "original requested group has K=0 only",
            }
        )

    artifact_claimed = False
    for case_index, condition in enumerate(conditions):
        result["cases"][condition.name] = {
            "config": asdict(condition),
            "summary": summarize_runs([], bootstrap_seed=40_000 + 100 * case_index),
            "runs": [],
        }
        runs = result["cases"][condition.name]["runs"]
        for seed in seeds:
            try:
                run = run_condition(
                    condition,
                    seed,
                    budget=budget,
                    eval_interval_transitions=args.eval_interval_transitions,
                    eval_interval_updates=args.eval_interval_updates,
                    eval_n=args.eval_n,
                )
            except Exception as error:  # retain invalid numerical/implementation runs
                run = {
                    "seed": seed,
                    "numeric_valid": False,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                    "partial_progress_available": False,
                }
                result["run_failures"].append(
                    {
                        "condition": condition.name,
                        **run,
                    }
                )
            runs.append(run)
            result["cases"][condition.name]["summary"] = summarize_runs(
                runs, bootstrap_seed=40_000 + 100 * case_index
            )
            # Atomic incremental checkpoints retain every completed or invalid
            # seed even if a later condition fails or the process is interrupted.
            _write_json_exclusive(
                output,
                result,
                overwrite=args.overwrite or artifact_claimed,
            )
            artifact_claimed = True
        summary = result["cases"][condition.name]["summary"]
        if summary["n_valid"]:
            print(
                f"{condition.name:45s} "
                f"AUC_t={summary['auc_mean_pass_by_transitions']['mean']:.3f} "
                f"final={summary['final_mean_pass']['mean']:.3f} "
                f"updates={summary['optimizer_updates']['mean']:.1f} "
                f"transitions={summary['transitions']['mean']:.0f} "
                f"invalid={summary['n_failed']}",
                flush=True,
            )

    analysis_ready = all(
        [run.get("seed") for run in record["runs"]] == seeds
        and all(run.get("numeric_valid", False) for run in record["runs"])
        for record in result["cases"].values()
    )
    if not analysis_ready:
        result["analysis_status"] = {
            "performed": False,
            "reason": "at least one predeclared run is missing or invalid",
        }
    elif args.stage == "pilot":
        result["pilot_selection"] = select_pilot_learning_rate(result)
        result["analysis_status"] = {"performed": True}
    elif args.stage == "core":
        if core_analysis_mode == "six_cell_transfer_and_capacity":
            attach_core_analysis(result)
            result["analysis_status"] = {
                "performed": True,
                "mode": core_analysis_mode,
            }
        elif core_analysis_mode == "two_cell_shared_efficacy_only":
            attach_shared_core_analysis(result)
            result["analysis_status"] = {
                "performed": True,
                "mode": core_analysis_mode,
            }
        else:
            result["analysis_status"] = {
                "performed": False,
                "mode": core_analysis_mode,
                "reason": "exploratory subset has no registered inferential contrast",
            }
    else:
        attach_scale_analysis(result)
        result["analysis_status"] = {"performed": True}

    result["artifact_state"] = (
        "complete" if analysis_ready else "complete_with_invalid_runs"
    )
    _write_json_exclusive(output, result, overwrite=True)
    print(f"wrote {output.resolve()}")
    if not analysis_ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
