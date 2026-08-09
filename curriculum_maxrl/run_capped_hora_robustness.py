"""Locked runner for the capped-HORA SkillChain robustness multiverse.

The retained scientific design is frozen in ``CAPPED_HORA_ROBUSTNESS_PROTOCOL.md``.
This module also exposes two non-scientific validation modes:

``audit``
    Run the complete 50-cell matrix on engineering seeds 90 and 91.
``overlap``
    Re-run only the six cells shared with the completed HORA factorial on
    logical seeds 0..15 and require exact equality of its core fields.

The 16-seed, 50-cell ``full`` mode deliberately requires an explicit command
line authorization flag.  Audit and overlap artifacts should be written
outside the repository until the independent review is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from curriculum_maxrl.estimators import weights_maxrl
from curriculum_maxrl.teachers import Teacher
from curriculum_maxrl.testbed import SkillChainEnv


RAW_SCHEMA = "curriculum-maxrl/capped-hora-robustness-raw/v1"
LOCK_SCHEMA = "curriculum-maxrl/capped-hora-robustness-lock/v1"
LOCK_PATH = Path(__file__).with_name("CAPPED_HORA_ROBUSTNESS_LOCK.json")
PROTOCOL_PATH = Path(__file__).with_name("CAPPED_HORA_ROBUSTNESS_PROTOCOL.md")
OLD_RESULT_PATH = Path(__file__).with_name("results_postguidance_hora_factorial.json")

SAMPLERS = ("uniform", "u_16")
ALLOCATORS = ("hora_hit", "fresh_group_mass_proxy")
CAPS: tuple[Optional[int], ...] = (24, 32, 48, None)
INFORMATION_SOURCES = ("same_step", "history_plus_probe", "oracle_preupdate")
SCIENTIFIC_SEEDS = tuple(range(16))
AUDIT_SEEDS = (90, 91)
PINNED_RUNTIME = {
    "python_implementation": "CPython",
    "python": "3.9.6",
    "numpy": "1.26.4",
}

SOURCE_RELATIVE_PATHS = (
    "curriculum_maxrl/CAPPED_HORA_ROBUSTNESS_PROTOCOL.md",
    "curriculum_maxrl/run_capped_hora_robustness.py",
    "curriculum_maxrl/analyze_capped_hora_robustness.py",
    "curriculum_maxrl/test_capped_hora_robustness.py",
    "curriculum_maxrl/estimators.py",
    "curriculum_maxrl/teachers.py",
    "curriculum_maxrl/testbed.py",
)

OVERLAP_FIELDS = (
    "steps",
    "groups",
    "completions",
    "probe_completions",
    "phase_b_completions",
    "teacher_observed_completions",
    "phase_a_successes",
    "phase_b_successes",
    "sampled_successes",
    "dead_groups",
    "mixed_groups",
    "all_pass_groups",
    "coefficient_l1_mass",
    "coefficient_l1_mass_per_completion",
    "mean_group_size",
    "minimum_group_size",
    "maximum_group_size",
    "group_size_histogram",
    "requested_task_counts",
    "normalized_auc_mean_pass",
    "normalized_auc_pass_at_8",
    "final_mean_pass",
    "final_pass_at_8",
    "checkpoints",
)

RNG_AND_TIE_RULES = {
    "environment_seed": "logical_seed",
    "task_teacher_seed": "logical_seed+1000",
    "task_sampling": "with replacement",
    "adaptive_allocation_rng": "none",
    "tie_break": "lowest batch position via NumPy first-index argmax",
    "behavior_snapshot": (
        "policy, exact pass probabilities, and teacher pseudo-count state are "
        "snapshotted before probes"
    ),
    "within_batch_history": (
        "duplicates share the pre-batch task snapshot and add only their own probes"
    ),
    "update_order": (
        "all probes and Phase-B completions use the snapshot; evidence update and "
        "one synchronous policy update follow collection"
    ),
}


@dataclass(frozen=True)
class RobustnessConfig:
    total_completions: int = 51_200
    checkpoint_every: int = 2_560
    tasks_per_step: int = 8
    average_n: int = 16
    probe_g0: int = 4
    learning_rate: float = 0.5
    uniform_floor: float = 0.1
    reference_group_decay: float = 0.9
    reference_n: int = 16
    eval_k: int = 8
    prior_alpha: float = 1.0
    prior_beta: float = 1.0


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    sampler: str
    allocator: str
    information_source: Optional[str]
    cap: Optional[int]

    @property
    def is_fixed(self) -> bool:
        return self.allocator == "fixed"


def _cap_label(cap: Optional[int]) -> str:
    return "uncapped" if cap is None else str(int(cap))


def fixed_cell(sampler: str) -> CellSpec:
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}")
    return CellSpec(f"{sampler}/fixed_n16", sampler, "fixed", None, 16)


def adaptive_cell(
    sampler: str,
    allocator: str,
    cap: Optional[int],
    information_source: str,
) -> CellSpec:
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}")
    if allocator not in ALLOCATORS:
        raise ValueError(f"unknown allocator {allocator!r}")
    if cap not in CAPS:
        raise ValueError(f"unknown cap {cap!r}")
    if information_source not in INFORMATION_SOURCES:
        raise ValueError(f"unknown information source {information_source!r}")
    cell_id = (
        f"{sampler}/{allocator}/{information_source}/cap_{_cap_label(cap)}"
    )
    return CellSpec(cell_id, sampler, allocator, information_source, cap)


def full_cell_specs() -> tuple[CellSpec, ...]:
    cells: list[CellSpec] = []
    for sampler in SAMPLERS:
        cells.append(fixed_cell(sampler))
        for allocator in ALLOCATORS:
            for cap in CAPS:
                for information_source in INFORMATION_SOURCES:
                    cells.append(
                        adaptive_cell(sampler, allocator, cap, information_source)
                    )
    result = tuple(cells)
    if len(result) != 50 or len({cell.cell_id for cell in result}) != 50:
        raise AssertionError("the frozen matrix must contain exactly 50 unique cells")
    return result


def overlap_cell_specs() -> tuple[CellSpec, ...]:
    cells: list[CellSpec] = []
    for sampler in SAMPLERS:
        cells.extend(
            (
                fixed_cell(sampler),
                adaptive_cell(sampler, "hora_hit", None, "same_step"),
                adaptive_cell(
                    sampler, "fresh_group_mass_proxy", None, "same_step"
                ),
            )
        )
    return tuple(cells)


def matrix_manifest() -> dict:
    return {
        "samplers": list(SAMPLERS),
        "fixed_anchor": {"final_group_size": 16, "count": 2},
        "adaptive_allocators": list(ALLOCATORS),
        "caps": [24, 32, 48, "uncapped"],
        "uncapped_mathematical_maximum": 100,
        "information_sources": list(INFORMATION_SOURCES),
        "cell_count": 50,
        "cells": [asdict(cell) for cell in full_cell_specs()],
    }


def validate_design(config: RobustnessConfig) -> None:
    integer_fields = (
        "total_completions",
        "checkpoint_every",
        "tasks_per_step",
        "average_n",
        "probe_g0",
        "reference_n",
        "eval_k",
    )
    for name in integer_fields:
        value = getattr(config, name)
        if int(value) != value or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.tasks_per_step != 8 or config.average_n != 16 or config.probe_g0 != 4:
        raise ValueError("the frozen allocator requires 8 positions, average N=16, G0=4")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= config.uniform_floor < 1.0:
        raise ValueError("uniform_floor must lie in [0,1)")
    if not 0.0 < config.reference_group_decay <= 1.0:
        raise ValueError("reference_group_decay must lie in (0,1]")
    if config.prior_alpha <= 0.0 or config.prior_beta <= 0.0:
        raise ValueError("Beta prior parameters must be positive")
    completions_per_step = config.tasks_per_step * config.average_n
    if config.total_completions % completions_per_step:
        raise ValueError("total_completions must contain whole batches")
    if config.checkpoint_every % completions_per_step:
        raise ValueError("checkpoint_every must align with whole batches")
    if config.total_completions % config.checkpoint_every:
        raise ValueError("total_completions must align with checkpoints")


def u_n(p: np.ndarray, n_rollouts: int) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.maximum(1.0 - (1.0 - p) ** n_rollouts - p, 0.0)


class CompletionClockTeacher(Teacher):
    """Discounted-Beta task sampler with completion-normalized decay."""

    def __init__(
        self,
        n_tasks: int,
        *,
        sampler: str,
        n_rollouts: int,
        seed: int,
        uniform_floor: float,
        reference_group_decay: float,
        reference_n: int,
    ) -> None:
        super().__init__(n_tasks, seed=seed)
        if sampler not in SAMPLERS:
            raise ValueError(f"unknown sampler {sampler!r}")
        self.sampler = sampler
        self.n_rollouts = int(n_rollouts)
        self.uniform_floor = float(uniform_floor)
        self.reference_group_decay = float(reference_group_decay)
        self.reference_n = int(reference_n)
        self.observed_completions = 0

    def distribution(self) -> np.ndarray:
        if self.sampler == "uniform":
            return np.full(self.n_tasks, 1.0 / self.n_tasks)
        draws = np.asarray(
            [self.rng.beta(*stats.alpha_beta) for stats in self.stats], dtype=float
        )
        scores = u_n(draws, self.n_rollouts)
        if scores.sum() <= 1e-12:
            scores = np.ones(self.n_tasks)
        priority = scores / scores.sum()
        uniform = np.full(self.n_tasks, 1.0 / self.n_tasks)
        return (1.0 - self.uniform_floor) * priority + self.uniform_floor * uniform

    def observe(self, task_id: int, rewards: np.ndarray) -> None:
        rewards = np.asarray(rewards, dtype=float)
        if rewards.ndim != 1 or len(rewards) == 0:
            raise ValueError("rewards must be a non-empty vector")
        stats = self.stats[int(task_id)]
        alpha, beta = stats.alpha_beta
        decay = self.reference_group_decay ** (len(rewards) / self.reference_n)
        successes = float(rewards.sum())
        stats.alpha_beta = (
            1.0 + (alpha - 1.0) * decay + successes,
            1.0 + (beta - 1.0) * decay + len(rewards) - successes,
        )
        stats.visits += 1
        self.observed_completions += len(rewards)


def beta_bernoulli_moment(alpha: float, beta: float, exponent: int) -> float:
    """Return the exact product form of B(a+1,b+e)/B(a,b)."""
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")
    if int(exponent) != exponent or exponent < 0:
        raise ValueError("exponent must be a non-negative integer")
    value = alpha / (alpha + beta)
    for j in range(int(exponent)):
        value *= (beta + j) / (alpha + beta + j + 1.0)
    return float(value)


def score_exponent(allocator: str, additional_so_far: int, probe_g0: int) -> int:
    if int(additional_so_far) != additional_so_far or additional_so_far < 0:
        raise ValueError("additional_so_far must be a non-negative integer")
    if allocator == "hora_hit":
        return int(additional_so_far)
    if allocator == "fresh_group_mass_proxy":
        return int(probe_g0 + additional_so_far)
    raise ValueError("adaptive score requested for an unknown allocator")


def information_state(
    information_source: str,
    correct_counts: np.ndarray,
    history_alpha: np.ndarray,
    history_beta: np.ndarray,
    exact_probabilities: np.ndarray,
    config: RobustnessConfig,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    counts = np.asarray(correct_counts, dtype=float)
    history_alpha = np.asarray(history_alpha, dtype=float)
    history_beta = np.asarray(history_beta, dtype=float)
    exact_probabilities = np.asarray(exact_probabilities, dtype=float)
    expected_shape = (config.tasks_per_step,)
    if not all(
        array.shape == expected_shape
        for array in (counts, history_alpha, history_beta, exact_probabilities)
    ):
        raise ValueError("information vectors must have one value per batch position")
    if (counts < 0).any() or (counts > config.probe_g0).any():
        raise ValueError("probe counts lie outside [0,G0]")
    if information_source == "same_step":
        alpha = config.prior_alpha + counts
        beta = config.prior_beta + config.probe_g0 - counts
        prediction = alpha / (alpha + beta)
        return alpha, beta, prediction
    if information_source == "history_plus_probe":
        alpha = history_alpha + counts
        beta = history_beta + config.probe_g0 - counts
        prediction = alpha / (alpha + beta)
        return alpha, beta, prediction
    if information_source == "oracle_preupdate":
        if (exact_probabilities < 0.0).any() or (exact_probabilities > 1.0).any():
            raise ValueError("oracle probabilities must lie in [0,1]")
        return None, None, exact_probabilities.copy()
    raise ValueError(f"unknown information source {information_source!r}")


def allocate_group_sizes(
    correct_counts: np.ndarray,
    *,
    allocator: str,
    information_source: str,
    cap: Optional[int],
    history_alpha: np.ndarray,
    history_beta: np.ndarray,
    exact_probabilities: np.ndarray,
    config: RobustnessConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """Greedily allocate the exact Phase-B budget and return audit sums."""
    if allocator not in ALLOCATORS:
        raise ValueError(f"unknown allocator {allocator!r}")
    if information_source not in INFORMATION_SOURCES:
        raise ValueError(f"unknown information source {information_source!r}")
    if cap not in CAPS:
        raise ValueError(f"unknown cap {cap!r}")
    maximum = 100 if cap is None else int(cap)
    if maximum < config.average_n:
        raise ValueError("cap cannot hold the registered average group size")

    alpha, beta, prediction = information_state(
        information_source,
        correct_counts,
        history_alpha,
        history_beta,
        exact_probabilities,
        config,
    )
    additional = np.zeros(config.tasks_per_step, dtype=int)
    remaining = config.tasks_per_step * (config.average_n - config.probe_g0)
    score_abs_error_sum = 0.0
    score_error_count = 0
    chosen_oracle_regret_sum = 0.0

    for _ in range(remaining):
        group_sizes = config.probe_g0 + additional
        eligible = group_sizes < maximum
        if not bool(np.any(eligible)):
            raise AssertionError("registered cap exhausted before budget allocation")
        deployed = np.full(config.tasks_per_step, -np.inf, dtype=float)
        oracle = np.full(config.tasks_per_step, -np.inf, dtype=float)
        for position in np.flatnonzero(eligible):
            exponent = score_exponent(
                allocator, int(additional[position]), config.probe_g0
            )
            p = float(exact_probabilities[position])
            oracle[position] = p * (1.0 - p) ** exponent
            if information_source == "oracle_preupdate":
                deployed[position] = oracle[position]
            else:
                assert alpha is not None and beta is not None
                deployed[position] = beta_bernoulli_moment(
                    float(alpha[position]), float(beta[position]), exponent
                )
            score_abs_error_sum += abs(deployed[position] - oracle[position])
            score_error_count += 1
        selected = int(np.argmax(deployed))
        best_oracle = float(np.max(oracle[eligible]))
        chosen_oracle_regret_sum += best_oracle - float(oracle[selected])
        additional[selected] += 1

    sizes = config.probe_g0 + additional
    if int(sizes.sum()) != config.tasks_per_step * config.average_n:
        raise AssertionError("allocator drifted from the exact batch budget")
    if (sizes < config.probe_g0).any() or (sizes > maximum).any():
        raise AssertionError("allocator violated a probe minimum or cap")
    probability_error = prediction - np.asarray(exact_probabilities, dtype=float)
    return sizes, {
        "probability_absolute_error_sum": float(np.abs(probability_error).sum()),
        "probability_squared_error_sum": float(np.square(probability_error).sum()),
        "probability_error_count": int(config.tasks_per_step),
        "marginal_score_absolute_error_sum": float(score_abs_error_sum),
        "marginal_score_error_count": int(score_error_count),
        "chosen_oracle_regret_sum": float(chosen_oracle_regret_sum),
        "allocation_decision_count": int(remaining),
    }


def policy_metrics(env: SkillChainEnv, *, eval_k: int) -> dict[str, float]:
    p = env.true_pass_rates()
    return {
        "mean_pass": float(p.mean()),
        "pass_at_8": float((1.0 - (1.0 - p) ** eval_k).mean()),
    }


def normalized_auc(checkpoints: list[dict], metric: str, budget: int) -> float:
    x = np.asarray([row["completions"] for row in checkpoints], dtype=float)
    y = np.asarray([row[metric] for row in checkpoints], dtype=float)
    return float(np.trapz(y, x) / budget)


def collect_batch(
    env: SkillChainEnv,
    task_ids: np.ndarray,
    spec: CellSpec,
    config: RobustnessConfig,
    history_snapshot: np.ndarray,
    exact_task_probabilities: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, int, int, dict]:
    probes: list[tuple[np.ndarray, np.ndarray]] = []
    for task_id in task_ids:
        probes.append(env.rollout(int(task_id), config.probe_g0))
    counts = np.asarray([int(rewards.sum()) for _, rewards in probes], dtype=int)

    if spec.is_fixed:
        sizes = np.full(config.tasks_per_step, config.average_n, dtype=int)
        diagnostics = {
            "probability_absolute_error_sum": 0.0,
            "probability_squared_error_sum": 0.0,
            "probability_error_count": 0,
            "marginal_score_absolute_error_sum": 0.0,
            "marginal_score_error_count": 0,
            "chosen_oracle_regret_sum": 0.0,
            "allocation_decision_count": 0,
        }
    else:
        positions = np.asarray(task_ids, dtype=int)
        history_alpha = history_snapshot[positions, 0]
        history_beta = history_snapshot[positions, 1]
        exact_probabilities = exact_task_probabilities[positions]
        sizes, diagnostics = allocate_group_sizes(
            counts,
            allocator=spec.allocator,
            information_source=str(spec.information_source),
            cap=spec.cap,
            history_alpha=history_alpha,
            history_beta=history_beta,
            exact_probabilities=exact_probabilities,
            config=config,
        )

    groups: list[tuple[np.ndarray, np.ndarray]] = []
    phase_b_successes = 0
    for position, task_id in enumerate(task_ids):
        probe_actions, probe_rewards = probes[position]
        n_additional = int(sizes[position] - config.probe_g0)
        if n_additional:
            add_actions, add_rewards = env.rollout(int(task_id), n_additional)
            actions = np.concatenate((probe_actions, add_actions), axis=0)
            rewards = np.concatenate((probe_rewards, add_rewards), axis=0)
            phase_b_successes += int(add_rewards.sum())
        else:
            actions, rewards = probe_actions, probe_rewards
        groups.append((actions, rewards))
    return groups, sizes, int(counts.sum()), phase_b_successes, diagnostics


def apply_synchronous_maxrl_update(
    env: SkillChainEnv,
    task_ids: np.ndarray,
    groups: list[tuple[np.ndarray, np.ndarray]],
    behavior_probs: np.ndarray,
    learning_rate: float,
) -> tuple[float, tuple[int, int, int]]:
    gradient = np.zeros_like(env.theta)
    coefficient_mass = 0.0
    dead = mixed = all_pass = 0
    for task_id_raw, (actions, rewards) in zip(task_ids, groups):
        task_id = int(task_id_raw)
        weights = weights_maxrl(rewards)
        coefficient_mass += float(np.abs(weights).sum())
        successes = int(rewards.sum())
        if successes == 0:
            dead += 1
        elif successes == len(rewards):
            all_pass += 1
        else:
            mixed += 1
        if not np.any(weights):
            continue
        required = env.tasks[task_id]
        probs = behavior_probs[required]
        onehot = np.zeros((len(rewards), len(required), env.n_actions))
        onehot[
            np.arange(len(rewards))[:, None],
            np.arange(len(required))[None, :],
            actions,
        ] = 1.0
        score = onehot - probs[None]
        group_gradient = np.einsum("j,jla->la", weights, score)
        gradient[required] += group_gradient
    env.theta += learning_rate * gradient
    return coefficient_mass, (dead, mixed, all_pass)


def nearest_rank_p95(histogram: Counter[int]) -> int:
    count = sum(histogram.values())
    if count <= 0:
        raise ValueError("group-size histogram is empty")
    target = int(math.ceil(0.95 * count))
    cumulative = 0
    for size in sorted(histogram):
        cumulative += histogram[size]
        if cumulative >= target:
            return int(size)
    raise AssertionError("nearest-rank scan failed")


def group_size_gini(histogram: Counter[int]) -> float:
    count = int(sum(histogram.values()))
    weighted_sum = int(sum(size * frequency for size, frequency in histogram.items()))
    if count <= 0 or weighted_sum <= 0:
        raise ValueError("group sizes must be positive and non-empty")
    pair_difference_sum = 0
    items = sorted(histogram.items())
    for size_i, frequency_i in items:
        for size_j, frequency_j in items:
            pair_difference_sum += (
                abs(size_i - size_j) * frequency_i * frequency_j
            )
    return float(pair_difference_sum / (2.0 * count * weighted_sum))


def run_cell(spec: CellSpec, seed: int, config: RobustnessConfig) -> dict:
    validate_design(config)
    if spec not in full_cell_specs():
        raise ValueError("cell is not in the frozen 50-cell matrix")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative primitive int")

    env = SkillChainEnv(seed=seed)
    teacher = CompletionClockTeacher(
        env.n_tasks,
        sampler=spec.sampler,
        n_rollouts=config.average_n,
        seed=seed + 1000,
        uniform_floor=config.uniform_floor,
        reference_group_decay=config.reference_group_decay,
        reference_n=config.reference_n,
    )
    completions_per_step = config.tasks_per_step * config.average_n
    steps = config.total_completions // completions_per_step
    checkpoints = [{"completions": 0, **policy_metrics(env, eval_k=config.eval_k)}]
    requested_task_counts = np.zeros(env.n_tasks, dtype=np.int64)
    group_size_histogram: Counter[int] = Counter()
    coefficient_l1_mass = 0.0
    phase_a_successes = phase_b_successes = 0
    dead_groups = mixed_groups = all_pass_groups = 0
    diagnostic_totals = {
        "probability_absolute_error_sum": 0.0,
        "probability_squared_error_sum": 0.0,
        "probability_error_count": 0,
        "marginal_score_absolute_error_sum": 0.0,
        "marginal_score_error_count": 0,
        "chosen_oracle_regret_sum": 0.0,
        "allocation_decision_count": 0,
    }

    for step in range(steps):
        behavior_probs = env.skill_probs()
        exact_task_probabilities = env.true_pass_rates()
        history_snapshot = np.asarray(
            [stats.alpha_beta for stats in teacher.stats], dtype=float
        )
        task_ids = teacher.sample_tasks(config.tasks_per_step)
        groups, sizes, probe_successes, add_successes, diagnostics = collect_batch(
            env,
            task_ids,
            spec,
            config,
            history_snapshot,
            exact_task_probabilities,
        )
        if int(sizes.sum()) != completions_per_step:
            raise AssertionError("batch did not use exactly 128 completions")
        if (sizes < config.probe_g0).any():
            raise AssertionError("batch dropped paid probes")
        if spec.is_fixed and not np.all(sizes == config.average_n):
            raise AssertionError("fixed anchor moved away from N=16")
        if not spec.is_fixed:
            maximum = 100 if spec.cap is None else int(spec.cap)
            if (sizes > maximum).any():
                raise AssertionError("adaptive batch exceeded its cap")
        phase_a_successes += probe_successes
        phase_b_successes += add_successes
        group_size_histogram.update(int(size) for size in sizes)
        for key, value in diagnostics.items():
            diagnostic_totals[key] += value

        for task_id_raw, (_, rewards) in zip(task_ids, groups):
            task_id = int(task_id_raw)
            requested_task_counts[task_id] += 1
            teacher.observe(task_id, rewards)
        mass, categories = apply_synchronous_maxrl_update(
            env, task_ids, groups, behavior_probs, config.learning_rate
        )
        coefficient_l1_mass += mass
        dead_groups += categories[0]
        mixed_groups += categories[1]
        all_pass_groups += categories[2]

        completions = (step + 1) * completions_per_step
        if completions % config.checkpoint_every == 0:
            checkpoints.append(
                {"completions": completions, **policy_metrics(env, eval_k=config.eval_k)}
            )

    groups_count = steps * config.tasks_per_step
    allocated_completions = int(
        sum(size * count for size, count in group_size_histogram.items())
    )
    if allocated_completions != config.total_completions:
        raise AssertionError("group-size accounting differs from paid budget")
    if teacher.observed_completions != config.total_completions:
        raise AssertionError("teacher evidence differs from paid budget")
    if groups_count != dead_groups + mixed_groups + all_pass_groups:
        raise AssertionError("group outcome counts do not sum")
    expected_checkpoints = list(
        range(0, config.total_completions + 1, config.checkpoint_every)
    )
    if [row["completions"] for row in checkpoints] != expected_checkpoints:
        raise AssertionError("checkpoint schedule drifted")
    if int(requested_task_counts.sum()) != groups_count:
        raise AssertionError("task-request counts do not sum")

    probability_count = int(diagnostic_totals["probability_error_count"])
    marginal_count = int(diagnostic_totals["marginal_score_error_count"])
    decision_count = int(diagnostic_totals["allocation_decision_count"])
    if spec.is_fixed:
        if probability_count or marginal_count or decision_count:
            raise AssertionError("fixed anchors must mark allocation diagnostics N/A")
        probability_mae = probability_mse = None
        marginal_score_mae = chosen_oracle_regret_mean = None
    else:
        if probability_count != groups_count:
            raise AssertionError("probability diagnostics must weight positions equally")
        if decision_count != steps * (
            config.tasks_per_step * (config.average_n - config.probe_g0)
        ):
            raise AssertionError("allocation-decision diagnostics are incomplete")
        if marginal_count < decision_count:
            raise AssertionError("eligible-position diagnostics are incomplete")
        probability_mae = (
            diagnostic_totals["probability_absolute_error_sum"] / probability_count
        )
        probability_mse = (
            diagnostic_totals["probability_squared_error_sum"] / probability_count
        )
        marginal_score_mae = (
            diagnostic_totals["marginal_score_absolute_error_sum"] / marginal_count
        )
        chosen_oracle_regret_mean = (
            diagnostic_totals["chosen_oracle_regret_sum"] / decision_count
        )

    final = checkpoints[-1]
    return {
        "cell_id": spec.cell_id,
        "sampler": spec.sampler,
        "allocator": spec.allocator,
        "information_source": spec.information_source,
        "cap": spec.cap,
        "seed": seed,
        "steps": steps,
        "groups": groups_count,
        "completions": allocated_completions,
        "probe_completions": groups_count * config.probe_g0,
        "phase_b_completions": allocated_completions
        - groups_count * config.probe_g0,
        "teacher_observed_completions": teacher.observed_completions,
        "phase_a_successes": phase_a_successes,
        "phase_b_successes": phase_b_successes,
        "sampled_successes": phase_a_successes + phase_b_successes,
        "dead_groups": dead_groups,
        "mixed_groups": mixed_groups,
        "all_pass_groups": all_pass_groups,
        "coefficient_l1_mass": coefficient_l1_mass,
        "coefficient_l1_mass_per_completion": coefficient_l1_mass
        / allocated_completions,
        "mean_group_size": allocated_completions / groups_count,
        "minimum_group_size": min(group_size_histogram),
        "maximum_group_size": max(group_size_histogram),
        "nearest_rank_p95_group_size": nearest_rank_p95(group_size_histogram),
        "group_size_gini": group_size_gini(group_size_histogram),
        "group_size_histogram": {
            str(size): group_size_histogram[size]
            for size in sorted(group_size_histogram)
        },
        "requested_task_counts": requested_task_counts.tolist(),
        "mean_absolute_probability_error": probability_mae,
        "mean_squared_probability_error": probability_mse,
        "probability_error_position_count": probability_count,
        "marginal_score_mae": marginal_score_mae,
        "marginal_score_eligible_position_decision_count": marginal_count,
        "chosen_oracle_regret_mean": chosen_oracle_regret_mean,
        "allocation_decision_count": decision_count,
        "allocation_diagnostics_applicable": not spec.is_fixed,
        "normalized_auc_mean_pass": normalized_auc(
            checkpoints, "mean_pass", config.total_completions
        ),
        "normalized_auc_pass_at_8": normalized_auc(
            checkpoints, "pass_at_8", config.total_completions
        ),
        "final_mean_pass": final["mean_pass"],
        "final_pass_at_8": final["pass_at_8"],
        "checkpoints": checkpoints,
    }


def _run_cell_tuple(args: tuple[CellSpec, int, RobustnessConfig]) -> dict:
    return run_cell(*args)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def source_hashes() -> dict[str, str]:
    missing = [
        relative
        for relative in SOURCE_RELATIVE_PATHS
        if not (PROJECT_ROOT / relative).is_file()
    ]
    if missing:
        raise RuntimeError("missing locked sources: " + ", ".join(missing))
    return {
        relative: _sha256(PROJECT_ROOT / relative)
        for relative in SOURCE_RELATIVE_PATHS
    }


def canonical_json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def expected_lock_static_fields() -> dict:
    return {
        "schema": LOCK_SCHEMA,
        "experiment": "capped_hora_skillchain_robustness",
        "frozen_on": "2026-08-08",
        "status": "exploratory multiverse; not confirmatory",
        "runtime": dict(PINNED_RUNTIME),
        "config": asdict(RobustnessConfig()),
        "matrix": matrix_manifest(),
        "scientific_seeds": list(SCIENTIFIC_SEEDS),
        "engineering_audit_seeds": list(AUDIT_SEEDS),
        "rng_and_tie_rules": dict(RNG_AND_TIE_RULES),
        "overlap": {
            "source_result": "curriculum_maxrl/results_postguidance_hora_factorial.json",
            "source_result_sha256": _sha256(OLD_RESULT_PATH),
            "cells": [cell.cell_id for cell in overlap_cell_specs()],
            "fields": list(OVERLAP_FIELDS),
            "logical_seeds": list(SCIENTIFIC_SEEDS),
            "required_exact_comparisons": 96,
        },
        "claim_boundary": (
            "Synthetic cap/information sensitivity only; does not validate HORA, "
            "establish neural RLVR performance, prove mass mediation, or add a "
            "confirmed paper contribution."
        ),
    }


def load_and_verify_lock(path: Path = LOCK_PATH) -> tuple[dict, str]:
    path = path.resolve()
    if path != LOCK_PATH.resolve():
        raise RuntimeError(f"canonical lock required: {LOCK_PATH.resolve()}")
    if not path.is_file():
        raise RuntimeError(f"canonical lock is missing: {path}")
    raw = path.read_bytes()
    lock = json.loads(raw.decode("utf-8"))
    errors: list[str] = []
    if raw != canonical_json_bytes(lock):
        errors.append("lock JSON is not canonical")
    live_runtime = _runtime()
    if live_runtime != PINNED_RUNTIME:
        errors.append(f"live runtime is not pinned: {live_runtime!r}")
    expected = expected_lock_static_fields()
    for key, value in expected.items():
        if lock.get(key) != value:
            errors.append(f"locked {key} mismatch")
    hashes = source_hashes()
    if set(lock.get("source_sha256", {})) != set(SOURCE_RELATIVE_PATHS):
        errors.append("source hash key set mismatch")
    if lock.get("source_sha256") != hashes:
        errors.append("source hash mismatch")
    if set(lock) != set(expected) | {"source_sha256"}:
        errors.append("lock top-level key set mismatch")
    if errors:
        raise RuntimeError("capped-HORA source/runtime lock failed: " + "; ".join(errors))
    return lock, hashlib.sha256(raw).hexdigest()


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _old_overlap_key(spec: CellSpec) -> tuple[str, str]:
    old_sampler = "u_n" if spec.sampler == "u_16" else "uniform"
    if spec.allocator == "fresh_group_mass_proxy":
        old_allocator = "mass_aware"
    else:
        old_allocator = spec.allocator
    return old_sampler, old_allocator


def verify_overlap_runs(runs: Sequence[dict]) -> dict:
    old = json.loads(OLD_RESULT_PATH.read_text(encoding="utf-8"))
    old_lookup = {}
    for cell in old["cells"].values():
        for run in cell["seed_runs"]:
            old_lookup[(run["sampler"], run["allocator"], run["seed"])] = run
    new_lookup = {(run["cell_id"], run["seed"]): run for run in runs}
    mismatches = []
    comparisons = 0
    for spec in overlap_cell_specs():
        old_sampler, old_allocator = _old_overlap_key(spec)
        for seed in SCIENTIFIC_SEEDS:
            comparisons += 1
            new_run = new_lookup.get((spec.cell_id, seed))
            old_run = old_lookup.get((old_sampler, old_allocator, seed))
            if new_run is None or old_run is None:
                mismatches.append(
                    {"cell_id": spec.cell_id, "seed": seed, "field": "missing_run"}
                )
                continue
            for field in OVERLAP_FIELDS:
                if new_run.get(field) != old_run.get(field):
                    mismatches.append(
                        {
                            "cell_id": spec.cell_id,
                            "seed": seed,
                            "field": field,
                            "new": new_run.get(field),
                            "old": old_run.get(field),
                        }
                    )
    return {
        "passed": comparisons == 96 and not mismatches,
        "source_result": str(OLD_RESULT_PATH.relative_to(PROJECT_ROOT)),
        "source_result_sha256": _sha256(OLD_RESULT_PATH),
        "fields": list(OVERLAP_FIELDS),
        "comparisons": comparisons,
        "expected_comparisons": 96,
        "mismatches": mismatches,
    }


def build_raw_artifact(
    mode: str,
    runs: list[dict],
    specs: Sequence[CellSpec],
    seeds: Sequence[int],
    config: RobustnessConfig,
    lock: dict,
    lock_sha256: str,
) -> dict:
    expected_keys = {
        (spec.cell_id, int(seed)) for spec in specs for seed in seeds
    }
    actual_keys = {(run["cell_id"], run["seed"]) for run in runs}
    if actual_keys != expected_keys or len(runs) != len(expected_keys):
        raise AssertionError("raw run matrix is incomplete or duplicated")
    completion_sequences = {
        tuple(row["completions"] for row in run["checkpoints"]) for run in runs
    }
    checks = {
        "all_requested_cells_present_once": True,
        "all_runs_exact_completion_budget": all(
            run["completions"] == config.total_completions for run in runs
        ),
        "all_runs_exact_average_group_size": all(
            run["mean_group_size"] == config.average_n for run in runs
        ),
        "all_runs_share_completion_checkpoints": len(completion_sequences) == 1,
        "fixed_anchors_always_n16": all(
            run["minimum_group_size"] == 16 and run["maximum_group_size"] == 16
            for run in runs
            if run["allocator"] == "fixed"
        ),
        "all_adaptive_caps_respected": all(
            run["maximum_group_size"]
            <= (100 if run["cap"] is None else run["cap"])
            for run in runs
            if run["allocator"] != "fixed"
        ),
        "all_group_outcomes_sum": all(
            run["dead_groups"] + run["mixed_groups"] + run["all_pass_groups"]
            == run["groups"]
            for run in runs
        ),
        "all_teacher_evidence_charged_once": all(
            run["teacher_observed_completions"] == config.total_completions
            for run in runs
        ),
        "allocation_diagnostic_applicability_exact": all(
            (run["allocator"] == "fixed")
            == (not run["allocation_diagnostics_applicable"])
            for run in runs
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"raw accounting checks failed: {checks!r}")
    artifact = {
        "schema": RAW_SCHEMA,
        "experiment": "capped_hora_skillchain_robustness",
        "mode": mode,
        "claim_status": "exploratory multiverse; not confirmatory",
        "config": {**asdict(config), "seeds": list(seeds)},
        "matrix": {
            "cell_count": len(specs),
            "cells": [asdict(spec) for spec in specs],
        },
        "provenance": {
            "source_lock_enforced": True,
            "source_lock_relative_path": str(LOCK_PATH.relative_to(PROJECT_ROOT)),
            "source_lock_sha256": lock_sha256,
            "source_sha256": lock["source_sha256"],
            "runtime": _runtime(),
            "git_commit": git_commit(),
            "rng_and_tie_rules": dict(RNG_AND_TIE_RULES),
        },
        "checks": checks,
        "runs": runs,
    }
    if mode == "overlap":
        artifact["overlap_verification"] = verify_overlap_runs(runs)
        if not artifact["overlap_verification"]["passed"]:
            raise AssertionError("six-cell overlap verification failed")
    return artifact


def run_study(
    mode: str,
    *,
    config: RobustnessConfig = RobustnessConfig(),
    workers: int = 1,
    lock_path: Path = LOCK_PATH,
) -> dict:
    if workers < 1:
        raise ValueError("workers must be at least one")
    validate_design(config)
    lock, lock_sha256 = load_and_verify_lock(lock_path)
    if mode == "audit":
        specs, seeds = full_cell_specs(), AUDIT_SEEDS
    elif mode == "overlap":
        specs, seeds = overlap_cell_specs(), SCIENTIFIC_SEEDS
    elif mode == "full":
        specs, seeds = full_cell_specs(), SCIENTIFIC_SEEDS
    else:
        raise ValueError(f"unknown mode {mode!r}")
    jobs = [(spec, seed, config) for spec in specs for seed in seeds]
    if workers == 1:
        runs = [_run_cell_tuple(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            runs = list(pool.map(_run_cell_tuple, jobs))
    return build_raw_artifact(mode, runs, specs, seeds, config, lock, lock_sha256)


def print_summary(artifact: dict) -> None:
    print("\nCapped-HORA robustness raw run")
    print(f"mode: {artifact['mode']}")
    print(f"cells: {artifact['matrix']['cell_count']}")
    print(f"seeds: {artifact['config']['seeds']}")
    print(f"runs: {len(artifact['runs'])}")
    for name, value in artifact["checks"].items():
        print(f"  {name}: {value}")
    if artifact["mode"] == "overlap":
        overlap = artifact["overlap_verification"]
        print(
            "overlap: "
            f"{overlap['comparisons']}/{overlap['expected_comparisons']} "
            f"exact run comparisons; passed={overlap['passed']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "overlap", "full"))
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--authorize-retained-full-matrix",
        action="store_true",
        help="required for the 16-seed 50-cell full mode",
    )
    args = parser.parse_args()
    if args.mode == "full" and not args.authorize_retained_full_matrix:
        parser.error("full mode requires --authorize-retained-full-matrix")
    if args.mode != "full" and args.authorize_retained_full_matrix:
        parser.error("the full-matrix authorization flag is valid only in full mode")
    artifact = run_study(args.mode, workers=args.workers, lock_path=args.lock)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical_json_bytes(artifact))
    print_summary(artifact)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
