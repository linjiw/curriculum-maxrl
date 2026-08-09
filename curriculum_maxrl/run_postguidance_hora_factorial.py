"""Post-guidance CPU factorial: task sampling x rollout allocation.

This is a deliberately small, synthetic SkillChain experiment motivated by
the HORA paper (Wang et al., 2026, arXiv:2605.07114).  It is *not* a faithful
reproduction of HORA's neural RLVR experiments.  It isolates a narrower
question: when the downstream estimator is practical, dropped-group MaxRL,
does rollout allocation improve when its utility counts the probe rollouts
that are already part of the final group?

Factorial design
----------------
Task sampler: ``uniform`` or ``u_n`` (Thompson priority at average N=16).
Rollout allocation within each eight-prompt batch:

* ``fixed``: four probe rollouts plus twelve more for every prompt;
* ``hora_hit``: the published HORA same-step Beta(1,1) hit-utility marginal,
  E[p(1-p)^ell], for the ell-th additional rollout; and
* ``mass_aware``: the posterior marginal of practical MaxRL coefficient mass,
  E[p(1-p)^(G0+ell)], which counts the G0 probes already spent.

All six cells receive exactly 51,200 sampled completions, use the same
completion-indexed checkpoints, practical group-size-specific MaxRL weights,
and no hindsight relabeling.  Rollouts for a batch are drawn before one
synchronous sum-of-prompt-gradients update.  This simplified batching is why
the artifact is described as HORA-style rather than as a HORA reproduction.

The hypotheses below were written into this source before any outcome from
this new factorial was computed.  They are post-guidance hypotheses, not a
preregistration and not independent confirmation.

Run from the repository root:

    python3 curriculum_maxrl/run_postguidance_hora_factorial.py
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
from typing import Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from curriculum_maxrl.estimators import weights_maxrl
from curriculum_maxrl.teachers import Teacher
from curriculum_maxrl.testbed import SkillChainEnv


SAMPLERS = ("uniform", "u_n")
ALLOCATORS = ("fixed", "hora_hit", "mass_aware")
PRIMARY_METRIC = "normalized_auc_pass_at_8"
METRICS = (
    "coefficient_l1_mass_per_completion",
    "normalized_auc_mean_pass",
    "normalized_auc_pass_at_8",
    "final_mean_pass",
    "final_pass_at_8",
)

# Frozen before execution on 2026-08-07.  These are explicitly post-guidance:
# the HORA paper and the MaxRL mass identity both informed their direction.
FROZEN_HYPOTHESES = {
    "frozen_on": "2026-08-07",
    "status": "post-guidance; frozen in source before first factorial execution",
    "H1_primary_mechanistic": (
        "Averaged over task samplers, mass_aware exceeds hora_hit in realized "
        "practical-MaxRL coefficient L1 mass per sampled completion."
    ),
    "H2_primary_performance": (
        "Averaged over task samplers, mass_aware exceeds hora_hit in normalized "
        "AUC of exact pass@8 versus sampled completions."
    ),
    "H3_sampler_main_effect": (
        "Averaged over rollout allocators, u_n exceeds uniform in normalized "
        "AUC of exact pass@8 versus sampled completions."
    ),
    "H4_exploratory_interaction": (
        "The mass_aware-minus-hora_hit coverage benefit is smaller with u_n "
        "sampling than with uniform sampling because both target related mass."
    ),
    "mean_pass_direction": (
        "No directional hypothesis was frozen for mean-pass AUC or final "
        "mean pass; these are required secondary safety metrics."
    ),
}


@dataclass(frozen=True)
class FactorialConfig:
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


def validate_design(config: FactorialConfig) -> None:
    """Reject any design that could silently violate matched accounting."""
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
    if config.probe_g0 >= config.average_n:
        raise ValueError("probe_g0 must be smaller than average_n")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= config.uniform_floor < 1.0:
        raise ValueError("uniform_floor must lie in [0, 1)")
    if not 0.0 < config.reference_group_decay <= 1.0:
        raise ValueError("reference_group_decay must lie in (0, 1]")
    if config.prior_alpha <= 0.0 or config.prior_beta <= 0.0:
        raise ValueError("Beta-prior parameters must be positive")

    completions_per_step = config.tasks_per_step * config.average_n
    if config.total_completions % completions_per_step:
        raise ValueError("total_completions must contain an integer number of steps")
    if config.checkpoint_every % completions_per_step:
        raise ValueError("checkpoint_every must align to whole batch steps")
    if config.total_completions % config.checkpoint_every:
        raise ValueError("total_completions must align to the checkpoint schedule")


def u_n(p: np.ndarray, n_rollouts: int) -> np.ndarray:
    """Expected practical-MaxRL half coefficient mass at fixed group size N."""
    p = np.asarray(p, dtype=float)
    return np.maximum(1.0 - (1.0 - p) ** n_rollouts - p, 0.0)


class CompletionClockTeacher(Teacher):
    """Thompson task sampler whose evidence decay follows completion count."""

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


def beta_bernoulli_marginal(alpha: float, beta: float, exponent: int) -> float:
    """Return E[P (1-P)^exponent] for P ~ Beta(alpha, beta).

    This equals B(alpha+1, beta+exponent) / B(alpha, beta).  The product form
    is stable for the small integer exponents used by the allocator.
    """
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")
    if int(exponent) != exponent or exponent < 0:
        raise ValueError("exponent must be a non-negative integer")
    value = alpha / (alpha + beta)
    for j in range(int(exponent)):
        value *= (beta + j) / (alpha + beta + j + 1.0)
    return float(value)


def allocation_marginal(
    allocator: str,
    alpha: float,
    beta: float,
    additional_so_far: int,
    probe_g0: int,
) -> float:
    """Posterior gain from one more Phase-B rollout.

    Published HORA's hit utility starts its exponent at zero because its event
    concerns only Phase-B rollouts.  Practical-MaxRL mass depends on the final
    pooled group size, so the mass-aware exponent starts at ``probe_g0``.
    """
    if allocator == "hora_hit":
        exponent = additional_so_far
    elif allocator == "mass_aware":
        exponent = probe_g0 + additional_so_far
    else:
        raise ValueError("allocation marginals exist only for adaptive allocators")
    return beta_bernoulli_marginal(alpha, beta, exponent)


def allocate_group_sizes(
    correct_counts: np.ndarray,
    allocator: str,
    config: FactorialConfig,
) -> np.ndarray:
    """Allocate an exactly matched Phase-B budget across batch positions."""
    counts = np.asarray(correct_counts, dtype=int)
    if counts.shape != (config.tasks_per_step,):
        raise ValueError("correct_counts must have one entry per batch position")
    if (counts < 0).any() or (counts > config.probe_g0).any():
        raise ValueError("probe correct counts fall outside [0, probe_g0]")
    if allocator not in ALLOCATORS:
        raise ValueError(f"unknown allocator {allocator!r}")

    additional = np.zeros(config.tasks_per_step, dtype=int)
    remaining = config.tasks_per_step * (config.average_n - config.probe_g0)
    if allocator == "fixed":
        additional[:] = config.average_n - config.probe_g0
    else:
        alpha = config.prior_alpha + counts
        beta = config.prior_beta + config.probe_g0 - counts
        for _ in range(remaining):
            marginals = np.asarray(
                [
                    allocation_marginal(
                        allocator,
                        float(alpha[i]),
                        float(beta[i]),
                        int(additional[i]),
                        config.probe_g0,
                    )
                    for i in range(config.tasks_per_step)
                ]
            )
            additional[int(np.argmax(marginals))] += 1

    sizes = config.probe_g0 + additional
    if int(sizes.sum()) != config.tasks_per_step * config.average_n:
        raise AssertionError("allocator drifted from the exact batch budget")
    if (sizes < config.probe_g0).any():
        raise AssertionError("allocator removed already-spent probe rollouts")
    return sizes


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
    allocator: str,
    config: FactorialConfig,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, int, int]:
    """Draw probes, allocate, then draw Phase B before any policy update."""
    probes: list[tuple[np.ndarray, np.ndarray]] = []
    for task_id in task_ids:
        probes.append(env.rollout(int(task_id), config.probe_g0))
    counts = np.asarray([int(rewards.sum()) for _, rewards in probes], dtype=int)
    sizes = allocate_group_sizes(counts, allocator, config)

    groups: list[tuple[np.ndarray, np.ndarray]] = []
    phase_b_successes = 0
    for position, task_id in enumerate(task_ids):
        pre_actions, pre_rewards = probes[position]
        n_additional = int(sizes[position] - config.probe_g0)
        if n_additional:
            add_actions, add_rewards = env.rollout(int(task_id), n_additional)
            actions = np.concatenate((pre_actions, add_actions), axis=0)
            rewards = np.concatenate((pre_rewards, add_rewards), axis=0)
            phase_b_successes += int(add_rewards.sum())
        else:
            actions, rewards = pre_actions, pre_rewards
        groups.append((actions, rewards))
    return groups, sizes, int(counts.sum()), phase_b_successes


def apply_synchronous_maxrl_update(
    env: SkillChainEnv,
    task_ids: np.ndarray,
    groups: list[tuple[np.ndarray, np.ndarray]],
    behavior_probs: np.ndarray,
    learning_rate: float,
) -> tuple[float, tuple[int, int, int]]:
    """Apply one summed batch update from group-specific practical MaxRL."""
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


def run_cell(
    sampler: str,
    allocator: str,
    seed: int,
    config: FactorialConfig,
) -> dict:
    """Run one paired factorial cell with exact completion accounting."""
    validate_design(config)
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}")
    if allocator not in ALLOCATORS:
        raise ValueError(f"unknown allocator {allocator!r}")

    env = SkillChainEnv(seed=seed)
    teacher = CompletionClockTeacher(
        env.n_tasks,
        sampler=sampler,
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
    phase_a_successes = 0
    phase_b_successes = 0
    dead_groups = mixed_groups = all_pass_groups = 0

    for step in range(steps):
        task_ids = teacher.sample_tasks(config.tasks_per_step)
        behavior_probs = env.skill_probs()
        groups, sizes, pre_successes, add_successes = collect_batch(
            env, task_ids, allocator, config
        )
        phase_a_successes += pre_successes
        phase_b_successes += add_successes
        group_size_histogram.update(int(size) for size in sizes)

        for task_id_raw, (_, rewards) in zip(task_ids, groups):
            task_id = int(task_id_raw)
            requested_task_counts[task_id] += 1
            teacher.observe(task_id, rewards)

        mass, categories = apply_synchronous_maxrl_update(
            env,
            task_ids,
            groups,
            behavior_probs,
            config.learning_rate,
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
    allocated_completions = sum(size * count for size, count in group_size_histogram.items())
    if allocated_completions != config.total_completions:
        raise AssertionError("group-size accounting does not match completion budget")
    if teacher.observed_completions != config.total_completions:
        raise AssertionError("teacher evidence does not match completion budget")
    if groups_count != dead_groups + mixed_groups + all_pass_groups:
        raise AssertionError("group outcome accounting does not sum")
    expected_checkpoints = list(
        range(0, config.total_completions + 1, config.checkpoint_every)
    )
    if [row["completions"] for row in checkpoints] != expected_checkpoints:
        raise AssertionError("completion-indexed checkpoints drifted")
    if int(requested_task_counts.sum()) != groups_count:
        raise AssertionError("task-request accounting does not sum")

    final = checkpoints[-1]
    return {
        "sampler": sampler,
        "allocator": allocator,
        "seed": seed,
        "steps": steps,
        "groups": groups_count,
        "completions": allocated_completions,
        "probe_completions": groups_count * config.probe_g0,
        "phase_b_completions": allocated_completions - groups_count * config.probe_g0,
        "teacher_observed_completions": teacher.observed_completions,
        "phase_a_successes": phase_a_successes,
        "phase_b_successes": phase_b_successes,
        "sampled_successes": phase_a_successes + phase_b_successes,
        "dead_groups": dead_groups,
        "mixed_groups": mixed_groups,
        "all_pass_groups": all_pass_groups,
        "coefficient_l1_mass": coefficient_l1_mass,
        "coefficient_l1_mass_per_completion": (
            coefficient_l1_mass / allocated_completions
        ),
        "mean_group_size": allocated_completions / groups_count,
        "minimum_group_size": min(group_size_histogram),
        "maximum_group_size": max(group_size_histogram),
        "group_size_histogram": {
            str(size): group_size_histogram[size]
            for size in sorted(group_size_histogram)
        },
        "requested_task_counts": requested_task_counts.tolist(),
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


def _run_cell_tuple(args: tuple[str, str, int, FactorialConfig]) -> dict:
    return run_cell(*args)


def summarize(values: Iterable[float]) -> dict[str, object]:
    x = np.asarray(list(values), dtype=float)
    if len(x) == 0:
        raise ValueError("cannot summarize an empty collection")
    sample_sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return {
        "mean": float(x.mean()),
        "sample_sd": sample_sd,
        "standard_error": float(sample_sd / math.sqrt(len(x))),
        "min": float(x.min()),
        "max": float(x.max()),
        "values_by_seed": x.tolist(),
    }


def exact_two_sided_sign_p(positive: int, negative: int) -> float | None:
    nonzero = positive + negative
    if nonzero == 0:
        return None
    tail = min(positive, negative)
    p_value = 2.0 * sum(math.comb(nonzero, k) for k in range(tail + 1)) / 2**nonzero
    return min(1.0, p_value)


def paired_summary(values: Sequence[float]) -> dict[str, object]:
    row = summarize(values)
    array = np.asarray(values)
    row.update(
        {
            "positive_seeds": int((array > 0.0).sum()),
            "zero_seeds": int((array == 0.0).sum()),
            "negative_seeds": int((array < 0.0).sum()),
        }
    )
    row["two_sided_exact_sign_p"] = exact_two_sided_sign_p(
        row["positive_seeds"], row["negative_seeds"]
    )
    return row


def metric_differences(
    lookup: dict[tuple[str, str, int], dict],
    lhs: tuple[str, str],
    rhs: tuple[str, str],
    seeds: Sequence[int],
    metric: str,
) -> list[float]:
    return [
        lookup[(lhs[0], lhs[1], seed)][metric]
        - lookup[(rhs[0], rhs[1], seed)][metric]
        for seed in seeds
    ]


def build_results(
    runs: list[dict], config: FactorialConfig, seeds: Sequence[int]
) -> dict:
    lookup = {
        (run["sampler"], run["allocator"], run["seed"]): run for run in runs
    }
    expected_keys = {
        (sampler, allocator, seed)
        for sampler in SAMPLERS
        for allocator in ALLOCATORS
        for seed in seeds
    }
    if set(lookup) != expected_keys:
        raise AssertionError("factorial cells are incomplete or duplicated")

    cells: dict[str, object] = {}
    for sampler in SAMPLERS:
        for allocator in ALLOCATORS:
            seed_runs = [lookup[(sampler, allocator, seed)] for seed in seeds]
            cells[f"{sampler}/{allocator}"] = {
                "summary": {
                    metric: summarize(run[metric] for run in seed_runs)
                    for metric in METRICS
                },
                "allocation_summary": {
                    metric: summarize(run[metric] for run in seed_runs)
                    for metric in ("minimum_group_size", "maximum_group_size")
                },
                "seed_runs": seed_runs,
            }

    within_sampler: dict[str, object] = {}
    allocation_pairs = (
        ("mass_aware_minus_hora_hit", "mass_aware", "hora_hit"),
        ("mass_aware_minus_fixed", "mass_aware", "fixed"),
        ("hora_hit_minus_fixed", "hora_hit", "fixed"),
    )
    for sampler in SAMPLERS:
        for name, lhs, rhs in allocation_pairs:
            within_sampler[f"{sampler}/{name}"] = {
                metric: paired_summary(
                    metric_differences(
                        lookup, (sampler, lhs), (sampler, rhs), seeds, metric
                    )
                )
                for metric in METRICS
            }

    within_allocator: dict[str, object] = {}
    for allocator in ALLOCATORS:
        within_allocator[f"{allocator}/u_n_minus_uniform"] = {
            metric: paired_summary(
                metric_differences(
                    lookup,
                    ("u_n", allocator),
                    ("uniform", allocator),
                    seeds,
                    metric,
                )
            )
            for metric in METRICS
        }

    allocation_main_effects: dict[str, object] = {}
    for name, lhs, rhs in allocation_pairs:
        allocation_main_effects[name] = {}
        for metric in METRICS:
            differences = []
            for seed in seeds:
                lhs_mean = np.mean(
                    [lookup[(sampler, lhs, seed)][metric] for sampler in SAMPLERS]
                )
                rhs_mean = np.mean(
                    [lookup[(sampler, rhs, seed)][metric] for sampler in SAMPLERS]
                )
                differences.append(float(lhs_mean - rhs_mean))
            allocation_main_effects[name][metric] = paired_summary(differences)

    sampler_main_effect: dict[str, object] = {}
    for metric in METRICS:
        differences = []
        for seed in seeds:
            u_mean = np.mean(
                [lookup[("u_n", allocator, seed)][metric] for allocator in ALLOCATORS]
            )
            uniform_mean = np.mean(
                [lookup[("uniform", allocator, seed)][metric] for allocator in ALLOCATORS]
            )
            differences.append(float(u_mean - uniform_mean))
        sampler_main_effect[metric] = paired_summary(differences)

    interaction: dict[str, object] = {}
    for metric in METRICS:
        values = []
        for seed in seeds:
            u_benefit = (
                lookup[("u_n", "mass_aware", seed)][metric]
                - lookup[("u_n", "hora_hit", seed)][metric]
            )
            uniform_benefit = (
                lookup[("uniform", "mass_aware", seed)][metric]
                - lookup[("uniform", "hora_hit", seed)][metric]
            )
            values.append(float(u_benefit - uniform_benefit))
        interaction[metric] = paired_summary(values)

    h1 = allocation_main_effects["mass_aware_minus_hora_hit"][
        "coefficient_l1_mass_per_completion"
    ]
    h2 = allocation_main_effects["mass_aware_minus_hora_hit"][PRIMARY_METRIC]
    h3 = sampler_main_effect[PRIMARY_METRIC]
    h4 = interaction[PRIMARY_METRIC]
    hypothesis_readout = {
        "H1_primary_mechanistic_direction_met": h1["mean"] > 0.0,
        "H2_primary_performance_direction_met": h2["mean"] > 0.0,
        "H3_sampler_main_effect_direction_met": h3["mean"] > 0.0,
        "H4_exploratory_smaller_with_u_n_direction_met": h4["mean"] < 0.0,
        "guardrail": (
            "Direction checks are descriptive readouts of post-guidance "
            "hypotheses, not confirmatory decisions."
        ),
    }

    exact_budget = all(run["completions"] == config.total_completions for run in runs)
    exact_mean_n = all(run["mean_group_size"] == config.average_n for run in runs)
    common_checkpoints = len(
        {
            tuple(row["completions"] for row in run["checkpoints"])
            for run in runs
        }
    ) == 1
    fixed_groups = all(
        run["minimum_group_size"] == config.average_n
        and run["maximum_group_size"] == config.average_n
        for run in runs
        if run["allocator"] == "fixed"
    )

    return {
        "schema_version": 1,
        "experiment": "skillchain_postguidance_hora_mass_factorial",
        "analysis_status": (
            "Synthetic post-guidance CPU experiment; hypotheses frozen in source "
            "before execution, but not preregistered or independent confirmation."
        ),
        "reproduction_guardrail": (
            "The hora_hit arm implements the paper's same-step posterior marginal, "
            "but this simplified SkillChain batch simulator is not a published-HORA "
            "reproduction and does not establish neural-RLVR performance."
        ),
        "primary_metric": PRIMARY_METRIC,
        "frozen_hypotheses": FROZEN_HYPOTHESES,
        "hypothesis_readout": hypothesis_readout,
        "protocol": {
            "environment": "SkillChainEnv: 3 nested chains x 12 levels x 10 actions",
            "factorial": {
                "samplers": list(SAMPLERS),
                "allocators": list(ALLOCATORS),
            },
            "estimator": "practical dropped-group MaxRL at each realized group size",
            "hindsight": False,
            "batching": (
                "all probes and Phase-B rollouts use one behavior policy snapshot; "
                "one synchronous sum of eight equal-prompt group gradients follows"
            ),
            "published_hora_source": "https://arxiv.org/abs/2605.07114",
            "published_hora_marginal": "E[p*(1-p)^ell | probe outcomes]",
            "mass_aware_marginal": "E[p*(1-p)^(G0+ell) | probe outcomes]",
            "checkpoint_clock": "sampled completions",
        },
        "config": {**asdict(config), "seeds": list(seeds)},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "git_commit": git_commit(),
            "source_sha256": source_checksums(),
        },
        "checks": {
            "all_cells_exact_completion_budget": exact_budget,
            "all_cells_exact_average_n": exact_mean_n,
            "all_cells_share_completion_checkpoints": common_checkpoints,
            "fixed_allocator_always_n": fixed_groups,
            "all_factorial_cells_present": len(runs)
            == len(SAMPLERS) * len(ALLOCATORS) * len(seeds),
        },
        "cells": cells,
        "contrasts": {
            "within_sampler_allocation": within_sampler,
            "within_allocator_sampler": within_allocator,
            "allocation_main_effects_averaged_over_samplers": allocation_main_effects,
            "sampler_main_effect_averaged_over_allocators": sampler_main_effect,
            "interaction_u_n_minus_uniform_of_mass_minus_hora": interaction,
        },
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_checksums() -> dict[str, str]:
    relative_paths = (
        "curriculum_maxrl/run_postguidance_hora_factorial.py",
        "curriculum_maxrl/estimators.py",
        "curriculum_maxrl/teachers.py",
        "curriculum_maxrl/testbed.py",
    )
    return {
        relative: hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
        for relative in relative_paths
    }


def run_factorial(
    config: FactorialConfig,
    *,
    seeds: Sequence[int] = tuple(range(16)),
    workers: int = 1,
) -> dict:
    validate_design(config)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty sequence of unique integers")
    jobs = [
        (sampler, allocator, int(seed), config)
        for sampler in SAMPLERS
        for allocator in ALLOCATORS
        for seed in seeds
    ]
    if workers == 1:
        runs = [_run_cell_tuple(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            runs = list(pool.map(_run_cell_tuple, jobs))
    return build_results(runs, config, seeds)


def parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def print_summary(results: dict) -> None:
    print("\nPost-guidance SkillChain rollout-allocation factorial")
    print("cell                    mass/completion   mean-pass AUC   pass@8 AUC")
    for name, cell in results["cells"].items():
        summary = cell["summary"]
        print(
            f"{name:24s} "
            f"{summary['coefficient_l1_mass_per_completion']['mean']:15.6f} "
            f"{summary['normalized_auc_mean_pass']['mean']:15.6f} "
            f"{summary['normalized_auc_pass_at_8']['mean']:12.6f}"
        )
    print("\nFrozen-hypothesis directional readout:")
    for name, value in results["hypothesis_readout"].items():
        print(f"  {name}: {value}")
    print("\nAccounting checks:")
    for name, value in results["checks"].items():
        print(f"  {name}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=parse_int_csv, default=tuple(range(16)))
    parser.add_argument("--total-completions", type=int, default=51_200)
    parser.add_argument("--checkpoint-every", type=int, default=2_560)
    parser.add_argument("--tasks-per-step", type=int, default=8)
    parser.add_argument("--average-n", type=int, default=16)
    parser.add_argument("--probe-g0", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--uniform-floor", type=float, default=0.1)
    parser.add_argument("--reference-group-decay", type=float, default=0.9)
    parser.add_argument("--reference-n", type=int, default=16)
    parser.add_argument("--eval-k", type=int, default=8)
    parser.add_argument("--prior-alpha", type=float, default=1.0)
    parser.add_argument("--prior-beta", type=float, default=1.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="parallel CPU processes; results are deterministic across values",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("results_postguidance_hora_factorial.json"),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    config = FactorialConfig(
        total_completions=args.total_completions,
        checkpoint_every=args.checkpoint_every,
        tasks_per_step=args.tasks_per_step,
        average_n=args.average_n,
        probe_g0=args.probe_g0,
        learning_rate=args.learning_rate,
        uniform_floor=args.uniform_floor,
        reference_group_decay=args.reference_group_decay,
        reference_n=args.reference_n,
        eval_k=args.eval_k,
        prior_alpha=args.prior_alpha,
        prior_beta=args.prior_beta,
    )
    results = run_factorial(config, seeds=args.seeds, workers=args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print_summary(results)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
