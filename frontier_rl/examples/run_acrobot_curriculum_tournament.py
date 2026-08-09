"""Raw runner for the source-locked three-arm Acrobot CPU tournament.

This module deliberately delegates actor/environment/update/evaluation mechanics
to :mod:`run_acrobot_neural`.  Its only algorithmic extension is an explicit,
temporary teacher-factory patch adding the p(1-p) sampler and richer raw logs.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import gymnasium
import numpy as np

from frontier_rl.examples import run_acrobot_neural as engine
from frontier_rl.teacher import FrontierTeacher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCHEMA = "curriculum-maxrl/acrobot-curriculum-tournament-raw/v2"
LOCK_SCHEMA = "curriculum-maxrl/acrobot-curriculum-tournament-lock/v2"
DEVELOPMENT_GATE_SCHEMA = (
    "curriculum-maxrl/acrobot-curriculum-tournament-development-gates/v2"
)
LOCK_PATH = HERE / "ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json"
PROTOCOL_PATH = HERE / "ACROBOT_CURRICULUM_TOURNAMENT_PROTOCOL.md"

CONFIRMATORY_SEEDS = tuple(range(20_000, 20_020))
DEVELOPMENT_SEEDS = tuple(range(20_100, 20_103))
QUICK_SEEDS = (20_200,)
TRANSITION_BUDGET = 2_000_000
EVAL_INTERVAL = 100_000
EVAL_N = 32
DEVELOPMENT_TRANSITION_BUDGET = 200_000
DEVELOPMENT_EVAL_INTERVAL = 50_000
DEVELOPMENT_EVAL_N = 16
QUICK_TRANSITION_BUDGET = 8_000
QUICK_EVAL_INTERVAL = 4_000
QUICK_EVAL_N = 2
LEARNING_RATE = 3e-4
N_ROLLOUTS = 16
TEACHER_DECAY = 0.7
TEACHER_FLOOR = 0.1
TEACHER_GAMMA = 1.0
EVALUATION_SEED_BASE = 1_000_000
MAX_COMPLETE_GROUP_TRANSITIONS = 16 * 500
ENGINE_MASTER_BASE = 50_000_000_000
ENGINE_MASTER_STRIDE = 10_000_000
RNG_DOMAIN_OFFSETS = {
    "actor_parameter": 0,
    "actor_action": 1,
    "teacher": 10_000,
    "environment_reset_rng": 11_003,
    "evaluation_episode": 1_000_000,
    "evaluation_action": 1_000_001,
}
ENVIRONMENT_ADAPTER_SEED_OFFSET = 1_000
DEVELOPMENT_GATE_NAMES = (
    "all_runs_accounting_numeric_verifier_parameter_cadence_valid",
    "all_tasks_visited_pooled_across_arms_and_seeds",
    "p1mp_sampler_exhibits_nonuniform_distribution",
    "u16_sampler_exhibits_nonuniform_distribution",
    "pooled_dead_mixed_all_pass_regimes_observed",
    "native_success_checkpoint_values_vary",
)
DEVELOPMENT_GATE_POLICY = {
    "outcome_blind": True,
    "uses_arm_contrasts": False,
    "uses_effect_direction": False,
    "uses_p_values_or_intervals": False,
    "uses_minimum_effect": False,
}
PINNED_RUNTIME_VERSIONS = {
    "python_implementation": "CPython",
    "python": "3.12.13",
    "numpy": "2.5.1",
    "gymnasium": "1.3.0",
}

CONDITIONS = (
    engine.Condition(
        "uniform_shared_h64", "tournament", "uniform", "shared", 64,
        LEARNING_RATE,
    ),
    engine.Condition(
        "p1mp_shared_h64", "tournament", "p1mp", "shared", 64,
        LEARNING_RATE,
    ),
    engine.Condition(
        "u16_shared_h64", "tournament", "u16", "shared", 64,
        LEARNING_RATE,
    ),
)

PRIOR_TRAINING_SEED_BLOCKS = {
    "legacy_acrobot_core": tuple(range(20)),
    "legacy_acrobot_scale": tuple(range(100, 110)),
    "acrobot_v1_pilot": tuple(range(10_000, 10_003)),
    "acrobot_v2_development": tuple(range(11_000, 11_003)),
    "acrobot_v3_confirmation": tuple(range(12_000, 12_020)),
    "acrobot_hindsight_v4a": tuple(range(13_000, 13_003)),
    "acrobot_hindsight_v4b": tuple(range(14_000, 14_010)),
    "acrobot_hindsight_v5a": tuple(range(15_000, 15_003)),
    "acrobot_hindsight_v5b": tuple(range(16_000, 16_020)),
    "mountaincar_v1_development": tuple(range(17_000, 17_003)),
    "mountaincar_v1_reserved_confirmation": tuple(range(18_000, 18_020)),
    "aborted_tournament_confirmation_rng_overlap": tuple(range(19_000, 19_020)),
    "aborted_tournament_development_rng_overlap": tuple(range(19_100, 19_103)),
}

SOURCE_RELATIVE_PATHS = (
    "frontier_rl/examples/run_acrobot_curriculum_tournament.py",
    "frontier_rl/examples/analyze_acrobot_curriculum_tournament.py",
    "frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_PROTOCOL.md",
    (
        "frontier_rl/examples/"
        "ACROBOT_CURRICULUM_TOURNAMENT_PROTOCOL_V1_ABORTED_PRE_OUTCOME_AUDIT.md"
    ),
    (
        "frontier_rl/examples/"
        "ACROBOT_CURRICULUM_TOURNAMENT_LOCK_V1_ABORTED_PRE_OUTCOME_AUDIT.json"
    ),
    "frontier_rl/examples/test_run_acrobot_curriculum_tournament.py",
    "frontier_rl/examples/test_analyze_acrobot_curriculum_tournament.py",
    "frontier_rl/examples/run_acrobot_neural.py",
    "frontier_rl/examples/test_run_acrobot_neural.py",
    "frontier_rl/__init__.py",
    "frontier_rl/adapters/__init__.py",
    "frontier_rl/adapters/acrobot_neural.py",
    "frontier_rl/teacher.py",
    "frontier_rl/estimators.py",
    "frontier_rl/interfaces.py",
    "frontier_rl/trainer.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
    }


def _runtime_versions(runtime: dict[str, str]) -> dict[str, str]:
    return {key: runtime[key] for key in PINNED_RUNTIME_VERSIONS}


def _source_hashes(*, require_all: bool = True) -> dict[str, str]:
    hashes = {}
    missing = []
    for relative in SOURCE_RELATIVE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            missing.append(relative)
        else:
            hashes[relative] = _sha256(path)
    if require_all and missing:
        raise RuntimeError("missing locked tournament sources: " + ", ".join(missing))
    return hashes


def engine_master_seed(logical_seed: int) -> int:
    """Map a human-scale paired seed to its isolated engine RNG namespace."""
    if type(logical_seed) is not int or logical_seed < 0:
        raise ValueError("logical_seed must be a non-negative primitive int")
    return ENGINE_MASTER_BASE + logical_seed * ENGINE_MASTER_STRIDE


def rng_domain_record(logical_seed: int) -> dict:
    master = engine_master_seed(logical_seed)
    return {
        "logical_seed": logical_seed,
        "engine_master_seed": master,
        "environment_adapter_seed_argument": (
            master + ENVIRONMENT_ADAPTER_SEED_OFFSET
        ),
        "rng_roots": {
            domain: master + offset for domain, offset in RNG_DOMAIN_OFFSETS.items()
        },
    }


def seed_collision_audit() -> dict:
    sealed = set(CONFIRMATORY_SEEDS)
    development = set(DEVELOPMENT_SEEDS)
    quick = set(QUICK_SEEDS)
    prior = set().union(*(set(v) for v in PRIOR_TRAINING_SEED_BLOCKS.values()))
    collisions = {
        "confirmatory_vs_prior": sorted(sealed & prior),
        "confirmatory_vs_development": sorted(sealed & development),
        "confirmatory_vs_quick": sorted(sealed & quick),
        "development_vs_prior": sorted(development & prior),
        "development_vs_quick": sorted(development & quick),
        "quick_vs_prior": sorted(quick & prior),
    }
    registered = tuple(CONFIRMATORY_SEEDS + DEVELOPMENT_SEEDS + QUICK_SEEDS)
    records = {str(seed): rng_domain_record(seed) for seed in registered}
    root_owner: dict[int, tuple[int, str]] = {}
    derived_collisions = []
    for seed in registered:
        for domain, root in records[str(seed)]["rng_roots"].items():
            prior_owner = root_owner.get(root)
            if prior_owner is not None:
                derived_collisions.append(
                    {
                        "root": root,
                        "first_logical_seed": prior_owner[0],
                        "first_domain": prior_owner[1],
                        "second_logical_seed": seed,
                        "second_domain": domain,
                    }
                )
            root_owner[root] = (seed, domain)
    all_logical_roots = prior | set(registered)
    derived_vs_logical = sorted(set(root_owner) & all_logical_roots)
    expected_root_count = len(registered) * len(RNG_DOMAIN_OFFSETS)
    passed = (
        not any(collisions.values())
        and not derived_collisions
        and not derived_vs_logical
        and len(root_owner) == expected_root_count
        and len(sealed) == 20
        and len(development) == 3
        and len(quick) == 1
    )
    result = {
        "passed": passed,
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "quick_seeds": list(QUICK_SEEDS),
        "prior_training_seed_blocks": {
            key: list(value) for key, value in PRIOR_TRAINING_SEED_BLOCKS.items()
        },
        "logical_seed_collisions": collisions,
        "derived_root_collisions": derived_collisions,
        "derived_roots_vs_all_logical_training_seeds": derived_vs_logical,
        "scope": (
            "global uniqueness over every registered logical-seed/domain pair; "
            "the three arms intentionally reuse one pair's roots as paired CRNs"
        ),
        "mapping": {
            "engine_master_base": ENGINE_MASTER_BASE,
            "engine_master_stride": ENGINE_MASTER_STRIDE,
            "environment_adapter_seed_offset": ENVIRONMENT_ADAPTER_SEED_OFFSET,
            "rng_domain_offsets": dict(RNG_DOMAIN_OFFSETS),
            "adapter_internal_reset_rng_rule": (
                "engine master + 1000 adapter argument + 10003 internal offset "
                "= engine master + 11003"
            ),
        },
        "per_logical_seed": records,
        "unique_derived_root_count": len(root_owner),
        "expected_unique_derived_root_count": expected_root_count,
    }
    if not passed:
        raise RuntimeError(
            "tournament seed collision audit failed: "
            f"logical={collisions}, derived={derived_collisions}, "
            f"derived_vs_logical={derived_vs_logical}"
        )
    return result


def _locked_schedule() -> dict:
    return {
        "condition_names": [c.name for c in CONDITIONS],
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "quick_seeds": list(QUICK_SEEDS),
        "engine_master_base": ENGINE_MASTER_BASE,
        "engine_master_stride": ENGINE_MASTER_STRIDE,
        "rng_domain_offsets": dict(RNG_DOMAIN_OFFSETS),
        "environment_adapter_seed_offset": ENVIRONMENT_ADAPTER_SEED_OFFSET,
        "confirmatory_engine_master_seeds": {
            str(seed): engine_master_seed(seed) for seed in CONFIRMATORY_SEEDS
        },
        "development_engine_master_seeds": {
            str(seed): engine_master_seed(seed) for seed in DEVELOPMENT_SEEDS
        },
        "quick_engine_master_seeds": {
            str(seed): engine_master_seed(seed) for seed in QUICK_SEEDS
        },
        "transition_budget": TRANSITION_BUDGET,
        "eval_interval_transitions": EVAL_INTERVAL,
        "eval_n_shared_trajectories": EVAL_N,
        "development_transition_budget": DEVELOPMENT_TRANSITION_BUDGET,
        "development_eval_interval_transitions": DEVELOPMENT_EVAL_INTERVAL,
        "development_eval_n_shared_trajectories": DEVELOPMENT_EVAL_N,
        "quick_transition_budget": QUICK_TRANSITION_BUDGET,
        "quick_eval_interval_transitions": QUICK_EVAL_INTERVAL,
        "quick_eval_n_shared_trajectories": QUICK_EVAL_N,
        "n_rollouts": N_ROLLOUTS,
        "learning_rate": LEARNING_RATE,
        "architecture": "shared_h64_task_blind",
        "teacher_decay": TEACHER_DECAY,
        "teacher_floor": TEACHER_FLOOR,
        "teacher_gamma": TEACHER_GAMMA,
        "hindsight_scale": 0.0,
    }


def _load_and_verify_lock(path: Path = LOCK_PATH) -> tuple[dict, str]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"source lock is missing: {path}")
    lock = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if lock.get("schema") != LOCK_SCHEMA:
        errors.append("lock schema mismatch")
    live_runtime = _runtime()
    if _runtime_versions(live_runtime) != PINNED_RUNTIME_VERSIONS:
        errors.append(
            "live runtime is not the preregistered pinned runtime: "
            f"{live_runtime!r}"
        )
    if lock.get("runtime") != live_runtime:
        errors.append(f"runtime mismatch: live={live_runtime!r}")
    if lock.get("schedule") != _locked_schedule():
        errors.append("locked schedule mismatch")
    live_audit = seed_collision_audit()
    if lock.get("seed_collision_audit") != live_audit:
        errors.append("seed collision audit mismatch")
    live_hashes = _source_hashes(require_all=True)
    if set(lock.get("source_sha256", {})) != set(SOURCE_RELATIVE_PATHS):
        errors.append("source lock key set is not exact")
    if lock.get("source_sha256") != live_hashes:
        errors.append("source hash mismatch")
    if errors:
        raise RuntimeError("tournament source/runtime lock failed: " + "; ".join(errors))
    return lock, _sha256(path)


def _require_canonical_lock_path(path: Path) -> None:
    """Reject content-equivalent lock copies for evidence-bearing runs."""
    if path.resolve() != LOCK_PATH.resolve():
        raise RuntimeError(
            "V2 development and confirmation require the canonical source lock: "
            f"{LOCK_PATH.resolve()}"
        )


class _TournamentTeacher(FrontierTeacher):
    """Canonical tracker/distribution with one tournament-local utility switch."""

    def __init__(self, sampling: str, seed: int):
        super().__init__(
            len(engine.THRESHOLDS), N_ROLLOUTS, decay=TEACHER_DECAY,
            floor=TEACHER_FLOOR, gamma=TEACHER_GAMMA, seed=seed,
        )
        if sampling not in {"uniform", "p1mp", "u16"}:
            raise ValueError(f"unknown tournament sampler {sampling!r}")
        self.sampling = sampling
        self.distribution_records: list[dict] = []

    def utility(self, p: np.ndarray) -> np.ndarray:
        if self.sampling == "p1mp":
            return np.maximum(p * (1.0 - p), 0.0)
        return super().utility(p)

    def distribution(self) -> np.ndarray:
        posterior = self.pass_rate_estimates().copy()
        if self.sampling == "uniform":
            probabilities = np.full(self.n_tasks, 1.0 / self.n_tasks)
        else:
            probabilities = super().distribution()
        self.distribution_records.append(
            {
                "posterior_mean_pass_rates_before_group": posterior.tolist(),
                "task_probabilities": probabilities.tolist(),
            }
        )
        return probabilities


_BASE_TEACHER_FACTORY = engine._teacher_for
_ACTIVE_FACTORY_CAPTURE: list[_TournamentTeacher] | None = None


def _tournament_teacher_factory(condition: engine.Condition, seed: int):
    if _ACTIVE_FACTORY_CAPTURE is None:
        raise RuntimeError("tournament teacher factory called outside its patch")
    teacher = _TournamentTeacher(condition.sampling, seed + 10_000)
    _ACTIVE_FACTORY_CAPTURE.append(teacher)
    return teacher


@contextlib.contextmanager
def _patched_teacher_factory() -> Iterator[list[_TournamentTeacher]]:
    """Install and always restore the sequential process-local factory patch."""
    global _ACTIVE_FACTORY_CAPTURE
    if engine._teacher_for is not _BASE_TEACHER_FACTORY or _ACTIVE_FACTORY_CAPTURE is not None:
        raise RuntimeError("Acrobot teacher factory is already patched")
    capture: list[_TournamentTeacher] = []
    _ACTIVE_FACTORY_CAPTURE = capture
    engine._teacher_for = _tournament_teacher_factory
    try:
        yield capture
    finally:
        engine._teacher_for = _BASE_TEACHER_FACTORY
        _ACTIVE_FACTORY_CAPTURE = None


def practical_maxrl_mass(success_count: int, group_size: int = N_ROLLOUTS) -> float:
    if not 0 <= success_count <= group_size or group_size < 1:
        raise ValueError("success_count must lie in [0, group_size]")
    if success_count in (0, group_size):
        return 0.0
    return 2.0 * (group_size - success_count) / group_size


def _augment_run(
    run: dict,
    teacher: _TournamentTeacher,
    *,
    transition_budget: int,
    eval_interval: int,
    eval_n: int,
) -> dict:
    groups = run["group_diagnostics"]
    if len(groups) != len(teacher.distribution_records):
        raise RuntimeError("teacher distribution log is not aligned to raw groups")
    masses = []
    for group, sampler_record in zip(groups, teacher.distribution_records):
        probabilities = sampler_record["task_probabilities"]
        if not math.isclose(
            probabilities[group["task_id"]], group["sampled_task_probability"],
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise RuntimeError("selected-task probability audit failed")
        mass = practical_maxrl_mass(int(group["success_count"]))
        group.update(sampler_record)
        group["realized_practical_maxrl_abs_coefficient_mass"] = mass
        masses.append(mass)

    curves = (
        "x_transitions", "x_optimizer_updates", "pass_rate_curve",
        "mean_pass_curve", "hardest_pass_curve", "native_success_rate_curve",
        "mean_native_return_curve", "mean_censored_time_to_goal_curve",
        "mean_policy_entropy_curve", "evaluation_rng_preserved",
    )
    lengths = {len(run[key]) for key in curves}
    if len(lengths) != 1:
        raise RuntimeError("evaluation curves are not checkpoint-aligned")
    group_at_transition = {
        int(group["transition_end"]): int(group["group"]) for group in groups
    }
    sampled_group_axis = []
    checkpoints = []
    for index in range(lengths.pop()):
        transition_coordinate = int(run["x_transitions"][index])
        sampled_groups = (
            0
            if transition_coordinate == 0
            else group_at_transition.get(transition_coordinate)
        )
        if sampled_groups is None:
            raise RuntimeError("checkpoint does not end on a retained complete group")
        sampled_group_axis.append(sampled_groups)
        checkpoints.append(
            {
                "checkpoint": index,
                "transitions": transition_coordinate,
                "sampled_groups": sampled_groups,
                "optimizer_updates": run["x_optimizer_updates"][index],
                "evaluation_shared_trajectories": eval_n,
                "pass_rates": run["pass_rate_curve"][index],
                "target_uniform_mean_pass_rate": run["mean_pass_curve"][index],
                "hardest_pass_rate": run["hardest_pass_curve"][index],
                "native_success_rate": run["native_success_rate_curve"][index],
                "mean_native_return": run["mean_native_return_curve"][index],
                "mean_censored_time_to_goal": run[
                    "mean_censored_time_to_goal_curve"
                ][index],
                "mean_policy_entropy": run["mean_policy_entropy_curve"][index],
                "training_rng_preserved": run["evaluation_rng_preserved"][index],
                "evaluation_seed": EVALUATION_SEED_BASE + run["seed"],
            }
        )
    if run["x_transitions"][0] != 0 or run["transitions"] < transition_budget:
        raise RuntimeError("run lacks initial or terminal transition checkpoint")
    if run["transitions"] > transition_budget + MAX_COMPLETE_GROUP_TRANSITIONS:
        raise RuntimeError("complete-group transition overshoot is impossible")
    if (
        eval_interval == EVAL_INTERVAL
        and transition_budget == TRANSITION_BUDGET
        and len(checkpoints) != TRANSITION_BUDGET // EVAL_INTERVAL + 1
    ):
        raise RuntimeError("confirmatory evaluation cadence is incomplete")
    mass = np.asarray(masses, dtype=np.float64)
    run["checkpoint_records"] = checkpoints
    run["x_sampled_groups"] = sampled_group_axis
    run["realized_coefficient_mass_total"] = float(mass.sum())
    run["realized_coefficient_mass_per_group"] = float(mass.mean())
    run["realized_coefficient_mass_per_million_transitions"] = float(
        mass.sum() * 1_000_000.0 / run["transitions"]
    )
    run["nonzero_coefficient_mass_group_fraction"] = float(np.count_nonzero(mass) / len(mass))
    run["auc_native_success_by_transitions"] = engine.normalized_trapezoid(
        run["native_success_rate_curve"], run["x_transitions"]
    )
    run["auc_native_return_by_transitions"] = engine.normalized_trapezoid(
        run["mean_native_return_curve"], run["x_transitions"]
    )
    run["auc_mean_pass_by_sampled_groups"] = engine.normalized_trapezoid(
        run["mean_pass_curve"], sampled_group_axis
    )
    run["sampled_groups_per_million_transitions"] = float(
        run["sampled_groups"] * 1_000_000.0 / run["transitions"]
    )
    run["optimizer_updates_per_million_transitions"] = float(
        run["optimizer_updates"] * 1_000_000.0 / run["transitions"]
    )
    return run


def run_one(
    condition: engine.Condition,
    seed: int,
    *,
    mode: str,
    lock_path: Path = LOCK_PATH,
) -> dict:
    if condition not in CONDITIONS:
        raise ValueError("condition is not a registered tournament arm")
    schedules = {
        "confirmatory": (CONFIRMATORY_SEEDS, TRANSITION_BUDGET, EVAL_INTERVAL, EVAL_N),
        "development": (
            DEVELOPMENT_SEEDS, DEVELOPMENT_TRANSITION_BUDGET,
            DEVELOPMENT_EVAL_INTERVAL, DEVELOPMENT_EVAL_N,
        ),
        "quick": (QUICK_SEEDS, QUICK_TRANSITION_BUDGET, QUICK_EVAL_INTERVAL, QUICK_EVAL_N),
    }
    if mode not in schedules:
        raise ValueError(f"unknown tournament mode {mode!r}")
    seeds, budget, interval, eval_n = schedules[mode]
    if type(seed) is not int or seed not in seeds:
        raise RuntimeError(f"seed {seed!r} is not registered for {mode}")
    if mode != "quick":
        _require_canonical_lock_path(lock_path)
        _load_and_verify_lock(lock_path)
    logical_seed = seed
    master_seed = engine_master_seed(logical_seed)
    with _patched_teacher_factory() as capture:
        run = engine.run_condition(
            condition,
            master_seed,
            budget=engine.RunBudget(transition_budget=budget),
            eval_interval_transitions=interval,
            eval_interval_updates=1,
            eval_n=eval_n,
            eval_seed_base=EVALUATION_SEED_BASE,
        )
    if len(capture) != 1:
        raise RuntimeError("exactly one tournament teacher must be constructed")
    run = _augment_run(
        run,
        capture[0],
        transition_budget=budget,
        eval_interval=interval,
        eval_n=eval_n,
    )
    if run.get("seed") != master_seed:
        raise RuntimeError("frozen engine did not retain its master seed")
    domain_record = rng_domain_record(logical_seed)
    run["seed"] = logical_seed
    run["logical_seed"] = logical_seed
    run["engine_master_seed"] = master_seed
    run["environment_adapter_seed_argument"] = domain_record[
        "environment_adapter_seed_argument"
    ]
    run["rng_roots"] = domain_record["rng_roots"]
    return run


def _mode_schedule(mode: str) -> tuple[tuple[int, ...], int, int, int]:
    if mode == "confirmatory":
        return CONFIRMATORY_SEEDS, TRANSITION_BUDGET, EVAL_INTERVAL, EVAL_N
    if mode == "development":
        return (
            DEVELOPMENT_SEEDS, DEVELOPMENT_TRANSITION_BUDGET,
            DEVELOPMENT_EVAL_INTERVAL, DEVELOPMENT_EVAL_N,
        )
    return QUICK_SEEDS, QUICK_TRANSITION_BUDGET, QUICK_EVAL_INTERVAL, QUICK_EVAL_N


def _protocol(mode: str, *, gate_record: dict | None = None) -> dict:
    seeds, budget, interval, eval_n = _mode_schedule(mode)
    return {
        "study": "acrobot_curriculum_tournament",
        "mode": mode,
        "status": "confirmatory" if mode == "confirmatory" else "development_only",
        "protocol_document": str(PROTOCOL_PATH.relative_to(PROJECT_ROOT)),
        "condition_names": [c.name for c in CONDITIONS],
        "paired_seeds": list(seeds),
        "logical_to_engine_master_seed": {
            str(seed): engine_master_seed(seed) for seed in seeds
        },
        "thresholds": list(engine.THRESHOLDS),
        "n_rollouts": N_ROLLOUTS,
        "transition_budget": budget,
        "complete_final_group": True,
        "eval_interval_transitions": interval,
        "eval_n_shared_trajectories": eval_n,
        "evaluation_threshold_scoring": (
            "shared nested trajectories reused across all eight thresholds"
        ),
        "evaluation_seed_base": EVALUATION_SEED_BASE,
        "fixed_evaluation_common_random_numbers": True,
        "rng_domain_contract": {
            "engine_master_base": ENGINE_MASTER_BASE,
            "engine_master_stride": ENGINE_MASTER_STRIDE,
            "environment_adapter_seed_offset": ENVIRONMENT_ADAPTER_SEED_OFFSET,
            "rng_domain_offsets": dict(RNG_DOMAIN_OFFSETS),
            "adapter_internal_environment_reset_rng": (
                "master + 1000 adapter seed argument + 10003 internal offset"
            ),
            "pairing": (
                "all three arms intentionally share roots within logical seed; "
                "roots are globally unique across logical-seed/domain pairs"
            ),
        },
        "architecture": "shared_h64_task_blind",
        "total_parameters": 640,
        "learning_rate": LEARNING_RATE,
        "optimizer": "plain SGD ascent",
        "estimator": "practical dropped-group MaxRL",
        "hindsight_scale": 0.0,
        "teacher": {
            "tracking": "discounted Beta Thompson sampling",
            "decay": TEACHER_DECAY,
            "floor": TEACHER_FLOOR,
            "gamma": TEACHER_GAMMA,
            "utilities": {
                "uniform_shared_h64": "constant target-uniform 1/8",
                "p1mp_shared_h64": "p(1-p)",
                "u16_shared_h64": "1-(1-p)^16-p",
            },
        },
        "primary": "u16 minus p(1-p) target-uniform transition AUC",
        "primary_test": "exact two-sided 2^20 paired sign flip",
        "primary_support": "20,000-resample paired-seed bootstrap interval",
        "primary_sesoi": 0.01,
        "primary_decision": (
            "supported iff mean u16-p1mp AUC >= +0.01 and exact two-sided p <= 0.05"
        ),
        "secondary_uniform_tests": (
            "p(1-p)-uniform and u16-uniform; Holm family"
        ),
        "development_gate": gate_record,
        "raw_only": True,
    }


def _case_summary(runs: list[dict]) -> dict:
    valid = [run for run in runs if run.get("numeric_valid")]
    metrics = (
        "auc_mean_pass_by_transitions", "final_native_success_rate",
        "final_mean_native_return", "optimizer_updates", "transitions",
        "realized_coefficient_mass_per_group",
    )
    return {
        "n_attempted": len(runs),
        "n_valid": len(valid),
        "n_failed": len(runs) - len(valid),
        "means_descriptive_only": {
            metric: (float(np.mean([run[metric] for run in valid])) if valid else None)
            for metric in metrics
        },
    }


def _write_json(path: Path, payload: dict, *, overwrite: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
        check=False,
    ).stdout.strip()


def _provenance(mode: str, lock_path: Path) -> dict:
    if mode == "quick":
        lock_hash = None
        hashes = _source_hashes(require_all=False)
    else:
        lock, lock_hash = _load_and_verify_lock(lock_path)
        hashes = lock["source_sha256"]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime(),
        "source_lock_relative_path": str(lock_path.resolve().relative_to(PROJECT_ROOT.resolve())),
        "source_lock_sha256": lock_hash,
        "source_lock_enforced": mode != "quick",
        "source_sha256": hashes,
        "git_commit": _git("rev-parse", "HEAD") or None,
        "git_status_porcelain": _git("status", "--porcelain").splitlines(),
        "seed_collision_audit": seed_collision_audit(),
    }


def _project_relative_file(path: Path, label: str) -> tuple[Path, str]:
    resolved = path.resolve()
    try:
        relative = str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError as error:
        raise RuntimeError(f"{label} must be inside the project") from error
    if not resolved.is_file():
        raise RuntimeError(f"{label} is missing: {resolved}")
    return resolved, relative


def _load_development_gate(
    path: Path,
    lock: dict,
    lock_hash: str,
) -> dict:
    resolved_gate, relative_gate = _project_relative_file(
        path, "development gate artifact"
    )
    record = json.loads(resolved_gate.read_text(encoding="utf-8"))
    if record.get("schema") != DEVELOPMENT_GATE_SCHEMA:
        raise RuntimeError("development gate artifact schema mismatch")
    if record.get("mode") != "development":
        raise RuntimeError("development gate artifact mode mismatch")
    if record.get("all_gates_passed") is not True:
        raise RuntimeError("development gate artifact did not pass")
    if record.get("source_lock_sha256") != lock_hash:
        raise RuntimeError("development gate artifact used a different source lock")
    if tuple(record.get("gates", {})) != DEVELOPMENT_GATE_NAMES:
        raise RuntimeError("development gate key set/order mismatch")
    if any(value is not True for value in record["gates"].values()):
        raise RuntimeError("development gate contains a failed required check")
    if record.get("gate_policy") != DEVELOPMENT_GATE_POLICY:
        raise RuntimeError("development gate policy mismatch")
    expected_lock_verification = {
        "passed": True,
        "runtime": _runtime(),
        "source_lock_sha256": lock_hash,
        "checked_source_files": sorted(lock["source_sha256"]),
    }
    if record.get("source_lock_verification") != expected_lock_verification:
        raise RuntimeError("development gate source-lock verification mismatch")
    raw_relative = record.get("raw_artifact_relative_path")
    if not isinstance(raw_relative, str):
        raise RuntimeError("development gate lacks a project-relative raw artifact")
    raw_path, observed_raw_relative = _project_relative_file(
        PROJECT_ROOT / raw_relative, "development raw artifact"
    )
    if observed_raw_relative != raw_relative:
        raise RuntimeError("development raw artifact relative path is not canonical")
    raw_hash = _sha256(raw_path)
    if record.get("raw_artifact_sha256") != raw_hash:
        raise RuntimeError("development gate raw artifact hash mismatch")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if (
        raw.get("schema") != SCHEMA
        or raw.get("artifact_state") != "complete"
        or raw.get("run_failures") != []
        or raw.get("protocol", {}).get("mode") != "development"
        or raw.get("protocol", {}).get("paired_seeds")
        != list(DEVELOPMENT_SEEDS)
        or raw.get("provenance", {}).get("source_lock_sha256") != lock_hash
        or raw.get("provenance", {}).get("source_lock_enforced") is not True
        or raw.get("provenance", {}).get("source_lock_relative_path")
        != str(LOCK_PATH.resolve().relative_to(PROJECT_ROOT.resolve()))
        or raw.get("provenance", {}).get("source_sha256")
        != lock.get("source_sha256")
        or raw.get("provenance", {}).get("runtime") != _runtime()
        or raw.get("provenance", {}).get("seed_collision_audit")
        != lock.get("seed_collision_audit")
    ):
        raise RuntimeError("development raw artifact is not the exact locked run")
    from frontier_rl.examples import (  # local import avoids runner/analyzer cycle
        analyze_acrobot_curriculum_tournament as independent_analysis,
    )

    validated = independent_analysis._validate_raw_artifact(raw)
    recomputed = independent_analysis.development_gates(
        validated, expected_lock_verification
    )
    for key in (
        "schema",
        "mode",
        "all_gates_passed",
        "source_lock_sha256",
        "source_lock_verification",
        "gates",
        "diagnostics",
        "gate_policy",
    ):
        if record.get(key) != recomputed.get(key):
            raise RuntimeError(f"development gate does not recompute for {key}")
    return {
        "relative_path": relative_gate,
        "sha256": _sha256(resolved_gate),
        "raw_artifact_relative_path": raw_relative,
        "raw_artifact_sha256": raw_hash,
        "all_gates_passed": True,
    }


def run_tournament(
    *,
    mode: str,
    output: Path,
    lock_path: Path = LOCK_PATH,
    overwrite: bool = False,
    development_gate: Path | None = None,
) -> dict:
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    if mode != "quick":
        _require_canonical_lock_path(lock_path)
    gate_record = None
    if mode == "confirmatory":
        lock, lock_hash = _load_and_verify_lock(lock_path)
        if development_gate is not None:
            gate_record = _load_development_gate(
                development_gate, lock, lock_hash
            )
        else:
            raise RuntimeError(
                "confirmation requires the fresh passing V2 --development-gate"
            )
    elif development_gate is not None:
        raise ValueError("the development gate option is confirmatory-only")

    artifact = {
        "schema": SCHEMA,
        "artifact_state": "in_progress",
        "provenance": _provenance(mode, lock_path),
        "protocol": _protocol(mode, gate_record=gate_record),
        "run_failures": [],
        "cases": {},
    }
    seeds, _, _, _ = _mode_schedule(mode)
    claimed = False
    for condition in CONDITIONS:
        case = {
            "config": asdict(condition),
            "sampler": artifact["protocol"]["teacher"]["utilities"][condition.name],
            "summary": _case_summary([]),
            "runs": [],
        }
        artifact["cases"][condition.name] = case
        for seed in seeds:
            try:
                run = run_one(condition, seed, mode=mode, lock_path=lock_path)
            except Exception as error:  # noqa: BLE001 - retain every invalid run
                run = {
                    **rng_domain_record(seed),
                    "seed": seed,
                    "numeric_valid": False,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                }
                artifact["run_failures"].append(
                    {"condition": condition.name, **run}
                )
            case["runs"].append(run)
            case["summary"] = _case_summary(case["runs"])
            _write_json(output, artifact, overwrite=overwrite or claimed)
            claimed = True
        print(
            f"{condition.name}: valid={case['summary']['n_valid']}/"
            f"{case['summary']['n_attempted']}", flush=True,
        )
    if mode != "quick":
        _load_and_verify_lock(lock_path)
    complete = not artifact["run_failures"] and all(
        [run["seed"] for run in case["runs"]] == list(seeds)
        and all(run.get("numeric_valid") for run in case["runs"])
        for case in artifact["cases"].values()
    )
    artifact["artifact_state"] = "complete" if complete else "complete_with_invalid_runs"
    _write_json(output, artifact, overwrite=True)
    return artifact


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--development", action="store_true")
    modes.add_argument("--quick", action="store_true")
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--development-gate", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    mode = "quick" if args.quick else "development" if args.development else "confirmatory"
    if args.output is None:
        args.output = HERE / f"acrobot_curriculum_tournament_{mode}.json"
    try:
        artifact = run_tournament(
            mode=mode,
            output=args.output,
            lock_path=args.lock,
            overwrite=args.overwrite,
            development_gate=args.development_gate,
        )
    except (ValueError, RuntimeError, FileExistsError) as error:
        parser.error(str(error))
    print(f"wrote {args.output.resolve()}")
    if artifact["artifact_state"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
