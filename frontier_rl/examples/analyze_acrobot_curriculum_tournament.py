"""Independent verifier/analyzer for the Acrobot curriculum tournament.

The raw runner never writes inferential results.  This module first verifies
the source lock and every retained ledger, then emits either outcome-blind
development launch gates or the registered confirmatory analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
from collections.abc import Sequence
from pathlib import Path

import gymnasium
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_LOCK = HERE / "ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json"
RAW_SCHEMA = "curriculum-maxrl/acrobot-curriculum-tournament-raw/v2"
LOCK_SCHEMA = "curriculum-maxrl/acrobot-curriculum-tournament-lock/v2"
GATE_SCHEMA = "curriculum-maxrl/acrobot-curriculum-tournament-development-gates/v2"
REPORT_SCHEMA = "curriculum-maxrl/acrobot-curriculum-tournament-analysis/v2"
EXPECTED_CASES = (
    "uniform_shared_h64",
    "p1mp_shared_h64",
    "u16_shared_h64",
)
EXPECTED_SAMPLING = {
    "uniform_shared_h64": "uniform",
    "p1mp_shared_h64": "p1mp",
    "u16_shared_h64": "u16",
}
THRESHOLDS = (-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.7, 1.0)
EXPECTED_SAMPLER_LABEL = {
    "uniform_shared_h64": "constant target-uniform 1/8",
    "p1mp_shared_h64": "p(1-p)",
    "u16_shared_h64": "1-(1-p)^16-p",
}
SCHEDULES = {
    "confirmatory": (tuple(range(20_000, 20_020)), 2_000_000, 100_000, 32),
    "development": (tuple(range(20_100, 20_103)), 200_000, 50_000, 16),
    "quick": ((20_200,), 8_000, 4_000, 2),
}
N_ROLLOUTS = 16
TEACHER_DECAY = 0.7
TEACHER_FLOOR = 0.1
MAX_COMPLETE_GROUP_TRANSITIONS = 8_000
EVALUATION_SEED_BASE = 1_000_000
ENGINE_MASTER_BASE = 50_000_000_000
ENGINE_MASTER_STRIDE = 10_000_000
ENVIRONMENT_ADAPTER_SEED_OFFSET = 1_000
RNG_DOMAIN_OFFSETS = {
    "actor_parameter": 0,
    "actor_action": 1,
    "teacher": 10_000,
    "environment_reset_rng": 11_003,
    "evaluation_episode": 1_000_000,
    "evaluation_action": 1_000_001,
}
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
EXPECTED_SOURCE_RELATIVE_PATHS = (
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
PRIMARY_SESOI = 0.01
PRIMARY_BOOTSTRAP_SEED = 20_016_001
N_BOOTSTRAP = 20_000


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


def _engine_master_seed(logical_seed: int) -> int:
    if type(logical_seed) is not int or logical_seed < 0:
        raise ValueError("logical seed must be a non-negative primitive int")
    return ENGINE_MASTER_BASE + logical_seed * ENGINE_MASTER_STRIDE


def _rng_domain_record(logical_seed: int) -> dict:
    master = _engine_master_seed(logical_seed)
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


def _independent_seed_collision_audit() -> dict:
    confirmatory, development, quick = (
        SCHEDULES["confirmatory"][0],
        SCHEDULES["development"][0],
        SCHEDULES["quick"][0],
    )
    sealed = set(confirmatory)
    development_set = set(development)
    quick_set = set(quick)
    prior = set().union(*(set(v) for v in PRIOR_TRAINING_SEED_BLOCKS.values()))
    collisions = {
        "confirmatory_vs_prior": sorted(sealed & prior),
        "confirmatory_vs_development": sorted(sealed & development_set),
        "confirmatory_vs_quick": sorted(sealed & quick_set),
        "development_vs_prior": sorted(development_set & prior),
        "development_vs_quick": sorted(development_set & quick_set),
        "quick_vs_prior": sorted(quick_set & prior),
    }
    registered = tuple(confirmatory + development + quick)
    records = {str(seed): _rng_domain_record(seed) for seed in registered}
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
        and len(development_set) == 3
        and len(quick_set) == 1
    )
    return {
        "passed": passed,
        "confirmatory_seeds": list(confirmatory),
        "development_seeds": list(development),
        "quick_seeds": list(quick),
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


def _independent_locked_schedule() -> dict:
    return {
        "condition_names": list(EXPECTED_CASES),
        "confirmatory_seeds": list(SCHEDULES["confirmatory"][0]),
        "development_seeds": list(SCHEDULES["development"][0]),
        "quick_seeds": list(SCHEDULES["quick"][0]),
        "engine_master_base": ENGINE_MASTER_BASE,
        "engine_master_stride": ENGINE_MASTER_STRIDE,
        "rng_domain_offsets": dict(RNG_DOMAIN_OFFSETS),
        "environment_adapter_seed_offset": ENVIRONMENT_ADAPTER_SEED_OFFSET,
        "confirmatory_engine_master_seeds": {
            str(seed): _engine_master_seed(seed)
            for seed in SCHEDULES["confirmatory"][0]
        },
        "development_engine_master_seeds": {
            str(seed): _engine_master_seed(seed)
            for seed in SCHEDULES["development"][0]
        },
        "quick_engine_master_seeds": {
            str(seed): _engine_master_seed(seed) for seed in SCHEDULES["quick"][0]
        },
        "transition_budget": SCHEDULES["confirmatory"][1],
        "eval_interval_transitions": SCHEDULES["confirmatory"][2],
        "eval_n_shared_trajectories": SCHEDULES["confirmatory"][3],
        "development_transition_budget": SCHEDULES["development"][1],
        "development_eval_interval_transitions": SCHEDULES["development"][2],
        "development_eval_n_shared_trajectories": SCHEDULES["development"][3],
        "quick_transition_budget": SCHEDULES["quick"][1],
        "quick_eval_interval_transitions": SCHEDULES["quick"][2],
        "quick_eval_n_shared_trajectories": SCHEDULES["quick"][3],
        "n_rollouts": N_ROLLOUTS,
        "learning_rate": 3e-4,
        "architecture": "shared_h64_task_blind",
        "teacher_decay": TEACHER_DECAY,
        "teacher_floor": TEACHER_FLOOR,
        "teacher_gamma": 1.0,
        "hindsight_scale": 0.0,
    }


def normalized_trapezoid(values: Sequence[float], coordinates: Sequence[int]) -> float:
    y = np.asarray(values, dtype=np.float64)
    x = np.asarray(coordinates, dtype=np.float64)
    if (
        x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2
        or x[0] != 0.0 or np.any(np.diff(x) < 0.0)
        or not np.isfinite(x).all() or not np.isfinite(y).all()
    ):
        raise ValueError("invalid curve for normalized transition AUC")
    if x[-1] == 0.0:
        return float(y[-1])
    return float(np.trapezoid(y, x) / x[-1])


def practical_maxrl_mass(success_count: int, group_size: int = N_ROLLOUTS) -> float:
    if not 0 <= success_count <= group_size or group_size < 1:
        raise ValueError("success_count must lie in [0, group_size]")
    if success_count in (0, group_size):
        return 0.0
    return 2.0 * (group_size - success_count) / group_size


def exact_two_sided_sign_flip_p(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not 1 <= len(values) <= 20 or not np.isfinite(values).all():
        raise ValueError("exact sign flip requires 1..20 finite paired values")
    signed_sums = np.zeros(1, dtype=np.float64)
    for value in values:
        signed_sums = np.concatenate((signed_sums - value, signed_sums + value))
    observed = abs(float(values.mean()))
    return float(np.mean(np.abs(signed_sums / len(values)) >= observed - 1e-15))


def paired_bootstrap_ci(
    values: Sequence[float], *, seed: int, n_boot: int = N_BOOTSTRAP
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("bootstrap requires finite paired values")
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    draws = values[
        rng.integers(0, len(values), size=(n_boot, len(values)))
    ].mean(axis=1)
    return [float(value) for value in np.quantile(draws, (0.025, 0.975))]


def holm_adjust(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    ordered = sorted((float(value), name) for name, value in p_values.items())
    running_adjusted = 0.0
    rejection_open = True
    result = {}
    for rank, (p_value, name) in enumerate(ordered, start=1):
        multiplier = len(ordered) - rank + 1
        running_adjusted = max(running_adjusted, multiplier * p_value)
        reject = rejection_open and p_value <= alpha / multiplier
        if not reject:
            rejection_open = False
        result[name] = {
            "raw_p": p_value,
            "holm_adjusted_p": float(min(1.0, running_adjusted)),
            "reject_familywise_0.05": bool(reject),
        }
    return result


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _require_int(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be a primitive integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} is below its allowed minimum")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} is above its allowed maximum")
    return value


def _require_int_vector(value: object, label: str, length: int = 8) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise ValueError(f"{label} must be a length-{length} nonnegative int vector")
    return np.asarray(value, dtype=np.int64)


def _verify_lock(artifact: dict, lock: dict, lock_path: Path) -> dict:
    if lock.get("schema") != LOCK_SCHEMA:
        raise ValueError("source-lock schema mismatch")
    runtime = _runtime()
    if _runtime_versions(runtime) != PINNED_RUNTIME_VERSIONS:
        raise ValueError(f"analysis runtime is not the pinned runtime: {runtime!r}")
    if lock.get("runtime") != runtime:
        raise ValueError(f"analysis runtime differs from source lock: {runtime!r}")
    if lock.get("schedule") != _independent_locked_schedule():
        raise ValueError("source-lock schedule does not independently reproduce")
    provenance = artifact.get("provenance", {})
    if provenance.get("runtime") != runtime:
        raise ValueError("artifact runtime differs from source lock")
    lock_hash = _sha256(lock_path)
    if provenance.get("source_lock_sha256") != lock_hash:
        raise ValueError("artifact was created under a different source lock")
    try:
        lock_relative = str(
            lock_path.resolve().relative_to(PROJECT_ROOT.resolve())
        )
    except ValueError as error:
        raise ValueError("source lock must be inside the project") from error
    if (
        provenance.get("source_lock_enforced") is not True
        or provenance.get("source_lock_relative_path") != lock_relative
    ):
        raise ValueError("artifact source-lock path/enforcement provenance mismatch")
    locked_hashes = lock.get("source_sha256")
    if (
        not isinstance(locked_hashes, dict)
        or set(locked_hashes) != set(EXPECTED_SOURCE_RELATIVE_PATHS)
    ):
        raise ValueError("source lock does not have the exact frozen source manifest")
    if provenance.get("source_sha256") != locked_hashes:
        raise ValueError("artifact source manifest differs from source lock")
    for relative, expected in locked_hashes.items():
        path = (PROJECT_ROOT / relative).resolve()
        try:
            path.relative_to(PROJECT_ROOT.resolve())
        except ValueError as error:
            raise ValueError(f"locked path escapes project: {relative}") from error
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"live source hash mismatch for {relative}")
    audit = _independent_seed_collision_audit()
    if audit.get("passed") is not True:
        raise ValueError("independent seed/RNG-root collision audit did not pass")
    if lock.get("seed_collision_audit") != audit:
        raise ValueError("source-lock seed/RNG-root audit does not reproduce")
    if provenance.get("seed_collision_audit") != audit:
        raise ValueError("artifact seed/RNG-root audit does not reproduce")
    return {
        "passed": True,
        "runtime": runtime,
        "source_lock_sha256": lock_hash,
        "checked_source_files": sorted(locked_hashes),
    }


def _project_relative_file(relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} must be a nonempty project-relative path")
    resolved = (PROJECT_ROOT / relative).resolve()
    try:
        observed = str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError as error:
        raise ValueError(f"{label} escapes the project") from error
    if observed != relative or not resolved.is_file():
        raise ValueError(f"{label} is noncanonical or missing")
    return resolved


def _run_metrics(
    run: dict,
    *,
    budget: int,
    interval: int,
    eval_n: int,
    mode: str,
    sampling: str,
) -> dict[str, object]:
    if not all(
        run.get(key) is True
        for key in (
            "numeric_valid",
            "accounting_valid",
            "verifier_relabel_checks_valid",
            "evaluation_cadence_invariant",
        )
    ):
        raise ValueError(f"invalid run flags for seed {run.get('seed')}")
    logical_seed = run.get("seed")
    expected_domains = _rng_domain_record(logical_seed)
    if (
        run.get("logical_seed") != logical_seed
        or run.get("engine_master_seed")
        != expected_domains["engine_master_seed"]
        or run.get("environment_adapter_seed_argument")
        != expected_domains["environment_adapter_seed_argument"]
        or run.get("rng_roots") != expected_domains["rng_roots"]
    ):
        raise ValueError("run RNG-domain roots do not independently reproduce")
    if run.get("total_parameters") != 640 or run.get(
        "active_parameters_per_task"
    ) != 640:
        raise ValueError("shared-H64 parameter contract failed")
    if run.get("relabeled_groups") != 0 or run.get("relabel_candidates") != 0:
        raise ValueError("tournament run contains hindsight activity")
    transitions = _require_int(run.get("transitions"), "terminal transitions", minimum=1)
    if not budget <= transitions <= budget + MAX_COMPLETE_GROUP_TRANSITIONS:
        raise ValueError("terminal coordinate violates complete-group budget")

    groups = run["group_diagnostics"]
    sampled_groups = _require_int(
        run.get("sampled_groups"), "sampled groups", minimum=1
    )
    optimizer_updates = _require_int(
        run.get("optimizer_updates"), "optimizer updates", minimum=0
    )
    live_groups = _require_int(run.get("live_groups"), "live groups", minimum=0)
    dead_groups = _require_int(run.get("dead_groups"), "dead groups", minimum=0)
    all_pass_groups = _require_int(
        run.get("all_pass_groups"), "all-pass groups", minimum=0
    )
    if not isinstance(groups, list) or len(groups) != sampled_groups or not groups:
        raise ValueError("raw group count mismatch")
    if (
        _require_int(run.get("rollout_attempts"), "rollout attempts", minimum=0)
        != sampled_groups * N_ROLLOUTS
        or dead_groups + live_groups + all_pass_groups != sampled_groups
        or run.get("reached_optimizer_update_budget") is not True
        or run.get("transition_cap_censored") is not False
    ):
        raise ValueError("run-level rollout/group budget accounting mismatch")
    previous_end = 0
    previous_updates = 0
    next_boundary = interval
    expected_x = [0]
    expected_sampled_group_axis = [0]
    expected_update_axis = [0]
    masses = []
    recomputed_tvs = []
    task_groups = np.zeros(8, dtype=np.int64)
    task_rollouts = np.zeros(8, dtype=np.int64)
    task_successes = np.zeros(8, dtype=np.int64)
    task_transitions = np.zeros(8, dtype=np.int64)
    regimes = []
    expected_update_identities = []
    expected_zero_gradient_identities = []
    teacher_rng = np.random.default_rng(
        expected_domains["rng_roots"]["teacher"]
    )
    teacher_alpha = np.ones(8, dtype=np.float64)
    teacher_beta = np.ones(8, dtype=np.float64)
    for index, group in enumerate(groups, start=1):
        group_number = _require_int(group.get("group"), "group number", minimum=1)
        transition_start = _require_int(
            group.get("transition_start"), "group transition start", minimum=0
        )
        transition_end = _require_int(
            group.get("transition_end"), "group transition end", minimum=1
        )
        group_transitions = _require_int(
            group.get("n_transitions"),
            "group transition count",
            minimum=N_ROLLOUTS,
            maximum=MAX_COMPLETE_GROUP_TRANSITIONS,
        )
        if group_number != index or transition_start != previous_end:
            raise ValueError("group ledger is not ordered and contiguous")
        if transition_end - previous_end != group_transitions:
            raise ValueError("group transition ledger is inconsistent")
        previous_end = transition_end
        updates_before = previous_updates
        updates_after = _require_int(
            group.get("optimizer_updates_after_group"),
            "optimizer updates after group",
            minimum=0,
            maximum=index,
        )
        if updates_after < updates_before or updates_after - updates_before > 1:
            raise ValueError("group optimizer-update ledger is invalid")
        previous_updates = updates_after
        task = _require_int(group.get("task_id"), "group task", minimum=0, maximum=7)
        count = _require_int(
            group.get("success_count"),
            "group success count",
            minimum=0,
            maximum=N_ROLLOUTS,
        )
        task_groups[task] += 1
        task_rollouts[task] += N_ROLLOUTS
        task_successes[task] += count
        task_transitions[task] += group_transitions
        probabilities = np.asarray(group["task_probabilities"], dtype=np.float64)
        posterior = np.asarray(
            group["posterior_mean_pass_rates_before_group"], dtype=np.float64
        )
        if (
            probabilities.shape != (8,)
            or posterior.shape != (8,)
            or not np.isfinite(probabilities).all()
            or not np.isfinite(posterior).all()
            or np.any((probabilities < 0.0) | (probabilities > 1.0))
            or np.any((posterior < 0.0) | (posterior > 1.0))
            or not math.isclose(
                float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise ValueError("saved sampler state is invalid")
        expected_posterior = teacher_alpha / (teacher_alpha + teacher_beta)
        if not np.allclose(
            posterior, expected_posterior, rtol=0.0, atol=1e-12
        ):
            raise ValueError("teacher posterior means do not replay")
        if sampling == "uniform":
            expected_probabilities = np.full(8, 1.0 / 8.0)
        else:
            thompson_draw = teacher_rng.beta(teacher_alpha, teacher_beta)
            if sampling == "p1mp":
                utility = np.maximum(
                    thompson_draw * (1.0 - thompson_draw), 0.0
                )
            elif sampling == "u16":
                utility = np.maximum(
                    1.0 - (1.0 - thompson_draw) ** N_ROLLOUTS
                    - thompson_draw,
                    0.0,
                )
            else:
                raise ValueError(f"unknown sampler for teacher replay: {sampling}")
            if float(utility.sum()) <= 1e-12:
                utility = np.ones(8, dtype=np.float64)
            expected_probabilities = (
                (1.0 - TEACHER_FLOOR) * utility / utility.sum()
                + TEACHER_FLOOR / 8.0
            )
        if not np.allclose(
            probabilities, expected_probabilities, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"{sampling} task probabilities do not replay from teacher RNG"
            )
        replayed_task = int(teacher_rng.choice(8, p=expected_probabilities))
        if replayed_task != task:
            raise ValueError("sampled task does not replay from teacher RNG")
        _assert_close(
            probabilities[task],
            group["sampled_task_probability"],
            "selected task probability",
        )
        recomputed_tv = float(
            0.5 * np.abs(probabilities - 1.0 / len(probabilities)).sum()
        )
        _assert_close(
            recomputed_tv,
            group["teacher_tv_from_uniform"],
            "teacher TV from saved probability vector",
        )
        recomputed_tvs.append(recomputed_tv)
        regime = (
            "dead"
            if count == 0
            else "all_pass"
            if count == N_ROLLOUTS
            else "mixed"
        )
        if group["regime"] != regime:
            raise ValueError("group regime disagrees with success count")
        update_source = group.get("update_source")
        update_applied = updates_after == updates_before + 1
        if update_applied:
            if regime != "mixed" or update_source != "requested_live":
                raise ValueError("applied update source/regime is invalid")
            expected_update_identities.append(
                {
                    "optimizer_update": updates_after,
                    "after_group": index,
                    "transitions": transition_end,
                    "source": "requested_live",
                    "requested_task": task,
                    "credited_task": task,
                }
            )
        elif update_source is not None:
            raise ValueError("non-applied group has an update source")
        elif regime == "mixed":
            expected_zero_gradient_identities.append(
                {
                    "after_group": index,
                    "transitions": transition_end,
                    "source": "requested_live",
                    "requested_task": task,
                    "credited_task": task,
                }
            )
        regimes.append(regime)
        mass = practical_maxrl_mass(count)
        _assert_close(
            group["realized_practical_maxrl_abs_coefficient_mass"],
            mass,
            "saved realized coefficient mass",
        )
        masses.append(mass)
        teacher_alpha[task] = (
            1.0 + (teacher_alpha[task] - 1.0) * TEACHER_DECAY + count
        )
        teacher_beta[task] = (
            1.0
            + (teacher_beta[task] - 1.0) * TEACHER_DECAY
            + (N_ROLLOUTS - count)
        )

        due = False
        if transition_end >= next_boundary:
            due = True
            while next_boundary <= transition_end:
                next_boundary += interval
        if transition_end >= budget:
            due = True
            if index != len(groups):
                raise ValueError("training continued after the complete budget group")
        if due:
            expected_x.append(transition_end)
            expected_sampled_group_axis.append(index)
            expected_update_axis.append(updates_after)

    if previous_end != transitions:
        raise ValueError("group ledger does not reach terminal transitions")
    if previous_updates != optimizer_updates:
        raise ValueError("group ledger does not reach terminal optimizer updates")
    accounting_vectors = {
        "task_groups": task_groups,
        "task_rollouts": task_rollouts,
        "task_successes": task_successes,
        "task_transitions": task_transitions,
    }
    for key, recomputed in accounting_vectors.items():
        saved = _require_int_vector(run.get(key), key)
        if not np.array_equal(recomputed, saved):
            raise ValueError(f"group ledger does not reproduce {key}")
    if regimes.count("dead") != dead_groups:
        raise ValueError("dead-group count mismatch")
    if regimes.count("mixed") != live_groups:
        raise ValueError("mixed/live-group count mismatch")
    if regimes.count("all_pass") != all_pass_groups:
        raise ValueError("all-pass-group count mismatch")
    mass = np.asarray(masses, dtype=np.float64)
    if int(np.count_nonzero(mass)) != live_groups:
        raise ValueError("nonzero mass groups do not reproduce live groups")
    if (
        _require_int(
            run.get("live_applied_updates"),
            "live applied updates",
            minimum=0,
        )
        != optimizer_updates
        or _require_int(
            run.get("zero_gradient_update_attempts"),
            "zero-gradient update attempts",
            minimum=0,
        )
        != len(expected_zero_gradient_identities)
        or _require_int(
            run.get("unscaled_aux_gradient_previews"),
            "unscaled auxiliary gradient previews",
            minimum=0,
        )
        != 0
        or run.get("auxiliary_gradient_diagnostics") != []
    ):
        raise ValueError("run-level optimizer/hindsight accounting mismatch")

    update_records = run.get("update_diagnostics")
    zero_gradient_records = run.get("zero_gradient_diagnostics")
    if (
        not isinstance(update_records, list)
        or not isinstance(zero_gradient_records, list)
        or len(update_records) != len(expected_update_identities)
        or len(zero_gradient_records) != len(expected_zero_gradient_identities)
    ):
        raise ValueError("optimizer diagnostic ledger count mismatch")
    for label, records, expected_records in (
        ("optimizer update", update_records, expected_update_identities),
        (
            "zero-gradient update",
            zero_gradient_records,
            expected_zero_gradient_identities,
        ),
    ):
        for record, expected_record in zip(records, expected_records):
            if any(record.get(key) != value for key, value in expected_record.items()):
                raise ValueError(f"{label} diagnostic identity mismatch")
            for numeric_key in (
                "gradient_norm",
                "update_norm",
                "mean_policy_entropy",
            ):
                numeric = float(record[numeric_key])
                if not math.isfinite(numeric) or numeric < 0.0:
                    raise ValueError(f"{label} diagnostic numeric field is invalid")
    _assert_close(
        run["realized_coefficient_mass_total"],
        mass.sum(),
        "realized coefficient mass total",
    )

    x = _require_int_vector(run.get("x_transitions"), "transition axis", len(expected_x))
    if x.tolist() != expected_x:
        raise ValueError("checkpoint crossings do not reproduce from group ledger")
    n_checkpoints = len(x)
    sampled_group_axis = _require_int_vector(
        run.get("x_sampled_groups"), "sampled-group axis", n_checkpoints
    )
    if sampled_group_axis.tolist() != expected_sampled_group_axis:
        raise ValueError("sampled-group checkpoint axis does not reproduce")
    x_updates = _require_int_vector(
        run.get("x_optimizer_updates"), "optimizer-update axis", n_checkpoints
    )
    if x_updates.tolist() != expected_update_axis:
        raise ValueError("optimizer-update checkpoint axis does not reproduce")
    pass_rates = np.asarray(run["pass_rate_curve"], dtype=np.float64)
    if (
        pass_rates.shape != (n_checkpoints, 8)
        or not np.isfinite(pass_rates).all()
        or np.any((pass_rates < 0.0) | (pass_rates > 1.0))
        or np.any(np.diff(pass_rates, axis=1) > 1e-12)
        or not np.allclose(
            pass_rates * eval_n,
            np.rint(pass_rates * eval_n),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "pass-rate curve must be finite, bounded, and nested over 8 tasks"
        )
    mean_pass = np.asarray(run["mean_pass_curve"], dtype=np.float64)
    hardest_pass = np.asarray(run["hardest_pass_curve"], dtype=np.float64)
    native_success = np.asarray(run["native_success_rate_curve"], dtype=np.float64)
    for name, curve in (
        ("mean pass", mean_pass),
        ("hardest pass", hardest_pass),
        ("native success", native_success),
    ):
        if (
            curve.shape != (n_checkpoints,)
            or not np.isfinite(curve).all()
            or np.any((curve < 0.0) | (curve > 1.0))
        ):
            raise ValueError(f"{name} curve is invalid")
    if not np.allclose(
        mean_pass, pass_rates.mean(axis=1), rtol=0.0, atol=1e-12
    ):
        raise ValueError("mean-pass curve does not reproduce from pass rates")
    if not np.allclose(
        hardest_pass, pass_rates[:, 7], rtol=0.0, atol=1e-12
    ):
        raise ValueError("hardest-pass curve is not task-7 pass rate")
    if not np.allclose(
        native_success, hardest_pass, rtol=0.0, atol=1e-12
    ):
        raise ValueError("native-success curve differs from task-7 pass rate")
    native_return = np.asarray(run["mean_native_return_curve"], dtype=np.float64)
    censored_time = np.asarray(
        run["mean_censored_time_to_goal_curve"], dtype=np.float64
    )
    entropy = np.asarray(run["mean_policy_entropy_curve"], dtype=np.float64)
    if (
        native_return.shape != (n_checkpoints,)
        or not np.isfinite(native_return).all()
        or np.any((native_return < -500.0) | (native_return > -1.0))
    ):
        raise ValueError("native-return curve is invalid")
    if (
        censored_time.shape != (n_checkpoints,)
        or not np.isfinite(censored_time).all()
        or np.any((censored_time < 1.0) | (censored_time > 500.0))
    ):
        raise ValueError("censored-time curve is invalid")
    if (
        entropy.shape != (n_checkpoints,)
        or not np.isfinite(entropy).all()
        or np.any((entropy < 0.0) | (entropy > math.log(3.0) + 1e-12))
    ):
        raise ValueError("policy-entropy curve is invalid")
    rng_preserved = run["evaluation_rng_preserved"]
    if rng_preserved != [True] * n_checkpoints:
        raise ValueError("evaluation RNG-preservation curve is invalid")

    checkpoints = run["checkpoint_records"]
    curve_keys = {
        "pass_rates": "pass_rate_curve",
        "target_uniform_mean_pass_rate": "mean_pass_curve",
        "hardest_pass_rate": "hardest_pass_curve",
        "native_success_rate": "native_success_rate_curve",
        "mean_native_return": "mean_native_return_curve",
        "mean_censored_time_to_goal": "mean_censored_time_to_goal_curve",
        "mean_policy_entropy": "mean_policy_entropy_curve",
        "training_rng_preserved": "evaluation_rng_preserved",
    }
    if len(checkpoints) != n_checkpoints:
        raise ValueError("raw checkpoint record count mismatch")
    expected_evaluation_seed = expected_domains["rng_roots"]["evaluation_episode"]
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        if (
            checkpoint["checkpoint"] != checkpoint_index
            or checkpoint["transitions"] != int(x[checkpoint_index])
            or checkpoint["sampled_groups"]
            != int(sampled_group_axis[checkpoint_index])
            or checkpoint["optimizer_updates"]
            != int(x_updates[checkpoint_index])
            or checkpoint.get("evaluation_shared_trajectories") != eval_n
            or checkpoint["evaluation_seed"] != expected_evaluation_seed
            or checkpoint["training_rng_preserved"] is not True
        ):
            raise ValueError("checkpoint identity/cadence/CRN record mismatch")
        for checkpoint_key, curve_key in curve_keys.items():
            saved = checkpoint[checkpoint_key]
            expected = run[curve_key][checkpoint_index]
            if isinstance(saved, list):
                if not np.allclose(saved, expected, rtol=0.0, atol=1e-12):
                    raise ValueError(f"checkpoint {checkpoint_key} mismatch")
            elif isinstance(saved, bool) or saved is None or expected is None:
                if saved is not expected:
                    raise ValueError(f"checkpoint {checkpoint_key} mismatch")
            else:
                _assert_close(saved, expected, f"checkpoint {checkpoint_key}")

    final_fields = {
        "final_mean_pass": mean_pass[-1],
        "final_hardest_pass": hardest_pass[-1],
        "final_native_success_rate": native_success[-1],
        "final_mean_native_return": native_return[-1],
        "final_mean_censored_time_to_goal": censored_time[-1],
    }
    for key, recomputed in final_fields.items():
        _assert_close(run[key], recomputed, key)
    target_auc = normalized_trapezoid(mean_pass, x)
    native_success_auc = normalized_trapezoid(native_success, x)
    native_return_auc = normalized_trapezoid(native_return, x)
    sampled_group_auc = normalized_trapezoid(mean_pass, sampled_group_axis)
    optimizer_update_auc = normalized_trapezoid(mean_pass, x_updates)
    derived = {
        "target_uniform_transition_auc": target_auc,
        "native_success_auc": native_success_auc,
        "native_return_auc": native_return_auc,
        "target_uniform_sampled_group_auc": sampled_group_auc,
        "target_uniform_optimizer_update_auc": optimizer_update_auc,
        "sampled_groups_per_million_transitions": float(
            run["sampled_groups"] * 1_000_000.0 / transitions
        ),
        "optimizer_updates_per_million_transitions": float(
            run["optimizer_updates"] * 1_000_000.0 / transitions
        ),
        "final_native_success_rate": float(native_success[-1]),
        "final_native_return": float(native_return[-1]),
        "coefficient_mass_per_group": float(mass.mean()),
        "coefficient_mass_per_million_transitions": float(
            mass.sum() * 1_000_000.0 / transitions
        ),
        "nonzero_mass_group_fraction": float(np.count_nonzero(mass) / len(mass)),
        "teacher_max_tv_from_uniform": float(max(recomputed_tvs)),
    }
    saved_map = {
        "target_uniform_transition_auc": "auc_mean_pass_by_transitions",
        "native_success_auc": "auc_native_success_by_transitions",
        "native_return_auc": "auc_native_return_by_transitions",
        "target_uniform_sampled_group_auc": "auc_mean_pass_by_sampled_groups",
        "target_uniform_optimizer_update_auc": (
            "auc_mean_pass_by_optimizer_updates"
        ),
        "sampled_groups_per_million_transitions": (
            "sampled_groups_per_million_transitions"
        ),
        "optimizer_updates_per_million_transitions": (
            "optimizer_updates_per_million_transitions"
        ),
        "coefficient_mass_per_group": "realized_coefficient_mass_per_group",
        "coefficient_mass_per_million_transitions": (
            "realized_coefficient_mass_per_million_transitions"
        ),
        "nonzero_mass_group_fraction": "nonzero_coefficient_mass_group_fraction",
    }
    for derived_key, saved_key in saved_map.items():
        _assert_close(derived[derived_key], run[saved_key], saved_key)
    derived["regimes"] = regimes
    derived["task_groups"] = task_groups.tolist()
    derived["native_success_curve"] = native_success.tolist()
    return derived


def _validate_raw_artifact(artifact: dict) -> dict:
    if artifact.get("schema") != RAW_SCHEMA:
        raise ValueError("raw artifact schema mismatch")
    if (
        artifact.get("artifact_state") != "complete"
        or artifact.get("run_failures") != []
    ):
        raise ValueError("raw artifact is incomplete or contains failed runs")
    protocol = artifact.get("protocol", {})
    mode = protocol.get("mode")
    if mode not in SCHEDULES:
        raise ValueError("unknown raw tournament mode")
    seeds, budget, interval, eval_n = SCHEDULES[mode]
    expected_protocol = {
        "study": "acrobot_curriculum_tournament",
        "mode": mode,
        "status": "confirmatory" if mode == "confirmatory" else "development_only",
        "protocol_document": (
            "frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_PROTOCOL.md"
        ),
        "condition_names": list(EXPECTED_CASES),
        "paired_seeds": list(seeds),
        "logical_to_engine_master_seed": {
            str(seed): _engine_master_seed(seed) for seed in seeds
        },
        "thresholds": list(THRESHOLDS),
        "n_rollouts": 16,
        "transition_budget": budget,
        "complete_final_group": True,
        "eval_interval_transitions": interval,
        "eval_n_shared_trajectories": eval_n,
        "evaluation_threshold_scoring": (
            "shared nested trajectories reused across all eight thresholds"
        ),
        "evaluation_seed_base": EVALUATION_SEED_BASE,
        "fixed_evaluation_common_random_numbers": True,
        "architecture": "shared_h64_task_blind",
        "total_parameters": 640,
        "learning_rate": 3e-4,
        "optimizer": "plain SGD ascent",
        "estimator": "practical dropped-group MaxRL",
        "hindsight_scale": 0.0,
        "teacher": {
            "tracking": "discounted Beta Thompson sampling",
            "decay": TEACHER_DECAY,
            "floor": TEACHER_FLOOR,
            "gamma": 1.0,
            "utilities": dict(EXPECTED_SAMPLER_LABEL),
        },
        "primary": "u16 minus p(1-p) target-uniform transition AUC",
        "primary_test": "exact two-sided 2^20 paired sign flip",
        "primary_support": "20,000-resample paired-seed bootstrap interval",
        "primary_sesoi": PRIMARY_SESOI,
        "primary_decision": (
            "supported iff mean u16-p1mp AUC >= +0.01 and exact two-sided p <= 0.05"
        ),
        "secondary_uniform_tests": "p(1-p)-uniform and u16-uniform; Holm family",
        "raw_only": True,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"protocol mismatch for {key}")
    if "v3_adequacy_waiver" in protocol:
        raise ValueError("V2 protocol forbids a V3 adequacy waiver")
    if mode == "confirmatory":
        if not isinstance(protocol.get("development_gate"), dict):
            raise ValueError("confirmatory protocol lacks its development-gate binding")
    elif protocol.get("development_gate") is not None:
        raise ValueError("non-confirmatory raw artifact contains a launch gate")
    if tuple(artifact.get("cases", {})) != EXPECTED_CASES:
        raise ValueError("raw artifact arm set/order mismatch")
    expected_rng_contract = {
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
    }
    if protocol.get("rng_domain_contract") != expected_rng_contract:
        raise ValueError("protocol RNG-domain contract mismatch")

    by_case = {}
    for case_name in EXPECTED_CASES:
        case = artifact["cases"][case_name]
        config = case["config"]
        if (
            case.get("sampler") != EXPECTED_SAMPLER_LABEL[case_name]
            or
            config.get("name") != case_name
            or config.get("stage") != "tournament"
            or config.get("sampling") != EXPECTED_SAMPLING[case_name]
            or config.get("architecture") != "shared"
            or config.get("hidden_size") != 64
            or config.get("learning_rate") != 3e-4
            or config.get("hindsight_scale") != 0.0
        ):
            raise ValueError(f"arm configuration mismatch for {case_name}")
        runs = case["runs"]
        if [run.get("seed") for run in runs] != list(seeds):
            raise ValueError(f"paired seed order mismatch for {case_name}")
        by_case[case_name] = {
            run["seed"]: _run_metrics(
                run,
                budget=budget,
                interval=interval,
                eval_n=eval_n,
                mode=mode,
                sampling=EXPECTED_SAMPLING[case_name],
            )
            for run in runs
        }
    return {
        "mode": mode,
        "seeds": list(seeds),
        "budget": budget,
        "eval_interval": interval,
        "eval_n": eval_n,
        "by_case": by_case,
    }


def development_gates(validated: dict, source_lock: dict) -> dict:
    if validated["mode"] != "development":
        raise ValueError("development gates require the registered development mode")
    all_metrics = [
        metrics
        for case in EXPECTED_CASES
        for metrics in validated["by_case"][case].values()
    ]
    pooled_task_groups = np.sum(
        [metrics["task_groups"] for metrics in all_metrics], axis=0
    ).astype(int)
    observed_regimes = sorted(
        {regime for metrics in all_metrics for regime in metrics["regimes"]}
    )
    adaptive_nonuniform = {
        case: max(
            metrics["teacher_max_tv_from_uniform"]
            for metrics in validated["by_case"][case].values()
        ) > 1e-12
        for case in ("p1mp_shared_h64", "u16_shared_h64")
    }
    native_values = np.asarray(
        [
            value
            for metrics in all_metrics
            for value in metrics["native_success_curve"]
        ],
        dtype=np.float64,
    )
    gates = {
        "all_runs_accounting_numeric_verifier_parameter_cadence_valid": True,
        "all_tasks_visited_pooled_across_arms_and_seeds": bool(
            np.all(pooled_task_groups > 0)
        ),
        "p1mp_sampler_exhibits_nonuniform_distribution": adaptive_nonuniform[
            "p1mp_shared_h64"
        ],
        "u16_sampler_exhibits_nonuniform_distribution": adaptive_nonuniform[
            "u16_shared_h64"
        ],
        "pooled_dead_mixed_all_pass_regimes_observed": observed_regimes
        == ["all_pass", "dead", "mixed"],
        "native_success_checkpoint_values_vary": bool(
            len(native_values) > 1 and np.ptp(native_values) > 0.0
        ),
    }
    return {
        "schema": GATE_SCHEMA,
        "mode": "development",
        "all_gates_passed": all(gates.values()),
        "source_lock_sha256": source_lock["source_lock_sha256"],
        "source_lock_verification": source_lock,
        "gates": gates,
        "diagnostics": {
            "pooled_task_groups": pooled_task_groups.tolist(),
            "observed_group_regimes": observed_regimes,
            "adaptive_max_tv_from_uniform": {
                case: max(
                    metrics["teacher_max_tv_from_uniform"]
                    for metrics in validated["by_case"][case].values()
                )
                for case in adaptive_nonuniform
            },
            "native_success_min": float(native_values.min()),
            "native_success_max": float(native_values.max()),
        },
        "gate_policy": dict(DEVELOPMENT_GATE_POLICY),
    }


def _verify_confirmatory_development_gate(
    artifact: dict,
    lock: dict,
    lock_path: Path,
    source_lock: dict,
) -> dict:
    binding = artifact.get("protocol", {}).get("development_gate")
    expected_binding_keys = {
        "relative_path",
        "sha256",
        "raw_artifact_relative_path",
        "raw_artifact_sha256",
        "all_gates_passed",
    }
    if not isinstance(binding, dict) or set(binding) != expected_binding_keys:
        raise ValueError("confirmatory protocol lacks the exact development-gate binding")
    if binding.get("all_gates_passed") is not True:
        raise ValueError("confirmatory development-gate binding is not passing")

    gate_path = _project_relative_file(
        binding.get("relative_path"), "confirmatory development gate"
    )
    if binding.get("sha256") != _sha256(gate_path):
        raise ValueError("confirmatory development-gate hash mismatch")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        gate.get("schema") != GATE_SCHEMA
        or gate.get("mode") != "development"
        or gate.get("all_gates_passed") is not True
        or gate.get("source_lock_sha256") != source_lock["source_lock_sha256"]
        or tuple(gate.get("gates", {})) != DEVELOPMENT_GATE_NAMES
        or any(value is not True for value in gate["gates"].values())
        or gate.get("gate_policy") != DEVELOPMENT_GATE_POLICY
        or gate.get("source_lock_verification") != source_lock
    ):
        raise ValueError("bound development gate has invalid locked policy/provenance")
    if (
        gate.get("raw_artifact_relative_path")
        != binding.get("raw_artifact_relative_path")
        or gate.get("raw_artifact_sha256") != binding.get("raw_artifact_sha256")
    ):
        raise ValueError("confirmatory gate/raw binding mismatch")

    raw_path = _project_relative_file(
        binding.get("raw_artifact_relative_path"), "bound development raw artifact"
    )
    if binding.get("raw_artifact_sha256") != _sha256(raw_path):
        raise ValueError("bound development raw artifact hash mismatch")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    development_lock = _verify_lock(raw, lock, lock_path)
    if development_lock != source_lock:
        raise ValueError("development and confirmatory lock verification differ")
    validated_development = _validate_raw_artifact(raw)
    recomputed = development_gates(validated_development, development_lock)
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
        if gate.get(key) != recomputed.get(key):
            raise ValueError(f"development gate does not recompute for {key}")
    return {
        "passed": True,
        "development_gate_relative_path": binding["relative_path"],
        "development_gate_sha256": binding["sha256"],
        "development_raw_relative_path": binding["raw_artifact_relative_path"],
        "development_raw_sha256": binding["raw_artifact_sha256"],
        "gates_recomputed_from_raw": True,
    }


def _paired_values(validated: dict, left: str, right: str, metric: str) -> np.ndarray:
    seeds = validated["seeds"]
    return np.asarray(
        [
            validated["by_case"][left][seed][metric]
            - validated["by_case"][right][seed][metric]
            for seed in seeds
        ],
        dtype=np.float64,
    )


def _paired_summary(
    validated: dict,
    left: str,
    right: str,
    metric: str,
    *,
    bootstrap_seed: int,
    include_test: bool,
) -> dict:
    values = _paired_values(validated, left, right, metric)
    result = {
        "estimand": f"{left} minus {right}",
        "metric": metric,
        "n_paired_seeds": len(values),
        "mean_paired_difference": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "paired_differences": values.tolist(),
        "paired_bootstrap_ci95_20000": paired_bootstrap_ci(
            values, seed=bootstrap_seed
        ),
    }
    if include_test:
        result["exact_two_sided_sign_flip_p"] = exact_two_sided_sign_flip_p(values)
    return result


def confirmatory_analysis(validated: dict, source_lock: dict) -> dict:
    if validated["mode"] != "confirmatory" or len(validated["seeds"]) != 20:
        raise ValueError("confirmatory analysis requires exactly 20 sealed pairs")
    primary = _paired_summary(
        validated,
        "u16_shared_h64",
        "p1mp_shared_h64",
        "target_uniform_transition_auc",
        bootstrap_seed=PRIMARY_BOOTSTRAP_SEED,
        include_test=True,
    )
    primary["test_assignments"] = 2**20
    primary["minimum_practical_effect_sesoi"] = PRIMARY_SESOI
    primary["sesoi_scale"] = "normalized [0,1] target-uniform transition AUC"
    primary["sesoi_paid_horizon_interpretation"] = (
        "one average percentage point over 2,000,000 paid transitions, equal "
        "to 20,000 pass-rate-by-transition units"
    )
    primary["sesoi_checkpoint_resolution_interpretation"] = (
        "1/(32*8)=0.00390625 per checkpoint mean-pass unit; +0.01 is 2.56 "
        "such resolution units"
    )
    primary["sesoi_provenance"] = (
        "judgment-based convention fixed before V2 primary-arm outcomes and "
        "not derived from V1 or aborted primary outcomes"
    )
    primary["positive_mean_direction"] = bool(
        primary["mean_paired_difference"] > 0.0
    )
    primary["statistically_significant_two_sided_0.05"] = bool(
        primary["exact_two_sided_sign_flip_p"] <= 0.05
    )
    primary["meets_sesoi"] = bool(
        primary["mean_paired_difference"] >= PRIMARY_SESOI
    )
    primary["efficacy_supported"] = bool(
        primary["positive_mean_direction"]
        and primary["statistically_significant_two_sided_0.05"]
        and primary["meets_sesoi"]
    )
    primary["decision_label"] = (
        "confirmed" if primary["efficacy_supported"] else "not confirmed"
    )
    primary["sign_exchangeability_caveat"] = (
        "The exact paired sign-flip randomization interpretation requires the "
        "paired effects to be sign-exchangeable under the sharp null; pairing "
        "and common random numbers do not make this assumption automatic."
    )
    primary["bootstrap_role"] = (
        "20,000-resample paired-seed percentile interval for estimation support; "
        "it is not a separate decision rule and failure is not equivalence"
    )
    secondary_specs = (
        ("p1mp_minus_uniform", "p1mp_shared_h64", "uniform_shared_h64"),
        ("u16_minus_uniform", "u16_shared_h64", "uniform_shared_h64"),
    )
    secondary_tests = {}
    for index, (name, left, right) in enumerate(secondary_specs):
        secondary_tests[name] = _paired_summary(
            validated,
            left,
            right,
            "target_uniform_transition_auc",
            bootstrap_seed=PRIMARY_BOOTSTRAP_SEED + 10 + index,
            include_test=True,
        )
    adjusted = holm_adjust(
        {
            name: record["exact_two_sided_sign_flip_p"]
            for name, record in secondary_tests.items()
        }
    )
    for name, correction in adjusted.items():
        secondary_tests[name].update(correction)
        secondary_tests[name]["positive_mean_direction"] = bool(
            secondary_tests[name]["mean_paired_difference"] > 0.0
        )
        secondary_tests[name]["efficacy_supported"] = bool(
            secondary_tests[name]["positive_mean_direction"]
            and secondary_tests[name]["reject_familywise_0.05"]
        )

    metrics = (
        "native_success_auc",
        "native_return_auc",
        "target_uniform_sampled_group_auc",
        "target_uniform_optimizer_update_auc",
        "sampled_groups_per_million_transitions",
        "optimizer_updates_per_million_transitions",
        "final_native_success_rate",
        "final_native_return",
        "coefficient_mass_per_group",
        "coefficient_mass_per_million_transitions",
        "nonzero_mass_group_fraction",
    )
    descriptive = {}
    pairs = (
        ("p1mp_minus_uniform", "p1mp_shared_h64", "uniform_shared_h64"),
        ("u16_minus_uniform", "u16_shared_h64", "uniform_shared_h64"),
        ("u16_minus_p1mp", "u16_shared_h64", "p1mp_shared_h64"),
    )
    for metric_index, metric in enumerate(metrics):
        arm_values = {
            case: [
                validated["by_case"][case][seed][metric]
                for seed in validated["seeds"]
            ]
            for case in EXPECTED_CASES
        }
        descriptive[metric] = {
            "arm_means": {
                case: float(np.mean(values)) for case, values in arm_values.items()
            },
            "paired_descriptive_contrasts": {
                name: _paired_summary(
                    validated,
                    left,
                    right,
                    metric,
                    bootstrap_seed=(
                        PRIMARY_BOOTSTRAP_SEED + 100 + 10 * metric_index + pair_index
                    ),
                    include_test=False,
                )
                for pair_index, (name, left, right) in enumerate(pairs)
            },
        }
    return {
        "schema": REPORT_SCHEMA,
        "mode": "confirmatory",
        "all_checks_passed": True,
        "source_lock": source_lock,
        "primary": primary,
        "primary_multiplicity": "one registered test; no adjustment",
        "secondary_uniform_auc_tests": secondary_tests,
        "secondary_uniform_multiplicity": {
            "family": [name for name, _, _ in secondary_specs],
            "method": "Holm step-down",
            "familywise_alpha": 0.05,
        },
        "secondary_descriptive_metrics": descriptive,
        "claim_boundary": (
            "native and realized-mass endpoints are descriptive and cannot rescue "
            "the primary; sampled-group/update axes and their rates are "
            "non-confirmatory cost-composition diagnostics; no PLR/ALP/PAIRED/"
            "ACCEL or hindsight claim"
        ),
    }


def analyze(
    artifact: dict,
    lock: dict | None = None,
    lock_path: Path | None = None,
) -> dict:
    validated = _validate_raw_artifact(artifact)
    if validated["mode"] == "quick":
        return {
            "schema": REPORT_SCHEMA,
            "mode": "quick",
            "all_checks_passed": True,
            "inference_performed": False,
            "reason": (
                "quick smoke is development-only; raw ledgers were validated "
                "without requiring or consulting a source lock"
            ),
        }
    if lock is None or lock_path is None:
        raise ValueError("development/confirmatory analysis requires a source lock")
    source_lock = _verify_lock(artifact, lock, lock_path)
    if validated["mode"] == "development":
        return development_gates(validated, source_lock)
    gate_verification = _verify_confirmatory_development_gate(
        artifact, lock, lock_path, source_lock
    )
    report = confirmatory_analysis(validated, source_lock)
    report["development_gate_verification"] = gate_verification
    return report


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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
        is_quick = artifact.get("protocol", {}).get("mode") == "quick"
        lock = (
            None
            if is_quick
            else json.loads(args.lock.read_text(encoding="utf-8"))
        )
        report = analyze(
            artifact,
            lock,
            None if is_quick else args.lock.resolve(),
        )
        report["raw_artifact_path"] = str(args.artifact.resolve())
        report["raw_artifact_sha256"] = _sha256(args.artifact)
        try:
            report["raw_artifact_relative_path"] = str(
                args.artifact.resolve().relative_to(PROJECT_ROOT.resolve())
            )
        except ValueError:
            report["raw_artifact_relative_path"] = None
        if not is_quick:
            report["source_lock_path"] = str(args.lock.resolve())
            report["source_lock_relative_path"] = str(
                args.lock.resolve().relative_to(PROJECT_ROOT.resolve())
            )
            report["source_lock_sha256"] = _sha256(args.lock)
        _write_json(args.output, report, overwrite=args.overwrite)
    except (ValueError, RuntimeError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
