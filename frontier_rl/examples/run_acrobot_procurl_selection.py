"""Raw runner for the source-faithful Acrobot ProCuRL selection study.

This is a new attachment around the byte-unchanged Acrobot learner.  It does
not import or copy upstream ProCuRL code.  Quick mode is engineering-only;
development and confirmation fail closed unless the canonical source/runtime
lock (and, for confirmation, the outcome-blind development gate) verify.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import platform
import subprocess
import tempfile
import time
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np

from frontier_rl.adapters.acrobot_neural import (
    MAX_EPISODE_STEPS,
    AcrobotNeuralSpace,
    TanhCategoricalActor,
    normalize_observation,
    tip_height,
)
from frontier_rl.examples import run_acrobot_neural as engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RAW_SCHEMA = "curriculum-maxrl/acrobot-procurl-selection-raw/v1"
LOCK_SCHEMA = "curriculum-maxrl/acrobot-procurl-selection-lock/v1"
GATE_SCHEMA = "curriculum-maxrl/acrobot-procurl-selection-development-gates/v1"
LOCK_PATH = HERE / "ACROBOT_PROCURL_SELECTION_LOCK.json"
PROTOCOL_PATH = HERE / "ACROBOT_PROCURL_SELECTION_PROTOCOL.md"
PROVENANCE_PATH = HERE / "PROCURL_PRIMARY_SOURCE_PROVENANCE.md"
V2_LOCK_PATH = HERE / "ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json"
V2_DEPENDENCY_PATHS = (
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
LOCK_KEYS = {
    "schema",
    "status",
    "created_utc",
    "purpose",
    "runtime",
    "schedule",
    "seed_collision_audit",
    "source_sha256",
    "v2_dependency_audit",
}

UPSTREAM_PROCURL_COMMIT = "17904f1d7b9b29e089d4f70ae7aadf1da50ba6b2"
CONFIRMATORY_SEEDS = tuple(range(21_000, 21_080))
DEVELOPMENT_SEEDS = tuple(range(21_300, 21_303))
QUICK_SEEDS = (21_400,)
CONFIRMATORY_PAID_BUDGET = 2_000_000
DEVELOPMENT_PAID_BUDGET = 400_000
QUICK_PAID_BUDGET = 100_000
REGULAR_EVAL_INTERVAL_PAID = 100_000
CONFIRMATORY_EVAL_N = 32
DEVELOPMENT_EVAL_N = 32
QUICK_EVAL_N = 2
N_ROLLOUTS = 16
LEARNING_RATE = 3e-4
PROBES_PER_TASK = 20
REFRESH_STUDENT_TRANSITIONS = 5_120
PROCURL_BETA = 20.0
U16_BETA_CONTINUOUS_RANGE_MATCHED = 6.416133525771289
U16_LATTICE_MAX_LOGIT = 4.97730861318145
PRIMARY_SESOI = 0.02
BOOTSTRAP_RESAMPLES = 20_000
SIGN_FLIP_MONTE_CARLO_DRAWS = 1_000_000
MAX_STUDENT_GROUP_TRANSITIONS = N_ROLLOUTS * MAX_EPISODE_STEPS
MAX_PROBE_SWEEP_TRANSITIONS = (
    len(engine.THRESHOLDS) * PROBES_PER_TASK * MAX_EPISODE_STEPS
)

ENGINE_MASTER_BASE = 50_000_000_000
ENGINE_MASTER_STRIDE = 10_000_000
ENVIRONMENT_ADAPTER_SEED_OFFSET = 1_000
RNG_DOMAIN_OFFSETS = {
    "actor_parameter": 0,
    "actor_action": 1,
    "selection": 10_000,
    "environment_reset_rng": 11_003,
    "evaluation_episode": 1_000_000,
    "evaluation_action": 1_000_001,
    "probe_episode_reset": 2_000_000,
    "probe_episode_action": 3_000_000,
}

# Collision-free coordinate encodings.  With at most 391 refresh boundaries
# under the 2M paid budget, 512 sweep slots are sufficient with a hard margin.
# The reset namespaces are disjoint and remain below Gymnasium's 2**31 limit;
# action namespaces remain below NumPy's signed 64-bit seed limit.
MAX_ENCODED_PROBE_SWEEPS = 512
PROBE_COORDINATE_BLOCK = MAX_ENCODED_PROBE_SWEEPS * 8 * PROBES_PER_TASK
EVALUATION_RESET_NAMESPACE_BASE = 1_800_000_000
EVALUATION_ACTION_NAMESPACE_BASE = 4_000_000_000_000_000_000
PROBE_ACTION_NAMESPACE_BASE = 3_000_000_000_000_000_000

PINNED_RUNTIME_VERSIONS = {
    "python_implementation": "CPython",
    "python": "3.12.13",
    "numpy": "2.5.1",
    "gymnasium": "1.3.0",
}

DEVELOPMENT_GATE_NAMES = (
    "all_runs_source_numeric_parameter_rng_ledger_valid",
    "all_sweeps_exact_probe_count_and_bounded_transitions",
    "all_p_hat_values_are_multiples_of_0p05",
    "initial_and_crossed_boundary_sweep_schedule_exact",
    "probes_preserve_actor_optimizer_and_training_rng",
    "paid_equals_student_plus_probe",
    "uniform_arms_exact_and_ordinary_has_no_probes",
    "adaptive_probabilities_recompute_and_nonuniform_once",
    "each_probed_run_has_20k_student_transitions_and_update",
    "pooled_dead_mixed_all_pass_regimes_observed",
    "pooled_native_evaluation_values_vary",
)
DEVELOPMENT_GATE_POLICY = {
    "outcome_blind": True,
    "uses_arm_contrasts": False,
    "uses_effect_direction": False,
    "uses_confidence_intervals": False,
    "uses_p_values": False,
    "uses_minimum_effect": False,
}


@dataclass(frozen=True)
class Arm:
    """One registered selection rule; learner fields are intentionally absent."""

    name: str
    selection: str
    probes: bool


ARMS = (
    Arm("procurl_env_b20_f5120", "procurl_p1mp_softmax", True),
    Arm("probe_sham_uniform_f5120", "uniform_sham", True),
    Arm("ordinary_uniform", "uniform_ordinary", False),
    Arm("u16_probe_range_matched_f5120", "u16_softmax", True),
)
ARM_BY_NAME = {arm.name: arm for arm in ARMS}

PRIOR_LOGICAL_SEED_BLOCKS = {
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
    "aborted_tournament_confirmation": tuple(range(19_000, 19_020)),
    "aborted_tournament_development": tuple(range(19_100, 19_103)),
    "acrobot_tournament_confirmation": tuple(range(20_000, 20_020)),
    "acrobot_tournament_development": tuple(range(20_100, 20_103)),
    "acrobot_tournament_quick": (20_200,),
    "invalid_procurl_selection_development_pre_gate_entropy_sum_mismatch": tuple(
        range(21_100, 21_103)
    ),
    "invalid_procurl_selection_quick_pre_gate_entropy_sum_mismatch": (21_200,),
}

SOURCE_RELATIVE_PATHS = (
    "frontier_rl/examples/run_acrobot_procurl_selection.py",
    "frontier_rl/examples/analyze_acrobot_procurl_selection.py",
    "frontier_rl/examples/build_acrobot_procurl_selection_lock.py",
    "frontier_rl/examples/verify_acrobot_procurl_selection_portable.py",
    "frontier_rl/examples/test_run_acrobot_procurl_selection.py",
    "frontier_rl/examples/test_analyze_acrobot_procurl_selection.py",
    "frontier_rl/examples/test_build_acrobot_procurl_selection_lock.py",
    "frontier_rl/examples/test_verify_acrobot_procurl_selection_portable.py",
    "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_PROTOCOL.md",
    "frontier_rl/examples/PROCURL_PRIMARY_SOURCE_PROVENANCE.md",
    "frontier_rl/examples/INVALID_ABORTED_PRE_GATE_ENTROPY_SUM_MISMATCH/INCIDENT.json",
    "frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json",
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


def _is_utc_iso8601(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(
        None
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _load_strict_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} as strict JSON: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain exactly one JSON object")
    return payload


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
    hashes: dict[str, str] = {}
    missing = []
    for relative in SOURCE_RELATIVE_PATHS:
        path = PROJECT_ROOT / relative
        if path.is_file():
            hashes[relative] = _sha256(path)
        else:
            missing.append(relative)
    if require_all and missing:
        raise RuntimeError(
            "missing locked ProCuRL-study sources: " + ", ".join(missing)
        )
    return hashes


def _v2_dependency_audit() -> dict:
    v2 = _load_strict_json(V2_LOCK_PATH, "V2 Acrobot dependency lock")
    if v2.get("schema") != "curriculum-maxrl/acrobot-curriculum-tournament-lock/v2":
        raise RuntimeError("V2 dependency lock schema mismatch")
    v2_hashes = v2.get("source_sha256")
    if not isinstance(v2_hashes, dict) or not set(V2_DEPENDENCY_PATHS) <= set(
        v2_hashes
    ):
        raise RuntimeError("V2 dependency lock lacks required transitive sources")
    live = {path: _sha256(PROJECT_ROOT / path) for path in V2_DEPENDENCY_PATHS}
    frozen = {path: v2_hashes[path] for path in V2_DEPENDENCY_PATHS}
    if live != frozen:
        raise RuntimeError("live ProCuRL dependencies differ from the V2 source lock")
    return {
        "passed": True,
        "v2_lock_relative_path": str(V2_LOCK_PATH.relative_to(PROJECT_ROOT)),
        "v2_lock_sha256": _sha256(V2_LOCK_PATH),
        "v2_lock_schema": v2["schema"],
        "dependency_paths": list(V2_DEPENDENCY_PATHS),
        "live_dependency_sha256": live,
        "v2_locked_dependency_sha256": frozen,
        "all_live_dependencies_match_v2": True,
    }


def engine_master_seed(logical_seed: int) -> int:
    if type(logical_seed) is not int or logical_seed < 0:
        raise ValueError("logical_seed must be a non-negative primitive int")
    return ENGINE_MASTER_BASE + logical_seed * ENGINE_MASTER_STRIDE


def rng_domain_record(logical_seed: int) -> dict:
    master = engine_master_seed(logical_seed)
    return {
        "logical_seed": logical_seed,
        "engine_master_seed": master,
        "environment_adapter_seed_argument": master + ENVIRONMENT_ADAPTER_SEED_OFFSET,
        "rng_roots": {
            name: master + offset for name, offset in RNG_DOMAIN_OFFSETS.items()
        },
    }


def seed_collision_audit() -> dict:
    registered = CONFIRMATORY_SEEDS + DEVELOPMENT_SEEDS + QUICK_SEEDS
    prior = set().union(*(set(values) for values in PRIOR_LOGICAL_SEED_BLOCKS.values()))
    blocks = {
        "confirmatory_vs_prior": sorted(set(CONFIRMATORY_SEEDS) & prior),
        "development_vs_prior": sorted(set(DEVELOPMENT_SEEDS) & prior),
        "quick_vs_prior": sorted(set(QUICK_SEEDS) & prior),
        "confirmatory_vs_development": sorted(
            set(CONFIRMATORY_SEEDS) & set(DEVELOPMENT_SEEDS)
        ),
        "confirmatory_vs_quick": sorted(set(CONFIRMATORY_SEEDS) & set(QUICK_SEEDS)),
        "development_vs_quick": sorted(set(DEVELOPMENT_SEEDS) & set(QUICK_SEEDS)),
    }
    records = {str(seed): rng_domain_record(seed) for seed in registered}
    owners: dict[int, tuple[int, str]] = {}
    derived_collisions = []
    for seed in registered:
        for domain, root in records[str(seed)]["rng_roots"].items():
            if root in owners:
                derived_collisions.append(
                    {
                        "root": root,
                        "first": list(owners[root]),
                        "second": [seed, domain],
                    }
                )
            owners[root] = (seed, domain)
    derived_vs_logical = sorted(set(owners) & (prior | set(registered)))
    expected = len(registered) * len(RNG_DOMAIN_OFFSETS)
    passed = (
        not any(blocks.values())
        and not derived_collisions
        and not derived_vs_logical
        and len(owners) == expected
        and len(CONFIRMATORY_SEEDS) == 80
        and len(DEVELOPMENT_SEEDS) == 3
        and len(QUICK_SEEDS) == 1
    )
    result = {
        "passed": passed,
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "quick_seeds": list(QUICK_SEEDS),
        "prior_logical_seed_blocks": {
            key: list(value) for key, value in PRIOR_LOGICAL_SEED_BLOCKS.items()
        },
        "logical_collisions": blocks,
        "derived_root_collisions": derived_collisions,
        "derived_roots_vs_logical_seeds": derived_vs_logical,
        "unique_derived_root_count": len(owners),
        "expected_unique_derived_root_count": expected,
        "per_logical_seed": records,
        "pairing": "all four arms deliberately reuse roots within logical seed",
        "collision_free_episode_seed_encoding": {
            "passed": True,
            "probe_reset_formula": (
                "((logical_seed*512+sweep_ordinal-1)*8+task_id)*20+episode"
            ),
            "probe_sweep_slots_per_seed": MAX_ENCODED_PROBE_SWEEPS,
            "probe_coordinate_block_per_seed": PROBE_COORDINATE_BLOCK,
            "maximum_registered_probe_reset_seed": (
                ((max(registered) * MAX_ENCODED_PROBE_SWEEPS + 511) * 8 + 7)
                * PROBES_PER_TASK
                + 19
            ),
            "evaluation_reset_namespace_base": EVALUATION_RESET_NAMESPACE_BASE,
            "probe_action_namespace_base": PROBE_ACTION_NAMESPACE_BASE,
            "evaluation_action_namespace_base": EVALUATION_ACTION_NAMESPACE_BASE,
            "coordinate_ranges_disjoint": True,
        },
    }
    if not passed:
        raise RuntimeError(f"ProCuRL-study seed collision audit failed: {result!r}")
    return result


def _mode_schedule(mode: str) -> tuple[tuple[int, ...], int, int]:
    if mode == "confirmatory":
        return CONFIRMATORY_SEEDS, CONFIRMATORY_PAID_BUDGET, CONFIRMATORY_EVAL_N
    if mode == "development":
        return DEVELOPMENT_SEEDS, DEVELOPMENT_PAID_BUDGET, DEVELOPMENT_EVAL_N
    if mode == "quick":
        return QUICK_SEEDS, QUICK_PAID_BUDGET, QUICK_EVAL_N
    raise ValueError(f"unknown mode {mode!r}")


def _locked_schedule() -> dict:
    return {
        "arm_names": [arm.name for arm in ARMS],
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "quick_seeds": list(QUICK_SEEDS),
        "confirmatory_paid_budget": CONFIRMATORY_PAID_BUDGET,
        "development_paid_budget": DEVELOPMENT_PAID_BUDGET,
        "quick_paid_budget": QUICK_PAID_BUDGET,
        "regular_eval_interval_paid": REGULAR_EVAL_INTERVAL_PAID,
        "confirmatory_eval_n": CONFIRMATORY_EVAL_N,
        "development_eval_n": DEVELOPMENT_EVAL_N,
        "quick_eval_n": QUICK_EVAL_N,
        "n_rollouts": N_ROLLOUTS,
        "learning_rate": LEARNING_RATE,
        "probes_per_task": PROBES_PER_TASK,
        "refresh_student_transitions": REFRESH_STUDENT_TRANSITIONS,
        "procurl_beta": PROCURL_BETA,
        "u16_beta_continuous_range_matched": U16_BETA_CONTINUOUS_RANGE_MATCHED,
        "u16_lattice_max_logit": U16_LATTICE_MAX_LOGIT,
        "engine_master_base": ENGINE_MASTER_BASE,
        "engine_master_stride": ENGINE_MASTER_STRIDE,
        "rng_domain_offsets": dict(RNG_DOMAIN_OFFSETS),
        "environment_adapter_seed_offset": ENVIRONMENT_ADAPTER_SEED_OFFSET,
        "upstream_procurl_commit": UPSTREAM_PROCURL_COMMIT,
    }


def _require_canonical_lock_path(path: Path) -> None:
    if path.resolve() != LOCK_PATH.resolve():
        raise RuntimeError(
            "development and confirmation require the canonical ProCuRL-study lock: "
            f"{LOCK_PATH.resolve()}"
        )


def _load_and_verify_lock(path: Path = LOCK_PATH) -> tuple[dict, str]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"source lock is missing: {path}")
    lock = _load_strict_json(path, "ProCuRL-study source lock")
    errors = []
    if set(lock) != LOCK_KEYS:
        errors.append("lock top-level schema is not closed")
    if lock.get("schema") != LOCK_SCHEMA:
        errors.append("lock schema mismatch")
    if lock.get("status") != "sealed_before_any_quick_development_or_confirmation":
        errors.append("lock status mismatch")
    if lock.get("purpose") != (
        "Canonical pre-execution source/runtime lock for the Acrobot "
        "ProCuRL selection-semantic study."
    ):
        errors.append("lock purpose mismatch")
    if not _is_utc_iso8601(lock.get("created_utc")):
        errors.append("lock creation timestamp invalid")
    live_runtime = _runtime()
    if _runtime_versions(live_runtime) != PINNED_RUNTIME_VERSIONS:
        errors.append(f"live runtime is not pinned: {live_runtime!r}")
    if lock.get("runtime") != live_runtime:
        errors.append("runtime mismatch")
    if lock.get("schedule") != _locked_schedule():
        errors.append("locked schedule mismatch")
    if lock.get("seed_collision_audit") != seed_collision_audit():
        errors.append("seed collision audit mismatch")
    try:
        live_v2_audit = _v2_dependency_audit()
    except (ValueError, RuntimeError) as error:
        errors.append(f"V2 dependency audit failed: {error}")
    else:
        if lock.get("v2_dependency_audit") != live_v2_audit:
            errors.append("V2 dependency audit mismatch")
    live_hashes = _source_hashes(require_all=True)
    if set(lock.get("source_sha256", {})) != set(SOURCE_RELATIVE_PATHS):
        errors.append("source lock key set is not exact")
    if lock.get("source_sha256") != live_hashes:
        errors.append("source hash mismatch")
    if errors:
        raise RuntimeError(
            "ProCuRL-study source/runtime lock failed: " + "; ".join(errors)
        )
    return lock, _sha256(path)


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(engine.THRESHOLDS),) or not np.isfinite(logits).all():
        raise ValueError("selection logits must be a finite eight-vector")
    shifted = logits - float(np.max(logits))
    weights = np.exp(shifted)
    probabilities = weights / float(np.sum(weights))
    if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-15):
        raise FloatingPointError("stable softmax failed to normalize")
    return probabilities


def selection_distribution(arm: Arm, p_hat: Sequence[float] | None) -> dict:
    """Compute the registered distribution from one complete latest sweep."""
    if arm.name not in ARM_BY_NAME or ARM_BY_NAME[arm.name] != arm:
        raise ValueError("arm is not registered")
    uniform = np.full(len(engine.THRESHOLDS), 1.0 / len(engine.THRESHOLDS))
    if arm.selection in {"uniform_sham", "uniform_ordinary"}:
        return {
            "p_hat": None if p_hat is None else [float(x) for x in p_hat],
            "utility": None,
            "logits": None,
            "probabilities": uniform.tolist(),
            "estimates_used_for_selection": False,
        }
    if p_hat is None:
        raise ValueError("adaptive arms require a complete p_hat vector")
    p = np.asarray(p_hat, dtype=np.float64)
    if p.shape != (len(engine.THRESHOLDS),) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_hat must be an eight-vector in [0,1]")
    if arm.selection == "procurl_p1mp_softmax":
        utility = p * (1.0 - p)
        logits = PROCURL_BETA * utility
    elif arm.selection == "u16_softmax":
        utility = 1.0 - np.power(1.0 - p, N_ROLLOUTS) - p
        logits = U16_BETA_CONTINUOUS_RANGE_MATCHED * utility
    else:  # pragma: no cover - Arm table is frozen and tests guard it
        raise ValueError(f"unknown selection rule {arm.selection!r}")
    probabilities = _stable_softmax(logits)
    return {
        "p_hat": p.tolist(),
        "utility": utility.tolist(),
        "logits": logits.tolist(),
        "probabilities": probabilities.tolist(),
        "estimates_used_for_selection": True,
    }


def crossed_refresh_boundaries(student_before: int, student_after: int) -> list[int]:
    if not (0 <= student_before <= student_after):
        raise ValueError(
            "student transition coordinates must be ordered and non-negative"
        )
    left = student_before // REFRESH_STUDENT_TRANSITIONS
    right = student_after // REFRESH_STUDENT_TRANSITIONS
    return [index * REFRESH_STUDENT_TRANSITIONS for index in range(left + 1, right + 1)]


def probe_episode_seeds(
    logical_seed: int, sweep_ordinal: int, task_id: int, episode: int
) -> tuple[int, int]:
    if any(
        type(value) is not int
        for value in (logical_seed, sweep_ordinal, task_id, episode)
    ):
        raise ValueError("probe coordinates must be primitive integers")
    if logical_seed not in CONFIRMATORY_SEEDS + DEVELOPMENT_SEEDS + QUICK_SEEDS:
        raise ValueError("probe logical seed is not registered")
    if not 1 <= sweep_ordinal <= MAX_ENCODED_PROBE_SWEEPS:
        raise ValueError("probe sweep ordinal exceeds collision-free namespace")
    if not 0 <= task_id < 8 or not 0 <= episode < PROBES_PER_TASK:
        raise ValueError("probe task/episode coordinate is outside the namespace")
    coordinate = (
        (logical_seed * MAX_ENCODED_PROBE_SWEEPS + (sweep_ordinal - 1)) * 8 + task_id
    ) * PROBES_PER_TASK + episode
    reset_seed = coordinate
    action_seed = PROBE_ACTION_NAMESPACE_BASE + coordinate
    if not (0 <= reset_seed < EVALUATION_RESET_NAMESPACE_BASE):
        raise RuntimeError("probe reset namespace overlaps evaluation namespace")
    if not (0 <= action_seed < 2**63 - 1):
        raise RuntimeError("probe action seed exceeds signed 64-bit range")
    return reset_seed, action_seed


def _training_state_fingerprint(
    space: AcrobotNeuralSpace,
    actor: TanhCategoricalActor,
    selection_rng: np.random.Generator,
    counters: dict[str, int],
) -> str:
    """Hash all state a probe or evaluation is forbidden to mutate."""
    encoded = pickle.dumps(
        {
            "engine_state": engine._training_state_fingerprint(space, actor),
            "selection_rng": copy.deepcopy(selection_rng.bit_generator.state),
            "counters": {key: int(value) for key, value in sorted(counters.items())},
        },
        protocol=5,
    )
    return hashlib.sha256(encoded).hexdigest()


def _parameter_fingerprint(actor: TanhCategoricalActor) -> str:
    return hashlib.sha256(actor.parameter_vector().tobytes(order="C")).hexdigest()


def _actions_sha256(actions: Sequence[int]) -> str:
    if any(type(action) is not int or action not in {0, 1, 2} for action in actions):
        raise ValueError("Acrobot action digest requires primitive actions in {0,1,2}")
    return hashlib.sha256(bytes(actions)).hexdigest()


def _trajectory_actions(trajectory: Sequence) -> list[int]:
    actions = []
    for transition in trajectory:
        if hasattr(transition, "action"):
            action = int(transition.action)
        elif isinstance(transition, dict) and "action" in transition:
            action = int(transition["action"])
        else:
            action = int(transition[1])
        actions.append(action)
    return actions


def _rng_state_fingerprint(rng: np.random.Generator) -> str:
    return hashlib.sha256(
        pickle.dumps(copy.deepcopy(rng.bit_generator.state), protocol=5)
    ).hexdigest()


def _probe_episode(
    probe_env,
    actor: TanhCategoricalActor,
    *,
    threshold: float,
    reset_seed: int,
    action_seed: int,
) -> dict:
    observation, _ = probe_env.reset(seed=int(reset_seed))
    action_rng = np.random.default_rng(int(action_seed))
    success = False
    max_height = -math.inf
    transitions = 0
    actions: list[int] = []
    for _ in range(MAX_EPISODE_STEPS):
        probabilities = actor.probabilities(normalize_observation(observation), 0)
        action = int(action_rng.choice(3, p=probabilities))
        actions.append(action)
        observation, _, terminated, truncated, _ = probe_env.step(action)
        transitions += 1
        height = tip_height(observation)
        max_height = max(max_height, height)
        if height > threshold:
            success = True
            break
        if terminated or truncated:
            break
    if not 1 <= transitions <= MAX_EPISODE_STEPS or not math.isfinite(max_height):
        raise RuntimeError("invalid probe episode accounting")
    return {
        "success": success,
        "transitions": transitions,
        "max_height": float(max_height),
        "action_count": len(actions),
        "action_sha256": _actions_sha256(actions),
    }


def run_probe_sweep(
    arm: Arm,
    actor: TanhCategoricalActor,
    space: AcrobotNeuralSpace,
    selection_rng: np.random.Generator,
    *,
    logical_seed: int,
    sweep_ordinal: int,
    trigger: str,
    crossed_boundary: int | None,
    student_transitions: int,
    paid_before: int,
    sampled_groups: int,
    optimizer_updates: int,
) -> dict:
    """Execute one complete paid probe sweep without touching training state."""
    if not arm.probes:
        raise ValueError("ordinary uniform has no probe sweep")
    if trigger not in {"initial", "refresh"}:
        raise ValueError("probe trigger must be initial or refresh")
    counters = {
        "student_transitions": student_transitions,
        "paid_transitions": paid_before,
        "sampled_groups": sampled_groups,
        "optimizer_updates": optimizer_updates,
    }
    training_before = _training_state_fingerprint(space, actor, selection_rng, counters)
    parameters_before = _parameter_fingerprint(actor)
    actor_update_calls_before = int(actor.update_calls)
    actor_applied_updates_before = int(actor.applied_updates)
    task_records = []
    probe_env = gymnasium.make("Acrobot-v1")
    try:
        for task_id, threshold in enumerate(engine.THRESHOLDS):
            episodes = []
            for episode in range(PROBES_PER_TASK):
                reset_seed, action_seed = probe_episode_seeds(
                    logical_seed, sweep_ordinal, task_id, episode
                )
                episodes.append(
                    {
                        "episode": episode,
                        "reset_seed": reset_seed,
                        "action_seed": action_seed,
                        **_probe_episode(
                            probe_env,
                            actor,
                            threshold=float(threshold),
                            reset_seed=reset_seed,
                            action_seed=action_seed,
                        ),
                    }
                )
            successes = int(sum(record["success"] for record in episodes))
            task_records.append(
                {
                    "task_id": task_id,
                    "threshold": float(threshold),
                    "n_episodes": PROBES_PER_TASK,
                    "success_count": successes,
                    "p_hat": successes / PROBES_PER_TASK,
                    "transitions": int(
                        sum(record["transitions"] for record in episodes)
                    ),
                    "episodes": episodes,
                }
            )
    finally:
        probe_env.close()
    probe_transitions = int(sum(record["transitions"] for record in task_records))
    if not 1 <= probe_transitions <= MAX_PROBE_SWEEP_TRANSITIONS:
        raise RuntimeError("probe sweep exceeded its atomic transition bound")
    p_hat = [record["p_hat"] for record in task_records]
    selection = selection_distribution(arm, p_hat)
    training_after = _training_state_fingerprint(space, actor, selection_rng, counters)
    parameters_after = _parameter_fingerprint(actor)
    preserved = (
        training_after == training_before
        and parameters_after == parameters_before
        and int(actor.update_calls) == actor_update_calls_before
        and int(actor.applied_updates) == actor_applied_updates_before
    )
    if not preserved:
        raise RuntimeError("probe sweep mutated forbidden training state")
    return {
        "sweep_ordinal": sweep_ordinal,
        "trigger": trigger,
        "crossed_boundary_student_transition": crossed_boundary,
        "student_transitions": student_transitions,
        "sampled_groups": sampled_groups,
        "optimizer_updates": optimizer_updates,
        "paid_before": paid_before,
        "paid_after": paid_before + probe_transitions,
        "probe_transitions": probe_transitions,
        "task_records": task_records,
        "p_hat": p_hat,
        "selection_after_sweep": selection,
        "training_state_fingerprint_before": training_before,
        "training_state_fingerprint_after": training_after,
        "parameter_fingerprint_before": parameters_before,
        "parameter_fingerprint_after": parameters_after,
        "actor_update_calls_before": actor_update_calls_before,
        "actor_update_calls_after": int(actor.update_calls),
        "actor_applied_updates_before": actor_applied_updates_before,
        "actor_applied_updates_after": int(actor.applied_updates),
        "training_state_preserved": preserved,
    }


def _evaluation_episode_seeds(logical_seed: int, episode: int) -> tuple[int, int]:
    if type(logical_seed) is not int or type(episode) is not int:
        raise ValueError("evaluation coordinates must be primitive integers")
    if logical_seed not in CONFIRMATORY_SEEDS + DEVELOPMENT_SEEDS + QUICK_SEEDS:
        raise ValueError("evaluation logical seed is not registered")
    if not 0 <= episode < CONFIRMATORY_EVAL_N:
        raise ValueError("evaluation episode exceeds collision-free namespace")
    local_seed_index = logical_seed - min(CONFIRMATORY_SEEDS)
    coordinate = local_seed_index * CONFIRMATORY_EVAL_N + episode
    reset = EVALUATION_RESET_NAMESPACE_BASE + coordinate
    action = EVALUATION_ACTION_NAMESPACE_BASE + coordinate
    if not (EVALUATION_RESET_NAMESPACE_BASE <= reset < 2**31 - 1):
        raise RuntimeError("evaluation reset seed exceeds Gymnasium namespace")
    if not (0 <= action < 2**63 - 1):
        raise RuntimeError("evaluation action seed exceeds signed 64-bit range")
    return reset, action


def evaluate_actor_full_horizon(
    actor: TanhCategoricalActor,
    space: AcrobotNeuralSpace,
    selection_rng: np.random.Generator,
    *,
    logical_seed: int,
    eval_n: int,
    counters: dict[str, int],
) -> dict:
    """Evaluate shared full-horizon CRN trajectories without paid accounting."""
    if eval_n < 1:
        raise ValueError("eval_n must be positive")
    before = _training_state_fingerprint(space, actor, selection_rng, counters)
    episode_records = []
    eval_env = gymnasium.make("Acrobot-v1")
    try:
        for episode in range(eval_n):
            reset_seed, action_seed = _evaluation_episode_seeds(logical_seed, episode)
            action_rng = np.random.default_rng(action_seed)
            observation, _ = eval_env.reset(seed=reset_seed)
            max_height = -math.inf
            native_return = 0.0
            native_success = False
            time_to_goal = MAX_EPISODE_STEPS
            entropy_sum = 0.0
            entropy_count = 0
            actions: list[int] = []
            for step_index in range(1, MAX_EPISODE_STEPS + 1):
                probabilities = actor.probabilities(
                    normalize_observation(observation), 0
                )
                entropy_sum += -float(
                    np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
                )
                entropy_count += 1
                action = int(action_rng.choice(3, p=probabilities))
                actions.append(action)
                observation, reward, terminated, truncated, _ = eval_env.step(action)
                max_height = max(max_height, tip_height(observation))
                native_return += float(reward)
                if terminated:
                    native_success = True
                    time_to_goal = step_index
                    break
                if truncated:
                    break
            episode_records.append(
                {
                    "episode": episode,
                    "reset_seed": reset_seed,
                    "action_seed": action_seed,
                    "transitions": len(actions),
                    "native_success": bool(native_success),
                    "native_return": float(native_return),
                    "censored_time_to_goal": float(time_to_goal),
                    "max_height": float(max_height),
                    "policy_entropy_sum": float(entropy_sum),
                    "policy_entropy_count": entropy_count,
                    "action_count": len(actions),
                    "action_sha256": _actions_sha256(actions),
                }
            )
    finally:
        eval_env.close()
    after = _training_state_fingerprint(space, actor, selection_rng, counters)
    if before != after:
        raise RuntimeError("evaluation mutated forbidden training state")
    max_array = np.asarray(
        [record["max_height"] for record in episode_records], dtype=np.float64
    )
    native_successes = np.asarray(
        [record["native_success"] for record in episode_records], dtype=np.float64
    )
    native_returns = np.asarray(
        [record["native_return"] for record in episode_records], dtype=np.float64
    )
    censored_times = np.asarray(
        [record["censored_time_to_goal"] for record in episode_records],
        dtype=np.float64,
    )
    entropy_sum = sum(record["policy_entropy_sum"] for record in episode_records)
    entropy_count = sum(record["policy_entropy_count"] for record in episode_records)
    thresholds = np.asarray(engine.THRESHOLDS, dtype=np.float64)
    midpoints = (thresholds[:-1] + thresholds[1:]) / 2.0
    pass_rates = [float(np.mean(max_array > threshold)) for threshold in thresholds]
    midpoint_rates = [float(np.mean(max_array > threshold)) for threshold in midpoints]
    return {
        "evaluation_shared_full_horizon_trajectories": eval_n,
        "episode_records": episode_records,
        "max_heights": max_array.tolist(),
        "pass_rates": pass_rates,
        "target_uniform_mean_success": float(np.mean(pass_rates)),
        "midpoint_thresholds": midpoints.tolist(),
        "midpoint_pass_rates": midpoint_rates,
        "native_success_rate": float(np.mean(native_successes)),
        "mean_native_return": float(np.mean(native_returns)),
        "mean_censored_time_to_goal": float(np.mean(censored_times)),
        "policy_entropy_sum": float(entropy_sum),
        "policy_entropy_count": int(entropy_count),
        "mean_policy_entropy": float(entropy_sum / entropy_count),
        "actor_parameter_fingerprint": _parameter_fingerprint(actor),
        "training_state_fingerprint_before": before,
        "training_state_fingerprint_after": after,
        "training_state_preserved": True,
    }


def _evaluation_record(
    actor: TanhCategoricalActor,
    space: AcrobotNeuralSpace,
    selection_rng: np.random.Generator,
    *,
    logical_seed: int,
    eval_n: int,
    kind: str,
    paid_transitions: int,
    student_transitions: int,
    probe_transitions: int,
    sampled_groups: int,
    optimizer_updates: int,
    sweep_count: int,
    evaluation_id: int,
    crossed_regular_paid_thresholds: list[int] | None = None,
) -> dict:
    counters = {
        "paid_transitions": paid_transitions,
        "student_transitions": student_transitions,
        "probe_transitions": probe_transitions,
        "sampled_groups": sampled_groups,
        "optimizer_updates": optimizer_updates,
        "sweep_count": sweep_count,
    }
    score = evaluate_actor_full_horizon(
        actor,
        space,
        selection_rng,
        logical_seed=logical_seed,
        eval_n=eval_n,
        counters=counters,
    )
    return {
        "evaluation_id": evaluation_id,
        "kind": kind,
        "paid_transitions": paid_transitions,
        "student_transitions": student_transitions,
        "probe_transitions": probe_transitions,
        "sampled_groups": sampled_groups,
        "optimizer_updates": optimizer_updates,
        "sweep_count": sweep_count,
        "crossed_regular_paid_thresholds": crossed_regular_paid_thresholds or [],
        "copied_from_evaluation_id": None,
        "evaluation_was_executed": True,
        **score,
    }


def _copy_evaluation_to_paid_coordinate(
    source: dict,
    *,
    evaluation_id: int,
    paid_transitions: int,
    probe_transitions: int,
    sweep_count: int,
) -> dict:
    copied = copy.deepcopy(source)
    copied.update(
        {
            "evaluation_id": evaluation_id,
            "kind": "post_probe_copy",
            "paid_transitions": paid_transitions,
            "probe_transitions": probe_transitions,
            "sweep_count": sweep_count,
            "crossed_regular_paid_thresholds": [],
            "copied_from_evaluation_id": source["evaluation_id"],
            "evaluation_was_executed": False,
        }
    )
    return copied


def _practical_maxrl_mass(success_count: int) -> float:
    if not 0 <= success_count <= N_ROLLOUTS:
        raise ValueError("success_count outside registered group")
    if success_count in {0, N_ROLLOUTS}:
        return 0.0
    return 2.0 * (N_ROLLOUTS - success_count) / N_ROLLOUTS


def run_one(
    arm: Arm,
    logical_seed: int,
    *,
    mode: str,
    lock_path: Path = LOCK_PATH,
) -> dict:
    """Own one environment resource and close it at every failure stage."""
    if arm.name not in ARM_BY_NAME or ARM_BY_NAME[arm.name] != arm:
        raise ValueError("arm is not registered")
    seeds, _, _ = _mode_schedule(mode)
    if type(logical_seed) is not int or logical_seed not in seeds:
        raise RuntimeError(f"seed {logical_seed!r} is not registered for {mode}")
    if mode != "quick":
        _require_canonical_lock_path(lock_path)
        _load_and_verify_lock(lock_path)

    master = engine_master_seed(logical_seed)
    domain_record = rng_domain_record(logical_seed)
    actor = TanhCategoricalActor(
        n_tasks=len(engine.THRESHOLDS),
        hidden_size=64,
        learning_rate=LEARNING_RATE,
        seed=master,
        mode="shared",
    )
    space = AcrobotNeuralSpace(
        actor=actor,
        thresholds=engine.THRESHOLDS,
        seed=master + ENVIRONMENT_ADAPTER_SEED_OFFSET,
    )
    try:
        return _run_one_open_space(
            arm,
            logical_seed,
            mode=mode,
            actor=actor,
            space=space,
            domain_record=domain_record,
        )
    finally:
        space.close()


def _run_one_open_space(
    arm: Arm,
    logical_seed: int,
    *,
    mode: str,
    actor: TanhCategoricalActor,
    space: AcrobotNeuralSpace,
    domain_record: dict,
) -> dict:
    """Execute a validated run inside the public wrapper's close boundary."""
    _, paid_budget, eval_n = _mode_schedule(mode)
    selection_rng = np.random.default_rng(domain_record["rng_roots"]["selection"])
    if actor.parameter_count != 640 or actor.active_parameter_count != 640:
        raise RuntimeError("registered task-blind H64 actor must have 640 parameters")

    paid_transitions = 0
    student_transitions = 0
    probe_transitions = 0
    sampled_groups = 0
    optimizer_updates = 0
    live_groups = 0
    dead_groups = 0
    all_pass_groups = 0
    zero_gradient_update_attempts = 0
    task_groups = np.zeros(len(engine.THRESHOLDS), dtype=np.int64)
    task_rollouts = np.zeros(len(engine.THRESHOLDS), dtype=np.int64)
    task_successes = np.zeros(len(engine.THRESHOLDS), dtype=np.int64)
    task_student_transitions = np.zeros(len(engine.THRESHOLDS), dtype=np.int64)
    group_records: list[dict] = []
    sweep_records: list[dict] = []
    evaluation_records: list[dict] = []
    latest_selection: dict | None = (
        None if arm.probes else selection_distribution(arm, None)
    )
    latest_sweep_ordinal: int | None = None
    next_regular_evaluation = REGULAR_EVAL_INTERVAL_PAID
    wall_start = time.perf_counter()

    def append_actual_evaluation(
        kind: str, crossed_regular: list[int] | None = None
    ) -> dict:
        record = _evaluation_record(
            actor,
            space,
            selection_rng,
            logical_seed=logical_seed,
            eval_n=eval_n,
            kind=kind,
            paid_transitions=paid_transitions,
            student_transitions=student_transitions,
            probe_transitions=probe_transitions,
            sampled_groups=sampled_groups,
            optimizer_updates=optimizer_updates,
            sweep_count=len(sweep_records),
            evaluation_id=len(evaluation_records),
            crossed_regular_paid_thresholds=crossed_regular,
        )
        evaluation_records.append(record)
        return record

    try:
        initial_evaluation = append_actual_evaluation("initial")
        if arm.probes:
            sweep = run_probe_sweep(
                arm,
                actor,
                space,
                selection_rng,
                logical_seed=logical_seed,
                sweep_ordinal=1,
                trigger="initial",
                crossed_boundary=None,
                student_transitions=0,
                paid_before=0,
                sampled_groups=0,
                optimizer_updates=0,
            )
            sweep["pre_probe_evaluation_id"] = initial_evaluation["evaluation_id"]
            probe_transitions += sweep["probe_transitions"]
            paid_transitions += sweep["probe_transitions"]
            if paid_transitions != sweep["paid_after"]:
                raise RuntimeError("initial probe paid ledger mismatch")
            sweep_records.append(sweep)
            latest_selection = sweep["selection_after_sweep"]
            latest_sweep_ordinal = sweep["sweep_ordinal"]
            copied = _copy_evaluation_to_paid_coordinate(
                initial_evaluation,
                evaluation_id=len(evaluation_records),
                paid_transitions=paid_transitions,
                probe_transitions=probe_transitions,
                sweep_count=len(sweep_records),
            )
            evaluation_records.append(copied)
            sweep["post_probe_copy_evaluation_id"] = copied["evaluation_id"]

        while paid_transitions < paid_budget:
            if latest_selection is None:
                raise RuntimeError("probed arm lacks its mandatory initial sweep")
            selection_snapshot = copy.deepcopy(latest_selection)
            selection_source_before_group = latest_sweep_ordinal
            probabilities = np.asarray(
                selection_snapshot["probabilities"], dtype=np.float64
            )
            if (
                probabilities.shape != (len(engine.THRESHOLDS),)
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0.0)
                or not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-12)
            ):
                raise RuntimeError("invalid task-selection distribution")
            selection_draw_index = sampled_groups
            selection_rng_before = _rng_state_fingerprint(selection_rng)
            selection_uniform = float(selection_rng.random())
            task_id = int(
                min(
                    np.searchsorted(
                        np.cumsum(probabilities), selection_uniform, side="right"
                    ),
                    len(engine.THRESHOLDS) - 1,
                )
            )
            selection_rng_after = _rng_state_fingerprint(selection_rng)
            paid_before_group = paid_transitions
            student_before = student_transitions
            actor_parameter_before_group = _parameter_fingerprint(actor)
            actor_action_rng_before_student = _rng_state_fingerprint(actor.action_rng)
            environment_reset_rng_before_student = _rng_state_fingerprint(space.rng)
            group = space.rollout_group(task_id, N_ROLLOUTS)
            actor_action_rng_after_student = _rng_state_fingerprint(actor.action_rng)
            environment_reset_rng_after_student = _rng_state_fingerprint(space.rng)
            rewards = np.asarray(group.rewards, dtype=np.float64)
            if rewards.shape != (N_ROLLOUTS,) or not np.all(
                (rewards == 0.0) | (rewards == 1.0)
            ):
                raise RuntimeError("student group rewards must be a binary N-vector")
            group_student_transitions = engine._group_transitions(group)
            if not 1 <= group_student_transitions <= MAX_STUDENT_GROUP_TRANSITIONS:
                raise RuntimeError("student group transition count is invalid")
            student_transitions += group_student_transitions
            paid_transitions += group_student_transitions
            sampled_groups += 1
            student_after = student_transitions
            paid_after_student_group = paid_transitions
            task_groups[task_id] += 1
            task_rollouts[task_id] += N_ROLLOUTS
            task_successes[task_id] += int(rewards.sum())
            task_student_transitions[task_id] += group_student_transitions
            student_rollout_records = []
            for rollout_index, (trajectory, info, reward) in enumerate(
                zip(group.trajectories, group.infos, rewards)
            ):
                actions = _trajectory_actions(trajectory)
                reset_draw_index = (sampled_groups - 1) * N_ROLLOUTS + rollout_index
                max_height = float(info["max_height"])
                success = bool(reward == 1.0)
                if success != bool(max_height > engine.THRESHOLDS[task_id]):
                    raise RuntimeError("student threshold-success predicate mismatch")
                student_rollout_records.append(
                    {
                        "rollout": rollout_index,
                        "student_reset_draw_index": reset_draw_index,
                        "reset_seed": int(info["reset_seed"]),
                        "success": success,
                        "transitions": int(info["n_steps"]),
                        "max_height": max_height,
                        "action_count": len(actions),
                        "action_sha256": _actions_sha256(actions),
                    }
                )

            boundaries = crossed_refresh_boundaries(student_before, student_after)
            sweep_ordinals_for_group = []
            for boundary in boundaries if arm.probes else []:
                pre_probe = append_actual_evaluation("pre_probe")
                ordinal = len(sweep_records) + 1
                sweep = run_probe_sweep(
                    arm,
                    actor,
                    space,
                    selection_rng,
                    logical_seed=logical_seed,
                    sweep_ordinal=ordinal,
                    trigger="refresh",
                    crossed_boundary=boundary,
                    student_transitions=student_transitions,
                    paid_before=paid_transitions,
                    sampled_groups=sampled_groups,
                    optimizer_updates=optimizer_updates,
                )
                sweep["pre_probe_evaluation_id"] = pre_probe["evaluation_id"]
                probe_transitions += sweep["probe_transitions"]
                paid_transitions += sweep["probe_transitions"]
                if paid_transitions != sweep["paid_after"]:
                    raise RuntimeError("refresh probe paid ledger mismatch")
                sweep_records.append(sweep)
                latest_selection = sweep["selection_after_sweep"]
                latest_sweep_ordinal = sweep["sweep_ordinal"]
                sweep_ordinals_for_group.append(ordinal)
                copied = _copy_evaluation_to_paid_coordinate(
                    pre_probe,
                    evaluation_id=len(evaluation_records),
                    paid_transitions=paid_transitions,
                    probe_transitions=probe_transitions,
                    sweep_count=len(sweep_records),
                )
                evaluation_records.append(copied)
                sweep["post_probe_copy_evaluation_id"] = copied["evaluation_id"]

            k = int(rewards.sum())
            regime = "dead" if k == 0 else "all_pass" if k == N_ROLLOUTS else "mixed"
            weights = engine._weights(rewards, "maxrl")
            update_parameter_before = _parameter_fingerprint(actor)
            update_calls_before = int(actor.update_calls)
            applied_updates_before = int(actor.applied_updates)
            update_diagnostics = None
            update_requested = regime == "mixed"
            update_applied = False
            if regime == "mixed":
                live_groups += 1
                diagnostics = engine._update(
                    actor, task_id, group.trajectories, weights
                )
                applied = bool(diagnostics.pop("applied"))
                update_applied = applied
                update_diagnostics = diagnostics
                if applied:
                    optimizer_updates += 1
                else:
                    zero_gradient_update_attempts += 1
            elif regime == "dead":
                dead_groups += 1
            else:
                all_pass_groups += 1
            update_record = {
                "source": "practical_dropped_group_maxrl",
                "eligible": regime == "mixed",
                "requested": update_requested,
                "weights": np.asarray(weights, dtype=np.float64).tolist(),
                "weight_sum": float(np.sum(weights)),
                "weight_l1": float(np.sum(np.abs(weights))),
                "parameter_fingerprint_before": update_parameter_before,
                "parameter_fingerprint_after": _parameter_fingerprint(actor),
                "actor_update_calls_before": update_calls_before,
                "actor_update_calls_after": int(actor.update_calls),
                "actor_applied_updates_before": applied_updates_before,
                "actor_applied_updates_after": int(actor.applied_updates),
                "applied": update_applied,
                "diagnostics": update_diagnostics,
            }

            crossed_regular = []
            while next_regular_evaluation <= paid_transitions:
                crossed_regular.append(next_regular_evaluation)
                next_regular_evaluation += REGULAR_EVAL_INTERVAL_PAID
            if crossed_regular:
                append_actual_evaluation("regular_after_update", crossed_regular)

            group_records.append(
                {
                    "group": sampled_groups,
                    "task_id": task_id,
                    "threshold": float(engine.THRESHOLDS[task_id]),
                    "selection_source_sweep_ordinal": selection_source_before_group,
                    "selection_draw_index": selection_draw_index,
                    "selection_uniform": selection_uniform,
                    "selection_rng_fingerprint_before": selection_rng_before,
                    "selection_rng_fingerprint_after": selection_rng_after,
                    "selection_probabilities_before_group": probabilities.tolist(),
                    "selected_task_probability": float(probabilities[task_id]),
                    "p_hat_used_before_group": selection_snapshot["p_hat"],
                    "utility_used_before_group": selection_snapshot["utility"],
                    "logits_used_before_group": selection_snapshot["logits"],
                    "paid_before_group": paid_before_group,
                    "paid_after_student_group": paid_after_student_group,
                    "paid_after_required_sweeps": paid_transitions,
                    "student_transition_start": student_before,
                    "student_transition_end": student_after,
                    "student_transitions": group_student_transitions,
                    "student_success_flags": [bool(value) for value in rewards],
                    "student_rollout_records": student_rollout_records,
                    "required_crossed_boundaries": boundaries if arm.probes else [],
                    "required_sweep_ordinals": sweep_ordinals_for_group,
                    "success_count": k,
                    "regime": regime,
                    "realized_practical_maxrl_abs_coefficient_mass": (
                        _practical_maxrl_mass(k)
                    ),
                    "optimizer_updates_after_group": optimizer_updates,
                    "update": update_record,
                    "actor_parameter_fingerprint_before_group": (
                        actor_parameter_before_group
                    ),
                    "actor_parameter_fingerprint_after_group": (
                        _parameter_fingerprint(actor)
                    ),
                    "actor_action_rng_fingerprint_before_student": (
                        actor_action_rng_before_student
                    ),
                    "actor_action_rng_fingerprint_after_student": (
                        actor_action_rng_after_student
                    ),
                    "environment_reset_rng_fingerprint_before_student": (
                        environment_reset_rng_before_student
                    ),
                    "environment_reset_rng_fingerprint_after_student": (
                        environment_reset_rng_after_student
                    ),
                }
            )

        append_actual_evaluation("terminal")
    finally:
        # The public run_one wrapper owns the immediate try/finally close.
        pass
    wall_seconds = time.perf_counter() - wall_start

    if paid_transitions != student_transitions + probe_transitions:
        raise RuntimeError("paid ledger does not equal student plus probe")
    if int(task_groups.sum()) != sampled_groups:
        raise RuntimeError("task-group ledger mismatch")
    if int(task_rollouts.sum()) != sampled_groups * N_ROLLOUTS:
        raise RuntimeError("student rollout ledger mismatch")
    if int(task_student_transitions.sum()) != student_transitions:
        raise RuntimeError("task student-transition ledger mismatch")
    if sampled_groups != live_groups + dead_groups + all_pass_groups:
        raise RuntimeError("student regime ledger mismatch")
    expected_sweeps = (
        1 + student_transitions // REFRESH_STUDENT_TRANSITIONS if arm.probes else 0
    )
    if len(sweep_records) != expected_sweeps:
        raise RuntimeError("probe sweep cadence ledger mismatch")
    if not arm.probes and probe_transitions != 0:
        raise RuntimeError("ordinary uniform accumulated probe transitions")
    paid_axis = [record["paid_transitions"] for record in evaluation_records]
    if paid_axis != sorted(paid_axis):
        raise RuntimeError("evaluation paid coordinate is not monotone")
    if paid_transitions < paid_budget:
        raise RuntimeError("run ended before the nominal paid budget")
    if paid_transitions > (
        paid_budget + MAX_STUDENT_GROUP_TRANSITIONS + 2 * MAX_PROBE_SWEEP_TRANSITIONS
    ):
        raise RuntimeError("terminal atomic overshoot exceeds the mechanical bound")
    if not all(record["training_state_preserved"] for record in evaluation_records):
        raise RuntimeError("evaluation state-preservation ledger failed")

    selection_matrix = np.asarray(
        [record["selection_probabilities_before_group"] for record in group_records],
        dtype=np.float64,
    )
    realized_task_fraction = task_groups.astype(np.float64) / sampled_groups
    selection_diagnostics = {
        "mean_selection_entropy": float(
            np.mean(
                -np.sum(
                    selection_matrix * np.log(np.maximum(selection_matrix, 1e-300)),
                    axis=1,
                )
            )
        ),
        "mean_selection_tv_from_uniform": float(
            np.mean(0.5 * np.abs(selection_matrix - 1.0 / 8.0).sum(axis=1))
        ),
        "mean_max_task_probability": float(np.mean(np.max(selection_matrix, axis=1))),
        "mean_assigned_probability_per_task": np.mean(
            selection_matrix, axis=0
        ).tolist(),
        "realized_task_fraction": realized_task_fraction.tolist(),
        "realized_task_tv_from_uniform": float(
            0.5 * np.abs(realized_task_fraction - 1.0 / 8.0).sum()
        ),
        "student_fraction_of_paid": student_transitions / paid_transitions,
        "probe_fraction_of_paid": probe_transitions / paid_transitions,
        "paid_budget_overshoot": paid_transitions - paid_budget,
    }

    return {
        **domain_record,
        "seed": logical_seed,
        "numeric_valid": True,
        "paid_budget_nominal": paid_budget,
        "paid_transitions": paid_transitions,
        "paid_budget_overshoot": paid_transitions - paid_budget,
        "student_transitions": student_transitions,
        "probe_transitions": probe_transitions,
        "probe_fraction_of_paid": probe_transitions / paid_transitions,
        "sampled_groups": sampled_groups,
        "student_rollouts": sampled_groups * N_ROLLOUTS,
        "probe_episodes": len(sweep_records) * len(engine.THRESHOLDS) * PROBES_PER_TASK,
        "probe_sweeps": len(sweep_records),
        "optimizer_updates": optimizer_updates,
        "live_groups": live_groups,
        "dead_groups": dead_groups,
        "all_pass_groups": all_pass_groups,
        "zero_gradient_update_attempts": zero_gradient_update_attempts,
        "task_groups": task_groups.tolist(),
        "task_rollouts": task_rollouts.tolist(),
        "task_successes": task_successes.tolist(),
        "task_student_transitions": task_student_transitions.tolist(),
        "total_parameters": actor.parameter_count,
        "active_parameters_per_task": actor.active_parameter_count,
        "final_parameter_fingerprint": _parameter_fingerprint(actor),
        "initial_parameter_fingerprint": evaluation_records[0][
            "actor_parameter_fingerprint"
        ],
        "wall_seconds": wall_seconds,
        "paid_transitions_per_wall_second": (
            paid_transitions / wall_seconds if wall_seconds > 0.0 else None
        ),
        "group_records": group_records,
        "probe_sweep_records": sweep_records,
        "evaluation_records": evaluation_records,
        "selection_diagnostics": selection_diagnostics,
        "accounting_valid": True,
        "evaluation_rng_preserved": True,
        "probe_training_state_preserved": all(
            record["training_state_preserved"] for record in sweep_records
        ),
    }


def _protocol(mode: str, *, development_gate: dict | None = None) -> dict:
    seeds, paid_budget, eval_n = _mode_schedule(mode)
    return {
        "study": "acrobot_procurl_selection_semantics",
        "mode": mode,
        "status": (
            "confirmatory"
            if mode == "confirmatory"
            else "engineering_only_no_scientific_inference"
            if mode == "quick"
            else "development_only"
        ),
        "protocol_document": str(PROTOCOL_PATH.relative_to(PROJECT_ROOT)),
        "primary_source_provenance": str(PROVENANCE_PATH.relative_to(PROJECT_ROOT)),
        "upstream_procurl_commit": UPSTREAM_PROCURL_COMMIT,
        "arm_names": [arm.name for arm in ARMS],
        "paired_logical_seeds": list(seeds),
        "logical_to_engine_master_seed": {
            str(seed): engine_master_seed(seed) for seed in seeds
        },
        "paid_budget_nominal": paid_budget,
        "complete_atomic_overshoot_retained": True,
        "paid_definition": "student transitions plus probe transitions; evaluation excluded",
        "student_group_size": N_ROLLOUTS,
        "student_estimator": "practical N=16 dropped-group MaxRL",
        "architecture": "task-blind shared H64, 640 parameters",
        "learning_rate": LEARNING_RATE,
        "optimizer": "plain SGD ascent",
        "hindsight": False,
        "thresholds": list(engine.THRESHOLDS),
        "probe": {
            "episodes_per_task_per_sweep": PROBES_PER_TASK,
            "tasks": len(engine.THRESHOLDS),
            "initial_sweep": True,
            "refresh_clock": "student transitions",
            "refresh_every": REFRESH_STUDENT_TRANSITIONS,
            "boundary_mapping": (
                "after complete student group and before its update, run one complete "
                "sweep per crossed boundary"
            ),
            "paid": True,
            "actor_and_training_rng_nonmutating": True,
            "latest_complete_sweep_replaces_previous_estimates": True,
        },
        "selection_rules": {
            "procurl_env_b20_f5120": "softmax(20*p_hat*(1-p_hat))",
            "probe_sham_uniform_f5120": "exact 1/8 after discarded probes",
            "ordinary_uniform": "exact 1/8 with no probes",
            "u16_probe_range_matched_f5120": (
                f"softmax({U16_BETA_CONTINUOUS_RANGE_MATCHED}*"
                "(1-(1-p_hat)^16-p_hat)); continuous-domain max logit 5; "
                f"0.05-lattice max logit {U16_LATTICE_MAX_LOGIT}"
            ),
        },
        "evaluation": {
            "shared_full_horizon_trajectories": eval_n,
            "common_episode_reset_and_action_streams": True,
            "primary_score": "unweighted mean success over eight thresholds",
            "raw_max_heights_retained": True,
            "seven_midpoint_thresholds_retained": True,
            "regular_interval_paid": REGULAR_EVAL_INTERVAL_PAID,
            "initial_terminal_and_around_every_sweep": True,
            "post_probe_score_is_pre_probe_copy": True,
        },
        "primary": {
            "contrast": ("u16_probe_range_matched_f5120 minus procurl_env_b20_f5120"),
            "metric": "fixed-nominal-paid-budget target-uniform mean-success AUC",
            "test": "two-sided paired t-test",
            "alpha": 0.05,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": 31_000,
            "bootstrap_quantiles": [0.025, 0.975],
            "bootstrap_quantile_method": "linear",
            "sesoi": PRIMARY_SESOI,
            "decision": "mean >= +0.02 and p <= 0.05",
            "robustness": (
                f"deterministic Monte Carlo paired sign flip, "
                f"{SIGN_FLIP_MONTE_CARLO_DRAWS} draws, seed 31001, plus-one correction"
            ),
        },
        "secondary_holm_family": [
            "procurl-minus-sham",
            "u16-minus-sham",
            "procurl-minus-ordinary",
            "u16-minus-ordinary",
            "sham-minus-ordinary",
        ],
        "secondary_statistics": {
            "paired_t_alpha": 0.05,
            "holm_familywise_alpha": 0.05,
            "bootstrap_seeds_in_family_order": list(range(31_100, 31_105)),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_quantiles": [0.025, 0.975],
            "bootstrap_quantile_method": "linear",
        },
        "auc_convention": {
            "integration": "trapezoid on nondecreasing recorded coordinates",
            "duplicate_coordinates": (
                "retain in record order; zero width contributes zero area; "
                "last duplicate governs the next positive-width segment"
            ),
            "nominal_cutoff": "linear interpolation within the crossing segment",
            "normalization": "divide by cutoff minus zero",
        },
        "development_gate": development_gate,
        "rng_domain_contract": {
            "engine_master_base": ENGINE_MASTER_BASE,
            "engine_master_stride": ENGINE_MASTER_STRIDE,
            "rng_domain_offsets": dict(RNG_DOMAIN_OFFSETS),
            "environment_adapter_seed_offset": ENVIRONMENT_ADAPTER_SEED_OFFSET,
            "pairing": "all four arms share roots within a logical seed",
        },
        "no_public_preexecution_commit": True,
        "claim_scope": (
            "source-faithful ProCuRL selection semantics attached to this learner; "
            "not full PPO ProCuRL reproduction"
        ),
        "raw_only": True,
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _provenance(mode: str, lock_path: Path) -> dict:
    if mode == "quick":
        source_hashes = _source_hashes(require_all=False)
        lock_hash = None
    else:
        lock, lock_hash = _load_and_verify_lock(lock_path)
        source_hashes = lock["source_sha256"]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime(),
        "source_lock_relative_path": str(
            lock_path.resolve().relative_to(PROJECT_ROOT.resolve())
        ),
        "source_lock_sha256": lock_hash,
        "source_lock_enforced": mode != "quick",
        "source_sha256": source_hashes,
        "git_commit": _git("rev-parse", "HEAD") or None,
        "git_status_porcelain": _git("status", "--porcelain").splitlines(),
        "seed_collision_audit": seed_collision_audit(),
        "upstream_procurl_commit": UPSTREAM_PROCURL_COMMIT,
        "upstream_code_copied": False,
        "public_preexecution_registration": False,
    }


def _write_json(path: Path, payload: dict, *, overwrite: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _project_relative_file(path: Path, label: str) -> tuple[Path, str]:
    resolved = path.resolve()
    try:
        relative = str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError as error:
        raise RuntimeError(f"{label} must be inside the project") from error
    if not resolved.is_file():
        raise RuntimeError(f"{label} is missing: {resolved}")
    return resolved, relative


def _load_development_gate(path: Path, lock: dict, lock_hash: str) -> dict:
    gate_path, gate_relative = _project_relative_file(path, "development gate artifact")
    gate = _load_strict_json(gate_path, "development gate")
    if set(gate) != {
        "schema",
        "mode",
        "all_gates_passed",
        "source_lock_sha256",
        "source_lock_verification",
        "gates",
        "diagnostics",
        "gate_policy",
        "raw_artifact_relative_path",
        "raw_artifact_sha256",
    }:
        raise RuntimeError("development gate top-level schema is not closed")
    if gate.get("schema") != GATE_SCHEMA or gate.get("mode") != "development":
        raise RuntimeError("development gate schema/mode mismatch")
    if gate.get("all_gates_passed") is not True:
        raise RuntimeError("development gate did not pass")
    if gate.get("source_lock_sha256") != lock_hash:
        raise RuntimeError("development gate used a different source lock")
    if tuple(gate.get("gates", {})) != DEVELOPMENT_GATE_NAMES:
        raise RuntimeError("development gate key set/order mismatch")
    if any(value is not True for value in gate["gates"].values()):
        raise RuntimeError("development gate contains a failed required check")
    if gate.get("gate_policy") != DEVELOPMENT_GATE_POLICY:
        raise RuntimeError("development gate policy mismatch")
    raw_relative = gate.get("raw_artifact_relative_path")
    if not isinstance(raw_relative, str):
        raise TypeError("development gate lacks a project-relative raw artifact")
    raw_path, observed_relative = _project_relative_file(
        PROJECT_ROOT / raw_relative, "development raw artifact"
    )
    if raw_relative != observed_relative or gate.get("raw_artifact_sha256") != _sha256(
        raw_path
    ):
        raise RuntimeError("development gate raw binding mismatch")

    from frontier_rl.examples import analyze_acrobot_procurl_selection as analyzer

    raw = _load_strict_json(raw_path, "development raw artifact")
    source_verification = analyzer.verify_source_lock(raw, lock, LOCK_PATH)
    validated = analyzer.validate_raw_artifact(raw)
    recomputed = analyzer.development_gates(
        validated,
        source_verification,
        raw_artifact_relative_path=raw_relative,
        raw_artifact_sha256=_sha256(raw_path),
    )
    if gate != recomputed:
        raise RuntimeError("development gate does not independently recompute exactly")
    return {
        "relative_path": gate_relative,
        "sha256": _sha256(gate_path),
        "raw_artifact_relative_path": raw_relative,
        "raw_artifact_sha256": _sha256(raw_path),
        "all_gates_passed": True,
    }


def _case_summary(runs: list[dict]) -> dict:
    valid = [run for run in runs if run.get("numeric_valid")]
    return {
        "n_attempted": len(runs),
        "n_valid": len(valid),
        "n_failed": len(runs) - len(valid),
        "ledger_means_descriptive_only": {
            key: (float(np.mean([run[key] for run in valid])) if valid else None)
            for key in (
                "paid_transitions",
                "student_transitions",
                "probe_transitions",
                "probe_sweeps",
                "optimizer_updates",
            )
        },
    }


def run_study(
    *,
    mode: str,
    output: Path,
    lock_path: Path = LOCK_PATH,
    development_gate: Path | None = None,
    overwrite: bool = False,
) -> dict:
    runtime = _runtime()
    if _runtime_versions(runtime) != PINNED_RUNTIME_VERSIONS:
        raise RuntimeError(
            "study execution requires the exact pinned runtime: "
            f"expected={PINNED_RUNTIME_VERSIONS!r}, observed={runtime!r}"
        )
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    if mode != "quick":
        _require_canonical_lock_path(lock_path)
    gate_binding = None
    if mode == "confirmatory":
        lock, lock_hash = _load_and_verify_lock(lock_path)
        if development_gate is None:
            raise RuntimeError("confirmation requires a fresh passing development gate")
        gate_binding = _load_development_gate(development_gate, lock, lock_hash)
    elif development_gate is not None:
        raise ValueError("--development-gate is confirmatory-only")

    seeds, _, _ = _mode_schedule(mode)
    artifact = {
        "schema": RAW_SCHEMA,
        "artifact_state": "in_progress",
        "provenance": _provenance(mode, lock_path),
        "protocol": _protocol(mode, development_gate=gate_binding),
        "run_failures": [],
        "cases": {},
    }
    claimed = False
    for arm in ARMS:
        case = {
            "config": asdict(arm),
            "summary": _case_summary([]),
            "runs": [],
        }
        artifact["cases"][arm.name] = case
        for seed in seeds:
            try:
                run = run_one(arm, seed, mode=mode, lock_path=lock_path)
            except Exception as error:  # noqa: BLE001 - failures are raw evidence
                run = {
                    **rng_domain_record(seed),
                    "seed": seed,
                    "numeric_valid": False,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                }
                artifact["run_failures"].append({"arm": arm.name, **run})
            case["runs"].append(run)
            case["summary"] = _case_summary(case["runs"])
            _write_json(output, artifact, overwrite=overwrite or claimed)
            claimed = True
        print(
            f"{arm.name}: valid={case['summary']['n_valid']}/"
            f"{case['summary']['n_attempted']}",
            flush=True,
        )
    if mode != "quick":
        _load_and_verify_lock(lock_path)
    complete = not artifact["run_failures"] and all(
        [run["seed"] for run in case["runs"]] == list(seeds)
        and all(run.get("numeric_valid") for run in case["runs"])
        for case in artifact["cases"].values()
    )
    artifact["artifact_state"] = (
        "complete" if complete else "complete_with_invalid_runs"
    )
    _write_json(output, artifact, overwrite=True)
    return artifact


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--development", action="store_true")
    modes.add_argument("--quick", action="store_true")
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--development-gate", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    mode = (
        "quick" if args.quick else "development" if args.development else "confirmatory"
    )
    if args.output is None:
        args.output = HERE / f"acrobot_procurl_selection_{mode}.json"
    try:
        artifact = run_study(
            mode=mode,
            output=args.output,
            lock_path=args.lock,
            development_gate=args.development_gate,
            overwrite=args.overwrite,
        )
    except (ValueError, TypeError, RuntimeError, FileExistsError) as error:
        parser.error(str(error))
    print(f"wrote {args.output.resolve()}")
    if artifact["artifact_state"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
