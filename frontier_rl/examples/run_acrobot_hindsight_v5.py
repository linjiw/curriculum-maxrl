"""Fail-closed runner for the Acrobot hindsight V5 study.

V5 is a clean successor to the immutable V4A calibration.  V4A is retained as
failed calibration evidence and is never pooled with V5.  V5A uses fresh
development seeds in the full learning-rate by hindsight-scale grid and makes
only learning-outcome-field-blind, natural-mechanics feasibility decisions.  V5B is created
only after both the runner and the independent analyzer authorize it.

The numerical training implementation is deliberately reused, byte-for-byte,
from :mod:`run_acrobot_hindsight_v4`, which instruments the frozen neural
Acrobot loop in :mod:`run_acrobot_neural`.  Every imported implementation file
is named and hashed in the V5 source lock and artifact provenance.  V5 uses
new schemas throughout; no V4 artifact is relabeled as V5.

Run all entry points as modules, for example::

    python -m frontier_rl.examples.run_acrobot_hindsight_v5 seal-a --output LOCK.json
    python -m frontier_rl.examples.run_acrobot_hindsight_v5 stage-a \
        --lock LOCK.json --output A.json
    python -m frontier_rl.examples.analyze_acrobot_hindsight_v5 A.json \
        --lock LOCK.json --output A_VERIFY.json

Interrupted artifacts are resumed only at complete-run boundaries.  A saved
run is immutable: resume verifies the exact source/runtime lock, condition
configuration, case order, and seed-prefix order before doing any new work.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, replace
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import gymnasium
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontier_rl.examples import run_acrobot_hindsight_v4 as engine
from frontier_rl.adapters import acrobot_neural as acrobot_adapter


locked = engine.locked

SCHEMA_A = "curriculum-maxrl/acrobot-hindsight-v5a-artifact/v1"
SCHEMA_B = "curriculum-maxrl/acrobot-hindsight-v5b-artifact/v1"
LOCK_SCHEMA_A = "curriculum-maxrl/acrobot-hindsight-v5a-source-lock/v1"
LOCK_SCHEMA_B = "curriculum-maxrl/acrobot-hindsight-v5b-source-lock/v1"
AMENDMENT_SCHEMA_B = "curriculum-maxrl/acrobot-hindsight-v5b-amendment/v1"
VERIFICATION_SCHEMA_A = "curriculum-maxrl/acrobot-hindsight-v5a-verification/v1"

PROTOCOL_PATH = Path(__file__).with_name("ACROBOT_HINDSIGHT_PROTOCOL_V5.md")
ANALYZER_MODULE = "frontier_rl.examples.analyze_acrobot_hindsight_v5"
DEFAULT_LOCK_A = Path(__file__).with_name("ACROBOT_HINDSIGHT_V5A_LOCK.json")
DEFAULT_ARTIFACT_A = Path(__file__).with_name("acrobot_hindsight_v5a_feasibility.json")
DEFAULT_VERIFICATION_A = Path(__file__).with_name(
    "acrobot_hindsight_v5a_verification.json"
)
DEFAULT_AMENDMENT_B = Path(__file__).with_name("ACROBOT_HINDSIGHT_V5B_AMENDMENT.json")
DEFAULT_LOCK_B = Path(__file__).with_name("ACROBOT_HINDSIGHT_V5B_LOCK.json")
DEFAULT_ARTIFACT_B = Path(__file__).with_name("acrobot_hindsight_v5b_factorial.json")
DEFAULT_VERIFICATION_B = Path(__file__).with_name(
    "acrobot_hindsight_v5b_verification.json"
)

BASE_LEARNING_RATE = 3e-4
LR_MULTIPLIERS = (0.5, 1.0, 2.0)
HINDSIGHT_SCALES = (0.0, 1.0, 2.0)
STAGE_A_SEEDS = tuple(range(15_000, 15_003))
STAGE_B_SEEDS = tuple(range(16_000, 16_020))
TARGET_UPDATES_A = 400
FALLBACK_UPDATES_A = 250
TRANSITION_GROUP_START_CAP = 4_000_000
MAX_COMPLETE_GROUP_OVERSHOOT = locked.N_ROLLOUTS * 500
EVAL_INTERVAL_UPDATES = 50
EVAL_EPISODES_PER_TASK = 32
TV_WARMUP_TRANSITIONS = 200_000
MIN_NATURAL_RELABELS_PER_CELL = 1
MAX_PROJECTED_HOURS_180 = 18.0

# Already-executed exploratory smoke seed; intentionally outside fresh V5 blocks.
SHADOW_SEED = 100
SHADOW_UPDATE_BUDGET = 3
SHADOW_TRANSITION_CAP = 80_000
SHADOW_EVAL_EPISODES = 4

SOURCE_RELATIVE_PATHS = (
    "frontier_rl/examples/run_acrobot_hindsight_v5.py",
    "frontier_rl/examples/analyze_acrobot_hindsight_v5.py",
    "frontier_rl/examples/test_run_acrobot_hindsight_v5.py",
    "frontier_rl/examples/test_analyze_acrobot_hindsight_v5.py",
    "frontier_rl/examples/ACROBOT_HINDSIGHT_PROTOCOL_V5.md",
    # Explicitly locked reused implementation provenance.
    "frontier_rl/examples/run_acrobot_hindsight_v4.py",
    "frontier_rl/examples/run_acrobot_neural.py",
    "frontier_rl/adapters/acrobot_neural.py",
    "frontier_rl/teacher.py",
    "frontier_rl/estimators.py",
    "frontier_rl/interfaces.py",
    # Frozen implementation tests are part of the executable evidence chain.
    "frontier_rl/examples/test_acrobot_neural.py",
    "frontier_rl/examples/test_run_acrobot_neural.py",
    "frontier_rl/examples/test_run_acrobot_hindsight_v4.py",
    # Package initializers and their eager trainer import execute on import.
    "frontier_rl/__init__.py",
    "frontier_rl/trainer.py",
    "frontier_rl/adapters/__init__.py",
)

PRIOR_ACROBOT_SEED_BLOCKS = {
    "v1_core": tuple(range(0, 12)),
    "v2_core": tuple(range(0, 20)),
    "v1_v2_scale": tuple(range(100, 110)),
    "v1_pilot": tuple(range(10_000, 10_003)),
    "v2_development": tuple(range(11_000, 11_003)),
    "v3_confirmatory": tuple(range(12_000, 12_020)),
    "v4a_feasibility": tuple(range(13_000, 13_003)),
    "v4b_registered": tuple(range(14_000, 14_010)),
}

# These fields may be validated after a launch decision, but they are never an
# input to ``compute_stage_a_gates``.  Group regimes, natural relabel events,
# source counters, teacher TV, and resource coordinates are feasibility
# mechanics rather than learning-effect estimands.
OUTCOME_FIELD_NAMES = frozenset(
    {
        "pass_rate_curve",
        "mean_pass_curve",
        "hardest_pass_curve",
        "native_success_rate_curve",
        "mean_native_return_curve",
        "mean_censored_time_to_goal_curve",
        "mean_policy_entropy_curve",
        "auc_mean_pass_by_transitions",
        "auc_mean_pass_by_optimizer_updates",
        "initial_mean_pass",
        "final_mean_pass",
        "final_hardest_pass",
        "final_native_success_rate",
        "final_mean_native_return",
        "final_mean_censored_time_to_goal",
    }
)


def _float_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _case_name(multiplier: float, scale: float) -> str:
    return f"lr_mult_{_float_label(multiplier)}_hs_{_float_label(scale)}"


CASES = tuple(
    _case_name(multiplier, scale)
    for multiplier in LR_MULTIPLIERS
    for scale in HINDSIGHT_SCALES
)
STAGE_A_CASES = CASES
STAGE_B_CASES = CASES


def _stage_a_schedule() -> dict:
    return {
        "v5_stage": "stage_a_natural_feasibility",
        "paired_seeds": list(STAGE_A_SEEDS),
        "base_learning_rate": BASE_LEARNING_RATE,
        "learning_rate_multipliers": list(LR_MULTIPLIERS),
        "hindsight_scales": list(HINDSIGHT_SCALES),
        "condition_names": list(STAGE_A_CASES),
        "optimizer_update_target": TARGET_UPDATES_A,
        "single_allowed_fallback_update_target": FALLBACK_UPDATES_A,
        "transition_group_start_cap": TRANSITION_GROUP_START_CAP,
        "maximum_complete_group_overshoot": MAX_COMPLETE_GROUP_OVERSHOOT,
        "eval_interval_optimizer_updates": EVAL_INTERVAL_UPDATES,
        "eval_episodes_per_task": EVAL_EPISODES_PER_TASK,
        "fresh_budget_selection_population": "all 27 V5A runs",
        "shadow_test": {
            "seed": SHADOW_SEED,
            "optimizer_update_budget": SHADOW_UPDATE_BUDGET,
            "transition_group_start_cap": SHADOW_TRANSITION_CAP,
            "eval_episodes_per_task": SHADOW_EVAL_EPISODES,
        },
    }


def _stage_b_schedule(selected_update_budget: int) -> dict:
    if selected_update_budget not in (FALLBACK_UPDATES_A, TARGET_UPDATES_A):
        raise ValueError("V5B update budget must be exactly 250 or 400")
    return {
        "v5_stage": "stage_b_confirmatory_factorial",
        "paired_seeds": list(STAGE_B_SEEDS),
        "base_learning_rate": BASE_LEARNING_RATE,
        "learning_rate_multipliers": list(LR_MULTIPLIERS),
        "hindsight_scales": list(HINDSIGHT_SCALES),
        "condition_names": list(STAGE_B_CASES),
        "optimizer_update_target": int(selected_update_budget),
        "transition_group_start_cap": TRANSITION_GROUP_START_CAP,
        "maximum_complete_group_overshoot": MAX_COMPLETE_GROUP_OVERSHOOT,
        "eval_interval_optimizer_updates": EVAL_INTERVAL_UPDATES,
        "eval_episodes_per_task": EVAL_EPISODES_PER_TASK,
        "run_count": 180,
    }


def _select_update_budget(update_counts: Sequence[int], expected_runs: int = 27) -> int | None:
    counts = [int(value) for value in update_counts]
    if len(counts) == expected_runs and all(value >= TARGET_UPDATES_A for value in counts):
        return TARGET_UPDATES_A
    if len(counts) == expected_runs and all(value >= FALLBACK_UPDATES_A for value in counts):
        return FALLBACK_UPDATES_A
    return None


def _teacher_tv_mean(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        return None
    exact_mean = sum(Decimal(str(float(value))) for value in array) / len(array)
    return float(exact_mean)


def _teacher_tv_pass(values: Sequence[float]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        return False
    exact_mean = sum(Decimal(str(float(value))) for value in array) / len(array)
    return bool(exact_mean > Decimal("0.05"))


def _projected_hours_180(slowest_prefix_wall_seconds: float) -> float:
    value = float(slowest_prefix_wall_seconds)
    if not math.isfinite(value) or value <= 0.0:
        return math.inf
    return 180.0 * value / 3600.0


def _runtime() -> dict:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
    }


def _engine_contract() -> dict:
    """Assert the semantic constants assumed by the locked V4 instrumentation."""

    actor = locked.TanhCategoricalActor(
        n_tasks=8,
        hidden_size=64,
        learning_rate=BASE_LEARNING_RATE,
        seed=9_907,
        mode="shared",
    )
    contract = {
        "instrumentation_checkpoints": [
            int(engine.FALLBACK_UPDATES_A),
            int(engine.TARGET_UPDATES_A),
        ],
        "instrumentation_eval_interval_updates": int(engine.EVAL_INTERVAL_UPDATES),
        "instrumentation_transition_group_start_cap": int(
            engine.TRANSITION_GROUP_START_CAP
        ),
        "instrumentation_maximum_complete_group_overshoot": int(
            engine.MAX_COMPLETE_GROUP_OVERSHOOT
        ),
        "same_locked_neural_module_object": engine.locked is locked,
        "n_rollouts": int(locked.N_ROLLOUTS),
        "thresholds": list(locked.THRESHOLDS),
        "teacher_gamma": float(locked.TEACHER_GAMMA),
        "teacher_decay": float(locked.TEACHER_DECAY),
        "teacher_floor": float(locked.TEACHER_FLOOR),
        "max_episode_steps": int(acrobot_adapter.MAX_EPISODE_STEPS),
        "shared_h64_total_parameters": int(actor.parameter_count),
        "shared_h64_active_parameters": int(actor.active_parameter_count),
    }
    expected = {
        "instrumentation_checkpoints": [250, 400],
        "instrumentation_eval_interval_updates": 50,
        "instrumentation_transition_group_start_cap": 4_000_000,
        "instrumentation_maximum_complete_group_overshoot": 8_000,
        "same_locked_neural_module_object": True,
        "n_rollouts": 16,
        "thresholds": [-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.7, 1.0],
        "teacher_gamma": 1.0,
        "teacher_decay": 0.7,
        "teacher_floor": 0.1,
        "max_episode_steps": 500,
        "shared_h64_total_parameters": 640,
        "shared_h64_active_parameters": 640,
    }
    if contract != expected:
        raise RuntimeError(
            f"V5 imported-engine semantic contract changed: {contract!r} != {expected!r}"
        )
    return contract


def _seed_collision_audit() -> dict:
    """Fail closed if either fresh V5 block intersects a registered prior block."""

    prior_union = {
        seed for seeds in PRIOR_ACROBOT_SEED_BLOCKS.values() for seed in seeds
    }
    stage_a = set(STAGE_A_SEEDS)
    stage_b = set(STAGE_B_SEEDS)
    collisions = {
        "v5a_vs_prior": sorted(stage_a & prior_union),
        "v5b_vs_prior": sorted(stage_b & prior_union),
        "v5a_vs_v5b": sorted(stage_a & stage_b),
    }
    audit = {
        "prior_registered_blocks": {
            name: [min(seeds), max(seeds), len(seeds)]
            for name, seeds in PRIOR_ACROBOT_SEED_BLOCKS.items()
        },
        "v5a_block": [min(STAGE_A_SEEDS), max(STAGE_A_SEEDS), len(STAGE_A_SEEDS)],
        "v5b_block": [min(STAGE_B_SEEDS), max(STAGE_B_SEEDS), len(STAGE_B_SEEDS)],
        "collisions": collisions,
        "passed": not any(collisions.values()),
        "scope": (
            "explicit ledger of Acrobot seed blocks registered by V1--V4 source/protocols; "
            "the source/protocol files carrying that ledger are themselves hash-locked"
        ),
    }
    if audit["passed"] is not True:
        raise RuntimeError(f"V5 seed collision audit failed: {collisions!r}")
    return audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_hashes() -> dict[str, str]:
    missing = [
        relative
        for relative in SOURCE_RELATIVE_PATHS
        if not (PROJECT_ROOT / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "cannot form/verify the V5 source lock; missing: " + ", ".join(missing)
        )
    return {
        relative: _sha256(PROJECT_ROOT / relative)
        for relative in SOURCE_RELATIVE_PATHS
    }


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"required preregistered file is missing: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _atomic_write(path: Path, payload: dict, *, must_not_exist: bool) -> None:
    """Durably replace one complete JSON object in the destination directory."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if must_not_exist and path.exists():
        raise FileExistsError(f"refusing to overwrite existing file {path}")
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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


def _verify_source_lock(
    lock_path: Path,
    *,
    schema: str,
    v5_stage: str,
    schedule: dict,
    expected_lock_sha256: str | None = None,
    amendment_path: Path | None = None,
) -> tuple[dict, str]:
    """Verify runtime, exact schedule, and all source bytes before any seed."""

    lock_path = lock_path.resolve()
    lock = _read_json(lock_path)
    lock_hash = _sha256(lock_path)
    errors: list[str] = []
    if expected_lock_sha256 is not None and lock_hash != expected_lock_sha256:
        errors.append("source-lock file hash changed since artifact creation")
    if lock.get("schema") != schema:
        errors.append(f"lock schema must be {schema!r}")
    if lock.get("v5_stage") != v5_stage:
        errors.append(f"lock v5_stage must be {v5_stage!r}")
    if lock.get("runtime") != _runtime():
        errors.append("live runtime differs from the frozen runtime")
    if lock.get("registered_schedule") != schedule:
        errors.append("registered schedule differs from the frozen runner")
    try:
        if lock.get("engine_contract") != _engine_contract():
            errors.append("imported engine semantic contract differs from the lock")
        live_seed_audit = _seed_collision_audit()
        if live_seed_audit.get("passed") is not True:
            errors.append("live seed collision audit did not pass")
        if lock.get("seed_collision_audit") != live_seed_audit:
            errors.append("seed collision audit differs from the lock")
    except Exception as error:
        errors.append(str(error))
    sources = lock.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_RELATIVE_PATHS):
        errors.append("source lock does not exactly cover the V5 evidence chain")
    else:
        try:
            if sources != _source_hashes():
                errors.append("one or more live source bytes differ from the lock")
        except Exception as error:  # fail closed with a useful aggregate message
            errors.append(str(error))
    if amendment_path is not None:
        amendment_path = amendment_path.resolve()
        if not amendment_path.is_file() or lock.get("amendment_sha256") != _sha256(
            amendment_path
        ):
            errors.append("V5B amendment is missing or differs from its lock")
    if errors:
        raise RuntimeError("V5 source/runtime lock verification failed: " + "; ".join(errors))
    return lock, lock_hash


def _module_command(module: str, *parts: object) -> str:
    return " ".join([sys.executable, "-m", module, *(str(part) for part in parts)])


def _seal_stage_a(output: Path) -> None:
    output = output.resolve()
    engine_contract = _engine_contract()
    seed_collision_audit = _seed_collision_audit()
    if seed_collision_audit.get("passed") is not True:
        raise RuntimeError("refusing to seal V5A with a failed seed collision audit")
    lock = {
        "schema": LOCK_SCHEMA_A,
        "v5_stage": "stage_a_natural_feasibility",
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Pre-execution source/runtime lock for fresh full-grid, learning-outcome-"
            "field-blind "
            "V5A natural feasibility. V4A remains immutable failed calibration."
        ),
        "runtime": _runtime(),
        "engine_contract": engine_contract,
        "seed_collision_audit": seed_collision_audit,
        "source_sha256": _source_hashes(),
        "reused_engine": {
            "module": "frontier_rl.examples.run_acrobot_hindsight_v4",
            "relative_path": "frontier_rl/examples/run_acrobot_hindsight_v4.py",
            "sha256": _sha256(
                PROJECT_ROOT / "frontier_rl/examples/run_acrobot_hindsight_v4.py"
            ),
            "scope": "instrumented execution only; no V4 schema or V4 gate is reused",
        },
        "registered_schedule": _stage_a_schedule(),
        "exact_commands": {
            "runner": _module_command(
                "frontier_rl.examples.run_acrobot_hindsight_v5",
                "stage-a",
                "--lock",
                output,
                "--output",
                DEFAULT_ARTIFACT_A.resolve(),
            ),
            "independent_verifier": _module_command(
                ANALYZER_MODULE,
                DEFAULT_ARTIFACT_A.resolve(),
                "--lock",
                output,
                "--output",
                DEFAULT_VERIFICATION_A.resolve(),
            ),
            "preflight_tests": _module_command(
                "pytest",
                "frontier_rl/examples/test_acrobot_neural.py",
                "frontier_rl/examples/test_run_acrobot_neural.py",
                "frontier_rl/examples/test_run_acrobot_hindsight_v4.py",
                "frontier_rl/examples/test_run_acrobot_hindsight_v5.py",
                "frontier_rl/examples/test_analyze_acrobot_hindsight_v5.py",
            ),
        },
        "equality_rule": (
            "V5A refuses start/resume unless runtime, schedule, source key set, "
            "and every source byte hash match exactly."
        ),
    }
    _atomic_write(output, lock, must_not_exist=True)
    print(f"sealed V5A source/runtime lock: {output}")


def _conditions() -> tuple[locked.Condition, ...]:
    return tuple(
        locked.Condition(
            name=_case_name(multiplier, scale),
            stage="scale",
            sampling="teacher",
            architecture="shared",
            hidden_size=64,
            learning_rate=BASE_LEARNING_RATE * multiplier,
            hindsight_scale=scale,
            lr_multiplier=multiplier,
        )
        for multiplier in LR_MULTIPLIERS
        for scale in HINDSIGHT_SCALES
    )


def _protocol(stage: str, schedule: dict) -> dict:
    protocol = {
        "v5_stage": stage,
        "protocol_document": str(PROTOCOL_PATH.relative_to(PROJECT_ROOT)),
        "registered_schedule": schedule,
        "predecessor_status": (
            "V4A is immutable failed calibration evidence; no V4 run selects V5 U* "
            "and no V4 outcome is pooled with V5"
        ),
        "execution_engine": {
            "module": "frontier_rl.examples.run_acrobot_hindsight_v4",
            "function": "_instrumented_run",
            "semantic_base": "frontier_rl.examples.run_acrobot_neural.run_condition",
            "provenance_rule": "all engine/base bytes are present in source_sha256",
        },
        "gymnasium_environment": "Acrobot-v1",
        "thresholds": list(locked.THRESHOLDS),
        "verifier": "strict post-transition Acrobot tip height > threshold",
        "max_episode_steps": 500,
        "n_rollouts": locked.N_ROLLOUTS,
        "sampling": "frontier-u_16 teacher",
        "teacher_utility": "1-(1-p)^N-p",
        "teacher_gamma": locked.TEACHER_GAMMA,
        "teacher_decay": locked.TEACHER_DECAY,
        "teacher_floor": locked.TEACHER_FLOOR,
        "teacher_evidence": "requested task and original binary outcomes only",
        "actor": "task-agnostic shared H64; 640 total and active parameters",
        "optimizer": "plain SGD ascent",
        "hindsight": "hardest verifier-valid mixed lower predicate; first-hit truncation",
        "matching_axis": "nonzero applied optimizer updates",
        "evaluation": "update 0 and each 50 nonzero updates; 32 episodes/task",
        "group_cap": (
            "no group starts at transitions >=4,000,000; a started 16-rollout group "
            "finishes, so overshoot is at most 8,000 transitions"
        ),
    }
    if stage == "stage_a_natural_feasibility":
        protocol.update(
            {
                "status": "development_learning_outcome_field_blind",
                "learning_outcome_fields_excluded_from_launch_rule": True,
                "budget_rule": (
                    "U*=400 iff all 27 runs reach 400; else U*=250 iff all 27 "
                    "reach 250; else STOP. Every gate uses first exact U* prefixes."
                ),
                "natural_gate_rule": (
                    "registered runs contain no forced relabels: every cell must naturally "
                    "produce an accepted "
                    "candidate pooled across its three seeds; positive-scale candidates "
                    "must all be positive finite applied auxiliary updates"
                ),
                "runtime_projection": (
                    "180 * slowest selected-prefix wall seconds / 3600 <= 18"
                ),
            }
        )
    elif stage == "stage_b_confirmatory_factorial":
        protocol.update(
            {
                "status": "confirmatory",
                "primary_metric": (
                    "normalized target-uniform mean-pass AUC over nonzero optimizer "
                    "updates, including update zero"
                ),
                "primary_family": ["C1", "C2", "C3", "C4"],
                "family_rule": "all 180 runs valid or no member of the family is analyzed",
                "test": "exact 2^20 two-sided paired sign-flip with Holm FWER 0.05",
                "bootstrap": "20,000 paired-seed resamples, descriptive",
                "bootstrap_seed_by_contrast": {
                    "C1": 55_000,
                    "C2": 55_001,
                    "C3": 55_002,
                    "C4": 55_003,
                },
                "sign_exchangeability_assumption": True,
            }
        )
    else:
        raise ValueError(f"unknown V5 stage {stage!r}")
    return protocol


def _provenance(lock_path: Path, lock_hash: str) -> dict:
    source_hashes = _source_hashes()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime(),
        "platform": platform.platform(),
        "source_sha256": source_hashes,
        "source_lock_path": str(lock_path.resolve()),
        "source_lock_sha256": lock_hash,
        "reused_engine_sha256": source_hashes[
            "frontier_rl/examples/run_acrobot_hindsight_v4.py"
        ],
        "v4_artifacts_reused": False,
    }


def _new_artifact(*, stage: str, schedule: dict, lock_path: Path, lock_hash: str) -> dict:
    return {
        "schema": SCHEMA_A if stage == "stage_a_natural_feasibility" else SCHEMA_B,
        "provenance": _provenance(lock_path, lock_hash),
        "protocol": _protocol(stage, schedule),
        "artifact_state": "in_progress",
        "run_failures": [],
        "cases": {
            condition.name: {"config": asdict(condition), "runs": []}
            for condition in _conditions()
        },
    }


def _instrumented_run(
    condition: locked.Condition,
    seed: int,
    *,
    budget: locked.RunBudget,
    eval_n: int,
) -> dict:
    """Call the source-locked V4 instrumentation without changing semantics."""

    return engine._instrumented_run(condition, seed, budget=budget, eval_n=eval_n)


def _deterministic_scale_fixture() -> dict:
    """Exercise the exact auxiliary gradient in every ``(a,s)`` cell.

    All actors begin from identical parameters and receive the same synthetic
    frozen trajectory group.  For positive scale, plain SGD must produce
    ``delta(theta)=base_lr*a*s*g`` exactly to numerical tolerance.  Scale zero
    must compute a positive preview while preserving parameters, counters, and
    the action RNG.  This forced *mechanical* fixture is separate from the
    registered V5A runs, whose natural-relabel coverage is never forced.
    """

    trajectories = [
        [(np.array([0.20, -0.10, 0.30, -0.20, 0.01, -0.02]), 0)],
        [(np.array([-0.30, 0.25, -0.15, 0.10, -0.03, 0.04]), 1)],
        [(np.array([0.15, 0.05, -0.25, 0.35, 0.02, 0.01]), 2)],
        [(np.array([-0.05, -0.20, 0.10, 0.30, -0.01, -0.03]), 0)],
    ]
    unscaled_weights = np.asarray([-0.75, 0.25, 0.50, 1.00], dtype=np.float64)
    credited_task = 2
    requested_task = 3
    seed = 9_907
    cells: dict[str, dict] = {}
    all_pass = True
    reference_scaled_norms: list[float] = []
    for multiplier in LR_MULTIPLIERS:
        for scale in HINDSIGHT_SCALES:
            actor = locked.TanhCategoricalActor(
                n_tasks=8,
                hidden_size=64,
                learning_rate=BASE_LEARNING_RATE * multiplier,
                seed=seed,
                mode="shared",
            )
            before = actor.parameter_vector()
            before_rng = _canonical_hash(actor.action_rng.bit_generator.state)
            before_counters = [int(actor.update_calls), int(actor.applied_updates)]
            gradient = actor.group_gradient(
                credited_task, trajectories, unscaled_weights
            )
            flat_gradient = np.concatenate(
                [gradient["W_in"].ravel(), gradient["b_hidden"].ravel(), gradient["W_out"].ravel()]
            )
            gradient_norm = float(np.linalg.norm(flat_gradient))
            if scale == 0.0:
                preview = actor.gradient_diagnostics(
                    credited_task, trajectories, unscaled_weights
                )
                after = actor.parameter_vector()
                delta = after - before
                expected = np.zeros_like(delta)
                preview_positive = bool(
                    math.isfinite(float(preview["gradient_norm"]))
                    and float(preview["gradient_norm"]) > 0.0
                )
            else:
                actor.update(
                    credited_task, trajectories, unscaled_weights * scale
                )
                after = actor.parameter_vector()
                delta = after - before
                expected = BASE_LEARNING_RATE * multiplier * scale * flat_gradient
                preview_positive = None
                reference_scaled_norms.append(
                    float(np.linalg.norm(delta) / (multiplier * scale))
                )
            after_rng = _canonical_hash(actor.action_rng.bit_generator.state)
            after_counters = [int(actor.update_calls), int(actor.applied_updates)]
            delta_matches = bool(np.allclose(delta, expected, rtol=1e-12, atol=1e-15))
            state_rule = bool(
                np.array_equal(after, before)
                and before_rng == after_rng
                and before_counters == after_counters
            ) if scale == 0.0 else bool(
                after_counters == [1, 1]
                and before_rng == after_rng
                and float(np.linalg.norm(delta)) > 0.0
            )
            passed = bool(
                gradient_norm > 0.0
                and requested_task > credited_task
                and delta_matches
                and state_rule
                and (preview_positive is not False)
            )
            all_pass = all_pass and passed
            cells[_case_name(multiplier, scale)] = {
                "passed": passed,
                "learning_rate_multiplier": multiplier,
                "hindsight_scale": scale,
                "requested_task": requested_task,
                "credited_task": credited_task,
                "source": "hindsight_relabel" if scale > 0.0 else "scale_zero_preview",
                "unscaled_gradient_norm": gradient_norm,
                "parameter_delta_norm": float(np.linalg.norm(delta)),
                "expected_parameter_delta_norm": float(np.linalg.norm(expected)),
                "delta_theta_equals_base_lr_times_a_times_s_times_g": delta_matches,
                "parameter_counter_rng_rule_passed": state_rule,
                "positive_nonmutating_preview": preview_positive,
            }
    common_ratio = bool(
        reference_scaled_norms
        and np.allclose(
            reference_scaled_norms,
            np.full(len(reference_scaled_norms), reference_scaled_norms[0]),
            rtol=1e-12,
            atol=1e-15,
        )
    )
    return {
        "schema": "curriculum-maxrl/acrobot-hindsight-v5-scale-fixture/v1",
        "synthetic_seed": seed,
        "registered_seed_touched": False,
        "forced_fixture_only": True,
        "natural_stage_a_events_forced": False,
        "same_initial_parameters_and_group_in_all_cells": True,
        "credited_task_strictly_lower_than_requested": credited_task < requested_task,
        "positive_cell_delta_norm_over_a_times_s_constant": common_ratio,
        "cells": cells,
        "passed": bool(all_pass and common_ratio and len(cells) == 9),
    }


def _shadow_projection(run: dict) -> dict:
    keys = (
        "transitions",
        "sampled_groups",
        "optimizer_updates",
        "live_groups",
        "live_applied_updates",
        "dead_groups",
        "all_pass_groups",
        "task_groups",
        "task_rollouts",
        "task_transitions",
        "x_transitions",
        "x_optimizer_updates",
        "update_diagnostics",
        "group_diagnostics",
        "training_group_trace_groups",
        "training_group_trace_sha256",
        "final_training_state_sha256",
    )
    return {key: run[key] for key in keys}


def _preview_shadow_equivalence() -> dict:
    preview_condition = locked.Condition(
        name="v5_preview_shadow_scale_zero",
        stage="scale",
        sampling="teacher",
        architecture="shared",
        hidden_size=64,
        learning_rate=BASE_LEARNING_RATE,
        hindsight_scale=0.0,
        lr_multiplier=1.0,
    )
    no_hindsight = replace(
        preview_condition, name="v5_preview_shadow_no_hindsight", stage="core"
    )
    budget = locked.RunBudget(
        optimizer_update_budget=SHADOW_UPDATE_BUDGET,
        transition_safety_cap=SHADOW_TRANSITION_CAP,
    )
    config = {
        "seed": SHADOW_SEED,
        "seed_block_status": "already-executed exploratory smoke; outside fresh V5 blocks",
        "learning_rate": BASE_LEARNING_RATE,
        "optimizer_update_budget": SHADOW_UPDATE_BUDGET,
        "transition_group_start_cap": SHADOW_TRANSITION_CAP,
        "eval_episodes_per_task": SHADOW_EVAL_EPISODES,
    }
    try:
        preview = _instrumented_run(
            preview_condition, SHADOW_SEED, budget=budget, eval_n=SHADOW_EVAL_EPISODES
        )
        control = _instrumented_run(
            no_hindsight, SHADOW_SEED, budget=budget, eval_n=SHADOW_EVAL_EPISODES
        )
        candidates = preview["eligible_relabel_candidate_groups"]
        records = preview["auxiliary_gradient_diagnostics"]
        record_groups = [int(record["after_group"]) for record in records]
        group_by_id = {
            int(group["group"]): group for group in preview["group_diagnostics"]
        }
        previews_valid = bool(candidates) and candidates == record_groups and all(
            record.get("applied") is False
            and record.get("mutated") is False
            and record.get("frozen_group_parameters") is True
            and math.isfinite(float(record.get("gradient_norm", math.nan)))
            and float(record["gradient_norm"]) > 0.0
            and math.isfinite(
                float(record.get("hypothetical_update_norm", math.nan))
            )
            and float(record["hypothetical_update_norm"]) > 0.0
            and int(record.get("after_group", -1)) in group_by_id
            and int(record.get("requested_task", -1))
            == int(group_by_id[int(record["after_group"])]["task_id"])
            and 0
            <= int(record.get("credited_task", -1))
            < int(record.get("requested_task", -1))
            < len(locked.THRESHOLDS)
            for record in records
        )
        trace_equal = (
            preview["training_group_trace_sha256"]
            == control["training_group_trace_sha256"]
        )
        state_equal = (
            preview["final_training_state_sha256"]
            == control["final_training_state_sha256"]
        )
        projection_equal = _shadow_projection(preview) == _shadow_projection(control)
        preview_projection = _shadow_projection(preview)
        control_projection = _shadow_projection(control)
        return {
            "test_config": config,
            "passed": bool(previews_valid and trace_equal and state_equal and projection_equal),
            "identical_training_group_trace": trace_equal,
            "identical_final_training_state": state_equal,
            "identical_saved_training_projection": projection_equal,
            "eligible_preview_exercised": bool(candidates),
            "preview_candidate_groups": candidates,
            "preview_diagnostic_groups": record_groups,
            "preview_auxiliary_gradient_diagnostics": records,
            "preview_mechanical_projection": preview_projection,
            "control_mechanical_projection": control_projection,
            "preview_mechanical_projection_sha256": _canonical_hash(preview_projection),
            "control_mechanical_projection_sha256": _canonical_hash(control_projection),
            "preview_training_group_trace_sha256": preview[
                "training_group_trace_sha256"
            ],
            "shadow_training_group_trace_sha256": control[
                "training_group_trace_sha256"
            ],
            "preview_final_training_state_sha256": preview[
                "final_training_state_sha256"
            ],
            "shadow_final_training_state_sha256": control[
                "final_training_state_sha256"
            ],
            "preview_transitions": preview["transitions"],
            "shadow_transitions": control["transitions"],
        }
    except Exception as error:
        return {
            "test_config": config,
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }


def _source_step_norms(records: Sequence[dict]) -> dict:
    output = {}
    for source in ("requested_live", "hindsight_relabel"):
        selected = [record for record in records if record.get("source") == source]
        norms = np.asarray(
            [float(record.get("update_norm", math.nan)) for record in selected],
            dtype=np.float64,
        )
        if len(norms) and (not np.isfinite(norms).all() or np.any(norms <= 0.0)):
            raise ValueError(f"{source} contains a non-positive/non-finite applied step")
        output[source] = {
            "count": len(selected),
            "cumulative_step_norm_M": float(norms.sum()) if len(norms) else 0.0,
            "cumulative_squared_step_norm_Q": (
                float(np.dot(norms, norms)) if len(norms) else 0.0
            ),
        }
    return output


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, np.integer)):
        return True
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_all_finite(key) and _all_finite(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return False


def _technical_gate_projection(value: Any) -> Any:
    """Copy a Stage-A object while structurally excluding all outcome fields."""

    if isinstance(value, dict):
        return {
            key: _technical_gate_projection(item)
            for key, item in value.items()
            if key not in OUTCOME_FIELD_NAMES
        }
    if isinstance(value, list):
        return [_technical_gate_projection(item) for item in value]
    return copy.deepcopy(value)


def _technical_run_errors(run: dict) -> list[str]:
    """Validate mechanics; launch gates use the prefix-specific variant below."""

    errors: list[str] = []
    if run.get("numeric_valid") is not True:
        return ["numeric_valid is not true"]
    technical = _technical_gate_projection(run)
    if not _all_finite(technical):
        errors.append("technical run projection contains non-finite data")
    if not all(
        run.get(key) is True
        for key in (
            "accounting_valid",
            "verifier_relabel_checks_valid",
            "evaluation_cadence_invariant",
        )
    ):
        errors.append("saved implementation invariant flag is not true")
    if (run.get("total_parameters"), run.get("active_parameters_per_task")) != (
        640,
        640,
    ):
        errors.append("shared-H64 parameter-count contract failed")
    groups = run.get("group_diagnostics")
    if not isinstance(groups, list) or len(groups) != run.get("sampled_groups"):
        return errors + ["sampled-group diagnostics are missing or inconsistent"]
    previous_end = 0
    previous_updates = 0
    regimes = {"dead": 0, "mixed": 0, "all_pass": 0}
    for expected_id, group in enumerate(groups, start=1):
        start = group.get("transition_start")
        end = group.get("transition_end")
        count = group.get("n_transitions")
        success = group.get("success_count")
        task_id = group.get("task_id")
        updates = group.get("optimizer_updates_after_group")
        if group.get("group") != expected_id:
            errors.append("group ids are not consecutive")
            break
        if (
            not all(isinstance(value, int) for value in (start, end, count))
            or start != previous_end
            or end - start != count
            or not 1 <= count <= MAX_COMPLETE_GROUP_OVERSHOOT
            or start >= TRANSITION_GROUP_START_CAP
        ):
            errors.append("group transition accounting/cap failed")
            break
        expected_regime = (
            "dead"
            if success == 0
            else "all_pass"
            if success == locked.N_ROLLOUTS
            else "mixed"
        )
        if (
            not isinstance(success, int)
            or not 0 <= success <= locked.N_ROLLOUTS
            or group.get("regime") != expected_regime
            or not isinstance(task_id, int)
            or not 0 <= task_id < len(locked.THRESHOLDS)
        ):
            errors.append("binary verifier/task/regime accounting failed")
            break
        if not isinstance(updates, int) or updates - previous_updates not in (0, 1):
            errors.append("optimizer counter changes by more than one in a group")
            break
        source = group.get("update_source")
        if (updates == previous_updates) != (source is None):
            errors.append("group update source and counter disagree")
            break
        if source not in (None, "requested_live", "hindsight_relabel"):
            errors.append("unknown update source")
            break
        if source == "requested_live" and expected_regime != "mixed":
            errors.append("requested-live update did not originate from a mixed group")
            break
        if source == "hindsight_relabel" and expected_regime != "dead":
            errors.append("hindsight update did not originate from a dead group")
            break
        regimes[expected_regime] += 1
        previous_end = end
        previous_updates = updates
    if previous_end != run.get("transitions"):
        errors.append("group transitions do not reproduce the terminal count")
    if previous_updates != run.get("optimizer_updates"):
        errors.append("group counters do not reproduce optimizer updates")
    if run.get("transitions", 0) > TRANSITION_GROUP_START_CAP + MAX_COMPLETE_GROUP_OVERSHOOT:
        errors.append("complete-group overshoot exceeds 8,000 transitions")
    for key, expected in (
        ("dead_groups", regimes["dead"]),
        ("live_groups", regimes["mixed"]),
        ("all_pass_groups", regimes["all_pass"]),
    ):
        if run.get(key) != expected:
            errors.append(f"{key} does not reproduce from group records")
    updates = run.get("update_diagnostics")
    if not isinstance(updates, list) or len(updates) != run.get("optimizer_updates"):
        return errors + ["optimizer update diagnostics are missing/inconsistent"]
    by_group = {int(record.get("after_group", -1)): record for record in updates}
    for index, record in enumerate(updates, start=1):
        source = record.get("source")
        norm = float(record.get("update_norm", math.nan))
        gradient = float(record.get("gradient_norm", math.nan))
        if (
            record.get("optimizer_update") != index
            or source not in ("requested_live", "hindsight_relabel")
            or not math.isfinite(norm)
            or norm <= 0.0
            or not math.isfinite(gradient)
            or gradient <= 0.0
        ):
            errors.append("applied update order/source/norm is invalid")
            break
        group_id = int(record.get("after_group", -1))
        if not 1 <= group_id <= len(groups) or groups[group_id - 1].get(
            "update_source"
        ) != source or record.get("transitions") != groups[group_id - 1].get(
            "transition_end"
        ):
            errors.append("applied update does not match its source group")
            break
        group = groups[group_id - 1]
        requested = int(record.get("requested_task", -1))
        credited = int(record.get("credited_task", -1))
        if source == "requested_live" and not (
            requested == credited == int(group["task_id"])
        ):
            errors.append("requested-live task metadata is invalid")
            break
        if source == "hindsight_relabel" and not (
            requested == int(group["task_id"])
            and 0 <= credited < requested < len(locked.THRESHOLDS)
        ):
            errors.append("hindsight task metadata is invalid")
            break
    if len(by_group) != len(updates):
        errors.append("more than one optimizer update is recorded for a group")
    if run.get("optimizer_updates") != run.get("live_applied_updates", 0) + run.get(
        "relabeled_groups", 0
    ):
        errors.append("optimizer source counters do not sum")
    source_counts = {
        source: sum(record.get("source") == source for record in updates)
        for source in ("requested_live", "hindsight_relabel")
    }
    if source_counts["requested_live"] != run.get("live_applied_updates"):
        errors.append("requested-live applied counter does not reproduce")
    if source_counts["hindsight_relabel"] != run.get("relabeled_groups"):
        errors.append("hindsight applied counter does not reproduce")
    x_updates = run.get("x_optimizer_updates")
    x_transitions = run.get("x_transitions")
    if (
        not isinstance(x_updates, list)
        or not isinstance(x_transitions, list)
        or len(x_updates) != len(x_transitions)
        or not x_updates
        or x_updates[0] != 0
        or x_transitions[0] != 0
        or any(right <= left for left, right in zip(x_transitions, x_transitions[1:]))
    ):
        errors.append("evaluation resource coordinates are invalid")
    if not all(run.get("evaluation_rng_preserved", [])):
        errors.append("evaluation changed training state")
    if run.get("training_group_trace_groups") != run.get("sampled_groups"):
        errors.append("instrumented trajectory trace count mismatch")
    for digest_key in ("training_group_trace_sha256", "final_training_state_sha256"):
        digest = run.get(digest_key)
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append(f"missing/malformed {digest_key}")
    return errors


def _prefix(run: dict, target: int) -> dict:
    groups = run["group_diagnostics"]
    terminal_index = next(
        (
            index
            for index, group in enumerate(groups)
            if int(group["optimizer_updates_after_group"]) == target
            and all(
                int(previous["optimizer_updates_after_group"]) < target
                for previous in groups[:index]
            )
        ),
        None,
    )
    if terminal_index is None:
        raise ValueError(f"run has no first exact {target}-update prefix")
    prefix_groups = groups[: terminal_index + 1]
    terminal_group = int(prefix_groups[-1]["group"])
    # Exactly one applied update can occur per group, so the first U* update
    # diagnostics are the only possible selected-prefix update records.  This
    # prevents malformed post-prefix records with reused ids from leaking into
    # a fallback launch decision.
    updates = list(run["update_diagnostics"][:target])

    # Candidate/preview ids are emitted in group order.  Retain only the
    # strictly increasing leading segment inside the selected boundary; once
    # an id leaves or repeats the prefix, every later record is post-prefix for
    # launch purposes and is audited separately outside the decision.
    candidates = [
        int(raw_group_id)
        for raw_group_id in run.get("eligible_relabel_candidate_groups", [])
        if int(raw_group_id) <= terminal_group
    ]
    previews = [
        record
        for record in run.get("auxiliary_gradient_diagnostics", [])
        if int(record["after_group"]) <= terminal_group
    ]
    evaluation_indices = list(range(target // EVAL_INTERVAL_UPDATES + 1))
    timing = run.get("wall_seconds_at_optimizer_updates", {}).get(str(target))
    if timing is None:
        raise ValueError(f"run lacks exact wall time at update {target}")
    return {
        "target_updates": target,
        "terminal_group": terminal_group,
        "terminal_transitions": int(prefix_groups[-1]["transition_end"]),
        "groups": prefix_groups,
        "updates": updates,
        "candidate_groups": candidates,
        "previews": previews,
        "evaluation_indices": evaluation_indices,
        "wall_seconds": float(timing),
        "source_step_norms": _source_step_norms(updates),
    }


def _prefix_errors(prefix: dict, run: dict) -> list[str]:
    errors: list[str] = []
    target = prefix["target_updates"]
    groups = prefix["groups"]
    updates = prefix["updates"]
    if int(groups[-1]["optimizer_updates_after_group"]) != target or any(
        int(group["optimizer_updates_after_group"]) >= target for group in groups[:-1]
    ):
        errors.append("prefix is not the first exact selected-update prefix")
    if len(updates) != target or [record.get("optimizer_update") for record in updates] != list(
        range(1, target + 1)
    ):
        errors.append("selected prefix does not contain exactly U* ordered updates")
    if not math.isfinite(prefix["wall_seconds"]) or prefix["wall_seconds"] <= 0.0:
        errors.append("selected-prefix wall time is invalid")
    # Evaluation *coordinates* are allowed; evaluation outcomes are not read.
    expected_coordinates = list(range(0, target + 1, EVAL_INTERVAL_UPDATES))
    x_updates = run.get("x_optimizer_updates", [])
    x_transitions = run.get("x_transitions", [])
    selected_indices = prefix["evaluation_indices"]
    if (
        len(x_updates) < len(selected_indices)
        or len(x_transitions) < len(selected_indices)
        or not selected_indices
        or selected_indices != list(range(len(selected_indices)))
    ):
        errors.append("selected prefix lacks paired evaluation resource records")
        return errors
    observed = [
        int(x_updates[index]) for index in selected_indices
    ]
    if observed != expected_coordinates:
        errors.append("selected prefix lacks the exact 50-update resource cadence")
    selected_transitions = [x_transitions[index] for index in selected_indices]
    if (
        any(not isinstance(value, int) or value < 0 for value in selected_transitions)
        or selected_transitions[0] != 0
        or any(
            right <= left
            for left, right in zip(selected_transitions, selected_transitions[1:])
        )
        or selected_transitions[-1] != prefix["terminal_transitions"]
    ):
        errors.append("selected prefix evaluation transition coordinates are invalid")
    return errors


def _prefix_technical_errors(prefix: dict, run: dict) -> list[str]:
    """Recompute launch invariants using only the first-exact-U* prefix."""

    errors = _prefix_errors(prefix, run)
    if (run.get("total_parameters"), run.get("active_parameters_per_task")) != (640, 640):
        errors.append("shared-H64 parameter-count contract failed")
    groups = prefix["groups"]
    updates = prefix["updates"]
    if not _all_finite(
        _technical_gate_projection(
            {
                "groups": groups,
                "updates": updates,
                "previews": prefix["previews"],
                "wall_seconds": prefix["wall_seconds"],
            }
        )
    ):
        errors.append("selected technical prefix contains non-finite data")
    previous_end = 0
    previous_updates = 0
    update_by_group = {int(record.get("after_group", -1)): record for record in updates}
    if len(update_by_group) != len(updates) or any(
        group_id < 1 or group_id > len(groups) for group_id in update_by_group
    ):
        errors.append("prefix update after_group ids are duplicate or out of range")
    for expected_id, group in enumerate(groups, start=1):
        start = group.get("transition_start")
        end = group.get("transition_end")
        count = group.get("n_transitions")
        success = group.get("success_count")
        task_id = group.get("task_id")
        update_count = group.get("optimizer_updates_after_group")
        expected_regime = (
            "dead"
            if success == 0
            else "all_pass"
            if success == locked.N_ROLLOUTS
            else "mixed"
        )
        if group.get("group") != expected_id:
            errors.append("prefix group identifiers are not consecutive")
            break
        if (
            not all(isinstance(value, int) for value in (start, end, count))
            or start != previous_end
            or end - start != count
            or not 1 <= count <= MAX_COMPLETE_GROUP_OVERSHOOT
            or start >= TRANSITION_GROUP_START_CAP
        ):
            errors.append("prefix transition/group accounting failed")
            break
        if (
            not isinstance(success, int)
            or not 0 <= success <= locked.N_ROLLOUTS
            or group.get("regime") != expected_regime
            or not isinstance(task_id, int)
            or not 0 <= task_id < len(locked.THRESHOLDS)
        ):
            errors.append("prefix verifier/task/regime accounting failed")
            break
        if not isinstance(update_count, int) or update_count - previous_updates not in (0, 1):
            errors.append("prefix optimizer counter jump failed")
            break
        diagnostic = update_by_group.get(expected_id)
        expected_source = None if diagnostic is None else diagnostic.get("source")
        incremented = update_count == previous_updates + 1
        if incremented != (diagnostic is not None) or group.get(
            "update_source"
        ) != expected_source:
            errors.append("prefix group/update source mismatch")
            break
        if expected_source == "requested_live" and expected_regime != "mixed":
            errors.append("prefix requested-live source is not a mixed group")
            break
        if expected_source == "hindsight_relabel" and expected_regime != "dead":
            errors.append("prefix hindsight source is not a dead group")
            break
        previous_end = end
        previous_updates = update_count
    if previous_end != prefix["terminal_transitions"] or previous_updates != prefix[
        "target_updates"
    ]:
        errors.append("prefix terminal resource counters do not reproduce")
    if prefix["terminal_transitions"] > TRANSITION_GROUP_START_CAP + MAX_COMPLETE_GROUP_OVERSHOOT:
        errors.append("prefix cap overshoot exceeds one complete group")
    for index, record in enumerate(updates, start=1):
        source = record.get("source")
        norm = float(record.get("update_norm", math.nan))
        gradient = float(record.get("gradient_norm", math.nan))
        if (
            record.get("optimizer_update") != index
            or source not in ("requested_live", "hindsight_relabel")
            or not math.isfinite(norm)
            or norm <= 0.0
            or not math.isfinite(gradient)
            or gradient <= 0.0
        ):
            errors.append("prefix applied-update order/source/norm failed")
            break
        group = groups[int(record["after_group"]) - 1]
        if record.get("transitions") != group.get("transition_end"):
            errors.append("prefix applied-update transition coordinate is invalid")
            break
        requested = int(record.get("requested_task", -1))
        credited = int(record.get("credited_task", -1))
        if source == "requested_live" and not (
            requested == credited == int(group["task_id"])
        ):
            errors.append("prefix requested-live task metadata is invalid")
            break
        if source == "hindsight_relabel" and not (
            requested == int(group["task_id"])
            and 0 <= credited < requested < len(locked.THRESHOLDS)
        ):
            errors.append("prefix hindsight task metadata is invalid")
            break
    candidates = prefix["candidate_groups"]
    if (
        candidates != sorted(candidates)
        or len(candidates) != len(set(candidates))
        or any(group_id < 1 or group_id > len(groups) for group_id in candidates)
        or any(groups[group_id - 1].get("regime") != "dead" for group_id in candidates)
    ):
        errors.append("prefix eligible relabel-candidate ids are invalid")
    evaluation_indices = prefix["evaluation_indices"]
    preserved = run.get("evaluation_rng_preserved", [])
    if any(index >= len(preserved) or preserved[index] is not True for index in evaluation_indices):
        errors.append("prefix evaluation changed training state")
    return errors


def _stage_a_lock_check(artifact: dict, lock_path: Path) -> tuple[bool, str | None]:
    try:
        _, observed_hash = _verify_source_lock(
            lock_path,
            schema=LOCK_SCHEMA_A,
            v5_stage="stage_a_natural_feasibility",
            schedule=_stage_a_schedule(),
            expected_lock_sha256=artifact["provenance"]["source_lock_sha256"],
        )
        provenance = artifact.get("provenance", {})
        if provenance.get("runtime") != _runtime():
            raise RuntimeError("artifact runtime differs from live locked runtime")
        if provenance.get("source_sha256") != _source_hashes():
            raise RuntimeError("artifact source hashes differ from live locked sources")
        if observed_hash != provenance.get("source_lock_sha256"):
            raise RuntimeError("artifact source-lock hash mismatch")
        return True, None
    except Exception as error:
        return False, str(error)


def _scale_zero_prefix_preview_valid(prefix: dict) -> tuple[bool, bool]:
    """Return (all preview mechanics valid, all task metadata valid)."""

    groups = {int(group["group"]): group for group in prefix["groups"]}
    previews = prefix["previews"]
    mechanical = all(
        record.get("applied") is False
        and record.get("mutated") is False
        and record.get("frozen_group_parameters") is True
        and math.isfinite(float(record.get("gradient_norm", math.nan)))
        and float(record["gradient_norm"]) > 0.0
        and math.isfinite(float(record.get("hypothetical_update_norm", math.nan)))
        and float(record["hypothetical_update_norm"]) > 0.0
        for record in previews
    )
    metadata = all(
        int(record.get("after_group", -1)) in groups
        and record.get("transitions")
        == groups[int(record["after_group"])].get("transition_end")
        and int(record.get("requested_task", -1))
        == int(groups[int(record["after_group"])]["task_id"])
        and 0
        <= int(record.get("credited_task", -1))
        < int(record.get("requested_task", -1))
        < len(locked.THRESHOLDS)
        for record in previews
    )
    return mechanical, metadata


def compute_stage_a_gates(artifact: dict, lock_path: Path) -> dict:
    """Compute V5A launch gates without reading any learning outcome field."""

    lock_ok, lock_error = _stage_a_lock_check(artifact, lock_path)
    recomputed_fixture = _deterministic_scale_fixture()
    saved_fixture = artifact.get("deterministic_scale_fixture")
    fixture_ok = saved_fixture == recomputed_fixture and recomputed_fixture.get("passed") is True
    expected_configs = {condition.name: asdict(condition) for condition in _conditions()}
    schedule_errors: list[str] = []
    if artifact.get("schema") != SCHEMA_A:
        schedule_errors.append("artifact schema mismatch")
    if artifact.get("protocol") != _protocol(
        "stage_a_natural_feasibility", _stage_a_schedule()
    ):
        schedule_errors.append("artifact protocol mismatch")
    if tuple(artifact.get("cases", {})) != STAGE_A_CASES:
        schedule_errors.append("V5A case order/set mismatch")
    all_runs: list[tuple[str, float, dict]] = []
    for multiplier in LR_MULTIPLIERS:
        for scale in HINDSIGHT_SCALES:
            name = _case_name(multiplier, scale)
            record = artifact.get("cases", {}).get(name, {})
            if record.get("config") != expected_configs[name]:
                schedule_errors.append(f"condition config mismatch for {name}")
            runs = record.get("runs", [])
            if [run.get("seed") for run in runs] != list(STAGE_A_SEEDS):
                schedule_errors.append(f"seed order mismatch for {name}")
            all_runs.extend((name, scale, run) for run in runs)

    counts = [
        int(run.get("optimizer_updates", -1))
        for _, _, run in all_runs
        if run.get("numeric_valid") is True
    ]
    selected = _select_update_budget(counts)

    prefixes: dict[str, dict] = {}
    invariant_details: dict[str, dict] = {}
    if selected is not None:
        for case_name, scale, run in all_runs:
            label = f"{case_name}/seed_{run.get('seed')}"
            errors: list[str] = []
            try:
                prefix = _prefix(run, selected)
                errors.extend(_prefix_technical_errors(prefix, run))
                prefixes[label] = {"scale": scale, **prefix}
            except Exception as error:
                errors.append(str(error))
            invariant_details[label] = {"passed": not errors, "errors": errors}

    gate1 = {
        "passed": bool(
            lock_ok
            and fixture_ok
            and not schedule_errors
            and selected is not None
            and len(all_runs) == 27
            and len(invariant_details) == 27
            and all(item["passed"] for item in invariant_details.values())
        ),
        "source_runtime_lock_passed": lock_ok,
        "source_runtime_lock_error": lock_error,
        "deterministic_nine_cell_scale_fixture_passed": fixture_ok,
        "deterministic_nine_cell_scale_fixture": saved_fixture,
        "schedule_errors": schedule_errors,
        "per_run_technical_invariants": invariant_details,
    }
    gate2 = {
        "passed": bool(selected is not None and len(prefixes) == 27),
        "selection_rule": (
            "400 iff all 27 reach 400; else 250 iff all 27 reach 250; else STOP"
        ),
        "full_run_update_counts": counts,
        "selected_update_budget": selected,
        "selection_population_size": len(counts),
    }

    scale_zero_details: dict[str, dict] = {}
    scale_zero_passes: list[bool] = []
    if gate2["passed"]:
        for label, prefix in prefixes.items():
            if prefix["scale"] != 0.0:
                continue
            candidates = prefix["candidate_groups"]
            previews = prefix["previews"]
            preview_ids = [int(record.get("after_group", -1)) for record in previews]
            valid, task_metadata_valid = _scale_zero_prefix_preview_valid(prefix)
            source_counts = prefix["source_step_norms"]
            passed = bool(
                candidates == preview_ids
                and valid
                and task_metadata_valid
                and source_counts["hindsight_relabel"]["count"] == 0
                and source_counts["requested_live"]["count"] == selected
            )
            scale_zero_passes.append(passed)
            scale_zero_details[label] = {
                "passed": passed,
                "candidate_count": len(candidates),
                "preview_count": len(previews),
                "candidate_groups_equal_preview_groups": candidates == preview_ids,
                "all_previews_positive_finite_nonmutating": valid,
                "all_preview_requested_and_credited_tasks_valid": task_metadata_valid,
                "source_step_norms": source_counts,
            }
    shadow = artifact.get("preview_shadow_equivalence", {})
    gate3 = {
        "passed": bool(
            gate2["passed"]
            and len(scale_zero_passes) == 9
            and all(scale_zero_passes)
            and shadow.get("passed") is True
        ),
        "per_scale_zero_run": scale_zero_details,
        "preview_shadow_equivalence": shadow,
    }

    positive_details: dict[str, dict] = {}
    positive_passes: list[bool] = []
    natural_cell_details: dict[str, dict] = {}
    natural_cell_passes: list[bool] = []
    if gate2["passed"]:
        for label, prefix in prefixes.items():
            if prefix["scale"] <= 0.0:
                continue
            candidates = prefix["candidate_groups"]
            updates_by_group = {
                int(record["after_group"]): record
                for record in prefix["updates"]
                if record.get("source") == "hindsight_relabel"
            }
            valid_records = all(
                group_id in updates_by_group
                and math.isfinite(
                    float(updates_by_group[group_id].get("gradient_norm", math.nan))
                )
                and float(updates_by_group[group_id]["gradient_norm"]) > 0.0
                and math.isfinite(
                    float(updates_by_group[group_id].get("update_norm", math.nan))
                )
                and float(updates_by_group[group_id]["update_norm"]) > 0.0
                and int(updates_by_group[group_id].get("credited_task", -1))
                < int(updates_by_group[group_id].get("requested_task", -1))
                for group_id in candidates
            )
            one_to_one = set(candidates) == set(updates_by_group) and len(candidates) == len(
                updates_by_group
            )
            no_preview_only = not prefix["previews"]
            passed = bool(valid_records and one_to_one and no_preview_only)
            positive_passes.append(passed)
            positive_details[label] = {
                "passed": passed,
                "accepted_candidate_count": len(candidates),
                "applied_hindsight_update_count": len(updates_by_group),
                "one_to_one_candidate_to_applied_update": one_to_one,
                "all_applied_auxiliary_updates_positive_finite": valid_records,
                "preview_only_record_count": len(prefix["previews"]),
            }
        for case_name in STAGE_A_CASES:
            candidate_count = sum(
                len(prefix["candidate_groups"])
                for label, prefix in prefixes.items()
                if label.startswith(case_name + "/")
            )
            passed = candidate_count >= MIN_NATURAL_RELABELS_PER_CELL
            natural_cell_passes.append(passed)
            scale = float(artifact["cases"][case_name]["config"]["hindsight_scale"])
            natural_cell_details[case_name] = {
                "passed": passed,
                "natural_event_type": (
                    "eligible_nonmutating_preview_candidate"
                    if scale == 0.0
                    else "finite_positive_applied_hindsight_update"
                ),
                "natural_events_pooled_across_three_seeds": candidate_count,
                "minimum_required": MIN_NATURAL_RELABELS_PER_CELL,
            }
    gate4 = {
        "passed": bool(
            gate2["passed"]
            and len(positive_passes) == 18
            and all(positive_passes)
            and len(natural_cell_passes) == 9
            and all(natural_cell_passes)
        ),
        "per_positive_scale_run": positive_details,
        "per_cell_natural_relabel_coverage": natural_cell_details,
    }

    regime_details: dict[str, dict] = {}
    regime_passes: list[bool] = []
    if gate2["passed"]:
        for case_name in STAGE_A_CASES:
            observed = sorted(
                {
                    group["regime"]
                    for label, prefix in prefixes.items()
                    if label.startswith(case_name + "/")
                    for group in prefix["groups"]
                }
            )
            passed = set(observed) == {"dead", "mixed", "all_pass"}
            regime_passes.append(passed)
            regime_details[case_name] = {"passed": passed, "observed_regimes": observed}
    gate5 = {
        "passed": bool(
            gate2["passed"] and len(regime_passes) == 9 and all(regime_passes)
        ),
        "per_cell": regime_details,
    }

    tv_details: dict[str, dict] = {}
    tv_passes: list[bool] = []
    if gate2["passed"]:
        for label, prefix in prefixes.items():
            values = [
                float(group["teacher_tv_from_uniform"])
                for group in prefix["groups"]
                if int(group["transition_start"]) >= TV_WARMUP_TRANSITIONS
            ]
            mean_tv = _teacher_tv_mean(values)
            passed = _teacher_tv_pass(values)
            tv_passes.append(passed)
            tv_details[label] = {
                "passed": passed,
                "n_groups": len(values),
                "mean_teacher_tv_from_uniform": mean_tv,
            }
    gate6 = {
        "passed": bool(gate2["passed"] and len(tv_passes) == 27 and all(tv_passes)),
        "warmup_transition_start_inclusive": TV_WARMUP_TRANSITIONS,
        "strict_minimum_mean_tv": 0.05,
        "per_run": tv_details,
    }

    prefix_walls = {label: prefix["wall_seconds"] for label, prefix in prefixes.items()}
    slowest = max(prefix_walls.values()) if len(prefix_walls) == 27 else None
    projected = None if slowest is None else _projected_hours_180(slowest)
    gate7 = {
        "passed": bool(projected is not None and projected <= MAX_PROJECTED_HOURS_180),
        "formula": "180 * max_j(wall_seconds_through_tau_j(U*)) / 3600",
        "per_run_prefix_wall_seconds": prefix_walls,
        "maximum_prefix_wall_seconds": slowest,
        "projected_hours_180": projected,
        "maximum_allowed_hours": MAX_PROJECTED_HOURS_180,
    }

    gates = {
        "learning_outcome_field_blind": True,
        "learning_outcome_fields_read": False,
        "uses_hindsight_outcome_contrast": False,
        "uses_v4_outcomes_or_budget": False,
        "selected_update_budget": selected,
        "gate_1_lock_schedule_and_technical_invariants": gate1,
        "gate_2_fresh_all_27_first_exact_prefix": gate2,
        "gate_3_scale_zero_preview_and_shadow": gate3,
        "gate_4_positive_updates_and_natural_relabel_coverage": gate4,
        "gate_5_per_cell_natural_regimes": gate5,
        "gate_6_per_run_teacher_tv": gate6,
        "gate_7_serial_runtime_projection": gate7,
        "outcome_exclusion": {
            "forbidden_field_names": sorted(OUTCOME_FIELD_NAMES),
            "gate_projection_contains_forbidden_field": False,
        },
    }
    gate_names = tuple(key for key in gates if key.startswith("gate_"))
    gates["all_pass"] = bool(gate_names and all(gates[key]["passed"] for key in gate_names))
    gates["stage_b_authorized"] = gates["all_pass"]
    gates["technical_prefix_digest_sha256"] = _canonical_hash(
        {
            label: {
                "terminal_group": prefix["terminal_group"],
                "terminal_transitions": prefix["terminal_transitions"],
                "candidate_groups": prefix["candidate_groups"],
                "source_step_norms": prefix["source_step_norms"],
            }
            for label, prefix in prefixes.items()
        }
    )
    return gates


def _artifact_identity(
    artifact: dict, *, stage: str, schedule: dict, lock_hash: str
) -> None:
    expected_schema = SCHEMA_A if stage == "stage_a_natural_feasibility" else SCHEMA_B
    if artifact.get("schema") != expected_schema:
        raise RuntimeError("resume artifact schema mismatch")
    if artifact.get("protocol") != _protocol(stage, schedule):
        raise RuntimeError("resume artifact protocol mismatch")
    provenance = artifact.get("provenance", {})
    if provenance.get("source_lock_sha256") != lock_hash:
        raise RuntimeError("resume artifact was created under a different source lock")
    if provenance.get("runtime") != _runtime() or provenance.get(
        "source_sha256"
    ) != _source_hashes():
        raise RuntimeError("resume artifact runtime/source mismatch")
    if tuple(artifact.get("cases", {})) != CASES:
        raise RuntimeError("resume artifact case order/set mismatch")
    expected_configs = {condition.name: asdict(condition) for condition in _conditions()}
    for name in CASES:
        if artifact["cases"][name].get("config") != expected_configs[name]:
            raise RuntimeError(f"resume condition mismatch for {name}")


def _exact_seed_prefix(runs: Sequence[dict], seeds: Sequence[int], label: str) -> int:
    observed = [run.get("seed") for run in runs]
    if observed != list(seeds[: len(observed)]):
        raise RuntimeError(f"resume records for {label} are not an exact seed prefix")
    return len(observed)


def _verify_stage_b_authorization(
    *,
    lock: dict,
    stage_a_artifact_path: Path,
    amendment_path: Path,
    selected_update_budget: int,
) -> tuple[dict, Path, str]:
    amendment = _read_json(amendment_path)
    if amendment.get("schema") != AMENDMENT_SCHEMA_B:
        raise RuntimeError("unexpected V5B amendment schema")
    if amendment.get("explicit_stage_b_authorization") is not True:
        raise RuntimeError("V5B amendment lacks explicit authorization")
    if amendment.get("selected_update_budget") != selected_update_budget:
        raise RuntimeError("V5B amendment carries a different selected budget")
    if amendment.get("registered_schedule") != _stage_b_schedule(selected_update_budget):
        raise RuntimeError("V5B amendment schedule mismatch")
    stage_a_artifact_path = stage_a_artifact_path.resolve()
    if amendment.get("stage_a_artifact_sha256") != _sha256(stage_a_artifact_path):
        raise RuntimeError("V5A artifact changed after V5B authorization")
    stage_a = _read_json(stage_a_artifact_path)
    saved_gates = stage_a.get("stage_a_learning_outcome_blind_gates", {})
    if (
        stage_a.get("schema") != SCHEMA_A
        or stage_a.get("artifact_state") != "complete"
        or saved_gates.get("all_pass") is not True
        or saved_gates.get("selected_update_budget") != selected_update_budget
    ):
        raise RuntimeError("V5A artifact does not authorize this V5B budget")
    verification_path = Path(amendment.get("stage_a_independent_verification", "")).resolve()
    verification_hash = _sha256(verification_path)
    if amendment.get("stage_a_independent_verification_sha256") != verification_hash:
        raise RuntimeError("independent V5A report changed after authorization")
    verification = _read_json(verification_path)
    requirements = (
        verification.get("schema") == VERIFICATION_SCHEMA_A,
        verification.get("v5_stage") == "stage_a_natural_feasibility",
        verification.get("all_checks_passed") is True,
        verification.get("stage_b_factorial_authorized") is True,
        verification.get("selected_optimizer_update_budget") == selected_update_budget,
        verification.get("artifact_sha256") == _sha256(stage_a_artifact_path),
        verification.get("runner_saved_gates_sha256") == _canonical_hash(saved_gates),
        verification.get("outcome_exclusion_audit", {}).get("passed") is True,
    )
    if not all(requirements):
        raise RuntimeError("independent V5A report no longer authorizes V5B")
    if lock.get("stage_a_artifact_sha256") != _sha256(stage_a_artifact_path):
        raise RuntimeError("V5B lock carries a different V5A artifact hash")
    if lock.get("stage_a_independent_verification_sha256") != verification_hash:
        raise RuntimeError("V5B lock carries a different independent report hash")
    if lock.get("stage_a_gates_sha256") != _canonical_hash(saved_gates):
        raise RuntimeError("V5B lock carries a different V5A gate hash")
    return amendment, verification_path, verification_hash


def _run_stage(
    *,
    stage: str,
    lock_path: Path,
    output: Path,
    resume: bool,
    selected_update_budget: int,
    stage_a_artifact_path: Path | None = None,
    amendment_path: Path | None = None,
) -> dict:
    schedule = (
        _stage_a_schedule()
        if stage == "stage_a_natural_feasibility"
        else _stage_b_schedule(selected_update_budget)
    )
    schema = LOCK_SCHEMA_A if stage == "stage_a_natural_feasibility" else LOCK_SCHEMA_B
    lock, lock_hash = _verify_source_lock(
        lock_path,
        schema=schema,
        v5_stage=stage,
        schedule=schedule,
        amendment_path=amendment_path,
    )
    verification_path: Path | None = None
    verification_hash: str | None = None
    if stage == "stage_b_confirmatory_factorial":
        if stage_a_artifact_path is None or amendment_path is None:
            raise ValueError("V5B requires V5A artifact and amendment paths")
        _, verification_path, verification_hash = _verify_stage_b_authorization(
            lock=lock,
            stage_a_artifact_path=stage_a_artifact_path,
            amendment_path=amendment_path,
            selected_update_budget=selected_update_budget,
        )

    output = output.resolve()
    if output.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}; use --resume")
        artifact = _read_json(output)
        _artifact_identity(artifact, stage=stage, schedule=schedule, lock_hash=lock_hash)
        if artifact.get("artifact_state") != "in_progress":
            print(f"artifact is already terminal: {output}")
            return artifact
    else:
        if resume:
            raise FileNotFoundError(f"cannot resume missing artifact {output}")
        artifact = _new_artifact(
            stage=stage, schedule=schedule, lock_path=lock_path, lock_hash=lock_hash
        )
        if stage == "stage_b_confirmatory_factorial":
            artifact.update(
                {
                    "stage_a_artifact": str(stage_a_artifact_path.resolve()),
                    "stage_a_artifact_sha256": _sha256(stage_a_artifact_path),
                    "stage_b_amendment": str(amendment_path.resolve()),
                    "stage_b_amendment_sha256": _sha256(amendment_path),
                    "stage_a_independent_verification": str(verification_path),
                    "stage_a_independent_verification_sha256": verification_hash,
                }
            )
        _atomic_write(output, artifact, must_not_exist=True)

    if stage == "stage_a_natural_feasibility":
        live_fixture = _deterministic_scale_fixture()
        if "deterministic_scale_fixture" not in artifact:
            artifact["deterministic_scale_fixture"] = live_fixture
            _atomic_write(output, artifact, must_not_exist=False)
            print(
                "completed V5 deterministic nine-cell scale fixture: "
                f"passed={live_fixture['passed']}",
                flush=True,
            )
        elif artifact["deterministic_scale_fixture"] != live_fixture:
            raise RuntimeError(
                "saved deterministic scale fixture differs from exact recomputation"
            )
        if live_fixture.get("passed") is not True:
            raise RuntimeError(
                "deterministic nine-cell scale fixture failed before registered seeds"
            )

    if stage == "stage_a_natural_feasibility":
        live_shadow = _preview_shadow_equivalence()
        if "preview_shadow_equivalence" not in artifact:
            artifact["preview_shadow_equivalence"] = live_shadow
            _atomic_write(output, artifact, must_not_exist=False)
            print(
                "completed V5 deterministic scale-zero/no-hindsight shadow: "
                f"passed={live_shadow['passed']}",
                flush=True,
            )
        elif artifact["preview_shadow_equivalence"] != live_shadow:
            raise RuntimeError(
                "saved preview shadow differs from exact deterministic recomputation"
            )
        if live_shadow.get("passed") is not True:
            raise RuntimeError("preview/no-hindsight shadow failed before registered seeds")

    seeds = STAGE_A_SEEDS if stage == "stage_a_natural_feasibility" else STAGE_B_SEEDS
    budget = locked.RunBudget(
        optimizer_update_budget=selected_update_budget,
        transition_safety_cap=TRANSITION_GROUP_START_CAP,
    )
    total = len(CASES) * len(seeds)
    completed = sum(len(record["runs"]) for record in artifact["cases"].values())
    for condition in _conditions():
        runs = artifact["cases"][condition.name]["runs"]
        start = _exact_seed_prefix(runs, seeds, condition.name)
        for seed in seeds[start:]:
            try:
                run = _instrumented_run(
                    condition, seed, budget=budget, eval_n=EVAL_EPISODES_PER_TASK
                )
            except Exception as error:
                run = {
                    "seed": seed,
                    "numeric_valid": False,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                    "partial_progress_available": False,
                }
                artifact["run_failures"].append({"condition": condition.name, **run})
            runs.append(run)
            completed += 1
            _atomic_write(output, artifact, must_not_exist=False)
            print(
                f"V5 {stage} technical progress: {completed}/{total}; "
                f"latest={condition.name}/seed_{seed}; valid={run.get('numeric_valid', False)}",
                flush=True,
            )

    if stage == "stage_a_natural_feasibility":
        artifact["stage_a_learning_outcome_blind_gates"] = compute_stage_a_gates(
            artifact, lock_path
        )
        artifact["analysis_status"] = {
            "performed": True,
            "type": "learning-outcome-field-blind natural-mechanics feasibility only",
            "learning_outcome_fields_used": False,
        }
        artifact["artifact_state"] = (
            "complete" if not artifact["run_failures"] else "complete_with_invalid_runs"
        )
    else:
        _attach_stage_b_analysis(artifact, lock_path, amendment_path)
    _atomic_write(output, artifact, must_not_exist=False)
    print(f"wrote {output}")
    return artifact


def _full_relabel_errors(run: dict, scale: float) -> list[str]:
    """Reconstruct full-run relabel mechanics for confirmatory validity."""

    errors: list[str] = []
    candidates = run.get("eligible_relabel_candidate_groups")
    if (
        not isinstance(candidates, list)
        or candidates != sorted(set(candidates))
        or len(candidates) != run.get("relabel_candidates")
    ):
        return ["eligible relabel candidate ids/count are invalid"]
    groups = {group["group"]: group for group in run.get("group_diagnostics", [])}
    if any(
        group_id not in groups or groups[group_id].get("regime") != "dead"
        for group_id in candidates
    ):
        errors.append("eligible relabel candidate does not name a dead group")
    updates = {
        int(record["after_group"]): record
        for record in run.get("update_diagnostics", [])
        if record.get("source") == "hindsight_relabel"
    }
    previews = run.get("auxiliary_gradient_diagnostics")
    if not isinstance(previews, list):
        return errors + ["auxiliary-gradient diagnostics are missing"]
    zero_records = run.get("zero_gradient_diagnostics")
    if (
        not isinstance(zero_records, list)
        or len(zero_records) != run.get("zero_gradient_update_attempts")
    ):
        return errors + ["zero-gradient attempt diagnostics/count are invalid"]
    zero_groups: set[int] = set()
    zero_hindsight: dict[int, dict] = {}
    for record in zero_records:
        group_id = int(record.get("after_group", -1))
        source = record.get("source")
        group = groups.get(group_id)
        gradient = float(record.get("gradient_norm", math.nan))
        update_norm = float(record.get("update_norm", math.nan))
        requested = int(record.get("requested_task", -1))
        credited = int(record.get("credited_task", -1))
        if (
            group is None
            or group_id in zero_groups
            or group.get("update_source") is not None
            or source not in ("requested_live", "hindsight_relabel")
            or record.get("transitions") != group.get("transition_end")
            or not math.isfinite(gradient)
            or gradient != 0.0
            or not math.isfinite(update_norm)
            or update_norm != 0.0
        ):
            errors.append("zero-gradient attempt record is malformed")
            continue
        if source == "requested_live" and not (
            group.get("regime") == "mixed"
            and requested == credited == int(group["task_id"])
        ):
            errors.append("zero-gradient requested-live metadata/regime is invalid")
        if source == "hindsight_relabel" and not (
            group.get("regime") == "dead"
            and group_id in candidates
            and requested == int(group["task_id"])
            and 0 <= credited < requested < len(locked.THRESHOLDS)
        ):
            errors.append("zero-gradient hindsight metadata/regime is invalid")
        zero_groups.add(group_id)
        if source == "hindsight_relabel":
            zero_hindsight[group_id] = record
    if scale == 0.0:
        preview_ids = [int(record.get("after_group", -1)) for record in previews]
        valid = all(
            record.get("applied") is False
            and record.get("mutated") is False
            and record.get("frozen_group_parameters") is True
            and math.isfinite(float(record.get("gradient_norm", math.nan)))
            and float(record["gradient_norm"]) > 0.0
            and math.isfinite(float(record.get("hypothetical_update_norm", math.nan)))
            and float(record["hypothetical_update_norm"]) > 0.0
            and int(record.get("requested_task", -1))
            == int(groups.get(int(record.get("after_group", -1)), {}).get("task_id", -2))
            and record.get("transitions")
            == groups.get(int(record.get("after_group", -1)), {}).get("transition_end")
            and 0
            <= int(record.get("credited_task", -1))
            < int(record.get("requested_task", -1))
            < len(locked.THRESHOLDS)
            for record in previews
        )
        if candidates != preview_ids or not valid:
            errors.append("scale-zero candidates/previews are not exact positive nonmutating pairs")
        if updates or run.get("relabeled_groups") != 0:
            errors.append("scale-zero run applied a hindsight update")
    else:
        valid = all(
            math.isfinite(float(record.get("gradient_norm", math.nan)))
            and float(record["gradient_norm"]) > 0.0
            and math.isfinite(float(record.get("update_norm", math.nan)))
            and float(record["update_norm"]) > 0.0
            and int(record.get("requested_task", -1))
            == int(groups.get(int(record.get("after_group", -1)), {}).get("task_id", -2))
            and record.get("transitions")
            == groups.get(int(record.get("after_group", -1)), {}).get("transition_end")
            and 0
            <= int(record.get("credited_task", -1))
            < int(record.get("requested_task", -1))
            < len(locked.THRESHOLDS)
            for record in updates.values()
        )
        if (
            set(updates) & set(zero_hindsight)
            or candidates != sorted([*updates, *zero_hindsight])
            or len(candidates) != len(updates) + len(zero_hindsight)
            or not valid
        ):
            errors.append(
                "positive-scale candidates do not partition into valid applied and zero-gradient hindsight attempts"
            )
        if previews or run.get("unscaled_aux_gradient_previews") != 0:
            errors.append("positive-scale run contains preview-only diagnostics")
    return errors


def _stage_b_runs_ready(artifact: dict, target: int) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if tuple(artifact.get("cases", {})) != STAGE_B_CASES:
        return False, ["V5B case order/set mismatch"]
    for case_name in STAGE_B_CASES:
        scale = float(artifact["cases"][case_name].get("config", {}).get("hindsight_scale", math.nan))
        runs = artifact["cases"][case_name].get("runs", [])
        if [run.get("seed") for run in runs] != list(STAGE_B_SEEDS):
            errors.append(f"seed order/completeness mismatch for {case_name}")
            continue
        for run in runs:
            label = f"{case_name}/seed_{run.get('seed')}"
            technical_errors = _technical_run_errors(run)
            technical_errors.extend(_full_relabel_errors(run, scale))
            if technical_errors:
                errors.append(f"{label}: " + "; ".join(technical_errors))
            if run.get("transition_cap_censored") is not False:
                errors.append(f"{label}: transition-cap censored")
            if run.get("reached_optimizer_update_budget") is not True:
                errors.append(f"{label}: did not reach frozen update target")
            if run.get("optimizer_updates") != target:
                errors.append(f"{label}: final update coordinate is not U*")
            expected_updates = list(range(0, target + 1, EVAL_INTERVAL_UPDATES))
            if run.get("x_optimizer_updates") != expected_updates:
                errors.append(
                    f"{label}: evaluation updates are not exactly {expected_updates}"
                )
            x_transitions = run.get("x_transitions", [])
            if (
                len(x_transitions) != len(expected_updates)
                or not x_transitions
                or x_transitions[0] != 0
                or x_transitions[-1] != run.get("transitions")
                or any(
                    right <= left for left, right in zip(x_transitions, x_transitions[1:])
                )
            ):
                errors.append(f"{label}: evaluation transition coordinates are invalid")
    return not errors, errors


def _recomputed_update_auc(run: dict, target: int) -> float:
    x = np.asarray(run["x_optimizer_updates"], dtype=np.float64)
    y = np.asarray(run["mean_pass_curve"], dtype=np.float64)
    if (
        len(x) != len(y)
        or len(x) < 2
        or x[0] != 0
        or x[-1] != target
        or np.any(np.diff(x) <= 0)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.any((y < 0.0) | (y > 1.0))
    ):
        raise ValueError("invalid update-indexed mean-pass AUC curve")
    return float(np.trapezoid(y, x) / target)


def _contrast_values(
    per_case: dict[str, dict[int, float]], coefficients: dict[str, float]
) -> np.ndarray:
    return np.asarray(
        [
            sum(
                coefficient * per_case[case_name][seed]
                for case_name, coefficient in coefficients.items()
            )
            for seed in STAGE_B_SEEDS
        ],
        dtype=np.float64,
    )


def _exact_sign_flip_p(values: Sequence[float]) -> float:
    """Exact two-sided sign-flip p-value over all 2^20 assignments."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (20,) or not np.isfinite(array).all():
        raise ValueError("V5B exact sign-flip requires exactly 20 finite pairs")
    observed = abs(float(array.mean()))
    extreme = 0
    total = 1 << 20
    powers = np.arange(20, dtype=np.uint64)
    for start in range(0, total, 1 << 15):
        masks = np.arange(start, min(start + (1 << 15), total), dtype=np.uint64)
        bits = ((masks[:, None] >> powers[None, :]) & 1).astype(np.float64)
        signs = bits * 2.0 - 1.0
        statistics = np.abs(signs @ array / 20.0)
        extreme += int(np.count_nonzero(statistics >= observed - 1e-15))
    return float(extreme / total)


def _holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    ordered = sorted((float(value), name) for name, value in p_values.items())
    running = 0.0
    still_rejecting = True
    result: dict[str, dict] = {}
    count = len(ordered)
    for rank, (p_value, name) in enumerate(ordered, start=1):
        multiplier = count - rank + 1
        running = max(running, multiplier * p_value)
        reject = still_rejecting and p_value <= alpha / multiplier
        if not reject:
            still_rejecting = False
        result[name] = {
            "raw_p": p_value,
            "holm_adjusted_p": float(min(running, 1.0)),
            "reject_familywise_0.05": bool(reject),
        }
    return result


CONTRAST_SPECS = {
    "C1": {_case_name(1.0, 1.0): 1.0, _case_name(1.0, 0.0): -1.0},
    "C2": {_case_name(1.0, 2.0): 1.0, _case_name(1.0, 1.0): -1.0},
    "C3": {
        _case_name(0.5, 2.0): 1.0,
        _case_name(0.5, 0.0): -1.0,
        _case_name(1.0, 1.0): -1.0,
        _case_name(1.0, 0.0): 1.0,
    },
    "C4": {
        _case_name(1.0, 2.0): 1.0,
        _case_name(1.0, 0.0): -1.0,
        _case_name(2.0, 1.0): -1.0,
        _case_name(2.0, 0.0): 1.0,
    },
}
CONTRAST_DESCRIPTIONS = {
    "C1": "base-LR scale 1 minus scale 0",
    "C2": "base-LR scale 2 minus scale 1",
    "C3": "half/base LR restricted-separability departure",
    "C4": "base/double LR restricted-separability departure",
}


def _predeclared_scale_decision(contrasts: dict[str, dict]) -> dict:
    return {
        "C1_directional_local_improvement_supported": bool(
            contrasts["C1"]["mean_contrast"] >= 0.03
            and contrasts["C1"]["reject_familywise_0.05"]
        ),
        "C2_directional_increment_supported": bool(
            contrasts["C2"]["mean_contrast"] >= 0.03
            and contrasts["C2"]["reject_familywise_0.05"]
        ),
        "C3_material_restricted_separability_departure": bool(
            abs(contrasts["C3"]["mean_contrast"]) >= 0.03
            and contrasts["C3"]["reject_familywise_0.05"]
        ),
        "C4_material_restricted_separability_departure": bool(
            abs(contrasts["C4"]["mean_contrast"]) >= 0.03
            and contrasts["C4"]["reject_familywise_0.05"]
        ),
        "directional_minimum_mean": 0.03,
        "restricted_departure_minimum_absolute_mean": 0.03,
        "interpretation_boundary": (
            "C3/C4 test departure from Y(a,s)=F(a)+G(a*s); they do not isolate "
            "semantic data value from optimizer magnitude, update-source composition, "
            "natural relabel frequency, or the induced policy path."
        ),
    }


def _attach_stage_b_analysis(
    artifact: dict, lock_path: Path, amendment_path: Path
) -> None:
    target = int(artifact["protocol"]["registered_schedule"]["optimizer_update_target"])
    errors: list[str] = []
    try:
        _verify_source_lock(
            lock_path,
            schema=LOCK_SCHEMA_B,
            v5_stage="stage_b_confirmatory_factorial",
            schedule=_stage_b_schedule(target),
            expected_lock_sha256=artifact["provenance"]["source_lock_sha256"],
            amendment_path=amendment_path,
        )
    except Exception as error:
        errors.append(str(error))
    ready, readiness_errors = _stage_b_runs_ready(artifact, target)
    errors.extend(readiness_errors)
    if not ready or errors:
        artifact["analysis_status"] = {
            "performed": False,
            "all_or_nothing_family_invalidated": True,
            "reason": "at least one of the 180 registered runs/locks failed",
            "errors": errors,
        }
        artifact["artifact_state"] = "complete_with_invalid_primary_family"
        return

    per_case: dict[str, dict[int, float]] = {}
    summaries: dict[str, dict] = {}
    for case_name in STAGE_B_CASES:
        seed_values: dict[int, float] = {}
        source_diagnostics: dict[str, dict] = {}
        transitions: dict[str, int] = {}
        for run in artifact["cases"][case_name]["runs"]:
            seed = int(run["seed"])
            auc = _recomputed_update_auc(run, target)
            if not math.isclose(
                auc,
                float(run["auc_mean_pass_by_optimizer_updates"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(f"saved update AUC mismatch for {case_name}/seed_{seed}")
            seed_values[seed] = auc
            source_diagnostics[str(seed)] = _source_step_norms(run["update_diagnostics"])
            transitions[str(seed)] = int(run["transitions"])
        values = np.asarray([seed_values[seed] for seed in STAGE_B_SEEDS])
        per_case[case_name] = seed_values
        summaries[case_name] = {
            "metric": "auc_mean_pass_by_optimizer_updates",
            "n_seeds": 20,
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "per_seed": values.tolist(),
            "source_step_norms_per_seed": source_diagnostics,
            "transitions_to_target_per_seed": transitions,
        }

    contrasts: dict[str, dict] = {}
    raw_p: dict[str, float] = {}
    for index, (name, coefficients) in enumerate(CONTRAST_SPECS.items()):
        values = _contrast_values(per_case, coefficients)
        p_value = _exact_sign_flip_p(values)
        contrasts[name] = {
            "description": CONTRAST_DESCRIPTIONS[name],
            "coefficients": coefficients,
            "metric": "auc_mean_pass_by_optimizer_updates",
            "n_pairs": 20,
            "sign_assignments_enumerated": 1 << 20,
            "mean_contrast": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "mean_ci95_paired_seed_bootstrap": locked.bootstrap_mean_ci(
                values, seed=55_000 + index, n_boot=20_000
            ),
            "exact_paired_sign_flip_p_two_sided": p_value,
            "per_seed_contrast": values.tolist(),
        }
        raw_p[name] = p_value
    adjusted = _holm(raw_p)
    for name, correction in adjusted.items():
        contrasts[name].update(correction)

    artifact["stage_b_case_summaries"] = summaries
    artifact["paired_scale_contrasts"] = contrasts
    artifact["scale_multiplicity"] = {
        "family": ["C1", "C2", "C3", "C4"],
        "all_or_nothing_180_run_validation": True,
        "metric": "auc_mean_pass_by_optimizer_updates",
        "method": "Holm step-down",
        "familywise_alpha": 0.05,
        "test": "exact two-sided paired sign-flip over all 2^20 assignments",
    }
    artifact["predeclared_scale_decision"] = _predeclared_scale_decision(contrasts)
    artifact["analysis_status"] = {
        "performed": True,
        "all_180_runs_valid": True,
        "all_final_update_coordinates_equal_selected_budget": True,
        "selected_update_budget": target,
    }
    artifact["artifact_state"] = "complete"


def _authorize_stage_b(
    *,
    stage_a_artifact_path: Path,
    stage_a_verification_path: Path,
    amendment_output: Path,
    lock_output: Path,
    explicit_authorization: bool,
) -> None:
    if not explicit_authorization:
        raise RuntimeError("refusing V5B authorization without --authorize-stage-b")
    stage_a_artifact_path = stage_a_artifact_path.resolve()
    stage_a_verification_path = stage_a_verification_path.resolve()
    stage_a = _read_json(stage_a_artifact_path)
    if stage_a.get("artifact_state") != "complete":
        raise RuntimeError("V5A artifact is not complete and valid")
    lock_a_path = Path(stage_a["provenance"]["source_lock_path"])
    recomputed = compute_stage_a_gates(stage_a, lock_a_path)
    saved = stage_a.get("stage_a_learning_outcome_blind_gates")
    if saved != recomputed or recomputed.get("all_pass") is not True:
        raise RuntimeError("saved/recomputed V5A gates do not authorize V5B")
    selected = int(recomputed["selected_update_budget"])
    stage_a_hash = _sha256(stage_a_artifact_path)
    verification = _read_json(stage_a_verification_path)
    required = (
        verification.get("schema") == VERIFICATION_SCHEMA_A,
        verification.get("v5_stage") == "stage_a_natural_feasibility",
        verification.get("all_checks_passed") is True,
        verification.get("stage_b_factorial_authorized") is True,
        verification.get("selected_optimizer_update_budget") == selected,
        verification.get("artifact_sha256") == stage_a_hash,
        verification.get("lock_sha256")
        == stage_a["provenance"]["source_lock_sha256"],
        verification.get("runner_saved_gates_sha256") == _canonical_hash(saved),
        verification.get("outcome_exclusion_audit", {}).get("passed") is True,
        Path(verification.get("artifact", "")).resolve() == stage_a_artifact_path,
    )
    if not all(required):
        raise RuntimeError("independent V5A report is not authorization-ready")
    verification_hash = _sha256(stage_a_verification_path)
    schedule = _stage_b_schedule(selected)
    amendment = {
        "schema": AMENDMENT_SCHEMA_B,
        "v5_stage": "stage_b_confirmatory_factorial",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "explicit_stage_b_authorization": True,
        "stage_a_artifact": str(stage_a_artifact_path),
        "stage_a_artifact_sha256": stage_a_hash,
        "stage_a_independent_verification": str(stage_a_verification_path),
        "stage_a_independent_verification_sha256": verification_hash,
        "stage_a_gates_sha256": _canonical_hash(saved),
        "selected_update_budget": selected,
        "registered_schedule": schedule,
        "frozen_claim_rule": (
            "C1/C2 require mean>=+0.03 and Holm rejection; C3/C4 require "
            "abs(mean)>=0.03 and Holm rejection; complete 180-run family only."
        ),
        "exact_commands": {
            "stage_a_independent_verifier": _module_command(
                ANALYZER_MODULE,
                stage_a_artifact_path,
                "--lock",
                lock_a_path.resolve(),
                "--output",
                stage_a_verification_path,
            ),
            "runner": _module_command(
                "frontier_rl.examples.run_acrobot_hindsight_v5",
                "stage-b",
                "--stage-a-artifact",
                stage_a_artifact_path,
                "--amendment",
                amendment_output.resolve(),
                "--lock",
                lock_output.resolve(),
                "--output",
                DEFAULT_ARTIFACT_B.resolve(),
            ),
            "independent_verifier": _module_command(
                ANALYZER_MODULE,
                DEFAULT_ARTIFACT_B.resolve(),
                "--lock",
                lock_output.resolve(),
                "--output",
                DEFAULT_VERIFICATION_B.resolve(),
            ),
        },
    }
    _atomic_write(amendment_output, amendment, must_not_exist=True)
    lock = {
        "schema": LOCK_SCHEMA_B,
        "v5_stage": "stage_b_confirmatory_factorial",
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Pre-execution source/runtime/amendment lock for V5B.",
        "runtime": _runtime(),
        "engine_contract": _engine_contract(),
        "seed_collision_audit": _seed_collision_audit(),
        "source_sha256": _source_hashes(),
        "registered_schedule": schedule,
        "amendment_sha256": _sha256(amendment_output),
        "stage_a_artifact_sha256": stage_a_hash,
        "stage_a_independent_verification_sha256": verification_hash,
        "stage_a_gates_sha256": _canonical_hash(saved),
        "equality_rule": (
            "V5B refuses start/resume unless runtime, sources, schedule, amendment, "
            "V5A artifact/report, and gate hashes all match exactly."
        ),
    }
    _atomic_write(lock_output, lock, must_not_exist=True)
    print(f"wrote explicit V5B amendment: {amendment_output.resolve()}")
    print(f"sealed V5B source/runtime lock: {lock_output.resolve()}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal_a = subparsers.add_parser("seal-a", help="write a V5A source/runtime lock")
    seal_a.add_argument("--output", type=Path, default=DEFAULT_LOCK_A)

    stage_a = subparsers.add_parser(
        "stage-a", help="run fresh learning-outcome-field-blind V5A"
    )
    stage_a.add_argument("--lock", type=Path, default=DEFAULT_LOCK_A)
    stage_a.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_A)
    stage_a.add_argument("--resume", action="store_true")

    authorize_b = subparsers.add_parser(
        "authorize-b", help="write the post-gate V5B amendment and source lock"
    )
    authorize_b.add_argument("--stage-a-artifact", type=Path, required=True)
    authorize_b.add_argument("--stage-a-verification", type=Path, required=True)
    authorize_b.add_argument("--amendment-output", type=Path, default=DEFAULT_AMENDMENT_B)
    authorize_b.add_argument("--lock-output", type=Path, default=DEFAULT_LOCK_B)
    authorize_b.add_argument("--authorize-stage-b", action="store_true")

    stage_b = subparsers.add_parser("stage-b", help="run amendment-authorized V5B")
    stage_b.add_argument("--stage-a-artifact", type=Path, required=True)
    stage_b.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT_B)
    stage_b.add_argument("--lock", type=Path, default=DEFAULT_LOCK_B)
    stage_b.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_B)
    stage_b.add_argument("--resume", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "seal-a":
            _seal_stage_a(args.output)
        elif args.command == "stage-a":
            _run_stage(
                stage="stage_a_natural_feasibility",
                lock_path=args.lock,
                output=args.output,
                resume=args.resume,
                selected_update_budget=TARGET_UPDATES_A,
            )
        elif args.command == "authorize-b":
            _authorize_stage_b(
                stage_a_artifact_path=args.stage_a_artifact,
                stage_a_verification_path=args.stage_a_verification,
                amendment_output=args.amendment_output,
                lock_output=args.lock_output,
                explicit_authorization=args.authorize_stage_b,
            )
        elif args.command == "stage-b":
            amendment = _read_json(args.amendment)
            selected = int(amendment.get("selected_update_budget", -1))
            _run_stage(
                stage="stage_b_confirmatory_factorial",
                lock_path=args.lock,
                output=args.output,
                resume=args.resume,
                selected_update_budget=selected,
                stage_a_artifact_path=args.stage_a_artifact,
                amendment_path=args.amendment,
            )
        else:  # pragma: no cover - argparse makes this unreachable
            raise AssertionError(f"unhandled command {args.command!r}")
    except (FileNotFoundError, FileExistsError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
