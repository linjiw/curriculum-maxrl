"""Fail-closed runner for the preregistered Acrobot hindsight V4 study.

The numerical training loop remains the locked implementation in
``run_acrobot_neural.py``.  This module supplies only V4 orchestration,
instrumentation, source-lock enforcement, prefix-only feasibility gates, and
the frozen Stage-B analysis.

Typical workflow::

    # Only after this runner and the independent analyzer are final.
    python run_acrobot_hindsight_v4.py seal-a --output LOCK.json
    python run_acrobot_hindsight_v4.py stage-a --lock LOCK.json --output A.json

    # Only when every saved Stage-A effect-blind gate passes.
    python run_acrobot_hindsight_v4.py authorize-b \
        --stage-a-artifact A.json --stage-a-verification A_VERIFY.json \
        --amendment-output AMENDMENT.json \
        --lock-output B_LOCK.json --authorize-stage-b
    python run_acrobot_hindsight_v4.py stage-b --stage-a-artifact A.json \
        --amendment AMENDMENT.json --lock B_LOCK.json --output B.json

Neither execution command accepts seed, cell, budget, cadence, or model
overrides.  An interrupted artifact can be continued with ``--resume``;
completed and failed seed records are never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import sys
import tempfile
import time as _time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import gymnasium
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import frontier_rl.examples.run_acrobot_neural as locked


SCHEMA_A = "curriculum-maxrl/acrobot-hindsight-v4a-artifact/v1"
SCHEMA_B = "curriculum-maxrl/acrobot-hindsight-v4b-artifact/v1"
LOCK_SCHEMA_A = "curriculum-maxrl/acrobot-hindsight-v4a-source-lock/v1"
LOCK_SCHEMA_B = "curriculum-maxrl/acrobot-hindsight-v4b-source-lock/v1"
AMENDMENT_SCHEMA_B = "curriculum-maxrl/acrobot-hindsight-v4b-amendment/v1"

PROTOCOL_PATH = Path(__file__).with_name("ACROBOT_HINDSIGHT_PROTOCOL_V4.md")
ANALYZER_PATH = Path(__file__).with_name("analyze_acrobot_hindsight_v4.py")
DEFAULT_LOCK_A = Path(__file__).with_name("ACROBOT_HINDSIGHT_V4A_LOCK.json")
DEFAULT_ARTIFACT_A = Path(__file__).with_name(
    "acrobot_hindsight_v4a_feasibility.json"
)
DEFAULT_VERIFICATION_A = Path(__file__).with_name(
    "acrobot_hindsight_v4a_verification.json"
)
DEFAULT_AMENDMENT_B = Path(__file__).with_name(
    "ACROBOT_HINDSIGHT_V4B_AMENDMENT.json"
)
DEFAULT_LOCK_B = Path(__file__).with_name("ACROBOT_HINDSIGHT_V4B_LOCK.json")
DEFAULT_ARTIFACT_B = Path(__file__).with_name(
    "acrobot_hindsight_v4b_factorial.json"
)
DEFAULT_VERIFICATION_B = Path(__file__).with_name(
    "acrobot_hindsight_v4b_verification.json"
)

BASE_LEARNING_RATE = 3e-4
LR_MULTIPLIERS = (0.5, 1.0, 2.0)
HINDSIGHT_SCALES_A = (0.0,)
HINDSIGHT_SCALES_B = (0.0, 1.0, 2.0)
STAGE_A_SEEDS = tuple(range(13_000, 13_003))
STAGE_B_SEEDS = tuple(range(14_000, 14_010))
TARGET_UPDATES_A = 400
FALLBACK_UPDATES_A = 250
TRANSITION_GROUP_START_CAP = 4_000_000
MAX_COMPLETE_GROUP_OVERSHOOT = locked.N_ROLLOUTS * 500
EVAL_INTERVAL_UPDATES = 50
EVAL_EPISODES_PER_TASK = 32
TV_WARMUP_TRANSITIONS = 200_000
MIN_PREVIEWS_PER_RUN = 10
MAX_PROJECTED_HOURS_90 = 24.0

# This audit seed was already used by the earlier exploratory scale smoke run.
# It is deliberately outside both fresh V4 seed blocks.
SHADOW_SEED = 100
SHADOW_UPDATE_BUDGET = 3
SHADOW_TRANSITION_CAP = 80_000
SHADOW_EVAL_EPISODES = 4

SOURCE_RELATIVE_PATHS = (
    "frontier_rl/examples/run_acrobot_hindsight_v4.py",
    "frontier_rl/examples/analyze_acrobot_hindsight_v4.py",
    "frontier_rl/examples/test_analyze_acrobot_hindsight_v4.py",
    "frontier_rl/examples/test_run_acrobot_hindsight_v4.py",
    "frontier_rl/examples/ACROBOT_HINDSIGHT_PROTOCOL_V4.md",
    "frontier_rl/examples/run_acrobot_neural.py",
    "frontier_rl/adapters/acrobot_neural.py",
    "frontier_rl/teacher.py",
    "frontier_rl/estimators.py",
    "frontier_rl/interfaces.py",
    "frontier_rl/examples/test_acrobot_neural.py",
    "frontier_rl/examples/test_run_acrobot_neural.py",
)


def _float_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _case_name(multiplier: float, scale: float) -> str:
    return f"lr_mult_{_float_label(multiplier)}_hs_{_float_label(scale)}"


STAGE_A_CASES = tuple(_case_name(multiplier, 0.0) for multiplier in LR_MULTIPLIERS)
STAGE_B_CASES = tuple(
    _case_name(multiplier, scale)
    for multiplier in LR_MULTIPLIERS
    for scale in HINDSIGHT_SCALES_B
)


def _stage_a_schedule() -> dict:
    return {
        "v4_stage": "stage_a_feasibility",
        "paired_seeds": list(STAGE_A_SEEDS),
        "base_learning_rate": BASE_LEARNING_RATE,
        "learning_rate_multipliers": list(LR_MULTIPLIERS),
        "hindsight_scales": list(HINDSIGHT_SCALES_A),
        "condition_names": list(STAGE_A_CASES),
        "optimizer_update_target": TARGET_UPDATES_A,
        "single_allowed_fallback_update_target": FALLBACK_UPDATES_A,
        "transition_group_start_cap": TRANSITION_GROUP_START_CAP,
        "maximum_complete_group_overshoot": MAX_COMPLETE_GROUP_OVERSHOOT,
        "eval_interval_optimizer_updates": EVAL_INTERVAL_UPDATES,
        "eval_episodes_per_task": EVAL_EPISODES_PER_TASK,
        "shadow_test": {
            "seed": SHADOW_SEED,
            "learning_rate_multiplier": 1.0,
            "optimizer_update_budget": SHADOW_UPDATE_BUDGET,
            "transition_group_start_cap": SHADOW_TRANSITION_CAP,
            "eval_episodes_per_task": SHADOW_EVAL_EPISODES,
        },
    }


def _stage_b_schedule(selected_update_budget: int) -> dict:
    if selected_update_budget not in (FALLBACK_UPDATES_A, TARGET_UPDATES_A):
        raise ValueError("Stage B update budget must be the frozen 250 or 400")
    return {
        "v4_stage": "stage_b_factorial",
        "paired_seeds": list(STAGE_B_SEEDS),
        "base_learning_rate": BASE_LEARNING_RATE,
        "learning_rate_multipliers": list(LR_MULTIPLIERS),
        "hindsight_scales": list(HINDSIGHT_SCALES_B),
        "condition_names": list(STAGE_B_CASES),
        "optimizer_update_target": int(selected_update_budget),
        "transition_group_start_cap": TRANSITION_GROUP_START_CAP,
        "maximum_complete_group_overshoot": MAX_COMPLETE_GROUP_OVERSHOOT,
        "eval_interval_optimizer_updates": EVAL_INTERVAL_UPDATES,
        "eval_episodes_per_task": EVAL_EPISODES_PER_TASK,
    }


def _runtime() -> dict:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    missing = [relative for relative in SOURCE_RELATIVE_PATHS if not (PROJECT_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "cannot form/verify the V4 source lock; missing: " + ", ".join(missing)
        )
    return {
        relative: _sha256(PROJECT_ROOT / relative)
        for relative in SOURCE_RELATIVE_PATHS
    }


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"required preregistered file is missing: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _atomic_write(path: Path, payload: dict, *, must_not_exist: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if must_not_exist and path.exists():
        raise FileExistsError(f"refusing to overwrite existing file {path}")
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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


def _verify_source_lock(
    lock_path: Path,
    *,
    schema: str,
    v4_stage: str,
    schedule: dict,
    expected_lock_sha256: str | None = None,
    amendment_path: Path | None = None,
) -> tuple[dict, str]:
    """Verify live bytes and runtime before any registered seed is touched."""
    lock_path = lock_path.resolve()
    lock = _read_json(lock_path)
    lock_hash = _sha256(lock_path)
    errors = []
    if expected_lock_sha256 is not None and lock_hash != expected_lock_sha256:
        errors.append("source-lock file hash changed since artifact creation")
    if lock.get("schema") != schema:
        errors.append(f"lock schema must be {schema!r}")
    if lock.get("v4_stage") != v4_stage:
        errors.append(f"lock v4_stage must be {v4_stage!r}")
    if lock.get("runtime") != _runtime():
        errors.append(
            f"runtime mismatch: current={_runtime()!r}, locked={lock.get('runtime')!r}"
        )
    if lock.get("registered_schedule") != schedule:
        errors.append("registered schedule does not exactly match the frozen runner")
    locked_sources = lock.get("source_sha256")
    if not isinstance(locked_sources, dict):
        errors.append("lock source_sha256 must be an object")
    else:
        if set(locked_sources) != set(SOURCE_RELATIVE_PATHS):
            errors.append(
                "lock source key set must exactly cover runner, independent "
                "analyzer/test, protocol, imported locked sources, and base tests"
            )
        try:
            live_sources = _source_hashes()
        except Exception as error:
            errors.append(str(error))
        else:
            if locked_sources != live_sources:
                errors.append("one or more live source bytes differ from the lock")
    if amendment_path is not None:
        amendment_path = amendment_path.resolve()
        if lock.get("amendment_sha256") != _sha256(amendment_path):
            errors.append("Stage-B amendment hash differs from the source lock")
    if errors:
        raise RuntimeError("V4 source/runtime lock verification failed: " + "; ".join(errors))
    return lock, lock_hash


def _seal_stage_a(output: Path) -> None:
    output = output.resolve()
    lock = {
        "schema": LOCK_SCHEMA_A,
        "v4_stage": "stage_a_feasibility",
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Pre-execution source/runtime lock for hindsight-effect-blind V4A "
            "feasibility; creating this file does not touch V4 seeds."
        ),
        "runtime": _runtime(),
        "source_sha256": _source_hashes(),
        "registered_schedule": _stage_a_schedule(),
        "exact_commands": {
            "runner": (
                f"{sys.executable} {Path(__file__).resolve()} stage-a "
                f"--lock {output} --output {DEFAULT_ARTIFACT_A.resolve()}"
            ),
            "independent_verifier": (
                f"{sys.executable} {ANALYZER_PATH.resolve()} "
                f"{DEFAULT_ARTIFACT_A.resolve()} --lock {output} "
                f"--output {DEFAULT_VERIFICATION_A.resolve()}"
            ),
            "preflight_tests": (
                f"{sys.executable} -m pytest "
                "frontier_rl/examples/test_acrobot_neural.py "
                "frontier_rl/examples/test_run_acrobot_neural.py "
                "frontier_rl/examples/test_run_acrobot_hindsight_v4.py "
                "frontier_rl/examples/test_analyze_acrobot_hindsight_v4.py"
            ),
        },
        "equality_rule": (
            "Stage A must refuse to start or resume unless runtime, schedule, "
            "source list, and every source hash match exactly."
        ),
    }
    _atomic_write(output, lock, must_not_exist=True)
    print(f"sealed V4A source/runtime lock: {output}")


@dataclass
class _Instrumentation:
    trace_hasher: Any
    group_count: int = 0
    eligible_relabel_candidate_groups: list[int] | None = None
    wall_start: float | None = None
    wall_seconds_at_optimizer_updates: dict[str, float] | None = None
    actor: Any = None

    def __post_init__(self) -> None:
        if self.eligible_relabel_candidate_groups is None:
            self.eligible_relabel_candidate_groups = []
        if self.wall_seconds_at_optimizer_updates is None:
            self.wall_seconds_at_optimizer_updates = {}


class _TimeProxy:
    """Capture the exact same first perf-counter value used by the locked loop."""

    def __init__(self, instrumentation: _Instrumentation):
        self._instrumentation = instrumentation

    def perf_counter(self) -> float:
        value = _time.perf_counter()
        if self._instrumentation.wall_start is None:
            self._instrumentation.wall_start = value
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(_time, name)


def _instrumented_run(
    condition: locked.Condition,
    seed: int,
    *,
    budget: locked.RunBudget,
    eval_n: int,
) -> dict:
    """Run the locked loop while observing, never altering, its semantics."""
    instrumentation = _Instrumentation(trace_hasher=hashlib.sha256())
    original_actor = locked.TanhCategoricalActor
    original_space = locked.AcrobotNeuralSpace
    original_update = locked._update
    original_time = locked.time

    class InstrumentedActor(original_actor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instrumentation.actor = self

    class InstrumentedSpace(original_space):
        def rollout_group(self, task_id: int, n_rollouts: int):
            group = super().rollout_group(task_id, n_rollouts)
            instrumentation.group_count += 1
            payload = pickle.dumps(group, protocol=5)
            instrumentation.trace_hasher.update(len(payload).to_bytes(8, "little"))
            instrumentation.trace_hasher.update(payload)
            return group

        def relabel(self, group):
            result = super().relabel(group)
            if result is not None:
                instrumentation.eligible_relabel_candidate_groups.append(
                    instrumentation.group_count
                )
            return result

    def timed_update(actor, task_id: int, trajectories: list, weights: np.ndarray):
        diagnostics = original_update(actor, task_id, trajectories, weights)
        if diagnostics.get("applied"):
            update_count = int(actor.applied_updates)
            if update_count in (FALLBACK_UPDATES_A, TARGET_UPDATES_A):
                key = str(update_count)
                if key not in instrumentation.wall_seconds_at_optimizer_updates:
                    if instrumentation.wall_start is None:
                        raise RuntimeError("locked run wall clock was not initialized")
                    instrumentation.wall_seconds_at_optimizer_updates[key] = float(
                        _time.perf_counter() - instrumentation.wall_start
                    )
        return diagnostics

    locked.TanhCategoricalActor = InstrumentedActor
    locked.AcrobotNeuralSpace = InstrumentedSpace
    locked._update = timed_update
    locked.time = _TimeProxy(instrumentation)
    try:
        run = locked.run_condition(
            condition,
            seed,
            budget=budget,
            eval_interval_transitions=100_000,
            eval_interval_updates=EVAL_INTERVAL_UPDATES,
            eval_n=eval_n,
        )
    finally:
        locked.TanhCategoricalActor = original_actor
        locked.AcrobotNeuralSpace = original_space
        locked._update = original_update
        locked.time = original_time

    if instrumentation.actor is None:
        raise RuntimeError("V4 instrumentation did not observe the actor")
    run["wall_seconds_at_optimizer_updates"] = dict(
        instrumentation.wall_seconds_at_optimizer_updates
    )
    run["wall_timing_definition"] = (
        "perf_counter offset from the locked run's own wall_start to immediately "
        "after the applied optimizer update returns; rollout group is complete "
        "and scheduled evaluation has not yet begun"
    )
    run["eligible_relabel_candidate_groups"] = list(
        instrumentation.eligible_relabel_candidate_groups
    )
    run["training_group_trace_groups"] = instrumentation.group_count
    run["training_group_trace_sha256"] = instrumentation.trace_hasher.hexdigest()
    run["final_training_state_sha256"] = locked._training_state_fingerprint(
        None, instrumentation.actor
    )
    run["source_step_norms_full_run"] = _source_step_norms(
        run["update_diagnostics"]
    )
    return run


def _source_step_norms(update_records: Sequence[dict]) -> dict:
    output = {}
    for source in ("requested_live", "hindsight_relabel"):
        norms = np.asarray(
            [
                float(record["update_norm"])
                for record in update_records
                if record.get("source") == source
            ],
            dtype=np.float64,
        )
        if len(norms) and (not np.isfinite(norms).all() or np.any(norms <= 0.0)):
            raise FloatingPointError(f"invalid applied-step norm for {source}")
        output[source] = {
            "count": int(len(norms)),
            "cumulative_step_norm_M": float(norms.sum()) if len(norms) else 0.0,
            "cumulative_squared_step_norm_Q": (
                float(np.dot(norms, norms)) if len(norms) else 0.0
            ),
        }
    return output


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
        "task_successes",
        "task_transitions",
        "x_transitions",
        "x_optimizer_updates",
        "pass_rate_curve",
        "mean_pass_curve",
        "hardest_pass_curve",
        "native_success_rate_curve",
        "mean_native_return_curve",
        "mean_censored_time_to_goal_curve",
        "mean_policy_entropy_curve",
        "update_diagnostics",
        "group_diagnostics",
        "training_group_trace_groups",
        "training_group_trace_sha256",
        "final_training_state_sha256",
    )
    return {key: run[key] for key in keys}


def _preview_shadow_equivalence() -> dict:
    preview_condition = locked.Condition(
        name="v4_preview_shadow_scale_zero",
        stage="scale",
        sampling="teacher",
        architecture="shared",
        hidden_size=64,
        learning_rate=BASE_LEARNING_RATE,
        hindsight_scale=0.0,
        lr_multiplier=1.0,
    )
    no_hindsight_condition = replace(
        preview_condition,
        name="v4_preview_shadow_no_hindsight",
        stage="core",
    )
    budget = locked.RunBudget(
        optimizer_update_budget=SHADOW_UPDATE_BUDGET,
        transition_safety_cap=SHADOW_TRANSITION_CAP,
    )
    try:
        preview = _instrumented_run(
            preview_condition,
            SHADOW_SEED,
            budget=budget,
            eval_n=SHADOW_EVAL_EPISODES,
        )
        shadow = _instrumented_run(
            no_hindsight_condition,
            SHADOW_SEED,
            budget=budget,
            eval_n=SHADOW_EVAL_EPISODES,
        )
        preview_ids = preview["eligible_relabel_candidate_groups"]
        diagnostic_ids = [
            int(record["after_group"])
            for record in preview["auxiliary_gradient_diagnostics"]
        ]
        trace_equal = (
            preview["training_group_trace_sha256"]
            == shadow["training_group_trace_sha256"]
        )
        state_equal = (
            preview["final_training_state_sha256"]
            == shadow["final_training_state_sha256"]
        )
        projection_equal = _shadow_projection(preview) == _shadow_projection(shadow)
        previews_valid = bool(preview_ids) and preview_ids == diagnostic_ids and all(
            math.isfinite(float(record.get("gradient_norm", math.nan)))
            and float(record["gradient_norm"]) > 0.0
            and record.get("mutated") is False
            and record.get("applied") is False
            and record.get("frozen_group_parameters") is True
            for record in preview["auxiliary_gradient_diagnostics"]
        )
        return {
            "test_config": {
                "seed": SHADOW_SEED,
                "seed_block_status": "historical exploratory; outside V4 blocks",
                "learning_rate": BASE_LEARNING_RATE,
                "optimizer_update_budget": SHADOW_UPDATE_BUDGET,
                "transition_group_start_cap": SHADOW_TRANSITION_CAP,
                "eval_episodes_per_task": SHADOW_EVAL_EPISODES,
            },
            "passed": bool(
                trace_equal and state_equal and projection_equal and previews_valid
            ),
            "identical_training_group_trace": trace_equal,
            "identical_final_training_state": state_equal,
            "identical_saved_training_projection": projection_equal,
            "eligible_preview_exercised": bool(preview_ids),
            "preview_candidate_groups": preview_ids,
            "preview_diagnostic_groups": diagnostic_ids,
            "preview_training_group_trace_sha256": preview[
                "training_group_trace_sha256"
            ],
            "shadow_training_group_trace_sha256": shadow[
                "training_group_trace_sha256"
            ],
            "preview_final_training_state_sha256": preview[
                "final_training_state_sha256"
            ],
            "shadow_final_training_state_sha256": shadow[
                "final_training_state_sha256"
            ],
            "preview_transitions": preview["transitions"],
            "shadow_transitions": shadow["transitions"],
        }
    except Exception as error:
        return {
            "test_config": {
                "seed": SHADOW_SEED,
                "optimizer_update_budget": SHADOW_UPDATE_BUDGET,
                "transition_group_start_cap": SHADOW_TRANSITION_CAP,
            },
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }


def _conditions(scales: Sequence[float]) -> tuple[locked.Condition, ...]:
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
        for scale in scales
    )


def _protocol(stage: str, schedule: dict) -> dict:
    common = {
        "v4_stage": stage,
        "protocol_document": str(PROTOCOL_PATH.relative_to(PROJECT_ROOT)),
        "registered_schedule": schedule,
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
        "actor": "shared H64, 640 total and active parameters",
        "optimizer": "plain SGD ascent",
        "score_reduction": "sum over trajectory transitions at frozen group parameters",
        "hindsight_description": "verifier-valid centered auxiliary update",
        "matching_axis": "nonzero applied optimizer updates",
        "group_start_cap_semantics": (
            "no group begins at transitions >= 4,000,000; every started 16-rollout "
            "group completes, permitting at most 8,000 transition overshoot"
        ),
        "evaluation": (
            "initialization and every 50 nonzero updates with fixed per-seed "
            "common random numbers and training-state invariance"
        ),
    }
    if stage == "stage_a_feasibility":
        common.update(
            {
                "status": "development_effect_blind",
                "effect_blind_to_stage_b_hindsight_estimands": True,
                "gate_prefix_rule": (
                    "select U*=400 if all runs reach 400, else U*=250 if all "
                    "reach 250, else STOP; all gates use only the exact U* prefix"
                ),
                "runtime_projection": (
                    "90 * max_j(wall_seconds_through_tau_j(U*)) / 3600 <= 24"
                ),
            }
        )
    else:
        common.update(
            {
                "status": "confirmatory",
                "primary_metric": (
                    "normalized target-uniform mean-pass AUC over nonzero optimizer "
                    "updates, including update zero"
                ),
                "primary_family": ["C1", "C2", "C3", "C4"],
                "test": "exact two-sided paired sign-flip with Holm FWER 0.05",
                "bootstrap": "20,000 paired-seed resamples",
                "bootstrap_seed_by_contrast": {
                    "C1": 45_000,
                    "C2": 45_001,
                    "C3": 45_002,
                    "C4": 45_003,
                },
                "sign_exchangeability_assumption": True,
            }
        )
    return common


def _provenance(lock_path: Path, lock_hash: str) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime(),
        "platform": platform.platform(),
        "source_sha256": _source_hashes(),
        "source_lock_path": str(lock_path.resolve()),
        "source_lock_sha256": lock_hash,
    }


def _new_artifact(
    *, stage: str, schedule: dict, lock_path: Path, lock_hash: str
) -> dict:
    conditions = _conditions(
        HINDSIGHT_SCALES_A if stage == "stage_a_feasibility" else HINDSIGHT_SCALES_B
    )
    return {
        "schema": SCHEMA_A if stage == "stage_a_feasibility" else SCHEMA_B,
        "provenance": _provenance(lock_path, lock_hash),
        "protocol": _protocol(stage, schedule),
        "artifact_state": "in_progress",
        "run_failures": [],
        "cases": {
            condition.name: {"config": asdict(condition), "runs": []}
            for condition in conditions
        },
    }


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


def _evaluation_coordinates_valid(run: dict) -> bool:
    updates = [int(value) for value in run.get("x_optimizer_updates", [])]
    transitions = [int(value) for value in run.get("x_transitions", [])]
    if not updates or updates[0] != 0 or len(updates) != len(transitions):
        return False
    if any(right <= left for left, right in zip(transitions, transitions[1:])):
        return False
    terminal = int(run.get("optimizer_updates", -1))
    expected = list(range(0, terminal + 1, EVAL_INTERVAL_UPDATES))
    if expected[-1] != terminal:
        expected.append(terminal)
    # A cap-censored run may have evaluated exactly at its last scheduled
    # multiple and then complete later all-fail/all-pass groups without another
    # update.  The locked loop retains one final evaluation at the same update
    # coordinate.  This duplicate is post-prefix and is allowed only once.
    if updates not in (expected, expected + [terminal]):
        return False
    curve_keys = (
        "pass_rate_curve",
        "mean_pass_curve",
        "hardest_pass_curve",
        "evaluation_rng_preserved",
        "native_success_rate_curve",
        "mean_native_return_curve",
        "mean_censored_time_to_goal_curve",
        "mean_policy_entropy_curve",
    )
    return all(len(run.get(key, [])) == len(updates) for key in curve_keys)


def _full_run_errors(run: dict) -> list[str]:
    errors = []
    if not run.get("numeric_valid", False):
        return ["numeric_valid is not true"]
    if not _all_finite(run):
        errors.append("run contains a non-finite or unsupported value")
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
    groups = run.get("group_diagnostics", [])
    if len(groups) != run.get("sampled_groups"):
        errors.append("sampled-group record count mismatch")
    regimes = {"dead": 0, "mixed": 0, "all_pass": 0}
    task_groups = np.zeros(len(locked.THRESHOLDS), dtype=np.int64)
    task_successes = np.zeros(len(locked.THRESHOLDS), dtype=np.int64)
    task_transitions = np.zeros(len(locked.THRESHOLDS), dtype=np.int64)
    transition_total = 0
    previous_end = 0
    for expected_group, group in enumerate(groups, start=1):
        if group.get("group") != expected_group:
            errors.append("group identifiers are not consecutive")
            break
        start = int(group.get("transition_start", -1))
        end = int(group.get("transition_end", -1))
        count = int(group.get("n_transitions", -1))
        success_count = int(group.get("success_count", -1))
        regime = group.get("regime")
        expected_regime = (
            "dead" if success_count == 0 else "all_pass"
            if success_count == locked.N_ROLLOUTS else "mixed"
        )
        if start != previous_end or end - start != count or not 1 <= count <= 8_000:
            errors.append("transition/group accounting mismatch")
            break
        if start >= TRANSITION_GROUP_START_CAP:
            errors.append("a group began at or beyond the group-start cap")
            break
        if not 0 <= success_count <= locked.N_ROLLOUTS or regime != expected_regime:
            errors.append("binary verifier/regime accounting mismatch")
            break
        if regime not in regimes:
            errors.append("unknown group regime")
            break
        task_id = int(group.get("task_id", -1))
        if not 0 <= task_id < len(locked.THRESHOLDS):
            errors.append("group task id is outside the registered task set")
            break
        regimes[regime] += 1
        task_groups[task_id] += 1
        task_successes[task_id] += success_count
        task_transitions[task_id] += count
        transition_total += count
        previous_end = end
    if transition_total != run.get("transitions"):
        errors.append("raw transition total mismatch")
    if run.get("transitions", 0) > (
        TRANSITION_GROUP_START_CAP + MAX_COMPLETE_GROUP_OVERSHOOT
    ):
        errors.append("complete-group cap overshoot exceeded 8,000 transitions")
    if run.get("sampled_groups") != sum(regimes.values()):
        errors.append("group-regime total mismatch")
    if run.get("dead_groups") != regimes["dead"]:
        errors.append("dead-group total mismatch")
    if run.get("live_groups") != regimes["mixed"]:
        errors.append("mixed-group total mismatch")
    if run.get("all_pass_groups") != regimes["all_pass"]:
        errors.append("all-pass-group total mismatch")
    if sum(run.get("task_groups", [])) != run.get("sampled_groups"):
        errors.append("per-task group total mismatch")
    if list(task_groups) != run.get("task_groups"):
        errors.append("per-task group records do not reproduce")
    if sum(run.get("task_rollouts", [])) != (
        run.get("sampled_groups", 0) * locked.N_ROLLOUTS
    ):
        errors.append("per-task rollout total mismatch")
    if list(task_groups * locked.N_ROLLOUTS) != run.get("task_rollouts"):
        errors.append("per-task rollout records do not reproduce")
    if list(task_successes) != run.get("task_successes"):
        errors.append("per-task binary-success records do not reproduce")
    if sum(run.get("task_transitions", [])) != run.get("transitions"):
        errors.append("per-task transition total mismatch")
    if list(task_transitions) != run.get("task_transitions"):
        errors.append("per-task transition records do not reproduce")
    update_records = run.get("update_diagnostics", [])
    if len(update_records) != run.get("optimizer_updates"):
        errors.append("optimizer-update diagnostic count mismatch")
    if [record.get("optimizer_update") for record in update_records] != list(
        range(1, len(update_records) + 1)
    ):
        errors.append("optimizer-update identifiers are not consecutive")
    if run.get("optimizer_updates") != (
        run.get("live_applied_updates", 0) + run.get("relabeled_groups", 0)
    ):
        errors.append("optimizer source count mismatch")
    recorded_update_groups = {
        int(record["after_group"]): record.get("source") for record in update_records
    }
    for group in groups:
        expected_source = recorded_update_groups.get(int(group["group"]))
        if group.get("update_source") != expected_source:
            errors.append("group/update source records do not reproduce")
            break
    if run.get("training_group_trace_groups") != run.get("sampled_groups"):
        errors.append("instrumented trajectory group count mismatch")
    if not _evaluation_coordinates_valid(run):
        errors.append("evaluation cadence/coordinate invariant failed")
    if not all(run.get("evaluation_rng_preserved", [])):
        errors.append("evaluation changed training state")
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
    updates = [
        record
        for record in run["update_diagnostics"]
        if int(record["optimizer_update"]) <= target
    ]
    candidates = [
        int(group_id)
        for group_id in run["eligible_relabel_candidate_groups"]
        if int(group_id) <= terminal_group
    ]
    previews = [
        record
        for record in run["auxiliary_gradient_diagnostics"]
        if int(record["after_group"]) <= terminal_group
    ]
    evaluation_indices = [
        index
        for index, (update, transitions) in enumerate(
            zip(run["x_optimizer_updates"], run["x_transitions"])
        )
        if int(update) <= target
        and int(transitions) <= int(prefix_groups[-1]["transition_end"])
    ]
    return {
        "target_updates": target,
        "terminal_group": terminal_group,
        "terminal_transitions": int(prefix_groups[-1]["transition_end"]),
        "groups": prefix_groups,
        "updates": updates,
        "candidate_groups": candidates,
        "previews": previews,
        "evaluation_indices": evaluation_indices,
        "wall_seconds": float(run["wall_seconds_at_optimizer_updates"][str(target)]),
        "source_step_norms": _source_step_norms(updates),
    }


def _prefix_errors(prefix: dict, run: dict) -> list[str]:
    errors = []
    target = prefix["target_updates"]
    groups = prefix["groups"]
    updates = prefix["updates"]
    if int(groups[-1]["optimizer_updates_after_group"]) != target:
        errors.append("prefix does not end at exact selected update count")
    if any(
        int(group["optimizer_updates_after_group"]) >= target for group in groups[:-1]
    ):
        errors.append("selected prefix extends beyond the first exact target group")
    if len(updates) != target:
        errors.append("selected prefix update records do not equal U*")
    expected_eval_updates = list(range(0, target + 1, EVAL_INTERVAL_UPDATES))
    observed_eval_updates = [
        int(run["x_optimizer_updates"][index])
        for index in prefix["evaluation_indices"]
    ]
    if observed_eval_updates != expected_eval_updates:
        errors.append("selected prefix lacks the exact 50-update evaluation cadence")
    if not math.isfinite(prefix["wall_seconds"]) or prefix["wall_seconds"] <= 0.0:
        errors.append("selected-prefix wall time is missing or invalid")
    if prefix["wall_seconds"] > float(run["wall_seconds"]):
        errors.append("selected-prefix wall time exceeds full-run wall time")
    return errors


def _stage_a_prefix_invariant_errors(prefix: dict, run: dict) -> list[str]:
    """Recompute Gate-1 implementation invariants on the selected prefix only."""
    errors = _prefix_errors(prefix, run)
    if run.get("numeric_valid") is not True:
        errors.append("numeric_valid is not true")
    terminal_updates = int(run.get("optimizer_updates", -1))
    reached = run.get("reached_optimizer_update_budget") is True
    censored = run.get("transition_cap_censored") is True
    if prefix["target_updates"] == TARGET_UPDATES_A:
        if terminal_updates != TARGET_UPDATES_A or not reached or censored:
            errors.append("400-update reach/censoring flags are inconsistent")
    elif reached:
        if terminal_updates != TARGET_UPDATES_A or censored:
            errors.append("reached-target fallback run has inconsistent flags")
    elif (
        not censored
        or not FALLBACK_UPDATES_A <= terminal_updates < TARGET_UPDATES_A
        or int(run.get("transitions", -1)) < TRANSITION_GROUP_START_CAP
    ):
        errors.append("censored fallback run has inconsistent flags/cap")
    if (run.get("total_parameters"), run.get("active_parameters_per_task")) != (
        640,
        640,
    ):
        errors.append("shared-H64 parameter-count contract failed")
    evaluation_indices = prefix["evaluation_indices"]
    evaluation_curve_keys = (
        "x_transitions",
        "x_optimizer_updates",
        "pass_rate_curve",
        "mean_pass_curve",
        "hardest_pass_curve",
        "native_success_rate_curve",
        "mean_native_return_curve",
        "mean_censored_time_to_goal_curve",
        "mean_policy_entropy_curve",
        "evaluation_rng_preserved",
    )
    selected_evaluations = {
        key: [run[key][index] for index in evaluation_indices]
        for key in evaluation_curve_keys
    }
    selected_view = {
        "groups": prefix["groups"],
        "updates": prefix["updates"],
        "previews": prefix["previews"],
        "wall_seconds": prefix["wall_seconds"],
        "evaluations": selected_evaluations,
    }
    if not _all_finite(selected_view):
        errors.append("selected prefix contains a non-finite value")

    groups = prefix["groups"]
    updates = prefix["updates"]
    transition_total = 0
    previous_end = 0
    task_groups = np.zeros(len(locked.THRESHOLDS), dtype=np.int64)
    task_successes = np.zeros(len(locked.THRESHOLDS), dtype=np.int64)
    task_transitions = np.zeros(len(locked.THRESHOLDS), dtype=np.int64)
    update_by_group = {
        int(record["after_group"]): record.get("source") for record in updates
    }
    prior_update_count = 0
    for expected_group, group in enumerate(groups, start=1):
        start = int(group.get("transition_start", -1))
        end = int(group.get("transition_end", -1))
        count = int(group.get("n_transitions", -1))
        success_count = int(group.get("success_count", -1))
        task_id = int(group.get("task_id", -1))
        update_count = int(group.get("optimizer_updates_after_group", -1))
        expected_regime = (
            "dead"
            if success_count == 0
            else "all_pass"
            if success_count == locked.N_ROLLOUTS
            else "mixed"
        )
        if group.get("group") != expected_group:
            errors.append("prefix group identifiers are not consecutive")
            break
        if start != previous_end or end - start != count or not 1 <= count <= 8_000:
            errors.append("prefix transition/group accounting mismatch")
            break
        if start >= TRANSITION_GROUP_START_CAP:
            errors.append("prefix group began at or beyond the group-start cap")
            break
        if not 0 <= success_count <= locked.N_ROLLOUTS:
            errors.append("prefix reward is not a binary-count aggregate")
            break
        if group.get("regime") != expected_regime:
            errors.append("prefix binary verifier/regime mismatch")
            break
        if not 0 <= task_id < len(locked.THRESHOLDS):
            errors.append("prefix task id is outside the task set")
            break
        if update_count - prior_update_count not in (0, 1):
            errors.append("prefix applies more than one update in a group")
            break
        if group.get("update_source") != update_by_group.get(expected_group):
            errors.append("prefix group/update source mismatch")
            break
        transition_total += count
        previous_end = end
        prior_update_count = update_count
        task_groups[task_id] += 1
        task_successes[task_id] += success_count
        task_transitions[task_id] += count
    if transition_total != prefix["terminal_transitions"]:
        errors.append("prefix transition total does not reproduce")
    if prefix["terminal_transitions"] > (
        TRANSITION_GROUP_START_CAP + MAX_COMPLETE_GROUP_OVERSHOOT
    ):
        errors.append("prefix complete-group cap overshoot exceeds 8,000")
    if [record.get("optimizer_update") for record in updates] != list(
        range(1, prefix["target_updates"] + 1)
    ):
        errors.append("prefix optimizer update ids do not reproduce")
    if not all(run["evaluation_rng_preserved"][index] for index in evaluation_indices):
        errors.append("prefix evaluation-state invariance failed")
    pass_rates = np.asarray(selected_evaluations["pass_rate_curve"], dtype=np.float64)
    bounded_curves = np.asarray(
        [
            selected_evaluations["mean_pass_curve"],
            selected_evaluations["hardest_pass_curve"],
            selected_evaluations["native_success_rate_curve"],
        ],
        dtype=np.float64,
    )
    censored_times = np.asarray(
        selected_evaluations["mean_censored_time_to_goal_curve"],
        dtype=np.float64,
    )
    entropies = np.asarray(
        selected_evaluations["mean_policy_entropy_curve"], dtype=np.float64
    )
    if pass_rates.shape != (len(evaluation_indices), len(locked.THRESHOLDS)):
        errors.append("prefix evaluation pass-rate shape mismatch")
    elif np.any((pass_rates < 0.0) | (pass_rates > 1.0)):
        errors.append("prefix evaluation pass rate is outside [0,1]")
    if np.any((bounded_curves < 0.0) | (bounded_curves > 1.0)):
        errors.append("prefix bounded evaluation summary is outside [0,1]")
    if np.any((censored_times < 0.0) | (censored_times > 500.0)):
        errors.append("prefix censored time is outside [0,500]")
    if np.any((entropies < 0.0) | (entropies > math.log(3.0) + 1e-12)):
        errors.append("prefix categorical entropy is outside [0,log(3)]")
    # Expose recomputed task ledgers for independent verification without using
    # the post-prefix aggregate ledgers saved by the inherited loop.
    prefix["recomputed_task_groups"] = task_groups.astype(int).tolist()
    prefix["recomputed_task_rollouts"] = (
        task_groups * locked.N_ROLLOUTS
    ).astype(int).tolist()
    prefix["recomputed_task_successes"] = task_successes.astype(int).tolist()
    prefix["recomputed_task_transitions"] = task_transitions.astype(int).tolist()
    return errors


def _stage_a_lock_check(artifact: dict, lock_path: Path) -> tuple[bool, str | None]:
    try:
        _, lock_hash = _verify_source_lock(
            lock_path,
            schema=LOCK_SCHEMA_A,
            v4_stage="stage_a_feasibility",
            schedule=_stage_a_schedule(),
            expected_lock_sha256=artifact["provenance"]["source_lock_sha256"],
        )
        if artifact["provenance"].get("runtime") != _runtime():
            raise RuntimeError("artifact runtime differs from the live locked runtime")
        if artifact["provenance"].get("source_sha256") != _source_hashes():
            raise RuntimeError("artifact source hashes differ from live locked sources")
        if lock_hash != artifact["provenance"]["source_lock_sha256"]:
            raise RuntimeError("artifact source-lock hash mismatch")
        return True, None
    except Exception as error:
        return False, str(error)


def compute_stage_a_gates(artifact: dict, lock_path: Path) -> dict:
    """Compute only the frozen hindsight-effect-blind feasibility gates."""
    lock_ok, lock_error = _stage_a_lock_check(artifact, lock_path)
    expected_configs = {
        condition.name: asdict(condition) for condition in _conditions(HINDSIGHT_SCALES_A)
    }
    schedule_errors = []
    if artifact.get("schema") != SCHEMA_A:
        schedule_errors.append("artifact schema mismatch")
    if artifact.get("protocol") != _protocol(
        "stage_a_feasibility", _stage_a_schedule()
    ):
        schedule_errors.append("artifact protocol mismatch")
    if tuple(artifact.get("cases", {})) != STAGE_A_CASES:
        schedule_errors.append("Stage-A case order/set mismatch")
    all_runs = []
    for case_name in STAGE_A_CASES:
        record = artifact.get("cases", {}).get(case_name, {})
        if record.get("config") != expected_configs[case_name]:
            schedule_errors.append(f"condition config mismatch for {case_name}")
        runs = record.get("runs", [])
        if [run.get("seed") for run in runs] != list(STAGE_A_SEEDS):
            schedule_errors.append(f"seed order mismatch for {case_name}")
        for run in runs:
            all_runs.append((case_name, run))

    # The protocol selects U* from terminal update counts before evaluating any
    # launch gate.  Terminal performance and all post-U* records are ignored.
    update_counts = [
        int(run.get("optimizer_updates", -1))
        for _, run in all_runs
        if run.get("numeric_valid", False)
    ]
    selected = None
    if len(update_counts) == 9 and all(value >= TARGET_UPDATES_A for value in update_counts):
        selected = TARGET_UPDATES_A
    elif len(update_counts) == 9 and all(
        value >= FALLBACK_UPDATES_A for value in update_counts
    ):
        selected = FALLBACK_UPDATES_A

    prefixes: dict[str, dict] = {}
    prefix_errors = {}
    prefix_invariant_checks = {}
    if selected is not None:
        for case_name, run in all_runs:
            label = f"{case_name}/seed_{run.get('seed')}"
            try:
                run_prefix = _prefix(run, selected)
                invariant_errors = _stage_a_prefix_invariant_errors(
                    run_prefix, run
                )
                prefixes[label] = run_prefix
                prefix_errors[label] = _prefix_errors(run_prefix, run)
                prefix_invariant_checks[label] = {
                    "passed": not invariant_errors,
                    "errors": invariant_errors,
                    "terminal_group": run_prefix["terminal_group"],
                    "terminal_transitions": run_prefix["terminal_transitions"],
                    "recomputed_task_groups": run_prefix[
                        "recomputed_task_groups"
                    ],
                    "recomputed_task_rollouts": run_prefix[
                        "recomputed_task_rollouts"
                    ],
                    "recomputed_task_successes": run_prefix[
                        "recomputed_task_successes"
                    ],
                    "recomputed_task_transitions": run_prefix[
                        "recomputed_task_transitions"
                    ],
                }
            except Exception as error:
                prefix_errors[label] = [str(error)]
                prefix_invariant_checks[label] = {
                    "passed": False,
                    "errors": [str(error)],
                }

    gate1_passed = bool(
        lock_ok
        and not schedule_errors
        and selected is not None
        and len(all_runs) == len(STAGE_A_CASES) * len(STAGE_A_SEEDS)
        and len(prefix_invariant_checks) == 9
        and all(item["passed"] for item in prefix_invariant_checks.values())
    )
    gate1 = {
        "passed": gate1_passed,
        "source_runtime_lock_passed": lock_ok,
        "source_runtime_lock_error": lock_error,
        "schedule_errors": schedule_errors,
        "prefix_only_when_fallback_selected": True,
        "per_run_selected_prefix": prefix_invariant_checks,
    }

    gate2_passed = bool(
        selected is not None
        and len(prefixes) == 9
        and all(not errors for errors in prefix_errors.values())
    )
    gate2 = {
        "passed": gate2_passed,
        "selection_rule": "400 if all nine reach 400; else 250 if all nine reach 250; else STOP",
        "full_run_update_counts": update_counts,
        "selected_update_budget": selected,
        "per_run_prefix_errors": prefix_errors,
    }

    preview_details = {}
    gate3_run_passes = []
    if gate2_passed:
        for label, prefix in prefixes.items():
            candidates = prefix["candidate_groups"]
            previews = prefix["previews"]
            preview_ids = [int(record["after_group"]) for record in previews]
            preview_valid = all(
                math.isfinite(float(record.get("gradient_norm", math.nan)))
                and float(record["gradient_norm"]) > 0.0
                and record.get("mutated") is False
                and record.get("applied") is False
                and record.get("frozen_group_parameters") is True
                for record in previews
            )
            requested_updates = prefix["source_step_norms"]["requested_live"]["count"]
            hindsight_updates = prefix["source_step_norms"]["hindsight_relabel"]["count"]
            passed = bool(
                len(candidates) >= MIN_PREVIEWS_PER_RUN
                and candidates == preview_ids
                and preview_valid
                and hindsight_updates == 0
                and requested_updates == selected
            )
            gate3_run_passes.append(passed)
            preview_details[label] = {
                "passed": passed,
                "eligible_relabel_candidate_count": len(candidates),
                "preview_count": len(previews),
                "candidate_groups_equal_preview_groups": candidates == preview_ids,
                "all_previews_finite_strictly_positive_nonmutating": preview_valid,
                "requested_live_updates": requested_updates,
                "relabeled_updates": hindsight_updates,
                "source_step_norms_through_selected_prefix": prefix[
                    "source_step_norms"
                ],
            }
    shadow = artifact.get("preview_shadow_equivalence", {})
    gate3 = {
        "passed": bool(
            gate2_passed
            and len(gate3_run_passes) == 9
            and all(gate3_run_passes)
            and shadow.get("passed") is True
        ),
        "minimum_previews_per_run": MIN_PREVIEWS_PER_RUN,
        "per_run": preview_details,
        "preview_shadow_equivalence": shadow,
    }

    regime_details = {}
    gate4_cells = []
    if gate2_passed:
        for case_name in STAGE_A_CASES:
            cell_groups = [
                group
                for label, prefix in prefixes.items()
                if label.startswith(case_name + "/")
                for group in prefix["groups"]
            ]
            observed = sorted({group["regime"] for group in cell_groups})
            passed = set(observed) == {"dead", "mixed", "all_pass"}
            gate4_cells.append(passed)
            regime_details[case_name] = {"passed": passed, "observed_regimes": observed}
    gate4 = {
        "passed": bool(gate2_passed and len(gate4_cells) == 3 and all(gate4_cells)),
        "per_cell": regime_details,
    }

    tv_details = {}
    gate5_runs = []
    if gate2_passed:
        for label, prefix in prefixes.items():
            values = [
                float(group["teacher_tv_from_uniform"])
                for group in prefix["groups"]
                if int(group["transition_start"]) >= TV_WARMUP_TRANSITIONS
            ]
            mean_tv = float(np.mean(values)) if values else None
            passed = bool(mean_tv is not None and mean_tv > 0.05)
            gate5_runs.append(passed)
            tv_details[label] = {
                "passed": passed,
                "n_groups": len(values),
                "mean_teacher_tv_from_uniform": mean_tv,
            }
    gate5 = {
        "passed": bool(gate2_passed and len(gate5_runs) == 9 and all(gate5_runs)),
        "warmup_transition_start_inclusive": TV_WARMUP_TRANSITIONS,
        "strict_minimum_mean_tv": 0.05,
        "per_run": tv_details,
    }

    prefix_wall_seconds = {
        label: prefix["wall_seconds"] for label, prefix in prefixes.items()
    }
    max_wall = max(prefix_wall_seconds.values()) if len(prefix_wall_seconds) == 9 else None
    projected_hours = None if max_wall is None else 90.0 * max_wall / 3600.0
    gate6 = {
        "passed": bool(projected_hours is not None and projected_hours <= 24.0),
        "formula": "90 * max_j(wall_seconds_through_tau_j(U*)) / 3600",
        "per_run_prefix_wall_seconds": prefix_wall_seconds,
        "maximum_prefix_wall_seconds": max_wall,
        "projected_hours_90": projected_hours,
        "maximum_allowed_hours": MAX_PROJECTED_HOURS_90,
    }

    gates = {
        "effect_blind": True,
        "uses_evaluation_performance": False,
        "uses_hindsight_contrast": False,
        "uses_v3_outcome": False,
        "selected_update_budget": selected,
        "gate_1_lock_and_implementation_invariants": gate1,
        "gate_2_exact_selected_prefix": gate2,
        "gate_3_preview_mechanics_and_shadow": gate3,
        "gate_4_requested_group_regimes": gate4,
        "gate_5_per_run_teacher_tv": gate5,
        "gate_6_serial_runtime_projection": gate6,
    }
    gates["all_pass"] = all(
        gates[key]["passed"]
        for key in (
            "gate_1_lock_and_implementation_invariants",
            "gate_2_exact_selected_prefix",
            "gate_3_preview_mechanics_and_shadow",
            "gate_4_requested_group_regimes",
            "gate_5_per_run_teacher_tv",
            "gate_6_serial_runtime_projection",
        )
    )
    gates["stage_b_authorized"] = gates["all_pass"]
    gates["prefix_diagnostics_sha256"] = _canonical_hash(
        {
            label: {
                "terminal_group": prefix["terminal_group"],
                "terminal_transitions": prefix["terminal_transitions"],
                "source_step_norms": prefix["source_step_norms"],
            }
            for label, prefix in prefixes.items()
        }
    )
    return gates


def _artifact_identity(
    artifact: dict,
    *,
    stage: str,
    schedule: dict,
    lock_hash: str,
) -> None:
    expected_schema = SCHEMA_A if stage == "stage_a_feasibility" else SCHEMA_B
    expected_cases = STAGE_A_CASES if stage == "stage_a_feasibility" else STAGE_B_CASES
    expected_scales = (
        HINDSIGHT_SCALES_A if stage == "stage_a_feasibility" else HINDSIGHT_SCALES_B
    )
    if artifact.get("schema") != expected_schema:
        raise RuntimeError("resume artifact schema mismatch")
    if artifact.get("protocol") != _protocol(stage, schedule):
        raise RuntimeError("resume artifact protocol mismatch")
    if artifact.get("provenance", {}).get("source_lock_sha256") != lock_hash:
        raise RuntimeError("resume artifact was created under a different source lock")
    if artifact.get("provenance", {}).get("runtime") != _runtime():
        raise RuntimeError("resume artifact runtime mismatch")
    if artifact.get("provenance", {}).get("source_sha256") != _source_hashes():
        raise RuntimeError("resume artifact source mismatch")
    if tuple(artifact.get("cases", {})) != expected_cases:
        raise RuntimeError("resume artifact case order/set mismatch")
    expected_configs = {
        condition.name: asdict(condition) for condition in _conditions(expected_scales)
    }
    for case_name in expected_cases:
        if artifact["cases"][case_name].get("config") != expected_configs[case_name]:
            raise RuntimeError(f"resume condition mismatch for {case_name}")


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
        if stage == "stage_a_feasibility"
        else _stage_b_schedule(selected_update_budget)
    )
    schema = LOCK_SCHEMA_A if stage == "stage_a_feasibility" else LOCK_SCHEMA_B
    lock, lock_hash = _verify_source_lock(
        lock_path,
        schema=schema,
        v4_stage=stage,
        schedule=schedule,
        amendment_path=amendment_path,
    )
    if stage == "stage_b_factorial":
        if amendment_path is None or stage_a_artifact_path is None:
            raise ValueError("Stage B requires amendment and Stage-A artifact paths")
        amendment = _read_json(amendment_path)
        if amendment.get("schema") != AMENDMENT_SCHEMA_B:
            raise RuntimeError("unexpected Stage-B amendment schema")
        if amendment.get("explicit_stage_b_authorization") is not True:
            raise RuntimeError("Stage-B amendment lacks explicit authorization")
        if amendment.get("stage_a_all_effect_blind_gates_passed") is not True:
            raise RuntimeError("Stage-B amendment does not record passing V4A gates")
        if amendment.get("selected_update_budget") != selected_update_budget:
            raise RuntimeError("Stage-B amendment carries a different selected budget")
        if amendment.get("registered_schedule") != schedule:
            raise RuntimeError("Stage-B amendment schedule mismatch")
        if amendment.get("stage_a_artifact_sha256") != _sha256(stage_a_artifact_path):
            raise RuntimeError("Stage-A artifact changed after the Stage-B amendment")
        verification_path_value = amendment.get("stage_a_independent_verification")
        if not isinstance(verification_path_value, str):
            raise RuntimeError("Stage-B amendment lacks the independent V4A report path")
        verification_path = Path(verification_path_value).resolve()
        verification_hash = _sha256(verification_path)
        if amendment.get(
            "stage_a_independent_verification_sha256"
        ) != verification_hash:
            raise RuntimeError(
                "independent V4A verification changed after the Stage-B amendment"
            )
        verification = _read_json(verification_path)
        stage_a_artifact = _read_json(stage_a_artifact_path)
        saved_gates = stage_a_artifact.get("stage_a_effect_blind_gates", {})
        if (
            stage_a_artifact.get("schema") != SCHEMA_A
            or stage_a_artifact.get("artifact_state") != "complete"
            or saved_gates.get("all_pass") is not True
            or saved_gates.get("selected_update_budget") != selected_update_budget
        ):
            raise RuntimeError("Stage-A artifact does not authorize this Stage-B budget")
        gates_hash = _canonical_hash(saved_gates)
        if (
            amendment.get("stage_a_gates_sha256") != gates_hash
            or lock.get("stage_a_gates_sha256") != gates_hash
        ):
            raise RuntimeError("Stage-A gate hash differs across artifact/amendment/lock")
        if not all(
            (
                verification.get("schema")
                == "curriculum-maxrl/acrobot-hindsight-v4a-verification/v1",
                verification.get("v4_stage") == "stage_a_feasibility",
                verification.get("all_checks_passed") is True,
                verification.get("stage_b_factorial_authorized") is True,
                verification.get("saved_runner_gates_verified") is True,
                verification.get("runner_saved_gates") == saved_gates,
                verification.get("selected_optimizer_update_budget")
                == selected_update_budget,
                verification.get("artifact_sha256")
                == _sha256(stage_a_artifact_path),
                verification.get("lock_sha256")
                == stage_a_artifact.get("provenance", {}).get(
                    "source_lock_sha256"
                ),
                Path(verification.get("artifact", "")).resolve()
                == stage_a_artifact_path.resolve(),
            )
        ):
            raise RuntimeError(
                "independent V4A verification no longer authorizes this Stage B"
            )
        if lock.get("stage_a_artifact_sha256") != _sha256(stage_a_artifact_path):
            raise RuntimeError("Stage-B lock has a different Stage-A artifact hash")
        if lock.get(
            "stage_a_independent_verification_sha256"
        ) != verification_hash:
            raise RuntimeError(
                "Stage-B lock has a different independent V4A verification hash"
            )

    output = output.resolve()
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"refusing to overwrite {output}; use --resume for this exact artifact"
            )
        artifact = _read_json(output)
        _artifact_identity(
            artifact, stage=stage, schedule=schedule, lock_hash=lock_hash
        )
        if artifact.get("artifact_state") != "in_progress":
            print(f"artifact is already terminal: {output}")
            return artifact
    else:
        if resume:
            raise FileNotFoundError(f"cannot resume missing artifact {output}")
        artifact = _new_artifact(
            stage=stage, schedule=schedule, lock_path=lock_path, lock_hash=lock_hash
        )
        if stage == "stage_b_factorial":
            artifact["stage_a_artifact"] = str(stage_a_artifact_path.resolve())
            artifact["stage_a_artifact_sha256"] = _sha256(stage_a_artifact_path)
            artifact["stage_b_amendment"] = str(amendment_path.resolve())
            artifact["stage_b_amendment_sha256"] = _sha256(amendment_path)
            artifact["stage_a_independent_verification"] = str(verification_path)
            artifact[
                "stage_a_independent_verification_sha256"
            ] = verification_hash
        _atomic_write(output, artifact, must_not_exist=True)

    if stage == "stage_a_feasibility" and "preview_shadow_equivalence" not in artifact:
        artifact["preview_shadow_equivalence"] = _preview_shadow_equivalence()
        _atomic_write(output, artifact, must_not_exist=False)
        print(
            "completed deterministic preview/no-hindsight shadow: "
            f"passed={artifact['preview_shadow_equivalence']['passed']}",
            flush=True,
        )

    seeds = STAGE_A_SEEDS if stage == "stage_a_feasibility" else STAGE_B_SEEDS
    scales = HINDSIGHT_SCALES_A if stage == "stage_a_feasibility" else HINDSIGHT_SCALES_B
    conditions = _conditions(scales)
    budget = locked.RunBudget(
        optimizer_update_budget=selected_update_budget,
        transition_safety_cap=TRANSITION_GROUP_START_CAP,
    )
    total_runs = len(conditions) * len(seeds)
    completed_before = sum(
        len(record["runs"]) for record in artifact["cases"].values()
    )
    completed = completed_before
    for condition in conditions:
        runs = artifact["cases"][condition.name]["runs"]
        existing_seeds = [run.get("seed") for run in runs]
        if existing_seeds != list(seeds[: len(existing_seeds)]):
            raise RuntimeError(
                f"resume records for {condition.name} are not an exact seed prefix"
            )
        for seed in seeds[len(runs) :]:
            try:
                run = _instrumented_run(
                    condition,
                    seed,
                    budget=budget,
                    eval_n=EVAL_EPISODES_PER_TASK,
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
                artifact["run_failures"].append(
                    {"condition": condition.name, **run}
                )
            runs.append(run)
            completed += 1
            _atomic_write(output, artifact, must_not_exist=False)
            print(
                f"V4 {stage} technical progress: {completed}/{total_runs} runs retained; "
                f"latest={condition.name}/seed_{seed}; valid={run.get('numeric_valid', False)}",
                flush=True,
            )

    if stage == "stage_a_feasibility":
        artifact["stage_a_effect_blind_gates"] = compute_stage_a_gates(
            artifact, lock_path
        )
        artifact["analysis_status"] = {
            "performed": True,
            "type": "hindsight-effect-blind feasibility gates only",
            "learning_performance_used": False,
        }
        artifact["artifact_state"] = (
            "complete"
            if not artifact["run_failures"]
            else "complete_with_invalid_runs"
        )
    else:
        _attach_stage_b_analysis(artifact, lock_path, amendment_path)
    _atomic_write(output, artifact, must_not_exist=False)
    print(f"wrote {output}")
    return artifact


def _stage_b_runs_ready(artifact: dict, target: int) -> tuple[bool, list[str]]:
    errors = []
    if tuple(artifact.get("cases", {})) != STAGE_B_CASES:
        errors.append("Stage-B case order/set mismatch")
        return False, errors
    for case_name in STAGE_B_CASES:
        runs = artifact["cases"][case_name].get("runs", [])
        if [run.get("seed") for run in runs] != list(STAGE_B_SEEDS):
            errors.append(f"seed order/completeness mismatch for {case_name}")
            continue
        for run in runs:
            label = f"{case_name}/seed_{run.get('seed')}"
            run_errors = _full_run_errors(run)
            if run_errors:
                errors.append(f"{label}: " + "; ".join(run_errors))
                continue
            if run.get("transition_cap_censored") is not False:
                errors.append(f"{label}: transition-cap censored")
            if run.get("reached_optimizer_update_budget") is not True:
                errors.append(f"{label}: did not reach frozen update target")
            if run.get("optimizer_updates") != target:
                errors.append(f"{label}: final update coordinate is not exactly U*")
            if run.get("x_optimizer_updates", [])[-1:] != [target]:
                errors.append(f"{label}: terminal evaluation is not exactly at U*")
    return not errors, errors


def _recomputed_update_auc(run: dict, target: int) -> float:
    x = np.asarray(run["x_optimizer_updates"], dtype=np.float64)
    y = np.asarray(run["mean_pass_curve"], dtype=np.float64)
    if len(x) != len(y) or len(x) < 2 or x[0] != 0 or x[-1] != target:
        raise ValueError("invalid update-indexed AUC curve")
    if np.any(np.diff(x) <= 0) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("update-indexed AUC inputs are invalid")
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


def _attach_stage_b_analysis(
    artifact: dict, lock_path: Path, amendment_path: Path
) -> None:
    target = int(artifact["protocol"]["registered_schedule"]["optimizer_update_target"])
    lock_errors = []
    try:
        _verify_source_lock(
            lock_path,
            schema=LOCK_SCHEMA_B,
            v4_stage="stage_b_factorial",
            schedule=_stage_b_schedule(target),
            expected_lock_sha256=artifact["provenance"]["source_lock_sha256"],
            amendment_path=amendment_path,
        )
    except Exception as error:
        lock_errors.append(str(error))
    ready, readiness_errors = _stage_b_runs_ready(artifact, target)
    if lock_errors or not ready:
        artifact["analysis_status"] = {
            "performed": False,
            "reason": (
                "the complete paired primary analysis is invalidated by a source-lock, "
                "missing-run, invalid-run, cap-censoring, or terminal-coordinate failure"
            ),
            "errors": lock_errors + readiness_errors,
        }
        artifact["artifact_state"] = "complete_with_invalid_primary_analysis"
        return

    per_case: dict[str, dict[int, float]] = {}
    case_summaries = {}
    for case_name in STAGE_B_CASES:
        seed_values = {}
        source_diagnostics = {}
        transitions = {}
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
            source_diagnostics[str(seed)] = _source_step_norms(
                run["update_diagnostics"]
            )
            transitions[str(seed)] = int(run["transitions"])
        per_case[case_name] = seed_values
        values = np.asarray([seed_values[seed] for seed in STAGE_B_SEEDS])
        case_summaries[case_name] = {
            "metric": "auc_mean_pass_by_optimizer_updates",
            "n_seeds": len(values),
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "per_seed": values.tolist(),
            "source_step_norms_per_seed": source_diagnostics,
            "transitions_to_target_per_seed": transitions,
        }

    specs = {
        "C1": {
            _case_name(1.0, 1.0): 1.0,
            _case_name(1.0, 0.0): -1.0,
        },
        "C2": {
            _case_name(1.0, 2.0): 1.0,
            _case_name(1.0, 1.0): -1.0,
        },
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
    descriptions = {
        "C1": "base-LR scale 1 minus scale 0",
        "C2": "base-LR scale 2 minus scale 1",
        "C3": "half/base LR restricted-separability diagnostic",
        "C4": "base/double LR restricted-separability diagnostic",
    }
    contrasts = {}
    raw_p = {}
    for index, (name, coefficients) in enumerate(specs.items()):
        values = _contrast_values(per_case, coefficients)
        contrasts[name] = {
            "description": descriptions[name],
            "coefficients": coefficients,
            "metric": "auc_mean_pass_by_optimizer_updates",
            "n_pairs": len(STAGE_B_SEEDS),
            "mean_contrast": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "mean_ci95_paired_seed_bootstrap": locked.bootstrap_mean_ci(
                values, seed=45_000 + index, n_boot=20_000
            ),
            "exact_paired_sign_flip_p_two_sided": locked.exact_sign_flip_p(values),
            "per_seed_contrast": values.tolist(),
        }
        raw_p[name] = contrasts[name]["exact_paired_sign_flip_p_two_sided"]
    adjusted = locked.holm_adjust(raw_p, alpha=0.05)
    for name in specs:
        contrasts[name].update(adjusted[name])

    directional_support = {
        name: bool(
            contrasts[name]["mean_contrast"] >= 0.03
            and contrasts[name]["reject_familywise_0.05"]
        )
        for name in ("C1", "C2")
    }
    restricted_departure = {
        name: bool(
            abs(contrasts[name]["mean_contrast"]) >= 0.03
            and contrasts[name]["reject_familywise_0.05"]
        )
        for name in ("C3", "C4")
    }
    artifact["stage_b_case_summaries"] = case_summaries
    artifact["paired_scale_contrasts"] = contrasts
    artifact["scale_multiplicity"] = {
        "family": ["C1", "C2", "C3", "C4"],
        "metric": "auc_mean_pass_by_optimizer_updates",
        "method": "Holm step-down",
        "familywise_alpha": 0.05,
        "test": "exact two-sided paired sign-flip randomization",
        "sign_exchangeability_assumption": (
            "independent seed-level contrasts have sign-exchangeable null distributions"
        ),
    }
    artifact["predeclared_scale_decision"] = {
        "C1_directional_local_improvement_supported": directional_support["C1"],
        "C2_directional_increment_supported": directional_support["C2"],
        "C3_material_restricted_separability_departure": restricted_departure["C3"],
        "C4_material_restricted_separability_departure": restricted_departure["C4"],
        "directional_minimum_mean": 0.03,
        "restricted_departure_minimum_absolute_mean": 0.03,
        "interpretation_boundary": (
            "C3/C4 diagnose departure from Y(a,s)=F(a)+G(a*s); they do not "
            "identify semantic data value separately from optimizer scale, update "
            "source composition, relabel frequency, or policy trajectory."
        ),
    }
    artifact["analysis_status"] = {
        "performed": True,
        "all_ten_seed_pairs_complete": True,
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
        raise RuntimeError(
            "refusing to authorize Stage B without the literal --authorize-stage-b flag"
        )
    stage_a_artifact_path = stage_a_artifact_path.resolve()
    stage_a_verification_path = stage_a_verification_path.resolve()
    stage_a = _read_json(stage_a_artifact_path)
    if stage_a.get("artifact_state") != "complete":
        raise RuntimeError("Stage-A artifact is not complete and valid")
    lock_a_path = Path(stage_a["provenance"]["source_lock_path"])
    recomputed = compute_stage_a_gates(stage_a, lock_a_path)
    saved = stage_a.get("stage_a_effect_blind_gates")
    if saved != recomputed:
        raise RuntimeError("saved Stage-A gates do not match deterministic recomputation")
    if recomputed.get("all_pass") is not True:
        raise RuntimeError("Stage-A gates do not authorize the factorial")
    selected = int(recomputed["selected_update_budget"])
    schedule = _stage_b_schedule(selected)
    stage_a_hash = _sha256(stage_a_artifact_path)
    verification = _read_json(stage_a_verification_path)
    verification_errors = []
    if verification.get("schema") != (
        "curriculum-maxrl/acrobot-hindsight-v4a-verification/v1"
    ):
        verification_errors.append("unexpected independent V4A verification schema")
    if verification.get("v4_stage") != "stage_a_feasibility":
        verification_errors.append("independent report is not for V4A")
    if verification.get("all_checks_passed") is not True:
        verification_errors.append("independent V4A verification did not pass")
    if verification.get("stage_b_factorial_authorized") is not True:
        verification_errors.append("independent verifier did not authorize Stage B")
    if verification.get("saved_runner_gates_verified") is not True:
        verification_errors.append("independent verifier did not validate saved gates")
    if verification.get("runner_saved_gates") != saved:
        verification_errors.append("independent report carries different saved gates")
    if verification.get("selected_optimizer_update_budget") != selected:
        verification_errors.append("independent verifier selected a different U*")
    if verification.get("artifact_sha256") != stage_a_hash:
        verification_errors.append("independent report has a different Stage-A hash")
    if verification.get("lock_sha256") != stage_a["provenance"][
        "source_lock_sha256"
    ]:
        verification_errors.append("independent report has a different V4A lock hash")
    report_artifact = verification.get("artifact")
    if report_artifact is None or Path(report_artifact).resolve() != stage_a_artifact_path:
        verification_errors.append("independent report names a different artifact path")
    if verification_errors:
        raise RuntimeError(
            "independent V4A verification is not authorization-ready: "
            + "; ".join(verification_errors)
        )
    verification_hash = _sha256(stage_a_verification_path)
    amendment = {
        "schema": AMENDMENT_SCHEMA_B,
        "v4_stage": "stage_b_factorial",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "explicit_stage_b_authorization": True,
        "stage_a_artifact": str(stage_a_artifact_path),
        "stage_a_artifact_sha256": stage_a_hash,
        "stage_a_independent_verification": str(stage_a_verification_path),
        "stage_a_independent_verification_sha256": verification_hash,
        "stage_a_gates_sha256": _canonical_hash(saved),
        "stage_a_all_effect_blind_gates_passed": True,
        "selected_update_budget": selected,
        "registered_schedule": schedule,
        "frozen_claim_rule": (
            "C1/C2 require mean >= +0.03 and Holm rejection; C3/C4 require "
            "absolute mean >= 0.03 and Holm rejection; no secondary rescue."
        ),
        "exact_commands": {
            "stage_a_independent_verifier": (
                f"{sys.executable} {ANALYZER_PATH.resolve()} "
                f"{stage_a_artifact_path} "
                f"--lock {Path(stage_a['provenance']['source_lock_path']).resolve()} "
                f"--output {stage_a_verification_path}"
            ),
            "runner": (
                f"{sys.executable} {Path(__file__).resolve()} stage-b "
                f"--stage-a-artifact {stage_a_artifact_path} "
                f"--amendment {amendment_output.resolve()} "
                f"--lock {lock_output.resolve()} "
                f"--output {DEFAULT_ARTIFACT_B.resolve()}"
            ),
            "independent_verifier": (
                f"{sys.executable} {ANALYZER_PATH.resolve()} "
                f"{DEFAULT_ARTIFACT_B.resolve()} "
                f"--lock {lock_output.resolve()} "
                f"--output {DEFAULT_VERIFICATION_B.resolve()}"
            ),
        },
    }
    _atomic_write(amendment_output, amendment, must_not_exist=True)
    lock = {
        "schema": LOCK_SCHEMA_B,
        "v4_stage": "stage_b_factorial",
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Pre-execution source/runtime/amendment lock for V4B.",
        "runtime": _runtime(),
        "source_sha256": _source_hashes(),
        "registered_schedule": schedule,
        "amendment_sha256": _sha256(amendment_output),
        "stage_a_artifact_sha256": stage_a_hash,
        "stage_a_independent_verification_sha256": verification_hash,
        "stage_a_gates_sha256": _canonical_hash(saved),
        "equality_rule": (
            "Stage B must refuse to start/resume unless runtime, sources, "
            "schedule, amendment, and Stage-A artifact hashes match exactly."
        ),
    }
    try:
        _atomic_write(lock_output, lock, must_not_exist=True)
    except BaseException:
        # The retained amendment is harmless and cannot launch without its lock.
        raise
    print(f"wrote explicit V4B amendment: {amendment_output.resolve()}")
    print(f"sealed V4B source/runtime lock: {lock_output.resolve()}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal_a = subparsers.add_parser(
        "seal-a", help="write the pre-existing V4A source/runtime lock"
    )
    seal_a.add_argument("--output", type=Path, default=DEFAULT_LOCK_A)

    stage_a = subparsers.add_parser(
        "stage-a", help="run exact effect-blind V4A feasibility"
    )
    stage_a.add_argument("--lock", type=Path, default=DEFAULT_LOCK_A)
    stage_a.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_A)
    stage_a.add_argument("--resume", action="store_true")

    authorize_b = subparsers.add_parser(
        "authorize-b", help="write the post-gate V4B amendment and source lock"
    )
    authorize_b.add_argument("--stage-a-artifact", type=Path, required=True)
    authorize_b.add_argument(
        "--stage-a-verification", type=Path, required=True
    )
    authorize_b.add_argument(
        "--amendment-output", type=Path, default=DEFAULT_AMENDMENT_B
    )
    authorize_b.add_argument("--lock-output", type=Path, default=DEFAULT_LOCK_B)
    authorize_b.add_argument("--authorize-stage-b", action="store_true")

    stage_b = subparsers.add_parser(
        "stage-b", help="run the exact amendment-authorized V4B factorial"
    )
    stage_b.add_argument("--stage-a-artifact", type=Path, required=True)
    stage_b.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT_B)
    stage_b.add_argument("--lock", type=Path, default=DEFAULT_LOCK_B)
    stage_b.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_B)
    stage_b.add_argument("--resume", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "seal-a":
            _seal_stage_a(args.output)
            return
        if args.command == "stage-a":
            _run_stage(
                stage="stage_a_feasibility",
                lock_path=args.lock,
                output=args.output,
                resume=args.resume,
                selected_update_budget=TARGET_UPDATES_A,
            )
            return
        if args.command == "authorize-b":
            _authorize_stage_b(
                stage_a_artifact_path=args.stage_a_artifact,
                stage_a_verification_path=args.stage_a_verification,
                amendment_output=args.amendment_output,
                lock_output=args.lock_output,
                explicit_authorization=args.authorize_stage_b,
            )
            return
        if args.command == "stage-b":
            amendment = _read_json(args.amendment)
            selected = int(amendment.get("selected_update_budget", -1))
            _run_stage(
                stage="stage_b_factorial",
                lock_path=args.lock,
                output=args.output,
                resume=args.resume,
                selected_update_budget=selected,
                stage_a_artifact_path=args.stage_a_artifact,
                amendment_path=args.amendment,
            )
            return
        raise AssertionError(f"unhandled command {args.command!r}")
    except (FileNotFoundError, FileExistsError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
