"""Independent read-only verifier for Acrobot hindsight V5 artifacts.

This module deliberately owns its V5 schemas, schedules, prefix gates, run
validation, exact 20-pair randomization test, and Holm implementation.  It does
not import the V5 runner or V4 analyzer.  Run it as a module so the project root
is established before any ``frontier_rl`` import::

    python -m frontier_rl.examples.analyze_acrobot_hindsight_v5 ARTIFACT.json \
        --lock LOCK.json --output VERIFICATION.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

import gymnasium
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontier_rl.adapters.acrobot_neural import TanhCategoricalActor
from frontier_rl.teacher import FrontierTeacher


SCHEMA_A = "curriculum-maxrl/acrobot-hindsight-v5a-artifact/v1"
SCHEMA_B = "curriculum-maxrl/acrobot-hindsight-v5b-artifact/v1"
LOCK_SCHEMA_A = "curriculum-maxrl/acrobot-hindsight-v5a-source-lock/v1"
LOCK_SCHEMA_B = "curriculum-maxrl/acrobot-hindsight-v5b-source-lock/v1"
REPORT_SCHEMA_A = "curriculum-maxrl/acrobot-hindsight-v5a-verification/v1"
REPORT_SCHEMA_B = "curriculum-maxrl/acrobot-hindsight-v5b-verification/v1"
AMENDMENT_SCHEMA_B = "curriculum-maxrl/acrobot-hindsight-v5b-amendment/v1"

RUNNER_RELATIVE = "frontier_rl/examples/run_acrobot_hindsight_v5.py"
PROTOCOL_RELATIVE = "frontier_rl/examples/ACROBOT_HINDSIGHT_PROTOCOL_V5.md"
BASE_LEARNING_RATE = 3e-4
LR_MULTIPLIERS = (0.5, 1.0, 2.0)
HINDSIGHT_SCALES = (0.0, 1.0, 2.0)
STAGE_A_SEEDS = tuple(range(15_000, 15_003))
STAGE_B_SEEDS = tuple(range(16_000, 16_020))
TARGET_UPDATES = 400
FALLBACK_UPDATES = 250
TRANSITION_CAP = 4_000_000
MAX_OVERSHOOT = 8_000
EVAL_INTERVAL = 50
EVAL_N = 32
TV_WARMUP = 200_000
THRESHOLDS = (-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.7, 1.0)
N_ROLLOUTS = 16
SHADOW_SEED = 100

REQUIRED_SOURCE_FILES = {
    "frontier_rl/examples/run_acrobot_hindsight_v5.py",
    "frontier_rl/examples/analyze_acrobot_hindsight_v5.py",
    "frontier_rl/examples/test_run_acrobot_hindsight_v5.py",
    "frontier_rl/examples/test_analyze_acrobot_hindsight_v5.py",
    "frontier_rl/examples/ACROBOT_HINDSIGHT_PROTOCOL_V5.md",
    "frontier_rl/examples/run_acrobot_hindsight_v4.py",
    "frontier_rl/examples/run_acrobot_neural.py",
    "frontier_rl/adapters/acrobot_neural.py",
    "frontier_rl/teacher.py",
    "frontier_rl/estimators.py",
    "frontier_rl/interfaces.py",
    "frontier_rl/examples/test_acrobot_neural.py",
    "frontier_rl/examples/test_run_acrobot_neural.py",
    "frontier_rl/examples/test_run_acrobot_hindsight_v4.py",
    "frontier_rl/__init__.py",
    "frontier_rl/trainer.py",
    "frontier_rl/adapters/__init__.py",
}

OUTCOME_FIELDS = {
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


def _float_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _case_name(multiplier: float, scale: float) -> str:
    return f"lr_mult_{_float_label(multiplier)}_hs_{_float_label(scale)}"


CASES = tuple(
    _case_name(multiplier, scale)
    for multiplier in LR_MULTIPLIERS
    for scale in HINDSIGHT_SCALES
)


def _schedule_a() -> dict:
    return {
        "v5_stage": "stage_a_natural_feasibility",
        "paired_seeds": list(STAGE_A_SEEDS),
        "base_learning_rate": BASE_LEARNING_RATE,
        "learning_rate_multipliers": list(LR_MULTIPLIERS),
        "hindsight_scales": list(HINDSIGHT_SCALES),
        "condition_names": list(CASES),
        "optimizer_update_target": TARGET_UPDATES,
        "single_allowed_fallback_update_target": FALLBACK_UPDATES,
        "transition_group_start_cap": TRANSITION_CAP,
        "maximum_complete_group_overshoot": MAX_OVERSHOOT,
        "eval_interval_optimizer_updates": EVAL_INTERVAL,
        "eval_episodes_per_task": EVAL_N,
        "fresh_budget_selection_population": "all 27 V5A runs",
        "shadow_test": {
            "seed": SHADOW_SEED,
            "optimizer_update_budget": 3,
            "transition_group_start_cap": 80_000,
            "eval_episodes_per_task": 4,
        },
    }


def _schedule_b(target: int) -> dict:
    if target not in (FALLBACK_UPDATES, TARGET_UPDATES):
        raise ValueError("V5B target must be exactly 250 or 400")
    return {
        "v5_stage": "stage_b_confirmatory_factorial",
        "paired_seeds": list(STAGE_B_SEEDS),
        "base_learning_rate": BASE_LEARNING_RATE,
        "learning_rate_multipliers": list(LR_MULTIPLIERS),
        "hindsight_scales": list(HINDSIGHT_SCALES),
        "condition_names": list(CASES),
        "optimizer_update_target": target,
        "transition_group_start_cap": TRANSITION_CAP,
        "maximum_complete_group_overshoot": MAX_OVERSHOOT,
        "eval_interval_optimizer_updates": EVAL_INTERVAL,
        "eval_episodes_per_task": EVAL_N,
        "run_count": 180,
    }


def _select_update_budget(update_counts: Sequence[int], expected_runs: int = 27) -> int | None:
    counts = [int(value) for value in update_counts]
    if len(counts) == expected_runs and all(value >= TARGET_UPDATES for value in counts):
        return TARGET_UPDATES
    if len(counts) == expected_runs and all(value >= FALLBACK_UPDATES for value in counts):
        return FALLBACK_UPDATES
    return None


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _project_file(relative: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"source path escapes project: {relative}") from error
    if not path.is_file():
        raise ValueError(f"locked source is not a file: {relative}")
    return path


def _assert_close(actual: Any, expected: Any, label: str, atol: float = 1e-12) -> None:
    try:
        left, right = float(actual), float(expected)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"{label} mismatch: {left!r} != {right!r}")


def _finite(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value in {label}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{label}[{index}]")
        return
    raise ValueError(f"unsupported value type in {label}: {type(value).__name__}")


def _verify_lock(artifact: dict, lock: dict, lock_path: Path, stage: str) -> dict:
    expected_schema = LOCK_SCHEMA_A if stage == "stage_a_natural_feasibility" else LOCK_SCHEMA_B
    if lock.get("schema") != expected_schema or lock.get("v5_stage") != stage:
        raise ValueError("V5 lock schema/stage mismatch")
    if lock.get("runtime") != _runtime() or artifact.get("provenance", {}).get(
        "runtime"
    ) != _runtime():
        raise ValueError("live/artifact runtime differs from source lock")
    lock_hash = _sha256(lock_path)
    provenance = artifact.get("provenance", {})
    if provenance.get("source_lock_sha256") != lock_hash:
        raise ValueError("artifact was created under a different lock file")
    if Path(provenance.get("source_lock_path", "")).resolve() != lock_path.resolve():
        raise ValueError("artifact provenance names a different source-lock path")
    schedule = _schedule_a() if stage == "stage_a_natural_feasibility" else _schedule_b(
        int(lock.get("registered_schedule", {}).get("optimizer_update_target", -1))
    )
    if lock.get("registered_schedule") != schedule:
        raise ValueError("locked schedule differs from independent V5 schedule")
    locked_hashes = lock.get("source_sha256")
    artifact_hashes = artifact.get("provenance", {}).get("source_sha256")
    if not isinstance(locked_hashes, dict) or set(locked_hashes) != REQUIRED_SOURCE_FILES:
        raise ValueError("source lock does not exactly cover the V5 evidence chain")
    if artifact_hashes != locked_hashes:
        raise ValueError("artifact source hashes differ from lock")
    if provenance.get("reused_engine_sha256") != locked_hashes.get(
        "frontier_rl/examples/run_acrobot_hindsight_v4.py"
    ):
        raise ValueError("artifact reused-engine provenance hash mismatch")
    if provenance.get("v4_artifacts_reused") is not False:
        raise ValueError("artifact does not affirm that V4 artifacts were not reused")
    for relative, digest in locked_hashes.items():
        if _sha256(_project_file(relative)) != digest:
            raise ValueError(f"live source differs from lock: {relative}")
    seed_audit = lock.get("seed_collision_audit")
    if not isinstance(seed_audit, dict) or seed_audit.get("passed") is not True:
        raise ValueError("locked seed collision audit did not pass")
    contract = lock.get("engine_contract")
    required_contract = {
        "instrumentation_checkpoints": [250, 400],
        "instrumentation_eval_interval_updates": 50,
        "instrumentation_transition_group_start_cap": 4_000_000,
        "instrumentation_maximum_complete_group_overshoot": 8_000,
        "same_locked_neural_module_object": True,
        "n_rollouts": 16,
        "thresholds": list(THRESHOLDS),
        "teacher_gamma": 1.0,
        "teacher_decay": 0.7,
        "teacher_floor": 0.1,
        "max_episode_steps": 500,
        "shared_h64_total_parameters": 640,
        "shared_h64_active_parameters": 640,
    }
    if contract != required_contract:
        raise ValueError("locked imported-engine contract differs from V5")
    return {
        "passed": True,
        "lock_sha256": lock_hash,
        "runtime": _runtime(),
        "registered_schedule": schedule,
        "checked_source_files": sorted(locked_hashes),
        "engine_contract": contract,
        "seed_collision_audit": seed_audit,
    }


def _expected_config(name: str, multiplier: float, scale: float) -> dict:
    return {
        "name": name,
        "stage": "scale",
        "sampling": "teacher",
        "architecture": "shared",
        "hidden_size": 64,
        "learning_rate": BASE_LEARNING_RATE * multiplier,
        "hindsight_scale": scale,
        "hindsight_estimator": "maxrl",
        "gamma": 1.0,
        "lr_multiplier": multiplier,
    }


def _validate_protocol(artifact: dict, stage: str, schedule: dict) -> None:
    protocol = artifact.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("artifact protocol is missing")
    required = {
        "v5_stage": stage,
        "protocol_document": PROTOCOL_RELATIVE,
        "registered_schedule": schedule,
        "gymnasium_environment": "Acrobot-v1",
        "thresholds": list(THRESHOLDS),
        "n_rollouts": N_ROLLOUTS,
        "matching_axis": "nonzero applied optimizer updates",
        "teacher_evidence": "requested task and original binary outcomes only",
    }
    for key, value in required.items():
        if protocol.get(key) != value:
            raise ValueError(f"protocol mismatch for {key}")
    status = (
        "development_learning_outcome_field_blind"
        if stage == "stage_a_natural_feasibility"
        else "confirmatory"
    )
    if protocol.get("status") != status:
        raise ValueError("protocol status mismatch")
    if stage == "stage_a_natural_feasibility":
        stage_required = {
            "learning_outcome_fields_excluded_from_launch_rule": True,
            "budget_rule": (
                "U*=400 iff all 27 runs reach 400; else U*=250 iff all 27 "
                "reach 250; else STOP. Every gate uses first exact U* prefixes."
            ),
            "natural_gate_rule": (
                "registered runs contain no forced relabels: every cell must naturally "
                "produce an accepted candidate pooled across its three seeds; positive-scale candidates "
                "must all be positive finite applied auxiliary updates"
            ),
            "runtime_projection": "180 * slowest selected-prefix wall seconds / 3600 <= 18",
        }
    else:
        stage_required = {
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
    for key, value in stage_required.items():
        if protocol.get(key) != value:
            raise ValueError(f"stage-specific protocol mismatch for {key}")


def _validate_curves(run: dict, label: str, *, exact_target: int | None) -> None:
    x_updates = run.get("x_optimizer_updates")
    x_transitions = run.get("x_transitions")
    mean_pass = run.get("mean_pass_curve")
    pass_rates = run.get("pass_rate_curve")
    if not all(isinstance(item, list) for item in (x_updates, x_transitions, mean_pass, pass_rates)):
        raise ValueError(f"missing curves in {label}")
    length = len(x_updates)
    if length < 2 or not all(len(item) == length for item in (x_transitions, mean_pass, pass_rates)):
        raise ValueError(f"curve length mismatch in {label}")
    if x_updates[0] != 0 or x_transitions[0] != 0 or any(
        right <= left for left, right in zip(x_transitions, x_transitions[1:])
    ):
        raise ValueError(f"resource coordinates invalid in {label}")
    if exact_target is not None:
        expected = list(range(0, exact_target + 1, EVAL_INTERVAL))
        if x_updates != expected or x_transitions[-1] != run.get("transitions"):
            raise ValueError(f"confirmatory evaluation coordinates invalid in {label}")
    for index, row in enumerate(pass_rates):
        if not isinstance(row, list) or len(row) != 8 or any(
            not 0.0 <= float(value) <= 1.0 for value in row
        ):
            raise ValueError(f"pass-rate row invalid in {label}")
        _assert_close(mean_pass[index], np.mean(row), f"{label}.mean_pass[{index}]")
        _assert_close(run["hardest_pass_curve"][index], row[-1], f"{label}.hardest[{index}]")
    if any(not 0.0 <= float(value) <= 1.0 for value in mean_pass):
        raise ValueError(f"mean pass outside [0,1] in {label}")
    if len(run.get("evaluation_rng_preserved", [])) != length or not all(
        run["evaluation_rng_preserved"]
    ):
        raise ValueError(f"evaluation changed training RNG in {label}")
    for key, value in run.items():
        if key.endswith("_curve") and isinstance(value, list) and len(value) != length:
            raise ValueError(f"curve length mismatch for {label}.{key}")


def _validate_groups(run: dict, label: str) -> set[str]:
    groups = run.get("group_diagnostics")
    if not isinstance(groups, list) or len(groups) != run.get("sampled_groups"):
        raise ValueError(f"group diagnostics mismatch in {label}")
    teacher = FrontierTeacher(
        8, 16, decay=0.7, floor=0.1, gamma=1.0, seed=int(run["seed"]) + 10_000
    )
    previous_end = 0
    previous_updates = 0
    regimes: set[str] = set()
    counts = {"dead": 0, "mixed": 0, "all_pass": 0}
    task_groups = [0] * 8
    task_successes = [0] * 8
    task_transitions = [0] * 8
    for index, group in enumerate(groups, start=1):
        start, end, size = (
            group.get("transition_start"),
            group.get("transition_end"),
            group.get("n_transitions"),
        )
        if (
            group.get("group") != index
            or start != previous_end
            or end - start != size
            or not 0 < size <= MAX_OVERSHOOT
            or start >= TRANSITION_CAP
        ):
            raise ValueError(f"group transition accounting invalid in {label}")
        task, success = group.get("task_id"), group.get("success_count")
        if not isinstance(task, int) or not 0 <= task < 8 or not isinstance(
            success, int
        ) or not 0 <= success <= 16:
            raise ValueError(f"group task/success invalid in {label}")
        regime = "dead" if success == 0 else "all_pass" if success == 16 else "mixed"
        if group.get("regime") != regime:
            raise ValueError(f"group regime invalid in {label}")
        probabilities = np.asarray(teacher.distribution(), dtype=np.float64)
        expected_tv = float(0.5 * np.abs(probabilities - 0.125).sum())
        expected_task = int(teacher.rng.choice(8, p=probabilities))
        if task != expected_task:
            raise ValueError(f"teacher task trace mismatch in {label}")
        _assert_close(group.get("teacher_tv_from_uniform"), expected_tv, f"{label}.tv")
        _assert_close(group.get("sampled_task_probability"), probabilities[task], f"{label}.p")
        # Reconstruct teacher evidence from the original requested outcome only.
        original_rewards = np.zeros(16, dtype=np.float64)
        original_rewards[:success] = 1.0
        teacher.observe(task, original_rewards)
        updates = group.get("optimizer_updates_after_group")
        source = group.get("update_source")
        if not isinstance(updates, int) or updates - previous_updates not in (0, 1):
            raise ValueError(f"optimizer counter jump in {label}")
        if (updates == previous_updates) != (source is None) or source not in (
            None,
            "requested_live",
            "hindsight_relabel",
        ):
            raise ValueError(f"optimizer source mismatch in {label}")
        if source == "requested_live" and regime != "mixed":
            raise ValueError(f"requested-live source is not a mixed group in {label}")
        if source == "hindsight_relabel" and regime != "dead":
            raise ValueError(f"hindsight source is not a dead group in {label}")
        regimes.add(regime)
        counts[regime] += 1
        task_groups[task] += 1
        task_successes[task] += success
        task_transitions[task] += size
        previous_end, previous_updates = end, updates
    if previous_end != run.get("transitions") or previous_updates != run.get(
        "optimizer_updates"
    ):
        raise ValueError(f"terminal group counters mismatch in {label}")
    if run.get("transitions", 0) > TRANSITION_CAP + MAX_OVERSHOOT:
        raise ValueError(f"cap overshoot invalid in {label}")
    expected_counts = {
        "dead_groups": counts["dead"],
        "live_groups": counts["mixed"],
        "all_pass_groups": counts["all_pass"],
        "task_groups": task_groups,
        "task_rollouts": [value * 16 for value in task_groups],
        "task_successes": task_successes,
        "task_transitions": task_transitions,
    }
    for key, expected in expected_counts.items():
        if run.get(key) != expected:
            raise ValueError(f"raw accounting mismatch for {label}.{key}")
    return regimes


def _source_norms(run: dict, label: str) -> dict:
    updates = run.get("update_diagnostics")
    if not isinstance(updates, list) or len(updates) != run.get("optimizer_updates"):
        raise ValueError(f"update diagnostics mismatch in {label}")
    groups = {group["group"]: group for group in run["group_diagnostics"]}
    totals = {
        "requested_live": {"count": 0, "M": 0.0, "Q": 0.0},
        "hindsight_relabel": {"count": 0, "M": 0.0, "Q": 0.0},
    }
    seen_groups = set()
    for index, record in enumerate(updates, start=1):
        source, after_group = record.get("source"), record.get("after_group")
        norm, gradient = float(record.get("update_norm", math.nan)), float(
            record.get("gradient_norm", math.nan)
        )
        if (
            record.get("optimizer_update") != index
            or source not in totals
            or after_group in seen_groups
            or after_group not in groups
            or groups[after_group].get("update_source") != source
            or record.get("transitions") != groups[after_group].get("transition_end")
            or not math.isfinite(norm)
            or norm <= 0.0
            or not math.isfinite(gradient)
            or gradient <= 0.0
        ):
            raise ValueError(f"applied update invalid in {label}")
        group = groups[after_group]
        requested = int(record.get("requested_task", -1))
        credited = int(record.get("credited_task", -1))
        if source == "requested_live" and not (
            requested == credited == int(group["task_id"])
        ):
            raise ValueError(f"requested-live task metadata invalid in {label}")
        if source == "hindsight_relabel" and not (
            requested == int(group["task_id"]) and 0 <= credited < requested < 8
        ):
            raise ValueError(f"hindsight task metadata invalid in {label}")
        seen_groups.add(after_group)
        totals[source]["count"] += 1
        totals[source]["M"] += norm
        totals[source]["Q"] += norm * norm
    if run.get("optimizer_updates") != run.get("live_applied_updates") + run.get(
        "relabeled_groups"
    ):
        raise ValueError(f"update source totals mismatch in {label}")
    if totals["requested_live"]["count"] != run.get("live_applied_updates"):
        raise ValueError(f"requested-live counter mismatch in {label}")
    if totals["hindsight_relabel"]["count"] != run.get("relabeled_groups"):
        raise ValueError(f"hindsight counter mismatch in {label}")
    return totals


def _validate_relabels(run: dict, label: str, scale: float) -> None:
    eligible = run.get("eligible_relabel_candidate_groups")
    if not isinstance(eligible, list) or len(eligible) != run.get("relabel_candidates"):
        raise ValueError(f"candidate accounting mismatch in {label}")
    if eligible != sorted(set(eligible)):
        raise ValueError(f"candidate ids duplicate/nonmonotone in {label}")
    groups = {group["group"]: group for group in run["group_diagnostics"]}
    if any(group_id not in groups or groups[group_id]["regime"] != "dead" for group_id in eligible):
        raise ValueError(f"candidate does not name a dead group in {label}")
    hindsight_groups = [
        record["after_group"]
        for record in run["update_diagnostics"]
        if record["source"] == "hindsight_relabel"
    ]
    if any(group_id not in eligible for group_id in hindsight_groups):
        raise ValueError(f"hindsight update lacks candidate in {label}")
    previews = run.get("auxiliary_gradient_diagnostics")
    if not isinstance(previews, list):
        raise ValueError(f"preview diagnostics missing in {label}")
    zero_records = run.get("zero_gradient_diagnostics")
    if not isinstance(zero_records, list) or len(zero_records) != run.get(
        "zero_gradient_update_attempts"
    ):
        raise ValueError(f"zero-gradient diagnostics/count invalid in {label}")
    zero_groups = set()
    zero_hindsight = []
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
            raise ValueError(f"malformed zero-gradient diagnostic in {label}")
        if source == "requested_live" and not (
            group.get("regime") == "mixed"
            and requested == credited == int(group["task_id"])
        ):
            raise ValueError(f"zero-gradient requested-live metadata invalid in {label}")
        if source == "hindsight_relabel" and not (
            group.get("regime") == "dead"
            and group_id in eligible
            and requested == int(group["task_id"])
            and 0 <= credited < requested < 8
        ):
            raise ValueError(f"zero-gradient hindsight metadata invalid in {label}")
        zero_groups.add(group_id)
        if source == "hindsight_relabel":
            zero_hindsight.append(group_id)
    if scale == 0.0:
        if [record.get("after_group") for record in previews] != eligible:
            raise ValueError(f"scale-zero preview/candidate mismatch in {label}")
        if hindsight_groups or run.get("relabeled_groups") != 0:
            raise ValueError(f"scale-zero applied hindsight update in {label}")
        for record in previews:
            group_id = int(record.get("after_group", -1))
            gradient = float(record.get("gradient_norm", math.nan))
            hypothetical = float(record.get("hypothetical_update_norm", math.nan))
            if (
                record.get("applied") is not False
                or record.get("mutated") is not False
                or record.get("frozen_group_parameters") is not True
                or not math.isfinite(gradient)
                or gradient <= 0.0
                or not math.isfinite(hypothetical)
                or hypothetical <= 0.0
                or group_id not in groups
                or record.get("transitions") != groups[group_id].get("transition_end")
                or int(record.get("requested_task", -1))
                != int(groups[group_id]["task_id"])
                or not 0
                <= int(record.get("credited_task", -1))
                < int(record.get("requested_task", -1))
                < 8
            ):
                raise ValueError(f"scale-zero preview invalid in {label}")
    else:
        if (
            set(hindsight_groups) & set(zero_hindsight)
            or sorted([*hindsight_groups, *zero_hindsight]) != eligible
            or len(hindsight_groups) + len(zero_hindsight) != len(eligible)
        ):
            raise ValueError(
                f"positive-scale candidate partition differs from applied/zero attempts in {label}"
            )
        if previews or run.get("unscaled_aux_gradient_previews") != 0:
            raise ValueError(f"positive-scale run contains preview-only records in {label}")


def _validate_run(run: dict, label: str, scale: float, *, exact_target: int | None) -> set[str]:
    if not all(
        run.get(key) is True
        for key in (
            "numeric_valid",
            "accounting_valid",
            "verifier_relabel_checks_valid",
            "evaluation_cadence_invariant",
        )
    ):
        raise ValueError(f"invalid saved run flags in {label}")
    if (run.get("total_parameters"), run.get("active_parameters_per_task")) != (640, 640):
        raise ValueError(f"shared-H64 parameter contract failed in {label}")
    _finite(run, label)
    regimes = _validate_groups(run, label)
    _source_norms(run, label)
    _validate_relabels(run, label, scale)
    _validate_curves(run, label, exact_target=exact_target)
    if run.get("training_group_trace_groups") != run.get("sampled_groups"):
        raise ValueError(f"training trace count mismatch in {label}")
    for key in ("training_group_trace_sha256", "final_training_state_sha256"):
        digest = run.get(key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"malformed {key} in {label}")
    return regimes


def _fixture() -> dict:
    trajectories = [
        [(np.array([0.20, -0.10, 0.30, -0.20, 0.01, -0.02]), 0)],
        [(np.array([-0.30, 0.25, -0.15, 0.10, -0.03, 0.04]), 1)],
        [(np.array([0.15, 0.05, -0.25, 0.35, 0.02, 0.01]), 2)],
        [(np.array([-0.05, -0.20, 0.10, 0.30, -0.01, -0.03]), 0)],
    ]
    weights = np.asarray([-0.75, 0.25, 0.50, 1.00], dtype=np.float64)
    cells = {}
    ratios = []
    all_pass = True
    for multiplier in LR_MULTIPLIERS:
        for scale in HINDSIGHT_SCALES:
            actor = TanhCategoricalActor(8, 64, BASE_LEARNING_RATE * multiplier, 9_907, "shared")
            before = actor.parameter_vector()
            before_rng = _canonical_hash(actor.action_rng.bit_generator.state)
            before_counts = [actor.update_calls, actor.applied_updates]
            gradient = actor.group_gradient(2, trajectories, weights)
            flat = np.concatenate(
                [gradient["W_in"].ravel(), gradient["b_hidden"].ravel(), gradient["W_out"].ravel()]
            )
            gradient_norm = float(np.linalg.norm(flat))
            if scale == 0.0:
                preview = actor.gradient_diagnostics(2, trajectories, weights)
                expected = np.zeros_like(flat)
                positive_preview = bool(float(preview["gradient_norm"]) > 0.0)
            else:
                actor.update(2, trajectories, weights * scale)
                expected = BASE_LEARNING_RATE * multiplier * scale * flat
                positive_preview = None
            delta = actor.parameter_vector() - before
            delta_matches = bool(np.allclose(delta, expected, rtol=1e-12, atol=1e-15))
            state_rule = bool(
                np.array_equal(actor.parameter_vector(), before)
                and before_counts == [actor.update_calls, actor.applied_updates]
                and before_rng == _canonical_hash(actor.action_rng.bit_generator.state)
            ) if scale == 0.0 else bool(
                [actor.update_calls, actor.applied_updates] == [1, 1]
                and before_rng == _canonical_hash(actor.action_rng.bit_generator.state)
                and np.linalg.norm(delta) > 0.0
            )
            if scale > 0.0:
                ratios.append(float(np.linalg.norm(delta) / (multiplier * scale)))
            passed = bool(gradient_norm > 0.0 and delta_matches and state_rule and positive_preview is not False)
            all_pass &= passed
            cells[_case_name(multiplier, scale)] = {
                "passed": passed,
                "learning_rate_multiplier": multiplier,
                "hindsight_scale": scale,
                "requested_task": 3,
                "credited_task": 2,
                "source": "hindsight_relabel" if scale > 0 else "scale_zero_preview",
                "unscaled_gradient_norm": gradient_norm,
                "parameter_delta_norm": float(np.linalg.norm(delta)),
                "expected_parameter_delta_norm": float(np.linalg.norm(expected)),
                "delta_theta_equals_base_lr_times_a_times_s_times_g": delta_matches,
                "parameter_counter_rng_rule_passed": state_rule,
                "positive_nonmutating_preview": positive_preview,
            }
    common = bool(np.allclose(ratios, np.full(len(ratios), ratios[0]), rtol=1e-12, atol=1e-15))
    return {
        "schema": "curriculum-maxrl/acrobot-hindsight-v5-scale-fixture/v1",
        "synthetic_seed": 9_907,
        "registered_seed_touched": False,
        "forced_fixture_only": True,
        "natural_stage_a_events_forced": False,
        "same_initial_parameters_and_group_in_all_cells": True,
        "credited_task_strictly_lower_than_requested": True,
        "positive_cell_delta_norm_over_a_times_s_constant": common,
        "cells": cells,
        "passed": bool(all_pass and common and len(cells) == 9),
    }


def _prefix(run: dict, target: int, label: str) -> dict:
    terminal_index = next(
        (
            index
            for index, group in enumerate(run["group_diagnostics"])
            if group["optimizer_updates_after_group"] == target
            and all(previous["optimizer_updates_after_group"] < target for previous in run["group_diagnostics"][:index])
        ),
        None,
    )
    if terminal_index is None:
        raise ValueError(f"missing first exact selected prefix in {label}")
    groups = run["group_diagnostics"][: terminal_index + 1]
    terminal_group = groups[-1]["group"]
    updates = run["update_diagnostics"][:target]
    if len(updates) != target or any(record["after_group"] > terminal_group for record in updates):
        raise ValueError(f"selected-prefix update records invalid in {label}")
    candidates = [
        group_id
        for group_id in run["eligible_relabel_candidate_groups"]
        if group_id <= terminal_group
    ]
    previews = [
        record
        for record in run["auxiliary_gradient_diagnostics"]
        if record["after_group"] <= terminal_group
    ]
    eval_count = target // EVAL_INTERVAL + 1
    if run["x_optimizer_updates"][:eval_count] != list(range(0, target + 1, EVAL_INTERVAL)):
        raise ValueError(f"prefix evaluation update coordinates invalid in {label}")
    if run["x_transitions"][eval_count - 1] != groups[-1]["transition_end"]:
        raise ValueError(f"prefix evaluation transition coordinate invalid in {label}")
    timing = float(run.get("wall_seconds_at_optimizer_updates", {}).get(str(target), math.nan))
    if not math.isfinite(timing) or timing <= 0.0:
        raise ValueError(f"prefix wall time invalid in {label}")
    if [record["optimizer_update"] for record in updates] != list(range(1, target + 1)):
        raise ValueError(f"prefix optimizer update ids invalid in {label}")
    if candidates != sorted(set(candidates)):
        raise ValueError(f"prefix candidate ids duplicate/nonmonotone in {label}")
    return {
        "groups": groups,
        "updates": updates,
        "candidates": candidates,
        "previews": previews,
        "wall_seconds": timing,
        "terminal_group": terminal_group,
        "terminal_transitions": groups[-1]["transition_end"],
    }


def _validate_prefix(prefix: dict, run: dict, target: int, label: str) -> None:
    """Independently validate only the retained first-exact-U* mechanics."""

    groups, updates = prefix["groups"], prefix["updates"]
    if (run.get("total_parameters"), run.get("active_parameters_per_task")) != (640, 640):
        raise ValueError(f"prefix parameter contract failed in {label}")
    update_by_group = {int(record.get("after_group", -1)): record for record in updates}
    if len(update_by_group) != len(updates) or any(
        group_id < 1 or group_id > len(groups) for group_id in update_by_group
    ):
        raise ValueError(f"prefix update group ids invalid in {label}")
    teacher = FrontierTeacher(8, 16, decay=0.7, floor=0.1, gamma=1.0, seed=int(run["seed"]) + 10_000)
    previous_end = 0
    previous_updates = 0
    for index, group in enumerate(groups, start=1):
        start, end, size = group.get("transition_start"), group.get("transition_end"), group.get("n_transitions")
        success, task = group.get("success_count"), group.get("task_id")
        if (
            group.get("group") != index
            or not all(isinstance(value, int) for value in (start, end, size, success, task))
            or start != previous_end
            or end - start != size
            or not 0 < size <= MAX_OVERSHOOT
            or start >= TRANSITION_CAP
            or not 0 <= success <= 16
            or not 0 <= task < 8
        ):
            raise ValueError(f"prefix group accounting invalid in {label}")
        regime = "dead" if success == 0 else "all_pass" if success == 16 else "mixed"
        if group.get("regime") != regime:
            raise ValueError(f"prefix group regime invalid in {label}")
        probabilities = np.asarray(teacher.distribution(), dtype=np.float64)
        expected_task = int(teacher.rng.choice(8, p=probabilities))
        expected_tv = float(0.5 * np.abs(probabilities - 0.125).sum())
        if task != expected_task:
            raise ValueError(f"prefix teacher trace mismatch in {label}")
        _assert_close(group.get("teacher_tv_from_uniform"), expected_tv, f"{label}.prefix_tv")
        _assert_close(group.get("sampled_task_probability"), probabilities[task], f"{label}.prefix_p")
        original = np.zeros(16, dtype=np.float64)
        original[:success] = 1.0
        teacher.observe(task, original)
        count = group.get("optimizer_updates_after_group")
        diagnostic = update_by_group.get(index)
        incremented = count == previous_updates + 1
        if not isinstance(count, int) or count - previous_updates not in (0, 1) or incremented != (diagnostic is not None):
            raise ValueError(f"prefix optimizer counter invalid in {label}")
        source = None if diagnostic is None else diagnostic.get("source")
        if group.get("update_source") != source:
            raise ValueError(f"prefix source/group mismatch in {label}")
        if source == "requested_live" and regime != "mixed":
            raise ValueError(f"prefix requested source regime invalid in {label}")
        if source == "hindsight_relabel" and regime != "dead":
            raise ValueError(f"prefix hindsight source regime invalid in {label}")
        previous_end, previous_updates = end, count
    if previous_updates != target or previous_end != prefix["terminal_transitions"]:
        raise ValueError(f"prefix terminal counters invalid in {label}")
    for index, record in enumerate(updates, start=1):
        source, group_id = record.get("source"), int(record.get("after_group", -1))
        norm, gradient = float(record.get("update_norm", math.nan)), float(record.get("gradient_norm", math.nan))
        if (
            record.get("optimizer_update") != index
            or source not in ("requested_live", "hindsight_relabel")
            or not math.isfinite(norm)
            or norm <= 0.0
            or not math.isfinite(gradient)
            or gradient <= 0.0
        ):
            raise ValueError(f"prefix applied update invalid in {label}")
        group = groups[group_id - 1]
        if record.get("transitions") != group.get("transition_end"):
            raise ValueError(f"prefix applied update transition invalid in {label}")
        requested, credited = int(record.get("requested_task", -1)), int(record.get("credited_task", -1))
        if source == "requested_live" and not (requested == credited == group["task_id"]):
            raise ValueError(f"prefix requested task metadata invalid in {label}")
        if source == "hindsight_relabel" and not (
            requested == group["task_id"] and 0 <= credited < requested < 8
        ):
            raise ValueError(f"prefix hindsight task metadata invalid in {label}")
    if prefix["candidates"] != sorted(set(prefix["candidates"])) or any(
        group_id < 1
        or group_id > len(groups)
        or groups[group_id - 1]["regime"] != "dead"
        for group_id in prefix["candidates"]
    ):
        raise ValueError(f"prefix candidates invalid in {label}")
    eval_count = target // EVAL_INTERVAL + 1
    transitions = run.get("x_transitions", [])[:eval_count]
    if (
        len(transitions) != eval_count
        or transitions[0] != 0
        or any(not isinstance(value, int) or value < 0 for value in transitions)
        or any(right <= left for left, right in zip(transitions, transitions[1:]))
        or transitions[-1] != prefix["terminal_transitions"]
        or run.get("evaluation_rng_preserved", [])[:eval_count] != [True] * eval_count
    ):
        raise ValueError(f"prefix evaluation resource/state trace invalid in {label}")


def _validate_shadow(artifact: dict) -> dict:
    shadow = artifact.get("preview_shadow_equivalence")
    if not isinstance(shadow, dict):
        raise ValueError("preview shadow is missing")
    required_true = (
        "passed",
        "identical_training_group_trace",
        "identical_final_training_state",
        "identical_saved_training_projection",
        "eligible_preview_exercised",
    )
    if not all(shadow.get(key) is True for key in required_true):
        raise ValueError("preview shadow did not pass")
    candidates = shadow.get("preview_candidate_groups")
    if not isinstance(candidates, list) or not candidates or candidates != shadow.get(
        "preview_diagnostic_groups"
    ):
        raise ValueError("preview shadow candidate evidence invalid")
    expected_config = {
        "seed": SHADOW_SEED,
        "seed_block_status": "already-executed exploratory smoke; outside fresh V5 blocks",
        "learning_rate": BASE_LEARNING_RATE,
        "optimizer_update_budget": 3,
        "transition_group_start_cap": 80_000,
        "eval_episodes_per_task": 4,
    }
    if shadow.get("test_config") != expected_config:
        raise ValueError("preview shadow configuration mismatch")
    preview_projection = shadow.get("preview_mechanical_projection")
    control_projection = shadow.get("control_mechanical_projection")
    if not isinstance(preview_projection, dict) or not isinstance(control_projection, dict):
        raise ValueError("preview shadow lacks raw mechanical projections")
    required_projection_keys = {
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
    }
    if set(preview_projection) != required_projection_keys or set(
        control_projection
    ) != required_projection_keys:
        raise ValueError("preview shadow mechanical projection key set mismatch")
    if preview_projection != control_projection:
        raise ValueError("preview/control raw mechanical projections differ")
    if shadow.get("preview_mechanical_projection_sha256") != _canonical_hash(
        preview_projection
    ) or shadow.get("control_mechanical_projection_sha256") != _canonical_hash(
        control_projection
    ):
        raise ValueError("preview shadow projection digest does not reproduce")
    projection_links = {
        "preview_transitions": preview_projection.get("transitions"),
        "shadow_transitions": control_projection.get("transitions"),
        "preview_training_group_trace_sha256": preview_projection.get(
            "training_group_trace_sha256"
        ),
        "shadow_training_group_trace_sha256": control_projection.get(
            "training_group_trace_sha256"
        ),
        "preview_final_training_state_sha256": preview_projection.get(
            "final_training_state_sha256"
        ),
        "shadow_final_training_state_sha256": control_projection.get(
            "final_training_state_sha256"
        ),
    }
    if any(shadow.get(key) != expected for key, expected in projection_links.items()):
        raise ValueError("preview shadow summary does not bind to raw projections")
    records = shadow.get("preview_auxiliary_gradient_diagnostics")
    if (
        not isinstance(records, list)
        or candidates != sorted(set(candidates))
        or [record.get("after_group") for record in records] != candidates
    ):
        raise ValueError("preview shadow lacks exact raw preview diagnostics")
    groups = {
        int(group["group"]): group
        for group in preview_projection.get("group_diagnostics", [])
    }
    for record in records:
        group_id = int(record.get("after_group", -1))
        gradient = float(record.get("gradient_norm", math.nan))
        hypothetical = float(record.get("hypothetical_update_norm", math.nan))
        if (
            group_id not in groups
            or groups[group_id].get("regime") != "dead"
            or record.get("applied") is not False
            or record.get("mutated") is not False
            or record.get("frozen_group_parameters") is not True
            or not math.isfinite(gradient)
            or gradient <= 0.0
            or not math.isfinite(hypothetical)
            or hypothetical <= 0.0
            or int(record.get("requested_task", -1)) != int(groups[group_id]["task_id"])
            or not 0
            <= int(record.get("credited_task", -1))
            < int(record.get("requested_task", -1))
            < 8
        ):
            raise ValueError("preview shadow raw diagnostic is invalid")
    for left, right in (
        ("preview_training_group_trace_sha256", "shadow_training_group_trace_sha256"),
        ("preview_final_training_state_sha256", "shadow_final_training_state_sha256"),
        ("preview_transitions", "shadow_transitions"),
    ):
        if shadow.get(left) != shadow.get(right):
            raise ValueError(f"preview shadow independent equality failed: {left}/{right}")
    for key in (
        "preview_training_group_trace_sha256",
        "shadow_training_group_trace_sha256",
        "preview_final_training_state_sha256",
        "shadow_final_training_state_sha256",
    ):
        digest = shadow.get(key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"preview shadow digest malformed: {key}")
    return shadow


def _scale_zero_prefix_preview_valid(prefix: dict) -> bool:
    groups = {int(group["group"]): group for group in prefix["groups"]}
    return all(
        record.get("applied") is False
        and record.get("mutated") is False
        and record.get("frozen_group_parameters") is True
        and math.isfinite(float(record.get("gradient_norm", math.nan)))
        and float(record.get("gradient_norm", 0.0)) > 0.0
        and math.isfinite(float(record.get("hypothetical_update_norm", math.nan)))
        and float(record.get("hypothetical_update_norm", 0.0)) > 0.0
        and int(record.get("after_group", -1)) in groups
        and record.get("transitions")
        == groups[int(record["after_group"])].get("transition_end")
        and int(record.get("requested_task", -1))
        == int(groups[int(record["after_group"])]["task_id"])
        and 0 <= int(record.get("credited_task", -1))
        < int(record.get("requested_task", -1)) < 8
        for record in prefix["previews"]
    )


def _static_outcome_exclusion_audit() -> dict:
    source = _project_file(RUNNER_RELATIVE).read_text(encoding="utf-8")
    tree = ast.parse(source)
    gate_functions = {
        "compute_stage_a_gates",
        "_prefix",
        "_prefix_errors",
        "_prefix_technical_errors",
        "_technical_gate_projection",
    }
    found = set()
    checked = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in gate_functions:
            checked.append(node.name)
            for nested in ast.walk(node):
                if isinstance(nested, ast.Constant) and isinstance(nested.value, str) and nested.value in OUTCOME_FIELDS:
                    found.add(nested.value)
    missing = sorted(gate_functions - set(checked))
    passed = not found and not missing
    if not passed:
        raise ValueError(
            f"runner launch code outcome-exclusion audit failed: found={sorted(found)}, missing={missing}"
        )
    return {
        "passed": True,
        "method": "AST scan of V5 launch functions for forbidden outcome-field literals",
        "checked_functions": sorted(checked),
        "forbidden_fields": sorted(OUTCOME_FIELDS),
        "found_forbidden_fields": [],
    }


def _stage_a_independent_gates(artifact: dict) -> dict:
    counts = [
        int(run.get("optimizer_updates", -1))
        for case_name in CASES
        for run in artifact["cases"][case_name]["runs"]
    ]
    selected = _select_update_budget(counts)
    if selected is None:
        return {"selected_update_budget": None, "all_pass": False, "gate_passes": {}}
    prefixes = {}
    prefix_invariants_pass = True
    for multiplier in LR_MULTIPLIERS:
        for scale in HINDSIGHT_SCALES:
            name = _case_name(multiplier, scale)
            for run in artifact["cases"][name]["runs"]:
                label = f"{name}/seed_{run['seed']}"
                prefix = _prefix(run, selected, label)
                _validate_prefix(prefix, run, selected, label)
                prefixes[label] = {"scale": scale, **prefix}
    fixture_pass = artifact.get("deterministic_scale_fixture") == _fixture()
    shadow_pass = _validate_shadow(artifact).get("passed") is True
    scale_zero_pass = True
    positive_pass = True
    relevance_pass = True
    regime_pass = True
    tv_pass = True
    for name in CASES:
        scale = float(artifact["cases"][name]["config"]["hindsight_scale"])
        cell = [prefix for label, prefix in prefixes.items() if label.startswith(name + "/")]
        natural_count = 0
        for prefix in cell:
            hindsight = {
                record["after_group"]: record
                for record in prefix["updates"]
                if record["source"] == "hindsight_relabel"
            }
            if scale == 0.0:
                preview_ids = [record["after_group"] for record in prefix["previews"]]
                valid = prefix["candidates"] == preview_ids and _scale_zero_prefix_preview_valid(
                    prefix
                ) and not hindsight and sum(
                    record["source"] == "requested_live" for record in prefix["updates"]
                ) == selected
                scale_zero_pass &= valid
                natural_count += len(prefix["candidates"])
            else:
                valid = set(prefix["candidates"]) == set(hindsight) and len(
                    prefix["candidates"]
                ) == len(hindsight) and not prefix["previews"] and all(
                    float(record.get("gradient_norm", 0.0)) > 0.0
                    and float(record.get("update_norm", 0.0)) > 0.0
                    and record.get("credited_task", -1) < record.get("requested_task", -1)
                    for record in hindsight.values()
                )
                positive_pass &= valid
                natural_count += len(hindsight)
            values = [
                float(group["teacher_tv_from_uniform"])
                for group in prefix["groups"]
                if group["transition_start"] >= TV_WARMUP
            ]
            tv_pass &= _teacher_tv_pass(values)
        relevance_pass &= natural_count >= 1
        observed = {
            group["regime"] for prefix in cell for group in prefix["groups"]
        }
        regime_pass &= observed == {"dead", "mixed", "all_pass"}
    projected = _projected_hours_180(
        max(prefix["wall_seconds"] for prefix in prefixes.values())
    )
    gate_passes = {
        "gate_1_lock_schedule_and_technical_invariants": fixture_pass and prefix_invariants_pass,
        "gate_2_fresh_all_27_first_exact_prefix": len(prefixes) == 27,
        "gate_3_scale_zero_preview_and_shadow": scale_zero_pass and shadow_pass,
        "gate_4_positive_updates_and_natural_relabel_coverage": positive_pass and relevance_pass,
        "gate_5_per_cell_natural_regimes": regime_pass,
        "gate_6_per_run_teacher_tv": tv_pass,
        "gate_7_serial_runtime_projection": projected <= 18.0,
    }
    projection = {
        label: {
            "groups": prefix["groups"],
            "updates": prefix["updates"],
            "candidates": prefix["candidates"],
            "previews": prefix["previews"],
            "wall_seconds": prefix["wall_seconds"],
        }
        for label, prefix in prefixes.items()
    }
    serialized_projection = json.dumps(projection, sort_keys=True, allow_nan=False)
    present = sorted(field for field in OUTCOME_FIELDS if f'"{field}"' in serialized_projection)
    if present:
        raise ValueError(f"technical gate projection contains outcome fields: {present}")
    return {
        "selected_update_budget": selected,
        "gate_passes": gate_passes,
        "all_pass": all(gate_passes.values()),
        "projected_hours_180": projected,
        "technical_projection_sha256": _canonical_hash(projection),
        "technical_projection_forbidden_fields_present": present,
    }


def _verify_stage_a(artifact: dict, lock: dict, lock_path: Path) -> dict:
    if artifact.get("schema") != SCHEMA_A or artifact.get("artifact_state") != "complete":
        raise ValueError("V5A artifact schema/state invalid")
    if artifact.get("run_failures"):
        raise ValueError("V5A artifact contains failed runs")
    _validate_protocol(artifact, "stage_a_natural_feasibility", _schedule_a())
    lock_report = _verify_lock(artifact, lock, lock_path, "stage_a_natural_feasibility")
    if tuple(artifact.get("cases", {})) != CASES:
        raise ValueError("V5A case order/set mismatch")
    descriptive_full_run_errors = {}
    for multiplier in LR_MULTIPLIERS:
        for scale in HINDSIGHT_SCALES:
            name = _case_name(multiplier, scale)
            record = artifact["cases"][name]
            if record.get("config") != _expected_config(name, multiplier, scale):
                raise ValueError(f"V5A config mismatch for {name}")
            if [run.get("seed") for run in record.get("runs", [])] != list(STAGE_A_SEEDS):
                raise ValueError(f"V5A seed order mismatch for {name}")
            for run in record["runs"]:
                label = f"{name}/seed_{run['seed']}"
                try:
                    _validate_run(run, label, scale, exact_target=None)
                except Exception as error:
                    # Full post-prefix integrity is reported, but launch is
                    # determined independently from the retained exact prefix.
                    descriptive_full_run_errors[label] = str(error)
    if artifact.get("deterministic_scale_fixture") != _fixture():
        raise ValueError("saved nine-cell deterministic fixture does not reproduce")
    independent = _stage_a_independent_gates(artifact)
    static_audit = _static_outcome_exclusion_audit()
    saved = artifact.get("stage_a_learning_outcome_blind_gates")
    if not isinstance(saved, dict):
        raise ValueError("runner V5A gates are missing")
    if saved.get("selected_update_budget") != independent["selected_update_budget"]:
        raise ValueError("runner and independent V5A budget selection differ")
    if saved.get("all_pass") is not independent["all_pass"] or saved.get(
        "stage_b_authorized"
    ) is not independent["all_pass"]:
        raise ValueError("runner and independent V5A authorization differ")
    for name, passed in independent["gate_passes"].items():
        if saved.get(name, {}).get("passed") is not passed:
            raise ValueError(f"runner and independent V5A gate differ: {name}")
    return {
        "schema": REPORT_SCHEMA_A,
        "v5_stage": "stage_a_natural_feasibility",
        "artifact": None,
        "artifact_sha256": None,
        "lock_sha256": lock_report["lock_sha256"],
        "all_checks_passed": True,
        "selected_optimizer_update_budget": independent["selected_update_budget"],
        "stage_b_factorial_authorized": independent["all_pass"],
        "runner_saved_gates_sha256": _canonical_hash(saved),
        "independent_gate_recomputation": independent,
        "outcome_exclusion_audit": static_audit,
        "source_runtime_lock": lock_report,
        "descriptive_full_run_audit": {
            "authorizing": False,
            "passed": not descriptive_full_run_errors,
            "errors": descriptive_full_run_errors,
        },
        "claim_boundary": (
            "V5A verifies implementation, natural mechanics, and serial feasibility; "
            "it does not estimate a learning-performance effect."
        ),
    }


def _update_auc(run: dict, target: int, label: str) -> float:
    x = np.asarray(run["x_optimizer_updates"], dtype=np.float64)
    y = np.asarray(run["mean_pass_curve"], dtype=np.float64)
    if x.tolist() != list(range(0, target + 1, EVAL_INTERVAL)):
        raise ValueError(f"AUC coordinates invalid in {label}")
    return float(np.trapezoid(y, x) / target)


def _verify_linked_stage_b_authorization(
    artifact: dict, lock: dict, target: int
) -> dict:
    amendment_path = Path(artifact.get("stage_b_amendment", "")).resolve()
    amendment_hash = _sha256(amendment_path)
    if (
        artifact.get("stage_b_amendment_sha256") != amendment_hash
        or lock.get("amendment_sha256") != amendment_hash
    ):
        raise ValueError("V5B amendment hash/link mismatch")
    amendment = _read_json(amendment_path)
    if (
        amendment.get("schema") != AMENDMENT_SCHEMA_B
        or amendment.get("v5_stage") != "stage_b_confirmatory_factorial"
        or amendment.get("explicit_stage_b_authorization") is not True
        or amendment.get("selected_update_budget") != target
        or amendment.get("registered_schedule") != _schedule_b(target)
        or amendment.get("frozen_claim_rule")
        != (
            "C1/C2 require mean>=+0.03 and Holm rejection; C3/C4 require "
            "abs(mean)>=0.03 and Holm rejection; complete 180-run family only."
        )
    ):
        raise ValueError("V5B amendment is not exact/authorized")
    stage_a_path = Path(amendment.get("stage_a_artifact", "")).resolve()
    stage_a_hash = _sha256(stage_a_path)
    if (
        amendment.get("stage_a_artifact_sha256") != stage_a_hash
        or lock.get("stage_a_artifact_sha256") != stage_a_hash
        or artifact.get("stage_a_artifact") != str(stage_a_path)
        or artifact.get("stage_a_artifact_sha256") != stage_a_hash
    ):
        raise ValueError("linked V5A artifact hash/path mismatch")
    stage_a = _read_json(stage_a_path)
    gates = stage_a.get("stage_a_learning_outcome_blind_gates")
    gates_hash = _canonical_hash(gates)
    if (
        stage_a.get("schema") != SCHEMA_A
        or stage_a.get("artifact_state") != "complete"
        or not isinstance(gates, dict)
        or gates.get("all_pass") is not True
        or gates.get("stage_b_authorized") is not True
        or gates.get("selected_update_budget") != target
        or amendment.get("stage_a_gates_sha256") != gates_hash
        or lock.get("stage_a_gates_sha256") != gates_hash
    ):
        raise ValueError("linked V5A gates do not authorize V5B")
    report_path = Path(amendment.get("stage_a_independent_verification", "")).resolve()
    report_hash = _sha256(report_path)
    if (
        amendment.get("stage_a_independent_verification_sha256") != report_hash
        or lock.get("stage_a_independent_verification_sha256") != report_hash
        or artifact.get("stage_a_independent_verification") != str(report_path)
        or artifact.get("stage_a_independent_verification_sha256") != report_hash
    ):
        raise ValueError("linked independent V5A report hash/path mismatch")
    report = _read_json(report_path)
    if not all(
        (
            report.get("schema") == REPORT_SCHEMA_A,
            report.get("v5_stage") == "stage_a_natural_feasibility",
            report.get("all_checks_passed") is True,
            report.get("stage_b_factorial_authorized") is True,
            report.get("selected_optimizer_update_budget") == target,
            report.get("artifact") == str(stage_a_path),
            report.get("artifact_sha256") == stage_a_hash,
            report.get("runner_saved_gates_sha256") == gates_hash,
            report.get("outcome_exclusion_audit", {}).get("passed") is True,
            report.get("lock_sha256")
            == stage_a.get("provenance", {}).get("source_lock_sha256"),
        )
    ):
        raise ValueError("linked independent V5A report does not authorize V5B")
    return {
        "passed": True,
        "amendment": str(amendment_path),
        "amendment_sha256": amendment_hash,
        "stage_a_artifact": str(stage_a_path),
        "stage_a_artifact_sha256": stage_a_hash,
        "stage_a_verification": str(report_path),
        "stage_a_verification_sha256": report_hash,
        "stage_a_gates_sha256": gates_hash,
    }


def _exact_sign_flip(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (20,) or not np.isfinite(array).all():
        raise ValueError("exact V5B sign flip requires 20 finite pairs")
    observed = abs(float(array.mean()))
    extreme = 0
    powers = np.arange(20, dtype=np.uint64)
    for start in range(0, 1 << 20, 1 << 15):
        masks = np.arange(start, start + (1 << 15), dtype=np.uint64)
        signs = ((((masks[:, None] >> powers) & 1).astype(np.float64)) * 2.0) - 1.0
        extreme += int(np.count_nonzero(np.abs(signs @ array / 20.0) >= observed - 1e-15))
    return float(extreme / (1 << 20))


def _bootstrap(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, 20, size=(20_000, 20))].mean(axis=1)
    return [float(value) for value in np.quantile(draws, (0.025, 0.975))]


def _holm(p_values: dict[str, float]) -> dict[str, dict]:
    ordered = sorted((value, name) for name, value in p_values.items())
    running = 0.0
    rejecting = True
    output = {}
    for rank, (value, name) in enumerate(ordered, start=1):
        multiplier = len(ordered) - rank + 1
        running = max(running, multiplier * value)
        reject = rejecting and value <= 0.05 / multiplier
        if not reject:
            rejecting = False
        output[name] = {
            "raw_p": value,
            "holm_adjusted_p": min(running, 1.0),
            "reject_familywise_0.05": bool(reject),
        }
    return output


CONTRASTS = {
    "C1": {_case_name(1, 1): 1.0, _case_name(1, 0): -1.0},
    "C2": {_case_name(1, 2): 1.0, _case_name(1, 1): -1.0},
    "C3": {
        _case_name(0.5, 2): 1.0,
        _case_name(0.5, 0): -1.0,
        _case_name(1, 1): -1.0,
        _case_name(1, 0): 1.0,
    },
    "C4": {
        _case_name(1, 2): 1.0,
        _case_name(1, 0): -1.0,
        _case_name(2, 1): -1.0,
        _case_name(2, 0): 1.0,
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


def _verify_stage_b(artifact: dict, lock: dict, lock_path: Path) -> dict:
    if artifact.get("schema") != SCHEMA_B or artifact.get("artifact_state") != "complete":
        raise ValueError("V5B artifact schema/state invalid")
    if artifact.get("run_failures"):
        raise ValueError("V5B contains failed runs; all-or-nothing family invalid")
    target = int(lock.get("registered_schedule", {}).get("optimizer_update_target", -1))
    schedule = _schedule_b(target)
    _validate_protocol(artifact, "stage_b_confirmatory_factorial", schedule)
    lock_report = _verify_lock(artifact, lock, lock_path, "stage_b_confirmatory_factorial")
    authorization_report = _verify_linked_stage_b_authorization(artifact, lock, target)
    if tuple(artifact.get("cases", {})) != CASES:
        raise ValueError("V5B case order/set mismatch")
    per_case = {}
    independent_case_summaries = {}
    for multiplier in LR_MULTIPLIERS:
        for scale in HINDSIGHT_SCALES:
            name = _case_name(multiplier, scale)
            record = artifact["cases"][name]
            if record.get("config") != _expected_config(name, multiplier, scale):
                raise ValueError(f"V5B config mismatch for {name}")
            runs = record.get("runs", [])
            if [run.get("seed") for run in runs] != list(STAGE_B_SEEDS):
                raise ValueError(f"V5B seed order mismatch for {name}")
            values = {}
            source_by_seed = {}
            transitions_by_seed = {}
            for run in runs:
                label = f"{name}/seed_{run['seed']}"
                _validate_run(run, label, scale, exact_target=target)
                norms = _source_norms(run, label)
                if run.get("transition_cap_censored") is not False or run.get(
                    "reached_optimizer_update_budget"
                ) is not True or run.get("optimizer_updates") != target:
                    raise ValueError(f"V5B terminal budget flags invalid in {label}")
                auc = _update_auc(run, target, label)
                _assert_close(run.get("auc_mean_pass_by_optimizer_updates"), auc, f"{label}.auc")
                values[int(run["seed"])] = auc
                source_by_seed[str(run["seed"])] = {
                    source: {
                        "count": record["count"],
                        "cumulative_step_norm_M": record["M"],
                        "cumulative_squared_step_norm_Q": record["Q"],
                    }
                    for source, record in norms.items()
                }
                transitions_by_seed[str(run["seed"])] = int(run["transitions"])
            per_case[name] = values
            vector = np.asarray([values[seed] for seed in STAGE_B_SEEDS])
            independent_case_summaries[name] = {
                "metric": "auc_mean_pass_by_optimizer_updates",
                "n_seeds": 20,
                "mean": float(vector.mean()),
                "sample_std": float(vector.std(ddof=1)),
                "per_seed": vector.tolist(),
                "source_step_norms_per_seed": source_by_seed,
                "transitions_to_target_per_seed": transitions_by_seed,
            }
    results = {}
    raw_p = {}
    for index, (name, coefficients) in enumerate(CONTRASTS.items()):
        values = np.asarray(
            [
                sum(coefficient * per_case[cell][seed] for cell, coefficient in coefficients.items())
                for seed in STAGE_B_SEEDS
            ],
            dtype=np.float64,
        )
        p_value = _exact_sign_flip(values)
        results[name] = {
            "description": CONTRAST_DESCRIPTIONS[name],
            "coefficients": coefficients,
            "metric": "auc_mean_pass_by_optimizer_updates",
            "n_pairs": 20,
            "sign_assignments_enumerated": 1 << 20,
            "mean_contrast": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "mean_ci95_paired_seed_bootstrap": _bootstrap(values, 55_000 + index),
            "exact_paired_sign_flip_p_two_sided": p_value,
            "per_seed_contrast": values.tolist(),
        }
        raw_p[name] = p_value
    corrections = _holm(raw_p)
    for name in results:
        results[name].update(corrections[name])
    runner = artifact.get("paired_scale_contrasts")
    if not isinstance(runner, dict) or set(runner) != set(results):
        raise ValueError("runner V5B contrast family missing/different")
    for name, independent in results.items():
        saved = runner[name]
        if (
            saved.get("coefficients") != independent["coefficients"]
            or saved.get("description") != independent["description"]
            or saved.get("metric") != independent["metric"]
        ):
            raise ValueError(f"runner/independent V5B mismatch {name}.coefficients")
        for key in (
            "n_pairs",
            "sign_assignments_enumerated",
            "mean_contrast",
            "sample_std",
            "mean_ci95_paired_seed_bootstrap",
            "exact_paired_sign_flip_p_two_sided",
            "raw_p",
            "holm_adjusted_p",
            "reject_familywise_0.05",
            "per_seed_contrast",
        ):
            if isinstance(independent[key], list):
                if not np.allclose(saved.get(key), independent[key], rtol=0.0, atol=1e-12):
                    raise ValueError(f"runner/independent V5B mismatch {name}.{key}")
            elif isinstance(independent[key], bool):
                if saved.get(key) is not independent[key]:
                    raise ValueError(f"runner/independent V5B mismatch {name}.{key}")
            else:
                _assert_close(saved.get(key), independent[key], f"{name}.{key}")
    saved_summaries = artifact.get("stage_b_case_summaries")
    if not isinstance(saved_summaries, dict) or set(saved_summaries) != set(CASES):
        raise ValueError("runner V5B case summaries are missing/different")
    for name, expected in independent_case_summaries.items():
        saved = saved_summaries[name]
        if (
            saved.get("metric") != expected["metric"]
            or saved.get("n_seeds") != 20
            or saved.get("source_step_norms_per_seed") != expected["source_step_norms_per_seed"]
            or saved.get("transitions_to_target_per_seed") != expected["transitions_to_target_per_seed"]
        ):
            raise ValueError(f"runner V5B case diagnostic summary mismatch: {name}")
        for key in ("mean", "sample_std"):
            _assert_close(saved.get(key), expected[key], f"{name}.{key}")
        if not np.allclose(saved.get("per_seed"), expected["per_seed"], rtol=0.0, atol=1e-12):
            raise ValueError(f"runner V5B per-seed AUC summary mismatch: {name}")
    expected_multiplicity = {
        "family": ["C1", "C2", "C3", "C4"],
        "all_or_nothing_180_run_validation": True,
        "metric": "auc_mean_pass_by_optimizer_updates",
        "method": "Holm step-down",
        "familywise_alpha": 0.05,
        "test": "exact two-sided paired sign-flip over all 2^20 assignments",
    }
    if artifact.get("scale_multiplicity") != expected_multiplicity:
        raise ValueError("runner V5B multiplicity metadata mismatch")
    independent_decision = _predeclared_scale_decision(results)
    if artifact.get("predeclared_scale_decision") != independent_decision:
        raise ValueError("runner V5B predeclared decision mismatch")
    if artifact.get("analysis_status") != {
        "performed": True,
        "all_180_runs_valid": True,
        "all_final_update_coordinates_equal_selected_budget": True,
        "selected_update_budget": target,
    }:
        raise ValueError("runner V5B analysis-status metadata mismatch")
    return {
        "schema": REPORT_SCHEMA_B,
        "v5_stage": "stage_b_confirmatory_factorial",
        "artifact": None,
        "artifact_sha256": None,
        "lock_sha256": lock_report["lock_sha256"],
        "all_checks_passed": True,
        "all_or_nothing_180_run_family_valid": True,
        "selected_optimizer_update_budget": target,
        "independent_contrasts": results,
        "independent_case_summaries": independent_case_summaries,
        "independent_predeclared_scale_decision": independent_decision,
        "source_runtime_lock": lock_report,
        "linked_stage_b_authorization": authorization_report,
        "assumption": (
            "exact 2^20 enumeration is conditional on independent seed-level pairs "
            "and sign-exchangeable null contrasts"
        ),
    }


def verify(artifact_path: Path, lock_path: Path) -> dict:
    artifact_path, lock_path = artifact_path.resolve(), lock_path.resolve()
    artifact, lock = _read_json(artifact_path), _read_json(lock_path)
    schema = artifact.get("schema")
    if schema == SCHEMA_A:
        report = _verify_stage_a(artifact, lock, lock_path)
    elif schema == SCHEMA_B:
        report = _verify_stage_b(artifact, lock, lock_path)
    else:
        raise ValueError(f"unknown V5 artifact schema {schema!r}")
    report["artifact"] = str(artifact_path)
    report["artifact_sha256"] = _sha256(artifact_path)
    report["lock"] = str(lock_path)
    return report


def _write_exclusive(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite verification report {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
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
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify(args.artifact, args.lock)
        if args.output is None:
            print(json.dumps(report, indent=2, allow_nan=False))
        else:
            _write_exclusive(args.output, report)
            print(f"wrote independent V5 verification: {args.output.resolve()}")
    except (FileNotFoundError, FileExistsError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
