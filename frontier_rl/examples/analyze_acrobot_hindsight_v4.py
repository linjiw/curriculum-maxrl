"""Independent read-only verifier for Acrobot hindsight V4 artifacts.

Stage A verifies the three-cell, scale-zero feasibility artifact and applies
only the effect-blind launch gates frozen in ``ACROBOT_HINDSIGHT_PROTOCOL_V4``.
Stage B verifies the complete paired 3-by-3 factorial and independently
recomputes its update-indexed AUCs, exact sign-flip tests, bootstrap intervals,
and Holm family.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import tempfile
from pathlib import Path
import gymnasium
import numpy as np

from frontier_rl.teacher import FrontierTeacher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path(__file__).with_name("ACROBOT_HINDSIGHT_PROTOCOL_V4.md")
PROTOCOL_RELATIVE = "frontier_rl/examples/ACROBOT_HINDSIGHT_PROTOCOL_V4.md"
SCHEMA_A = "curriculum-maxrl/acrobot-hindsight-v4a-artifact/v1"
SCHEMA_B = "curriculum-maxrl/acrobot-hindsight-v4b-artifact/v1"
LOCK_SCHEMA_A = "curriculum-maxrl/acrobot-hindsight-v4a-source-lock/v1"
LOCK_SCHEMA_B = "curriculum-maxrl/acrobot-hindsight-v4b-source-lock/v1"
BASE_LEARNING_RATE = 3e-4
LR_MULTIPLIERS = (0.5, 1.0, 2.0)
HINDSIGHT_SCALES = (0.0, 1.0, 2.0)
STAGE_A_SEEDS = list(range(13_000, 13_003))
STAGE_B_SEEDS = list(range(14_000, 14_010))
STAGE_A_TARGET_UPDATES = 400
ALLOWED_FALLBACK_UPDATES = 250
TRANSITION_CAP = 4_000_000
EVAL_INTERVAL_UPDATES = 50
EVAL_EPISODES_PER_TASK = 32
WARMUP_TRANSITIONS = 200_000
PRIMARY_METRIC = "auc_mean_pass_by_optimizer_updates"
THRESHOLDS = (-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.7, 1.0)
SHADOW_SEED = 100
SHADOW_UPDATE_BUDGET = 3
SHADOW_TRANSITION_CAP = 80_000
SHADOW_EVAL_EPISODES = 4
REQUIRED_SOURCE_FILES = {
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
}


def _float_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _case_name(multiplier: float, scale: float) -> str:
    return f"lr_mult_{_float_label(multiplier)}_hs_{_float_label(scale)}"


STAGE_A_CASES = tuple(_case_name(multiplier, 0.0) for multiplier in LR_MULTIPLIERS)
STAGE_B_CASES = tuple(
    _case_name(multiplier, scale)
    for multiplier in LR_MULTIPLIERS
    for scale in HINDSIGHT_SCALES
)
CONTRAST_SPECS = {
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
CONTRAST_DESCRIPTIONS = {
    "C1": "base-LR scale 1 minus scale 0",
    "C2": "base-LR scale 2 minus scale 1",
    "C3": "half/base LR restricted-separability diagnostic",
    "C4": "base/double LR restricted-separability diagnostic",
}


def _expected_schedule(stage: str, selected_budget: int | None = None) -> dict:
    common = {
        "paired_seeds": STAGE_A_SEEDS if stage == "stage_a_feasibility" else STAGE_B_SEEDS,
        "base_learning_rate": BASE_LEARNING_RATE,
        "learning_rate_multipliers": list(LR_MULTIPLIERS),
        "hindsight_scales": (
            [0.0] if stage == "stage_a_feasibility" else list(HINDSIGHT_SCALES)
        ),
        "condition_names": (
            list(STAGE_A_CASES)
            if stage == "stage_a_feasibility"
            else list(STAGE_B_CASES)
        ),
        "optimizer_update_target": (
            STAGE_A_TARGET_UPDATES
            if stage == "stage_a_feasibility"
            else selected_budget
        ),
        "transition_group_start_cap": TRANSITION_CAP,
        "maximum_complete_group_overshoot": 8_000,
        "eval_interval_optimizer_updates": EVAL_INTERVAL_UPDATES,
        "eval_episodes_per_task": EVAL_EPISODES_PER_TASK,
    }
    if stage == "stage_a_feasibility":
        common.update(
            {
                "v4_stage": stage,
                "single_allowed_fallback_update_target": ALLOWED_FALLBACK_UPDATES,
                "shadow_test": {
                    "seed": SHADOW_SEED,
                    "learning_rate_multiplier": 1.0,
                    "optimizer_update_budget": SHADOW_UPDATE_BUDGET,
                    "transition_group_start_cap": SHADOW_TRANSITION_CAP,
                    "eval_episodes_per_task": SHADOW_EVAL_EPISODES,
                },
            }
        )
        # Match the runner's intentionally frozen insertion order only for
        # readable reports; equality itself is key-order independent.
        ordered = {
            "v4_stage": common.pop("v4_stage"),
            **common,
        }
        return ordered
    if selected_budget not in (ALLOWED_FALLBACK_UPDATES, STAGE_A_TARGET_UPDATES):
        raise ValueError("V4B selected optimizer budget must be exactly 250 or 400")
    return {"v4_stage": stage, **common}


def _expected_protocol(stage: str, schedule: dict) -> dict:
    protocol = {
        "v4_stage": stage,
        "protocol_document": PROTOCOL_RELATIVE,
        "registered_schedule": schedule,
        "gymnasium_environment": "Acrobot-v1",
        "thresholds": list(THRESHOLDS),
        "verifier": "strict post-transition Acrobot tip height > threshold",
        "max_episode_steps": 500,
        "n_rollouts": 16,
        "sampling": "frontier-u_16 teacher",
        "teacher_utility": "1-(1-p)^N-p",
        "teacher_gamma": 1.0,
        "teacher_decay": 0.7,
        "teacher_floor": 0.1,
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
        protocol.update(
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
        protocol.update(
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
    return protocol


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_close(actual, expected, label: str, atol: float = 1e-12) -> None:
    try:
        actual_float = float(actual)
        expected_float = float(expected)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {actual!r}") from error
    if not math.isclose(actual_float, expected_float, rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _assert_nested_equal(actual, expected, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{label} object keys mismatch")
        for key, value in expected.items():
            _assert_nested_equal(actual[key], value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{label} list length mismatch")
        for index, value in enumerate(expected):
            _assert_nested_equal(actual[index], value, f"{label}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")
        return
    if isinstance(expected, (int, float, np.integer, np.floating)):
        _assert_close(actual, expected, label)
        return
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _require_finite(value, label: str) -> None:
    """Reject JSON non-finite numbers recursively while ignoring text/nulls."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, np.integer)):
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite value in {label}")
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            _require_finite(nested, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _require_finite(nested, f"{label}[{index}]")


def _live_runtime() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
    }


def _project_file(relative: str) -> Path:
    candidate = (PROJECT_ROOT / relative).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"source-lock path escapes project: {relative}") from error
    if not candidate.is_file():
        raise ValueError(f"source-lock path is not a file: {relative}")
    return candidate


def _verify_lock(
    artifact: dict,
    lock: dict,
    stage: str,
    *,
    lock_sha256: str | None,
) -> dict:
    runtime = _live_runtime()
    expected_schema = LOCK_SCHEMA_A if stage == "stage_a_feasibility" else LOCK_SCHEMA_B
    if lock.get("schema") != expected_schema or lock.get("v4_stage") != stage:
        raise ValueError("source lock schema/stage differs from the V4 artifact")
    if lock.get("runtime") != runtime:
        raise ValueError(
            f"analysis runtime differs from source lock: {runtime!r} != "
            f"{lock.get('runtime')!r}"
        )
    provenance = artifact.get("provenance", {})
    artifact_runtime = provenance.get("runtime")
    if artifact_runtime != runtime:
        raise ValueError("artifact runtime differs from source lock and live runtime")

    recorded_lock_hash = provenance.get("source_lock_sha256")
    if recorded_lock_hash is not None:
        if lock_sha256 is None:
            raise ValueError("source-lock file hash is required for this V4 artifact")
        if recorded_lock_hash != lock_sha256:
            raise ValueError("artifact was created under a different source-lock file")

    if stage == "stage_a_feasibility":
        expected_schedule = _expected_schedule(stage)
    else:
        schedule = lock.get("registered_schedule")
        if not isinstance(schedule, dict):
            raise ValueError("V4B lock lacks its explicit registered schedule")
        expected_schedule = _expected_schedule(
            stage, schedule.get("optimizer_update_target")
        )
    if lock.get("registered_schedule") != expected_schedule:
        raise ValueError("source-lock schedule differs from the exact frozen V4 schedule")

    expected_hashes = lock.get("source_sha256")
    recorded_hashes = provenance.get("source_sha256")
    if not isinstance(expected_hashes, dict) or not isinstance(recorded_hashes, dict):
        raise ValueError("source lock and artifact must contain source_sha256 objects")
    if set(expected_hashes) != REQUIRED_SOURCE_FILES:
        raise ValueError("source lock does not exactly cover the frozen V4 source/test set")
    if set(recorded_hashes) != set(expected_hashes):
        raise ValueError("artifact source hash key set differs from source lock")
    for relative, expected in expected_hashes.items():
        if recorded_hashes[relative] != expected:
            raise ValueError(f"artifact source-lock mismatch for {relative}")
        if _sha256(_project_file(relative)) != expected:
            raise ValueError(f"current source-lock mismatch for {relative}")
    return {
        "passed": True,
        "runtime": runtime,
        "lock_schema": expected_schema,
        "registered_schedule": expected_schedule,
        "checked_source_files": sorted(expected_hashes),
    }


def _protocol_stage(protocol: dict) -> str:
    stage = protocol.get("v4_stage")
    if stage not in {"stage_a_feasibility", "stage_b_factorial"}:
        raise ValueError(f"unknown or noncanonical V4 stage: {stage!r}")
    return stage


def _check_protocol_fields(protocol: dict, expected: dict) -> None:
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(
                f"protocol mismatch for {key}: {protocol.get(key)!r} != {value!r}"
            )


def _expected_config(case_name: str, multiplier: float, scale: float) -> dict:
    return {
        "name": case_name,
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


def _validate_config(config: dict, case_name: str, multiplier: float, scale: float) -> None:
    expected = _expected_config(case_name, multiplier, scale)
    if set(config) != set(expected):
        raise ValueError(f"configuration key mismatch for {case_name}")
    for key, value in expected.items():
        actual = config[key]
        if isinstance(value, float):
            _assert_close(actual, value, f"{case_name}.{key}", atol=1e-15)
        elif actual != value:
            raise ValueError(
                f"configuration mismatch for {case_name}.{key}: "
                f"{actual!r} != {value!r}"
            )


def _regime(success_count: int) -> str:
    return "dead" if success_count == 0 else "all_pass" if success_count == 16 else "mixed"


def _validate_curve_coordinates(
    run: dict, case_name: str, *, allow_duplicate_terminal_update: bool
) -> None:
    x_transitions = run.get("x_transitions")
    x_updates = run.get("x_optimizer_updates")
    mean_pass = run.get("mean_pass_curve")
    if not all(isinstance(values, list) for values in (x_transitions, x_updates, mean_pass)):
        raise ValueError(f"missing evaluation curves in {case_name}")
    if len(x_transitions) != len(x_updates) or len(x_updates) != len(mean_pass):
        raise ValueError(f"evaluation curve length mismatch in {case_name}")
    if len(x_updates) < 2 or x_transitions[0] != 0 or x_updates[0] != 0:
        raise ValueError(f"evaluation curves lack initialization in {case_name}")
    if any(right <= left for left, right in zip(x_transitions, x_transitions[1:])):
        raise ValueError(f"transition evaluation coordinates are not increasing in {case_name}")
    update_differences = [right - left for left, right in zip(x_updates, x_updates[1:])]
    allowed_terminal_duplicate = (
        allow_duplicate_terminal_update
        and update_differences
        and update_differences[-1] == 0
        and all(difference > 0 for difference in update_differences[:-1])
    )
    if not all(difference > 0 for difference in update_differences) and not allowed_terminal_duplicate:
        raise ValueError(f"update evaluation coordinates are not monotone in {case_name}")
    expected_updates = list(range(0, int(run.get("optimizer_updates")) + 1, EVAL_INTERVAL_UPDATES))
    if expected_updates[-1] != run.get("optimizer_updates"):
        expected_updates.append(run.get("optimizer_updates"))
    allowed_cadences = [expected_updates]
    if allow_duplicate_terminal_update:
        allowed_cadences.append(expected_updates + [run.get("optimizer_updates")])
    if x_updates not in allowed_cadences:
        raise ValueError(f"evaluation update cadence mismatch in {case_name}")
    if x_transitions[-1] != run.get("transitions"):
        raise ValueError(f"terminal transition coordinate mismatch in {case_name}")
    if x_updates[-1] != run.get("optimizer_updates"):
        raise ValueError(f"terminal optimizer coordinate mismatch in {case_name}")
    if any(not 0.0 <= float(value) <= 1.0 for value in mean_pass):
        raise ValueError(f"mean-pass curve leaves [0,1] in {case_name}")
    pass_rates = run.get("pass_rate_curve")
    hardest = run.get("hardest_pass_curve")
    if (
        not isinstance(pass_rates, list)
        or not isinstance(hardest, list)
        or len(pass_rates) != len(mean_pass)
        or len(hardest) != len(mean_pass)
    ):
        raise ValueError(f"missing per-task evaluation curve in {case_name}")
    for index, row in enumerate(pass_rates):
        if (
            not isinstance(row, list)
            or len(row) != 8
            or any(not 0.0 <= float(value) <= 1.0 for value in row)
        ):
            raise ValueError(f"invalid per-task pass rates in {case_name}")
        _assert_close(mean_pass[index], np.mean(row), f"{case_name} mean pass {index}")
        _assert_close(hardest[index], row[-1], f"{case_name} hardest pass {index}")
    if "initial_mean_pass" in run:
        _assert_close(run["initial_mean_pass"], mean_pass[0], f"{case_name} initial mean")
    if "final_mean_pass" in run:
        _assert_close(run["final_mean_pass"], mean_pass[-1], f"{case_name} final mean")
    for key, values in run.items():
        if key.endswith("_curve") and isinstance(values, list) and len(values) != len(mean_pass):
            raise ValueError(f"evaluation curve length mismatch for {case_name}.{key}")
    preserved = run.get("evaluation_rng_preserved")
    if preserved is not None and (
        len(preserved) != len(mean_pass) or not all(value is True for value in preserved)
    ):
        raise ValueError(f"evaluation RNG preservation trace failed in {case_name}")


def _validate_groups(run: dict, case_name: str) -> set[str]:
    groups = run.get("group_diagnostics")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"missing group diagnostics in {case_name}")
    if len(groups) != run.get("sampled_groups"):
        raise ValueError(f"sampled-group count mismatch in {case_name}")
    regimes: set[str] = set()
    previous_end = 0
    previous_updates = 0
    task_groups = [0] * 8
    task_rollouts = [0] * 8
    task_successes = [0] * 8
    task_transitions = [0] * 8
    live = dead = all_pass = 0
    seed = run.get("seed")
    if not isinstance(seed, int):
        raise ValueError(f"invalid seed in {case_name}")
    teacher = FrontierTeacher(
        8,
        16,
        decay=0.7,
        floor=0.1,
        gamma=1.0,
        seed=seed + 10_000,
    )
    for index, group in enumerate(groups, start=1):
        if group.get("group") != index:
            raise ValueError(f"group index mismatch in {case_name}")
        start = group.get("transition_start")
        end = group.get("transition_end")
        n_transitions = group.get("n_transitions")
        if (
            start != previous_end
            or end - start != n_transitions
            or not 0 < n_transitions <= 8_000
        ):
            raise ValueError(f"noncontiguous group transitions in {case_name}")
        previous_end = end
        task = group.get("task_id")
        success = group.get("success_count")
        if not isinstance(task, int) or not 0 <= task < 8:
            raise ValueError(f"invalid task id in {case_name}")
        if not isinstance(success, int) or not 0 <= success <= 16:
            raise ValueError(f"invalid success count in {case_name}")
        expected_regime = _regime(success)
        if group.get("regime") != expected_regime:
            raise ValueError(f"group regime mismatch in {case_name}")
        regimes.add(expected_regime)
        live += expected_regime == "mixed"
        dead += expected_regime == "dead"
        all_pass += expected_regime == "all_pass"
        task_groups[task] += 1
        task_rollouts[task] += 16
        task_successes[task] += success
        task_transitions[task] += n_transitions
        tv = float(group.get("teacher_tv_from_uniform"))
        probability = float(group.get("sampled_task_probability"))
        if not (math.isfinite(tv) and 0.0 <= tv <= 1.0):
            raise ValueError(f"invalid teacher TV in {case_name}")
        if not (math.isfinite(probability) and 0.0 <= probability <= 1.0):
            raise ValueError(f"invalid sampled-task probability in {case_name}")
        expected_probabilities = np.asarray(teacher.distribution(), dtype=np.float64)
        expected_tv = float(0.5 * np.abs(expected_probabilities - 0.125).sum())
        expected_task = int(teacher.rng.choice(8, p=expected_probabilities))
        if task != expected_task:
            raise ValueError(f"frontier-teacher task trace mismatch in {case_name}")
        _assert_close(tv, expected_tv, f"{case_name} group {index} teacher TV")
        _assert_close(
            probability,
            expected_probabilities[task],
            f"{case_name} group {index} sampled-task probability",
        )
        replay_rewards = np.zeros(16, dtype=np.float64)
        replay_rewards[:success] = 1.0
        teacher.observe(task, replay_rewards)
        updates = group.get("optimizer_updates_after_group")
        if not isinstance(updates, int) or not previous_updates <= updates <= previous_updates + 1:
            raise ValueError(f"optimizer counter jump in {case_name}")
        source = group.get("update_source")
        if updates == previous_updates and source is not None:
            raise ValueError(f"uncounted update source in {case_name}")
        if updates == previous_updates + 1 and source not in {
            "requested_live",
            "hindsight_relabel",
        }:
            raise ValueError(f"missing update source in {case_name}")
        previous_updates = updates

    if previous_end != run.get("transitions"):
        raise ValueError(f"group transition total mismatch in {case_name}")
    if previous_updates != run.get("optimizer_updates"):
        raise ValueError(f"group optimizer total mismatch in {case_name}")
    expected_counts = {
        "live_groups": live,
        "dead_groups": dead,
        "all_pass_groups": all_pass,
        "task_groups": task_groups,
        "task_rollouts": task_rollouts,
        "task_successes": task_successes,
        "task_transitions": task_transitions,
    }
    for key, expected in expected_counts.items():
        if run.get(key) != expected:
            raise ValueError(f"raw accounting mismatch for {case_name}.{key}")
    if run.get("rollout_attempts") != 16 * len(groups):
        raise ValueError(f"rollout accounting mismatch in {case_name}")
    if run.get("optimizer_updates") != (
        run.get("live_applied_updates") + run.get("relabeled_groups")
    ):
        raise ValueError(f"update-source accounting mismatch in {case_name}")
    update_diagnostics = run.get("update_diagnostics", [])
    if len(update_diagnostics) != run.get("optimizer_updates"):
        raise ValueError(f"update diagnostic count mismatch in {case_name}")
    group_by_id = {group["group"]: group for group in groups}
    for optimizer_update, diagnostic in enumerate(update_diagnostics, start=1):
        if diagnostic.get("optimizer_update") != optimizer_update:
            raise ValueError(f"update diagnostic order mismatch in {case_name}")
        after_group = diagnostic.get("after_group")
        if after_group not in group_by_id:
            raise ValueError(f"update diagnostic names an unknown group in {case_name}")
        group = group_by_id[after_group]
        if (
            diagnostic.get("source") != group.get("update_source")
            or diagnostic.get("transitions") != group.get("transition_end")
            or diagnostic.get("requested_task") != group.get("task_id")
        ):
            raise ValueError(f"group/update diagnostic mismatch in {case_name}")
    source_counts = {
        source: sum(group.get("update_source") == source for group in groups)
        for source in ("requested_live", "hindsight_relabel")
    }
    if (
        source_counts["requested_live"] != run.get("live_applied_updates")
        or source_counts["hindsight_relabel"] != run.get("relabeled_groups")
    ):
        raise ValueError(f"applied-update source totals do not reproduce in {case_name}")
    return regimes


def _validate_common_run(
    run: dict, case_name: str, *, allow_duplicate_terminal_update: bool = False
) -> set[str]:
    required_flags = (
        "numeric_valid",
        "accounting_valid",
        "verifier_relabel_checks_valid",
        "evaluation_cadence_invariant",
    )
    if not all(run.get(key) is True for key in required_flags):
        raise ValueError(f"invalid run in {case_name}, seed {run.get('seed')}")
    if run.get("total_parameters") != 640 or run.get("active_parameters_per_task") != 640:
        raise ValueError(f"shared-H64 parameter-count invariant failed in {case_name}")
    if run.get("training_group_trace_groups") != run.get("sampled_groups"):
        raise ValueError(f"training group trace count mismatch in {case_name}")
    for key in ("training_group_trace_sha256", "final_training_state_sha256"):
        digest = run.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"missing or malformed {key} in {case_name}")
    if not isinstance(run.get("wall_seconds"), (int, float)) or run["wall_seconds"] <= 0:
        raise ValueError(f"invalid wall time in {case_name}")
    _require_finite(run, f"{case_name}.seed_{run.get('seed')}")
    regimes = _validate_groups(run, case_name)
    _step_norm_diagnostics(run, case_name)
    _validate_curve_coordinates(
        run,
        case_name,
        allow_duplicate_terminal_update=allow_duplicate_terminal_update,
    )
    if run["transitions"] > TRANSITION_CAP + 8_000:
        raise ValueError(f"transition-cap overshoot exceeds one complete group in {case_name}")
    if any(
        group["transition_start"] >= TRANSITION_CAP
        for group in run["group_diagnostics"]
    ):
        raise ValueError(f"a group began at or after the frozen cap in {case_name}")
    return regimes


def _normalized_update_auc(run: dict, case_name: str, target_updates: int) -> float:
    x = np.asarray(run["x_optimizer_updates"], dtype=np.float64)
    y = np.asarray(run["mean_pass_curve"], dtype=np.float64)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError(f"AUC curve length mismatch in {case_name}")
    if x[0] != 0 or x[-1] != target_updates or np.any(np.diff(x) <= 0):
        raise ValueError(f"AUC update coordinates do not cover the frozen target in {case_name}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f"non-finite AUC curve in {case_name}")
    return float(np.trapezoid(y, x) / target_updates)


def _step_norm_diagnostics(run: dict, case_name: str) -> dict[str, dict[str, float | int]]:
    totals = {
        "requested_live": {
            "count": 0,
            "cumulative_step_norm_M": 0.0,
            "cumulative_squared_step_norm_Q": 0.0,
        },
        "hindsight_relabel": {
            "count": 0,
            "cumulative_step_norm_M": 0.0,
            "cumulative_squared_step_norm_Q": 0.0,
        },
    }
    for index, diagnostic in enumerate(run["update_diagnostics"], start=1):
        if diagnostic.get("optimizer_update") != index:
            raise ValueError(f"update diagnostic index mismatch in {case_name}")
        source = diagnostic.get("source")
        if source not in totals:
            raise ValueError(f"unknown applied-update source in {case_name}")
        norm = float(diagnostic.get("update_norm", 0.0))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"applied update lacks a positive finite norm in {case_name}")
        totals[source]["count"] += 1
        totals[source]["cumulative_step_norm_M"] += norm
        totals[source]["cumulative_squared_step_norm_Q"] += norm * norm
    saved = run.get("source_step_norms_full_run")
    if not isinstance(saved, dict) or set(saved) != set(totals):
        raise ValueError(f"missing saved source-wise step norms in {case_name}")
    for source, expected in totals.items():
        actual = saved.get(source, {})
        if actual.get("count") != expected["count"]:
            raise ValueError(f"saved source step count mismatch in {case_name}")
        _assert_close(
            actual.get("cumulative_step_norm_M"),
            expected["cumulative_step_norm_M"],
            f"{case_name}.{source}.M",
        )
        _assert_close(
            actual.get("cumulative_squared_step_norm_Q"),
            expected["cumulative_squared_step_norm_Q"],
            f"{case_name}.{source}.Q",
        )
    return totals


def _qualifying_previews(run: dict, case_name: str) -> list[dict]:
    previews = run.get("auxiliary_gradient_diagnostics")
    if not isinstance(previews, list):
        raise ValueError(f"missing auxiliary previews in {case_name}")
    if run.get("unscaled_aux_gradient_previews") != len(previews):
        raise ValueError(f"auxiliary preview count mismatch in {case_name}")
    if run.get("relabel_candidates") != len(previews):
        raise ValueError(f"scale-zero relabel/preview mismatch in {case_name}")
    eligible_groups = run.get("eligible_relabel_candidate_groups")
    preview_groups = [preview.get("after_group") for preview in previews]
    if eligible_groups != preview_groups:
        raise ValueError(f"eligible relabel groups do not match previews in {case_name}")
    if any(
        not isinstance(group, int) or group < 1
        for group in eligible_groups
    ) or any(right <= left for left, right in zip(eligible_groups, eligible_groups[1:])):
        raise ValueError(f"eligible relabel group ids are invalid in {case_name}")
    qualifying = []
    for preview in previews:
        _require_finite(preview, f"{case_name}.auxiliary_preview")
        if (
            preview.get("applied") is not False
            or preview.get("mutated") is not False
            or preview.get("frozen_group_parameters") is not True
        ):
            raise ValueError(f"scale-zero preview mutated or applied an update in {case_name}")
        gradient_norm = float(preview.get("gradient_norm", 0.0))
        hypothetical_norm = float(preview.get("hypothetical_update_norm", 0.0))
        if gradient_norm > 0.0 and hypothetical_norm > 0.0:
            qualifying.append(preview)
    return qualifying


def _validate_relabel_accounting(run: dict, case_name: str, scale: float) -> None:
    eligible = run.get("eligible_relabel_candidate_groups")
    if not isinstance(eligible, list) or len(eligible) != run.get("relabel_candidates"):
        raise ValueError(f"eligible relabel candidate total mismatch in {case_name}")
    if (
        any(not isinstance(group, int) for group in eligible)
        or len(set(eligible)) != len(eligible)
        or eligible != sorted(eligible)
    ):
        raise ValueError(f"eligible relabel candidate ids are invalid in {case_name}")
    groups = {group["group"]: group for group in run["group_diagnostics"]}
    if any(
        group not in groups or groups[group]["regime"] != "dead"
        for group in eligible
    ):
        raise ValueError(f"relabel candidate does not name a dead group in {case_name}")
    hindsight_update_groups = [
        diagnostic["after_group"]
        for diagnostic in run["update_diagnostics"]
        if diagnostic["source"] == "hindsight_relabel"
    ]
    if any(group not in eligible for group in hindsight_update_groups):
        raise ValueError(f"hindsight update lacks an eligible candidate in {case_name}")
    previews = run.get("auxiliary_gradient_diagnostics")
    if not isinstance(previews, list):
        raise ValueError(f"missing auxiliary-gradient diagnostics in {case_name}")
    if scale == 0.0:
        _qualifying_previews(run, case_name)
        if run.get("relabeled_groups") != 0 or hindsight_update_groups:
            raise ValueError(f"scale-zero run applied a hindsight update in {case_name}")
    elif previews or run.get("unscaled_aux_gradient_previews") != 0:
        raise ValueError(f"positive-scale run unexpectedly saved preview-only records in {case_name}")


def _selected_prefix(run: dict, selected_updates: int, case_name: str) -> tuple[list[dict], int]:
    """Return groups through the first completed group reaching selected_updates."""

    if run["optimizer_updates"] < selected_updates:
        raise ValueError(f"run does not reach selected update prefix in {case_name}")
    previous = 0
    for index, group in enumerate(run["group_diagnostics"]):
        updates = group["optimizer_updates_after_group"]
        if updates == selected_updates:
            if previous != selected_updates - 1:
                raise ValueError(f"selected prefix is not the first target crossing in {case_name}")
            return run["group_diagnostics"][: index + 1], int(group["group"])
        previous = updates
    raise ValueError(f"selected update prefix is absent in {case_name}")


def _prefix_wall_seconds(run: dict, selected_updates: int, case_name: str) -> float:
    timings = run.get("wall_seconds_at_optimizer_updates")
    if not isinstance(timings, dict):
        raise ValueError(f"missing exact prefix wall-time record in {case_name}")
    key = str(selected_updates)
    if key not in timings:
        raise ValueError(f"missing wall time through update {selected_updates} in {case_name}")
    value = float(timings[key])
    if not math.isfinite(value) or value <= 0.0 or value > float(run["wall_seconds"]):
        raise ValueError(f"invalid prefix wall time in {case_name}")
    if "250" in timings and "400" in timings:
        if not 0.0 < float(timings["250"]) <= float(timings["400"]):
            raise ValueError(f"nonmonotone prefix wall times in {case_name}")
    return value


def _verify_artifact_shell(
    artifact: dict, expected_cases: tuple[str, ...], expected_schema: str
) -> None:
    if artifact.get("schema") != expected_schema:
        raise ValueError("artifact schema does not match its canonical V4 stage")
    if artifact.get("artifact_state") != "complete":
        raise ValueError("artifact is not complete")
    if artifact.get("run_failures"):
        raise ValueError("artifact contains failed runs")
    cases = artifact.get("cases")
    if not isinstance(cases, dict) or tuple(cases) != expected_cases:
        raise ValueError("artifact does not contain exactly the ordered V4 cases")


def _stage_a_protocol(artifact: dict) -> None:
    protocol = artifact["protocol"]
    expected = _expected_protocol(
        "stage_a_feasibility", _expected_schedule("stage_a_feasibility")
    )
    if protocol != expected:
        raise ValueError("V4 Stage A protocol does not exactly match the frozen contract")


def _validate_preview_shadow(artifact: dict) -> dict:
    shadow = artifact.get("preview_shadow_equivalence")
    expected_keys = {
        "test_config",
        "passed",
        "identical_training_group_trace",
        "identical_final_training_state",
        "identical_saved_training_projection",
        "eligible_preview_exercised",
        "preview_candidate_groups",
        "preview_diagnostic_groups",
        "preview_training_group_trace_sha256",
        "shadow_training_group_trace_sha256",
        "preview_final_training_state_sha256",
        "shadow_final_training_state_sha256",
        "preview_transitions",
        "shadow_transitions",
    }
    if not isinstance(shadow, dict) or set(shadow) != expected_keys:
        raise ValueError("missing or malformed deterministic preview-shadow evidence")
    expected_config = {
        "seed": SHADOW_SEED,
        "seed_block_status": "historical exploratory; outside V4 blocks",
        "learning_rate": BASE_LEARNING_RATE,
        "optimizer_update_budget": SHADOW_UPDATE_BUDGET,
        "transition_group_start_cap": SHADOW_TRANSITION_CAP,
        "eval_episodes_per_task": SHADOW_EVAL_EPISODES,
    }
    if shadow.get("test_config") != expected_config:
        raise ValueError("preview-shadow test configuration differs from the lock")
    required_true = (
        "passed",
        "identical_training_group_trace",
        "identical_final_training_state",
        "identical_saved_training_projection",
        "eligible_preview_exercised",
    )
    if not all(shadow.get(key) is True for key in required_true):
        raise ValueError("deterministic preview-only/no-hindsight shadow did not pass")
    candidate_groups = shadow.get("preview_candidate_groups")
    diagnostic_groups = shadow.get("preview_diagnostic_groups")
    if (
        not isinstance(candidate_groups, list)
        or not candidate_groups
        or candidate_groups != diagnostic_groups
        or any(not isinstance(group, int) or group < 1 for group in candidate_groups)
    ):
        raise ValueError("preview-shadow candidate/diagnostic group evidence is invalid")
    hash_pairs = (
        (
            "preview_training_group_trace_sha256",
            "shadow_training_group_trace_sha256",
        ),
        ("preview_final_training_state_sha256", "shadow_final_training_state_sha256"),
    )
    for left, right in hash_pairs:
        left_hash = shadow.get(left)
        right_hash = shadow.get(right)
        if (
            not isinstance(left_hash, str)
            or len(left_hash) != 64
            or any(character not in "0123456789abcdef" for character in left_hash)
            or left_hash != right_hash
        ):
            raise ValueError(f"preview-shadow hash evidence mismatch for {left}")
    if (
        not isinstance(shadow.get("preview_transitions"), int)
        or shadow["preview_transitions"] <= 0
        or shadow["preview_transitions"] != shadow.get("shadow_transitions")
    ):
        raise ValueError("preview-shadow transition evidence differs")
    return shadow


def _validate_saved_stage_a_gates(
    artifact: dict, computed_gates: dict, selected_budget: int | None
) -> dict:
    saved = artifact.get("stage_a_effect_blind_gates")
    expected_keys = {
        "effect_blind",
        "uses_evaluation_performance",
        "uses_hindsight_contrast",
        "uses_v3_outcome",
        "selected_update_budget",
        "gate_1_lock_and_implementation_invariants",
        "gate_2_exact_selected_prefix",
        "gate_3_preview_mechanics_and_shadow",
        "gate_4_requested_group_regimes",
        "gate_5_per_run_teacher_tv",
        "gate_6_serial_runtime_projection",
        "all_pass",
        "stage_b_authorized",
        "prefix_diagnostics_sha256",
    }
    if not isinstance(saved, dict) or set(saved) != expected_keys:
        raise ValueError("artifact lacks the exact saved Stage-A gate family")
    if (
        saved.get("effect_blind") is not True
        or saved.get("uses_evaluation_performance") is not False
        or saved.get("uses_hindsight_contrast") is not False
        or saved.get("uses_v3_outcome") is not False
        or saved.get("selected_update_budget") != selected_budget
    ):
        raise ValueError("saved Stage-A effect-blind declaration or U* is invalid")
    gate_mapping = {
        "gate_1_lock_and_implementation_invariants": "source_runtime_schedule_and_run_invariants",
        "gate_2_exact_selected_prefix": "every_selected_prefix_ends_at_selected_update_budget",
        "gate_3_preview_mechanics_and_shadow": "every_run_has_ten_one_to_one_positive_nonmutating_previews",
        "gate_4_requested_group_regimes": "every_cell_observes_all_regimes",
        "gate_5_per_run_teacher_tv": "every_run_post_warmup_teacher_tv_above_0p05",
        "gate_6_serial_runtime_projection": "projected_90_run_serial_runtime_at_most_24h",
    }
    for saved_name, computed_name in gate_mapping.items():
        record = saved.get(saved_name)
        if (
            not isinstance(record, dict)
            or record.get("passed") is not computed_gates[computed_name]
        ):
            raise ValueError(f"saved Stage-A decision does not reproduce for {saved_name}")
    if saved["gate_2_exact_selected_prefix"].get("selected_update_budget") != selected_budget:
        raise ValueError("saved Stage-A prefix gate carries a different U*")
    if saved["gate_3_preview_mechanics_and_shadow"].get(
        "preview_shadow_equivalence"
    ) != artifact["preview_shadow_equivalence"]:
        raise ValueError("saved Stage-A Gate 3 shadow evidence differs from artifact")
    expected_all = computed_gates["all_pass"]
    if saved.get("all_pass") is not expected_all or saved.get(
        "stage_b_authorized"
    ) is not expected_all:
        raise ValueError("saved Stage-A all-pass/authorization decision does not reproduce")
    digest = saved.get("prefix_diagnostics_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("saved Stage-A prefix diagnostic digest is malformed")
    _assert_nested_equal(
        artifact.get("analysis_status"),
        {
            "performed": True,
            "type": "hindsight-effect-blind feasibility gates only",
            "learning_performance_used": False,
        },
        "saved Stage-A analysis status",
    )
    return saved


def _verify_stage_a(artifact: dict, source_lock: dict) -> dict:
    _verify_artifact_shell(artifact, STAGE_A_CASES, SCHEMA_A)
    _stage_a_protocol(artifact)
    shadow = _validate_preview_shadow(artifact)
    validated_runs: dict[str, list[dict]] = {}
    all_reached_400 = True
    every_run_at_least_250 = True
    for multiplier, case_name in zip(LR_MULTIPLIERS, STAGE_A_CASES):
        record = artifact["cases"][case_name]
        _validate_config(record["config"], case_name, multiplier, 0.0)
        runs = record.get("runs", [])
        if [run.get("seed") for run in runs] != STAGE_A_SEEDS:
            raise ValueError(f"seed order mismatch for {case_name}")
        for run in runs:
            _validate_common_run(
                run, case_name, allow_duplicate_terminal_update=True
            )
            if run.get("relabeled_groups") != 0:
                raise ValueError(f"scale-zero run applied hindsight in {case_name}")
            if any(
                group.get("update_source") == "hindsight_relabel"
                for group in run["group_diagnostics"]
            ):
                raise ValueError(f"scale-zero group counted hindsight in {case_name}")
            _validate_relabel_accounting(run, case_name, 0.0)
            reached = run.get("reached_optimizer_update_budget") is True
            censored = run.get("transition_cap_censored") is True
            if reached:
                if run["optimizer_updates"] != STAGE_A_TARGET_UPDATES or censored:
                    raise ValueError(f"inconsistent reached-target flags in {case_name}")
            else:
                if not censored or run["optimizer_updates"] >= STAGE_A_TARGET_UPDATES:
                    raise ValueError(f"inconsistent cap-censoring flags in {case_name}")
                if run["transitions"] < TRANSITION_CAP:
                    raise ValueError(f"censored run stopped before transition cap in {case_name}")
            all_reached_400 &= reached
            every_run_at_least_250 &= run["optimizer_updates"] >= ALLOWED_FALLBACK_UPDATES
        validated_runs[case_name] = runs

    if all_reached_400:
        selected_budget = STAGE_A_TARGET_UPDATES
    elif every_run_at_least_250:
        selected_budget = ALLOWED_FALLBACK_UPDATES
    else:
        selected_budget = None

    per_cell = {}
    every_run_preview_gate = selected_budget is not None and shadow["passed"] is True
    every_cell_regimes = selected_budget is not None
    every_run_teacher_tv = selected_budget is not None
    prefix_wall_seconds: list[float] = []
    if selected_budget is not None:
        for case_name, runs in validated_runs.items():
            per_run = []
            cell_regimes: set[str] = set()
            for run in runs:
                prefix, final_group = _selected_prefix(run, selected_budget, case_name)
                if prefix[-1]["optimizer_updates_after_group"] != selected_budget:
                    raise ValueError(f"selected prefix does not end at U* in {case_name}")
                prefix_eval_updates = [
                    int(update)
                    for update, transition in zip(
                        run["x_optimizer_updates"], run["x_transitions"]
                    )
                    if int(transition) <= int(prefix[-1]["transition_end"])
                    and int(update) <= selected_budget
                ]
                expected_prefix_grid = list(
                    range(0, selected_budget + 1, EVAL_INTERVAL_UPDATES)
                )
                if prefix_eval_updates != expected_prefix_grid:
                    raise ValueError(
                        f"selected prefix lacks exact evaluation grid in {case_name}"
                    )
                prefix_regimes = {group["regime"] for group in prefix}
                cell_regimes |= prefix_regimes
                requested_updates = sum(
                    group["update_source"] == "requested_live" for group in prefix
                )
                hindsight_updates = sum(
                    group["update_source"] == "hindsight_relabel" for group in prefix
                )
                prefix_previews = [
                    preview
                    for preview in run["auxiliary_gradient_diagnostics"]
                    if preview["after_group"] <= final_group
                ]
                prefix_candidate_groups = [
                    group
                    for group in run["eligible_relabel_candidate_groups"]
                    if group <= final_group
                ]
                qualifying_previews = [
                    preview
                    for preview in prefix_previews
                    if float(preview.get("gradient_norm", 0.0)) > 0.0
                    and float(preview.get("hypothetical_update_norm", 0.0)) > 0.0
                    and preview.get("applied") is False
                    and preview.get("mutated") is False
                    and preview.get("frozen_group_parameters") is True
                ]
                preview_gate = (
                    len(prefix_previews) >= 10
                    and prefix_candidate_groups
                    == [preview["after_group"] for preview in prefix_previews]
                    and len(qualifying_previews) == len(prefix_previews)
                    and requested_updates == selected_budget
                    and hindsight_updates == 0
                )
                every_run_preview_gate &= preview_gate
                post_warmup = [
                    group
                    for group in prefix
                    if group["transition_start"] >= WARMUP_TRANSITIONS
                ]
                mean_tv = (
                    float(
                        np.mean(
                            np.asarray(
                                [group["teacher_tv_from_uniform"] for group in post_warmup],
                                dtype=np.float64,
                            )
                        )
                    )
                    if post_warmup
                    else 0.0
                )
                tv_pass = bool(post_warmup) and mean_tv > 0.05
                every_run_teacher_tv &= tv_pass
                wall = _prefix_wall_seconds(run, selected_budget, case_name)
                prefix_wall_seconds.append(wall)
                per_run.append(
                    {
                        "seed": run["seed"],
                        "selected_prefix_final_group": final_group,
                        "selected_prefix_transition_end": prefix[-1]["transition_end"],
                        "selected_prefix_optimizer_updates": selected_budget,
                        "requested_live_updates": requested_updates,
                        "hindsight_updates": hindsight_updates,
                        "relabel_candidate_count": len(prefix_previews),
                        "positive_nonmutating_preview_count": len(qualifying_previews),
                        "preview_gate_pass": preview_gate,
                        "observed_regimes": sorted(prefix_regimes),
                        "post_warmup_group_count": len(post_warmup),
                        "post_warmup_mean_teacher_tv_from_uniform": mean_tv,
                        "teacher_tv_pass": tv_pass,
                        "wall_seconds_through_selected_prefix": wall,
                    }
                )
            regimes_pass = cell_regimes == {"dead", "mixed", "all_pass"}
            every_cell_regimes &= regimes_pass
            per_cell[case_name] = {
                "n_runs": len(runs),
                "selected_prefix_update_budget": selected_budget,
                "per_run": per_run,
                "selected_prefix_observed_regimes": sorted(cell_regimes),
                "all_three_regimes_pass": regimes_pass,
            }

    slowest_prefix_seconds = max(prefix_wall_seconds) if prefix_wall_seconds else None
    projected_hours = (
        90.0 * slowest_prefix_seconds / 3600.0
        if slowest_prefix_seconds is not None
        else None
    )
    gates = {
        "source_runtime_schedule_and_run_invariants": True,
        "all_runs_reach_400_or_every_run_reaches_250": selected_budget is not None,
        "every_selected_prefix_ends_at_selected_update_budget": selected_budget is not None,
        "every_run_has_ten_one_to_one_positive_nonmutating_previews": every_run_preview_gate,
        "every_cell_observes_all_regimes": every_cell_regimes,
        "every_run_post_warmup_teacher_tv_above_0p05": every_run_teacher_tv,
        "projected_90_run_serial_runtime_at_most_24h": (
            projected_hours is not None and projected_hours <= 24.0
        ),
    }
    gates["all_pass"] = all(gates.values())
    saved_gates = _validate_saved_stage_a_gates(
        artifact, gates, selected_budget
    )
    return {
        "schema": "curriculum-maxrl/acrobot-hindsight-v4a-verification/v1",
        "v4_stage": "stage_a_feasibility",
        "all_checks_passed": True,
        "source_lock": source_lock,
        "protocol_checks_passed": True,
        "analysis_contract": (
            "effect-blind V4 Stage A gates; no learning outcome enters the launch decision"
        ),
        "per_cell": per_cell,
        "runtime": {
            "slowest_run_wall_seconds_through_selected_prefix": slowest_prefix_seconds,
            "projected_90_run_factorial_serial_hours": projected_hours,
            "limit_hours": 24.0,
        },
        "selected_optimizer_update_budget": selected_budget,
        "gates": gates,
        "runner_saved_gates": saved_gates,
        "saved_runner_gates_verified": True,
        "stage_b_factorial_authorized": gates["all_pass"],
    }


def _exact_sign_flip_p(values: np.ndarray) -> float:
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError("the V4 Stage B sign-flip test requires ten finite pairs")
    observed = abs(float(values.mean()))
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=10):
        statistic = abs(float(np.dot(signs, values) / 10.0))
        extreme += statistic >= observed - 1e-15
    return float(extreme / (2**10))


def _bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, 10, size=(20_000, 10))].mean(axis=1)
    return [float(value) for value in np.quantile(draws, (0.025, 0.975))]


def _holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    ordered = sorted((float(value), name) for name, value in p_values.items())
    running = 0.0
    still_rejecting = True
    result = {}
    for rank, (p_value, name) in enumerate(ordered, start=1):
        multiplier = len(ordered) - rank + 1
        running = max(running, multiplier * p_value)
        reject = still_rejecting and p_value <= alpha / multiplier
        if not reject:
            still_rejecting = False
        result[name] = {
            "raw_p": p_value,
            "holm_adjusted_p": min(running, 1.0),
            "reject_familywise_0.05": bool(reject),
        }
    return result


def _selected_stage_b_budget(protocol: dict, lock: dict) -> int:
    protocol_schedule = protocol.get("registered_schedule")
    lock_schedule = lock.get("registered_schedule")
    protocol_budget = (
        protocol_schedule.get("optimizer_update_target")
        if isinstance(protocol_schedule, dict)
        else None
    )
    lock_budget = (
        lock_schedule.get("optimizer_update_target")
        if isinstance(lock_schedule, dict)
        else None
    )
    if protocol_budget not in (250, 400) or lock_budget != protocol_budget:
        raise ValueError("V4B selected optimizer budget is absent, invalid, or unlocked")
    return int(protocol_budget)


def _stage_b_protocol(artifact: dict, lock: dict) -> int:
    protocol = artifact["protocol"]
    selected_budget = _selected_stage_b_budget(protocol, lock)
    expected = _expected_protocol(
        "stage_b_factorial",
        _expected_schedule("stage_b_factorial", selected_budget),
    )
    if protocol != expected:
        raise ValueError("V4 Stage B protocol does not exactly match the frozen contract")
    return selected_budget


def _contrast_values(
    aucs: dict[str, np.ndarray], coefficients: dict[str, float]
) -> np.ndarray:
    values = np.zeros(10, dtype=np.float64)
    for case_name, coefficient in coefficients.items():
        values += coefficient * aucs[case_name]
    return values


def _validate_saved_stage_b(artifact: dict, analyses: dict, corrections: dict) -> bool:
    saved = artifact.get("paired_scale_contrasts")
    if saved is None:
        raise ValueError("artifact lacks the required saved V4 paired analysis")
    if tuple(saved) != tuple(CONTRAST_SPECS):
        raise ValueError("saved V4 contrast family differs from the frozen family")
    for name in CONTRAST_SPECS:
        actual = saved[name]
        expected = analyses[name]
        expected_keys = {
            "description",
            "coefficients",
            "metric",
            "n_pairs",
            "mean_contrast",
            "sample_std",
            "mean_ci95_paired_seed_bootstrap",
            "exact_paired_sign_flip_p_two_sided",
            "per_seed_contrast",
            "raw_p",
            "holm_adjusted_p",
            "reject_familywise_0.05",
        }
        if set(actual) != expected_keys:
            raise ValueError(f"saved contrast field set mismatch for {name}")
        if actual.get("description") != CONTRAST_DESCRIPTIONS[name]:
            raise ValueError(f"saved contrast description mismatch for {name}")
        if actual.get("metric") != PRIMARY_METRIC:
            raise ValueError(f"saved metric mismatch for {name}")
        if actual.get("n_pairs") != 10:
            raise ValueError(f"saved pair count mismatch for {name}")
        if actual.get("coefficients") != CONTRAST_SPECS[name]:
            raise ValueError(f"saved coefficients mismatch for {name}")
        if not np.allclose(
            actual.get("per_seed_contrast"),
            expected["per_seed_contrast"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"saved paired differences do not reproduce for {name}")
        for key in (
            "mean_contrast",
            "sample_std",
            "exact_paired_sign_flip_p_two_sided",
        ):
            _assert_close(actual.get(key), expected[key], f"{name}.{key}")
        if not np.allclose(
            actual.get("mean_ci95_paired_seed_bootstrap"),
            expected["mean_ci95_paired_seed_bootstrap"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"saved bootstrap interval does not reproduce for {name}")
        for key, expected_value in corrections[name].items():
            actual_value = actual.get(key)
            if isinstance(expected_value, bool):
                if actual_value is not expected_value:
                    raise ValueError(f"saved Holm decision mismatch for {name}")
            else:
                _assert_close(actual_value, expected_value, f"{name}.{key}")

    multiplicity = artifact.get("scale_multiplicity")
    expected_multiplicity = {
        "family": list(CONTRAST_SPECS),
        "metric": PRIMARY_METRIC,
        "method": "Holm step-down",
        "familywise_alpha": 0.05,
        "test": "exact two-sided paired sign-flip randomization",
        "sign_exchangeability_assumption": (
            "independent seed-level contrasts have sign-exchangeable null distributions"
        ),
    }
    if multiplicity != expected_multiplicity:
        raise ValueError("saved V4 multiplicity declaration mismatch")
    return True


def _read_linked_json(path_value, expected_hash, label: str) -> tuple[Path, dict]:
    if not isinstance(path_value, str) or not isinstance(expected_hash, str):
        raise ValueError(f"V4B artifact lacks its {label} path/hash")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file() or _sha256(path) != expected_hash:
        raise ValueError(f"linked {label} is missing or changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"linked {label} is not a JSON object")
    return path, value


def _validate_stage_b_authorization(
    artifact: dict, lock: dict, selected_budget: int
) -> dict:
    required_lock_hashes = (
        "amendment_sha256",
        "stage_a_artifact_sha256",
        "stage_a_independent_verification_sha256",
        "stage_a_gates_sha256",
    )
    for key in required_lock_hashes:
        value = lock.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"V4B lock lacks a valid {key}")
    linkage = {
        "stage_a_artifact_sha256": artifact.get("stage_a_artifact_sha256"),
        "amendment_sha256": artifact.get("stage_b_amendment_sha256"),
        "stage_a_independent_verification_sha256": artifact.get(
            "stage_a_independent_verification_sha256"
        ),
    }
    if any(lock.get(key) != value for key, value in linkage.items()):
        raise ValueError("V4B artifact authorization hashes differ from its lock")

    stage_a_path, stage_a = _read_linked_json(
        artifact.get("stage_a_artifact"),
        linkage["stage_a_artifact_sha256"],
        "Stage-A artifact",
    )
    amendment_path, amendment = _read_linked_json(
        artifact.get("stage_b_amendment"),
        linkage["amendment_sha256"],
        "Stage-B amendment",
    )
    verification_path, verification = _read_linked_json(
        artifact.get("stage_a_independent_verification"),
        linkage["stage_a_independent_verification_sha256"],
        "independent Stage-A verification",
    )
    if (
        amendment.get("schema")
        != "curriculum-maxrl/acrobot-hindsight-v4b-amendment/v1"
        or amendment.get("v4_stage") != "stage_b_factorial"
        or amendment.get("explicit_stage_b_authorization") is not True
        or amendment.get("stage_a_all_effect_blind_gates_passed") is not True
        or amendment.get("selected_update_budget") != selected_budget
        or amendment.get("registered_schedule")
        != _expected_schedule("stage_b_factorial", selected_budget)
    ):
        raise ValueError("linked Stage-B amendment is not the exact authorization")
    amendment_links = {
        "stage_a_artifact_sha256": linkage["stage_a_artifact_sha256"],
        "stage_a_independent_verification_sha256": linkage[
            "stage_a_independent_verification_sha256"
        ],
        "stage_a_gates_sha256": lock["stage_a_gates_sha256"],
    }
    if any(amendment.get(key) != value for key, value in amendment_links.items()):
        raise ValueError("Stage-B amendment hashes differ from the V4B lock")
    if Path(amendment.get("stage_a_artifact", "")).resolve() != stage_a_path:
        raise ValueError("Stage-B amendment names a different Stage-A artifact")
    if Path(
        amendment.get("stage_a_independent_verification", "")
    ).resolve() != verification_path:
        raise ValueError("Stage-B amendment names a different independent report")

    saved_stage_a_gates = stage_a.get("stage_a_effect_blind_gates")
    if (
        stage_a.get("schema") != SCHEMA_A
        or stage_a.get("artifact_state") != "complete"
        or not isinstance(saved_stage_a_gates, dict)
        or saved_stage_a_gates.get("all_pass") is not True
        or saved_stage_a_gates.get("stage_b_authorized") is not True
        or saved_stage_a_gates.get("selected_update_budget") != selected_budget
        or _canonical_hash(saved_stage_a_gates) != lock["stage_a_gates_sha256"]
    ):
        raise ValueError("linked Stage-A artifact no longer authorizes V4B")
    if (
        verification.get("schema")
        != "curriculum-maxrl/acrobot-hindsight-v4a-verification/v1"
        or verification.get("v4_stage") != "stage_a_feasibility"
        or verification.get("all_checks_passed") is not True
        or verification.get("saved_runner_gates_verified") is not True
        or verification.get("runner_saved_gates") != saved_stage_a_gates
        or verification.get("stage_b_factorial_authorized") is not True
        or verification.get("selected_optimizer_update_budget") != selected_budget
        or verification.get("artifact_sha256") != linkage["stage_a_artifact_sha256"]
        or verification.get("lock_sha256")
        != stage_a.get("provenance", {}).get("source_lock_sha256")
    ):
        raise ValueError("linked independent Stage-A report does not authorize V4B")
    if Path(verification.get("artifact", "")).resolve() != stage_a_path:
        raise ValueError("independent Stage-A report names a different artifact")
    return {
        "passed": True,
        "stage_a_artifact": str(stage_a_path),
        "stage_b_amendment": str(amendment_path),
        "stage_a_independent_verification": str(verification_path),
    }


def _verify_stage_b(artifact: dict, lock: dict, source_lock: dict) -> dict:
    _verify_artifact_shell(artifact, STAGE_B_CASES, SCHEMA_B)
    selected_budget = _stage_b_protocol(artifact, lock)
    authorization = _validate_stage_b_authorization(
        artifact, lock, selected_budget
    )
    aucs: dict[str, np.ndarray] = {}
    per_cell = {}
    expected_case_summaries = {}
    for multiplier in LR_MULTIPLIERS:
        for scale in HINDSIGHT_SCALES:
            case_name = _case_name(multiplier, scale)
            record = artifact["cases"][case_name]
            _validate_config(record["config"], case_name, multiplier, scale)
            runs = record.get("runs", [])
            if [run.get("seed") for run in runs] != STAGE_B_SEEDS:
                raise ValueError(f"seed order mismatch for {case_name}")
            cell_aucs = []
            cell_step_norms = {}
            cell_transitions = {}
            for run in runs:
                _validate_common_run(run, case_name)
                _validate_relabel_accounting(run, case_name, scale)
                if (
                    run.get("optimizer_updates") != selected_budget
                    or run.get("reached_optimizer_update_budget") is not True
                    or run.get("transition_cap_censored") is not False
                ):
                    raise ValueError(f"incomplete or censored primary pair in {case_name}")
                auc = _normalized_update_auc(run, case_name, selected_budget)
                _assert_close(auc, run.get(PRIMARY_METRIC), f"run AUC {case_name}")
                cell_aucs.append(auc)
                cell_step_norms[str(run["seed"])] = _step_norm_diagnostics(
                    run, case_name
                )
                cell_transitions[str(run["seed"])] = int(run["transitions"])
            aucs[case_name] = np.asarray(cell_aucs, dtype=np.float64)
            expected_case_summaries[case_name] = {
                "metric": PRIMARY_METRIC,
                "n_seeds": len(cell_aucs),
                "mean": float(np.mean(cell_aucs)),
                "sample_std": float(np.std(cell_aucs, ddof=1)),
                "per_seed": list(cell_aucs),
                "source_step_norms_per_seed": cell_step_norms,
                "transitions_to_target_per_seed": cell_transitions,
            }
            per_cell[case_name] = {
                "n_runs": len(runs),
                "all_complete": True,
                "mean_recomputed_update_auc": float(np.mean(cell_aucs)),
                "descriptive_applied_step_norms_by_seed": cell_step_norms,
            }

    _assert_nested_equal(
        artifact.get("stage_b_case_summaries"),
        expected_case_summaries,
        "saved Stage-B case summaries",
    )

    analyses = {}
    raw_p_values = {}
    for index, (name, coefficients) in enumerate(CONTRAST_SPECS.items()):
        values = _contrast_values(aucs, coefficients)
        p_value = _exact_sign_flip_p(values)
        analyses[name] = {
            "description": CONTRAST_DESCRIPTIONS[name],
            "metric": PRIMARY_METRIC,
            "coefficients": coefficients,
            "n_pairs": 10,
            "per_seed_contrast": values.tolist(),
            "mean_contrast": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "mean_ci95_paired_seed_bootstrap": _bootstrap_ci(
                values, 45_000 + index
            ),
            "exact_paired_sign_flip_p_two_sided": p_value,
        }
        raw_p_values[name] = p_value
    corrections = _holm(raw_p_values)
    for name, correction in corrections.items():
        analyses[name].update(correction)
    saved_verified = _validate_saved_stage_b(artifact, analyses, corrections)

    decisions = {
        "scale1_mean_at_least_0p03_and_holm_significant": bool(
            analyses["C1"]["mean_contrast"] >= 0.03
            and analyses["C1"]["reject_familywise_0.05"]
        ),
        "scale2_increment_mean_at_least_0p03_and_holm_significant": bool(
            analyses["C2"]["mean_contrast"] >= 0.03
            and analyses["C2"]["reject_familywise_0.05"]
        ),
        "c3_material_and_holm_significant": bool(
            abs(analyses["C3"]["mean_contrast"]) >= 0.03
            and analyses["C3"]["reject_familywise_0.05"]
        ),
        "c4_material_and_holm_significant": bool(
            abs(analyses["C4"]["mean_contrast"])
            >= 0.03
            and analyses["C4"]["reject_familywise_0.05"]
        ),
    }
    expected_saved_decision = {
        "C1_directional_local_improvement_supported": decisions[
            "scale1_mean_at_least_0p03_and_holm_significant"
        ],
        "C2_directional_increment_supported": decisions[
            "scale2_increment_mean_at_least_0p03_and_holm_significant"
        ],
        "C3_material_restricted_separability_departure": decisions[
            "c3_material_and_holm_significant"
        ],
        "C4_material_restricted_separability_departure": decisions[
            "c4_material_and_holm_significant"
        ],
        "directional_minimum_mean": 0.03,
        "restricted_departure_minimum_absolute_mean": 0.03,
        "interpretation_boundary": (
            "C3/C4 diagnose departure from Y(a,s)=F(a)+G(a*s); they do not "
            "identify semantic data value separately from optimizer scale, update "
            "source composition, relabel frequency, or policy trajectory."
        ),
    }
    _assert_nested_equal(
        artifact.get("predeclared_scale_decision"),
        expected_saved_decision,
        "saved predeclared Stage-B decision",
    )
    _assert_nested_equal(
        artifact.get("analysis_status"),
        {
            "performed": True,
            "all_ten_seed_pairs_complete": True,
            "all_final_update_coordinates_equal_selected_budget": True,
            "selected_update_budget": selected_budget,
        },
        "saved Stage-B analysis status",
    )
    return {
        "schema": "curriculum-maxrl/acrobot-hindsight-v4b-verification/v1",
        "v4_stage": "stage_b_factorial",
        "all_checks_passed": True,
        "source_lock": source_lock,
        "protocol_checks_passed": True,
        "stage_b_authorization_chain": authorization,
        "selected_optimizer_update_budget": selected_budget,
        "per_cell": per_cell,
        "primary_metric": PRIMARY_METRIC,
        "paired_contrasts": analyses,
        "holm_familywise_alpha": 0.05,
        "saved_analysis_verified": saved_verified,
        "decisions": decisions,
    }


def verify(
    artifact: dict, lock: dict, *, lock_sha256: str | None = None
) -> dict:
    """Verify one complete V4A or V4B artifact without modifying it."""

    protocol = artifact.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("artifact lacks a protocol object")
    stage = _protocol_stage(protocol)
    source_lock = _verify_lock(
        artifact, lock, stage, lock_sha256=lock_sha256
    )
    if stage == "stage_a_feasibility":
        return _verify_stage_a(artifact, source_lock)
    return _verify_stage_b(artifact, lock, source_lock)


def _write_exclusive(path: Path, payload: dict, overwrite: bool) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    report = verify(artifact, lock, lock_sha256=_sha256(args.lock))
    report["artifact"] = str(args.artifact.resolve())
    report["artifact_sha256"] = _sha256(args.artifact)
    report["lock"] = str(args.lock.resolve())
    report["lock_sha256"] = _sha256(args.lock)
    _write_exclusive(args.output, report, args.overwrite)
    if report["v4_stage"] == "stage_a_feasibility":
        print(
            "V4A verification complete: "
            f"authorized={report['stage_b_factorial_authorized']}, "
            f"selected_updates={report['selected_optimizer_update_budget']}"
        )
    else:
        print("V4B verification passed: exact paired/Holm family reproduced")


if __name__ == "__main__":
    main()
