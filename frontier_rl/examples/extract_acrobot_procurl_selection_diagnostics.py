"""Extract frozen-protocol descriptive diagnostics from completed Acrobot raw.

This unlocked post-run utility first delegates all scientific-ledger validation
to the locked analyzer.  It then reports only protocol-required per-seed values
and ordinary arm-level descriptive summaries.  It adds no hypothesis test,
interval, multiplicity decision, threshold, or causal interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from statistics import fmean, stdev
from types import ModuleType
from typing import Any


DIAGNOSTICS_SCHEMA = (
    "curriculum-maxrl/acrobot-procurl-selection-descriptive-diagnostics/v1"
)
VERIFICATION_SCHEMA = (
    "curriculum-maxrl/acrobot-procurl-selection-descriptive-diagnostics-verification/v1"
)
STATUS = "descriptive_only_no_new_inference"

PER_SEED_SCALAR_FIELDS = (
    "auc_target_uniform_mean_success_full_atomic_paid",
    "auc_target_uniform_mean_success_by_student_transitions",
    "final_target_uniform_mean_success",
    "final_native_success_rate",
    "student_fraction_of_paid",
    "probe_fraction_of_paid",
    "probe_sweeps",
    "optimizer_updates",
    "mean_selection_entropy",
    "mean_selection_tv_from_uniform",
    "mean_max_task_probability",
    "realized_task_tv_from_uniform",
    "paid_budget_overshoot",
)
PER_SEED_VECTOR_FIELDS = (
    "mean_assigned_probability_per_task",
    "realized_task_fraction",
)
UNIT_INTERVAL_SCALAR_FIELDS = {
    "auc_target_uniform_mean_success_full_atomic_paid",
    "auc_target_uniform_mean_success_by_student_transitions",
    "final_target_uniform_mean_success",
    "final_native_success_rate",
    "student_fraction_of_paid",
    "probe_fraction_of_paid",
    "mean_selection_tv_from_uniform",
    "mean_max_task_probability",
    "realized_task_tv_from_uniform",
}
NONNEGATIVE_INTEGER_FIELDS = {
    "probe_sweeps",
    "optimizer_updates",
    "paid_budget_overshoot",
}
PER_SEED_KEYS = {"seed", *PER_SEED_SCALAR_FIELDS, *PER_SEED_VECTOR_FIELDS}
SUMMARY_KEYS = {"scalar_metrics", "per_task_metrics"}
SCALAR_SUMMARY_KEYS = {"mean", "sample_std"}
VECTOR_SUMMARY_KEYS = {"mean_per_task", "sample_std_per_task"}
RAW_BINDING_KEYS = {"logical_path", "size_bytes", "sha256", "schema"}
SCHEDULE_KEYS = {"arms", "seeds", "run_count"}
REPORT_KEYS = {
    "schema",
    "mode",
    "status",
    "raw_artifact",
    "source_lock",
    "development_gate",
    "schedule",
    "metric_policy",
    "arms",
}
METRIC_POLICY = {
    "protocol_source": "ACROBOT_PROCURL_SELECTION_PROTOCOL.md",
    "aggregation": "per-seed values and arm-level mean/sample standard deviation",
    "overshoot_auc": "complete atomic paid-transition endpoint retained",
    "student_auc": "full curve indexed by student transitions",
    "selection_values": "copied from strictly revalidated run diagnostics",
    "new_inferential_statistics": False,
}

RAW_SCHEMA = "curriculum-maxrl/acrobot-procurl-selection-raw/v1"
LOCK_SCHEMA = "curriculum-maxrl/acrobot-procurl-selection-lock/v1"
GATE_SCHEMA = "curriculum-maxrl/acrobot-procurl-selection-development-gates/v1"
ARM_NAMES = (
    "procurl_env_b20_f5120",
    "probe_sham_uniform_f5120",
    "ordinary_uniform",
    "u16_probe_range_matched_f5120",
)
CONFIRMATORY_SEEDS = tuple(range(21_000, 21_080))
PINNED_REANALYSIS_RUNTIME = {
    "python_implementation": "CPython",
    "python": "3.12.13",
    "numpy": "2.5.1",
    "gymnasium": "1.3.0",
}
RUNTIME_KEYS = {
    "python_implementation",
    "python",
    "platform",
    "machine",
    "numpy",
    "gymnasium",
}
EXPECTED_LOCK_SCHEDULE = {
    "arm_names": list(ARM_NAMES),
    "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
    "development_seeds": [21_300, 21_301, 21_302],
    "quick_seeds": [21_400],
    "confirmatory_paid_budget": 2_000_000,
    "development_paid_budget": 400_000,
    "quick_paid_budget": 100_000,
    "regular_eval_interval_paid": 100_000,
    "confirmatory_eval_n": 32,
    "development_eval_n": 32,
    "quick_eval_n": 2,
    "n_rollouts": 16,
    "learning_rate": 3e-4,
    "probes_per_task": 20,
    "refresh_student_transitions": 5_120,
    "procurl_beta": 20.0,
    "u16_beta_continuous_range_matched": 6.416133525771289,
    "u16_lattice_max_logit": 4.97730861318145,
    "engine_master_base": 50_000_000_000,
    "engine_master_stride": 10_000_000,
    "rng_domain_offsets": {
        "actor_parameter": 0,
        "actor_action": 1,
        "selection": 10_000,
        "environment_reset_rng": 11_003,
        "evaluation_episode": 1_000_000,
        "evaluation_action": 1_000_001,
        "probe_episode_reset": 2_000_000,
        "probe_episode_action": 3_000_000,
    },
    "environment_adapter_seed_offset": 1_000,
    "upstream_procurl_commit": "17904f1d7b9b29e089d4f70ae7aadf1da50ba6b2",
}
EXPECTED_SOURCE_RELATIVE_PATHS = (
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
GATE_KEYS = {
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
}
RAW_GATE_BINDING_KEYS = {
    "relative_path",
    "sha256",
    "raw_artifact_relative_path",
    "raw_artifact_sha256",
    "all_gates_passed",
}
SOURCE_VERIFICATION_KEYS = {
    "passed",
    "runtime",
    "source_lock_sha256",
    "checked_source_files",
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
SELECTION_DIAGNOSTIC_KEYS = {
    "mean_selection_entropy",
    "mean_selection_tv_from_uniform",
    "mean_max_task_probability",
    "mean_assigned_probability_per_task",
    "realized_task_fraction",
    "realized_task_tv_from_uniform",
    "student_fraction_of_paid",
    "probe_fraction_of_paid",
    "paid_budget_overshoot",
}
ENTROPY_SUM_SENTINEL_TERMS = (
    549.289282936898, 549.2857356245377, 549.2872495862491,
    549.2861638667421, 549.2857443317533, 549.2867282464132,
    549.2895726406483, 549.2865534076619, 549.2860469113732,
    549.288622325428, 549.2862002480819, 549.2873496418205,
    549.2863843248118, 549.2858695926706, 549.2856971184226,
    549.2867986978171, 549.2863129624436, 549.2865341586298,
    549.2861813786084, 549.2860759557776, 549.2857224574934,
    549.2861795828343, 549.2859069840187, 549.2864419313281,
    549.2864596960767, 549.2865119613786, 549.28707379166,
    549.2865645008778, 549.2859656700225, 549.2861366690263,
    549.2856495032845, 549.2865620090922,
)
EXPECTED_ENTROPY_SUM_SENTINEL_HEX = "0x1.12a4ae5d8b0d5p+14"
LOCKED_ANALYZER_LOGICAL_PATH = (
    "frontier_rl/examples/analyze_acrobot_procurl_selection.py"
)
CANONICAL_LOCK_LOGICAL_PATH = (
    "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_LOCK.json"
)
DEFAULT_ANALYZER_PATH = Path(__file__).resolve().parent / (
    "analyze_acrobot_procurl_selection.py"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _typed_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return list(left) == list(right) and all(
            _typed_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _typed_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class _CapturedJson:
    """One immutable byte capture and the strict JSON object parsed from it."""

    path: Path
    data: bytes
    payload: dict[str, Any]
    size_bytes: int
    sha256: str


def _parse_strict_json(data: bytes, label: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"cannot read {label} as strict JSON") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"cannot read {label} as strict JSON") from error
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
    _require(_all_finite(value), f"{label} contains a non-finite number")
    assert isinstance(value, dict)
    return value


def _capture_json(path: Path, label: str) -> _CapturedJson:
    path = _regular_file(path, label)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {label} as strict JSON") from error
    payload = _parse_strict_json(data, label)
    return _CapturedJson(
        path=path,
        data=data,
        payload=payload,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def load_strict_json(path: Path, label: str) -> dict[str, Any]:
    """Compatibility entry point; production flows carry `_CapturedJson`."""
    return _capture_json(path, label).payload


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _portable_logical_path(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be a string")
    assert isinstance(value, str)
    _require("\\" not in value and "\x00" not in value, f"{label} is not portable")
    _require(
        not any(ord(character) < 32 or ord(character) == 127 for character in value),
        f"{label} contains a control character",
    )
    _require(
        not any(character in '<>"|?*' for character in value),
        f"{label} contains a Windows-invalid filename character",
    )
    _require(":" not in value and "//" not in value, f"{label} is not portable")
    _require(not value.startswith(("/", "~")), f"{label} is not relative")
    parts = value.split("/")
    _require(
        all(part not in {"", ".", ".."} for part in parts),
        f"{label} is not normalized",
    )
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}"
        for prefix in ("COM", "LPT")
        for number in range(1, 10)
    }
    for part in parts:
        _require(not part.endswith((".", " ")), f"{label} has a trailing dot/space")
        _require(
            part.split(".", 1)[0].rstrip(" .").upper() not in reserved,
            f"{label} contains a reserved device name",
        )
    path = PurePosixPath(value)
    _require(path.as_posix() == value and not path.is_absolute(), f"{label} is invalid")
    return value


def _regular_file(path: Path, label: str) -> Path:
    path = path.absolute()
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    _require(path.is_file(), f"{label} is missing or not a regular file")
    return path


def _is_utc_iso8601(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(None)


def _live_reanalysis_runtime() -> dict[str, str]:
    try:
        numpy_version = importlib.metadata.version("numpy")
        gymnasium_version = importlib.metadata.version("gymnasium")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("exact pinned NumPy and Gymnasium are required") from error
    return {
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "numpy": numpy_version,
        "gymnasium": gymnasium_version,
    }


def _verify_source_lock_before_import(
    lock: dict[str, Any], analyzer_path: Path
) -> str:
    _require(set(lock) == LOCK_KEYS, "source lock field set mismatch")
    _require(lock.get("schema") == LOCK_SCHEMA, "source lock schema mismatch")
    _require(
        lock.get("status") == "sealed_before_any_quick_development_or_confirmation",
        "source lock status mismatch",
    )
    _require(
        lock.get("purpose")
        == (
            "Canonical pre-execution source/runtime lock for the Acrobot "
            "ProCuRL selection-semantic study."
        ),
        "source lock purpose mismatch",
    )
    _require(_is_utc_iso8601(lock.get("created_utc")), "source lock timestamp invalid")
    _require(
        _typed_equal(lock.get("schedule"), EXPECTED_LOCK_SCHEDULE),
        "source lock schedule mismatch",
    )
    runtime = lock.get("runtime")
    _require(
        isinstance(runtime, dict) and set(runtime) == RUNTIME_KEYS,
        "source lock runtime field set mismatch",
    )
    assert isinstance(runtime, dict)
    recorded = {key: runtime.get(key) for key in PINNED_REANALYSIS_RUNTIME}
    _require(
        _typed_equal(recorded, PINNED_REANALYSIS_RUNTIME),
        "recorded runtime is not exact",
    )
    _require(
        _typed_equal(_live_reanalysis_runtime(), PINNED_REANALYSIS_RUNTIME),
        "live runtime is not exact",
    )
    _require(
        sum(ENTROPY_SUM_SENTINEL_TERMS).hex()
        == EXPECTED_ENTROPY_SUM_SENTINEL_HEX,
        "compensated-sum sentinel mismatch",
    )
    audit = lock.get("seed_collision_audit")
    _require(isinstance(audit, dict) and audit.get("passed") is True, "seed audit failed")
    sources = lock.get("source_sha256")
    _require(isinstance(sources, dict), "source lock manifest missing")
    assert isinstance(sources, dict)
    _require(
        tuple(sources) == EXPECTED_SOURCE_RELATIVE_PATHS,
        "source lock manifest key/order mismatch",
    )
    _require(all(_valid_sha256(value) for value in sources.values()), "source SHA invalid")
    expected = sources[LOCKED_ANALYZER_LOGICAL_PATH]
    analyzer_path = _regular_file(analyzer_path, "locked analyzer")
    _require(_sha256(analyzer_path) == expected, "analyzer does not match source lock")
    return expected


def _load_verified_analyzer(path: Path, expected_sha256: str) -> ModuleType:
    path = _regular_file(path, "locked analyzer")
    source = path.read_bytes()
    _require(
        hashlib.sha256(source).hexdigest() == expected_sha256,
        "analyzer source buffer hash mismatch before execution",
    )
    code = compile(source, str(path), "exec", dont_inherit=True)
    module = ModuleType(f"_locked_acrobot_diagnostics_{expected_sha256[:16]}")
    module.__file__ = str(path)
    module.__package__ = "frontier_rl.examples"
    module.__loader__ = None
    module.__spec__ = None
    module.__cached__ = None
    exec(code, module.__dict__)
    return module


def _all_finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _strict_confirmatory(
    raw: dict[str, Any], analyzer: ModuleType
) -> dict[str, Any]:
    validated = analyzer.validate_raw_artifact(raw)
    _require(validated.get("strict_valid") is True, "locked raw validation failed")
    _require(validated.get("mode") == "confirmatory", "raw is not confirmatory")
    _require(
        _typed_equal(validated.get("seeds"), list(CONFIRMATORY_SEEDS)),
        "validated seed schedule mismatch",
    )
    _require(
        tuple(validated.get("by_case", {})) == ARM_NAMES,
        "validated arm schedule mismatch",
    )
    return validated


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_raw_lock_and_gate_before_import(
    raw_capture: _CapturedJson,
    lock_capture: _CapturedJson,
    gate_capture: _CapturedJson,
) -> str:
    raw = raw_capture.payload
    lock = lock_capture.payload
    lock_sha = lock_capture.sha256
    provenance = raw.get("provenance")
    _require(isinstance(provenance, dict), "raw provenance is missing")
    assert isinstance(provenance, dict)
    _require(
        provenance.get("source_lock_relative_path") == CANONICAL_LOCK_LOGICAL_PATH,
        "raw source-lock logical path is not canonical",
    )
    _require(
        provenance.get("source_lock_sha256") == lock_sha,
        "raw source-lock SHA-256 mismatch",
    )
    _require(provenance.get("source_lock_enforced") is True, "raw did not enforce lock")
    _require(
        _typed_equal(provenance.get("runtime"), lock["runtime"]),
        "raw runtime differs from source lock",
    )
    _require(
        _typed_equal(provenance.get("source_sha256"), lock["source_sha256"]),
        "raw source manifest differs from source lock",
    )
    _require(
        _typed_equal(
            provenance.get("seed_collision_audit"), lock["seed_collision_audit"]
        ),
        "raw seed audit differs from source lock",
    )

    lock_resolved = lock_capture.path.resolve()
    lock_parts = PurePosixPath(CANONICAL_LOCK_LOGICAL_PATH).parts
    project_root = lock_resolved.parents[len(lock_parts) - 1]
    _require(
        lock_resolved.relative_to(project_root).as_posix()
        == CANONICAL_LOCK_LOGICAL_PATH,
        "source lock is not at its canonical project-relative path",
    )
    try:
        observed_gate_logical = (
            gate_capture.path.resolve().relative_to(project_root).as_posix()
        )
    except ValueError as error:
        raise ValueError("development gate is outside the source project") from error
    observed_gate_logical = _portable_logical_path(
        observed_gate_logical, "observed development gate logical path"
    )
    gate = gate_capture.payload
    _require(set(gate) == GATE_KEYS, "development gate field set mismatch")
    _require(
        gate.get("schema") == GATE_SCHEMA
        and gate.get("mode") == "development"
        and gate.get("all_gates_passed") is True,
        "development gate is not a passing development artifact",
    )
    _require(
        gate.get("source_lock_sha256") == lock_sha,
        "development gate source-lock mismatch",
    )
    source = gate.get("source_lock_verification")
    _require(
        isinstance(source, dict) and set(source) == SOURCE_VERIFICATION_KEYS,
        "gate source-lock verification field set mismatch",
    )
    assert isinstance(source, dict)
    _require(
        source.get("passed") is True
        and source.get("source_lock_sha256") == lock_sha
        and _typed_equal(source.get("runtime"), lock["runtime"])
        and _typed_equal(
            source.get("checked_source_files"), sorted(lock["source_sha256"])
        ),
        "gate source-lock verification mismatch",
    )
    gates = gate.get("gates")
    _require(
        isinstance(gates, dict)
        and tuple(gates) == DEVELOPMENT_GATE_NAMES
        and all(value is True for value in gates.values()),
        "development gate required checks are missing or false",
    )
    _require(
        _typed_equal(gate.get("gate_policy"), DEVELOPMENT_GATE_POLICY),
        "development gate policy mismatch",
    )
    _require(isinstance(gate.get("diagnostics"), dict), "gate diagnostics invalid")
    development_raw_logical = _portable_logical_path(
        gate.get("raw_artifact_relative_path"), "development raw logical path"
    )
    _require(
        _valid_sha256(gate.get("raw_artifact_sha256")),
        "development raw SHA-256 invalid",
    )

    protocol = raw.get("protocol")
    _require(
        isinstance(protocol, dict) and protocol.get("mode") == "confirmatory",
        "raw protocol is not confirmatory",
    )
    assert isinstance(protocol, dict)
    binding = protocol.get("development_gate")
    _require(
        isinstance(binding, dict) and set(binding) == RAW_GATE_BINDING_KEYS,
        "raw development binding field set mismatch",
    )
    assert isinstance(binding, dict)
    gate_logical = _portable_logical_path(
        binding.get("relative_path"), "development gate logical path"
    )
    _require(
        binding.get("relative_path") == gate_logical == observed_gate_logical
        and binding.get("sha256") == gate_capture.sha256
        and binding.get("raw_artifact_relative_path") == development_raw_logical
        and binding.get("raw_artifact_sha256") == gate["raw_artifact_sha256"]
        and binding.get("all_gates_passed") is True,
        "raw-to-development-gate five-field binding mismatch",
    )
    return gate_logical


def _per_seed(record: dict[str, Any]) -> dict[str, Any]:
    raw = record["raw"]
    derived = record["derived"]
    selection = raw["selection_diagnostics"]
    _require(
        set(selection) == SELECTION_DIAGNOSTIC_KEYS,
        "selection diagnostic field set mismatch after locked validation",
    )
    result = {
        "seed": record["seed"],
        "auc_target_uniform_mean_success_full_atomic_paid": derived[
            "auc_target_uniform_mean_success_full_atomic_paid"
        ],
        "auc_target_uniform_mean_success_by_student_transitions": derived[
            "auc_target_uniform_mean_success_by_student_transitions"
        ],
        "final_target_uniform_mean_success": derived[
            "final_target_uniform_mean_success"
        ],
        "final_native_success_rate": derived["final_native_success_rate"],
        "student_fraction_of_paid": selection["student_fraction_of_paid"],
        "probe_fraction_of_paid": selection["probe_fraction_of_paid"],
        "probe_sweeps": raw["probe_sweeps"],
        "optimizer_updates": raw["optimizer_updates"],
        "mean_selection_entropy": selection["mean_selection_entropy"],
        "mean_selection_tv_from_uniform": selection[
            "mean_selection_tv_from_uniform"
        ],
        "mean_max_task_probability": selection["mean_max_task_probability"],
        "mean_assigned_probability_per_task": list(
            selection["mean_assigned_probability_per_task"]
        ),
        "realized_task_fraction": list(selection["realized_task_fraction"]),
        "realized_task_tv_from_uniform": selection[
            "realized_task_tv_from_uniform"
        ],
        "paid_budget_overshoot": selection["paid_budget_overshoot"],
    }
    _require(set(result) == PER_SEED_KEYS, "internal per-seed field mismatch")
    _require(_all_finite(result), "per-seed diagnostics contain non-finite values")
    for key in PER_SEED_VECTOR_FIELDS:
        _require(
            isinstance(result[key], list) and len(result[key]) == 8,
            f"{key} must contain exactly eight tasks",
        )
    return result


def _descriptive_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    _require(len(records) == 80, "descriptive summary requires exactly 80 seeds")
    scalar_metrics: dict[str, dict[str, float]] = {}
    for key in PER_SEED_SCALAR_FIELDS:
        values = [float(record[key]) for record in records]
        _require(len(values) == 80 and all(math.isfinite(value) for value in values), f"{key} invalid")
        scalar_metrics[key] = {
            "mean": float(fmean(values)),
            "sample_std": float(stdev(values)),
        }
    per_task_metrics: dict[str, dict[str, list[float]]] = {}
    for key in PER_SEED_VECTOR_FIELDS:
        values = [[float(value) for value in record[key]] for record in records]
        _require(
            len(values) == 80
            and all(len(row) == 8 for row in values)
            and all(math.isfinite(value) for row in values for value in row),
            f"{key} invalid",
        )
        columns = list(zip(*values, strict=True))
        per_task_metrics[key] = {
            "mean_per_task": [float(fmean(column)) for column in columns],
            "sample_std_per_task": [float(stdev(column)) for column in columns],
        }
    return {
        "scalar_metrics": scalar_metrics,
        "per_task_metrics": per_task_metrics,
    }


def extract_diagnostics(
    raw_path: Path,
    *,
    raw_logical_path: str,
    lock_path: Path,
    development_gate_path: Path,
    analyzer_path: Path | None = None,
) -> dict[str, Any]:
    """Return the deterministic descriptive report after locked validation."""
    logical = _portable_logical_path(raw_logical_path, "raw logical path")
    raw_capture = _capture_json(raw_path, "confirmatory raw artifact")
    lock_capture = _capture_json(lock_path, "source lock")
    gate_capture = _capture_json(development_gate_path, "development gate")
    raw = raw_capture.payload
    lock = lock_capture.payload
    _require(raw.get("schema") == RAW_SCHEMA, "confirmatory raw schema mismatch")
    analyzer_path = DEFAULT_ANALYZER_PATH if analyzer_path is None else analyzer_path
    expected_analyzer_sha = _verify_source_lock_before_import(lock, analyzer_path)
    gate_logical = _validate_raw_lock_and_gate_before_import(
        raw_capture,
        lock_capture,
        gate_capture,
    )
    analyzer = _load_verified_analyzer(analyzer_path, expected_analyzer_sha)
    validated = _strict_confirmatory(raw, analyzer)
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        records = [_per_seed(record) for record in validated["by_case"][arm]]
        _require(
            all(type(record["seed"]) is int for record in records)
            and [record["seed"] for record in records] == list(CONFIRMATORY_SEEDS),
            f"{arm} per-seed order mismatch",
        )
        arms[arm] = {
            "per_seed": records,
            "descriptive_summary": _descriptive_summary(records),
        }
    report = {
        "schema": DIAGNOSTICS_SCHEMA,
        "mode": "confirmatory",
        "status": STATUS,
        "raw_artifact": {
            "logical_path": logical,
            "size_bytes": raw_capture.size_bytes,
            "sha256": raw_capture.sha256,
            "schema": RAW_SCHEMA,
        },
        "source_lock": {
            "logical_path": CANONICAL_LOCK_LOGICAL_PATH,
            "size_bytes": lock_capture.size_bytes,
            "sha256": lock_capture.sha256,
            "schema": LOCK_SCHEMA,
        },
        "development_gate": {
            "logical_path": gate_logical,
            "size_bytes": gate_capture.size_bytes,
            "sha256": gate_capture.sha256,
            "schema": GATE_SCHEMA,
        },
        "schedule": {
            "arms": list(ARM_NAMES),
            "seeds": list(CONFIRMATORY_SEEDS),
            "run_count": 320,
        },
        "metric_policy": METRIC_POLICY,
        "arms": arms,
    }
    validate_report_shape(report)
    return report


def _require_float(value: object, label: str, *, low: float, high: float | None) -> float:
    _require(type(value) is float and math.isfinite(value), f"{label} must be a finite float")
    assert isinstance(value, float)
    _require(value >= low and (high is None or value <= high), f"{label} is out of range")
    return value


def _validate_per_seed_record(record: dict[str, Any], label: str) -> None:
    _require(set(record) == PER_SEED_KEYS, f"{label} field set mismatch")
    _require(type(record["seed"]) is int, f"{label} seed must be an integer")
    for key in UNIT_INTERVAL_SCALAR_FIELDS:
        _require_float(record[key], f"{label} {key}", low=0.0, high=1.0)
    _require(
        record["mean_max_task_probability"] >= 1.0 / 8.0,
        f"{label} mean_max_task_probability is below uniform",
    )
    _require_float(
        record["mean_selection_entropy"],
        f"{label} mean_selection_entropy",
        low=0.0,
        high=math.log(8.0),
    )
    for key in NONNEGATIVE_INTEGER_FIELDS:
        _require(
            type(record[key]) is int and record[key] >= 0,
            f"{label} {key} must be a nonnegative integer",
        )
    _require(
        math.isclose(
            record["student_fraction_of_paid"] + record["probe_fraction_of_paid"],
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"{label} paid fractions do not sum to one",
    )
    for key in PER_SEED_VECTOR_FIELDS:
        vector = record[key]
        _require(
            isinstance(vector, list) and len(vector) == 8,
            f"{label} {key} task count mismatch",
        )
        for task, value in enumerate(vector):
            _require_float(
                value, f"{label} {key}[{task}]", low=0.0, high=1.0
            )
        _require(
            math.isclose(sum(vector), 1.0, rel_tol=0.0, abs_tol=1e-9),
            f"{label} {key} does not sum to one",
        )


def _validate_descriptive_summary(summary: dict[str, Any], label: str) -> None:
    _require(set(summary) == SUMMARY_KEYS, f"{label} field set mismatch")
    scalars = summary["scalar_metrics"]
    _require(
        isinstance(scalars, dict) and tuple(scalars) == PER_SEED_SCALAR_FIELDS,
        f"{label} scalar summary fields mismatch",
    )
    for key, value in scalars.items():
        _require(
            isinstance(value, dict) and set(value) == SCALAR_SUMMARY_KEYS,
            f"{label} {key} shape mismatch",
        )
        high = 1.0 if key in UNIT_INTERVAL_SCALAR_FIELDS else (
            math.log(8.0) if key == "mean_selection_entropy" else None
        )
        _require_float(value["mean"], f"{label} {key} mean", low=0.0, high=high)
        if key == "mean_max_task_probability":
            _require(
                value["mean"] >= 1.0 / 8.0,
                f"{label} {key} mean is below uniform",
            )
        _require_float(
            value["sample_std"], f"{label} {key} sample_std", low=0.0, high=None
        )
    vectors = summary["per_task_metrics"]
    _require(
        isinstance(vectors, dict) and tuple(vectors) == PER_SEED_VECTOR_FIELDS,
        f"{label} vector summary fields mismatch",
    )
    for key, value in vectors.items():
        _require(
            isinstance(value, dict) and set(value) == VECTOR_SUMMARY_KEYS,
            f"{label} {key} shape mismatch",
        )
        means = value["mean_per_task"]
        standard_deviations = value["sample_std_per_task"]
        _require(
            isinstance(means, list)
            and isinstance(standard_deviations, list)
            and len(means) == len(standard_deviations) == 8,
            f"{label} {key} task count mismatch",
        )
        for task, item in enumerate(means):
            _require_float(
                item, f"{label} {key} mean_per_task[{task}]", low=0.0, high=1.0
            )
        for task, item in enumerate(standard_deviations):
            _require_float(
                item,
                f"{label} {key} sample_std_per_task[{task}]",
                low=0.0,
                high=None,
            )
        _require(
            math.isclose(sum(means), 1.0, rel_tol=0.0, abs_tol=1e-9),
            f"{label} {key} mean task probabilities do not sum to one",
        )


def validate_report_shape(report: dict[str, Any]) -> None:
    _require(set(report) == REPORT_KEYS, "diagnostics report field set mismatch")
    _require(report["schema"] == DIAGNOSTICS_SCHEMA, "diagnostics schema mismatch")
    _require(report["mode"] == "confirmatory", "diagnostics mode mismatch")
    _require(report["status"] == STATUS, "diagnostics status mismatch")
    _require(
        _typed_equal(report["metric_policy"], METRIC_POLICY),
        "metric policy mismatch",
    )
    _require(_all_finite(report), "diagnostics report contains non-finite values")
    for key, schema in (
        ("raw_artifact", RAW_SCHEMA),
        ("source_lock", LOCK_SCHEMA),
        ("development_gate", GATE_SCHEMA),
    ):
        binding = report[key]
        _require(
            isinstance(binding, dict) and set(binding) == RAW_BINDING_KEYS,
            f"{key} binding field set mismatch",
        )
        _portable_logical_path(
            binding["logical_path"], f"{key} logical path"
        )
        _require(
            type(binding["size_bytes"]) is int and binding["size_bytes"] > 0,
            f"{key} byte count invalid",
        )
        _require(
            _valid_sha256(binding["sha256"]),
            f"{key} SHA-256 invalid",
        )
        _require(binding["schema"] == schema, f"{key} schema binding mismatch")
    schedule = report["schedule"]
    _require(
        isinstance(schedule, dict) and set(schedule) == SCHEDULE_KEYS,
        "schedule field set mismatch",
    )
    _require(
        _typed_equal(schedule["arms"], list(ARM_NAMES)),
        "schedule arms mismatch",
    )
    _require(
        _typed_equal(schedule["seeds"], list(CONFIRMATORY_SEEDS)),
        "schedule seeds mismatch",
    )
    _require(
        type(schedule["run_count"]) is int and schedule["run_count"] == 320,
        "schedule run count mismatch",
    )
    arms = report["arms"]
    _require(
        isinstance(arms, dict) and tuple(arms) == ARM_NAMES,
        "arm key/order mismatch",
    )
    for arm in ARM_NAMES:
        arm_record = arms[arm]
        _require(
            isinstance(arm_record, dict)
            and set(arm_record) == {"per_seed", "descriptive_summary"},
            f"{arm} field set mismatch",
        )
        per_seed = arm_record["per_seed"]
        _require(
            isinstance(per_seed, list) and len(per_seed) == 80,
            f"{arm} seed count mismatch",
        )
        for position, record in enumerate(per_seed):
            _require(isinstance(record, dict), f"{arm} per-seed {position} is not an object")
            _validate_per_seed_record(record, f"{arm} per-seed {position}")
        _require(
            _typed_equal(
                [record["seed"] for record in per_seed], list(CONFIRMATORY_SEEDS)
            ),
            f"{arm} missing, duplicate, reordered, or extra seed records",
        )
        summary = arm_record["descriptive_summary"]
        _require(isinstance(summary, dict), f"{arm} summary is not an object")
        _validate_descriptive_summary(summary, f"{arm} summary")
        _require(
            _typed_equal(summary, _descriptive_summary(per_seed)),
            f"{arm} summary does not exactly recompute",
        )


def verify_diagnostics(
    report_path: Path,
    raw_path: Path,
    *,
    lock_path: Path,
    development_gate_path: Path,
    raw_logical_path: str | None = None,
    analyzer_path: Path | None = None,
) -> dict[str, Any]:
    report_capture = _capture_json(report_path, "descriptive diagnostics")
    report = report_capture.payload
    validate_report_shape(report)
    logical = (
        report["raw_artifact"]["logical_path"]
        if raw_logical_path is None
        else raw_logical_path
    )
    expected = extract_diagnostics(
        raw_path,
        raw_logical_path=logical,
        lock_path=lock_path,
        development_gate_path=development_gate_path,
        analyzer_path=analyzer_path,
    )
    _require(report == expected, "descriptive diagnostics do not exactly recompute")
    return {
        "schema": VERIFICATION_SCHEMA,
        "all_checks_passed": True,
        "run_count": 320,
        "arm_count": 4,
        "locked_raw_validation_replayed": True,
        "new_inferential_statistics": False,
    }


def write_json(path: Path, payload: dict[str, Any], *, overwrite: bool = False) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="extract diagnostics")
    build.add_argument("--raw", type=Path, required=True)
    build.add_argument("--lock", type=Path, required=True)
    build.add_argument("--development-gate", type=Path, required=True)
    build.add_argument("--analyzer", type=Path)
    build.add_argument("--raw-logical-path", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--overwrite", action="store_true")
    check = commands.add_parser("check", help="recompute and verify diagnostics")
    check.add_argument("--report", type=Path, required=True)
    check.add_argument("--raw", type=Path, required=True)
    check.add_argument("--lock", type=Path, required=True)
    check.add_argument("--development-gate", type=Path, required=True)
    check.add_argument("--analyzer", type=Path)
    check.add_argument("--raw-logical-path")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            report = extract_diagnostics(
                args.raw,
                raw_logical_path=args.raw_logical_path,
                lock_path=args.lock,
                development_gate_path=args.development_gate,
                analyzer_path=args.analyzer,
            )
            write_json(args.output, report, overwrite=args.overwrite)
            result: dict[str, Any] = {
                "schema": VERIFICATION_SCHEMA,
                "all_checks_passed": True,
                "mode": "build",
                "run_count": 320,
                "new_inferential_statistics": False,
            }
        else:
            result = verify_diagnostics(
                args.report,
                args.raw,
                lock_path=args.lock,
                development_gate_path=args.development_gate,
                raw_logical_path=args.raw_logical_path,
                analyzer_path=args.analyzer,
            )
    except (FileExistsError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
