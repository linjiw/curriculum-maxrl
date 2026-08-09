"""Build and verify the external Acrobot ProCuRL confirmatory-raw manifest.

The single full raw JSON is intentionally not copied into the source bundle.  The
manifest is a deterministic compact receipt for that file and for the four small
artifacts that bind it to the frozen protocol.  Compact verification needs only
the manifest and the bound small artifacts.  Full verification additionally
requires the external raw JSON, runs the locked strict validator, and reconciles
all 320 arm/seed records with the manifest's canonical per-run hashes.

Only caller-supplied portable logical paths are serialized.  Host filesystem
paths are never written to the manifest or verification result.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any


MANIFEST_SCHEMA = (
    "curriculum-maxrl/acrobot-procurl-selection-external-raw-manifest/v1"
)
VERIFICATION_SCHEMA = (
    "curriculum-maxrl/acrobot-procurl-selection-external-raw-verification/v1"
)
STUDY = "acrobot_procurl_selection_semantics"
CANONICAL_RUN_ENCODING = "json-utf8-sorted-keys-compact-separators-no-nan/v1"
INDEX_ORDER = "arm-major-then-frozen-seed-order"
RAW_SCHEMA = "curriculum-maxrl/acrobot-procurl-selection-raw/v1"
LOCK_SCHEMA = "curriculum-maxrl/acrobot-procurl-selection-lock/v1"
GATE_SCHEMA = "curriculum-maxrl/acrobot-procurl-selection-development-gates/v1"
ANALYSIS_SCHEMA = "curriculum-maxrl/acrobot-procurl-selection-analysis/v1"
PORTABLE_SCHEMA = (
    "curriculum-maxrl/acrobot-procurl-selection-portable-verification/v1"
)

ARM_NAMES = (
    "procurl_env_b20_f5120",
    "probe_sham_uniform_f5120",
    "ordinary_uniform",
    "u16_probe_range_matched_f5120",
)
CONFIRMATORY_SEEDS = tuple(range(21_000, 21_080))
DEVELOPMENT_SEEDS = (21_300, 21_301, 21_302)
QUICK_SEEDS = (21_400,)
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
    "development_seeds": list(DEVELOPMENT_SEEDS),
    "quick_seeds": list(QUICK_SEEDS),
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
BINDING_PAYLOAD_KEYS = {
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
    549.289282936898,
    549.2857356245377,
    549.2872495862491,
    549.2861638667421,
    549.2857443317533,
    549.2867282464132,
    549.2895726406483,
    549.2865534076619,
    549.2860469113732,
    549.288622325428,
    549.2862002480819,
    549.2873496418205,
    549.2863843248118,
    549.2858695926706,
    549.2856971184226,
    549.2867986978171,
    549.2863129624436,
    549.2865341586298,
    549.2861813786084,
    549.2860759557776,
    549.2857224574934,
    549.2861795828343,
    549.2859069840187,
    549.2864419313281,
    549.2864596960767,
    549.2865119613786,
    549.28707379166,
    549.2865645008778,
    549.2859656700225,
    549.2861366690263,
    549.2856495032845,
    549.2865620090922,
)
EXPECTED_ENTROPY_SUM_SENTINEL_HEX = "0x1.12a4ae5d8b0d5p+14"
NAIVE_ENTROPY_SUM_SENTINEL_HEX = "0x1.12a4ae5d8b0d3p+14"

ROLE_SCHEMAS = {
    "source_lock": LOCK_SCHEMA,
    "development_gate": GATE_SCHEMA,
    "confirmatory_analysis": ANALYSIS_SCHEMA,
    "portable_verification": PORTABLE_SCHEMA,
}
ROLE_ORDER = tuple(ROLE_SCHEMAS)

MANIFEST_KEYS = {
    "schema",
    "study",
    "mode",
    "raw_artifact",
    "canonical_run_encoding",
    "schedule",
    "run_index",
    "bindings",
}
BINDING_KEYS = {"logical_path", "size_bytes", "sha256", "schema"}
SCHEDULE_KEYS = {"arms", "seeds", "run_count", "index_order"}
RUN_INDEX_KEYS = {
    "ordinal",
    "arm",
    "seed",
    "canonical_json_size_bytes",
    "canonical_json_sha256",
}
ANALYSIS_KEYS = {
    "schema",
    "mode",
    "strict_validation_passed",
    "source_lock_verification",
    "development_gate_binding_verification",
    "primary",
    "secondary_holm_family",
    "secondary_multiplicity",
    "arm_descriptives",
    "statistical_conventions",
    "raw_artifact_relative_path",
    "raw_artifact_sha256",
}
DEVELOPMENT_VERIFICATION_KEYS = {
    "passed",
    "binding_exact",
    "development_gate_relative_path",
    "development_gate_sha256",
    "development_raw_relative_path",
    "development_raw_sha256",
    "same_source_lock",
    "raw_revalidated",
    "gate_recomputed_exactly",
    "all_gates_passed",
}
PORTABLE_KEYS = {
    "schema",
    "all_checks_passed",
    "recorded_execution_runtime",
    "source_lock_sha256",
    "source_manifest_verification",
    "live_reanalysis_runtime_verification",
    "invalid_pre_gate_archive_verification",
    "raw_ledger_validation",
    "development_gate_binding_verification",
    "stored_analysis_comparison",
    "scope",
}
SOURCE_MANIFEST_VERIFICATION_KEYS = {
    "passed",
    "checked_source_files",
    "all_live_hashes_match",
    "analyzer_hashed_before_import",
    "exact_manifest_key_set",
    "imported_analyzer_hash_matches_lock",
}
LIVE_RUNTIME_VERIFICATION_KEYS = {
    "passed",
    "checked_before_analyzer_import",
    "recorded_runtime",
    "live_runtime",
    "entropy_sum_sentinel_hex",
    "expected_entropy_sum_sentinel_hex",
    "known_naive_entropy_sum_sentinel_hex",
}
INVALID_ARCHIVE_VERIFICATION_KEYS = {
    "passed",
    "incident_relative_path",
    "incident_sha256",
    "archived_artifacts_checked",
    "outcome_blind",
    "development_gate_absent",
    "contrasts_uninspected",
}
RAW_LEDGER_VERIFICATION_KEYS = {
    "passed",
    "paired_seed_count",
    "arm_count",
    "cross_arm_crn_invariants",
}
CROSS_ARM_CRN_KEYS = {
    "passed",
    "paired_seeds_checked",
    "selection_rng_stream_replayed",
    "student_reset_stream_paired",
    "probe_coordinates_paired",
    "uniform_mechanics_paired_on_overlap",
    "same_actor_evaluations_paired",
}
INCIDENT_LOGICAL_PATH = (
    "frontier_rl/examples/INVALID_ABORTED_PRE_GATE_ENTROPY_SUM_MISMATCH/INCIDENT.json"
)
EXPECTED_INVALID_ARCHIVE_PATHS = [
    (
        "frontier_rl/examples/INVALID_ABORTED_PRE_GATE_ENTROPY_SUM_MISMATCH/"
        "ACROBOT_PROCURL_SELECTION_LOCK.json"
    ),
    (
        "frontier_rl/examples/INVALID_ABORTED_PRE_GATE_ENTROPY_SUM_MISMATCH/"
        "acrobot_procurl_selection_quick.json"
    ),
    (
        "frontier_rl/examples/INVALID_ABORTED_PRE_GATE_ENTROPY_SUM_MISMATCH/"
        "acrobot_procurl_selection_quick_analysis.json"
    ),
    (
        "frontier_rl/examples/INVALID_ABORTED_PRE_GATE_ENTROPY_SUM_MISMATCH/"
        "acrobot_procurl_selection_development.json"
    ),
]
EXPECTED_PORTABLE_SCOPE = (
    "verifies the replacement frozen bundle, exact pinned live reanalysis "
    "runtime and compensated-sum semantics, the outcome-blind invalid-wave "
    "archive, and deterministic reanalysis; excludes invalid-wave outcomes "
    "from evidence and does not reproduce the training execution"
)
LOCKED_ANALYZER_LOGICAL_PATH = (
    "frontier_rl/examples/analyze_acrobot_procurl_selection.py"
)
EXPECTED_LOCK_LOGICAL_PATH = (
    "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_LOCK.json"
)
HERE = Path(__file__).resolve().parent
DEFAULT_ANALYZER_PATH = HERE / "analyze_acrobot_procurl_selection.py"


def _fail(message: str) -> None:
    raise ValueError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _typed_equal(left: object, right: object) -> bool:
    """Equality that never treats bool/int/float identities as interchangeable."""
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


def _require_exact_keys(value: object, keys: set[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    assert isinstance(value, dict)
    _require(set(value) == keys, f"{label} field set mismatch")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _all_numbers_finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    return True


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
    _require(_all_numbers_finite(value), f"{label} contains a non-finite number")
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
        sha256=_sha256_bytes(data),
    )


def load_strict_json(path: Path, label: str) -> dict[str, Any]:
    """Compatibility entry point; production flows carry `_CapturedJson`."""
    return _capture_json(path, label).payload


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


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
    raw_parts = value.split("/")
    _require(
        all(part not in {"", ".", ".."} for part in raw_parts),
        f"{label} is not a normalized portable logical path",
    )
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}"
        for prefix in ("COM", "LPT")
        for number in range(1, 10)
    }
    for part in raw_parts:
        _require(
            not part.endswith((".", " ")),
            f"{label} contains a trailing dot or space",
        )
        device = part.split(".", 1)[0].rstrip(" .").upper()
        _require(device not in reserved, f"{label} contains a reserved device name")
    path = PurePosixPath(value)
    _require(
        path.as_posix() == value
        and bool(path.parts)
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{label} is not a normalized portable logical path",
    )
    return value


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


def _live_reanalysis_runtime() -> dict[str, str]:
    try:
        numpy_version = importlib.metadata.version("numpy")
        gymnasium_version = importlib.metadata.version("gymnasium")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError(
            "locked replay requires the exact pinned NumPy and Gymnasium"
        ) from error
    return {
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "numpy": numpy_version,
        "gymnasium": gymnasium_version,
    }


def _verify_runtime_before_import(lock: dict[str, Any]) -> dict[str, Any]:
    recorded = lock.get("runtime")
    _require(
        isinstance(recorded, dict) and set(recorded) == RUNTIME_KEYS,
        "source lock runtime field set mismatch",
    )
    assert isinstance(recorded, dict)
    recorded_relevant = {
        key: recorded.get(key) for key in PINNED_REANALYSIS_RUNTIME
    }
    _require(
        _typed_equal(recorded_relevant, PINNED_REANALYSIS_RUNTIME),
        "recorded runtime is not the exact pinned reanalysis runtime",
    )
    observed = _live_reanalysis_runtime()
    _require(
        _typed_equal(observed, PINNED_REANALYSIS_RUNTIME),
        "live runtime is not the exact pinned reanalysis runtime",
    )
    sentinel = sum(ENTROPY_SUM_SENTINEL_TERMS).hex()
    _require(
        sentinel == EXPECTED_ENTROPY_SUM_SENTINEL_HEX,
        "live built-in sum failed the frozen compensated-sum sentinel",
    )
    return {
        "passed": True,
        "checked_before_analyzer_import": True,
        "recorded_runtime": recorded_relevant,
        "live_runtime": observed,
        "entropy_sum_sentinel_hex": sentinel,
        "expected_entropy_sum_sentinel_hex": EXPECTED_ENTROPY_SUM_SENTINEL_HEX,
        "known_naive_entropy_sum_sentinel_hex": NAIVE_ENTROPY_SUM_SENTINEL_HEX,
    }


def _regular_file(path: Path, label: str) -> Path:
    lexical = path.absolute()
    _require(not lexical.is_symlink(), f"{label} must not be a symlink")
    _require(lexical.is_file(), f"{label} is missing or not a regular file")
    return lexical


def _resolve_logical(root: Path, logical: str, label: str) -> Path:
    logical = _portable_logical_path(logical, f"{label} logical path")
    root = root.resolve()
    candidate = root.joinpath(*PurePosixPath(logical).parts)
    current = root
    for part in PurePosixPath(logical).parts:
        current = current / part
        _require(not current.is_symlink(), f"{label} path contains a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes artifact root") from error
    return _regular_file(resolved, label)


def _artifact_binding(
    captured: _CapturedJson, logical_path: str, schema: str
) -> dict[str, Any]:
    logical_path = _portable_logical_path(logical_path, "artifact logical path")
    _require(
        captured.payload.get("schema") == schema,
        f"{logical_path} schema mismatch",
    )
    return {
        "logical_path": logical_path,
        "size_bytes": captured.size_bytes,
        "sha256": captured.sha256,
        "schema": schema,
    }


def _canonical_run_bytes(run: dict[str, Any]) -> bytes:
    try:
        text = json.dumps(
            run,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("run cannot be encoded as canonical strict JSON") from error
    return text.encode("utf-8")


def _raw_run_index(raw: dict[str, Any]) -> list[dict[str, Any]]:
    cases = raw.get("cases")
    _require(isinstance(cases, dict), "raw cases must be an object")
    assert isinstance(cases, dict)
    _require(tuple(cases) == ARM_NAMES, "raw arm key/order mismatch")
    records: list[dict[str, Any]] = []
    ordinal = 0
    for arm in ARM_NAMES:
        case = cases[arm]
        _require(isinstance(case, dict), f"raw case {arm} must be an object")
        runs = case.get("runs")
        _require(isinstance(runs, list), f"raw case {arm} runs must be a list")
        _require(
            all(isinstance(run, dict) and type(run.get("seed")) is int for run in runs)
            and [run["seed"] for run in runs] == list(CONFIRMATORY_SEEDS),
            f"raw case {arm} has missing, duplicate, reordered, or extra records",
        )
        for seed, run in zip(CONFIRMATORY_SEEDS, runs, strict=True):
            _require(isinstance(run, dict), f"raw run {arm}/{seed} must be an object")
            encoded = _canonical_run_bytes(run)
            records.append(
                {
                    "ordinal": ordinal,
                    "arm": arm,
                    "seed": seed,
                    "canonical_json_size_bytes": len(encoded),
                    "canonical_json_sha256": _sha256_bytes(encoded),
                }
            )
            ordinal += 1
    _require(ordinal == 320, "raw run count is not exactly 320")
    return records


def _strict_validate_confirmatory_raw(
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


def _validate_binding_shape(
    binding: object, *, label: str, expected_schema: str
) -> dict[str, Any]:
    record = _require_exact_keys(binding, BINDING_KEYS, label)
    _portable_logical_path(record["logical_path"], f"{label} logical path")
    _require(
        type(record["size_bytes"]) is int and record["size_bytes"] > 0,
        f"{label} size is invalid",
    )
    _require(_valid_sha256(record["sha256"]), f"{label} SHA-256 is invalid")
    _require(record["schema"] == expected_schema, f"{label} schema mismatch")
    return record


def validate_manifest_shape(manifest: dict[str, Any]) -> dict[str, Any]:
    _require(_all_numbers_finite(manifest), "manifest contains a non-finite number")
    record = _require_exact_keys(manifest, MANIFEST_KEYS, "external manifest")
    _require(record["schema"] == MANIFEST_SCHEMA, "manifest schema mismatch")
    _require(record["study"] == STUDY, "manifest study mismatch")
    _require(record["mode"] == "confirmatory", "manifest mode mismatch")
    _require(
        record["canonical_run_encoding"] == CANONICAL_RUN_ENCODING,
        "manifest canonical run encoding mismatch",
    )
    raw_binding = _validate_binding_shape(
        record["raw_artifact"], label="raw artifact binding", expected_schema=RAW_SCHEMA
    )
    schedule = _require_exact_keys(record["schedule"], SCHEDULE_KEYS, "schedule")
    expected_count = len(ARM_NAMES) * len(CONFIRMATORY_SEEDS)
    _require(_typed_equal(schedule["arms"], list(ARM_NAMES)), "schedule arms mismatch")
    _require(
        _typed_equal(schedule["seeds"], list(CONFIRMATORY_SEEDS)),
        "schedule seeds mismatch",
    )
    _require(
        type(schedule["run_count"]) is int
        and schedule["run_count"] == expected_count == 320,
        "schedule count mismatch",
    )
    _require(schedule["index_order"] == INDEX_ORDER, "schedule order mismatch")

    index = record["run_index"]
    _require(isinstance(index, list) and len(index) == 320, "run index length mismatch")
    expected_coordinates = [
        (arm, seed)
        for arm in ARM_NAMES
        for seed in CONFIRMATORY_SEEDS
    ]
    for ordinal, (entry, (arm, seed)) in enumerate(
        zip(index, expected_coordinates, strict=True)
    ):
        entry = _require_exact_keys(entry, RUN_INDEX_KEYS, f"run index {ordinal}")
        _require(
            type(entry["ordinal"]) is int and entry["ordinal"] == ordinal,
            f"run index {ordinal} ordinal mismatch",
        )
        _require(entry["arm"] == arm, f"run index {ordinal} arm mismatch")
        _require(
            type(entry["seed"]) is int and entry["seed"] == seed,
            f"run index {ordinal} seed mismatch",
        )
        _require(
            type(entry["canonical_json_size_bytes"]) is int
            and entry["canonical_json_size_bytes"] > 0,
            f"run index {ordinal} size invalid",
        )
        _require(
            _valid_sha256(entry["canonical_json_sha256"]),
            f"run index {ordinal} SHA-256 invalid",
        )

    bindings = _require_exact_keys(record["bindings"], set(ROLE_ORDER), "bindings")
    _require(tuple(bindings) == ROLE_ORDER, "binding role order mismatch")
    paths = [raw_binding["logical_path"]]
    for role in ROLE_ORDER:
        binding = _validate_binding_shape(
            bindings[role], label=f"{role} binding", expected_schema=ROLE_SCHEMAS[role]
        )
        paths.append(binding["logical_path"])
    _require(len(paths) == len(set(paths)), "artifact logical paths are not unique")
    return record


def _verify_file_binding(
    captured: _CapturedJson, binding: dict[str, Any], label: str
) -> None:
    _require(
        captured.size_bytes == binding["size_bytes"],
        f"{label} byte count mismatch",
    )
    _require(captured.sha256 == binding["sha256"], f"{label} SHA-256 mismatch")
    _require(
        captured.payload.get("schema") == binding["schema"],
        f"{label} schema mismatch",
    )


def _verify_source_lock_before_import(
    lock: dict[str, Any], analyzer_path: Path
) -> tuple[str, dict[str, Any]]:
    _require_exact_keys(lock, LOCK_KEYS, "source lock")
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
    seed_audit = lock.get("seed_collision_audit")
    _require(
        isinstance(seed_audit, dict) and seed_audit.get("passed") is True,
        "source lock seed-collision audit is not positive",
    )
    source_manifest = lock.get("source_sha256")
    _require(isinstance(source_manifest, dict), "source lock lacks a source manifest")
    assert isinstance(source_manifest, dict)
    _require(
        tuple(source_manifest) == EXPECTED_SOURCE_RELATIVE_PATHS,
        "source lock manifest key/order mismatch",
    )
    _require(
        all(_valid_sha256(digest) for digest in source_manifest.values()),
        "source lock manifest contains an invalid SHA-256",
    )
    expected = source_manifest[LOCKED_ANALYZER_LOGICAL_PATH]
    _require(
        _sha256(_regular_file(analyzer_path, "locked analyzer")) == expected,
        "bundle analyzer does not match the source lock",
    )
    runtime_check = _verify_runtime_before_import(lock)
    return expected, runtime_check


def _load_verified_analyzer(analyzer_path: Path, expected_sha256: str) -> ModuleType:
    """Compile and execute the exact source buffer whose SHA-256 was verified.

    Deliberately bypass importlib loaders, ``sys.modules``, and adjacent bytecode
    caches.  Reading once also closes the hash-to-exec source-swap window.
    """
    analyzer_path = _regular_file(analyzer_path, "locked analyzer")
    source_bytes = analyzer_path.read_bytes()
    _require(
        _sha256_bytes(source_bytes) == expected_sha256,
        "analyzer source buffer hash mismatch before execution",
    )
    name = f"_locked_acrobot_procurl_analysis_{expected_sha256[:16]}"
    try:
        code = compile(source_bytes, str(analyzer_path), "exec", dont_inherit=True)
    except (SyntaxError, UnicodeError, ValueError) as error:
        raise ValueError("verified analyzer source cannot be compiled") from error
    module = ModuleType(name)
    module.__file__ = str(analyzer_path)
    module.__package__ = "frontier_rl.examples"
    module.__loader__ = None
    module.__spec__ = None
    module.__cached__ = None
    exec(code, module.__dict__)
    return module


def _validate_bound_payloads(
    manifest: dict[str, Any],
    captured_artifacts: Mapping[str, _CapturedJson],
    *,
    analyzer_path: Path,
) -> tuple[dict[str, _CapturedJson], str]:
    _require(
        set(captured_artifacts) == set(ROLE_ORDER),
        "bound artifact role set mismatch",
    )
    for role in ROLE_ORDER:
        _verify_file_binding(
            captured_artifacts[role],
            manifest["bindings"][role],
            role.replace("_", " "),
        )

    lock = captured_artifacts["source_lock"].payload
    expected_analyzer_sha, runtime_check = _verify_source_lock_before_import(
        lock, analyzer_path
    )
    lock_sha = manifest["bindings"]["source_lock"]["sha256"]

    gate = _require_exact_keys(
        captured_artifacts["development_gate"].payload,
        GATE_KEYS,
        "development gate",
    )
    _require(
        gate["schema"] == GATE_SCHEMA
        and gate.get("mode") == "development"
        and gate.get("all_gates_passed") is True,
        "development gate is not a passing development artifact",
    )
    _require(gate.get("source_lock_sha256") == lock_sha, "gate source-lock mismatch")
    _portable_logical_path(gate.get("raw_artifact_relative_path"), "development raw path")
    _require(
        _valid_sha256(gate.get("raw_artifact_sha256")),
        "development raw SHA-256 is invalid",
    )

    stored = _require_exact_keys(
        captured_artifacts["confirmatory_analysis"].payload,
        ANALYSIS_KEYS,
        "confirmatory analysis",
    )
    _require(
        stored["schema"] == ANALYSIS_SCHEMA
        and stored.get("mode") == "confirmatory"
        and stored.get("strict_validation_passed") is True,
        "stored analysis is not a strict confirmatory analysis",
    )
    source = _require_exact_keys(
        stored.get("source_lock_verification"),
        SOURCE_VERIFICATION_KEYS,
        "analysis source-lock verification",
    )
    _require(
        source.get("passed") is True and source.get("source_lock_sha256") == lock_sha,
        "analysis source-lock binding mismatch",
    )
    _require(
        _typed_equal(source.get("runtime"), lock["runtime"])
        and _typed_equal(
            source.get("checked_source_files"), sorted(lock["source_sha256"])
        ),
        "analysis source-lock verification payload mismatch",
    )
    development = _require_exact_keys(
        stored.get("development_gate_binding_verification"),
        DEVELOPMENT_VERIFICATION_KEYS,
        "analysis development-gate verification",
    )
    _require(
        all(
            development[key] is True
            for key in (
                "passed",
                "binding_exact",
                "same_source_lock",
                "raw_revalidated",
                "gate_recomputed_exactly",
                "all_gates_passed",
            )
        ),
        "analysis development-gate verification is not positive",
    )
    _require(
        development["development_gate_relative_path"]
        == manifest["bindings"]["development_gate"]["logical_path"]
        and development["development_gate_sha256"]
        == manifest["bindings"]["development_gate"]["sha256"],
        "analysis development-gate artifact binding mismatch",
    )
    _require(
        development["development_raw_relative_path"]
        == gate["raw_artifact_relative_path"]
        and development["development_raw_sha256"] == gate["raw_artifact_sha256"],
        "analysis development-raw binding mismatch",
    )
    _require(
        stored.get("raw_artifact_relative_path")
        == manifest["raw_artifact"]["logical_path"]
        and stored.get("raw_artifact_sha256") == manifest["raw_artifact"]["sha256"],
        "analysis confirmatory-raw binding mismatch",
    )

    portable = _require_exact_keys(
        captured_artifacts["portable_verification"].payload,
        PORTABLE_KEYS,
        "portable verification",
    )
    _require(
        portable["schema"] == PORTABLE_SCHEMA
        and portable.get("all_checks_passed") is True
        and portable.get("scope") == EXPECTED_PORTABLE_SCOPE,
        "portable verification is not positive",
    )
    _require(
        portable.get("source_lock_sha256") == lock_sha,
        "portable source-lock binding mismatch",
    )
    _require(
        _typed_equal(portable.get("recorded_execution_runtime"), lock["runtime"]),
        "portable recorded execution runtime mismatch",
    )
    source_manifest_check = _require_exact_keys(
        portable.get("source_manifest_verification"),
        SOURCE_MANIFEST_VERIFICATION_KEYS,
        "portable source-manifest verification",
    )
    _require(
        all(
            source_manifest_check[key] is True
            for key in (
                "passed",
                "all_live_hashes_match",
                "analyzer_hashed_before_import",
                "exact_manifest_key_set",
                "imported_analyzer_hash_matches_lock",
            )
        )
        and _typed_equal(
            source_manifest_check.get("checked_source_files"),
            sorted(lock["source_sha256"]),
        ),
        "portable source-manifest verification is incomplete",
    )
    live_runtime = _require_exact_keys(
        portable.get("live_reanalysis_runtime_verification"),
        LIVE_RUNTIME_VERIFICATION_KEYS,
        "portable live-runtime verification",
    )
    _require(
        live_runtime.get("passed") is True
        and live_runtime.get("checked_before_analyzer_import") is True
        and _typed_equal(live_runtime.get("recorded_runtime"), PINNED_REANALYSIS_RUNTIME)
        and _typed_equal(live_runtime.get("live_runtime"), PINNED_REANALYSIS_RUNTIME)
        and live_runtime.get("entropy_sum_sentinel_hex")
        == EXPECTED_ENTROPY_SUM_SENTINEL_HEX
        and live_runtime.get("expected_entropy_sum_sentinel_hex")
        == EXPECTED_ENTROPY_SUM_SENTINEL_HEX
        and live_runtime.get("known_naive_entropy_sum_sentinel_hex")
        == NAIVE_ENTROPY_SUM_SENTINEL_HEX,
        "portable live-runtime verification mismatch",
    )
    _require(
        _typed_equal(runtime_check, live_runtime),
        "portable live-runtime verification does not match this process",
    )
    invalid_archive = _require_exact_keys(
        portable.get("invalid_pre_gate_archive_verification"),
        INVALID_ARCHIVE_VERIFICATION_KEYS,
        "portable invalid-archive verification",
    )
    _require(
        invalid_archive.get("passed") is True
        and invalid_archive.get("outcome_blind") is True
        and invalid_archive.get("development_gate_absent") is True
        and invalid_archive.get("contrasts_uninspected") is True
        and invalid_archive.get("incident_relative_path") == INCIDENT_LOGICAL_PATH
        and invalid_archive.get("incident_sha256")
        == lock["source_sha256"][INCIDENT_LOGICAL_PATH]
        and _typed_equal(
            invalid_archive.get("archived_artifacts_checked"),
            EXPECTED_INVALID_ARCHIVE_PATHS,
        ),
        "portable invalid-archive verification mismatch",
    )
    _require(
        _typed_equal(
            portable.get("development_gate_binding_verification"), development
        ),
        "portable development-gate binding mismatch",
    )
    ledger = _require_exact_keys(
        portable.get("raw_ledger_validation"),
        RAW_LEDGER_VERIFICATION_KEYS,
        "portable raw-ledger validation",
    )
    _require(
        ledger.get("passed") is True
        and type(ledger.get("paired_seed_count")) is int
        and ledger.get("paired_seed_count") == 80
        and type(ledger.get("arm_count")) is int
        and ledger.get("arm_count") == 4,
        "portable raw-ledger count mismatch",
    )
    crn = _require_exact_keys(
        ledger.get("cross_arm_crn_invariants"),
        CROSS_ARM_CRN_KEYS,
        "portable cross-arm CRN verification",
    )
    _require(
        all(
            crn[key] is True
            for key in CROSS_ARM_CRN_KEYS - {"paired_seeds_checked"}
        )
        and _typed_equal(crn.get("paired_seeds_checked"), list(CONFIRMATORY_SEEDS)),
        "portable cross-arm CRN verification mismatch",
    )
    comparison = _require_exact_keys(
        portable.get("stored_analysis_comparison"),
        {"passed", "all_recomputed_fields_match", "stored_analysis_sha256"},
        "portable stored-analysis comparison",
    )
    _require(
        comparison.get("passed") is True
        and comparison.get("all_recomputed_fields_match") is True
        and comparison.get("stored_analysis_sha256")
        == manifest["bindings"]["confirmatory_analysis"]["sha256"],
        "portable stored-analysis binding mismatch",
    )
    return dict(captured_artifacts), expected_analyzer_sha


def build_manifest(
    *,
    raw_path: Path,
    raw_logical_path: str,
    bound_paths: Mapping[str, Path],
    bound_logical_paths: Mapping[str, str],
    analyzer_path: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic manifest after full strict validation."""
    _require(set(bound_paths) == set(ROLE_ORDER), "bound path role set mismatch")
    _require(
        set(bound_logical_paths) == set(ROLE_ORDER),
        "bound logical-path role set mismatch",
    )
    raw_capture = _capture_json(raw_path, "external raw artifact")
    index = _raw_run_index(raw_capture.payload)
    raw_binding = _artifact_binding(raw_capture, raw_logical_path, RAW_SCHEMA)
    captured_artifacts = {
        role: _capture_json(bound_paths[role], role.replace("_", " "))
        for role in ROLE_ORDER
    }
    bindings = {
        role: _artifact_binding(
            captured_artifacts[role],
            bound_logical_paths[role],
            ROLE_SCHEMAS[role],
        )
        for role in ROLE_ORDER
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "study": STUDY,
        "mode": "confirmatory",
        "raw_artifact": raw_binding,
        "canonical_run_encoding": CANONICAL_RUN_ENCODING,
        "schedule": {
            "arms": list(ARM_NAMES),
            "seeds": list(CONFIRMATORY_SEEDS),
            "run_count": 320,
            "index_order": INDEX_ORDER,
        },
        "run_index": index,
        "bindings": bindings,
    }
    validate_manifest_shape(manifest)
    analyzer_path = DEFAULT_ANALYZER_PATH if analyzer_path is None else analyzer_path
    captured_artifacts, expected_analyzer_sha = _validate_bound_payloads(
        manifest, captured_artifacts, analyzer_path=analyzer_path
    )
    _verify_raw_before_import(
        manifest,
        raw_capture,
        bound_artifacts=captured_artifacts,
    )
    analyzer = _load_verified_analyzer(analyzer_path, expected_analyzer_sha)
    _strict_validate_confirmatory_raw(raw_capture.payload, analyzer)
    return manifest


def _verify_raw_before_import(
    manifest: dict[str, Any],
    raw_capture: _CapturedJson,
    *,
    bound_artifacts: Mapping[str, _CapturedJson],
) -> dict[str, Any]:
    binding = manifest["raw_artifact"]
    _require(
        raw_capture.size_bytes == binding["size_bytes"],
        "external raw byte count mismatch",
    )
    _require(
        raw_capture.sha256 == binding["sha256"],
        "external raw SHA-256 mismatch",
    )
    raw = raw_capture.payload
    _require(raw.get("schema") == binding["schema"], "external raw schema mismatch")
    _require(_raw_run_index(raw) == manifest["run_index"], "320-run index mismatch")
    lock_binding = manifest["bindings"]["source_lock"]
    gate_binding = manifest["bindings"]["development_gate"]
    provenance = raw.get("provenance", {})
    bound_lock = bound_artifacts["source_lock"].payload
    _require(
        isinstance(provenance, dict)
        and provenance.get("source_lock_relative_path")
        == lock_binding["logical_path"]
        == EXPECTED_LOCK_LOGICAL_PATH
        and provenance.get("source_lock_sha256") == lock_binding["sha256"],
        "raw source-lock binding mismatch",
    )
    _require(
        provenance.get("source_lock_enforced") is True
        and _typed_equal(provenance.get("runtime"), bound_lock["runtime"])
        and _typed_equal(
            provenance.get("source_sha256"), bound_lock["source_sha256"]
        )
        and _typed_equal(
            provenance.get("seed_collision_audit"),
            bound_lock["seed_collision_audit"],
        ),
        "raw provenance does not exactly match the source lock",
    )
    development = raw.get("protocol", {}).get("development_gate")
    bound_gate = bound_artifacts["development_gate"].payload
    stored_development = bound_artifacts["confirmatory_analysis"].payload[
        "development_gate_binding_verification"
    ]
    portable_development = bound_artifacts["portable_verification"].payload[
        "development_gate_binding_verification"
    ]
    _require(
        _typed_equal(stored_development, portable_development),
        "stored and portable development bindings diverge",
    )
    _require(
        isinstance(development, dict)
        and set(development) == BINDING_PAYLOAD_KEYS,
        "raw development-gate binding field set mismatch",
    )
    _require(
        development.get("relative_path") == gate_binding["logical_path"]
        and development.get("sha256") == gate_binding["sha256"]
        and development.get("raw_artifact_relative_path")
        == bound_gate["raw_artifact_relative_path"]
        == stored_development["development_raw_relative_path"]
        and development.get("raw_artifact_sha256")
        == bound_gate["raw_artifact_sha256"]
        == stored_development["development_raw_sha256"]
        and development.get("all_gates_passed") is True,
        "raw development-gate/development-raw binding mismatch",
    )
    return raw


def verify_manifest(
    manifest_path: Path,
    *,
    artifact_root: Path,
    mode: str,
    raw_path: Path | None = None,
) -> dict[str, Any]:
    """Verify a manifest in compact mode or with the complete external raw."""
    _require(mode in {"compact", "full"}, "verification mode must be compact or full")
    if mode == "compact":
        _require(raw_path is None, "compact verification must not read a raw artifact")
    else:
        _require(raw_path is not None, "full verification requires the raw artifact")
    manifest_capture = _capture_json(manifest_path, "external manifest")
    manifest = manifest_capture.payload
    validate_manifest_shape(manifest)
    paths = {
        role: _resolve_logical(
            artifact_root,
            manifest["bindings"][role]["logical_path"],
            role.replace("_", " "),
        )
        for role in ROLE_ORDER
    }
    analyzer_path = _resolve_logical(
        artifact_root, LOCKED_ANALYZER_LOGICAL_PATH, "locked analyzer"
    )
    captured_artifacts = {
        role: _capture_json(paths[role], role.replace("_", " "))
        for role in ROLE_ORDER
    }
    captured_artifacts, expected_analyzer_sha = _validate_bound_payloads(
        manifest, captured_artifacts, analyzer_path=analyzer_path
    )
    if mode == "full":
        assert raw_path is not None
        raw_capture = _capture_json(raw_path, "external raw artifact")
        raw = _verify_raw_before_import(
            manifest,
            raw_capture,
            bound_artifacts=captured_artifacts,
        )
        analyzer = _load_verified_analyzer(analyzer_path, expected_analyzer_sha)
        _strict_validate_confirmatory_raw(raw, analyzer)
    return {
        "schema": VERIFICATION_SCHEMA,
        "mode": mode,
        "all_checks_passed": True,
        "bound_artifact_roles": list(ROLE_ORDER),
        "bound_artifact_count": len(ROLE_ORDER),
        "run_index_count": 320,
        "raw_bytes_verified": mode == "full",
        "locked_raw_validation_replayed": mode == "full",
        "all_run_index_records_reconciled": mode == "full",
        "host_paths_serialized": False,
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


def _role_arguments(parser: argparse.ArgumentParser) -> None:
    for role in ROLE_ORDER:
        option = role.replace("_", "-")
        parser.add_argument(f"--{option}", type=Path, required=True)
        parser.add_argument(f"--{option}-logical-path", required=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build a manifest from the full raw")
    build.add_argument("--raw", type=Path, required=True)
    build.add_argument("--raw-logical-path", required=True)
    _role_arguments(build)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--overwrite", action="store_true")

    check = commands.add_parser("check", help="verify an existing manifest")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--artifact-root", type=Path, required=True)
    modes = check.add_mutually_exclusive_group(required=True)
    modes.add_argument("--compact", action="store_true")
    modes.add_argument("--full", action="store_true")
    check.add_argument("--raw", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            bound_paths = {
                role: getattr(args, role) for role in ROLE_ORDER
            }
            bound_logical_paths = {
                role: getattr(args, f"{role}_logical_path") for role in ROLE_ORDER
            }
            manifest = build_manifest(
                raw_path=args.raw,
                raw_logical_path=args.raw_logical_path,
                bound_paths=bound_paths,
                bound_logical_paths=bound_logical_paths,
            )
            write_json(args.output, manifest, overwrite=args.overwrite)
            result: dict[str, Any] = {
                "schema": VERIFICATION_SCHEMA,
                "mode": "build",
                "all_checks_passed": True,
                "run_index_count": 320,
                "host_paths_serialized": False,
            }
        else:
            mode = "full" if args.full else "compact"
            result = verify_manifest(
                args.manifest,
                artifact_root=args.artifact_root,
                mode=mode,
                raw_path=args.raw,
            )
    except (FileExistsError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
