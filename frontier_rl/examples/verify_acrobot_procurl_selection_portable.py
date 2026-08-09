"""Read-only verifier for the completed ProCuRL-selection bundle.

The analyzer is located and SHA-256 checked against the source lock *before*
Python executes any of its bytes.  Before that import, the verifier also
requires the exact pinned reanalysis runtime and checks Python 3.12's
compensated built-in summation with a frozen sentinel.  It proves bundle
integrity and deterministic reanalysis; it does not re-execute trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

SCHEMA = "curriculum-maxrl/acrobot-procurl-selection-portable-verification/v1"
EXPECTED_LOCK_RELATIVE_PATH = "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_LOCK.json"
LOCKED_ANALYZER_RELATIVE_PATH = (
    "frontier_rl/examples/analyze_acrobot_procurl_selection.py"
)
INVALID_INCIDENT_RELATIVE_PATH = (
    "frontier_rl/examples/INVALID_ABORTED_PRE_GATE_ENTROPY_SUM_MISMATCH/INCIDENT.json"
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
REPORT_METADATA_KEYS = {"raw_artifact_relative_path", "raw_artifact_sha256"}
EXPECTED_REANALYSIS_RUNTIME = {
    "python_implementation": "CPython",
    "python": "3.12.13",
    "numpy": "2.5.1",
    "gymnasium": "1.3.0",
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


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} as strict JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain exactly one JSON object")
    return value


def _live_reanalysis_runtime() -> dict[str, str]:
    try:
        numpy_version = importlib.metadata.version("numpy")
        gymnasium_version = importlib.metadata.version("gymnasium")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError(
            "portable reanalysis requires the exact pinned NumPy and Gymnasium"
        ) from error
    return {
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "numpy": numpy_version,
        "gymnasium": gymnasium_version,
    }


def _entropy_sum_sentinel_hex() -> str:
    return sum(ENTROPY_SUM_SENTINEL_TERMS).hex()


def _verify_reanalysis_runtime_before_import(lock: dict) -> dict:
    recorded = lock.get("runtime")
    if not isinstance(recorded, dict):
        raise TypeError("source lock lacks a recorded reanalysis runtime")
    recorded_relevant = {key: recorded.get(key) for key in EXPECTED_REANALYSIS_RUNTIME}
    if recorded_relevant != EXPECTED_REANALYSIS_RUNTIME:
        raise ValueError("recorded runtime is not the exact pinned reanalysis runtime")
    observed = _live_reanalysis_runtime()
    if observed != EXPECTED_REANALYSIS_RUNTIME:
        raise ValueError(
            "live runtime is not the exact pinned reanalysis runtime: "
            f"expected={EXPECTED_REANALYSIS_RUNTIME!r}, observed={observed!r}"
        )
    sentinel_hex = _entropy_sum_sentinel_hex()
    if sentinel_hex != EXPECTED_ENTROPY_SUM_SENTINEL_HEX:
        raise ValueError(
            "live built-in sum failed the frozen Python 3.12 compensated-sum sentinel"
        )
    return {
        "passed": True,
        "checked_before_analyzer_import": True,
        "recorded_runtime": recorded_relevant,
        "live_runtime": observed,
        "entropy_sum_sentinel_hex": sentinel_hex,
        "expected_entropy_sum_sentinel_hex": EXPECTED_ENTROPY_SUM_SENTINEL_HEX,
        "known_naive_entropy_sum_sentinel_hex": NAIVE_ENTROPY_SUM_SENTINEL_HEX,
    }


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_relative(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a canonical project-relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{label} must be a canonical project-relative path")
    return path


def _source_file(source_root: Path, relative: object, label: str) -> Path:
    pure = _safe_relative(relative, label)
    root = source_root.resolve()
    path = root.joinpath(*pure.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the source root") from error
    if not path.is_file():
        raise ValueError(f"{label} is missing from source bundle: {relative}")
    return path


def _discover_source_root(lock_path: Path, manifest: dict[str, str]) -> Path:
    for candidate in (lock_path.resolve().parent, *lock_path.resolve().parents):
        try:
            if all(
                _source_file(candidate, relative, "locked source").is_file()
                for relative in manifest
            ):
                return candidate
        except ValueError:
            continue
    raise ValueError("could not discover source root; pass --source-root")


def _verify_manifest_bytes(lock: dict, source_root: Path) -> dict:
    manifest = lock.get("source_sha256")
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("source lock lacks a nonempty manifest")
    if LOCKED_ANALYZER_RELATIVE_PATH not in manifest:
        raise ValueError("source lock omits the analyzer")
    checked = []
    for relative, expected in manifest.items():
        if not _valid_sha256(expected):
            raise ValueError(f"invalid source SHA-256: {relative}")
        path = _source_file(source_root, relative, "locked source")
        if _sha256(path) != expected:
            raise ValueError(f"locked source hash mismatch: {relative}")
        checked.append(relative)
    return {
        "passed": True,
        "checked_source_files": sorted(checked),
        "all_live_hashes_match": True,
        "analyzer_hashed_before_import": True,
    }


def _load_verified_analyzer(analyzer_path: Path, expected_sha256: str) -> ModuleType:
    """Hash first, then and only then execute the analyzer module."""
    if _sha256(analyzer_path) != expected_sha256:
        raise ValueError("analyzer hash changed before import")
    module_name = f"_locked_acrobot_procurl_analysis_{expected_sha256[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, analyzer_path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot create an import specification for locked analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_v2_audit(lock: dict, source_root: Path, analysis: ModuleType) -> None:
    v2_path = _source_file(
        source_root,
        "frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json",
        "V2 dependency lock",
    )
    v2 = _load_json(v2_path, "V2 dependency lock")
    if v2.get("schema") != "curriculum-maxrl/acrobot-curriculum-tournament-lock/v2":
        raise ValueError("V2 dependency lock schema mismatch")
    v2_hashes = v2.get("source_sha256")
    paths = tuple(analysis.V2_DEPENDENCY_PATHS)
    if not isinstance(v2_hashes, dict) or not set(paths) <= set(v2_hashes):
        raise ValueError("V2 dependency lock lacks transitive dependencies")
    live = {
        relative: _sha256(_source_file(source_root, relative, "V2 dependency"))
        for relative in paths
    }
    frozen = {relative: v2_hashes[relative] for relative in paths}
    expected = {
        "passed": True,
        "v2_lock_relative_path": (
            "frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json"
        ),
        "v2_lock_sha256": _sha256(v2_path),
        "v2_lock_schema": v2["schema"],
        "dependency_paths": list(paths),
        "live_dependency_sha256": live,
        "v2_locked_dependency_sha256": frozen,
        "all_live_dependencies_match_v2": True,
    }
    if live != frozen or lock.get("v2_dependency_audit") != expected:
        raise ValueError("V2 transitive dependency audit mismatch")


def _verify_invalid_incident(source_root: Path) -> dict:
    incident_path = _source_file(
        source_root, INVALID_INCIDENT_RELATIVE_PATH, "invalid-wave incident"
    )
    record = _load_json(incident_path, "invalid-wave incident")
    expected_top = {
        "schema",
        "status",
        "created_utc",
        "outcome_blind",
        "incident",
        "execution_accounting",
        "burned_seed_blocks",
        "replacement_registration",
        "archived_artifacts",
    }
    if set(record) != expected_top:
        raise ValueError("invalid-wave incident schema is not closed")
    if (
        record["schema"]
        != "curriculum-maxrl/acrobot-procurl-selection-invalid-abort/v1"
        or record["status"] != "invalid_aborted_pre_gate_entropy_sum_mismatch"
        or record["outcome_blind"] is not True
        or not _is_utc_iso8601(record["created_utc"])
    ):
        raise ValueError("invalid-wave incident identity mismatch")
    incident = record["incident"]
    incident_keys = {
        "stage",
        "reason",
        "mismatched_evaluation_records",
        "maximum_absolute_difference",
        "registered_absolute_tolerance",
        "stored_aggregates_match_python312_builtin_sum_for_all_records",
        "development_gate_created",
        "confirmation_launched",
        "arm_contrasts_inspected",
        "effect_directions_inspected",
        "confidence_intervals_inspected",
        "p_values_inspected",
        "minimum_effect_decisions_inspected",
    }
    if (
        not isinstance(incident, dict)
        or set(incident) != incident_keys
        or incident.get("stage") != "development_independent_validation_before_gate"
        or incident.get("mismatched_evaluation_records") != 73
        or incident.get("maximum_absolute_difference") != 7.275957614183426e-12
        or incident.get("registered_absolute_tolerance") != 1e-12
        or incident.get("stored_aggregates_match_python312_builtin_sum_for_all_records")
        is not True
        or incident.get("development_gate_created") is not False
        or incident.get("confirmation_launched") is not False
        or any(
            incident.get(key) is not False
            for key in (
                "arm_contrasts_inspected",
                "effect_directions_inspected",
                "confidence_intervals_inspected",
                "p_values_inspected",
                "minimum_effect_decisions_inspected",
            )
        )
    ):
        raise ValueError("invalid-wave outcome-blind incident facts mismatch")
    if record["execution_accounting"] != {
        "quick_seed": 21_200,
        "quick_runner_valid_runs": 4,
        "quick_expected_runs": 4,
        "quick_strict_analysis_passed": True,
        "development_seeds": [21_100, 21_101, 21_102],
        "development_runner_valid_runs": 12,
        "development_expected_runs": 12,
        "development_strict_analysis_passed": False,
        "scientific_evidence_status": "invalid_not_usable",
    }:
        raise ValueError("invalid-wave execution accounting mismatch")
    if record["burned_seed_blocks"] != {
        "invalid_quick": [21_200],
        "invalid_development": [21_100, 21_101, 21_102],
    }:
        raise ValueError("invalid-wave burned seed record mismatch")
    replacement = record["replacement_registration"]
    if (
        not isinstance(replacement, dict)
        or set(replacement)
        != {
            "confirmation_seeds_unchanged",
            "fresh_development_seeds",
            "fresh_quick_seed",
            "replacement_lock_created",
            "replacement_execution_started",
        }
        or replacement.get("confirmation_seeds_unchanged") != "21000-21079"
        or replacement.get("fresh_development_seeds") != [21_300, 21_301, 21_302]
        or replacement.get("fresh_quick_seed") != [21_400]
        or replacement.get("replacement_lock_created") is not False
        or replacement.get("replacement_execution_started") is not False
    ):
        raise ValueError("invalid-wave replacement registration mismatch")
    artifacts = record["archived_artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 4:
        raise ValueError("invalid-wave archive manifest length mismatch")
    expected_locations = {
        "invalid_source_lock": (
            "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_LOCK.json",
            (
                "frontier_rl/examples/"
                "INVALID_ABORTED_PRE_GATE_ENTROPY_SUM_MISMATCH/"
                "ACROBOT_PROCURL_SELECTION_LOCK.json"
            ),
        ),
        "invalid_quick_raw": (
            "frontier_rl/examples/acrobot_procurl_selection_quick.json",
            (
                "frontier_rl/examples/"
                "INVALID_ABORTED_PRE_GATE_ENTROPY_SUM_MISMATCH/"
                "acrobot_procurl_selection_quick.json"
            ),
        ),
        "invalid_quick_analysis": (
            "frontier_rl/examples/acrobot_procurl_selection_quick_analysis.json",
            (
                "frontier_rl/examples/"
                "INVALID_ABORTED_PRE_GATE_ENTROPY_SUM_MISMATCH/"
                "acrobot_procurl_selection_quick_analysis.json"
            ),
        ),
        "invalid_development_raw": (
            "frontier_rl/examples/acrobot_procurl_selection_development.json",
            (
                "frontier_rl/examples/"
                "INVALID_ABORTED_PRE_GATE_ENTROPY_SUM_MISMATCH/"
                "acrobot_procurl_selection_development.json"
            ),
        ),
    }
    checked = []
    observed_roles = set()
    for artifact in artifacts:
        if set(artifact) != {
            "role",
            "original_relative_path",
            "archive_relative_path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("invalid-wave archive entry schema mismatch")
        role = artifact["role"]
        if (
            role not in expected_locations
            or role in observed_roles
            or (
                artifact["original_relative_path"],
                artifact["archive_relative_path"],
            )
            != expected_locations[role]
            or type(artifact["size_bytes"]) is not int
            or artifact["size_bytes"] <= 0
        ):
            raise ValueError("invalid-wave archive identity mismatch")
        observed_roles.add(role)
        path = _source_file(
            source_root, artifact["archive_relative_path"], "invalid-wave artifact"
        )
        if (
            not _valid_sha256(artifact["sha256"])
            or _sha256(path) != artifact["sha256"]
            or path.stat().st_size != artifact["size_bytes"]
        ):
            raise ValueError("invalid-wave archived bytes mismatch")
        checked.append(artifact["archive_relative_path"])
    if observed_roles != set(expected_locations):
        raise ValueError("invalid-wave archive roles mismatch")
    return {
        "passed": True,
        "incident_relative_path": INVALID_INCIDENT_RELATIVE_PATH,
        "incident_sha256": _sha256(incident_path),
        "archived_artifacts_checked": checked,
        "outcome_blind": True,
        "development_gate_absent": True,
        "contrasts_uninspected": True,
    }


def _source_record(lock: dict, lock_hash: str) -> dict:
    return {
        "passed": True,
        "runtime": lock["runtime"],
        "source_lock_sha256": lock_hash,
        "checked_source_files": sorted(lock["source_sha256"]),
    }


def _verify_lock(
    lock: dict,
    lock_hash: str,
    manifest_check: dict,
    source_root: Path,
    analysis: ModuleType,
) -> dict:
    if set(lock) != LOCK_KEYS or set(lock) != analysis.LOCK_KEYS:
        raise ValueError("source-lock top-level schema is not closed")
    if lock.get("schema") != analysis.LOCK_SCHEMA:
        raise ValueError("source-lock schema mismatch")
    if lock.get("status") != "sealed_before_any_quick_development_or_confirmation":
        raise ValueError("source-lock status mismatch")
    if lock.get("purpose") != (
        "Canonical pre-execution source/runtime lock for the Acrobot "
        "ProCuRL selection-semantic study."
    ):
        raise ValueError("source-lock purpose mismatch")
    if not _is_utc_iso8601(lock.get("created_utc")):
        raise ValueError("source-lock timestamp invalid")
    if not _valid_sha256(lock_hash):
        raise ValueError("source-lock digest invalid")
    runtime = lock.get("runtime")
    if (
        not isinstance(runtime, dict)
        or {key: runtime.get(key) for key in analysis.PINNED_RUNTIME_VERSIONS}
        != analysis.PINNED_RUNTIME_VERSIONS
    ):
        raise ValueError("source lock does not record the pinned runtime")
    if lock.get("schedule") != analysis._independent_locked_schedule():
        raise ValueError("source-lock schedule mismatch")
    audit = analysis._independent_seed_collision_audit()
    if audit.get("passed") is not True or lock.get("seed_collision_audit") != audit:
        raise ValueError("source-lock seed/RNG audit mismatch")
    if set(lock["source_sha256"]) != set(analysis.EXPECTED_SOURCE_RELATIVE_PATHS):
        raise ValueError("source-lock manifest key set mismatch")
    if manifest_check.get("passed") is not True:
        raise ValueError("source manifest was not verified")
    _verify_v2_audit(lock, source_root, analysis)
    return _source_record(lock, lock_hash)


def _verify_raw_provenance(
    raw: dict,
    lock: dict,
    lock_hash: str,
    analysis: ModuleType,
    *,
    label: str,
) -> None:
    if raw.get("schema") != analysis.RAW_SCHEMA:
        raise ValueError(f"{label} raw schema mismatch")
    if raw.get("artifact_state") != "complete" or raw.get("run_failures") != []:
        raise ValueError(f"{label} raw artifact is incomplete or failed")
    provenance = raw.get("provenance", {})
    if provenance.get("runtime") != lock.get("runtime"):
        raise ValueError(f"{label} recorded runtime differs from lock")
    if provenance.get("source_lock_sha256") != lock_hash:
        raise ValueError(f"{label} source-lock hash differs")
    if provenance.get("source_lock_enforced") is not True:
        raise ValueError(f"{label} did not enforce source lock")
    if provenance.get("source_lock_relative_path") != EXPECTED_LOCK_RELATIVE_PATH:
        raise ValueError(f"{label} source-lock path is not canonical")
    if provenance.get("source_sha256") != lock.get("source_sha256"):
        raise ValueError(f"{label} source manifest differs")
    if provenance.get("seed_collision_audit") != lock.get("seed_collision_audit"):
        raise ValueError(f"{label} seed audit differs")


def _verify_development_binding(
    confirmatory_raw: dict,
    lock: dict,
    lock_hash: str,
    source_root: Path,
    source_record: dict,
    analysis: ModuleType,
) -> dict:
    binding = confirmatory_raw.get("protocol", {}).get("development_gate")
    if not isinstance(binding, dict) or set(binding) != analysis.BINDING_KEYS:
        raise ValueError("confirmatory raw lacks exact development-gate binding")
    if binding.get("all_gates_passed") is not True:
        raise ValueError("bound development gate is not passing")
    gate_path = _source_file(source_root, binding["relative_path"], "development gate")
    development_path = _source_file(
        source_root, binding["raw_artifact_relative_path"], "development raw"
    )
    if _sha256(gate_path) != binding["sha256"]:
        raise ValueError("bound development-gate hash mismatch")
    if _sha256(development_path) != binding["raw_artifact_sha256"]:
        raise ValueError("bound development-raw hash mismatch")
    gate = _load_json(gate_path, "development gate")
    if set(gate) != analysis.GATE_KEYS:
        raise ValueError("development gate field set mismatch")
    development_raw = _load_json(development_path, "development raw")
    _verify_raw_provenance(
        development_raw, lock, lock_hash, analysis, label="development"
    )
    validated = analysis.validate_raw_artifact(development_raw)
    expected = analysis.development_gates(
        validated,
        source_record,
        raw_artifact_relative_path=binding["raw_artifact_relative_path"],
        raw_artifact_sha256=binding["raw_artifact_sha256"],
    )
    if gate != expected:
        raise ValueError("development gate does not exactly recompute")
    return {
        "passed": True,
        "binding_exact": True,
        "development_gate_relative_path": binding["relative_path"],
        "development_gate_sha256": binding["sha256"],
        "development_raw_relative_path": binding["raw_artifact_relative_path"],
        "development_raw_sha256": binding["raw_artifact_sha256"],
        "same_source_lock": True,
        "raw_revalidated": True,
        "gate_recomputed_exactly": True,
        "all_gates_passed": True,
    }


def verify_portable(
    lock_path: Path,
    raw_path: Path,
    stored_analysis_path: Path,
    *,
    source_root: Path | None = None,
) -> dict:
    lock_path = lock_path.resolve()
    raw_path = raw_path.resolve()
    stored_analysis_path = stored_analysis_path.resolve()
    lock = _load_json(lock_path, "source lock")
    raw = _load_json(raw_path, "confirmatory raw")
    stored = _load_json(stored_analysis_path, "stored analysis")
    manifest = lock.get("source_sha256")
    if not isinstance(manifest, dict):
        raise TypeError("source lock lacks a manifest")
    source_root = (
        _discover_source_root(lock_path, manifest)
        if source_root is None
        else source_root.resolve()
    )

    # Crucial order: verify all bytes, and specifically the analyzer, before import.
    manifest_check = _verify_manifest_bytes(lock, source_root)
    runtime_check = _verify_reanalysis_runtime_before_import(lock)
    analyzer_path = _source_file(
        source_root, LOCKED_ANALYZER_RELATIVE_PATH, "locked analyzer"
    )
    analysis = _load_verified_analyzer(
        analyzer_path, manifest[LOCKED_ANALYZER_RELATIVE_PATH]
    )
    lock_hash = _sha256(lock_path)
    source_record = _verify_lock(lock, lock_hash, manifest_check, source_root, analysis)
    invalid_archive_check = _verify_invalid_incident(source_root)
    _verify_raw_provenance(raw, lock, lock_hash, analysis, label="confirmatory")
    validated = analysis.validate_raw_artifact(raw)
    if validated.get("mode") != "confirmatory":
        raise ValueError("raw artifact is not confirmatory mode")
    gate_check = _verify_development_binding(
        raw, lock, lock_hash, source_root, source_record, analysis
    )
    expected_analysis = analysis.confirmatory_analysis(
        validated, source_record, gate_check
    )
    if set(stored) != set(expected_analysis) | REPORT_METADATA_KEYS:
        raise ValueError("stored analysis field set mismatch")
    for key, value in expected_analysis.items():
        if stored.get(key) != value:
            raise ValueError(f"stored analysis reanalysis mismatch: {key}")
    if stored.get("raw_artifact_sha256") != _sha256(raw_path):
        raise ValueError("stored analysis raw-artifact hash mismatch")
    stored_raw_relative = _safe_relative(
        stored.get("raw_artifact_relative_path"), "stored raw path"
    ).as_posix()
    try:
        observed_raw_relative = raw_path.relative_to(source_root).as_posix()
    except ValueError as error:
        raise ValueError("confirmatory raw is outside the source bundle") from error
    if stored_raw_relative != observed_raw_relative:
        raise ValueError("stored analysis raw-artifact relative path mismatch")
    return {
        "schema": SCHEMA,
        "all_checks_passed": True,
        "recorded_execution_runtime": lock["runtime"],
        "source_lock_sha256": lock_hash,
        "source_manifest_verification": {
            **manifest_check,
            "exact_manifest_key_set": True,
            "imported_analyzer_hash_matches_lock": True,
        },
        "live_reanalysis_runtime_verification": runtime_check,
        "invalid_pre_gate_archive_verification": invalid_archive_check,
        "raw_ledger_validation": {
            "passed": True,
            "paired_seed_count": len(validated["seeds"]),
            "arm_count": len(validated["by_case"]),
            "cross_arm_crn_invariants": validated["cross_arm_crn_invariants"],
        },
        "development_gate_binding_verification": gate_check,
        "stored_analysis_comparison": {
            "passed": True,
            "all_recomputed_fields_match": True,
            "stored_analysis_sha256": _sha256(stored_analysis_path),
        },
        "scope": (
            "verifies the replacement frozen bundle, exact pinned live reanalysis "
            "runtime and compensated-sum semantics, the outcome-blind invalid-wave "
            "archive, and deterministic reanalysis; excludes invalid-wave outcomes "
            "from evidence and does not reproduce the training execution"
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--source-root", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = verify_portable(
            args.lock,
            args.raw,
            args.analysis,
            source_root=args.source_root,
        )
    except (ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
