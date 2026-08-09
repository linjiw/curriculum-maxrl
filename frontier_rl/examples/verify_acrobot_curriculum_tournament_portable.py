"""Portable, read-only verifier for a completed Acrobot tournament artifact.

This verifier checks a frozen source bundle, its source lock, the completed raw
ledger, the bound development gate, and a stored confirmatory analysis.  It
then reuses the tournament analyzer's deterministic ledger-validation and
statistical functions to compare a fresh analysis with the stored report.

The check is intentionally location- and platform-independent: the runtime
recorded in the raw artifacts must equal the runtime recorded in the lock, but
the machine running this verifier need not have that runtime or platform
identity.  Passing therefore establishes evidence-bundle integrity and
deterministic reanalysis, not reproduction of the training execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from frontier_rl.examples import analyze_acrobot_curriculum_tournament as analysis


SCHEMA = "curriculum-maxrl/acrobot-curriculum-tournament-portable-verification/v1"
EXPECTED_LOCK_RELATIVE_PATH = (
    "frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json"
)
LOCKED_ANALYZER_RELATIVE_PATH = (
    "frontier_rl/examples/analyze_acrobot_curriculum_tournament.py"
)
STORED_ANALYSIS_METADATA_KEYS = {
    "raw_artifact_path",
    "raw_artifact_sha256",
    "raw_artifact_relative_path",
    "source_lock_path",
    "source_lock_relative_path",
    "source_lock_sha256",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} as strict JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_relative_path(relative: object, label: str) -> PurePosixPath:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a nonempty project-relative path")
    if "\\" in relative:
        raise ValueError(f"{label} is not a canonical POSIX path")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} is not a canonical project-relative path")
    if path.as_posix() != relative:
        raise ValueError(f"{label} is not a canonical POSIX path")
    return path


def _source_file(source_root: Path, relative: object, label: str) -> Path:
    pure = _safe_relative_path(relative, label)
    root = source_root.resolve()
    path = root.joinpath(*pure.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the source root") from error
    if not path.is_file():
        raise ValueError(f"{label} is missing from the source bundle: {relative}")
    return path


def _discover_source_root(lock_path: Path, manifest: dict[str, str]) -> Path:
    candidates: list[Path] = []
    for start in (lock_path.resolve().parent, analysis.PROJECT_ROOT.resolve()):
        for candidate in (start, *start.parents):
            if candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        try:
            if all(
                _source_file(candidate, relative, "locked source").is_file()
                for relative in manifest
            ):
                return candidate
        except ValueError:
            continue
    raise ValueError(
        "could not locate the locked source bundle; pass --source-root explicitly"
    )


def _verify_source_manifest(
    lock: dict[str, Any], source_root: Path
) -> dict[str, Any]:
    manifest = lock.get("source_sha256")
    expected_paths = set(analysis.EXPECTED_SOURCE_RELATIVE_PATHS)
    if not isinstance(manifest, dict) or set(manifest) != expected_paths:
        raise ValueError(
            "source lock does not contain the exact frozen source manifest"
        )
    if LOCKED_ANALYZER_RELATIVE_PATH not in manifest:
        raise ValueError("source manifest omits the analyzer used for reanalysis")
    checked: list[str] = []
    for relative, expected_hash in manifest.items():
        if not _valid_sha256(expected_hash):
            raise ValueError(f"invalid SHA-256 in source manifest: {relative}")
        path = _source_file(source_root, relative, "locked source")
        if _sha256(path) != expected_hash:
            raise ValueError(f"locked source hash mismatch: {relative}")
        checked.append(relative)
    imported_analyzer_path = Path(analysis.__file__).resolve()
    if (
        not imported_analyzer_path.is_file()
        or _sha256(imported_analyzer_path)
        != manifest[LOCKED_ANALYZER_RELATIVE_PATH]
    ):
        raise ValueError("imported analyzer bytes differ from the locked analyzer")
    return {
        "passed": True,
        "source_root_checked": True,
        "checked_source_files": sorted(checked),
        "exact_manifest_key_set": True,
        "all_live_hashes_match": True,
        "imported_analyzer_hash_matches_lock": True,
    }


def _expected_source_lock_record(
    lock: dict[str, Any], lock_sha256: str
) -> dict[str, Any]:
    return {
        "passed": True,
        "runtime": lock["runtime"],
        "source_lock_sha256": lock_sha256,
        "checked_source_files": sorted(lock["source_sha256"]),
    }


def _verify_lock_and_artifact_provenance(
    artifact: dict[str, Any],
    lock: dict[str, Any],
    lock_sha256: str,
    source_manifest: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if lock.get("schema") != analysis.LOCK_SCHEMA:
        raise ValueError("source-lock schema mismatch")
    if not _valid_sha256(lock_sha256):
        raise ValueError("source-lock digest is invalid")
    runtime = lock.get("runtime")
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError("source lock lacks a recorded execution runtime")
    recorded_versions = {
        key: runtime.get(key) for key in analysis.PINNED_RUNTIME_VERSIONS
    }
    if recorded_versions != analysis.PINNED_RUNTIME_VERSIONS:
        raise ValueError("source lock does not record the pinned runtime versions")
    if lock.get("schedule") != analysis._independent_locked_schedule():
        raise ValueError("source-lock schedule does not match the frozen protocol")
    seed_audit = analysis._independent_seed_collision_audit()
    if seed_audit.get("passed") is not True:
        raise ValueError("independent seed/RNG-root collision audit failed")
    if lock.get("seed_collision_audit") != seed_audit:
        raise ValueError("source-lock seed/RNG-root audit does not match the protocol")
    if source_manifest.get("passed") is not True:
        raise ValueError("source manifest was not verified")

    if artifact.get("schema") != analysis.RAW_SCHEMA:
        raise ValueError(f"{label} raw schema mismatch")
    if artifact.get("artifact_state") != "complete" or artifact.get(
        "run_failures"
    ) != []:
        raise ValueError(f"{label} raw artifact is incomplete or contains failed runs")
    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{label} raw artifact lacks provenance")
    if provenance.get("runtime") != runtime:
        raise ValueError(f"{label} recorded runtime differs from the source lock")
    if provenance.get("source_lock_sha256") != lock_sha256:
        raise ValueError(f"{label} raw artifact was created under a different lock")
    if provenance.get("source_lock_enforced") is not True:
        raise ValueError(f"{label} raw artifact does not record lock enforcement")
    if provenance.get("source_lock_relative_path") != EXPECTED_LOCK_RELATIVE_PATH:
        raise ValueError(f"{label} source-lock path provenance is not canonical")
    if provenance.get("source_sha256") != lock.get("source_sha256"):
        raise ValueError(f"{label} source manifest differs from the source lock")
    if provenance.get("seed_collision_audit") != seed_audit:
        raise ValueError(f"{label} seed/RNG-root provenance differs from the lock")
    return _expected_source_lock_record(lock, lock_sha256)


def _verify_development_gate_binding(
    artifact: dict[str, Any],
    lock: dict[str, Any],
    lock_sha256: str,
    source_root: Path,
    source_manifest: dict[str, Any],
    source_lock: dict[str, Any],
) -> dict[str, Any]:
    binding = artifact.get("protocol", {}).get("development_gate")
    expected_binding_keys = {
        "relative_path",
        "sha256",
        "raw_artifact_relative_path",
        "raw_artifact_sha256",
        "all_gates_passed",
    }
    if not isinstance(binding, dict) or set(binding) != expected_binding_keys:
        raise ValueError(
            "confirmatory artifact lacks the exact development-gate binding"
        )
    if binding.get("all_gates_passed") is not True:
        raise ValueError("bound development gate is not passing")
    if not _valid_sha256(binding.get("sha256")) or not _valid_sha256(
        binding.get("raw_artifact_sha256")
    ):
        raise ValueError("development-gate binding contains an invalid digest")

    gate_path = _source_file(
        source_root, binding["relative_path"], "bound development gate"
    )
    if _sha256(gate_path) != binding["sha256"]:
        raise ValueError("bound development-gate hash mismatch")
    gate = _load_json(gate_path, "bound development gate")

    development_raw_path = _source_file(
        source_root,
        binding["raw_artifact_relative_path"],
        "bound development raw artifact",
    )
    if _sha256(development_raw_path) != binding["raw_artifact_sha256"]:
        raise ValueError("bound development raw-artifact hash mismatch")
    development_raw = _load_json(
        development_raw_path, "bound development raw artifact"
    )
    development_source_lock = _verify_lock_and_artifact_provenance(
        development_raw,
        lock,
        lock_sha256,
        source_manifest,
        label="development",
    )
    if development_source_lock != source_lock:
        raise ValueError("development and confirmatory provenance do not bind equally")
    validated = analysis._validate_raw_artifact(development_raw)
    if validated.get("mode") != "development":
        raise ValueError("bound development raw artifact is not development mode")
    expected_gate = analysis.development_gates(validated, development_source_lock)
    expected_gate_keys = set(expected_gate) | STORED_ANALYSIS_METADATA_KEYS
    if set(gate) != expected_gate_keys:
        raise ValueError("bound development gate has an unexpected field set")
    for key, value in expected_gate.items():
        if gate.get(key) != value:
            raise ValueError(
                f"bound development gate does not reanalyze for field: {key}"
            )
    if (
        gate.get("raw_artifact_relative_path")
        != binding["raw_artifact_relative_path"]
        or gate.get("raw_artifact_sha256") != binding["raw_artifact_sha256"]
    ):
        raise ValueError("bound development gate/raw-artifact provenance mismatch")
    if gate.get("source_lock_relative_path") != EXPECTED_LOCK_RELATIVE_PATH:
        raise ValueError("bound development gate source-lock path is not canonical")
    if gate.get("source_lock_sha256") != lock_sha256:
        raise ValueError("bound development gate is tied to a different source lock")
    for key in ("raw_artifact_path", "source_lock_path"):
        if not isinstance(gate.get(key), str) or not gate[key]:
            raise ValueError(f"bound development gate lacks path metadata: {key}")
    return {
        "passed": True,
        "development_gate_relative_path": binding["relative_path"],
        "development_gate_sha256": binding["sha256"],
        "development_raw_relative_path": binding["raw_artifact_relative_path"],
        "development_raw_sha256": binding["raw_artifact_sha256"],
        "gates_recomputed_from_raw": True,
    }


def _compare_stored_analysis(
    stored: dict[str, Any],
    expected: dict[str, Any],
    *,
    raw_sha256: str,
    lock_sha256: str,
) -> dict[str, Any]:
    expected_keys = set(expected) | STORED_ANALYSIS_METADATA_KEYS
    if set(stored) != expected_keys:
        missing = sorted(expected_keys - set(stored))
        extra = sorted(set(stored) - expected_keys)
        raise ValueError(
            f"stored analysis field set mismatch; missing={missing!r}, extra={extra!r}"
        )
    for key, value in expected.items():
        if stored.get(key) != value:
            raise ValueError(
                f"stored analysis does not match reanalysis for field: {key}"
            )
    if stored.get("raw_artifact_sha256") != raw_sha256:
        raise ValueError("stored analysis is bound to a different raw artifact")
    if stored.get("source_lock_sha256") != lock_sha256:
        raise ValueError("stored analysis is bound to a different source lock")
    if stored.get("source_lock_relative_path") != EXPECTED_LOCK_RELATIVE_PATH:
        raise ValueError("stored analysis source-lock path is not canonical")
    for key in ("raw_artifact_path", "source_lock_path"):
        if not isinstance(stored.get(key), str) or not stored[key]:
            raise ValueError(f"stored analysis lacks recorded path metadata: {key}")
    raw_relative = stored.get("raw_artifact_relative_path")
    if raw_relative is not None:
        _safe_relative_path(raw_relative, "stored raw-artifact relative path")
    return {
        "passed": True,
        "all_recomputed_fields_match": True,
        "matched_top_level_fields": sorted(expected),
        "raw_artifact_sha256_matches": True,
        "source_lock_sha256_matches": True,
    }


def verify_portable(
    lock_path: Path,
    raw_path: Path,
    stored_analysis_path: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Verify one completed confirmatory evidence bundle without modifying it."""
    lock_path = lock_path.resolve()
    raw_path = raw_path.resolve()
    stored_analysis_path = stored_analysis_path.resolve()
    lock = _load_json(lock_path, "source lock")
    raw = _load_json(raw_path, "completed raw artifact")
    stored = _load_json(stored_analysis_path, "stored analysis")
    lock_sha256 = _sha256(lock_path)
    raw_sha256 = _sha256(raw_path)
    stored_analysis_sha256 = _sha256(stored_analysis_path)

    manifest = lock.get("source_sha256")
    if not isinstance(manifest, dict):
        raise ValueError("source lock lacks a source manifest")
    if source_root is None:
        source_root = _discover_source_root(lock_path, manifest)
    source_manifest = _verify_source_manifest(lock, source_root)
    source_lock = _verify_lock_and_artifact_provenance(
        raw,
        lock,
        lock_sha256,
        source_manifest,
        label="confirmatory",
    )
    validated = analysis._validate_raw_artifact(raw)
    if validated.get("mode") != "confirmatory":
        raise ValueError("portable verification requires a confirmatory raw artifact")
    gate_verification = _verify_development_gate_binding(
        raw,
        lock,
        lock_sha256,
        source_root.resolve(),
        source_manifest,
        source_lock,
    )
    expected_report = analysis.confirmatory_analysis(validated, source_lock)
    expected_report["development_gate_verification"] = gate_verification
    comparison = _compare_stored_analysis(
        stored,
        expected_report,
        raw_sha256=raw_sha256,
        lock_sha256=lock_sha256,
    )

    return {
        "schema": SCHEMA,
        "all_checks_passed": True,
        "scope": (
            "Evidence-bundle integrity and deterministic reanalysis passed; "
            "this does not reproduce the training execution."
        ),
        "input_sha256": {
            "source_lock": lock_sha256,
            "completed_raw_artifact": raw_sha256,
            "stored_analysis": stored_analysis_sha256,
        },
        "recorded_execution_runtime": lock["runtime"],
        "verifier_runtime_not_used_as_an_execution_identity_check": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "portable_verifier_source_sha256_unlocked": _sha256(
            Path(__file__).resolve()
        ),
        "source_manifest_verification": source_manifest,
        "raw_ledger_validation": {
            "passed": True,
            "mode": validated["mode"],
            "paired_seed_count": len(validated["seeds"]),
            "all_registered_arms_checked": True,
        },
        "development_gate_binding_verification": gate_verification,
        "stored_analysis_comparison": comparison,
        "limitations": [
            (
                "This verifies retained JSON ledgers and locked source bytes; it "
                "does not rerun the environment, optimizer, or training process."
            ),
            (
                "It reuses pure validation and statistical functions from the "
                "locked analyzer, so it is not a second implementation of those "
                "algorithms."
            ),
            (
                "This portable verifier was added after the execution lock and is "
                "not itself in that pre-execution source manifest; its own digest "
                "is reported for separate review."
            ),
            (
                "Successful reanalysis requires dependencies compatible with the "
                "locked analyzer's deterministic NumPy operations, even though "
                "the live OS, machine, and version strings are not equality gates."
            ),
            (
                "Hash agreement cannot establish that unrecorded evaluation "
                "trajectories or external machine state were faithfully captured."
            ),
            (
                "The hashes establish consistency relative to the supplied lock; "
                "authenticity and pre-execution timing still require a separately "
                "trusted copy of the lock digest or repository history."
            ),
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("stored_analysis", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="root of the source bundle; auto-detected when omitted",
    )
    args = parser.parse_args(argv)
    try:
        result = verify_portable(
            args.lock,
            args.raw,
            args.stored_analysis,
            source_root=args.source_root,
        )
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
