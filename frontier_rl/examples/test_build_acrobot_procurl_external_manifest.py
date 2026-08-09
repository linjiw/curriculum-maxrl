"""Synthetic/adversarial tests for the external confirmatory-raw manifest."""

from __future__ import annotations

import copy
import ast
import hashlib
import json
import os
import py_compile
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("gymnasium")

from frontier_rl.examples import analyze_acrobot_procurl_selection as analysis
from frontier_rl.examples import build_acrobot_procurl_external_manifest as manifest


RAW_LOGICAL = "external/acrobot-procurl-selection-confirmatory.json"
LOGICAL_PATHS = {
    "source_lock": "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_LOCK.json",
    "development_gate": "frontier_rl/examples/synthetic-development-gate.json",
    "confirmatory_analysis": "frontier_rl/examples/synthetic-analysis.json",
    "portable_verification": "frontier_rl/examples/synthetic-portable.json",
}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _swap_after_first_read(
    monkeypatch, target: Path, replacement: bytes
) -> tuple[dict[str, int], object]:
    """Swap a file after returning its first buffer; forbid a second read."""
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    target_absolute = target.absolute()
    state = {"reads": 0, "legacy_text_reads": 0}

    def racing_read_text(path: Path, *args, **kwargs) -> str:
        text = original_read_text(path, *args, **kwargs)
        if path.absolute() == target_absolute:
            state["legacy_text_reads"] += 1
            target.write_bytes(replacement)
        return text

    def racing_read_bytes(path: Path) -> bytes:
        data = original_read_bytes(path)
        if path.absolute() == target_absolute:
            state["reads"] += 1
            if state["reads"] > 1:
                raise AssertionError(f"artifact was read more than once: {target}")
            target.write_bytes(replacement)
        return data

    monkeypatch.setattr(Path, "read_text", racing_read_text)
    monkeypatch.setattr(Path, "read_bytes", racing_read_bytes)
    return state, original_read_bytes


def _synthetic_validated() -> dict:
    return {
        "strict_valid": True,
        "mode": "confirmatory",
        "seeds": list(analysis.CONFIRMATORY_SEEDS),
        "by_case": {arm: [] for arm in analysis.ARM_NAMES},
    }


def _raw(
    lock_logical: str,
    lock_sha: str,
    gate_logical: str,
    gate_sha: str,
    lock: dict,
) -> dict:
    return {
        "schema": analysis.RAW_SCHEMA,
        "artifact_state": "complete",
        "provenance": {
            "source_lock_relative_path": lock_logical,
            "source_lock_sha256": lock_sha,
            "source_lock_enforced": True,
            "runtime": lock["runtime"],
            "source_sha256": lock["source_sha256"],
            "seed_collision_audit": lock["seed_collision_audit"],
        },
        "protocol": {
            "mode": "confirmatory",
            "development_gate": {
                "relative_path": gate_logical,
                "sha256": gate_sha,
                "raw_artifact_relative_path": "evidence/synthetic-development.json",
                "raw_artifact_sha256": "d" * 64,
                "all_gates_passed": True,
            },
        },
        "run_failures": [],
        "cases": {
            arm: {
                "runs": [
                    {"seed": seed, "synthetic_arm": arm, "synthetic_value": seed - 21_000}
                    for seed in analysis.CONFIRMATORY_SEEDS
                ]
            }
            for arm in analysis.ARM_NAMES
        },
    }


def _development_verification(gate_logical: str, gate_sha: str) -> dict:
    return {
        "passed": True,
        "binding_exact": True,
        "development_gate_relative_path": gate_logical,
        "development_gate_sha256": gate_sha,
        "development_raw_relative_path": "evidence/synthetic-development.json",
        "development_raw_sha256": "d" * 64,
        "same_source_lock": True,
        "raw_revalidated": True,
        "gate_recomputed_exactly": True,
        "all_gates_passed": True,
    }


@pytest.fixture
def synthetic_bundle(tmp_path: Path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    analyzer_copy = root / manifest.LOCKED_ANALYZER_LOGICAL_PATH
    analyzer_copy.parent.mkdir(parents=True)
    shutil.copyfile(Path(analysis.__file__), analyzer_copy)
    source_sha = {relative: "a" * 64 for relative in analysis.EXPECTED_SOURCE_RELATIVE_PATHS}
    source_sha[manifest.LOCKED_ANALYZER_LOGICAL_PATH] = _sha256(analyzer_copy)
    runtime = {
        **manifest.PINNED_REANALYSIS_RUNTIME,
        "platform": "synthetic",
        "machine": "synthetic",
    }
    lock = {
        "schema": analysis.LOCK_SCHEMA,
        "status": "sealed_before_any_quick_development_or_confirmation",
        "created_utc": "2026-08-09T00:00:00+00:00",
        "purpose": (
            "Canonical pre-execution source/runtime lock for the Acrobot "
            "ProCuRL selection-semantic study."
        ),
        "runtime": runtime,
        "schedule": manifest.EXPECTED_LOCK_SCHEDULE,
        "seed_collision_audit": {"passed": True},
        "source_sha256": source_sha,
        "v2_dependency_audit": {},
    }
    lock_path = root / LOGICAL_PATHS["source_lock"]
    _write_json(lock_path, lock)
    lock_sha = _sha256(lock_path)

    gate = {
        "schema": analysis.GATE_SCHEMA,
        "mode": "development",
        "all_gates_passed": True,
        "source_lock_sha256": lock_sha,
        "source_lock_verification": {},
        "gates": {},
        "diagnostics": {},
        "gate_policy": {},
        "raw_artifact_relative_path": "evidence/synthetic-development.json",
        "raw_artifact_sha256": "d" * 64,
    }
    gate_path = root / LOGICAL_PATHS["development_gate"]
    _write_json(gate_path, gate)
    gate_sha = _sha256(gate_path)

    raw_path = tmp_path / "external" / "confirmatory.json"
    _write_json(
        raw_path,
        _raw(
            LOGICAL_PATHS["source_lock"],
            lock_sha,
            LOGICAL_PATHS["development_gate"],
            gate_sha,
            lock,
        ),
    )
    raw_sha = _sha256(raw_path)
    development = _development_verification(
        LOGICAL_PATHS["development_gate"], gate_sha
    )
    stored_analysis = {
        "schema": analysis.ANALYSIS_SCHEMA,
        "mode": "confirmatory",
        "strict_validation_passed": True,
        "source_lock_verification": {
            "passed": True,
            "runtime": runtime,
            "source_lock_sha256": lock_sha,
            "checked_source_files": sorted(source_sha),
        },
        "development_gate_binding_verification": development,
        "primary": {},
        "secondary_holm_family": {},
        "secondary_multiplicity": {},
        "arm_descriptives": {},
        "statistical_conventions": {},
        "raw_artifact_relative_path": RAW_LOGICAL,
        "raw_artifact_sha256": raw_sha,
    }
    analysis_path = root / LOGICAL_PATHS["confirmatory_analysis"]
    _write_json(analysis_path, stored_analysis)
    analysis_sha = _sha256(analysis_path)
    portable = {
        "schema": manifest.PORTABLE_SCHEMA,
        "all_checks_passed": True,
        "recorded_execution_runtime": runtime,
        "source_lock_sha256": lock_sha,
        "source_manifest_verification": {
            "passed": True,
            "checked_source_files": sorted(source_sha),
            "all_live_hashes_match": True,
            "analyzer_hashed_before_import": True,
            "exact_manifest_key_set": True,
            "imported_analyzer_hash_matches_lock": True,
        },
        "live_reanalysis_runtime_verification": {
            "passed": True,
            "checked_before_analyzer_import": True,
            "recorded_runtime": manifest.PINNED_REANALYSIS_RUNTIME,
            "live_runtime": manifest.PINNED_REANALYSIS_RUNTIME,
            "entropy_sum_sentinel_hex": manifest.EXPECTED_ENTROPY_SUM_SENTINEL_HEX,
            "expected_entropy_sum_sentinel_hex": (
                manifest.EXPECTED_ENTROPY_SUM_SENTINEL_HEX
            ),
            "known_naive_entropy_sum_sentinel_hex": (
                manifest.NAIVE_ENTROPY_SUM_SENTINEL_HEX
            ),
        },
        "invalid_pre_gate_archive_verification": {
            "passed": True,
            "incident_relative_path": manifest.INCIDENT_LOGICAL_PATH,
            "incident_sha256": source_sha[manifest.INCIDENT_LOGICAL_PATH],
            "archived_artifacts_checked": manifest.EXPECTED_INVALID_ARCHIVE_PATHS,
            "outcome_blind": True,
            "development_gate_absent": True,
            "contrasts_uninspected": True,
        },
        "raw_ledger_validation": {
            "passed": True,
            "paired_seed_count": 80,
            "arm_count": 4,
            "cross_arm_crn_invariants": {
                "passed": True,
                "paired_seeds_checked": list(manifest.CONFIRMATORY_SEEDS),
                "selection_rng_stream_replayed": True,
                "student_reset_stream_paired": True,
                "probe_coordinates_paired": True,
                "uniform_mechanics_paired_on_overlap": True,
                "same_actor_evaluations_paired": True,
            },
        },
        "development_gate_binding_verification": development,
        "stored_analysis_comparison": {
            "passed": True,
            "all_recomputed_fields_match": True,
            "stored_analysis_sha256": analysis_sha,
        },
        "scope": manifest.EXPECTED_PORTABLE_SCOPE,
    }
    portable_path = root / LOGICAL_PATHS["portable_verification"]
    _write_json(portable_path, portable)
    bound_paths = {
        "source_lock": lock_path,
        "development_gate": gate_path,
        "confirmatory_analysis": analysis_path,
        "portable_verification": portable_path,
    }
    fake_analyzer = SimpleNamespace(
        validate_raw_artifact=lambda raw: _synthetic_validated()
    )
    monkeypatch.setattr(
        manifest, "_load_verified_analyzer", lambda path, digest: fake_analyzer
    )
    built = manifest.build_manifest(
        raw_path=raw_path,
        raw_logical_path=RAW_LOGICAL,
        bound_paths=bound_paths,
        bound_logical_paths=LOGICAL_PATHS,
        analyzer_path=analyzer_copy,
    )
    manifest_path = root / "evidence" / "external-manifest.json"
    manifest.write_json(manifest_path, built)
    return {
        "root": root,
        "raw": raw_path,
        "manifest": manifest_path,
        "payload": built,
        "bound_paths": bound_paths,
        "analyzer": analyzer_copy,
    }


def test_build_is_deterministic_and_compact_and_full_verify(synthetic_bundle):
    fixture = synthetic_bundle
    rebuilt = manifest.build_manifest(
        raw_path=fixture["raw"],
        raw_logical_path=RAW_LOGICAL,
        bound_paths=fixture["bound_paths"],
        bound_logical_paths=LOGICAL_PATHS,
        analyzer_path=fixture["analyzer"],
    )
    assert rebuilt == fixture["payload"]
    assert len(rebuilt["run_index"]) == 320
    assert rebuilt["run_index"][0]["arm"] == analysis.ARM_NAMES[0]
    assert rebuilt["run_index"][-1]["seed"] == analysis.CONFIRMATORY_SEEDS[-1]
    serialized = json.dumps(rebuilt, allow_nan=False)
    assert str(fixture["raw"].parent) not in serialized
    compact = manifest.verify_manifest(
        fixture["manifest"], artifact_root=fixture["root"], mode="compact"
    )
    full = manifest.verify_manifest(
        fixture["manifest"],
        artifact_root=fixture["root"],
        mode="full",
        raw_path=fixture["raw"],
    )
    assert compact["raw_bytes_verified"] is False
    assert compact["all_run_index_records_reconciled"] is False
    assert full["raw_bytes_verified"] is True
    assert full["all_run_index_records_reconciled"] is True


@pytest.mark.parametrize("target_role", ["raw", *manifest.ROLE_ORDER])
def test_builder_binds_and_validates_each_exact_single_artifact_capture(
    synthetic_bundle, monkeypatch, target_role
):
    fixture = synthetic_bundle
    target = (
        fixture["raw"]
        if target_role == "raw"
        else fixture["bound_paths"][target_role]
    )
    original = target.read_bytes()
    replacement = b'{"schema":"swapped","schema":"invalid"}\n'
    state, original_read_bytes = _swap_after_first_read(
        monkeypatch, target, replacement
    )
    rebuilt = manifest.build_manifest(
        raw_path=fixture["raw"],
        raw_logical_path=RAW_LOGICAL,
        bound_paths=fixture["bound_paths"],
        bound_logical_paths=LOGICAL_PATHS,
        analyzer_path=fixture["analyzer"],
    )
    binding = (
        rebuilt["raw_artifact"]
        if target_role == "raw"
        else rebuilt["bindings"][target_role]
    )
    assert state["reads"] == 1
    assert state["legacy_text_reads"] == 0
    assert binding["size_bytes"] == len(original)
    assert binding["sha256"] == hashlib.sha256(original).hexdigest()
    assert original_read_bytes(target) == replacement


@pytest.mark.parametrize("target_role", ["raw", *manifest.ROLE_ORDER])
def test_verifier_uses_each_exact_single_artifact_capture(
    synthetic_bundle, monkeypatch, target_role
):
    fixture = synthetic_bundle
    target = (
        fixture["raw"]
        if target_role == "raw"
        else fixture["bound_paths"][target_role]
    )
    replacement = b'{"schema":"swapped","schema":"invalid"}\n'
    state, original_read_bytes = _swap_after_first_read(
        monkeypatch, target, replacement
    )
    result = manifest.verify_manifest(
        fixture["manifest"],
        artifact_root=fixture["root"],
        mode="full" if target_role == "raw" else "compact",
        raw_path=fixture["raw"] if target_role == "raw" else None,
    )
    assert result["all_checks_passed"] is True
    assert state["reads"] == 1
    assert state["legacy_text_reads"] == 0
    assert original_read_bytes(target) == replacement


def test_compact_mode_does_not_require_or_read_raw(synthetic_bundle, monkeypatch):
    fixture = synthetic_bundle
    real_loader = manifest.load_strict_json

    def guarded(path: Path, label: str):
        if "raw artifact" in label:
            raise AssertionError("compact verification attempted to read raw")
        return real_loader(path, label)

    monkeypatch.setattr(manifest, "load_strict_json", guarded)
    result = manifest.verify_manifest(
        fixture["manifest"], artifact_root=fixture["root"], mode="compact"
    )
    assert result["all_checks_passed"] is True


def test_full_mode_rejects_raw_byte_or_index_tampering(synthetic_bundle):
    fixture = synthetic_bundle
    raw = json.loads(fixture["raw"].read_text(encoding="utf-8"))
    raw["cases"][analysis.ARM_NAMES[0]]["runs"][0]["synthetic_value"] = 999
    _write_json(fixture["raw"], raw)
    with pytest.raises(ValueError, match="byte count mismatch|SHA-256 mismatch"):
        manifest.verify_manifest(
            fixture["manifest"],
            artifact_root=fixture["root"],
            mode="full",
            raw_path=fixture["raw"],
        )


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        (lambda value: value["run_index"].append(copy.deepcopy(value["run_index"][-1])), "length"),
        (lambda value: value["run_index"][0].__setitem__("seed", 99), "seed mismatch"),
        (lambda value: value["run_index"][0].__setitem__("seed", 21_000.0), "seed mismatch"),
        (lambda value: value["run_index"][0].__setitem__("ordinal", False), "ordinal mismatch"),
        (lambda value: value["run_index"][0].__setitem__("extra", True), "field set"),
        (lambda value: value["schedule"].__setitem__("run_count", 320.0), "count mismatch"),
        (lambda value: value["schedule"]["seeds"].__setitem__(0, 21_000.0), "seeds mismatch"),
        (lambda value: value["raw_artifact"].__setitem__("size_bytes", True), "size is invalid"),
        (lambda value: value["schedule"].__setitem__("extra", True), "field set"),
    ],
)
def test_compact_manifest_rejects_extra_or_forged_index(
    synthetic_bundle, mutation, pattern
):
    forged = copy.deepcopy(synthetic_bundle["payload"])
    mutation(forged)
    with pytest.raises(ValueError, match=pattern):
        manifest.validate_manifest_shape(forged)


@pytest.mark.parametrize(
    "logical",
    [
        "/Users/example/raw.json",
        "C:/Users/example/raw.json",
        "../raw.json",
        "file://host/raw.json",
        "folder\\raw.json",
        "evidence/CON",
        "evidence/NUL.json",
        "evidence/CON .txt",
        "evidence/name.",
        "evidence/name ",
        "evidence/name\n.json",
        "evidence/name<.json",
        "evidence/name>.json",
        'evidence/name".json',
        "evidence/name|.json",
        "evidence/name?.json",
        "evidence/name*.json",
    ],
)
def test_local_or_nonportable_logical_paths_fail_closed(synthetic_bundle, logical):
    forged = copy.deepcopy(synthetic_bundle["payload"])
    forged["raw_artifact"]["logical_path"] = logical
    with pytest.raises(
        ValueError,
        match=(
            "portable|relative|normalized|control|trailing|reserved|"
            "Windows-invalid"
        ),
    ):
        manifest.validate_manifest_shape(forged)


def test_strict_manifest_loader_rejects_duplicate_and_nonfinite(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        manifest.load_strict_json(duplicate, "manifest")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":1e999}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        manifest.load_strict_json(nonfinite, "manifest")


def test_compact_rejects_bound_extra_field_even_with_updated_receipt(synthetic_bundle):
    fixture = synthetic_bundle
    gate_path = fixture["bound_paths"]["development_gate"]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["unregistered_extra"] = True
    _write_json(gate_path, gate)
    forged = copy.deepcopy(fixture["payload"])
    binding = forged["bindings"]["development_gate"]
    binding["size_bytes"] = gate_path.stat().st_size
    binding["sha256"] = _sha256(gate_path)
    forged_path = fixture["root"] / "evidence" / "forged-manifest.json"
    manifest.write_json(forged_path, forged)
    with pytest.raises(ValueError, match="field set mismatch"):
        manifest.verify_manifest(
            forged_path, artifact_root=fixture["root"], mode="compact"
        )


def test_full_requires_raw_and_compact_forbids_raw(synthetic_bundle):
    fixture = synthetic_bundle
    with pytest.raises(ValueError, match="requires"):
        manifest.verify_manifest(
            fixture["manifest"], artifact_root=fixture["root"], mode="full"
        )
    with pytest.raises(ValueError, match="must not read"):
        manifest.verify_manifest(
            fixture["manifest"],
            artifact_root=fixture["root"],
            mode="compact",
            raw_path=fixture["raw"],
        )


def _write_forged_bound_payload(fixture, role: str, payload: dict, suffix: str) -> Path:
    path = fixture["bound_paths"][role]
    _write_json(path, payload)
    forged = copy.deepcopy(fixture["payload"])
    forged["bindings"][role]["size_bytes"] = path.stat().st_size
    forged["bindings"][role]["sha256"] = _sha256(path)
    forged_path = fixture["root"] / "evidence" / f"forged-{suffix}.json"
    manifest.write_json(forged_path, forged)
    return forged_path


DEVELOPMENT_TYPE_ALIASES = (
    ("passed", 1),
    ("binding_exact", 1.0),
    ("development_gate_relative_path", True),
    ("development_gate_sha256", 1),
    ("development_raw_relative_path", 1.0),
    ("development_raw_sha256", False),
    ("same_source_lock", 1),
    ("raw_revalidated", 1.0),
    ("gate_recomputed_exactly", "true"),
    ("all_gates_passed", 1),
)


@pytest.mark.parametrize("field,alias", DEVELOPMENT_TYPE_ALIASES)
def test_compact_rejects_type_alias_in_each_portable_development_field(
    synthetic_bundle, field, alias
):
    fixture = synthetic_bundle
    portable_path = fixture["bound_paths"]["portable_verification"]
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    portable["development_gate_binding_verification"][field] = alias
    forged_path = _write_forged_bound_payload(
        fixture,
        "portable_verification",
        portable,
        f"portable-development-{field}",
    )
    with pytest.raises(ValueError, match="portable development-gate binding"):
        manifest.verify_manifest(
            forged_path,
            artifact_root=fixture["root"],
            mode="compact",
        )


@pytest.mark.parametrize("field,alias", DEVELOPMENT_TYPE_ALIASES)
def test_raw_preimport_rejects_type_alias_in_each_portable_development_field(
    synthetic_bundle, field, alias
):
    fixture = synthetic_bundle
    portable_path = fixture["bound_paths"]["portable_verification"]
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    portable["development_gate_binding_verification"][field] = alias
    _write_json(portable_path, portable)
    raw_capture = manifest._capture_json(fixture["raw"], "external raw artifact")
    bound_artifacts = {
        role: manifest._capture_json(path, role.replace("_", " "))
        for role, path in fixture["bound_paths"].items()
    }
    with pytest.raises(ValueError, match="stored and portable development bindings"):
        manifest._verify_raw_before_import(
            fixture["payload"],
            raw_capture,
            bound_artifacts=bound_artifacts,
        )


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        (
            lambda value: value["source_manifest_verification"].__setitem__(
                "all_live_hashes_match", False
            ),
            "source-manifest",
        ),
        (
            lambda value: value["source_manifest_verification"].__setitem__(
                "extra", True
            ),
            "field set",
        ),
        (
            lambda value: value["live_reanalysis_runtime_verification"][
                "live_runtime"
            ].__setitem__("numpy", "0.0"),
            "live-runtime",
        ),
        (
            lambda value: value["invalid_pre_gate_archive_verification"].__setitem__(
                "outcome_blind", False
            ),
            "invalid-archive",
        ),
        (
            lambda value: value["raw_ledger_validation"][
                "cross_arm_crn_invariants"
            ].__setitem__("same_actor_evaluations_paired", False),
            "cross-arm CRN",
        ),
        (
            lambda value: value["raw_ledger_validation"][
                "cross_arm_crn_invariants"
            ].__setitem__("extra", True),
            "field set",
        ),
        (
            lambda value: value["raw_ledger_validation"].__setitem__(
                "paired_seed_count", 80.0
            ),
            "count mismatch",
        ),
        (
            lambda value: value["raw_ledger_validation"].__setitem__(
                "arm_count", True
            ),
            "count mismatch",
        ),
        (
            lambda value: value["raw_ledger_validation"][
                "cross_arm_crn_invariants"
            ]["paired_seeds_checked"].__setitem__(0, 21_000.0),
            "cross-arm CRN",
        ),
    ],
)
def test_compact_deeply_rejects_forged_portable_checks(
    synthetic_bundle, mutation, pattern
):
    fixture = synthetic_bundle
    portable_path = fixture["bound_paths"]["portable_verification"]
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    mutation(portable)
    forged_path = _write_forged_bound_payload(
        fixture, "portable_verification", portable, "portable"
    )
    with pytest.raises(ValueError, match=pattern):
        manifest.verify_manifest(
            forged_path, artifact_root=fixture["root"], mode="compact"
        )


@pytest.mark.parametrize("failure", ["runtime", "sentinel"])
def test_runtime_and_entropy_checks_precede_analyzer_import(
    synthetic_bundle, monkeypatch, failure
):
    called = False

    def forbidden_loader(path, digest):
        nonlocal called
        called = True
        raise AssertionError("analyzer import occurred before trust checks")

    monkeypatch.setattr(manifest, "_load_verified_analyzer", forbidden_loader)
    if failure == "runtime":
        monkeypatch.setattr(
            manifest,
            "_live_reanalysis_runtime",
            lambda: {**manifest.PINNED_REANALYSIS_RUNTIME, "numpy": "0.0"},
        )
        pattern = "live runtime"
    else:
        monkeypatch.setattr(
            manifest, "EXPECTED_ENTROPY_SUM_SENTINEL_HEX", "0x0.0p+0"
        )
        pattern = "compensated-sum sentinel"
    with pytest.raises(ValueError, match=pattern):
        manifest.build_manifest(
            raw_path=synthetic_bundle["raw"],
            raw_logical_path=RAW_LOGICAL,
            bound_paths=synthetic_bundle["bound_paths"],
            bound_logical_paths=LOGICAL_PATHS,
            analyzer_path=synthetic_bundle["analyzer"],
        )
    assert called is False


def test_full_raw_index_rejects_float_seed_identity(synthetic_bundle):
    raw = json.loads(synthetic_bundle["raw"].read_text(encoding="utf-8"))
    raw["cases"][manifest.ARM_NAMES[0]]["runs"][0]["seed"] = 21_000.0
    with pytest.raises(ValueError, match="missing, duplicate, reordered, or extra"):
        manifest._raw_run_index(raw)


def test_production_builder_has_no_static_analyzer_import():
    tree = ast.parse(Path(manifest.__file__).read_text(encoding="utf-8"))
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any(
        "analyze_acrobot_procurl_selection" in ast.unparse(node)
        for node in imports
    )


def test_verified_source_buffer_ignores_timestamp_valid_malicious_pyc(
    tmp_path: Path,
):
    source_path = tmp_path / "cached_analyzer.py"
    malicious = b'VALUE = "evil"\n'
    verified = b'VALUE = "safe"\n'
    assert len(malicious) == len(verified)
    source_path.write_bytes(malicious)
    pyc_path = Path(
        py_compile.compile(
            str(source_path),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
        )
    )
    timestamp = source_path.stat()
    source_path.write_bytes(verified)
    os.utime(
        source_path,
        ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns),
    )
    cached = pyc_path.read_bytes()
    assert int.from_bytes(cached[4:8], "little") == 0
    assert int.from_bytes(cached[8:12], "little") == int(timestamp.st_mtime)
    assert int.from_bytes(cached[12:16], "little") == len(verified)

    module = manifest._load_verified_analyzer(
        source_path, hashlib.sha256(verified).hexdigest()
    )
    assert module.VALUE == "safe"


def test_verified_source_buffer_closes_hash_to_exec_source_swap(
    tmp_path: Path, monkeypatch
):
    source_path = tmp_path / "racing_analyzer.py"
    verified = b'VALUE = "safe"\n'
    replacement = b'VALUE = "evil"\n'
    source_path.write_bytes(verified)
    expected = hashlib.sha256(verified).hexdigest()
    original_read_bytes = Path.read_bytes
    reads = 0

    def racing_read_bytes(path: Path) -> bytes:
        nonlocal reads
        data = original_read_bytes(path)
        if path.resolve() == source_path.resolve():
            reads += 1
            if reads == 1:
                source_path.write_bytes(replacement)
        return data

    monkeypatch.setattr(Path, "read_bytes", racing_read_bytes)
    module = manifest._load_verified_analyzer(source_path, expected)
    assert reads == 1
    assert module.VALUE == "safe"
    assert original_read_bytes(source_path) == replacement


def test_full_rejects_coherently_receipted_different_development_raw_before_import(
    synthetic_bundle, monkeypatch
):
    fixture = synthetic_bundle
    raw = json.loads(fixture["raw"].read_text(encoding="utf-8"))
    development = raw["protocol"]["development_gate"]
    development["raw_artifact_relative_path"] = (
        "evidence/different-development.json"
    )
    development["raw_artifact_sha256"] = "e" * 64
    _write_json(fixture["raw"], raw)

    forged = copy.deepcopy(fixture["payload"])
    forged["raw_artifact"]["size_bytes"] = fixture["raw"].stat().st_size
    forged["raw_artifact"]["sha256"] = _sha256(fixture["raw"])

    stored_path = fixture["bound_paths"]["confirmatory_analysis"]
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    stored["raw_artifact_sha256"] = forged["raw_artifact"]["sha256"]
    _write_json(stored_path, stored)
    forged["bindings"]["confirmatory_analysis"]["size_bytes"] = (
        stored_path.stat().st_size
    )
    forged["bindings"]["confirmatory_analysis"]["sha256"] = _sha256(stored_path)

    portable_path = fixture["bound_paths"]["portable_verification"]
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    portable["stored_analysis_comparison"]["stored_analysis_sha256"] = _sha256(
        stored_path
    )
    _write_json(portable_path, portable)
    forged["bindings"]["portable_verification"]["size_bytes"] = (
        portable_path.stat().st_size
    )
    forged["bindings"]["portable_verification"]["sha256"] = _sha256(
        portable_path
    )

    forged_path = fixture["root"] / "evidence" / "coherent-different-dev.json"
    manifest.write_json(forged_path, forged)
    imported = False

    def forbidden_loader(path, digest):
        nonlocal imported
        imported = True
        raise AssertionError("analyzer imported before development binding rejection")

    monkeypatch.setattr(manifest, "_load_verified_analyzer", forbidden_loader)
    with pytest.raises(ValueError, match="development-raw binding mismatch"):
        manifest.verify_manifest(
            forged_path,
            artifact_root=fixture["root"],
            mode="full",
            raw_path=fixture["raw"],
        )
    assert imported is False


def _refresh_coherent_raw_receipt_chain(fixture, raw: dict, suffix: str) -> Path:
    _write_json(fixture["raw"], raw)
    forged = copy.deepcopy(fixture["payload"])
    forged["raw_artifact"]["size_bytes"] = fixture["raw"].stat().st_size
    forged["raw_artifact"]["sha256"] = _sha256(fixture["raw"])
    stored_path = fixture["bound_paths"]["confirmatory_analysis"]
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    stored["raw_artifact_sha256"] = forged["raw_artifact"]["sha256"]
    _write_json(stored_path, stored)
    forged["bindings"]["confirmatory_analysis"]["size_bytes"] = (
        stored_path.stat().st_size
    )
    forged["bindings"]["confirmatory_analysis"]["sha256"] = _sha256(stored_path)
    portable_path = fixture["bound_paths"]["portable_verification"]
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    portable["stored_analysis_comparison"]["stored_analysis_sha256"] = _sha256(
        stored_path
    )
    _write_json(portable_path, portable)
    forged["bindings"]["portable_verification"]["size_bytes"] = (
        portable_path.stat().st_size
    )
    forged["bindings"]["portable_verification"]["sha256"] = _sha256(
        portable_path
    )
    forged_path = fixture["root"] / "evidence" / f"coherent-{suffix}.json"
    manifest.write_json(forged_path, forged)
    return forged_path


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        (
            lambda raw: raw["provenance"]["runtime"].__setitem__("numpy", "0.0"),
            "provenance does not exactly match",
        ),
        (
            lambda raw: raw["provenance"]["source_sha256"].__setitem__(
                manifest.LOCKED_ANALYZER_LOGICAL_PATH, "f" * 64
            ),
            "provenance does not exactly match",
        ),
        (
            lambda raw: raw["provenance"]["seed_collision_audit"].__setitem__(
                "tampered", True
            ),
            "provenance does not exactly match",
        ),
        (
            lambda raw: raw["provenance"].__setitem__(
                "source_lock_enforced", False
            ),
            "provenance does not exactly match",
        ),
        (
            lambda raw: raw["provenance"].__setitem__(
                "source_lock_relative_path", "frontier_rl/examples/other-lock.json"
            ),
            "source-lock binding mismatch",
        ),
        (
            lambda raw: raw["provenance"].__setitem__(
                "source_lock_sha256", "f" * 64
            ),
            "source-lock binding mismatch",
        ),
    ],
)
def test_full_rejects_coherently_receipted_raw_provenance_tamper_before_import(
    synthetic_bundle, monkeypatch, mutation, pattern
):
    raw = json.loads(synthetic_bundle["raw"].read_text(encoding="utf-8"))
    mutation(raw)
    forged_path = _refresh_coherent_raw_receipt_chain(
        synthetic_bundle, raw, "provenance"
    )
    imported = False

    def forbidden_loader(path, digest):
        nonlocal imported
        imported = True
        raise AssertionError("analyzer imported before raw provenance rejection")

    monkeypatch.setattr(manifest, "_load_verified_analyzer", forbidden_loader)
    with pytest.raises(ValueError, match=pattern):
        manifest.verify_manifest(
            forged_path,
            artifact_root=synthetic_bundle["root"],
            mode="full",
            raw_path=synthetic_bundle["raw"],
        )
    assert imported is False
