"""Synthetic-only adversarial tests for the portable verifier."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from frontier_rl.examples import analyze_acrobot_procurl_selection as analysis
from frontier_rl.examples import verify_acrobot_procurl_selection_portable as portable


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def synthetic_bundle(monkeypatch, tmp_path):
    source_root = tmp_path / "source"
    analyzer_path = source_root / portable.LOCKED_ANALYZER_RELATIVE_PATH
    analyzer_path.parent.mkdir(parents=True)
    analyzer_path.write_text("FROZEN = True\n", encoding="utf-8")
    manifest = {portable.LOCKED_ANALYZER_RELATIVE_PATH: portable._sha256(analyzer_path)}
    schedule = {"synthetic": "schedule"}
    seed_audit = {"passed": True, "synthetic": "audit"}
    runtime = {
        "python_implementation": "CPython",
        "python": "3.12.13",
        "platform": "SyntheticOS",
        "machine": "synthetic",
        "numpy": "2.5.1",
        "gymnasium": "1.3.0",
    }
    monkeypatch.setattr(analysis, "EXPECTED_SOURCE_RELATIVE_PATHS", tuple(manifest))
    monkeypatch.setattr(analysis, "_independent_locked_schedule", lambda: schedule)
    monkeypatch.setattr(
        analysis, "_independent_seed_collision_audit", lambda: seed_audit
    )
    monkeypatch.setattr(
        portable, "_load_verified_analyzer", lambda path, digest: analysis
    )
    monkeypatch.setattr(portable, "_verify_v2_audit", lambda *args: None)
    monkeypatch.setattr(
        portable,
        "_verify_invalid_incident",
        lambda *args: {"passed": True, "synthetic": True},
    )
    lock = {
        "schema": analysis.LOCK_SCHEMA,
        "status": "sealed_before_any_quick_development_or_confirmation",
        "created_utc": "2026-08-08T00:00:00+00:00",
        "purpose": (
            "Canonical pre-execution source/runtime lock for the Acrobot "
            "ProCuRL selection-semantic study."
        ),
        "runtime": runtime,
        "schedule": schedule,
        "seed_collision_audit": seed_audit,
        "source_sha256": manifest,
        "v2_dependency_audit": {"passed": True, "synthetic": True},
    }
    lock_path = tmp_path / "inputs" / "lock.json"
    _write(lock_path, lock)
    lock_hash = portable._sha256(lock_path)
    source_record = {
        "passed": True,
        "runtime": runtime,
        "source_lock_sha256": lock_hash,
        "checked_source_files": sorted(manifest),
    }

    def provenance():
        return {
            "runtime": runtime,
            "source_lock_sha256": lock_hash,
            "source_lock_enforced": True,
            "source_lock_relative_path": portable.EXPECTED_LOCK_RELATIVE_PATH,
            "source_sha256": manifest,
            "seed_collision_audit": seed_audit,
        }

    development_raw = {
        "schema": analysis.RAW_SCHEMA,
        "artifact_state": "complete",
        "run_failures": [],
        "protocol": {"mode": "development"},
        "provenance": provenance(),
    }
    development_relative = "evidence/development.json"
    development_path = source_root / development_relative
    _write(development_path, development_raw)
    development_hash = portable._sha256(development_path)
    gate = {
        "schema": analysis.GATE_SCHEMA,
        "mode": "development",
        "all_gates_passed": True,
        "source_lock_sha256": lock_hash,
        "source_lock_verification": source_record,
        "gates": {"synthetic": True},
        "diagnostics": {"synthetic": True},
        "gate_policy": {"outcome_blind": True},
        "raw_artifact_relative_path": development_relative,
        "raw_artifact_sha256": development_hash,
    }
    gate_relative = "evidence/development_gate.json"
    gate_path = source_root / gate_relative
    _write(gate_path, gate)
    binding = {
        "relative_path": gate_relative,
        "sha256": portable._sha256(gate_path),
        "raw_artifact_relative_path": development_relative,
        "raw_artifact_sha256": development_hash,
        "all_gates_passed": True,
    }
    confirmatory_raw = {
        "schema": analysis.RAW_SCHEMA,
        "artifact_state": "complete",
        "run_failures": [],
        "protocol": {"mode": "confirmatory", "development_gate": binding},
        "provenance": provenance(),
    }
    raw_path = source_root / "evidence" / "confirmatory.json"
    _write(raw_path, confirmatory_raw)
    development_validated = {"mode": "development", "synthetic": True}
    confirmation_validated = {
        "mode": "confirmatory",
        "seeds": list(range(80)),
        "by_case": {name: [] for name in analysis.ARM_NAMES},
        "cross_arm_crn_invariants": {"passed": True},
    }

    def validate(raw):
        if raw["protocol"]["mode"] == "development":
            return copy.deepcopy(development_validated)
        return copy.deepcopy(confirmation_validated)

    def development_gates(
        validated,
        observed_source,
        *,
        raw_artifact_relative_path,
        raw_artifact_sha256,
    ):
        assert validated == development_validated
        assert observed_source == source_record
        expected = copy.deepcopy(gate)
        expected["raw_artifact_relative_path"] = raw_artifact_relative_path
        expected["raw_artifact_sha256"] = raw_artifact_sha256
        return expected

    def confirmation(validated, observed_source, gate_check):
        assert validated == confirmation_validated
        assert observed_source == source_record
        assert gate_check["gate_recomputed_exactly"] is True
        return {
            "schema": analysis.ANALYSIS_SCHEMA,
            "mode": "confirmatory",
            "strict_validation_passed": True,
            "source_lock_verification": source_record,
            "development_gate_binding_verification": gate_check,
            "primary": {"mean_contrast": 0.03},
            "secondary_holm_family": {},
            "secondary_multiplicity": {},
            "arm_descriptives": {},
            "statistical_conventions": {},
        }

    monkeypatch.setattr(analysis, "validate_raw_artifact", validate)
    monkeypatch.setattr(analysis, "development_gates", development_gates)
    monkeypatch.setattr(analysis, "confirmatory_analysis", confirmation)
    gate_check = {
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
    stored = {
        **confirmation(confirmation_validated, source_record, gate_check),
        "raw_artifact_relative_path": "evidence/confirmatory.json",
        "raw_artifact_sha256": portable._sha256(raw_path),
    }
    stored_path = tmp_path / "inputs" / "analysis.json"
    _write(stored_path, stored)
    return {
        "source_root": source_root,
        "analyzer_path": analyzer_path,
        "lock_path": lock_path,
        "raw_path": raw_path,
        "stored_path": stored_path,
        "gate_path": gate_path,
    }


def _verify(bundle):
    return portable.verify_portable(
        bundle["lock_path"],
        bundle["raw_path"],
        bundle["stored_path"],
        source_root=bundle["source_root"],
    )


def test_portable_verifier_requires_exact_live_runtime_before_reanalysis(
    synthetic_bundle,
):
    result = _verify(synthetic_bundle)
    assert result["all_checks_passed"] is True
    assert result["recorded_execution_runtime"]["platform"] == "SyntheticOS"
    assert result["live_reanalysis_runtime_verification"] == {
        "passed": True,
        "checked_before_analyzer_import": True,
        "recorded_runtime": portable.EXPECTED_REANALYSIS_RUNTIME,
        "live_runtime": portable.EXPECTED_REANALYSIS_RUNTIME,
        "entropy_sum_sentinel_hex": portable.EXPECTED_ENTROPY_SUM_SENTINEL_HEX,
        "expected_entropy_sum_sentinel_hex": (
            portable.EXPECTED_ENTROPY_SUM_SENTINEL_HEX
        ),
        "known_naive_entropy_sum_sentinel_hex": (
            portable.NAIVE_ENTROPY_SUM_SENTINEL_HEX
        ),
    }
    assert (
        result["source_manifest_verification"]["analyzer_hashed_before_import"] is True
    )
    assert (
        result["development_gate_binding_verification"]["gate_recomputed_exactly"]
        is True
    )
    assert result["invalid_pre_gate_archive_verification"] == {
        "passed": True,
        "synthetic": True,
    }


def test_wrong_recorded_runtime_fails_before_analyzer_import(
    synthetic_bundle, monkeypatch
):
    lock = json.loads(synthetic_bundle["lock_path"].read_text(encoding="utf-8"))
    lock["runtime"]["python"] = "3.11.9"
    _write(synthetic_bundle["lock_path"], lock)
    monkeypatch.setattr(
        portable,
        "_load_verified_analyzer",
        lambda *args: pytest.fail("analyzer imported before recorded-runtime guard"),
    )
    with pytest.raises(ValueError, match="recorded runtime is not the exact pinned"):
        _verify(synthetic_bundle)


def test_wrong_live_runtime_fails_before_analyzer_import(synthetic_bundle, monkeypatch):
    wrong = dict(portable.EXPECTED_REANALYSIS_RUNTIME)
    wrong["python"] = "3.11.9"
    monkeypatch.setattr(portable, "_live_reanalysis_runtime", lambda: wrong)
    monkeypatch.setattr(
        portable,
        "_load_verified_analyzer",
        lambda *args: pytest.fail("analyzer imported before live-runtime guard"),
    )
    with pytest.raises(ValueError, match="live runtime is not the exact pinned"):
        _verify(synthetic_bundle)


def test_naive_sum_sentinel_fails_before_analyzer_import(synthetic_bundle, monkeypatch):
    monkeypatch.setattr(
        portable,
        "_entropy_sum_sentinel_hex",
        lambda: portable.NAIVE_ENTROPY_SUM_SENTINEL_HEX,
    )
    monkeypatch.setattr(
        portable,
        "_load_verified_analyzer",
        lambda *args: pytest.fail("analyzer imported before sum-semantics guard"),
    )
    with pytest.raises(ValueError, match="compensated-sum sentinel"):
        _verify(synthetic_bundle)


def test_real_invalid_wave_incident_verifies_hashes_sizes_and_blindness():
    result = portable._verify_invalid_incident(analysis.PROJECT_ROOT)
    assert result["passed"] is True
    assert result["outcome_blind"] is True
    assert result["development_gate_absent"] is True
    assert result["contrasts_uninspected"] is True
    assert len(result["archived_artifacts_checked"]) == 4


def test_invalid_wave_archive_tamper_is_detected(tmp_path):
    record = portable._load_json(
        analysis.PROJECT_ROOT / portable.INVALID_INCIDENT_RELATIVE_PATH,
        "registered invalid-wave incident",
    )
    archive_paths = []
    for artifact in record["archived_artifacts"]:
        path = tmp_path / artifact["archive_relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(artifact["role"].encode("utf-8"))
        artifact["sha256"] = portable._sha256(path)
        artifact["size_bytes"] = path.stat().st_size
        archive_paths.append(path)
    _write(tmp_path / portable.INVALID_INCIDENT_RELATIVE_PATH, record)
    assert portable._verify_invalid_incident(tmp_path)["passed"] is True

    archive_paths[0].write_bytes(archive_paths[0].read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="archived bytes mismatch"):
        portable._verify_invalid_incident(tmp_path)


def test_analyzer_hash_is_checked_before_any_module_execution(tmp_path):
    sentinel = tmp_path / "executed.txt"
    analyzer = tmp_path / "analyzer.py"
    analyzer.write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash changed before import"):
        portable._load_verified_analyzer(analyzer, "0" * 64)
    assert not sentinel.exists()
    portable._load_verified_analyzer(analyzer, portable._sha256(analyzer))
    assert sentinel.read_text(encoding="utf-8") == "executed"


def test_source_tampering_fails_before_reanalysis(synthetic_bundle):
    synthetic_bundle["analyzer_path"].write_text("FROZEN = False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="locked source hash mismatch"):
        _verify(synthetic_bundle)


def test_raw_runtime_tampering_fails(synthetic_bundle):
    raw = json.loads(synthetic_bundle["raw_path"].read_text(encoding="utf-8"))
    raw["provenance"]["runtime"]["platform"] = "tampered"
    _write(synthetic_bundle["raw_path"], raw)
    with pytest.raises(ValueError, match="recorded runtime differs"):
        _verify(synthetic_bundle)


def test_bound_gate_tampering_fails_hash(synthetic_bundle):
    gate = json.loads(synthetic_bundle["gate_path"].read_text(encoding="utf-8"))
    gate["diagnostics"] = {"tampered": True}
    _write(synthetic_bundle["gate_path"], gate)
    with pytest.raises(ValueError, match="development-gate hash mismatch"):
        _verify(synthetic_bundle)


def test_polluted_gate_schema_fails_after_exact_rebinding(synthetic_bundle):
    gate = json.loads(synthetic_bundle["gate_path"].read_text(encoding="utf-8"))
    gate["pollution"] = True
    _write(synthetic_bundle["gate_path"], gate)
    raw = json.loads(synthetic_bundle["raw_path"].read_text(encoding="utf-8"))
    raw["protocol"]["development_gate"]["sha256"] = portable._sha256(
        synthetic_bundle["gate_path"]
    )
    _write(synthetic_bundle["raw_path"], raw)
    with pytest.raises(ValueError, match="field set mismatch"):
        _verify(synthetic_bundle)


def test_confirmation_without_gate_binding_fails(synthetic_bundle):
    raw = json.loads(synthetic_bundle["raw_path"].read_text(encoding="utf-8"))
    raw["protocol"]["development_gate"] = None
    _write(synthetic_bundle["raw_path"], raw)
    with pytest.raises(ValueError, match="lacks exact development-gate binding"):
        _verify(synthetic_bundle)


def test_stored_analysis_tampering_fails_reanalysis(synthetic_bundle):
    stored = json.loads(synthetic_bundle["stored_path"].read_text(encoding="utf-8"))
    stored["primary"]["mean_contrast"] = -9.0
    _write(synthetic_bundle["stored_path"], stored)
    with pytest.raises(ValueError, match="stored analysis reanalysis mismatch"):
        _verify(synthetic_bundle)


def test_portable_loader_rejects_duplicate_keys_and_nonfinite(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"x": 1, "x": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        portable._load_json(path, "bad")
    path.write_text('{"x": Infinity}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        portable._load_json(path, "bad")
