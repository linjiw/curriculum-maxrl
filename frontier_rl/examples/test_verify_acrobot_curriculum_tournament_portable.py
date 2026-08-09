"""Synthetic-only tests for the portable Acrobot tournament verifier."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from frontier_rl.examples import analyze_acrobot_curriculum_tournament as analysis
from frontier_rl.examples import (
    verify_acrobot_curriculum_tournament_portable as portable,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def synthetic_bundle(monkeypatch, tmp_path):
    source_root = tmp_path / "source"
    source = source_root / portable.LOCKED_ANALYZER_RELATIVE_PATH
    source.parent.mkdir(parents=True)
    source.write_text("FROZEN = True\n", encoding="utf-8")
    manifest = {portable.LOCKED_ANALYZER_RELATIVE_PATH: portable._sha256(source)}
    schedule = {"synthetic_schedule": True}
    seed_audit = {"passed": True, "synthetic_seed_audit": True}
    recorded_runtime = {
        "python_implementation": "CPython",
        "python": "3.12.13",
        "platform": "SyntheticOS-that-is-not-the-test-host",
        "machine": "synthetic-cpu",
        "numpy": "2.5.1",
        "gymnasium": "1.3.0",
    }
    monkeypatch.setattr(analysis, "EXPECTED_SOURCE_RELATIVE_PATHS", tuple(manifest))
    monkeypatch.setattr(analysis, "__file__", str(source))
    monkeypatch.setattr(analysis, "_independent_locked_schedule", lambda: schedule)
    monkeypatch.setattr(
        analysis, "_independent_seed_collision_audit", lambda: seed_audit
    )

    lock = {
        "schema": analysis.LOCK_SCHEMA,
        "runtime": recorded_runtime,
        "schedule": schedule,
        "seed_collision_audit": seed_audit,
        "source_sha256": manifest,
    }
    lock_path = tmp_path / "inputs" / "lock.json"
    _write(lock_path, lock)
    lock_sha = portable._sha256(lock_path)
    source_lock = {
        "passed": True,
        "runtime": recorded_runtime,
        "source_lock_sha256": lock_sha,
        "checked_source_files": sorted(manifest),
    }

    def raw_provenance():
        return {
            "runtime": recorded_runtime,
            "source_lock_sha256": lock_sha,
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
        "provenance": raw_provenance(),
        "synthetic_payload": "development-ledgers",
    }
    development_relative = "evidence/development.json"
    development_path = source_root / development_relative
    _write(development_path, development_raw)
    development_sha = portable._sha256(development_path)

    gate_core = {
        "schema": analysis.GATE_SCHEMA,
        "mode": "development",
        "all_gates_passed": True,
        "source_lock_sha256": lock_sha,
        "source_lock_verification": source_lock,
        "gates": {name: True for name in analysis.DEVELOPMENT_GATE_NAMES},
        "diagnostics": {"synthetic_reanalysis": True},
        "gate_policy": dict(analysis.DEVELOPMENT_GATE_POLICY),
    }
    gate = {
        **gate_core,
        "raw_artifact_path": "/original/machine/development.json",
        "raw_artifact_relative_path": development_relative,
        "raw_artifact_sha256": development_sha,
        "source_lock_path": "/original/machine/lock.json",
        "source_lock_relative_path": portable.EXPECTED_LOCK_RELATIVE_PATH,
        "source_lock_sha256": lock_sha,
    }
    gate_relative = "evidence/development_gate.json"
    gate_path = source_root / gate_relative
    _write(gate_path, gate)
    gate_sha = portable._sha256(gate_path)
    gate_binding = {
        "relative_path": gate_relative,
        "sha256": gate_sha,
        "raw_artifact_relative_path": development_relative,
        "raw_artifact_sha256": development_sha,
        "all_gates_passed": True,
    }

    confirmatory_raw = {
        "schema": analysis.RAW_SCHEMA,
        "artifact_state": "complete",
        "run_failures": [],
        "protocol": {
            "mode": "confirmatory",
            "development_gate": gate_binding,
        },
        "provenance": raw_provenance(),
        "synthetic_payload": "confirmatory-ledgers",
    }
    raw_path = tmp_path / "inputs" / "confirmatory.json"
    _write(raw_path, confirmatory_raw)
    raw_sha = portable._sha256(raw_path)

    development_validated = {
        "mode": "development",
        "seeds": [20100, 20101, 20102],
        "by_case": {"synthetic": True},
    }
    confirmatory_validated = {
        "mode": "confirmatory",
        "seeds": list(range(20000, 20020)),
        "by_case": {"synthetic": True},
    }

    def validate(raw):
        mode = raw.get("protocol", {}).get("mode")
        if mode == "development":
            return copy.deepcopy(development_validated)
        if mode == "confirmatory":
            return copy.deepcopy(confirmatory_validated)
        raise ValueError("synthetic fixture has an unknown mode")

    def development_gates(validated, observed_source_lock):
        assert validated == development_validated
        assert observed_source_lock == source_lock
        return copy.deepcopy(gate_core)

    confirmatory_core = {
        "schema": analysis.REPORT_SCHEMA,
        "mode": "confirmatory",
        "all_checks_passed": True,
        "source_lock": source_lock,
        "primary": {
            "mean_paired_difference": 0.0125,
            "exact_two_sided_sign_flip_p": 0.03125,
        },
        "secondary_uniform_auc_tests": {"synthetic": True},
    }

    def confirmatory_analysis(validated, observed_source_lock):
        assert validated == confirmatory_validated
        assert observed_source_lock == source_lock
        return copy.deepcopy(confirmatory_core)

    monkeypatch.setattr(analysis, "_validate_raw_artifact", validate)
    monkeypatch.setattr(analysis, "development_gates", development_gates)
    monkeypatch.setattr(analysis, "confirmatory_analysis", confirmatory_analysis)

    gate_verification = {
        "passed": True,
        "development_gate_relative_path": gate_relative,
        "development_gate_sha256": gate_sha,
        "development_raw_relative_path": development_relative,
        "development_raw_sha256": development_sha,
        "gates_recomputed_from_raw": True,
    }
    stored = {
        **confirmatory_core,
        "development_gate_verification": gate_verification,
        "raw_artifact_path": "/original/machine/confirmatory.json",
        "raw_artifact_sha256": raw_sha,
        "raw_artifact_relative_path": "evidence/confirmatory.json",
        "source_lock_path": "/original/machine/lock.json",
        "source_lock_relative_path": portable.EXPECTED_LOCK_RELATIVE_PATH,
        "source_lock_sha256": lock_sha,
    }
    stored_path = tmp_path / "inputs" / "analysis.json"
    _write(stored_path, stored)
    return {
        "source_root": source_root,
        "source": source,
        "lock_path": lock_path,
        "raw_path": raw_path,
        "stored_path": stored_path,
        "gate_path": gate_path,
        "development_path": development_path,
    }


def _verify(bundle):
    return portable.verify_portable(
        bundle["lock_path"],
        bundle["raw_path"],
        bundle["stored_path"],
        source_root=bundle["source_root"],
    )


def test_portable_verifier_checks_bundle_without_live_runtime_equality(
    synthetic_bundle,
):
    result = _verify(synthetic_bundle)
    assert result["all_checks_passed"] is True
    assert result["recorded_execution_runtime"]["platform"].startswith("SyntheticOS")
    assert result["source_manifest_verification"]["all_live_hashes_match"] is True
    assert result["raw_ledger_validation"]["paired_seed_count"] == 20
    assert result["development_gate_binding_verification"][
        "gates_recomputed_from_raw"
    ] is True
    assert result["stored_analysis_comparison"][
        "all_recomputed_fields_match"
    ] is True
    assert "does not reproduce the training execution" in result["scope"]
    assert "execution_reproduced" not in json.dumps(result)


def test_incomplete_raw_artifact_fails_closed(synthetic_bundle):
    raw = json.loads(synthetic_bundle["raw_path"].read_text(encoding="utf-8"))
    raw["artifact_state"] = "in_progress"
    _write(synthetic_bundle["raw_path"], raw)
    with pytest.raises(ValueError, match="raw artifact is incomplete"):
        _verify(synthetic_bundle)


def test_source_byte_tampering_fails_manifest_hash_check(synthetic_bundle):
    synthetic_bundle["source"].write_text("FROZEN = False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="locked source hash mismatch"):
        _verify(synthetic_bundle)


def test_recorded_runtime_must_still_bind_raw_to_lock(synthetic_bundle):
    raw = json.loads(synthetic_bundle["raw_path"].read_text(encoding="utf-8"))
    raw["provenance"]["runtime"]["platform"] = "tampered"
    _write(synthetic_bundle["raw_path"], raw)
    with pytest.raises(ValueError, match="recorded runtime differs"):
        _verify(synthetic_bundle)


def test_bound_development_gate_tampering_fails_hash_check(synthetic_bundle):
    gate = json.loads(synthetic_bundle["gate_path"].read_text(encoding="utf-8"))
    gate["diagnostics"] = {"tampered": True}
    _write(synthetic_bundle["gate_path"], gate)
    with pytest.raises(ValueError, match="development-gate hash mismatch"):
        _verify(synthetic_bundle)


def test_stored_statistic_tampering_fails_reanalysis_comparison(synthetic_bundle):
    stored = json.loads(synthetic_bundle["stored_path"].read_text(encoding="utf-8"))
    stored["primary"]["mean_paired_difference"] = 999.0
    _write(synthetic_bundle["stored_path"], stored)
    with pytest.raises(
        ValueError, match="does not match reanalysis for field: primary"
    ):
        _verify(synthetic_bundle)


def test_stored_raw_hash_binding_is_checked(synthetic_bundle):
    stored = json.loads(synthetic_bundle["stored_path"].read_text(encoding="utf-8"))
    stored["raw_artifact_sha256"] = "0" * 64
    _write(synthetic_bundle["stored_path"], stored)
    with pytest.raises(ValueError, match="bound to a different raw artifact"):
        _verify(synthetic_bundle)


def test_cli_prints_read_only_verification_json(synthetic_bundle, capsys):
    portable.main(
        [
            str(synthetic_bundle["lock_path"]),
            str(synthetic_bundle["raw_path"]),
            str(synthetic_bundle["stored_path"]),
            "--source-root",
            str(synthetic_bundle["source_root"]),
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert output["schema"] == portable.SCHEMA
    assert output["all_checks_passed"] is True
