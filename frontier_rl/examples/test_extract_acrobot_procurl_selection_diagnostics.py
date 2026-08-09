"""Synthetic/adversarial tests for the post-run descriptive extractor."""

from __future__ import annotations

import copy
import ast
import hashlib
import json
import os
import py_compile
import shutil
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")

from frontier_rl.examples import analyze_acrobot_procurl_selection as analysis
from frontier_rl.examples import extract_acrobot_procurl_selection_diagnostics as diagnostics


RAW_LOGICAL = "external/acrobot-procurl-selection-confirmatory.json"


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


def _selection(seed_index: int, arm_index: int) -> dict:
    offset = (seed_index + arm_index) / 100_000.0
    assigned = [1.0 / 8.0 + offset] * 4 + [1.0 / 8.0 - offset] * 4
    realized_offset = ((seed_index % 5) - 2) / 10_000.0
    realized = [1.0 / 8.0 + realized_offset] * 4 + [
        1.0 / 8.0 - realized_offset
    ] * 4
    probe_fraction = 0.2 if arm_index != 2 else 0.0
    return {
        "mean_selection_entropy": 2.0 - offset,
        "mean_selection_tv_from_uniform": 4.0 * offset,
        "mean_max_task_probability": 1.0 / 8.0 + offset,
        "mean_assigned_probability_per_task": assigned,
        "realized_task_fraction": realized,
        "realized_task_tv_from_uniform": 4.0 * abs(realized_offset),
        "student_fraction_of_paid": 1.0 - probe_fraction,
        "probe_fraction_of_paid": probe_fraction,
        "paid_budget_overshoot": seed_index + arm_index,
    }


def _validated() -> dict:
    by_case = {}
    for arm_index, arm in enumerate(analysis.ARM_NAMES):
        records = []
        for seed_index, seed in enumerate(analysis.CONFIRMATORY_SEEDS):
            success = (seed_index + arm_index) / 100.0
            records.append(
                {
                    "seed": seed,
                    "raw": {
                        "probe_sweeps": 0 if arm_index == 2 else 10 + seed_index,
                        "optimizer_updates": 200 + seed_index + arm_index,
                        "selection_diagnostics": _selection(seed_index, arm_index),
                    },
                    "derived": {
                        "auc_target_uniform_mean_success_full_atomic_paid": success,
                        "auc_target_uniform_mean_success_by_student_transitions": success
                        + 0.01,
                        "final_target_uniform_mean_success": min(success + 0.02, 1.0),
                        "final_native_success_rate": min(success + 0.03, 1.0),
                    },
                }
            )
        by_case[arm] = records
    return {
        "strict_valid": True,
        "mode": "confirmatory",
        "seeds": list(analysis.CONFIRMATORY_SEEDS),
        "by_case": by_case,
    }


@pytest.fixture
def synthetic(tmp_path: Path, monkeypatch):
    root = tmp_path / "bundle"
    analyzer_copy = root / diagnostics.LOCKED_ANALYZER_LOGICAL_PATH
    analyzer_copy.parent.mkdir(parents=True)
    shutil.copyfile(Path(analysis.__file__), analyzer_copy)
    source_sha = {
        relative: "a" * 64
        for relative in diagnostics.EXPECTED_SOURCE_RELATIVE_PATHS
    }
    source_sha[diagnostics.LOCKED_ANALYZER_LOGICAL_PATH] = _sha256(
        analyzer_copy
    )
    runtime = {
        **diagnostics.PINNED_REANALYSIS_RUNTIME,
        "platform": "synthetic",
        "machine": "synthetic",
    }
    lock = {
        "schema": diagnostics.LOCK_SCHEMA,
        "status": "sealed_before_any_quick_development_or_confirmation",
        "created_utc": "2026-08-09T00:00:00+00:00",
        "purpose": (
            "Canonical pre-execution source/runtime lock for the Acrobot "
            "ProCuRL selection-semantic study."
        ),
        "runtime": runtime,
        "schedule": diagnostics.EXPECTED_LOCK_SCHEDULE,
        "seed_collision_audit": {"passed": True},
        "source_sha256": source_sha,
        "v2_dependency_audit": {},
    }
    lock_logical = diagnostics.CANONICAL_LOCK_LOGICAL_PATH
    lock_path = root / lock_logical
    _write_json(lock_path, lock)
    lock_sha = _sha256(lock_path)
    gate_logical = "frontier_rl/examples/synthetic-development-gates.json"
    gate = {
        "schema": diagnostics.GATE_SCHEMA,
        "mode": "development",
        "all_gates_passed": True,
        "source_lock_sha256": lock_sha,
        "source_lock_verification": {
            "passed": True,
            "runtime": runtime,
            "source_lock_sha256": lock_sha,
            "checked_source_files": sorted(source_sha),
        },
        "gates": {name: True for name in diagnostics.DEVELOPMENT_GATE_NAMES},
        "diagnostics": {},
        "gate_policy": diagnostics.DEVELOPMENT_GATE_POLICY,
        "raw_artifact_relative_path": "evidence/synthetic-development.json",
        "raw_artifact_sha256": "d" * 64,
    }
    gate_path = root / gate_logical
    _write_json(gate_path, gate)
    gate_sha = _sha256(gate_path)
    raw_path = tmp_path / "external" / "raw.json"
    _write_json(
        raw_path,
        {
            "schema": analysis.RAW_SCHEMA,
            "synthetic": True,
            "provenance": {
                "source_lock_relative_path": lock_logical,
                "source_lock_sha256": lock_sha,
                "source_lock_enforced": True,
                "runtime": runtime,
                "source_sha256": source_sha,
                "seed_collision_audit": lock["seed_collision_audit"],
            },
            "protocol": {
                "mode": "confirmatory",
                "development_gate": {
                    "relative_path": gate_logical,
                    "sha256": gate_sha,
                    "raw_artifact_relative_path": gate[
                        "raw_artifact_relative_path"
                    ],
                    "raw_artifact_sha256": gate["raw_artifact_sha256"],
                    "all_gates_passed": True,
                },
            },
        },
    )
    validated = _validated()
    monkeypatch.setattr(analysis, "validate_raw_artifact", lambda raw: validated)
    monkeypatch.setattr(
        diagnostics,
        "_load_verified_analyzer",
        lambda path, digest: analysis,
    )
    report = diagnostics.extract_diagnostics(
        raw_path,
        raw_logical_path=RAW_LOGICAL,
        lock_path=lock_path,
        development_gate_path=gate_path,
        analyzer_path=analyzer_copy,
    )
    report_path = tmp_path / "report.json"
    diagnostics.write_json(report_path, report)
    return {
        "raw": raw_path,
        "validated": validated,
        "report": report,
        "report_path": report_path,
        "lock": lock_path,
        "gate": gate_path,
        "analyzer": analyzer_copy,
    }


def test_extracts_all_protocol_diagnostics_without_new_inference(synthetic):
    report = synthetic["report"]
    assert report["status"] == diagnostics.STATUS
    assert report["metric_policy"]["new_inferential_statistics"] is False
    assert report["schedule"]["run_count"] == 320
    assert report["development_gate"]["sha256"] == _sha256(synthetic["gate"])
    assert tuple(report["arms"]) == analysis.ARM_NAMES
    first = report["arms"][analysis.ARM_NAMES[0]]["per_seed"][0]
    assert set(first) == diagnostics.PER_SEED_KEYS
    assert first["auc_target_uniform_mean_success_full_atomic_paid"] == 0.0
    assert first["auc_target_uniform_mean_success_by_student_transitions"] == 0.01
    assert first["student_fraction_of_paid"] == 0.8
    assert first["probe_fraction_of_paid"] == 0.2
    assert len(first["mean_assigned_probability_per_task"]) == 8
    serialized = json.dumps(report, allow_nan=False).lower()
    for forbidden in ("p_value", "confidence_interval", "supported", "significant"):
        assert forbidden not in serialized


def test_extraction_and_verification_are_deterministic(synthetic):
    rebuilt = diagnostics.extract_diagnostics(
        synthetic["raw"],
        raw_logical_path=RAW_LOGICAL,
        lock_path=synthetic["lock"],
        development_gate_path=synthetic["gate"],
        analyzer_path=synthetic["analyzer"],
    )
    assert rebuilt == synthetic["report"]
    result = diagnostics.verify_diagnostics(
        synthetic["report_path"],
        synthetic["raw"],
        lock_path=synthetic["lock"],
        development_gate_path=synthetic["gate"],
        analyzer_path=synthetic["analyzer"],
    )
    assert result == {
        "schema": diagnostics.VERIFICATION_SCHEMA,
        "all_checks_passed": True,
        "run_count": 320,
        "arm_count": 4,
        "locked_raw_validation_replayed": True,
        "new_inferential_statistics": False,
    }


@pytest.mark.parametrize("target_role", ["raw", "lock", "gate"])
def test_diagnostics_binds_and_validates_each_exact_single_artifact_capture(
    synthetic, monkeypatch, target_role
):
    target = synthetic[target_role]
    original = target.read_bytes()
    replacement = b'{"schema":"swapped","schema":"invalid"}\n'
    state, original_read_bytes = _swap_after_first_read(
        monkeypatch, target, replacement
    )
    report = diagnostics.extract_diagnostics(
        synthetic["raw"],
        raw_logical_path=RAW_LOGICAL,
        lock_path=synthetic["lock"],
        development_gate_path=synthetic["gate"],
        analyzer_path=synthetic["analyzer"],
    )
    report_key = {
        "raw": "raw_artifact",
        "lock": "source_lock",
        "gate": "development_gate",
    }[target_role]
    assert state["reads"] == 1
    assert state["legacy_text_reads"] == 0
    assert report[report_key]["size_bytes"] == len(original)
    assert report[report_key]["sha256"] == hashlib.sha256(original).hexdigest()
    assert original_read_bytes(target) == replacement


def test_descriptive_mean_and_sample_std_recompute(synthetic):
    arm = analysis.ARM_NAMES[0]
    report = synthetic["report"]["arms"][arm]
    auc = report["descriptive_summary"]["scalar_metrics"][
        "auc_target_uniform_mean_success_full_atomic_paid"
    ]
    assert auc["mean"] == pytest.approx(0.395)
    assert auc["sample_std"] > 0.0
    assigned = report["descriptive_summary"]["per_task_metrics"][
        "mean_assigned_probability_per_task"
    ]
    assert len(assigned["mean_per_task"]) == 8
    assert len(assigned["sample_std_per_task"]) == 8


def test_missing_extra_or_reordered_seed_records_fail_closed(synthetic, monkeypatch):
    forged = copy.deepcopy(synthetic["validated"])
    forged["by_case"][analysis.ARM_NAMES[0]].append(
        copy.deepcopy(forged["by_case"][analysis.ARM_NAMES[0]][-1])
    )
    monkeypatch.setattr(analysis, "validate_raw_artifact", lambda raw: forged)
    with pytest.raises(ValueError, match="per-seed order mismatch"):
        diagnostics.extract_diagnostics(
            synthetic["raw"],
            raw_logical_path=RAW_LOGICAL,
            lock_path=synthetic["lock"],
            development_gate_path=synthetic["gate"],
            analyzer_path=synthetic["analyzer"],
        )


def test_extra_selection_field_fails_even_after_synthetic_validator(synthetic, monkeypatch):
    forged = copy.deepcopy(synthetic["validated"])
    forged["by_case"][analysis.ARM_NAMES[0]][0]["raw"]["selection_diagnostics"][
        "extra"
    ] = True
    monkeypatch.setattr(analysis, "validate_raw_artifact", lambda raw: forged)
    with pytest.raises(ValueError, match="selection diagnostic field set"):
        diagnostics.extract_diagnostics(
            synthetic["raw"],
            raw_logical_path=RAW_LOGICAL,
            lock_path=synthetic["lock"],
            development_gate_path=synthetic["gate"],
            analyzer_path=synthetic["analyzer"],
        )


def test_report_extra_field_and_changed_value_fail_closed(synthetic):
    forged = copy.deepcopy(synthetic["report"])
    forged["unregistered_extra"] = True
    with pytest.raises(ValueError, match="field set"):
        diagnostics.validate_report_shape(forged)
    changed = copy.deepcopy(synthetic["report"])
    changed["arms"][analysis.ARM_NAMES[0]]["per_seed"][0][
        "paid_budget_overshoot"
    ] += 1
    changed_path = synthetic["report_path"].with_name("changed.json")
    diagnostics.write_json(changed_path, changed)
    with pytest.raises(ValueError, match="summary does not exactly recompute"):
        diagnostics.verify_diagnostics(
            changed_path,
            synthetic["raw"],
            lock_path=synthetic["lock"],
            development_gate_path=synthetic["gate"],
            analyzer_path=synthetic["analyzer"],
        )


def test_strict_raw_and_report_loading_reject_duplicate_and_nonfinite(
    tmp_path: Path, synthetic
):
    duplicate = tmp_path / "duplicate-raw.json"
    duplicate.write_text('{"schema":"a","schema":"b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        diagnostics.extract_diagnostics(
            duplicate,
            raw_logical_path=RAW_LOGICAL,
            lock_path=synthetic["lock"],
            development_gate_path=synthetic["gate"],
            analyzer_path=synthetic["analyzer"],
        )
    nonfinite = tmp_path / "nonfinite-raw.json"
    nonfinite.write_text('{"value":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        diagnostics.extract_diagnostics(
            nonfinite,
            raw_logical_path=RAW_LOGICAL,
            lock_path=synthetic["lock"],
            development_gate_path=synthetic["gate"],
            analyzer_path=synthetic["analyzer"],
        )
    duplicate_report = tmp_path / "duplicate-report.json"
    duplicate_report.write_text('{"schema":"a","schema":"b"}\n', encoding="utf-8")
    valid_raw = tmp_path / "valid-raw.json"
    _write_json(valid_raw, {"schema": analysis.RAW_SCHEMA})
    with pytest.raises(ValueError, match="duplicate JSON key"):
        diagnostics.verify_diagnostics(
            duplicate_report,
            valid_raw,
            lock_path=synthetic["lock"],
            development_gate_path=synthetic["gate"],
            analyzer_path=synthetic["analyzer"],
        )


@pytest.mark.parametrize(
    "logical",
    [
        "/Users/example/raw.json",
        "evidence/name<.json",
        "evidence/name>.json",
        'evidence/name".json',
        "evidence/name|.json",
        "evidence/name?.json",
        "evidence/name*.json",
    ],
)
def test_nonportable_raw_logical_path_rejected(synthetic, logical):
    with pytest.raises(ValueError, match="relative|Windows-invalid"):
        diagnostics.extract_diagnostics(
            synthetic["raw"],
            raw_logical_path=logical,
            lock_path=synthetic["lock"],
            development_gate_path=synthetic["gate"],
            analyzer_path=synthetic["analyzer"],
        )


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        (
            lambda value: value["arms"][analysis.ARM_NAMES[0]]["per_seed"][0].__setitem__(
                "auc_target_uniform_mean_success_full_atomic_paid", True
            ),
            "finite float",
        ),
        (
            lambda value: value["arms"][analysis.ARM_NAMES[0]]["per_seed"][0].__setitem__(
                "student_fraction_of_paid", "0.8"
            ),
            "finite float",
        ),
        (
            lambda value: value["arms"][analysis.ARM_NAMES[0]]["per_seed"][0].__setitem__(
                "paid_budget_overshoot", 0.0
            ),
            "nonnegative integer",
        ),
        (
            lambda value: value["arms"][analysis.ARM_NAMES[0]]["per_seed"][0].__setitem__(
                "seed", 21_000.0
            ),
            "seed must be an integer",
        ),
        (
            lambda value: value["arms"][analysis.ARM_NAMES[0]]["per_seed"][0].__setitem__(
                "mean_selection_entropy", 99.0
            ),
            "out of range",
        ),
        (
            lambda value: value["arms"][analysis.ARM_NAMES[0]]["per_seed"][0][
                "realized_task_fraction"
            ].__setitem__(0, False),
            "finite float",
        ),
        (
            lambda value: value["arms"][analysis.ARM_NAMES[0]][
                "descriptive_summary"
            ]["scalar_metrics"]["final_native_success_rate"].__setitem__(
                "mean", "0.5"
            ),
            "finite float",
        ),
        (
            lambda value: value["arms"][analysis.ARM_NAMES[0]][
                "descriptive_summary"
            ]["scalar_metrics"]["optimizer_updates"].__setitem__(
                "sample_std", False
            ),
            "finite float",
        ),
        (
            lambda value: value["arms"][analysis.ARM_NAMES[0]][
                "descriptive_summary"
            ]["per_task_metrics"]["realized_task_fraction"][
                "mean_per_task"
            ].__setitem__(0, 2.0),
            "out of range",
        ),
        (
            lambda value: value["schedule"].__setitem__("run_count", 320.0),
            "run count mismatch",
        ),
        (
            lambda value: value["schedule"]["seeds"].__setitem__(0, 21_000.0),
            "schedule seeds mismatch",
        ),
    ],
)
def test_shape_validation_rejects_wrong_numeric_types_and_ranges(
    synthetic, mutation, pattern
):
    forged = copy.deepcopy(synthetic["report"])
    mutation(forged)
    with pytest.raises(ValueError, match=pattern):
        diagnostics.validate_report_shape(forged)


@pytest.mark.parametrize("failure", ["runtime", "sentinel"])
def test_diagnostics_trust_checks_precede_analyzer_import(
    synthetic, monkeypatch, failure
):
    called = False

    def forbidden_loader(path, digest):
        nonlocal called
        called = True
        raise AssertionError("analyzer import occurred before trust checks")

    monkeypatch.setattr(diagnostics, "_load_verified_analyzer", forbidden_loader)
    if failure == "runtime":
        monkeypatch.setattr(
            diagnostics,
            "_live_reanalysis_runtime",
            lambda: {
                **diagnostics.PINNED_REANALYSIS_RUNTIME,
                "gymnasium": "0.0",
            },
        )
        pattern = "live runtime"
    else:
        monkeypatch.setattr(
            diagnostics,
            "EXPECTED_ENTROPY_SUM_SENTINEL_HEX",
            "0x0.0p+0",
        )
        pattern = "compensated-sum sentinel"
    with pytest.raises(ValueError, match=pattern):
        diagnostics.extract_diagnostics(
            synthetic["raw"],
            raw_logical_path=RAW_LOGICAL,
            lock_path=synthetic["lock"],
            development_gate_path=synthetic["gate"],
            analyzer_path=synthetic["analyzer"],
        )
    assert called is False


def test_diagnostics_rejects_analyzer_bytes_not_bound_by_lock(synthetic):
    synthetic["analyzer"].write_text("# tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="analyzer does not match"):
        diagnostics.extract_diagnostics(
            synthetic["raw"],
            raw_logical_path=RAW_LOGICAL,
            lock_path=synthetic["lock"],
            development_gate_path=synthetic["gate"],
            analyzer_path=synthetic["analyzer"],
        )


def test_production_diagnostics_has_no_static_analyzer_import():
    tree = ast.parse(Path(diagnostics.__file__).read_text(encoding="utf-8"))
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any(
        "analyze_acrobot_procurl_selection" in ast.unparse(node)
        for node in imports
    )
    assert not any(
        "build_acrobot_procurl_external_manifest" in ast.unparse(node)
        for node in imports
    )


def _forbid_diagnostics_analyzer_import(monkeypatch):
    state = {"called": False}

    def forbidden(path, digest):
        state["called"] = True
        raise AssertionError("analyzer imported before evidence binding rejection")

    monkeypatch.setattr(diagnostics, "_load_verified_analyzer", forbidden)
    return state


def test_diagnostics_requires_present_development_gate(synthetic, monkeypatch):
    state = _forbid_diagnostics_analyzer_import(monkeypatch)
    missing = synthetic["gate"].with_name("missing-gate.json")
    with pytest.raises(ValueError, match="development gate is missing"):
        diagnostics.extract_diagnostics(
            synthetic["raw"],
            raw_logical_path=RAW_LOGICAL,
            lock_path=synthetic["lock"],
            development_gate_path=missing,
            analyzer_path=synthetic["analyzer"],
        )
    assert state["called"] is False


def test_diagnostics_rejects_forged_gate_even_with_refreshed_raw_hash(
    synthetic, monkeypatch
):
    gate = json.loads(synthetic["gate"].read_text(encoding="utf-8"))
    gate["gates"][diagnostics.DEVELOPMENT_GATE_NAMES[0]] = False
    _write_json(synthetic["gate"], gate)
    raw = json.loads(synthetic["raw"].read_text(encoding="utf-8"))
    raw["protocol"]["development_gate"]["sha256"] = _sha256(synthetic["gate"])
    _write_json(synthetic["raw"], raw)
    state = _forbid_diagnostics_analyzer_import(monkeypatch)
    with pytest.raises(ValueError, match="required checks"):
        diagnostics.extract_diagnostics(
            synthetic["raw"],
            raw_logical_path=RAW_LOGICAL,
            lock_path=synthetic["lock"],
            development_gate_path=synthetic["gate"],
            analyzer_path=synthetic["analyzer"],
        )
    assert state["called"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda binding: binding.__setitem__(
            "relative_path", "frontier_rl/examples/other-gate.json"
        ),
        lambda binding: binding.__setitem__("sha256", "e" * 64),
        lambda binding: binding.__setitem__(
            "raw_artifact_relative_path", "evidence/other-development.json"
        ),
        lambda binding: binding.__setitem__("raw_artifact_sha256", "e" * 64),
        lambda binding: binding.__setitem__("all_gates_passed", False),
    ],
)
def test_diagnostics_rejects_each_raw_gate_binding_field_before_import(
    synthetic, monkeypatch, mutation
):
    raw = json.loads(synthetic["raw"].read_text(encoding="utf-8"))
    mutation(raw["protocol"]["development_gate"])
    _write_json(synthetic["raw"], raw)
    state = _forbid_diagnostics_analyzer_import(monkeypatch)
    with pytest.raises(ValueError, match="five-field binding mismatch"):
        diagnostics.extract_diagnostics(
            synthetic["raw"],
            raw_logical_path=RAW_LOGICAL,
            lock_path=synthetic["lock"],
            development_gate_path=synthetic["gate"],
            analyzer_path=synthetic["analyzer"],
        )
    assert state["called"] is False


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        (
            lambda provenance: provenance["runtime"].__setitem__("numpy", "0.0"),
            "runtime differs",
        ),
        (
            lambda provenance: provenance["source_sha256"].__setitem__(
                diagnostics.LOCKED_ANALYZER_LOGICAL_PATH, "e" * 64
            ),
            "source manifest differs",
        ),
        (
            lambda provenance: provenance["seed_collision_audit"].__setitem__(
                "tampered", True
            ),
            "seed audit differs",
        ),
        (
            lambda provenance: provenance.__setitem__("source_lock_enforced", False),
            "did not enforce",
        ),
        (
            lambda provenance: provenance.__setitem__(
                "source_lock_relative_path", "frontier_rl/examples/other-lock.json"
            ),
            "not canonical",
        ),
        (
            lambda provenance: provenance.__setitem__(
                "source_lock_sha256", "e" * 64
            ),
            "SHA-256 mismatch",
        ),
    ],
)
def test_diagnostics_rejects_raw_provenance_tamper_before_import(
    synthetic, monkeypatch, mutation, pattern
):
    raw = json.loads(synthetic["raw"].read_text(encoding="utf-8"))
    mutation(raw["provenance"])
    _write_json(synthetic["raw"], raw)
    state = _forbid_diagnostics_analyzer_import(monkeypatch)
    with pytest.raises(ValueError, match=pattern):
        diagnostics.extract_diagnostics(
            synthetic["raw"],
            raw_logical_path=RAW_LOGICAL,
            lock_path=synthetic["lock"],
            development_gate_path=synthetic["gate"],
            analyzer_path=synthetic["analyzer"],
        )
    assert state["called"] is False


def test_diagnostics_verified_buffer_ignores_timestamp_valid_malicious_pyc(
    tmp_path: Path,
):
    source_path = tmp_path / "cached_diagnostics_analyzer.py"
    malicious = b'VALUE = "evil"\n'
    verified = b'VALUE = "safe"\n'
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
    os.utime(source_path, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
    assert pyc_path.is_file()
    module = diagnostics._load_verified_analyzer(
        source_path, hashlib.sha256(verified).hexdigest()
    )
    assert module.VALUE == "safe"
