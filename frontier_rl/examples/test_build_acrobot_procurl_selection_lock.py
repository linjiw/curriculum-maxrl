"""Synthetic tests for the separate source/runtime lock builder.

These tests never write the canonical study lock.
"""

from __future__ import annotations

import copy
import json

import pytest

from frontier_rl.examples import build_acrobot_procurl_selection_lock as builder
from frontier_rl.examples import run_acrobot_procurl_selection as runner


def _synthetic_lock_dependencies(monkeypatch, tmp_path):
    source = tmp_path / "source.py"
    source.write_text("FROZEN = True\n", encoding="utf-8")
    runtime = runner._runtime()
    schedule = {"schedule": "synthetic"}
    seed_audit = {"passed": True, "audit": "synthetic"}
    v2_audit = {"passed": True, "audit": "v2"}
    monkeypatch.setattr(runner, "LOCK_PATH", tmp_path / "LOCK.json")
    monkeypatch.setattr(
        runner, "PINNED_RUNTIME_VERSIONS", runner._runtime_versions(runtime)
    )
    monkeypatch.setattr(runner, "_runtime", lambda: copy.deepcopy(runtime))
    monkeypatch.setattr(runner, "_locked_schedule", lambda: copy.deepcopy(schedule))
    monkeypatch.setattr(
        runner, "seed_collision_audit", lambda: copy.deepcopy(seed_audit)
    )
    monkeypatch.setattr(runner, "_v2_dependency_audit", lambda: copy.deepcopy(v2_audit))
    monkeypatch.setattr(
        runner,
        "_source_hashes",
        lambda require_all=True: {"source.py": runner._sha256(source)},
    )
    return source, runtime, schedule, seed_audit, v2_audit


def test_lock_payload_has_exact_closed_schema(monkeypatch, tmp_path):
    _synthetic_lock_dependencies(monkeypatch, tmp_path)
    payload = builder.build_lock_payload()
    assert set(payload) == runner.LOCK_KEYS
    assert payload["schema"] == runner.LOCK_SCHEMA
    assert payload["status"] == "sealed_before_any_quick_development_or_confirmation"
    assert payload["v2_dependency_audit"]["passed"] is True


def test_atomic_writer_refuses_overwrite_and_preserves_existing_bytes(
    monkeypatch, tmp_path
):
    _synthetic_lock_dependencies(monkeypatch, tmp_path)
    output = runner.LOCK_PATH
    first = builder.write_lock_atomic_refuse_overwrite(output)
    before = output.read_bytes()
    assert json.loads(before)["schema"] == first["schema"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.write_lock_atomic_refuse_overwrite(output)
    assert output.read_bytes() == before


def test_builder_fails_closed_when_runtime_seed_or_v2_audit_fails(
    monkeypatch, tmp_path
):
    _synthetic_lock_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "PINNED_RUNTIME_VERSIONS", {"python": "impossible"})
    with pytest.raises(RuntimeError, match="exact pinned runtime"):
        builder.build_lock_payload()

    _synthetic_lock_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "seed_collision_audit", lambda: {"passed": False})
    with pytest.raises(RuntimeError, match="seed/RNG audit"):
        builder.build_lock_payload()

    _synthetic_lock_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_v2_dependency_audit", lambda: {"passed": False})
    with pytest.raises(RuntimeError, match="V2 transitive"):
        builder.build_lock_payload()


def test_v2_transitive_dependency_tamper_is_detected(monkeypatch, tmp_path):
    dependency = tmp_path / "dependency.py"
    dependency.write_text("VERSION = 1\n", encoding="utf-8")
    relative = "dependency.py"
    v2_lock = tmp_path / "V2.json"
    v2_lock.write_text(
        json.dumps(
            {
                "schema": "curriculum-maxrl/acrobot-curriculum-tournament-lock/v2",
                "source_sha256": {relative: runner._sha256(dependency)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "V2_LOCK_PATH", v2_lock)
    monkeypatch.setattr(runner, "V2_DEPENDENCY_PATHS", (relative,))
    audit = runner._v2_dependency_audit()
    assert audit["passed"] is True
    dependency.write_text("VERSION = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differ from the V2 source lock"):
        runner._v2_dependency_audit()


def test_source_lock_verifier_detects_source_tamper_and_closed_schema(
    monkeypatch, tmp_path
):
    source, runtime, schedule, seed_audit, v2_audit = _synthetic_lock_dependencies(
        monkeypatch, tmp_path
    )
    monkeypatch.setattr(runner, "SOURCE_RELATIVE_PATHS", ("source.py",))
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    lock = {
        "schema": runner.LOCK_SCHEMA,
        "status": "sealed_before_any_quick_development_or_confirmation",
        "created_utc": "2026-08-08T00:00:00+00:00",
        "purpose": (
            "Canonical pre-execution source/runtime lock for the Acrobot "
            "ProCuRL selection-semantic study."
        ),
        "runtime": runtime,
        "schedule": schedule,
        "seed_collision_audit": seed_audit,
        "source_sha256": {"source.py": runner._sha256(source)},
        "v2_dependency_audit": v2_audit,
    }
    runner.LOCK_PATH.write_text(json.dumps(lock), encoding="utf-8")
    observed, digest = runner._load_and_verify_lock(runner.LOCK_PATH)
    assert observed == lock and digest == runner._sha256(runner.LOCK_PATH)

    polluted = copy.deepcopy(lock)
    polluted["unexpected"] = True
    runner.LOCK_PATH.write_text(json.dumps(polluted), encoding="utf-8")
    with pytest.raises(RuntimeError, match="top-level schema"):
        runner._load_and_verify_lock(runner.LOCK_PATH)

    runner.LOCK_PATH.write_text(json.dumps(lock), encoding="utf-8")
    source.write_text("FROZEN = False\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        runner._load_and_verify_lock(runner.LOCK_PATH)
