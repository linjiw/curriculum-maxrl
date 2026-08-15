"""Focused contracts for the three-arm Acrobot tournament runner."""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from frontier_rl.examples import run_acrobot_curriculum_tournament as tournament
from frontier_rl.examples import analyze_acrobot_curriculum_tournament as analysis
from frontier_rl.examples import run_acrobot_neural as engine
from frontier_rl.teacher import FrontierTeacher


def test_registered_arm_and_schedule_contracts():
    assert [condition.name for condition in tournament.CONDITIONS] == [
        "uniform_shared_h64",
        "p1mp_shared_h64",
        "u16_shared_h64",
    ]
    assert {condition.sampling for condition in tournament.CONDITIONS} == {
        "uniform", "p1mp", "u16",
    }
    assert all(
        condition.architecture == "shared"
        and condition.hidden_size == 64
        and condition.learning_rate == 3e-4
        and condition.hindsight_scale == 0.0
        for condition in tournament.CONDITIONS
    )
    assert tournament.CONFIRMATORY_SEEDS == tuple(range(20_000, 20_020))
    assert tournament.DEVELOPMENT_SEEDS == tuple(range(20_100, 20_103))
    assert tournament.QUICK_SEEDS == (20_200,)
    assert tournament._mode_schedule("confirmatory") == (
        tournament.CONFIRMATORY_SEEDS, 2_000_000, 100_000, 32,
    )
    assert tournament._mode_schedule("development") == (
        tournament.DEVELOPMENT_SEEDS, 200_000, 50_000, 16,
    )
    assert tournament._mode_schedule("quick") == (
        tournament.QUICK_SEEDS, 8_000, 4_000, 2,
    )


def test_seed_collision_audit_is_clean_and_explicit():
    audit = tournament.seed_collision_audit()
    assert audit["passed"] is True
    assert audit["confirmatory_seeds"] == list(range(20_000, 20_020))
    assert not any(audit["logical_seed_collisions"].values())
    assert audit["derived_root_collisions"] == []
    assert audit["derived_roots_vs_all_logical_training_seeds"] == []
    assert audit["unique_derived_root_count"] == (
        audit["expected_unique_derived_root_count"]
    )
    assert list(range(19_000, 19_020)) == audit["prior_training_seed_blocks"][
        "aborted_tournament_confirmation_rng_overlap"
    ]
    roots = [
        root
        for record in audit["per_logical_seed"].values()
        for root in record["rng_roots"].values()
    ]
    assert len(roots) == len(set(roots))
    assert len(roots) == 24 * len(tournament.RNG_DOMAIN_OFFSETS)
    logical_seeds = {
        *tournament.CONFIRMATORY_SEEDS,
        *tournament.DEVELOPMENT_SEEDS,
        *tournament.QUICK_SEEDS,
    }
    assert set(roots).isdisjoint(logical_seeds)


def test_engine_master_stride_and_adapter_internal_roots_are_exact():
    left = tournament.rng_domain_record(20_000)
    right = tournament.rng_domain_record(20_001)
    assert right["engine_master_seed"] - left["engine_master_seed"] == 10_000_000
    master = left["engine_master_seed"]
    assert left["environment_adapter_seed_argument"] == master + 1_000
    assert left["rng_roots"] == {
        "actor_parameter": master,
        "actor_action": master + 1,
        "teacher": master + 10_000,
        "environment_reset_rng": master + 11_003,
        "evaluation_episode": master + 1_000_000,
        "evaluation_action": master + 1_000_001,
    }
    assert set(left["rng_roots"].values()).isdisjoint(
        right["rng_roots"].values()
    )


def test_p1mp_changes_only_utility_and_u16_matches_canonical_teacher():
    p = np.asarray([0.0, 0.1, 0.5, 0.9, 1.0])
    learnability = tournament._TournamentTeacher("p1mp", seed=7)
    assert np.allclose(learnability.utility(p), p * (1.0 - p))

    recorded = tournament._TournamentTeacher("u16", seed=11)
    canonical = FrontierTeacher(
        len(engine.THRESHOLDS), 16, decay=0.7, floor=0.1, gamma=1.0, seed=11
    )
    for task, rewards in (
        (0, np.zeros(16)),
        (3, np.r_[np.ones(7), np.zeros(9)]),
        (7, np.ones(16)),
    ):
        assert np.allclose(recorded.distribution(), canonical.distribution())
        recorded.observe(task, rewards)
        canonical.observe(task, rewards)
    assert len(recorded.distribution_records) == 3


def test_uniform_sampler_is_exact_and_does_not_consume_thompson_draw():
    teacher = tournament._TournamentTeacher("uniform", seed=13)
    state = teacher.rng.bit_generator.state
    probabilities = teacher.distribution()
    assert np.array_equal(probabilities, np.full(8, 1.0 / 8.0))
    assert teacher.rng.bit_generator.state == state


def test_process_local_factory_patch_is_restored_even_on_error():
    original = engine._teacher_for
    with (
        pytest.raises(RuntimeError, match="synthetic"),
        tournament._patched_teacher_factory() as capture,
    ):
        assert engine._teacher_for is tournament._tournament_teacher_factory
        condition = tournament.CONDITIONS[1]
        teacher = engine._teacher_for(condition, 123)
        assert capture == [teacher]
        raise RuntimeError("synthetic")
    assert engine._teacher_for is original
    assert tournament._ACTIVE_FACTORY_CAPTURE is None


def test_practical_mass_extremes_and_mixed_group():
    assert tournament.practical_maxrl_mass(0) == 0.0
    assert tournament.practical_maxrl_mass(16) == 0.0
    assert tournament.practical_maxrl_mass(4) == 1.5
    with pytest.raises(ValueError):
        tournament.practical_maxrl_mass(17)


def test_run_one_passes_engine_master_and_retains_logical_seed(monkeypatch):
    observed = {}

    @contextmanager
    def fake_patch():
        yield [object()]

    def fake_engine_run(condition, seed, **kwargs):
        observed["condition"] = condition
        observed["seed"] = seed
        observed["kwargs"] = kwargs
        return {"seed": seed}

    monkeypatch.setattr(tournament, "_patched_teacher_factory", fake_patch)
    monkeypatch.setattr(tournament.engine, "run_condition", fake_engine_run)
    monkeypatch.setattr(
        tournament,
        "_augment_run",
        lambda run, teacher, **kwargs: run,
    )
    logical = tournament.QUICK_SEEDS[0]
    run = tournament.run_one(
        tournament.CONDITIONS[0], logical, mode="quick"
    )
    master = tournament.engine_master_seed(logical)
    assert observed["seed"] == master
    assert run["seed"] == logical
    assert run["logical_seed"] == logical
    assert run["engine_master_seed"] == master
    assert run["rng_roots"] == tournament.rng_domain_record(logical)["rng_roots"]


def test_lock_verifier_checks_exact_runtime_schedule_sources_and_audit(
    monkeypatch, tmp_path
):
    relative = "synthetic.py"
    source = tmp_path / relative
    source.write_text("# synthetic tournament lock test\n", encoding="utf-8")
    monkeypatch.setattr(tournament, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(tournament, "SOURCE_RELATIVE_PATHS", (relative,))
    lock = {
        "schema": tournament.LOCK_SCHEMA,
        "runtime": tournament._runtime(),
        "schedule": tournament._locked_schedule(),
        "seed_collision_audit": tournament.seed_collision_audit(),
        "source_sha256": {relative: tournament._sha256(source)},
    }
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    observed, digest = tournament._load_and_verify_lock(path)
    assert observed == lock
    assert digest == tournament._sha256(path)

    tamper_cases = (
        (lambda item: item.update(schema="wrong"), "lock schema mismatch"),
        (
            lambda item: item.update(
                runtime={**item["runtime"], "numpy": "wrong"}
            ),
            "runtime mismatch",
        ),
        (lambda item: item.update(schedule={}), "locked schedule mismatch"),
        (
            lambda item: item.update(seed_collision_audit={}),
            "seed collision audit mismatch",
        ),
        (
            lambda item: item.update(source_sha256={"other.py": "0" * 64}),
            "source lock key set is not exact",
        ),
        (
            lambda item: item["source_sha256"].update({relative: "0" * 64}),
            "source hash mismatch",
        ),
    )
    for tamper, message in tamper_cases:
        changed = copy.deepcopy(lock)
        tamper(changed)
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(RuntimeError, match=message):
            tournament._load_and_verify_lock(path)


def test_confirmation_requires_fresh_v2_development_gate(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        tournament, "_load_and_verify_lock", lambda path: ({"source_sha256": {}}, "h")
    )
    with pytest.raises(RuntimeError, match="requires the fresh passing V2"):
        tournament.run_tournament(
            mode="confirmatory", output=tmp_path / "blocked.json"
        )


def test_nonquick_runs_require_the_canonical_lock_path(monkeypatch, tmp_path):
    canonical = tmp_path / "canonical-lock.json"
    copy_path = tmp_path / "content-equivalent-copy.json"
    canonical.write_text("{}\n", encoding="utf-8")
    copy_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(tournament, "LOCK_PATH", canonical)
    tournament._require_canonical_lock_path(canonical)
    with pytest.raises(RuntimeError, match="canonical source lock"):
        tournament.run_one(
            tournament.CONDITIONS[0],
            tournament.DEVELOPMENT_SEEDS[0],
            mode="development",
            lock_path=copy_path,
        )
    with pytest.raises(RuntimeError, match="canonical source lock"):
        tournament.run_tournament(
            mode="development",
            output=tmp_path / "blocked.json",
            lock_path=copy_path,
        )


def test_gate_loader_recomputes_raw_gates_and_rejects_strict_tampering(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(tournament, "PROJECT_ROOT", tmp_path)
    canonical_lock = tmp_path / "LOCK.json"
    monkeypatch.setattr(tournament, "LOCK_PATH", canonical_lock)
    lock_hash = "a" * 64
    lock = {
        "source_sha256": {"synthetic.py": "b" * 64},
        "seed_collision_audit": {"passed": True},
    }
    raw = {
        "schema": tournament.SCHEMA,
        "artifact_state": "complete",
        "run_failures": [],
        "protocol": {
            "mode": "development",
            "paired_seeds": list(tournament.DEVELOPMENT_SEEDS),
        },
        "provenance": {
            "source_lock_sha256": lock_hash,
            "source_lock_enforced": True,
            "source_lock_relative_path": "LOCK.json",
            "source_sha256": lock["source_sha256"],
            "runtime": tournament._runtime(),
            "seed_collision_audit": lock["seed_collision_audit"],
        },
    }
    raw_path = tmp_path / "development.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    lock_verification = {
        "passed": True,
        "runtime": tournament._runtime(),
        "source_lock_sha256": lock_hash,
        "checked_source_files": ["synthetic.py"],
    }
    recomputed_gate = {
        "schema": tournament.DEVELOPMENT_GATE_SCHEMA,
        "mode": "development",
        "all_gates_passed": True,
        "source_lock_sha256": lock_hash,
        "source_lock_verification": lock_verification,
        "gates": {name: True for name in tournament.DEVELOPMENT_GATE_NAMES},
        "diagnostics": {"recomputed_from_raw": True},
        "gate_policy": dict(tournament.DEVELOPMENT_GATE_POLICY),
    }
    gate = {
        **recomputed_gate,
        "raw_artifact_relative_path": "development.json",
        "raw_artifact_sha256": tournament._sha256(raw_path),
    }
    validated_marker = {"validated": True}
    observed = {}

    def validate(payload):
        observed["raw"] = payload
        return validated_marker

    def recompute(validated, source_lock):
        observed["validated"] = validated
        observed["source_lock"] = source_lock
        return recomputed_gate

    monkeypatch.setattr(analysis, "_validate_raw_artifact", validate)
    monkeypatch.setattr(analysis, "development_gates", recompute)
    gate_path = tmp_path / "gates.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    accepted = tournament._load_development_gate(gate_path, lock, lock_hash)
    assert accepted["raw_artifact_sha256"] == tournament._sha256(raw_path)
    assert observed == {
        "raw": raw,
        "validated": validated_marker,
        "source_lock": lock_verification,
    }

    tamper_cases = (
        ("schema", "wrong", "schema mismatch"),
        ("mode", "quick", "mode mismatch"),
        ("all_gates_passed", False, "did not pass"),
        ("source_lock_sha256", "wrong", "different source lock"),
        ("gates", {"missing": True}, "key set/order"),
        (
            "gates",
            {
                name: name != tournament.DEVELOPMENT_GATE_NAMES[0]
                for name in tournament.DEVELOPMENT_GATE_NAMES
            },
            "failed required check",
        ),
        ("gate_policy", {}, "policy"),
        ("source_lock_verification", {"passed": True}, "source-lock verification"),
        ("raw_artifact_relative_path", "nested/../development.json", "canonical"),
        ("raw_artifact_sha256", "wrong", "raw artifact hash"),
    )
    for key, value, message in tamper_cases:
        tampered = {**gate, key: value}
        gate_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(RuntimeError, match=message):
            tournament._load_development_gate(gate_path, lock, lock_hash)

    for provenance_key, value in (
        ("source_lock_enforced", False),
        ("source_lock_relative_path", "copy.json"),
        ("runtime", {}),
        ("seed_collision_audit", {}),
    ):
        tampered_raw = copy.deepcopy(raw)
        tampered_raw["provenance"][provenance_key] = value
        raw_path.write_text(json.dumps(tampered_raw), encoding="utf-8")
        tampered_gate = {
            **gate,
            "raw_artifact_sha256": tournament._sha256(raw_path),
        }
        gate_path.write_text(json.dumps(tampered_gate), encoding="utf-8")
        with pytest.raises(RuntimeError, match="not the exact locked run"):
            tournament._load_development_gate(gate_path, lock, lock_hash)

    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    nonreproducing_gate = {
        **gate,
        "diagnostics": {"recomputed_from_raw": False},
        "raw_artifact_sha256": tournament._sha256(raw_path),
    }
    gate_path.write_text(json.dumps(nonreproducing_gate), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not recompute for diagnostics"):
        tournament._load_development_gate(gate_path, lock, lock_hash)


def test_protocol_records_outcome_blind_development_and_raw_only_analysis():
    protocol = tournament._protocol("development")
    assert protocol["paired_seeds"] == [20_100, 20_101, 20_102]
    assert protocol["transition_budget"] == 200_000
    assert protocol["raw_only"] is True
    assert "u16 minus p(1-p)" in protocol["primary"]
    assert protocol["primary_sesoi"] == 0.01
    assert protocol["rng_domain_contract"]["rng_domain_offsets"] == (
        tournament.RNG_DOMAIN_OFFSETS
    )
