"""Fail-closed orchestration tests for the Acrobot hindsight V4 runner.

All records in this module are synthetic.  Registered V4 seeds are represented
only as integer labels; no Gymnasium rollout is executed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("gymnasium")

from frontier_rl.examples import run_acrobot_hindsight_v4 as v4


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_source_lock_requires_exact_runtime_sources_schedule_and_file_hash(
    monkeypatch, tmp_path
):
    runtime = {"runtime": "synthetic"}
    sources = {"synthetic-source.py": "abc123"}
    schedule = v4._stage_a_schedule()
    monkeypatch.setattr(v4, "SOURCE_RELATIVE_PATHS", tuple(sources))
    monkeypatch.setattr(v4, "_runtime", lambda: runtime)
    monkeypatch.setattr(v4, "_source_hashes", lambda: dict(sources))

    lock_path = tmp_path / "lock.json"
    lock = {
        "schema": v4.LOCK_SCHEMA_A,
        "v4_stage": "stage_a_feasibility",
        "runtime": runtime,
        "source_sha256": sources,
        "registered_schedule": schedule,
    }
    _write_json(lock_path, lock)
    lock_hash = v4._sha256(lock_path)

    _, observed_hash = v4._verify_source_lock(
        lock_path,
        schema=v4.LOCK_SCHEMA_A,
        v4_stage="stage_a_feasibility",
        schedule=schedule,
        expected_lock_sha256=lock_hash,
    )
    assert observed_hash == lock_hash

    wrong_schedule = copy.deepcopy(schedule)
    wrong_schedule["paired_seeds"] = [99]
    with pytest.raises(RuntimeError, match="schedule"):
        v4._verify_source_lock(
            lock_path,
            schema=v4.LOCK_SCHEMA_A,
            v4_stage="stage_a_feasibility",
            schedule=wrong_schedule,
        )

    monkeypatch.setattr(
        v4, "_source_hashes", lambda: {"synthetic-source.py": "changed"}
    )
    with pytest.raises(RuntimeError, match="source bytes"):
        v4._verify_source_lock(
            lock_path,
            schema=v4.LOCK_SCHEMA_A,
            v4_stage="stage_a_feasibility",
            schedule=schedule,
        )

    with pytest.raises(RuntimeError, match="lock file hash"):
        v4._verify_source_lock(
            lock_path,
            schema=v4.LOCK_SCHEMA_A,
            v4_stage="stage_a_feasibility",
            schedule=schedule,
            expected_lock_sha256="0" * 64,
        )


def _stage_a_artifact(update_counts: list[int], *, shadow_passed: bool = True) -> dict:
    assert len(update_counts) == len(v4.STAGE_A_CASES) * len(v4.STAGE_A_SEEDS)
    cursor = 0
    cases = {}
    for condition in v4._conditions(v4.HINDSIGHT_SCALES_A):
        runs = []
        for seed in v4.STAGE_A_SEEDS:
            runs.append(
                {
                    "seed": seed,
                    "numeric_valid": True,
                    "optimizer_updates": update_counts[cursor],
                }
            )
            cursor += 1
        cases[condition.name] = {"config": v4.asdict(condition), "runs": runs}
    return {
        "schema": v4.SCHEMA_A,
        "protocol": v4._protocol("stage_a_feasibility", v4._stage_a_schedule()),
        "cases": cases,
        "preview_shadow_equivalence": {"passed": shadow_passed},
    }


def _patch_synthetic_stage_a_prefixes(monkeypatch, observed_targets: list[int]) -> None:
    monkeypatch.setattr(v4, "_stage_a_lock_check", lambda artifact, path: (True, None))

    def fake_prefix(run: dict, target: int) -> dict:
        observed_targets.append(target)
        groups = [
            {
                "group": 1,
                "regime": "dead",
                "transition_start": 210_000,
                "transition_end": 211_000,
                "teacher_tv_from_uniform": 0.20,
            },
            {
                "group": 2,
                "regime": "mixed",
                "transition_start": 211_000,
                "transition_end": 212_000,
                "teacher_tv_from_uniform": 0.20,
            },
            {
                "group": 3,
                "regime": "all_pass",
                "transition_start": 212_000,
                "transition_end": 213_000,
                "teacher_tv_from_uniform": 0.20,
            },
        ]
        candidates = list(range(10, 20))
        previews = [
            {
                "after_group": group_id,
                "gradient_norm": 1.0,
                "mutated": False,
                "applied": False,
                "frozen_group_parameters": True,
            }
            for group_id in candidates
        ]
        return {
            "target_updates": target,
            "terminal_group": 20,
            "terminal_transitions": 213_000,
            "groups": groups,
            "updates": [],
            "candidate_groups": candidates,
            "previews": previews,
            "evaluation_indices": [],
            "wall_seconds": 1.0,
            "source_step_norms": {
                "requested_live": {"count": target},
                "hindsight_relabel": {"count": 0},
            },
        }

    def fake_prefix_invariants(prefix: dict, run: dict) -> list[str]:
        zeros = [0] * len(v4.locked.THRESHOLDS)
        prefix.update(
            {
                "recomputed_task_groups": zeros,
                "recomputed_task_rollouts": zeros,
                "recomputed_task_successes": zeros,
                "recomputed_task_transitions": zeros,
            }
        )
        return []

    monkeypatch.setattr(v4, "_prefix", fake_prefix)
    monkeypatch.setattr(v4, "_prefix_errors", lambda prefix, run: [])
    monkeypatch.setattr(
        v4, "_stage_a_prefix_invariant_errors", fake_prefix_invariants
    )


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([400] * 9, 400),
        ([400] * 8 + [399], 250),
        ([400] * 8 + [249], None),
    ],
)
def test_stage_a_selects_exact_400_250_or_stop(
    monkeypatch, tmp_path, counts, expected
):
    observed_targets: list[int] = []
    _patch_synthetic_stage_a_prefixes(monkeypatch, observed_targets)
    gates = v4.compute_stage_a_gates(
        _stage_a_artifact(counts), tmp_path / "unused-lock.json"
    )
    assert gates["selected_update_budget"] == expected
    assert gates["gate_2_exact_selected_prefix"]["selected_update_budget"] == expected
    assert gates["stage_b_authorized"] is (expected is not None)
    if expected is None:
        assert observed_targets == []
    else:
        assert observed_targets == [expected] * 9


def test_stage_a_preview_shadow_is_a_required_fail_closed_gate(monkeypatch, tmp_path):
    observed_targets: list[int] = []
    _patch_synthetic_stage_a_prefixes(monkeypatch, observed_targets)
    gates = v4.compute_stage_a_gates(
        _stage_a_artifact([400] * 9, shadow_passed=False),
        tmp_path / "unused-lock.json",
    )
    assert gates["gate_2_exact_selected_prefix"]["passed"]
    assert not gates["gate_3_preview_mechanics_and_shadow"]["passed"]
    assert not gates["all_pass"]
    assert not gates["stage_b_authorized"]


def _stage_b_readiness_artifact(target: int) -> dict:
    return {
        "cases": {
            case_name: {
                "runs": [
                    {
                        "seed": seed,
                        "numeric_valid": True,
                        "transition_cap_censored": False,
                        "reached_optimizer_update_budget": True,
                        "optimizer_updates": target,
                        "x_optimizer_updates": [0, target],
                    }
                    for seed in v4.STAGE_B_SEEDS
                ]
            }
            for case_name in v4.STAGE_B_CASES
        }
    }


def test_stage_b_cap_censoring_invalidates_the_entire_primary_analysis(monkeypatch):
    target = 250
    artifact = _stage_b_readiness_artifact(target)
    monkeypatch.setattr(v4, "_full_run_errors", lambda run: [])
    ready, errors = v4._stage_b_runs_ready(artifact, target)
    assert ready and errors == []

    artifact["cases"][v4.STAGE_B_CASES[0]]["runs"][0][
        "transition_cap_censored"
    ] = True
    artifact["cases"][v4.STAGE_B_CASES[0]]["runs"][0][
        "reached_optimizer_update_budget"
    ] = False
    ready, errors = v4._stage_b_runs_ready(artifact, target)
    assert not ready
    assert any("transition-cap censored" in error for error in errors)
    assert any("did not reach frozen update target" in error for error in errors)


def _stage_b_analysis_artifact(values: dict[str, float], target: int = 250) -> dict:
    return {
        "protocol": {
            "registered_schedule": {"optimizer_update_target": target}
        },
        "provenance": {"source_lock_sha256": "synthetic-lock"},
        "cases": {
            case_name: {
                "runs": [
                    {
                        "seed": seed,
                        "synthetic_auc": values[case_name],
                        "auc_mean_pass_by_optimizer_updates": values[case_name],
                        "update_diagnostics": [],
                        "transitions": 1_000,
                    }
                    for seed in v4.STAGE_B_SEEDS
                ]
            }
            for case_name in v4.STAGE_B_CASES
        },
    }


def _contrast_cell_values(*, c1: float = 0.04) -> dict[str, float]:
    values = {
        v4._case_name(0.5, 0.0): 0.100,
        v4._case_name(0.5, 1.0): 0.130,
        v4._case_name(0.5, 2.0): 0.175,
        v4._case_name(1.0, 0.0): 0.200,
        v4._case_name(1.0, 1.0): 0.200 + c1,
        v4._case_name(1.0, 2.0): 0.290,
        v4._case_name(2.0, 0.0): 0.300,
        v4._case_name(2.0, 1.0): 0.355,
        v4._case_name(2.0, 2.0): 0.400,
    }
    assert set(values) == set(v4.STAGE_B_CASES)
    return values


def _patch_fast_stage_b_analysis(monkeypatch):
    monkeypatch.setattr(v4, "_verify_source_lock", lambda *args, **kwargs: ({}, "h"))
    monkeypatch.setattr(v4, "_stage_b_runs_ready", lambda artifact, target: (True, []))
    monkeypatch.setattr(
        v4, "_recomputed_update_auc", lambda run, target: run["synthetic_auc"]
    )
    monkeypatch.setattr(
        v4.locked,
        "bootstrap_mean_ci",
        lambda values, **kwargs: [float(np.mean(values)), float(np.mean(values))],
    )


def test_stage_b_uses_exact_four_contrasts_holm_and_material_decisions(
    monkeypatch, tmp_path
):
    _patch_fast_stage_b_analysis(monkeypatch)
    artifact = _stage_b_analysis_artifact(_contrast_cell_values())
    v4._attach_stage_b_analysis(
        artifact, tmp_path / "lock.json", tmp_path / "amendment.json"
    )

    expected_coefficients = {
        "C1": {
            v4._case_name(1.0, 1.0): 1.0,
            v4._case_name(1.0, 0.0): -1.0,
        },
        "C2": {
            v4._case_name(1.0, 2.0): 1.0,
            v4._case_name(1.0, 1.0): -1.0,
        },
        "C3": {
            v4._case_name(0.5, 2.0): 1.0,
            v4._case_name(0.5, 0.0): -1.0,
            v4._case_name(1.0, 1.0): -1.0,
            v4._case_name(1.0, 0.0): 1.0,
        },
        "C4": {
            v4._case_name(1.0, 2.0): 1.0,
            v4._case_name(1.0, 0.0): -1.0,
            v4._case_name(2.0, 1.0): -1.0,
            v4._case_name(2.0, 0.0): 1.0,
        },
    }
    assert artifact["scale_multiplicity"]["family"] == ["C1", "C2", "C3", "C4"]
    for name, coefficients in expected_coefficients.items():
        contrast = artifact["paired_scale_contrasts"][name]
        assert contrast["coefficients"] == coefficients
        assert contrast["reject_familywise_0.05"]
        assert contrast["holm_adjusted_p"] <= 0.05

    decision = artifact["predeclared_scale_decision"]
    assert decision["directional_minimum_mean"] == pytest.approx(0.03)
    assert decision["restricted_departure_minimum_absolute_mean"] == pytest.approx(
        0.03
    )
    assert decision["C1_directional_local_improvement_supported"]
    assert decision["C2_directional_increment_supported"]
    assert decision["C3_material_restricted_separability_departure"]
    assert decision["C4_material_restricted_separability_departure"]

    below_material = _stage_b_analysis_artifact(_contrast_cell_values(c1=0.029))
    v4._attach_stage_b_analysis(
        below_material, tmp_path / "lock.json", tmp_path / "amendment.json"
    )
    assert below_material["paired_scale_contrasts"]["C1"][
        "reject_familywise_0.05"
    ]
    assert not below_material["predeclared_scale_decision"][
        "C1_directional_local_improvement_supported"
    ]


def test_resume_keeps_existing_seed_records_and_never_replaces_them(
    monkeypatch, tmp_path
):
    runtime = {"runtime": "synthetic"}
    sources = {"source": "hash"}
    lock_hash = "a" * 64
    lock_path = tmp_path / "lock.json"
    output = tmp_path / "stage-a.json"
    monkeypatch.setattr(v4, "_runtime", lambda: runtime)
    monkeypatch.setattr(v4, "_source_hashes", lambda: dict(sources))
    monkeypatch.setattr(
        v4, "_verify_source_lock", lambda *args, **kwargs: ({}, lock_hash)
    )
    monkeypatch.setattr(
        v4,
        "compute_stage_a_gates",
        lambda artifact, path: {
            "all_pass": True,
            "stage_b_authorized": True,
            "selected_update_budget": 400,
        },
    )

    artifact = v4._new_artifact(
        stage="stage_a_feasibility",
        schedule=v4._stage_a_schedule(),
        lock_path=lock_path,
        lock_hash=lock_hash,
    )
    artifact["preview_shadow_equivalence"] = {"passed": True}
    artifact["cases"][v4.STAGE_A_CASES[0]]["runs"].append(
        {"seed": v4.STAGE_A_SEEDS[0], "numeric_valid": True, "marker": "retain"}
    )
    v4._atomic_write(output, artifact, must_not_exist=True)

    calls: list[tuple[str, int]] = []

    def fake_run(condition, seed, **kwargs):
        calls.append((condition.name, seed))
        return {"seed": seed, "numeric_valid": True, "marker": "new"}

    monkeypatch.setattr(v4, "_instrumented_run", fake_run)
    completed = v4._run_stage(
        stage="stage_a_feasibility",
        lock_path=lock_path,
        output=output,
        resume=True,
        selected_update_budget=400,
    )
    retained = completed["cases"][v4.STAGE_A_CASES[0]]["runs"][0]
    assert retained == {
        "seed": v4.STAGE_A_SEEDS[0],
        "numeric_valid": True,
        "marker": "retain",
    }
    assert (v4.STAGE_A_CASES[0], v4.STAGE_A_SEEDS[0]) not in calls
    assert len(calls) == 8

    calls.clear()
    terminal = v4._run_stage(
        stage="stage_a_feasibility",
        lock_path=lock_path,
        output=output,
        resume=True,
        selected_update_budget=400,
    )
    assert terminal["artifact_state"] == "complete"
    assert calls == []
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        v4._run_stage(
            stage="stage_a_feasibility",
            lock_path=lock_path,
            output=output,
            resume=False,
            selected_update_budget=400,
        )


def test_authorization_requires_a_passing_hash_bound_independent_report(
    monkeypatch, tmp_path
):
    stage_a_path = tmp_path / "stage-a.json"
    verification_path = tmp_path / "verification.json"
    amendment_path = tmp_path / "amendment.json"
    lock_path = tmp_path / "stage-b-lock.json"
    saved_gates = {
        "all_pass": True,
        "stage_b_authorized": True,
        "selected_update_budget": 250,
    }
    stage_a = {
        "artifact_state": "complete",
        "provenance": {
            "source_lock_path": str(tmp_path / "stage-a-lock.json"),
            "source_lock_sha256": "stage-a-lock-hash",
        },
        "stage_a_effect_blind_gates": saved_gates,
    }
    _write_json(stage_a_path, stage_a)
    monkeypatch.setattr(
        v4, "compute_stage_a_gates", lambda artifact, lock: copy.deepcopy(saved_gates)
    )

    with pytest.raises(FileNotFoundError, match="verification"):
        v4._authorize_stage_b(
            stage_a_artifact_path=stage_a_path,
            stage_a_verification_path=verification_path,
            amendment_output=amendment_path,
            lock_output=lock_path,
            explicit_authorization=True,
        )

    report = {
        "schema": "curriculum-maxrl/acrobot-hindsight-v4a-verification/v1",
        "v4_stage": "stage_a_feasibility",
        "all_checks_passed": False,
        "stage_b_factorial_authorized": True,
        "selected_optimizer_update_budget": 250,
        "artifact_sha256": v4._sha256(stage_a_path),
        "lock_sha256": "stage-a-lock-hash",
        "artifact": str(stage_a_path.resolve()),
    }
    _write_json(verification_path, report)
    with pytest.raises(RuntimeError, match="verification did not pass"):
        v4._authorize_stage_b(
            stage_a_artifact_path=stage_a_path,
            stage_a_verification_path=verification_path,
            amendment_output=amendment_path,
            lock_output=lock_path,
            explicit_authorization=True,
        )

    report["all_checks_passed"] = True
    report["artifact_sha256"] = "tampered-artifact-hash"
    _write_json(verification_path, report)
    with pytest.raises(RuntimeError, match="different Stage-A hash"):
        v4._authorize_stage_b(
            stage_a_artifact_path=stage_a_path,
            stage_a_verification_path=verification_path,
            amendment_output=amendment_path,
            lock_output=lock_path,
            explicit_authorization=True,
        )
    assert not amendment_path.exists()
    assert not lock_path.exists()


def test_cli_and_authorization_refuse_missing_verification_or_amendment(
    monkeypatch, tmp_path, capsys
):
    stage_a = tmp_path / "stage-a.json"
    amendment = tmp_path / "amendment.json"
    lock = tmp_path / "lock.json"
    _write_json(stage_a, {})

    with pytest.raises(SystemExit):
        v4.main(
            [
                "authorize-b",
                "--stage-a-artifact",
                str(stage_a),
                "--amendment-output",
                str(amendment),
                "--lock-output",
                str(lock),
                "--authorize-stage-b",
            ]
        )
    assert "--stage-a-verification" in capsys.readouterr().err

    called = False

    def forbidden_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("registered training must not start")

    monkeypatch.setattr(v4, "_instrumented_run", forbidden_run)
    with pytest.raises(SystemExit):
        v4.main(
            [
                "stage-b",
                "--stage-a-artifact",
                str(stage_a),
                "--amendment",
                str(tmp_path / "missing-amendment.json"),
                "--lock",
                str(lock),
                "--output",
                str(tmp_path / "stage-b.json"),
            ]
        )
    assert not called
