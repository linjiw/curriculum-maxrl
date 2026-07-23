"""Deterministic/synthetic tests for the V5 hindsight runner.

No registered V5 seed is executed by this module.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("gymnasium")

from frontier_rl.adapters.acrobot_neural import (
    AcrobotNeuralSpace,
    AcrobotTransition,
    TanhCategoricalActor,
    tip_height,
)
from frontier_rl.interfaces import GroupResult
from frontier_rl.examples import run_acrobot_hindsight_v5 as v5


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_frozen_schedule_engine_seed_and_source_contracts():
    assert v5.STAGE_A_SEEDS == (15_000, 15_001, 15_002)
    assert v5.STAGE_B_SEEDS == tuple(range(16_000, 16_020))
    assert len(v5.STAGE_A_CASES) == 9
    assert v5.CASES == tuple(
        v5._case_name(multiplier, scale)
        for multiplier in (0.5, 1.0, 2.0)
        for scale in (0.0, 1.0, 2.0)
    )
    assert len(v5.CASES) * len(v5.STAGE_A_SEEDS) == 27
    assert len(v5.CASES) * len(v5.STAGE_B_SEEDS) == 180
    assert v5._stage_a_schedule()["fresh_budget_selection_population"] == "all 27 V5A runs"
    assert v5._stage_b_schedule(250)["run_count"] == 180
    assert v5._engine_contract()["instrumentation_checkpoints"] == [250, 400]
    assert v5._seed_collision_audit()["passed"]
    assert {
        "frontier_rl/__init__.py",
        "frontier_rl/trainer.py",
        "frontier_rl/adapters/__init__.py",
    } <= set(v5.SOURCE_RELATIVE_PATHS)


def test_budget_tv_runtime_and_materiality_boundaries_are_inclusive_or_strict_as_registered():
    assert v5._select_update_budget([400] * 27) == 400
    assert v5._select_update_budget([400] * 26 + [399]) == 250
    assert v5._select_update_budget([400] * 26 + [249]) is None
    assert v5._select_update_budget([400] * 26) is None
    assert not v5._teacher_tv_pass([0.05])
    assert v5._teacher_tv_pass([np.nextafter(0.05, 1.0)])
    assert v5._teacher_tv_pass([0.05, np.nextafter(0.05, 1.0)])
    assert v5._projected_hours_180(360.0) == 18.0
    assert v5._projected_hours_180(np.nextafter(360.0, np.inf)) > 18.0
    contrasts = {
        "C1": {"mean_contrast": 0.03, "reject_familywise_0.05": True},
        "C2": {"mean_contrast": np.nextafter(0.03, 0.0), "reject_familywise_0.05": True},
        "C3": {"mean_contrast": -0.03, "reject_familywise_0.05": True},
        "C4": {"mean_contrast": 0.03, "reject_familywise_0.05": False},
    }
    decision = v5._predeclared_scale_decision(contrasts)
    assert decision["C1_directional_local_improvement_supported"]
    assert not decision["C2_directional_increment_supported"]
    assert decision["C3_material_restricted_separability_departure"]
    assert not decision["C4_material_restricted_separability_departure"]
    assert set(v5.CONTRAST_SPECS) == {"C1", "C2", "C3", "C4"}
    assert v5.CONTRAST_SPECS["C1"] == {
        v5._case_name(1.0, 1.0): 1.0,
        v5._case_name(1.0, 0.0): -1.0,
    }


def test_seed_collision_audit_raises_fail_closed(monkeypatch):
    monkeypatch.setattr(v5, "STAGE_A_SEEDS", (100, 15_001, 15_002))
    with pytest.raises(RuntimeError, match="collision"):
        v5._seed_collision_audit()


def _synthetic_stage_a_artifact(counts: list[int]) -> dict:
    assert len(counts) == 27
    cursor = 0
    cases = {}
    for condition in v5._conditions():
        runs = []
        for seed in v5.STAGE_A_SEEDS:
            runs.append(
                {
                    "seed": seed,
                    "numeric_valid": True,
                    "optimizer_updates": counts[cursor],
                    "_scale": condition.hindsight_scale,
                    "_tv": np.nextafter(0.05, 1.0),
                    "_wall": 360.0,
                    "_natural": True,
                    "_all_regimes": True,
                }
            )
            cursor += 1
        cases[condition.name] = {"config": v5.asdict(condition), "runs": runs}
    return {
        "schema": v5.SCHEMA_A,
        "protocol": v5._protocol(
            "stage_a_natural_feasibility", v5._stage_a_schedule()
        ),
        "deterministic_scale_fixture": v5._deterministic_scale_fixture(),
        "preview_shadow_equivalence": {"passed": True},
        "cases": cases,
    }


def _patch_synthetic_prefixes(monkeypatch):
    monkeypatch.setattr(v5, "_stage_a_lock_check", lambda artifact, path: (True, None))
    monkeypatch.setattr(v5, "_prefix_technical_errors", lambda prefix, run: [])

    def fake_prefix(run: dict, target: int) -> dict:
        regimes = ["dead", "mixed", "all_pass"] if run["_all_regimes"] else ["dead", "mixed"]
        groups = []
        for index, regime in enumerate(regimes, start=1):
            groups.append(
                {
                    "group": index,
                    "task_id": 2,
                    "regime": regime,
                    "transition_start": 200_000 + index,
                    "transition_end": 200_001 + index,
                    "teacher_tv_from_uniform": run["_tv"],
                }
            )
        candidates = [1] if run["_natural"] else []
        if run["_scale"] == 0.0:
            previews = [
                {
                    "after_group": 1,
                    "transitions": 200_002,
                    "requested_task": 2,
                    "credited_task": 1,
                    "gradient_norm": 1.0,
                    "hypothetical_update_norm": 0.001,
                    "applied": False,
                    "mutated": False,
                    "frozen_group_parameters": True,
                }
            ] if candidates else []
            updates = []
            source_norms = {
                "requested_live": {"count": target},
                "hindsight_relabel": {"count": 0},
            }
        else:
            previews = []
            updates = [
                {
                    "after_group": 1,
                    "transitions": 200_002,
                    "source": "hindsight_relabel",
                    "requested_task": 2,
                    "credited_task": 1,
                    "gradient_norm": 1.0,
                    "update_norm": 0.001,
                }
            ] if candidates else []
            source_norms = {
                "requested_live": {"count": target - len(updates)},
                "hindsight_relabel": {"count": len(updates)},
            }
        return {
            "target_updates": target,
            "terminal_group": len(groups),
            "terminal_transitions": groups[-1]["transition_end"],
            "groups": groups,
            "updates": updates,
            "candidate_groups": candidates,
            "previews": previews,
            "evaluation_indices": [],
            "wall_seconds": run["_wall"],
            "source_step_norms": source_norms,
        }

    monkeypatch.setattr(v5, "_prefix", fake_prefix)


def test_all_27_stage_a_gate_branches_boundaries_and_outcome_invariance(monkeypatch, tmp_path):
    _patch_synthetic_prefixes(monkeypatch)
    artifact = _synthetic_stage_a_artifact([400] * 27)
    gates = v5.compute_stage_a_gates(artifact, tmp_path / "unused.json")
    assert gates["selected_update_budget"] == 400
    assert gates["all_pass"]

    outcome_mutation = copy.deepcopy(artifact)
    for record in outcome_mutation["cases"].values():
        for run in record["runs"]:
            run["mean_pass_curve"] = [0.999]
            run["final_mean_pass"] = -123.0
    assert v5.compute_stage_a_gates(outcome_mutation, tmp_path / "unused.json") == gates

    fallback = _synthetic_stage_a_artifact([400] * 26 + [399])
    assert v5.compute_stage_a_gates(fallback, tmp_path / "unused.json")[
        "selected_update_budget"
    ] == 250
    stop = _synthetic_stage_a_artifact([400] * 26 + [249])
    assert v5.compute_stage_a_gates(stop, tmp_path / "unused.json")[
        "selected_update_budget"
    ] is None

    tv_boundary = _synthetic_stage_a_artifact([400] * 27)
    for record in tv_boundary["cases"].values():
        for run in record["runs"]:
            run["_tv"] = 0.05
    assert not v5.compute_stage_a_gates(tv_boundary, tmp_path / "unused.json")[
        "gate_6_per_run_teacher_tv"
    ]["passed"]

    runtime_boundary = _synthetic_stage_a_artifact([400] * 27)
    runtime_boundary["cases"][v5.CASES[0]]["runs"][0]["_wall"] = np.nextafter(
        360.0, np.inf
    )
    assert not v5.compute_stage_a_gates(runtime_boundary, tmp_path / "unused.json")[
        "gate_7_serial_runtime_projection"
    ]["passed"]

    no_relevance = _synthetic_stage_a_artifact([400] * 27)
    for run in no_relevance["cases"][v5.CASES[0]]["runs"]:
        run["_natural"] = False
    assert not v5.compute_stage_a_gates(no_relevance, tmp_path / "unused.json")[
        "gate_4_positive_updates_and_natural_relabel_coverage"
    ]["passed"]

    missing_regime = _synthetic_stage_a_artifact([400] * 27)
    for run in missing_regime["cases"][v5.CASES[0]]["runs"]:
        run["_all_regimes"] = False
    assert not v5.compute_stage_a_gates(missing_regime, tmp_path / "unused.json")[
        "gate_5_per_cell_natural_regimes"
    ]["passed"]


def test_source_lock_requires_exact_runtime_engine_seed_schedule_and_hash(monkeypatch, tmp_path):
    runtime = {"runtime": "fixture"}
    sources = {"fixture.py": "abc"}
    engine = {"engine": "fixture"}
    seed_audit = {"passed": True}
    schedule = v5._stage_a_schedule()
    monkeypatch.setattr(v5, "SOURCE_RELATIVE_PATHS", tuple(sources))
    monkeypatch.setattr(v5, "_source_hashes", lambda: dict(sources))
    monkeypatch.setattr(v5, "_runtime", lambda: runtime)
    monkeypatch.setattr(v5, "_engine_contract", lambda: engine)
    monkeypatch.setattr(v5, "_seed_collision_audit", lambda: seed_audit)
    path = tmp_path / "lock.json"
    lock = {
        "schema": v5.LOCK_SCHEMA_A,
        "v5_stage": "stage_a_natural_feasibility",
        "runtime": runtime,
        "engine_contract": engine,
        "seed_collision_audit": seed_audit,
        "registered_schedule": schedule,
        "source_sha256": sources,
    }
    _write(path, lock)
    _, digest = v5._verify_source_lock(
        path,
        schema=v5.LOCK_SCHEMA_A,
        v5_stage="stage_a_natural_feasibility",
        schedule=schedule,
    )
    assert digest == v5._sha256(path)
    wrong_schedule = copy.deepcopy(schedule)
    wrong_schedule["paired_seeds"] = [99]
    with pytest.raises(RuntimeError, match="schedule"):
        v5._verify_source_lock(
            path,
            schema=v5.LOCK_SCHEMA_A,
            v5_stage="stage_a_natural_feasibility",
            schedule=wrong_schedule,
        )
    monkeypatch.setattr(v5, "_runtime", lambda: {"runtime": "changed"})
    with pytest.raises(RuntimeError, match="runtime"):
        v5._verify_source_lock(
            path,
            schema=v5.LOCK_SCHEMA_A,
            v5_stage="stage_a_natural_feasibility",
            schedule=schedule,
        )
    monkeypatch.setattr(v5, "_runtime", lambda: runtime)
    monkeypatch.setattr(v5, "_engine_contract", lambda: {"engine": "changed"})
    with pytest.raises(RuntimeError, match="engine"):
        v5._verify_source_lock(
            path,
            schema=v5.LOCK_SCHEMA_A,
            v5_stage="stage_a_natural_feasibility",
            schedule=schedule,
        )
    monkeypatch.setattr(v5, "_engine_contract", lambda: engine)
    monkeypatch.setattr(v5, "_source_hashes", lambda: {"fixture.py": "changed"})
    with pytest.raises(RuntimeError, match="source bytes"):
        v5._verify_source_lock(
            path,
            schema=v5.LOCK_SCHEMA_A,
            v5_stage="stage_a_natural_feasibility",
            schedule=schedule,
        )
    monkeypatch.setattr(v5, "_source_hashes", lambda: dict(sources))
    with pytest.raises(RuntimeError, match="lock file hash"):
        v5._verify_source_lock(
            path,
            schema=v5.LOCK_SCHEMA_A,
            v5_stage="stage_a_natural_feasibility",
            schedule=schedule,
            expected_lock_sha256="0" * 64,
        )
    monkeypatch.setattr(v5, "_seed_collision_audit", lambda: {"passed": False})
    with pytest.raises(RuntimeError, match="seed collision"):
        v5._verify_source_lock(
            path,
            schema=v5.LOCK_SCHEMA_A,
            v5_stage="stage_a_natural_feasibility",
            schedule=schedule,
        )


def test_nine_cell_fixture_checks_delta_theta_scale_zero_and_counters():
    fixture = v5._deterministic_scale_fixture()
    assert fixture["passed"]
    assert set(fixture["cells"]) == set(v5.CASES)
    for cell in fixture["cells"].values():
        assert cell["delta_theta_equals_base_lr_times_a_times_s_times_g"]
        assert cell["parameter_counter_rng_rule_passed"]
        if cell["hindsight_scale"] == 0:
            assert cell["parameter_delta_norm"] == 0
            assert cell["positive_nonmutating_preview"]
        else:
            assert cell["parameter_delta_norm"] > 0
            assert cell["source"] == "hindsight_relabel"


def _prefix_run(target: int = 250) -> dict:
    groups, updates = [], []
    for index in range(1, target + 1):
        groups.append(
            {
                "group": index,
                "transition_start": index - 1,
                "transition_end": index,
                "n_transitions": 1,
                "task_id": 2,
                "success_count": 8,
                "regime": "mixed",
                "teacher_tv_from_uniform": 0.2,
                "sampled_task_probability": 0.2,
                "optimizer_updates_after_group": index,
                "update_source": "requested_live",
            }
        )
        updates.append(
            {
                "optimizer_update": index,
                "after_group": index,
                "transitions": index,
                "source": "requested_live",
                "requested_task": 2,
                "credited_task": 2,
                "gradient_norm": 1.0,
                "update_norm": 0.001,
            }
        )
    coordinates = list(range(0, target + 1, 50))
    return {
        "seed": 15_000,
        "numeric_valid": True,
        "transitions": target,
        "sampled_groups": target,
        "optimizer_updates": target,
        "live_groups": target,
        "dead_groups": 0,
        "all_pass_groups": 0,
        "live_applied_updates": target,
        "relabeled_groups": 0,
        "relabel_candidates": 0,
        "eligible_relabel_candidate_groups": [],
        "unscaled_aux_gradient_previews": 0,
        "zero_gradient_update_attempts": 0,
        "zero_gradient_diagnostics": [],
        "auxiliary_gradient_diagnostics": [],
        "group_diagnostics": groups,
        "update_diagnostics": updates,
        "x_optimizer_updates": coordinates,
        "x_transitions": coordinates,
        "evaluation_rng_preserved": [True] * len(coordinates),
        "wall_seconds_at_optimizer_updates": {str(target): 1.0},
        "wall_seconds": 1.5,
        "total_parameters": 640,
        "active_parameters_per_task": 640,
        "pass_rate_curve": [[0.0] * 8 for _ in coordinates],
        "mean_pass_curve": [0.0] * len(coordinates),
    }


def test_prefix_isolation_ignores_saved_postprefix_fields_but_rejects_selected_corruption():
    run = _prefix_run()
    prefix = v5._prefix(run, 250)
    assert v5._prefix_technical_errors(prefix, run) == []

    post = copy.deepcopy(run)
    post["wall_seconds"] = -999
    post["final_mean_pass"] = float("nan")
    post["group_diagnostics"].append(
        {
            "group": 251,
            "transition_start": 250,
            "transition_end": 999_999,
            "n_transitions": -1,
            "optimizer_updates_after_group": 999,
        }
    )
    # First exact prefix, including its decision-relevant digest, is unchanged.
    post_prefix = v5._prefix(post, 250)
    assert post_prefix == prefix
    assert v5._prefix_technical_errors(post_prefix, post) == []

    selected = copy.deepcopy(run)
    selected["update_diagnostics"][9]["after_group"] = 9
    errors = v5._prefix_technical_errors(v5._prefix(selected, 250), selected)
    assert any("duplicate or out of range" in error for error in errors)


def test_prefix_rejects_bad_evaluation_coordinates_and_wrong_source_regime():
    run = _prefix_run()
    run["x_transitions"][2] = run["x_transitions"][1]
    assert any("evaluation transition" in error for error in v5._prefix_technical_errors(v5._prefix(run, 250), run))
    run = _prefix_run()
    run["group_diagnostics"][0]["regime"] = "dead"
    run["group_diagnostics"][0]["success_count"] = 0
    assert any("requested-live source" in error for error in v5._prefix_technical_errors(v5._prefix(run, 250), run))
    run = _prefix_run()
    run["eligible_relabel_candidate_groups"] = [1, 251, 2]
    assert v5._prefix(run, 250)["candidate_groups"] == [1, 2]


def test_scale_aware_full_relabel_validation_reconstructs_one_to_one_sources():
    scale_zero = _prefix_run(1)
    scale_zero["group_diagnostics"][0].update(
        success_count=0, regime="dead", update_source=None, optimizer_updates_after_group=0
    )
    scale_zero["optimizer_updates"] = 0
    scale_zero["live_applied_updates"] = 0
    scale_zero["update_diagnostics"] = []
    scale_zero["relabel_candidates"] = 1
    scale_zero["eligible_relabel_candidate_groups"] = [1]
    scale_zero["unscaled_aux_gradient_previews"] = 1
    scale_zero["auxiliary_gradient_diagnostics"] = [
        {
            "after_group": 1,
            "transitions": 1,
            "requested_task": 2,
            "credited_task": 1,
            "gradient_norm": 1.0,
            "hypothetical_update_norm": 0.001,
            "applied": False,
            "mutated": False,
            "frozen_group_parameters": True,
        }
    ]
    assert v5._full_relabel_errors(scale_zero, 0.0) == []
    scale_zero["auxiliary_gradient_diagnostics"][0]["mutated"] = True
    assert v5._full_relabel_errors(scale_zero, 0.0)

    positive = copy.deepcopy(scale_zero)
    positive["unscaled_aux_gradient_previews"] = 0
    positive["auxiliary_gradient_diagnostics"] = []
    positive["relabeled_groups"] = 1
    positive["update_diagnostics"] = [
        {
            "after_group": 1,
            "transitions": 1,
            "source": "hindsight_relabel",
            "gradient_norm": 1.0,
            "update_norm": 0.001,
            "requested_task": 2,
            "credited_task": 1,
        }
    ]
    assert v5._full_relabel_errors(positive, 1.0) == []
    positive["update_diagnostics"] = []
    assert v5._full_relabel_errors(positive, 1.0)
    positive["relabeled_groups"] = 0
    positive["group_diagnostics"][0]["update_source"] = None
    positive["zero_gradient_update_attempts"] = 1
    positive["zero_gradient_diagnostics"] = [
        {
            "after_group": 1,
            "transitions": 1,
            "source": "hindsight_relabel",
            "requested_task": 2,
            "credited_task": 1,
            "gradient_norm": 0.0,
            "update_norm": 0.0,
        }
    ]
    assert v5._full_relabel_errors(positive, 1.0) == []
    positive["zero_gradient_diagnostics"][0]["credited_task"] = 2
    assert v5._full_relabel_errors(positive, 1.0)


def test_scale_zero_gate_preview_binds_requested_and_credited_task_metadata():
    prefix = {
        "groups": [{"group": 1, "task_id": 2, "transition_end": 1}],
        "previews": [
            {
                "after_group": 1,
                "transitions": 1,
                "requested_task": 2,
                "credited_task": 1,
                "gradient_norm": 1.0,
                "hypothetical_update_norm": 0.001,
                "applied": False,
                "mutated": False,
                "frozen_group_parameters": True,
            }
        ],
    }
    assert v5._scale_zero_prefix_preview_valid(prefix) == (True, True)
    prefix["previews"][0]["requested_task"] = 3
    assert v5._scale_zero_prefix_preview_valid(prefix) == (True, False)


def test_teacher_observes_only_original_requested_rewards_in_base_loop():
    tree = ast.parse(inspect.getsource(v5.locked.run_condition))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "teacher"
        and node.func.attr == "observe"
    ]
    assert len(calls) == 1
    assert [argument.id for argument in calls[0].args if isinstance(argument, ast.Name)] == [
        "task_id",
        "rewards",
    ]


def _observation_for_height(height: float) -> np.ndarray:
    # t2=0 gives height=-2*cos(t1).
    cosine = -height / 2.0
    sine = float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))
    return np.asarray([cosine, sine, 1.0, 0.0, 0.0, 0.0])


def test_relabel_recomputes_verifier_and_truncates_at_first_strict_hit():
    actor = TanhCategoricalActor(seed=7, learning_rate=3e-4)
    space = AcrobotNeuralSpace(actor=actor, thresholds=v5.locked.THRESHOLDS, seed=8)
    try:
        heights = [[-1.2, -0.9, -0.4], [-1.3, -1.1, -1.05]]
        trajectories = []
        for row in heights:
            trajectory = []
            for height in row:
                observation = _observation_for_height(height)
                trajectory.append(
                    AcrobotTransition(
                        obs_before=observation,
                        action=0,
                        obs_after=observation,
                        native_reward=-1.0,
                        native_terminated=False,
                        truncated=False,
                        height_after=height,
                    )
                )
            trajectories.append(trajectory)
        group = GroupResult(
            task_id=4,
            rewards=np.zeros(2),
            trajectories=trajectories,
            infos=[{}, {}],
        )
        credited, rewards, rewritten = space.relabel(group)
        assert credited < group.task_id
        assert 0 < rewards.sum() < len(rewards)
        threshold = v5.locked.THRESHOLDS[credited]
        for reward, trajectory in zip(rewards, rewritten):
            verified = [tip_height(step.obs_after) for step in trajectory]
            if reward == 1:
                assert verified[-1] > threshold
                assert all(value <= threshold for value in verified[:-1])
    finally:
        space.close()


def test_atomic_resume_seed_prefix_never_replaces_saved_record(tmp_path):
    path = tmp_path / "artifact.json"
    saved = {"seed": 15_000, "fingerprint": "immutable"}
    payload = {"cases": {v5.CASES[0]: {"runs": [saved]}}}
    v5._atomic_write(path, payload, must_not_exist=True)
    loaded = v5._read_json(path)
    assert v5._exact_seed_prefix(
        loaded["cases"][v5.CASES[0]]["runs"], v5.STAGE_A_SEEDS, v5.CASES[0]
    ) == 1
    assert loaded["cases"][v5.CASES[0]]["runs"][0] == saved
    loaded["cases"][v5.CASES[0]]["runs"].append({"seed": 99})
    with pytest.raises(RuntimeError, match="exact seed prefix"):
        v5._exact_seed_prefix(
            loaded["cases"][v5.CASES[0]]["runs"], v5.STAGE_A_SEEDS, v5.CASES[0]
        )


def test_resume_identity_fails_closed_on_config_runtime_source_and_lock(monkeypatch):
    runtime = {"runtime": "fixture"}
    sources = {"source.py": "hash"}
    monkeypatch.setattr(v5, "_runtime", lambda: runtime)
    monkeypatch.setattr(v5, "_source_hashes", lambda: sources)
    schedule = v5._stage_a_schedule()
    artifact = {
        "schema": v5.SCHEMA_A,
        "protocol": v5._protocol("stage_a_natural_feasibility", schedule),
        "provenance": {
            "source_lock_sha256": "lock-hash",
            "runtime": runtime,
            "source_sha256": sources,
        },
        "cases": {
            condition.name: {"config": v5.asdict(condition), "runs": []}
            for condition in v5._conditions()
        },
    }
    v5._artifact_identity(
        artifact,
        stage="stage_a_natural_feasibility",
        schedule=schedule,
        lock_hash="lock-hash",
    )
    bad = copy.deepcopy(artifact)
    bad["cases"][v5.CASES[0]]["config"]["hidden_size"] = 63
    with pytest.raises(RuntimeError, match="condition"):
        v5._artifact_identity(
            bad,
            stage="stage_a_natural_feasibility",
            schedule=schedule,
            lock_hash="lock-hash",
        )
    with pytest.raises(RuntimeError, match="different source lock"):
        v5._artifact_identity(
            artifact,
            stage="stage_a_natural_feasibility",
            schedule=schedule,
            lock_hash="other",
        )


def test_module_commands_always_use_python_dash_m():
    command = v5._module_command(v5.ANALYZER_MODULE, "a.json", "--lock", "l.json")
    assert " -m frontier_rl.examples.analyze_acrobot_hindsight_v5 " in command
    completed = subprocess.run(
        [sys.executable, "-m", v5.ANALYZER_MODULE, "--help"],
        cwd=v5.PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "Independent read-only verifier" in completed.stdout


def test_stage_b_readiness_requires_exact_evaluation_coordinates(monkeypatch):
    target = 250
    artifact = {
        "cases": {
            name: {
                "config": {"hindsight_scale": float(name.rsplit("_hs_", 1)[1].replace("p", "."))},
                "runs": [
                    {
                        "seed": seed,
                        "transition_cap_censored": False,
                        "reached_optimizer_update_budget": True,
                        "optimizer_updates": target,
                        "transitions": target,
                        "x_optimizer_updates": list(range(0, target + 1, 50)),
                        "x_transitions": list(range(0, target + 1, 50)),
                    }
                    for seed in v5.STAGE_B_SEEDS
                ],
            }
            for name in v5.STAGE_B_CASES
        }
    }
    monkeypatch.setattr(v5, "_technical_run_errors", lambda run: [])
    monkeypatch.setattr(v5, "_full_relabel_errors", lambda run, scale: [])
    ready, errors = v5._stage_b_runs_ready(artifact, target)
    assert ready and not errors
    artifact["cases"][v5.STAGE_B_CASES[0]]["runs"][0]["x_optimizer_updates"][2] = 99
    ready, errors = v5._stage_b_runs_ready(artifact, target)
    assert not ready and any("evaluation updates" in error for error in errors)
