"""Focused independent-verifier tests for Acrobot hindsight V5."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("gymnasium")

from frontier_rl.examples import analyze_acrobot_hindsight_v5 as verifier
from frontier_rl.examples import run_acrobot_hindsight_v5 as runner


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_independent_fixture_schedule_protocol_and_outcome_ast_audit():
    assert verifier._fixture()["passed"]
    assert verifier._schedule_a() == runner._stage_a_schedule()
    assert verifier._schedule_b(250) == runner._stage_b_schedule(250)
    verifier._validate_protocol(
        {"protocol": runner._protocol("stage_a_natural_feasibility", runner._stage_a_schedule())},
        "stage_a_natural_feasibility",
        verifier._schedule_a(),
    )
    verifier._validate_protocol(
        {"protocol": runner._protocol("stage_b_confirmatory_factorial", runner._stage_b_schedule(400))},
        "stage_b_confirmatory_factorial",
        verifier._schedule_b(400),
    )
    assert verifier._static_outcome_exclusion_audit()["passed"]


def test_exact_twenty_pair_sign_flip_enumerates_all_assignments_and_holm():
    # Only the all-positive and all-negative assignments are at least as
    # extreme as the observed mean.
    assert verifier._exact_sign_flip(np.ones(20)) == 2 / (2**20)
    correction = verifier._holm({"C1": 0.001, "C2": 0.02, "C3": 0.03, "C4": 0.2})
    assert correction["C1"]["reject_familywise_0.05"]
    assert correction["C4"]["reject_familywise_0.05"] is False
    assert correction["C1"]["holm_adjusted_p"] == 0.004
    assert correction["C2"]["holm_adjusted_p"] == 0.06
    assert correction["C3"]["holm_adjusted_p"] == 0.06
    assert correction["C4"]["holm_adjusted_p"] == 0.2
    nontrivial = np.asarray([1.0] * 18 + [-1.0] * 2)
    assert verifier._exact_sign_flip(nontrivial) == 422 / (2**20)


def test_independent_budget_tv_runtime_contrast_and_materiality_boundaries():
    assert verifier._select_update_budget([400] * 27) == 400
    assert verifier._select_update_budget([400] * 26 + [399]) == 250
    assert verifier._select_update_budget([250] * 26 + [249]) is None
    assert not verifier._teacher_tv_pass([0.05])
    assert verifier._teacher_tv_pass([np.nextafter(0.05, 1.0)])
    assert verifier._teacher_tv_pass([0.05, np.nextafter(0.05, 1.0)])
    assert verifier._teacher_tv_pass(
        [0.05, np.nextafter(0.05, 1.0)]
    ) == runner._teacher_tv_pass([0.05, np.nextafter(0.05, 1.0)])
    assert verifier._projected_hours_180(360.0) == 18.0
    contrasts = {
        "C1": {"mean_contrast": 0.03, "reject_familywise_0.05": True},
        "C2": {"mean_contrast": np.nextafter(0.03, 0.0), "reject_familywise_0.05": True},
        "C3": {"mean_contrast": -0.03, "reject_familywise_0.05": True},
        "C4": {"mean_contrast": 0.03, "reject_familywise_0.05": False},
    }
    decision = verifier._predeclared_scale_decision(contrasts)
    assert decision["C1_directional_local_improvement_supported"]
    assert not decision["C2_directional_increment_supported"]
    assert decision["C3_material_restricted_separability_departure"]
    assert not decision["C4_material_restricted_separability_departure"]
    assert verifier.CONTRASTS == runner.CONTRAST_SPECS


def test_exact_sign_flip_rejects_wrong_pair_count_and_nonfinite():
    with pytest.raises(ValueError, match="20 finite"):
        verifier._exact_sign_flip(np.ones(19))
    values = np.ones(20)
    values[3] = np.nan
    with pytest.raises(ValueError, match="20 finite"):
        verifier._exact_sign_flip(values)


def _shadow() -> dict:
    projection = {
        "transitions": 10,
        "sampled_groups": 2,
        "optimizer_updates": 1,
        "live_groups": 1,
        "live_applied_updates": 1,
        "dead_groups": 1,
        "all_pass_groups": 0,
        "task_groups": [0] * 8,
        "task_rollouts": [0] * 8,
        "task_transitions": [0] * 8,
        "x_transitions": [0, 10],
        "x_optimizer_updates": [0, 1],
        "update_diagnostics": [],
        "group_diagnostics": [
            {"group": 2, "task_id": 3, "regime": "dead", "transition_end": 10}
        ],
        "training_group_trace_groups": 2,
        "training_group_trace_sha256": "1" * 64,
        "final_training_state_sha256": "2" * 64,
    }
    records = [
        {
            "after_group": 2,
            "transitions": 10,
            "requested_task": 3,
            "credited_task": 2,
            "gradient_norm": 1.0,
            "hypothetical_update_norm": 0.001,
            "applied": False,
            "mutated": False,
            "frozen_group_parameters": True,
        }
    ]
    return {
        "test_config": {
            "seed": 100,
            "seed_block_status": "already-executed exploratory smoke; outside fresh V5 blocks",
            "learning_rate": 3e-4,
            "optimizer_update_budget": 3,
            "transition_group_start_cap": 80_000,
            "eval_episodes_per_task": 4,
        },
        "passed": True,
        "identical_training_group_trace": True,
        "identical_final_training_state": True,
        "identical_saved_training_projection": True,
        "eligible_preview_exercised": True,
        "preview_candidate_groups": [2],
        "preview_diagnostic_groups": [2],
        "preview_auxiliary_gradient_diagnostics": records,
        "preview_mechanical_projection": projection,
        "control_mechanical_projection": copy.deepcopy(projection),
        "preview_mechanical_projection_sha256": verifier._canonical_hash(projection),
        "control_mechanical_projection_sha256": verifier._canonical_hash(projection),
        "preview_training_group_trace_sha256": "1" * 64,
        "shadow_training_group_trace_sha256": "1" * 64,
        "preview_final_training_state_sha256": "2" * 64,
        "shadow_final_training_state_sha256": "2" * 64,
        "preview_transitions": 10,
        "shadow_transitions": 10,
    }


def test_shadow_validation_recomputes_saved_equalities_instead_of_trusting_boolean():
    artifact = {"preview_shadow_equivalence": _shadow()}
    assert verifier._validate_shadow(artifact)["passed"]
    artifact["preview_shadow_equivalence"]["shadow_final_training_state_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="summary does not bind"):
        verifier._validate_shadow(artifact)


def test_independent_scale_zero_gate_binds_preview_task_metadata():
    prefix = {
        "groups": [{"group": 4, "task_id": 3, "transition_end": 1}],
        "previews": [
            {
                "after_group": 4,
                "transitions": 1,
                "requested_task": 3,
                "credited_task": 2,
                "gradient_norm": 1.0,
                "hypothetical_update_norm": 0.001,
                "applied": False,
                "mutated": False,
                "frozen_group_parameters": True,
            }
        ],
    }
    assert verifier._scale_zero_prefix_preview_valid(prefix)
    prefix["previews"][0]["credited_task"] = 3
    assert not verifier._scale_zero_prefix_preview_valid(prefix)


def _one_update_run() -> dict:
    return {
        "seed": 16_000,
        "optimizer_updates": 1,
        "live_applied_updates": 1,
        "relabeled_groups": 0,
        "relabel_candidates": 0,
        "eligible_relabel_candidate_groups": [],
        "unscaled_aux_gradient_previews": 0,
        "auxiliary_gradient_diagnostics": [],
        "zero_gradient_update_attempts": 0,
        "zero_gradient_diagnostics": [],
        "group_diagnostics": [
            {
                "group": 1,
                "task_id": 2,
                "regime": "mixed",
                "update_source": "requested_live",
                "transition_end": 1,
            }
        ],
        "update_diagnostics": [
            {
                "optimizer_update": 1,
                "after_group": 1,
                "transitions": 1,
                "source": "requested_live",
                "requested_task": 2,
                "credited_task": 2,
                "gradient_norm": 1.0,
                "update_norm": 0.1,
            }
        ],
    }


def test_source_norms_bind_each_saved_counter_and_task_metadata():
    run = _one_update_run()
    assert verifier._source_norms(run, "fixture")["requested_live"]["count"] == 1
    run["live_applied_updates"] = 0
    run["relabeled_groups"] = 1
    with pytest.raises(ValueError, match="requested-live counter"):
        verifier._source_norms(run, "fixture")


def test_positive_relabel_validation_requires_ordered_one_to_one_candidates():
    run = _one_update_run()
    run.update(
        {
            "live_applied_updates": 0,
            "relabeled_groups": 1,
            "relabel_candidates": 1,
            "eligible_relabel_candidate_groups": [1],
        }
    )
    run["group_diagnostics"][0].update(regime="dead", update_source="hindsight_relabel")
    run["update_diagnostics"][0].update(
        source="hindsight_relabel", requested_task=2, credited_task=1
    )
    verifier._validate_relabels(run, "fixture", 1.0)
    run["eligible_relabel_candidate_groups"].append(2)
    run["relabel_candidates"] = 2
    with pytest.raises(ValueError, match="candidate"):
        verifier._validate_relabels(run, "fixture", 1.0)


def test_positive_relabel_validation_accepts_audited_zero_gradient_partition():
    run = _one_update_run()
    run.update(
        {
            "optimizer_updates": 0,
            "live_applied_updates": 0,
            "relabeled_groups": 0,
            "relabel_candidates": 1,
            "eligible_relabel_candidate_groups": [1],
            "update_diagnostics": [],
            "zero_gradient_update_attempts": 1,
            "zero_gradient_diagnostics": [
                {
                    "after_group": 1,
                    "transitions": 1,
                    "source": "hindsight_relabel",
                    "requested_task": 2,
                    "credited_task": 1,
                    "gradient_norm": 0.0,
                    "update_norm": 0.0,
                }
            ],
        }
    )
    run["group_diagnostics"][0].update(regime="dead", update_source=None)
    verifier._validate_relabels(run, "fixture", 1.0)
    run["zero_gradient_diagnostics"][0]["transitions"] = 2
    with pytest.raises(ValueError, match="zero-gradient"):
        verifier._validate_relabels(run, "fixture", 1.0)


def _linked_authorization(tmp_path: Path) -> tuple[dict, dict, int]:
    target = 250
    gates = {
        "all_pass": True,
        "stage_b_authorized": True,
        "selected_update_budget": target,
    }
    stage_a_path = tmp_path / "a.json"
    stage_a = {
        "schema": verifier.SCHEMA_A,
        "artifact_state": "complete",
        "provenance": {"source_lock_sha256": "a-lock"},
        "stage_a_learning_outcome_blind_gates": gates,
    }
    _write(stage_a_path, stage_a)
    stage_a_hash = verifier._sha256(stage_a_path)
    report_path = tmp_path / "a_verify.json"
    report = {
        "schema": verifier.REPORT_SCHEMA_A,
        "v5_stage": "stage_a_natural_feasibility",
        "all_checks_passed": True,
        "stage_b_factorial_authorized": True,
        "selected_optimizer_update_budget": target,
        "artifact": str(stage_a_path.resolve()),
        "artifact_sha256": stage_a_hash,
        "runner_saved_gates_sha256": verifier._canonical_hash(gates),
        "outcome_exclusion_audit": {"passed": True},
        "lock_sha256": "a-lock",
    }
    _write(report_path, report)
    report_hash = verifier._sha256(report_path)
    amendment_path = tmp_path / "amendment.json"
    amendment = {
        "schema": verifier.AMENDMENT_SCHEMA_B,
        "v5_stage": "stage_b_confirmatory_factorial",
        "explicit_stage_b_authorization": True,
        "selected_update_budget": target,
        "registered_schedule": verifier._schedule_b(target),
        "stage_a_artifact": str(stage_a_path.resolve()),
        "stage_a_artifact_sha256": stage_a_hash,
        "stage_a_independent_verification": str(report_path.resolve()),
        "stage_a_independent_verification_sha256": report_hash,
        "stage_a_gates_sha256": verifier._canonical_hash(gates),
        "frozen_claim_rule": (
            "C1/C2 require mean>=+0.03 and Holm rejection; C3/C4 require "
            "abs(mean)>=0.03 and Holm rejection; complete 180-run family only."
        ),
    }
    _write(amendment_path, amendment)
    amendment_hash = verifier._sha256(amendment_path)
    artifact = {
        "stage_b_amendment": str(amendment_path.resolve()),
        "stage_b_amendment_sha256": amendment_hash,
        "stage_a_artifact": str(stage_a_path.resolve()),
        "stage_a_artifact_sha256": stage_a_hash,
        "stage_a_independent_verification": str(report_path.resolve()),
        "stage_a_independent_verification_sha256": report_hash,
    }
    lock = {
        "amendment_sha256": amendment_hash,
        "stage_a_artifact_sha256": stage_a_hash,
        "stage_a_independent_verification_sha256": report_hash,
        "stage_a_gates_sha256": verifier._canonical_hash(gates),
    }
    return artifact, lock, target


def test_linked_stage_b_authorization_binds_every_hash_and_claim_rule(tmp_path):
    artifact, lock, target = _linked_authorization(tmp_path)
    assert verifier._verify_linked_stage_b_authorization(artifact, lock, target)["passed"]
    lock["stage_a_gates_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="gates"):
        verifier._verify_linked_stage_b_authorization(artifact, lock, target)


def test_verification_report_write_is_exclusive_and_atomic(tmp_path):
    path = tmp_path / "report.json"
    verifier._write_exclusive(path, {"passed": True})
    assert verifier._read_json(path) == {"passed": True}
    with pytest.raises(FileExistsError, match="overwrite"):
        verifier._write_exclusive(path, {"passed": False})
