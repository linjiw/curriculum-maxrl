"""Small synthetic tests for the independent Acrobot hindsight V4 verifier."""

from __future__ import annotations

import copy
import json
import platform

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from frontier_rl.examples import analyze_acrobot_hindsight_v4 as verifier


ORIGINAL_VALIDATE_STAGE_B_AUTHORIZATION = verifier._validate_stage_b_authorization


@pytest.fixture(autouse=True)
def _synthetic_source_lock(monkeypatch):
    """Synthetic artifacts exercise logic without depending on a not-yet-frozen lock."""

    monkeypatch.setattr(verifier, "REQUIRED_SOURCE_FILES", set())
    monkeypatch.setattr(
        verifier,
        "_validate_stage_b_authorization",
        lambda artifact, lock, selected_budget: {"passed": True, "synthetic": True},
    )


def _lock(*, selected_budget: int | None = None) -> dict:
    stage = (
        "stage_a_feasibility"
        if selected_budget is None
        else "stage_b_factorial"
    )
    result = {
        "schema": verifier.LOCK_SCHEMA_A if selected_budget is None else verifier.LOCK_SCHEMA_B,
        "v4_stage": stage,
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
        },
        "source_sha256": {},
        "registered_schedule": verifier._expected_schedule(stage, selected_budget),
    }
    return result


def _config(multiplier: float, scale: float) -> dict:
    name = verifier._case_name(multiplier, scale)
    return verifier._expected_config(name, multiplier, scale)


def _run(
    seed: int,
    *,
    optimizer_updates: int,
    mean_pass: float,
    scale_zero_previews: bool,
    reached_target: bool = True,
) -> dict:
    groups = []
    update_diagnostics = []
    auxiliary = []
    transitions = 0
    update_count = 0
    teacher = verifier.FrontierTeacher(
        8, 16, decay=0.7, floor=0.1, gamma=1.0, seed=seed + 10_000
    )

    def append_group(success_count: int, source: str | None = None) -> int:
        nonlocal transitions, update_count
        probabilities = np.asarray(teacher.distribution(), dtype=np.float64)
        task = int(teacher.rng.choice(8, p=probabilities))
        teacher_tv = float(0.5 * np.abs(probabilities - 0.125).sum())
        start = transitions
        transitions += 1_000
        if source is not None:
            update_count += 1
            update_diagnostics.append(
                {
                    "optimizer_update": update_count,
                    "after_group": len(groups) + 1,
                    "transitions": transitions,
                    "source": source,
                    "requested_task": task,
                    "credited_task": task,
                    "gradient_norm": 1.0,
                    "update_norm": 0.001,
                }
            )
        groups.append(
            {
                "group": len(groups) + 1,
                "transition_start": start,
                "transition_end": transitions,
                "n_transitions": 1_000,
                "task_id": task,
                "success_count": success_count,
                "regime": verifier._regime(success_count),
                "teacher_tv_from_uniform": teacher_tv,
                "sampled_task_probability": float(probabilities[task]),
                "optimizer_updates_after_group": update_count,
                "update_source": source,
            }
        )
        rewards = np.zeros(16, dtype=np.float64)
        rewards[:success_count] = 1.0
        teacher.observe(task, rewards)
        return len(groups)

    eligible_groups = []
    for _ in range(12):
        group_number = append_group(0)
        if scale_zero_previews:
            eligible_groups.append(group_number)
            auxiliary.append(
                {
                    "after_group": group_number,
                    "transitions": transitions,
                    "requested_task": 1,
                    "credited_task": 0,
                    "applied": False,
                    "gradient_norm": 2.0,
                    "hypothetical_update_norm": 0.002,
                    "frozen_group_parameters": True,
                    "mutated": False,
                }
            )
    append_group(16)
    evaluation_transition_by_update = {0: 0}
    for update in range(1, optimizer_updates + 1):
        append_group(1, "requested_live")
        if update % verifier.EVAL_INTERVAL_UPDATES == 0:
            evaluation_transition_by_update[update] = transitions

    if not reached_target:
        while transitions < verifier.TRANSITION_CAP:
            append_group(16)

    x_updates = list(
        range(0, optimizer_updates + 1, verifier.EVAL_INTERVAL_UPDATES)
    )
    if x_updates[-1] != optimizer_updates:
        x_updates.append(optimizer_updates)
        evaluation_transition_by_update[optimizer_updates] = transitions
    x_transitions = [evaluation_transition_by_update[value] for value in x_updates]
    if not reached_target:
        x_updates.append(optimizer_updates)
        x_transitions.append(transitions)
    task_groups = [sum(group["task_id"] == task for group in groups) for task in range(8)]
    task_rollouts = [16 * count for count in task_groups]
    task_successes = [
        sum(group["success_count"] for group in groups if group["task_id"] == task)
        for task in range(8)
    ]
    task_transitions = [
        sum(group["n_transitions"] for group in groups if group["task_id"] == task)
        for task in range(8)
    ]
    return {
        "seed": seed,
        "numeric_valid": True,
        "accounting_valid": True,
        "verifier_relabel_checks_valid": True,
        "evaluation_cadence_invariant": True,
        "transitions": transitions,
        "sampled_groups": len(groups),
        "rollout_attempts": 16 * len(groups),
        "optimizer_updates": optimizer_updates,
        "reached_optimizer_update_budget": reached_target,
        "transition_cap_censored": not reached_target,
        "live_groups": sum(group["regime"] == "mixed" for group in groups),
        "live_applied_updates": optimizer_updates,
        "dead_groups": sum(group["regime"] == "dead" for group in groups),
        "all_pass_groups": sum(group["regime"] == "all_pass" for group in groups),
        "relabeled_groups": 0,
        "relabel_candidates": len(auxiliary),
        "eligible_relabel_candidate_groups": eligible_groups,
        "unscaled_aux_gradient_previews": len(auxiliary),
        "zero_gradient_update_attempts": 0,
        "task_groups": task_groups,
        "task_rollouts": task_rollouts,
        "task_successes": task_successes,
        "task_transitions": task_transitions,
        "total_parameters": 640,
        "active_parameters_per_task": 640,
        "wall_seconds": 1.0,
        "wall_seconds_at_optimizer_updates": {
            key: value
            for key, value in {"250": 0.5, "400": 1.0}.items()
            if int(key) <= optimizer_updates
        },
        "x_transitions": x_transitions,
        "x_optimizer_updates": x_updates,
        "mean_pass_curve": [mean_pass] * len(x_updates),
        "pass_rate_curve": [[mean_pass] * 8 for _ in x_updates],
        "hardest_pass_curve": [mean_pass] * len(x_updates),
        "evaluation_rng_preserved": [True] * len(x_updates),
        "auc_mean_pass_by_optimizer_updates": mean_pass,
        "initial_mean_pass": mean_pass,
        "final_mean_pass": mean_pass,
        "update_diagnostics": update_diagnostics,
        "zero_gradient_diagnostics": [],
        "auxiliary_gradient_diagnostics": auxiliary,
        "group_diagnostics": groups,
        "training_group_trace_groups": len(groups),
        "training_group_trace_sha256": "1" * 64,
        "final_training_state_sha256": "2" * 64,
        "source_step_norms_full_run": {
            "requested_live": {
                "count": optimizer_updates,
                "cumulative_step_norm_M": optimizer_updates * 0.001,
                "cumulative_squared_step_norm_Q": optimizer_updates * 0.001**2,
            },
            "hindsight_relabel": {
                "count": 0,
                "cumulative_step_norm_M": 0.0,
                "cumulative_squared_step_norm_Q": 0.0,
            },
        },
    }


def _provenance() -> dict:
    return {
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
        },
        "source_sha256": {},
    }


def _stage_a_artifact() -> dict:
    cases = {}
    for multiplier in verifier.LR_MULTIPLIERS:
        name = verifier._case_name(multiplier, 0.0)
        cases[name] = {
            "config": _config(multiplier, 0.0),
            "runs": [
                _run(
                    seed,
                    optimizer_updates=400,
                    mean_pass=0.1 + seed % 3 * 0.01,
                    scale_zero_previews=True,
                )
                for seed in verifier.STAGE_A_SEEDS
            ],
        }
    artifact = {
        "schema": verifier.SCHEMA_A,
        "provenance": _provenance(),
        "protocol": verifier._expected_protocol(
            "stage_a_feasibility",
            verifier._expected_schedule("stage_a_feasibility"),
        ),
        "artifact_state": "complete",
        "run_failures": [],
        "cases": cases,
        "preview_shadow_equivalence": {
            "test_config": {
                "seed": verifier.SHADOW_SEED,
                "seed_block_status": "historical exploratory; outside V4 blocks",
                "learning_rate": verifier.BASE_LEARNING_RATE,
                "optimizer_update_budget": verifier.SHADOW_UPDATE_BUDGET,
                "transition_group_start_cap": verifier.SHADOW_TRANSITION_CAP,
                "eval_episodes_per_task": verifier.SHADOW_EVAL_EPISODES,
            },
            "passed": True,
            "identical_training_group_trace": True,
            "identical_final_training_state": True,
            "identical_saved_training_projection": True,
            "eligible_preview_exercised": True,
            "preview_candidate_groups": [1],
            "preview_diagnostic_groups": [1],
            "preview_training_group_trace_sha256": "3" * 64,
            "shadow_training_group_trace_sha256": "3" * 64,
            "preview_final_training_state_sha256": "4" * 64,
            "shadow_final_training_state_sha256": "4" * 64,
            "preview_transitions": 1_000,
            "shadow_transitions": 1_000,
        },
    }
    _attach_stage_a_saved_gates(artifact, selected_budget=400)
    return artifact


def _attach_stage_a_saved_gates(
    artifact: dict, *, selected_budget: int, gate3_passed: bool = True
) -> None:
    all_pass = gate3_passed
    shadow = copy.deepcopy(artifact["preview_shadow_equivalence"])
    artifact["stage_a_effect_blind_gates"] = {
        "effect_blind": True,
        "uses_evaluation_performance": False,
        "uses_hindsight_contrast": False,
        "uses_v3_outcome": False,
        "selected_update_budget": selected_budget,
        "gate_1_lock_and_implementation_invariants": {"passed": True},
        "gate_2_exact_selected_prefix": {
            "passed": True,
            "selected_update_budget": selected_budget,
        },
        "gate_3_preview_mechanics_and_shadow": {
            "passed": gate3_passed,
            "preview_shadow_equivalence": shadow,
        },
        "gate_4_requested_group_regimes": {"passed": True},
        "gate_5_per_run_teacher_tv": {"passed": True},
        "gate_6_serial_runtime_projection": {"passed": True},
        "all_pass": all_pass,
        "stage_b_authorized": all_pass,
        "prefix_diagnostics_sha256": "5" * 64,
    }
    artifact["analysis_status"] = {
        "performed": True,
        "type": "hindsight-effect-blind feasibility gates only",
        "learning_performance_used": False,
    }


GRID_AUCS = {
    (0.5, 0.0): 0.10,
    (0.5, 1.0): 0.11,
    (0.5, 2.0): 0.18,
    (1.0, 0.0): 0.10,
    (1.0, 1.0): 0.15,
    (1.0, 2.0): 0.20,
    (2.0, 0.0): 0.10,
    (2.0, 1.0): 0.14,
    (2.0, 2.0): 0.18,
}


def _stage_b_artifact(*, with_saved_analysis: bool = True) -> dict:
    selected = 250
    cases = {}
    for multiplier in verifier.LR_MULTIPLIERS:
        for scale in verifier.HINDSIGHT_SCALES:
            name = verifier._case_name(multiplier, scale)
            cases[name] = {
                "config": _config(multiplier, scale),
                "runs": [
                    _run(
                        seed,
                        optimizer_updates=selected,
                        mean_pass=GRID_AUCS[(multiplier, scale)],
                        scale_zero_previews=scale == 0.0,
                    )
                    for seed in verifier.STAGE_B_SEEDS
                ],
            }
    artifact = {
        "schema": verifier.SCHEMA_B,
        "provenance": _provenance(),
        "protocol": verifier._expected_protocol(
            "stage_b_factorial",
            verifier._expected_schedule("stage_b_factorial", selected),
        ),
        "artifact_state": "complete",
        "run_failures": [],
        "cases": cases,
    }
    artifact["stage_b_case_summaries"] = {}
    for case_name, record in cases.items():
        values = np.asarray(
            [run[verifier.PRIMARY_METRIC] for run in record["runs"]],
            dtype=np.float64,
        )
        artifact["stage_b_case_summaries"][case_name] = {
            "metric": verifier.PRIMARY_METRIC,
            "n_seeds": 10,
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "per_seed": values.tolist(),
            "source_step_norms_per_seed": {
                str(run["seed"]): copy.deepcopy(run["source_step_norms_full_run"])
                for run in record["runs"]
            },
            "transitions_to_target_per_seed": {
                str(run["seed"]): run["transitions"] for run in record["runs"]
            },
        }
    if with_saved_analysis:
        artifact["paired_scale_contrasts"] = {}
        raw_p_values = {}
        for index, (name, coefficients) in enumerate(
            verifier.CONTRAST_SPECS.items()
        ):
            values = np.zeros(10, dtype=np.float64)
            for case_name, coefficient in coefficients.items():
                values += coefficient * np.asarray(
                    [
                        run[verifier.PRIMARY_METRIC]
                        for run in artifact["cases"][case_name]["runs"]
                    ],
                    dtype=np.float64,
                )
            p_value = verifier._exact_sign_flip_p(values)
            artifact["paired_scale_contrasts"][name] = {
                "description": verifier.CONTRAST_DESCRIPTIONS[name],
                "metric": verifier.PRIMARY_METRIC,
                "coefficients": copy.deepcopy(coefficients),
                "n_pairs": 10,
                "per_seed_contrast": values.tolist(),
                "mean_contrast": float(values.mean()),
                "sample_std": float(values.std(ddof=1)),
                "mean_ci95_paired_seed_bootstrap": verifier._bootstrap_ci(
                    values, 45_000 + index
                ),
                "exact_paired_sign_flip_p_two_sided": p_value,
            }
            raw_p_values[name] = p_value
        corrections = verifier._holm(raw_p_values)
        for name, correction in corrections.items():
            artifact["paired_scale_contrasts"][name].update(correction)
        artifact["scale_multiplicity"] = {
            "family": list(verifier.CONTRAST_SPECS),
            "metric": verifier.PRIMARY_METRIC,
            "method": "Holm step-down",
            "familywise_alpha": 0.05,
            "test": "exact two-sided paired sign-flip randomization",
            "sign_exchangeability_assumption": (
                "independent seed-level contrasts have sign-exchangeable null distributions"
            ),
        }
        c = artifact["paired_scale_contrasts"]
        artifact["predeclared_scale_decision"] = {
            "C1_directional_local_improvement_supported": bool(
                c["C1"]["mean_contrast"] >= 0.03
                and c["C1"]["reject_familywise_0.05"]
            ),
            "C2_directional_increment_supported": bool(
                c["C2"]["mean_contrast"] >= 0.03
                and c["C2"]["reject_familywise_0.05"]
            ),
            "C3_material_restricted_separability_departure": bool(
                abs(c["C3"]["mean_contrast"]) >= 0.03
                and c["C3"]["reject_familywise_0.05"]
            ),
            "C4_material_restricted_separability_departure": bool(
                abs(c["C4"]["mean_contrast"]) >= 0.03
                and c["C4"]["reject_familywise_0.05"]
            ),
            "directional_minimum_mean": 0.03,
            "restricted_departure_minimum_absolute_mean": 0.03,
            "interpretation_boundary": (
                "C3/C4 diagnose departure from Y(a,s)=F(a)+G(a*s); they do not "
                "identify semantic data value separately from optimizer scale, update "
                "source composition, relabel frequency, or policy trajectory."
            ),
        }
        artifact["analysis_status"] = {
            "performed": True,
            "all_ten_seed_pairs_complete": True,
            "all_final_update_coordinates_equal_selected_budget": True,
            "selected_update_budget": selected,
        }
    return artifact


def test_stage_a_effect_blind_gates_pass_and_select_400():
    artifact = _stage_a_artifact()
    report = verifier.verify(artifact, _lock())
    assert report["gates"]["all_pass"]
    assert report["stage_b_factorial_authorized"]
    assert report["selected_optimizer_update_budget"] == 400
    assert report["runtime"]["projected_90_run_factorial_serial_hours"] == pytest.approx(
        0.025
    )


def test_stage_a_is_outcome_blind_but_rejects_preview_tampering():
    artifact = _stage_a_artifact()
    baseline = verifier.verify(artifact, _lock())["gates"]
    for case in artifact["cases"].values():
        for run in case["runs"]:
            run["mean_pass_curve"] = [0.9] * len(run["mean_pass_curve"])
            run["pass_rate_curve"] = [
                [0.9] * 8 for _ in run["mean_pass_curve"]
            ]
            run["hardest_pass_curve"] = [0.9] * len(run["mean_pass_curve"])
            run["initial_mean_pass"] = 0.9
            run["final_mean_pass"] = 0.9
            run["auc_mean_pass_by_optimizer_updates"] = 0.9
    assert verifier.verify(artifact, _lock())["gates"] == baseline

    artifact["cases"][verifier.STAGE_A_CASES[0]]["runs"][0][
        "auxiliary_gradient_diagnostics"
    ][0]["gradient_norm"] = 0.0
    _attach_stage_a_saved_gates(
        artifact, selected_budget=400, gate3_passed=False
    )
    report = verifier.verify(artifact, _lock())
    assert not report["gates"][
        "every_run_has_ten_one_to_one_positive_nonmutating_previews"
    ]
    assert not report["stage_b_factorial_authorized"]


def test_stage_a_fallback_uses_exact_250_update_prefix():
    artifact = _stage_a_artifact()
    for case in artifact["cases"].values():
        case["runs"] = [
            _run(
                seed,
                optimizer_updates=300,
                mean_pass=0.2,
                scale_zero_previews=True,
                reached_target=False,
            )
            for seed in verifier.STAGE_A_SEEDS
        ]
    _attach_stage_a_saved_gates(artifact, selected_budget=250)
    report = verifier.verify(artifact, _lock())
    assert report["selected_optimizer_update_budget"] == 250
    assert report["gates"][
        "every_selected_prefix_ends_at_selected_update_budget"
    ]
    assert all(
        run["selected_prefix_optimizer_updates"] == 250
        for cell in report["per_cell"].values()
        for run in cell["per_run"]
    )


def test_missing_stage_a_cell_is_rejected():
    artifact = _stage_a_artifact()
    artifact["cases"].pop(verifier.STAGE_A_CASES[-1])
    with pytest.raises(ValueError, match="exactly the ordered V4 cases"):
        verifier.verify(artifact, _lock())


def test_stage_b_recomputes_four_test_holm_family_and_decisions():
    artifact = _stage_b_artifact()
    report = verifier.verify(artifact, _lock(selected_budget=250))
    assert report["all_checks_passed"]
    assert report["saved_analysis_verified"]
    assert tuple(report["paired_contrasts"]) == tuple(verifier.CONTRAST_SPECS)
    assert report["decisions"]["scale1_mean_at_least_0p03_and_holm_significant"]
    assert report["decisions"][
        "scale2_increment_mean_at_least_0p03_and_holm_significant"
    ]
    assert report["decisions"]["c3_material_and_holm_significant"]
    assert report["decisions"]["c4_material_and_holm_significant"]


def test_stage_b_requires_budget_in_lock_and_saved_analysis():
    artifact = _stage_b_artifact()
    unlocked = _lock(selected_budget=250)
    unlocked["registered_schedule"].pop("optimizer_update_target")
    with pytest.raises(ValueError, match="250 or 400|schedule"):
        verifier.verify(artifact, unlocked)

    artifact = _stage_b_artifact(with_saved_analysis=False)
    with pytest.raises(ValueError, match="required saved"):
        verifier.verify(artifact, _lock(selected_budget=250))


def test_stage_b_authorization_chain_rechecks_linked_hashes(tmp_path):
    selected = 250
    saved_gates = {
        "all_pass": True,
        "stage_b_authorized": True,
        "selected_update_budget": selected,
    }
    stage_a_path = tmp_path / "stage_a.json"
    stage_a = {
        "schema": verifier.SCHEMA_A,
        "artifact_state": "complete",
        "provenance": {"source_lock_sha256": "a" * 64},
        "stage_a_effect_blind_gates": saved_gates,
    }
    stage_a_path.write_text(json.dumps(stage_a), encoding="utf-8")
    stage_a_hash = verifier._sha256(stage_a_path)

    report_path = tmp_path / "stage_a_verification.json"
    report = {
        "schema": "curriculum-maxrl/acrobot-hindsight-v4a-verification/v1",
        "v4_stage": "stage_a_feasibility",
        "all_checks_passed": True,
        "saved_runner_gates_verified": True,
        "runner_saved_gates": saved_gates,
        "stage_b_factorial_authorized": True,
        "selected_optimizer_update_budget": selected,
        "artifact": str(stage_a_path),
        "artifact_sha256": stage_a_hash,
        "lock_sha256": "a" * 64,
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_hash = verifier._sha256(report_path)

    amendment_path = tmp_path / "amendment.json"
    gates_hash = verifier._canonical_hash(saved_gates)
    amendment = {
        "schema": "curriculum-maxrl/acrobot-hindsight-v4b-amendment/v1",
        "v4_stage": "stage_b_factorial",
        "explicit_stage_b_authorization": True,
        "stage_a_all_effect_blind_gates_passed": True,
        "selected_update_budget": selected,
        "registered_schedule": verifier._expected_schedule(
            "stage_b_factorial", selected
        ),
        "stage_a_artifact": str(stage_a_path),
        "stage_a_artifact_sha256": stage_a_hash,
        "stage_a_independent_verification": str(report_path),
        "stage_a_independent_verification_sha256": report_hash,
        "stage_a_gates_sha256": gates_hash,
    }
    amendment_path.write_text(json.dumps(amendment), encoding="utf-8")
    amendment_hash = verifier._sha256(amendment_path)
    artifact = {
        "stage_a_artifact": str(stage_a_path),
        "stage_a_artifact_sha256": stage_a_hash,
        "stage_b_amendment": str(amendment_path),
        "stage_b_amendment_sha256": amendment_hash,
        "stage_a_independent_verification": str(report_path),
        "stage_a_independent_verification_sha256": report_hash,
    }
    lock = {
        "amendment_sha256": amendment_hash,
        "stage_a_artifact_sha256": stage_a_hash,
        "stage_a_independent_verification_sha256": report_hash,
        "stage_a_gates_sha256": gates_hash,
    }
    assert ORIGINAL_VALIDATE_STAGE_B_AUTHORIZATION(
        artifact, lock, selected
    )["passed"]

    report["stage_b_factorial_authorized"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        ORIGINAL_VALIDATE_STAGE_B_AUTHORIZATION(artifact, lock, selected)


def test_stage_b_rejects_tampered_auc_censoring_and_holm():
    artifact = _stage_b_artifact()
    first_case = verifier.STAGE_B_CASES[0]
    artifact["cases"][first_case]["runs"][0][verifier.PRIMARY_METRIC] += 0.1
    with pytest.raises(ValueError, match="run AUC"):
        verifier.verify(artifact, _lock(selected_budget=250))

    artifact = _stage_b_artifact()
    artifact["cases"][first_case]["runs"][0]["transition_cap_censored"] = True
    with pytest.raises(ValueError, match="incomplete or censored"):
        verifier.verify(artifact, _lock(selected_budget=250))

    artifact = _stage_b_artifact()
    contrast = next(iter(verifier.CONTRAST_SPECS))
    artifact["paired_scale_contrasts"][contrast]["holm_adjusted_p"] = 0.99
    with pytest.raises(ValueError, match="holm_adjusted_p"):
        verifier.verify(artifact, _lock(selected_budget=250))
