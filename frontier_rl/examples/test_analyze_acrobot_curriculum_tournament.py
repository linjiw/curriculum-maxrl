"""Independent-analysis and tamper tests for the Acrobot V2 tournament."""

from __future__ import annotations

import copy
import itertools
import json

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from frontier_rl.examples import analyze_acrobot_curriculum_tournament as analysis


def _sampler_state(sampling: str, logical_seed: int) -> tuple[list[float], int, float]:
    roots = analysis._rng_domain_record(logical_seed)["rng_roots"]
    rng = np.random.default_rng(roots["teacher"])
    if sampling == "uniform":
        probabilities = np.full(8, 1.0 / 8.0)
    else:
        draw = rng.beta(np.ones(8), np.ones(8))
        if sampling == "p1mp":
            utility = np.maximum(draw * (1.0 - draw), 0.0)
        else:
            utility = np.maximum(1.0 - (1.0 - draw) ** 16 - draw, 0.0)
        probabilities = 0.9 * utility / utility.sum() + 0.1 / 8.0
    task = int(rng.choice(8, p=probabilities))
    tv = float(0.5 * np.abs(probabilities - 1.0 / 8.0).sum())
    return probabilities.tolist(), task, tv


def _quick_run(sampling: str = "uniform") -> dict:
    logical_seed = analysis.SCHEDULES["quick"][0][0]
    domains = analysis._rng_domain_record(logical_seed)
    probabilities, task, tv = _sampler_state(sampling, logical_seed)
    success_count = 4
    mass = analysis.practical_maxrl_mass(success_count)
    pass_rates = [
        [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
    ]
    mean_pass = [float(np.mean(row)) for row in pass_rates]
    hardest = [row[7] for row in pass_rates]
    native_return = [-500.0, -450.0]
    censored_time = [500.0, 450.0]
    entropy = [1.0, 0.9]
    x = [0, 8_000]
    sampled_axis = [0, 1]
    update_axis = [0, 1]
    task_groups = [int(index == task) for index in range(8)]
    task_rollouts = [16 * value for value in task_groups]
    task_successes = [success_count * value for value in task_groups]
    task_transitions = [8_000 * value for value in task_groups]
    run = {
        "seed": logical_seed,
        "logical_seed": logical_seed,
        "engine_master_seed": domains["engine_master_seed"],
        "environment_adapter_seed_argument": domains[
            "environment_adapter_seed_argument"
        ],
        "rng_roots": domains["rng_roots"],
        "numeric_valid": True,
        "accounting_valid": True,
        "verifier_relabel_checks_valid": True,
        "evaluation_cadence_invariant": True,
        "total_parameters": 640,
        "active_parameters_per_task": 640,
        "relabeled_groups": 0,
        "relabel_candidates": 0,
        "transitions": 8_000,
        "sampled_groups": 1,
        "rollout_attempts": 16,
        "optimizer_updates": 1,
        "reached_optimizer_update_budget": True,
        "transition_cap_censored": False,
        "live_groups": 1,
        "live_applied_updates": 1,
        "dead_groups": 0,
        "all_pass_groups": 0,
        "zero_gradient_update_attempts": 0,
        "unscaled_aux_gradient_previews": 0,
        "task_groups": task_groups,
        "task_rollouts": task_rollouts,
        "task_successes": task_successes,
        "task_transitions": task_transitions,
        "update_diagnostics": [
            {
                "optimizer_update": 1,
                "after_group": 1,
                "transitions": 8_000,
                "source": "requested_live",
                "requested_task": task,
                "credited_task": task,
                "gradient_norm": 1.0,
                "update_norm": 0.1,
                "mean_policy_entropy": 0.9,
            }
        ],
        "zero_gradient_diagnostics": [],
        "auxiliary_gradient_diagnostics": [],
        "group_diagnostics": [
            {
                "group": 1,
                "transition_start": 0,
                "transition_end": 8_000,
                "n_transitions": 8_000,
                "task_id": task,
                "success_count": success_count,
                "regime": "mixed",
                "task_probabilities": probabilities,
                "posterior_mean_pass_rates_before_group": [0.5] * 8,
                "sampled_task_probability": probabilities[task],
                "teacher_tv_from_uniform": tv,
                "optimizer_updates_after_group": 1,
                "update_source": "requested_live",
                "realized_practical_maxrl_abs_coefficient_mass": mass,
            }
        ],
        "x_transitions": x,
        "x_sampled_groups": sampled_axis,
        "x_optimizer_updates": update_axis,
        "pass_rate_curve": pass_rates,
        "mean_pass_curve": mean_pass,
        "hardest_pass_curve": list(hardest),
        "native_success_rate_curve": list(hardest),
        "mean_native_return_curve": native_return,
        "mean_censored_time_to_goal_curve": censored_time,
        "mean_policy_entropy_curve": entropy,
        "evaluation_rng_preserved": [True, True],
        "final_mean_pass": mean_pass[-1],
        "final_hardest_pass": hardest[-1],
        "final_native_success_rate": hardest[-1],
        "final_mean_native_return": native_return[-1],
        "final_mean_censored_time_to_goal": censored_time[-1],
        "realized_coefficient_mass_total": mass,
        "realized_coefficient_mass_per_group": mass,
        "realized_coefficient_mass_per_million_transitions": mass * 125.0,
        "nonzero_coefficient_mass_group_fraction": 1.0,
        "sampled_groups_per_million_transitions": 125.0,
        "optimizer_updates_per_million_transitions": 125.0,
    }
    run["checkpoint_records"] = [
        {
            "checkpoint": index,
            "transitions": x[index],
            "sampled_groups": sampled_axis[index],
            "optimizer_updates": update_axis[index],
            "evaluation_shared_trajectories": 2,
            "pass_rates": pass_rates[index],
            "target_uniform_mean_pass_rate": mean_pass[index],
            "hardest_pass_rate": hardest[index],
            "native_success_rate": hardest[index],
            "mean_native_return": native_return[index],
            "mean_censored_time_to_goal": censored_time[index],
            "mean_policy_entropy": entropy[index],
            "training_rng_preserved": True,
            "evaluation_seed": domains["rng_roots"]["evaluation_episode"],
        }
        for index in range(2)
    ]
    run["auc_mean_pass_by_transitions"] = analysis.normalized_trapezoid(
        mean_pass, x
    )
    run["auc_native_success_by_transitions"] = analysis.normalized_trapezoid(
        hardest, x
    )
    run["auc_native_return_by_transitions"] = analysis.normalized_trapezoid(
        native_return, x
    )
    run["auc_mean_pass_by_sampled_groups"] = analysis.normalized_trapezoid(
        mean_pass, sampled_axis
    )
    run["auc_mean_pass_by_optimizer_updates"] = analysis.normalized_trapezoid(
        mean_pass, update_axis
    )
    return run


def _quick_artifact() -> dict:
    seeds, budget, interval, eval_n = analysis.SCHEDULES["quick"]
    protocol = {
        "study": "acrobot_curriculum_tournament",
        "mode": "quick",
        "status": "development_only",
        "protocol_document": (
            "frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_PROTOCOL.md"
        ),
        "condition_names": list(analysis.EXPECTED_CASES),
        "paired_seeds": list(seeds),
        "logical_to_engine_master_seed": {
            str(seed): analysis._engine_master_seed(seed) for seed in seeds
        },
        "thresholds": list(analysis.THRESHOLDS),
        "n_rollouts": 16,
        "transition_budget": budget,
        "complete_final_group": True,
        "eval_interval_transitions": interval,
        "eval_n_shared_trajectories": eval_n,
        "evaluation_threshold_scoring": (
            "shared nested trajectories reused across all eight thresholds"
        ),
        "evaluation_seed_base": analysis.EVALUATION_SEED_BASE,
        "fixed_evaluation_common_random_numbers": True,
        "rng_domain_contract": {
            "engine_master_base": analysis.ENGINE_MASTER_BASE,
            "engine_master_stride": analysis.ENGINE_MASTER_STRIDE,
            "environment_adapter_seed_offset": (
                analysis.ENVIRONMENT_ADAPTER_SEED_OFFSET
            ),
            "rng_domain_offsets": dict(analysis.RNG_DOMAIN_OFFSETS),
            "adapter_internal_environment_reset_rng": (
                "master + 1000 adapter seed argument + 10003 internal offset"
            ),
            "pairing": (
                "all three arms intentionally share roots within logical seed; "
                "roots are globally unique across logical-seed/domain pairs"
            ),
        },
        "architecture": "shared_h64_task_blind",
        "total_parameters": 640,
        "learning_rate": 3e-4,
        "optimizer": "plain SGD ascent",
        "estimator": "practical dropped-group MaxRL",
        "hindsight_scale": 0.0,
        "teacher": {
            "tracking": "discounted Beta Thompson sampling",
            "decay": analysis.TEACHER_DECAY,
            "floor": analysis.TEACHER_FLOOR,
            "gamma": 1.0,
            "utilities": dict(analysis.EXPECTED_SAMPLER_LABEL),
        },
        "primary": "u16 minus p(1-p) target-uniform transition AUC",
        "primary_test": "exact two-sided 2^20 paired sign flip",
        "primary_support": "20,000-resample paired-seed bootstrap interval",
        "primary_sesoi": analysis.PRIMARY_SESOI,
        "primary_decision": (
            "supported iff mean u16-p1mp AUC >= +0.01 and exact two-sided p <= 0.05"
        ),
        "secondary_uniform_tests": (
            "p(1-p)-uniform and u16-uniform; Holm family"
        ),
        "development_gate": None,
        "raw_only": True,
    }
    cases = {}
    for case in analysis.EXPECTED_CASES:
        sampling = analysis.EXPECTED_SAMPLING[case]
        cases[case] = {
            "sampler": analysis.EXPECTED_SAMPLER_LABEL[case],
            "config": {
                "name": case,
                "stage": "tournament",
                "sampling": sampling,
                "architecture": "shared",
                "hidden_size": 64,
                "learning_rate": 3e-4,
                "hindsight_scale": 0.0,
            },
            "runs": [_quick_run(sampling)],
        }
    return {
        "schema": analysis.RAW_SCHEMA,
        "artifact_state": "complete",
        "run_failures": [],
        "protocol": protocol,
        "cases": cases,
    }


def test_exact_sign_flip_matches_direct_enumeration_and_uses_all_assignments():
    values = np.asarray([0.2, -0.1, 0.4])
    observed = abs(float(values.mean()))
    expected = np.mean(
        [
            abs(float(np.dot(signs, values) / len(values))) >= observed - 1e-15
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ]
    )
    assert analysis.exact_two_sided_sign_flip_p(values) == expected
    assert analysis.exact_two_sided_sign_flip_p(np.zeros(20)) == 1.0


def test_holm_is_monotone_and_step_down():
    adjusted = analysis.holm_adjust({"a": 0.01, "b": 0.03})
    assert adjusted["a"]["holm_adjusted_p"] == 0.02
    assert adjusted["b"]["holm_adjusted_p"] == 0.03
    assert adjusted["a"]["reject_familywise_0.05"] is True
    assert adjusted["b"]["reject_familywise_0.05"] is True


@pytest.mark.parametrize("sampling", ("uniform", "p1mp", "u16"))
def test_raw_run_recomputes_auc_mass_rng_teacher_and_cost_axes(sampling):
    run = _quick_run(sampling)
    metrics = analysis._run_metrics(
        run,
        budget=8_000,
        interval=4_000,
        eval_n=2,
        mode="quick",
        sampling=sampling,
    )
    assert metrics["target_uniform_transition_auc"] == pytest.approx(0.28125)
    assert metrics["target_uniform_sampled_group_auc"] == pytest.approx(0.28125)
    assert metrics["target_uniform_optimizer_update_auc"] == pytest.approx(0.28125)
    assert metrics["coefficient_mass_per_group"] == 1.5
    assert metrics["sampled_groups_per_million_transitions"] == 125.0
    assert metrics["teacher_max_tv_from_uniform"] == pytest.approx(
        run["group_diagnostics"][0]["teacher_tv_from_uniform"]
    )


def _mutate(path: tuple[object, ...], value: object):
    def apply(run: dict) -> None:
        cursor = run
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value

    return apply


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (_mutate(("numeric_valid",), False), "invalid run flags"),
        (_mutate(("rng_roots", "teacher"), 0), "RNG-domain roots"),
        (_mutate(("total_parameters",), 641), "parameter contract"),
        (_mutate(("relabeled_groups",), 1), "hindsight activity"),
        (_mutate(("rollout_attempts",), 15), "rollout/group budget"),
        (
            _mutate(("group_diagnostics", 0, "transition_start"), 1),
            "ordered and contiguous",
        ),
        (
            _mutate(("group_diagnostics", 0, "optimizer_updates_after_group"), 0),
            "non-applied group has an update source",
        ),
        (
            _mutate(("group_diagnostics", 0, "teacher_tv_from_uniform"), 0.2),
            "teacher TV",
        ),
        (
            _mutate(("group_diagnostics", 0, "success_count"), 5),
            "saved realized coefficient mass",
        ),
        (_mutate(("x_transitions", 1), 7_999), "checkpoint crossings"),
        (_mutate(("x_sampled_groups", 1), 0), "sampled-group checkpoint axis"),
        (_mutate(("x_optimizer_updates", 1), 0), "optimizer-update checkpoint axis"),
        (
            _mutate(("pass_rate_curve", 1), [0.0, 1.0] + [0.0] * 6),
            "pass-rate curve",
        ),
        (_mutate(("mean_pass_curve", 1), 0.2), "mean-pass curve"),
        (_mutate(("hardest_pass_curve", 1), 0.5), "hardest-pass curve"),
        (_mutate(("native_success_rate_curve", 1), 0.5), "native-success curve"),
        (_mutate(("mean_native_return_curve", 1), 0.0), "native-return curve"),
        (_mutate(("mean_censored_time_to_goal_curve", 1), 0.0), "censored-time curve"),
        (_mutate(("mean_policy_entropy_curve", 1), 2.0), "policy-entropy curve"),
        (_mutate(("evaluation_rng_preserved", 1), False), "RNG-preservation"),
        (
            _mutate(("checkpoint_records", 1, "evaluation_shared_trajectories"), 3),
            "identity/cadence/CRN",
        ),
        (_mutate(("final_mean_pass",), 0.0), "final_mean_pass mismatch"),
        (_mutate(("auc_mean_pass_by_transitions",), 0.0), "auc_mean_pass"),
    ),
)
def test_strict_run_validation_rejects_curve_axis_cadence_final_and_tv_tampering(
    mutator, message
):
    run = _quick_run("uniform")
    mutator(run)
    with pytest.raises(ValueError, match=message):
        analysis._run_metrics(
            run,
            budget=8_000,
            interval=4_000,
            eval_n=2,
            mode="quick",
            sampling="uniform",
        )


def test_quick_full_artifact_analysis_is_lockless_but_strict():
    artifact = _quick_artifact()
    report = analysis.analyze(artifact)
    assert report == {
        "schema": analysis.REPORT_SCHEMA,
        "mode": "quick",
        "all_checks_passed": True,
        "inference_performed": False,
        "reason": (
            "quick smoke is development-only; raw ledgers were validated "
            "without requiring or consulting a source lock"
        ),
    }

    tampered = copy.deepcopy(artifact)
    tampered["protocol"]["primary_sesoi"] = 0.0
    with pytest.raises(ValueError, match="protocol mismatch for primary_sesoi"):
        analysis.analyze(tampered)


@pytest.mark.parametrize(
    "key",
    (
        "study",
        "mode",
        "status",
        "protocol_document",
        "condition_names",
        "paired_seeds",
        "logical_to_engine_master_seed",
        "thresholds",
        "n_rollouts",
        "transition_budget",
        "complete_final_group",
        "eval_interval_transitions",
        "eval_n_shared_trajectories",
        "evaluation_threshold_scoring",
        "evaluation_seed_base",
        "fixed_evaluation_common_random_numbers",
        "architecture",
        "total_parameters",
        "learning_rate",
        "optimizer",
        "estimator",
        "hindsight_scale",
        "teacher",
        "primary",
        "primary_test",
        "primary_support",
        "primary_sesoi",
        "primary_decision",
        "secondary_uniform_tests",
        "raw_only",
    ),
)
def test_every_locked_protocol_field_is_strict(key):
    artifact = _quick_artifact()
    artifact["protocol"][key] = None
    expected = "unknown raw tournament mode" if key == "mode" else f"protocol mismatch for {key}"
    with pytest.raises(ValueError, match=expected):
        analysis._validate_raw_artifact(artifact)


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda item: item.update(schema="wrong"), "raw artifact schema"),
        (
            lambda item: item.update(artifact_state="in_progress"),
            "raw artifact is incomplete",
        ),
        (
            lambda item: item["run_failures"].append({"failed": True}),
            "contains failed runs",
        ),
        (
            lambda item: item["protocol"].update(
                rng_domain_contract={"tampered": True}
            ),
            "RNG-domain contract",
        ),
        (
            lambda item: item["protocol"].update(
                v3_adequacy_waiver={"reason": "forbidden"}
            ),
            "forbids a V3 adequacy waiver",
        ),
        (
            lambda item: item["protocol"].update(
                development_gate={"unexpected": True}
            ),
            "non-confirmatory raw artifact contains a launch gate",
        ),
        (
            lambda item: item.update(cases=dict(reversed(item["cases"].items()))),
            "arm set/order",
        ),
        (
            lambda item: item["cases"]["p1mp_shared_h64"].update(
                sampler="wrong"
            ),
            "arm configuration",
        ),
        (
            lambda item: item["cases"]["u16_shared_h64"]["config"].update(
                learning_rate=1e-3
            ),
            "arm configuration",
        ),
        (
            lambda item: item["cases"]["uniform_shared_h64"]["runs"][0].update(
                seed=20_201
            ),
            "paired seed order",
        ),
    ),
)
def test_raw_artifact_envelope_arm_and_order_tampering_is_rejected(mutator, message):
    artifact = _quick_artifact()
    mutator(artifact)
    with pytest.raises(ValueError, match=message):
        analysis._validate_raw_artifact(artifact)


def test_project_relative_artifact_paths_must_be_canonical(monkeypatch, tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(analysis, "PROJECT_ROOT", tmp_path)
    assert analysis._project_relative_file("raw.json", "raw") == raw
    with pytest.raises(ValueError, match="noncanonical"):
        analysis._project_relative_file("nested/../raw.json", "raw")
    with pytest.raises(ValueError, match="project-relative"):
        analysis._project_relative_file(str(raw.resolve()), "raw")


def _development_validated() -> dict:
    by_case = {}
    seeds = analysis.SCHEDULES["development"][0]
    for case_index, case in enumerate(analysis.EXPECTED_CASES):
        by_case[case] = {}
        for seed_index, seed in enumerate(seeds):
            task_groups = [0] * 8
            task_groups[(case_index * 3 + seed_index) % 8] = 1
            by_case[case][seed] = {
                "task_groups": task_groups,
                "regimes": ["dead", "mixed", "all_pass"],
                "teacher_max_tv_from_uniform": (
                    0.0 if case == "uniform_shared_h64" else 0.2
                ),
                "native_success_curve": [0.0, 0.25 + 0.01 * seed_index],
            }
    pooled = [metrics for case in by_case.values() for metrics in case.values()]
    for task, metrics in enumerate(pooled[:8]):
        metrics["task_groups"] = [int(index == task) for index in range(8)]
    return {"mode": "development", "seeds": list(seeds), "by_case": by_case}


def test_development_gates_are_outcome_blind_and_cover_registered_checks():
    report = analysis.development_gates(
        _development_validated(), {"source_lock_sha256": "a" * 64}
    )
    assert report["schema"] == analysis.GATE_SCHEMA
    assert report["all_gates_passed"] is True
    assert tuple(report["gates"]) == analysis.DEVELOPMENT_GATE_NAMES
    assert all(report["gates"].values())
    assert report["gate_policy"] == analysis.DEVELOPMENT_GATE_POLICY
    assert "contrast" not in report["diagnostics"]


def test_bound_development_gate_is_recomputed_from_raw(monkeypatch, tmp_path):
    monkeypatch.setattr(analysis, "PROJECT_ROOT", tmp_path)
    raw_path = tmp_path / "development.json"
    raw_path.write_text(json.dumps({"raw": True}), encoding="utf-8")
    source_lock = {
        "passed": True,
        "runtime": {"synthetic": True},
        "source_lock_sha256": "a" * 64,
        "checked_source_files": ["synthetic.py"],
    }
    canonical_gate = {
        "schema": analysis.GATE_SCHEMA,
        "mode": "development",
        "all_gates_passed": True,
        "source_lock_sha256": source_lock["source_lock_sha256"],
        "source_lock_verification": source_lock,
        "gates": {name: True for name in analysis.DEVELOPMENT_GATE_NAMES},
        "diagnostics": {"raw_recomputed": True},
        "gate_policy": dict(analysis.DEVELOPMENT_GATE_POLICY),
    }
    gate = {
        **canonical_gate,
        "raw_artifact_relative_path": "development.json",
        "raw_artifact_sha256": analysis._sha256(raw_path),
    }
    gate_path = tmp_path / "gates.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    binding = {
        "relative_path": "gates.json",
        "sha256": analysis._sha256(gate_path),
        "raw_artifact_relative_path": "development.json",
        "raw_artifact_sha256": analysis._sha256(raw_path),
        "all_gates_passed": True,
    }
    artifact = {"protocol": {"development_gate": binding}}
    marker = {"mode": "development", "validated": True}
    monkeypatch.setattr(analysis, "_verify_lock", lambda *args: source_lock)
    monkeypatch.setattr(analysis, "_validate_raw_artifact", lambda raw: marker)
    monkeypatch.setattr(
        analysis,
        "development_gates",
        lambda validated, lock: canonical_gate,
    )
    report = analysis._verify_confirmatory_development_gate(
        artifact, {}, tmp_path / "lock.json", source_lock
    )
    assert report["gates_recomputed_from_raw"] is True

    gate["diagnostics"] = {"raw_recomputed": False}
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    artifact["protocol"]["development_gate"]["sha256"] = analysis._sha256(
        gate_path
    )
    with pytest.raises(ValueError, match="does not recompute for diagnostics"):
        analysis._verify_confirmatory_development_gate(
            artifact, {}, tmp_path / "lock.json", source_lock
        )


def _confirmatory_validated(primary_delta: float = 0.02) -> dict:
    metric_names = (
        "target_uniform_transition_auc",
        "native_success_auc",
        "native_return_auc",
        "target_uniform_sampled_group_auc",
        "target_uniform_optimizer_update_auc",
        "sampled_groups_per_million_transitions",
        "optimizer_updates_per_million_transitions",
        "final_native_success_rate",
        "final_native_return",
        "coefficient_mass_per_group",
        "coefficient_mass_per_million_transitions",
        "nonzero_mass_group_fraction",
    )
    p1mp_offset = 0.01
    offsets = {
        "uniform_shared_h64": 0.0,
        "p1mp_shared_h64": p1mp_offset,
        "u16_shared_h64": p1mp_offset + primary_delta,
    }
    seeds = analysis.SCHEDULES["confirmatory"][0]
    by_case = {}
    for case in analysis.EXPECTED_CASES:
        by_case[case] = {}
        for index, seed in enumerate(seeds):
            base = 0.2 + index / 1_000.0 + offsets[case]
            by_case[case][seed] = {metric: base for metric in metric_names}
    return {"mode": "confirmatory", "seeds": list(seeds), "by_case": by_case}


@pytest.mark.parametrize(
    ("delta", "positive", "meets_sesoi", "supported"),
    ((0.02, True, True, True), (0.005, True, False, False), (-0.02, False, False, False)),
)
def test_registered_primary_direction_sesoi_and_conjunctive_decision_rule(
    delta, positive, meets_sesoi, supported
):
    report = analysis.confirmatory_analysis(
        _confirmatory_validated(delta), {"source_lock_sha256": "b" * 64}
    )
    primary = report["primary"]
    assert primary["estimand"] == "u16_shared_h64 minus p1mp_shared_h64"
    assert primary["test_assignments"] == 2**20
    assert primary["mean_paired_difference"] == pytest.approx(delta)
    assert primary["exact_two_sided_sign_flip_p"] == 2 / 2**20
    assert primary["positive_mean_direction"] is positive
    assert primary["meets_sesoi"] is meets_sesoi
    assert primary["efficacy_supported"] is supported
    assert primary["decision_label"] == ("confirmed" if supported else "not confirmed")
    assert set(report["secondary_uniform_auc_tests"]) == {
        "p1mp_minus_uniform",
        "u16_minus_uniform",
    }
    assert report["secondary_uniform_multiplicity"]["method"] == "Holm step-down"
