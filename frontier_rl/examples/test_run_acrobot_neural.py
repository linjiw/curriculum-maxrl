"""Focused regression tests for the preregistered neural Acrobot runner."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from frontier_rl.interfaces import GroupResult
from frontier_rl.examples import run_acrobot_neural as runner


def test_registered_matrices_and_default_seed_sets():
    core = runner.core_conditions(1e-3)
    assert len(core) == 6
    assert {
        (case.sampling, case.architecture, case.hidden_size) for case in core
    } == {
        (sampling, architecture, hidden)
        for sampling in ("uniform", "teacher")
        for architecture, hidden in (
            ("shared", 64),
            ("disjoint_total_budget", 8),
            ("disjoint_active_capacity", 64),
        )
    }
    scale = runner.scale_conditions(1e-3)
    assert len(scale) == 9
    assert {case.hindsight_scale for case in scale} == {0.0, 1.0, 2.0}

    common = dict(
        seeds=None,
        seed_start=None,
        transition_budget=None,
        update_budget=None,
        transition_cap=None,
        quick=False,
    )
    seeds, budget = runner._stage_defaults(
        SimpleNamespace(stage="pilot", **common)
    )
    assert seeds == [10_000, 10_001, 10_002]
    assert budget.transition_budget == 1_000_000
    seeds, budget = runner._stage_defaults(
        SimpleNamespace(stage="core", **common)
    )
    assert seeds == list(range(20))
    assert budget.transition_budget == 2_000_000
    seeds, budget = runner._stage_defaults(
        SimpleNamespace(stage="scale", **common)
    )
    assert seeds == list(range(100, 110))
    assert budget.optimizer_update_budget == 400
    assert budget.transition_safety_cap == 4_000_000


def test_core_architecture_filter_is_explicit_and_validated():
    shared = runner.core_conditions(3e-4, ("shared",))
    assert [case.name for case in shared] == [
        "uniform_shared_h64",
        "teacher_shared_h64",
    ]
    assert runner._parse_core_architectures("shared") == ("shared",)
    with pytest.raises(ValueError, match="unknown core architectures"):
        runner.core_conditions(3e-4, ("missing",))
    with pytest.raises(Exception, match="unique architecture subset"):
        runner._parse_core_architectures("shared,shared")


def test_explicit_exploratory_cli_preserves_defaults_and_hashes_selected_protocol(
    monkeypatch, tmp_path
):
    alternate_protocol = runner.PROJECT_ROOT / "FRAMEWORK.md"
    output = tmp_path / "exploratory_core.json"
    monkeypatch.setattr(runner, "core_conditions", lambda base_lr, architectures: ())
    monkeypatch.setattr(runner, "attach_core_analysis", lambda result: None)

    runner.main(
        [
            "core",
            "--exploratory",
            "--protocol-document",
            str(alternate_protocol),
            "--output",
            str(output),
        ]
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    protocol = artifact["protocol"]
    relative_protocol = str(alternate_protocol.relative_to(runner.PROJECT_ROOT))
    assert protocol["status"] == "exploratory"
    assert protocol["exploratory"] is True
    assert protocol["explicit_exploratory"] is True
    assert protocol["quick_is_exploratory"] is False
    assert protocol["pilot_is_exploratory"] is False
    assert protocol["paired_seeds"] == list(range(20))
    assert protocol["budget"]["transition_budget"] == 2_000_000
    assert protocol["protocol_document"] == relative_protocol
    assert artifact["provenance"]["source_sha256"][relative_protocol] == (
        runner.sha256_file(alternate_protocol)
    )
    default_protocol = str(runner.PROTOCOL_PATH.relative_to(runner.PROJECT_ROOT))
    assert default_protocol not in artifact["provenance"]["source_sha256"]


def test_shared_only_core_cli_records_narrow_claim(monkeypatch, tmp_path):
    output = tmp_path / "shared_core.json"
    monkeypatch.setattr(
        runner, "core_conditions", lambda base_lr, architectures: ()
    )
    monkeypatch.setattr(runner, "attach_shared_core_analysis", lambda result: None)

    runner.main(
        [
            "core",
            "--core-architectures",
            "shared",
            "--seed-start",
            "12000",
            "--protocol-document",
            str(runner.PROTOCOL_V3_PATH),
            "--output",
            str(output),
        ]
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    protocol = artifact["protocol"]
    assert protocol["status"] == "confirmatory"
    assert protocol["paired_seeds"] == list(range(12_000, 12_020))
    assert protocol["core_architectures"] == ["shared"]
    assert protocol["core_analysis_mode"] == "two_cell_shared_efficacy_only"
    assert protocol["transfer_claim_evaluated"] is False


def test_nonshared_partial_core_requires_exploratory(tmp_path):
    with pytest.raises(SystemExit):
        runner.main(
            [
                "core",
                "--core-architectures",
                "disjoint_total_budget",
                "--output",
                str(tmp_path / "invalid.json"),
            ]
        )


def test_v2_full_core_confirmation_is_procedurally_blocked(tmp_path):
    with pytest.raises(SystemExit):
        runner.main(["core", "--output", str(tmp_path / "blocked.json")])


def test_v3_confirmation_rejects_schedule_drift(tmp_path):
    with pytest.raises(SystemExit):
        runner.main(
            [
                "core",
                "--core-architectures",
                "shared",
                "--seed-start",
                "12000",
                "--base-lr",
                "0.001",
                "--protocol-document",
                str(runner.PROTOCOL_V3_PATH),
                "--output",
                str(tmp_path / "drift.json"),
            ]
        )


def test_pilot_and_quick_remain_exploratory_without_explicit_flag():
    common = dict(
        seeds=None,
        seed_start=None,
        transition_budget=None,
        update_budget=None,
        transition_cap=None,
        eval_interval_transitions=100_000,
        eval_interval_updates=50,
        eval_n=32,
        exploratory=False,
    )
    pilot_args = SimpleNamespace(stage="pilot", quick=False, **common)
    pilot_seeds, pilot_budget = runner._stage_defaults(pilot_args)
    pilot = runner._artifact_protocol(
        "pilot", pilot_args, pilot_seeds, pilot_budget
    )
    assert pilot["status"] == "exploratory"
    assert pilot["exploratory"] is True
    assert pilot["explicit_exploratory"] is False

    quick_args = SimpleNamespace(stage="core", quick=True, **common)
    quick_seeds, quick_budget = runner._stage_defaults(quick_args)
    quick = runner._artifact_protocol(
        "core", quick_args, quick_seeds, quick_budget
    )
    assert quick["status"] == "exploratory"
    assert quick["exploratory"] is True
    assert quick["explicit_exploratory"] is False


def test_protocol_document_must_exist_inside_project(tmp_path):
    outside = tmp_path / "outside_protocol.md"
    outside.write_text("frozen", encoding="utf-8")
    with pytest.raises(ValueError, match="inside the project"):
        runner._resolve_protocol_document(outside)
    with pytest.raises(ValueError, match="existing file"):
        runner._resolve_protocol_document(
            runner.PROJECT_ROOT / "missing_protocol_document.md"
        )


def test_auc_exact_sign_flip_and_holm_are_resource_explicit():
    assert runner.normalized_trapezoid([0.0, 1.0], [0, 10]) == pytest.approx(0.5)
    assert runner.exact_sign_flip_p([1.0, 1.0]) == pytest.approx(0.5)
    adjusted = runner.holm_adjust({"a": 0.01, "b": 0.03, "c": 0.5})
    assert adjusted["a"]["holm_adjusted_p"] == pytest.approx(0.03)
    assert adjusted["b"]["holm_adjusted_p"] == pytest.approx(0.06)
    assert not adjusted["b"]["reject_familywise_0.05"]


def _case_runs(values):
    return {"runs": [
        {"seed": seed, "auc_mean_pass_by_transitions": float(value)}
        for seed, value in enumerate(values)
    ]}


def test_core_analysis_uses_one_registered_five_contrast_family():
    n = 12
    result = {
        "cases": {
            "uniform_shared_h64": _case_runs(np.zeros(n)),
            "teacher_shared_h64": _case_runs(np.full(n, 0.20)),
            "uniform_disjoint_total_h8": _case_runs(np.zeros(n)),
            "teacher_disjoint_total_h8": _case_runs(np.full(n, 0.02)),
            "uniform_disjoint_active_h64": _case_runs(np.zeros(n)),
            "teacher_disjoint_active_h64": _case_runs(np.full(n, 0.01)),
        }
    }
    runner.attach_core_analysis(result)
    assert len(result["primary_multiplicity"]["family"]) == 5
    assert result["predeclared_core_decision"]["efficacy_supported"]
    assert result["predeclared_core_decision"]["strong_transfer_supported"]


def test_shared_core_analysis_is_one_test_and_disables_transfer_claim():
    n = 20
    result = {
        "cases": {
            "uniform_shared_h64": _case_runs(np.zeros(n)),
            "teacher_shared_h64": _case_runs(np.full(n, 0.04)),
        }
    }
    runner.attach_shared_core_analysis(result)
    assert result["primary_multiplicity"]["family"] == [
        "curriculum_efficacy_shared"
    ]
    assert result["predeclared_core_decision"]["efficacy_supported"]
    assert result["predeclared_core_decision"]["transfer_claim_evaluated"] is False
    assert result["predeclared_core_decision"]["transfer_claim_supported"] is None


def test_scale_analysis_has_only_the_three_registered_update_auc_contrasts():
    cases = {}
    for condition in runner.scale_conditions(1e-3):
        value = condition.hindsight_scale * condition.learning_rate
        cases[condition.name] = {
            "runs": [
                {
                    "seed": seed,
                    "auc_mean_pass_by_optimizer_updates": value,
                }
                for seed in (100, 101)
            ]
        }
    result = {"cases": cases}
    runner.attach_scale_analysis(result)
    assert set(result["paired_scale_contrasts"]) == {
        "scale1_minus_scale0_base_lr",
        "scale2_minus_scale1_base_lr",
        "iso_auxiliary_step_interaction",
    }
    assert result["scale_multiplicity"]["metric"] == (
        "auc_mean_pass_by_optimizer_updates"
    )


def _pilot_run(seed, final, auc, sampling):
    regimes = ("dead", "mixed", "all_pass")
    return {
        "seed": seed,
        "numeric_valid": True,
        "accounting_valid": True,
        "verifier_relabel_checks_valid": True,
        "evaluation_cadence_invariant": True,
        "optimizer_updates": 500,
        "initial_mean_pass": 0.20,
        "final_mean_pass": final,
        "auc_mean_pass_by_transitions": auc,
        "transitions": 1_000_000,
        "wall_seconds": 1.0,
        "x_transitions": [0, 1_000_000],
        "mean_pass_curve": [0.20, final],
        "group_diagnostics": [
            {
                "transition_start": 300_000 + 10_000 * index,
                "regime": regime,
                "teacher_tv_from_uniform": 0.2 if sampling == "teacher" else 0.0,
            }
            for index, regime in enumerate(regimes)
        ],
    }


def test_pilot_selection_applies_tie_rule_and_strengthened_gates():
    cases = {}
    for lr, pooled_final in ((1e-4, 0.34), (3e-4, 0.345)):
        for sampling in ("uniform", "teacher"):
            final = pooled_final + (0.02 if sampling == "teacher" else 0.0)
            auc = 0.30 + (0.04 if sampling == "teacher" else 0.0)
            name = f"lr_{runner._float_label(lr)}_{sampling}_shared_h64"
            cases[name] = {
                "config": {
                    "learning_rate": lr,
                    "sampling": sampling,
                },
                "runs": [
                    _pilot_run(seed, final, auc, sampling)
                    for seed in (10_000, 10_001, 10_002)
                ],
            }
    result = {
        "cases": cases,
        "protocol": {
            "eval_interval_transitions": 100_000,
            "eval_n_per_task": 32,
        },
    }
    selection = runner.select_pilot_learning_rate(result)
    assert selection["selected_learning_rate"] == pytest.approx(1e-4)
    assert selection["gates"]["all_pass"]
    assert selection["scale_budget_freeze"]["selected_nonzero_update_budget"] == 400


class _FakeActor:
    def __init__(self, *args, **kwargs):
        self.parameter_count = 640
        self.active_parameter_count = 640
        self.theta = np.zeros(1)
        self.rng = np.random.default_rng(7)
        self.last_update_stats = {}

    def update(self, task_id, trajectories, weights):
        self.theta += 1.0
        self.last_update_stats = {
            "gradient_norm": 1.0,
            "update_norm": 1.0,
            "entropy": 1.0,
            "applied": True,
        }

    def gradient_diagnostics(self, task_id, trajectories, weights):
        return {
            "gradient_norm": 2.0,
            "hypothetical_update_norm": 2.0,
            "mutated": False,
        }


class _FakeSpace:
    def __init__(self, actor, **kwargs):
        self.actor = actor
        self.rng = np.random.default_rng(8)
        self.calls = 0

    def evaluate(self, n, seed):
        return {
            "pass_rates": [0.25] * 8,
            "native_success_rate": 0.25,
            "mean_native_return": -400.0,
            "mean_time_to_goal": 400.0,
            "mean_policy_entropy": 1.0,
        }

    def rollout_group(self, task_id, n_rollouts):
        self.calls += 1
        rewards = np.zeros(n_rollouts) if self.calls == 1 else np.r_[1.0, np.zeros(15)]
        trajectories = [[(np.zeros(6), 0)] for _ in range(n_rollouts)]
        infos = [{"n_steps": 1} for _ in range(n_rollouts)]
        return GroupResult(task_id, rewards, trajectories, infos)

    def relabel(self, group):
        return 6, np.r_[np.ones(8), np.zeros(8)], group.trajectories

    def close(self):
        pass


class _FixedTeacher:
    def __init__(self):
        self.rng = np.random.default_rng(9)

    def distribution(self):
        probabilities = np.zeros(8)
        probabilities[-1] = 1.0
        return probabilities

    def observe(self, task_id, rewards):
        pass


def test_scale_zero_previews_relabel_without_applying_or_counting(monkeypatch):
    monkeypatch.setattr(runner, "TanhCategoricalActor", _FakeActor)
    monkeypatch.setattr(runner, "AcrobotNeuralSpace", _FakeSpace)
    monkeypatch.setattr(runner, "_teacher_for", lambda condition, seed: _FixedTeacher())
    condition = next(
        case for case in runner.scale_conditions(1e-3)
        if case.lr_multiplier == 1.0 and case.hindsight_scale == 0.0
    )
    run = runner.run_condition(
        condition,
        seed=100,
        budget=runner.RunBudget(
            optimizer_update_budget=1,
            transition_safety_cap=100,
        ),
        eval_interval_updates=1,
        eval_n=1,
    )
    assert run["optimizer_updates"] == 1
    assert run["live_applied_updates"] == 1
    assert run["relabel_candidates"] == 1
    assert run["unscaled_aux_gradient_previews"] == 1
    assert run["relabeled_groups"] == 0
    assert not run["auxiliary_gradient_diagnostics"][0]["applied"]


def test_evaluation_cadence_check_covers_parameters_and_counters():
    actor = _FakeActor()

    class MutatingEvaluationSpace(_FakeSpace):
        def evaluate(self, n, seed):
            self.actor.theta += 1.0
            self.actor.last_update_stats = {"illicit_eval_counter": 1}
            return super().evaluate(n, seed)

    env = MutatingEvaluationSpace(actor)
    with pytest.raises(RuntimeError, match="training state"):
        runner._evaluate(env, n=1, seed=123)


def test_artifact_writer_refuses_implicit_overwrite(tmp_path):
    output = tmp_path / "artifact.json"
    runner._write_json_exclusive(output, {"first": True}, overwrite=False)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner._write_json_exclusive(output, {"second": True}, overwrite=False)
