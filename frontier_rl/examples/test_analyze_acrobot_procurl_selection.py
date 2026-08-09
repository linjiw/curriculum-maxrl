"""Adversarial, synthetic-only tests for the independent analyzer."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict

import pytest

pytest.importorskip("gymnasium")

from frontier_rl.examples import analyze_acrobot_procurl_selection as analysis
from frontier_rl.examples import run_acrobot_procurl_selection as runner
from frontier_rl.examples.test_run_acrobot_procurl_selection import (
    OneStepEnv,
    TerminalSuccessEnv,
    make_all_four_synthetic_runs,
)

# Outcome-free episode entropy terms from the invalid pre-gate wave.  This
# fixed vector exercises Python 3.12's compensated built-in sum: a naive
# iterative accumulator differs by more than the registered 1e-12 tolerance.
ENTROPY_SUM_REGRESSION_TERMS = [
    549.289282936898,
    549.2857356245377,
    549.2872495862491,
    549.2861638667421,
    549.2857443317533,
    549.2867282464132,
    549.2895726406483,
    549.2865534076619,
    549.2860469113732,
    549.288622325428,
    549.2862002480819,
    549.2873496418205,
    549.2863843248118,
    549.2858695926706,
    549.2856971184226,
    549.2867986978171,
    549.2863129624436,
    549.2865341586298,
    549.2861813786084,
    549.2860759557776,
    549.2857224574934,
    549.2861795828343,
    549.2859069840187,
    549.2864419313281,
    549.2864596960767,
    549.2865119613786,
    549.28707379166,
    549.2865645008778,
    549.2859656700225,
    549.2861366690263,
    549.2856495032845,
    549.2865620090922,
]


def _quick_raw(monkeypatch, *, environment_class=OneStepEnv, eval_n=2) -> dict:
    monkeypatch.setattr(runner, "QUICK_EVAL_N", eval_n)
    monkeypatch.setattr(analysis, "QUICK_EVAL_N", eval_n)
    runs = make_all_four_synthetic_runs(
        monkeypatch, environment_class=environment_class
    )
    monkeypatch.setattr(analysis, "QUICK_PAID_BUDGET", 6_500)
    monkeypatch.setattr(analysis, "REGULAR_EVAL_INTERVAL_PAID", 5_000)
    runtime = {
        "python_implementation": "CPython",
        "python": "3.12.13",
        "platform": "synthetic",
        "machine": "synthetic",
        "numpy": "2.5.1",
        "gymnasium": "1.3.0",
    }
    source = {
        relative: "0" * 64 for relative in analysis.EXPECTED_SOURCE_RELATIVE_PATHS
    }
    return {
        "schema": analysis.RAW_SCHEMA,
        "artifact_state": "complete",
        "provenance": {
            "created_utc": "2026-08-08T00:00:00+00:00",
            "runtime": runtime,
            "source_lock_relative_path": (
                "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_LOCK.json"
            ),
            "source_lock_sha256": None,
            "source_lock_enforced": False,
            "source_sha256": source,
            "git_commit": None,
            "git_status_porcelain": [],
            "seed_collision_audit": analysis._independent_seed_collision_audit(),
            "upstream_procurl_commit": analysis.UPSTREAM_PROCURL_COMMIT,
            "upstream_code_copied": False,
            "public_preexecution_registration": False,
        },
        "protocol": runner._protocol("quick"),
        "run_failures": [],
        "cases": {
            arm.name: {
                "config": asdict(arm),
                "summary": runner._case_summary([runs[arm.name]]),
                "runs": [runs[arm.name]],
            }
            for arm in runner.ARMS
        },
    }


@pytest.fixture
def validated_raw(monkeypatch):
    raw = _quick_raw(monkeypatch)
    validated = analysis.validate_raw_artifact(raw)
    assert validated["strict_valid"] is True
    return raw, validated


def _reject(raw: dict, pattern: str | None = None):
    context = (
        pytest.raises(ValueError, match=pattern)
        if pattern
        else pytest.raises(ValueError)
    )
    with context:
        analysis.validate_raw_artifact(raw)


def _naive_iterative_sum(values: list[float]) -> float:
    total = 0.0
    for value in values:
        total += value
    return total


def _install_entropy_regression_vector(raw: dict) -> None:
    compensated = sum(ENTROPY_SUM_REGRESSION_TERMS)
    count = 500 * len(ENTROPY_SUM_REGRESSION_TERMS)
    for case in raw["cases"].values():
        for run in case["runs"]:
            for evaluation in run["evaluation_records"]:
                assert len(evaluation["episode_records"]) == len(
                    ENTROPY_SUM_REGRESSION_TERMS
                )
                for episode, entropy_term in zip(
                    evaluation["episode_records"],
                    ENTROPY_SUM_REGRESSION_TERMS,
                    strict=True,
                ):
                    episode["transitions"] = 500
                    episode["native_return"] = -500.0
                    episode["censored_time_to_goal"] = 500.0
                    episode["policy_entropy_sum"] = entropy_term
                    episode["policy_entropy_count"] = 500
                    episode["action_count"] = 500
                evaluation["mean_native_return"] = -500.0
                evaluation["mean_censored_time_to_goal"] = 500.0
                evaluation["policy_entropy_sum"] = compensated
                evaluation["policy_entropy_count"] = count
                evaluation["mean_policy_entropy"] = compensated / count


def test_independent_constants_protocol_and_source_manifest_match_runner():
    assert analysis.ARM_NAMES == tuple(arm.name for arm in runner.ARMS)
    assert analysis._independent_locked_schedule() == runner._locked_schedule()
    assert analysis._independent_seed_collision_audit() == runner.seed_collision_audit()
    assert analysis.EXPECTED_SOURCE_RELATIVE_PATHS == runner.SOURCE_RELATIVE_PATHS
    assert analysis._independent_protocol("quick", None) == runner._protocol("quick")


def test_all_four_ledgers_replay_and_cross_arm_crn_is_positive(validated_raw):
    _, validated = validated_raw
    assert validated["cross_arm_crn_invariants"] == {
        "passed": True,
        "paired_seeds_checked": [21_400],
        "selection_rng_stream_replayed": True,
        "student_reset_stream_paired": True,
        "probe_coordinates_paired": True,
        "uniform_mechanics_paired_on_overlap": True,
        "same_actor_evaluations_paired": True,
    }
    for arm in analysis.ARM_NAMES:
        record = validated["by_case"][arm][0]
        assert record["strict_valid"] is True
        assert (
            0.0
            <= record["derived"]["auc_target_uniform_mean_success_fixed_paid_budget"]
            <= 1.0
        )


def test_terminal_success_return_strict_validation_and_native_variation_gate(
    monkeypatch, validated_raw
):
    _, no_success_validated = validated_raw
    terminal_raw = _quick_raw(monkeypatch, environment_class=TerminalSuccessEnv)
    terminal_validated = analysis.validate_raw_artifact(terminal_raw)
    for arm in analysis.ARM_NAMES:
        record = terminal_validated["by_case"][arm][0]
        assert record["strict_valid"] is True
        for evaluation in record["raw"]["evaluation_records"]:
            assert evaluation["native_success_rate"] == 1.0
            assert evaluation["mean_native_return"] == 0.0
            assert all(
                episode["native_success"] is True
                and episode["transitions"] == 1
                and episode["native_return"] == 0.0
                for episode in evaluation["episode_records"]
            )

    # The gate consumes only already-strict diagnostics. Pool one strictly
    # all-success artifact with one strictly no-success diagnostic stream.
    gate_validated = copy.deepcopy(terminal_validated)
    gate_validated["mode"] = "development"
    gate_validated["by_case"]["ordinary_uniform"][0]["evaluation_diagnostics"][
        "native_values"
    ] = no_success_validated["by_case"]["ordinary_uniform"][0][
        "evaluation_diagnostics"
    ]["native_values"]
    source_verification = {
        "passed": True,
        "runtime": terminal_raw["provenance"]["runtime"],
        "source_lock_sha256": "6" * 64,
        "checked_source_files": sorted(terminal_raw["provenance"]["source_sha256"]),
    }
    gate = analysis.development_gates(
        gate_validated,
        source_verification,
        raw_artifact_relative_path="evidence/synthetic-development.json",
        raw_artifact_sha256="7" * 64,
    )
    assert gate["gates"]["pooled_native_evaluation_values_vary"] is True
    assert gate["diagnostics"]["n_distinct_native_evaluation_values"] == 2


def test_python312_compensated_entropy_sum_fixed_regression_vector():
    compensated = sum(ENTROPY_SUM_REGRESSION_TERMS)
    naive = _naive_iterative_sum(ENTROPY_SUM_REGRESSION_TERMS)
    assert compensated == 17577.170278713882
    assert naive == 17577.170278713875
    assert compensated - naive == 7.275957614183426e-12


def test_eval_n32_all_arms_survives_json_roundtrip_and_strict_replay(monkeypatch):
    raw = _quick_raw(monkeypatch, eval_n=32)
    roundtripped = json.loads(json.dumps(raw, allow_nan=False))
    validated = analysis.validate_raw_artifact(roundtripped)
    assert validated["strict_valid"] is True
    for arm_name in analysis.ARM_NAMES:
        run = validated["by_case"][arm_name][0]["raw"]
        assert all(
            record["evaluation_shared_full_horizon_trajectories"] == 32
            and len(record["episode_records"]) == 32
            for record in run["evaluation_records"]
        )


def test_strict_replay_accepts_builtin_sum_and_rejects_naive_sum(monkeypatch):
    raw = _quick_raw(monkeypatch, eval_n=32)
    _install_entropy_regression_vector(raw)
    roundtripped = json.loads(json.dumps(raw, allow_nan=False))
    assert analysis.validate_raw_artifact(roundtripped)["strict_valid"] is True

    forged = copy.deepcopy(roundtripped)
    evaluation = forged["cases"]["ordinary_uniform"]["runs"][0]["evaluation_records"][0]
    evaluation["policy_entropy_sum"] = _naive_iterative_sum(
        ENTROPY_SUM_REGRESSION_TERMS
    )
    _reject(forged, "evaluation policy_entropy_sum mismatch")


def test_forged_extra_evaluation_cannot_change_auc(validated_raw):
    raw, _ = validated_raw
    forged = copy.deepcopy(raw)
    run = forged["cases"]["ordinary_uniform"]["runs"][0]
    extra = copy.deepcopy(run["evaluation_records"][-1])
    extra["evaluation_id"] = len(run["evaluation_records"])
    run["evaluation_records"].append(extra)
    _reject(forged, "missing or extra")


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        (
            lambda run: run["evaluation_records"][-1].__setitem__("kind", "mystery"),
            "unknown",
        ),
        (
            lambda run: run["evaluation_records"][-1].__setitem__(
                "paid_transitions",
                run["evaluation_records"][-1]["paid_transitions"] + 1,
            ),
            "paid ledger",
        ),
        (
            lambda run: run["group_records"][0].__setitem__(
                "selection_probabilities_before_group", 0.125
            ),
            "shape",
        ),
        (
            lambda run: run["group_records"][0]["update"].__setitem__("weights", 0.0),
            "weights mismatch",
        ),
    ],
)
def test_unknown_impossible_and_scalar_broadcast_tampers_fail(
    validated_raw, mutation, pattern
):
    raw, _ = validated_raw
    forged = copy.deepcopy(raw)
    mutation(forged["cases"]["ordinary_uniform"]["runs"][0])
    _reject(forged, pattern)


def test_selection_rng_task_is_replayed_not_merely_range_checked(validated_raw):
    raw, _ = validated_raw
    forged = copy.deepcopy(raw)
    group = forged["cases"]["ordinary_uniform"]["runs"][0]["group_records"][0]
    group["task_id"] = (group["task_id"] + 1) % 8
    group["threshold"] = analysis.THRESHOLDS[group["task_id"]]
    _reject(forged, "uniform/task replay")


def test_cross_arm_pairing_detects_mechanical_divergence(validated_raw):
    raw, _ = validated_raw
    forged = copy.deepcopy(raw)
    rollout = forged["cases"]["probe_sham_uniform_f5120"]["runs"][0]["group_records"][
        0
    ]["student_rollout_records"][0]
    rollout["action_sha256"] = "1" * 64
    _reject(forged, "sham/ordinary overlapping")


def test_same_actor_fingerprint_requires_identical_episode_outputs(validated_raw):
    raw, _ = validated_raw
    forged = copy.deepcopy(raw)
    evaluations = forged["cases"]["procurl_env_b20_f5120"]["runs"][0][
        "evaluation_records"
    ]
    pre = next(record for record in evaluations if record["kind"] == "pre_probe")
    post = evaluations[pre["evaluation_id"] + 1]
    pre["episode_records"][0]["action_sha256"] = "2" * 64
    post["episode_records"][0]["action_sha256"] = "2" * 64
    _reject(forged, "same actor fingerprint")


def test_exact_top_nested_and_case_summary_schemas_reject_pollution(validated_raw):
    raw, _ = validated_raw
    for mutate, pattern in (
        (lambda value: value.__setitem__("pollution", True), "raw artifact schema"),
        (
            lambda value: value["cases"]["ordinary_uniform"]["config"].__setitem__(
                "pollution", True
            ),
            "config schema",
        ),
        (
            lambda value: value["cases"]["ordinary_uniform"]["summary"].__setitem__(
                "n_valid", 99
            ),
            "summary does not exactly recompute",
        ),
        (
            lambda value: value["cases"]["ordinary_uniform"]["runs"][0][
                "selection_diagnostics"
            ].__setitem__("pollution", True),
            "selection diagnostics schema",
        ),
    ):
        forged = copy.deepcopy(raw)
        mutate(forged)
        _reject(forged, pattern)


def test_confirmation_without_exact_positive_gate_binding_fails(validated_raw):
    raw, _ = validated_raw
    forged = copy.deepcopy(raw)
    forged["protocol"] = runner._protocol("confirmatory", development_gate=None)
    forged["provenance"]["source_lock_enforced"] = True
    forged["provenance"]["source_lock_sha256"] = "3" * 64
    _reject(forged, "development gate binding")


def test_shared_analyzer_recomputes_exact_bound_development_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(analysis, "PROJECT_ROOT", tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    development_path = evidence / "development.json"
    development_raw = {"synthetic": "development"}
    development_path.write_text(json.dumps(development_raw), encoding="utf-8")
    development_relative = "evidence/development.json"
    development_hash = analysis._sha256(development_path)
    lock_hash = "4" * 64
    source_record = {
        "passed": True,
        "runtime": {
            "python_implementation": "CPython",
            "python": "3.12.13",
            "platform": "synthetic",
            "machine": "synthetic",
            "numpy": "2.5.1",
            "gymnasium": "1.3.0",
        },
        "source_lock_sha256": lock_hash,
        "checked_source_files": ["synthetic.py"],
    }
    gate = {
        "schema": analysis.GATE_SCHEMA,
        "mode": "development",
        "all_gates_passed": True,
        "source_lock_sha256": lock_hash,
        "source_lock_verification": source_record,
        "gates": {name: True for name in analysis.DEVELOPMENT_GATE_NAMES},
        "diagnostics": {"synthetic": True},
        "gate_policy": analysis.DEVELOPMENT_GATE_POLICY,
        "raw_artifact_relative_path": development_relative,
        "raw_artifact_sha256": development_hash,
    }
    gate_path = evidence / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    binding = {
        "relative_path": "evidence/gate.json",
        "sha256": analysis._sha256(gate_path),
        "raw_artifact_relative_path": development_relative,
        "raw_artifact_sha256": development_hash,
        "all_gates_passed": True,
    }
    confirmatory_raw = {
        "schema": analysis.RAW_SCHEMA,
        "artifact_state": "complete",
        "provenance": {"source_lock_sha256": lock_hash},
        "protocol": {"mode": "confirmatory", "development_gate": binding},
        "run_failures": [],
        "cases": {},
    }
    monkeypatch.setattr(
        analysis, "verify_source_lock", lambda raw, lock, path: source_record
    )
    monkeypatch.setattr(
        analysis, "validate_raw_artifact", lambda raw: {"mode": "development"}
    )
    monkeypatch.setattr(
        analysis,
        "development_gates",
        lambda validated, source, **metadata: copy.deepcopy(gate),
    )
    verification = analysis.verify_confirmation_development_gate(
        confirmatory_raw, {}, tmp_path / "lock.json"
    )
    assert verification["passed"] is True
    assert verification["gate_recomputed_exactly"] is True
    assert verification["same_source_lock"] is True


def test_confirmatory_statistics_cannot_run_without_positive_gate_record():
    source = {
        "passed": True,
        "runtime": {key: "synthetic" for key in analysis.RUNTIME_KEYS},
        "source_lock_sha256": "5" * 64,
        "checked_source_files": [],
    }
    with pytest.raises(ValueError, match="requires positive"):
        analysis.confirmatory_analysis(
            {"mode": "confirmatory"}, source, development_gate_verification=None
        )


def test_analyzer_strict_loader_rejects_duplicate_and_nonfinite_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"x": 1, "x": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        analysis.load_strict_json(path, "bad")
    path.write_text('{"x": -Infinity}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        analysis.load_strict_json(path, "bad")


def test_extra_postbudget_group_is_rejected_by_exact_stop_rule(monkeypatch):
    from frontier_rl.examples.test_run_acrobot_procurl_selection import (
        install_synthetic_engine,
    )

    install_synthetic_engine(monkeypatch, budget=15_000, eval_interval=5_000)
    run = runner.run_one(runner.ARMS[2], 21_400, mode="quick")
    assert len(run["group_records"]) >= 3
    run["paid_budget_nominal"] = 6_500
    run["paid_budget_overshoot"] = run["paid_transitions"] - 6_500
    run["selection_diagnostics"]["paid_budget_overshoot"] = (
        run["paid_transitions"] - 6_500
    )
    monkeypatch.setattr(analysis, "QUICK_PAID_BUDGET", 6_500)
    monkeypatch.setattr(analysis, "REGULAR_EVAL_INTERVAL_PAID", 5_000)
    with pytest.raises(
        ValueError,
        match="nonterminal group already exhausted|extra student group started",
    ):
        analysis._validate_run(
            "ordinary_uniform", run, mode="quick", budget=6_500, eval_n=2
        )


def test_paid_auc_duplicate_and_cutoff_policy_is_frozen():
    records = [
        {"paid_transitions": 0, "target_uniform_mean_success": 0.0},
        {"paid_transitions": 20, "target_uniform_mean_success": 0.0},
        {"paid_transitions": 20, "target_uniform_mean_success": 1.0},
        {"paid_transitions": 100, "target_uniform_mean_success": 1.0},
        {"paid_transitions": 120, "target_uniform_mean_success": 1.0},
    ]
    assert math.isclose(
        analysis._normalized_truncated_auc(
            records, axis="paid_transitions", cutoff=100
        ),
        0.8,
    )
    assert math.isclose(
        analysis._normalized_full_auc(records, axis="paid_transitions"),
        100.0 / 120.0,
    )


def test_statistical_rng_quantile_signflip_and_holm_are_deterministic():
    values = [0.1, -0.02, 0.04, 0.08]
    assert analysis.paired_bootstrap_ci(values, seed=31_000, n_resamples=200) == (
        analysis.paired_bootstrap_ci(values, seed=31_000, n_resamples=200)
    )
    assert analysis.monte_carlo_sign_flip_p(
        values, seed=31_001, n_draws=1_000
    ) == analysis.monte_carlo_sign_flip_p(values, seed=31_001, n_draws=1_000)
    holm = analysis._holm({"b": 0.02, "a": 0.01, "c": 0.5}, alpha=0.05)
    assert holm["a"]["holm_adjusted_p"] == 0.03
    assert holm["b"]["holm_adjusted_p"] == 0.04
    assert holm["c"]["holm_adjusted_p"] == 0.5


def test_development_gate_policy_is_strictly_outcome_blind():
    assert analysis.DEVELOPMENT_GATE_POLICY == {
        "outcome_blind": True,
        "uses_arm_contrasts": False,
        "uses_effect_direction": False,
        "uses_confidence_intervals": False,
        "uses_p_values": False,
        "uses_minimum_effect": False,
    }
    assert "arm_contrast" not in " ".join(analysis.DEVELOPMENT_GATE_NAMES)
