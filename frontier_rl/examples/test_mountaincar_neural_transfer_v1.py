"""Preflight tests for MountainCar neural transfer V1.

Reserved development and confirmatory seeds are never executed here.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from frontier_rl.examples import mountaincar_neural_transfer_v1_core as core
from frontier_rl.examples import run_mountaincar_neural_transfer_v1 as runner


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_registered_matrix_capacity_arithmetic_and_fresh_seed_blocks():
    assert runner.DEVELOPMENT_SEEDS == (17_000, 17_001, 17_002)
    assert runner.CONFIRMATORY_SEEDS == tuple(range(18_000, 18_020))
    audit = runner.seed_collision_audit()
    assert audit["passed"]
    assert not any(audit["collisions"]["rng_roots_vs_training"].values())
    assert not any(audit["collisions"]["between_rng_root_blocks"].values())
    assert [condition.name for condition in runner.CONDITIONS] == [
        "frontier_shared_h64",
        "uniform_shared_h64",
        "hardest_shared_h64",
        "uniform_disjoint_total_h8x8",
        "uniform_disjoint_active_h64x8",
    ]
    assert [condition.sampling for condition in runner.CONDITIONS] == [
        "frontier_u16",
        "uniform",
        "hardest_only",
        "uniform",
        "uniform",
    ]
    observed = {}
    for mode in runner.CAPACITY_CONTRACT:
        actor = core.MountainCarNeuralActor(mode=mode)
        observed[mode] = (actor.parameter_count, actor.active_parameter_count)
    assert observed == {
        core.MountainCarNeuralActor.SHARED: (384, 384),
        core.MountainCarNeuralActor.DISJOINT_TOTAL: (384, 48),
        core.MountainCarNeuralActor.DISJOINT_ACTIVE: (3072, 384),
    }


def test_v5_lock_protected_bytes_remain_unchanged():
    lock_path = PROJECT_ROOT / "frontier_rl/examples/ACROBOT_HINDSIGHT_V5B_LOCK.json"
    lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    assert lock_hash == (
        "dfc930bbaf8e51c96fd1dab5851179457fce4f151def8c138ddf0cf17402bcf2"
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert set(lock["source_sha256"]) == {
        "frontier_rl/examples/run_acrobot_hindsight_v5.py",
        "frontier_rl/examples/analyze_acrobot_hindsight_v5.py",
        "frontier_rl/examples/test_run_acrobot_hindsight_v5.py",
        "frontier_rl/examples/test_analyze_acrobot_hindsight_v5.py",
        "frontier_rl/examples/ACROBOT_HINDSIGHT_PROTOCOL_V5.md",
        "frontier_rl/examples/run_acrobot_hindsight_v4.py",
        "frontier_rl/examples/run_acrobot_neural.py",
        "frontier_rl/adapters/acrobot_neural.py",
        "frontier_rl/teacher.py",
        "frontier_rl/estimators.py",
        "frontier_rl/interfaces.py",
        "frontier_rl/examples/test_acrobot_neural.py",
        "frontier_rl/examples/test_run_acrobot_neural.py",
        "frontier_rl/examples/test_run_acrobot_hindsight_v4.py",
        "frontier_rl/__init__.py",
        "frontier_rl/trainer.py",
        "frontier_rl/adapters/__init__.py",
    }
    for relative, expected in lock["source_sha256"].items():
        observed = hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected, relative
    examples = PROJECT_ROOT / "frontier_rl/examples"
    stage_b = json.loads(
        (examples / "acrobot_hindsight_v5b_factorial.json").read_text(encoding="utf-8")
    )
    assert stage_b["provenance"]["source_lock_sha256"] == lock_hash
    assert stage_b["provenance"]["source_sha256"] == lock["source_sha256"]
    dependencies = {
        "amendment_sha256": examples / "ACROBOT_HINDSIGHT_V5B_AMENDMENT.json",
        "stage_a_artifact_sha256": examples / "acrobot_hindsight_v5a_feasibility.json",
        "stage_a_independent_verification_sha256": examples
        / "acrobot_hindsight_v5a_verification.json",
    }
    for key, path in dependencies.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == lock[key]
    stage_a = json.loads(
        dependencies["stage_a_artifact_sha256"].read_text(encoding="utf-8")
    )
    gates_payload = json.dumps(
        stage_a["stage_a_learning_outcome_blind_gates"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(gates_payload).hexdigest() == lock["stage_a_gates_sha256"]


def test_thresholds_are_strictly_nested_and_end_at_native_flag():
    assert core.THRESHOLDS == (
        -0.375,
        -0.250,
        -0.125,
        0.0,
        0.125,
        0.250,
        0.375,
        0.500,
    )
    assert len(core.THRESHOLDS) == 8
    assert all(
        left < right for left, right in zip(core.THRESHOLDS, core.THRESHOLDS[1:])
    )
    assert core.THRESHOLDS[0] > -0.4
    assert core.THRESHOLDS[-1] == 0.5


def test_source_manifest_is_the_exact_eleven_file_evidence_chain():
    expected = (
        "frontier_rl/examples/mountaincar_neural_transfer_v1_core.py",
        "frontier_rl/examples/run_mountaincar_neural_transfer_v1.py",
        "frontier_rl/examples/analyze_mountaincar_neural_transfer_v1.py",
        "frontier_rl/examples/MOUNTAINCAR_NEURAL_TRANSFER_PROTOCOL_V1.md",
        "frontier_rl/examples/test_mountaincar_neural_transfer_v1.py",
        "frontier_rl/examples/test_analyze_mountaincar_neural_transfer_v1.py",
        "frontier_rl/__init__.py",
        "frontier_rl/interfaces.py",
        "frontier_rl/teacher.py",
        "frontier_rl/estimators.py",
        "frontier_rl/trainer.py",
    )
    assert runner.SOURCE_RELATIVE_PATHS == expected
    assert tuple(runner._source_hashes()) == expected


def test_practical_maxrl_coefficients_have_registered_group_contract():
    rewards = np.array([1.0] * 8 + [0.0] * 8)
    weights = core.practical_maxrl_weights(rewards)
    assert np.all(weights[:8] == 1.0 / 16.0)
    assert np.all(weights[8:] == -1.0 / 16.0)
    assert float(weights.sum()) == 0.0
    assert np.array_equal(core.practical_maxrl_weights(np.zeros(16)), np.zeros(16))
    assert np.array_equal(core.practical_maxrl_weights(np.ones(16)), np.zeros(16))
    for successes in (1, 4, 8, 15):
        rng = np.random.default_rng(successes)
        rewards = np.array([1.0] * successes + [0.0] * (16 - successes))
        rng.shuffle(rewards)
        expected = rewards / successes - 1.0 / 16.0
        assert np.array_equal(core.practical_maxrl_weights(rewards), expected)
    with pytest.raises(ValueError, match="exactly 16"):
        core.practical_maxrl_weights(np.ones(15))


def _correlated_group():
    observations = [np.array([-0.5, 0.01], dtype=np.float64)] * 16
    trajectories = [
        [(observation.copy(), 2 if index < 8 else 0)]
        for index, observation in enumerate(observations)
    ]
    rewards = np.array([1.0] * 8 + [0.0] * 8)
    return trajectories, core.practical_maxrl_weights(rewards)


def test_shared_actor_is_task_agnostic_and_disjoint_update_is_slot_local():
    observation = np.array([-0.5, 0.01])
    trajectories, weights = _correlated_group()

    shared = core.MountainCarNeuralActor(
        mode=core.MountainCarNeuralActor.SHARED, parameter_seed=7
    )
    assert np.array_equal(
        shared.probabilities(observation, 0), shared.probabilities(observation, 7)
    )
    stats = shared.update(3, trajectories, weights)
    assert stats["applied"]
    assert np.array_equal(
        shared.slot_parameter_vector(0), shared.slot_parameter_vector(7)
    )

    for mode in (
        core.MountainCarNeuralActor.DISJOINT_TOTAL,
        core.MountainCarNeuralActor.DISJOINT_ACTIVE,
    ):
        disjoint = core.MountainCarNeuralActor(mode=mode, parameter_seed=7)
        before = [disjoint.slot_parameter_vector(task) for task in range(8)]
        stats = disjoint.update(3, trajectories, weights)
        assert stats["applied"]
        for task in range(8):
            changed = not np.array_equal(
                disjoint.slot_parameter_vector(task), before[task]
            )
            assert changed == (task == 3)


def test_frozen_group_gradient_does_not_mutate_actor():
    actor = core.MountainCarNeuralActor(parameter_seed=4)
    trajectories, weights = _correlated_group()
    before = actor.parameter_vector()
    gradient, diagnostics = actor.group_gradient(0, trajectories, weights)
    assert np.array_equal(actor.parameter_vector(), before)
    assert diagnostics["frozen_group_parameters"]
    assert diagnostics["gradient_norm"] > 0
    assert set(gradient) == {"W_in", "b_hidden", "W_out"}
    snapshot = actor.parameter_state()
    actor.W_out += 1.0
    actor.load_parameter_state(snapshot)
    assert np.array_equal(actor.parameter_vector(), before)


def test_evaluation_is_deterministic_nested_and_preserves_training_state():
    space = core.MountainCarSparseGoalSpace(seed=913)
    try:
        parameter_before = space.actor.parameter_sha256()
        episode_before = copy.deepcopy(space.episode_rng.bit_generator.state)
        action_before = copy.deepcopy(space.actor.action_rng.bit_generator.state)
        first = space.evaluate(n=2, seed=700_913)
        second = space.evaluate(n=2, seed=700_913)
        assert first == second
        assert first["shared_nested_pass_rates"] is True
        samples = np.asarray(first["max_position_samples"])
        assert samples.shape == (8, 2)
        assert np.array_equal(samples, np.repeat(samples[:1], 8, axis=0))
        recomputed = [
            float(np.mean(samples[task] >= threshold))
            for task, threshold in enumerate(core.THRESHOLDS)
        ]
        assert recomputed == first["pass_rates"]
        assert all(
            first[key] is True
            for key in (
                "training_episode_rng_preserved",
                "training_action_rng_preserved",
                "training_parameters_preserved",
            )
        )
        assert space.actor.parameter_sha256() == parameter_before
        assert space.episode_rng.bit_generator.state == episode_before
        assert space.actor.action_rng.bit_generator.state == action_before
        assert not hasattr(space, "relabel")
    finally:
        space.close()


def test_fixed_budget_auc_interpolates_back_from_complete_group_overshoot():
    x = [0, 100, 210]
    y = [0.0, 0.5, 1.0]
    at_200 = 0.5 + (100.0 / 110.0) * 0.5
    expected = (100 * 0.25 + 100 * (0.5 + at_200) / 2.0) / 200.0
    assert runner.fixed_budget_auc(y, x, 200) == pytest.approx(expected)
    assert runner.fixed_budget_value(y, x, 200) == pytest.approx(at_200)
    with pytest.raises(ValueError, match="invalid curve"):
        runner.fixed_budget_auc(y, [0, 100, 100], 100)


def test_sampler_contracts_are_deterministic_and_frontier_has_floor():
    condition = runner.CONDITIONS[0]
    left = runner._sampler(condition, 777)
    right = runner._sampler(condition, 777)
    for successes in (0, 8, 16):
        left_task, left_p = left.draw()
        right_task, right_p = right.draw()
        assert left_task == right_task
        assert np.array_equal(left_p, right_p)
        assert left_p.min() >= runner.TEACHER_FLOOR / core.N_TASKS
        rewards = np.array([1.0] * successes + [0.0] * (16 - successes))
        left.observe(left_task, rewards)
        right.observe(right_task, rewards)

    reference_rng = np.random.default_rng(123 + runner.TEACHER_SEED_BASE)
    sampled_pass = reference_rng.beta(np.ones(8), np.ones(8))
    utility = np.maximum(1.0 - (1.0 - sampled_pass) ** 16 - sampled_pass, 0.0) ** 4
    expected_p = 0.9 * utility / utility.sum() + 0.1 / 8
    expected_task = int(reference_rng.choice(8, p=expected_p))
    observed_task, observed_p = runner._sampler(condition, 123).draw()
    assert observed_task == expected_task
    assert np.allclose(observed_p, expected_p, rtol=0.0, atol=1e-15)

    uniform = runner._sampler(runner.CONDITIONS[1], 123)
    uniform_task, uniform_p = uniform.draw()
    uniform_rng = np.random.default_rng(123 + runner.TEACHER_SEED_BASE)
    expected_uniform = np.full(8, 1.0 / 8)
    assert uniform_task == int(uniform_rng.choice(8, p=expected_uniform))
    assert np.array_equal(uniform_p, expected_uniform)

    hardest = runner._sampler(runner.CONDITIONS[2], 777)
    task, probabilities = hardest.draw()
    assert task == 7
    assert np.array_equal(probabilities, np.eye(8)[-1])


def test_all_uniform_controls_use_one_outcome_independent_registered_sequence():
    seed = 17_000
    expected = runner._uniform_task_sequence(seed, 128)
    assert len(expected) == 128
    for condition in (runner.CONDITIONS[1], runner.CONDITIONS[3], runner.CONDITIONS[4]):
        sampler = runner._sampler(condition, seed)
        observed = []
        for group in range(128):
            task, _ = sampler.draw()
            observed.append(task)
            # Deliberately incompatible outcome histories must not alter later draws.
            successes = group % 17
            sampler.observe(
                task,
                np.array([1.0] * min(successes, 16) + [0.0] * max(16 - successes, 0)),
            )
        assert tuple(observed) == expected
    assert runner._uniform_task_schedule_sha256(seed) == runner._canonical_hash(
        list(runner._uniform_task_sequence(seed))
    )


def test_protocol_makes_hardest_goal_primary_and_mean_pass_supporting():
    protocol = runner.schedule()["protocol"]
    assert protocol["primary_metric"].startswith("hardest-goal pass AUC")
    assert protocol["primary_estimand"].startswith("paired mean method difference")
    assert protocol["supporting_metric"].startswith("target-uniform mean-pass AUC")
    assert "cannot rescue" in protocol["claim_rule"]
    assert protocol["uniform_task_schedule"]["registered_length"] == 31_250


def test_excluded_tiny_gymnasium_run_obeys_transition_and_no_hindsight_contract():
    run = runner.run_condition(
        runner.CONDITIONS[0],
        runner.SMOKE_SEED,
        transition_budget=8_000,
        eval_interval_transitions=4_000,
        eval_n=1,
    )
    assert run["seed"] not in runner.DEVELOPMENT_SEEDS
    assert run["seed"] not in runner.CONFIRMATORY_SEEDS
    assert run["hindsight"] is False
    assert run["relabel_candidates"] == run["relabeled_groups"] == 0
    assert 8_000 <= run["transitions"] <= 11_200
    assert run["x_transitions"][0] == 0
    assert run["x_transitions"] == [0, 4_000, 8_000]
    assert run["evaluation_trigger_transitions"][-1] == run["transitions"]
    assert run["evaluation_policy_sources"][-1] == "pre_crossing_group"
    assert len(run["x_transitions"]) == 3
    assert all(run["evaluation_rng_preserved"])
    assert run["actor"]["parameter_count"] == 384
    assert run["numeric_valid"]
    expected_episode_rng = np.random.default_rng(
        runner.SMOKE_SEED + runner.TRAINING_EPISODE_SEED_OFFSET
    )
    for group in run["group_diagnostics"]:
        assert len(group["rollouts"]) == 16
        assert group["n_transitions"] == sum(
            rollout["n_steps"] for rollout in group["rollouts"]
        )
        assert group["success_count"] == sum(
            rollout["success"] for rollout in group["rollouts"]
        )
        threshold = core.THRESHOLDS[group["task_id"]]
        for rollout in group["rollouts"]:
            assert rollout["episode_seed"] == int(
                expected_episode_rng.integers(0, 2**31 - 1)
            )
            assert rollout["native_reward_sum"] == -rollout["n_steps"]
            if rollout["success"]:
                assert rollout["max_position_before_final"] < threshold
                assert rollout["final_position"] >= threshold
            else:
                assert rollout["max_position"] < threshold
                assert rollout["n_steps"] == 200
                assert rollout["native_terminated"] or rollout["native_truncated"]
    samples = np.asarray(run["evaluation_max_position_samples"])
    assert samples.shape == (3, 8, 1)
    for checkpoint, matrix in enumerate(samples):
        recomputed = [
            float(np.mean(matrix[task] >= threshold))
            for task, threshold in enumerate(core.THRESHOLDS)
        ]
        assert recomputed == run["pass_rate_curve"][checkpoint]


def test_default_cli_is_inspection_only_and_development_requires_lock(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        runner,
        "run_condition",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("seed executed")),
    )
    assert runner.main([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert "unsealed" in output["registration_status"]
    assert output["confirmatory_execution_available"] is False
    with pytest.raises(SystemExit):
        runner.main(["--mode", "development", "--output", "/tmp/never.json"])


def test_development_lock_is_exact_and_public_runner_fails_closed_before_env_creation(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        core.gym,
        "make",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("env created")),
    )
    monkeypatch.setattr(
        runner,
        "MountainCarNeuralActor",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("actor created")),
    )
    for seed in runner.CONFIRMATORY_SEEDS:
        with pytest.raises(RuntimeError, match="no V1 execution path"):
            runner.run_condition(runner.CONDITIONS[0], seed)
    for seed in runner.DEVELOPMENT_SEEDS:
        with pytest.raises(RuntimeError, match="requires an exact V1 lock"):
            runner.run_condition(runner.CONDITIONS[0], seed)
    missing = tmp_path / "missing-lock.json"
    with pytest.raises(RuntimeError, match="lock is missing"):
        runner.run_development(
            tmp_path / "never-created.json", development_lock=missing
        )

    lock = tmp_path / "development-lock.json"
    record = runner.seal_development(lock)
    assert record["payload"] == runner._development_lock_payload()
    assert record["payload"]["scope"] == "development_only"
    assert record["payload"]["confirmatory_execution_available"] is False
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.seal_development(lock)
    tampered = json.loads(lock.read_text(encoding="utf-8"))
    tampered["protocol"]["learning_rate"] = 1e-3
    lock.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from current"):
        runner._load_development_lock(lock)


def test_registered_development_rejects_custom_config_before_lock_or_actor(monkeypatch):
    monkeypatch.setattr(
        runner,
        "MountainCarNeuralActor",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("actor created")),
    )
    with pytest.raises(RuntimeError, match="locked V1 config"):
        runner.run_condition(
            runner.CONDITIONS[0],
            runner.DEVELOPMENT_SEEDS[0],
            transition_budget=8_000,
            eval_interval_transitions=4_000,
            eval_n=1,
        )
    custom = runner.Condition("custom", "uniform", core.MountainCarNeuralActor.SHARED)
    with pytest.raises(RuntimeError, match="locked V1 condition"):
        runner.run_condition(custom, runner.DEVELOPMENT_SEEDS[0])


def test_seed_is_one_immutable_primitive_before_any_guard_or_actor(monkeypatch):
    class StatefulIntegerLike:
        def __init__(self):
            self.calls = 0

        def __int__(self):
            self.calls += 1
            return runner.SMOKE_SEED if self.calls == 1 else runner.DEVELOPMENT_SEEDS[0]

    value = StatefulIntegerLike()
    monkeypatch.setattr(
        runner,
        "MountainCarNeuralActor",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("actor created")),
    )
    with pytest.raises(TypeError, match="primitive Python int"):
        runner.run_condition(runner.CONDITIONS[0], value)
    assert value.calls == 0
    with pytest.raises(TypeError, match="primitive Python int"):
        runner.run_condition(runner.CONDITIONS[0], True)


def test_core_cannot_create_an_environment_for_reserved_seeds_without_authorization(
    monkeypatch,
):
    monkeypatch.setattr(
        core.gym,
        "make",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("env created")),
    )
    for seed in runner.DEVELOPMENT_SEEDS:
        with pytest.raises(RuntimeError, match="lacks validated-lock authorization"):
            core.MountainCarSparseGoalSpace(seed=seed)
    for seed in runner.CONFIRMATORY_SEEDS:
        with pytest.raises(RuntimeError, match="no MountainCar V1 core path"):
            core.MountainCarSparseGoalSpace(seed=seed)
