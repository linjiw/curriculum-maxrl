"""Focused, outcome-free contracts for the ProCuRL-selection runner."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from frontier_rl.examples import run_acrobot_procurl_selection as runner


class SyntheticActor:
    bad_capacity = False

    def __init__(
        self,
        n_tasks=8,
        hidden_size=64,
        learning_rate=3e-4,
        seed=0,
        mode="shared",
    ):
        del n_tasks, hidden_size, learning_rate, mode
        self.parameters = np.asarray([float(seed % 97), 0.0], dtype=np.float64)
        self.action_rng = np.random.default_rng(seed + 1)
        self.rng = self.action_rng
        self.update_calls = 0
        self.applied_updates = 0
        self.parameter_count = 639 if self.bad_capacity else 640
        self.active_parameter_count = 640
        self.fail_update = False

    def parameter_vector(self):
        return self.parameters.copy()

    def probabilities(self, observation, task_id=0):
        del observation, task_id
        return np.full(3, 1.0 / 3.0)

    def update(self, task_id, trajectories, weights):
        del trajectories, weights
        if self.fail_update:
            raise RuntimeError("synthetic update failure")
        self.update_calls += 1
        self.applied_updates += 1
        self.parameters[1] += 0.001 * (int(task_id) + 1)
        return {
            "gradient_norm": 1.0,
            "update_norm": 3e-4,
            "mean_policy_entropy": 1.0,
            "applied": True,
        }


class SyntheticSpace:
    instances: ClassVar[list[SyntheticSpace]] = []
    fail_rollout = False
    actor_should_fail_update = False

    def __init__(self, actor, thresholds, seed):
        self.actor = actor
        self.thresholds = tuple(thresholds)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed + 10_003)
        self.closed = False
        actor.fail_update = self.actor_should_fail_update
        self.instances.append(self)

    def rollout_group(self, task_id, n_rollouts):
        if self.fail_rollout:
            raise RuntimeError("synthetic rollout failure")
        trajectories, infos, rewards = [], [], []
        threshold = self.thresholds[task_id]
        for rollout in range(n_rollouts):
            reset_seed = int(self.rng.integers(0, 2**31 - 1))
            actions = self.actor.action_rng.choice(
                3, size=400, p=np.full(3, 1.0 / 3.0)
            ).tolist()
            trajectories.append([{"action": int(action)} for action in actions])
            success = rollout % 2 == 0
            infos.append(
                {
                    "n_steps": len(actions),
                    "reset_seed": reset_seed,
                    "max_height": threshold + (0.1 if success else -0.1),
                }
            )
            rewards.append(float(success))
        return SimpleNamespace(
            task_id=task_id,
            rewards=np.asarray(rewards, dtype=np.float64),
            trajectories=trajectories,
            infos=infos,
        )

    def close(self):
        self.closed = True


class OneStepEnv:
    def __init__(self):
        self.closed = False

    def reset(self, seed):
        del seed
        return np.asarray([1.0, 0.0, 1.0, 0.0, 0.0, 0.0]), {}

    def step(self, action):
        del action
        # Tip height is exactly 1.0: strict success for tasks 0--6 only.
        observation = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        return observation, -1.0, False, True, {}

    def close(self):
        self.closed = True


class TerminalSuccessEnv(OneStepEnv):
    """One-step Acrobot terminal with Gymnasium 1.3's zero terminal reward."""

    def step(self, action):
        del action
        observation = np.asarray([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        return observation, 0.0, True, False, {}


def install_synthetic_engine(
    monkeypatch,
    *,
    budget=6_500,
    eval_interval=5_000,
    environment_class=OneStepEnv,
):
    SyntheticSpace.instances = []
    SyntheticSpace.fail_rollout = False
    SyntheticSpace.actor_should_fail_update = False
    SyntheticActor.bad_capacity = False
    monkeypatch.setattr(runner, "TanhCategoricalActor", SyntheticActor)
    monkeypatch.setattr(runner, "AcrobotNeuralSpace", SyntheticSpace)
    monkeypatch.setattr(runner.gymnasium, "make", lambda name: environment_class())
    monkeypatch.setattr(runner, "QUICK_PAID_BUDGET", budget)
    monkeypatch.setattr(runner, "REGULAR_EVAL_INTERVAL_PAID", eval_interval)


def make_all_four_synthetic_runs(monkeypatch, *, environment_class=OneStepEnv):
    install_synthetic_engine(monkeypatch, environment_class=environment_class)
    return {arm.name: runner.run_one(arm, 21_400, mode="quick") for arm in runner.ARMS}


def test_registered_design_and_continuous_range_matching():
    assert [arm.name for arm in runner.ARMS] == [
        "procurl_env_b20_f5120",
        "probe_sham_uniform_f5120",
        "ordinary_uniform",
        "u16_probe_range_matched_f5120",
    ]
    assert [arm.probes for arm in runner.ARMS] == [True, True, False, True]
    assert runner.CONFIRMATORY_SEEDS == tuple(range(21_000, 21_080))
    assert runner.DEVELOPMENT_SEEDS == tuple(range(21_300, 21_303))
    assert runner.QUICK_SEEDS == (21_400,)
    assert runner.PRIOR_LOGICAL_SEED_BLOCKS[
        "invalid_procurl_selection_development_pre_gate_entropy_sum_mismatch"
    ] == tuple(range(21_100, 21_103))
    assert runner.PRIOR_LOGICAL_SEED_BLOCKS[
        "invalid_procurl_selection_quick_pre_gate_entropy_sum_mismatch"
    ] == (21_200,)
    assert runner.PROBES_PER_TASK == 20
    assert runner.REFRESH_STUDENT_TRANSITIONS == 5_120
    assert runner.PROCURL_BETA == 20.0
    assert runner.U16_BETA_CONTINUOUS_RANGE_MATCHED == 6.416133525771289
    assert runner.U16_LATTICE_MAX_LOGIT == 4.97730861318145

    p_star = 1.0 - 16.0 ** (-1.0 / 15.0)
    u_star = 1.0 - (1.0 - p_star) ** 16 - p_star
    assert np.isclose(
        runner.U16_BETA_CONTINUOUS_RANGE_MATCHED * u_star,
        5.0,
        rtol=0.0,
        atol=1e-14,
    )
    lattice = np.arange(21, dtype=np.float64) / 20.0
    lattice_logits = runner.U16_BETA_CONTINUOUS_RANGE_MATCHED * (
        1.0 - (1.0 - lattice) ** 16 - lattice
    )
    assert np.isclose(lattice_logits.max(), runner.U16_LATTICE_MAX_LOGIT, atol=1e-14)


def test_selection_rules_and_refresh_mapping_are_exact():
    p = np.asarray([0.0, 0.05, 0.2, 0.5, 0.7, 0.9, 0.95, 1.0])
    procurl = runner.selection_distribution(runner.ARMS[0], p)
    logits = 20.0 * p * (1.0 - p)
    expected = np.exp(logits - logits.max())
    expected /= expected.sum()
    assert np.allclose(procurl["logits"], logits, rtol=0.0, atol=1e-15)
    assert np.allclose(procurl["probabilities"], expected, rtol=0.0, atol=1e-15)
    assert np.array_equal(
        runner.selection_distribution(runner.ARMS[1], p)["probabilities"],
        np.full(8, 1.0 / 8.0),
    )
    assert np.array_equal(
        runner.selection_distribution(runner.ARMS[2], None)["probabilities"],
        np.full(8, 1.0 / 8.0),
    )
    u16 = runner.selection_distribution(runner.ARMS[3], p)
    utility = 1.0 - (1.0 - p) ** 16 - p
    assert np.allclose(u16["utility"], utility, rtol=0.0, atol=1e-15)
    assert runner.crossed_refresh_boundaries(4_000, 11_000) == [5_120, 10_240]


def test_collision_free_coordinate_encoding_is_injective_and_bounded():
    audit = runner.seed_collision_audit()
    assert audit["passed"] is True
    assert audit["development_seeds"] == [21_300, 21_301, 21_302]
    assert audit["quick_seeds"] == [21_400]
    assert all(not collisions for collisions in audit["logical_collisions"].values())
    assert audit["prior_logical_seed_blocks"][
        "invalid_procurl_selection_development_pre_gate_entropy_sum_mismatch"
    ] == [21_100, 21_101, 21_102]
    assert audit["prior_logical_seed_blocks"][
        "invalid_procurl_selection_quick_pre_gate_entropy_sum_mismatch"
    ] == [21_200]
    coordinates = {
        runner.probe_episode_seeds(seed, sweep, task, episode)
        for seed in (21_000, 21_079, 21_300, 21_302, 21_400)
        for sweep in (1, 2, runner.MAX_ENCODED_PROBE_SWEEPS)
        for task in (0, 7)
        for episode in (0, 19)
    }
    assert len(coordinates) == 5 * 3 * 2 * 2
    assert all(
        0 <= reset < runner.EVALUATION_RESET_NAMESPACE_BASE for reset, _ in coordinates
    )
    assert all(0 <= action < 2**63 - 1 for _, action in coordinates)
    assert runner._evaluation_episode_seeds(21_400, 31)[0] < 2**31 - 1
    with pytest.raises(ValueError, match="not registered"):
        runner._evaluation_episode_seeds(21_200, 0)
    with pytest.raises(ValueError, match="primitive integers"):
        runner.probe_episode_seeds(True, 1, 0, 0)


def test_all_four_run_one_arms_reach_refresh_and_close(monkeypatch):
    runs = make_all_four_synthetic_runs(monkeypatch)
    assert set(runs) == {arm.name for arm in runner.ARMS}
    for arm in runner.ARMS:
        run = runs[arm.name]
        assert run["paid_transitions"] >= 6_500
        if arm.probes:
            assert run["probe_sweeps"] == 2
            assert run["probe_sweep_records"][1]["trigger"] == "refresh"
            assert (
                run["probe_sweep_records"][1]["crossed_boundary_student_transition"]
                == 5_120
            )
        else:
            assert run["probe_sweeps"] == 0
    assert len(SyntheticSpace.instances) == 4
    assert all(space.closed for space in SyntheticSpace.instances)


@pytest.mark.parametrize(
    "stage,arm_index",
    [
        ("capacity", 0),
        ("initial_evaluation", 0),
        ("initial_probe", 0),
        ("rollout", 0),
        ("refresh_pre_evaluation", 0),
        ("refresh_probe", 0),
        ("update", 0),
        ("terminal_evaluation", 2),
    ],
)
def test_run_one_closes_space_at_every_failure_stage(monkeypatch, stage, arm_index):
    install_synthetic_engine(
        monkeypatch,
        budget=6_500,
        eval_interval=100_000 if stage == "terminal_evaluation" else 5_000,
    )
    if stage == "capacity":
        SyntheticActor.bad_capacity = True
    original_evaluate = runner.evaluate_actor_full_horizon
    evaluation_calls = 0

    def failing_evaluate(*args, **kwargs):
        nonlocal evaluation_calls
        evaluation_calls += 1
        target = 1 if stage == "initial_evaluation" else 2
        if (
            stage
            in {"initial_evaluation", "refresh_pre_evaluation", "terminal_evaluation"}
            and evaluation_calls == target
        ):
            raise RuntimeError(f"synthetic {stage} failure")
        return original_evaluate(*args, **kwargs)

    if stage in {"initial_evaluation", "refresh_pre_evaluation", "terminal_evaluation"}:
        monkeypatch.setattr(runner, "evaluate_actor_full_horizon", failing_evaluate)
    original_sweep = runner.run_probe_sweep

    def failing_sweep(*args, **kwargs):
        trigger = kwargs["trigger"]
        if (stage == "initial_probe" and trigger == "initial") or (
            stage == "refresh_probe" and trigger == "refresh"
        ):
            raise RuntimeError(f"synthetic {stage} failure")
        return original_sweep(*args, **kwargs)

    if stage in {"initial_probe", "refresh_probe"}:
        monkeypatch.setattr(runner, "run_probe_sweep", failing_sweep)
    SyntheticSpace.fail_rollout = stage == "rollout"
    SyntheticSpace.actor_should_fail_update = stage == "update"
    with pytest.raises(RuntimeError):
        runner.run_one(runner.ARMS[arm_index], 21_400, mode="quick")
    assert len(SyntheticSpace.instances) == 1
    assert SyntheticSpace.instances[0].closed is True


def test_probe_and_evaluation_fresh_environments_close_on_failure(monkeypatch):
    actor = SyntheticActor(seed=1)
    space = SyntheticSpace(actor, runner.engine.THRESHOLDS, 2)
    selection_rng = np.random.default_rng(3)
    environments: list[OneStepEnv] = []

    class FailingEnv(OneStepEnv):
        def step(self, action):
            raise RuntimeError("environment failure")

    def make(_name):
        env = FailingEnv()
        environments.append(env)
        return env

    monkeypatch.setattr(runner.gymnasium, "make", make)
    with pytest.raises(RuntimeError, match="environment failure"):
        runner.run_probe_sweep(
            runner.ARMS[0],
            actor,
            space,
            selection_rng,
            logical_seed=21_400,
            sweep_ordinal=1,
            trigger="initial",
            crossed_boundary=None,
            student_transitions=0,
            paid_before=0,
            sampled_groups=0,
            optimizer_updates=0,
        )
    with pytest.raises(RuntimeError, match="environment failure"):
        runner.evaluate_actor_full_horizon(
            actor,
            space,
            selection_rng,
            logical_seed=21_400,
            eval_n=2,
            counters={"paid_transitions": 0},
        )
    assert environments and all(env.closed for env in environments)


def test_terminal_success_uses_zero_terminal_reward_in_episode_and_aggregate(
    monkeypatch,
):
    environments: list[TerminalSuccessEnv] = []

    def make(_name):
        env = TerminalSuccessEnv()
        environments.append(env)
        return env

    monkeypatch.setattr(runner.gymnasium, "make", make)
    actor = SyntheticActor(seed=1)
    space = SyntheticSpace(actor, runner.engine.THRESHOLDS, 2)
    score = runner.evaluate_actor_full_horizon(
        actor,
        space,
        np.random.default_rng(3),
        logical_seed=21_400,
        eval_n=2,
        counters={"paid_transitions": 0},
    )
    assert [episode["native_success"] for episode in score["episode_records"]] == [
        True,
        True,
    ]
    assert [episode["transitions"] for episode in score["episode_records"]] == [
        1,
        1,
    ]
    assert [episode["native_return"] for episode in score["episode_records"]] == [
        0.0,
        0.0,
    ]
    assert score["native_success_rate"] == 1.0
    assert score["mean_native_return"] == 0.0
    assert score["mean_censored_time_to_goal"] == 1.0
    assert environments and all(env.closed for env in environments)


def test_strict_json_loader_rejects_duplicate_keys_and_nonfinite(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"x": 1, "x": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner._load_strict_json(path, "bad")
    path.write_text('{"x": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        runner._load_strict_json(path, "bad")


def test_training_state_fingerprint_does_not_change_during_probe(monkeypatch):
    monkeypatch.setattr(runner.gymnasium, "make", lambda name: OneStepEnv())
    actor = SyntheticActor(seed=1)
    space = SyntheticSpace(actor, runner.engine.THRESHOLDS, 2)
    selection_rng = np.random.default_rng(3)
    before = copy.deepcopy(selection_rng.bit_generator.state)
    sweep = runner.run_probe_sweep(
        runner.ARMS[0],
        actor,
        space,
        selection_rng,
        logical_seed=21_400,
        sweep_ordinal=1,
        trigger="initial",
        crossed_boundary=None,
        student_transitions=0,
        paid_before=0,
        sampled_groups=0,
        optimizer_updates=0,
    )
    assert sweep["training_state_preserved"] is True
    assert selection_rng.bit_generator.state == before
