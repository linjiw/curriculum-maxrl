"""Unit tests for frontier_rl. Run: python3 frontier_rl/test_framework.py"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from math import comb

import numpy as np

from frontier_rl import (FrontierTeacher, FrontierTrainer, TrainerConfig,
                         grpo_weights, maxrl_success_weights,
                         maxrl_unbiased_cv_weights, maxrl_weights, rloo_weights)
from frontier_rl.adapters.skill_chain import SkillChainSpace
from frontier_rl.adapters.grid_reach import GridReachSpace, MOVES
from frontier_rl.adapters.cosmos_libero import (CosmosLiberoSpace,
                                                MasteryFrontierTeacher,
                                                PoisonRateMeter)


def test_estimators():
    r = np.array([1., 0., 0., 1.])
    w = maxrl_weights(r)
    assert abs(w.sum()) < 1e-12 and abs(w[0] - (0.5 - 0.25)) < 1e-12
    assert not maxrl_weights(np.zeros(4)).any()
    assert abs(rloo_weights(r).sum()) < 1e-12
    assert abs(grpo_weights(r).sum()) < 1e-12
    print("estimators OK")


def test_positive_part_estimator():
    # success weights kept, failure weights zeroed
    r = np.array([1., 0., 0., 1.])
    w = maxrl_weights(r, positive_part=True)
    assert abs(w[0] - (0.5 - 0.25)) < 1e-12 and w[1] == 0.0 and w[2] == 0.0
    # all-pass self-retirement: K=N => 1/K - 1/N = 0 everywhere
    assert not maxrl_weights(np.ones(8), positive_part=True).any()
    # dead groups still dead
    assert not maxrl_weights(np.zeros(8), positive_part=True).any()
    # exact identity (COSMOS3 Q1): E[sum w+] = pass@N - pass@1 = u(p),
    # so the teacher utility governs the weighted-RFT update exactly
    rng = np.random.default_rng(0)
    N = 8
    for p in (0.05, 0.2, 0.5):
        K = rng.binomial(N, p, 400_000)
        # sum of positive weights given K>=1 is K*(1/K - 1/N) = 1 - K/N
        mass = np.where(K >= 1, 1.0 - K / N, 0.0).mean()
        u = (1.0 - (1.0 - p) ** N) - p
        assert abs(mass - u) < 3e-3, (p, mass, u)
    print("positive-part estimator OK")


def test_maxrl_population_orders():
    """Exact binomial enumeration catches the practical estimator's N-1 shift."""
    for n in (2, 3, 8):
        for p in (0.03, 0.2, 0.7):
            got = {"drop": 0.0, "raw": 0.0, "cv": 0.0, "mass": 0.0}
            for k in range(n + 1):
                prob = comb(n, k) * p ** k * (1 - p) ** (n - k)
                r = np.array([1.0] * k + [0.0] * (n - k))
                score = r - p  # Bernoulli-logit score; grad p = p(1-p)
                wd = maxrl_weights(r)
                got["drop"] += prob * float(wd @ score)
                got["raw"] += prob * float(maxrl_success_weights(r) @ score)
                got["cv"] += prob * float(maxrl_unbiased_cv_weights(r) @ score)
                got["mass"] += prob * float(np.abs(wd).sum())
            grad_p = p * (1 - p)
            w_n = (1 - (1 - p) ** n) / p
            w_nm1 = (1 - (1 - p) ** (n - 1)) / p
            assert np.isclose(got["raw"], w_n * grad_p)
            assert np.isclose(got["cv"], w_n * grad_p)
            assert np.isclose(got["drop"], w_nm1 * grad_p)
            assert np.isclose(
                got["mass"], 2 * ((1 - (1 - p) ** n) - p)
            )
    print("MaxRL population-order identities OK")


def test_teacher_pseudocounts_and_utility():
    t = FrontierTeacher(n_tasks=20, n_rollouts=16, seed=0)
    for _ in range(6):
        t.observe(3, np.array([1.]*4 + [0.]*12))   # frontier p~0.25
        t.observe(4, np.ones(16))                   # mastered
        t.observe(5, np.zeros(16))                  # dead
    d = np.zeros(20)
    for _ in range(300):
        d += t.distribution()
    d /= 300
    assert d[3] > d[4] and d[3] > d[5], (d[3], d[4], d[5])
    assert abs(d.sum() - 1) < 1e-9
    # utility peak location ~ln(N)/N
    p = np.linspace(1e-4, 0.999, 4000)
    peak = p[np.argmax(t.utility(p))]
    assert abs(peak - (1 - 16 ** (-1/15))) < 2e-3
    print("teacher OK")


def test_teacher_state_roundtrip():
    t = FrontierTeacher(5, 8, seed=0)
    t.observe(2, np.array([1., 0., 1., 0., 0., 0., 0., 0.]))
    t2 = FrontierTeacher(5, 8, seed=0)
    t2.load_state_dict(t.state_dict())
    assert np.allclose(t.alpha, t2.alpha) and np.allclose(t.beta, t2.beta)
    assert np.allclose(t.distribution(), t2.distribution())
    print("state roundtrip OK")


def test_trainer_on_skill_chain():
    env = SkillChainSpace(seed=0)
    trainer = FrontierTrainer(env, env, TrainerConfig(seed=0, hindsight=True,
                                                       teacher_gamma=4.0))
    before = env.true_pass_rates().mean()
    stats = trainer.train(steps=60)
    after = env.true_pass_rates().mean()
    assert after > before + 0.15, (before, after)
    assert any(s.relabeled_groups > 0 for s in stats), "hindsight never fired"
    print(f"trainer on skill chain OK ({before:.3f} -> {after:.3f})")


def test_hindsight_contract_gridworld():
    env = GridReachSpace(radius=6, seed=0)
    g = env.rollout_group(5, 16)   # ring 6: hard from scratch
    if g.rewards.sum() == 0:
        rel = env.relabel(g)
        assert rel is not None
        new_task, new_r, new_trajs = rel
        assert 0 <= new_task < env.n_tasks
        assert new_r.sum() >= 1, "relabel must create at least one success"
        ring = new_task + 1
        relabeled_goal = new_trajs[0]["goal"]
        for r, nt in zip(new_r, new_trajs):
            assert np.array_equal(nt["goal"], relabeled_goal)
            if r == 1.0:
                # Semantic validity and fresh-task stopping: the final credited
                # transition is the first hit of the concrete relabeled goal.
                reached = nt["positions"][-1] + MOVES[nt["actions"][-1]]
                assert np.array_equal(reached, nt["goal"])
                assert max(abs(reached[0]), abs(reached[1])) == ring
    print("hindsight contract OK")


def _mock_libero(success_rates, achieved_by_task, achieved_prob=1.0):
    """A 3-task mock predicate world for the cosmos adapter.

    success_rates[t]: sim pass rate of task t; achieved_by_task[t]: predicate
    set a FAILED rollout of t leaves true w.p. achieved_prob per rollout
    (recorded in info["achieved"] at rollout time — what the oracle verifier
    reads back, mirroring predicate extraction from final frames).
    """
    rng = np.random.default_rng(0)

    def rollout_fn(template, init_bin, n):
        tid = TEMPLATE_TO_TID[template]
        r = (rng.random(n) < success_rates[tid]).astype(float)
        trajs = [{"language_goal": template, "actions": np.zeros(3)}
                 for _ in range(n)]
        infos = [{"task": tid, "success": bool(s),
                  "achieved": (set(achieved_by_task[tid])
                               if rng.random() < achieved_prob else set())}
                 for s in r]
        return r, trajs, infos

    def verifier_fn(info):
        return info["achieved"]

    return rollout_fn, verifier_fn


TASKS = [
    (["open(microwave)"], "open the microwave"),
    (["open(microwave)", "in(bowl,microwave)"],
     "put the bowl in the microwave"),
    (["on(plate,table)"], "put the plate on the table"),
]
TEMPLATE_TO_TID = {t: i for i, (_, t) in enumerate(TASKS)}


def test_cosmos_libero_relabel_contracts():
    # task 1 always fails but its rollouts verifiably open the microwave —
    # the proposal's own motivating example
    rollout_fn, verifier_fn = _mock_libero(
        success_rates=[0.5, 0.0, 0.5],
        achieved_by_task={1: ["open(microwave)"], 0: [], 2: []})
    env = CosmosLiberoSpace(TASKS, rollout_fn, verifier_fn)
    g = env.rollout_group(1, 8)
    assert g.rewards.sum() == 0
    new_task, new_r, new_trajs = env.relabel(g)
    # relabeled to the achieved strict sub-conjunction = pool task 0
    assert new_task == 0 and new_r.sum() == 8
    # contract 2: conditioning rewritten to the target's canonical template
    assert all(nt["language_goal"] == "open the microwave" for nt in new_trajs)
    # originals untouched (relabel must not mutate the source group)
    assert all(t["language_goal"] == "put the bowl in the microwave"
               for t in g.trajectories)
    # Q3.4: a failure can never be upgraded to the ORIGINAL task's success —
    # even if the verifier (over-)reports the full goal achieved
    rollout_fn2, _ = _mock_libero([0.5, 0.0, 0.5], {1: [], 0: [], 2: []})
    env2 = CosmosLiberoSpace(
        TASKS, rollout_fn2,
        lambda info: {"open(microwave)", "in(bowl,microwave)"})
    rel = env2.relabel(env2.rollout_group(1, 8))
    assert rel is not None and rel[0] != 1, "upgraded failure to original task"
    print("cosmos relabel contracts OK")


def test_cosmos_poison_meter_gates_vocabulary():
    meter = PoisonRateMeter(precision_gate=0.9)
    # self-verifier: perfect on "open", hallucinates "on" half the time
    for i in range(100):
        oracle = {"open(microwave)"} | ({"on(bowl,plate)"} if i % 2 else set())
        self_p = {"open(microwave)", "on(bowl,plate)"}
        meter.observe(self_p, oracle)
    allowed = meter.allowed_vocabulary()
    assert "open" in allowed and "on" not in allowed
    # the gated adapter then refuses relabels resting on the poisoned class
    rollout_fn, _ = _mock_libero([0.5, 0.0, 0.5], {1: [], 0: [], 2: []})
    env = CosmosLiberoSpace(TASKS, rollout_fn,
                            lambda info: {"on(plate,table)"},   # poisoned class
                            allowed_classes=allowed)
    assert env.relabel(env.rollout_group(1, 8)) is None
    print("poison meter vocabulary gate OK")


def test_cosmos_mastery_split_and_shrinkage():
    rollout_fn, verifier_fn = _mock_libero(
        success_rates=[0.98, 0.0, 0.5],
        achieved_by_task={1: ["open(microwave)"], 0: [], 2: []})
    env = CosmosLiberoSpace(TASKS, rollout_fn, verifier_fn)
    teacher = MasteryFrontierTeacher(env.n_tasks, n_rollouts=8, seed=0)
    for _ in range(8):
        g = env.rollout_group(0, 8)
        teacher.observe(0, g.rewards)
    new_ids = env.split_mastered(teacher, n_bins=4, lam=0.3)
    assert len(new_ids) == 4 and teacher.n_tasks == env.n_tasks
    # children shrink toward the (saturated) parent but far weaker than it:
    # lam scales pseudo-counts, so child evidence dominates after ~2 groups
    child = new_ids[0]
    assert teacher.alpha[child] - 1 < 0.5 * (teacher.alpha[0] - 1)
    p_child = teacher.pass_rate_estimates()[child]
    assert p_child > 0.7, "child prior should inherit the parent's high p"
    # split arms carry init bins; only once per parent
    assert env.tasks[child].init_bin is not None
    assert env.split_mastered(teacher) == []
    # teacher can sample the grown arm set without index errors
    assert teacher.sample_tasks(16).max() < teacher.n_tasks
    print("mastery split + shrinkage OK")


def test_cosmos_posterior_hygiene_end_to_end():
    """Relabels route gradient to the relabeled task but never its posterior."""
    # achieved_prob<1 keeps relabeled groups contrasted (K<N): an all-success
    # relabeled group has zero MaxRL weight by the same K=N self-retirement
    # that retires mastered live tasks — mock must not be degenerate
    rollout_fn, verifier_fn = _mock_libero(
        success_rates=[0.0, 0.0, 0.5],
        achieved_by_task={0: [], 1: ["open(microwave)"], 2: []},
        achieved_prob=0.7)

    updates = []
    class SpyPolicy:
        def update(self, task_id, trajectories, weights):
            updates.append(task_id)

    env = CosmosLiberoSpace(TASKS, rollout_fn, verifier_fn)
    teacher = MasteryFrontierTeacher(env.n_tasks, n_rollouts=8, seed=0)
    trainer = FrontierTrainer(env, SpyPolicy(),
                              TrainerConfig(n_rollouts=8, tasks_per_step=4,
                                            positive_weights=True, seed=0),
                              teacher=teacher)
    trainer.train(steps=15)
    assert 0 in updates, "relabeled gradient never reached task 0"
    # task 0's sim pass rate is 0.0: every direct group fails, so ONLY
    # relabels could have inflated its posterior — alpha must still be prior+0
    assert teacher.alpha[0] == 1.0, "relabel leaked into the posterior (V4)"
    print("posterior hygiene end-to-end OK")


def test_grid_group_update_is_permutation_invariant():
    env_a = GridReachSpace(radius=3, seed=0)
    env_b = GridReachSpace(radius=3, seed=0)
    trajectories = [
        {"positions": [np.array([0, 0]), np.array([1, 0])],
         "actions": [0, 2], "goal": np.array([2, 1])},
        {"positions": [np.array([0, 0])],
         "actions": [1], "goal": np.array([2, 1])},
        {"positions": [np.array([0, 0]), np.array([0, 1])],
         "actions": [3, 2], "goal": np.array([2, 1])},
    ]
    weights = np.array([0.5, -0.25, -0.25])
    env_a.update(1, trajectories, weights)
    order = np.array([2, 0, 1])
    env_b.update(1, [trajectories[i] for i in order], weights[order])
    assert np.allclose(env_a.theta, env_b.theta)
    print("grid batch-gradient order invariance OK")


def test_dead_group_without_relabel_is_skipped():
    class NoRelabelEnv(SkillChainSpace):
        def relabel(self, group):
            return None
    env = NoRelabelEnv(seed=0)
    trainer = FrontierTrainer(env, env, TrainerConfig(seed=0, hindsight=True))
    stats = trainer.train(steps=10)
    assert all(s.relabeled_groups == 0 for s in stats)
    print("no-relabel fallback OK")


def test_all_pass_group_is_not_relabelled():
    class AllPassEnv(SkillChainSpace):
        def __init__(self):
            super().__init__(n_chains=1, n_levels=1, n_actions=2, seed=0)
            self.relabel_calls = 0

        def rollout_group(self, task_id, n_rollouts):
            group = super().rollout_group(task_id, n_rollouts)
            group.rewards[:] = 1.0
            return group

        def relabel(self, group):
            self.relabel_calls += 1
            return super().relabel(group)

    env = AllPassEnv()
    trainer = FrontierTrainer(
        env, env, TrainerConfig(n_rollouts=4, tasks_per_step=1, hindsight=True)
    )
    stats = trainer.step()
    assert stats.all_pass_groups == 1
    assert stats.dead_groups == 0 and stats.relabeled_groups == 0
    assert env.relabel_calls == 0
    print("all-pass gate OK")


if __name__ == "__main__":
    test_estimators()
    test_positive_part_estimator()
    test_maxrl_population_orders()
    test_teacher_pseudocounts_and_utility()
    test_teacher_state_roundtrip()
    test_trainer_on_skill_chain()
    test_hindsight_contract_gridworld()
    test_grid_group_update_is_permutation_invariant()
    test_cosmos_libero_relabel_contracts()
    test_cosmos_poison_meter_gates_vocabulary()
    test_cosmos_mastery_split_and_shrinkage()
    test_cosmos_posterior_hygiene_end_to_end()
    test_dead_group_without_relabel_is_skipped()
    test_all_pass_group_is_not_relabelled()
    print("\nALL TESTS PASSED")
