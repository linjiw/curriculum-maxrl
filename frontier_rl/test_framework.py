"""Unit tests for frontier_rl. Run: python3 frontier_rl/test_framework.py"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from frontier_rl import (FrontierTeacher, FrontierTrainer, TrainerConfig,
                         maxrl_weights, grpo_weights, rloo_weights)
from frontier_rl.adapters.skill_chain import SkillChainSpace
from frontier_rl.adapters.grid_reach import GridReachSpace
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


def test_teacher_posterior_and_utility():
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
        for r, info, nt in zip(new_r, g.infos, new_trajs):
            end_ring = max(abs(info["final_pos"][0]), abs(info["final_pos"][1]))
            if r == 1.0:
                # exactness (P6 contract 1): success truly ended on that ring
                assert end_ring == ring
                # conditioning (contract 2): goal rewritten to achieved cell
                assert np.array_equal(nt["goal"], info["final_pos"])
    print("hindsight contract OK")


def test_evaluation_harness():
    from frontier_rl.evaluation import (pass_at_k, TaskEval, summarize,
                                        RunLedger, teacher_calibration)
    # unbiased pass@k against exhaustive enumeration: n=6, c=2, k=3
    # P(no success in 3 of 6 w/o replacement) = C(4,3)/C(6,3) = 4/20
    assert abs(pass_at_k(6, 2, 3) - (1 - 4 / 20)) < 1e-12
    assert pass_at_k(8, 0, 4) == 0.0 and pass_at_k(8, 8, 1) == 1.0
    assert pass_at_k(8, 6, 4) == 1.0          # n-c < k => certain
    # MC agreement: estimator == empirical mean over resamples
    rng = np.random.default_rng(0)
    n, c, k = 16, 5, 8
    draws = [rng.choice([1] * c + [0] * (n - c), k, replace=False).max()
             for _ in range(20000)]
    assert abs(pass_at_k(n, c, k) - np.mean(draws)) < 0.01

    evals = [TaskEval(0, 16, 16), TaskEval(1, 16, 4), TaskEval(2, 16, 0)]
    s = summarize(evals, ks=(1, 8), baseline_rates={0: 0.9, 1: 0.3, 2: 0.0})
    assert abs(s["mean_success"] - (1.0 + 0.25 + 0.0) / 3) < 1e-12
    assert s["easy_decile_retention"] == 1.0       # probe = easiest task (0)
    assert s["success@8"] > s["success@1"]         # coverage sees the tail

    # calibration flags an inflated posterior
    t = FrontierTeacher(3, 8, seed=0)
    for _ in range(4):
        t.observe(1, np.ones(8))                   # posterior thinks p~1
    cal = teacher_calibration(t, evals)            # eval says task1 = 0.25
    assert cal["teacher/calibration_bias"] > 0.3

    # ledger accounting
    led = RunLedger()
    led.observe_group(8, 800, live=True, server_batches=10)
    led.observe_group(8, 1600, live=False, relabeled=True, server_batches=20)
    row = led.snapshot(step=1)
    assert row["episodes"] == 16 and row["updates_relabel"] == 1
    assert row["dead_group_rate"] == 0.5 and row["relabel_yield"] == 1.0
    print("evaluation harness OK")


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


def test_pilot0_instruments():
    from frontier_rl.pilot0 import (GroupVarianceProbe, run_poison_probe,
                                    surrogate_fidelity, pilot0_verdict)
    rng = np.random.default_rng(0)
    # 0a: healthy sampler (contrasted groups, real action spread)
    probe = GroupVarianceProbe()
    for _ in range(20):
        probe.observe(rng.binomial(1, 0.3, 8),
                      [rng.normal(0, 0.1, (4, 7)) for _ in range(8)])
    rep = probe.report()
    assert rep["gate_contrast"] and rep["gate_action_variance"]
    # 0a: deterministic sampler fails both gates
    dead = GroupVarianceProbe()
    for _ in range(20):
        dead.observe(np.zeros(8), [np.zeros((4, 7))] * 8)
    repd = dead.report()
    assert not repd["gate_contrast"] and not repd["gate_action_variance"]

    # 0b: clean class allowed; poisoned pruned; low-support class UNMEASURED
    episodes = []
    for i in range(120):
        oracle = {"open(m)"} if i % 2 else set()
        episodes.append({"info": {"oracle": oracle},
                         "oracle_predicates": oracle})
    def self_v(info):
        preds = set(info["oracle"])
        preds.add("on(a,b)")           # hallucinated every time
        return preds
    rep_b = run_poison_probe(episodes, self_v, min_positive_support=20)
    assert "open" in rep_b["allowed_vocabulary"]
    assert "on" not in rep_b["allowed_vocabulary"]
    assert "on" in rep_b["unmeasured_classes"] or rep_b["precision"]["on"] < 0.9

    # 0c: aligned directions pass; orthogonal fail
    g = [rng.normal(0, 1, 32) for _ in range(30)]
    aligned = [(v + rng.normal(0, 0.2, 32), v) for v in g]
    rep_c = surrogate_fidelity(aligned)
    assert rep_c["gate_fidelity"] and rep_c["mean_direction_cosine"] > 0.9
    orth = [(rng.normal(0, 1, 32), rng.normal(0, 1, 32)) for _ in range(30)]
    assert not surrogate_fidelity(orth)["gate_fidelity"]

    verdict = pilot0_verdict(rep, rep_b, rep_c)
    assert "0a PASS" in verdict and "0c PASS" in verdict
    print("pilot0 instruments OK")


def test_cosmos_live_glue():
    """LiveRolloutBackend wave loop + WeightedCFMBuffer with fake client/venv."""
    import tempfile, json as _json
    from frontier_rl.adapters.cosmos_live import (LiveRolloutBackend,
                                                  WeightedCFMBuffer,
                                                  goal_predicates_of)
    from frontier_rl.evaluation import RunLedger

    # goal_state parsing -> canonical predicate strings
    parsed = {"goal_state": [["Open", "microwave_1"],
                             ["In", "bowl_1", "microwave_1"]]}
    assert goal_predicates_of(parsed) == ["open(microwave_1)",
                                          "in(bowl_1,microwave_1)"]

    class FakeClient:
        prompt = ""
        def __init__(self):
            self.batch_calls = 0
        def predict_batch(self, imgs):
            self.batch_calls += 1
            # constant 4-step chunk per env
            return [[[0.1] * 7 for _ in range(4)] for _ in imgs]

    class FakeVenv:
        """Env 0 succeeds at sim step 6; env 1 never does."""
        def reset(self, id=None): pass
        def set_init_state(self, states, id=None):
            self.step_count = 0
            return [{"slot": s} for s in id]
        def step(self, actions, id=None):
            self.step_count += 1
            info = [{"success": (s == 0 and self.step_count >= 6)}
                    for s in id]
            d = np.array([i["success"] for i in info])
            return [{"slot": s} for s in id], None, d, info

    venv = FakeVenv()
    client = FakeClient()
    ledger = RunLedger()
    backend = LiveRolloutBackend(
        client=client,
        venv_of_task=lambda tid: (venv, parsed, 12),
        init_states_of=lambda tid, b, n: np.zeros((n, 3)),
        get_images=lambda obs: [np.zeros((8, 8, 3), np.uint8)],
        action_to_env=lambda row: list(row),
        predicate_snapshot=lambda env, pp: {"open(microwave_1)"},
        action_horizon=4, warmup_steps=2, record_frames=False,
        ledger=ledger)

    rewards, trajs, infos = backend("put the bowl in the microwave", None, 2,
                                    task_id=5)
    assert client.prompt == "put the bowl in the microwave"
    assert rewards.tolist() == [1.0, 0.0]
    # success env carries no predicate snapshot; failed env does
    assert infos[0]["achieved_predicates"] == set()
    assert infos[1]["achieved_predicates"] == {"open(microwave_1)"}
    assert trajs[0]["task_id"] == 5
    assert ledger.episodes == 2 and ledger.server_batches >= 1

    # manifest: zero-weight rows dropped; relabel flag from task_id mismatch
    with tempfile.TemporaryDirectory() as td:
        buf = WeightedCFMBuffer(f"{td}/round0.jsonl")
        buf.update(3, trajs, np.array([0.875, 0.0]))     # traj task_id = 5
        rows = [_json.loads(l) for l in open(f"{td}/round0.jsonl")]
        assert len(rows) == 1 and rows[0]["weight"] == 0.875
        assert rows[0]["relabeled"] == 1
        assert rows[0]["language_goal"] == "put the bowl in the microwave"
    print("cosmos live glue OK")


def test_baseline_estimator_arms():
    # estimator swap reaches the update path
    seen_w = []
    class SpyPolicy:
        def update(self, task_id, trajectories, weights):
            seen_w.append(np.asarray(weights))
    env = SkillChainSpace(seed=0)
    tr = FrontierTrainer(env, SpyPolicy(),
                         TrainerConfig(seed=0, estimator="grpo",
                                       hindsight=False))
    tr.train(steps=5)
    assert seen_w and all(abs(w.sum()) < 1e-9 for w in seen_w), \
        "grpo weights must be mean-zero"
    # DAPO redraws convert dead draws into (counted) extra generation cost
    env2 = SkillChainSpace(seed=0)
    tr2 = FrontierTrainer(env2, env2,
                          TrainerConfig(seed=0, hindsight=False,
                                        dapo_max_redraws=4))
    stats = tr2.train(steps=10)
    total_groups = sum(s.live_groups + s.dead_groups for s in stats)
    assert total_groups > 10 * tr2.cfg.tasks_per_step * 0.9
    # unknown estimator fails loudly
    try:
        FrontierTrainer(env, env, TrainerConfig(estimator="ppo")).step()
        raise AssertionError("should have raised")
    except ValueError:
        pass
    print("baseline estimator arms OK")


def test_dead_group_without_relabel_is_skipped():
    class NoRelabelEnv(SkillChainSpace):
        def relabel(self, group):
            return None
    env = NoRelabelEnv(seed=0)
    trainer = FrontierTrainer(env, env, TrainerConfig(seed=0, hindsight=True))
    stats = trainer.train(steps=10)
    assert all(s.relabeled_groups == 0 for s in stats)
    print("no-relabel fallback OK")


def test_isaaclab_adapter():
    """Reset-stream teacher: the behaviors ISAACLAB_DESIGN.md claims."""
    from frontier_rl.adapters.isaaclab_curriculum import FrontierBinTeacher

    t = FrontierBinTeacher(n_bins=6, decay_half_life=64.0, seed=0)
    # visit every bin (unvisited bins sit at the optimistic prior p=0.5 and
    # tie for max utility by design); pass rates 0 / .5 / .1 / 0 / .9 / 1
    fails = {0: [True] * 8, 1: [False, True] * 4, 2: [True] * 7 + [False],
             3: [True] * 8, 4: [False] * 7 + [True], 5: [False] * 8}
    for _ in range(20):
        for b, f in fails.items():
            t.observe_resets(np.full(8, b), np.array(f))
    probs = t.sampling_probs()
    # frontier bin (p~0.5 under learnability) out-samples dead + mastered
    assert probs[1] > probs[3] and probs[1] > probs[5], probs
    assert abs(probs.sum() - 1) < 1e-9
    assert t.argmax_utility() == 1
    # decayed evidence (~15 episode-equivalents) leaves all-fail bins at
    # p̂≈0.06 under the Beta prior — probe with a matching threshold
    assert t.dead_fraction(threshold=0.1) > 0 and t.mastered_fraction() > 0
    # evidence-scaled decay is exact: one half-life of events halves counts
    t2 = FrontierBinTeacher(n_bins=2, decay_half_life=10.0, seed=0)
    t2.observe_resets(np.zeros(10, dtype=int), np.zeros(10, dtype=bool))
    s_before = t2.succ[0]
    t2.observe_resets(np.ones(10, dtype=int), np.ones(10, dtype=bool))
    assert abs(t2.succ[0] - s_before / 2) < 1e-12
    # sample_bins respects the floor: every bin reachable
    seen = set(t.sample_bins(2000).tolist())
    assert seen == set(range(6)), seen
    # state roundtrip
    t3 = FrontierBinTeacher(n_bins=6, decay_half_life=64.0, seed=0)
    t3.load_state_dict(t.state_dict())
    assert np.allclose(t3.succ, t.succ) and np.allclose(t3.fail, t.fail)
    print("isaaclab reset-stream teacher OK")



def test_countdown_llm_adapter():
    """E-LLM-2 contracts: exact verify, exact relabel, template rewrite."""
    from frontier_rl.adapters.countdown_llm import (CountdownInstance,
                                                    CountdownSpace,
                                                    achieved_value, verify,
                                                    rewrite_prompt)
    # verifier: strict TinyZero semantics
    ok = "<think>x</think><answer>(3 + 7) * 2</answer>"
    assert verify(ok, [3, 7, 2], 20) and not verify(ok, [3, 7, 2], 21)
    assert not verify(ok, [3, 7, 5], 20), "numbers-once check failed"
    assert not verify("<answer>__import__('os')</answer>", [1], 1)
    # relabel map: failed trace -> the integer it actually reached
    assert achieved_value(ok, [3, 7, 2]) == 20
    assert achieved_value(ok, [3, 7, 5]) is None      # contract 1: true task only
    assert achieved_value("<answer>7 / 2</answer>", [7, 2]) is None  # non-integer
    # contract 2: rewrite hits the template slot only
    p = "Using the numbers [3, 7, 2], create an equation that equals 21. 21 rules."
    r = rewrite_prompt(p, 21, 20)
    assert "equals 20" in r and r.endswith("21 rules.")
    assert rewrite_prompt("no slot here 21", 21, 20) is None

    # end-to-end: mock LLM that always produces (a+b) regardless of target
    pool = [CountdownInstance([3, 7], 10, 0), CountdownInstance([3, 7], 99, 0),
            CountdownInstance([2, 5, 8], 80, 1)]
    def mock_llm(prompt, n):
        nums = re.findall(r"\[([\d, ]+)\]", prompt)[0].split(",")
        a, b = int(nums[0]), int(nums[1])
        return [f"<think>.</think><answer>{a} + {b}</answer>"] * n
    import re
    env = CountdownSpace(pool, tiers=[2, 3], rollout_fn=mock_llm, seed=0)
    # target-99 group: all fail, but every trace reached 10 -> dense relabel
    g = env.rollout_group(0, 4)
    if g.rewards.sum() == 0:                       # instance was target=99
        rel = env.relabel(g)
        assert rel is not None
        new_task, new_r, new_trajs = rel
        assert new_task == 0 and new_r.sum() == 4
        assert all("equals 10" in t["prompt"] for t in new_trajs)
        assert all(t["relabeled_target"] == 10 for t in new_trajs)
    print("countdown LLM adapter OK")


if __name__ == "__main__":
    test_estimators()
    test_positive_part_estimator()
    test_teacher_posterior_and_utility()
    test_teacher_state_roundtrip()
    test_trainer_on_skill_chain()
    test_hindsight_contract_gridworld()
    test_evaluation_harness()
    test_cosmos_libero_relabel_contracts()
    test_cosmos_poison_meter_gates_vocabulary()
    test_cosmos_mastery_split_and_shrinkage()
    test_cosmos_posterior_hygiene_end_to_end()
    test_pilot0_instruments()
    test_cosmos_live_glue()
    test_baseline_estimator_arms()
    test_dead_group_without_relabel_is_skipped()
    test_isaaclab_adapter()
    test_countdown_llm_adapter()
    print("\nALL TESTS PASSED")
