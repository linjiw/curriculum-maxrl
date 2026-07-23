"""Simulated Phase-1 pilot for the Cosmos3/LIBERO integration.

Runs the four preregistered arms of COSMOS3_RESPONSE.md II.3 on a mock
predicate world (exact pass rates, CPU-only), so the full loop — teacher,
positive-part weights, template rewrites, poison gating — is exercised
end-to-end before any GPU hour is spent.  Swapping `make_world`'s rollout_fn
for the real policy-server wave is the whole Phase-1 port.

The mock: predicates are Bernoulli skills q_p (shared across tasks — the
transfer channel curricula require); a task's goal is a conjunction, its
exact pass rate Π q_p.  The pool is few-demo/frontier-heavy-shaped (V5's
regime, the few-demo LIBERO-Long analogue): 2 pair tasks near the frontier
(p ≈ q² ≈ 6·10⁻³) and 8 triple tasks that are effectively dead at N=8
(p ≈ q³ ≈ 5·10⁻⁴).  Registered sub-goal templates (singletons + pairs
beneath the pool) are the "curriculum below the pool" hindsight is allowed
to invent — no pool task is learnable without it at this budget.

Preregistered predictions (from the regime map / V5 / response 0.1):
  1. uniform ≈ teacher ≈ 0 (frontier-heavy: nothing to allocate — the
     teacher-alone arm CANNOT win here, response 0.1);
  2. relabel arms ignite the pool (the categorical V5 result);
  3. ignition: relabels burst early then decay as live groups take over;
  4. an ungated self-verifier with a poisoned predicate class does worse
     than oracle — and per-class gating (PoisonRateMeter) recovers most of
     the gap by pruning the vocabulary, not lowering the bar.  The poison
     mechanism is concrete here: a hallucinated "success" credits behavior
     that did not achieve the predicate, pushing its skill DOWN.

Run: python3 frontier_rl/examples/run_cosmos_pilot.py       (~1 min CPU)
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np

from frontier_rl import FrontierTrainer, TrainerConfig
from frontier_rl.adapters.cosmos_libero import (CosmosLiberoSpace,
                                                MasteryFrontierTeacher,
                                                PoisonRateMeter)

# ---------------------------------------------------------------------------
# mock world
# ---------------------------------------------------------------------------
PREDICATES = ([f"open(obj{i})" for i in range(4)]
              + [f"in(obj{i},obj{i+1})" for i in range(4)]
              + [f"on(obj{i},obj{i+1})" for i in range(4)])


def template_of(goal) -> str:
    """Canonical instruction per goal — the fixed template vocabulary (Q3.2)."""
    return " and ".join(sorted(goal))


def goal_of(template: str) -> frozenset:
    return frozenset(template.split(" and "))


def build_pool(rng):
    """Frontier-heavy: 10 triple-conjunction tasks, all dead at N=8
    (p = q³ ≈ 6·10⁻⁵ at init).  The learnable curriculum exists only BELOW
    the pool — singletons/pairs reachable through hindsight's sub-goal
    vocabulary (V5's frontier-heavy construction: nothing to sample toward)."""
    goals, seen = [], set()
    while len(goals) < 10:
        g = frozenset(rng.choice(PREDICATES, 3, replace=False))
        if g not in seen:
            goals.append(g); seen.add(g)
    return goals


def subgoal_vocabulary(pool_goals):
    """Singletons + pairs beneath the pool — admissible relabel targets."""
    from itertools import combinations
    vocab = {}
    for g in pool_goals:
        for size in (1, 2):
            for c in combinations(sorted(g), size):
                sub = frozenset(c)
                if sub not in pool_goals:
                    vocab[sub] = template_of(sub)
    return vocab


class PredicateSkillPolicy:
    """Bernoulli-skill policy with exact REINFORCE — the mock 'flow head'.

    theta[p] -> q_p = sigmoid(theta[p]); a rollout of goal G samples each
    predicate independently, succeeds iff all hold.  update() credits the
    skills named in the trajectory's (possibly rewritten) language goal —
    which is exactly what makes contract 2 matter here.
    """

    # init_q=0.015: pool pass rate q³ ≈ 3·10⁻⁶ makes the expected number of
    # live groups over the whole budget ≈ 0.2 (V5's construction — nothing
    # to sample toward), while singleton sub-goals (p=q=0.015, pass@8 ≈ 0.11)
    # sit exactly on the N=8 frontier band once hindsight invents them
    def __init__(self, init_q=0.015, lr=0.5, seed=0):
        self.idx = {p: i for i, p in enumerate(PREDICATES)}
        self.theta = np.full(len(PREDICATES), np.log(init_q / (1 - init_q)))
        self.lr = lr
        self.rng = np.random.default_rng(seed)

    def q(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.theta))

    def pass_rate(self, goal) -> float:
        q = self.q()
        return float(np.prod([q[self.idx[p]] for p in goal]))

    # Policy protocol
    def update(self, task_id, trajectories, weights) -> None:
        q = self.q()
        for traj, w in zip(trajectories, np.asarray(weights)):
            if w == 0.0:
                continue
            for p in goal_of(traj["language_goal"]):
                i = self.idx[p]
                a = traj["achieved"].get(p)
                if a is None:        # predicate outside the sampled goal
                    continue
                self.theta[i] += self.lr * w * (a - q[i])


def make_world(policy, seed, fp_rate_by_class=None):
    """rollout_fn + oracle/self verifier_fns over the shared policy."""
    rng = np.random.default_rng(seed)
    fp = fp_rate_by_class or {}

    def rollout_fn(template, init_bin, n):
        goal = sorted(goal_of(template))
        q = np.array([policy.q()[policy.idx[p]] for p in goal])
        a = rng.random((n, len(goal))) < q
        rewards = a.all(axis=1).astype(float)
        trajs = [{"language_goal": template,
                  "achieved": {p: float(a[j, k]) for k, p in enumerate(goal)}}
                 for j in range(n)]
        infos = [{"achieved_true": {p for k, p in enumerate(goal) if a[j, k]},
                  "goal": goal} for j in range(n)]
        return rewards, trajs, infos

    def oracle_verifier(info):
        return set(info["achieved_true"])

    def self_verifier(info):
        """Oracle + class-conditional false positives (the poison channel)."""
        preds = set(info["achieved_true"])
        for p in info["goal"]:
            if p not in preds:
                cls = PoisonRateMeter.predicate_class(p)
                if rng.random() < fp.get(cls, 0.0):
                    preds.add(p)     # hallucinated achievement
        return preds

    return rollout_fn, oracle_verifier, self_verifier


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
def run_arm(arm, seed, steps=100, n_rollouts=8, tasks_per_step=8):
    rng = np.random.default_rng(7)          # fixed pool across arms/seeds
    pool_goals = build_pool(rng)
    tasks = [(g, template_of(g)) for g in pool_goals]
    vocab = subgoal_vocabulary(set(pool_goals))

    policy = PredicateSkillPolicy(seed=seed)
    # "on" is the poisoned class.  NOTE a base-rate effect the real Pilot 0
    # must budget for: precision is measured on the verifier's POSITIVE
    # calls, and with rare true achievements (q≈0.015) false-positive
    # opportunities outnumber true ones ~65:1 — so even a 0.3% per-call
    # hallucination rate caps a class's precision at ~0.84, below any 90%
    # gate.  A gate on few-demo rollouts therefore needs either near-zero
    # hallucination classes (as mocked here) or a probe set enriched with
    # successes (mixed success/failure rollouts, as the proposal specifies).
    fp = {"on": 0.5, "in": 0.0, "open": 0.0}
    rollout_fn, oracle_v, self_v = make_world(policy, seed + 100,
                                              fp_rate_by_class=fp)

    allowed = None
    verifier = oracle_v
    if arm in ("self", "self+gate"):
        verifier = self_v
        if arm == "self+gate":
            # Pilot-0b in miniature: measure the verifier against oracle on
            # probe rollouts, then gate the relabel vocabulary per class
            # probe budget matters: at q≈0.04 true achievements are rare, so
            # a starved probe (32/task) mis-prunes CLEAN classes on sampling
            # noise — the F3 "starved regime" lesson applied to the meter
            meter = PoisonRateMeter(precision_gate=0.9)
            for g in pool_goals:
                _, _, infos = rollout_fn(template_of(g), None, 128)
                for info in infos:
                    meter.observe(self_v(info), oracle_v(info))
            allowed = meter.allowed_vocabulary()

    env = CosmosLiberoSpace(tasks, rollout_fn, verifier,
                            subgoal_templates=vocab, allowed_classes=allowed)
    # sub-goal arms are relabel-only (no BDDL task exists for them on the
    # real stack) — the teacher must not roll them out directly
    teacher = MasteryFrontierTeacher(env.n_tasks, n_rollouts,
                                     samplable=env.samplable_mask(),
                                     seed=seed + 1000)
    if arm == "uniform":
        mask = env.samplable_mask()
        teacher.distribution = lambda: mask / mask.sum()

    cfg = TrainerConfig(n_rollouts=n_rollouts, tasks_per_step=tasks_per_step,
                        hindsight=(arm not in ("uniform", "teacher")),
                        positive_weights=True, seed=seed)
    trainer = FrontierTrainer(env, policy, cfg, teacher=teacher)

    curve = []
    def on_eval(i):
        curve.append(float(np.mean([policy.pass_rate(g)
                                    for g in pool_goals])))
    stats = trainer.train(steps, on_eval=on_eval, eval_every=5)
    relabels = [s.relabeled_groups for s in stats]
    third = max(len(relabels) // 3, 1)
    return {
        "auc": float(np.mean(curve)),
        "final": curve[-1],
        "dead_rate": float(np.mean([s.dead_groups / tasks_per_step
                                    for s in stats])),
        "relabels_early": int(np.sum(relabels[:third])),
        "relabels_late": int(np.sum(relabels[-third:])),
        "allowed": allowed,
    }


def main():
    seeds = range(3)
    arms = ["uniform", "teacher", "oracle", "self", "self+gate"]
    print(f"{'arm':12s} {'AUC':>14s} {'final':>7s} "
          f"{'dead/8':>7s} {'relabel e->l':>13s}")
    results = {}
    for arm in arms:
        rs = [run_arm(arm, s) for s in seeds]
        results[arm] = rs
        auc = [r["auc"] for r in rs]
        print(f"{arm:12s} {np.mean(auc):7.3f}±{np.std(auc):.3f} "
              f"{np.mean([r['final'] for r in rs]):7.3f} "
              f"{np.mean([r['dead_rate'] for r in rs]):7.2f} "
              f"{np.mean([r['relabels_early'] for r in rs]):6.0f} -> "
              f"{np.mean([r['relabels_late'] for r in rs]):.0f}")
    gate = results["self+gate"][0]["allowed"]
    print(f"\ngated vocabulary (poisoned 'on' class pruned): {sorted(gate)}")

    # preregistered checks (II.3 / response 0.1)
    a = {k: np.mean([r["auc"] for r in v]) for k, v in results.items()}
    f = {k: np.mean([r["final"] for r in v]) for k, v in results.items()}
    checks = [
        ("frontier-heavy: pool ~dead without relabeling (rare lucky "
         "ignitions only)",
         f["uniform"] < 0.1 and f["teacher"] < 0.1),
        ("relabeling ignites the pool (oracle final >> non-relabel arms)",
         f["oracle"] > 5 * max(f["uniform"], f["teacher"], 1e-9)),
        ("ignition: relabels decay (early > late)",
         np.mean([r["relabels_early"] for r in results["oracle"]])
         > np.mean([r["relabels_late"] for r in results["oracle"]])),
        ("poison hurts; per-class gate recovers",
         a["self"] < a["oracle"] and a["self+gate"] > a["self"]),
    ]
    print()
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


if __name__ == "__main__":
    main()
