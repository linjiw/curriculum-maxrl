"""Cosmos3/LIBERO adapter: the frontier schedule on a flow-policy VLA stack.

Implements the design agreed in COSMOS3_RESPONSE.md (Part II) for the Cosmos3
team's "Self-Verified Frontier RL" proposal:

  arm       = (task, init-state-bin) — Phase 1 starts with 90 task-level arms;
              init-state bins are created by MASTERY SPLITS (Q2.2), not day 0
  rollout   = one wave of N parallel episodes against the policy server +
              SubprocVectorEnv (`rollout_fn` hook — this module never imports
              cosmos/LIBERO, mirroring the isaaclab adapter's lazy pattern)
  verifier  = binary sim `info["success"]` for LIVE groups (never the model);
              a predicate verifier (oracle sim-BDDL, or the model's own
              ID-prefilter + reasoner query) only ever scores DEAD groups
  relabel   = deepest achieved sub-conjunction of the failed task's own goal,
              falling back to pool/singleton predicate tasks (Q3.3), with the
              language conditioning rewritten from a FIXED TEMPLATE per
              predicate goal (Q3.2 — never free-generated), and the hard rule
              that a failed rollout is NEVER upgraded to a success of the
              original task (Q3.4)
  estimator = positive-part MaxRL weights (TrainerConfig.positive_weights=True
              — weighted RFT; E[Σw⁺] = u(p) exactly, so P1 governs sampling
              unchanged; see estimators.maxrl_weights)
  teacher   = MasteryFrontierTeacher — FrontierTeacher plus dynamic arm growth
              with hierarchical pseudo-count shrinkage toward the parent task
              (Q2.3: alpha_child = 1 + lam*(alpha_parent-1) + own evidence)

Poison-rate measurement (Pilot 0b) ships here too: `PoisonRateMeter` compares
a self-verifier's achieved-predicate sets against oracle truth per predicate
class and prunes the relabel vocabulary at a precision gate — the action on a
failing class is removal from the vocabulary, not lowering the gate.

Everything is unit-tested on CPU with a mock predicate world
(test_framework.py::test_cosmos_libero_*).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, FrozenSet, Optional, Sequence

import numpy as np

from frontier_rl.interfaces import GroupResult
from frontier_rl.teacher import FrontierTeacher

Predicate = str
Goal = FrozenSet[Predicate]


# ---------------------------------------------------------------------------
# task table
# ---------------------------------------------------------------------------
@dataclass
class PredicateTask:
    """One arm: a predicate-conjunction goal + its canonical instruction.

    `template` is the fixed language conditioning for this goal (contract 2's
    closed-vocabulary rewrite target — one human-reviewed string per goal).
    `init_bin` is None for task-level arms; mastery splits create children
    with concrete bins.  `source` records provenance for telemetry.
    """
    goal: Goal
    template: str
    init_bin: Optional[int] = None
    parent: Optional[int] = None          # arm id this was split from
    source: str = "pool"                  # "pool" | "subgoal" | "split"


class PoisonRateMeter:
    """Per-predicate-class precision/recall of a self-verifier vs oracle.

    Pilot 0b's instrument: feed (self_predicates, oracle_predicates) pairs per
    rollout; `precision()` returns per-class precision, `allowed_vocabulary`
    the classes above the gate.  Class of a predicate = its head symbol
    ("open(microwave)" -> "open"), because verifier competence is systematic
    by relation type, not by instance.
    """

    def __init__(self, precision_gate: float = 0.9):
        self.gate = precision_gate
        self.true_pos = defaultdict(int)   # self said yes, oracle agrees
        self.false_pos = defaultdict(int)  # self said yes, oracle disagrees
        self.false_neg = defaultdict(int)  # oracle yes, self missed

    @staticmethod
    def predicate_class(pred: Predicate) -> str:
        return pred.split("(", 1)[0]

    def observe(self, self_preds: set, oracle_preds: set) -> None:
        for p in self_preds:
            c = self.predicate_class(p)
            if p in oracle_preds:
                self.true_pos[c] += 1
            else:
                self.false_pos[c] += 1
        for p in oracle_preds - self_preds:
            self.false_neg[self.predicate_class(p)] += 1

    def precision(self) -> dict:
        out = {}
        for c in set(self.true_pos) | set(self.false_pos):
            tp, fp = self.true_pos[c], self.false_pos[c]
            out[c] = tp / (tp + fp) if tp + fp else 1.0
        return out

    def recall(self) -> dict:
        out = {}
        for c in set(self.true_pos) | set(self.false_neg):
            tp, fn = self.true_pos[c], self.false_neg[c]
            out[c] = tp / (tp + fn) if tp + fn else 1.0
        return out

    def allowed_vocabulary(self) -> set:
        """Predicate classes clean enough to relabel with (precision >= gate).

        Recall costs only recyclable signal; precision costs contract 1 —
        so only precision gates the vocabulary."""
        return {c for c, prec in self.precision().items() if prec >= self.gate}


# ---------------------------------------------------------------------------
# teacher with mastery splits + hierarchical shrinkage
# ---------------------------------------------------------------------------
class MasteryFrontierTeacher(FrontierTeacher):
    """FrontierTeacher with relabel-only arms and mastery splits (Q2.2/Q2.3).

    `samplable`: bool mask over arms.  Non-samplable arms are RELABEL-ONLY —
    auxiliary sub-goal tasks that exist so relabeled gradient has a stable
    (task_id, template) to be credited to, but that cannot be rolled out
    directly (no BDDL task file exists for an arbitrary sub-conjunction).
    They get zero sampling probability, zero floor share, and — since they
    are never rolled out — an empty posterior, which is posterior hygiene by
    construction.  Without this mask the teacher quietly trains the invented
    curriculum directly, which is not available on the real robot stack (and
    turns the frontier-heavy regime into a balanced one — measured in
    run_cosmos_pilot.py during development).

    `split(parent, k, lam)` appends k samplable child arms whose posteriors
    are shrunk toward the parent's by pseudo-count fraction lam (default
    0.3): the cheap hierarchical-Beta — children start informed, own
    evidence dominates after a few groups.  The parent arm stays; its
    saturated posterior retires it through u(p)->0 while the floor keeps
    retention probes on it.
    """

    def __init__(self, n_tasks: int, n_rollouts: int = 16, *,
                 samplable: Optional[np.ndarray] = None, **kwargs):
        super().__init__(n_tasks, n_rollouts, **kwargs)
        self.samplable = (np.ones(n_tasks, dtype=bool) if samplable is None
                          else np.asarray(samplable, dtype=bool).copy())

    def distribution(self) -> np.ndarray:
        p = self.rng.beta(self.alpha, self.beta)
        u = self.utility(p) ** self.gamma
        u[~self.samplable] = 0.0
        if u.sum() <= 1e-12:
            u = self.samplable.astype(float)
        probs = u / u.sum()
        uniform = self.samplable / self.samplable.sum()
        return (1.0 - self.floor) * probs + self.floor * uniform

    def split(self, parent: int, k: int, lam: float = 0.3) -> list[int]:
        a0 = 1.0 + lam * (self.alpha[parent] - 1.0)
        b0 = 1.0 + lam * (self.beta[parent] - 1.0)
        first = self.n_tasks
        self.alpha = np.concatenate([self.alpha, np.full(k, a0)])
        self.beta = np.concatenate([self.beta, np.full(k, b0)])
        self.visits = np.concatenate([self.visits, np.zeros(k, dtype=np.int64)])
        self.samplable = np.concatenate([self.samplable, np.ones(k, dtype=bool)])
        self.n_tasks += k
        return list(range(first, first + k))

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["samplable"] = self.samplable.copy()
        return d

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        if "samplable" in state:
            self.samplable = np.asarray(state["samplable"], dtype=bool)


# ---------------------------------------------------------------------------
# the TaskSpace
# ---------------------------------------------------------------------------
class CosmosLiberoSpace:
    """TaskSpace over predicate-goal manipulation tasks (libero_90 pattern).

    Args:
      tasks: the pool — list of (goal predicates, language template). Order
        defines arm ids 0..len-1.
      rollout_fn: (template, init_bin, n) -> (rewards, trajectories, infos).
        In production this posts one /predict_batch wave through the policy
        server against SubprocVectorEnv; rewards are the sim's binary
        info["success"] (the ONLY verifier live groups ever see); each info
        must carry whatever the predicate verifier needs (final frames /
        sim state handle). Trajectories must be dicts with a "language_goal"
        key (the conditioning the relabel rewrites).
      verifier_fn: (info) -> set of achieved predicates, called on dead
        groups only. Oracle arm: sim BDDL evaluation. Self-verified arm:
        ID-consistency prefilter + reasoner predicate query.
      subgoal_templates: template per admissible relabel goal beyond the pool
        (typically singleton predicates). Goals absent from pool+subgoals are
        never relabel targets. This is the closed template vocabulary of Q3.2.
      allowed_classes: predicate classes the verifier is trusted on (from
        PoisonRateMeter.allowed_vocabulary(); None = trust all — oracle arm).
      max_subconj_size: cap on sub-conjunction enumeration (goals are 1-4
        predicates in LIBERO; the cap only guards degenerate inputs).
    """

    def __init__(self, tasks: Sequence[tuple], rollout_fn: Callable,
                 verifier_fn: Callable, *,
                 subgoal_templates: Optional[dict] = None,
                 allowed_classes: Optional[set] = None,
                 max_subconj_size: int = 4):
        self.tasks: list[PredicateTask] = [
            PredicateTask(goal=frozenset(g), template=t) for g, t in tasks]
        self.rollout_fn = rollout_fn
        self.verifier_fn = verifier_fn
        self.allowed_classes = allowed_classes
        self.max_subconj_size = max_subconj_size
        # relabel-target registry: goal -> arm id (pool tasks first, then
        # auxiliary sub-goal tasks — the "curriculum below the pool" arms)
        self._task_of_goal: dict[Goal, int] = {
            t.goal: i for i, t in enumerate(self.tasks)}
        for goal, template in (subgoal_templates or {}).items():
            goal = frozenset(goal) if not isinstance(goal, frozenset) else goal
            if goal not in self._task_of_goal:
                self._task_of_goal[goal] = len(self.tasks)
                self.tasks.append(PredicateTask(goal=goal, template=template,
                                                source="subgoal"))
        # telemetry (II.4)
        self.relabel_attempts = 0
        self.relabel_successes = 0
        self.relabels_by_target = defaultdict(int)

    # ---- TaskSpace -------------------------------------------------------
    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    def samplable_mask(self) -> np.ndarray:
        """True for arms the teacher may roll out (pool + mastery splits);
        False for relabel-only sub-goal arms. Pass to MasteryFrontierTeacher."""
        return np.array([t.source != "subgoal" for t in self.tasks])

    def rollout_group(self, task_id: int, n_rollouts: int) -> GroupResult:
        t = self.tasks[task_id]
        rewards, trajs, infos = self.rollout_fn(t.template, t.init_bin,
                                                n_rollouts)
        return GroupResult(task_id, np.asarray(rewards, dtype=float),
                           list(trajs), list(infos))

    def relabel(self, group: GroupResult) -> Optional[tuple]:
        """Deepest achieved sub-conjunction, both P6 contracts enforced.

        Search order (Q3.3): (a) largest strict sub-conjunction of the failed
        task's own goal achieved by >=1 rollout — the on-the-path target that
        compounds toward the sampled task; (b) largest registered cross-task /
        singleton goal achieved.  The original task's full goal is EXCLUDED by
        construction (strict subsets only + the group was all-fail): the
        verifier can never upgrade a failure into a success of the task the
        sim already scored (Q3.4).
        """
        # NB: if every rollout achieves the target sub-goal the relabeled
        # group comes back all-success and maxrl_weights are zero (K=N
        # self-retirement) — an uncontrasted group carries no likelihood
        # signal, and a sub-goal every failure reaches is effectively
        # mastered.  This is the framework-wide semantics (skill_chain and
        # grid_reach behave identically), not an adapter quirk.
        self.relabel_attempts += 1
        own_goal = self.tasks[group.task_id].goal
        achieved = []
        for info in group.infos:
            preds = set(self.verifier_fn(info))
            if self.allowed_classes is not None:
                preds = {p for p in preds
                         if PoisonRateMeter.predicate_class(p)
                         in self.allowed_classes}
            achieved.append(preds)

        target_goal = self._deepest_target(own_goal, achieved)
        if target_goal is None:
            return None
        new_task = self._task_of_goal[target_goal]
        new_rewards = np.array([1.0 if target_goal <= a else 0.0
                                for a in achieved])
        # contract 2: rebuild the conditioning from the target's canonical
        # template — for every rollout in the group (they all train as
        # attempts at the relabeled task)
        template = self.tasks[new_task].template
        new_trajs = []
        for traj in group.trajectories:
            nt = dict(traj)
            nt["language_goal"] = template
            new_trajs.append(nt)
        self.relabel_successes += 1
        self.relabels_by_target[new_task] += 1
        return new_task, new_rewards, new_trajs

    def _deepest_target(self, own_goal: Goal,
                        achieved: list[set]) -> Optional[Goal]:
        any_achieved = set().union(*achieved) if achieved else set()
        # (a) strict sub-conjunctions of the failed task's goal, deepest first
        own = sorted(own_goal)
        for size in range(min(len(own) - 1, self.max_subconj_size), 0, -1):
            best = None
            for combo in combinations(own, size):
                g = frozenset(combo)
                if g in self._task_of_goal and any(g <= a for a in achieved):
                    best = g if best is None else best
            if best is not None:
                return best
        # (b) registered cross-task / singleton goals, deepest first
        candidates = [g for g in self._task_of_goal
                      if g != own_goal and g <= any_achieved
                      and any(g <= a for a in achieved)]
        if not candidates:
            return None
        return max(candidates, key=len)

    # ---- mastery splits (Q2.2) --------------------------------------------
    def split_mastered(self, teacher: MasteryFrontierTeacher, *,
                       n_bins: int = 4, p_threshold: float = 0.9,
                       min_visits: int = 5, lam: float = 0.3) -> list[int]:
        """Create init-state-bin child arms for saturated task-level arms.

        Call periodically (e.g. from the trainer's on_eval hook). A task
        splits once, when its posterior mean exceeds p_threshold with enough
        visits — manufacturing new frontier from mastered material instead of
        starting with a starved 450-arm pool (COSMOS3 response 0.2).
        """
        p = teacher.pass_rate_estimates()
        new_ids = []
        for tid, task in enumerate(list(self.tasks)):
            if (task.source == "split" or task.init_bin is not None
                    or tid >= len(p)):
                continue
            already = any(c.parent == tid for c in self.tasks)
            if already or p[tid] < p_threshold or teacher.visits[tid] < min_visits:
                continue
            child_ids = teacher.split(tid, n_bins, lam=lam)
            for b, cid in enumerate(child_ids):
                self.tasks.append(PredicateTask(
                    goal=task.goal, template=task.template, init_bin=b,
                    parent=tid, source="split"))
                assert len(self.tasks) - 1 == cid, "env/teacher arm ids diverged"
            new_ids += child_ids
        return new_ids

    # ---- telemetry (II.4) --------------------------------------------------
    def metrics(self) -> dict:
        return {
            "env/n_arms": self.n_tasks,
            "env/relabel_attempts": self.relabel_attempts,
            "env/relabel_successes": self.relabel_successes,
            "env/n_split_arms": sum(t.source == "split" for t in self.tasks),
        }
