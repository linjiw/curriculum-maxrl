"""Evaluation harness for frontier-RL runs: the metrics a robotics claim
needs to be clean (COSMOS3_RESPONSE.md Q4), environment-agnostic.

The meter lesson (EVIDENCE.md §3) drove this module's design: three separate
times a metric hid what the method was doing — fixed-step comparisons hid the
teacher's speed, peakedness hid its targeting, pass@1 hid the deep-frontier
march.  So the harness reports, for every arm:

  1. success@k per task (unbiased Chen et al. 2021 estimator) — coverage
     currency, where likelihood-style training moves first;
  2. easy-decile retention — the collapse tripwire (H6's signature is easy
     tasks decaying while the frontier improves);
  3. teacher calibration — posterior p̂ vs held-out eval pass rate per arm
     (the V4/GPU-C posterior-inflation detector);
  4. budget accounting in BOTH currencies — rollouts (episodes + sim steps)
     and gradient updates, split by live/relabeled, so matched-rollout
     comparisons can't hide relabeling's extra updates and matched-step
     comparisons can't hide the teacher's early-termination speedup.

numpy-only; no simulator import.  `EvalProtocol` is the piece a TaskSpace
plugs into; `RunLedger` is threaded through training via the trainer's
on_eval hook or manual calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# coverage currency
# ---------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c successes (Chen et al. 2021).

    P(at least one success in k draws without replacement) =
    1 - C(n-c, k)/C(n, k) = 1 - prod_{i=0..k-1} (n-c-i)/(n-i).
    """
    if k > n:
        raise ValueError(f"pass@{k} needs at least k samples, got n={n}")
    if n - c < k:
        return 1.0
    ratio = 1.0
    for i in range(k):
        ratio *= (n - c - i) / (n - i)
    return 1.0 - ratio


@dataclass
class TaskEval:
    task_id: int
    n: int                    # eval episodes run
    c: int                    # successes
    steps: int = 0            # total sim steps consumed

    def rate(self) -> float:
        return self.c / self.n if self.n else 0.0

    def coverage(self, k: int) -> float:
        return pass_at_k(self.n, self.c, k)


class EvalProtocol:
    """Fixed-seed, fixed-init-state eval sweep over a task list.

    eval_fn(task_id, n_episodes) -> (successes, sim_steps): the env side —
    for CosmosLiberoSpace, a rollout_group call with the teacher bypassed
    and a HELD-OUT init-state set (never the training states).  The same
    seeds/init states must be reused across arms and checkpoints — pairing
    is what makes small deltas readable (REPORT.md multi-seed protocol).
    """

    def __init__(self, task_ids: Sequence[int], eval_fn: Callable,
                 n_episodes: int = 16, ks: Sequence[int] = (1, 4, 8)):
        self.task_ids = list(task_ids)
        self.eval_fn = eval_fn
        self.n_episodes = n_episodes
        self.ks = [k for k in ks if k <= n_episodes]

    def run(self) -> dict:
        evals = []
        for tid in self.task_ids:
            c, steps = self.eval_fn(tid, self.n_episodes)
            evals.append(TaskEval(tid, self.n_episodes, int(c), int(steps)))
        return summarize(evals, ks=self.ks)


def summarize(evals: Sequence[TaskEval], *, ks: Sequence[int] = (1, 4, 8),
              easy_set: Optional[Sequence[int]] = None,
              baseline_rates: Optional[dict] = None) -> dict:
    """Aggregate one eval sweep.

    easy_set: task ids of the retention probe.  If absent and
    baseline_rates given (task_id -> SFT-baseline rate), the easiest decile
    by baseline is used — fix it ONCE per experiment, from the baseline
    checkpoint, never re-derived per arm (else each arm gets its own probe
    and retention numbers stop being comparable).
    """
    rates = np.array([e.rate() for e in evals])
    out = {
        "mean_success": float(rates.mean()) if len(rates) else 0.0,
        "n_tasks": len(evals),
        "eval_episodes": int(sum(e.n for e in evals)),
        "eval_sim_steps": int(sum(e.steps for e in evals)),
    }
    for k in ks:
        out[f"success@{k}"] = float(np.mean([e.coverage(k) for e in evals]))
    if easy_set is None and baseline_rates:
        ordered = sorted(baseline_rates, key=baseline_rates.get, reverse=True)
        easy_set = ordered[:max(1, len(ordered) // 10)]
    if easy_set is not None:
        easy = [e for e in evals if e.task_id in set(easy_set)]
        if easy:
            out["easy_decile_retention"] = float(np.mean([e.rate()
                                                          for e in easy]))
    return out


# ---------------------------------------------------------------------------
# teacher calibration
# ---------------------------------------------------------------------------
def teacher_calibration(teacher, evals: Sequence[TaskEval],
                        min_visits: int = 2) -> dict:
    """Posterior p̂ vs held-out eval rate — the posterior-inflation detector.

    The V4/GPU-C failure signature was p̂ 0.81 vs eval 0.47: relabels (or any
    off-policy evidence) leaking into the posterior show up here as a large
    positive bias long before they show up as a worse final.  Alert on
    |bias| drift, not absolute value (Thompson optimism gives a small
    positive bias by design).
    """
    p_hat = teacher.pass_rate_estimates()
    pairs = [(p_hat[e.task_id], e.rate()) for e in evals
             if e.task_id < len(p_hat)
             and teacher.visits[e.task_id] >= min_visits]
    if not pairs:
        return {"teacher/calibration_n": 0}
    est, obs = np.array(pairs).T
    return {
        "teacher/calibration_n": len(pairs),
        "teacher/calibration_bias": float((est - obs).mean()),
        "teacher/calibration_mae": float(np.abs(est - obs).mean()),
    }


# ---------------------------------------------------------------------------
# budget accounting (both currencies)
# ---------------------------------------------------------------------------
@dataclass
class RunLedger:
    """Counts everything both matched-budget protocols need.

    Matched-rollouts hides the teacher's early-termination speedup and
    relabeling's extra gradient updates; matched-wall-clock hides nothing
    but needs exclusive hardware.  Record enough for both tables.
    """
    episodes: int = 0
    sim_steps: int = 0
    server_batches: int = 0          # /predict_batch calls (diffusion forwards)
    live_groups: int = 0
    dead_groups: int = 0
    relabeled_groups: int = 0
    live_updates: int = 0            # policy.update calls from live groups
    relabel_updates: int = 0         # ... from relabeled groups (the "free
                                     # signal" that is NOT free in update count)
    wall_seconds: float = 0.0
    history: list = field(default_factory=list)

    def observe_group(self, n_episodes: int, sim_steps: int, *,
                      live: bool, relabeled: bool = False,
                      server_batches: int = 0) -> None:
        # relabeled only applies to dead groups (a live group's rewards are
        # its own) — assert to keep callers honest (review fix 3).
        assert not (live and relabeled), \
            "relabeled=True requires live=False (dead groups only)"
        self.episodes += n_episodes
        self.sim_steps += sim_steps
        self.server_batches += server_batches
        if live:
            self.live_groups += 1
            self.live_updates += 1
        else:
            self.dead_groups += 1
            if relabeled:
                self.relabeled_groups += 1
                self.relabel_updates += 1

    def snapshot(self, step: int, extra: Optional[dict] = None) -> dict:
        row = {
            "step": step,
            "episodes": self.episodes,
            "sim_steps": self.sim_steps,
            "server_batches": self.server_batches,
            "updates_live": self.live_updates,
            "updates_relabel": self.relabel_updates,
            "dead_group_rate": (self.dead_groups
                                / max(self.live_groups + self.dead_groups, 1)),
            "relabel_yield": (self.relabeled_groups
                              / max(self.dead_groups, 1)),
            "wall_seconds": self.wall_seconds,
        }
        if extra:
            row.update(extra)
        self.history.append(row)
        return row


def matched_budget_report(arms: dict, *, currency: str = "episodes") -> str:
    """Render arm -> (ledger, final_eval_summary) at a common budget.

    Truncates every arm's snapshot history to the largest budget all arms
    reached in `currency` ('episodes' | 'sim_steps' | 'wall_seconds'), then
    tabulates the last common snapshot — the fair matched-budget row — and
    each arm's own final row for the wall-clock story.
    """
    if not arms:
        return "(no arms)"
    common = min(l.history[-1][currency] for l, _ in arms.values()
                 if l.history)
    lines = [f"matched budget: {currency} = {common}",
             f"{'arm':24s} {'mean':>6s} {'s@8':>6s} {'retain':>7s} "
             f"{'dead%':>6s} {'upd(L/R)':>10s} {'episodes':>9s}"]
    for name, (ledger, final_eval) in arms.items():
        at = [r for r in ledger.history if r[currency] <= common]
        row = at[-1] if at else (ledger.history[-1] if ledger.history else {})
        ev = row.get("eval", final_eval)
        lines.append(
            f"{name:24s} {ev.get('mean_success', float('nan')):6.3f} "
            f"{ev.get('success@8', float('nan')):6.3f} "
            f"{ev.get('easy_decile_retention', float('nan')):7.3f} "
            f"{100 * row.get('dead_group_rate', 0):5.1f}% "
            f"{row.get('updates_live', 0):4d}/{row.get('updates_relabel', 0):<4d} "
            f"{row.get('episodes', 0):9d}")
    return "\n".join(lines)
