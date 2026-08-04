"""Countdown (TinyZero/reasoning-gym) as a TaskSpace — the E-LLM-2 design.

A failed trace that uses the numbers correctly still evaluates to a value
``v``. The corrected adapter chooses one achieved value as a shared anchor,
rewrites every row to that target, and re-verifies the whole group. This
produces a genuine fixed-destination contrast when ``0 < K' < N``. Historical
Countdown runs used per-trace targets instead; they do not validate this
corrected contract.

Task bins for the teacher = difficulty tiers (operand count), matching the
maze's goal-distance axis. Individual instances stream within a tier; the
posterior lives at tier level (the verl integration keeps per-prompt rows —
both are supported by the same teacher).

The rollout_fn hook keeps this module free of any LLM dependency (mirrors
cosmos_libero): the real integration passes a vllm/HF generate call; tests
pass a mock. Verification is pure python (ported from TinyZero's
reward_score/countdown.py, hardened: regex-gated eval with no builtins).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from frontier_rl.interfaces import GroupResult

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ALLOWED_EXPR = re.compile(r"^[\d+\-*/().\s]+$")
_INT_RE = re.compile(r"\d+")

PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number "
    "can only be used once. Show your work in <think> </think> tags. And "
    "return the final answer in <answer> </answer> tags, for example "
    "<answer> (1 + 2) / 3 </answer>."
)


def extract_equation(solution_str: str) -> Optional[str]:
    """Last <answer>...</answer> block, or None."""
    matches = _ANSWER_RE.findall(solution_str)
    return matches[-1].strip() if matches else None


def uses_numbers_once(equation: str, numbers: list[int]) -> bool:
    try:
        used = sorted(int(n) for n in _INT_RE.findall(equation))
    except ValueError:
        return False
    return used == sorted(numbers)


def safe_eval(equation: str) -> Optional[float]:
    """Evaluate an arithmetic expression; None on any violation/error."""
    if not _ALLOWED_EXPR.match(equation):
        return None
    try:
        result = eval(equation, {"__builtins__": None}, {})  # noqa: S307
        return float(result)
    except Exception:
        return None


def verify(solution_str: str, numbers: list[int], target: float,
           atol: float = 1e-5) -> bool:
    """TinyZero's strict verifier, binary form (no format band)."""
    eq = extract_equation(solution_str)
    if eq is None or not uses_numbers_once(eq, numbers):
        return False
    v = safe_eval(eq)
    return v is not None and abs(v - target) < atol


def achieved_value(solution_str: str, numbers: list[int]) -> Optional[float]:
    """The exact-relabel map: what target did this trace actually reach?

    Returns v if the trace's equation is well-formed AND uses each number
    exactly once (P6 contract 1 requires the relabeled instance to be a TRUE
    task of the space — same numbers, achievable target v). None otherwise.
    Integer-valued v only: countdown targets are integers, and relabeling to
    e.g. 7.5 would leave the tier's task distribution.
    """
    eq = extract_equation(solution_str)
    if eq is None or not uses_numbers_once(eq, numbers):
        return None
    v = safe_eval(eq)
    if v is None or not np.isfinite(v):
        return None
    if abs(v - round(v)) > 1e-9:
        return None
    v = int(round(v))
    return v if 0 < v <= 100_000 else None


def rewrite_prompt(prompt: str, old_target: int, new_target: int) -> Optional[str]:
    """Contract 2: swap the target in the conditioning, template-exact.

    Replaces the standalone occurrence of old_target that follows 'equals'
    (the template slot), not any digit substring. None if the slot isn't
    found — the caller must then DROP the relabel (never train with stale
    conditioning; the gridworld ablation measured that mistake at -0.06 AUC).
    """
    pattern = re.compile(r"(equals )" + re.escape(str(old_target)) + r"\b")
    out, n = pattern.subn(r"\g<1>" + str(new_target), prompt, count=1)
    return out if n == 1 else None


@dataclass
class CountdownInstance:
    numbers: list[int]
    target: int
    tier: int          # index into the tier list (operand-count buckets)


class CountdownSpace:
    """TaskSpace over difficulty tiers of a fixed Countdown pool.

    rollout_fn(prompt, n) -> list[str]: n sampled completions (the LLM hook).
    Tiers are operand counts (e.g. [2, 3, 4]); each rollout_group draws one
    instance from the tier's pool slice — group-level evidence matches the
    verl integration's per-prompt groups aggregated to tiers.
    """

    def __init__(self, pool: list[CountdownInstance], tiers: list[int],
                 rollout_fn: Callable[[str, int], list[str]], seed: int = 0):
        self.pool = pool
        self.tiers = tiers
        self.rollout_fn = rollout_fn
        self.rng = np.random.default_rng(seed)
        self._by_tier = {t: [i for i, inst in enumerate(pool) if inst.tier == t]
                         for t in range(len(tiers))}
        for t, idxs in self._by_tier.items():
            if not idxs:
                raise ValueError(f"tier {t} (operands={tiers[t]}) has no instances")

    @property
    def n_tasks(self) -> int:
        return len(self.tiers)

    def _prompt(self, inst: CountdownInstance) -> str:
        return PROMPT_TEMPLATE.format(numbers=inst.numbers, target=inst.target)

    def rollout_group(self, task_id: int, n_rollouts: int) -> GroupResult:
        idx = int(self.rng.choice(self._by_tier[task_id]))
        inst = self.pool[idx]
        prompt = self._prompt(inst)
        completions = self.rollout_fn(prompt, n_rollouts)
        rewards = np.array([float(verify(c, inst.numbers, inst.target))
                            for c in completions])
        trajs = [{"prompt": prompt, "completion": c} for c in completions]
        infos = [{"pool_index": idx, "numbers": inst.numbers,
                  "target": inst.target, "tier": inst.tier} for _ in completions]
        return GroupResult(task_id, rewards, trajectories=trajs, infos=infos)

    def relabel(self, group: GroupResult):
        """Relabel the entire group to one achieved integer target.

        The most frequent admissible achieved value is used as the shared
        anchor (ties prefer the value closest to the requested target, then
        the smaller value). Every prompt is rewritten and every completion
        is re-verified under that anchor. Constant destination groups are
        rejected because a centered estimator requires ``0 < K' < N``.
        """
        info = group.infos[0]
        numbers, old_target = info["numbers"], info["target"]
        achieved = [achieved_value(t["completion"], numbers)
                    for t in group.trajectories]
        counts = {
            value: achieved.count(value)
            for value in set(achieved)
            if value is not None and value != old_target
        }
        n = len(group.trajectories)
        candidates = [value for value, count in counts.items()
                      if 0 < count < n]
        if not candidates:
            return None
        anchor = min(
            candidates,
            key=lambda value: (-counts[value], abs(value - old_target), value),
        )

        new_trajs = []
        for traj in group.trajectories:
            rewritten = rewrite_prompt(traj["prompt"], old_target, anchor)
            if rewritten is None:
                return None
            new_trajs.append({
                "prompt": rewritten,
                "completion": traj["completion"],
                "relabeled_target": anchor,
            })
        new_rewards = np.asarray([
            float(verify(t["completion"], numbers, anchor))
            for t in new_trajs
        ])
        if not 0 < new_rewards.sum() < n:
            return None
        return group.task_id, np.asarray(new_rewards), new_trajs
