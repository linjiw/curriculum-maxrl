"""Jugs reward for verl (custom_reward_function hook) + relabel helpers.

Strict-binary, dependency-free (verl workers import only this file), same
contract as curriculum_maxrl/countdown/countdown_reward.py:

    custom_reward_function.path=curriculum_maxrl/jugs/jugs_reward.py
    custom_reward_function.name=compute_score

ground_truth is {"target": int, "jug_capacities": list[int]} carried in
the parquet's reward_model.ground_truth column (see prep_jugs.py).

Relabel semantics (E-LLM-3): `achieved_amounts` returns the set of
amounts present in the FINAL state of a parsed move sequence — relabeling
to any of them keeps the response text a verified success as-is (no
response rewrite; the <answer> artifact is coherent under the new goal,
think-text may reference the old goal — the same pre-registered deviation
as Countdown E-LLM-2). Prefix-achieved amounts (richer map, needs
response truncation) are exposed via `achieved_prefix_amounts` for a
later ablation; v1 uses final-state only.
"""

from __future__ import annotations

import re

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_MOVE_RE = re.compile(r"^(fill [A-Z]|empty [A-Z]|pour [A-Z]->[A-Z])$")
_MAX_MOVES = 64


def _parse_moves(solution_str: str, n_jugs: int):
    m = _ANSWER_RE.search(solution_str or "")
    if not m:
        return None
    valid = {chr(ord("A") + i) for i in range(n_jugs)}
    moves = []
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if not _MOVE_RE.match(line):
            return None
        if any(tok not in valid for tok in re.findall(r"[A-Z]", line)):
            return None
        moves.append(line)
        if len(moves) > _MAX_MOVES:
            return None
    return moves or None


def _simulate(moves, capacities):
    """Exact simulator. Returns list of states (incl. initial) or None."""
    n = len(capacities)
    idx = {chr(ord("A") + i): i for i in range(n)}
    state = [0] * n
    states = [tuple(state)]
    for mv in moves:
        parts = mv.split()
        if parts[0] == "fill":
            state[idx[parts[1]]] = capacities[idx[parts[1]]]
        elif parts[0] == "empty":
            state[idx[parts[1]]] = 0
        else:  # pour X->Y
            src, dst = parts[1].split("->")
            i, j = idx[src], idx[dst]
            amt = min(state[i], capacities[j] - state[j])
            state[i] -= amt
            state[j] += amt
        states.append(tuple(state))
    return states


def compute_score(data_source=None, solution_str=None, ground_truth=None,
                  extra_info=None, **kwargs) -> float:
    """Binary: 1.0 iff the move sequence is valid and the final state has
    some jug at exactly the target amount."""
    caps = list(ground_truth["jug_capacities"])
    target = int(ground_truth["target"])
    moves = _parse_moves(solution_str, len(caps))
    if moves is None:
        return 0.0
    states = _simulate(moves, caps)
    return 1.0 if target in states[-1] else 0.0


# ---------------------------------------------------------------- hindsight
def achieved_amounts(solution_str: str, jug_capacities) -> list[int]:
    """Amounts (>0) present in the FINAL state of a parsed sequence.
    Relabeling the target to any of these keeps the response a verified
    success without touching response tokens."""
    caps = list(jug_capacities)
    moves = _parse_moves(solution_str, len(caps))
    if moves is None:
        return []
    final = _simulate(moves, caps)[-1]
    return sorted({int(a) for a in final if a > 0})


def achieved_prefix_amounts(solution_str: str, jug_capacities):
    """Richer map for the v2 ablation: {amount: shortest prefix length}
    over ALL prefixes. Using these requires truncating the response to
    the prefix (response rewrite) — not used in v1."""
    caps = list(jug_capacities)
    moves = _parse_moves(solution_str, len(caps))
    if moves is None:
        return {}
    states = _simulate(moves, caps)
    out: dict[int, int] = {}
    for k, st in enumerate(states[1:], start=1):
        for a in st:
            if a > 0 and a not in out:
                out[int(a)] = k
    return out


TARGET_SLOT_RE_TEMPLATE = r"(contain exactly )%d( litres)"


def rewrite_prompt_text(text: str, old_target: int, new_target: int):
    """Swap the goal slot in a decoded Jugs prompt (P6 contract 2).
    Returns rewritten text or None if the slot is missing/ambiguous."""
    pattern = re.compile(TARGET_SLOT_RE_TEMPLATE % old_target)
    new_text, n = pattern.subn(r"\g<1>" + str(new_target) + r"\g<2>",
                               text, count=1)
    return new_text if n == 1 else None
