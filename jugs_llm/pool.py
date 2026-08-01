"""Jugs pool for E-LLM-3: tiered generation, exact verifier, relabel map.

Wraps the vendored reasoning-gym jugs task (exact move-sequence simulator)
into the shape our RLVR loop needs:

- tiered pool: tiers are (num_jugs, min_required_moves) cells
- prompt format: fixed instruction + JSON answer contract (mirrors the
  Countdown setup: answer inside <answer>...</answer>, one move per line)
- verify(): exact rule-based check (no LLM judge) — contract 1
- relabel(): the exact relabel map — a *failed* but parseable move
  sequence is replayed; every amount any jug held at any prefix is a
  verified achieved target, so the group can be relabeled to the
  achieved target nearest the requested one (with the prompt's goal
  conditioning rewritten) — contract 2. Returns (new_target,
  truncated_moves, prefix_len) or None when nothing was achieved
  (e.g. unparseable output).

Design notes recorded for the prereg: target selection = the achieved
value with the LOWEST posterior pass-rate estimate among candidates
(the gate then decides admission); here we only expose candidates.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from random import Random

RG_PATH = Path(__file__).resolve().parents[2] / "external" / "reasoning-gym"


def _load_jugs_module():
    """Load ONLY the jugs functions, skipping the reasoning_gym package
    __init__ (whose import chain needs py>=3.10 dataclass kw_only)."""
    import types

    src = (RG_PATH / "reasoning_gym" / "algorithmic" / "jugs.py").read_text()
    # keep everything above the package-relative dataset plumbing
    # (cut before the @dataclass decorating JugsConfig)
    cut = src.index("class JugsConfig")
    head = src[:cut].rsplit("@dataclass", 1)[0]
    # drop package-relative imports (need the full package otherwise)
    head = "\n".join(l for l in head.splitlines()
                     if not l.startswith("from .."))
    mod = types.ModuleType("rg_jugs_core")
    exec(compile(head, "rg_jugs_core", "exec"), mod.__dict__)
    return mod


_jugs = _load_jugs_module()
generate_puzzle = _jugs.generate_puzzle
min_moves_n = _jugs.min_moves_n
verify_solution = _jugs.verify_solution

PROMPT_TEMPLATE = """You have {n} water jugs with capacities {caps} litres. All jugs start empty.
Allowed moves (one per line):
  fill X       (fill jug X to its capacity)
  empty X      (empty jug X)
  pour X->Y    (pour from X into Y until X is empty or Y is full)
Jugs are labelled {labels}.
Goal: make any one jug contain exactly {target} litres.
Think step by step, then give ONLY the move list inside <answer> tags,
one move per line, like:
<answer>
fill A
pour A->B
</answer>
"""

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
MOVE_RE = re.compile(r"^(fill [A-Z]|empty [A-Z]|pour [A-Z]->[A-Z])$")


@dataclass
class JugsTask:
    jug_capacities: list[int]
    target: int
    min_moves: int
    tier: str

    def prompt(self) -> str:
        n = len(self.jug_capacities)
        labels = ", ".join(chr(ord("A") + i) for i in range(n))
        return PROMPT_TEMPLATE.format(
            n=n, caps=self.jug_capacities, labels=labels, target=self.target)


def parse_moves(completion: str, n_jugs: int) -> list[str] | None:
    """Extract a syntactically valid move list, or None."""
    m = ANSWER_RE.search(completion)
    if not m:
        return None
    valid_labels = {chr(ord("A") + i) for i in range(n_jugs)}
    moves = []
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if not MOVE_RE.match(line):
            return None
        for tok in re.findall(r"[A-Z]", line):
            if tok not in valid_labels:
                return None
        moves.append(line)
    return moves or None


def verify(task: JugsTask, completion: str) -> bool:
    moves = parse_moves(completion, len(task.jug_capacities))
    if moves is None:
        return False
    try:
        ok, _ = verify_solution(
            {"jug_capacities": task.jug_capacities, "target": task.target},
            moves)
    except (ValueError, KeyError, IndexError):
        return False
    return bool(ok)


def relabel_candidates(task: JugsTask, completion: str):
    """Exact relabel map for a FAILED rollout.

    Returns list of (achieved_target, moves_prefix) — every distinct
    nonzero amount any jug held after any prefix of the parsed moves,
    excluding the requested target (that would be a success, not a
    failure) — each certified by the same simulator that verifies
    successes. Empty list if the output didn't parse.
    """
    moves = parse_moves(completion, len(task.jug_capacities))
    if moves is None:
        return []
    try:
        _, states = verify_solution(
            {"jug_capacities": task.jug_capacities, "target": task.target},
            moves)
    except (ValueError, KeyError, IndexError):
        return []
    out, seen = [], set()
    for k, state in enumerate(states[1:], start=1):  # states[0] = all empty
        for amount in state:
            if amount > 0 and amount != task.target and amount not in seen:
                seen.add(amount)
                out.append((int(amount), moves[:k]))
    return out


def relabeled_task(task: JugsTask, new_target: int) -> JugsTask:
    """Conditioning rewrite (contract 2): same jugs, new goal."""
    mm = min_moves_n(task.jug_capacities, new_target)
    return JugsTask(task.jug_capacities, int(new_target),
                    -1 if mm is None else int(mm), task.tier + "_relabeled")


# ------------------------------------------------------------------ pool
# name, num_jugs, capacity range, min_moves band [lo, hi)
# Capacity range is decoupled from the moves band (the vendored
# generate_puzzle ties caps to difficulty, which collapses low tiers to
# a handful of distinct puzzles); tasks are deduped on (caps, target).
TIER_GRID = [
    ("t0", 2, (3, 20), (2, 5)),   # 330 unique tasks exist in this cell
    ("t1", 3, (3, 12), (4, 8)),
    ("t2", 3, (5, 15), (8, 12)),
    ("t3", 4, (5, 15), (12, 16)),
    ("t4", 5, (7, 19), (16, 32)),
]


def build_pool(per_tier: int = 200, seed: int = 0,
               max_attempts: int = 200000) -> list[JugsTask]:
    from functools import reduce
    from math import gcd

    rng = Random(seed)
    pool = []
    for name, n_jugs, (cap_lo, cap_hi), (mm_lo, mm_hi) in TIER_GRID:
        seen: set = set()
        tier_tasks: list[JugsTask] = []
        attempts = 0
        while len(tier_tasks) < per_tier and attempts < max_attempts:
            attempts += 1
            caps = sorted(rng.randint(cap_lo, cap_hi) for _ in range(n_jugs))
            g = reduce(gcd, caps)
            targets = [t for t in range(1, max(caps) + 1) if t % g == 0]
            if not targets:
                continue
            target = rng.choice(targets)
            key = (tuple(caps), target)
            if key in seen:
                continue
            seen.add(key)
            mm = min_moves_n(caps, target)
            if mm is None or not (mm_lo <= mm < mm_hi):
                continue
            tier_tasks.append(JugsTask(list(caps), target, mm, name))
        if len(tier_tasks) < per_tier:
            print(f"warning: {name} exhausted at {len(tier_tasks)} unique "
                  f"tasks after {attempts} attempts")
        pool += tier_tasks
    return pool


def main():
    pool = build_pool(per_tier=200, seed=0)
    path = Path(__file__).with_name("pool_v1.jsonl")
    with open(path, "w") as f:
        for t in pool:
            f.write(json.dumps({"jug_capacities": t.jug_capacities,
                                "target": t.target, "min_moves": t.min_moves,
                                "tier": t.tier}) + "\n")
    from collections import Counter
    tiers = Counter(t.tier for t in pool)
    print(f"wrote {path}: {dict(tiers)}")
    for name, *_ in TIER_GRID:
        v = [t.min_moves for t in pool if t.tier == name]
        uniq = len({(tuple(t.jug_capacities), t.target)
                    for t in pool if t.tier == name})
        if v:
            print(f"  {name}: min_moves {min(v)}-{max(v)}, unique {uniq}/{len(v)}")


if __name__ == "__main__":
    main()
