"""Build a deterministic real-structure Countdown v2 dataset.

The prior execution fork's pool builder and SFT artifact were not vendored.
This replacement freezes the same public task definition with explicit
provenance and integrity checks:

* tiers are 2, 3, and 4 operands;
* 10,000 train tasks (2k/4k/4k) and 128 held-out tasks per tier;
* task identity is ``(target, sorted operand multiset)``;
* 3/4-operand candidates come from ``Jiayi-Pan/Countdown-Tasks-3to4``;
* every task is independently solved using all permutations, binary-tree
  parenthesizations, and exact rational division;
* SFT examples are drawn only from the RL training split.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import pandas as pd
from datasets import load_dataset

SYSTEM = "You are a helpful assistant. Think step by step, then provide the final answer."
PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number "
    "can only be used once. Show your work in <think> </think> tags. And "
    "return the final answer in <answer> </answer> tags, for example "
    "<answer> (1 + 2) / 3 </answer>."
)
TRAIN_COUNTS = {0: 2000, 1: 4000, 2: 4000}
TEST_PER_TIER = 128
SFT_COUNTS = {0: 1000, 1: 2500, 2: 2500}


def task_key(target: int, numbers: list[int]) -> tuple[int, tuple[int, ...]]:
    return int(target), tuple(sorted(int(n) for n in numbers))


@lru_cache(maxsize=100_000)
def _solve(values: tuple[tuple[int, int, str], ...], target: int) -> str | None:
    """Exact subset-DP over value/expression states (n <= 4)."""
    if len(values) == 1:
        num, den, expr = values[0]
        return expr if Fraction(num, den) == target else None

    n = len(values)
    for mask in range(1, (1 << n) - 1):
        if not (mask & 1):
            continue
        left = [values[i] for i in range(n) if mask & (1 << i)]
        right = [values[i] for i in range(n) if not mask & (1 << i)]
        left_states = _all_results(tuple(left))
        right_states = _all_results(tuple(right))
        for a, ea in left_states.items():
            for b, eb in right_states.items():
                candidates = [
                    (a + b, f"({ea} + {eb})"),
                    (a - b, f"({ea} - {eb})"),
                    (b - a, f"({eb} - {ea})"),
                    (a * b, f"({ea} * {eb})"),
                ]
                if b:
                    candidates.append((a / b, f"({ea} / {eb})"))
                if a:
                    candidates.append((b / a, f"({eb} / {ea})"))
                for value, expr in candidates:
                    if value == target:
                        return expr
    return None


@lru_cache(maxsize=250_000)
def _all_results(values: tuple[tuple[int, int, str], ...]) -> dict[Fraction, str]:
    if len(values) == 1:
        num, den, expr = values[0]
        return {Fraction(num, den): expr}
    out: dict[Fraction, str] = {}
    n = len(values)
    for mask in range(1, (1 << n) - 1):
        if not (mask & 1):
            continue
        left = tuple(values[i] for i in range(n) if mask & (1 << i))
        right = tuple(values[i] for i in range(n) if not mask & (1 << i))
        for a, ea in _all_results(left).items():
            for b, eb in _all_results(right).items():
                options = [
                    (a + b, f"({ea} + {eb})"),
                    (a - b, f"({ea} - {eb})"),
                    (b - a, f"({eb} - {ea})"),
                    (a * b, f"({ea} * {eb})"),
                ]
                if b:
                    options.append((a / b, f"({ea} / {eb})"))
                if a:
                    options.append((b / a, f"({eb} / {ea})"))
                for value, expr in options:
                    if abs(value) <= 100_000:
                        out.setdefault(value, expr)
    return out


def solve_countdown(numbers: list[int], target: int) -> str | None:
    values = tuple((int(n), 1, str(int(n))) for n in numbers)
    direct = _all_results(values).get(Fraction(int(target)))
    return direct


def _two_operand_candidates(rng: random.Random):
    while True:
        numbers = [rng.randint(1, 100), rng.randint(1, 100)]
        a, b = map(Fraction, numbers)
        options = [a + b, a - b, b - a, a * b]
        if b:
            options.append(a / b)
        if a:
            options.append(b / a)
        integers = [int(v) for v in options if v.denominator == 1 and 0 < v <= 1000]
        if integers:
            yield rng.choice(integers), numbers


def collect_tasks(seed: int) -> dict[int, list[dict]]:
    rng = random.Random(seed)
    needed = {tier: TRAIN_COUNTS[tier] + TEST_PER_TIER for tier in TRAIN_COUNTS}
    by_tier: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    seen: set[tuple[int, tuple[int, ...]]] = set()

    for target, numbers in _two_operand_candidates(rng):
        key = task_key(target, numbers)
        if key in seen:
            continue
        solution = solve_countdown(numbers, target)
        if solution:
            seen.add(key)
            by_tier[0].append({"target": target, "numbers": numbers, "solution": solution})
        if len(by_tier[0]) >= needed[0]:
            break

    source = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    order = list(range(len(source)))
    rng.shuffle(order)
    for idx in order:
        numbers = [int(n) for n in source[idx]["nums"]]
        tier = len(numbers) - 2
        if tier not in (1, 2) or len(by_tier[tier]) >= needed[tier]:
            continue
        target = int(source[idx]["target"])
        key = task_key(target, numbers)
        if key in seen:
            continue
        solution = solve_countdown(numbers, target)
        if solution is None:
            continue
        seen.add(key)
        by_tier[tier].append({"target": target, "numbers": numbers, "solution": solution})
        if all(len(by_tier[t]) >= needed[t] for t in by_tier):
            break

    for tier, count in needed.items():
        if len(by_tier[tier]) < count:
            raise RuntimeError(f"tier {tier}: collected {len(by_tier[tier])}, need {count}")
        rng.shuffle(by_tier[tier])
    return by_tier


def rl_row(task: dict, tier: int, split: str, index: int) -> dict:
    prompt = PROMPT_TEMPLATE.format(numbers=task["numbers"], target=task["target"])
    return {
        "data_source": f"countdown_tier{tier}",
        "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": {"target": task["target"], "numbers": task["numbers"]}},
        "extra_info": {"split": split, "index": index, "tier": tier},
    }


def build(out_dir: Path, seed: int) -> dict:
    by_tier = collect_tasks(seed)
    train_tasks: list[tuple[int, dict]] = []
    test_tasks: list[tuple[int, dict]] = []
    for tier in sorted(by_tier):
        test_tasks.extend((tier, task) for task in by_tier[tier][:TEST_PER_TIER])
        train_tasks.extend((tier, task) for task in by_tier[tier][TEST_PER_TIER : TEST_PER_TIER + TRAIN_COUNTS[tier]])

    train_rows = [rl_row(task, tier, "train", i) for i, (tier, task) in enumerate(train_tasks)]
    test_rows = [rl_row(task, tier, "test", 10_000_000 + i) for i, (tier, task) in enumerate(test_tasks)]

    rng = random.Random(seed + 1)
    sft_rows = []
    for tier in sorted(SFT_COUNTS):
        candidates = [task for task_tier, task in train_tasks if task_tier == tier]
        rng.shuffle(candidates)
        for task in candidates[: SFT_COUNTS[tier]]:
            prompt = PROMPT_TEMPLATE.format(numbers=task["numbers"], target=task["target"])
            answer = f"<think>I will combine every supplied number exactly once.</think>\n<answer>{task['solution']}</answer>"
            sft_rows.append({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "tier": tier,
                "target": task["target"],
                "numbers": task["numbers"],
            })
    rng.shuffle(sft_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(out_dir / "train.parquet", index=False)
    pd.DataFrame(test_rows).to_parquet(out_dir / "test.parquet", index=False)
    with (out_dir / "sft_train.jsonl").open("w") as handle:
        for row in sft_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    train_keys = {task_key(t["target"], t["numbers"]) for _, t in train_tasks}
    test_keys = {task_key(t["target"], t["numbers"]) for _, t in test_tasks}
    sft_keys = {task_key(r["target"], r["numbers"]) for r in sft_rows}
    manifest = {
        "seed": seed,
        "source_3to4": "Jiayi-Pan/Countdown-Tasks-3to4",
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "sft_rows": len(sft_rows),
        "train_tiers": dict(sorted(Counter(r["data_source"] for r in train_rows).items())),
        "test_tiers": dict(sorted(Counter(r["data_source"] for r in test_rows).items())),
        "train_unique": len(train_keys),
        "test_unique": len(test_keys),
        "train_test_overlap": len(train_keys & test_keys),
        "sft_test_overlap": len(sft_keys & test_keys),
        "task_identity": "(target, sorted operand multiset)",
    }
    with (out_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=os.path.expanduser("~/data/countdown_v2_rebuilt"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(build(Path(args.out_dir), args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

