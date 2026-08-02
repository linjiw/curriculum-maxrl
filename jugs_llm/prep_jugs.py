"""Build the E-LLM-3 Jugs pool in verl-parquet format.

Mirrors countdown/prep_countdown.py: per-tier data_source
(jugs_tier0..4) so verl's validation metrics resolve per tier; fixed
pool (fixedness is the hindsight-compounding regime); train/test split
deduped on (capacities, target).

Reads pool_v1.jsonl (200 unique tasks/tier from pool.py) and emits
~/data/jugs_v1/{train,test}.parquet. Test = 40/tier held out.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import pandas as pd

SYSTEM = ("You are a helpful assistant. You first think about the reasoning "
          "process step by step and then provide the user with the answer.")

# Two-shot exemplars baked into every prompt (feasibility showed the
# 2-shot base model has the ideal tier landscape, while SFT format-priming
# collapsed the policy onto short generic sequences — see DESIGN_E_LLM3).
# CRITICAL: the exemplar goals use "obtain ... litres in one jug", NOT the
# task's "contain exactly {t} litres" phrasing — the hindsight prompt
# rewrite matches the task slot by that exact phrase and must never hit an
# exemplar (verified in test_jugs_hindsight_verl.py::two_shot case).
TWO_SHOT_PREFIX = """Example 1:
Two jugs with capacities [3, 5]. Objective: obtain 4 litres in one jug.
<answer>
fill B
pour B->A
empty A
pour B->A
fill B
pour B->A
</answer>

Example 2:
Two jugs with capacities [4, 9]. Objective: obtain 9 litres in one jug.
<answer>
fill B
</answer>

Now solve this one:
"""

PROMPT_TEMPLATE = """You have {n} water jugs with capacities {caps} litres. All jugs start empty.
Allowed moves (one per line):
  fill X       (fill jug X to its capacity)
  empty X      (empty jug X)
  pour X->Y    (pour from X into Y until X is empty or Y is full)
Jugs are labelled {labels}.
Goal: make any one jug contain exactly {target} litres.
Show your work in <think> </think> tags. Then give ONLY the move list inside <answer> tags, one move per line, like:
<answer>
fill A
pour A->B
</answer>
"""


def row_for(d: dict, split: str, index: int) -> dict:
    caps = d["jug_capacities"]
    n = len(caps)
    labels = ", ".join(chr(ord("A") + i) for i in range(n))
    q = TWO_SHOT_PREFIX + PROMPT_TEMPLATE.format(n=n, caps=caps,
                                                 labels=labels,
                                                 target=d["target"])
    tier_idx = int(d["tier"][1:])
    return {
        "data_source": f"jugs_tier{tier_idx}",
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": q}],
        "ability": "planning",
        "reward_model": {"style": "rule",
                         "ground_truth": {"target": d["target"],
                                          "jug_capacities": caps}},
        "extra_info": {"split": split, "index": index,
                       "tier": tier_idx, "min_moves": d["min_moves"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pool_v1.jsonl"))
    ap.add_argument("--out_dir", default=os.path.expanduser("~/data/jugs_v1"))
    ap.add_argument("--test_per_tier", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    by_tier: dict[str, list[dict]] = {}
    with open(args.pool) as f:
        for line in f:
            d = json.loads(line)
            by_tier.setdefault(d["tier"], []).append(d)

    rng = random.Random(args.seed)
    train, test, idx = [], [], 0
    for tier in sorted(by_tier):
        rows = by_tier[tier][:]
        rng.shuffle(rows)
        for d in rows[:args.test_per_tier]:
            test.append(row_for(d, "test", 10_000_000 + len(test)))
        for d in rows[args.test_per_tier:]:
            train.append(row_for(d, "train", idx))
            idx += 1
    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(train).to_parquet(os.path.join(args.out_dir, "train.parquet"))
    pd.DataFrame(test).to_parquet(os.path.join(args.out_dir, "test.parquet"))
    from collections import Counter
    print(f"train {len(train)} {Counter(r['data_source'] for r in train)}")
    print(f"test  {len(test)} {Counter(r['data_source'] for r in test)}")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
