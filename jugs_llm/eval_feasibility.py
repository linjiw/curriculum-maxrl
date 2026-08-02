"""Feasibility gate for E-LLM-3: base pass@1/pass@16 per Jugs tier.

Runs candidate base models over pool_v1.jsonl with N samples/task and the
exact verifier; also measures relabel yield on failures (fraction of
failed rollouts that produce >=1 relabel candidate — the recycling arm's
fuel gauge). Writes feasibility_<model_tag>.json.

Decision rule (DESIGN_E_LLM3.md): need >=1 tier with pass@1 in [1%,40%]
and >=1 tier with pass@16 ~ 0. Try SmolLM2-360M-Instruct first, then
Qwen2.5-0.5B/1.5B-Instruct.

Usage (needs ~6-16GB GPU depending on model; waits for free GPU):
  python3 eval_feasibility.py --model HuggingFaceTB/SmolLM2-360M-Instruct \
      --per-tier 50 --n 16 [--two-shot]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pool import JugsTask, relabel_candidates, verify  # noqa: E402
# one source of truth: the exemplars the RL parquet actually uses
# (exemplar goals deliberately avoid the "contain exactly N litres"
# phrase so the hindsight rewrite can never match them)
from prep_jugs import TWO_SHOT_PREFIX  # noqa: E402


def load_pool(path: str, per_tier: int) -> list[JugsTask]:
    by_tier = defaultdict(list)
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            t = JugsTask(d["jug_capacities"], d["target"], d["min_moves"],
                         d["tier"])
            if len(by_tier[t.tier]) < per_tier:
                by_tier[t.tier].append(t)
    return [t for ts in by_tier.values() for t in ts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--pool", default=str(Path(__file__).with_name("pool_v1.jsonl")))
    ap.add_argument("--per-tier", type=int, default=50)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--two-shot", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--min-free-mb", type=int, default=8000)
    args = ap.parse_args()

    import subprocess
    while True:
        free = int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"]).split()[0])
        if free > args.min_free_mb:
            break
        print(f"waiting for GPU ({free}MB free)...", flush=True)
        time.sleep(300)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tasks = load_pool(args.pool, args.per_tier)
    print(f"{len(tasks)} tasks, N={args.n}", flush=True)

    stats = defaultdict(lambda: {"n_tasks": 0, "pass1_hits": 0,
                                 "passk_hits": 0, "n_fail_rollouts": 0,
                                 "n_fail_with_relabel": 0,
                                 "n_parseable": 0, "n_rollouts": 0})
    t_start = time.time()
    for ti, task in enumerate(tasks):
        prompt = task.prompt()
        if args.two_shot:
            prompt = TWO_SHOT_PREFIX + prompt
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        enc = tok(chat, return_tensors="pt").to("cuda")
        outs = []
        remaining = args.n
        while remaining > 0:
            b = min(args.batch, remaining)
            with torch.no_grad():
                gen = model.generate(
                    **enc, do_sample=True, temperature=args.temperature,
                    top_p=0.95, num_return_sequences=b,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tok.pad_token_id)
            outs += [tok.decode(g[enc.input_ids.shape[1]:],
                                skip_special_tokens=True) for g in gen]
            remaining -= b
        s = stats[task.tier]
        s["n_tasks"] += 1
        hits = 0
        for o in outs:
            s["n_rollouts"] += 1
            ok = verify(task, o)
            if ok:
                hits += 1
            else:
                s["n_fail_rollouts"] += 1
                cands = relabel_candidates(task, o)
                if cands:
                    s["n_fail_with_relabel"] += 1
                    s["n_parseable"] += 1
                elif "<answer>" in o.lower():
                    pass  # unparseable answer block
        s["pass1_hits"] += hits
        s["passk_hits"] += int(hits > 0)
        if (ti + 1) % 10 == 0:
            el = time.time() - t_start
            print(f"[{ti+1}/{len(tasks)}] {el:.0f}s "
                  + " ".join(f"{k}:p1={v['pass1_hits']/max(1,v['n_rollouts']):.3f}"
                             f",p@{args.n}={v['passk_hits']/max(1,v['n_tasks']):.2f}"
                             for k, v in sorted(stats.items())), flush=True)

    result = {"model": args.model, "n": args.n, "two_shot": args.two_shot,
              "per_tier": args.per_tier, "temperature": args.temperature,
              "tiers": {}}
    for tier, s in sorted(stats.items()):
        result["tiers"][tier] = {
            "pass1": s["pass1_hits"] / max(1, s["n_rollouts"]),
            f"pass{args.n}": s["passk_hits"] / max(1, s["n_tasks"]),
            "relabel_yield_on_fail":
                s["n_fail_with_relabel"] / max(1, s["n_fail_rollouts"]),
            "n_tasks": s["n_tasks"]}
        print(tier, result["tiers"][tier], flush=True)
    tag = args.model.split("/")[-1].replace(".", "_") \
        + ("_2shot" if args.two_shot else "")
    out = Path(__file__).with_name(f"feasibility_{tag}.json")
    json.dump(result, open(out, "w"), indent=1)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
