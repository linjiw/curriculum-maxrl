#!/usr/bin/env python3
"""GATE-DR sealed endpoint evaluation (prereg amendment 2026-08-13b).

Evaluates the 12 saved step-60 hf_model checkpoints on the frozen 384-task
Countdown test set (128 per tier): n=16 samples/task, temperature 1.0, top_p 1.0,
max 128 new tokens, vLLM engine seed 20260813, exact compute_score verifier.

Retains the full 16-bit binary outcome vector per task per run, from which it
reports per tier: mean@16, STANDARD observed-set pass@16 (fraction of tasks with
>=1 success among the 16), and the legacy VERL-style bootstrap best@16 proxy
(1000 with-replacement resamples of size 16, rng seed 20260813) for continuity
with historical numbers. Decision rules in the prereg are applied to the proxy
exactly as frozen; standard pass@16 is reported beside it.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, "/scratch/lwang44/curriculum-maxrl")
from curriculum_maxrl.countdown.countdown_reward import compute_score

import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

RUNTIME = "/scratch/lwang44/curriculum-maxrl-runtime"
OUT_DIR = f"{RUNTIME}/checkpoints/gate_dr_eval"
os.makedirs(OUT_DIR, exist_ok=True)
ARMS = ["b1h", "g0", "g085", "g070"]
SEEDS = [1, 2, 3]
ENGINE_SEED = 20260813

df = pd.read_parquet(f"{RUNTIME}/data/countdown_v2_rebuilt/test.parquet")
records = df.to_dict("records")
print(f"eval set: {len(records)} tasks", flush=True)

results = {}
for arm in ARMS:
    for seed in SEEDS:
        run = f"gatedr_{arm}_s{seed}"
        ckpt = f"{RUNTIME}/checkpoints/{run}/global_step_60/actor/huggingface"
        assert os.path.isdir(ckpt), f"missing checkpoint: {ckpt}"
        tok = AutoTokenizer.from_pretrained(ckpt)
        prompts = [tok.apply_chat_template(list(r["prompt"]), add_generation_prompt=True,
                                           tokenize=False) for r in records]
        llm = LLM(model=ckpt, gpu_memory_utilization=0.5, dtype="bfloat16",
                  seed=ENGINE_SEED)
        sp = SamplingParams(n=16, temperature=1.0, top_p=1.0, max_tokens=128)
        outs = llm.generate(prompts, sp, use_tqdm=False)
        per_task = []
        for r, o in zip(records, outs):
            gt = r["reward_model"]["ground_truth"]
            gt = {"target": int(gt["target"]), "numbers": list(map(int, gt["numbers"]))}
            bits = [int(compute_score(data_source=r["data_source"],
                                      solution_str=c.text, ground_truth=gt))
                    for c in o.outputs]
            assert len(bits) == 16
            per_task.append({"data_source": r["data_source"], "outcomes": bits})
        del llm  # free GPU before next checkpoint

        rng = np.random.default_rng(ENGINE_SEED)
        tiers = {}
        for t in (0, 1, 2):
            rows = [p["outcomes"] for p in per_task
                    if p["data_source"] == f"countdown_tier{t}"]
            a = np.array(rows)                                # (tasks, 16)
            mean16 = float(a.mean())
            pass16 = float((a.sum(axis=1) > 0).mean())        # standard observed-set
            # legacy VERL-style bootstrap proxy
            idx = rng.integers(0, 16, size=(len(rows), 1000, 16))
            boot = np.take_along_axis(a[:, None, :].repeat(1000, 1), idx, axis=2)
            proxy = float((boot.sum(axis=2) > 0).mean())
            tiers[f"t{t}"] = {"mean16": mean16, "pass16_standard": pass16,
                              "proxy16_bootstrap": proxy, "n_tasks": len(rows)}
        results[run] = {"tiers": tiers}
        json.dump({"run": run, "engine_seed": ENGINE_SEED, "tiers": tiers,
                   "per_task": per_task},
                  open(f"{OUT_DIR}/{run}_eval.json", "w"))
        print(run, json.dumps(tiers["t1"]), flush=True)

json.dump({"engine_seed": ENGINE_SEED, "results": results},
          open(f"{OUT_DIR}/gate_dr_eval_master.json", "w"), indent=1)
print("GATE-DR EVAL DONE")
