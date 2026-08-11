"""GPU pass@k and relabel-yield probe for Countdown checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from curriculum_maxrl.countdown.countdown_reward import achieved_value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_object(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: normalize_object(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_object(item) for item in value]
    return value


def summarize(rows: list[dict], k: int) -> dict:
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_tier[row["data_source"]].append(row)
    out = {}
    for tier, tier_rows in sorted(by_tier.items()):
        rewards = [reward for row in tier_rows for reward in row["rewards"]]
        relabelable = [flag for row in tier_rows for flag in row["relabelable_failures"]]
        failures = sum(1 - reward for reward in rewards)
        out[tier] = {
            "tasks": len(tier_rows),
            "samples": len(rewards),
            f"mean@{k}": sum(rewards) / len(rewards),
            f"pass@{k}": sum(any(row["rewards"]) for row in tier_rows) / len(tier_rows),
            "relabelable_failures": sum(relabelable),
            "failures": failures,
            "relabel_yield_on_failure": sum(relabelable) / failures if failures else 0.0,
            "mean_new_tokens": sum(length for row in tier_rows for length in row["new_tokens"]) / len(rewards),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8, help="prompts per generation call")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    data_path = Path(args.data).resolve()
    output_path = Path(args.output).resolve()
    model_path = Path(args.model).resolve()
    evaluator_path = Path(__file__).resolve()
    reward_path = evaluator_path.with_name("countdown_reward.py")
    model_files = {}
    for name in ("config.json", "model.safetensors"):
        path = model_path / name
        if not path.is_file():
            raise ValueError(f"evaluated model artifact is missing: {path}")
        model_files[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    frame = pd.read_parquet(data_path)
    if args.limit:
        frame = frame.head(args.limit)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda").eval()
    # Pair generation RNG after model loading so checkpoint-load internals
    # cannot consume a different prefix of the nominally shared seed stream.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows: list[dict] = []
    started = time.time()
    for start in range(0, len(frame), args.batch_size):
        batch = frame.iloc[start : start + args.batch_size]
        messages = [normalize_object(prompt) for prompt in batch["prompt"]]
        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.k,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuation_ids = generated[:, encoded.input_ids.shape[1] :].cpu()
        completions = tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)
        lengths = (continuation_ids != tokenizer.pad_token_id).sum(dim=1).tolist()

        for offset, (_, source_row) in enumerate(batch.iterrows()):
            ground_truth = normalize_object(source_row["reward_model"])["ground_truth"]
            task_completions = completions[offset * args.k : (offset + 1) * args.k]
            task_lengths = lengths[offset * args.k : (offset + 1) * args.k]
            values = [achieved_value(text, ground_truth["numbers"]) for text in task_completions]
            rewards = [int(value == int(ground_truth["target"])) for value in values]
            rows.append({
                "data_source": source_row["data_source"],
                "ground_truth": ground_truth,
                "completions": task_completions,
                "achieved_values": values,
                "rewards": rewards,
                "relabelable_failures": [int(value is not None and not reward) for value, reward in zip(values, rewards)],
                "new_tokens": task_lengths,
            })
        print(f"probe {min(start + len(batch), len(frame))}/{len(frame)} elapsed={time.time() - started:.1f}s", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_suffix(".jsonl")
    with raw_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "model": str(model_path),
        "model_files": model_files,
        "evaluator": str(evaluator_path),
        "evaluator_sha256": sha256(evaluator_path),
        "reward_implementation": str(reward_path),
        "reward_sha256": sha256(reward_path),
        "data": str(data_path),
        "data_sha256": sha256(data_path),
        "raw_outcomes": str(raw_path),
        "k": args.k,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20,
        "tiers": summarize(rows, args.k),
    }
    with output_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
