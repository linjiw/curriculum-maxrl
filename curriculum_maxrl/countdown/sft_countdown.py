"""Completion-only SFT warmstart for the rebuilt Countdown v2 split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CompletionDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.examples = []
        truncated = 0
        for row in self.rows:
            messages = row["messages"]
            prompt_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
            full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            common = 0
            for prompt_token, full_token in zip(prompt_ids, full_ids):
                if prompt_token != full_token:
                    break
                common += 1
            if common != len(prompt_ids):
                raise ValueError("chat template's generation prompt is not a prefix of the completed conversation")
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]
                truncated += 1
            labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
            if not any(label != -100 for label in labels):
                raise ValueError("SFT row has no assistant tokens after truncation")
            self.examples.append({"input_ids": full_ids, "labels": labels})
        self.truncated = truncated

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate(examples, pad_token_id: int):
    width = max(len(example["input_ids"]) for example in examples)
    input_ids, attention_mask, labels = [], [], []
    for example in examples:
        padding = width - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad_token_id] * padding)
        attention_mask.append([1] * len(example["input_ids"]) + [0] * padding)
        labels.append(example["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    data_path = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = CompletionDataset(data_path, tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda examples: collate(examples, tokenizer.pad_token_id),
        pin_memory=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    optimizer_steps = math.ceil(len(loader) * args.epochs / args.gradient_accumulation)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(optimizer_steps * args.warmup_ratio),
        num_training_steps=optimizer_steps,
    )

    history = []
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    running_loss = 0.0
    for epoch in range(args.epochs):
        for batch in loader:
            micro_step += 1
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            output = model(**batch, use_cache=False)
            loss = output.loss / args.gradient_accumulation
            loss.backward()
            running_loss += output.loss.detach().float().item()
            is_boundary = micro_step % args.gradient_accumulation == 0 or micro_step == len(loader) * args.epochs
            if is_boundary:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step = len(history) + 1
                row = {
                    "optimizer_step": step,
                    "micro_step": micro_step,
                    "loss": running_loss / (args.gradient_accumulation if micro_step % args.gradient_accumulation == 0 else 1),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "elapsed_seconds": time.time() - started,
                }
                history.append(row)
                running_loss = 0.0
                if step == 1 or step % 10 == 0 or step == optimizer_steps:
                    print(json.dumps(row, sort_keys=True), flush=True)

    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    metrics = {
        "base_model": args.model,
        "data": str(data_path),
        "data_sha256": sha256(data_path),
        "rows": len(dataset),
        "truncated_rows": dataset.truncated,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "optimizer_steps": len(history),
        "final_loss": history[-1]["loss"],
        "elapsed_seconds": time.time() - started,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20,
        "history": history,
    }
    with (output_dir / "training_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: value for key, value in metrics.items() if key != "history"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

