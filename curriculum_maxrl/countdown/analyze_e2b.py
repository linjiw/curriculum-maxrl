"""Validate E2b recent-buffer delivery and paired held-out endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curriculum_maxrl.countdown.analyze_e2 import (
    DATA_SHA256,
    SEEDS,
    paired_summary,
    read_jsonl,
)


def run_id(arm: str, seed: int) -> str:
    if arm == "rb":
        return f"e2b_buffer_replay_s{seed}_260809"
    return f"e2_clean_{arm}_s{seed}_260809"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result_dir = args.iteration_dir / "e2_results"
    checkpoint_root = args.runtime_root / "checkpoints"
    records: dict[str, dict[str, dict]] = {}
    delivery: dict[str, dict] = {}

    for seed in SEEDS:
        seed_key = str(seed)
        records[seed_key] = {}
        for arm in ("b1", "b2", "rb"):
            identifier = run_id(arm, seed)
            path = result_dir / f"{identifier}_eval.json"
            row = json.loads(path.read_text())
            assert row["data_sha256"] == DATA_SHA256
            assert row["k"] == 16 and row["seed"] == 10000 + seed
            assert set(row["tiers"]) == {
                "countdown_tier0", "countdown_tier1", "countdown_tier2"
            }
            records[seed_key][arm] = {
                "evaluation_path": str(path.resolve()),
                "model": row["model"],
                "elapsed_seconds": row["elapsed_seconds"],
                "tiers": row["tiers"],
            }

        b2_path = (checkpoint_root / run_id("b2", seed) /
                   "dose_accounting.jsonl")
        rb_path = (checkpoint_root / run_id("rb", seed) /
                   "replay_accounting.jsonl")
        b2_rows = read_jsonl(b2_path)
        rb_rows = read_jsonl(rb_path)
        assert len(b2_rows) == args.steps and len(rb_rows) == args.steps
        expected_steps = list(range(1, args.steps + 1))
        assert [row["global_step"] for row in b2_rows] == expected_steps
        assert [row["global_step"] for row in rb_rows] == expected_steps

        scheduled = 0
        displaced = 0
        buffer_sources = 0
        max_source_age = 0
        max_buffer_groups = 0
        max_aux_mismatch = 0.0
        max_optimizer_mismatch = 0.0
        for b2, rb in zip(b2_rows, rb_rows):
            assert rb["replay/groups"] == b2["hindsight/relabeled_groups"]
            assert rb["replay/optimizer_rows_total"] == 128
            assert rb["replay/fallback_slots"] == 0
            aux_mismatch = rb["replay/cumulative_token_mismatch_fraction"]
            optimizer_mismatch = rb[
                "replay/cumulative_optimizer_token_mismatch_fraction"]
            assert aux_mismatch <= 0.05 and optimizer_mismatch <= 0.05
            expected_indices = [group["dataset_index"]
                                for group in b2["accepted_groups"]]
            groups = rb["replay_groups"]
            assert [group["replaced_dataset_index"]
                    for group in groups] == expected_indices
            for group in groups:
                assert group["source_dataset_index"] != group[
                    "replaced_dataset_index"]
                if group["source_kind"] == "current":
                    assert group["source_age_steps"] == 0
                else:
                    assert group["source_kind"] == "buffer"
                    assert 1 <= group["source_age_steps"] <= 8
            assert rb["replay/buffer_groups_after_step"] <= 64

            scheduled += rb["replay/groups"]
            displaced += rb["replay/displaced_live_slots"]
            buffer_sources += rb["replay/buffer_sources_used"]
            max_source_age = max(max_source_age,
                                 rb["replay/max_source_age_steps"])
            max_buffer_groups = max(max_buffer_groups,
                                    rb["replay/buffer_groups_after_step"])
            max_aux_mismatch = max(max_aux_mismatch, aux_mismatch)
            max_optimizer_mismatch = max(
                max_optimizer_mismatch, optimizer_mismatch)

        displacement_fraction = displaced / scheduled if scheduled else 0.0
        delivery[seed_key] = {
            "valid": displacement_fraction <= 0.25,
            "b2_audit": str(b2_path),
            "rb_audit": str(rb_path),
            "steps": args.steps,
            "scheduled_groups": scheduled,
            "buffer_sources_used": buffer_sources,
            "displaced_live_slots": displaced,
            "displaced_live_fraction": displacement_fraction,
            "max_source_age_steps": max_source_age,
            "max_buffer_groups": max_buffer_groups,
            "max_cumulative_aux_token_mismatch_fraction": max_aux_mismatch,
            "max_cumulative_optimizer_token_mismatch_fraction":
                max_optimizer_mismatch,
        }

    contrasts = {}
    for tier in ("countdown_tier0", "countdown_tier1", "countdown_tier2"):
        contrasts[tier] = {}
        for metric in ("mean@16", "pass@16"):
            contrasts[tier][metric] = {}
            for left, right in (("b2", "b1"), ("rb", "b1"),
                                ("rb", "b2")):
                values = [
                    records[str(seed)][left]["tiers"][tier][metric] -
                    records[str(seed)][right]["tiers"][tier][metric]
                    for seed in SEEDS
                ]
                contrasts[tier][metric][f"{left}_minus_{right}"] = (
                    paired_summary(values))

    output = {
        "protocol": "E2b recent-buffer dose-matched live replay",
        "seeds": list(SEEDS),
        "data_sha256": DATA_SHA256,
        "all_delivery_valid": all(row["valid"] for row in delivery.values()),
        "delivery": delivery,
        "per_seed": records,
        "paired_contrasts": contrasts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
