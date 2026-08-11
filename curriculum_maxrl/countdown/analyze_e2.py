"""Validate E2 treatment delivery and aggregate paired held-out endpoints."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ARMS = ("b1", "b2", "replay")
SEEDS = (1, 2, 3)
DATA_SHA256 = "95b1456fc3f49bc6f463614fef92900d748a07e4b429fd9383bbcf5edcb4e489"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def paired_summary(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result_dir = args.iteration_dir / "e2_results"
    records: dict[str, dict[str, dict]] = {}
    delivery: dict[str, dict] = {}

    for seed in SEEDS:
        seed_key = str(seed)
        records[seed_key] = {}
        for arm in ARMS:
            run_id = f"e2_clean_{arm}_s{seed}_260809"
            path = result_dir / f"{run_id}_eval.json"
            row = json.loads(path.read_text())
            assert row["data_sha256"] == DATA_SHA256, (path, row["data_sha256"])
            assert row["k"] == 16 and row["seed"] == 10000 + seed, path
            tiers = row["tiers"]
            assert set(tiers) == {
                "countdown_tier0", "countdown_tier1", "countdown_tier2"
            }, (path, tiers.keys())
            records[seed_key][arm] = {
                "evaluation_path": str(path.resolve()),
                "model": row["model"],
                "elapsed_seconds": row["elapsed_seconds"],
                "tiers": tiers,
            }

        checkpoint_root = args.runtime_root / "checkpoints"
        b2_path = (checkpoint_root / f"e2_clean_b2_s{seed}_260809" /
                   "dose_accounting.jsonl")
        replay_path = (checkpoint_root / f"e2_clean_replay_s{seed}_260809" /
                       "replay_accounting.jsonl")
        b2_rows = read_jsonl(b2_path)
        replay_rows = read_jsonl(replay_path)
        assert len(b2_rows) == args.steps and len(replay_rows) == args.steps
        assert [row["global_step"] for row in b2_rows] == list(
            range(1, args.steps + 1))
        assert [row["global_step"] for row in replay_rows] == list(
            range(1, args.steps + 1))

        displaced = 0
        scheduled = 0
        max_aux_mismatch = 0.0
        max_optimizer_mismatch = 0.0
        for b2, replay in zip(b2_rows, replay_rows):
            assert replay["replay/groups"] == b2["hindsight/relabeled_groups"]
            assert replay["replay/optimizer_rows_total"] == 128
            assert replay["replay/fallback_slots"] == 0
            aux_mismatch = replay["replay/cumulative_token_mismatch_fraction"]
            optimizer_mismatch = replay[
                "replay/cumulative_optimizer_token_mismatch_fraction"]
            assert aux_mismatch <= 0.05 and optimizer_mismatch <= 0.05
            expected_indices = [group["dataset_index"]
                                for group in b2["accepted_groups"]]
            replay_groups = replay["replay_groups"]
            actual_indices = [group["replaced_dataset_index"]
                              for group in replay_groups]
            assert actual_indices == expected_indices
            assert all(group["source_dataset_index"] !=
                       group["replaced_dataset_index"]
                       for group in replay_groups)
            displaced += replay["replay/displaced_live_slots"]
            scheduled += replay["replay/groups"]
            max_aux_mismatch = max(max_aux_mismatch, aux_mismatch)
            max_optimizer_mismatch = max(
                max_optimizer_mismatch, optimizer_mismatch)

        displacement_fraction = displaced / scheduled if scheduled else 0.0
        delivery[seed_key] = {
            "valid": displacement_fraction <= 0.25,
            "b2_audit": str(b2_path),
            "replay_audit": str(replay_path),
            "steps": args.steps,
            "scheduled_groups": scheduled,
            "displaced_live_slots": displaced,
            "displaced_live_fraction": displacement_fraction,
            "max_cumulative_aux_token_mismatch_fraction": max_aux_mismatch,
            "max_cumulative_optimizer_token_mismatch_fraction":
                max_optimizer_mismatch,
        }

    contrasts = {}
    for tier in ("countdown_tier0", "countdown_tier1", "countdown_tier2"):
        contrasts[tier] = {}
        for metric in ("mean@16", "pass@16"):
            contrasts[tier][metric] = {}
            for left, right in (("b2", "b1"), ("replay", "b1"),
                                ("replay", "b2")):
                values = [
                    records[str(seed)][left]["tiers"][tier][metric] -
                    records[str(seed)][right]["tiers"][tier][metric]
                    for seed in SEEDS
                ]
                contrasts[tier][metric][f"{left}_minus_{right}"] = (
                    paired_summary(values))

    output = {
        "protocol": "E2 dose-matched fixed-slot live-group replay",
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
