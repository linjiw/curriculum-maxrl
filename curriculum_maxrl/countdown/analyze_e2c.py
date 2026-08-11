"""Validate E2c delivery before permitting held-out endpoint evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from curriculum_maxrl.countdown.e2c_protocol import (
    FROZEN_DISPLACED_SLOT_LIMIT,
    FROZEN_MISMATCH_LIMIT,
    FROZEN_OPTIMIZER_ROWS,
    FROZEN_STEPS,
    require_frozen_scalar,
    require_frozen_seeds,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_seed(
    seed: int,
    audit_path: Path,
    schedule_path: Path,
    expected_steps: int,
    mismatch_limit: float,
    expected_rows: int,
) -> dict:
    audit = read_jsonl(audit_path)
    schedule = read_jsonl(schedule_path)
    expected_step_set = set(range(1, expected_steps + 1))
    audit_by_step = {int(row["global_step"]): row for row in audit}
    schedule_by_step = {int(row["global_step"]): row for row in schedule}
    if len(audit_by_step) != len(audit):
        raise ValueError(f"seed {seed}: duplicate E2c audit step")
    if len(schedule_by_step) != len(schedule):
        raise ValueError(f"seed {seed}: duplicate B2 schedule step")
    if set(audit_by_step) != expected_step_set:
        raise ValueError(f"seed {seed}: E2c audit is not {expected_steps} steps")
    if set(schedule_by_step) != expected_step_set:
        raise ValueError(f"seed {seed}: B2 schedule is not {expected_steps} steps")

    scheduled_groups = 0
    fallback_slots = 0
    displaced_slots = 0
    maximum_aux_mismatch = 0.0
    maximum_optimizer_mismatch = 0.0
    source_indices = set()
    cumulative_target_aux = 0
    cumulative_actual_aux = 0
    cumulative_target_optimizer = 0
    cumulative_actual_optimizer = 0
    for step in range(1, expected_steps + 1):
        row = audit_by_step[step]
        source = schedule_by_step[step]
        expected_groups = len(source.get("accepted_groups", []))
        groups = int(row["replay/groups"])
        if groups != expected_groups:
            raise ValueError(
                f"seed {seed} step {step}: replay groups {groups} != B2 "
                f"{expected_groups}")
        if int(row["replay/reservoir_sources_used"]) != groups:
            raise ValueError(
                f"seed {seed} step {step}: non-reservoir replay source used")
        if int(row["replay/buffer_sources_used"]) != 0:
            raise ValueError(
                f"seed {seed} step {step}: buffered replay source used")
        if int(row["replay/fallback_slots"]) != 0:
            raise ValueError(
                f"seed {seed} step {step}: replay slot fallback used")
        if int(row["replay/optimizer_rows_total"]) != expected_rows:
            raise ValueError(
                f"seed {seed} step {step}: optimizer row mismatch")
        aux_mismatch = float(
            row["replay/cumulative_token_mismatch_fraction"])
        optimizer_mismatch = float(
            row["replay/cumulative_optimizer_response_token_mismatch_fraction"])
        if aux_mismatch > mismatch_limit:
            raise ValueError(
                f"seed {seed} step {step}: auxiliary mismatch "
                f"{aux_mismatch:.4%}")
        if optimizer_mismatch > mismatch_limit:
            raise ValueError(
                f"seed {seed} step {step}: optimizer mismatch "
                f"{optimizer_mismatch:.4%}")
        records = row.get("replay_groups", [])
        if len(records) != groups:
            raise ValueError(
                f"seed {seed} step {step}: replay record count mismatch")
        expected_indices = [
            int(group["dataset_index"])
            for group in source.get("accepted_groups", [])]
        expected_aux_counts = [
            int(group["response_tokens"])
            for group in source.get("accepted_groups", [])]
        replaced_indices = [
            int(record["replaced_dataset_index"]) for record in records]
        target_aux_counts = [
            int(record["target_response_tokens"]) for record in records]
        actual_aux_counts = [
            int(record["replay_response_tokens"]) for record in records]
        if replaced_indices != expected_indices:
            raise ValueError(
                f"seed {seed} step {step}: replay did not use exact B2 slots")
        if target_aux_counts != expected_aux_counts:
            raise ValueError(
                f"seed {seed} step {step}: target token metadata drifted")
        for record in records:
            if record["source_kind"] != "reservoir":
                raise ValueError(
                    f"seed {seed} step {step}: source is not reservoir")
            if record["source_step"] is not None:
                raise ValueError(
                    f"seed {seed} step {step}: reservoir source has step")
            if record["source_age_steps"] is not None:
                raise ValueError(
                    f"seed {seed} step {step}: reservoir source has age")
            if (record["source_dataset_index"] ==
                    record["replaced_dataset_index"]):
                raise ValueError(
                    f"seed {seed} step {step}: self-source replay")
            source_indices.add(int(record["source_dataset_index"]))

        target_aux = sum(expected_aux_counts)
        actual_aux = sum(actual_aux_counts)
        if int(row["replay/target_aux_response_tokens"]) != target_aux:
            raise ValueError(
                f"seed {seed} step {step}: target auxiliary-token total drifted")
        if int(row["replay/aux_response_tokens"]) != actual_aux:
            raise ValueError(
                f"seed {seed} step {step}: replay auxiliary-token total drifted")
        cumulative_target_aux += target_aux
        cumulative_actual_aux += actual_aux
        aux_delta = cumulative_actual_aux - cumulative_target_aux
        recomputed_aux_fraction = (
            abs(aux_delta) / max(cumulative_target_aux, 1))
        expected_optimizer = int(
            source["hindsight/optimizer_response_tokens_total"])
        if int(row[
                "replay/target_optimizer_response_tokens_total"]) != \
                expected_optimizer:
            raise ValueError(
                f"seed {seed} step {step}: target optimizer-token total drifted")
        actual_optimizer = int(row["replay/optimizer_response_tokens_total"])
        cumulative_target_optimizer += expected_optimizer
        cumulative_actual_optimizer += actual_optimizer
        optimizer_delta = (
            cumulative_actual_optimizer - cumulative_target_optimizer)
        recomputed_optimizer_fraction = (
            abs(optimizer_delta) / max(cumulative_target_optimizer, 1))
        exact_integer_fields = {
            "replay/cumulative_target_aux_response_tokens":
                cumulative_target_aux,
            "replay/cumulative_aux_response_tokens": cumulative_actual_aux,
            "replay/cumulative_token_delta": aux_delta,
            "replay/cumulative_target_optimizer_response_tokens":
                cumulative_target_optimizer,
            "replay/cumulative_optimizer_response_tokens":
                cumulative_actual_optimizer,
            "replay/cumulative_optimizer_token_delta": optimizer_delta,
            "replay/optimizer_response_token_delta":
                actual_optimizer - expected_optimizer,
        }
        for field, expected_value in exact_integer_fields.items():
            if int(row[field]) != expected_value:
                raise ValueError(
                    f"seed {seed} step {step}: {field} is inconsistent")
        if not math.isclose(
                aux_mismatch, recomputed_aux_fraction,
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"seed {seed} step {step}: auxiliary mismatch is inconsistent")
        if not math.isclose(
                optimizer_mismatch, recomputed_optimizer_fraction,
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"seed {seed} step {step}: optimizer mismatch is inconsistent")
        displaced_from_records = sum(
            record.get("replaced_group_status") == "informative"
            for record in records)
        if int(row["replay/displaced_live_slots"]) != displaced_from_records:
            raise ValueError(
                f"seed {seed} step {step}: displaced-slot count is inconsistent")
        scheduled_groups += groups
        fallback_slots += int(row["replay/fallback_slots"])
        displaced_slots += int(row["replay/displaced_live_slots"])
        maximum_aux_mismatch = max(maximum_aux_mismatch, aux_mismatch)
        maximum_optimizer_mismatch = max(
            maximum_optimizer_mismatch, optimizer_mismatch)

    return {
        "seed": seed,
        "status": "pass",
        "steps": expected_steps,
        "scheduled_groups": scheduled_groups,
        "unique_reservoir_sources_used": len(source_indices),
        "fallback_slots": fallback_slots,
        "displaced_live_slots": displaced_slots,
        "maximum_cumulative_aux_token_mismatch_fraction":
            maximum_aux_mismatch,
        "maximum_cumulative_optimizer_token_mismatch_fraction":
            maximum_optimizer_mismatch,
        "audit": str(audit_path.resolve()),
        "schedule": str(schedule_path.resolve()),
    }


def summarize_displacement(seeds: list[dict]) -> dict:
    """Apply E2's frozen interpretation threshold without hiding endpoints."""
    per_seed = []
    total_groups = 0
    total_displaced = 0
    for item in seeds:
        groups = int(item["scheduled_groups"])
        displaced = int(item["displaced_live_slots"])
        fraction = displaced / groups if groups else 0.0
        per_seed.append({
            "seed": int(item["seed"]),
            "scheduled_groups": groups,
            "displaced_live_slots": displaced,
            "fraction": fraction,
            "within_frozen_25pct_threshold": (
                fraction <= FROZEN_DISPLACED_SLOT_LIMIT),
        })
        total_groups += groups
        total_displaced += displaced
    combined_fraction = (
        total_displaced / total_groups if total_groups else 0.0)
    within_threshold = combined_fraction <= FROZEN_DISPLACED_SLOT_LIMIT
    return {
        "frozen_threshold": FROZEN_DISPLACED_SLOT_LIMIT,
        "per_seed": per_seed,
        "scheduled_groups": total_groups,
        "displaced_live_slots": total_displaced,
        "combined_fraction": combined_fraction,
        "within_frozen_threshold": within_threshold,
        "interpretation": (
            "dose_matched_fixed_slot_control" if within_threshold else
            "fixed_slot_direction_substitution_only"),
        "note": (
            "Exceeding 25% does not expose or invalidate endpoints; it forbids "
            "a pure extra-dose interpretation, as frozen in E2."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="append", required=True)
    parser.add_argument("--schedule", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-steps", type=int, default=60)
    parser.add_argument("--mismatch-limit", type=float, default=0.05)
    parser.add_argument("--expected-rows", type=int, default=128)
    args = parser.parse_args()
    if not (len(args.audit) == len(args.schedule) == len(args.seed)):
        raise ValueError("provide one audit, schedule, and seed per arm")
    require_frozen_seeds(args.seed)
    require_frozen_scalar("steps", args.expected_steps, FROZEN_STEPS)
    require_frozen_scalar(
        "mismatch limit", args.mismatch_limit, FROZEN_MISMATCH_LIMIT)
    require_frozen_scalar(
        "optimizer rows", args.expected_rows, FROZEN_OPTIMIZER_ROWS)
    seeds = [validate_seed(
        seed,
        Path(audit),
        Path(schedule),
        args.expected_steps,
        args.mismatch_limit,
        args.expected_rows,
    ) for audit, schedule, seed in zip(
        args.audit, args.schedule, args.seed)]
    displacement = summarize_displacement(seeds)
    report = {
        "status": "pass",
        "endpoint_evaluation_permitted": True,
        "seeds": seeds,
        "displaced_slot_diagnostic": displacement,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
