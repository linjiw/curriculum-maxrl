"""Static provenance and token-support preflight for E2c.

This script intentionally reads no held-out model endpoints. It verifies the
frozen replay reservoir, the three B2 delivery schedules, and the exact
deterministic reservoir matcher before E2c training is allowed to start.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from curriculum_maxrl.countdown.e2c_protocol import (
    FROZEN_ASSET_SHA256,
    FROZEN_BASE_MODEL_REVISION,
    FROZEN_COLLECTOR_SEED,
    FROZEN_GROUP_SIZE,
    FROZEN_MAXIMUM_GROUPS,
    FROZEN_MAXRL_COMMIT,
    FROZEN_MINIMUM_GROUPS,
    FROZEN_MINIMUM_TOKEN_COUNTS,
    FROZEN_MISMATCH_LIMIT,
    FROZEN_STEPS,
    require_frozen_scalar,
    require_frozen_seeds,
)
from verl_integration.vendored.hindsight import DoseMatchedLiveReplay


def _normalize(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def _extra_index(row) -> int:
    extra = _normalize(row["extra_info"])
    if not isinstance(extra, dict) or "index" not in extra:
        raise ValueError("dataset row lacks extra_info.index")
    return int(extra["index"])


def _task_key_from_row(row) -> tuple[int, tuple[int, ...]]:
    reward_model = _normalize(row["reward_model"])
    truth = reward_model["ground_truth"]
    return int(truth["target"]), tuple(sorted(int(v) for v in truth["numbers"]))


def _task_key_from_group(group) -> tuple[int, tuple[int, ...]]:
    reward_rows = group["payload"]["non_tensor"].get("reward_model")
    if not reward_rows:
        raise ValueError(
            f"reservoir group {group['dataset_index']} lacks reward_model")
    truth = _normalize(reward_rows[0])["ground_truth"]
    return int(truth["target"]), tuple(sorted(int(v) for v in truth["numbers"]))


def _nested_keys(value):
    value = _normalize(value)
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _nested_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _nested_keys(item)


def load_reservoir(path: Path, expected_sha256: str | None = None) -> dict:
    digest = DoseMatchedLiveReplay._sha256(path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(
            f"reservoir SHA-256 {digest} != expected {expected_sha256}")
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")
    if artifact.get("format_version") != 1:
        raise ValueError("unsupported reservoir format")
    artifact["sha256"] = digest
    return artifact


def validate_reservoir(
    artifact: dict,
    train_path: Path,
    test_path: Path,
    minimum_groups: int,
    minimum_token_counts: int,
    expected_group_size: int,
) -> dict:
    import pandas as pd

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    train_by_index = {
        _extra_index(row): {
            "task": _task_key_from_row(row),
            "data_source": str(row["data_source"]),
        } for _, row in train.iterrows()}
    if len(train_by_index) != len(train):
        raise ValueError("train split has duplicate extra_info.index values")
    train_tasks = [item["task"] for item in train_by_index.values()]
    if len(set(train_tasks)) != len(train_tasks):
        raise ValueError("train split has duplicate task identities")
    test_keys = {_task_key_from_row(row) for _, row in test.iterrows()}
    if len(test_keys) != len(test):
        raise ValueError("held-out split has duplicate task identities")
    groups = artifact.get("groups", [])
    if len(groups) < minimum_groups:
        raise ValueError(
            f"reservoir has {len(groups)} groups, needs {minimum_groups}")
    indices = set()
    token_counts = set()
    for position, group in enumerate(groups):
        dataset_index = int(group["dataset_index"])
        if dataset_index in indices:
            raise ValueError(
                f"duplicate reservoir dataset index {dataset_index}")
        indices.add(dataset_index)
        if dataset_index not in train_by_index:
            raise ValueError(
                f"reservoir index {dataset_index} is not in train split")
        group_key = _task_key_from_group(group)
        if group_key != train_by_index[dataset_index]["task"]:
            raise ValueError(
                f"reservoir group {dataset_index} task metadata mismatch")
        if str(group.get("data_source")) != train_by_index[
                dataset_index]["data_source"]:
            raise ValueError(
                f"reservoir group {dataset_index} data-source mismatch")
        if group_key in test_keys:
            raise ValueError(
                f"reservoir group {dataset_index} overlaps held-out task")
        if group.get("status") != "informative":
            raise ValueError(f"reservoir group {position} is not informative")
        group_size = int(group["group_size"])
        if group_size != expected_group_size:
            raise ValueError(
                f"reservoir group {position} size {group_size} != "
                f"{expected_group_size}")
        rewards = group["payload"]["reward"].sum(dim=-1)
        payload = group["payload"]
        for key, tensor in payload.get("batch", {}).items():
            if not hasattr(tensor, "shape") or int(tensor.shape[0]) != group_size:
                raise ValueError(
                    f"reservoir group {position} batch field {key} has "
                    "the wrong leading dimension")
        if int(payload["reward"].shape[0]) != group_size:
            raise ValueError(
                f"reservoir group {position} reward has wrong row count")
        non_tensor = payload.get("non_tensor", {})
        banned_fragments = ("hindsight", "relabel", "achieved_target")
        contaminated_keys = sorted({
            key for key in _nested_keys(non_tensor)
            if any(fragment in key.lower() for fragment in banned_fragments)})
        if contaminated_keys:
            raise ValueError(
                f"reservoir group {position} contains relabel metadata: "
                f"{contaminated_keys}")
        for required_key in ("reward_model", "index", "data_source"):
            values = _normalize(non_tensor.get(required_key))
            if not isinstance(values, list) or len(values) != group_size:
                raise ValueError(
                    f"reservoir group {position} lacks {group_size} "
                    f"{required_key} rows")
        reward_rows = _normalize(non_tensor["reward_model"])
        if any(_task_key_from_row({"reward_model": item}) != group_key
               for item in reward_rows):
            raise ValueError(
                f"reservoir group {position} has mixed reward-model tasks")
        if any(int(value) != dataset_index
               for value in _normalize(non_tensor["index"])):
            raise ValueError(
                f"reservoir group {position} has mixed dataset indices")
        if any(str(value) != str(group["data_source"])
               for value in _normalize(non_tensor["data_source"])):
            raise ValueError(
                f"reservoir group {position} has mixed data sources")
        successes = int((rewards > 0).sum().item())
        if not 0 < successes < group_size:
            raise ValueError(
                f"reservoir group {position} reward payload is not informative")
        if successes != int(group["success_rollouts"]):
            raise ValueError(
                f"reservoir group {position} success count mismatch")
        payload_tokens = int(
            group["payload"]["batch"]["response_mask"].sum().item())
        if payload_tokens != int(group["tokens"]):
            raise ValueError(
                f"reservoir group {position} response-token mismatch")
        token_counts.add(payload_tokens)
    if len(token_counts) < minimum_token_counts:
        raise ValueError(
            f"reservoir has {len(token_counts)} distinct response-token "
            f"counts, needs {minimum_token_counts}")
    return {
        "groups": len(groups),
        "unique_dataset_indices": len(indices),
        "distinct_response_token_counts": len(token_counts),
        "response_token_min": min(token_counts),
        "response_token_max": max(token_counts),
        "train_rows": len(train),
        "test_rows": len(test),
        "task_overlap_with_test": 0,
    }


def validate_schedule_support(
    schedule_path: Path,
    reservoir_path: Path,
    reservoir_sha256: str,
    seed: int,
    expected_steps: int,
    mismatch_limit: float,
) -> dict:
    replay = DoseMatchedLiveReplay(
        str(schedule_path),
        seed=seed,
        max_cumulative_token_mismatch_fraction=mismatch_limit,
        strict=True,
        reservoir_path=str(reservoir_path),
        reservoir_sha256=reservoir_sha256,
    )
    expected_step_set = set(range(1, expected_steps + 1))
    if set(replay.schedule) != expected_step_set:
        missing = sorted(expected_step_set - set(replay.schedule))
        extra = sorted(set(replay.schedule) - expected_step_set)
        raise ValueError(
            f"schedule {schedule_path} steps mismatch; missing={missing}, "
            f"extra={extra}")

    maximum_aux_mismatch = 0.0
    maximum_optimizer_mismatch = 0.0
    scheduled_groups = 0
    for step in range(1, expected_steps + 1):
        row = replay.schedule[step]
        accepted = row.get("accepted_groups", [])
        counts = [int(group["response_tokens"]) for group in accepted]
        fallback_counts = [int(value) for value in
                           row.get("accepted_group_token_counts", [])]
        if not counts and fallback_counts:
            raise ValueError(
                f"schedule {schedule_path} step {step} lacks accepted-slot "
                "metadata")
        if len(counts) != len(fallback_counts):
            raise ValueError(
                f"schedule {schedule_path} step {step} accepted-group count "
                "mismatch")
        indices = [group.get("dataset_index") for group in accepted]
        if any(index is None for index in indices):
            raise ValueError(
                f"schedule {schedule_path} step {step} has missing slot index")
        if len(indices) != len(set(indices)):
            raise ValueError(
                f"schedule {schedule_path} step {step} has duplicate slots")
        if len(counts) > 8:
            raise ValueError(
                f"schedule {schedule_path} step {step} requests "
                f"{len(counts)} groups")
        expected_rows = int(row.get("hindsight/optimizer_rows_total", -1))
        if expected_rows != 128:
            raise ValueError(
                f"schedule {schedule_path} step {step} optimizer rows "
                f"{expected_rows} != 128")

        target_groups = [{
            "uid": f"target:{dataset_index}",
            "dataset_index": int(dataset_index),
            "tokens": token_count,
        } for dataset_index, token_count in zip(indices, counts)]
        target_step_tokens = sum(counts)
        expected_optimizer_tokens = int(
            row["hindsight/optimizer_response_tokens_total"])
        target_aux_after = (replay.cumulative_target_tokens +
                            target_step_tokens)
        target_optimizer_after = (
            replay.cumulative_target_optimizer_tokens +
            expected_optimizer_tokens)
        selected = replay._select_sources(
            replay.reservoir_groups,
            target_groups,
            target_aux_after,
            target_optimizer_after,
            expected_optimizer_tokens,
            step,
        )
        selected_tokens = sum(group["tokens"] for group in selected)
        replay.cumulative_target_tokens = target_aux_after
        replay.cumulative_replay_tokens += selected_tokens
        replay.cumulative_target_optimizer_tokens = target_optimizer_after
        # Conditional preflight assumes non-replaced E2c response-token dose
        # equals B2. Under that parity, auxiliary and optimizer deltas coincide.
        replay.cumulative_optimizer_tokens += (
            expected_optimizer_tokens - target_step_tokens + selected_tokens)
        aux_mismatch = abs(
            replay.cumulative_replay_tokens -
            replay.cumulative_target_tokens) / max(
                replay.cumulative_target_tokens, 1)
        optimizer_mismatch = abs(
            replay.cumulative_optimizer_tokens -
            replay.cumulative_target_optimizer_tokens) / max(
                replay.cumulative_target_optimizer_tokens, 1)
        maximum_aux_mismatch = max(maximum_aux_mismatch, aux_mismatch)
        maximum_optimizer_mismatch = max(
            maximum_optimizer_mismatch, optimizer_mismatch)
        if aux_mismatch > mismatch_limit:
            raise ValueError(
                f"schedule {schedule_path} step {step} reservoir auxiliary "
                f"support mismatch {aux_mismatch:.4%} > {mismatch_limit:.4%}")
        if optimizer_mismatch > mismatch_limit:
            raise ValueError(
                f"schedule {schedule_path} step {step} reservoir optimizer "
                f"support mismatch {optimizer_mismatch:.4%} > "
                f"{mismatch_limit:.4%}")
        scheduled_groups += len(counts)

    return {
        "schedule": str(schedule_path.resolve()),
        "seed": seed,
        "steps": expected_steps,
        "scheduled_groups": scheduled_groups,
        "maximum_conditional_aux_token_mismatch_fraction":
            maximum_aux_mismatch,
        "maximum_conditional_optimizer_token_mismatch_fraction":
            maximum_optimizer_mismatch,
        "conditional_on_non_replaced_token_parity": True,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reservoir", required=True)
    parser.add_argument("--reservoir-sha256")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--maxrl-commit", required=True)
    parser.add_argument("--schedule", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-steps", type=int, default=60)
    parser.add_argument("--mismatch-limit", type=float, default=0.05)
    parser.add_argument("--minimum-groups", type=int, default=128)
    parser.add_argument("--minimum-token-counts", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=16)
    args = parser.parse_args()
    if len(args.schedule) != len(args.seed):
        raise ValueError("provide one --seed for each --schedule")
    require_frozen_seeds(args.seed)
    require_frozen_scalar("steps", args.expected_steps, FROZEN_STEPS)
    require_frozen_scalar(
        "mismatch limit", args.mismatch_limit, FROZEN_MISMATCH_LIMIT)
    require_frozen_scalar(
        "minimum reservoir groups", args.minimum_groups,
        FROZEN_MINIMUM_GROUPS)
    require_frozen_scalar(
        "minimum distinct token counts", args.minimum_token_counts,
        FROZEN_MINIMUM_TOKEN_COUNTS)
    require_frozen_scalar("group size", args.group_size, FROZEN_GROUP_SIZE)
    require_frozen_scalar(
        "MaxRL commit", args.maxrl_commit, FROZEN_MAXRL_COMMIT)

    reservoir_path = Path(args.reservoir).resolve()
    artifact = load_reservoir(
        reservoir_path, expected_sha256=args.reservoir_sha256)
    if int(artifact.get("collector_seed", -1)) != FROZEN_COLLECTOR_SEED:
        raise ValueError(
            "reservoir collector seed is not the frozen "
            f"{FROZEN_COLLECTOR_SEED}")
    if int(artifact.get("max_groups", -1)) != FROZEN_MAXIMUM_GROUPS:
        raise ValueError(
            f"reservoir max_groups is not the frozen {FROZEN_MAXIMUM_GROUPS}")
    if artifact.get("steps_seen") != list(range(1, FROZEN_STEPS + 1)):
        raise ValueError("reservoir was not collected over frozen steps 1..60")
    reservoir_report = validate_reservoir(
        artifact,
        Path(args.train_data).resolve(),
        Path(args.test_data).resolve(),
        args.minimum_groups,
        args.minimum_token_counts,
        args.group_size,
    )
    schedules = [validate_schedule_support(
        Path(schedule).resolve(),
        reservoir_path,
        artifact["sha256"],
        seed,
        args.expected_steps,
        args.mismatch_limit,
    ) for schedule, seed in zip(args.schedule, args.seed)]
    train_path = Path(args.train_data).resolve()
    test_path = Path(args.test_data).resolve()
    for path, label in ((train_path, "train.parquet"),
                        (test_path, "test.parquet")):
        digest = DoseMatchedLiveReplay._sha256(path)
        if digest != FROZEN_ASSET_SHA256[label]:
            raise ValueError(
                f"frozen {label} SHA-256 {digest} != "
                f"{FROZEN_ASSET_SHA256[label]}")

    model_path = Path(args.model).resolve()
    model_files = {}
    for name in ("config.json", "model.safetensors", "tokenizer.json",
                 "training_metrics.json"):
        path = model_path / name
        if not path.is_file():
            raise ValueError(f"frozen model file is missing: {path}")
        digest = DoseMatchedLiveReplay._sha256(path)
        if digest != FROZEN_ASSET_SHA256[name]:
            raise ValueError(
                f"frozen model {name} SHA-256 {digest} != "
                f"{FROZEN_ASSET_SHA256[name]}")
        model_files[name] = {
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    with (model_path / "training_metrics.json").open(
            encoding="utf-8") as handle:
        training_metrics = json.load(handle)
    base_model_path = str(training_metrics.get("base_model", ""))
    base_model_revision = Path(base_model_path).name
    if base_model_revision != FROZEN_BASE_MODEL_REVISION:
        raise ValueError(
            f"base model revision {base_model_revision!r} != frozen "
            f"{FROZEN_BASE_MODEL_REVISION!r}")
    report = {
        "status": "pass",
        "reservoir": str(reservoir_path),
        "reservoir_sha256": artifact["sha256"],
        "train_data": str(Path(args.train_data).resolve()),
        "train_data_sha256": FROZEN_ASSET_SHA256["train.parquet"],
        "test_data": str(Path(args.test_data).resolve()),
        "test_data_sha256": FROZEN_ASSET_SHA256["test.parquet"],
        "frozen_model": str(model_path),
        "frozen_model_files": model_files,
        "base_model_revision": base_model_revision,
        "maxrl_commit": args.maxrl_commit,
        "collection_protocol": {
            "seed": artifact["collector_seed"],
            "generation_steps": len(artifact["steps_seen"]),
            "prompt_groups_per_step": 8,
            "rollouts_per_group": artifact["expected_group_size"],
            "maximum_retained_groups": artifact["max_groups"],
            "learning_rate": 0,
        },
        "reservoir_validation": reservoir_report,
        "schedule_validation": schedules,
        "runtime_gates_still_required": [
            "exact accepted slots and group counts",
            "zero fallback",
            "reservoir-only non-self sources",
            "cumulative auxiliary-token mismatch <= 5%",
            "cumulative total optimizer-token mismatch <= 5%",
            "128 optimizer rows and matched optimizer/LR schedules",
        ],
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
