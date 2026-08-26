"""Fail-closed descriptive analyzer for E4 Countdown count-law calibration.

The analyzer consumes raw per-task binary outcomes. It performs no hypothesis
test and emits no verdict or decision field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SCHEMA = "curriculum-maxrl/llm-countlaw-calibration/v1"
EXPECTED_TIERS = {
    "countdown_tier0": 2,
    "countdown_tier1": 3,
    "countdown_tier2": 4,
}
EXPECTED_TASKS_PER_TIER = 128
EXPECTED_GROUP_SIZE = 16
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_903


class AnalysisError(ValueError):
    """Raised when the frozen raw-data contract is not satisfied."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_identity(row: dict) -> tuple[int, tuple[int, ...]]:
    truth = row.get("ground_truth")
    if not isinstance(truth, dict):
        raise AnalysisError("row has no ground_truth object")
    target = truth.get("target")
    numbers = truth.get("numbers")
    if not isinstance(target, int) or isinstance(target, bool):
        raise AnalysisError("ground_truth.target must be an integer")
    if not isinstance(numbers, list) or not numbers:
        raise AnalysisError("ground_truth.numbers must be a nonempty list")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in numbers):
        raise AnalysisError("ground_truth.numbers must contain integers")
    return target, tuple(sorted(numbers))


def _validate_vector(row: dict, name: str, n: int) -> list:
    value = row.get(name)
    if not isinstance(value, list) or len(value) != n:
        raise AnalysisError(f"{name} must be a length-{n} list")
    return value


def validate_rows(rows: list[dict]) -> dict[str, list[dict]]:
    expected_total = len(EXPECTED_TIERS) * EXPECTED_TASKS_PER_TIER
    if len(rows) != expected_total:
        raise AnalysisError(f"expected {expected_total} rows, found {len(rows)}")

    by_tier: dict[str, list[dict]] = defaultdict(list)
    identities: set[tuple[int, tuple[int, ...]]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise AnalysisError("every JSONL row must be an object")
        tier = row.get("data_source")
        if tier not in EXPECTED_TIERS:
            raise AnalysisError(f"unexpected data_source: {tier!r}")
        identity = _task_identity(row)
        if len(identity[1]) != EXPECTED_TIERS[tier]:
            raise AnalysisError(f"operand count disagrees with {tier}: {identity!r}")
        if identity in identities:
            raise AnalysisError(f"duplicate task identity: {identity!r}")
        identities.add(identity)

        rewards = _validate_vector(row, "rewards", EXPECTED_GROUP_SIZE)
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value not in (0, 1)
            for value in rewards
        ):
            raise AnalysisError("rewards must be binary integers")
        _validate_vector(row, "completions", EXPECTED_GROUP_SIZE)
        _validate_vector(row, "achieved_values", EXPECTED_GROUP_SIZE)
        _validate_vector(row, "new_tokens", EXPECTED_GROUP_SIZE)
        by_tier[tier].append(row)

    if set(by_tier) != set(EXPECTED_TIERS):
        raise AnalysisError("the three preregistered tiers are not complete")
    for tier, tier_rows in by_tier.items():
        if len(tier_rows) != EXPECTED_TASKS_PER_TIER:
            raise AnalysisError(
                f"{tier}: expected {EXPECTED_TASKS_PER_TIER} rows, found {len(tier_rows)}"
            )
        tier_rows.sort(key=_task_identity)
    return dict(by_tier)


def _metrics(rows: Iterable[dict]) -> dict[str, float | int | list[int]]:
    rows = list(rows)
    counts = [sum(row["rewards"]) for row in rows]
    tasks = len(counts)
    mean_pass_rate = sum(counts) / (tasks * EXPECTED_GROUP_SIZE)
    empirical_all_fail = sum(count == 0 for count in counts) / tasks
    plugin_all_fail = (1.0 - mean_pass_rate) ** EXPECTED_GROUP_SIZE
    gap = empirical_all_fail - plugin_all_fail
    count_law_activity = 2.0 * (1.0 - empirical_all_fail - mean_pass_rate)
    plugin_activity = 2.0 * (
        1.0 - mean_pass_rate - plugin_all_fail
    )
    histogram = Counter(counts)
    return {
        "tasks": tasks,
        "samples": tasks * EXPECTED_GROUP_SIZE,
        "mean_pass_rate": mean_pass_rate,
        "empirical_all_fail_probability": empirical_all_fail,
        "plugin_all_fail_probability": plugin_all_fail,
        "all_fail_gap": gap,
        "count_law_activity": count_law_activity,
        "plugin_activity": plugin_activity,
        "plugin_minus_count_law_activity": 2.0 * gap,
        "observed_pass_at_16": 1.0 - empirical_all_fail,
        "count_histogram_k_0_to_16": [histogram.get(k, 0) for k in range(17)],
    }


def _percentile(values: list[float], probability: float) -> float:
    """Type-7 linearly interpolated percentile, matching NumPy's default."""
    if not values:
        raise AnalysisError("cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_interval(rows: list[dict], seed: int) -> list[float]:
    rng = random.Random(seed)
    gaps = []
    n = len(rows)
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        gaps.append(float(_metrics(sample)["all_fail_gap"]))
    return [_percentile(gaps, 0.025), _percentile(gaps, 0.975)]


def analyze_rows(rows: list[dict]) -> dict:
    by_tier = validate_rows(rows)
    buckets = {}
    for offset, tier in enumerate(sorted(EXPECTED_TIERS)):
        metrics = _metrics(by_tier[tier])
        bucket_bootstrap_seed = BOOTSTRAP_SEED + offset
        metrics["all_fail_gap_task_bootstrap_95_interval"] = _bootstrap_interval(
            by_tier[tier], bucket_bootstrap_seed
        )
        metrics["bootstrap_seed"] = bucket_bootstrap_seed
        metrics["operand_count"] = EXPECTED_TIERS[tier]
        buckets[tier] = metrics
    return {
        "schema": SCHEMA,
        "evidence_tier": "Tier 2' controlled descriptive",
        "estimand": "Pr(K=0|z) - (1-p_bar_z)^16",
        "group_size": EXPECTED_GROUP_SIZE,
        "tasks_per_bucket": EXPECTED_TASKS_PER_TIER,
        "bootstrap": {
            "unit": "atomic task within operand-count bucket",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "95% percentile, type-7 linear interpolation",
        },
        "buckets": buckets,
        "inference": {
            "hypothesis_test": None,
            "p_value": None,
            "decision_rule": None,
            "verdict": None,
        },
    }


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise AnalysisError(f"blank JSONL line: {line_number}")
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"invalid JSONL line {line_number}: {exc}") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.raw.is_file() or args.raw.is_symlink():
        raise SystemExit(f"raw input must be a regular file: {args.raw}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite analysis: {args.output}")

    result = analyze_rows(read_jsonl(args.raw))
    result["raw_input"] = {
        "bytes": args.raw.stat().st_size,
        "sha256": sha256(args.raw),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
