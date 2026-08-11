#!/usr/bin/env python3
"""Audit Countdown SFT/evaluation overlap and derive clean tier-0 metrics.

The paper's Countdown task identity is ``(target, sorted operand multiset)``.
This script reads JSON, JSONL, or Parquet exports, applies that identity, and
reports overlap separately for every evaluation tier.  If raw per-task binary
outcomes are supplied, it also recomputes mean@N and pass@N after excluding
SFT-exposed tier-0 tasks.

Examples
--------
Overlap only::

    python3 curriculum_maxrl/audit_countdown_sft_overlap.py \
      --sft /path/to/sft_examples.parquet \
      --eval /path/to/frozen_eval.parquet \
      --output /tmp/countdown_sft_overlap.json

Overlap plus clean tier-0 endpoints::

    python3 curriculum_maxrl/audit_countdown_sft_overlap.py \
      --sft /path/to/sft_examples.parquet \
      --eval /path/to/frozen_eval.parquet \
      --outcomes /path/to/per_task_step60_outcomes.jsonl \
      --output /tmp/countdown_sft_overlap.json

Outcome rows must identify the task (``numbers`` and ``target``), the arm and
seed, and contain binary ``successes`` or ``rewards``.  ``step`` is optional;
when present, only ``--outcome-step`` is retained (60 by default).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


TASK_IDENTITY = "(integer target, sorted integer operand multiset)"
TIER_RE = re.compile(r"countdown[ _-]*tier[ _-]*(\d+)", re.IGNORECASE)
NUMBERS_RE = re.compile(r"(?:numbers|multiset)\s*(?:are|:|=)?\s*\[([^]]+)\]", re.I)
TARGET_RE = re.compile(r"(?:target\s*(?:is|:|=)?|equals)\s*(-?\d+(?:\.\d+)?)", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonlike(value: Any) -> Any:
    """Decode JSON stored inside a string while leaving ordinary text alone."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    value = _jsonlike(value)
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk_mappings(child)


def _walk_strings(value: Any) -> Iterable[str]:
    value = _jsonlike(value)
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk_strings(child)


def _as_numbers(value: Any) -> tuple[int, ...] | None:
    value = _jsonlike(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        pieces = re.findall(r"-?\d+", value)
        value = pieces if pieces else None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        numbers = tuple(sorted(int(item) for item in value))
    except (TypeError, ValueError):
        return None
    return numbers if numbers else None


def _as_integer(value: Any) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def extract_task(row: Mapping[str, Any]) -> tuple[int, tuple[int, ...]]:
    """Extract the canonical task key from common VERL/JSON schemas."""
    number_names = {"numbers", "nums", "operands"}
    target_names = {"target", "goal", "answer"}
    for candidate in _walk_mappings(row):
        lowered = {str(key).lower(): value for key, value in candidate.items()}
        numbers = next(
            (_as_numbers(lowered[name]) for name in number_names if name in lowered),
            None,
        )
        target = next(
            (_as_integer(lowered[name]) for name in target_names if name in lowered),
            None,
        )
        if numbers is not None and target is not None:
            return target, numbers

    # Some exports retain only the rendered prompt.  Restrict fallback parsing
    # to prompt/problem fields so solution arithmetic is never mistaken for the
    # operand multiset.
    for candidate in _walk_mappings(row):
        for key, value in candidate.items():
            if not any(token in str(key).lower() for token in ("prompt", "problem", "question")):
                continue
            for prompt_text in _walk_strings(value):
                numbers_match = NUMBERS_RE.search(prompt_text)
                target_match = TARGET_RE.search(prompt_text)
                if numbers_match and target_match:
                    numbers = _as_numbers(numbers_match.group(1))
                    target = _as_integer(target_match.group(1))
                    if numbers is not None and target is not None:
                        return target, numbers
    raise ValueError("row has no parseable Countdown numbers/target")


def format_task_key(task: tuple[int, tuple[int, ...]]) -> str:
    target, numbers = task
    return f"{target}|{','.join(str(number) for number in numbers)}"


def extract_tier(row: Mapping[str, Any], task: tuple[int, tuple[int, ...]]) -> str:
    for candidate in _walk_mappings(row):
        for key, value in candidate.items():
            key_lower = str(key).lower()
            if key_lower not in {"tier", "data_source", "difficulty", "ability"}:
                continue
            if isinstance(value, str):
                match = TIER_RE.search(value)
                if match:
                    return f"countdown_tier{int(match.group(1))}"
                if key_lower == "tier" and value.strip().isdigit():
                    return f"countdown_tier{int(value)}"
            elif key_lower == "tier" and _as_integer(value) is not None:
                return f"countdown_tier{_as_integer(value)}"

    # The v2 pool defines tiers 0/1/2 as 2/3/4 operands.  Keeping this fallback
    # local and explicit makes prompt-only exports auditable too.
    operand_count = len(task[1])
    if 2 <= operand_count <= 4:
        return f"countdown_tier{operand_count - 2}"
    raise ValueError(f"cannot infer tier for {operand_count}-operand task")


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, Mapping):
            for field in ("rows", "records", "data", "examples", "outcomes"):
                if isinstance(payload.get(field), list):
                    rows = payload[field]
                    break
            else:
                raise ValueError(f"{path}: JSON object has no row-list field")
        else:
            raise ValueError(f"{path}: expected a JSON list or object")
    elif suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Parquet input requires pandas and pyarrow") from error
        rows = pd.read_parquet(path).to_dict(orient="records")
    else:
        raise ValueError(f"{path}: supported inputs are .json, .jsonl, .parquet, and .pq")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{path}: every row must be a JSON object")
    return [dict(row) for row in rows]


def input_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _parse_dataset(paths: Sequence[Path], require_tier: bool) -> dict[str, Any]:
    tasks: set[str] = set()
    tasks_by_tier: dict[str, set[str]] = defaultdict(set)
    row_count = 0
    unparseable: list[dict[str, Any]] = []
    for path in paths:
        for row_index, row in enumerate(load_rows(path), start=1):
            row_count += 1
            try:
                task = extract_task(row)
                task_key = format_task_key(task)
                tasks.add(task_key)
                if require_tier:
                    tasks_by_tier[extract_tier(row, task)].add(task_key)
            except ValueError as error:
                unparseable.append({"path": str(path), "row": row_index, "error": str(error)})
    if unparseable:
        preview = "; ".join(
            f"{item['path']}:{item['row']} ({item['error']})" for item in unparseable[:5]
        )
        raise ValueError(f"{len(unparseable)}/{row_count} rows are unparseable: {preview}")
    return {
        "rows": row_count,
        "unique_tasks": len(tasks),
        "duplicate_task_rows": row_count - len(tasks),
        "tasks": tasks,
        "tasks_by_tier": tasks_by_tier,
    }


def _successes(row: Mapping[str, Any]) -> list[int]:
    value: Any = None
    for candidate in _walk_mappings(row):
        lowered = {str(key).lower(): item for key, item in candidate.items()}
        for name in ("successes", "rewards", "correct", "outcomes"):
            if name in lowered:
                value = _jsonlike(lowered[name])
                break
        if value is not None:
            break
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (bool, int, float)):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("outcome row has no binary successes/rewards array")
    parsed: list[int] = []
    for item in value:
        numeric = _as_integer(item)
        if numeric not in (0, 1):
            raise ValueError(f"outcome must be binary, got {item!r}")
        parsed.append(numeric)
    if not parsed:
        raise ValueError("outcome array is empty")
    return parsed


def _field(row: Mapping[str, Any], names: set[str], default: Any = None) -> Any:
    for candidate in _walk_mappings(row):
        for key, value in candidate.items():
            if str(key).lower() in names:
                return value
    return default


def clean_tier0_reanalysis(
    paths: Sequence[Path],
    clean_tasks: set[str],
    outcome_step: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rows_read = 0
    rows_retained = 0
    for path in paths:
        for row_index, row in enumerate(load_rows(path), start=1):
            rows_read += 1
            step_value = _field(row, {"step", "global_step"})
            if step_value is not None and _as_integer(step_value) != outcome_step:
                continue
            try:
                task = extract_task(row)
                task_key = format_task_key(task)
                tier = extract_tier(row, task)
                if tier != "countdown_tier0" or task_key not in clean_tasks:
                    continue
                arm = str(_field(row, {"arm", "cell", "condition"}, "unknown"))
                seed = str(_field(row, {"seed", "run_seed"}, "unknown"))
                grouped[(arm, seed, outcome_step)][task_key].extend(_successes(row))
                rows_retained += 1
            except ValueError as error:
                raise ValueError(f"{path}:{row_index}: {error}") from error

    summaries = []
    for (arm, seed, step), per_task in sorted(grouped.items()):
        sample_counts = sorted({len(values) for values in per_task.values()})
        total_samples = sum(len(values) for values in per_task.values())
        total_successes = sum(sum(values) for values in per_task.values())
        tasks_with_success = sum(any(values) for values in per_task.values())
        summary: dict[str, Any] = {
            "arm": arm,
            "seed": seed,
            "step": step,
            "clean_tasks_expected": len(clean_tasks),
            "clean_tasks_observed": len(per_task),
            "missing_clean_task_keys": sorted(clean_tasks - set(per_task)),
            "samples_per_task_values": sample_counts,
            "total_samples": total_samples,
            "mean": total_successes / total_samples,
            "pass": tasks_with_success / len(per_task),
            "complete": len(per_task) == len(clean_tasks) and len(sample_counts) == 1,
        }
        if len(sample_counts) == 1:
            n = sample_counts[0]
            summary[f"mean@{n}"] = summary.pop("mean")
            summary[f"pass@{n}"] = summary.pop("pass")
        summaries.append(summary)

    return {
        "status": (
            "complete"
            if summaries and all(item["complete"] for item in summaries)
            else "incomplete"
        ),
        "outcome_step": outcome_step,
        "rows_read": rows_read,
        "rows_retained": rows_retained,
        "summaries": summaries,
    }


def audit(
    sft_paths: Sequence[Path],
    eval_paths: Sequence[Path],
    outcome_paths: Sequence[Path] = (),
    outcome_step: int = 60,
) -> dict[str, Any]:
    sft = _parse_dataset(sft_paths, require_tier=False)
    evaluation = _parse_dataset(eval_paths, require_tier=True)
    tiers: dict[str, Any] = {}
    for tier, eval_tasks in sorted(evaluation["tasks_by_tier"].items()):
        overlap = eval_tasks & sft["tasks"]
        clean = eval_tasks - sft["tasks"]
        tiers[tier] = {
            "eval_unique_tasks": len(eval_tasks),
            "sft_overlap_unique_tasks": len(overlap),
            "clean_unique_tasks": len(clean),
            "overlap_fraction": len(overlap) / len(eval_tasks),
            "overlap_task_keys": sorted(overlap),
            "clean_task_keys": sorted(clean),
        }

    expected_tiers = {f"countdown_tier{tier}" for tier in range(3)}
    missing_tiers = sorted(expected_tiers - set(tiers))
    if missing_tiers:
        raise ValueError(f"evaluation inputs are missing required tiers: {missing_tiers}")
    tier0 = tiers["countdown_tier0"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "task_identity": TASK_IDENTITY,
        "inputs": {
            "sft": [input_record(path) for path in sft_paths],
            "evaluation": [input_record(path) for path in eval_paths],
            "outcomes": [input_record(path) for path in outcome_paths],
        },
        "sft": {
            "rows": sft["rows"],
            "unique_tasks": sft["unique_tasks"],
            "duplicate_task_rows": sft["duplicate_task_rows"],
        },
        "evaluation": {
            "rows": evaluation["rows"],
            "unique_tasks": evaluation["unique_tasks"],
            "duplicate_task_rows": evaluation["duplicate_task_rows"],
        },
        "sft_evaluation_overlap": {"tiers": tiers},
    }
    if outcome_paths:
        report["clean_tier0_reanalysis"] = clean_tier0_reanalysis(
            outcome_paths,
            set(tier0["clean_task_keys"]),
            outcome_step,
        )
    else:
        report["clean_tier0_reanalysis"] = {
            "status": "blocked_missing_per_task_outcomes",
            "required": (
                "For each retained arm/seed at the final evaluation step: the frozen task "
                "identity and all 16 binary verifier outcomes. Aggregate tier metrics are "
                "insufficient to remove the exposed tasks."
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", action="append", required=True, type=Path)
    parser.add_argument("--eval", action="append", required=True, type=Path)
    parser.add_argument("--outcomes", action="append", default=[], type=Path)
    parser.add_argument("--outcome-step", default=60, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    for path in [*args.sft, *args.eval, *args.outcomes]:
        if not path.is_file():
            parser.error(f"input does not exist: {path}")

    report = audit(args.sft, args.eval, args.outcomes, args.outcome_step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tiers": report["sft_evaluation_overlap"]["tiers"],
                "clean_tier0_reanalysis": report["clean_tier0_reanalysis"]["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
