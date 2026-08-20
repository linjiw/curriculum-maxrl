#!/usr/bin/env python3
"""Outcome-blind treatment-delivery replay for GROUP-LAW-FLIP v1.

The input is historical MAZE-SCORE ``un`` telemetry.  At each update, both
registered P0 scoring functionals inspect the same count-law sufficient
statistics.  We record the total-variation distance between their induced
level distributions, then feed both the same historical selected-level/count
pairs.  Evaluation records are neither accepted nor read.

This is a design calibration, not a counterfactual endpoint estimate: changing
the teacher would change later visits and model outcomes.  Its only purpose is
to choose/check a treatment-delivery threshold before P0 is frozen.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from curriculum_maxrl.count_law_stats import CountLawStats


N_LEVELS = 13
N_ROLLOUTS = 32
N_UPDATES = 250
TASKS_PER_UPDATE = 8
PRIOR_P0 = 0.5
PRIOR_MASS = 2.0 / N_ROLLOUTS
DECAY = 0.7
FLOOR = 0.15
_NAME_RE = re.compile(r"^mazescore_un_s(\d+)\.telemetry\.jsonl$")


class ReplayError(ValueError):
    """Raised when historical telemetry violates the replay contract."""


def _distribution(score: np.ndarray) -> np.ndarray:
    score = np.maximum(np.asarray(score, dtype=float), 0.0)
    if score.shape != (N_LEVELS,) or not np.all(np.isfinite(score)):
        raise ReplayError("score must be a finite 13-vector")
    if score.sum() <= 1e-12:
        score = np.ones(N_LEVELS, dtype=float)
    normalized = score / score.sum()
    return (1.0 - FLOOR) * normalized + FLOOR / N_LEVELS


def _load_records(path: Path) -> tuple[int, list[Mapping[str, Any]]]:
    match = _NAME_RE.fullmatch(path.name)
    if match is None:
        raise ReplayError(f"unexpected telemetry filename: {path.name}")
    seed = int(match.group(1))
    records: list[Mapping[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ReplayError(f"{path.name}:{line_number}: invalid JSON") from exc
        if not isinstance(record, dict) or record.get("record_type") != "telemetry":
            raise ReplayError(f"{path.name}:{line_number}: telemetry record required")
        if record.get("protocol") != "maze_score_v2":
            raise ReplayError(f"{path.name}:{line_number}: wrong protocol")
        records.append(record)
    updates = [record.get("completed_updates") for record in records]
    if updates != list(range(1, N_UPDATES + 1)):
        raise ReplayError(f"{path.name}: expected updates 1..{N_UPDATES}")
    return seed, records


def replay_path(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    seed, records = _load_records(input_path)
    bank = CountLawStats(
        N_LEVELS,
        N_ROLLOUTS,
        p0=PRIOR_P0,
        prior_mass=PRIOR_MASS,
        decay=DECAY,
    )
    tv_by_update: list[float] = []
    plugin_visit_expectation = np.zeros(N_LEVELS, dtype=float)
    grouplaw_visit_expectation = np.zeros(N_LEVELS, dtype=float)
    for record in records:
        plugin = _distribution(bank.plugin_activity("maxrl"))
        grouplaw = _distribution(bank.activity("maxrl"))
        tv_by_update.append(float(0.5 * np.abs(plugin - grouplaw).sum()))
        plugin_visit_expectation += plugin
        grouplaw_visit_expectation += grouplaw

        levels = record.get("selected_levels")
        counts = record.get("group_k")
        if not isinstance(levels, list) or not isinstance(counts, list):
            raise ReplayError(f"{input_path.name}: levels/counts must be lists")
        if len(levels) != TASKS_PER_UPDATE or len(counts) != TASKS_PER_UPDATE:
            raise ReplayError(
                f"{input_path.name}: expected {TASKS_PER_UPDATE} groups per update"
            )
        for level, count in zip(levels, counts, strict=True):
            if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level < N_LEVELS:
                raise ReplayError(f"{input_path.name}: invalid level {level!r}")
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= N_ROLLOUTS:
                raise ReplayError(f"{input_path.name}: invalid count {count!r}")
            bank.observe(level, count)

    values = np.asarray(tv_by_update, dtype=float)
    plugin_visit_expectation /= plugin_visit_expectation.sum()
    grouplaw_visit_expectation /= grouplaw_visit_expectation.sum()
    return {
        "seed": seed,
        "mean_tv": float(values.mean()),
        "median_tv": float(np.median(values)),
        "min_tv": float(values.min()),
        "max_tv": float(values.max()),
        "full_run_expected_visit_tv": float(
            0.5 * np.abs(plugin_visit_expectation - grouplaw_visit_expectation).sum()
        ),
    }


def replay_paths(paths: Sequence[str | Path]) -> dict[str, Any]:
    if not paths:
        raise ReplayError("at least one telemetry path is required")
    rows = [replay_path(path) for path in paths]
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ReplayError("duplicate seed telemetry")
    mean_tvs = np.asarray([row["mean_tv"] for row in rows], dtype=float)
    visit_tvs = np.asarray(
        [row["full_run_expected_visit_tv"] for row in rows], dtype=float
    )
    if not np.all(np.isfinite(mean_tvs)):
        raise ReplayError("non-finite replay result")
    return {
        "schema": "curriculum-maxrl/group-law-flip-delivery-replay/v1",
        "outcome_blind": True,
        "input_contract": "historical MAZE-SCORE un-arm telemetry only",
        "posterior": {
            "family": "count_law_moments",
            "sampling": "posterior_mean",
            "prior_p0": PRIOR_P0,
            "prior_mass_groups": PRIOR_MASS,
            "prior_mass_rollouts": PRIOR_MASS * N_ROLLOUTS,
            "decay": DECAY,
        },
        "teacher": {"floor": FLOOR, "n_rollouts": N_ROLLOUTS},
        "n_seed_blocks": len(rows),
        "seed_blocks": sorted(seeds),
        "mean_update_tv_across_blocks": float(mean_tvs.mean()),
        "min_block_mean_tv": float(mean_tvs.min()),
        "max_block_mean_tv": float(mean_tvs.max()),
        "mean_full_run_expected_visit_tv_across_blocks": float(visit_tvs.mean()),
        "min_block_full_run_expected_visit_tv": float(visit_tvs.min()),
        "max_block_full_run_expected_visit_tv": float(visit_tvs.max()),
        "blocks": {str(row["seed"]): row for row in sorted(rows, key=lambda x: x["seed"])},
        "interpretation": (
            "Design calibration only; replay does not estimate an endpoint because "
            "teacher-dependent visits and model states are not counterfactually replayed."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = replay_paths(args.inputs)
    output = Path(args.output)
    if output.exists():
        raise ReplayError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"replayed {result['n_seed_blocks']} outcome-blind blocks; "
        f"mean update TV={result['mean_update_tv_across_blocks']:.6f} -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
