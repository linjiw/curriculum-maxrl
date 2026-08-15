#!/usr/bin/env python3
"""Analyze the frozen MAZE-SCORE campaign only after it is complete.

Canonical input is one ``mazescore_{arm}_s{seed}.jsonl`` file for every
``arm in {un, learn, unif}`` and seed block 20--49.  Each file contains:

* exactly one ``record_type="config"`` record;
* exactly one post-SFT evaluation at ``completed_updates=0``; and
* exactly one RL evaluation at each completed-update count 25, 50, ..., 250.

The deliberately strict loader is an outcome firewall: it validates the full
30-block/90-cell campaign, source/config identity, paired SFT checkpoints, and every
timepoint before calculating or emitting any endpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL = "maze_score_v2"
ARMS = ("un", "learn", "unif")
EXPECTED_SEEDS = tuple(range(20, 50))
EXPECTED_UPDATES = tuple(range(25, 251, 25))
EXPECTED_LEVELS = tuple(str(level) for level in range(13))
EXPECTED_ROLLOUTS = 32
EXPECTED_STEPS = 250
EXPECTED_EVAL_EVERY = 25
EXPECTED_EVAL_TASKS_PER_LEVEL = 32
EXPECTED_TASKS_PER_STEP = 8
EXPECTED_SFT_STEPS = 600
EXPECTED_EVAL_SAMPLES = 8
EXPECTED_LR = 1e-4
EXPECTED_D_MODEL = 128
EXPECTED_N_LAYERS = 6
EXPECTED_TEACHER_POWER = 1.0
BOOTSTRAP_RESAMPLES = 10_000
RANDOM_SEED = 20_260_813
SESOI = 0.005
MAX_EXACT_SIGN_FLIP_N = 40

_FILENAME_RE = re.compile(r"^mazescore_(un|learn|unif)_s(\d+)\.jsonl$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AnalysisError(ValueError):
    """Raised when an input violates the frozen analysis contract."""


@dataclass(frozen=True)
class Run:
    """A validated run, before campaign-level cross-cell checks."""

    path: Path
    config: Mapping[str, Any]
    baseline_cov8: float
    update_cov8: Mapping[int, float]

    @property
    def arm(self) -> str:
        return str(self.config["arm"])

    @property
    def seed(self) -> int:
        return int(self.config["seed"])

    @property
    def cov_auc_delta(self) -> float:
        values = [self.update_cov8[update] for update in EXPECTED_UPDATES]
        return float(np.mean(values) - self.baseline_cov8)


def _reject_json_constant(value: str) -> None:
    raise AnalysisError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnalysisError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json_line(line: str, path: Path, line_number: int) -> Mapping[str, Any]:
    try:
        value = json.loads(
            line,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, AnalysisError) as exc:
        raise AnalysisError(f"{path.name}:{line_number}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{path.name}:{line_number}: record must be a JSON object")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(config: Mapping[str, Any], key: str, expected: int | None = None) -> int:
    value = config.get(key)
    if not _is_int(value):
        raise AnalysisError(f"config field {key!r} must be an integer")
    if expected is not None and value != expected:
        raise AnalysisError(f"config field {key!r} must equal {expected}, got {value}")
    return int(value)


def _require_nonempty(config: Mapping[str, Any], key: str) -> Any:
    if key not in config:
        raise AnalysisError(f"missing config field {key!r}")
    value = config[key]
    if value is None or value == "" or value == {} or value == []:
        raise AnalysisError(f"config field {key!r} must be nonempty")
    return value


def _require_number(
    config: Mapping[str, Any], key: str, expected: float
) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"config field {key!r} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric != expected:
        raise AnalysisError(
            f"config field {key!r} must equal {expected}, got {value}"
        )
    return numeric


def _require_false(config: Mapping[str, Any], key: str) -> None:
    if config.get(key) is not False:
        raise AnalysisError(f"config field {key!r} must be false")


def _validate_config(config: Mapping[str, Any], path: Path) -> None:
    if config.get("record_type") != "config":
        raise AnalysisError(f"{path.name}: malformed config record")
    if config.get("protocol") != PROTOCOL:
        raise AnalysisError(
            f"{path.name}: protocol must be {PROTOCOL!r}, got {config.get('protocol')!r}"
        )
    if config.get("estimator") != "maxrl":
        raise AnalysisError(f"{path.name}: estimator must be 'maxrl'")

    arm = config.get("arm")
    if arm not in ARMS:
        raise AnalysisError(f"{path.name}: arm must be one of {ARMS}, got {arm!r}")
    seed = _require_int(config, "seed")
    if seed not in EXPECTED_SEEDS:
        raise AnalysisError(f"{path.name}: seed {seed} is outside frozen blocks 20--49")

    seeds = config.get("seeds")
    if not isinstance(seeds, dict):
        raise AnalysisError(f"{path.name}: seeds must be an object")
    expected_seeds = {
        "base": seed,
        "sft": seed,
        "rl": seed,
        "teacher": seed + 77,
        "eval_tasks": 202_608_130 + seed,
        "eval_samples": 302_608_130 + seed,
    }
    for key, expected in expected_seeds.items():
        value = seeds.get(key)
        if not _is_int(value) or value != expected:
            raise AnalysisError(
                f"{path.name}: seeds.{key} must equal {expected}, got {value!r}"
            )

    match = _FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise AnalysisError(
            f"{path.name}: filename must match mazescore_{{arm}}_s{{seed}}.jsonl"
        )
    if match.group(1) != arm or int(match.group(2)) != seed:
        raise AnalysisError(f"{path.name}: filename and config arm/seed disagree")

    rollouts = _require_int(config, "rollouts", EXPECTED_ROLLOUTS)
    _require_int(config, "steps", EXPECTED_STEPS)
    _require_int(config, "eval_every", EXPECTED_EVAL_EVERY)
    _require_int(
        config, "eval_tasks_per_level", EXPECTED_EVAL_TASKS_PER_LEVEL
    )
    _require_int(config, "tasks_per_step", EXPECTED_TASKS_PER_STEP)
    _require_int(config, "sft_steps", EXPECTED_SFT_STEPS)
    _require_int(config, "eval_samples", EXPECTED_EVAL_SAMPLES)
    _require_int(config, "planned_rl_eval_count", len(EXPECTED_UPDATES))
    _require_int(config, "d_model", EXPECTED_D_MODEL)
    _require_int(config, "n_layers", EXPECTED_N_LAYERS)
    _require_number(config, "lr", EXPECTED_LR)
    _require_number(config, "teacher_power", EXPECTED_TEACHER_POWER)
    for key in ("hindsight", "hindsight_dense", "hindsight_to_teacher"):
        _require_false(config, key)
    campaign = _require_nonempty(config, "campaign")
    if not isinstance(campaign, str):
        raise AnalysisError(f"{path.name}: campaign must be a nonempty string")
    _require_nonempty(config, "source_manifest")

    checkpoint_hash = _require_nonempty(config, "sft_checkpoint_sha256")
    if not isinstance(checkpoint_hash, str) or not _SHA256_RE.fullmatch(checkpoint_hash):
        raise AnalysisError(
            f"{path.name}: sft_checkpoint_sha256 must be 64 lowercase hex characters"
        )

    exponent = config.get("effective_exponent")
    if arm == "un":
        if config.get("teacher") not in {"frontier_un", "coefficient_activity"}:
            raise AnalysisError(
                f"{path.name}: un teacher must select exact coefficient activity"
            )
        if config.get("score_family") != "coefficient_activity":
            raise AnalysisError(
                f"{path.name}: un score_family must be 'coefficient_activity'"
            )
        if not _is_int(exponent) or exponent != rollouts:
            raise AnalysisError(
                f"{path.name}: un effective_exponent must equal rollouts ({rollouts})"
            )
    elif arm == "learn":
        if config.get("teacher") != "learnability":
            raise AnalysisError(f"{path.name}: learn teacher must be 'learnability'")
        if config.get("score_family") != "coefficient_activity":
            raise AnalysisError(
                f"{path.name}: learn score_family must be 'coefficient_activity'"
            )
        if not _is_int(exponent) or exponent != 2:
            raise AnalysisError(f"{path.name}: learn effective_exponent must equal 2")
    else:
        if config.get("teacher") != "uniform":
            raise AnalysisError(f"{path.name}: unif teacher must be 'uniform'")
        if config.get("score_family") != "uniform":
            raise AnalysisError(f"{path.name}: uniform score_family must be 'uniform'")
        if exponent is not None:
            raise AnalysisError(f"{path.name}: uniform effective_exponent must be null")


def _mean_pass8(record: Mapping[str, Any], path: Path, line_number: int) -> float:
    passk = record.get("passk")
    if not isinstance(passk, dict):
        raise AnalysisError(f"{path.name}:{line_number}: passk must be an object")
    if set(passk) != set(EXPECTED_LEVELS):
        missing = sorted(set(EXPECTED_LEVELS) - set(passk), key=int)
        extra = sorted(set(passk) - set(EXPECTED_LEVELS))
        raise AnalysisError(
            f"{path.name}:{line_number}: passk levels mismatch; "
            f"missing={missing}, extra={extra}"
        )

    values: list[float] = []
    for level in EXPECTED_LEVELS:
        per_k = passk[level]
        if not isinstance(per_k, dict) or "8" not in per_k:
            raise AnalysisError(
                f"{path.name}:{line_number}: level {level} lacks pass@8"
            )
        value = per_k["8"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AnalysisError(
                f"{path.name}:{line_number}: level {level} pass@8 is not numeric"
            )
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise AnalysisError(
                f"{path.name}:{line_number}: level {level} pass@8 is outside [0, 1]"
            )
        values.append(numeric)
    return float(np.mean(values))


def load_run(path: str | os.PathLike[str]) -> Run:
    """Load and fully validate one canonical MAZE-SCORE JSONL file."""

    run_path = Path(path)
    if not run_path.is_file():
        raise AnalysisError(f"input is not a file: {run_path}")

    records: list[tuple[int, Mapping[str, Any]]] = []
    with run_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if raw.strip():
                records.append(
                    (line_number, _load_json_line(raw, run_path, line_number))
                )
    if not records:
        raise AnalysisError(f"{run_path.name}: empty input")

    config_records = [
        (line, rec)
        for line, rec in records
        if rec.get("record_type") == "config"
    ]
    if len(config_records) != 1:
        raise AnalysisError(
            f"{run_path.name}: expected exactly one config record, found {len(config_records)}"
        )
    config = config_records[0][1]
    _validate_config(config, run_path)

    evaluations: list[tuple[int, Mapping[str, Any]]] = []
    for line_number, record in records:
        record_type = record.get("record_type")
        if record_type == "evaluation":
            if record.get("protocol") != PROTOCOL:
                raise AnalysisError(
                    f"{run_path.name}:{line_number}: evaluation protocol must be {PROTOCOL!r}"
                )
            evaluations.append((line_number, record))
        elif record_type != "config":
            raise AnalysisError(
                f"{run_path.name}:{line_number}: unknown record_type {record_type!r}"
            )

    baselines = [
        (line, rec)
        for line, rec in evaluations
        if rec.get("phase") == "post_sft"
    ]
    if len(baselines) != 1:
        raise AnalysisError(
            f"{run_path.name}: expected exactly one post-SFT baseline, found {len(baselines)}"
        )
    baseline_line, baseline = baselines[0]
    if baseline.get("completed_updates") != 0:
        raise AnalysisError(
            f"{run_path.name}:{baseline_line}: post-SFT baseline must have completed_updates=0"
        )
    if baseline.get("final", False) is not False:
        raise AnalysisError(f"{run_path.name}:{baseline_line}: baseline cannot be final")
    baseline_cov8 = _mean_pass8(baseline, run_path, baseline_line)

    update_cov8: dict[int, float] = {}
    for line_number, record in evaluations:
        if record is baseline:
            continue
        if record.get("phase") != "rl":
            raise AnalysisError(
                f"{run_path.name}:{line_number}: non-baseline evaluation phase must be 'rl'"
            )
        update = record.get("completed_updates")
        if not _is_int(update):
            raise AnalysisError(
                f"{run_path.name}:{line_number}: completed_updates must be an integer"
            )
        if update in update_cov8:
            raise AnalysisError(
                f"{run_path.name}: duplicate RL evaluation at completed_updates={update}"
            )
        final = record.get("final", False)
        if not isinstance(final, bool):
            raise AnalysisError(
                f"{run_path.name}:{line_number}: final must be a boolean when present"
            )
        if update == EXPECTED_STEPS and final is not True:
            raise AnalysisError(
                f"{run_path.name}:{line_number}: completed_updates=250 must carry final=true"
            )
        if update != EXPECTED_STEPS and final is not False:
            raise AnalysisError(
                f"{run_path.name}:{line_number}: only completed_updates=250 may be final"
            )
        update_cov8[int(update)] = _mean_pass8(record, run_path, line_number)

    seen_updates = set(update_cov8)
    expected_updates = set(EXPECTED_UPDATES)
    if seen_updates != expected_updates:
        missing = sorted(expected_updates - seen_updates)
        extra = sorted(seen_updates - expected_updates)
        raise AnalysisError(
            f"{run_path.name}: RL timepoints mismatch; missing={missing}, extra={extra}"
        )

    return Run(
        path=run_path,
        config=dict(config),
        baseline_cov8=baseline_cov8,
        update_cov8=update_cov8,
    )


def _manifest_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"source_manifest is not canonical JSON: {exc}") from exc


def _validate_campaign(runs: Iterable[Run]) -> dict[tuple[int, str], Run]:
    cells: dict[tuple[int, str], Run] = {}
    campaign: str | None = None
    manifest_key: str | None = None
    shared_config: tuple[int, int, int, int] | None = None

    for run in runs:
        key = (run.seed, run.arm)
        if key in cells:
            raise AnalysisError(f"duplicate campaign cell arm={run.arm}, seed={run.seed}")
        cells[key] = run

        run_campaign = str(run.config["campaign"])
        run_manifest_key = _manifest_key(run.config["source_manifest"])
        run_shared_config = (
            int(run.config["rollouts"]),
            int(run.config["steps"]),
            int(run.config["eval_every"]),
            int(run.config["eval_tasks_per_level"]),
        )
        if campaign is None:
            campaign = run_campaign
            manifest_key = run_manifest_key
            shared_config = run_shared_config
        elif run_campaign != campaign:
            raise AnalysisError("campaign mismatch across input files")
        elif run_manifest_key != manifest_key:
            raise AnalysisError("source_manifest mismatch across input files")
        elif run_shared_config != shared_config:
            raise AnalysisError("shared protocol config mismatch across input files")

    expected_cells = {(seed, arm) for seed in EXPECTED_SEEDS for arm in ARMS}
    seen_cells = set(cells)
    if seen_cells != expected_cells:
        missing = [f"{arm}/s{seed}" for seed, arm in sorted(expected_cells - seen_cells)]
        extra = [f"{arm}/s{seed}" for seed, arm in sorted(seen_cells - expected_cells)]
        raise AnalysisError(f"incomplete campaign matrix; missing={missing}, extra={extra}")

    for seed in EXPECTED_SEEDS:
        hashes = {
            cells[(seed, arm)].config["sft_checkpoint_sha256"] for arm in ARMS
        }
        if len(hashes) != 1:
            raise AnalysisError(f"seed block {seed} does not share one SFT checkpoint hash")

    return cells


def exact_sign_flip_p(differences: Sequence[float]) -> float:
    """Exact two-sided randomization p-value for the paired mean.

    A meet-in-the-middle count evaluates all ``2**n`` sign assignments while
    storing and sorting only ``O(2**(n/2))`` signed sums.  The explicit upper
    bound prevents an accidental, unbounded memory request if this analyzer is
    reused outside its frozen 30-block design.
    """

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise AnalysisError("paired differences must be a nonempty finite vector")
    if values.size > MAX_EXACT_SIGN_FLIP_N:
        raise AnalysisError(
            "exact sign-flip test supports at most "
            f"{MAX_EXACT_SIGN_FLIP_N} pairs, got {values.size}"
        )

    def signed_sums(part: np.ndarray) -> np.ndarray:
        sums = np.zeros(1, dtype=float)
        for value in part:
            sums = np.concatenate((sums - value, sums + value))
        return sums

    split = values.size // 2
    left = signed_sums(values[:split])
    right = np.sort(signed_sums(values[split:]))
    observed_sum = abs(float(np.sum(values)))
    tolerance = max(1e-15, 1e-14 * max(1.0, observed_sum))
    threshold = max(0.0, observed_sum - tolerance)
    if threshold == 0.0:
        return 1.0

    # Count left+right <= -threshold or >= threshold.  These tails do not
    # overlap because threshold is positive.
    lower_count = int(
        np.sum(np.searchsorted(right, -threshold - left, side="right"), dtype=np.int64)
    )
    upper_indices = np.searchsorted(right, threshold - left, side="left")
    upper_count = int(
        np.sum(right.size - upper_indices, dtype=np.int64)
    )
    total = 1 << int(values.size)
    return float((lower_count + upper_count) / total)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values, preserving input labels."""

    if not p_values:
        return {}
    for label, value in p_values.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise AnalysisError(f"invalid p-value for {label!r}: {value}")

    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (label, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[label] = float(running)
    return {label: adjusted[label] for label in p_values}


def _contrast_result(
    label: str,
    differences: np.ndarray,
    bootstrap_indices: np.ndarray,
    raw_p: float,
    adjusted_p: float,
) -> dict[str, Any]:
    bootstrap_means = np.mean(differences[bootstrap_indices], axis=1)
    ci = np.percentile(bootstrap_means, [2.5, 97.5])
    mean = float(np.mean(differences))
    lower, upper = float(ci[0]), float(ci[1])

    supported = mean >= SESOI and lower > 0.0 and adjusted_p < 0.05
    practically_ruled_out = (not supported) and upper < SESOI
    if supported:
        decision = "supported"
    elif practically_ruled_out:
        decision = "practically_ruled_out"
    else:
        decision = "inconclusive"

    return {
        "label": label,
        "per_seed_difference": {
            str(seed): float(value)
            for seed, value in zip(EXPECTED_SEEDS, differences, strict=True)
        },
        "mean": mean,
        "bootstrap_ci_95": [lower, upper],
        "sign_flip_p_two_sided_exact": float(raw_p),
        "holm_adjusted_p": float(adjusted_p),
        "positive_pairs": int(np.sum(differences > 0.0)),
        "negative_pairs": int(np.sum(differences < 0.0)),
        "zero_pairs": int(np.sum(differences == 0.0)),
        "sesoi": SESOI,
        "supported": bool(supported),
        "practically_ruled_out": bool(practically_ruled_out),
        "decision": decision,
    }


def analyze_paths(paths: Sequence[str | os.PathLike[str]]) -> dict[str, Any]:
    """Validate the whole frozen campaign, then calculate its two contrasts."""

    if not paths:
        raise AnalysisError("no input files supplied")
    runs = [load_run(path) for path in paths]
    cells = _validate_campaign(runs)

    # Nothing above this line computes a campaign endpoint.  Completeness and
    # provenance are therefore established before outcomes can be released.
    cell_metrics = {
        (seed, arm): cells[(seed, arm)].cov_auc_delta
        for seed in EXPECTED_SEEDS
        for arm in ARMS
    }
    primary = np.asarray(
        [cell_metrics[(seed, "un")] - cell_metrics[(seed, "learn")] for seed in EXPECTED_SEEDS],
        dtype=float,
    )
    secondary = np.asarray(
        [cell_metrics[(seed, "un")] - cell_metrics[(seed, "unif")] for seed in EXPECTED_SEEDS],
        dtype=float,
    )

    raw_ps = {
        "primary_un_minus_learn": exact_sign_flip_p(primary),
        "secondary_un_minus_unif": exact_sign_flip_p(secondary),
    }
    adjusted_ps = holm_adjust(raw_ps)
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_indices = rng.integers(
        0, len(EXPECTED_SEEDS), size=(BOOTSTRAP_RESAMPLES, len(EXPECTED_SEEDS))
    )

    first = cells[(EXPECTED_SEEDS[0], ARMS[0])]
    output_cells: dict[str, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        output_cells[str(seed)] = {}
        for arm in ARMS:
            run = cells[(seed, arm)]
            output_cells[str(seed)][arm] = {
                "input": str(run.path),
                "sft_checkpoint_sha256": run.config["sft_checkpoint_sha256"],
                "post_sft_cov8": run.baseline_cov8,
                "mean_rl_cov8": float(
                    np.mean([run.update_cov8[update] for update in EXPECTED_UPDATES])
                ),
                "cov_auc_delta": run.cov_auc_delta,
            }

    return {
        "protocol": PROTOCOL,
        "campaign": first.config["campaign"],
        "source_manifest": first.config["source_manifest"],
        "complete": True,
        "seed_blocks": list(EXPECTED_SEEDS),
        "arms": list(ARMS),
        "endpoint": {
            "name": "cov_auc_delta",
            "evaluation_completed_updates": list(EXPECTED_UPDATES),
            "definition": "mean pass@8 coverage over ten RL evaluations minus post-SFT coverage",
        },
        "analysis": {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "random_seed": RANDOM_SEED,
            "bootstrap_interval": "paired percentile 95% (NumPy linear percentile)",
            "randomization_test": "exact two-sided paired sign-flip test on the mean",
            "multiplicity": "Holm adjustment across primary and secondary",
            "support_rule": "mean>=0.005 and CI lower>0 and Holm p<0.05",
            "practically_ruled_out_rule": "CI upper<0.005",
        },
        "cells": output_cells,
        "contrasts": {
            "primary_un_minus_learn": _contrast_result(
                "un - learn",
                primary,
                bootstrap_indices,
                raw_ps["primary_un_minus_learn"],
                adjusted_ps["primary_un_minus_learn"],
            ),
            "secondary_un_minus_unif": _contrast_result(
                "un - unif",
                secondary,
                bootstrap_indices,
                raw_ps["secondary_un_minus_unif"],
                adjusted_ps["secondary_un_minus_unif"],
            ),
        },
    }


def _atomic_write_json(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _default_inputs() -> list[Path]:
    here = Path(__file__).resolve().parent
    return sorted(here.glob("mazescore_*.jsonl"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        help="all 90 canonical mazescore_{arm}_s{seed}.jsonl files",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "maze_score_analysis.json"),
        help="JSON result path, or '-' for stdout",
    )
    args = parser.parse_args(argv)
    inputs = [Path(path) for path in args.inputs] if args.inputs else _default_inputs()

    try:
        result = analyze_paths(inputs)
        if args.output == "-":
            json.dump(result, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
            sys.stdout.write("\n")
        else:
            output = Path(args.output)
            resolved_inputs = {path.resolve() for path in inputs}
            if output.resolve() in resolved_inputs:
                raise AnalysisError("output path must not overwrite an input")
            _atomic_write_json(output, result)
            print(f"complete MAZE-SCORE analysis -> {output}")
        return 0
    except AnalysisError as exc:
        # No result object is printed or written when completeness fails.
        print(f"MAZE-SCORE analysis refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
