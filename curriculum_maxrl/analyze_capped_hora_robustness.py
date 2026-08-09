"""Independent verifier and analyzer for the capped-HORA robustness study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RAW_SCHEMA = "curriculum-maxrl/capped-hora-robustness-raw/v1"
LOCK_SCHEMA = "curriculum-maxrl/capped-hora-robustness-lock/v1"
ANALYSIS_SCHEMA = "curriculum-maxrl/capped-hora-robustness-analysis/v1"
LOCK_PATH = HERE / "CAPPED_HORA_ROBUSTNESS_LOCK.json"
OLD_RESULT_PATH = HERE / "results_postguidance_hora_factorial.json"

SAMPLERS = ("uniform", "u_16")
ALLOCATORS = ("hora_hit", "fresh_group_mass_proxy")
CAPS: tuple[Optional[int], ...] = (24, 32, 48, None)
INFORMATION_SOURCES = ("same_step", "history_plus_probe", "oracle_preupdate")
SCIENTIFIC_SEEDS = tuple(range(16))
AUDIT_SEEDS = (90, 91)
PINNED_RUNTIME = {
    "python_implementation": "CPython",
    "python": "3.9.6",
    "numpy": "1.26.4",
}
DEFAULT_CONFIG = {
    "total_completions": 51_200,
    "checkpoint_every": 2_560,
    "tasks_per_step": 8,
    "average_n": 16,
    "probe_g0": 4,
    "learning_rate": 0.5,
    "uniform_floor": 0.1,
    "reference_group_decay": 0.9,
    "reference_n": 16,
    "eval_k": 8,
    "prior_alpha": 1.0,
    "prior_beta": 1.0,
}
SOURCE_RELATIVE_PATHS = (
    "curriculum_maxrl/CAPPED_HORA_ROBUSTNESS_PROTOCOL.md",
    "curriculum_maxrl/run_capped_hora_robustness.py",
    "curriculum_maxrl/analyze_capped_hora_robustness.py",
    "curriculum_maxrl/test_capped_hora_robustness.py",
    "curriculum_maxrl/estimators.py",
    "curriculum_maxrl/teachers.py",
    "curriculum_maxrl/testbed.py",
)
RNG_AND_TIE_RULES = {
    "environment_seed": "logical_seed",
    "task_teacher_seed": "logical_seed+1000",
    "task_sampling": "with replacement",
    "adaptive_allocation_rng": "none",
    "tie_break": "lowest batch position via NumPy first-index argmax",
    "behavior_snapshot": (
        "policy, exact pass probabilities, and teacher pseudo-count state are "
        "snapshotted before probes"
    ),
    "within_batch_history": (
        "duplicates share the pre-batch task snapshot and add only their own probes"
    ),
    "update_order": (
        "all probes and Phase-B completions use the snapshot; evidence update and "
        "one synchronous policy update follow collection"
    ),
}
OVERLAP_FIELDS = (
    "steps",
    "groups",
    "completions",
    "probe_completions",
    "phase_b_completions",
    "teacher_observed_completions",
    "phase_a_successes",
    "phase_b_successes",
    "sampled_successes",
    "dead_groups",
    "mixed_groups",
    "all_pass_groups",
    "coefficient_l1_mass",
    "coefficient_l1_mass_per_completion",
    "mean_group_size",
    "minimum_group_size",
    "maximum_group_size",
    "group_size_histogram",
    "requested_task_counts",
    "normalized_auc_mean_pass",
    "normalized_auc_pass_at_8",
    "final_mean_pass",
    "final_pass_at_8",
    "checkpoints",
)
METRICS = (
    "normalized_auc_pass_at_8",
    "normalized_auc_mean_pass",
    "final_pass_at_8",
    "final_mean_pass",
    "coefficient_l1_mass_per_completion",
    "dead_groups",
    "mixed_groups",
    "all_pass_groups",
    "maximum_group_size",
    "nearest_rank_p95_group_size",
    "group_size_gini",
    "mean_absolute_probability_error",
    "mean_squared_probability_error",
    "marginal_score_mae",
    "chosen_oracle_regret_mean",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _runtime() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def _fixed_cell(sampler: str) -> dict:
    return {
        "cell_id": f"{sampler}/fixed_n16",
        "sampler": sampler,
        "allocator": "fixed",
        "information_source": None,
        "cap": 16,
    }


def _adaptive_cell(
    sampler: str,
    allocator: str,
    cap: Optional[int],
    information_source: str,
) -> dict:
    cap_label = "uncapped" if cap is None else str(cap)
    return {
        "cell_id": f"{sampler}/{allocator}/{information_source}/cap_{cap_label}",
        "sampler": sampler,
        "allocator": allocator,
        "information_source": information_source,
        "cap": cap,
    }


def _full_cells() -> list[dict]:
    cells = []
    for sampler in SAMPLERS:
        cells.append(_fixed_cell(sampler))
        for allocator in ALLOCATORS:
            for cap in CAPS:
                for information_source in INFORMATION_SOURCES:
                    cells.append(
                        _adaptive_cell(sampler, allocator, cap, information_source)
                    )
    if len(cells) != 50 or len({cell["cell_id"] for cell in cells}) != 50:
        raise AssertionError("independent matrix construction did not yield 50 cells")
    return cells


def _overlap_cells() -> list[dict]:
    cells = []
    for sampler in SAMPLERS:
        cells.extend(
            (
                _fixed_cell(sampler),
                _adaptive_cell(sampler, "hora_hit", None, "same_step"),
                _adaptive_cell(
                    sampler, "fresh_group_mass_proxy", None, "same_step"
                ),
            )
        )
    return cells


def _matrix_manifest() -> dict:
    return {
        "samplers": list(SAMPLERS),
        "fixed_anchor": {"final_group_size": 16, "count": 2},
        "adaptive_allocators": list(ALLOCATORS),
        "caps": [24, 32, 48, "uncapped"],
        "uncapped_mathematical_maximum": 100,
        "information_sources": list(INFORMATION_SOURCES),
        "cell_count": 50,
        "cells": _full_cells(),
    }


def _source_hashes() -> dict[str, str]:
    missing = [
        relative
        for relative in SOURCE_RELATIVE_PATHS
        if not (PROJECT_ROOT / relative).is_file()
    ]
    if missing:
        raise ValueError("locked source files are missing: " + ", ".join(missing))
    return {
        relative: _sha256(PROJECT_ROOT / relative)
        for relative in SOURCE_RELATIVE_PATHS
    }


def _expected_lock_fields() -> dict:
    return {
        "schema": LOCK_SCHEMA,
        "experiment": "capped_hora_skillchain_robustness",
        "frozen_on": "2026-08-08",
        "status": "exploratory multiverse; not confirmatory",
        "runtime": dict(PINNED_RUNTIME),
        "config": dict(DEFAULT_CONFIG),
        "matrix": _matrix_manifest(),
        "scientific_seeds": list(SCIENTIFIC_SEEDS),
        "engineering_audit_seeds": list(AUDIT_SEEDS),
        "rng_and_tie_rules": dict(RNG_AND_TIE_RULES),
        "overlap": {
            "source_result": "curriculum_maxrl/results_postguidance_hora_factorial.json",
            "source_result_sha256": _sha256(OLD_RESULT_PATH),
            "cells": [cell["cell_id"] for cell in _overlap_cells()],
            "fields": list(OVERLAP_FIELDS),
            "logical_seeds": list(SCIENTIFIC_SEEDS),
            "required_exact_comparisons": 96,
        },
        "claim_boundary": (
            "Synthetic cap/information sensitivity only; does not validate HORA, "
            "establish neural RLVR performance, prove mass mediation, or add a "
            "confirmed paper contribution."
        ),
    }


def independently_verify_lock(lock_path: Path = LOCK_PATH) -> tuple[dict, str]:
    path = lock_path.resolve()
    errors = []
    if path != LOCK_PATH.resolve():
        errors.append("noncanonical lock path")
    if not path.is_file():
        raise ValueError(f"canonical lock missing: {path}")
    raw = path.read_bytes()
    lock = json.loads(raw.decode("utf-8"))
    if raw != _canonical_json_bytes(lock):
        errors.append("lock JSON is not canonical")
    if _runtime() != PINNED_RUNTIME:
        errors.append("live runtime is not pinned")
    expected = _expected_lock_fields()
    for key, value in expected.items():
        if lock.get(key) != value:
            errors.append(f"locked {key} mismatch")
    hashes = _source_hashes()
    if set(lock.get("source_sha256", {})) != set(SOURCE_RELATIVE_PATHS):
        errors.append("source key set mismatch")
    if lock.get("source_sha256") != hashes:
        errors.append("source hashes mismatch")
    if set(lock) != set(expected) | {"source_sha256"}:
        errors.append("lock top-level key set mismatch")
    if errors:
        raise ValueError("independent source/runtime lock check failed: " + "; ".join(errors))
    return lock, hashlib.sha256(raw).hexdigest()


def _nearest_rank_p95(histogram: dict[str, int]) -> int:
    count = sum(histogram.values())
    target = int(math.ceil(0.95 * count))
    cumulative = 0
    for size_text in sorted(histogram, key=int):
        cumulative += histogram[size_text]
        if cumulative >= target:
            return int(size_text)
    raise ValueError("empty or malformed group-size histogram")


def _gini(histogram: dict[str, int]) -> float:
    parsed = [(int(size), int(frequency)) for size, frequency in histogram.items()]
    count = sum(frequency for _, frequency in parsed)
    weighted_sum = sum(size * frequency for size, frequency in parsed)
    pair_difference = sum(
        abs(size_i - size_j) * frequency_i * frequency_j
        for size_i, frequency_i in parsed
        for size_j, frequency_j in parsed
    )
    return float(pair_difference / (2.0 * count * weighted_sum))


def _auc(checkpoints: list[dict], metric: str, budget: int) -> float:
    x = np.asarray([row["completions"] for row in checkpoints], dtype=float)
    y = np.asarray([row[metric] for row in checkpoints], dtype=float)
    return float(np.trapz(y, x) / budget)


def _verify_run(run: dict, cell: dict, seed: int) -> list[str]:
    errors = []
    prefix = f"{cell['cell_id']} seed {seed}: "
    for key, value in cell.items():
        if run.get(key) != value:
            errors.append(prefix + f"{key} mismatch")
    if run.get("seed") != seed:
        errors.append(prefix + "seed mismatch")
    steps = DEFAULT_CONFIG["total_completions"] // (
        DEFAULT_CONFIG["tasks_per_step"] * DEFAULT_CONFIG["average_n"]
    )
    groups = steps * DEFAULT_CONFIG["tasks_per_step"]
    if run.get("steps") != steps or run.get("groups") != groups:
        errors.append(prefix + "step/group count mismatch")
    histogram = run.get("group_size_histogram", {})
    try:
        histogram_groups = sum(int(value) for value in histogram.values())
        histogram_completions = sum(
            int(size) * int(value) for size, value in histogram.items()
        )
        if histogram_groups != groups:
            errors.append(prefix + "histogram group count mismatch")
        if histogram_completions != DEFAULT_CONFIG["total_completions"]:
            errors.append(prefix + "histogram completion count mismatch")
        if run.get("minimum_group_size") != min(map(int, histogram)):
            errors.append(prefix + "minimum group size mismatch")
        if run.get("maximum_group_size") != max(map(int, histogram)):
            errors.append(prefix + "maximum group size mismatch")
        if run.get("nearest_rank_p95_group_size") != _nearest_rank_p95(histogram):
            errors.append(prefix + "nearest-rank P95 mismatch")
        if run.get("group_size_gini") != _gini(histogram):
            errors.append(prefix + "Gini mismatch")
    except (TypeError, ValueError, ZeroDivisionError):
        errors.append(prefix + "malformed group-size histogram")
    if run.get("completions") != DEFAULT_CONFIG["total_completions"]:
        errors.append(prefix + "paid budget mismatch")
    if run.get("probe_completions") != groups * DEFAULT_CONFIG["probe_g0"]:
        errors.append(prefix + "probe budget mismatch")
    if run.get("phase_b_completions") != (
        DEFAULT_CONFIG["total_completions"]
        - groups * DEFAULT_CONFIG["probe_g0"]
    ):
        errors.append(prefix + "Phase-B budget mismatch")
    if run.get("teacher_observed_completions") != DEFAULT_CONFIG["total_completions"]:
        errors.append(prefix + "teacher evidence budget mismatch")
    if run.get("mean_group_size") != 16.0:
        errors.append(prefix + "average group size mismatch")
    if sum(run.get("requested_task_counts", [])) != groups:
        errors.append(prefix + "task-request counts mismatch")
    if run.get("sampled_successes") != (
        run.get("phase_a_successes", -1) + run.get("phase_b_successes", -2)
    ):
        errors.append(prefix + "success accounting mismatch")
    if run.get("dead_groups", -1) + run.get("mixed_groups", -1) + run.get(
        "all_pass_groups", -1
    ) != groups:
        errors.append(prefix + "group-outcome accounting mismatch")
    checkpoints = run.get("checkpoints", [])
    expected_checkpoint_clock = list(
        range(
            0,
            DEFAULT_CONFIG["total_completions"] + 1,
            DEFAULT_CONFIG["checkpoint_every"],
        )
    )
    if [row.get("completions") for row in checkpoints] != expected_checkpoint_clock:
        errors.append(prefix + "checkpoint clock mismatch")
    else:
        if run.get("normalized_auc_mean_pass") != _auc(
            checkpoints, "mean_pass", DEFAULT_CONFIG["total_completions"]
        ):
            errors.append(prefix + "mean-pass AUC mismatch")
        if run.get("normalized_auc_pass_at_8") != _auc(
            checkpoints, "pass_at_8", DEFAULT_CONFIG["total_completions"]
        ):
            errors.append(prefix + "pass@8 AUC mismatch")
        if run.get("final_mean_pass") != checkpoints[-1].get("mean_pass"):
            errors.append(prefix + "terminal mean-pass mismatch")
        if run.get("final_pass_at_8") != checkpoints[-1].get("pass_at_8"):
            errors.append(prefix + "terminal pass@8 mismatch")

    fixed = cell["allocator"] == "fixed"
    if fixed:
        if run.get("minimum_group_size") != 16 or run.get("maximum_group_size") != 16:
            errors.append(prefix + "fixed anchor allocation mismatch")
        for key in (
            "mean_absolute_probability_error",
            "mean_squared_probability_error",
            "marginal_score_mae",
            "chosen_oracle_regret_mean",
        ):
            if run.get(key) is not None:
                errors.append(prefix + f"fixed {key} must be null")
        for key in (
            "probability_error_position_count",
            "marginal_score_eligible_position_decision_count",
            "allocation_decision_count",
        ):
            if run.get(key) != 0:
                errors.append(prefix + f"fixed {key} must be zero")
        if run.get("allocation_diagnostics_applicable") is not False:
            errors.append(prefix + "fixed diagnostic applicability mismatch")
    else:
        maximum = 100 if cell["cap"] is None else cell["cap"]
        if run.get("minimum_group_size", 0) < 4 or run.get(
            "maximum_group_size", 101
        ) > maximum:
            errors.append(prefix + "adaptive cap/probe bound mismatch")
        if run.get("probability_error_position_count") != groups:
            errors.append(prefix + "probability position count mismatch")
        if run.get("allocation_decision_count") != steps * 96:
            errors.append(prefix + "allocation decision count mismatch")
        if run.get("marginal_score_eligible_position_decision_count", 0) < steps * 96:
            errors.append(prefix + "eligible score count mismatch")
        if run.get("allocation_diagnostics_applicable") is not True:
            errors.append(prefix + "adaptive diagnostic applicability mismatch")
        for key in (
            "mean_absolute_probability_error",
            "mean_squared_probability_error",
            "marginal_score_mae",
            "chosen_oracle_regret_mean",
        ):
            value = run.get(key)
            if value is None or not np.isfinite(value) or value < 0.0:
                errors.append(prefix + f"invalid {key}")
        if cell["information_source"] == "oracle_preupdate":
            for key in (
                "mean_absolute_probability_error",
                "mean_squared_probability_error",
                "marginal_score_mae",
                "chosen_oracle_regret_mean",
            ):
                if run.get(key) != 0.0:
                    errors.append(prefix + f"oracle {key} must be exactly zero")
    return errors


def _old_overlap_key(cell: dict) -> tuple[str, str]:
    old_sampler = "u_n" if cell["sampler"] == "u_16" else "uniform"
    allocator = cell["allocator"]
    old_allocator = "mass_aware" if allocator == "fresh_group_mass_proxy" else allocator
    return old_sampler, old_allocator


def independently_verify_overlap(runs: Sequence[dict]) -> dict:
    old = json.loads(OLD_RESULT_PATH.read_text(encoding="utf-8"))
    old_lookup = {}
    for cell in old["cells"].values():
        for run in cell["seed_runs"]:
            old_lookup[(run["sampler"], run["allocator"], run["seed"])] = run
    new_lookup = {(run["cell_id"], run["seed"]): run for run in runs}
    mismatches = []
    comparisons = 0
    for cell in _overlap_cells():
        old_sampler, old_allocator = _old_overlap_key(cell)
        for seed in SCIENTIFIC_SEEDS:
            comparisons += 1
            new = new_lookup.get((cell["cell_id"], seed))
            prior = old_lookup.get((old_sampler, old_allocator, seed))
            if new is None or prior is None:
                mismatches.append(
                    {"cell_id": cell["cell_id"], "seed": seed, "field": "missing_run"}
                )
                continue
            for field in OVERLAP_FIELDS:
                if new.get(field) != prior.get(field):
                    mismatches.append(
                        {"cell_id": cell["cell_id"], "seed": seed, "field": field}
                    )
    return {
        "passed": comparisons == 96 and not mismatches,
        "source_result_sha256": _sha256(OLD_RESULT_PATH),
        "comparisons": comparisons,
        "expected_comparisons": 96,
        "mismatches": mismatches,
    }


def verify_raw_artifact(raw: dict, lock_path: Path = LOCK_PATH) -> dict:
    lock, lock_sha256 = independently_verify_lock(lock_path)
    errors = []
    if raw.get("schema") != RAW_SCHEMA:
        errors.append("raw schema mismatch")
    if raw.get("experiment") != "capped_hora_skillchain_robustness":
        errors.append("experiment name mismatch")
    if raw.get("claim_status") != "exploratory multiverse; not confirmatory":
        errors.append("claim status mismatch")
    mode = raw.get("mode")
    if mode == "audit":
        expected_cells, expected_seeds = _full_cells(), AUDIT_SEEDS
    elif mode == "overlap":
        expected_cells, expected_seeds = _overlap_cells(), SCIENTIFIC_SEEDS
    elif mode == "full":
        expected_cells, expected_seeds = _full_cells(), SCIENTIFIC_SEEDS
    else:
        errors.append("unknown raw mode")
        expected_cells, expected_seeds = [], ()
    expected_config = {**DEFAULT_CONFIG, "seeds": list(expected_seeds)}
    if raw.get("config") != expected_config:
        errors.append("raw configuration mismatch")
    if raw.get("matrix") != {
        "cell_count": len(expected_cells),
        "cells": expected_cells,
    }:
        errors.append("raw matrix mismatch")
    provenance = raw.get("provenance", {})
    if provenance.get("source_lock_enforced") is not True:
        errors.append("raw lock was not enforced")
    if provenance.get("source_lock_relative_path") != (
        "curriculum_maxrl/CAPPED_HORA_ROBUSTNESS_LOCK.json"
    ):
        errors.append("raw lock path mismatch")
    if provenance.get("source_lock_sha256") != lock_sha256:
        errors.append("raw lock hash mismatch")
    if provenance.get("source_sha256") != lock.get("source_sha256"):
        errors.append("raw source hashes mismatch")
    if provenance.get("runtime") != _runtime():
        errors.append("raw runtime mismatch")
    if provenance.get("rng_and_tie_rules") != RNG_AND_TIE_RULES:
        errors.append("raw RNG/tie rules mismatch")
    checks = raw.get("checks", {})
    if not checks or not all(value is True for value in checks.values()):
        errors.append("raw accounting checks are absent or failed")
    runs = raw.get("runs", [])
    lookup = {}
    for run in runs:
        key = (run.get("cell_id"), run.get("seed"))
        if key in lookup:
            errors.append(f"duplicate raw run {key!r}")
        lookup[key] = run
    expected_keys = {
        (cell["cell_id"], seed) for cell in expected_cells for seed in expected_seeds
    }
    if set(lookup) != expected_keys or len(runs) != len(expected_keys):
        errors.append("raw run matrix is incomplete")
    cell_by_id = {cell["cell_id"]: cell for cell in expected_cells}
    for (cell_id, seed), run in lookup.items():
        if cell_id in cell_by_id and seed in expected_seeds:
            errors.extend(_verify_run(run, cell_by_id[cell_id], seed))
    overlap = None
    if mode == "overlap" and set(lookup) == expected_keys:
        overlap = independently_verify_overlap(runs)
        if not overlap["passed"]:
            errors.append("independent six-cell overlap verification failed")
        runner_overlap = raw.get("overlap_verification", {})
        if runner_overlap.get("passed") is not True:
            errors.append("runner overlap check did not pass")
        if runner_overlap.get("comparisons") != 96:
            errors.append("runner overlap comparison count mismatch")
    if errors:
        raise ValueError("raw artifact verification failed: " + "; ".join(errors))
    return {
        "passed": True,
        "mode": mode,
        "run_count": len(runs),
        "cell_count": len(expected_cells),
        "seed_count": len(expected_seeds),
        "lock_sha256": lock_sha256,
        "overlap": overlap,
    }


def summarize(values: Iterable[Optional[float]]) -> dict:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot summarize an empty sequence")
    if any(value is None for value in materialized):
        if not all(value is None for value in materialized):
            raise ValueError("metric mixes applicable and non-applicable values")
        return {
            "status": "not_applicable",
            "reason": "fixed anchors have no allocation information source or choice",
            "n": len(materialized),
        }
    x = np.asarray(materialized, dtype=float)
    sample_sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return {
        "status": "ok",
        "n": len(x),
        "mean": float(x.mean()),
        "sample_sd": sample_sd,
        "min": float(x.min()),
        "max": float(x.max()),
        "values_by_seed": x.tolist(),
    }


def _t_critical_975(df: int) -> float:
    registered = {
        1: 12.706204736432095,
        15: 2.131449545559323,
    }
    if df not in registered:
        raise ValueError(f"no frozen t critical value for df={df}")
    return registered[df]


def paired_summary(
    lhs_runs: Sequence[dict], rhs_runs: Sequence[dict], metric: str
) -> dict:
    if len(lhs_runs) != len(rhs_runs) or len(lhs_runs) < 2:
        raise ValueError("paired contrasts require equally sized sequences of >=2")
    if [run["seed"] for run in lhs_runs] != [run["seed"] for run in rhs_runs]:
        raise ValueError("paired contrast seed order differs")
    lhs = [run[metric] for run in lhs_runs]
    rhs = [run[metric] for run in rhs_runs]
    if any(value is None for value in lhs + rhs):
        return {
            "status": "not_applicable",
            "reason": (
                "one side has no registered allocation diagnostic; learning, mass, "
                "outcome, and group-size metrics remain paired"
            ),
            "n": len(lhs),
        }
    difference = np.asarray(lhs, dtype=float) - np.asarray(rhs, dtype=float)
    n = len(difference)
    mean = float(difference.mean())
    sample_sd = float(difference.std(ddof=1))
    standard_error = sample_sd / math.sqrt(n)
    half_width = _t_critical_975(n - 1) * standard_error
    return {
        "status": "ok",
        "n": n,
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "student_t_95_ci": [mean - half_width, mean + half_width],
        "positive_seeds": int((difference > 0.0).sum()),
        "zero_seeds": int((difference == 0.0).sum()),
        "negative_seeds": int((difference < 0.0).sum()),
        "values_by_seed": difference.tolist(),
    }


def _contrast(
    contrast_id: str,
    lhs_id: str,
    rhs_id: str,
    lookup: dict[tuple[str, int], dict],
    seeds: Sequence[int],
) -> dict:
    lhs = [lookup[(lhs_id, seed)] for seed in seeds]
    rhs = [lookup[(rhs_id, seed)] for seed in seeds]
    return {
        "contrast_id": contrast_id,
        "lhs_cell": lhs_id,
        "rhs_cell": rhs_id,
        "metrics": {metric: paired_summary(lhs, rhs, metric) for metric in METRICS},
    }


def build_contrasts(runs: Sequence[dict], seeds: Sequence[int]) -> dict:
    lookup = {(run["cell_id"], run["seed"]): run for run in runs}
    cap_minus_uncapped = []
    allocator_fresh_minus_hora = []
    history_minus_same = []
    oracle_minus_deployable = []
    adaptive_minus_fixed = []
    for sampler in SAMPLERS:
        fixed_id = _fixed_cell(sampler)["cell_id"]
        for allocator in ALLOCATORS:
            for information in INFORMATION_SOURCES:
                uncapped = _adaptive_cell(sampler, allocator, None, information)[
                    "cell_id"
                ]
                for cap in (24, 32, 48):
                    capped = _adaptive_cell(sampler, allocator, cap, information)[
                        "cell_id"
                    ]
                    cap_minus_uncapped.append(
                        _contrast(
                            f"{capped}_minus_uncapped",
                            capped,
                            uncapped,
                            lookup,
                            seeds,
                        )
                    )
            for cap in CAPS:
                history = _adaptive_cell(
                    sampler, allocator, cap, "history_plus_probe"
                )["cell_id"]
                same = _adaptive_cell(sampler, allocator, cap, "same_step")[
                    "cell_id"
                ]
                history_minus_same.append(
                    _contrast(
                        f"{history}_minus_same_step",
                        history,
                        same,
                        lookup,
                        seeds,
                    )
                )
                oracle = _adaptive_cell(
                    sampler, allocator, cap, "oracle_preupdate"
                )["cell_id"]
                for deployable in ("same_step", "history_plus_probe"):
                    rhs = _adaptive_cell(sampler, allocator, cap, deployable)[
                        "cell_id"
                    ]
                    oracle_minus_deployable.append(
                        _contrast(
                            f"{oracle}_minus_{deployable}",
                            oracle,
                            rhs,
                            lookup,
                            seeds,
                        )
                    )
        for information in INFORMATION_SOURCES:
            for cap in CAPS:
                fresh = _adaptive_cell(
                    sampler, "fresh_group_mass_proxy", cap, information
                )["cell_id"]
                hora = _adaptive_cell(sampler, "hora_hit", cap, information)[
                    "cell_id"
                ]
                allocator_fresh_minus_hora.append(
                    _contrast(
                        f"{fresh}_minus_hora_hit",
                        fresh,
                        hora,
                        lookup,
                        seeds,
                    )
                )
        for allocator in ALLOCATORS:
            for cap in CAPS:
                for information in INFORMATION_SOURCES:
                    adaptive = _adaptive_cell(
                        sampler, allocator, cap, information
                    )["cell_id"]
                    adaptive_minus_fixed.append(
                        _contrast(
                            f"{adaptive}_minus_fixed_n16",
                            adaptive,
                            fixed_id,
                            lookup,
                            seeds,
                        )
                    )
    expected_counts = (36, 24, 16, 32, 48)
    observed_counts = tuple(
        len(rows)
        for rows in (
            cap_minus_uncapped,
            allocator_fresh_minus_hora,
            history_minus_same,
            oracle_minus_deployable,
            adaptive_minus_fixed,
        )
    )
    if observed_counts != expected_counts:
        raise AssertionError(
            f"contrast grid mismatch: observed {observed_counts}, expected {expected_counts}"
        )
    return {
        "cap_minus_uncapped_within_allocator": cap_minus_uncapped,
        "fresh_group_mass_proxy_minus_hora_hit_within_cap": allocator_fresh_minus_hora,
        "history_plus_probe_minus_same_step": history_minus_same,
        "oracle_preupdate_minus_deployable": oracle_minus_deployable,
        "adaptive_minus_fixed_anchor": adaptive_minus_fixed,
        "contrast_counts": {
            "cap_minus_uncapped_within_allocator": 36,
            "fresh_group_mass_proxy_minus_hora_hit_within_cap": 24,
            "history_plus_probe_minus_same_step": 16,
            "oracle_preupdate_minus_deployable": 32,
            "adaptive_minus_fixed_anchor": 48,
        },
        "inference": (
            "equal-seed descriptive paired differences with Student-t 95% intervals; "
            "no nominal contrast p-values"
        ),
    }


def engineering_cap_filter(runs: Sequence[dict], seeds: Sequence[int]) -> dict:
    if tuple(seeds) != SCIENTIFIC_SEEDS:
        return {
            "status": "not_evaluated",
            "reason": "the prospective filter is defined only on retained seeds 0..15",
        }
    lookup = {(run["cell_id"], run["seed"]): run for run in runs}
    rows = []
    for cap in (24, 32, 48):
        cap_max_by_seed = []
        uncapped_max_by_seed = []
        auc_difference_by_seed = []
        for seed in seeds:
            capped_runs = [
                lookup[
                    (
                        _adaptive_cell(
                            sampler,
                            "fresh_group_mass_proxy",
                            cap,
                            "same_step",
                        )["cell_id"],
                        seed,
                    )
                ]
                for sampler in SAMPLERS
            ]
            uncapped_runs = [
                lookup[
                    (
                        _adaptive_cell(
                            sampler,
                            "fresh_group_mass_proxy",
                            None,
                            "same_step",
                        )["cell_id"],
                        seed,
                    )
                ]
                for sampler in SAMPLERS
            ]
            cap_max_by_seed.append(
                float(np.mean([run["maximum_group_size"] for run in capped_runs]))
            )
            uncapped_max_by_seed.append(
                float(np.mean([run["maximum_group_size"] for run in uncapped_runs]))
            )
            auc_difference_by_seed.append(
                float(
                    np.mean(
                        [
                            capped["normalized_auc_pass_at_8"]
                            - uncapped["normalized_auc_pass_at_8"]
                            for capped, uncapped in zip(capped_runs, uncapped_runs)
                        ]
                    )
                )
            )
        reduction = 1.0 - float(np.mean(cap_max_by_seed)) / float(
            np.mean(uncapped_max_by_seed)
        )
        auc_change = float(np.mean(auc_difference_by_seed))
        eligible = reduction >= 0.25 and auc_change >= -0.005
        rows.append(
            {
                "cap": cap,
                "maximum_group_size_reduction": reduction,
                "pass_at_8_auc_change": auc_change,
                "eligible": eligible,
            }
        )
    eligible_caps = [row["cap"] for row in rows if row["eligible"]]
    return {
        "status": "evaluated",
        "rows": rows,
        "selected_cap": min(eligible_caps) if eligible_caps else "uncapped",
        "calibration_failed": not bool(eligible_caps),
        "claim_boundary": (
            "prospective engineering filter only; point estimates are not an "
            "equivalence test or paper-level success criterion"
        ),
    }


def build_analysis(raw: dict, raw_path: Path, lock_path: Path = LOCK_PATH) -> dict:
    on_disk = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw != on_disk:
        raise ValueError("supplied raw object does not match the artifact bytes")
    verification = verify_raw_artifact(raw, lock_path)
    seeds = tuple(raw["config"]["seeds"])
    runs = raw["runs"]
    lookup = {(run["cell_id"], run["seed"]): run for run in runs}
    cells = {}
    for cell in raw["matrix"]["cells"]:
        seed_runs = [lookup[(cell["cell_id"], seed)] for seed in seeds]
        cells[cell["cell_id"]] = {
            "spec": cell,
            "metrics": {
                metric: summarize(run[metric] for run in seed_runs)
                for metric in METRICS
            },
        }
    complete_matrix = raw["matrix"]["cell_count"] == 50
    contrasts = build_contrasts(runs, seeds) if complete_matrix else {
        "status": "not_applicable",
        "reason": "overlap mode contains only the six legacy-overlap cells",
    }
    cap_filter = (
        engineering_cap_filter(runs, seeds)
        if raw["mode"] == "full"
        else {
            "status": "not_evaluated",
            "reason": "only the retained full matrix may drive the prospective filter",
        }
    )
    return {
        "schema": ANALYSIS_SCHEMA,
        "experiment": "capped_hora_skillchain_robustness",
        "mode": raw["mode"],
        "analysis_status": (
            "exploratory multiverse; descriptive paired intervals; no nominal "
            "contrast p-values and no confirmatory promotion"
        ),
        "raw_artifact": str(raw_path),
        "raw_artifact_sha256": _sha256(raw_path),
        "verification": verification,
        "frozen_readout_metrics": list(METRICS),
        "cells": cells,
        "contrasts": contrasts,
        "pre_outcome_engineering_cap_filter": cap_filter,
        "claim_boundary": (
            "Tests cap/information sensitivity in one synthetic shared-skill testbed; "
            "does not validate HORA, establish neural RLVR performance, prove "
            "coefficient-mass mediation, or add a fourth confirmed contribution."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.artifact.read_text(encoding="utf-8"))
    analysis = build_analysis(raw, args.artifact.resolve(), args.lock)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(_canonical_json_bytes(analysis))
    print("Capped-HORA independent verification passed")
    print(f"mode: {analysis['mode']}")
    print(f"cells: {analysis['verification']['cell_count']}")
    print(f"runs: {analysis['verification']['run_count']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
