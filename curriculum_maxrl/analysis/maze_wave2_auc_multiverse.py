#!/usr/bin/env python3
"""Frozen AUC robustness multiverse for the wave-2 maze factorial.

The independent unit is always a seed/warm-start block.  Uniform and
``frontier_un`` are repeated sampler observations inside each block and are
reported separately; their average is a block-level descriptive summary, not
two additional replicates.

The committed repository currently contains cell-level AUC summaries, not the
per-checkpoint JSONL trajectories required for this analysis.  In that state
the program writes a deterministic insufficiency audit and exits with status
2.  Once all 24 raw logs are supplied, it validates them against the frozen
summary before evaluating the prespecified multiverse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_SUMMARY = (
    REPO / "curriculum_maxrl" / "maze_gpu_factorial"
    / "results_factorial_wave2.json"
)
DEFAULT_RAW_DIR = REPO / "curriculum_maxrl" / "maze_gpu_factorial"
DEFAULT_JSON_OUT = HERE / "maze_wave2_auc_multiverse.json"
DEFAULT_REPORT_OUT = HERE / "MAZE_WAVE2_AUC_MULTIVERSE.md"

SAMPLERS = ("uniform", "frontier_un")
ESTIMATORS = ("maxrl", "grpo")
SEEDS = tuple(range(6, 12))
LEVELS = tuple(str(level) for level in range(13))
CHECKPOINT_STEPS = tuple(range(0, 251, 25))
HORIZONS = {"early": 75, "mid": 150, "full": 250}
INTEGRATORS = ("simple_mean", "trapezoid")
WARMUP_OPTIONS = ("exclude", "include")
SIGN_TOLERANCE = 1e-12
SUMMARY_TOLERANCE = 1e-12


class DataValidationError(ValueError):
    """Raised when present raw data violate the frozen input contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def _cell_key(sampler: str, estimator: str, seed: int) -> str:
    return f"{sampler}/{estimator}/s{seed}"


def _raw_filename(sampler: str, estimator: str, seed: int) -> str:
    return f"fact250_{sampler}_{estimator}_s{seed}.jsonl"


def _sign(value: float) -> int:
    if abs(value) <= SIGN_TOLERANCE:
        return 0
    return 1 if value > 0 else -1


def _canonicalize(value: Any) -> Any:
    if isinstance(value, float):
        rounded = round(value, 15)
        return 0.0 if abs(rounded) <= SIGN_TOLERANCE else rounded
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in value.items()}
    return value


def _contrast_summary(values: Iterable[float]) -> dict[str, Any]:
    xs = [float(value) for value in values]
    return {
        "n_independent_blocks": len(xs),
        "n_positive": sum(_sign(value) > 0 for value in xs),
        "n_zero": sum(_sign(value) == 0 for value in xs),
        "n_negative": sum(_sign(value) < 0 for value in xs),
        "mean": statistics.mean(xs),
        "minimum": min(xs),
        "maximum": max(xs),
        "per_block": xs,
    }


def _load_summary(path: Path) -> dict[str, Any]:
    try:
        source = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise DataValidationError(f"summary file is missing: {path}") from exc
    cells = source.get("cells")
    if not isinstance(cells, dict):
        raise DataValidationError("summary must contain an object named 'cells'")
    expected = {
        _cell_key(sampler, estimator, seed)
        for sampler in SAMPLERS
        for estimator in ESTIMATORS
        for seed in SEEDS
    }
    missing = sorted(expected - set(cells))
    if missing:
        raise DataValidationError(
            f"summary is missing {len(missing)} required cells: {missing}"
        )
    for key in sorted(expected):
        value = cells[key].get("cov_auc_delta")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise DataValidationError(f"{key} lacks finite cov_auc_delta")
    return source


def _legacy_anchor(source: dict[str, Any]) -> dict[str, Any]:
    cells = source["cells"]
    per_sampler: dict[str, Any] = {}
    contrasts_by_sampler: dict[str, list[float]] = {}
    for sampler in SAMPLERS:
        values = [
            float(cells[_cell_key(sampler, "maxrl", seed)]["cov_auc_delta"])
            - float(cells[_cell_key(sampler, "grpo", seed)]["cov_auc_delta"])
            for seed in SEEDS
        ]
        contrasts_by_sampler[sampler] = values
        per_sampler[sampler] = _contrast_summary(values)
    block_average = [
        statistics.mean(contrasts_by_sampler[sampler][index]
                        for sampler in SAMPLERS)
        for index in range(len(SEEDS))
    ]
    return {
        "definition": (
            "Frozen summary cov_auc_delta: arithmetic mean over every "
            "in-training pass@8 record selected by fact_analyze.py minus "
            "post-SFT initialization; MaxRL minus GRPO within block."
        ),
        "per_sampler": per_sampler,
        "sampler_average_within_block": _contrast_summary(block_average),
    }


def _coverage(record: dict[str, Any], *, path: Path, line: int) -> float:
    passk = record.get("passk")
    if not isinstance(passk, dict):
        raise DataValidationError(f"{path}:{line}: missing passk object")
    values = []
    for level in LEVELS:
        level_values = passk.get(level)
        if not isinstance(level_values, dict) or "8" not in level_values:
            raise DataValidationError(
                f"{path}:{line}: missing passk[{level!r}]['8']"
            )
        value = level_values["8"]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise DataValidationError(
                f"{path}:{line}: pass@8 for level {level} is not finite"
            )
        if not 0.0 <= float(value) <= 1.0:
            raise DataValidationError(
                f"{path}:{line}: pass@8 for level {level} is outside [0,1]"
            )
        values.append(float(value))
    return statistics.mean(values)


def _read_raw_curve(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(record, dict):
            raise DataValidationError(f"{path}:{line_number}: row is not an object")
        if "passk" not in record:
            continue
        step = record.get("step")
        if not isinstance(step, int):
            raise DataValidationError(f"{path}:{line_number}: step is not an integer")
        rows.append({
            "step": step,
            "coverage": _coverage(record, path=path, line=line_number),
            "final": bool(record.get("final")),
            "line": line_number,
        })

    init_rows = [row for row in rows if row["step"] == -1]
    if len(init_rows) != 1:
        raise DataValidationError(
            f"{path}: expected exactly one post-SFT step -1 row; found {len(init_rows)}"
        )
    training_rows = [row for row in rows if row["step"] >= 0]
    if not training_rows:
        raise DataValidationError(f"{path}: no in-training pass@8 records")

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in training_rows:
        grouped.setdefault(row["step"], []).append(row)
    steps = tuple(sorted(grouped))
    if steps != CHECKPOINT_STEPS:
        missing = sorted(set(CHECKPOINT_STEPS) - set(steps))
        extra = sorted(set(steps) - set(CHECKPOINT_STEPS))
        raise DataValidationError(
            f"{path}: checkpoint grid mismatch; missing={missing}, extra={extra}"
        )

    # Frozen duplicate policy: checkpoint semantics are last-record-wins.
    # The legacy anchor below still uses every record exactly as the original
    # analyzer did, so duplicate weighting cannot pass unnoticed.
    checkpoints = {
        step: grouped[step][-1]["coverage"] for step in CHECKPOINT_STEPS
    }
    duplicate_steps = {
        str(step): [row["line"] for row in grouped[step]]
        for step in CHECKPOINT_STEPS if len(grouped[step]) > 1
    }
    init = init_rows[0]["coverage"]
    legacy_auc_delta = statistics.mean(
        row["coverage"] for row in training_rows
    ) - init
    return {
        "init": init,
        "checkpoints": checkpoints,
        "legacy_auc_delta": legacy_auc_delta,
        "n_legacy_records": len(training_rows),
        "duplicate_steps_last_record_wins": duplicate_steps,
    }


def _aggregate(
    curve: dict[str, Any], *, integration: str, warmup: str,
    horizon: int, omitted_step: int | None = None,
) -> float:
    points = [
        (step, float(curve["checkpoints"][step]) - float(curve["init"]))
        for step in CHECKPOINT_STEPS
        if step <= horizon and step != omitted_step
    ]
    if warmup == "include":
        points.insert(0, (CHECKPOINT_STEPS[0] - 25, 0.0))
    if not points:
        raise DataValidationError("a multiverse cell has no eligible checkpoints")
    if integration == "simple_mean":
        return statistics.mean(value for _, value in points)
    if integration != "trapezoid":
        raise DataValidationError(f"unknown integrator: {integration}")
    if len(points) < 2 or points[-1][0] == points[0][0]:
        raise DataValidationError("trapezoid integration requires two time points")
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for (left_x, left_y), (right_x, right_y)
        in zip(points, points[1:])
    )
    return area / (points[-1][0] - points[0][0])


def _variant(
    curves: dict[str, dict[str, Any]], *, integration: str, warmup: str,
    horizon_name: str, omitted_step: int | None = None,
) -> dict[str, Any]:
    horizon = HORIZONS[horizon_name]
    cell_values = {
        key: _aggregate(
            curve, integration=integration, warmup=warmup,
            horizon=horizon, omitted_step=omitted_step,
        )
        for key, curve in curves.items()
    }
    contrasts_by_sampler: dict[str, list[float]] = {}
    per_sampler: dict[str, Any] = {}
    for sampler in SAMPLERS:
        values = [
            cell_values[_cell_key(sampler, "maxrl", seed)]
            - cell_values[_cell_key(sampler, "grpo", seed)]
            for seed in SEEDS
        ]
        contrasts_by_sampler[sampler] = values
        per_sampler[sampler] = _contrast_summary(values)
    block_average = [
        statistics.mean(contrasts_by_sampler[sampler][index]
                        for sampler in SAMPLERS)
        for index in range(len(SEEDS))
    ]
    variant_id = f"{integration}__warmup-{warmup}__{horizon_name}"
    if omitted_step is not None:
        variant_id += f"__omit-{omitted_step}"
    return {
        "variant_id": variant_id,
        "integration": integration,
        "warmup": warmup,
        "horizon": horizon_name,
        "horizon_step": horizon,
        "omitted_checkpoint_step": omitted_step,
        "per_sampler": per_sampler,
        "sampler_average_within_block": _contrast_summary(block_average),
    }


def _range(values: Iterable[float]) -> list[float]:
    xs = list(values)
    return [min(xs), max(xs)]


def _robustness_summary(
    base: list[dict[str, Any]], loo: list[dict[str, Any]],
) -> dict[str, Any]:
    all_variants = base + loo
    views = {
        "uniform": lambda row: row["per_sampler"]["uniform"],
        "frontier_un": lambda row: row["per_sampler"]["frontier_un"],
        "sampler_average_within_block": (
            lambda row: row["sampler_average_within_block"]
        ),
    }
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            name: {
                "minimum_positive_blocks": min(
                    selector(row)["n_positive"] for row in rows
                ),
                "mean_effect_range": _range(
                    selector(row)["mean"] for row in rows
                ),
            }
            for name, selector in views.items()
        }
    return {
        "n_base_variants": len(base),
        "n_leave_one_checkpoint_out_variants": len(loo),
        "n_total_variants": len(all_variants),
        "base_variants": summarize(base),
        "base_plus_leave_one_checkpoint_out": summarize(all_variants),
        "interpretation_rule": (
            "Signs are counted over six independent seed/warm-start blocks. "
            "The two samplers are never pooled as 12 independent replicates."
        ),
    }


def _build_multiverse(curves: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base: list[dict[str, Any]] = []
    loo: list[dict[str, Any]] = []
    for horizon_name, horizon in HORIZONS.items():
        eligible_steps = [step for step in CHECKPOINT_STEPS if step <= horizon]
        for integration in INTEGRATORS:
            for warmup in WARMUP_OPTIONS:
                base.append(_variant(
                    curves, integration=integration, warmup=warmup,
                    horizon_name=horizon_name,
                ))
                for omitted_step in eligible_steps:
                    loo.append(_variant(
                        curves, integration=integration, warmup=warmup,
                        horizon_name=horizon_name,
                        omitted_step=omitted_step,
                    ))
    return {
        "base_variants": base,
        "leave_one_checkpoint_out_variants": loo,
        "robustness_summary": _robustness_summary(base, loo),
    }


def build_analysis(summary_path: Path, raw_dir: Path) -> dict[str, Any]:
    source = _load_summary(summary_path)
    anchor = _legacy_anchor(source)
    expected_paths = {
        _cell_key(sampler, estimator, seed): (
            raw_dir / _raw_filename(sampler, estimator, seed)
        )
        for sampler in SAMPLERS
        for estimator in ESTIMATORS
        for seed in SEEDS
    }
    missing = [path for path in expected_paths.values() if not path.is_file()]
    common = {
        "schema_version": 1,
        "analysis_id": "maze-wave2-frozen-auc-multiverse-v1",
        "generated_by": "curriculum_maxrl/analysis/maze_wave2_auc_multiverse.py",
        "frozen_specification": {
            "independent_unit": "seed/warm-start block (six blocks, seeds 6-11)",
            "within_block_repeated_factor": "sampler (uniform, frontier_un)",
            "contrast": "MaxRL minus GRPO",
            "metric": "mean pass@8 over maze levels 0-12, relative to post-SFT init",
            "checkpoint_grid": list(CHECKPOINT_STEPS),
            "horizons": HORIZONS,
            "integration_conventions": list(INTEGRATORS),
            "warmup_options": list(WARMUP_OPTIONS),
            "warmup_coordinate_for_trapezoid": -25,
            "duplicate_checkpoint_policy": "last record wins",
            "leave_one_checkpoint_out": (
                "omit each eligible in-training checkpoint in turn; never "
                "omit the warmup point"
            ),
        },
        "summary_source": {
            "path": _display_path(summary_path),
            "sha256": _sha256(summary_path),
            "n_required_cells": 24,
        },
        "legacy_summary_anchor": anchor,
    }
    if missing:
        return _canonicalize({
            **common,
            "status": "insufficient_raw_trajectories",
            "raw_data_audit": {
                "raw_directory": _display_path(raw_dir),
                "n_expected_files": len(expected_paths),
                "n_found_files": len(expected_paths) - len(missing),
                "n_missing_files": len(missing),
                "missing_files": [_display_path(path) for path in missing],
                "why_summaries_are_insufficient": [
                    "cov_auc_delta is one scalar mean and cannot identify checkpoint values",
                    "early and mid horizons cannot be reconstructed from init/final endpoints",
                    "trapezoid integration needs checkpoint time-value pairs",
                    "leave-one-checkpoint-out needs every checkpoint contribution",
                ],
                "required_record_schema": (
                    "one JSON object per line with integer step and passk levels "
                    "0-12 containing key '8'; exactly one step -1 warmup and "
                    "checkpoints 0,25,...,250"
                ),
            },
            "multiverse": None,
            "robustness_summary": {
                "minimum_positive_blocks": None,
                "mean_effect_range": None,
                "reason": "not identifiable from aggregate cell summaries",
            },
        })

    curves: dict[str, dict[str, Any]] = {}
    raw_sources = []
    validation_errors = []
    for key, path in expected_paths.items():
        try:
            curve = _read_raw_curve(path)
            expected_auc = float(source["cells"][key]["cov_auc_delta"])
            if not math.isclose(
                curve["legacy_auc_delta"], expected_auc,
                rel_tol=0.0, abs_tol=SUMMARY_TOLERANCE,
            ):
                raise DataValidationError(
                    f"{path}: legacy AUC {curve['legacy_auc_delta']:+.15f} "
                    f"does not match frozen summary {expected_auc:+.15f}"
                )
            curves[key] = curve
            raw_sources.append({
                "cell": key,
                "path": _display_path(path),
                "sha256": _sha256(path),
                "n_legacy_records": curve["n_legacy_records"],
                "duplicate_steps_last_record_wins": (
                    curve["duplicate_steps_last_record_wins"]
                ),
            })
        except (OSError, DataValidationError) as exc:
            validation_errors.append(str(exc))

    if validation_errors:
        return _canonicalize({
            **common,
            "status": "invalid_raw_trajectories",
            "raw_data_audit": {
                "raw_directory": _display_path(raw_dir),
                "n_expected_files": 24,
                "n_found_files": 24,
                "validation_errors": validation_errors,
            },
            "multiverse": None,
            "robustness_summary": {
                "minimum_positive_blocks": None,
                "mean_effect_range": None,
                "reason": "raw trajectories failed the frozen validation contract",
            },
        })

    multiverse = _build_multiverse(curves)
    return _canonicalize({
        **common,
        "status": "complete",
        "raw_data_audit": {
            "raw_directory": _display_path(raw_dir),
            "n_expected_files": 24,
            "n_found_files": 24,
            "all_legacy_auc_values_match_frozen_summary": True,
            "sources": raw_sources,
        },
        "multiverse": {
            "base_variants": multiverse["base_variants"],
            "leave_one_checkpoint_out_variants": (
                multiverse["leave_one_checkpoint_out_variants"]
            ),
        },
        "robustness_summary": multiverse["robustness_summary"],
    })


def _fmt(value: float | None) -> str:
    return "not estimable" if value is None else f"{value:+.5f}"


def render_report(analysis: dict[str, Any]) -> str:
    anchor = analysis["legacy_summary_anchor"]
    lines = [
        "# Frozen maze wave-2 AUC multiverse",
        "",
        f"**Status: `{analysis['status']}`.**",
        "",
        "The analysis unit is an independent seed/warm-start block. The two",
        "samplers are repeated observations inside each block and are never",
        "counted as twelve independent replicates.",
        "",
        "## Frozen summary anchor",
        "",
        "| view | positive blocks | mean MaxRL - GRPO | range |",
        "|---|---:|---:|---:|",
    ]
    views = {
        "uniform": anchor["per_sampler"]["uniform"],
        "frontier_un": anchor["per_sampler"]["frontier_un"],
        "sampler average within block": anchor["sampler_average_within_block"],
    }
    for name, row in views.items():
        lines.append(
            f"| {name} | {row['n_positive']}/{row['n_independent_blocks']} | "
            f"{row['mean']:+.5f} | [{row['minimum']:+.5f}, {row['maximum']:+.5f}] |"
        )

    if analysis["status"] != "complete":
        audit = analysis["raw_data_audit"]
        lines.extend([
            "",
            "## Why the requested robustness result cannot be computed",
            "",
            f"Only {audit.get('n_found_files', 0)}/24 required checkpoint JSONL files "
            "are present. The committed factorial JSON stores one AUC scalar plus",
            "initial/final endpoints per cell; that is not enough to reconstruct",
            "checkpoint order or contributions.",
            "",
            "Therefore simple-mean versus trapezoid, warmup inclusion, early/mid/full",
            "horizons, leave-one-checkpoint-out, minimum sign count, and effect range",
            "are **not estimable**. Reporting them from the summaries would invent data.",
            "",
            "Required next input: the 24 files",
            "`fact250_{uniform,frontier_un}_{maxrl,grpo}_s{6..11}.jsonl` from",
            "execution fork `9f7dd2e`. Re-running this script will first reproduce every",
            "legacy `cov_auc_delta` exactly, then execute the frozen specification.",
            "",
        ])
        if audit.get("validation_errors"):
            lines.extend(["### Validation errors", ""])
            lines.extend(f"- {error}" for error in audit["validation_errors"])
            lines.append("")
        lines.extend([
            "The machine-readable result enumerates every missing path and records the",
            "source checksum, so this failure is deterministic and auditable.",
        ])
        return "\n".join(lines) + "\n"

    summary = analysis["robustness_summary"]
    lines.extend([
        "",
        "## Multiverse result",
        "",
        f"The frozen grid contains {summary['n_base_variants']} base variants and "
        f"{summary['n_leave_one_checkpoint_out_variants']} leave-one-checkpoint-out "
        f"variants ({summary['n_total_variants']} total).",
        "",
        "| scope/view | minimum positive blocks | mean-effect range |",
        "|---|---:|---:|",
    ])
    for scope in ("base_variants", "base_plus_leave_one_checkpoint_out"):
        for view, row in summary[scope].items():
            lo, hi = row["mean_effect_range"]
            lines.append(
                f"| {scope} / {view} | {row['minimum_positive_blocks']}/6 | "
                f"[{lo:+.5f}, {hi:+.5f}] |"
            )
    lines.extend([
        "",
        "## Base variants",
        "",
        "| variant | uniform | frontier_un | block average |",
        "|---|---:|---:|---:|",
    ])
    for row in analysis["multiverse"]["base_variants"]:
        uniform = row["per_sampler"]["uniform"]
        frontier = row["per_sampler"]["frontier_un"]
        average = row["sampler_average_within_block"]
        lines.append(
            f"| `{row['variant_id']}` | {uniform['n_positive']}/6, "
            f"{uniform['mean']:+.5f} | {frontier['n_positive']}/6, "
            f"{frontier['mean']:+.5f} | {average['n_positive']}/6, "
            f"{average['mean']:+.5f} |"
        )
    return "\n".join(lines) + "\n"


def _json_text(analysis: dict[str, Any]) -> str:
    return json.dumps(analysis, indent=2, sort_keys=True) + "\n"


def _check_file(path: Path, expected: str) -> bool:
    return path.is_file() and path.read_text() == expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    try:
        analysis = build_analysis(args.summary, args.raw_dir)
    except DataValidationError as exc:
        print(f"fatal input error: {exc}", file=sys.stderr)
        return 2
    json_text = _json_text(analysis)
    report_text = render_report(analysis)

    if args.check:
        ok_json = _check_file(args.json_out, json_text)
        ok_report = _check_file(args.report_out, report_text)
        if not ok_json or not ok_report:
            print(
                f"output drift: json={ok_json}, report={ok_report}",
                file=sys.stderr,
            )
            return 1
    else:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text)
        args.report_out.write_text(report_text)

    if analysis["status"] != "complete":
        print(
            f"AUC multiverse failed explicitly: {analysis['status']}; "
            f"see {args.report_out}",
            file=sys.stderr,
        )
        return 2
    print(
        f"AUC multiverse complete: "
        f"{analysis['robustness_summary']['n_total_variants']} variants"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
