"""Post-hoc, non-authorizing forensic verifier for the sealed Acrobot V5B artifact.

This verifier does not repair or replace the registered V5 analyzer.  It explains
the registered analyzer's exact-comparison failure without emitting any primary
contrast, cell-outcome summary, or decision.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from frontier_rl.examples import analyze_acrobot_hindsight_v5 as frozen


SCHEMA = "curriculum-maxrl/acrobot-hindsight-v5b-forensic-verification/v1"
EXPECTED_ARTIFACT_SHA256 = (
    "c633886a121906ee2bceb03f3117e4bea5dc20ab314e43f9b702ef8d88f495ac"
)
EXPECTED_CANONICAL_FAILURE = (
    "runner V5B case diagnostic summary mismatch: lr_mult_0p5_hs_0"
)
FROZEN_ABSOLUTE_TOLERANCE = 1e-12
SOURCES = ("requested_live", "hindsight_relabel")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _ordered_float_bits(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    sign = 1 << 63
    mask = (1 << 64) - 1
    return (mask - bits) if bits & sign else (sign + bits)


def _ulp_distance(left: float, right: float) -> int:
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("ULP distance requires finite values")
    return abs(_ordered_float_bits(left) - _ordered_float_bits(right))


def _numpy_source_norms(run: dict) -> dict:
    """Reproduce the runner's frozen NumPy reduction semantics exactly."""

    output = {}
    for source in SOURCES:
        norms = np.asarray(
            [
                float(record["update_norm"])
                for record in run["update_diagnostics"]
                if record.get("source") == source
            ],
            dtype=np.float64,
        )
        output[source] = {
            "count": len(norms),
            "M": float(norms.sum()) if len(norms) else 0.0,
            "Q": float(np.dot(norms, norms)) if len(norms) else 0.0,
        }
    return output


def _mechanical_integrity(artifact: dict, lock: dict, lock_path: Path) -> tuple[dict, dict]:
    if artifact.get("schema") != frozen.SCHEMA_B or artifact.get("artifact_state") != "complete":
        raise ValueError("sealed V5B artifact is not complete")
    if artifact.get("run_failures"):
        raise ValueError("sealed V5B artifact contains run failures")
    target = int(lock.get("registered_schedule", {}).get("optimizer_update_target", -1))
    schedule = frozen._schedule_b(target)
    frozen._validate_protocol(artifact, "stage_b_confirmatory_factorial", schedule)
    lock_report = frozen._verify_lock(
        artifact, lock, lock_path, "stage_b_confirmatory_factorial"
    )
    authorization_report = frozen._verify_linked_stage_b_authorization(
        artifact, lock, target
    )
    if tuple(artifact.get("cases", {})) != frozen.CASES:
        raise ValueError("V5B case order/set mismatch")

    run_count = group_count = update_count = checkpoint_count = 0
    for multiplier in frozen.LR_MULTIPLIERS:
        for scale in frozen.HINDSIGHT_SCALES:
            name = frozen._case_name(multiplier, scale)
            record = artifact["cases"][name]
            if record.get("config") != frozen._expected_config(name, multiplier, scale):
                raise ValueError(f"V5B config mismatch for {name}")
            runs = record.get("runs", [])
            if [run.get("seed") for run in runs] != list(frozen.STAGE_B_SEEDS):
                raise ValueError(f"V5B seed order mismatch for {name}")
            for run in runs:
                label = f"{name}/seed_{run['seed']}"
                frozen._validate_run(run, label, scale, exact_target=target)
                if (
                    run.get("transition_cap_censored") is not False
                    or run.get("reached_optimizer_update_budget") is not True
                    or run.get("optimizer_updates") != target
                ):
                    raise ValueError(f"V5B terminal budget flags invalid in {label}")
                run_count += 1
                group_count += len(run["group_diagnostics"])
                update_count += len(run["update_diagnostics"])
                checkpoint_count += len(run["x_optimizer_updates"])

    if run_count != 180:
        raise ValueError(f"expected 180 registered runs, found {run_count}")
    return (
        {
            "passed": True,
            "complete_cases": len(frozen.CASES),
            "runs_per_case": len(frozen.STAGE_B_SEEDS),
            "registered_runs_validated": run_count,
            "raw_group_records_validated": group_count,
            "raw_update_records_validated": update_count,
            "evaluation_checkpoints_validated": checkpoint_count,
            "run_failures_empty": True,
            "selected_optimizer_update_budget": target,
            "frozen_protocol_validation_passed": True,
            "frozen_source_runtime_lock_validation_passed": lock_report.get("passed") is True,
            "frozen_linked_authorization_validation_passed": authorization_report.get("passed")
            is True,
        },
        {"lock": lock_report, "authorization": authorization_report},
    )


def _capture_canonical_failure(artifact_path: Path, lock_path: Path) -> dict:
    try:
        frozen.verify(artifact_path, lock_path)
    except ValueError as error:
        message = str(error)
        if message != EXPECTED_CANONICAL_FAILURE:
            raise ValueError(f"unexpected frozen-analyzer failure: {message}") from error
        return {
            "reproduced": True,
            "exception_type": type(error).__name__,
            "message": message,
        }
    raise ValueError("the original frozen analyzer unexpectedly passed")


def _reduction_forensics(artifact: dict) -> tuple[dict, dict[tuple[str, int, str], dict]]:
    total = numpy_matches = python_matches = 0
    max_absolute_difference = 0.0
    max_ulp_distance = 0
    python_by_run: dict[tuple[str, int, str], dict] = {}
    field_pairs = (
        ("cumulative_step_norm_M", "M"),
        ("cumulative_squared_step_norm_Q", "Q"),
    )
    for case_name in frozen.CASES:
        saved_by_seed = artifact["stage_b_case_summaries"][case_name][
            "source_step_norms_per_seed"
        ]
        for run in artifact["cases"][case_name]["runs"]:
            seed = int(run["seed"])
            numpy_totals = _numpy_source_norms(run)
            python_totals = frozen._source_norms(run, f"{case_name}/seed_{seed}")
            for source in SOURCES:
                saved = saved_by_seed[str(seed)][source]
                if saved.get("count") != numpy_totals[source]["count"]:
                    raise ValueError("saved runner source count differs from NumPy reconstruction")
                python_by_run[(case_name, seed, source)] = python_totals[source]
                for saved_key, total_key in field_pairs:
                    total += 1
                    saved_value = float(saved[saved_key])
                    numpy_value = float(numpy_totals[source][total_key])
                    python_value = float(python_totals[source][total_key])
                    if saved_value == numpy_value:
                        numpy_matches += 1
                    if saved_value == python_value:
                        python_matches += 1
                    difference = abs(saved_value - python_value)
                    max_absolute_difference = max(max_absolute_difference, difference)
                    max_ulp_distance = max(
                        max_ulp_distance, _ulp_distance(saved_value, python_value)
                    )

    numpy_mismatches = total - numpy_matches
    python_mismatches = total - python_matches
    if numpy_mismatches:
        raise ValueError(f"saved diagnostics differ from runner NumPy semantics in {numpy_mismatches} fields")
    within_tolerance = max_absolute_difference <= FROZEN_ABSOLUTE_TOLERANCE
    return (
        {
            "passed": True,
            "reduction_fields_checked": total,
            "saved_runner_numpy_exact_matches": numpy_matches,
            "saved_runner_numpy_exact_mismatches": numpy_mismatches,
            "analyzer_sequential_python_exact_matches": python_matches,
            "analyzer_sequential_python_exact_mismatches": python_mismatches,
            "maximum_absolute_difference": max_absolute_difference,
            "maximum_ulp_distance": max_ulp_distance,
            "frozen_numeric_absolute_tolerance": FROZEN_ABSOLUTE_TOLERANCE,
            "all_sequential_python_differences_within_frozen_numeric_tolerance": within_tolerance,
            "diagnosis": (
                "The runner saved NumPy sum/dot reductions exactly. The frozen analyzer "
                "reduced the same positive norms by sequential Python addition and then "
                "required exact dictionary equality for these diagnostic fields."
            ),
        },
        python_by_run,
    )


def _compatibility_check(
    artifact: dict, lock: dict, lock_path: Path, python_by_run: dict[tuple[str, int, str], dict]
) -> dict:
    normalized = copy.deepcopy(artifact)
    changed = 0
    for case_name in frozen.CASES:
        saved_by_seed = normalized["stage_b_case_summaries"][case_name][
            "source_step_norms_per_seed"
        ]
        for seed in frozen.STAGE_B_SEEDS:
            for source in SOURCES:
                python_totals = python_by_run[(case_name, seed, source)]
                saved = saved_by_seed[str(seed)][source]
                for saved_key, total_key in (
                    ("cumulative_step_norm_M", "M"),
                    ("cumulative_squared_step_norm_Q", "Q"),
                ):
                    value = float(python_totals[total_key])
                    changed += int(float(saved[saved_key]) != value)
                    saved[saved_key] = value

    # Invoke the complete frozen check, but retain only its top-level pass flag.
    # Primary contrast, cell-outcome, and decision subtrees are never copied into
    # or inspected by this forensic report.
    frozen_result = frozen._verify_stage_b(normalized, lock, lock_path)
    passed = frozen_result.get("all_checks_passed") is True
    frozen_result.clear()
    return {
        "performed_in_memory_only": True,
        "diagnostic_fields_normalized": 720,
        "diagnostic_values_changed": changed,
        "all_remaining_frozen_checks_passed": passed,
        "primary_outcome_subtrees_retained_or_emitted": False,
        "changes_original_artifact": False,
        "repairs_or_supersedes_registered_analyzer": False,
    }


def verify(artifact_path: Path, lock_path: Path) -> dict:
    artifact_path, lock_path = artifact_path.resolve(), lock_path.resolve()
    digest = _sha256(artifact_path)
    if digest != EXPECTED_ARTIFACT_SHA256:
        raise ValueError(
            f"forensic verifier is bound to artifact {EXPECTED_ARTIFACT_SHA256}, got {digest}"
        )
    artifact, lock = _read_json(artifact_path), _read_json(lock_path)
    mechanical, chain = _mechanical_integrity(artifact, lock, lock_path)
    canonical_failure = _capture_canonical_failure(artifact_path, lock_path)
    reductions, python_by_run = _reduction_forensics(artifact)
    compatibility = _compatibility_check(artifact, lock, lock_path, python_by_run)
    if not all(
        (
            mechanical["passed"],
            canonical_failure["reproduced"],
            reductions["passed"],
            compatibility["all_remaining_frozen_checks_passed"],
        )
    ):
        raise ValueError("forensic verification did not establish its diagnostic scope")
    report = {
        "schema": SCHEMA,
        "status": "post_hoc_non_authorizing_forensic_diagnostic",
        "artifact": str(artifact_path),
        "artifact_sha256": digest,
        "artifact_size_bytes": artifact_path.stat().st_size,
        "lock": str(lock_path),
        "lock_sha256": _sha256(lock_path),
        "registered_primary_family_authorized": False,
        "original_locked_analyzer_passed": False,
        "mechanical_integrity": mechanical,
        "canonical_locked_analyzer_failure": canonical_failure,
        "reduction_forensics": reductions,
        "compatibility_diagnostic": compatibility,
        "frozen_chain": {
            "source_runtime_lock_sha256": chain["lock"]["lock_sha256"],
            "source_files_checked": len(chain["lock"]["checked_source_files"]),
            "linked_amendment_sha256": chain["authorization"]["amendment_sha256"],
            "linked_stage_a_artifact_sha256": chain["authorization"][
                "stage_a_artifact_sha256"
            ],
            "linked_stage_a_verification_sha256": chain["authorization"][
                "stage_a_verification_sha256"
            ],
            "linked_stage_a_gates_sha256": chain["authorization"][
                "stage_a_gates_sha256"
            ],
            "passed": True,
        },
        "scope_boundary": (
            "This report diagnoses a post-hoc floating-point reduction-order mismatch. "
            "It neither authorizes the registered primary family nor reports, interprets, "
            "or rescues any primary contrast, cell outcome, or predeclared decision."
        ),
    }
    return report


def _write_exclusive(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite forensic report {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify(args.artifact, args.lock)
        if args.output is None:
            print(json.dumps(report, indent=2, allow_nan=False))
        else:
            _write_exclusive(args.output, report)
            print(f"wrote non-authorizing V5B forensic report: {args.output.resolve()}")
    except (FileNotFoundError, FileExistsError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
