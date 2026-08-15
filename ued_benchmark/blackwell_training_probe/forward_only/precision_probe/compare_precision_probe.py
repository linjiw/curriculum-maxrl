#!/usr/bin/env python3
"""Compare frozen CPU/GPU default and highest-precision LSTM forwards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


PROBE_ROOT = Path(__file__).resolve().parent
PROTOCOL = PROBE_ROOT / "PRECISION_PROTOCOL.json"
PAYLOAD = PROBE_ROOT.parent / "FORWARD_PAYLOAD.json"
PROTOCOL_SHA256 = "0abdb46a7b56986756a31f3d4cc1793af20fc6ca53d2b397720386aab7f5b820"
PAYLOAD_SHA256 = "845a34ae40fb762e72b4c6ec569ef16ab6531b241eeaf6cecbc0523059f3bc78"
RUN_ROOT = Path("/data/robotixx/ued_bench/runs/blackwell_precision_0abdb46a")


class CompareError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CompareError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_capture(directory: Path, backend: str) -> tuple[Path, dict[str, Any]]:
    directory = directory.resolve()
    require(directory.is_dir() and not directory.is_symlink(), "unsafe capture directory")
    path = directory / "capture.json"
    require(path.is_file() and not path.is_symlink(), "missing capture receipt")
    capture = json.loads(path.read_text())
    require(capture["schema_version"] == 1, "capture schema drift")
    require(capture["status"] == "captured_default_and_highest_forward_only", "bad capture")
    require(capture["lane"] == "modern" and capture["backend"] == backend, "lane drift")
    require(capture["hashes"]["protocol_sha256"] == PROTOCOL_SHA256, "protocol drift")
    require(capture["hashes"]["payload_sha256"] == PAYLOAD_SHA256, "payload drift")
    for field in (
        "training_steps",
        "experiment_step_calls",
        "agent_update_calls",
        "gradient_calculations",
        "gradient_transformation_proposals",
        "optimizer_applications",
        "parameter_mutations",
    ):
        require(capture[field] == 0, f"forbidden execution count: {field}")
    require(capture["rng_consumed"] is False, "RNG unexpectedly consumed")
    require(capture["payload"]["unchanged"] is True, "parameter mutation")
    require(
        capture["payload"]["parameter_sha256_before"]
        == capture["payload"]["parameter_sha256_after"],
        "parameter digest drift",
    )
    require(capture["record_count"] == len(capture["records"]), "record count drift")
    for index, record in enumerate(capture["records"]):
        require(record["index"] == index, "record ordering drift")
        relative = Path(record["file"])
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe array path")
        array_path = directory / relative
        require(array_path.is_file() and not array_path.is_symlink(), "missing array")
        require(sha256(array_path) == record["file_sha256"], "array digest drift")
    return path, capture


def structure_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record["stage"],
        record["label"],
        record["path"],
        tuple(record["shape"]),
        record["dtype"],
        record["size"],
    )


def compare_record(
    np: Any,
    reference_root: Path,
    candidate_root: Path,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    exact: bool,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    require(structure_key(reference) == structure_key(candidate), "record structure drift")
    left = np.load(reference_root / reference["file"], allow_pickle=False)
    right = np.load(candidate_root / candidate["file"], allow_pickle=False)
    require(left.shape == right.shape and left.dtype == right.dtype, "array structure drift")
    exact_bytes = reference["raw_sha256"] == candidate["raw_sha256"]
    result: dict[str, Any] = {
        "stage": reference["stage"],
        "label": reference["label"],
        "path": reference["path"],
        "shape": reference["shape"],
        "dtype": reference["dtype"],
        "required_exact": exact,
        "exact_bytes": exact_bytes,
        "passed": True,
        "element_failure_count": 0,
        "aggregate_failures": [],
        "max_element_absolute_error": 0.0,
        "max_element_relative_error": 0.0,
    }
    if exact:
        if not exact_bytes:
            unequal = np.not_equal(left, right)
            result["passed"] = False
            result["element_failure_count"] = int(unequal.sum())
            differing = np.flatnonzero(unequal.reshape(-1))
            if differing.size:
                flat = int(differing[0])
                result["first_failure_index"] = [
                    int(value) for value in np.unravel_index(flat, left.shape)
                ]
        return result
    finite = np.isfinite(left) & np.isfinite(right)
    sentinels_match = (
        np.array_equal(np.isnan(left), np.isnan(right))
        and np.array_equal(np.isposinf(left), np.isposinf(right))
        and np.array_equal(np.isneginf(left), np.isneginf(right))
    )
    close = np.ones(left.shape, dtype=np.bool_)
    close[finite] = np.isclose(left[finite], right[finite], rtol=rtol, atol=atol)
    close &= np.isnan(left) == np.isnan(right)
    close &= np.isposinf(left) == np.isposinf(right)
    close &= np.isneginf(left) == np.isneginf(right)
    failures = ~close
    result["element_failure_count"] = int(failures.sum())
    if finite.any():
        absolute = np.abs(
            left[finite].astype(np.float64, copy=False)
            - right[finite].astype(np.float64, copy=False)
        )
        relative = absolute / np.maximum(
            np.abs(left[finite].astype(np.float64, copy=False)), atol
        )
        result["max_element_absolute_error"] = float(absolute.max(initial=0.0))
        result["max_element_relative_error"] = float(relative.max(initial=0.0))
    for metric in ("abs_sum", "squared_l2", "max_abs"):
        left_metric = float(reference[metric])
        right_metric = float(candidate[metric])
        if not bool(np.isclose(left_metric, right_metric, rtol=rtol, atol=atol)):
            absolute_error = abs(left_metric - right_metric)
            result["aggregate_failures"].append(
                {
                    "metric": metric,
                    "reference": left_metric,
                    "candidate": right_metric,
                    "absolute_error": absolute_error,
                    "relative_error": absolute_error / max(abs(left_metric), atol),
                }
            )
    result["sentinel_match"] = sentinels_match
    result["passed"] = (
        sentinels_match
        and result["element_failure_count"] == 0
        and not result["aggregate_failures"]
    )
    if result["element_failure_count"]:
        flat = int(np.flatnonzero(failures.reshape(-1))[0])
        result["first_failure_index"] = [
            int(value) for value in np.unravel_index(flat, left.shape)
        ]
    return result


def within_backend_precision_delta(
    np: Any,
    root: Path,
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    pairs = (
        ("lstm_input_affine", "default_lstm_input_affine", "highest_lstm_input_affine"),
        ("lstm_hidden_affine", "default_lstm_hidden_affine", "highest_lstm_hidden_affine"),
        (
            "lstm_gate_preactivation",
            "default_lstm_gate_preactivation",
            "highest_lstm_gate_preactivation",
        ),
        ("lstm_gate_activation", "default_lstm_gate_activation", "highest_lstm_gate_activation"),
        ("lstm_cell_state", "default_lstm_cell_state", "highest_lstm_cell_state"),
        ("lstm_hidden_state", "default_lstm_hidden_state", "highest_lstm_hidden_state"),
        ("final_carry", "default_final_carry", "highest_final_carry"),
    )
    by_key = {
        (record["stage"], record["label"], record["path"]): record
        for record in capture["records"]
    }
    result: dict[str, Any] = {}
    for name, default_stage, highest_stage in pairs:
        default_records = [record for record in capture["records"] if record["stage"] == default_stage]
        stage_max = 0.0
        exact_records = 0
        record_count = 0
        for default_record in default_records:
            highest_record = by_key.get(
                (highest_stage, default_record["label"], default_record["path"])
            )
            require(highest_record is not None, f"missing precision pair: {name}")
            left = np.load(root / default_record["file"], allow_pickle=False)
            right = np.load(root / highest_record["file"], allow_pickle=False)
            require(left.shape == right.shape and left.dtype == right.dtype, "precision pair drift")
            stage_max = max(stage_max, float(np.max(np.abs(left - right), initial=0.0)))
            exact_records += int(default_record["raw_sha256"] == highest_record["raw_sha256"])
            record_count += 1
        result[name] = {
            "record_count": record_count,
            "exact_record_count": exact_records,
            "max_element_absolute_difference": stage_max,
        }
    return result


def compare(args: argparse.Namespace) -> dict[str, Any]:
    require(not os.environ.get("PYTHONPATH"), "PYTHONPATH must be unset")
    require(sha256(PROTOCOL) == PROTOCOL_SHA256, "precision protocol drift")
    require(sha256(PAYLOAD) == PAYLOAD_SHA256, "payload drift")
    protocol = json.loads(PROTOCOL.read_text())
    output = args.output.resolve()
    require(output.is_relative_to(RUN_ROOT.resolve()), "output outside precision run root")
    require(not output.exists(), "output already exists")
    require(output.parent.is_dir() and not output.parent.is_symlink(), "unsafe output parent")
    reference_path, reference = load_capture(args.reference, "cpu")
    candidate_path, candidate = load_capture(args.candidate, args.backend)
    require(
        reference["hashes"]["capture_script_sha256"]
        == candidate["hashes"]["capture_script_sha256"],
        "capture script drift",
    )
    require(reference["groups"] == candidate["groups"], "capture group drift")
    require(reference["record_count"] == candidate["record_count"], "record count drift")

    import numpy as np

    tolerance = protocol["comparison"][args.backend]
    rtol = float(tolerance["rtol"])
    atol = float(tolerance["atol"])
    exact_dtypes = set(protocol["comparison"]["exact_dtypes"])
    contracts = {item["name"]: item for item in protocol["ordered_stages"]}
    stages: dict[str, Any] = {
        item["name"]: {
            "mode": item["mode"],
            "status": "pass",
            "record_count": 0,
            "exact_record_count": 0,
            "failure_count": 0,
            "max_element_absolute_error": 0.0,
            "max_element_relative_error": 0.0,
            "failures": [],
        }
        for item in protocol["ordered_stages"]
    }
    reference_root = args.reference.resolve()
    candidate_root = args.candidate.resolve()
    for left, right in zip(reference["records"], candidate["records"]):
        require(structure_key(left) == structure_key(right), "record ordering drift")
        stage = left["stage"]
        require(stage in contracts, f"undeclared stage: {stage}")
        exact = (
            stage == "input_payload"
            or contracts[stage]["comparison"] == "exact"
            or left["dtype"] in exact_dtypes
        )
        record_result = compare_record(
            np,
            reference_root,
            candidate_root,
            left,
            right,
            exact=exact,
            rtol=rtol,
            atol=atol,
        )
        summary = stages[stage]
        summary["record_count"] += 1
        summary["exact_record_count"] += int(record_result["exact_bytes"])
        summary["max_element_absolute_error"] = max(
            summary["max_element_absolute_error"], record_result["max_element_absolute_error"]
        )
        summary["max_element_relative_error"] = max(
            summary["max_element_relative_error"], record_result["max_element_relative_error"]
        )
        if not record_result["passed"]:
            summary["status"] = "fail"
            summary["failure_count"] += 1
            summary["failures"].append(record_result)

    ordered = [item["name"] for item in protocol["ordered_stages"]]
    mode_results: dict[str, Any] = {}
    for mode in ("shared", "default", "highest", "canonical_default"):
        names = [name for name in ordered if stages[name]["mode"] == mode]
        earliest = next((name for name in names if stages[name]["status"] == "fail"), None)
        mode_results[mode] = {
            "status": "pass" if earliest is None else "fail",
            "earliest_failing_stage": earliest,
            "stages": {name: stages[name]["status"] for name in names},
        }
    closure_contract = protocol["closure_gate"]
    required_highest = closure_contract["required_highest_cross_backend_stages"]
    highest_required_pass = all(stages[name]["status"] == "pass" for name in required_highest)
    default_input_failure = stages["default_lstm_input_affine"]["status"] == "fail"
    default_canonical_failure = stages["canonical_default_output"]["status"] == "fail"
    input_affine_closed = (
        default_input_failure and stages["highest_lstm_input_affine"]["status"] == "pass"
    )
    carry_closed = (
        stages["default_final_carry"]["status"] == "fail"
        and stages["highest_final_carry"]["status"] == "pass"
    )
    closure_pass = (
        highest_required_pass
        and default_input_failure
        and default_canonical_failure
        and input_affine_closed
        and carry_closed
    )
    all_stages_pass = all(summary["status"] == "pass" for summary in stages.values())
    if args.backend == "cpu":
        status = "cpu_selfcheck_pass" if all_stages_pass else "fail_closed_cpu_selfcheck"
        closure_evaluated = False
        future_reconsideration = False
    else:
        status = (
            "precision_closure_pass_training_gate_closed"
            if closure_pass
            else "fail_closed_precision_did_not_close"
        )
        closure_evaluated = True
        future_reconsideration = closure_pass
    result = {
        "schema_version": 1,
        "status": status,
        "backend_compared": args.backend,
        "paper_evidence": False,
        "performance_endpoint": False,
        "training_steps": 0,
        "optimizer_applications": 0,
        "parameter_mutations": 0,
        "tolerances": {"rtol": rtol, "atol": atol},
        "tolerance_relaxed": False,
        "reference": {"capture": str(reference_path), "sha256": sha256(reference_path)},
        "candidate": {"capture": str(candidate_path), "sha256": sha256(candidate_path)},
        "hashes": {
            "protocol_sha256": PROTOCOL_SHA256,
            "payload_sha256": PAYLOAD_SHA256,
            "capture_script_sha256": reference["hashes"]["capture_script_sha256"],
            "compare_script_sha256": sha256(Path(__file__).resolve()),
        },
        "ordered_stages": ordered,
        "stages": stages,
        "mode_results": mode_results,
        "within_backend_default_vs_highest": {
            "cpu": within_backend_precision_delta(np, reference_root, reference),
            args.backend: within_backend_precision_delta(np, candidate_root, candidate),
        },
        "closure": {
            "evaluated": closure_evaluated,
            "highest_required_stages_pass": highest_required_pass,
            "default_input_affine_failure_reproduced": default_input_failure,
            "default_canonical_carry_failure_reproduced": default_canonical_failure,
            "input_affine_discrepancy_closed": input_affine_closed,
            "final_carry_discrepancy_closed": carry_closed,
            "closure_pass": closure_pass,
            "future_training_gate_may_be_reconsidered": future_reconsideration,
            "training_gate_open": False,
            "meaning": closure_contract["passing_effect"],
        },
        "training_gate_open": False,
        "next_action": (
            "CPU gate complete; GPU closure not evaluated"
            if args.backend == "cpu" and all_stages_pass
            else (
                "consider a source-level highest-precision dot patch under a new compatibility protocol"
                if closure_pass
                else "hold source changes and training; highest precision did not close parity"
            )
        ),
    }
    atomic_json(output, result)
    print(
        "PRECISION_COMPARE_"
        f"{'CPU_SELFCHECK_PASS' if args.backend == 'cpu' and all_stages_pass else ('CLOSURE_PASS_TRAINING_CLOSED' if closure_pass else 'FAIL_CLOSED')} "
        f"backend={args.backend} default={mode_results['default']['status']} "
        f"highest={mode_results['highest']['status']}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        compare(args)
    except (
        CompareError,
        AssertionError,
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"PRECISION_COMPARE_ERROR: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
