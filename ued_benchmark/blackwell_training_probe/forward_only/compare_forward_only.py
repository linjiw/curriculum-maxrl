#!/usr/bin/env python3
"""Compare frozen forward-only captures in operation order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


PROBE_ROOT = Path(__file__).resolve().parent
PROTOCOL = PROBE_ROOT / "FORWARD_ONLY_PROTOCOL.json"
PAYLOAD = PROBE_ROOT / "FORWARD_PAYLOAD.json"
PROTOCOL_SHA256 = "024239a6b659097198a6d902b1bb63698849d38e340ac033fa21537b0e5888ce"
PAYLOAD_SHA256 = "845a34ae40fb762e72b4c6ec569ef16ab6531b241eeaf6cecbc0523059f3bc78"
RUN_ROOT = Path("/data/robotixx/ued_bench/runs/blackwell_forward_only_024239a6")


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


def load_capture(directory: Path, expected_backend: str) -> tuple[Path, dict[str, Any]]:
    directory = directory.resolve()
    require(directory.is_dir() and not directory.is_symlink(), "unsafe capture directory")
    path = directory / "capture.json"
    require(path.is_file() and not path.is_symlink(), "missing capture receipt")
    capture = json.loads(path.read_text())
    require(capture["schema_version"] == 1, "capture schema drift")
    require(capture["status"] == "captured_forward_only_without_training", "capture failed")
    require(capture["lane"] == "modern", "capture lane drift")
    require(capture["backend"] == expected_backend, "capture backend drift")
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
    require(capture["rng_consumed"] is False, "forward unexpectedly consumed RNG")
    require(capture["payload"]["unchanged"] is True, "parameter digest drift")
    require(
        capture["payload"]["parameter_sha256_before"]
        == capture["payload"]["parameter_sha256_after"],
        "parameters changed",
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
    require(left.shape == right.shape and left.dtype == right.dtype, "loaded structure drift")
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
                result["first_failure_flat_index"] = flat
                result["first_failure_index"] = [
                    int(value) for value in np.unravel_index(flat, left.shape)
                ]
                result["reference_value"] = left.reshape(-1)[flat].item()
                result["candidate_value"] = right.reshape(-1)[flat].item()
        return result

    left_nan = np.isnan(left)
    right_nan = np.isnan(right)
    left_posinf = np.isposinf(left)
    right_posinf = np.isposinf(right)
    left_neginf = np.isneginf(left)
    right_neginf = np.isneginf(right)
    sentinels_match = (
        np.array_equal(left_nan, right_nan)
        and np.array_equal(left_posinf, right_posinf)
        and np.array_equal(left_neginf, right_neginf)
    )
    finite = np.isfinite(left) & np.isfinite(right)
    close = np.ones(left.shape, dtype=np.bool_)
    close[finite] = np.isclose(left[finite], right[finite], rtol=rtol, atol=atol)
    close &= left_nan == right_nan
    close &= left_posinf == right_posinf
    close &= left_neginf == right_neginf
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
        index = [int(value) for value in np.unravel_index(flat, left.shape)]
        result["first_failure_flat_index"] = flat
        result["first_failure_index"] = index
        result["time_index"] = index[0] if len(index) >= 3 else None
        result["reference_value"] = left.reshape(-1)[flat].item()
        result["candidate_value"] = right.reshape(-1)[flat].item()
    return result


def input_gemm_diagnostic(
    np: Any,
    reference_root: Path,
    candidate_root: Path,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Bound the effect of feature drift before attributing the input GEMM."""
    def load_selected(
        root: Path,
        capture: Mapping[str, Any],
        stage: str,
        label: str,
        path: str,
    ) -> Any:
        matches = [
            record
            for record in capture["records"]
            if record["stage"] == stage
            and record["label"] == label
            and record["path"] == path
        ]
        require(len(matches) == 1, f"selected diagnostic record drift: {stage}/{path}")
        return np.load(root / matches[0]["file"], allow_pickle=False)

    cpu_features = load_selected(
        reference_root,
        reference,
        "concatenated_features",
        "visual_plus_scalar",
        "",
    )
    gpu_features = load_selected(
        candidate_root,
        candidate,
        "concatenated_features",
        "visual_plus_scalar",
        "",
    )
    feature_delta = gpu_features.astype(np.float64) - cpu_features.astype(np.float64)
    result: dict[str, Any] = {
        "method": (
            "float64 propagation of the captured feature delta through the exact "
            "frozen kernel; read-only counterfactual"
        ),
        "feature_max_absolute_error": float(np.max(np.abs(feature_delta), initial=0.0)),
        "per_gate": {},
    }
    dominance_checks = []
    kernel_paths = {
        "f": "['params']['params']['rnn']['OptimizedLSTMCell_0']['if']['kernel']",
        "g": "['params']['params']['rnn']['OptimizedLSTMCell_0']['ig']['kernel']",
        "i": "['params']['params']['rnn']['OptimizedLSTMCell_0']['ii']['kernel']",
        "o": "['params']['params']['rnn']['OptimizedLSTMCell_0']['io']['kernel']",
    }
    for gate, kernel_path in kernel_paths.items():
        kernel = load_selected(
            reference_root,
            reference,
            "input_payload",
            "exact_inputs",
            kernel_path,
        )
        cpu_affine = load_selected(
            reference_root,
            reference,
            "lstm_input_affine",
            "concatenated_input_dot",
            f"['{gate}']",
        )
        gpu_affine = load_selected(
            candidate_root,
            candidate,
            "lstm_input_affine",
            "concatenated_input_dot",
            f"['{gate}']",
        )
        propagated_feature_delta = np.matmul(feature_delta, kernel.astype(np.float64))
        observed_delta = gpu_affine.astype(np.float64) - cpu_affine.astype(np.float64)
        residual = observed_delta - propagated_feature_delta
        propagated_max = float(np.max(np.abs(propagated_feature_delta), initial=0.0))
        observed_max = float(np.max(np.abs(observed_delta), initial=0.0))
        residual_max = float(np.max(np.abs(residual), initial=0.0))
        cpu_float32_reconstruction = np.matmul(cpu_features, kernel)
        cpu_reconstruction_error = float(
            np.max(np.abs(cpu_float32_reconstruction - cpu_affine), initial=0.0)
        )
        ratio = observed_max / max(propagated_max, np.finfo(np.float64).tiny)
        dominates = observed_max > 100.0 * propagated_max
        dominance_checks.append(dominates)
        result["per_gate"][gate] = {
            "propagated_feature_delta_max_abs": propagated_max,
            "observed_affine_delta_max_abs": observed_max,
            "residual_after_feature_delta_max_abs": residual_max,
            "observed_to_propagated_ratio": ratio,
            "cpu_float32_reconstruction_max_abs_error": cpu_reconstruction_error,
            "backend_gemm_arithmetic_dominates_feature_delta": dominates,
        }
    result["all_gate_dominance_checks_pass"] = all(dominance_checks)
    result["inference"] = (
        "default_precision_lstm_input_gemm_backend_arithmetic_dominates_feature_perturbation"
        if result["all_gate_dominance_checks_pass"]
        else "feature_perturbation_not_ruled_out"
    )
    return result


def compare(args: argparse.Namespace) -> dict[str, Any]:
    require(not os.environ.get("PYTHONPATH"), "PYTHONPATH must be unset")
    require(sha256(PROTOCOL) == PROTOCOL_SHA256, "forward protocol drift")
    require(sha256(PAYLOAD) == PAYLOAD_SHA256, "forward payload drift")
    protocol = json.loads(PROTOCOL.read_text())
    output = args.output.resolve()
    require(output.is_relative_to(RUN_ROOT.resolve()), "output outside forward run root")
    require(not output.exists(), "output already exists")
    require(output.parent.is_dir() and not output.parent.is_symlink(), "unsafe output parent")
    reference_path, reference = load_capture(args.reference, "cpu")
    candidate_path, candidate = load_capture(args.candidate, args.backend)
    require(
        reference["hashes"]["capture_script_sha256"]
        == candidate["hashes"]["capture_script_sha256"],
        "capture script drift",
    )
    require(reference["groups"] == candidate["groups"], "capture group structure drift")
    require(reference["record_count"] == candidate["record_count"], "record count drift")
    if args.backend == "cpu":
        require(
            reference["cpu_reproduction_gates"]
            == {
                "canonical_matches_frozen_capture": True,
                "manual_final_carry_matches_canonical": True,
            },
            "CPU reproduction gates not satisfied",
        )

    import numpy as np

    tolerance = protocol["comparison"][args.backend]
    rtol = float(tolerance["rtol"])
    atol = float(tolerance["atol"])
    exact_dtypes = set(protocol["comparison"]["exact_dtypes"])
    ordered = protocol["ordered_stages"]
    contracts = {item["name"]: item for item in ordered}
    stages: dict[str, Any] = {
        item["name"]: {
            "status": "pass",
            "classification": item["classification_on_first_failure"],
            "record_count": 0,
            "exact_record_count": 0,
            "failure_count": 0,
            "max_element_absolute_error": 0.0,
            "max_element_relative_error": 0.0,
            "failures": [],
        }
        for item in ordered
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
            summary["max_element_absolute_error"],
            record_result["max_element_absolute_error"],
        )
        summary["max_element_relative_error"] = max(
            summary["max_element_relative_error"],
            record_result["max_element_relative_error"],
        )
        if not record_result["passed"]:
            summary["status"] = "fail"
            summary["failure_count"] += 1
            summary["failures"].append(record_result)

    ordered_names = [item["name"] for item in ordered]
    earliest_stage = next(
        (name for name in ordered_names if stages[name]["status"] == "fail"),
        None,
    )
    if earliest_stage is None:
        localization = {
            "earliest_failing_stage": None,
            "classification": "no_divergence_within_frozen_tolerances",
            "tensor": None,
            "time_index": None,
        }
        status = "pass_diagnostic_only"
    else:
        first_failure = stages[earliest_stage]["failures"][0]
        localization = {
            "earliest_failing_stage": earliest_stage,
            "classification": contracts[earliest_stage]["classification_on_first_failure"],
            "tensor": {
                "label": first_failure["label"],
                "path": first_failure["path"],
                "shape": first_failure["shape"],
                "dtype": first_failure["dtype"],
            },
            "time_index": first_failure.get("time_index"),
            "first_failure_index": first_failure.get("first_failure_index"),
            "element_failure_count": first_failure["element_failure_count"],
            "max_element_absolute_error": first_failure["max_element_absolute_error"],
            "aggregate_failures": first_failure["aggregate_failures"],
        }
        status = "fail_closed"

    gemm_diagnostic = (
        input_gemm_diagnostic(
            np,
            reference_root,
            candidate_root,
            reference,
            candidate,
        )
        if args.backend == "gpu" and earliest_stage == "lstm_input_affine"
        else {"status": "not_applicable"}
    )

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
        "reference": {
            "capture": str(reference_path),
            "capture_sha256": sha256(reference_path),
        },
        "candidate": {
            "capture": str(candidate_path),
            "capture_sha256": sha256(candidate_path),
        },
        "hashes": {
            "protocol_sha256": PROTOCOL_SHA256,
            "payload_sha256": PAYLOAD_SHA256,
            "capture_script_sha256": reference["hashes"]["capture_script_sha256"],
            "compare_script_sha256": sha256(Path(__file__).resolve()),
        },
        "ordered_stages": ordered_names,
        "stages": stages,
        "localization": localization,
        "read_only_input_gemm_diagnostic": gemm_diagnostic,
        "training_gate_open": False,
        "next_action": (
            "none; forward diagnostic passed but prior training gate remains closed"
            if earliest_stage is None
            else f"hold training; earliest operation is {localization['classification']}"
        ),
    }
    atomic_json(output, result)
    print(
        "FORWARD_ONLY_COMPARE_"
        f"{'PASS_DIAGNOSTIC_ONLY' if earliest_stage is None else 'FAIL_CLOSED'} "
        f"backend={args.backend} earliest={earliest_stage} "
        f"classification={localization['classification']}"
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
        print(f"FORWARD_ONLY_COMPARE_ERROR: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
