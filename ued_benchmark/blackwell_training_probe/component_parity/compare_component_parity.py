#!/usr/bin/env python3
"""Compare non-updating Frontier PPO captures in frozen stage order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


PROTOCOL = Path(__file__).resolve().with_name("COMPONENT_PARITY_PROTOCOL.json")
PROTOCOL_SHA256 = "0f8c083202a189ec234f32c0e1c15e7c09753892fb05af0d6262b9ff0bf9f1a5"
RUN_ROOT = Path("/data/robotixx/ued_bench/runs")


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


def load_capture(directory: Path, expected_lane: str, expected_backend: str) -> tuple[Path, dict]:
    directory = directory.resolve()
    require(directory.is_dir() and not directory.is_symlink(), "unsafe capture directory")
    capture_path = directory / "capture.json"
    require(capture_path.is_file() and not capture_path.is_symlink(), "missing capture receipt")
    capture = json.loads(capture_path.read_text())
    require(capture["schema_version"] == 1, "capture schema drift")
    require(capture["status"] == "captured_without_optimizer_application", "capture failed")
    require(capture["lane"] == expected_lane, "capture lane drift")
    require(capture["backend"] == expected_backend, "capture backend drift")
    require(capture["hashes"]["protocol_sha256"] == PROTOCOL_SHA256, "protocol drift")
    require(capture["optimizer_applications"] == 0, "capture applied an optimizer")
    require(capture["parameter_mutations"] == 0, "capture mutated parameters")
    require(capture["cycle_two_experiment_steps"] == 0, "cycle-two step was called")
    require(capture["cycle_two_agent_updates"] == 0, "cycle-two agent update was called")
    require(capture["gradient_transformation_proposals"] == 1, "proposal count drift")
    require(capture["parameters"]["unchanged"] is True, "parameter digest drift")
    require(capture["record_count"] == len(capture["records"]), "record-count drift")
    for expected_index, record in enumerate(capture["records"]):
        require(record["index"] == expected_index, "record index drift")
        relative = Path(record["file"])
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe array path")
        array_path = directory / relative
        require(array_path.is_file() and not array_path.is_symlink(), "missing array")
        require(sha256(array_path) == record["file_sha256"], "array file digest drift")
    return capture_path, capture


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
    require(left.shape == right.shape, "loaded shape drift")
    require(left.dtype == right.dtype, "loaded dtype drift")
    exact_bytes = reference["raw_sha256"] == candidate["raw_sha256"]
    result: dict[str, Any] = {
        "stage": reference["stage"],
        "label": reference["label"],
        "path": reference["path"],
        "dtype": reference["dtype"],
        "shape": reference["shape"],
        "exact_bytes": exact_bytes,
        "required_exact": exact,
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
                first = int(differing[0])
                result["first_failure_flat_index"] = first
                result["reference_value"] = left.reshape(-1)[first].item()
                result["candidate_value"] = right.reshape(-1)[first].item()
        return result

    left_nan = np.isnan(left)
    right_nan = np.isnan(right)
    left_posinf = np.isposinf(left)
    right_posinf = np.isposinf(right)
    left_neginf = np.isneginf(left)
    right_neginf = np.isneginf(right)
    sentinel_match = (
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
    result["sentinel_match"] = sentinel_match
    result["passed"] = (
        sentinel_match
        and result["element_failure_count"] == 0
        and not result["aggregate_failures"]
    )
    if result["element_failure_count"]:
        first = int(np.flatnonzero(failures.reshape(-1))[0])
        result["first_failure_flat_index"] = first
        result["reference_value"] = left.reshape(-1)[first].item()
        result["candidate_value"] = right.reshape(-1)[first].item()
    return result


def causal_classification(stage_results: Mapping[str, Any], first_stage: str | None) -> str:
    if first_stage is None:
        return "no_divergence_within_frozen_tolerances"
    if first_stage == "cycle_one_control":
        failures = stage_results[first_stage]["failures"]
        if failures and all(
            failure["label"] == "environment_state"
            and failure["path"].startswith("[3]")
            for failure in failures
        ):
            # cycle_one_state[2:] index three is the recurrent carry. It is
            # produced by the LSTM forward pass, not by environment dynamics.
            return "forward_or_gemm_recurrent_carry"
    if first_stage == "rollout_action_stream":
        model_failures = [
            failure
            for failure in stage_results["ppo_forward"]["failures"]
            if failure["label"] == "model"
        ]
        old_value_failures = [
            failure
            for failure in stage_results["rollout_forward_stream"]["failures"]
            if failure["label"] == "old_values"
        ]
        if model_failures or old_value_failures:
            return "forward_or_gemm_with_downstream_sampling_divergence"
        return "sampling"
    return stage_results[first_stage]["classification"]


def first_step_adam_proposal(np: Any, clipped_gradient: Any) -> Any:
    """Return the zero-state Adam proposal from the post-clipping gradient."""
    return (
        -np.float32(0.0003)
        * clipped_gradient
        / (np.abs(clipped_gradient) + np.float32(1e-5))
    )


def first_step_adam_diagnostic(
    np: Any,
    raw_gradient: Any,
    clipped_gradient: Any,
    proposed_update: Any,
    clip_factor: Any,
) -> dict[str, Any]:
    """Bind the Adam formula to the captured output of gradient clipping."""
    expected_clipped_gradient = raw_gradient * np.asarray(
        clip_factor, dtype=raw_gradient.dtype
    )
    clipping_error = float(
        np.max(
            np.abs(expected_clipped_gradient - clipped_gradient),
            initial=0.0,
        )
    )
    expected_update = first_step_adam_proposal(np, clipped_gradient)
    formula_error = float(
        np.max(np.abs(expected_update - proposed_update), initial=0.0)
    )
    return {
        "analytic_input": "captured_clipped_gradient_tree",
        "captured_clipping_matches_raw_times_factor": bool(
            np.allclose(
                expected_clipped_gradient,
                clipped_gradient,
                rtol=1e-6,
                atol=1e-7,
            )
        ),
        "captured_clipping_max_abs_error": clipping_error,
        "first_step_adam_formula_matches": formula_error <= 1e-9,
        "analytic_adam_max_abs_error": formula_error,
    }


def selected_adam_diagnostics(
    np: Any,
    reference_root: Path,
    candidate_root: Path,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    def load_selected(
        root: Path,
        capture: Mapping[str, Any],
        stage: str,
        label: str,
        path: str,
    ) -> tuple[Any, Mapping[str, Any]]:
        matches = [
            record
            for record in capture["records"]
            if record["stage"] == stage
            and record["label"] == label
            and record["path"] == path
        ]
        require(len(matches) == 1, f"selected record missing or duplicated: {stage}/{path}")
        record = matches[0]
        return np.load(root / record["file"], allow_pickle=False), record

    diagnostics: dict[str, Any] = {}
    formula_matches = True
    clipping_checks_pass = True
    ref_clip_factor, _ref_clip_factor_record = load_selected(
        reference_root,
        reference,
        "clipping_and_global_norm",
        "scalars",
        "['clip_factor']",
    )
    gpu_clip_factor, _gpu_clip_factor_record = load_selected(
        candidate_root,
        candidate,
        "clipping_and_global_norm",
        "scalars",
        "['clip_factor']",
    )
    for short_name, path in (
        ("fc_pi_1_bias", "['params']['fc_pi_1']['bias']"),
        ("fc_pi_1_kernel", "['params']['fc_pi_1']['kernel']"),
    ):
        ref_gradient, ref_gradient_record = load_selected(
            reference_root, reference, "unclipped_gradients", "gradient_tree", path
        )
        gpu_gradient, gpu_gradient_record = load_selected(
            candidate_root, candidate, "unclipped_gradients", "gradient_tree", path
        )
        ref_clipped_gradient, ref_clipped_gradient_record = load_selected(
            reference_root,
            reference,
            "clipping_and_global_norm",
            "clipped_gradient_tree",
            path,
        )
        gpu_clipped_gradient, gpu_clipped_gradient_record = load_selected(
            candidate_root,
            candidate,
            "clipping_and_global_norm",
            "clipped_gradient_tree",
            path,
        )
        ref_update, ref_update_record = load_selected(
            reference_root, reference, "adam_proposal", "parameter_update_tree", path
        )
        gpu_update, gpu_update_record = load_selected(
            candidate_root, candidate, "adam_proposal", "parameter_update_tree", path
        )
        ref_formula = first_step_adam_diagnostic(
            np,
            ref_gradient,
            ref_clipped_gradient,
            ref_update,
            ref_clip_factor,
        )
        gpu_formula = first_step_adam_diagnostic(
            np,
            gpu_gradient,
            gpu_clipped_gradient,
            gpu_update,
            gpu_clip_factor,
        )
        ref_formula_error = ref_formula["analytic_adam_max_abs_error"]
        gpu_formula_error = gpu_formula["analytic_adam_max_abs_error"]
        path_formula_matches = (
            ref_formula["first_step_adam_formula_matches"]
            and gpu_formula["first_step_adam_formula_matches"]
        )
        path_clipping_checks_pass = (
            ref_formula["captured_clipping_matches_raw_times_factor"]
            and gpu_formula["captured_clipping_matches_raw_times_factor"]
        )
        formula_matches &= path_formula_matches
        clipping_checks_pass &= path_clipping_checks_pass
        gradient_abs_difference = abs(
            float(ref_gradient_record["abs_sum"])
            - float(gpu_gradient_record["abs_sum"])
        )
        update_abs_difference = abs(
            float(ref_update_record["abs_sum"])
            - float(gpu_update_record["abs_sum"])
        )
        diagnostics[short_name] = {
            "path": path,
            "cpu_gradient_abs_sum": float(ref_gradient_record["abs_sum"]),
            "gpu_gradient_abs_sum": float(gpu_gradient_record["abs_sum"]),
            "gradient_abs_sum_difference": gradient_abs_difference,
            "cpu_clipped_gradient_abs_sum": float(
                ref_clipped_gradient_record["abs_sum"]
            ),
            "gpu_clipped_gradient_abs_sum": float(
                gpu_clipped_gradient_record["abs_sum"]
            ),
            "cpu_proposed_update_abs_sum": float(ref_update_record["abs_sum"]),
            "gpu_proposed_update_abs_sum": float(gpu_update_record["abs_sum"]),
            "proposed_update_abs_sum_difference": update_abs_difference,
            "difference_amplification": (
                update_abs_difference / max(gradient_abs_difference, 1e-300)
            ),
            "cpu_analytic_adam_max_abs_error": ref_formula_error,
            "gpu_analytic_adam_max_abs_error": gpu_formula_error,
            "first_step_adam_formula_matches": path_formula_matches,
            "analytic_input": "captured_clipped_gradient_tree",
            "captured_clipping_matches_raw_times_factor": path_clipping_checks_pass,
            "cpu_captured_clipping_max_abs_error": ref_formula[
                "captured_clipping_max_abs_error"
            ],
            "gpu_captured_clipping_max_abs_error": gpu_formula[
                "captured_clipping_max_abs_error"
            ],
        }
    diagnostics["conclusion"] = (
        "upstream_numerical_drift_amplified_by_adam"
        if formula_matches
        else "adam_proposal_arithmetic_mismatch"
    )
    diagnostics["all_first_step_adam_formula_checks_pass"] = formula_matches
    diagnostics["all_captured_clipping_checks_pass"] = clipping_checks_pass
    diagnostics["analytic_input"] = "captured_clipped_gradient_tree"
    return diagnostics


def compare(args: argparse.Namespace) -> dict[str, Any]:
    require(not os.environ.get("PYTHONPATH"), "PYTHONPATH must be unset")
    require(sha256(PROTOCOL) == PROTOCOL_SHA256, "component protocol drift")
    protocol = json.loads(PROTOCOL.read_text())
    output = args.output.resolve()
    require(output.is_relative_to(RUN_ROOT.resolve()), "output must remain under /data run root")
    require(not output.exists(), "output already exists")
    require(output.parent.is_dir() and not output.parent.is_symlink(), "unsafe output parent")
    reference_path, reference = load_capture(args.reference, args.reference_lane, "cpu")
    candidate_path, candidate = load_capture(args.candidate, "modern", args.backend)
    require(
        reference["hashes"]["capture_script_sha256"]
        == candidate["hashes"]["capture_script_sha256"],
        "capture script digest drift",
    )
    require(reference["record_count"] == candidate["record_count"], "record count drift")
    require(reference["groups"] == candidate["groups"], "capture group structure drift")

    import numpy as np

    tolerance = protocol["comparison"][args.backend]
    rtol = float(tolerance["rtol"])
    atol = float(tolerance["atol"])
    exact_dtypes = set(protocol["comparison"]["exact_dtypes"])
    stage_contracts = {item["name"]: item for item in protocol["ordered_stages"]}
    stage_results: dict[str, Any] = {
        item["name"]: {
            "status": "pass",
            "classification": item["classification_on_first_failure"],
            "record_count": 0,
            "exact_record_count": 0,
            "max_element_absolute_error": 0.0,
            "max_element_relative_error": 0.0,
            "failure_count": 0,
            "failures": [],
        }
        for item in protocol["ordered_stages"]
    }
    reference_root = args.reference.resolve()
    candidate_root = args.candidate.resolve()
    for left, right in zip(reference["records"], candidate["records"]):
        require(structure_key(left) == structure_key(right), "record ordering drift")
        stage = left["stage"]
        require(stage in stage_contracts, f"undeclared stage: {stage}")
        exact = (
            left["dtype"] in exact_dtypes
            or stage_contracts[stage]["comparison"] == "exact"
        )
        result = compare_record(
            np,
            reference_root,
            candidate_root,
            left,
            right,
            exact=exact,
            rtol=rtol,
            atol=atol,
        )
        summary = stage_results[stage]
        summary["record_count"] += 1
        summary["exact_record_count"] += int(result["exact_bytes"])
        summary["max_element_absolute_error"] = max(
            summary["max_element_absolute_error"],
            result["max_element_absolute_error"],
        )
        summary["max_element_relative_error"] = max(
            summary["max_element_relative_error"],
            result["max_element_relative_error"],
        )
        if not result["passed"]:
            summary["status"] = "fail"
            summary["failure_count"] += 1
            summary["failures"].append(result)

    ordered_names = [item["name"] for item in protocol["ordered_stages"]]
    first_failing_stage = next(
        (name for name in ordered_names if stage_results[name]["status"] == "fail"),
        None,
    )
    classification = causal_classification(stage_results, first_failing_stage)
    status = "pass" if first_failing_stage is None else "fail_closed"
    adam_diagnostics = (
        selected_adam_diagnostics(
            np,
            reference_root,
            candidate_root,
            reference,
            candidate,
        )
        if args.backend == "gpu"
        else {"status": "not_applicable_to_cpu_comparison"}
    )
    result = {
        "schema_version": 1,
        "status": status,
        "paper_evidence": False,
        "performance_endpoint": False,
        "optimizer_applications": 0,
        "backend_compared": args.backend,
        "reference_lane": args.reference_lane,
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
            "capture_script_sha256": reference["hashes"]["capture_script_sha256"],
            "compare_script_sha256": sha256(Path(__file__).resolve()),
        },
        "ordered_stages": ordered_names,
        "stages": stage_results,
        "earliest_failing_stage": first_failing_stage,
        "classification": classification,
        "selected_adam_diagnostics": adam_diagnostics,
        "gpu_training_gate_open": False,
        "next_action": (
            "none; CPU compatibility gate passed"
            if status == "pass"
            else f"diagnose {classification} without applying an optimizer update"
        ),
    }
    atomic_json(output, result)
    print(
        "COMPONENT_COMPARE_"
        f"{'PASS' if status == 'pass' else 'FAIL_CLOSED'} "
        f"backend={args.backend} earliest={first_failing_stage} "
        f"classification={classification}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--reference-lane", choices=("reference", "modern"), required=True
    )
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
        print(f"COMPONENT_COMPARE_ERROR: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
