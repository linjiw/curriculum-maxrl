#!/usr/bin/env python3
"""Read-only recovery of the exhausted highest-precision CPU update budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ued_benchmark.blackwell_training_probe import run_parity_one_update as parity_base
from ued_benchmark.blackwell_training_probe.highest_precision_patch import (
    audit_one_update_aggregates as aggregate_audit,
)
from ued_benchmark.blackwell_training_probe.highest_precision_patch import (
    run_highest_precision_one_update as precision_runner,
)


PROTOCOL_SHA256 = "ba0b6fd30de472554d732308017cb8d3c28f7ddef0549631fc5fe907610ec4c3"
REFERENCE_RECEIPT_SHA256 = "1005e3c907c38061f23c46ef8b8b24016818603d4bf42bfd1555afe073b3c8e9"
RAW_RECEIPT_SHA256 = "98cba2e35bb79ef9037b6286c3605177b2b188a44ab7bb5dff5da75f50edfdf7"
INITIAL_CHECKPOINT_SHA256 = "6e095ca4637d3894717434dde1832dfabf486a3aeb915a87847f985649f98e08"
FINAL_CHECKPOINT_SHA256 = "d21e282ea7f2e0fa6721a090f9ab570bf6f53b59a76dcf8cf0d5676c595d5151"
AGGREGATE_REPORT_SHA256 = "a168816ee639a25f5ede95d5e17fb9516b4cbb720a934ba448b113f26915ce85"
AGGREGATE_AUDITOR_SHA256 = "b4bb2359f280f990b7b617d7f84976d87c818d43374a1dd400b2a517591ae6f0"
EXECUTED_WRAPPER_SHA256 = "dd9d6a27a6f99125ea0aac38d96cb00792bfa9163fbb9b3f91065b692045a0a0"
FIXED_FUTURE_WRAPPER_SHA256 = "dec1c6339f20885ecbae77caecc4c2765e35bc93db217db688b889654191d005"
SHIM_SHA256 = "f9b0cad2ca1a9bc1df5ded6925bd3bd7166e2efe33684d400244a2852ab16186"
GROUPED_RUNNER_SHA256 = "b34ac79ef05330e7209e9309f9d112bb9da9b60025566eb4e29d16b9bf8d9597"


class RecoveryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-receipt", type=Path, required=True)
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--aggregate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require(not os.environ.get("PYTHONPATH"), "PYTHONPATH must be unset")
    required_environment = {
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": "cpu",
        "JAX_THREEFRY_PARTITIONABLE": "false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    for name, expected in required_environment.items():
        require(os.environ.get(name, "").lower() == expected, f"{name} drift")

    source = args.source.resolve()
    reference_path = args.reference_receipt.resolve()
    raw_path = args.raw_receipt.resolve()
    initial_path = args.initial_checkpoint.resolve()
    final_path = args.final_checkpoint.resolve()
    aggregate_path = args.aggregate_report.resolve()
    output = args.output.resolve()
    require(not output.exists(), "recovery output already exists")
    for path, expected_hash, label in (
        (precision_runner.PROTOCOL, PROTOCOL_SHA256, "protocol"),
        (reference_path, REFERENCE_RECEIPT_SHA256, "reference receipt"),
        (raw_path, RAW_RECEIPT_SHA256, "raw receipt"),
        (initial_path, INITIAL_CHECKPOINT_SHA256, "initial checkpoint"),
        (final_path, FINAL_CHECKPOINT_SHA256, "final checkpoint"),
        (aggregate_path, AGGREGATE_REPORT_SHA256, "aggregate report"),
        (Path(aggregate_audit.__file__).resolve(), AGGREGATE_AUDITOR_SHA256, "aggregate auditor"),
        (Path(precision_runner.__file__).resolve(), FIXED_FUTURE_WRAPPER_SHA256, "future wrapper"),
        (ROOT / "assert_cycle_compat.py", SHIM_SHA256, "cycle API shim"),
        (REPO_ROOT / "ued_benchmark/scripts/run_grouped_one_update.py", GROUPED_RUNNER_SHA256, "grouped runner"),
    ):
        require(path.is_file() and not path.is_symlink(), f"unsafe {label}")
        require(sha256(path) == expected_hash, f"{label} digest drift")

    source_receipt = precision_runner.validate_source(source, "modern")
    raw = json.loads(raw_path.read_text())
    reference = json.loads(reference_path.read_text())
    aggregate_report = json.loads(aggregate_path.read_text())
    require(aggregate_report["summary"]["status"] == "passed", "aggregate audit failed")
    require(aggregate_report["summary"]["failure_count"] == 0, "aggregate failures remain")
    require(aggregate_report["summary"]["aggregate_gate_count"] == 546, "aggregate count drift")
    require(aggregate_report["summary"]["initial_exact_leaf_hash_count"] == 91, "initial hash gate drift")
    require(aggregate_report["summary"]["statistic_gate_count"] == 24, "statistic gate count drift")
    require(len(raw["cycles"]) == 2, "cycle records are missing")
    cycle_two_keys = sorted(raw["cycles"][1]["stats"])
    require(cycle_two_keys == sorted(reference["cycles"][1]["stats"]), "cycle-2 scalar keys drift")

    sys.path[:0] = [str(source / "src"), str(REPO_ROOT)]
    import jax
    import numpy as np
    from minimax.util.checkpoint import load_pkl_object
    from ued_benchmark.scripts import run_grouped_one_update as grouped

    require(jax.default_backend() == "cpu", "recovery backend is not CPU")
    devices = jax.devices("cpu")
    require(len(devices) == 1 and devices[0].platform == "cpu", "recovery device drift")
    require(jax.config.jax_threefry_partitionable is False, "PRNG mode drift")
    config = grouped.load_frozen_config(parity_base.CONFIG)
    parsed = grouped.configure_engineering_run(config, local_test_mode=True)
    experiment = grouped._make_experiment(parsed)

    def load_state(path: Path) -> tuple[Any, Any, Mapping[str, Any]]:
        checkpoint = load_pkl_object(str(path))
        fresh = experiment.runner.reset(jax.random.PRNGKey(parsed.seed))
        state = experiment.runner.load_checkpoint_state(fresh, checkpoint)
        grouped._block(state)
        return checkpoint, state, parity_base.leaf_signature(jax, np, state[1].state_dict)

    initial_checkpoint, initial_state, initial_signature = load_state(initial_path)
    final_checkpoint, final_state, final_signature = load_state(final_path)
    require(initial_signature == raw["numerical"]["initial"], "raw initial signature drift")
    require(final_signature == raw["numerical"]["final"], "raw final signature drift")
    initial_summary = grouped._state_summary(initial_state)
    final_summary = grouped._state_summary(final_state)
    initial_optimizer_steps = initial_summary.pop("optimizer_step_applications")
    final_optimizer_steps = final_summary.pop("optimizer_step_applications")
    require(initial_summary == raw["initial_checkpoint"]["state"], "initial state summary drift")
    require(final_summary == raw["final_state"], "final state summary drift")
    require(initial_optimizer_steps == 0, "initial optimizer count drift")
    require(final_optimizer_steps == 1, "final optimizer count drift")

    initial_checkpoint_signature = parity_base.leaf_signature(jax, np, initial_checkpoint)
    final_checkpoint_signature = parity_base.leaf_signature(jax, np, final_checkpoint)
    require(
        initial_checkpoint_signature["structure_sha256"]
        == raw["initial_checkpoint"]["structure_sha256"],
        "initial serialized checkpoint structure drift",
    )
    require(
        final_checkpoint_signature["structure_sha256"] == raw["checkpoint"]["structure_sha256"],
        "final serialized checkpoint structure drift",
    )
    reloaded_again = load_pkl_object(str(final_path))
    serialized_leaf_count = grouped._assert_same_pytree(
        final_checkpoint, reloaded_again, "read-only pickle recovery"
    )
    require(
        serialized_leaf_count == raw["checkpoint"]["serialized_leaf_count"],
        "serialized checkpoint leaf count drift",
    )
    second_fresh = grouped._make_experiment(parsed)
    second_reset = second_fresh.runner.reset(jax.random.PRNGKey(parsed.seed))
    second_resumed = second_fresh.runner.load_checkpoint_state(second_reset, reloaded_again)
    grouped._block(second_resumed)
    resumed_leaf_count = grouped._assert_same_pytree(
        final_state[1].state_dict,
        second_resumed[1].state_dict,
        "read-only fresh-runner resume",
    )
    require(resumed_leaf_count == raw["checkpoint"]["resumed_leaf_count"], "resume leaf count drift")

    record = {
        "schema_version": 1,
        "purpose": "read-only recovery of the exhausted highest-precision CPU one-update candidate",
        "paper_evidence": False,
        "performance_endpoint": False,
        "ood_evaluation": False,
        "additional_cpu_updates_during_recovery": 0,
        "gpu_attempted": False,
        "status": "recovered_parity_pass_but_original_gate_incomplete",
        "decision": {
            "cpu_numerical_parity": "passed",
            "original_wrapper_completion": "failed_after_raw_receipt_write",
            "bounded_training_gate": "INCOMPLETE",
            "project_gate": "HOLD",
            "gpu_gate": "NOT_ATTEMPTED",
            "reason": "the single complete CPU update exhausted the frozen budget before the wrapper could persist its final provenance-bearing receipt"
        },
        "source": source_receipt,
        "runtime": {
            "recovery_backend": jax.default_backend(),
            "recovery_device_kind": devices[0].device_kind,
            "jax": raw["runtime"]["jax"],
            "jaxlib": raw["runtime"]["jaxlib"],
            "pythonpath_cleared": True,
            "xla_preallocation": False,
        },
        "execution": {
            "complete_cpu_candidate_updates": 1,
            "final_optimizer_step_applications": final_optimizer_steps,
            "outer_cycles": len(raw["cycles"]),
            "final_state": final_summary,
            "cycle_two_scalar_keys": cycle_two_keys,
            "missing_cycle_two_scalar_keys": [],
            "missing_wrapper_only_fields": [
                "helper_api_compatibility.optimizer_step_observations",
                "preserved_gpu_process",
                "execution_budget",
                "wrapper parity summary"
            ],
        },
        "parity": aggregate_report["summary"],
        "checkpoint_recovery": {
            "initial_checkpoint_sha256": sha256(initial_path),
            "final_checkpoint_sha256": sha256(final_path),
            "initial_train_state_signature_exact_to_raw_receipt": True,
            "final_train_state_signature_exact_to_raw_receipt": True,
            "serialized_leaf_count": serialized_leaf_count,
            "resumed_leaf_count": resumed_leaf_count,
            "structure_sha256": final_checkpoint_signature["structure_sha256"],
            "exact_pickle_round_trip": True,
            "exact_fresh_runner_resume": True,
            "post_resume_update_executed": False,
        },
        "failures": {
            "preflight": "legacy _assert_cycle API mismatch; zero optimizer updates",
            "complete_candidate": "wrapper receipt rewrite referenced nonexistent base._atomic_json after raw receipt was already written",
            "invalid_auditor_v1": "incorrectly required final leaf byte hashes; superseded without changing tolerances"
        },
        "hashes": {
            "protocol_sha256": PROTOCOL_SHA256,
            "reference_receipt_sha256": sha256(reference_path),
            "raw_receipt_sha256": sha256(raw_path),
            "aggregate_report_sha256": sha256(aggregate_path),
            "aggregate_auditor_sha256": AGGREGATE_AUDITOR_SHA256,
            "executed_wrapper_sha256": EXECUTED_WRAPPER_SHA256,
            "fixed_future_wrapper_sha256": FIXED_FUTURE_WRAPPER_SHA256,
            "cycle_compatibility_shim_sha256": SHIM_SHA256,
            "shared_grouped_runner_sha256": GROUPED_RUNNER_SHA256,
            "recovery_script_sha256": sha256(Path(__file__).resolve()),
        },
    }
    atomic_json(output, record)
    print(
        "HIGHEST_PRECISION_CPU_RECOVERY_HOLD "
        f"updates={record['execution']['complete_cpu_candidate_updates']} "
        f"aggregates={record['parity']['aggregate_gate_count']} "
        f"failures={record['parity']['failure_count']} gate=INCOMPLETE"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        RecoveryError,
        AssertionError,
        ValueError,
        KeyError,
        OSError,
        TypeError,
    ) as error:
        print(f"HIGHEST_PRECISION_CPU_RECOVERY_ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
