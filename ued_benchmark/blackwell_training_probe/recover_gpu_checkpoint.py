#!/usr/bin/env python3
"""Read-only recovery and parity diagnosis for the sole RTX 5090 update."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]


class RecoveryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(np: Any, reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict:
    require(reference["structure_sha256"] == candidate["structure_sha256"], "structure drift")
    require(reference["leaf_count"] == candidate["leaf_count"], "leaf-count drift")
    exact = 0
    failures = []
    max_absolute = 0.0
    max_relative = 0.0
    for before, after in zip(reference["leaves"], candidate["leaves"]):
        require(
            (before["path"], before["shape"], before["dtype"], before["size"])
            == (after["path"], after["shape"], after["dtype"], after["size"]),
            f"leaf structure drift: {before['path']}",
        )
        if before["sha256"] == after["sha256"]:
            exact += 1
        for metric in (
            "nan_count",
            "infinite_count",
            "positive_inf_count",
            "negative_inf_count",
        ):
            if metric in before or metric in after:
                require(
                    before.get(metric) == after.get(metric),
                    f"non-finite sentinel drift: {before['path']}:{metric}",
                )
        for metric in ("abs_sum", "squared_l2", "max_abs"):
            if metric not in before:
                continue
            left = float(before[metric])
            right = float(after[metric])
            absolute = abs(left - right)
            relative = absolute / max(abs(left), 5e-5, 1e-300)
            max_absolute = max(max_absolute, absolute)
            max_relative = max(max_relative, relative)
            if not np.isclose(left, right, rtol=5e-4, atol=5e-5):
                failures.append(
                    {
                        "path": before["path"],
                        "metric": metric,
                        "reference": left,
                        "candidate": right,
                        "absolute_error": absolute,
                        "relative_error": relative,
                    }
                )
    return {
        "exact_leaf_hashes": exact,
        "leaf_count": candidate["leaf_count"],
        "failure_count": len(failures),
        "failures": failures,
        "max_aggregate_absolute_error": max_absolute,
        "max_aggregate_relative_error": max_relative,
        "rtol": 5e-4,
        "atol": 5e-5,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-receipt", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not os.environ.get("PYTHONPATH"), "PYTHONPATH must be unset")
    required = {
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": "cpu",
        "JAX_THREEFRY_PARTITIONABLE": "false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    for name, expected in required.items():
        require(os.environ.get(name, "").lower() == expected, f"{name} drift")
    output = args.output.resolve()
    require(not output.exists(), "recovery output already exists")
    source = args.source.resolve()
    sys.path[:0] = [str(source / "src"), str(REPO_ROOT)]

    import jax
    import numpy as np
    from minimax.util.checkpoint import load_pkl_object
    from ued_benchmark.blackwell_training_probe.run_parity_one_update import (
        PARITY_PROTOCOL_SHA256,
        leaf_signature,
    )
    from ued_benchmark.scripts import run_grouped_one_update as base

    require(jax.default_backend() == "cpu", "recovery backend is not CPU")
    require(jax.devices()[0].platform == "cpu", "recovery device is not CPU")
    require(jax.config.jax_threefry_partitionable is False, "PRNG mode drift")
    config = base.load_frozen_config(
        REPO_ROOT / "ued_benchmark/configs/maze_frontier_exact_grouped_n8.json"
    )
    parsed = base.configure_engineering_run(config, local_test_mode=True)
    experiment = base._make_experiment(parsed)

    def load_state(path: Path):
        require(path.is_file() and not path.is_symlink(), "unsafe checkpoint")
        checkpoint = load_pkl_object(str(path))
        fresh = experiment.runner.reset(jax.random.PRNGKey(parsed.seed))
        state = experiment.runner.load_checkpoint_state(fresh, checkpoint)
        base._block(state)
        return checkpoint, state, leaf_signature(jax, np, state[1].state_dict)

    initial_checkpoint, initial_state, initial_signature = load_state(
        args.initial_checkpoint.resolve()
    )
    final_checkpoint, final_state, final_signature = load_state(
        args.final_checkpoint.resolve()
    )
    reference_path = args.reference_receipt.resolve()
    reference = json.loads(reference_path.read_text())
    initial_parity = diagnose(np, reference["numerical"]["initial"], initial_signature)
    final_parity = diagnose(np, reference["numerical"]["final"], final_signature)
    require(initial_parity["failure_count"] == 0, "GPU initial checkpoint parity failed")
    require(
        initial_parity["exact_leaf_hashes"] == initial_parity["leaf_count"],
        "GPU initial checkpoint is not byte-exact",
    )
    require(final_parity["failure_count"] == 1, "unexpected GPU parity failure count")
    failure = final_parity["failures"][0]
    require(
        failure["path"] == "['params']['params']['fc_pi_1']['bias']"
        and failure["metric"] == "abs_sum",
        "unexpected GPU parity failure location",
    )
    initial_summary = base._state_summary(initial_state)
    final_summary = base._state_summary(final_state)
    require(initial_summary["n_updates"] == 0, "initial update counter drift")
    require(initial_summary["frontier_total_trials"] == 0, "initial posterior drift")
    require(final_summary["n_updates"] == 1, "GPU update count is not exactly one")
    require(final_summary["n_grad_updates"] == 1, "GPU gradient update count drift")
    require(final_summary["frontier_total_trials"] == 64, "GPU posterior trial drift")
    require(final_summary["frontier_incomplete_group_count"] == 0, "incomplete group")
    require(final_summary["frontier_duplicate_new_group_count"] == 0, "duplicate group")
    checkpoint_structure = leaf_signature(jax, np, final_checkpoint)
    require(
        checkpoint_structure["structure_sha256"]
        == reference["checkpoint"]["structure_sha256"],
        "GPU serialized checkpoint structure drift",
    )

    record = {
        "schema_version": 1,
        "purpose": "read-only recovery of the sole bounded RTX 5090 PPO update",
        "paper_evidence": False,
        "ood_evaluation": False,
        "gpu_updates_executed": 1,
        "additional_updates_during_recovery": 0,
        "status": "failed_gpu_numerical_parity",
        "original_gpu_backend": "gpu",
        "original_gpu_device_kind": "NVIDIA GeForce RTX 5090",
        "recovery_backend": "cpu",
        "reference_receipt_sha256": sha256(reference_path),
        "parity_protocol_sha256": PARITY_PROTOCOL_SHA256,
        "initial_checkpoint": {
            "sha256": sha256(args.initial_checkpoint.resolve()),
            "summary": initial_summary,
            "parity": initial_parity,
        },
        "final_checkpoint": {
            "sha256": sha256(args.final_checkpoint.resolve()),
            "summary": final_summary,
            "structure_sha256": checkpoint_structure["structure_sha256"],
            "parity": final_parity,
        },
        "original_process_receipt_present": False,
        "gpu_gate_open": False,
        "next_action": "diagnose GPU reduction/update drift without another optimizer update",
    }
    base._atomic_json(output, record)
    print(
        "GPU_RECOVERY_FAIL_CLOSED "
        f"updates={final_summary['n_updates']} "
        f"trials={final_summary['frontier_total_trials']} "
        f"parity_failures={final_parity['failure_count']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RecoveryError, AssertionError, ValueError, KeyError, OSError) as error:
        print(f"GPU_RECOVERY_ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
