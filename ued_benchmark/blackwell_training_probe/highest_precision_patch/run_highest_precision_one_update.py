#!/usr/bin/env python3
"""Run the frozen highest-precision LSTM one-update compatibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ued_benchmark.blackwell_training_probe import run_parity_one_update as base
from ued_benchmark.blackwell_training_probe.highest_precision_patch import (
    audit_one_update_aggregates as audit,
)
from ued_benchmark.blackwell_training_probe.highest_precision_patch import (
    assert_cycle_compat as cycle_compat,
)


PROTOCOL = ROOT / "HIGHEST_PRECISION_ONE_UPDATE_PROTOCOL.json"
PATCH_CONTRACT = ROOT / "PATCH_CONTRACT.json"
PATCH = ROOT / "minimax-highest-lstm.patch"
APPLICATOR = ROOT / "apply_highest_precision_patch.py"
PROTOCOL_SHA256 = "ba0b6fd30de472554d732308017cb8d3c28f7ddef0549631fc5fe907610ec4c3"
PATCH_CONTRACT_SHA256 = "7d8744ff34d064bd324cdc3d92b972b8050f492ff580edc6e44870bbf4aa969e"
PATCH_SHA256 = "a16f4394af0d89289314ab4a11ea43d3334ecba36a22e3c86ed11633d15fb9db"
APPLICATOR_SHA256 = "4fef0fdb4bee747b9794b06832db2ba87345e54e2d21fb1881536521104abd57"
BASE_RUNNER_SHA256 = "078ac8deb7dc79b2c3fa8ede65a7b818afae7a70f01221f1204ec170339358f2"
AUDITOR_SHA256 = "b4bb2359f280f990b7b617d7f84976d87c818d43374a1dd400b2a517591ae6f0"
CYCLE_COMPAT_SHA256 = "f9b0cad2ca1a9bc1df5ded6925bd3bd7166e2efe33684d400244a2852ab16186"
GROUPED_RUNNER_SHA256 = "b34ac79ef05330e7209e9309f9d112bb9da9b60025566eb4e29d16b9bf8d9597"
FRONTIER_MANIFEST_SHA256 = "d929efa2f059a93125e217ec4713ae81670c769d979c67abd2b10efc64268af3"
MODERN_MANIFEST_SHA256 = "ea5fb73c0072cd95829630344e559f02a83f65b0f8b479845ef4dff8921ff65c"
HIGHEST_MANIFEST_SHA256 = "10f276850036306d9838f44b3266626c8600b69b4a0dcf5757b2bd5468e4d050"
REFERENCE_RECEIPT_SHA256 = "1005e3c907c38061f23c46ef8b8b24016818603d4bf42bfd1555afe073b3c8e9"
INITIAL_CHECKPOINT_SHA256 = "4dd07bf02eeb7ec072e4ec72b3aa02180c3ae84284ba20b27174f3dfa9886187"
PRESERVED_GPU_PID = 2786996


class PrecisionGateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionGateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_highest_manifest(contract: Mapping[str, Any]) -> dict[str, Any]:
    target = "src/minimax/models/common.py"
    return {
        "schema_version": 1,
        "overlay": contract["overlay"],
        "contract_sha256": PATCH_CONTRACT_SHA256,
        "patch_sha256": PATCH_SHA256,
        "parent_modernization_manifest_sha256": MODERN_MANIFEST_SHA256,
        "parent_frontier_manifest_sha256": FRONTIER_MANIFEST_SHA256,
        "upstream_commit": base.UPSTREAM_COMMIT,
        "file_sha256": {target: contract["files"][target]["applied_sha256"]},
        "precision": "highest",
        "scope": "OptimizedLSTMCell call only",
        "paper_evidence": False,
    }


def validate_source(source: Path, lane: str) -> dict[str, Any]:
    require(lane == "modern", "highest-precision patch supports only the modern lane")
    require(source.is_dir() and not source.is_symlink(), "unsafe source directory")
    protocol = json.loads(PROTOCOL.read_text())
    require(source.resolve() == Path(protocol["source"]["isolated_clone"]), "source clone drift")
    require(base.git(source, "rev-parse", "HEAD") == base.UPSTREAM_COMMIT, "source commit drift")
    require(base.git(source, "rev-parse", "HEAD^{tree}") == base.UPSTREAM_TREE, "source tree drift")

    frontier_path = source / ".frontierrl_overlay.json"
    modern_path = source / ".blackwell_training_overlay.json"
    highest_path = source / ".blackwell_highest_lstm_overlay.json"
    for label, path, expected_hash in (
        ("Frontier", frontier_path, FRONTIER_MANIFEST_SHA256),
        ("modernization", modern_path, MODERN_MANIFEST_SHA256),
        ("highest-precision", highest_path, HIGHEST_MANIFEST_SHA256),
    ):
        require(path.is_file() and not path.is_symlink(), f"unsafe {label} manifest")
        require(sha256(path) == expected_hash, f"{label} manifest digest drift")

    frontier = json.loads(frontier_path.read_text())
    modern_contract = json.loads(base.MODERN_CONTRACT.read_text())
    modern = json.loads(modern_path.read_text())
    patch_contract = json.loads(PATCH_CONTRACT.read_text())
    highest = json.loads(highest_path.read_text())
    require(modern == base.expected_modern_manifest(modern_contract), "modern manifest drift")
    require(highest == expected_highest_manifest(patch_contract), "highest manifest drift")

    effective = dict(frontier["overlay_file_sha256"])
    effective.update(
        {
            relative: details["applied_sha256"]
            for relative, details in modern_contract["files"].items()
        }
    )
    target = "src/minimax/models/common.py"
    effective[target] = patch_contract["files"][target]["applied_sha256"]
    for relative, expected_hash in effective.items():
        path = source / relative
        require(path.is_file() and not path.is_symlink(), f"unsafe effective file: {relative}")
        require(sha256(path) == expected_hash, f"effective source digest drift: {relative}")

    target_text = (source / target).read_text()
    required_context = patch_contract["files"][target]["required_context"]
    require(target_text.count(required_context) == 1, "highest precision context count drift")
    require(target_text.count("nn.OptimizedLSTMCell(**rnn_kwargs)") == 1, "LSTM cell drift")
    require(target_text.count("nn.GRUCell(**rnn_kwargs)") == 1, "GRU cell drift")
    require(
        target_text.index(required_context) < target_text.index("nn.GRUCell(**rnn_kwargs)"),
        "precision context is not scoped before GRU",
    )
    removed_count = sum(
        path.read_text().count("jax.tree_map")
        for path in (source / "src/minimax").rglob("*.py")
    )
    require(removed_count == 0, f"modern source retains removed API: {removed_count}")
    return {
        "parent_manifest_sha256": sha256(frontier_path),
        "modernization_contract_sha256": sha256(base.MODERN_CONTRACT),
        "modernization_manifest_sha256": sha256(modern_path),
        "highest_precision_contract_sha256": sha256(PATCH_CONTRACT),
        "highest_precision_patch_sha256": sha256(PATCH),
        "highest_precision_manifest_sha256": sha256(highest_path),
        "highest_precision_target_sha256": sha256(source / target),
        "highest_precision_context_count": target_text.count(required_context),
        "removed_api_occurrences": removed_count,
        "effective_file_count": len(effective),
    }


def process_fingerprint(pid: int) -> dict[str, Any]:
    process = Path("/proc") / str(pid)
    require(process.is_dir(), f"preserved GPU PID {pid} is absent")
    stat = (process / "stat").read_text()
    suffix = stat.rsplit(")", 1)
    require(len(suffix) == 2, "cannot parse preserved process stat")
    fields_after_comm = suffix[1].split()
    require(len(fields_after_comm) > 19, "preserved process stat is truncated")
    cmdline = (process / "cmdline").read_bytes()
    return {
        "pid": pid,
        "start_time_ticks": fields_after_comm[19],
        "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
    }


def gpu_process_inventory() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run(cli: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected_hash, label in (
        (PROTOCOL, PROTOCOL_SHA256, "protocol"),
        (PATCH_CONTRACT, PATCH_CONTRACT_SHA256, "patch contract"),
        (PATCH, PATCH_SHA256, "patch"),
        (APPLICATOR, APPLICATOR_SHA256, "applicator"),
        (Path(base.__file__).resolve(), BASE_RUNNER_SHA256, "base runner"),
        (Path(audit.__file__).resolve(), AUDITOR_SHA256, "aggregate auditor"),
        (Path(cycle_compat.__file__).resolve(), CYCLE_COMPAT_SHA256, "cycle compatibility shim"),
    ):
        require(sha256(path) == expected_hash, f"{label} digest drift")
    protocol = json.loads(PROTOCOL.read_text())
    reference = cli.reference_receipt.resolve()
    initial = cli.initial_checkpoint.resolve()
    require(reference == Path(protocol["reference"]["receipt"]), "reference receipt path drift")
    require(initial == Path(protocol["reference"]["initial_checkpoint"]), "initial checkpoint path drift")
    require(reference.is_file() and not reference.is_symlink(), "unsafe reference receipt")
    require(initial.is_file() and not initial.is_symlink(), "unsafe initial checkpoint")
    require(sha256(reference) == REFERENCE_RECEIPT_SHA256, "reference receipt digest drift")
    require(sha256(initial) == INITIAL_CHECKPOINT_SHA256, "initial checkpoint digest drift")

    # The shared grouped-runner helper gained an optimizer-count argument after
    # the parent parity runner was frozen. Import it only after independently
    # enforcing backend selection, then install the isolated legacy-call shim.
    require(not os.environ.get("PYTHONPATH"), "PYTHONPATH must be unset")
    required_platform = "cpu" if cli.backend == "cpu" else "cuda"
    require(os.environ.get("JAX_PLATFORMS") == required_platform, "JAX_PLATFORMS drift")
    require(os.environ.get("JAX_PLATFORM_NAME") == cli.backend, "JAX_PLATFORM_NAME drift")
    require(
        os.environ.get("JAX_THREEFRY_PARTITIONABLE", "").lower() == "false",
        "source-era Threefry mode is required",
    )
    require(
        os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() == "false",
        "XLA preallocation must be disabled",
    )
    source = cli.source.resolve()
    validate_source(source, "modern")
    sys.path[:0] = [str(source / "src"), str(REPO_ROOT)]
    import jax
    from ued_benchmark.scripts import run_grouped_one_update as grouped_runner

    require(
        sha256(Path(grouped_runner.__file__).resolve()) == GROUPED_RUNNER_SHA256,
        "grouped-runner helper digest drift",
    )
    require(jax.default_backend() == cli.backend, "requested backend is not active in shim preflight")
    devices = jax.devices(cli.backend)
    require(len(devices) == 1 and devices[0].platform == cli.backend, "shim device isolation drift")
    shim_state = cycle_compat.install(grouped_runner)

    preserved_before = None
    inventory_before = None
    if cli.backend == "gpu":
        preserved_before = process_fingerprint(PRESERVED_GPU_PID)
        inventory_before = gpu_process_inventory()
        require(
            any(line.split(",", 1)[0].strip() == str(PRESERVED_GPU_PID) for line in inventory_before.splitlines()),
            "preserved PID is not present in the GPU compute inventory",
        )

    base.PARITY_PROTOCOL = PROTOCOL
    base.PARITY_PROTOCOL_SHA256 = PROTOCOL_SHA256
    base.validate_source = validate_source
    base_cli = argparse.Namespace(**vars(cli))
    # Capture the complete candidate receipt before exhaustive comparison so a
    # failing aggregate never suppresses later aggregate diagnostics.
    base_cli.reference_receipt = None
    receipt = base.run(base_cli)

    preserved_after = None
    inventory_after = None
    if cli.backend == "gpu":
        preserved_after = process_fingerprint(PRESERVED_GPU_PID)
        inventory_after = gpu_process_inventory()
        require(preserved_before == preserved_after, "preserved GPU process identity changed")
        require(
            any(line.split(",", 1)[0].strip() == str(PRESERVED_GPU_PID) for line in inventory_after.splitlines()),
            "preserved PID left the GPU compute inventory",
        )

    reference_document = json.loads(reference.read_text())
    comparison = audit.compare_documents(
        reference_document,
        receipt,
        backend=cli.backend,
        protocol=protocol,
    )
    receipt["purpose"] = "bounded highest-precision OptimizedLSTMCell Blackwell compatibility gate"
    receipt["status"] = "passed" if comparison["summary"]["status"] == "passed" else "failed"
    receipt["compatibility_patch"] = {
        "overlay": protocol["patch"]["overlay"],
        "precision": "highest",
        "semantic_scope": protocol["patch"]["semantic_scope"],
        "changed_files": protocol["patch"]["changed_files"],
    }
    optimizer_observations = list(shim_state["optimizer_step_observations"])
    require(optimizer_observations and optimizer_observations[-1] == 1, "optimizer shim count drift")
    require(set(optimizer_observations).issubset({0, 1}), "optimizer shim observed extra updates")
    receipt["helper_api_compatibility"] = {
        "shim_version": shim_state["shim_version"],
        "shim_sha256": CYCLE_COMPAT_SHA256,
        "shared_grouped_runner_sha256": GROUPED_RUNNER_SHA256,
        "reason": "frozen parent runner uses the legacy _assert_cycle(summary, cycle=...) call",
        "optimizer_step_observations": optimizer_observations,
        "final_optimizer_step_applications": optimizer_observations[-1],
        "semantics_weakened": False,
    }
    receipt["execution_budget"] = {
        "maximum_candidate_runs_this_backend": 1,
        "maximum_gpu_ppo_updates": 1,
        "performance_endpoint": False,
        "ood_evaluation": False,
        "multiseed": False,
        "throughput": False,
    }
    receipt["preserved_gpu_process"] = {
        "required": cli.backend == "gpu",
        "pid": PRESERVED_GPU_PID,
        "before": preserved_before,
        "after": preserved_after,
        "inventory_before": inventory_before,
        "inventory_after": inventory_after,
        "preserved": cli.backend != "gpu" or preserved_before == preserved_after,
    }
    receipt["parity"] = comparison["summary"]
    receipt["hashes"].update(
        {
            "script_sha256": sha256(Path(__file__).resolve()),
            "base_runner_sha256": BASE_RUNNER_SHA256,
            "aggregate_auditor_sha256": AUDITOR_SHA256,
            "cycle_compatibility_shim_sha256": CYCLE_COMPAT_SHA256,
            "shared_grouped_runner_sha256": GROUPED_RUNNER_SHA256,
            "highest_precision_protocol_sha256": PROTOCOL_SHA256,
            "highest_precision_patch_contract_sha256": PATCH_CONTRACT_SHA256,
            "highest_precision_patch_sha256": PATCH_SHA256,
            "highest_precision_applicator_sha256": APPLICATOR_SHA256,
            "highest_precision_applied_manifest_sha256": HIGHEST_MANIFEST_SHA256,
            "reference_receipt_sha256": REFERENCE_RECEIPT_SHA256,
            "reference_initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA256,
        }
    )
    receipt_path = cli.output.resolve() / "receipt.json"
    audit.atomic_json(receipt_path, receipt)
    exhaustive_report = audit.compare_paths(reference, receipt_path, backend=cli.backend)
    require(exhaustive_report["summary"] == comparison["summary"], "audit summary reproduction drift")
    audit.atomic_json(cli.output.resolve() / "aggregate-comparison.json", exhaustive_report)
    if comparison["summary"]["status"] != "passed":
        raise PrecisionGateError(
            f"aggregate parity failed: {comparison['summary']['first_failure']}"
        )
    return receipt, exhaustive_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", choices=("modern",), default="modern")
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--reference-receipt", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    cli = parser.parse_args()
    try:
        receipt, report = run(cli)
    except (
        PrecisionGateError,
        audit.AuditError,
        base.GateError,
        AssertionError,
        KeyError,
        OSError,
        ValueError,
        TypeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"HIGHEST_PRECISION_ONE_UPDATE_FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "HIGHEST_PRECISION_ONE_UPDATE_PASS "
        f"backend={receipt['backend']} updates={receipt['actual_student_updates']} "
        f"trials={receipt['final_state']['frontier_total_trials']} "
        f"aggregates={report['summary']['aggregate_gate_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
