#!/usr/bin/env python3
"""Fail-closed analysis for the first matched Frontier-vs-MaxMC dev gate.

This program accepts no loose CSVs.  A campaign manifest must bind the exact
protocol, analyzer, source bundle, overlay, environment, drivers, sbatch, GPU
shape, and ten pre-endpoint Slurm submissions.  Each deterministic run ID names
one atomic schema-2 package.  In addition to the checkpoint, logs, command,
scheduler receipt, and aggregate evaluation, the package carries the run
context, the complete training and evaluation receipts, both source
``SHA256SUMS``/``COMPLETE`` pairs, and all 30 raw episode records.  Frontier
packages also carry the source-manifest-bound safe replay snapshot.  The outer
``SHA256SUMS`` closes over every payload and its ``COMPLETE`` binds that
manifest digest.

The analyzer first validates every outer closure, both embedded source
closures, all receipt/context/checkpoint/provenance links, logs, terminal
accounting, and paired budgets.  It then validates all raw episode files.  It
does not parse a numeric aggregate CSV cell until all ten packages and all 30
records per package have passed.  The program never unpickles a checkpoint.

This is engineering/development analysis, not paper evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PROTOCOL_PATH = HERE / "development_protocol_v1.json"
ASSEMBLER_PATH = REPO_ROOT / "ued_benchmark/scripts/assemble_matched_run.py"
PROTOCOL_ID = "ued-dev-frontier-vs-maxmc-4x8-b500-v1"
PURPOSE = "engineering_development_only_not_paper_evidence"
ARMS = ("frontier", "maxmc")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
JOB_RE = re.compile(r"^[0-9]+(?:_[0-9]+)?$")
UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
MANIFEST_LINE_RE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$")
COMMON_PACKAGE_PAYLOADS = frozenset({
    "checkpoint.pkl",
    "command.txt",
    "endpoint.json",
    "evaluation-COMPLETE",
    "evaluation-SHA256SUMS",
    "evaluation-episodes.jsonl",
    "evaluation-receipt.json",
    "evaluation.csv",
    "logs.csv",
    "meta.json",
    "run-manifest.json",
    "run-context.json",
    "scheduler.json",
    "stderr.log",
    "stdout.log",
    "training-COMPLETE",
    "training-SHA256SUMS",
    "training-receipt.json",
})
FRONTIER_PACKAGE_PAYLOADS = frozenset({"training-frontier-buffer-snapshot.json"})
# Compatibility name for callers constructing MaxMC/common packages.
PACKAGE_PAYLOADS = COMMON_PACKAGE_PAYLOADS
SECRET_KEY_RE = re.compile(
    r"(?:^|_)(?:API_KEY|ACCESS_KEY|PRIVATE_KEY|PASSWORD|PASSWD|SECRET|TOKEN)(?:$|_)",
    re.IGNORECASE,
)


class GateError(RuntimeError):
    """Raised when an input violates the frozen development contract."""


@dataclass(frozen=True)
class ValidatedRun:
    arm: str
    seed: int
    run_id: str
    root: Path
    student_ppo_updates: int
    optimizer_step_applications: int
    outer_cycles: int
    training_transitions: int
    package_manifest_sha256: str


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _sha256(path: Path) -> str:
    _fail(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GateError("value is not canonical-JSON serializable") from exc
    return hashlib.sha256(payload).hexdigest()


def _json_no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        _fail(key not in document, f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _fail(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_no_duplicate_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"invalid {label}: {path}") from exc
    _fail(isinstance(document, dict), f"{label} must be a JSON object")
    return document


def _exact_keys(document: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    expected = set(keys)
    actual = set(document)
    _fail(actual == expected, f"{label} keys drift: expected {sorted(expected)}, got {sorted(actual)}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_hash(value: Any, label: str) -> str:
    _fail(isinstance(value, str) and HASH_RE.fullmatch(value) is not None,
          f"{label} is not a lowercase SHA-256")
    return value


def _run_id(seed: int, arm: str) -> str:
    return f"{PROTOCOL_ID}-s{seed}-{arm}"


def _load_protocol() -> tuple[dict[str, Any], str]:
    protocol = _load_json(PROTOCOL_PATH, "protocol")
    _fail(protocol.get("schema") == 1, "protocol schema drift")
    _fail(protocol.get("protocol_id") == PROTOCOL_ID, "protocol identity drift")
    _fail(protocol.get("purpose") == PURPOSE, "protocol purpose drift")
    _fail(protocol.get("training_seeds") == [101, 102, 103, 104, 105],
          "development seed drift")
    _fail(set(protocol.get("arms", {})) == set(ARMS), "protocol arm drift")
    _fail(protocol.get("evaluation", {}).get("max_episode_horizon") == 450,
          "evaluation horizon drift")
    budget = protocol.get("training_budget", {})
    _fail(budget.get("target_student_ppo_updates") == 30000,
          "student PPO-update budget drift")
    _fail(budget.get("ppo_epochs_per_student_update") == 5
          and budget.get("ppo_minibatches_per_epoch") == 1,
          "PPO optimizer schedule drift")
    student_rl = protocol.get("expected_static_meta_config", {}).get("student_rl_args", {})
    _fail(student_rl.get("n_epochs") == budget["ppo_epochs_per_student_update"]
          and student_rl.get("n_minibatches") == budget["ppo_minibatches_per_epoch"],
          "PPO optimizer schedule does not match the resolved config")
    _fail(budget.get("target_optimizer_step_applications")
          == budget["target_student_ppo_updates"]
          * budget["ppo_epochs_per_student_update"]
          * budget["ppo_minibatches_per_epoch"],
          "optimizer-step application budget does not reconcile")
    _fail(budget.get("upstream_integrity_counters", {}).get("n_updates") == 30000
          and budget["upstream_integrity_counters"].get("n_grad_updates") == 30000,
          "upstream PPO integrity-counter drift")
    package = protocol.get("run_package")
    _fail(isinstance(package, dict), "run-package protocol is missing")
    _exact_keys(package, {
        "schema", "analyzer_eligible_endpoint_class", "engineering_endpoint_class",
        "engineering_package_analyzer_eligible", "common_payloads",
        "frontier_additional_payloads", "source_closures", "numeric_csv_parse_order",
    }, "run-package protocol")
    _fail(package["schema"] == 2
          and package["analyzer_eligible_endpoint_class"] == "matched_development"
          and package["engineering_endpoint_class"] == "bounded_engineering_test"
          and package["engineering_package_analyzer_eligible"] is False,
          "run-package identity drift")
    _fail(package["common_payloads"] == sorted(COMMON_PACKAGE_PAYLOADS)
          and package["frontier_additional_payloads"]
          == sorted(FRONTIER_PACKAGE_PAYLOADS),
          "run-package payload closure drift")
    _fail(package["source_closures"] == {
        "evaluation": [
            "evaluation-episodes.jsonl", "evaluation.csv", "evaluation-receipt.json",
            "evaluation-SHA256SUMS", "evaluation-COMPLETE",
        ],
        "training_frontier": [
            "training-receipt.json", "training-frontier-buffer-snapshot.json",
            "training-SHA256SUMS", "training-COMPLETE",
        ],
        "training_maxmc": [
            "training-receipt.json", "training-SHA256SUMS", "training-COMPLETE",
        ],
    }, "embedded source-closure protocol drift")
    return protocol, _sha256(PROTOCOL_PATH)


def _singleton_config(path: Path) -> dict[str, Any]:
    document = _load_json(path, "authored config")
    _exact_keys(document, {"args"}, f"authored config {path.name}")
    args = document["args"]
    _fail(isinstance(args, dict), f"{path.name} args must be an object")
    flat: dict[str, Any] = {}
    for key, value in args.items():
        _fail(isinstance(value, list) and len(value) == 1,
              f"{path.name}:{key} must be a singleton grid value")
        flat[key] = value[0]
    return flat


def repository_preflight() -> dict[str, Any]:
    """Verify the authored pair without reading any run output."""
    protocol, protocol_sha = _load_protocol()
    authored: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for arm in ARMS:
        spec = protocol["arms"][arm]
        path = REPO_ROOT / spec["config_path"]
        digest = _sha256(path)
        _fail(digest == spec["config_sha256"], f"{arm} authored config hash drift")
        authored[arm] = _singleton_config(path)
        hashes[arm] = digest

    keys = set(authored["frontier"]) | set(authored["maxmc"])
    difference_keys = sorted(
        key for key in keys
        if authored["frontier"].get(key, object()) != authored["maxmc"].get(key, object())
    )
    expected_differences = sorted({
        "ued_score",
        "plr_frontier_n_rollouts",
        "plr_frontier_require_n_eval_match",
        "plr_frontier_prior_alpha",
        "plr_frontier_prior_beta",
        "plr_frontier_success_threshold",
        "plr_frontier_posterior_mode",
    })
    _fail(difference_keys == expected_differences,
          f"authored score-isolation drift: {difference_keys}")
    frontier = authored["frontier"]
    maxmc = authored["maxmc"]
    expected_common = {
        "n_total_updates": 30000,
        "n_devices": 1,
        "n_students": 1,
        "n_parallel": 4,
        "n_eval": 8,
        "n_rollout_steps": 256,
        "plr_buffer_size": 500,
        "plr_replay_prob": 0.5,
        "plr_min_fill_ratio": 0.5,
        "test_n_episodes": 10,
        "test_env_names": "Maze-SixteenRooms,Maze-Labyrinth,Maze-StandardMaze",
        "from_last_checkpoint": False,
    }
    for key, value in expected_common.items():
        _fail(frontier.get(key) == value and maxmc.get(key) == value,
              f"common authored field drift: {key}")
    _fail(frontier["ued_score"] == "coefficient_activity", "Frontier score drift")
    _fail(maxmc["ued_score"] == "max_mc", "MaxMC score drift")
    _fail(frontier["plr_frontier_n_rollouts"] == frontier["n_eval"] == 8,
          "Frontier exact-group contract drift")
    return {
        "status": "PASS",
        "scope": PURPOSE,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha,
        "analyzer_sha256": _sha256(Path(__file__).resolve()),
        "assembler_sha256": _sha256(ASSEMBLER_PATH),
        "authored_config_sha256": hashes,
        "authored_template_difference_keys": difference_keys,
        "budget_semantics": {
            "student_ppo_updates": 30000,
            "upstream_n_updates_integrity_counter": 30000,
            "upstream_n_grad_updates_integrity_counter": 30000,
            "optimizer_step_applications": 150000,
            "optimizer_step_formula": "30000 student PPO updates * 5 epochs * 1 minibatch",
        },
        "endpoint_accessed": False,
    }


def _validate_campaign(path: Path, expected_sha: str, protocol: dict[str, Any],
                       protocol_sha: str) -> tuple[dict[str, Any], str]:
    _require_hash(expected_sha, "expected campaign digest")
    actual_sha = _sha256(path)
    _fail(actual_sha == expected_sha, "campaign manifest digest mismatch")
    campaign = _load_json(path, "campaign manifest")
    _exact_keys(campaign, {
        "schema", "protocol_id", "purpose", "created_utc",
        "frozen_before_endpoint_access", "protocol_sha256", "analyzer_sha256",
        "provenance", "hardware", "submissions",
    }, "campaign manifest")
    _fail(campaign["schema"] == 1, "campaign schema drift")
    _fail(campaign["protocol_id"] == PROTOCOL_ID and campaign["purpose"] == PURPOSE,
          "campaign identity drift")
    _fail(campaign["frozen_before_endpoint_access"] is True,
          "campaign was not frozen before endpoint access")
    _fail(isinstance(campaign["created_utc"], str)
          and UTC_RE.fullmatch(campaign["created_utc"]) is not None,
          "campaign created_utc must be second-resolution UTC")
    _fail(campaign["protocol_sha256"] == protocol_sha, "campaign protocol drift")
    _fail(campaign["analyzer_sha256"] == _sha256(Path(__file__).resolve()),
          "campaign analyzer drift")

    provenance = campaign["provenance"]
    _fail(isinstance(provenance, dict), "campaign provenance must be an object")
    _exact_keys(provenance, {
        "base_commit", "base_tree", "overlay_contract_sha256",
        "bundle_manifest_sha256", "overlay_manifest_sha256",
        "applied_overlay_manifest_sha256", "environment_manifest_sha256",
        "training_driver_sha256", "evaluation_driver_sha256",
        "assembler_driver_sha256", "sbatch_sha256",
    }, "campaign provenance")
    frozen_provenance = protocol["provenance"]
    for field in ("base_commit", "base_tree", "overlay_contract_sha256"):
        _fail(provenance[field] == frozen_provenance[field],
              f"campaign {field} drift")
    for field, value in provenance.items():
        if field not in {"base_commit", "base_tree"}:
            _require_hash(value, f"campaign provenance {field}")
    _fail(provenance["assembler_driver_sha256"] == _sha256(ASSEMBLER_PATH),
          "campaign assembler drift")

    hardware = campaign["hardware"]
    _fail(isinstance(hardware, dict), "campaign hardware must be an object")
    _exact_keys(hardware, {
        "partition", "gpu_model", "gpu_profile", "gpu_count", "n_devices"
    }, "campaign hardware")
    for field in ("partition", "gpu_model", "gpu_profile"):
        _fail(isinstance(hardware[field], str) and hardware[field],
              f"campaign hardware {field} must be nonempty")
    _fail(hardware["gpu_count"] == 1 and hardware["n_devices"] == 1,
          "campaign must use one GPU and one JAX device")

    submissions = campaign["submissions"]
    seeds = protocol["training_seeds"]
    expected_cells = [(seed, arm) for seed in seeds for arm in ARMS]
    _fail(isinstance(submissions, list) and len(submissions) == len(expected_cells),
          "campaign must contain exactly ten submissions")
    seen_jobs: set[str] = set()
    for submission, (seed, arm) in zip(submissions, expected_cells):
        _fail(isinstance(submission, dict), "submission must be an object")
        _exact_keys(submission, {
            "arm", "training_seed", "evaluation_seed", "run_id", "job_id", "attempt"
        }, "submission")
        _fail(submission["arm"] == arm and submission["training_seed"] == seed,
              "submission order/cell drift")
        _fail(submission["evaluation_seed"] == 100000 + seed,
              "submission evaluation seed drift")
        _fail(submission["run_id"] == _run_id(seed, arm), "submission run ID drift")
        _fail(submission["attempt"] == 1, "v1 does not permit retries")
        job_id = submission["job_id"]
        _fail(isinstance(job_id, str) and JOB_RE.fullmatch(job_id) is not None,
              "invalid Slurm job ID")
        _fail(job_id not in seen_jobs, "duplicate Slurm job/task ID")
        seen_jobs.add(job_id)
    return campaign, actual_sha


def _package_payloads(arm: str) -> frozenset[str]:
    _fail(arm in ARMS, f"unknown arm for package closure: {arm}")
    if arm == "frontier":
        return COMMON_PACKAGE_PAYLOADS | FRONTIER_PACKAGE_PAYLOADS
    return COMMON_PACKAGE_PAYLOADS


def _read_manifest(path: Path, expected_names: set[str], label: str) -> dict[str, str]:
    _fail(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise GateError(f"invalid {label}: {path}") from exc
    listed: dict[str, str] = {}
    for number, line in enumerate(lines, 1):
        match = MANIFEST_LINE_RE.fullmatch(line)
        _fail(match is not None, f"unsafe {label} line {number}")
        digest, name = match.groups()
        _fail(name not in listed, f"duplicate {label} path: {name}")
        listed[name] = digest
    _fail(set(listed) == expected_names,
          f"{label} closure drift: expected {sorted(expected_names)}, got {sorted(listed)}")
    return listed


def _verify_package_closure(root: Path, run_id: str, arm: str) -> str:
    _fail(root.is_dir() and not root.is_symlink(), f"missing run directory: {run_id}")
    actual_names: set[str] = set()
    for path in root.iterdir():
        _fail(path.is_file() and not path.is_symlink(),
              f"run package contains non-regular entry: {path.name}")
        actual_names.add(path.name)
    payloads = _package_payloads(arm)
    expected_names = set(payloads) | {"SHA256SUMS", "COMPLETE"}
    _fail(actual_names == expected_names,
          f"run package closure drift for {run_id}: {sorted(actual_names)}")

    manifest = root / "SHA256SUMS"
    listed = _read_manifest(manifest, set(payloads), f"SHA256SUMS for {run_id}")
    for name, expected in listed.items():
        _fail(_sha256(root / name) == expected, f"payload hash mismatch: {run_id}/{name}")
    manifest_sha = _sha256(manifest)

    complete = _load_json(root / "COMPLETE", "COMPLETE")
    _exact_keys(complete, {"schema", "status", "run_id", "sha256sums_sha256", "file_count"},
                f"COMPLETE {run_id}")
    _fail(complete == {
        "schema": 2,
        "status": "complete",
        "run_id": run_id,
        "sha256sums_sha256": manifest_sha,
        "file_count": len(payloads),
    }, f"COMPLETE binding drift for {run_id}")
    return manifest_sha


RUN_CONTEXT_PROVENANCE_KEYS = frozenset({
    "base_commit",
    "base_tree",
    "overlay_contract_sha256",
    "bundle_manifest_sha256",
    "overlay_manifest_sha256",
    "applied_overlay_manifest_sha256",
    "environment_manifest_sha256",
    "training_driver_sha256",
    "evaluation_driver_sha256",
    "sbatch_sha256",
})


def _validate_run_context(
    path: Path,
    run_manifest: Mapping[str, Any],
    campaign: Mapping[str, Any],
    campaign_sha: str,
) -> dict[str, Any]:
    context = _load_json(path, "run context")
    _exact_keys(context, {
        "schema", "protocol_id", "purpose", "run_id", "arm", "training_seed",
        "job_id", "campaign_manifest_sha256", "provenance",
    }, f"run context {run_manifest['run_id']}")
    _fail(context["schema"] == 1
          and context["protocol_id"] == PROTOCOL_ID
          and context["purpose"] == PURPOSE,
          f"{run_manifest['run_id']} run-context identity drift")
    for field in ("run_id", "arm", "training_seed", "job_id"):
        _fail(context[field] == run_manifest[field],
              f"{run_manifest['run_id']} run-context drift: {field}")
    _fail(context["campaign_manifest_sha256"] == campaign_sha,
          f"{run_manifest['run_id']} run-context campaign drift")
    provenance = context["provenance"]
    _fail(isinstance(provenance, dict), "run-context provenance must be an object")
    _exact_keys(provenance, RUN_CONTEXT_PROVENANCE_KEYS,
                f"run-context provenance {run_manifest['run_id']}")
    expected = {
        field: campaign["provenance"][field]
        for field in RUN_CONTEXT_PROVENANCE_KEYS
    }
    _fail(provenance == expected,
          f"{run_manifest['run_id']} run-context provenance drift")
    return context


def _validate_source_receipt(
    source: Any,
    campaign: Mapping[str, Any],
    run_id: str,
    label: str,
    *,
    analyzer_eligible: bool,
) -> None:
    _fail(isinstance(source, dict), f"{run_id} {label} source receipt missing")
    _exact_keys(source, {
        "base_commit", "base_tree", "applied_overlay_manifest_sha256",
        "overlay_file_count", "worktree_status", "git_executable",
        "git_executable_sha256", "git_version",
    }, f"{label} source receipt {run_id}")
    provenance = campaign["provenance"]
    _fail(source["base_commit"] == provenance["base_commit"]
          and source["base_tree"] == provenance["base_tree"]
          and source["applied_overlay_manifest_sha256"]
          == provenance["applied_overlay_manifest_sha256"],
          f"{run_id} {label} source identity drift")
    _fail(_is_int(source["overlay_file_count"])
          and source["overlay_file_count"] >= 1,
          f"{run_id} {label} overlay file count drift")
    git_executable = source["git_executable"]
    _fail(isinstance(git_executable, str) and Path(git_executable).is_absolute()
          and ".." not in Path(git_executable).parts,
          f"{run_id} {label} Git executable receipt drift")
    _require_hash(source["git_executable_sha256"],
                  f"{run_id} {label} Git executable digest")
    _fail(isinstance(source["git_version"], str) and source["git_version"],
          f"{run_id} {label} Git version receipt drift")
    if analyzer_eligible:
        _fail(source["git_version"] == "git version 2.45.2",
              f"{run_id} {label} production Git version drift")
    status = source["worktree_status"]
    _fail(isinstance(status, dict)
          and len(status) == source["overlay_file_count"] + 1,
          f"{run_id} {label} worktree closure drift")
    for relative, state in status.items():
        _fail(isinstance(relative, str) and relative
              and not Path(relative).is_absolute()
              and ".." not in Path(relative).parts,
              f"{run_id} {label} unsafe worktree path")
        _fail(state in {" M", "??"},
              f"{run_id} {label} unsafe worktree status")


def _validate_training_source_package(
    root: Path,
    run_manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    campaign: Mapping[str, Any],
    protocol_sha: str,
    *,
    analyzer_eligible: bool,
) -> dict[str, Any]:
    run_id = str(run_manifest["run_id"])
    arm = str(run_manifest["arm"])
    binding = run_manifest["training_source_package"]
    _fail(isinstance(binding, dict), f"{run_id} training source binding missing")
    _exact_keys(binding, {
        "receipt_file", "receipt_sha256", "sha256sums_file",
        "sha256sums_sha256", "complete_file", "complete_sha256",
        "source_payload_count", "frontier_snapshot_file",
        "frontier_snapshot_sha256",
    }, f"training source binding {run_id}")
    expected_snapshot_file = (
        "training-frontier-buffer-snapshot.json" if arm == "frontier" else None
    )
    expected_source_names = {"training-receipt.json"}
    if arm == "frontier":
        expected_source_names.add("frontier-buffer-snapshot.json")
    expected_binding = {
        "receipt_file": "training-receipt.json",
        "receipt_sha256": _sha256(root / "training-receipt.json"),
        "sha256sums_file": "training-SHA256SUMS",
        "sha256sums_sha256": _sha256(root / "training-SHA256SUMS"),
        "complete_file": "training-COMPLETE",
        "complete_sha256": _sha256(root / "training-COMPLETE"),
        "source_payload_count": len(expected_source_names),
        "frontier_snapshot_file": expected_snapshot_file,
        "frontier_snapshot_sha256": (
            _sha256(root / expected_snapshot_file) if expected_snapshot_file else None
        ),
    }
    _fail(binding == expected_binding, f"{run_id} training source digest binding drift")

    listed = _read_manifest(
        root / "training-SHA256SUMS", expected_source_names,
        f"training source SHA256SUMS for {run_id}",
    )
    _fail(listed["training-receipt.json"] == _sha256(root / "training-receipt.json"),
          f"{run_id} training receipt source-manifest drift")
    if arm == "frontier":
        snapshot_path = root / "training-frontier-buffer-snapshot.json"
        _fail(listed["frontier-buffer-snapshot.json"] == _sha256(snapshot_path),
              f"{run_id} Frontier snapshot source-manifest drift")
        snapshot = _load_json(snapshot_path, "Frontier snapshot")
        _fail(snapshot.get("schema") == 1 and snapshot.get("status") == "completed"
              and snapshot.get("kind") == "frontier_plr_buffer_safe_snapshot"
              and snapshot.get("protocol_id") == PROTOCOL_ID
              and snapshot.get("purpose") == PURPOSE
              and snapshot.get("paper_evidence") is False
              and snapshot.get("run_id") == run_id
              and snapshot.get("arm") == arm
              and snapshot.get("training_seed") == run_manifest["training_seed"]
              and snapshot.get("checkpoint_sha256") == _sha256(root / "checkpoint.pkl"),
              f"{run_id} Frontier snapshot identity drift")

    source_manifest_sha = _sha256(root / "training-SHA256SUMS")
    complete = _load_json(root / "training-COMPLETE", "training COMPLETE")
    _exact_keys(complete, {
        "schema", "status", "run_id", "arm", "sha256sums_sha256", "file_count"
    }, f"training COMPLETE {run_id}")
    _fail(complete == {
        "schema": 1,
        "status": "complete",
        "run_id": run_id,
        "arm": arm,
        "sha256sums_sha256": source_manifest_sha,
        "file_count": len(expected_source_names),
    }, f"{run_id} training COMPLETE binding drift")

    receipt = _load_json(root / "training-receipt.json", "training receipt")
    _exact_keys(receipt, {
        "schema", "status", "protocol_id", "purpose", "paper_evidence",
        "endpoint_class", "run_id", "arm", "training_seed", "job_id", "resumed",
        "outer_cycles", "student_training_transitions", "transitions_per_outer_cycle",
        "n_updates", "upstream_n_grad_updates", "optimizer_step_applications",
        "periodic_evaluation_accounting", "optimizer_step_formula", "integrity",
        "engineering_test", "terminal_checkpoint", "config", "provenance", "endpoint",
        "wall_seconds", "frontier_snapshot",
    }, f"training receipt {run_id}")
    expected_endpoint_class = (
        "matched_development" if analyzer_eligible else "bounded_engineering_test"
    )
    _fail(receipt["schema"] == 1 and receipt["status"] == "completed"
          and receipt["protocol_id"] == PROTOCOL_ID and receipt["purpose"] == PURPOSE
          and receipt["paper_evidence"] is False
          and receipt["endpoint_class"] == expected_endpoint_class
          and receipt["resumed"] is False,
          f"{run_id} training receipt identity/status drift")
    for field in ("run_id", "arm", "training_seed", "job_id"):
        _fail(receipt[field] == context[field],
              f"{run_id} training receipt/context drift: {field}")
    terminal = receipt["terminal_checkpoint"]
    _fail(isinstance(terminal, dict), f"{run_id} terminal-checkpoint receipt missing")
    _exact_keys(terminal, {
        "path", "sha256", "saved_after_loop_termination",
        "periodic_checkpoint_used", "round_trip_counter_freshness",
    }, f"training terminal checkpoint {run_id}")
    _fail(terminal == {
        "path": "checkpoint.pkl",
        "sha256": _sha256(root / "checkpoint.pkl"),
        "saved_after_loop_termination": True,
        "periodic_checkpoint_used": False,
        "round_trip_counter_freshness": True,
    }, f"{run_id} training checkpoint receipt drift")
    endpoint_binding = receipt["endpoint"]
    _fail(endpoint_binding == {
        "path": "endpoint.json", "sha256": _sha256(root / "endpoint.json")
    }, f"{run_id} training endpoint receipt drift")
    config = receipt["config"]
    _fail(isinstance(config, dict), f"{run_id} training config receipt missing")
    _exact_keys(config, {
        "authored_path", "authored_sha256", "resolved",
        "resolved_canonical_sha256", "meta_sha256", "logs_sha256",
    }, f"training config receipt {run_id}")
    _fail(config["authored_sha256"] == run_manifest["config_template_sha256"]
          and config["authored_path"]
          == Path(run_manifest["config_template_path"]).name
          and config["resolved_canonical_sha256"]
          == _canonical_sha256(config["resolved"])
          and config["meta_sha256"] == _sha256(root / "meta.json")
          and config["logs_sha256"] == _sha256(root / "logs.csv"),
          f"{run_id} training config artifact binding drift")
    provenance = receipt["provenance"]
    _fail(isinstance(provenance, dict), f"{run_id} training provenance missing")
    _exact_keys(provenance, {
        "run_context", "run_context_sha256", "protocol_sha256",
        "training_driver_sha256", "source", "minimax_module", "backend",
        "devices",
    }, f"training provenance {run_id}")
    expected_backend = (
        "cpu"
        if not analyzer_eligible
        and isinstance(receipt.get("engineering_test"), dict)
        and receipt["engineering_test"].get("execution_mode") == "local"
        else "gpu"
    )
    _fail(provenance["run_context"] == context
          and provenance["run_context_sha256"] == _sha256(root / "run-context.json")
          and provenance["protocol_sha256"] == protocol_sha
          and provenance["training_driver_sha256"]
          == campaign["provenance"]["training_driver_sha256"]
          and provenance["backend"] == expected_backend,
          f"{run_id} training provenance drift")
    devices = provenance["devices"]
    _fail(isinstance(devices, list) and len(devices) == 1,
          f"{run_id} training device-count drift")
    device = devices[0]
    _fail(isinstance(device, dict)
          and set(device) == {"id", "platform", "device_kind"}
          and _is_int(device["id"])
          and device["id"] >= 0
          and device["platform"] == expected_backend
          and isinstance(device["device_kind"], str)
          and device["device_kind"],
          f"{run_id} training device receipt drift")
    _validate_source_receipt(
        provenance["source"], campaign, run_id, "training",
        analyzer_eligible=analyzer_eligible)
    engineering = receipt["engineering_test"]
    _fail(isinstance(engineering, dict)
          and set(engineering) == {"enabled", "execution_mode", "overrides"}
          and engineering["enabled"] is (not analyzer_eligible)
          and isinstance(engineering["overrides"], list)
          and (
              (analyzer_eligible
               and engineering["execution_mode"] == "production"
               and engineering["overrides"] == [])
              or (
                  not analyzer_eligible
                  and engineering["execution_mode"] in {"local", "slurm"}
                  and len(engineering["overrides"]) >= 1
              )
          ),
          f"{run_id} training engineering-mode receipt drift")
    if not analyzer_eligible and engineering["execution_mode"] == "slurm":
        _fail(provenance["source"]["git_version"] == "git version 2.45.2",
              f"{run_id} Slurm engineering Git version drift")
    _fail(isinstance(receipt["wall_seconds"], (int, float))
          and not isinstance(receipt["wall_seconds"], bool)
          and math.isfinite(float(receipt["wall_seconds"]))
          and float(receipt["wall_seconds"]) > 0.0,
          f"{run_id} training wall-time receipt drift")
    expected_snapshot = (
        {
            "path": "frontier-buffer-snapshot.json",
            "sha256": listed["frontier-buffer-snapshot.json"],
        }
        if arm == "frontier" else None
    )
    _fail(receipt["frontier_snapshot"] == expected_snapshot,
          f"{run_id} training snapshot receipt drift")
    return receipt


def _validate_evaluation_source_package(
    root: Path,
    run_manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    campaign: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_sha: str,
    training_receipt: Mapping[str, Any],
    *,
    analyzer_eligible: bool,
) -> dict[str, Any]:
    run_id = str(run_manifest["run_id"])
    binding = run_manifest["evaluation_source_package"]
    _fail(isinstance(binding, dict), f"{run_id} evaluation source binding missing")
    _exact_keys(binding, {
        "receipt_file", "receipt_sha256", "raw_results_file", "raw_results_sha256",
        "aggregate_results_file", "aggregate_results_sha256", "sha256sums_file",
        "sha256sums_sha256", "complete_file", "complete_sha256",
        "source_payload_count",
    }, f"evaluation source binding {run_id}")
    expected_binding = {
        "receipt_file": "evaluation-receipt.json",
        "receipt_sha256": _sha256(root / "evaluation-receipt.json"),
        "raw_results_file": "evaluation-episodes.jsonl",
        "raw_results_sha256": _sha256(root / "evaluation-episodes.jsonl"),
        "aggregate_results_file": "evaluation.csv",
        "aggregate_results_sha256": _sha256(root / "evaluation.csv"),
        "sha256sums_file": "evaluation-SHA256SUMS",
        "sha256sums_sha256": _sha256(root / "evaluation-SHA256SUMS"),
        "complete_file": "evaluation-COMPLETE",
        "complete_sha256": _sha256(root / "evaluation-COMPLETE"),
        "source_payload_count": 3,
    }
    _fail(binding == expected_binding, f"{run_id} evaluation source digest binding drift")
    source_names = {
        "evaluation-episodes.jsonl", "evaluation.csv", "evaluation-receipt.json"
    }
    listed = _read_manifest(
        root / "evaluation-SHA256SUMS", source_names,
        f"evaluation source SHA256SUMS for {run_id}",
    )
    for name in source_names:
        _fail(listed[name] == _sha256(root / name),
              f"{run_id} evaluation source-manifest drift: {name}")
    complete = _load_json(root / "evaluation-COMPLETE", "evaluation COMPLETE")
    _exact_keys(complete, {"schema", "status", "run_id", "sha256sums_sha256", "file_count"},
                f"evaluation COMPLETE {run_id}")
    _fail(complete == {
        "schema": 1,
        "status": "complete",
        "run_id": run_id,
        "sha256sums_sha256": _sha256(root / "evaluation-SHA256SUMS"),
        "file_count": 3,
    }, f"{run_id} evaluation COMPLETE binding drift")

    receipt = _load_json(root / "evaluation-receipt.json", "evaluation receipt")
    _exact_keys(receipt, {
        "schema", "status", "protocol_id", "purpose", "paper_evidence", "run_id",
        "arm", "training_seed", "evaluation_seed", "environments",
        "n_episodes_per_environment", "agent_indices", "synthetic_test_mode",
        "evaluation_transition_accounting", "terminal_checkpoint",
        "training_receipt_sha256", "meta_sha256", "provenance",
        "raw_results", "aggregate_results",
    }, f"evaluation receipt {run_id}")
    synthetic = receipt.get("synthetic_test_mode")
    _fail(receipt["schema"] == 1 and receipt["status"] == "completed"
          and receipt["protocol_id"] == PROTOCOL_ID and receipt["purpose"] == PURPOSE
          and receipt["paper_evidence"] is False
          and type(synthetic) is bool
          and (not analyzer_eligible or synthetic is False),
          f"{run_id} evaluation receipt identity/status drift")
    training_execution_mode = training_receipt["engineering_test"]["execution_mode"]
    _fail(not (training_execution_mode == "slurm" and synthetic is True),
          f"{run_id} Slurm engineering evaluation cannot be synthetic")
    for field in ("run_id", "arm", "training_seed"):
        _fail(receipt[field] == context[field],
              f"{run_id} evaluation receipt/context drift: {field}")
    expected_eval = protocol["evaluation"]
    _fail(receipt["evaluation_seed"] == 100000 + context["training_seed"]
          and receipt["environments"] == expected_eval["environments"]
          and receipt["n_episodes_per_environment"]
          == expected_eval["n_episodes_per_environment"]
          and receipt["agent_indices"] == [0],
          f"{run_id} evaluation parameter drift")
    _fail(receipt["terminal_checkpoint"]
          == {"sha256": _sha256(root / "checkpoint.pkl")}
          and receipt["training_receipt_sha256"]
          == _sha256(root / "training-receipt.json")
          and receipt["meta_sha256"] == _sha256(root / "meta.json"),
          f"{run_id} evaluation input binding drift")
    raw = receipt["raw_results"]
    aggregate = receipt["aggregate_results"]
    _fail(isinstance(raw, dict) and isinstance(aggregate, dict),
          f"{run_id} evaluation result receipt missing")
    _exact_keys(raw, {"path", "sha256", "record_count"}, f"raw result receipt {run_id}")
    _exact_keys(aggregate, {"path", "sha256", "values"},
                f"aggregate result receipt {run_id}")
    _fail(raw == {
        "path": "evaluation-episodes.jsonl",
        "sha256": _sha256(root / "evaluation-episodes.jsonl"),
        "record_count": 30,
    }, f"{run_id} raw evaluation receipt drift")
    _fail(aggregate["path"] == "evaluation.csv"
          and aggregate["sha256"] == _sha256(root / "evaluation.csv")
          and isinstance(aggregate["values"], dict),
          f"{run_id} aggregate evaluation receipt drift")
    provenance = receipt["provenance"]
    _fail(isinstance(provenance, dict), f"{run_id} evaluation provenance missing")
    _exact_keys(provenance, {
        "run_context", "run_context_sha256", "protocol_sha256",
        "evaluation_driver_sha256", "source", "runtime",
    }, f"evaluation provenance {run_id}")
    _fail(provenance["run_context"] == context
          and provenance["run_context_sha256"] == _sha256(root / "run-context.json")
          and provenance["protocol_sha256"] == protocol_sha
          and provenance["evaluation_driver_sha256"]
          == campaign["provenance"]["evaluation_driver_sha256"],
          f"{run_id} evaluation provenance drift")
    _validate_source_receipt(
        provenance["source"], campaign, run_id, "evaluation",
        analyzer_eligible=analyzer_eligible)
    _fail(provenance["source"] == training_receipt["provenance"]["source"],
          f"{run_id} evaluator/trainer source receipt drift")
    accounting = receipt["evaluation_transition_accounting"]
    _fail(isinstance(accounting, dict), f"{run_id} evaluation accounting missing")
    expected_effective = 0 if synthetic else 13500
    expected_scans = not synthetic
    independent_transitions = accounting.get("engineering_independent_verification_transitions")
    _fail(_is_int(independent_transitions)
          and independent_transitions in {0, 13500}
          and (analyzer_eligible is False or independent_transitions == 0),
          f"{run_id} independent evaluation accounting drift")
    _fail(accounting.get("environment_count") == 3
          and accounting.get("episodes_per_environment") == 10
          and accounting.get("max_episode_horizon") == 450
          and accounting.get("per_environment_max_episode_horizons") == [450, 450, 450]
          and accounting.get("budgeted_primary_max_transitions") == 13500
          and accounting.get("effective_primary_transitions") == expected_effective
          and accounting.get("primary_runner_scans_full_horizon") is expected_scans
          and accounting.get("total_runtime_transitions")
          == expected_effective + independent_transitions
          and accounting.get("excluded_from_student_training_transitions") is True,
          f"{run_id} production evaluation accounting drift")
    return receipt


def _validate_meta(meta: dict[str, Any], run_id: str, seed: int, job_id: str,
                   arm: str, protocol: dict[str, Any]) -> None:
    config = meta.get("config")
    _fail(isinstance(config, dict), f"{run_id} meta config missing")
    _fail(config.get("seed") == seed, f"{run_id} meta seed drift")
    _fail(config.get("xpid") == run_id, f"{run_id} meta xpid drift")
    _fail(isinstance(config.get("log_dir"), str) and config["log_dir"],
          f"{run_id} meta log_dir missing")
    wandb = config.get("wandb_args")
    _fail(isinstance(wandb, dict) and wandb.get("api_key") is None,
          f"{run_id} must disable external WandB logging")
    dynamic = {"seed", "xpid", "log_dir", "wandb_args"}
    static = {key: value for key, value in config.items() if key not in dynamic}
    expected = json.loads(json.dumps(protocol["expected_static_meta_config"]))
    expected["train_runner_args"]["ued_score"] = protocol["arms"][arm]["ued_score"]
    _fail(static == expected, f"{run_id} resolved meta config drift")

    _fail(meta.get("xpid") == run_id, f"{run_id} metadata xpid drift")
    git = meta.get("git")
    _fail(isinstance(git, dict)
          and git.get("commit") == protocol["provenance"]["base_commit"],
          f"{run_id} Git commit drift")
    slurm = meta.get("slurm")
    _fail(isinstance(slurm, dict) and str(slurm.get("job_id")) == job_id.split("_")[0],
          f"{run_id} Slurm metadata drift")
    environment = meta.get("env")
    _fail(isinstance(environment, dict), f"{run_id} environment metadata missing")
    leaking = sorted(
        key for key, value in environment.items()
        if SECRET_KEY_RE.search(str(key)) and value not in (None, "")
    )
    _fail(not leaking, f"{run_id} metadata contains secret-like environment keys: {leaking}")


def _integer_cell(value: Any, label: str) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"non-numeric integer cell {label}") from exc
    _fail(math.isfinite(parsed) and parsed.is_integer(), f"non-integral cell {label}")
    return int(parsed)


def _validate_logs(path: Path, run_id: str, arm: str, endpoint: dict[str, Any],
                   protocol: dict[str, Any]) -> None:
    with path.open("r", encoding="utf-8", newline="") as stream:
        header = stream.readline().rstrip("\r\n")
        _fail(header.startswith("# "), f"{run_id} logs.csv lacks upstream header")
        fieldnames = next(csv.reader([header[2:]]))
        _fail(len(fieldnames) == len(set(fieldnames)), f"{run_id} duplicate log headers")
        required = {"_tick", "n_updates", "steps", "real_steps"}
        frontier_fields = {
            "plr/frontier_n_rollouts",
            "plr/frontier_n_eval",
            "plr/frontier_group_size_match",
            "plr/frontier_incomplete_group_count",
            "plr/frontier_duplicate_new_group_count",
        }
        required |= frontier_fields if arm == "frontier" else set()
        _fail(required <= set(fieldnames), f"{run_id} missing required log fields")
        if arm == "maxmc":
            _fail(not (frontier_fields & set(fieldnames)),
                  f"{run_id} MaxMC logs expose Frontier-only fields")
        reader = csv.DictReader(stream, fieldnames=fieldnames)
        rows = list(reader)
    _fail(rows, f"{run_id} has no log rows")
    previous_tick = -1
    previous_updates = -1
    per_cycle = protocol["training_budget"]["training_transitions_per_outer_cycle"]
    target_ppo_updates = protocol["training_budget"]["target_student_ppo_updates"]
    for index, row in enumerate(rows, 1):
        _fail(None not in row, f"{run_id} malformed log row {index}")
        tick = _integer_cell(row["_tick"], f"{run_id}:row{index}:tick")
        updates = _integer_cell(row["n_updates"], f"{run_id}:row{index}:n_updates")
        steps = _integer_cell(row["steps"], f"{run_id}:row{index}:steps")
        real_steps = _integer_cell(row["real_steps"], f"{run_id}:row{index}:real_steps")
        _fail(tick > previous_tick and tick % 10 == 0, f"{run_id} log tick drift")
        _fail(previous_updates <= updates <= target_ppo_updates,
              f"{run_id} student PPO-update trajectory drift")
        _fail(steps == real_steps == tick * per_cycle, f"{run_id} logged transition drift")
        if arm == "frontier":
            expected_frontier = {
                "plr/frontier_n_rollouts": 8,
                "plr/frontier_n_eval": 8,
                "plr/frontier_group_size_match": 1,
                "plr/frontier_incomplete_group_count": 0,
                "plr/frontier_duplicate_new_group_count": 0,
            }
            for field, expected in expected_frontier.items():
                _fail(_integer_cell(row[field], f"{run_id}:row{index}:{field}") == expected,
                      f"{run_id} Frontier log integrity drift: {field}")
        previous_tick = tick
        previous_updates = updates
    outer = endpoint["outer_cycles"]
    expected_last_tick = (outer // 10) * 10
    _fail(previous_tick == expected_last_tick, f"{run_id} logs do not reach terminal interval")
    if previous_tick == outer:
        _fail(previous_updates == target_ppo_updates,
              f"{run_id} terminal logged student PPO-update drift")


def _validate_evaluation_shape(path: Path, run_id: str,
                               environments: list[str]) -> None:
    expected = [
        f"eval/a0:test_{metric}:{env}"
        for env in environments for metric in ("return", "solved_rate")
    ]
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _fail(reader.fieldnames == expected, f"{run_id} evaluation columns/order drift")
        _fail(len(reader.fieldnames) == len(set(reader.fieldnames)),
              f"{run_id} duplicate evaluation columns")
        rows = list(reader)
    _fail(len(rows) == 1 and None not in rows[0],
          f"{run_id} evaluation must contain exactly one complete row")


def _validate_run(root: Path, submission: dict[str, Any], campaign: dict[str, Any],
                  campaign_sha: str, protocol: dict[str, Any],
                  protocol_sha: str) -> ValidatedRun:
    arm = submission["arm"]
    seed = submission["training_seed"]
    run_id = submission["run_id"]
    job_id = submission["job_id"]
    manifest_sha = _verify_package_closure(root, run_id, arm)

    run_manifest = _load_json(root / "run-manifest.json", "run manifest")
    _exact_keys(run_manifest, {
        "schema", "protocol_id", "purpose", "paper_evidence",
        "analyzer_eligible", "endpoint_class",
        "campaign_manifest_sha256",
        "run_id", "arm", "training_seed", "evaluation_seed", "job_id",
        "config_template_path", "config_template_sha256", "provenance",
        "run_context", "training_source_package", "evaluation_source_package",
        "evaluation",
    }, f"run manifest {run_id}")
    _fail(run_manifest["schema"] == 2
          and run_manifest["protocol_id"] == PROTOCOL_ID
          and run_manifest["purpose"] == PURPOSE
          and run_manifest["paper_evidence"] is False
          and run_manifest["analyzer_eligible"] is True
          and run_manifest["endpoint_class"] == "matched_development",
          f"{run_id} manifest identity drift")
    expected_identity = {
        "campaign_manifest_sha256": campaign_sha,
        "run_id": run_id,
        "arm": arm,
        "training_seed": seed,
        "evaluation_seed": 100000 + seed,
        "job_id": job_id,
        "config_template_path": protocol["arms"][arm]["config_path"],
        "config_template_sha256": protocol["arms"][arm]["config_sha256"],
    }
    for field, value in expected_identity.items():
        _fail(run_manifest[field] == value, f"{run_id} run manifest drift: {field}")
    _fail(run_manifest["provenance"] == campaign["provenance"],
          f"{run_id} provenance drift")
    context_binding = run_manifest["run_context"]
    _fail(isinstance(context_binding, dict), f"{run_id} run-context binding missing")
    _exact_keys(context_binding, {"file", "sha256"}, f"run-context binding {run_id}")
    _fail(context_binding == {
        "file": "run-context.json", "sha256": _sha256(root / "run-context.json")
    }, f"{run_id} run-context digest binding drift")
    context = _validate_run_context(
        root / "run-context.json", run_manifest, campaign, campaign_sha)
    training_receipt = _validate_training_source_package(
        root, run_manifest, context, campaign, protocol_sha,
        analyzer_eligible=True)
    evaluation_receipt = _validate_evaluation_source_package(
        root, run_manifest, context, campaign, protocol, protocol_sha, training_receipt,
        analyzer_eligible=True)

    endpoint = _load_json(root / "endpoint.json", "endpoint receipt")
    _exact_keys(endpoint, {
        "schema", "status", "run_id", "arm", "training_seed", "n_updates",
        "n_grad_updates", "optimizer_step_applications", "outer_cycles",
        "student_training_transitions",
        "checkpoint_file", "checkpoint_sha256",
        "terminal_checkpoint_saved_after_training", "resumed", "frontier_integrity",
    }, f"endpoint {run_id}")
    _fail(endpoint["schema"] == 1 and endpoint["status"] == "completed",
          f"{run_id} endpoint is not complete")
    _fail(endpoint["run_id"] == run_id and endpoint["arm"] == arm
          and endpoint["training_seed"] == seed, f"{run_id} endpoint identity drift")
    budget = protocol["training_budget"]
    target_ppo_updates = budget["target_student_ppo_updates"]
    _fail(endpoint["n_updates"] == endpoint["n_grad_updates"] == target_ppo_updates,
          f"{run_id} student PPO-update integrity-counter drift")
    expected_optimizer_applications = (
        endpoint["n_updates"]
        * budget["ppo_epochs_per_student_update"]
        * budget["ppo_minibatches_per_epoch"]
    )
    _fail(endpoint["optimizer_step_applications"]
          == expected_optimizer_applications
          == budget["target_optimizer_step_applications"],
          f"{run_id} optimizer-step application accounting drift")
    outer = endpoint["outer_cycles"]
    _fail(_is_int(outer) and outer >= 30000, f"{run_id} invalid outer cycle count")
    transitions = endpoint["student_training_transitions"]
    per_cycle = protocol["training_budget"]["training_transitions_per_outer_cycle"]
    _fail(_is_int(transitions) and transitions == outer * per_cycle,
          f"{run_id} transition accounting drift")
    _fail(endpoint["checkpoint_file"] == "checkpoint.pkl"
          and endpoint["checkpoint_sha256"] == _sha256(root / "checkpoint.pkl"),
          f"{run_id} terminal checkpoint binding drift")
    _fail(endpoint["terminal_checkpoint_saved_after_training"] is True,
          f"{run_id} uses a periodic/nonterminal checkpoint")
    _fail(endpoint["resumed"] is False, f"{run_id} is a resumed run")
    receipt_counter_links = {
        "n_updates": "n_updates",
        "upstream_n_grad_updates": "n_grad_updates",
        "optimizer_step_applications": "optimizer_step_applications",
        "outer_cycles": "outer_cycles",
        "student_training_transitions": "student_training_transitions",
    }
    for receipt_field, endpoint_field in receipt_counter_links.items():
        _fail(training_receipt[receipt_field] == endpoint[endpoint_field],
              f"{run_id} training receipt/endpoint drift: {receipt_field}")
    _fail(training_receipt["transitions_per_outer_cycle"] == per_cycle,
          f"{run_id} training receipt transition-unit drift")
    formula = training_receipt["optimizer_step_formula"]
    _fail(formula == {
        "n_updates": target_ppo_updates,
        "student_n_epochs": budget["ppo_epochs_per_student_update"],
        "student_n_minibatches": budget["ppo_minibatches_per_epoch"],
    }, f"{run_id} optimizer receipt formula drift")

    integrity = endpoint["frontier_integrity"]
    if arm == "frontier":
        _fail(isinstance(integrity, dict), f"{run_id} lacks Frontier integrity")
        _exact_keys(integrity, {
            "n_rollouts", "n_eval", "group_size_match", "incomplete_group_count",
            "duplicate_new_group_count", "buffer_total_trials", "buffer_total_successes",
        }, f"Frontier integrity {run_id}")
        _fail(integrity["n_rollouts"] == integrity["n_eval"] == 8
              and integrity["group_size_match"] is True,
              f"{run_id} Frontier group contract drift")
        _fail(integrity["incomplete_group_count"] == 0
              and integrity["duplicate_new_group_count"] == 0,
              f"{run_id} nonzero Frontier delivery counters")
        trials = integrity["buffer_total_trials"]
        successes = integrity["buffer_total_successes"]
        _fail(_is_int(trials) and _is_int(successes) and 0 <= successes <= trials,
              f"{run_id} invalid Frontier posterior totals")
    else:
        _fail(integrity is None, f"{run_id} MaxMC endpoint has Frontier integrity state")

    scheduler = _load_json(root / "scheduler.json", "scheduler receipt")
    _exact_keys(scheduler, {
        "schema", "job_id", "state", "exit_code", "partition", "gpu_model",
        "gpu_profile", "gpu_count", "elapsed_seconds", "max_rss_bytes",
        "peak_gpu_memory_bytes", "terminal_sacct_retrieved_utc",
    }, f"scheduler {run_id}")
    _fail(scheduler["schema"] == 1 and scheduler["job_id"] == job_id,
          f"{run_id} scheduler identity drift")
    _fail(scheduler["state"] == "COMPLETED" and scheduler["exit_code"] == "0:0",
          f"{run_id} Slurm job not cleanly completed")
    hardware = campaign["hardware"]
    for field in ("partition", "gpu_model", "gpu_profile", "gpu_count"):
        _fail(scheduler[field] == hardware[field], f"{run_id} hardware drift: {field}")
    for field in ("elapsed_seconds", "max_rss_bytes", "peak_gpu_memory_bytes"):
        _fail(_is_int(scheduler[field]) and scheduler[field] >= 0,
              f"{run_id} invalid resource field: {field}")
    _fail(scheduler["elapsed_seconds"] > 0, f"{run_id} zero elapsed time")
    _fail(isinstance(scheduler["terminal_sacct_retrieved_utc"], str)
          and UTC_RE.fullmatch(scheduler["terminal_sacct_retrieved_utc"]) is not None,
          f"{run_id} terminal accounting timestamp drift")

    meta = _load_json(root / "meta.json", "upstream meta")
    _validate_meta(meta, run_id, seed, job_id, arm, protocol)
    _fail(training_receipt["config"]["resolved"] == meta["config"],
          f"{run_id} training receipt/resolved metadata drift")
    _validate_logs(root / "logs.csv", run_id, arm, endpoint, protocol)

    evaluation = run_manifest["evaluation"]
    _fail(isinstance(evaluation, dict), f"{run_id} evaluation receipt missing")
    _exact_keys(evaluation, {
        "seed", "n_episodes", "environments", "checkpoint_sha256",
        "results_file", "results_sha256", "raw_results_file",
        "raw_results_sha256", "raw_record_count", "receipt_file",
        "receipt_sha256",
    }, f"evaluation receipt {run_id}")
    expected_evaluation = protocol["evaluation"]
    _fail(evaluation["seed"] == 100000 + seed
          and evaluation["n_episodes"] == expected_evaluation["n_episodes_per_environment"]
          and evaluation["environments"] == expected_evaluation["environments"],
          f"{run_id} evaluator parameter drift")
    _fail(evaluation["checkpoint_sha256"] == endpoint["checkpoint_sha256"],
          f"{run_id} evaluator checkpoint drift")
    _fail(evaluation["results_file"] == "evaluation.csv"
          and evaluation["results_sha256"] == _sha256(root / "evaluation.csv"),
          f"{run_id} evaluator result binding drift")
    _fail(evaluation["raw_results_file"] == "evaluation-episodes.jsonl"
          and evaluation["raw_results_sha256"]
          == _sha256(root / "evaluation-episodes.jsonl")
          and evaluation["raw_record_count"] == 30
          and evaluation["receipt_file"] == "evaluation-receipt.json"
          and evaluation["receipt_sha256"]
          == _sha256(root / "evaluation-receipt.json"),
          f"{run_id} raw evaluator binding drift")
    _fail(evaluation_receipt["terminal_checkpoint"]["sha256"]
          == evaluation["checkpoint_sha256"],
          f"{run_id} evaluation receipt/checkpoint drift")
    _validate_evaluation_shape(root / "evaluation.csv", run_id,
                               expected_evaluation["environments"])

    return ValidatedRun(
        arm=arm,
        seed=seed,
        run_id=run_id,
        root=root,
        student_ppo_updates=endpoint["n_updates"],
        optimizer_step_applications=endpoint["optimizer_step_applications"],
        outer_cycles=outer,
        training_transitions=transitions,
        package_manifest_sha256=manifest_sha,
    )


def _read_raw_metrics(
    run: ValidatedRun,
    environments: list[str],
    n_episodes: int,
) -> dict[str, float]:
    """Validate the 30 raw records after every package has passed closure."""
    path = run.root / "evaluation-episodes.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise GateError(f"{run.run_id} raw evaluation is unreadable") from exc
    _fail(len(lines) == len(environments) * n_episodes,
          f"{run.run_id} raw evaluation record count drift")
    expected_order = [
        (environment, episode)
        for environment in environments
        for episode in range(n_episodes)
    ]
    records: list[dict[str, Any]] = []
    for index, (line, expected_identity) in enumerate(zip(lines, expected_order), 1):
        try:
            record = json.loads(line, object_pairs_hook=_json_no_duplicate_pairs)
        except json.JSONDecodeError as exc:
            raise GateError(f"{run.run_id} invalid raw JSON record {index}") from exc
        _fail(isinstance(record, dict), f"{run.run_id} raw record {index} is not an object")
        _exact_keys(record, {"environment", "episode", "agent_index", "solved", "return"},
                    f"raw evaluation record {run.run_id}:{index}")
        _fail((record["environment"], record["episode"]) == expected_identity,
              f"{run.run_id} raw evaluation order drift at record {index}")
        _fail(record["agent_index"] == 0 and type(record["solved"]) is bool,
              f"{run.run_id} raw evaluation identity/type drift at record {index}")
        value = record["return"]
        _fail(isinstance(value, (int, float)) and not isinstance(value, bool)
              and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0,
              f"{run.run_id} raw return drift at record {index}")
        _fail(bool(float(value) > 0.0) is record["solved"],
              f"{run.run_id} raw solved/return drift at record {index}")
        records.append(record)

    aggregate: dict[str, float] = {}
    for environment in environments:
        rows = [record for record in records if record["environment"] == environment]
        aggregate[f"eval/a0:test_return:{environment}"] = statistics.fmean(
            float(record["return"]) for record in rows)
        aggregate[f"eval/a0:test_solved_rate:{environment}"] = statistics.fmean(
            int(record["solved"]) for record in rows)

    receipt = _load_json(run.root / "evaluation-receipt.json", "evaluation receipt")
    receipt_values = receipt["aggregate_results"]["values"]
    _fail(set(receipt_values) == set(aggregate),
          f"{run.run_id} aggregate receipt columns drift")
    for field, expected in aggregate.items():
        value = receipt_values[field]
        _fail(isinstance(value, (int, float)) and not isinstance(value, bool)
              and math.isfinite(float(value))
              and abs(float(value) - expected) <= 2e-12,
              f"{run.run_id} aggregate receipt/raw mismatch: {field}")
    return aggregate


def _read_metrics(
    run: ValidatedRun,
    environments: list[str],
    n_episodes: int,
    raw_aggregate: Mapping[str, float],
) -> dict[str, float]:
    """Parse aggregate CSV numbers only after every raw file is validated."""
    with (run.root / "evaluation.csv").open("r", encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    solved: dict[str, float] = {}
    for env in environments:
        solved_key = f"eval/a0:test_solved_rate:{env}"
        return_key = f"eval/a0:test_return:{env}"
        try:
            solved_value = float(row[solved_key])
            return_value = float(row[return_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise GateError(f"{run.run_id} has a non-numeric evaluation endpoint") from exc
        _fail(math.isfinite(solved_value) and 0.0 <= solved_value <= 1.0,
              f"{run.run_id} solved rate out of range: {env}")
        _fail(abs(solved_value * n_episodes - round(solved_value * n_episodes)) <= 1e-7,
              f"{run.run_id} solved rate is incompatible with {n_episodes} episodes: {env}")
        _fail(math.isfinite(return_value) and 0.0 <= return_value <= 1.0,
              f"{run.run_id} return out of AMaze range: {env}")
        _fail(abs(solved_value - raw_aggregate[solved_key]) <= 2e-12
              and abs(return_value - raw_aggregate[return_key]) <= 2e-12,
              f"{run.run_id} aggregate CSV/raw mismatch: {env}")
        solved[env] = solved_value
    return solved


def _sign_flip_pvalue(differences: list[float]) -> float:
    observed = abs(statistics.fmean(differences))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted = abs(statistics.fmean(sign * value for sign, value in zip(signs, differences)))
        extreme += int(permuted >= observed - 1e-15)
        total += 1
    return extreme / total


def analyze(campaign_path: Path, expected_campaign_sha: str,
            runs_root: Path) -> dict[str, Any]:
    protocol, protocol_sha = _load_protocol()
    repository_preflight()
    campaign, campaign_sha = _validate_campaign(
        campaign_path, expected_campaign_sha, protocol, protocol_sha)
    _fail(runs_root.is_dir() and not runs_root.is_symlink(), "runs root is not a directory")

    expected_run_ids = {submission["run_id"] for submission in campaign["submissions"]}
    actual_entries = list(runs_root.iterdir())
    _fail(all(path.is_dir() and not path.is_symlink() for path in actual_entries),
          "runs root contains a non-directory entry")
    _fail({path.name for path in actual_entries} == expected_run_ids,
          "runs root has missing or extra run packages")

    # Integrity phase: no metric value is parsed or emitted here.
    validated: dict[tuple[int, str], ValidatedRun] = {}
    for submission in campaign["submissions"]:
        run = _validate_run(
            runs_root / submission["run_id"], submission, campaign, campaign_sha,
            protocol, protocol_sha)
        key = (run.seed, run.arm)
        _fail(key not in validated, f"duplicate validated cell: {key}")
        validated[key] = run

    for seed in protocol["training_seeds"]:
        frontier = validated[(seed, "frontier")]
        maxmc = validated[(seed, "maxmc")]
        budget = protocol["training_budget"]
        _fail(frontier.outer_cycles == maxmc.outer_cycles,
              f"seed {seed} outer-cycle budget mismatch")
        _fail(frontier.student_ppo_updates == maxmc.student_ppo_updates
              == budget["target_student_ppo_updates"],
              f"seed {seed} student PPO-update budget mismatch")
        _fail(frontier.optimizer_step_applications
              == maxmc.optimizer_step_applications
              == budget["target_optimizer_step_applications"],
              f"seed {seed} optimizer-step application budget mismatch")
        _fail(frontier.training_transitions == maxmc.training_transitions,
              f"seed {seed} transition budget mismatch")

    # Raw-record validation begins only after all ten packages and five budgets
    # pass.  No numeric aggregate CSV cell has been parsed at this point.
    environments = protocol["evaluation"]["environments"]
    n_episodes = protocol["evaluation"]["n_episodes_per_environment"]
    raw_aggregates = {
        key: _read_raw_metrics(run, environments, n_episodes)
        for key, run in validated.items()
    }
    # Aggregate endpoint parsing begins only after all 300 raw episode records
    # have passed schema, order, value, and receipt aggregation checks.
    metrics = {
        key: _read_metrics(run, environments, n_episodes, raw_aggregates[key])
        for key, run in validated.items()
    }
    paired_rows: list[dict[str, Any]] = []
    macro_differences: list[float] = []
    environment_differences: dict[str, list[float]] = {env: [] for env in environments}
    for seed in protocol["training_seeds"]:
        frontier_values = metrics[(seed, "frontier")]
        maxmc_values = metrics[(seed, "maxmc")]
        deltas = {env: frontier_values[env] - maxmc_values[env] for env in environments}
        macro_frontier = statistics.fmean(frontier_values.values())
        macro_maxmc = statistics.fmean(maxmc_values.values())
        macro_delta = macro_frontier - macro_maxmc
        macro_differences.append(macro_delta)
        for env in environments:
            environment_differences[env].append(deltas[env])
        paired_rows.append({
            "training_seed": seed,
            "evaluation_seed": 100000 + seed,
            "student_ppo_updates_per_arm": validated[(seed, "frontier")].student_ppo_updates,
            "optimizer_step_applications_per_arm": (
                validated[(seed, "frontier")].optimizer_step_applications
            ),
            "outer_cycles": validated[(seed, "frontier")].outer_cycles,
            "training_transitions_per_arm": validated[(seed, "frontier")].training_transitions,
            "frontier_solved_rate": frontier_values,
            "maxmc_solved_rate": maxmc_values,
            "per_environment_delta": deltas,
            "frontier_macro_solved_rate": macro_frontier,
            "maxmc_macro_solved_rate": macro_maxmc,
            "macro_delta": macro_delta,
        })

    mean_delta = statistics.fmean(macro_differences)
    standard_error = statistics.stdev(macro_differences) / math.sqrt(len(macro_differences))
    # Frozen n=5 => df=4. This constant avoids an analysis-time SciPy dependency.
    t_critical_df4_95 = 2.7764451051977987
    half_width = t_critical_df4_95 * standard_error
    per_environment_mean_delta = {
        env: statistics.fmean(values) for env, values in environment_differences.items()
    }
    min_env_delta = min(per_environment_mean_delta.values())
    advance = mean_delta > 0.0 and min_env_delta >= -0.05

    return {
        "schema": 1,
        "protocol_id": PROTOCOL_ID,
        "scope": PURPOSE,
        "paper_evidence": False,
        "integrity_gate": "PASS",
        "all_ten_packages_validated_before_metric_parse": True,
        "all_300_raw_episode_records_validated_before_aggregate_csv_parse": True,
        "budget_semantics": {
            "student_ppo_updates_per_arm": protocol["training_budget"][
                "target_student_ppo_updates"
            ],
            "upstream_n_updates_integrity_counter": 30000,
            "upstream_n_grad_updates_integrity_counter": 30000,
            "optimizer_step_applications_per_arm": protocol["training_budget"][
                "target_optimizer_step_applications"
            ],
            "optimizer_step_formula": (
                "n_updates * student PPO epochs * student PPO minibatches = 30000 * 5 * 1"
            ),
        },
        "hashes": {
            "protocol_sha256": protocol_sha,
            "analyzer_sha256": _sha256(Path(__file__).resolve()),
            "campaign_manifest_sha256": campaign_sha,
            "run_package_sha256sums_sha256": {
                run.run_id: run.package_manifest_sha256
                for run in sorted(validated.values(), key=lambda item: item.run_id)
            },
        },
        "paired_runs": paired_rows,
        "primary": {
            "metric": protocol["evaluation"]["primary_metric"],
            "direction": "Frontier-minus-MaxMC",
            "n_paired_training_seeds": len(macro_differences),
            "paired_differences": macro_differences,
            "mean_difference": mean_delta,
            "sample_standard_deviation": statistics.stdev(macro_differences),
            "standard_error": standard_error,
            "student_t_df": 4,
            "two_sided_95_percent_ci": [mean_delta - half_width, mean_delta + half_width],
            "exact_two_sided_sign_flip_pvalue_descriptive": _sign_flip_pvalue(macro_differences),
        },
        "safety": {
            "per_environment_mean_delta": per_environment_mean_delta,
            "minimum_per_environment_mean_delta": min_env_delta,
            "allowed_floor": -0.05,
        },
        "decision": {
            "advance_exact_grouped_frontier": advance,
            "positive_primary_mean": mean_delta > 0.0,
            "no_validation_environment_regresses_more_than_0.05": min_env_delta >= -0.05,
            "interpretation": (
                "development selection signal only; confirmatory panel and seeds remain sealed"
            ),
        },
    }


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    _fail(path.is_absolute(), "output path must be absolute")
    _fail(path.parent.is_dir(), "output parent directory does not exist")
    _fail(not path.exists() and not path.is_symlink(), "refusing to overwrite analysis output")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true",
                        help="verify only frozen repository/config inputs")
    parser.add_argument("--campaign-manifest", type=Path)
    parser.add_argument("--expected-campaign-sha256")
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.preflight:
        _fail(all(value is None for value in (
            args.campaign_manifest, args.expected_campaign_sha256, args.runs_root, args.output
        )), "--preflight cannot be combined with endpoint arguments")
        print(json.dumps(repository_preflight(), indent=2, sort_keys=True))
        return 0
    _fail(args.campaign_manifest is not None
          and args.expected_campaign_sha256 is not None
          and args.runs_root is not None,
          "sealed analysis requires campaign manifest, its expected hash, and runs root")
    result = analyze(args.campaign_manifest, args.expected_campaign_sha256, args.runs_root)
    if args.output is not None:
        _atomic_write_json(args.output, result)
        print(json.dumps({
            "status": "PASS",
            "output": str(args.output),
            "output_sha256": _sha256(args.output),
            "advance_exact_grouped_frontier": result["decision"]["advance_exact_grouped_frontier"],
        }, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
