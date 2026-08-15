#!/usr/bin/env python3
"""Atomically assemble one schema-2 matched-development analyzer package.

This is the only bridge from the terminal trainer and external evaluator into
the preregistered development analyzer.  It accepts no loose evaluation CSV:
the four-file training output, closed training sidecar, closed evaluation
package, run context, campaign, command/log files, and terminal scheduler
receipt are all validated and copied into a fresh temporary directory.  The
analyzer's own fail-closed validators are run against that directory before it
is atomically published.

The output remains engineering/development evidence only.  This program does
not submit jobs, evaluate policies, read existing endpoints, or unpickle model
checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ued_benchmark.analysis import preregistered_dev_analysis as gate


TRAINING_OUTPUT_NAMES = frozenset({
    "checkpoint.pkl", "endpoint.json", "logs.csv", "meta.json",
})
EVALUATION_PACKAGE_NAMES = frozenset({
    "evaluation-episodes.jsonl", "evaluation.csv", "evaluation-receipt.json",
    "SHA256SUMS", "COMPLETE",
})
ANCILLARY_OUTPUT_NAMES = {
    "command": "command.txt",
    "scheduler": "scheduler.json",
    "stdout": "stdout.log",
    "stderr": "stderr.log",
}


class AssemblyError(RuntimeError):
    """Raised when no atomic analyzer package can be safely assembled."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssemblyError(message)


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"unsafe or missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_absolute(path: Path, *, directory: bool, label: str) -> Path:
    require(path.is_absolute(), f"{label} must be an absolute path")
    require(".." not in path.parts, f"{label} contains path traversal")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise AssemblyError(f"missing {label}: {path}") from exc
    require(resolved == path, f"{label} contains a symlink or noncanonical component")
    if directory:
        require(path.is_dir() and not path.is_symlink(), f"unsafe {label}: {path}")
    else:
        require(path.is_file() and not path.is_symlink(), f"unsafe {label}: {path}")
    return path


def _validate_directory(root: Path, expected: set[str], label: str) -> None:
    actual: set[str] = set()
    for entry in root.iterdir():
        require(entry.is_file() and not entry.is_symlink(),
                f"{label} contains a non-regular entry: {entry.name}")
        actual.add(entry.name)
    require(actual == expected,
            f"{label} closure drift: expected {sorted(expected)}, got {sorted(actual)}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=gate._json_no_duplicate_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"invalid {label}: {path}") from exc
    require(isinstance(document, dict), f"{label} must be a JSON object")
    return document


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path.name}")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_text(path: Path, value: str) -> None:
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path.name}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_file(source: Path, destination: Path) -> None:
    require(not destination.exists() and not destination.is_symlink(),
            f"refusing to overwrite {destination.name}")
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    require(sha256(destination) == sha256(source),
            f"copy digest mismatch: {source.name} -> {destination.name}")


def _training_sidecar_names(arm: str) -> set[str]:
    names = {"training-receipt.json", "SHA256SUMS", "COMPLETE"}
    if arm == "frontier":
        names.add("frontier-buffer-snapshot.json")
    return names


def _context_identity(
    context: Mapping[str, Any],
    context_sha: str,
    campaign: Mapping[str, Any],
    campaign_sha: str,
) -> dict[str, Any]:
    expected_keys = {
        "schema", "protocol_id", "purpose", "run_id", "arm", "training_seed",
        "job_id", "campaign_manifest_sha256", "provenance",
    }
    require(set(context) == expected_keys, "run-context keys drift")
    require(context["schema"] == 1
            and context["protocol_id"] == gate.PROTOCOL_ID
            and context["purpose"] == gate.PURPOSE,
            "run-context identity drift")
    require(context["arm"] in gate.ARMS, "run-context arm drift")
    require(isinstance(context["training_seed"], int)
            and not isinstance(context["training_seed"], bool),
            "run-context training seed is invalid")
    require(context["campaign_manifest_sha256"] == campaign_sha,
            "run-context campaign digest drift")
    expected_provenance = {
        field: campaign["provenance"][field]
        for field in gate.RUN_CONTEXT_PROVENANCE_KEYS
    }
    require(context["provenance"] == expected_provenance,
            "run-context provenance drift")
    matching = [
        submission for submission in campaign["submissions"]
        if submission["run_id"] == context["run_id"]
    ]
    require(len(matching) == 1, "run context does not select exactly one campaign submission")
    submission = matching[0]
    for field in ("arm", "training_seed", "job_id"):
        require(context[field] == submission[field],
                f"run-context/submission drift: {field}")
    require(submission["evaluation_seed"] == 100000 + context["training_seed"],
            "submission evaluation seed drift")
    require(gate.HASH_RE.fullmatch(context_sha) is not None,
            "expected run-context SHA-256 is malformed")
    return submission


def _validate_engineering_campaign(
    path: Path,
    expected_sha: str,
    protocol_sha: str,
) -> tuple[dict[str, Any], str]:
    require(gate.HASH_RE.fullmatch(expected_sha or "") is not None,
            "expected engineering campaign SHA-256 is malformed")
    actual_sha = sha256(path)
    require(actual_sha == expected_sha, "engineering campaign digest mismatch")
    campaign = _load_json(path, "engineering campaign manifest")
    expected_keys = {
        "schema", "protocol_id", "purpose", "created_utc",
        "frozen_before_endpoint_access", "protocol_sha256", "analyzer_sha256",
        "provenance", "hardware", "submissions",
    }
    require(set(campaign) == expected_keys, "engineering campaign keys drift")
    require(campaign["schema"] == 1
            and campaign["protocol_id"] == gate.PROTOCOL_ID
            and campaign["purpose"] == gate.PURPOSE
            and campaign["frozen_before_endpoint_access"] is True,
            "engineering campaign identity drift")
    require(isinstance(campaign["created_utc"], str)
            and gate.UTC_RE.fullmatch(campaign["created_utc"]) is not None,
            "engineering campaign UTC timestamp drift")
    require(campaign["protocol_sha256"] == protocol_sha,
            "engineering campaign protocol drift")
    require(campaign["analyzer_sha256"] == gate._sha256(Path(gate.__file__).resolve()),
            "engineering campaign analyzer drift")
    provenance = campaign["provenance"]
    expected_provenance = set(gate.RUN_CONTEXT_PROVENANCE_KEYS) | {
        "assembler_driver_sha256"
    }
    require(isinstance(provenance, dict) and set(provenance) == expected_provenance,
            "engineering campaign provenance keys drift")
    protocol, _ = gate._load_protocol()
    for field in ("base_commit", "base_tree", "overlay_contract_sha256"):
        require(provenance[field] == protocol["provenance"][field],
                f"engineering campaign {field} drift")
    for field, value in provenance.items():
        if field not in {"base_commit", "base_tree"}:
            require(gate.HASH_RE.fullmatch(str(value)) is not None,
                    f"engineering campaign malformed provenance hash: {field}")
    hardware = campaign["hardware"]
    require(isinstance(hardware, dict)
            and set(hardware) == {
                "partition", "gpu_model", "gpu_profile", "gpu_count", "n_devices"
            }, "engineering campaign hardware keys drift")
    require(all(isinstance(hardware[field], str) and hardware[field]
                for field in ("partition", "gpu_model", "gpu_profile"))
            and hardware["gpu_count"] == hardware["n_devices"] == 1,
            "engineering campaign hardware drift")
    submissions = campaign["submissions"]
    require(isinstance(submissions, list) and len(submissions) == 1,
            "engineering campaign must contain exactly one submission")
    submission = submissions[0]
    require(isinstance(submission, dict)
            and set(submission) == {
                "arm", "training_seed", "evaluation_seed", "run_id", "job_id", "attempt"
            }, "engineering submission keys drift")
    require(submission["arm"] in gate.ARMS
            and isinstance(submission["training_seed"], int)
            and not isinstance(submission["training_seed"], bool)
            and submission["evaluation_seed"] == 100000 + submission["training_seed"]
            and isinstance(submission["run_id"], str)
            and submission["run_id"].startswith("engineering-")
            and isinstance(submission["job_id"], str)
            and gate.JOB_RE.fullmatch(submission["job_id"]) is not None
            and submission["attempt"] == 1,
            "engineering submission identity drift")
    return campaign, actual_sha


def _validate_engineering_package(
    root: Path,
    submission: Mapping[str, Any],
    campaign: Mapping[str, Any],
    campaign_sha: str,
    protocol: Mapping[str, Any],
    protocol_sha: str,
) -> None:
    """Validate a bounded package without invoking the 30k analyzer gate."""
    run_id = str(submission["run_id"])
    arm = str(submission["arm"])
    gate._verify_package_closure(root, run_id, arm)
    manifest = gate._load_json(root / "run-manifest.json", "engineering run manifest")
    expected_manifest_keys = {
        "schema", "protocol_id", "purpose", "paper_evidence", "analyzer_eligible",
        "endpoint_class", "campaign_manifest_sha256", "run_id", "arm",
        "training_seed", "evaluation_seed", "job_id", "config_template_path",
        "config_template_sha256", "provenance", "run_context",
        "training_source_package", "evaluation_source_package", "evaluation",
    }
    require(set(manifest) == expected_manifest_keys,
            "engineering run-manifest keys drift")
    require(manifest["schema"] == 2
            and manifest["protocol_id"] == gate.PROTOCOL_ID
            and manifest["purpose"] == gate.PURPOSE
            and manifest["paper_evidence"] is False
            and manifest["analyzer_eligible"] is False
            and manifest["endpoint_class"] == "bounded_engineering_test"
            and manifest["campaign_manifest_sha256"] == campaign_sha
            and manifest["run_id"] == run_id
            and manifest["arm"] == arm
            and manifest["training_seed"] == submission["training_seed"]
            and manifest["evaluation_seed"] == submission["evaluation_seed"]
            and manifest["job_id"] == submission["job_id"]
            and manifest["provenance"] == campaign["provenance"],
            "engineering run-manifest identity drift")
    require(manifest["config_template_path"] == protocol["arms"][arm]["config_path"]
            and manifest["config_template_sha256"]
            == protocol["arms"][arm]["config_sha256"],
            "engineering config-template drift")
    require(manifest["run_context"] == {
        "file": "run-context.json", "sha256": sha256(root / "run-context.json")
    }, "engineering run-context digest binding drift")
    context = gate._validate_run_context(
        root / "run-context.json", manifest, campaign, campaign_sha)
    training_receipt = gate._validate_training_source_package(
        root, manifest, context, campaign, protocol_sha, analyzer_eligible=False)
    evaluation_receipt = gate._validate_evaluation_source_package(
        root, manifest, context, campaign, protocol, protocol_sha, training_receipt,
        analyzer_eligible=False)

    endpoint = gate._load_json(root / "endpoint.json", "engineering endpoint")
    endpoint_keys = {
        "schema", "status", "run_id", "arm", "training_seed", "n_updates",
        "n_grad_updates", "optimizer_step_applications", "outer_cycles",
        "student_training_transitions", "checkpoint_file", "checkpoint_sha256",
        "terminal_checkpoint_saved_after_training", "resumed", "frontier_integrity",
    }
    require(set(endpoint) == endpoint_keys
            and endpoint["schema"] == 1 and endpoint["status"] == "completed"
            and endpoint["run_id"] == run_id and endpoint["arm"] == arm
            and endpoint["training_seed"] == submission["training_seed"]
            and endpoint["checkpoint_file"] == "checkpoint.pkl"
            and endpoint["checkpoint_sha256"] == sha256(root / "checkpoint.pkl")
            and endpoint["terminal_checkpoint_saved_after_training"] is True
            and endpoint["resumed"] is False,
            "engineering endpoint identity/binding drift")
    for key in ("n_updates", "n_grad_updates", "optimizer_step_applications",
                "outer_cycles", "student_training_transitions"):
        require(isinstance(endpoint[key], int) and not isinstance(endpoint[key], bool)
                and endpoint[key] >= 1,
                f"engineering endpoint counter drift: {key}")
    require(endpoint["n_updates"] == endpoint["n_grad_updates"]
            and endpoint["n_updates"] == training_receipt["n_updates"]
            and endpoint["n_grad_updates"] == training_receipt["upstream_n_grad_updates"]
            and endpoint["optimizer_step_applications"]
            == training_receipt["optimizer_step_applications"]
            and endpoint["outer_cycles"] == training_receipt["outer_cycles"]
            and endpoint["student_training_transitions"]
            == training_receipt["student_training_transitions"]
            and endpoint["student_training_transitions"]
            == endpoint["outer_cycles"] * training_receipt["transitions_per_outer_cycle"],
            "engineering endpoint/training receipt counter drift")
    formula = training_receipt["optimizer_step_formula"]
    require(isinstance(formula, dict)
            and set(formula) == {
                "n_updates", "student_n_epochs", "student_n_minibatches"
            }
            and formula["n_updates"] == endpoint["n_updates"]
            and endpoint["optimizer_step_applications"]
            == formula["n_updates"] * formula["student_n_epochs"]
            * formula["student_n_minibatches"],
            "engineering optimizer formula drift")
    if arm == "frontier":
        integrity = endpoint["frontier_integrity"]
        require(isinstance(integrity, dict)
                and integrity.get("n_rollouts") == integrity.get("n_eval") == 8
                and integrity.get("group_size_match") is True
                and integrity.get("incomplete_group_count") == 0
                and integrity.get("duplicate_new_group_count") == 0,
                "engineering Frontier delivery drift")
    else:
        require(endpoint["frontier_integrity"] is None,
                "engineering MaxMC has Frontier integrity state")

    meta = gate._load_json(root / "meta.json", "engineering metadata")
    require(meta.get("xpid") == run_id
            and isinstance(meta.get("config"), dict)
            and meta["config"] == training_receipt["config"]["resolved"],
            "engineering metadata/config drift")
    scheduler = gate._load_json(root / "scheduler.json", "engineering scheduler")
    scheduler_keys = {
        "schema", "job_id", "state", "exit_code", "partition", "gpu_model",
        "gpu_profile", "gpu_count", "elapsed_seconds", "max_rss_bytes",
        "peak_gpu_memory_bytes", "terminal_sacct_retrieved_utc",
    }
    require(set(scheduler) == scheduler_keys
            and scheduler["schema"] == 1
            and scheduler["job_id"] == submission["job_id"]
            and scheduler["state"] == "COMPLETED"
            and scheduler["exit_code"] == "0:0",
            "engineering scheduler receipt drift")
    for field in ("partition", "gpu_model", "gpu_profile", "gpu_count"):
        require(scheduler[field] == campaign["hardware"][field],
                f"engineering scheduler hardware drift: {field}")
    evaluation = manifest["evaluation"]
    require(isinstance(evaluation, dict)
            and evaluation.get("seed") == submission["evaluation_seed"]
            and evaluation.get("n_episodes") == 10
            and evaluation.get("environments") == protocol["evaluation"]["environments"]
            and evaluation.get("checkpoint_sha256") == endpoint["checkpoint_sha256"]
            and evaluation.get("results_file") == "evaluation.csv"
            and evaluation.get("results_sha256") == sha256(root / "evaluation.csv")
            and evaluation.get("raw_results_file") == "evaluation-episodes.jsonl"
            and evaluation.get("raw_results_sha256")
            == sha256(root / "evaluation-episodes.jsonl")
            and evaluation.get("raw_record_count") == 30
            and evaluation.get("receipt_file") == "evaluation-receipt.json"
            and evaluation.get("receipt_sha256")
            == sha256(root / "evaluation-receipt.json")
            and evaluation_receipt["terminal_checkpoint"]["sha256"]
            == endpoint["checkpoint_sha256"],
            "engineering evaluation manifest drift")
    validated = gate.ValidatedRun(
        arm=arm,
        seed=int(submission["training_seed"]),
        run_id=run_id,
        root=root,
        student_ppo_updates=endpoint["n_updates"],
        optimizer_step_applications=endpoint["optimizer_step_applications"],
        outer_cycles=endpoint["outer_cycles"],
        training_transitions=endpoint["student_training_transitions"],
        package_manifest_sha256=sha256(root / "SHA256SUMS"),
    )
    raw = gate._read_raw_metrics(
        validated, protocol["evaluation"]["environments"], 10)
    gate._read_metrics(validated, protocol["evaluation"]["environments"], 10, raw)


def validate_output(cli: argparse.Namespace) -> dict[str, Any]:
    """Revalidate one already-published package without modifying it."""
    package = _safe_absolute(cli.output_dir, directory=True, label="output package")
    campaign_path = _safe_absolute(
        cli.campaign_manifest, directory=False, label="campaign manifest")
    require(gate.HASH_RE.fullmatch(
        cli.expected_package_sha256sums_sha256 or "") is not None,
        "expected package SHA256SUMS digest is malformed")
    require(sha256(package / "SHA256SUMS")
            == cli.expected_package_sha256sums_sha256,
            "package SHA256SUMS digest mismatch")
    protocol, protocol_sha = gate._load_protocol()
    if cli.engineering_test_mode:
        campaign, campaign_sha = _validate_engineering_campaign(
            campaign_path, cli.expected_campaign_sha256, protocol_sha)
    else:
        try:
            campaign, campaign_sha = gate._validate_campaign(
                campaign_path, cli.expected_campaign_sha256, protocol, protocol_sha)
        except gate.GateError as exc:
            raise AssemblyError(str(exc)) from exc
    driver_sha = sha256(Path(__file__).resolve())
    require(campaign["provenance"]["assembler_driver_sha256"] == driver_sha,
            "campaign binds another assembler")
    manifest = _load_json(package / "run-manifest.json", "run manifest")
    matches = [
        submission for submission in campaign["submissions"]
        if submission["run_id"] == manifest.get("run_id")
    ]
    require(len(matches) == 1, "package does not select exactly one campaign submission")
    try:
        if cli.engineering_test_mode:
            _validate_engineering_package(
                package, matches[0], campaign, campaign_sha, protocol, protocol_sha)
        else:
            validated = gate._validate_run(
                package, matches[0], campaign, campaign_sha, protocol, protocol_sha)
            raw = gate._read_raw_metrics(
                validated, protocol["evaluation"]["environments"], 10)
            gate._read_metrics(
                validated, protocol["evaluation"]["environments"], 10, raw)
    except gate.GateError as exc:
        raise AssemblyError(str(exc)) from exc
    return {
        "schema": 1,
        "status": "PASS",
        "purpose": gate.PURPOSE,
        "paper_evidence": False,
        "analyzer_eligible": not cli.engineering_test_mode,
        "run_id": manifest["run_id"],
        "package_sha256sums_sha256": sha256(package / "SHA256SUMS"),
        "raw_record_count": 30,
    }


def assemble(cli: argparse.Namespace) -> dict[str, Any]:
    required_inputs = {
        "expected_assembler_sha256": cli.expected_assembler_sha256,
        "run_context": cli.run_context,
        "expected_run_context_sha256": cli.expected_run_context_sha256,
        "training_output_dir": cli.training_output_dir,
        "training_sidecar_dir": cli.training_sidecar_dir,
        "evaluation_package_dir": cli.evaluation_package_dir,
        "command": cli.command,
        "scheduler": cli.scheduler,
        "stdout": cli.stdout,
        "stderr": cli.stderr,
    }
    missing = sorted(key for key, value in required_inputs.items() if value is None)
    require(not missing, f"assembly is missing required arguments: {missing}")
    driver_path = Path(__file__).resolve()
    driver_sha = sha256(driver_path)
    require(gate.HASH_RE.fullmatch(cli.expected_assembler_sha256 or "") is not None,
            "expected assembler SHA-256 is malformed")
    require(driver_sha == cli.expected_assembler_sha256,
            "assembler SHA-256 mismatch")

    campaign_path = _safe_absolute(
        cli.campaign_manifest, directory=False, label="campaign manifest")
    run_context_path = _safe_absolute(
        cli.run_context, directory=False, label="run context")
    training_output = _safe_absolute(
        cli.training_output_dir, directory=True, label="training output")
    training_sidecar = _safe_absolute(
        cli.training_sidecar_dir, directory=True, label="training sidecar")
    evaluation_package = _safe_absolute(
        cli.evaluation_package_dir, directory=True, label="evaluation package")
    ancillary = {
        name: _safe_absolute(getattr(cli, name), directory=False, label=name)
        for name in ANCILLARY_OUTPUT_NAMES
    }
    output = cli.output_dir
    require(output.is_absolute(), "output directory must be absolute")
    require(".." not in output.parts, "output directory contains path traversal")
    require(not output.exists() and not output.is_symlink(), "output directory already exists")
    output_parent = _safe_absolute(output.parent, directory=True, label="output parent")
    source_roots = {training_output, training_sidecar, evaluation_package}
    require(all(output != source and not output.is_relative_to(source)
                for source in source_roots),
            "output may not overlap an input package")

    protocol, protocol_sha = gate._load_protocol()
    if cli.engineering_test_mode:
        campaign, campaign_sha = _validate_engineering_campaign(
            campaign_path, cli.expected_campaign_sha256, protocol_sha)
    else:
        try:
            campaign, campaign_sha = gate._validate_campaign(
                campaign_path, cli.expected_campaign_sha256, protocol, protocol_sha)
        except gate.GateError as exc:
            raise AssemblyError(str(exc)) from exc
    require(campaign["provenance"]["assembler_driver_sha256"] == driver_sha,
            "campaign binds another assembler")
    context_sha = sha256(run_context_path)
    require(context_sha == cli.expected_run_context_sha256,
            "run-context SHA-256 mismatch")
    context = _load_json(run_context_path, "run context")
    submission = _context_identity(
        context, cli.expected_run_context_sha256, campaign, campaign_sha)
    arm = str(context["arm"])
    run_id = str(context["run_id"])
    require(output.name == run_id, "output basename must equal the deterministic run ID")

    _validate_directory(training_output, set(TRAINING_OUTPUT_NAMES), "training output")
    _validate_directory(training_sidecar, _training_sidecar_names(arm), "training sidecar")
    _validate_directory(evaluation_package, set(EVALUATION_PACKAGE_NAMES),
                        "evaluation package")

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output_parent))
    try:
        for name in sorted(TRAINING_OUTPUT_NAMES):
            _copy_file(training_output / name, temporary / name)
        _copy_file(run_context_path, temporary / "run-context.json")
        _copy_file(training_sidecar / "training-receipt.json",
                   temporary / "training-receipt.json")
        _copy_file(training_sidecar / "SHA256SUMS", temporary / "training-SHA256SUMS")
        _copy_file(training_sidecar / "COMPLETE", temporary / "training-COMPLETE")
        if arm == "frontier":
            _copy_file(
                training_sidecar / "frontier-buffer-snapshot.json",
                temporary / "training-frontier-buffer-snapshot.json",
            )
        for name in ("evaluation-episodes.jsonl", "evaluation.csv", "evaluation-receipt.json"):
            _copy_file(evaluation_package / name, temporary / name)
        _copy_file(evaluation_package / "SHA256SUMS", temporary / "evaluation-SHA256SUMS")
        _copy_file(evaluation_package / "COMPLETE", temporary / "evaluation-COMPLETE")
        for key, destination_name in ANCILLARY_OUTPUT_NAMES.items():
            _copy_file(ancillary[key], temporary / destination_name)

        snapshot_file = (
            "training-frontier-buffer-snapshot.json" if arm == "frontier" else None
        )
        run_manifest = {
            "schema": 2,
            "protocol_id": gate.PROTOCOL_ID,
            "purpose": gate.PURPOSE,
            "paper_evidence": False,
            "analyzer_eligible": not cli.engineering_test_mode,
            "endpoint_class": (
                "bounded_engineering_test"
                if cli.engineering_test_mode else "matched_development"
            ),
            "campaign_manifest_sha256": campaign_sha,
            "run_id": run_id,
            "arm": arm,
            "training_seed": context["training_seed"],
            "evaluation_seed": submission["evaluation_seed"],
            "job_id": context["job_id"],
            "config_template_path": protocol["arms"][arm]["config_path"],
            "config_template_sha256": protocol["arms"][arm]["config_sha256"],
            "provenance": campaign["provenance"],
            "run_context": {
                "file": "run-context.json",
                "sha256": sha256(temporary / "run-context.json"),
            },
            "training_source_package": {
                "receipt_file": "training-receipt.json",
                "receipt_sha256": sha256(temporary / "training-receipt.json"),
                "sha256sums_file": "training-SHA256SUMS",
                "sha256sums_sha256": sha256(temporary / "training-SHA256SUMS"),
                "complete_file": "training-COMPLETE",
                "complete_sha256": sha256(temporary / "training-COMPLETE"),
                "source_payload_count": 2 if arm == "frontier" else 1,
                "frontier_snapshot_file": snapshot_file,
                "frontier_snapshot_sha256": (
                    sha256(temporary / snapshot_file) if snapshot_file else None
                ),
            },
            "evaluation_source_package": {
                "receipt_file": "evaluation-receipt.json",
                "receipt_sha256": sha256(temporary / "evaluation-receipt.json"),
                "raw_results_file": "evaluation-episodes.jsonl",
                "raw_results_sha256": sha256(temporary / "evaluation-episodes.jsonl"),
                "aggregate_results_file": "evaluation.csv",
                "aggregate_results_sha256": sha256(temporary / "evaluation.csv"),
                "sha256sums_file": "evaluation-SHA256SUMS",
                "sha256sums_sha256": sha256(temporary / "evaluation-SHA256SUMS"),
                "complete_file": "evaluation-COMPLETE",
                "complete_sha256": sha256(temporary / "evaluation-COMPLETE"),
                "source_payload_count": 3,
            },
            "evaluation": {
                "seed": submission["evaluation_seed"],
                "n_episodes": protocol["evaluation"]["n_episodes_per_environment"],
                "environments": protocol["evaluation"]["environments"],
                "checkpoint_sha256": sha256(temporary / "checkpoint.pkl"),
                "results_file": "evaluation.csv",
                "results_sha256": sha256(temporary / "evaluation.csv"),
                "raw_results_file": "evaluation-episodes.jsonl",
                "raw_results_sha256": sha256(temporary / "evaluation-episodes.jsonl"),
                "raw_record_count": 30,
                "receipt_file": "evaluation-receipt.json",
                "receipt_sha256": sha256(temporary / "evaluation-receipt.json"),
            },
        }
        _write_json(temporary / "run-manifest.json", run_manifest)
        payloads = gate._package_payloads(arm)
        manifest_text = "".join(
            f"{sha256(temporary / name)}  {name}\n" for name in sorted(payloads)
        )
        _write_text(temporary / "SHA256SUMS", manifest_text)
        _write_json(temporary / "COMPLETE", {
            "schema": 2,
            "status": "complete",
            "run_id": run_id,
            "sha256sums_sha256": sha256(temporary / "SHA256SUMS"),
            "file_count": len(payloads),
        })

        try:
            if cli.engineering_test_mode:
                _validate_engineering_package(
                    temporary, submission, campaign, campaign_sha, protocol, protocol_sha)
            else:
                validated = gate._validate_run(
                    temporary, submission, campaign, campaign_sha, protocol, protocol_sha)
                raw = gate._read_raw_metrics(
                    validated,
                    protocol["evaluation"]["environments"],
                    protocol["evaluation"]["n_episodes_per_environment"],
                )
                gate._read_metrics(
                    validated,
                    protocol["evaluation"]["environments"],
                    protocol["evaluation"]["n_episodes_per_environment"],
                    raw,
                )
        except gate.GateError as exc:
            raise AssemblyError(str(exc)) from exc

        os.replace(temporary, output)
        directory_fd = os.open(output_parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        try:
            if cli.engineering_test_mode:
                _validate_engineering_package(
                    output, submission, campaign, campaign_sha, protocol, protocol_sha)
            else:
                validated = gate._validate_run(
                    output, submission, campaign, campaign_sha, protocol, protocol_sha)
                raw = gate._read_raw_metrics(
                    validated,
                    protocol["evaluation"]["environments"],
                    protocol["evaluation"]["n_episodes_per_environment"],
                )
                gate._read_metrics(
                    validated,
                    protocol["evaluation"]["environments"],
                    protocol["evaluation"]["n_episodes_per_environment"],
                    raw,
                )
        except gate.GateError as exc:
            raise AssemblyError(f"published package revalidation failed: {exc}") from exc
        return {
            "schema": 1,
            "status": "complete",
            "purpose": gate.PURPOSE,
            "paper_evidence": False,
            "analyzer_eligible": not cli.engineering_test_mode,
            "run_id": run_id,
            "output_dir": str(output),
            "assembler_sha256": driver_sha,
            "package_sha256sums_sha256": sha256(output / "SHA256SUMS"),
            "training_source_sha256sums_sha256": sha256(output / "training-SHA256SUMS"),
            "evaluation_source_sha256sums_sha256": sha256(output / "evaluation-SHA256SUMS"),
            "raw_record_count": 30,
        }
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--expected-campaign-sha256", required=True)
    parser.add_argument("--run-context", type=Path)
    parser.add_argument("--expected-run-context-sha256")
    parser.add_argument("--expected-assembler-sha256")
    parser.add_argument("--training-output-dir", type=Path)
    parser.add_argument("--training-sidecar-dir", type=Path)
    parser.add_argument("--evaluation-package-dir", type=Path)
    parser.add_argument("--command", type=Path)
    parser.add_argument("--scheduler", type=Path)
    parser.add_argument("--stdout", type=Path)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--engineering-test-mode",
        action="store_true",
        help=(
            "assemble and structurally validate one bounded non-evidence run; "
            "the output is explicitly ineligible for the production analyzer"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="revalidate output-dir without writing; requires its expected SHA256SUMS digest",
    )
    parser.add_argument("--expected-package-sha256sums-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        cli = parse_cli(argv)
        if cli.validate_only:
            result = validate_output(cli)
        else:
            require(cli.expected_package_sha256sums_sha256 is None,
                    "expected package digest is valid only with --validate-only")
            result = assemble(cli)
    except (AssemblyError, gate.GateError, AssertionError, KeyError, TypeError, ValueError) as exc:
        print(f"MATCHED_RUN_ASSEMBLY_REFUSED: {exc}", file=sys.stderr)
        return 1
    print(
        ("MATCHED_RUN_VALIDATION_COMPLETE " if cli.validate_only
         else "MATCHED_RUN_ASSEMBLY_COMPLETE ")
        + f"run_id={result['run_id']} package={result['package_sha256sums_sha256']} "
        f"raw_records={result['raw_record_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
