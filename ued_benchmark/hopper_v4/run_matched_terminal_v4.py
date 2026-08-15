#!/usr/bin/env python3
"""Run one matched minimax UED arm and save the true terminal runner state.

This driver intentionally owns the ``ExperimentRunner.step`` loop.  The
upstream ``ExperimentRunner.train`` method returns no runner state, so a
checkpoint saved by a caller after that method returns cannot be guaranteed to
be the terminal state.  This program writes both an analyzer-compatible
``endpoint.json`` and a richer ``training-receipt.json`` only after the target
student PPO-update count has been reached and the terminal checkpoint has been
round-tripped.

The two proposed matched inputs are the v4 Frontier and group-matched MaxMC
templates in ``ued_benchmark/configs``.  This DRAFT driver accepts only bounded
local or Slurm engineering modes; overrides are typed, bounded, whitelisted,
recorded, and force ``paper_evidence=false``.  There is no resume path and no
authorization to create a matched-development or paper endpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np


PROTOCOL_ID = "ued-dev-frontier-vs-maxmc-4x8-b500-tie-aware-v2-draft"
PURPOSE = "draft_engineering_development_only_no_endpoint_authorization_not_paper_evidence"
BASE_COMMIT = "d053054c5290a04c1c4cd8b55704d999cad73e30"
BASE_TREE = "b0cace1fc54984e21a842f12d15d0b899e33d270"
OVERLAY_CONTRACT_SHA256 = (
    "3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b"
)
OVERLAY_VERSION = "frontier-activity-tie-aware-v4"
FLOAT32_DIAGNOSTIC_TOLERANCE = 2e-6
FLOAT32_DISTINCT_SCORE_EQUIVALENCE_TOLERANCE = 5e-7
FLOAT64_RECONSTRUCTION_TOLERANCE = 1e-10
PERIODIC_EVAL_ENVIRONMENTS = 3
PERIODIC_EVAL_EPISODES_PER_ENVIRONMENT = 10
PERIODIC_EVAL_HORIZON = 450
ARMS = ("frontier", "maxmc")
HASH_KEYS = {
    "bundle_manifest_sha256",
    "overlay_manifest_sha256",
    "applied_overlay_manifest_sha256",
    "environment_manifest_sha256",
    "training_driver_sha256",
    "evaluation_driver_sha256",
    "sbatch_sha256",
}
CAMPAIGN_PROVENANCE_KEYS = HASH_KEYS | {
    "base_commit",
    "base_tree",
    "overlay_contract_sha256",
    "assembler_driver_sha256",
}
CAMPAIGN_KEYS = {
    "schema",
    "protocol_id",
    "purpose",
    "created_utc",
    "frozen_before_endpoint_access",
    "protocol_sha256",
    "analyzer_sha256",
    "provenance",
    "hardware",
    "submissions",
}
CAMPAIGN_HARDWARE_KEYS = {
    "partition",
    "gpu_model",
    "gpu_profile",
    "gpu_count",
    "n_devices",
}
CAMPAIGN_SUBMISSION_KEYS = {
    "arm",
    "training_seed",
    "evaluation_seed",
    "run_id",
    "job_id",
    "attempt",
}
UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
SLURM_JOB = re.compile(r"^[0-9]+(?:_[0-9]+)?$")
SIDECAR_MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$")


class DriverError(RuntimeError):
    """Raised when a run cannot satisfy the frozen driver contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DriverError(message)


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"unsafe or missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"unsafe or missing {label}: {path}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            require(key not in value, f"duplicate JSON key in {label}: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DriverError(f"invalid {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(path.parent.is_dir() and not path.parent.is_symlink(), "unsafe output parent")
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, value: str) -> None:
    require(path.parent.is_dir() and not path.parent.is_symlink(), "unsafe output parent")
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_training_sidecar(root: Path, run_id: str, arm: str) -> str:
    require(root.is_dir() and not root.is_symlink(), "training sidecar is missing")
    payloads = {"training-receipt.json", "plr-replay-snapshot.json"}
    expected_names = payloads | {"SHA256SUMS", "COMPLETE"}
    actual_names: set[str] = set()
    for entry in root.iterdir():
        require(entry.is_file() and not entry.is_symlink(), f"unsafe sidecar entry: {entry.name}")
        actual_names.add(entry.name)
    require(actual_names == expected_names, "training sidecar closure drift")
    listed: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = SIDECAR_MANIFEST_LINE.fullmatch(line)
        require(match is not None, "unsafe training sidecar SHA256SUMS line")
        digest, name = match.groups()
        require(name not in listed, "duplicate training sidecar manifest path")
        listed[name] = digest
    require(set(listed) == payloads, "training sidecar manifest closure drift")
    for name, expected in listed.items():
        require(sha256(root / name) == expected, f"training sidecar hash drift: {name}")
    manifest_sha = sha256(root / "SHA256SUMS")
    complete = load_json(root / "COMPLETE", "training sidecar COMPLETE")
    require(
        complete
        == {
            "schema": 1,
            "status": "complete",
            "run_id": run_id,
            "arm": arm,
            "sha256sums_sha256": manifest_sha,
            "file_count": len(payloads),
        },
        "training sidecar COMPLETE binding drift",
    )
    return manifest_sha


def write_training_sidecar(
    output_dir: Path,
    receipt: dict[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    require(output_dir.is_absolute(), "training sidecar directory must be absolute")
    require(not output_dir.exists() and not output_dir.is_symlink(), "training sidecar exists")
    require(output_dir.parent.is_dir() and not output_dir.parent.is_symlink(), "unsafe sidecar parent")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        snapshot_path = temporary / "plr-replay-snapshot.json"
        atomic_json(snapshot_path, snapshot)
        receipt["plr_snapshot"] = {
            "path": snapshot_path.name,
            "sha256": sha256(snapshot_path),
        }
        payload_names = ["training-receipt.json", snapshot_path.name]
        atomic_json(temporary / "training-receipt.json", receipt)
        manifest = "".join(
            f"{sha256(temporary / name)}  {name}\n" for name in sorted(payload_names)
        )
        _atomic_text(temporary / "SHA256SUMS", manifest)
        complete = {
            "schema": 1,
            "status": "complete",
            "run_id": receipt["run_id"],
            "arm": receipt["arm"],
            "sha256sums_sha256": sha256(temporary / "SHA256SUMS"),
            "file_count": len(payload_names),
        }
        atomic_json(temporary / "COMPLETE", complete)
        validate_training_sidecar(temporary, receipt["run_id"], receipt["arm"])
        os.replace(temporary, output_dir)
        validate_training_sidecar(output_dir, receipt["run_id"], receipt["arm"])
        return receipt
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _is_hash(value: Any, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def active_slurm_job_id() -> str:
    """Return the canonical Slurm job or array-task identity."""

    job_id = os.environ.get("SLURM_JOB_ID")
    require(
        isinstance(job_id, str) and job_id.isdigit(),
        "active Slurm job ID is unavailable or invalid",
    )
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    require(
        (array_job_id is None) == (array_task_id is None),
        "partial Slurm array identity is invalid",
    )
    if array_job_id is None:
        return job_id
    require(
        array_job_id.isdigit() and array_task_id is not None and array_task_id.isdigit(),
        "Slurm array identity is invalid",
    )
    return f"{array_job_id}_{array_task_id}"


def validate_run_context(
    path: Path,
    expected_sha256: str,
    *,
    arm: str,
    engineering_test_mode: bool,
    slurm_engineering_test_mode: bool = False,
) -> dict[str, Any]:
    require(
        not (engineering_test_mode and slurm_engineering_test_mode),
        "local and Slurm engineering modes are mutually exclusive",
    )
    require(_is_hash(expected_sha256), "expected run-context SHA-256 is malformed")
    require(sha256(path) == expected_sha256, "run-context SHA-256 mismatch")
    context = load_json(path, "run context")
    expected_keys = {
        "schema",
        "protocol_id",
        "purpose",
        "run_id",
        "arm",
        "training_seed",
        "job_id",
        "campaign_manifest_sha256",
        "provenance",
    }
    require(set(context) == expected_keys, "run-context keys drift")
    require(context["schema"] == 1, "run-context schema drift")
    require(context["protocol_id"] == PROTOCOL_ID, "run-context protocol drift")
    require(context["purpose"] == PURPOSE, "run-context purpose drift")
    require(context["arm"] == arm, "run-context arm drift")
    require(
        isinstance(context["training_seed"], int)
        and not isinstance(context["training_seed"], bool),
        "training seed must be an integer",
    )
    require(
        isinstance(context["run_id"], str) and context["run_id"],
        "run ID must be nonempty",
    )
    require(
        _is_hash(context["campaign_manifest_sha256"]),
        "campaign manifest SHA-256 is malformed",
    )
    provenance = context["provenance"]
    require(isinstance(provenance, dict), "run-context provenance must be an object")
    expected_provenance = {"base_commit", "base_tree", "overlay_contract_sha256"} | HASH_KEYS
    require(set(provenance) == expected_provenance, "run-context provenance keys drift")
    require(provenance["base_commit"] == BASE_COMMIT, "base commit drift")
    require(provenance["base_tree"] == BASE_TREE, "base tree drift")
    require(
        provenance["overlay_contract_sha256"] == OVERLAY_CONTRACT_SHA256,
        "overlay contract drift",
    )
    for key in HASH_KEYS:
        require(_is_hash(provenance[key]), f"malformed provenance hash: {key}")

    job_id = context["job_id"]
    seed = context["training_seed"]
    if engineering_test_mode:
        require(job_id == "local-test", "engineering test context must use local-test job ID")
        require(
            context["run_id"].startswith("engineering-"),
            "engineering test run ID must be visibly labeled",
        )
        require("SLURM_JOB_ID" not in os.environ, "engineering test mode is forbidden under Slurm")
    elif slurm_engineering_test_mode:
        require(seed in (101, 102, 103, 104, 105), "Slurm engineering seed is outside the frozen set")
        require(
            isinstance(job_id, str) and SLURM_JOB.fullmatch(job_id) is not None,
            "Slurm engineering job ID is invalid",
        )
        require(active_slurm_job_id() == job_id, "Slurm engineering job ID/context mismatch")
        require(
            context["run_id"] == f"engineering-slurm-{job_id}-{arm}-s{seed}",
            "Slurm engineering run ID drift",
        )
    else:
        require(seed in (101, 102, 103, 104, 105), "training seed is outside the frozen set")
        require(
            context["run_id"] == f"{PROTOCOL_ID}-s{seed}-{arm}",
            "production run ID drift",
        )
        require(
            isinstance(job_id, str) and SLURM_JOB.fullmatch(job_id) is not None,
            "production job ID is invalid",
        )
        require(active_slurm_job_id() == job_id, "Slurm job ID/context mismatch")
    return context


def validate_source(
    source: Path,
    context: Mapping[str, Any],
    *,
    git_executable: Path | None = None,
    require_pinned_git: bool = False,
) -> dict[str, Any]:
    require(source.is_dir() and not source.is_symlink(), "unsafe patched source directory")
    if git_executable is None:
        require(not require_pinned_git, "pinned environment Git is required")
        discovered_git = shutil.which("git")
        require(discovered_git is not None, "Git executable is unavailable")
        git_path = Path(discovered_git).resolve()
    else:
        require(git_executable.is_absolute(), "Git executable path must be absolute")
        require(git_executable.is_file(), "Git executable is missing")
        git_path = git_executable.resolve()
        if require_pinned_git:
            require(
                git_path.parent == Path(sys.executable).resolve().parent,
                "Git executable is outside the active pinned environment",
            )
    require(git_path.is_file() and os.access(git_path, os.X_OK), "Git executable is not usable")
    try:
        git_version = subprocess.run(
            [str(git_path), "--version"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DriverError("could not execute pinned Git") from exc
    if require_pinned_git:
        require(git_version == "git version 2.45.2", "pinned Git version drift")
    overlay_receipt_path = source / ".frontierrl_overlay.json"
    overlay_receipt = load_json(overlay_receipt_path, "applied overlay receipt")
    provenance = context["provenance"]
    applied_sha = sha256(overlay_receipt_path)
    require(
        applied_sha == provenance["applied_overlay_manifest_sha256"],
        "applied overlay receipt drift",
    )
    require(overlay_receipt.get("base_commit") == BASE_COMMIT, "overlay base commit drift")
    require(overlay_receipt.get("overlay") == OVERLAY_VERSION, "overlay version drift")
    require(
        overlay_receipt.get("overlay_contract_sha256") == OVERLAY_CONTRACT_SHA256,
        "applied overlay contract drift",
    )
    overlay_files = overlay_receipt.get("overlay_files")
    overlay_hashes = overlay_receipt.get("overlay_file_sha256")
    require(
        isinstance(overlay_files, list)
        and isinstance(overlay_hashes, dict)
        and set(overlay_files) == set(overlay_hashes),
        "applied overlay file closure drift",
    )
    for relative in overlay_files:
        require(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            "unsafe overlay-relative path",
        )
        require(sha256(source / relative) == overlay_hashes[relative], f"overlay file drift: {relative}")

    try:
        commit = subprocess.run(
            [str(git_path), "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        tree = subprocess.run(
            [str(git_path), "-C", str(source), "rev-parse", "HEAD^{tree}"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        status_output = subprocess.run(
            [
                str(git_path),
                "-C",
                str(source),
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DriverError("could not validate pinned Git source") from exc
    require(commit == BASE_COMMIT and tree == BASE_TREE, "pinned Git identity drift")
    status: dict[str, str] = {}
    for raw_record in status_output.split(b"\0"):
        if not raw_record:
            continue
        require(len(raw_record) >= 4 and raw_record[2:3] == b" ", "unsafe Git status record")
        code = raw_record[:2].decode("ascii", errors="strict")
        relative = os.fsdecode(raw_record[3:])
        require(relative not in status, f"duplicate Git status path: {relative}")
        # Rename/copy records contain a second NUL path.  Neither is permitted
        # by this overlay, so accepting only these two exact status codes also
        # rejects ambiguous multi-path records.
        require(code in {" M", "??"}, f"inadmissible Git status {code!r}: {relative}")
        status[relative] = code
    expected_dirty = set(overlay_files) | {".frontierrl_overlay.json"}
    require(set(status) == expected_dirty, "patched source worktree closure drift")
    require(status[".frontierrl_overlay.json"] == "??", "overlay marker status drift")
    for relative in overlay_files:
        try:
            tracked = subprocess.run(
                [str(git_path), "-C", str(source), "cat-file", "-e", f"HEAD:{relative}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
        except OSError as exc:
            raise DriverError("could not validate overlay tracking state") from exc
        expected_status = " M" if tracked else "??"
        require(status[relative] == expected_status, f"overlay Git status drift: {relative}")
    return {
        "base_commit": commit,
        "base_tree": tree,
        "applied_overlay_manifest_sha256": applied_sha,
        "overlay_file_count": len(overlay_files),
        "worktree_status": dict(sorted(status.items())),
        "git_executable": str(git_path),
        "git_executable_sha256": sha256(git_path),
        "git_version": git_version,
    }


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = load_json(path, "development protocol")
    require(protocol.get("schema") == 1, "protocol schema drift")
    require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol identity drift")
    require(protocol.get("purpose") == PURPOSE, "protocol purpose drift")
    require(protocol.get("provenance", {}).get("base_commit") == BASE_COMMIT, "protocol commit drift")
    require(protocol.get("provenance", {}).get("base_tree") == BASE_TREE, "protocol tree drift")
    require(
        protocol.get("provenance", {}).get("overlay_contract_sha256")
        == OVERLAY_CONTRACT_SHA256,
        "protocol overlay drift",
    )
    require(set(protocol.get("arms", {})) == set(ARMS), "protocol arm drift")
    return protocol, sha256(path)


def validate_campaign_binding(
    path: Path | None,
    expected_sha256: str | None,
    *,
    context: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    engineering_test_mode: bool,
    slurm_engineering_test_mode: bool,
) -> dict[str, Any] | None:
    """Bind runtime inputs to the frozen campaign before importing minimax.

    A campaign is optional only for the bounded local fixture lane.  Both the
    Slurm engineering lane and production must validate the exact campaign,
    every executable analysis component, the complete provenance projection,
    and the submission selected by this run context before any endpoint code
    can execute.
    """

    require(
        not (engineering_test_mode and slurm_engineering_test_mode),
        "local and Slurm engineering modes are mutually exclusive",
    )
    supplied = path is not None or expected_sha256 is not None
    require(
        (path is None) == (expected_sha256 is None),
        "campaign manifest and expected digest must be supplied together",
    )
    if not supplied:
        require(
            engineering_test_mode,
            "campaign manifest binding is required under Slurm/production",
        )
        return None

    assert path is not None and expected_sha256 is not None
    require(_is_hash(expected_sha256), "expected campaign SHA-256 is malformed")
    require(path.is_absolute(), "campaign manifest path must be absolute")
    campaign_sha = sha256(path)
    campaign_path = path.resolve()
    require(campaign_sha == expected_sha256, "campaign manifest SHA-256 mismatch")
    require(
        campaign_sha == context["campaign_manifest_sha256"],
        "run context binds another campaign manifest",
    )
    campaign = load_json(campaign_path, "campaign manifest")
    require(set(campaign) == CAMPAIGN_KEYS, "campaign manifest keys drift")
    require(campaign["schema"] == 1, "campaign schema drift")
    require(campaign["protocol_id"] == PROTOCOL_ID, "campaign protocol identity drift")
    require(campaign["purpose"] == PURPOSE, "campaign purpose drift")
    require(
        isinstance(campaign["created_utc"], str)
        and UTC_TIMESTAMP.fullmatch(campaign["created_utc"]) is not None,
        "campaign created_utc must be second-resolution UTC",
    )
    require(
        campaign["frozen_before_endpoint_access"] is True,
        "campaign was not frozen before endpoint access",
    )
    require(campaign["protocol_sha256"] == protocol_sha256, "campaign protocol hash drift")

    benchmark_root = Path(__file__).resolve().parents[1]
    training_driver_path = Path(__file__).resolve()
    evaluation_driver_path = training_driver_path.parent / "evaluate_matched_terminal_v4.py"
    assembler_path = training_driver_path.parent / "assemble_matched_run_v4.py"
    analyzer_path = benchmark_root / "analysis" / "preregistered_dev_analysis.py"
    require(
        campaign["analyzer_sha256"] == sha256(analyzer_path),
        "campaign analyzer hash drift",
    )

    provenance = campaign["provenance"]
    require(isinstance(provenance, dict), "campaign provenance must be an object")
    require(set(provenance) == CAMPAIGN_PROVENANCE_KEYS, "campaign provenance keys drift")
    require(provenance["base_commit"] == BASE_COMMIT, "campaign base commit drift")
    require(provenance["base_tree"] == BASE_TREE, "campaign base tree drift")
    require(
        provenance["overlay_contract_sha256"] == OVERLAY_CONTRACT_SHA256,
        "campaign overlay contract drift",
    )
    for key, value in provenance.items():
        if key not in {"base_commit", "base_tree"}:
            require(_is_hash(value), f"malformed campaign provenance hash: {key}")
    require(
        provenance["training_driver_sha256"] == sha256(training_driver_path),
        "campaign training driver hash drift",
    )
    require(
        provenance["evaluation_driver_sha256"] == sha256(evaluation_driver_path),
        "campaign evaluation driver hash drift",
    )
    require(
        provenance["assembler_driver_sha256"] == sha256(assembler_path),
        "campaign assembler driver hash drift",
    )
    expected_context_provenance = {
        key: provenance[key] for key in ({"base_commit", "base_tree", "overlay_contract_sha256"} | HASH_KEYS)
    }
    require(
        context["provenance"] == expected_context_provenance,
        "run-context/campaign provenance drift",
    )
    protocol_provenance = protocol.get("provenance")
    require(isinstance(protocol_provenance, Mapping), "protocol provenance is missing")
    for key in ("base_commit", "base_tree", "overlay_contract_sha256"):
        require(
            provenance[key] == protocol_provenance.get(key),
            f"campaign/protocol provenance drift: {key}",
        )

    hardware = campaign["hardware"]
    require(
        isinstance(hardware, dict) and set(hardware) == CAMPAIGN_HARDWARE_KEYS,
        "campaign hardware keys drift",
    )
    for key in ("partition", "gpu_model", "gpu_profile"):
        require(isinstance(hardware[key], str) and hardware[key], f"campaign hardware {key} drift")
    require(
        hardware["gpu_count"] == 1 and hardware["n_devices"] == 1,
        "campaign must bind one GPU and one JAX device",
    )

    submissions = campaign["submissions"]
    require(isinstance(submissions, list), "campaign submissions must be a list")
    if engineering_test_mode or slurm_engineering_test_mode:
        require(len(submissions) == 1, "engineering campaign must contain one submission")
        expected_cells = [(context["training_seed"], context["arm"])]
    else:
        seeds = protocol.get("training_seeds")
        require(seeds == [101, 102, 103, 104, 105], "protocol training-seed set drift")
        expected_cells = [(seed, arm) for seed in seeds for arm in ARMS]
        require(
            len(submissions) == len(expected_cells),
            "production campaign must contain ten submissions",
        )

    seen_jobs: set[str] = set()
    matched_context = 0
    for submission, (seed, arm) in zip(submissions, expected_cells):
        require(
            isinstance(submission, dict) and set(submission) == CAMPAIGN_SUBMISSION_KEYS,
            "campaign submission keys drift",
        )
        require(
            submission["training_seed"] == seed and submission["arm"] == arm,
            "campaign submission order/cell drift",
        )
        require(
            submission["evaluation_seed"] == 100000 + seed,
            "campaign submission evaluation seed drift",
        )
        expected_run_id = (
            context["run_id"]
            if engineering_test_mode or slurm_engineering_test_mode
            else f"{PROTOCOL_ID}-s{seed}-{arm}"
        )
        require(submission["run_id"] == expected_run_id, "campaign submission run ID drift")
        require(submission["attempt"] == 1, "campaign retries are not permitted")
        job_id = submission["job_id"]
        if engineering_test_mode:
            require(job_id == "local-test", "local campaign job ID drift")
        else:
            require(
                isinstance(job_id, str) and SLURM_JOB.fullmatch(job_id) is not None,
                "campaign submission job ID drift",
            )
        require(job_id not in seen_jobs, "duplicate campaign submission job ID")
        seen_jobs.add(job_id)
        if all(submission[key] == context[key] for key in ("run_id", "arm", "training_seed", "job_id")):
            matched_context += 1
    require(matched_context == 1, "run context does not select exactly one campaign submission")
    return campaign


def load_authored_config(
    path: Path, arm: str, protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    expected = protocol["arms"][arm]
    digest = sha256(path)
    require(digest == expected["config_sha256"], "authored config SHA-256 drift")
    document = load_json(path, "authored config")
    require(set(document) == {"args"} and isinstance(document["args"], dict), "config shape drift")
    for key, values in document["args"].items():
        require(
            isinstance(key, str) and isinstance(values, list) and len(values) == 1,
            f"config field {key!r} is not a singleton",
        )
    require(document["args"]["ued_score"][0] == expected["ued_score"], "arm score drift")
    return document, digest


def parse_authored_args(document: Mapping[str, Any], parser: Any) -> Any:
    argv = ["run-matched-terminal"] + [
        f"--{key}={values[0]}" for key, values in document["args"].items()
    ]
    previous = sys.argv
    try:
        sys.argv = argv
        return parser.parse_args()
    finally:
        sys.argv = previous


def _as_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _as_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


_ENGINEERING_OVERRIDE_SPEC: dict[str, tuple[type, float, float]] = {
    "n_total_updates": (int, 1, 2),
    "test_interval": (int, 0, 10),
    "log_interval": (int, 1, 10),
    "train_runner_args.buffer_size": (int, 8, 64),
    "train_runner_args.replay_prob": (float, 0.5, 1.0),
    "train_runner_args.min_fill_ratio": (float, 0.0625, 1.0),
    "train_runner_args.n_rollout_steps": (int, 1, 16),
    "train_runner_args.n_unroll_rollout": (int, 1, 10),
    "env_args.max_episode_steps": (int, 1, 32),
    "student_rl_args.n_unroll_update": (int, 1, 5),
    "student_rl_args.n_epochs": (int, 1, 5),
    "student_model_args.hidden_dim": (int, 4, 32),
    "student_model_args.recurrent_hidden_dim": (int, 4, 64),
    "student_model_args.n_conv_filters": (int, 1, 16),
    "driver.max_outer_cycles": (int, 2, 1000),
}


def _strict_number(value: Any, wanted: type, label: str) -> int | float:
    if wanted is int:
        require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
        return value
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def parse_engineering_overrides(entries: Sequence[str]) -> dict[str, int | float]:
    overrides: dict[str, int | float] = {}
    for entry in entries:
        require("=" in entry, f"engineering override lacks '=': {entry!r}")
        key, encoded = entry.split("=", 1)
        require(key in _ENGINEERING_OVERRIDE_SPEC, f"engineering override is not allowed: {key}")
        require(key not in overrides, f"duplicate engineering override: {key}")
        try:
            raw_value = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise DriverError(f"engineering override is not JSON: {key}") from exc
        wanted, lower, upper = _ENGINEERING_OVERRIDE_SPEC[key]
        value = _strict_number(raw_value, wanted, key)
        require(lower <= value <= upper, f"engineering override out of bounds: {key}")
        overrides[key] = value
    return dict(sorted(overrides.items()))


def _get_nested(root: Any, dotted: str) -> Any:
    value = root
    for part in dotted.split("."):
        value = value[part] if isinstance(value, Mapping) else getattr(value, part)
    return value


def _set_nested(root: Any, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    target = root
    for part in parts[:-1]:
        target = target[part] if isinstance(target, Mapping) else getattr(target, part)
    if isinstance(target, MutableMapping):
        target[parts[-1]] = value
    else:
        setattr(target, parts[-1], value)


def apply_engineering_overrides(args: Any, overrides: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    max_outer_cycles = 1000
    for key, value in overrides.items():
        if key == "driver.max_outer_cycles":
            records.append({"field": key, "authored": None, "resolved": value})
            max_outer_cycles = int(value)
            continue
        before = _get_nested(args, key)
        _set_nested(args, key, value)
        records.append({"field": key, "authored": before, "resolved": value})
    return records, max_outer_cycles


def validate_resolved_args(args: Any, arm: str, *, engineering: bool) -> None:
    runner = args.train_runner_args
    require(args.train_runner == "plr" and args.agent_rl_algo == "ppo", "runner/algo drift")
    require(args.n_devices == 1 and runner.n_students == 1, "one-device/student contract drift")
    require(runner.n_parallel == 4 and runner.n_eval == 8, "4x8 group layout drift")
    require(runner.use_robust_plr is True and runner.force_unique is True, "robust PLR drift")
    require(runner.use_parallel_eval is False, "parallel-eval drift")
    require(runner.use_score_ranks is True, "score-rank mode drift")
    require(runner.tie_aware_score_ranks is True, "tie-aware score ranks disabled")
    require(args.from_last_checkpoint is False, "resume is forbidden")
    require(args.archive_init_checkpoint is False, "initial checkpoint archive is forbidden")
    require(args.archive_interval == 0, "archive path is forbidden")
    require(runner.frontier_overlay_version == OVERLAY_VERSION, "overlay version drift")
    require(
        runner.frontier_overlay_contract_sha256 == OVERLAY_CONTRACT_SHA256,
        "resolved overlay contract drift",
    )
    if arm == "frontier":
        require(runner.ued_score == "coefficient_activity", "Frontier score drift")
        require(runner.frontier_n_rollouts == runner.n_eval == 8, "Frontier N/n_eval drift")
        require(runner.frontier_require_n_eval_match is True, "Frontier strict grouping disabled")
        require(runner.frontier_posterior_mode == "expected_activity", "posterior mode drift")
    else:
        require(runner.ued_score == "max_mc", "MaxMC score drift")
    if not engineering:
        require(args.n_total_updates == 30000, "production PPO-update target drift")
        require(runner.n_rollout_steps == 256, "production rollout horizon drift")
        require(runner.buffer_size == 500, "production buffer drift")
        require(runner.replay_prob == 0.5 and runner.min_fill_ratio == 0.5, "PLR schedule drift")
        require(args.student_rl_args.n_epochs == 5, "production PPO epoch drift")
        require(args.student_rl_args.n_minibatches == 1, "production minibatch drift")


def make_experiment(args: Any, ExperimentRunner: Any) -> Any:
    resolved = copy.deepcopy(args)
    return ExperimentRunner(
        train_runner=resolved.train_runner,
        env_name=resolved.env_name,
        agent_rl_algo=resolved.agent_rl_algo,
        student_model_name=resolved.student_model_name,
        teacher_model_name=resolved.teacher_model_name,
        train_runner_kwargs=resolved.train_runner_args,
        env_kwargs=resolved.env_args,
        ued_env_kwargs=resolved.ued_env_args,
        student_rl_kwargs=resolved.student_rl_args,
        teacher_rl_kwargs=resolved.teacher_rl_args,
        student_model_kwargs=resolved.student_model_args,
        teacher_model_kwargs=resolved.teacher_model_args,
        eval_kwargs=resolved.eval_args,
        eval_env_kwargs=resolved.eval_env_args,
        n_devices=resolved.n_devices,
    )


def scalar_int(value: Any, label: str) -> int:
    array = np.asarray(value)
    require(array.size == 1, f"{label} is not scalar")
    result = array.reshape(-1)[0]
    require(np.issubdtype(array.dtype, np.integer), f"{label} is not integral")
    return int(result)


def block(tree: Any, jax: Any) -> None:
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


def state_summary(state: Sequence[Any], arm: str) -> dict[str, Any]:
    train_state = state[1]
    buffer = train_state.plr_buffer
    filled = np.asarray(buffer.filled, dtype=bool)
    scores = np.asarray(buffer.scores)
    nonfinite_filled = int(np.logical_and(filled, ~np.isfinite(scores)).sum())
    replay_draws = scalar_int(buffer.replay_group_draw_count, "replay_group_draw_count")
    replay_distinct = scalar_int(
        buffer.replay_distinct_group_count, "replay_distinct_group_count"
    )
    replay_duplicates = scalar_int(
        buffer.replay_duplicate_group_count, "replay_duplicate_group_count"
    )
    last_draws = scalar_int(buffer.last_replay_group_count, "last_replay_group_count")
    last_distinct = scalar_int(
        buffer.last_replay_distinct_group_count, "last_replay_distinct_group_count"
    )
    last_duplicates = scalar_int(
        buffer.last_replay_duplicate_group_count, "last_replay_duplicate_group_count"
    )
    result: dict[str, Any] = {
        "n_iters": scalar_int(train_state.n_iters, "n_iters"),
        "n_updates": scalar_int(train_state.n_updates, "n_updates"),
        "n_grad_updates": scalar_int(train_state.n_grad_updates, "n_grad_updates"),
        "replay_integrity": {
            "tie_aware_score_ranks": bool(buffer.tie_aware_score_ranks),
            "nonfinite_filled_score_count": nonfinite_filled,
            "nonfinite_score_rejection_count": scalar_int(
                buffer.nonfinite_score_rejection_count,
                "nonfinite_score_rejection_count",
            ),
            "replay_group_draw_count": replay_draws,
            "replay_distinct_group_count": replay_distinct,
            "replay_duplicate_group_count": replay_duplicates,
            "last_replay_group_count": last_draws,
            "last_replay_distinct_group_count": last_distinct,
            "last_replay_duplicate_group_count": last_duplicates,
            "force_unique_resamples_replay": False,
            "sample_identity": "replay buffer slot index",
        },
    }
    if arm == "frontier":
        result["frontier_integrity"] = {
            "n_rollouts": int(buffer.frontier_n_rollouts),
            "n_eval": int(buffer.frontier_n_eval),
            "group_size_match": bool(buffer.frontier_n_rollouts == buffer.frontier_n_eval),
            "incomplete_group_count": scalar_int(
                buffer.incomplete_group_count, "incomplete_group_count"
            ),
            "duplicate_new_group_count": scalar_int(
                buffer.duplicate_new_group_count, "duplicate_new_group_count"
            ),
            "buffer_total_trials": int(np.asarray(buffer.trial_counts).sum()),
            "buffer_total_successes": int(np.asarray(buffer.success_counts).sum()),
        }
    else:
        result["frontier_integrity"] = None
    return result


def validate_replay_integrity(summary: Mapping[str, Any]) -> None:
    integrity = summary["replay_integrity"]
    require(integrity["tie_aware_score_ranks"] is True, "tie-aware rank state drift")
    require(integrity["nonfinite_filled_score_count"] == 0, "nonfinite filled PLR score")
    require(integrity["nonfinite_score_rejection_count"] == 0, "nonfinite score rejected")
    require(
        integrity["replay_group_draw_count"]
        == integrity["replay_distinct_group_count"]
        + integrity["replay_duplicate_group_count"],
        "cumulative replay draw accounting drift",
    )
    require(
        integrity["last_replay_group_count"]
        == integrity["last_replay_distinct_group_count"]
        + integrity["last_replay_duplicate_group_count"],
        "last replay draw accounting drift",
    )
    require(
        integrity["replay_distinct_group_count"]
        <= integrity["replay_group_draw_count"],
        "cumulative distinct replay count exceeds draws",
    )
    require(
        integrity["last_replay_distinct_group_count"]
        <= integrity["last_replay_group_count"],
        "last distinct replay count exceeds draws",
    )
    require(integrity["force_unique_resamples_replay"] is False, "sampling semantics drift")


def _json_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in stats.items():
        if value is None:
            result[key] = None
            continue
        array = np.asarray(value)
        if array.size != 1:
            continue
        scalar = array.reshape(-1)[0]
        if np.issubdtype(array.dtype, np.bool_):
            result[key] = bool(scalar)
        elif np.issubdtype(array.dtype, np.integer):
            result[key] = int(scalar)
        elif np.issubdtype(array.dtype, np.floating):
            number = float(scalar)
            result[key] = number if math.isfinite(number) else None
    return result


def _materialize_level_identity_source(
    levels: Any, jax: Any
) -> tuple[str, list[np.ndarray]]:
    leaves, structure = jax.tree_util.tree_flatten(levels)
    # Materialize each full replay-buffer leaf exactly once.  Hashing up to 500
    # slots must not trigger one device-to-host transfer per leaf per slot.
    return str(structure), [np.asarray(leaf) for leaf in leaves]


def _level_identity(
    structure: str,
    host_leaves: Sequence[np.ndarray],
    student: int,
    slot: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(structure.encode("utf-8"))
    for index, leaf in enumerate(host_leaves):
        array = leaf[student, slot]
        contiguous = np.ascontiguousarray(array)
        descriptor = {
            "index": index,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        }
        digest.update(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def coefficient_scores(
    successes: int,
    trials: int,
    n_rollouts: int,
    prior_alpha: float,
    prior_beta: float,
) -> tuple[float, float]:
    require(0 <= successes <= trials, "invalid posterior sufficient statistics")
    a = successes + prior_alpha
    b = trials - successes + prior_beta
    failure_moment = 1.0
    for offset in range(n_rollouts):
        failure_moment *= (b + offset) / (a + b + offset)
    analytic = min(1.0, max(0.0, 1.0 - failure_moment - a / (a + b)))
    p = a / (a + b)
    plugin = min(1.0, max(0.0, 1.0 - (1.0 - p) ** n_rollouts - p))
    return analytic, plugin


def replay_distribution(
    scores: np.ndarray,
    ages: np.ndarray,
    filled: np.ndarray,
    *,
    temperature: float,
    staleness_coef: float,
    use_score_ranks: bool,
    tie_aware_score_ranks: bool,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    scores = np.asarray(scores, dtype=np.float64).copy()
    ages = np.asarray(ages, dtype=np.float64)
    filled = np.asarray(filled, dtype=bool)
    require(scores.ndim == ages.ndim == filled.ndim == 1, "replay arrays must be vectors")
    require(scores.shape == ages.shape == filled.shape, "replay array shape drift")
    require(filled.any(), "cannot snapshot an empty PLR buffer")
    require(temperature > 0.0 and 0.0 <= staleness_coef <= 1.0, "invalid replay parameters")
    require(not tie_aware_score_ranks or use_score_ranks, "invalid tie-aware rank activation")
    require(np.isfinite(scores[filled]).all(), "filled PLR score is non-finite")
    tie_block_sizes: list[int] = []
    if use_score_ranks:
        order = np.argsort(-np.where(filled, scores, -np.inf), kind="stable")
        sorted_filled = filled[order]
        filled_order = order[sorted_filled]
        rank_mass = np.power(
            1.0/(1.0 + np.arange(filled_order.size, dtype=np.float64)),
            1.0/temperature,
        )
        score_mass = np.zeros_like(scores)
        cursor = 0
        while cursor < filled_order.size:
            end = cursor + 1
            score = scores[filled_order[cursor]]
            while end < filled_order.size and scores[filled_order[end]] == score:
                end += 1
            size = end - cursor
            tie_block_sizes.append(size)
            if tie_aware_score_ranks:
                score_mass[filled_order[cursor:end]] = float(rank_mass[cursor:end].mean())
            else:
                score_mass[filled_order[cursor:end]] = rank_mass[cursor:end]
            cursor = end
    else:
        score_mass = np.power(scores * filled, 1.0/temperature)
        tie_block_sizes = [1] * int(filled.sum())
    score_total = float(
        np.sort(score_mass).sum()
        if tie_aware_score_ranks else score_mass.sum()
    )
    require(math.isfinite(score_total) and score_total > 0.0, "invalid score distribution")
    score_dist = score_mass / score_total
    stale_mass = ages * filled
    stale_total = float(
        np.sort(stale_mass).sum()
        if tie_aware_score_ranks else stale_mass.sum()
    )
    stale_dist = stale_mass / stale_total if stale_total > 0.0 else score_dist
    replay = (1.0 - staleness_coef) * score_dist + staleness_coef * stale_dist
    require(np.isfinite(replay).all() and (replay >= 0.0).all(), "invalid replay distribution")
    validate_probability_mass(
        replay,
        label="reconstructed replay distribution",
        tolerance=FLOAT64_RECONSTRUCTION_TOLERANCE,
    )
    return replay, score_dist, tie_block_sizes


def validate_probability_mass(values: Any, *, label: str, tolerance: float) -> float:
    probabilities = np.asarray(values, dtype=np.float64)
    require(probabilities.ndim == 1, f"{label} must be a vector")
    require(probabilities.size > 0, f"{label} is empty")
    require(np.isfinite(probabilities).all(), f"{label} is non-finite")
    require((probabilities >= 0.0).all(), f"{label} contains negative mass")
    # This is receipt validation, not source behavior.  Canonical value order
    # makes the recorded total independent of replay-buffer slot order.
    total = float(np.sum(np.sort(probabilities), dtype=np.float64))
    require(
        math.isfinite(tolerance) and tolerance > 0.0,
        f"{label} normalization tolerance is invalid",
    )
    require(
        abs(total - 1.0) <= tolerance,
        f"{label} is not normalized",
    )
    return total


def plr_replay_snapshot(
    state: Sequence[Any],
    args: Any,
    context: Mapping[str, Any],
    checkpoint_sha256: str,
    jax: Any,
    plr_manager: Any,
    arm: str,
) -> dict[str, Any]:
    buffer = state[1].plr_buffer
    filled = np.asarray(buffer.filled)
    scores = np.asarray(buffer.scores)
    ages = np.asarray(buffer.ages)
    successes = np.asarray(buffer.success_counts)
    trials = np.asarray(buffer.trial_counts)
    require(filled.ndim == 2 and filled.shape[0] == 1, "snapshot requires one student")
    replay, score_dist, tie_block_sizes = replay_distribution(
        scores[0],
        ages[0],
        filled[0],
        temperature=float(args.train_runner_args.temp),
        staleness_coef=float(args.train_runner_args.staleness_coef),
        use_score_ranks=bool(args.train_runner_args.use_score_ranks),
        tie_aware_score_ranks=bool(args.train_runner_args.tie_aware_score_ranks),
    )
    implementation_replay = np.asarray(
        plr_manager._get_replay_dist(buffer.scores[0], buffer.ages[0], buffer.filled[0]),
        dtype=np.float64,
    )
    require(implementation_replay.shape == replay.shape, "implementation replay shape drift")
    require(np.isfinite(implementation_replay).all(), "implementation replay is non-finite")
    require(
        np.max(np.abs(implementation_replay[~filled[0]]), initial=0.0)
        <= FLOAT32_DIAGNOSTIC_TOLERANCE,
        "implementation assigns replay mass to unfilled slots",
    )
    require(
        abs(float(np.sort(implementation_replay).sum()) - 1.0)
        <= FLOAT32_DIAGNOSTIC_TOLERANCE,
        "implementation replay distribution is not normalized",
    )
    replay_max_abs_error = float(np.max(np.abs(implementation_replay - replay)))
    require(
        replay_max_abs_error <= FLOAT32_DIAGNOSTIC_TOLERANCE,
        "reconstructed replay distribution disagrees with pinned implementation",
    )
    slots: list[dict[str, Any]] = []
    stored_score_max_abs_error = 0.0
    level_structure, host_level_leaves = _materialize_level_identity_source(
        buffer.levels, jax
    )
    for slot in np.flatnonzero(filled[0]):
        entry: dict[str, Any] = {
            "student_index": 0,
            "slot_index": int(slot),
            "level_sha256": _level_identity(
                level_structure, host_level_leaves, 0, int(slot)
            ),
            "stored_score": float(scores[0, slot]),
            "age": int(ages[0, slot]),
            "normalized_score_probability": float(score_dist[slot]),
            "normalized_replay_probability": float(replay[slot]),
        }
        if arm == "frontier":
            success_count = int(successes[0, slot])
            trial_count = int(trials[0, slot])
            analytic, plugin = coefficient_scores(
                success_count,
                trial_count,
                int(args.train_runner_args.frontier_n_rollouts),
                float(args.train_runner_args.frontier_prior_alpha),
                float(args.train_runner_args.frontier_prior_beta),
            )
            stored_score_error = abs(float(scores[0, slot]) - analytic)
            stored_score_max_abs_error = max(
                stored_score_max_abs_error, stored_score_error
            )
            require(
                stored_score_error <= FLOAT32_DIAGNOSTIC_TOLERANCE,
                f"stored Frontier score disagrees with analytic score at slot {slot}",
            )
            require(
                plugin + FLOAT32_DIAGNOSTIC_TOLERANCE >= analytic,
                "Jensen ordering drift",
            )
            entry.update(
                success_count=success_count,
                trial_count=trial_count,
                analytic_expected_activity_score=analytic,
                mean_plugin_score=plugin,
                jensen_gap=plugin - analytic,
                stored_score_abs_error=stored_score_error,
            )
        slots.append(entry)
    filled_count = scalar_int(buffer.filled_count, "buffer filled_count")
    require(len(slots) == filled_count, "snapshot filled-count drift")
    filled_slot_mass_sum = validate_probability_mass(
        np.asarray(
            [slot["normalized_replay_probability"] for slot in slots], dtype=np.float64
        ),
        label="filled-slot replay mass",
        tolerance=FLOAT64_RECONSTRUCTION_TOLERANCE,
    )
    score_effective_support = float(
        1.0/np.square(np.sort(score_dist)).sum())
    replay_effective_support = float(
        1.0/np.square(np.sort(replay)).sum())
    require(
        math.isfinite(score_effective_support)
        and 1.0 <= score_effective_support <= filled_count + FLOAT32_DIAGNOSTIC_TOLERANCE,
        "invalid score-distribution effective support",
    )
    require(
        math.isfinite(replay_effective_support)
        and 1.0 <= replay_effective_support <= filled_count + FLOAT32_DIAGNOSTIC_TOLERANCE,
        "invalid replay-distribution effective support",
    )
    replay_integrity = state_summary(state, arm)["replay_integrity"]
    validate_replay_integrity({"replay_integrity": replay_integrity})
    return {
        "schema": 1,
        "status": "completed",
        "kind": "tie_aware_plr_buffer_safe_snapshot",
        "protocol_id": PROTOCOL_ID,
        "purpose": PURPOSE,
        "paper_evidence": False,
        "run_id": context["run_id"],
        "arm": arm,
        "training_seed": context["training_seed"],
        "checkpoint_sha256": checkpoint_sha256,
        "buffer_size": int(args.train_runner_args.buffer_size),
        "filled_count": filled_count,
        "n_rollouts": (
            int(args.train_runner_args.frontier_n_rollouts)
            if arm == "frontier" else None
        ),
        "n_eval": int(args.train_runner_args.n_eval),
        "prior_alpha": (
            float(args.train_runner_args.frontier_prior_alpha)
            if arm == "frontier" else None
        ),
        "prior_beta": (
            float(args.train_runner_args.frontier_prior_beta)
            if arm == "frontier" else None
        ),
        "replay_distribution": {
            "use_score_ranks": bool(args.train_runner_args.use_score_ranks),
            "tie_aware_score_ranks": bool(
                args.train_runner_args.tie_aware_score_ranks
            ),
            "score_normalization_order": (
                "canonical_ascending_unnormalized_mass"
                if bool(args.train_runner_args.tie_aware_score_ranks)
                else "source_buffer_slot_order"
            ),
            "distinct_score_stable_equivalence_float32_abs_tolerance": (
                FLOAT32_DISTINCT_SCORE_EQUIVALENCE_TOLERANCE
                if bool(args.train_runner_args.tie_aware_score_ranks)
                else None
            ),
            "tie_equality": "exact filled-score equality; +0 and -0 tie",
            "tie_block_sizes_descending_score_order": tie_block_sizes,
            "tied_block_sizes_descending_score_order": [
                size for size in tie_block_sizes if size > 1
            ],
            "distinct_filled_score_count": len(tie_block_sizes),
            "score_effective_support": score_effective_support,
            "replay_effective_support": replay_effective_support,
            "temperature": float(args.train_runner_args.temp),
            "staleness_coef": float(args.train_runner_args.staleness_coef),
            "normalization_sum": float(np.sort(replay).sum()),
            "filled_slot_float64_normalization_sum": filled_slot_mass_sum,
            "float64_reconstruction_tolerance": FLOAT64_RECONSTRUCTION_TOLERANCE,
            "pinned_implementation_normalization_sum": float(
                np.sort(implementation_replay).sum()),
            "pinned_implementation_max_abs_error": replay_max_abs_error,
            "float32_validation_tolerance": FLOAT32_DIAGNOSTIC_TOLERANCE,
        },
        "sampling_diagnostics": replay_integrity,
        "sampling_semantics": {
            "replacement": "with_replacement",
            "force_unique_effect": "buffer-update deduplication only; no replay resampling",
            "distinct_identity": "replay buffer slot index",
            "last_counts_persist_until_next_actual_replay_batch": True,
        },
        "stored_score_validation": (
            {
                "max_abs_error": stored_score_max_abs_error,
                "float32_validation_tolerance": FLOAT32_DIAGNOSTIC_TOLERANCE,
            }
            if arm == "frontier" else None
        ),
        "level_identity": "sha256(canonical pytree structure, leaf dtype/shape/C-order bytes)",
        "level_identity_materialization": {
            "host_transfers_per_level_leaf": 1,
            "level_leaf_count": len(host_level_leaves),
        },
        "slots": slots,
    }


def _write_safe_meta(
    output_dir: Path,
    args: Any,
    context: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    successful: bool,
) -> None:
    meta_path = output_dir / "meta.json"
    value = {
        "date_start": None,
        "date_end": None,
        "successful": successful,
        "git": {
            "commit": source["base_commit"],
            "tree": source["base_tree"],
            "branch": None,
            "is_dirty": True,
            "path": None,
        },
        "slurm": {"job_id": context["job_id"]},
        # Deliberately do not serialize the process environment.  No runtime
        # secret is needed to reproduce or analyze the experiment.
        "env": {},
        "config": _as_plain(args),
        "xpid": context["run_id"],
    }
    if meta_path.exists():
        temporary = meta_path.with_name(f".{meta_path.name}.rewrite")
        require(not temporary.exists(), "stale metadata rewrite file")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, meta_path)
    else:
        atomic_json(meta_path, value)


def _fresh_checkpoint_summary(checkpoint: Any, arm: str) -> dict[str, Any]:
    require(isinstance(checkpoint, (list, tuple)) and len(checkpoint) >= 2, "checkpoint shape drift")
    train_state = checkpoint[1]
    require(isinstance(train_state, Mapping), "checkpoint train state is not a mapping")
    result = {
        "n_iters": scalar_int(train_state["n_iters"], "checkpoint n_iters"),
        "n_updates": scalar_int(train_state["n_updates"], "checkpoint n_updates"),
        "n_grad_updates": scalar_int(train_state["n_grad_updates"], "checkpoint n_grad_updates"),
    }
    plr = train_state.get("plr_buffer")
    require(plr is not None, "checkpoint PLR buffer missing")

    def plr_field(name: str) -> Any:
        return plr[name] if isinstance(plr, Mapping) else getattr(plr, name)

    filled = np.asarray(plr_field("filled"), dtype=bool)
    scores = np.asarray(plr_field("scores"))
    replay_integrity = {
        "tie_aware_score_ranks": bool(plr_field("tie_aware_score_ranks")),
        "nonfinite_filled_score_count": int(
            np.logical_and(filled, ~np.isfinite(scores)).sum()
        ),
        "nonfinite_score_rejection_count": scalar_int(
            plr_field("nonfinite_score_rejection_count"),
            "checkpoint nonfinite_score_rejection_count",
        ),
        "replay_group_draw_count": scalar_int(
            plr_field("replay_group_draw_count"), "checkpoint replay_group_draw_count"
        ),
        "replay_distinct_group_count": scalar_int(
            plr_field("replay_distinct_group_count"),
            "checkpoint replay_distinct_group_count",
        ),
        "replay_duplicate_group_count": scalar_int(
            plr_field("replay_duplicate_group_count"),
            "checkpoint replay_duplicate_group_count",
        ),
        "last_replay_group_count": scalar_int(
            plr_field("last_replay_group_count"), "checkpoint last_replay_group_count"
        ),
        "last_replay_distinct_group_count": scalar_int(
            plr_field("last_replay_distinct_group_count"),
            "checkpoint last_replay_distinct_group_count",
        ),
        "last_replay_duplicate_group_count": scalar_int(
            plr_field("last_replay_duplicate_group_count"),
            "checkpoint last_replay_duplicate_group_count",
        ),
        "force_unique_resamples_replay": False,
        "sample_identity": "replay buffer slot index",
    }
    validate_replay_integrity({"replay_integrity": replay_integrity})
    result["replay_integrity"] = replay_integrity
    if arm == "frontier":
        trial_counts = plr_field("trial_counts")
        success_counts = plr_field("success_counts")
        result["buffer_total_trials"] = int(np.asarray(trial_counts).sum())
        result["buffer_total_successes"] = int(np.asarray(success_counts).sum())
    return result


def run(cli: argparse.Namespace) -> dict[str, Any]:
    driver_path = Path(__file__).resolve()
    driver_sha = sha256(driver_path)
    require(_is_hash(cli.expected_driver_sha256), "expected driver SHA-256 is malformed")
    require(driver_sha == cli.expected_driver_sha256, "training driver SHA-256 mismatch")

    arm = cli.arm
    require(arm in ARMS, "unknown arm")
    require(
        not (cli.engineering_test_mode and cli.slurm_engineering_test_mode),
        "local and Slurm engineering modes are mutually exclusive",
    )
    engineering_mode = bool(
        cli.engineering_test_mode or cli.slurm_engineering_test_mode
    )
    require(
        engineering_mode,
        "draft v2 driver forbids matched-development/production endpoints",
    )
    context = validate_run_context(
        cli.run_context.resolve(),
        cli.expected_run_context_sha256,
        arm=arm,
        engineering_test_mode=cli.engineering_test_mode,
        slurm_engineering_test_mode=cli.slurm_engineering_test_mode,
    )
    require(
        context["provenance"]["training_driver_sha256"] == driver_sha,
        "run context binds another training driver",
    )
    evaluation_driver_path = driver_path.parent / "evaluate_matched_terminal_v4.py"
    require(
        context["provenance"]["evaluation_driver_sha256"]
        == sha256(evaluation_driver_path),
        "run context binds a drifted evaluation driver",
    )
    protocol, protocol_sha = load_protocol(cli.protocol.resolve())
    validate_campaign_binding(
        cli.campaign_manifest,
        cli.expected_campaign_manifest_sha256,
        context=context,
        protocol=protocol,
        protocol_sha256=protocol_sha,
        engineering_test_mode=cli.engineering_test_mode,
        slurm_engineering_test_mode=cli.slurm_engineering_test_mode,
    )
    config_document, config_sha = load_authored_config(cli.config.resolve(), arm, protocol)
    source_dir = cli.patched_source_dir.resolve()
    source_receipt = validate_source(
        source_dir,
        context,
        git_executable=cli.git_executable,
        require_pinned_git=not cli.engineering_test_mode,
    )

    source_module_dir = source_dir / "src"
    require(source_module_dir.is_dir(), "patched source lacks src directory")
    require("minimax" not in sys.modules, "minimax was imported before source validation")
    sys.path.insert(0, str(source_module_dir))
    import jax  # type: ignore
    import minimax  # type: ignore
    from minimax.arguments import parser as minimax_parser  # type: ignore
    from minimax.runners import ExperimentRunner  # type: ignore
    from minimax.util.checkpoint import load_pkl_object, safe_checkpoint  # type: ignore
    from minimax.util.loggers import Logger  # type: ignore

    minimax_path = Path(minimax.__file__).resolve()
    require(minimax_path.is_relative_to(source_dir), "minimax import escaped patched source")
    expected_backend = "cpu" if cli.engineering_test_mode else "gpu"
    require(jax.default_backend() == expected_backend, f"expected {expected_backend} backend")
    runtime_devices = jax.devices(expected_backend)
    require(len(runtime_devices) == 1, "run requires exactly one visible device")
    device_receipt = [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
        }
        for device in runtime_devices
    ]

    args = parse_authored_args(config_document, minimax_parser)
    args.seed = context["training_seed"]
    args.xpid = context["run_id"]
    args.wandb_args.api_key = None

    override_values = parse_engineering_overrides(cli.engineering_override)
    if engineering_mode:
        require(override_values, "engineering test mode requires explicit bounded overrides")
        override_records, max_outer_cycles = apply_engineering_overrides(args, override_values)
    else:
        require(not override_values, "production run forbids engineering overrides")
        override_records = []
        max_outer_cycles = int(args.n_total_updates) * 10 + 1000
    validate_resolved_args(args, arm, engineering=engineering_mode)
    require(max_outer_cycles >= int(args.n_total_updates), "outer-cycle ceiling is too small")

    output_dir = cli.output_dir
    require(output_dir.is_absolute(), "output directory must be absolute")
    require(not output_dir.exists() and not output_dir.is_symlink(), "output directory already exists")
    require(output_dir.parent.is_dir() and not output_dir.parent.is_symlink(), "unsafe output parent")
    sidecar_dir = cli.sidecar_dir
    require(sidecar_dir.is_absolute(), "sidecar directory must be absolute")
    require(not sidecar_dir.exists() and not sidecar_dir.is_symlink(), "sidecar directory exists")
    require(sidecar_dir.parent.is_dir() and not sidecar_dir.parent.is_symlink(), "unsafe sidecar parent")
    require(sidecar_dir != output_dir, "sidecar and training output must be separate")
    require(not sidecar_dir.is_relative_to(output_dir), "sidecar may not be nested in training output")
    args.log_dir = str(output_dir.parent)
    args.xpid = output_dir.name
    require(args.xpid == context["run_id"], "output basename/run ID mismatch")

    # Logger is used for source-format logs.csv only.  Metadata is written by
    # this driver from a secret-free whitelist.
    logger = Logger(
        log_dir=args.log_dir,
        xpid=args.xpid,
        xp_args=None,
        callback=None,
        from_last_checkpoint=False,
        verbose=False,
    )
    require(Path(logger.paths["xpid_dir"]).resolve() == output_dir, "logger output path drift")
    _write_safe_meta(output_dir, args, context, source_receipt, successful=False)

    experiment = make_experiment(args, ExperimentRunner)
    require(experiment.eval_runner is not None, "periodic evaluation runner is missing")
    periodic_eval_horizons = tuple(
        int(benv.env.max_episode_steps()) for benv in experiment.eval_runner.benvs
    )
    require(
        periodic_eval_horizons
        == (PERIODIC_EVAL_HORIZON,) * PERIODIC_EVAL_ENVIRONMENTS,
        "periodic evaluation horizon drift",
    )
    state = experiment.runner.reset(jax.random.PRNGKey(args.seed))
    block(state, jax)
    initial = state_summary(state, arm)
    require(initial["n_iters"] == initial["n_updates"] == initial["n_grad_updates"] == 0, "nonzero initial counters")
    validate_replay_integrity(initial)

    target_updates = int(args.n_total_updates)
    tick = 0
    train_steps = 0
    periodic_evaluation_calls = 0
    wall_started = time.monotonic()
    last_stats: dict[str, Any] = {}
    summary = initial
    while summary["n_updates"] < target_updates:
        require(tick < max_outer_cycles, "outer-cycle ceiling reached before update target")
        evaluate = args.test_interval > 0 and (tick + 1) % args.test_interval == 0
        step_started = time.monotonic()
        stats, eval_stats, *state = experiment.step(state, evaluate)
        block((stats, eval_stats, state), jax)
        if evaluate:
            stats.update(eval_stats)
            periodic_evaluation_calls += 1
        else:
            stats.update({key: None for key in eval_stats})

        dsteps = (
            experiment.runner.step_batch_size
            * experiment.runner.n_rollout_steps
            * experiment.n_devices
        )
        train_steps += int(dsteps // experiment.runner.n_students)
        elapsed = max(time.monotonic() - step_started, 1e-12)
        stats.update(
            {
                "steps": train_steps,
                "real_steps": train_steps,
                "sps": int(dsteps / elapsed),
                "real_sps": int((dsteps // experiment.runner.n_students) / elapsed),
            }
        )
        tick += 1
        summary = state_summary(state, arm)
        validate_replay_integrity(summary)
        require(summary["n_iters"] == tick, "upstream n_iters/outer-cycle drift")
        require(summary["n_updates"] <= target_updates, "student PPO-update target overshot")
        if arm == "frontier":
            integrity = summary["frontier_integrity"]
            require(integrity["n_rollouts"] == integrity["n_eval"] == 8, "Frontier grouping drift")
            require(integrity["group_size_match"] is True, "Frontier group mismatch")
            require(integrity["incomplete_group_count"] == 0, "incomplete Frontier group observed")
            require(integrity["duplicate_new_group_count"] == 0, "duplicate new Frontier group observed")
        if args.log_interval > 0 and tick % args.log_interval == 0:
            logger.log(stats, tick, ignore_val=-np.inf)
        last_stats = _json_stats(stats)

    final = summary
    require(final["n_updates"] == target_updates, "terminal student PPO-update drift")
    require(final["n_grad_updates"] == target_updates, "terminal upstream gradient-counter drift")
    require(final["n_iters"] == tick, "terminal n_iters drift")
    expected_periodic_evaluation_calls = (
        tick // int(args.test_interval) if int(args.test_interval) > 0 else 0
    )
    require(
        periodic_evaluation_calls == expected_periodic_evaluation_calls,
        "periodic evaluation call accounting drift",
    )
    periodic_evaluation_transitions = (
        periodic_evaluation_calls
        * PERIODIC_EVAL_EPISODES_PER_ENVIRONMENT
        * sum(periodic_eval_horizons)
    )
    transitions_per_cycle = int(
        args.train_runner_args.n_parallel
        * args.train_runner_args.n_eval
        * args.train_runner_args.n_rollout_steps
    )
    transitions = tick * transitions_per_cycle
    require(train_steps == transitions, "training transition accounting drift")
    optimizer_applications = int(
        final["n_updates"]
        * args.student_rl_args.n_epochs
        * args.student_rl_args.n_minibatches
    )

    # The first checkpoint written to this output directory is the state after
    # loop termination.  No periodic checkpoint or resume state is accepted.
    checkpoint_state = experiment.runner.get_checkpoint_state(state)
    checkpoint_path = output_dir / "checkpoint.pkl"
    require(not checkpoint_path.exists(), "a nonterminal checkpoint already exists")
    safe_checkpoint(checkpoint_state, str(output_dir), "checkpoint")
    require(checkpoint_path.is_file() and not checkpoint_path.is_symlink(), "terminal checkpoint missing")
    checkpoint_sha = sha256(checkpoint_path)
    reloaded = load_pkl_object(str(checkpoint_path))
    checkpoint_summary = _fresh_checkpoint_summary(reloaded, arm)
    require(
        checkpoint_summary["n_iters"] == final["n_iters"]
        and checkpoint_summary["n_updates"] == final["n_updates"]
        and checkpoint_summary["n_grad_updates"] == final["n_grad_updates"],
        "terminal checkpoint counter freshness failed",
    )
    require(
        checkpoint_summary["replay_integrity"] == final["replay_integrity"],
        "terminal checkpoint replay telemetry freshness failed",
    )
    if arm == "frontier":
        integrity = final["frontier_integrity"]
        require(
            checkpoint_summary["buffer_total_trials"] == integrity["buffer_total_trials"]
            and checkpoint_summary["buffer_total_successes"] == integrity["buffer_total_successes"],
            "terminal checkpoint posterior freshness failed",
        )

    snapshot = plr_replay_snapshot(
        state,
        args,
        context,
        checkpoint_sha,
        jax,
        experiment.runner.plr_mgr,
        arm,
    )

    if hasattr(logger, "_logfile"):
        logger._logfile.flush()
        os.fsync(logger._logfile.fileno())
        logger._logfile.close()
    logs_path = output_dir / "logs.csv"
    require(logs_path.is_file() and not logs_path.is_symlink(), "source-format logs are missing")
    _write_safe_meta(output_dir, args, context, source_receipt, successful=True)
    meta_path = output_dir / "meta.json"

    endpoint = {
        "schema": 1,
        "status": "completed",
        "protocol_id": PROTOCOL_ID,
        "purpose": PURPOSE,
        "paper_evidence": False,
        "endpoint_class": "bounded_engineering_test",
        "run_id": context["run_id"],
        "arm": arm,
        "training_seed": context["training_seed"],
        "n_updates": final["n_updates"],
        "n_grad_updates": final["n_grad_updates"],
        "optimizer_step_applications": optimizer_applications,
        "outer_cycles": tick,
        "student_training_transitions": transitions,
        "checkpoint_file": "checkpoint.pkl",
        "checkpoint_sha256": checkpoint_sha,
        "terminal_checkpoint_saved_after_training": True,
        "resumed": False,
        "replay_integrity": final["replay_integrity"],
        "frontier_integrity": final["frontier_integrity"],
    }
    endpoint_path = output_dir / "endpoint.json"
    atomic_json(endpoint_path, endpoint)

    resolved_config = _as_plain(args)
    receipt = {
        "schema": 1,
        "status": "completed",
        "protocol_id": PROTOCOL_ID,
        "purpose": PURPOSE,
        "paper_evidence": False,
        "endpoint_class": "bounded_engineering_test",
        "run_id": context["run_id"],
        "arm": arm,
        "training_seed": context["training_seed"],
        "job_id": context["job_id"],
        "resumed": False,
        "outer_cycles": tick,
        "student_training_transitions": transitions,
        "transitions_per_outer_cycle": transitions_per_cycle,
        "n_updates": final["n_updates"],
        "upstream_n_grad_updates": final["n_grad_updates"],
        "optimizer_step_applications": optimizer_applications,
        "periodic_evaluation_accounting": {
            "calls": periodic_evaluation_calls,
            "test_interval_outer_cycles": int(args.test_interval),
            "environment_count": PERIODIC_EVAL_ENVIRONMENTS,
            "episodes_per_environment": PERIODIC_EVAL_EPISODES_PER_ENVIRONMENT,
            "max_episode_horizon": PERIODIC_EVAL_HORIZON,
            "per_environment_max_episode_horizons": list(periodic_eval_horizons),
            "budgeted_max_transitions": periodic_evaluation_transitions,
            "runner_scans_full_horizon": True,
            "excluded_from_student_training_transitions": True,
        },
        "optimizer_step_formula": {
            "n_updates": final["n_updates"],
            "student_n_epochs": int(args.student_rl_args.n_epochs),
            "student_n_minibatches": int(args.student_rl_args.n_minibatches),
        },
        "integrity": {
            "initial": initial,
            "terminal": final,
            "checkpoint_round_trip": checkpoint_summary,
            "last_step_stats": last_stats,
            "max_outer_cycles": max_outer_cycles,
        },
        "engineering_test": {
            "enabled": engineering_mode,
            "execution_mode": (
                "slurm" if cli.slurm_engineering_test_mode
                else "local" if cli.engineering_test_mode
                else "production"
            ),
            "overrides": override_records,
        },
        "terminal_checkpoint": {
            "path": "checkpoint.pkl",
            "sha256": checkpoint_sha,
            "saved_after_loop_termination": True,
            "periodic_checkpoint_used": False,
            "round_trip_counter_freshness": True,
        },
        "config": {
            "authored_path": cli.config.resolve().name,
            "authored_sha256": config_sha,
            "resolved": resolved_config,
            "resolved_canonical_sha256": canonical_sha256(resolved_config),
            "meta_sha256": sha256(meta_path),
            "logs_sha256": sha256(logs_path),
        },
        "provenance": {
            "run_context": context,
            "run_context_sha256": cli.expected_run_context_sha256,
            "protocol_sha256": protocol_sha,
            "training_driver_sha256": driver_sha,
            "source": source_receipt,
            "minimax_module": str(minimax_path),
            "backend": jax.default_backend(),
            "devices": device_receipt,
        },
        "endpoint": {"path": endpoint_path.name, "sha256": sha256(endpoint_path)},
        "wall_seconds": time.monotonic() - wall_started,
    }
    return write_training_sidecar(sidecar_dir, receipt, snapshot)


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--campaign-manifest",
        type=Path,
        help="frozen campaign manifest; required under Slurm/production",
    )
    parser.add_argument(
        "--expected-campaign-manifest-sha256",
        help="exact digest of --campaign-manifest; required under Slurm/production",
    )
    parser.add_argument("--run-context", type=Path, required=True)
    parser.add_argument("--expected-run-context-sha256", required=True)
    parser.add_argument("--expected-driver-sha256", required=True)
    parser.add_argument("--patched-source-dir", type=Path, required=True)
    parser.add_argument(
        "--git-executable",
        type=Path,
        help="absolute Git executable; required from the active pinned environment under Slurm/production",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--engineering-test-mode", action="store_true")
    parser.add_argument(
        "--slurm-engineering-test-mode",
        action="store_true",
        help="bounded, permanently non-evidence engineering mode under Slurm",
    )
    parser.add_argument(
        "--engineering-override",
        action="append",
        default=[],
        metavar="FIELD=JSON",
        help="bounded test-only override; repeat for multiple fields",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        receipt = run(parse_cli(argv))
    except (DriverError, AssertionError, KeyError, TypeError, ValueError) as exc:
        print(f"MATCHED_TERMINAL_REFUSED: {exc}", file=sys.stderr)
        return 1
    print(
        "MATCHED_TERMINAL_COMPLETE "
        f"run_id={receipt['run_id']} updates={receipt['n_updates']} "
        f"outer_cycles={receipt['outer_cycles']} "
        f"checkpoint={receipt['terminal_checkpoint']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
