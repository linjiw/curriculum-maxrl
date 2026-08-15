#!/usr/bin/env python3
"""Finalize one terminal-chain engineering package after terminal Slurm accounting.

This is phase B of a deliberately two-phase workflow.  The Slurm job publishes
closed training/evaluation components and exits.  Only after the allocation is
terminal does this helper combine those fetched components with a hardened
``terminal-receipt`` capture and fetched Slurm logs.  It invokes the frozen
assembler in engineering mode; it never invokes the production analyzer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import site
import subprocess
import sys
import tempfile
from typing import Any, Iterable
from zoneinfo import ZoneInfo


HASH_RE = re.compile(r"[0-9a-f]{64}")
UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
SLURM_TIME_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
)
TERMINAL_HEADER = (
    "JobIDRaw|JobName|Partition|State|ExitCode|ElapsedRaw|AllocCPUS|ReqMem|"
    "NodeList|Submit|Start|End|AllocTRES|QOS|TimelimitRaw|Restarts|WorkDir|"
    "StdOut|StdErr|SubmitLine"
)
RESOURCE_HEADER = "JobIDRaw|MaxRSS|TRESUsageInMax"
SUBMISSION_HEADER = (
    "job_id\tutc\thost\tlocal_script\tlocal_sha256\tremote_script\t"
    "remote_sha256\toutput_path\tremote_receipt\tsbatch_args"
)
COMPONENT_ENDPOINT_CLASS = "bounded_engineering_terminal_chain_components"
EXPECTED_JOB_NAME = "ued-minimax-terminal-chain"
EXPECTED_PARTITION = "gpuq"
EXPECTED_GPU_PROFILE = "1g.10gb"
EXPECTED_EVALUATION_TRANSITIONS = 13_500


class FinalizationError(RuntimeError):
    """Raised when phase B cannot publish a complete, honest package."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizationError(message)


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"unsafe or missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FinalizationError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"invalid {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def safe_existing(path: Path, *, directory: bool, label: str) -> Path:
    require(path.is_absolute() and ".." not in path.parts, f"{label} must be canonical absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FinalizationError(f"missing {label}: {path}") from exc
    require(resolved == path, f"{label} contains a symlink or noncanonical component")
    if directory:
        require(path.is_dir() and not path.is_symlink(), f"unsafe {label}: {path}")
    else:
        require(path.is_file() and not path.is_symlink(), f"unsafe {label}: {path}")
    return path


def validate_manifest(root: Path, name: str, expected_sha: str) -> list[str]:
    require(HASH_RE.fullmatch(expected_sha or "") is not None, f"malformed expected {name} hash")
    manifest = root / name
    require(sha256(manifest) == expected_sha, f"{name} digest mismatch")
    entries: list[str] = []
    seen: set[str] = set()
    for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\\]+)", raw)
        require(match is not None, f"unsafe {name} line {number}")
        expected, rel_text = match.groups()
        rel = PurePosixPath(rel_text)
        require(
            not rel.is_absolute()
            and rel_text not in {"", "."}
            and all(part not in {"", ".", ".."} for part in rel.parts),
            f"unsafe {name} path: {rel_text}",
        )
        normalized = rel.as_posix()
        require(normalized not in seen, f"duplicate {name} path: {rel_text}")
        seen.add(normalized)
        target = root.joinpath(*rel.parts)
        require(target.is_file() and not target.is_symlink(), f"missing manifest payload: {rel_text}")
        require(target.resolve().is_relative_to(root), f"manifest payload escaped root: {rel_text}")
        require(sha256(target) == expected, f"payload digest mismatch: {rel_text}")
        entries.append(normalized)
    require(entries, f"empty {name}")
    return entries


def exact_file_tree(root: Path, allowed_unlisted: Iterable[str], listed: Iterable[str]) -> None:
    allowed = set(allowed_unlisted) | set(listed)
    actual: set[str] = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), f"symbolic link in closure: {path}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        else:
            require(path.is_dir(), f"non-regular closure entry: {path}")
    require(actual == allowed, "closure contains missing or unmanifested files")


def parse_terminal_receipt(path: Path, job_id: str) -> dict[str, Any]:
    values: dict[str, list[str]] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split("\t", 1)
        require(len(fields) == 2 and fields[0] and fields[1] != "", f"bad receipt line {number}")
        values.setdefault(fields[0], []).append(fields[1])
    require(set(values) == {
        "terminal_receipt_schema", "retrieved_utc", "retrieved_epoch",
        "terminal_end_epoch", "terminal_header", "terminal_row",
        "resource_header", "resource_row",
    }, "terminal receipt keys drift")
    for singleton in (
        "terminal_receipt_schema", "retrieved_utc", "retrieved_epoch",
        "terminal_end_epoch", "terminal_header", "terminal_row", "resource_header",
    ):
        require(len(values[singleton]) == 1, f"terminal receipt cardinality drift: {singleton}")
    require(values["terminal_receipt_schema"] == ["2"], "terminal receipt schema drift")
    retrieved = values["retrieved_utc"][0]
    require(UTC_RE.fullmatch(retrieved) is not None, "terminal retrieval UTC drift")
    require(values["retrieved_epoch"][0].isdigit(), "terminal retrieval epoch drift")
    require(values["terminal_end_epoch"][0].isdigit(), "terminal end epoch drift")
    retrieved_epoch = int(values["retrieved_epoch"][0])
    terminal_end_epoch = int(values["terminal_end_epoch"][0])
    utc_epoch = int(datetime.strptime(retrieved, "%Y-%m-%dT%H:%M:%SZ")
                    .replace(tzinfo=timezone.utc).timestamp())
    require(retrieved_epoch == utc_epoch, "terminal retrieval UTC/epoch mismatch")
    require(retrieved_epoch >= terminal_end_epoch, "terminal receipt predates job end")
    require(values["terminal_header"] == [TERMINAL_HEADER], "terminal receipt header drift")
    require(values["resource_header"] == [RESOURCE_HEADER], "resource receipt header drift")
    terminal = values["terminal_row"][0].split("|")
    require(len(terminal) == 20 and terminal[0] == job_id, "terminal allocation row drift")
    require(terminal[3] == "COMPLETED" and terminal[4] == "0:0", "job did not complete cleanly")
    require(terminal[5].isdigit() and int(terminal[5]) > 0, "invalid terminal elapsed time")
    require(terminal[6].isdigit() and int(terminal[6]) > 0, "invalid allocated CPU count")
    require(all(terminal[index] for index in (
        1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)),
            "terminal allocation fields are incomplete")
    require(terminal[14].isdigit() and int(terminal[14]) > 0,
            "invalid terminal time limit")
    require(terminal[15].isdigit(), "invalid terminal restart count")
    require(int(terminal[5]) <= int(terminal[14]) * 60,
            "elapsed time exceeds authoritative time limit")
    scheduler_times: list[int] = []
    for label, value in zip(("Submit", "Start", "End"), terminal[9:12]):
        require(SLURM_TIME_RE.fullmatch(value) is not None,
                f"invalid Slurm {label} timestamp")
        naive = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
        scheduler_zone = ZoneInfo("America/New_York")
        parsed = naive.replace(tzinfo=scheduler_zone, fold=0)
        alternate = naive.replace(tzinfo=scheduler_zone, fold=1)
        require(parsed.utcoffset() == alternate.utcoffset(),
                f"ambiguous or nonexistent Slurm {label} local timestamp")
        scheduler_times.append(int(parsed.timestamp()))
    submit_epoch, start_epoch, end_epoch = scheduler_times
    require(submit_epoch <= start_epoch <= end_epoch,
            "authoritative Slurm timestamps are unordered")
    require(end_epoch == terminal_end_epoch,
            "terminal End timestamp/epoch binding drift")
    require(end_epoch - start_epoch == int(terminal[5]),
            "terminal ElapsedRaw/Start/End binding drift")
    resources: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in values["resource_row"]:
        fields = raw.split("|")
        require(len(fields) == 3, "resource accounting row width drift")
        row_id = fields[0]
        require(row_id == job_id or row_id.startswith(job_id + "."), "foreign resource row")
        require(row_id not in seen, "duplicate resource row")
        seen.add(row_id)
        resources.append({"job_id_raw": row_id, "max_rss": fields[1], "tres_usage_in_max": fields[2]})
    require(resources, "terminal receipt has no resource rows")
    return {
        "retrieved_utc": retrieved,
        "retrieved_epoch": retrieved_epoch,
        "terminal_end_epoch": terminal_end_epoch,
        "job_id": terminal[0],
        "job_name": terminal[1],
        "partition": terminal[2],
        "state": terminal[3],
        "exit_code": terminal[4],
        "elapsed_seconds": int(terminal[5]),
        "alloc_cpus": int(terminal[6]),
        "req_mem": terminal[7],
        "node_list": terminal[8],
        "submit": terminal[9],
        "start": terminal[10],
        "end": terminal[11],
        "alloc_tres": terminal[12],
        "qos": terminal[13],
        "timelimit_minutes": int(terminal[14]),
        "restarts": int(terminal[15]),
        "work_dir": terminal[16],
        "stdout_path": terminal[17],
        "stderr_path": terminal[18],
        "submit_line": terminal[19],
        "resource_rows": resources,
    }


def parse_alloc_tres(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in value.split(","):
        fields = item.split("=", 1)
        require(len(fields) == 2 and all(fields), "malformed AllocTRES field")
        key, quantity = fields
        require(key not in result, f"duplicate AllocTRES field: {key}")
        result[key] = quantity
    return result


def parse_singleton_tsv(path: Path, expected_keys: set[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split("\t", 1)
        require(len(fields) == 2 and all(fields), f"bad {label} line {number}")
        key, value = fields
        require(key not in result, f"duplicate {label} key: {key}")
        result[key] = value
    require(set(result) == expected_keys, f"{label} keys drift")
    return result


def local_tree_digest(root: Path) -> str:
    lines: list[str] = []
    for path in sorted((path for path in root.rglob("*") if path.is_file()),
                       key=lambda path: path.relative_to(root).as_posix()):
        require(not path.is_symlink(), f"symbolic link in fetched tree: {path}")
        relative = path.relative_to(root).as_posix()
        require("\n" not in relative and "\\" not in relative,
                "unsupported fetched-tree path")
        lines.append(f"{sha256(path)}  ./{relative}\n")
    require(lines, "fetched directory is empty")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def parse_fetch_receipt(
    receipt_path: Path,
    *,
    local_path: Path,
    remote_path: str,
    remote_type: str,
    terminal_end_epoch: int,
    terminal_retrieved_epoch: int,
    terminal_receipt_sha256: str,
    require_manifest: bool,
) -> dict[str, str]:
    receipt = parse_singleton_tsv(receipt_path, {
        "fetch_receipt_schema", "fetch_started_utc", "fetch_started_epoch",
        "retrieved_utc", "retrieved_epoch", "terminal_end_epoch",
        "terminal_receipt_sha256", "remote_path", "remote_type", "remote_digest",
        "manifest_verified", "local_path", "local_digest",
    }, "fetch receipt")
    require(
        receipt["fetch_receipt_schema"] == "2"
        and UTC_RE.fullmatch(receipt["fetch_started_utc"]) is not None
        and UTC_RE.fullmatch(receipt["retrieved_utc"]) is not None
        and receipt["fetch_started_epoch"].isdigit()
        and receipt["retrieved_epoch"].isdigit()
        and receipt["terminal_end_epoch"].isdigit()
        and receipt["terminal_receipt_sha256"] == terminal_receipt_sha256
        and receipt["remote_path"] == remote_path
        and receipt["remote_type"] == remote_type
        and HASH_RE.fullmatch(receipt["remote_digest"]) is not None
        and HASH_RE.fullmatch(receipt["local_digest"]) is not None
        and receipt["remote_digest"] == receipt["local_digest"]
        and receipt["manifest_verified"] == ("1" if require_manifest else "0")
        and receipt["local_path"] == str(local_path),
        "fetch receipt identity/digest drift",
    )
    started_epoch = int(receipt["fetch_started_epoch"])
    retrieved_epoch = int(receipt["retrieved_epoch"])
    started_utc_epoch = int(datetime.strptime(
        receipt["fetch_started_utc"], "%Y-%m-%dT%H:%M:%SZ")
        .replace(tzinfo=timezone.utc).timestamp())
    retrieved_utc_epoch = int(datetime.strptime(
        receipt["retrieved_utc"], "%Y-%m-%dT%H:%M:%SZ")
        .replace(tzinfo=timezone.utc).timestamp())
    require(
        started_epoch == started_utc_epoch
        and retrieved_epoch == retrieved_utc_epoch
        and int(receipt["terminal_end_epoch"]) == terminal_end_epoch
        and started_epoch >= terminal_end_epoch
        and started_epoch >= terminal_retrieved_epoch
        and retrieved_epoch >= started_epoch,
        "fetch did not begin after authoritative job end",
    )
    local_digest = local_tree_digest(local_path) if remote_type == "dir" else sha256(local_path)
    require(local_digest == receipt["local_digest"], "fetched payload changed after retrieval")
    return receipt


def parse_submission_receipt(
    path: Path,
    *,
    job_id: str,
    sbatch_sha: str,
) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == 2 and lines[0] == SUBMISSION_HEADER,
            "submission receipt header/cardinality drift")
    fields = lines[1].split("\t")
    require(len(fields) == 10, "submission receipt row width drift")
    keys = SUBMISSION_HEADER.split("\t")
    receipt = dict(zip(keys, fields))
    require(all(receipt.values()), "submission receipt has an empty field")
    require(
        receipt["job_id"] == job_id
        and UTC_RE.fullmatch(receipt["utc"]) is not None
        and receipt["local_sha256"] == receipt["remote_sha256"] == sbatch_sha,
        "submission receipt job/script drift",
    )
    remote_script = receipt["remote_script"]
    require(
        re.fullmatch(
            rf"/scratch/[A-Za-z0-9._-]+/maxrl/sbatch/"
            rf"ued_minimax_terminal_chain_smoke-{sbatch_sha[:16]}-"
            rf"[0-9]{{8}}T[0-9]{{6}}Z-[0-9]+\.sbatch",
            remote_script,
        ) is not None,
        "submission receipt remote script path drift",
    )
    require(re.fullmatch(
        rf"/scratch/[A-Za-z0-9._-]+/maxrl/receipts/job-{job_id}-"
        rf"[0-9]{{8}}T[0-9]{{6}}Z\.tsv", receipt["remote_receipt"]) is not None,
            "submission receipt remote path drift")
    args = receipt["sbatch_args"]
    require(
        args.startswith("--export=UED_") and "ALL" not in args.split("=", 1)[1].split(","),
        "submission did not bind one explicit allowlisted export",
    )
    forbidden = (
        "--partition", "--qos", "--gres", "--gpus", "--nodes", "--ntasks",
        "--cpus-per-task", "--mem", "--time", "--requeue", "--job-name",
        "--output", "--error", "--chdir",
    )
    require(not any(token in args for token in forbidden),
            "submission receipt contains an identity/resource override")
    return receipt


def parse_submission_exports(value: str) -> dict[str, str]:
    require(value.startswith("--export=UED_"), "submission export syntax drift")
    result: dict[str, str] = {}
    for item in value.removeprefix("--export=").split(","):
        fields = item.split("=", 1)
        require(len(fields) == 2 and all(fields), "malformed submission export")
        key, exported = fields
        require(re.fullmatch(r"[A-Z][A-Z0-9_]+", key) is not None,
                "unsafe submission export name")
        require(key not in result, f"duplicate submission export: {key}")
        result[key] = exported
    return result


def expand_slurm_path(value: str, *, user: str, job_id: str) -> str:
    expanded = value.replace("%u", user).replace("%x", EXPECTED_JOB_NAME).replace("%j", job_id)
    require("%" not in expanded, "unsupported Slurm path token")
    return expanded


def validate_device(value: Any, physical_model: str, gpu_profile: str, label: str) -> None:
    expected_device_kind = f"{physical_model} MIG {gpu_profile}"
    require(
        isinstance(value, dict)
        and set(value) == {"id", "platform", "device_kind"}
        and isinstance(value["id"], int)
        and not isinstance(value["id"], bool)
        and value["id"] >= 0
        and value["platform"] == "gpu"
        and value["device_kind"] == expected_device_kind,
        f"{label} GPU device receipt drift",
    )


def parse_nvidia_smi(path: Path, label: str) -> tuple[str, str, str, str]:
    import csv

    try:
        rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise FinalizationError(f"invalid {label}") from exc
    require(len(rows) == 1 and len(rows[0]) == 4, f"{label} cardinality/width drift")
    row = tuple(field.strip() for field in rows[0])
    require(all(row), f"{label} has an empty field")
    require(re.fullmatch(r"GPU-[A-Za-z0-9-]+", row[0]) is not None,
            f"{label} GPU UUID drift")
    require(re.fullmatch(r"[0-9]+ MiB", row[2]) is not None,
            f"{label} memory receipt drift")
    require(re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", row[3]) is not None,
            f"{label} driver receipt drift")
    return row  # type: ignore[return-value]


def validate_phase_a_components(
    components: Path,
    complete: dict[str, Any],
    campaign: dict[str, Any],
    context: dict[str, Any],
    campaign_sha: str,
    context_sha: str,
    job_id: str,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, str, str, str]]:
    """Reject relabeled local/CPU/synthetic artifacts before assembly."""
    run_id = f"engineering-slurm-{job_id}-frontier-s101"
    require(complete["run_id"] == run_id, "component run ID is not the Slurm engineering ID")
    require(
        context.get("run_id") == run_id
        and context.get("job_id") == job_id
        and context.get("arm") == "frontier"
        and context.get("training_seed") == 101
        and context.get("campaign_manifest_sha256") == campaign_sha,
        "run-context Slurm identity drift",
    )
    submissions = campaign.get("submissions")
    require(isinstance(submissions, list) and len(submissions) == 1,
            "engineering campaign submission cardinality drift")
    require(submissions[0] == {
        "arm": "frontier", "training_seed": 101, "evaluation_seed": 100101,
        "run_id": run_id, "job_id": job_id, "attempt": 1,
    }, "engineering campaign submission identity drift")
    require(campaign.get("frozen_before_endpoint_access") is True,
            "campaign was not frozen before endpoint access")

    hardware = campaign.get("hardware")
    require(isinstance(hardware, dict) and hardware == {
        "partition": EXPECTED_PARTITION,
        "gpu_model": hardware.get("gpu_model"),
        "gpu_profile": EXPECTED_GPU_PROFILE,
        "gpu_count": 1,
        "n_devices": 1,
    } and isinstance(hardware["gpu_model"], str) and hardware["gpu_model"],
            "campaign hardware drift")
    nvidia_before = parse_nvidia_smi(components / "nvidia-smi-before.csv", "pre-run nvidia-smi")
    nvidia_after = parse_nvidia_smi(components / "nvidia-smi-after.csv", "post-run nvidia-smi")
    require(nvidia_before == nvidia_after, "GPU identity changed during phase A")
    require(nvidia_before[1] == hardware["gpu_model"], "campaign/nvidia-smi model drift")

    training = load_json(
        components / "training-sidecar/training-receipt.json", "training receipt")
    require(
        training.get("schema") == 1
        and training.get("status") == "completed"
        and training.get("paper_evidence") is False
        and training.get("endpoint_class") == "bounded_engineering_test"
        and training.get("run_id") == run_id
        and training.get("job_id") == job_id
        and training.get("arm") == "frontier"
        and training.get("training_seed") == 101
        and training.get("n_updates") == 1
        and training.get("upstream_n_grad_updates") == 1
        and training.get("optimizer_step_applications") == 1,
        "training Slurm identity/counter drift",
    )
    engineering = training.get("engineering_test")
    require(
        isinstance(engineering, dict)
        and set(engineering) == {"enabled", "execution_mode", "overrides"}
        and engineering["enabled"] is True
        and engineering["execution_mode"] == "slurm"
        and isinstance(engineering["overrides"], list)
        and len(engineering["overrides"]) >= 1,
        "training did not execute in Slurm engineering mode",
    )
    training_provenance = training.get("provenance")
    require(isinstance(training_provenance, dict), "training provenance missing")
    require(
        training_provenance.get("run_context") == context
        and training_provenance.get("run_context_sha256") == context_sha
        and training_provenance.get("backend") == "gpu",
        "training Slurm/GPU provenance drift",
    )
    training_devices = training_provenance.get("devices")
    require(isinstance(training_devices, list) and len(training_devices) == 1,
            "training GPU cardinality drift")
    validate_device(
        training_devices[0], hardware["gpu_model"], hardware["gpu_profile"], "training")
    source = training_provenance.get("source")
    require(
        isinstance(source, dict)
        and source.get("git_version") == "git version 2.45.2"
        and HASH_RE.fullmatch(str(source.get("git_executable_sha256", ""))) is not None
        and re.fullmatch(r"/scratch/[A-Za-z0-9._-]+/envs/[^/]+/bin/git",
                         str(source.get("git_executable", ""))) is not None,
        "training pinned-Git provenance drift",
    )

    evaluation = load_json(
        components / "evaluation-package/evaluation-receipt.json", "evaluation receipt")
    require(
        evaluation.get("schema") == 1
        and evaluation.get("status") == "completed"
        and evaluation.get("paper_evidence") is False
        and evaluation.get("run_id") == run_id
        and evaluation.get("arm") == "frontier"
        and evaluation.get("training_seed") == 101
        and evaluation.get("evaluation_seed") == 100101
        and evaluation.get("synthetic_test_mode") is False
        and evaluation.get("raw_results", {}).get("record_count") == 30,
        "actual evaluation identity drift",
    )
    accounting = evaluation.get("evaluation_transition_accounting")
    require(
        isinstance(accounting, dict)
        and accounting.get("environment_count") == 3
        and accounting.get("episodes_per_environment") == 10
        and accounting.get("max_episode_horizon") == 450
        and accounting.get("per_environment_max_episode_horizons") == [450, 450, 450]
        and accounting.get("budgeted_primary_max_transitions") == EXPECTED_EVALUATION_TRANSITIONS
        and accounting.get("effective_primary_transitions") == EXPECTED_EVALUATION_TRANSITIONS
        and accounting.get("primary_runner_scans_full_horizon") is True
        and accounting.get("engineering_independent_verification_transitions") == 0
        and accounting.get("total_runtime_transitions") == EXPECTED_EVALUATION_TRANSITIONS
        and accounting.get("excluded_from_student_training_transitions") is True,
        "actual evaluation transition accounting drift",
    )
    evaluation_provenance = evaluation.get("provenance")
    require(isinstance(evaluation_provenance, dict), "evaluation provenance missing")
    require(
        evaluation_provenance.get("run_context") == context
        and evaluation_provenance.get("run_context_sha256") == context_sha
        and evaluation_provenance.get("source") == source,
        "evaluation context/source drift",
    )
    runtime = evaluation_provenance.get("runtime")
    require(
        isinstance(runtime, dict)
        and runtime.get("backend") == "gpu"
        and runtime.get("device_count") == 1
        and runtime.get("per_environment_max_episode_horizons") == [450, 450, 450],
        "evaluation GPU runtime drift",
    )
    evaluation_devices = runtime.get("devices")
    require(isinstance(evaluation_devices, list) and len(evaluation_devices) == 1,
            "evaluation GPU cardinality drift")
    validate_device(
        evaluation_devices[0], hardware["gpu_model"], hardware["gpu_profile"], "evaluation")
    parity = runtime.get("raw_vs_independent_evalrunner")
    require(
        isinstance(parity, dict)
        and parity.get("checked") is False
        and parity.get("all_six_fields_checked") is False
        and parity.get("max_abs_error") is None
        and parity.get("per_field_abs_error") is None,
        "phase-A evaluator unexpectedly changed execution budget",
    )
    return training, evaluation, nvidia_before


def validate_completion_marker(
    stdout: Path,
    stderr: Path,
    job_id: str,
    components_sha: str,
    complete: dict[str, Any],
) -> None:
    marker = re.compile(
        r"UED_TERMINAL_COMPONENTS_COMPLETE job=([0-9]+) "
        r"manifest=([0-9a-f]{64}) result=([^ ]+) analyzer_eligible=false"
    )
    stdout_lines = stdout.read_text(encoding="utf-8").splitlines()
    stderr_lines = stderr.read_text(encoding="utf-8").splitlines()
    matches = [match for line in stdout_lines if (match := marker.fullmatch(line))]
    require(len(matches) == 1, "Slurm stdout must contain exactly one completion marker")
    require(not any(marker.fullmatch(line) for line in stderr_lines),
            "completion marker appeared in Slurm stderr")
    match = matches[0]
    require(match.group(1) == job_id, "completion marker job drift")
    require(match.group(2) == components_sha, "completion marker manifest drift")
    result_dir = match.group(3)
    require(result_dir == complete["result_dir"], "completion marker result path drift")
    closure_sha = complete["input_closure_sha256"]
    require(
        re.fullmatch(
            rf"/scratch/[A-Za-z0-9._-]+/maxrl/tests/ued-minimax-terminal-chain/"
            rf"{closure_sha[:20]}/job-{job_id}",
            result_dir,
        ) is not None,
        "completion marker result path is outside the input-addressed closure",
    )


def validate_input_closure(
    components: Path,
    complete: dict[str, Any],
    campaign: dict[str, Any],
    expected_sha: str,
    expected_assembler_sha: str,
    expected_finalizer_sha: str,
) -> dict[str, Any]:
    closure_path = components / "INPUT_CLOSURE.json"
    require(sha256(closure_path) == expected_sha == complete["input_closure_sha256"],
            "input-closure digest binding drift")
    closure = load_json(closure_path, "input closure")
    require(set(closure) == {
        "input_closure_schema", "purpose", "paper_evidence", "analyzer_eligible",
        "endpoint_class", "git", "two_phase", "resources", "schedule", "hashes",
    }, "input-closure keys drift")
    require(
        closure["input_closure_schema"] == 1
        and closure["purpose"] == "bounded Frontier terminal-chain Slurm engineering smoke"
        and closure["paper_evidence"] is False
        and closure["analyzer_eligible"] is False
        and closure["endpoint_class"] == COMPONENT_ENDPOINT_CLASS
        and closure["git"] == "git version 2.45.2",
        "input-closure identity drift",
    )
    require(closure["two_phase"] == {
        "phase_a": "slurm_components_only",
        "phase_b": "post_terminal_local_engineering_assembly",
        "production_analyzer_invoked": False,
        "terminal_sacct_required": True,
    }, "input-closure phase contract drift")
    require(closure["resources"] == {
        "partition": EXPECTED_PARTITION, "qos": "gpu", "gres": "gpu:1g.10gb:1",
        "nodes": 1, "ntasks": 1, "cpus_per_task": 2,
        "memory": "15G", "walltime": "00:30:00", "requeue": False,
    }, "input-closure resource contract drift")
    require(closure["schedule"] == {
        "arm": "frontier", "training_seed": 101, "evaluation_seed": 100101,
        "n_updates": 1, "n_grad_updates": 1, "optimizer_step_applications": 1,
        "outer_cycles": 2, "student_training_transitions": 128,
        "actual_external_evaluation": True, "evaluation_environments": 3,
        "episodes_per_environment": 10, "max_episode_horizon": 450,
        "primary_evaluation_transitions": EXPECTED_EVALUATION_TRANSITIONS,
        "independent_verification_transitions": 0,
    }, "input-closure schedule drift")
    hashes = closure["hashes"]
    expected_hash_keys = {
        "bundle_manifest_sha256", "upstream_commit", "upstream_tree_git_sha1",
        "upstream_git_bundle_sha256", "overlay_manifest_sha256",
        "terminal_chain_sbatch_sha256", "frontier_config_sha256",
        "overlay_contract_sha256", "protocol_sha256", "analyzer_sha256",
        "training_driver_sha256", "evaluation_driver_sha256", "assembler_sha256",
        "environment_lock_sha256", "environment_freeze_sha256",
        "environment_manifest_sha256", "environment_setup_script_sha256",
        "conda_explicit_sha256", "environment_json_sha256",
        "import_smoke_manifest_sha256", "one_update_manifest_sha256",
        "terminal_finalizer_sha256", "hopper_wrapper_sha256",
    }
    require(isinstance(hashes, dict) and set(hashes) == expected_hash_keys,
            "input-closure hash keys drift")
    for key, value in hashes.items():
        if key == "upstream_commit":
            require(re.fullmatch(r"[0-9a-f]{40}", str(value)) is not None,
                    "input-closure upstream commit malformed")
        elif key == "upstream_tree_git_sha1":
            require(re.fullmatch(r"[0-9a-f]{40}", str(value)) is not None,
                    "input-closure upstream tree malformed")
        else:
            require(HASH_RE.fullmatch(str(value)) is not None,
                    f"input-closure malformed hash: {key}")
    provenance = campaign.get("provenance")
    require(isinstance(provenance, dict), "campaign provenance missing")
    closure_to_campaign = {
        "bundle_manifest_sha256": "bundle_manifest_sha256",
        "overlay_manifest_sha256": "overlay_manifest_sha256",
        "terminal_chain_sbatch_sha256": "sbatch_sha256",
        "overlay_contract_sha256": "overlay_contract_sha256",
        "training_driver_sha256": "training_driver_sha256",
        "evaluation_driver_sha256": "evaluation_driver_sha256",
        "assembler_sha256": "assembler_driver_sha256",
        "environment_manifest_sha256": "environment_manifest_sha256",
        "upstream_commit": "base_commit",
        "upstream_tree_git_sha1": "base_tree",
    }
    for closure_key, campaign_key in closure_to_campaign.items():
        require(hashes[closure_key] == provenance.get(campaign_key),
                f"input-closure/campaign hash drift: {closure_key}")
    require(hashes["bundle_manifest_sha256"] == complete["bundle_manifest_sha256"],
            "input-closure/component bundle drift")
    require(hashes["protocol_sha256"] == campaign.get("protocol_sha256"),
            "input-closure campaign protocol drift")
    require(hashes["analyzer_sha256"] == campaign.get("analyzer_sha256"),
            "input-closure campaign analyzer drift")
    require(hashes["assembler_sha256"] == expected_assembler_sha,
            "input-closure assembler drift")
    require(hashes["terminal_finalizer_sha256"] == expected_finalizer_sha,
            "input-closure finalizer drift")
    require(sha256(components / "environment/ENVIRONMENT_SHA256SUMS")
            == hashes["environment_manifest_sha256"], "input-closure environment drift")
    require(sha256(components / "environment/ENVIRONMENT.freeze")
            == hashes["environment_freeze_sha256"], "input-closure freeze drift")
    require(sha256(components / "environment/CONDA_EXPLICIT.txt")
            == hashes["conda_explicit_sha256"], "input-closure Conda drift")
    require(sha256(components / "environment/ENVIRONMENT.json")
            == hashes["environment_json_sha256"], "input-closure environment JSON drift")
    require(sha256(components / "import-smoke/SHA256SUMS")
            == hashes["import_smoke_manifest_sha256"], "input-closure import gate drift")
    require(sha256(components / "one-update/SHA256SUMS")
            == hashes["one_update_manifest_sha256"], "input-closure one-update gate drift")
    return closure


def slurm_quantity_bytes(value: str) -> int:
    if value == "":
        return 0
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KkMmGgTtPp]?)", value)
    require(match is not None, f"unsupported Slurm quantity: {value}")
    number, suffix = match.groups()
    multiplier = {
        "": 1, "K": 1024, "M": 1024**2, "G": 1024**3,
        "T": 1024**4, "P": 1024**5,
    }[suffix.upper()]
    return int(float(number) * multiplier)


def resource_maxima(rows: list[dict[str, str]]) -> tuple[int, int]:
    max_rss = 0
    peak_gpu = 0
    for row in rows:
        max_rss = max(max_rss, slurm_quantity_bytes(row["max_rss"]))
        for item in row["tres_usage_in_max"].split(","):
            if item.startswith("gres/gpumem="):
                peak_gpu = max(peak_gpu, slurm_quantity_bytes(item.split("=", 1)[1]))
    return max_rss, peak_gpu


def write_json(path: Path, value: Any) -> None:
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def copy_file(source: Path, destination: Path) -> None:
    require(not destination.exists() and not destination.is_symlink(), f"refusing to overwrite {destination}")
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    require(sha256(source) == sha256(destination), f"copy digest mismatch: {source}")


def build_outer_manifest(root: Path) -> str:
    payloads = sorted(
        relative
        for path in root.rglob("*")
        if path.is_file()
        for relative in [path.relative_to(root).as_posix()]
        if relative not in {"SHA256SUMS", "COMPLETE"}
    )
    require(payloads, "empty finalization package")
    text = "".join(f"{sha256(root / rel)}  {rel}\n" for rel in payloads)
    manifest = root / "SHA256SUMS"
    with manifest.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(manifest)


def phase_b_environment(python: Path, expected_freeze_sha: str) -> tuple[dict[str, str], str]:
    environment = os.environ.copy()
    for name in (
        "PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP",
        "PYTHONOPTIMIZE", "LD_LIBRARY_PATH", "LD_PRELOAD",
        "JAX_DISABLE_JIT", "JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_COMPILATION_CACHE_DIR", "XLA_FLAGS", "NVIDIA_TF32_OVERRIDE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR",
        "CUDA_LAUNCH_BLOCKING", "TF_ENABLE_ONEDNN_OPTS",
    ):
        environment.pop(name, None)
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_CACHE_DIR": "1",
        "LC_ALL": "C",
    })
    check = subprocess.run(
        [str(python), "-I", "-B", "-m", "pip", "check"], env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
    )
    require(check.returncode == 0, "Phase-B pinned environment failed pip check")
    frozen = subprocess.run(
        [str(python), "-I", "-B", "-m", "pip", "freeze", "--all"],
        env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
    )
    require(frozen.returncode == 0, "could not resolve Phase-B package freeze")
    lines = sorted(line for line in frozen.stdout.splitlines() if line)
    require(lines, "Phase-B package freeze is empty")
    freeze_text = "".join(f"{line}\n" for line in lines)
    freeze_sha = hashlib.sha256(freeze_text.encode("utf-8")).hexdigest()
    require(freeze_sha == expected_freeze_sha, "Phase-B package freeze digest drift")
    return environment, freeze_text


def finalize(cli: argparse.Namespace) -> dict[str, Any]:
    job_id = cli.job_id
    require(
        sys.flags.isolated == 1
        and sys.flags.ignore_environment == 1
        and sys.flags.no_user_site == 1
        and sys.flags.optimize == 0
        and sys.dont_write_bytecode is True
        and site.ENABLE_USER_SITE is False
        and all(name not in os.environ for name in (
            "PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP",
            "PYTHONOPTIMIZE", "LD_LIBRARY_PATH", "LD_PRELOAD",
            "JAX_DISABLE_JIT", "JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION",
            "JAX_COMPILATION_CACHE_DIR", "XLA_FLAGS", "NVIDIA_TF32_OVERRIDE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR",
            "CUDA_LAUNCH_BLOCKING", "TF_ENABLE_ONEDNN_OPTS",
        )),
        "Phase-B finalizer requires an isolated, injection-free Python launch",
    )
    require(re.fullmatch(r"[0-9]+", job_id) is not None, "job ID must be numeric")
    for value, label in (
        (cli.expected_bundle_manifest_sha256, "bundle manifest"),
        (cli.expected_components_manifest_sha256, "components manifest"),
        (cli.expected_input_closure_sha256, "input closure"),
        (cli.expected_sbatch_sha256, "terminal-chain sbatch"),
        (cli.expected_assembler_sha256, "assembler"),
        (cli.expected_finalizer_sha256, "finalizer"),
        (cli.expected_python_sha256, "Python executable"),
        (cli.expected_python_freeze_sha256, "Python package freeze"),
        (cli.expected_python_venv_config_sha256, "Python venv config"),
    ):
        require(HASH_RE.fullmatch(value or "") is not None, f"malformed expected {label} hash")

    receipt_path = safe_existing(cli.terminal_receipt, directory=False, label="terminal receipt")
    terminal_receipt_sha = sha256(receipt_path)
    terminal = parse_terminal_receipt(receipt_path, job_id)
    require(
        terminal["job_name"] == EXPECTED_JOB_NAME
        and terminal["partition"] == EXPECTED_PARTITION
        and terminal["state"] == "COMPLETED"
        and terminal["exit_code"] == "0:0"
        and terminal["alloc_cpus"] == 2
        and terminal["req_mem"] == "15G"
        and terminal["qos"] == "gpu"
        and terminal["timelimit_minutes"] == 30
        and terminal["restarts"] == 0,
        "terminal job/resource request drift",
    )
    alloc_tres = parse_alloc_tres(terminal["alloc_tres"])
    require(
        set(alloc_tres) == {
            "billing", "cpu", "gres/gpu", "gres/gpu:1g.10gb", "mem", "node"
        }
        and alloc_tres["billing"] == "20"
        and alloc_tres["cpu"] == "2"
        and alloc_tres["gres/gpu"] == "1"
        and alloc_tres["gres/gpu:1g.10gb"] == "1"
        and alloc_tres["mem"] == "15G"
        and alloc_tres["node"] == "1",
        "terminal allocated TRES drift",
    )
    work_match = re.fullmatch(r"/scratch/([A-Za-z0-9._-]+)/maxrl", terminal["work_dir"])
    require(work_match is not None, "terminal work directory drift")
    remote_user = work_match.group(1)
    expected_remote_components = (
        f"/scratch/{remote_user}/maxrl/tests/ued-minimax-terminal-chain/"
        f"{cli.expected_input_closure_sha256[:20]}/job-{job_id}"
    )
    expected_stdout = (
        f"/scratch/{remote_user}/maxrl/tests/logs/{EXPECTED_JOB_NAME}_{job_id}.out"
    )
    expected_stderr = (
        f"/scratch/{remote_user}/maxrl/tests/logs/{EXPECTED_JOB_NAME}_{job_id}.err"
    )
    require(
        expand_slurm_path(terminal["stdout_path"], user=remote_user, job_id=job_id)
        == expected_stdout
        and expand_slurm_path(terminal["stderr_path"], user=remote_user, job_id=job_id)
        == expected_stderr,
        "authoritative Slurm log path drift",
    )

    # Attest the exact bundled finalizer and its isolated interpreter before
    # opening any fetched component or Slurm-log payload.
    bundle = safe_existing(cli.bundle_dir, directory=True, label="bundle directory")
    bundle_entries = validate_manifest(
        bundle, "SHA256SUMS", cli.expected_bundle_manifest_sha256)
    exact_file_tree(bundle, {"SHA256SUMS"}, bundle_entries)
    assembler = bundle / "ued_benchmark/scripts/assemble_matched_run.py"
    require(sha256(assembler) == cli.expected_assembler_sha256, "assembler digest mismatch")
    bundled_finalizer = bundle / "hopper/finalize_ued_minimax_terminal_chain.py"
    self_path = Path(__file__).resolve(strict=True)
    require(
        self_path == bundled_finalizer
        and sha256(self_path) == cli.expected_finalizer_sha256,
        "Phase-B did not execute the exact manifest-bound bundled finalizer",
    )
    python_launcher = cli.python
    require(python_launcher.is_absolute() and ".." not in python_launcher.parts,
            "Python must be a canonical absolute executable")
    try:
        python_binary = python_launcher.resolve(strict=True)
    except OSError as exc:
        raise FinalizationError("Python executable is missing") from exc
    require(python_binary.is_file() and not python_binary.is_symlink()
            and os.access(python_launcher, os.X_OK),
            "Python must resolve to a regular executable")
    require(Path(sys.executable) == python_launcher,
            "Phase-B helper was not launched through the requested venv")
    require(Path(sys.executable).resolve() == python_binary,
            "Phase-B helper must run under the requested pinned Python")
    python_prefix = Path(sys.prefix)
    require(python_prefix.is_absolute() and python_prefix.resolve() == python_prefix
            and python_launcher.parent.parent == python_prefix,
            "Phase-B Python is not bound to its virtual environment prefix")
    venv_config = python_prefix / "pyvenv.cfg"
    require(sha256(venv_config) == cli.expected_python_venv_config_sha256,
            "Phase-B virtual-environment config digest drift")
    require(cli.expected_python_version == "3.10.20",
            "Phase-B Python must be pinned to 3.10.20")
    require(platform.python_version() == "3.10.20",
            "Phase-B Python version drift")
    require(sha256(python_binary) == cli.expected_python_sha256,
            "Phase-B Python executable digest drift")
    environment, python_freeze = phase_b_environment(
        python_launcher, cli.expected_python_freeze_sha256)

    submission_path = safe_existing(
        cli.submission_receipt, directory=False, label="submission receipt")
    submission_receipt_sha = sha256(submission_path)
    submission = parse_submission_receipt(
        submission_path, job_id=job_id, sbatch_sha=cli.expected_sbatch_sha256)
    require(
        submission["output_path"] == expected_stdout
        and submission["remote_script"].startswith(
            f"/scratch/{remote_user}/maxrl/sbatch/")
        and submission["remote_receipt"].startswith(
            f"/scratch/{remote_user}/maxrl/receipts/")
        and PurePosixPath(submission["local_script"]).name
        == "ued_minimax_terminal_chain_smoke.sbatch",
        "submission receipt path/user drift",
    )
    submit_tokens = shlex.split(terminal["submit_line"])
    require(
        submit_tokens == [
            "sbatch", "--parsable", submission["sbatch_args"], submission["remote_script"]
        ],
        "authoritative SubmitLine/submission-receipt drift",
    )

    # The terminal receipt is validated before any endpoint tree or Slurm log
    # is opened.  Each hardened fetch receipt records its start before its first
    # remote probe and binds this exact terminal receipt and End timestamp.
    components = safe_existing(cli.components_dir, directory=True, label="components directory")
    slurm_stdout = safe_existing(cli.slurm_stdout, directory=False, label="Slurm stdout")
    slurm_stderr = safe_existing(cli.slurm_stderr, directory=False, label="Slurm stderr")
    components_fetch_path = safe_existing(
        cli.components_fetch_receipt, directory=False, label="components fetch receipt")
    stdout_fetch_path = safe_existing(
        cli.stdout_fetch_receipt, directory=False, label="stdout fetch receipt")
    stderr_fetch_path = safe_existing(
        cli.stderr_fetch_receipt, directory=False, label="stderr fetch receipt")
    submission_fetch_path = safe_existing(
        cli.submission_fetch_receipt, directory=False, label="submission fetch receipt")
    fetch_receipt_file_hashes = {
        "fetch-components.tsv": sha256(components_fetch_path),
        "fetch-slurm-stdout.tsv": sha256(stdout_fetch_path),
        "fetch-slurm-stderr.tsv": sha256(stderr_fetch_path),
        "fetch-submission-receipt.tsv": sha256(submission_fetch_path),
    }
    fetch_receipts = {
        "components": parse_fetch_receipt(
            components_fetch_path, local_path=components,
            remote_path=expected_remote_components, remote_type="dir",
            terminal_end_epoch=terminal["terminal_end_epoch"],
            terminal_retrieved_epoch=terminal["retrieved_epoch"],
            terminal_receipt_sha256=terminal_receipt_sha, require_manifest=True),
        "stdout": parse_fetch_receipt(
            stdout_fetch_path, local_path=slurm_stdout, remote_path=expected_stdout,
            remote_type="file", terminal_end_epoch=terminal["terminal_end_epoch"],
            terminal_retrieved_epoch=terminal["retrieved_epoch"],
            terminal_receipt_sha256=terminal_receipt_sha, require_manifest=False),
        "stderr": parse_fetch_receipt(
            stderr_fetch_path, local_path=slurm_stderr, remote_path=expected_stderr,
            remote_type="file", terminal_end_epoch=terminal["terminal_end_epoch"],
            terminal_retrieved_epoch=terminal["retrieved_epoch"],
            terminal_receipt_sha256=terminal_receipt_sha, require_manifest=False),
        "submission": parse_fetch_receipt(
            submission_fetch_path, local_path=submission_path,
            remote_path=submission["remote_receipt"], remote_type="file",
            terminal_end_epoch=terminal["terminal_end_epoch"],
            terminal_retrieved_epoch=terminal["retrieved_epoch"],
            terminal_receipt_sha256=terminal_receipt_sha, require_manifest=False),
    }

    output = cli.output_dir
    require(output.is_absolute() and ".." not in output.parts, "output directory must be canonical absolute")
    require(not output.exists() and not output.is_symlink(), "output directory already exists")
    output_parent = safe_existing(output.parent, directory=True, label="output parent")
    require(
        not output.is_relative_to(bundle)
        and not bundle.is_relative_to(output)
        and not output.is_relative_to(components)
        and not components.is_relative_to(output)
        and not output.is_relative_to(python_prefix)
        and not python_prefix.is_relative_to(output),
        "output directory overlaps an immutable input closure",
    )

    component_entries = validate_manifest(
        components, "SHA256SUMS", cli.expected_components_manifest_sha256)
    exact_file_tree(components, {"SHA256SUMS", "COMPONENTS_COMPLETE.json"}, component_entries)
    require(sha256(components / "bundle-state.json") == sha256(bundle / "BUNDLE_STATE.json"),
            "component bundle-state receipt does not match the exact bundle")
    complete_path = components / "COMPONENTS_COMPLETE.json"
    complete_sha = sha256(complete_path)
    complete = load_json(complete_path, "component COMPLETE")
    required_complete = {
        "schema", "status", "paper_evidence", "analyzer_eligible", "endpoint_class",
        "job_id", "run_id", "arm", "sha256sums_sha256", "file_count",
        "bundle_manifest_sha256", "campaign_manifest_sha256", "run_context_sha256",
        "training_sidecar_manifest_sha256", "evaluation_package_manifest_sha256",
        "actual_student_updates", "actual_external_evaluation", "raw_evaluation_records",
        "terminal_sacct_included", "phase_b_required", "input_closure_sha256",
        "result_dir",
    }
    require(set(complete) == required_complete, "component COMPLETE keys drift")
    require(
        complete["schema"] == 1
        and complete["status"] == "complete"
        and complete["paper_evidence"] is False
        and complete["analyzer_eligible"] is False
        and complete["endpoint_class"] == COMPONENT_ENDPOINT_CLASS
        and complete["job_id"] == job_id
        and complete["arm"] == "frontier"
        and complete["sha256sums_sha256"] == cli.expected_components_manifest_sha256
        and complete["file_count"] == len(component_entries)
        and complete["bundle_manifest_sha256"] == cli.expected_bundle_manifest_sha256
        and complete["actual_student_updates"] == 1
        and complete["actual_external_evaluation"] is True
        and complete["raw_evaluation_records"] == 30
        and complete["terminal_sacct_included"] is False
        and complete["phase_b_required"] is True,
        "component COMPLETE binding drift",
    )
    exact_component_paths = {
        "campaign-manifest.json", "run-context.json", "command.txt",
        "INPUT_CLOSURE.json", "resource-accounting.json", "nvidia-smi-before.csv",
        "nvidia-smi-after.csv", "bundle-state.json", "applied-overlay-manifest.json",
        "overlay-check.json", "overlay-apply.json", "overlay-postcheck.json",
        "training.stdout", "training.stderr", "evaluation.stdout", "evaluation.stderr",
        "training-output/checkpoint.pkl",
        "training-output/endpoint.json", "training-output/logs.csv",
        "training-output/meta.json", "training-sidecar/training-receipt.json",
        "training-sidecar/frontier-buffer-snapshot.json", "training-sidecar/SHA256SUMS",
        "training-sidecar/COMPLETE", "evaluation-package/evaluation-episodes.jsonl",
        "evaluation-package/evaluation.csv", "evaluation-package/evaluation-receipt.json",
        "evaluation-package/SHA256SUMS", "evaluation-package/COMPLETE",
    }
    nested_specs = (
        ("environment", "ENVIRONMENT_SHA256SUMS", {"ENVIRONMENT_COMPLETE"}),
        ("import-smoke", "SHA256SUMS", {"COMPLETE"}),
        ("one-update", "SHA256SUMS", {"COMPLETE"}),
    )
    for relative, manifest_name, unlisted in nested_specs:
        nested_root = components / relative
        nested_entries = validate_manifest(
            nested_root, manifest_name, sha256(nested_root / manifest_name))
        exact_file_tree(nested_root, {manifest_name} | unlisted, nested_entries)
        exact_component_paths.add(f"{relative}/{manifest_name}")
        exact_component_paths.update(f"{relative}/{name}" for name in unlisted)
        exact_component_paths.update(f"{relative}/{name}" for name in nested_entries)
    require(set(component_entries) == exact_component_paths,
            "component payload closure contains missing or extra files")
    require(
        sha256(components / "INPUT_CLOSURE.json") == cli.expected_input_closure_sha256
        == complete["input_closure_sha256"],
        "input-closure digest binding drift",
    )
    require(
        sha256(components / "training-sidecar/SHA256SUMS")
        == complete["training_sidecar_manifest_sha256"],
        "training-sidecar manifest binding drift",
    )
    require(
        sha256(components / "evaluation-package/SHA256SUMS")
        == complete["evaluation_package_manifest_sha256"],
        "evaluation-package manifest binding drift",
    )

    campaign_path = components / "campaign-manifest.json"
    context_path = components / "run-context.json"
    campaign_sha = sha256(campaign_path)
    context_sha = sha256(context_path)
    require(campaign_sha == complete["campaign_manifest_sha256"], "campaign binding drift")
    require(context_sha == complete["run_context_sha256"], "run-context binding drift")
    campaign = load_json(campaign_path, "engineering campaign")
    context = load_json(context_path, "run context")
    require(context.get("job_id") == job_id and context.get("run_id") == complete["run_id"],
            "component identity drift")
    require(context.get("campaign_manifest_sha256") == campaign_sha, "context/campaign drift")
    require(campaign.get("provenance", {}).get("assembler_driver_sha256")
            == cli.expected_assembler_sha256, "campaign binds another assembler")
    hardware = campaign.get("hardware")
    require(isinstance(hardware, dict), "engineering campaign hardware missing")
    closure = validate_input_closure(
        components, complete, campaign, cli.expected_input_closure_sha256,
        cli.expected_assembler_sha256, cli.expected_finalizer_sha256)
    require(closure["hashes"]["terminal_chain_sbatch_sha256"]
            == cli.expected_sbatch_sha256, "input-closure sbatch drift")
    require(sha256(bundle / "hopper/hopper.sh")
            == closure["hashes"]["hopper_wrapper_sha256"],
            "input-closure Hopper wrapper drift")
    require(complete["result_dir"] == expected_remote_components,
            "component COMPLETE remote result path drift")
    exported = parse_submission_exports(submission["sbatch_args"])
    expected_export_hashes = {
        "UED_BUNDLE_MANIFEST_SHA256": "bundle_manifest_sha256",
        "UED_UPSTREAM_COMMIT": "upstream_commit",
        "UED_UPSTREAM_TREE": "upstream_tree_git_sha1",
        "UED_UPSTREAM_BUNDLE_SHA256": "upstream_git_bundle_sha256",
        "UED_OVERLAY_MANIFEST_SHA256": "overlay_manifest_sha256",
        "UED_TERMINAL_CHAIN_SBATCH_SHA256": "terminal_chain_sbatch_sha256",
        "UED_FRONTIER_CONFIG_SHA256": "frontier_config_sha256",
        "UED_CONTRACT_SHA256": "overlay_contract_sha256",
        "UED_PROTOCOL_SHA256": "protocol_sha256",
        "UED_ANALYZER_SHA256": "analyzer_sha256",
        "UED_TRAINING_DRIVER_SHA256": "training_driver_sha256",
        "UED_EVALUATION_DRIVER_SHA256": "evaluation_driver_sha256",
        "UED_ASSEMBLER_SHA256": "assembler_sha256",
        "UED_ENV_LOCK_SHA256": "environment_lock_sha256",
        "UED_ENV_FREEZE_SHA256": "environment_freeze_sha256",
        "UED_ENV_MANIFEST_SHA256": "environment_manifest_sha256",
        "UED_IMPORT_SMOKE_MANIFEST_SHA256": "import_smoke_manifest_sha256",
        "UED_ONE_UPDATE_MANIFEST_SHA256": "one_update_manifest_sha256",
    }
    require(set(exported) == set(expected_export_hashes) | {
        "UED_BUNDLE_DIR", "UED_ENV_DIR", "UED_IMPORT_SMOKE_RESULT_DIR",
        "UED_ONE_UPDATE_RESULT_DIR",
    }, "submission export key set drift")
    for export_name, closure_name in expected_export_hashes.items():
        require(exported[export_name] == closure["hashes"][closure_name],
                f"submission export hash drift: {export_name}")
    require(
        re.fullmatch(rf"/scratch/{re.escape(remote_user)}/maxrl/bundles/ued_minimax/[0-9a-f]{{20}}",
                     exported["UED_BUNDLE_DIR"]) is not None
        and re.fullmatch(rf"/scratch/{re.escape(remote_user)}/envs/[A-Za-z0-9._-]+",
                         exported["UED_ENV_DIR"]) is not None
        and re.fullmatch(
            rf"/scratch/{re.escape(remote_user)}/maxrl/tests/ued-minimax-gpu-smoke/[0-9]+",
            exported["UED_IMPORT_SMOKE_RESULT_DIR"]) is not None
        and re.fullmatch(
            rf"/scratch/{re.escape(remote_user)}/maxrl/tests/ued-minimax-one-update/[0-9]+",
            exported["UED_ONE_UPDATE_RESULT_DIR"]) is not None,
        "submission export path drift",
    )
    _, _, nvidia_receipt = validate_phase_a_components(
        components, complete, campaign, context, campaign_sha, context_sha, job_id)
    validate_completion_marker(
        slurm_stdout, slurm_stderr, job_id,
        cli.expected_components_manifest_sha256, complete)

    require(
        hardware == {
            "partition": EXPECTED_PARTITION,
            "gpu_model": nvidia_receipt[1],
            "gpu_profile": EXPECTED_GPU_PROFILE,
            "gpu_count": 1,
            "n_devices": 1,
        },
        "terminal/campaign/nvidia-smi hardware drift",
    )
    max_rss, peak_gpu = resource_maxima(terminal["resource_rows"])
    require(max_rss > 0 and peak_gpu > 0,
            "terminal Slurm resource maxima are missing")
    phase_a_resource = load_json(components / "resource-accounting.json", "phase-A resource receipt")
    requested = phase_a_resource.get("requested")
    allocation = phase_a_resource.get("allocation")
    require(
        phase_a_resource.get("job_id") == job_id
        and phase_a_resource.get("external_accounting_authority") == "terminal_slurm_sacct"
        and phase_a_resource.get("terminal_sacct_included") is False
        and requested == {
            "partition": EXPECTED_PARTITION, "qos": "gpu",
            "gres": "gpu:1g.10gb:1", "cpus_per_task": 2,
            "memory": "15G", "walltime": "00:30:00",
        }
        and isinstance(allocation, dict)
        and allocation.get("SLURM_JOB_ID") == job_id
        and allocation.get("SLURM_JOB_PARTITION") == EXPECTED_PARTITION
        and allocation.get("SLURM_CPUS_PER_TASK") == "2"
        and allocation.get("SLURM_NTASKS") == "1"
        and allocation.get("SLURM_MEM_PER_NODE") == "15360"
        and allocation.get("SLURM_RESTART_COUNT") == "0",
        "phase-A resource receipt drift",
    )
    observed_peak = phase_a_resource.get("peak_gpu_memory_bytes_observed", 0)
    require(isinstance(observed_peak, int) and not isinstance(observed_peak, bool)
            and observed_peak >= 0, "invalid phase-A GPU memory observation")
    peak_gpu = max(peak_gpu, observed_peak)

    copy_sources = {
        "terminal-sacct.tsv": receipt_path,
        "submission-receipt.tsv": submission_path,
        "fetch-components.tsv": components_fetch_path,
        "fetch-slurm-stdout.tsv": stdout_fetch_path,
        "fetch-slurm-stderr.tsv": stderr_fetch_path,
        "fetch-submission-receipt.tsv": submission_fetch_path,
        "slurm-stdout.log": slurm_stdout,
        "slurm-stderr.log": slurm_stderr,
        "PHASE_B_PYVENV.cfg": venv_config,
    }
    expected_copy_hashes = {
        "terminal-sacct.tsv": terminal_receipt_sha,
        "submission-receipt.tsv": submission_receipt_sha,
        **fetch_receipt_file_hashes,
        "slurm-stdout.log": fetch_receipts["stdout"]["local_digest"],
        "slurm-stderr.log": fetch_receipts["stderr"]["local_digest"],
        "PHASE_B_PYVENV.cfg": cli.expected_python_venv_config_sha256,
    }

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output_parent))
    try:
        copied_components = temporary / "components"
        shutil.copytree(components, copied_components, symlinks=False)
        copied_entries = validate_manifest(
            copied_components, "SHA256SUMS", cli.expected_components_manifest_sha256)
        exact_file_tree(
            copied_components, {"SHA256SUMS", "COMPONENTS_COMPLETE.json"}, copied_entries)
        require(sha256(copied_components / "COMPONENTS_COMPLETE.json") == complete_sha,
                "component COMPLETE changed during archival copy")
        require(load_json(copied_components / "COMPONENTS_COMPLETE.json",
                          "copied component COMPLETE") == complete,
                "copied component COMPLETE semantic drift")
        for destination, source in copy_sources.items():
            copy_file(source, temporary / destination)
            require(sha256(temporary / destination) == expected_copy_hashes[destination],
                    f"validated source changed during archival copy: {destination}")
        freeze_path = temporary / "PHASE_B_PYTHON_FREEZE.txt"
        with freeze_path.open("x", encoding="utf-8") as stream:
            stream.write(python_freeze)
            stream.flush()
            os.fsync(stream.fileno())
        require(sha256(freeze_path) == cli.expected_python_freeze_sha256,
                "archived Phase-B package freeze digest drift")

        scheduler = {
            "schema": 1,
            "job_id": job_id,
            "state": terminal["state"],
            "exit_code": terminal["exit_code"],
            "partition": terminal["partition"],
            "gpu_model": hardware["gpu_model"],
            "gpu_profile": hardware["gpu_profile"],
            "gpu_count": hardware["gpu_count"],
            "elapsed_seconds": terminal["elapsed_seconds"],
            "max_rss_bytes": max_rss,
            "peak_gpu_memory_bytes": peak_gpu,
            "terminal_sacct_retrieved_utc": terminal["retrieved_utc"],
        }
        scheduler_path = temporary / "scheduler.json"
        write_json(scheduler_path, scheduler)
        scheduler_sha = sha256(scheduler_path)
        package_parent = temporary / "package"
        package_parent.mkdir()
        package = package_parent / complete["run_id"]
        command = [
            str(python_launcher), "-I", "-B", str(assembler),
            "--campaign-manifest", str(copied_components / "campaign-manifest.json"),
            "--expected-campaign-sha256", campaign_sha,
            "--run-context", str(copied_components / "run-context.json"),
            "--expected-run-context-sha256", context_sha,
            "--expected-assembler-sha256", cli.expected_assembler_sha256,
            "--training-output-dir", str(copied_components / "training-output"),
            "--training-sidecar-dir", str(copied_components / "training-sidecar"),
            "--evaluation-package-dir", str(copied_components / "evaluation-package"),
            "--command", str(copied_components / "command.txt"),
            "--scheduler", str(scheduler_path),
            "--stdout", str(temporary / "slurm-stdout.log"),
            "--stderr", str(temporary / "slurm-stderr.log"),
            "--output-dir", str(package),
            "--engineering-test-mode",
        ]
        environment = environment.copy()
        completed = subprocess.run(
            command, cwd=bundle, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
        )
        require(completed.returncode == 0,
                "engineering assembler refused:\n" + completed.stderr[-4000:])
        require("MATCHED_RUN_ASSEMBLY_COMPLETE" in completed.stdout,
                "engineering assembler omitted completion marker")
        package_manifest_sha = sha256(package / "SHA256SUMS")
        package_entries = validate_manifest(package, "SHA256SUMS", package_manifest_sha)
        exact_file_tree(package, {"SHA256SUMS", "COMPLETE"}, package_entries)
        validation_command = [
            str(python_launcher), "-I", "-B", str(assembler),
            "--engineering-test-mode", "--validate-only",
            "--campaign-manifest", str(copied_components / "campaign-manifest.json"),
            "--expected-campaign-sha256", campaign_sha,
            "--output-dir", str(package),
            "--expected-package-sha256sums-sha256", package_manifest_sha,
        ]
        validated = subprocess.run(
            validation_command, cwd=bundle, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
        )
        require(validated.returncode == 0,
                "read-only engineering validation refused:\n" + validated.stderr[-4000:])
        require("MATCHED_RUN_VALIDATION_COMPLETE" in validated.stdout,
                "read-only engineering validation omitted completion marker")
        # Isolated subprocesses must not create bytecode or otherwise mutate
        # the exact bundle after its initial self/manifest attestation.
        require(
            validate_manifest(bundle, "SHA256SUMS", cli.expected_bundle_manifest_sha256)
            == bundle_entries,
            "bundle manifest entries changed during Phase-B assembly",
        )
        exact_file_tree(bundle, {"SHA256SUMS"}, bundle_entries)
        require(
            validate_manifest(
                copied_components, "SHA256SUMS",
                cli.expected_components_manifest_sha256) == copied_entries,
            "copied component manifest changed during Phase-B assembly",
        )
        exact_file_tree(
            copied_components, {"SHA256SUMS", "COMPONENTS_COMPLETE.json"},
            copied_entries)
        require(
            sha256(copied_components / "COMPONENTS_COMPLETE.json") == complete_sha
            and load_json(copied_components / "COMPONENTS_COMPLETE.json",
                          "post-assembly copied component COMPLETE") == complete,
            "copied component COMPLETE changed during Phase-B assembly",
        )
        for destination, expected in expected_copy_hashes.items():
            require(sha256(temporary / destination) == expected,
                    f"archived input changed during Phase-B assembly: {destination}")
        require(
            sha256(freeze_path) == cli.expected_python_freeze_sha256
            and sha256(scheduler_path) == scheduler_sha,
            "Phase-B environment or scheduler receipt changed during assembly",
        )
        run_manifest = load_json(package / "run-manifest.json", "assembled run manifest")
        require(
            run_manifest.get("paper_evidence") is False
            and run_manifest.get("analyzer_eligible") is False
            and run_manifest.get("endpoint_class") == "bounded_engineering_test",
            "assembled engineering eligibility drift",
        )
        finalization = {
            "schema": 1,
            "status": "complete",
            "paper_evidence": False,
            "analyzer_eligible": False,
            "production_analyzer_invoked": False,
            "assembler_validate_only_passed": True,
            "phase": "post_terminal_local_engineering_assembly",
            "job_id": job_id,
            "run_id": complete["run_id"],
            "bundle_manifest_sha256": cli.expected_bundle_manifest_sha256,
            "components_manifest_sha256": cli.expected_components_manifest_sha256,
            "input_closure_sha256": cli.expected_input_closure_sha256,
            "terminal_receipt_sha256": sha256(temporary / "terminal-sacct.tsv"),
            "submission_receipt_sha256": sha256(temporary / "submission-receipt.tsv"),
            "post_terminal_fetch_receipts": {
                label: {
                    "receipt_sha256": sha256(temporary / destination),
                    "fetch_started_utc": fetch_receipts[label]["fetch_started_utc"],
                    "fetch_started_epoch": int(
                        fetch_receipts[label]["fetch_started_epoch"]),
                    "retrieved_utc": fetch_receipts[label]["retrieved_utc"],
                    "retrieved_epoch": int(fetch_receipts[label]["retrieved_epoch"]),
                    "remote_path": fetch_receipts[label]["remote_path"],
                    "payload_digest": fetch_receipts[label]["local_digest"],
                }
                for label, destination in {
                    "components": "fetch-components.tsv",
                    "stdout": "fetch-slurm-stdout.tsv",
                    "stderr": "fetch-slurm-stderr.tsv",
                    "submission": "fetch-submission-receipt.tsv",
                }.items()
            },
            "terminal_chain_sbatch_sha256": cli.expected_sbatch_sha256,
            "assembler_sha256": cli.expected_assembler_sha256,
            "finalizer_sha256": cli.expected_finalizer_sha256,
            "hopper_wrapper_sha256": closure["hashes"]["hopper_wrapper_sha256"],
            "phase_b_python": {
                "launcher_path": str(python_launcher),
                "resolved_binary_path": str(python_binary),
                "venv_prefix": str(python_prefix),
                "venv_config_sha256": cli.expected_python_venv_config_sha256,
                "sha256": cli.expected_python_sha256,
                "version": platform.python_version(),
                "package_freeze_sha256": cli.expected_python_freeze_sha256,
                "pip_check_passed": True,
                "user_site_enabled": False,
                "isolated": True,
                "ignore_environment": True,
                "no_user_site": True,
                "dont_write_bytecode": True,
                "optimize": 0,
            },
            "assembled_package_manifest_sha256": package_manifest_sha,
            "terminal_scheduler": scheduler,
            "finalized_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        write_json(temporary / "FINALIZATION.json", finalization)
        outer_manifest_sha = build_outer_manifest(temporary)
        write_json(temporary / "COMPLETE", {
            "schema": 1,
            "status": "complete",
            "paper_evidence": False,
            "analyzer_eligible": False,
            "job_id": job_id,
            "run_id": complete["run_id"],
            "bundle_manifest_sha256": cli.expected_bundle_manifest_sha256,
            "components_manifest_sha256": cli.expected_components_manifest_sha256,
            "input_closure_sha256": cli.expected_input_closure_sha256,
            "sha256sums_sha256": outer_manifest_sha,
            "assembled_package_manifest_sha256": package_manifest_sha,
        })
        outer_entries = validate_manifest(temporary, "SHA256SUMS", outer_manifest_sha)
        exact_file_tree(temporary, {"SHA256SUMS", "COMPLETE"}, outer_entries)
        os.replace(temporary, output)
        directory_fd = os.open(output_parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        published_entries = validate_manifest(output, "SHA256SUMS", outer_manifest_sha)
        exact_file_tree(output, {"SHA256SUMS", "COMPLETE"}, published_entries)
        return {
            "job_id": job_id,
            "run_id": complete["run_id"],
            "output": str(output),
            "outer_manifest_sha256": outer_manifest_sha,
            "package_manifest_sha256": package_manifest_sha,
        }
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--expected-bundle-manifest-sha256", required=True)
    parser.add_argument("--components-dir", type=Path, required=True)
    parser.add_argument("--expected-components-manifest-sha256", required=True)
    parser.add_argument("--expected-input-closure-sha256", required=True)
    parser.add_argument("--expected-sbatch-sha256", required=True)
    parser.add_argument("--terminal-receipt", type=Path, required=True)
    parser.add_argument("--submission-receipt", type=Path, required=True)
    parser.add_argument("--submission-fetch-receipt", type=Path, required=True)
    parser.add_argument("--components-fetch-receipt", type=Path, required=True)
    parser.add_argument("--slurm-stdout", type=Path, required=True)
    parser.add_argument("--stdout-fetch-receipt", type=Path, required=True)
    parser.add_argument("--slurm-stderr", type=Path, required=True)
    parser.add_argument("--stderr-fetch-receipt", type=Path, required=True)
    parser.add_argument("--expected-assembler-sha256", required=True)
    parser.add_argument("--expected-finalizer-sha256", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--expected-python-sha256", required=True)
    parser.add_argument("--expected-python-version", required=True)
    parser.add_argument("--expected-python-freeze-sha256", required=True)
    parser.add_argument("--expected-python-venv-config-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    os.umask(0o077)
    try:
        result = finalize(parse_cli(argv))
    except (FinalizationError, OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        print(f"UED_TERMINAL_FINALIZATION_REFUSED: {exc}", file=sys.stderr)
        return 1
    print(
        "UED_TERMINAL_FINALIZATION_COMPLETE "
        f"job={result['job_id']} run_id={result['run_id']} "
        f"outer={result['outer_manifest_sha256']} package={result['package_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
