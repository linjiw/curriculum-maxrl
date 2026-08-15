#!/usr/bin/env python3
"""Locally close one v4 engineering package after clean terminal accounting.

This program performs no network or scheduler operation.  It consumes an
already captured terminal receipt, exact submission receipt, and terminal-
gated fetch receipt.  Evaluation values remain sealed inside their closed
component; the production analyzer is neither imported nor invoked.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo


HASH_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
SLURM_TIME_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$")
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
EXPECTED_EXPORTS = {
    "UED_BUNDLE_DIR", "UED_BUNDLE_MANIFEST_SHA256", "UED_UPSTREAM_COMMIT",
    "UED_UPSTREAM_TREE", "UED_UPSTREAM_BUNDLE_SHA256",
    "UED_OVERLAY_MANIFEST_SHA256", "UED_ENV_DIR", "UED_ENV_LOCK_SHA256",
    "UED_ENV_FREEZE_SHA256", "UED_ENV_MANIFEST_SHA256", "UED_SBATCH_SHA256",
    "UED_IMPORT_SMOKE_RESULT_DIR", "UED_IMPORT_SMOKE_MANIFEST_SHA256",
    "UED_ONE_UPDATE_RESULT_DIR", "UED_ONE_UPDATE_MANIFEST_SHA256", "UED_ARM",
    "UED_CONFIG_SHA256", "UED_CONTRACT_SHA256", "UED_PROTOCOL_SHA256",
    "UED_PHASE_A_DRIVER_SHA256", "UED_TRAINING_DRIVER_SHA256",
    "UED_EVALUATION_DRIVER_SHA256", "UED_ASSEMBLER_SHA256",
    "UED_FINALIZER_SHA256",
}
PACKAGE_TOP_LEVEL = {
    "INPUT_CLOSURE.json", "components-COMPLETE.json",
    "components-SHA256SUMS", "campaign-manifest.json", "run-context.json",
    "scheduler.json", "run-manifest.json", "training-plr-replay-snapshot.json",
    "training-output", "training-sidecar", "evaluation-package",
    "evaluation-integrity.json", "SHA256SUMS", "COMPLETE",
    "phase-b-receipts",
}


class FinalizationError(RuntimeError):
    """Raised when the post-terminal v4 closure is not exact."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizationError(message)


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate {label} key: {key}")
            result[key] = value
        return result

    require(path.is_file() and not path.is_symlink(), f"unsafe {label}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def safe_existing(path: Path, *, directory: bool, label: str) -> Path:
    require(path.is_absolute() and ".." not in path.parts, f"{label} must be absolute")
    resolved = path.resolve(strict=True)
    require(resolved == path, f"{label} is noncanonical or contains a symlink")
    require((path.is_dir() if directory else path.is_file()) and not path.is_symlink(), f"unsafe {label}")
    return path


def safe_new_output(path: Path, components: Path) -> Path:
    require(path.is_absolute() and ".." not in path.parts, "output must be canonical absolute")
    require(path.name not in {"", ".", ".."}, "unsafe output basename")
    require(not path.exists() and not path.is_symlink(), "output exists")
    parent = safe_existing(path.parent, directory=True, label="output parent")
    canonical = parent / path.name
    require(canonical == path, "output path is noncanonical")
    require(
        not canonical.is_relative_to(components)
        and not components.is_relative_to(canonical),
        "output/components overlap",
    )
    return canonical


def parse_multivalue_tsv(path: Path, expected: set[str], label: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split("\t", 1)
        require(len(fields) == 2 and fields[0] and fields[1] != "", f"bad {label} line {number}")
        values.setdefault(fields[0], []).append(fields[1])
    require(set(values) == expected, f"{label} keys drift")
    return values


def parse_terminal(path: Path, job_id: str, *, local_test_mode: bool) -> dict[str, Any]:
    values = parse_multivalue_tsv(path, {
        "terminal_receipt_schema", "retrieved_utc", "retrieved_epoch",
        "terminal_end_epoch", "terminal_header", "terminal_row",
        "resource_header", "resource_row",
    }, "terminal receipt")
    for key in set(values) - {"resource_row"}:
        require(len(values[key]) == 1, f"terminal receipt cardinality drift: {key}")
    require(values["terminal_receipt_schema"] == ["2"], "terminal receipt schema drift")
    require(values["terminal_header"] == [TERMINAL_HEADER], "terminal header drift")
    require(values["resource_header"] == [RESOURCE_HEADER], "resource header drift")
    retrieved_utc = values["retrieved_utc"][0]
    require(UTC_RE.fullmatch(retrieved_utc) is not None, "terminal retrieval UTC drift")
    require(values["retrieved_epoch"][0].isdigit() and values["terminal_end_epoch"][0].isdigit(), "terminal epoch drift")
    retrieved_epoch = int(values["retrieved_epoch"][0])
    end_epoch = int(values["terminal_end_epoch"][0])
    require(
        retrieved_epoch == int(datetime.strptime(retrieved_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
        and retrieved_epoch >= end_epoch,
        "terminal retrieval time drift",
    )
    row = values["terminal_row"][0].split("|")
    require(len(row) == 20 and row[0] == job_id, "terminal allocation identity drift")
    require(row[1] == "ued-v4-terminal", "terminal job name drift")
    require(row[3] == "COMPLETED" and row[4] == "0:0", "job not cleanly complete")
    require(
        row[5].isdigit()
        and (0 <= int(row[5]) <= 2700 if local_test_mode else 0 < int(row[5]) <= 2700),
        "terminal elapsed drift",
    )
    require(row[6] == "2" and row[7] in {"15G", "15Gn", "15Gc"}, "terminal host resource drift")
    require(row[14] == "45" and row[15] == "0", "terminal time/restart drift")
    if local_test_mode:
        require(row[2] == "local" and row[13] == "local", "local terminal queue drift")
    else:
        require(row[2] == "gpuq" and row[13] == "gpu", "terminal queue drift")
    epochs: list[int] = []
    for value in row[9:12]:
        require(SLURM_TIME_RE.fullmatch(value) is not None, "Slurm timestamp drift")
        if local_test_mode:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        else:
            naive = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
            zone = ZoneInfo("America/New_York")
            parsed = naive.replace(tzinfo=zone, fold=0)
            alternate = naive.replace(tzinfo=zone, fold=1)
            require(parsed.utcoffset() == alternate.utcoffset(), "ambiguous Slurm time")
        epochs.append(int(parsed.timestamp()))
    require(epochs[0] <= epochs[1] <= epochs[2] == end_epoch, "Slurm time ordering drift")
    require(epochs[2] - epochs[1] == int(row[5]), "Slurm elapsed binding drift")
    tres: dict[str, str] = {}
    for encoded in row[12].split(","):
        fields = encoded.split("=", 1)
        require(len(fields) == 2 and all(fields) and fields[0] not in tres, "AllocTRES drift")
        tres[fields[0]] = fields[1]
    require(tres.get("cpu") == "2" and tres.get("gres/gpu") == "1", "allocated GPU/CPU drift")
    resources: list[dict[str, str]] = []
    seen: set[str] = set()
    for encoded in values["resource_row"]:
        fields = encoded.split("|")
        require(len(fields) == 3, "resource row width drift")
        require(fields[0] == job_id or fields[0].startswith(job_id + "."), "foreign resource row")
        require(fields[0] not in seen, "duplicate resource row")
        seen.add(fields[0])
        resources.append({"job_id": fields[0], "max_rss": fields[1], "tres_usage_in_max": fields[2]})
    require(resources, "resource rows missing")
    nonempty_sizes = [
        parsed
        for parsed in (slurm_size_bytes(resource["max_rss"]) for resource in resources)
        if parsed is not None
    ]
    require(
        nonempty_sizes and any(resource["tres_usage_in_max"] for resource in resources),
        "resource usage fields missing",
    )
    if not local_test_mode:
        require(
            f"{job_id}.batch" in seen and f"{job_id}.extern" in seen,
            "batch/extern resource rows missing",
        )
    return {
        "job_id": row[0], "job_name": row[1], "partition": row[2],
        "state": row[3], "exit_code": row[4], "elapsed_raw": int(row[5]),
        "cpus": int(row[6]), "memory": "15G", "qos": row[13],
        "restarts": int(row[15]), "stdout_path": row[17],
        "stderr_path": row[18], "submit_line": row[19],
        "submit_epoch": epochs[0],
        "terminal_end_epoch": end_epoch, "retrieved_epoch": retrieved_epoch,
        "resources": resources, "max_rss_bytes": max(nonempty_sizes),
    }


def parse_exports(argument: str) -> dict[str, str]:
    require(argument.startswith("--export="), "submission export argument missing")
    payload = argument.removeprefix("--export=")
    require(payload not in {"", "ALL", "NONE"}, "submission export sentinel forbidden")
    result: dict[str, str] = {}
    for encoded in payload.split(","):
        fields = encoded.split("=", 1)
        require(len(fields) == 2 and all(fields), "malformed submission export")
        key, value = fields
        require(re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is not None, "unsafe export key")
        require(key not in result, "duplicate submission export")
        result[key] = value
    require(set(result) == EXPECTED_EXPORTS, "submission export allowlist drift")
    return result


def slurm_size_bytes(value: str) -> int | None:
    if value == "":
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)", value)
    require(match is not None, f"malformed Slurm size: {value}")
    amount = float(match.group(1))
    exponent = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[match.group(2)]
    result = int(amount * (1024 ** exponent))
    require(result >= 0, "negative Slurm size")
    return result


def parse_submission(
    path: Path, job_id: str, sbatch_sha: str, terminal_submit_line: str,
    *, local_test_mode: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == 2 and lines[0] == SUBMISSION_HEADER, "submission receipt header/cardinality drift")
    fields = lines[1].split("\t")
    keys = SUBMISSION_HEADER.split("\t")
    require(len(fields) == len(keys) and all(fields), "submission receipt row drift")
    receipt = dict(zip(keys, fields))
    require(receipt["job_id"] == job_id and UTC_RE.fullmatch(receipt["utc"]) is not None, "submission identity/time drift")
    require(receipt["local_sha256"] == receipt["remote_sha256"] == sbatch_sha, "submitted sbatch drift")
    local_script = Path(receipt["local_script"])
    require(
        local_script.is_absolute()
        and local_script.is_file()
        and not local_script.is_symlink()
        and sha256(local_script) == sbatch_sha,
        "submission local sbatch bytes drift",
    )
    exports = parse_exports(receipt["sbatch_args"])
    forbidden = (
        "--partition", "--qos", "--gres", "--gpus", "--nodes", "--ntasks",
        "--cpus-per-task", "--mem", "--time", "--requeue", "--job-name",
        "--output", "--error", "--chdir", "--array",
    )
    require(not any(token in receipt["sbatch_args"] for token in forbidden), "submission resource/identity override")
    command = shlex.split(terminal_submit_line)
    require(command == ["sbatch", "--parsable", receipt["sbatch_args"], receipt["remote_script"]], "terminal SubmitLine/submission receipt drift")
    if not local_test_mode:
        require(re.fullmatch(r"/scratch/[A-Za-z0-9._-]+/maxrl/sbatch/ued_minimax_v4_terminal_chain_smoke-[0-9a-f]{16}-[0-9]{8}T[0-9]{6}Z-[0-9]+\.sbatch", receipt["remote_script"]) is not None, "remote sbatch path drift")
        require(
            re.fullmatch(rf"/scratch/[A-Za-z0-9._-]+/maxrl/receipts/job-{job_id}-[0-9]{{8}}T[0-9]{{6}}Z\.tsv", receipt["remote_receipt"])
            is not None,
            "remote submission receipt path drift",
        )
    return receipt, exports


def tree_digest(root: Path) -> str:
    lines: list[str] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix()):
        require(not path.is_symlink(), f"symlink in fetched tree: {path}")
        lines.append(f"{sha256(path)}  ./{path.relative_to(root).as_posix()}\n")
    require(lines, "empty fetched tree")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def parse_fetch(
    path: Path, components: Path, terminal: Mapping[str, Any], terminal_sha: str,
    *, arm: str, local_test_mode: bool,
) -> dict[str, str]:
    raw = parse_multivalue_tsv(path, {
        "fetch_receipt_schema", "fetch_started_utc", "fetch_started_epoch",
        "retrieved_utc", "retrieved_epoch", "terminal_end_epoch",
        "terminal_receipt_sha256", "remote_path", "remote_type", "remote_digest",
        "manifest_verified", "local_path", "local_digest",
    }, "fetch receipt")
    require(all(len(values) == 1 for values in raw.values()), "fetch receipt cardinality drift")
    receipt = {key: values[0] for key, values in raw.items()}
    require(receipt["fetch_receipt_schema"] == "2" and receipt["remote_type"] == "dir", "fetch schema/type drift")
    for utc_key, epoch_key in (
        ("fetch_started_utc", "fetch_started_epoch"),
        ("retrieved_utc", "retrieved_epoch"),
    ):
        require(UTC_RE.fullmatch(receipt[utc_key]) is not None, f"fetch {utc_key} drift")
        require(
            int(datetime.strptime(receipt[utc_key], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
            == int(receipt[epoch_key]),
            f"fetch {utc_key}/epoch mismatch",
        )
    require(receipt["manifest_verified"] == "1", "fetched component manifest was not verified")
    require(receipt["terminal_receipt_sha256"] == terminal_sha, "fetch terminal binding drift")
    require(receipt["local_path"] == str(components), "fetch local path drift")
    require(receipt["terminal_end_epoch"].isdigit() and receipt["fetch_started_epoch"].isdigit() and receipt["retrieved_epoch"].isdigit(), "fetch epoch drift")
    require(
        int(receipt["terminal_end_epoch"]) == terminal["terminal_end_epoch"]
        and int(receipt["fetch_started_epoch"]) >= terminal["retrieved_epoch"]
        and int(receipt["retrieved_epoch"]) >= int(receipt["fetch_started_epoch"]),
        "fetch occurred before terminal receipt",
    )
    if not local_test_mode:
        require(re.fullmatch(rf"/scratch/[A-Za-z0-9._-]+/maxrl/tests/ued-minimax-v4-terminal-components/{terminal['job_id']}-{arm}", receipt["remote_path"]) is not None, "fetch remote path drift")
    digest = tree_digest(components)
    require(receipt["remote_digest"] == receipt["local_digest"] == digest, "fetch tree digest drift")
    return receipt


def validate_export_bindings(exports: Mapping[str, str], closure: Mapping[str, Any], arm: str) -> None:
    mapping = {
        "UED_BUNDLE_MANIFEST_SHA256": "bundle_manifest_sha256",
        "UED_OVERLAY_MANIFEST_SHA256": "overlay_manifest_sha256",
        "UED_ENV_MANIFEST_SHA256": "environment_manifest_sha256",
        "UED_SBATCH_SHA256": "sbatch_sha256",
        "UED_CONFIG_SHA256": "config_sha256",
        "UED_PROTOCOL_SHA256": "protocol_sha256",
        "UED_PHASE_A_DRIVER_SHA256": "phase_a_driver_sha256",
        "UED_TRAINING_DRIVER_SHA256": "training_driver_sha256",
        "UED_EVALUATION_DRIVER_SHA256": "evaluation_driver_sha256",
        "UED_ASSEMBLER_SHA256": "assembler_sha256",
        "UED_FINALIZER_SHA256": "finalizer_sha256",
    }
    for exported, field in mapping.items():
        require(exports[exported] == closure.get(field), f"submission/input closure drift: {exported}")
    require(exports["UED_ARM"] == closure.get("arm") == arm, "submission arm drift")
    require(exports["UED_CONTRACT_SHA256"] == "3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b", "submission contract drift")
    require(exports["UED_UPSTREAM_COMMIT"] == closure.get("source_commit"), "submission commit drift")
    require(exports["UED_UPSTREAM_TREE"] == closure.get("source_tree"), "submission tree drift")
    prereq = closure.get("prerequisites")
    require(isinstance(prereq, dict), "input closure prerequisites missing")
    require(exports["UED_IMPORT_SMOKE_RESULT_DIR"] == prereq["import"]["result_dir"], "submission import path drift")
    require(exports["UED_IMPORT_SMOKE_MANIFEST_SHA256"] == prereq["import"]["manifest_sha256"], "submission import hash drift")
    require(exports["UED_ONE_UPDATE_RESULT_DIR"] == prereq["one_update"]["result_dir"], "submission one-update path drift")
    require(exports["UED_ONE_UPDATE_MANIFEST_SHA256"] == prereq["one_update"]["manifest_sha256"], "submission one-update hash drift")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def create_phase_b_receipt_archive(
    root: Path,
    terminal: Path,
    submission: Path,
    fetch: Path,
) -> str:
    require(not root.exists() and not root.is_symlink(), "receipt archive exists")
    root.mkdir()
    sources = {
        "terminal.tsv": terminal,
        "submission.tsv": submission,
        "fetch.tsv": fetch,
    }
    for name, source in sources.items():
        shutil.copy2(source, root / name, follow_symlinks=False)
    payloads = sorted(sources)
    manifest = root / "SHA256SUMS"
    manifest.write_text(
        "".join(f"{sha256(root / name)}  {name}\n" for name in payloads),
        encoding="utf-8",
    )
    manifest_sha = sha256(manifest)
    atomic_json(
        root / "COMPLETE",
        {
            "schema": 1,
            "status": "complete",
            "sha256sums_sha256": manifest_sha,
            "file_count": len(payloads),
        },
    )
    return manifest_sha


def validate_published_package(root: Path, arm: str) -> str:
    """Independently validate the assembler output without importing it."""
    require(root.is_dir() and not root.is_symlink(), "published package missing")
    require({entry.name for entry in root.iterdir()} == PACKAGE_TOP_LEVEL, "published top-level closure drift")
    manifest = root / "SHA256SUMS"
    complete_path = root / "COMPLETE"
    require(manifest.is_file() and not manifest.is_symlink(), "published manifest missing")
    require(complete_path.is_file() and not complete_path.is_symlink(), "published completion missing")
    listed: dict[str, str] = {}
    for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw)
        require(match is not None, f"unsafe published manifest line {number}")
        digest, text_path = match.groups()
        relative = PurePosixPath(text_path)
        require(
            text_path not in ("", ".")
            and not relative.is_absolute()
            and all(part not in ("", ".", "..") for part in relative.parts),
            "unsafe published manifest path",
        )
        canonical = relative.as_posix()
        require(canonical not in listed, "duplicate published manifest path")
        listed[canonical] = digest
    actual: set[str] = set()
    for target in root.rglob("*"):
        require(not target.is_symlink(), "symlink in published package")
        if target.is_file():
            relative = target.relative_to(root).as_posix()
            if relative not in {"SHA256SUMS", "COMPLETE"}:
                actual.add(relative)
        else:
            require(target.is_dir(), "non-file published package entry")
    require(set(listed) == actual, "published manifest file closure drift")
    for relative, digest in listed.items():
        require(sha256(root / relative) == digest, f"published payload drift: {relative}")
    manifest_sha = sha256(manifest)
    complete = load_json(complete_path, "published completion")
    require(
        set(complete)
        == {
            "schema", "status", "paper_evidence", "analyzer_eligible",
            "production_analyzer_invoked", "endpoint_class", "run_id", "arm",
            "sha256sums_sha256", "file_count",
        },
        "published completion keys drift",
    )
    require(
        complete["schema"] == 1
        and complete["status"] == "complete"
        and complete["paper_evidence"] is False
        and complete["analyzer_eligible"] is False
        and complete["production_analyzer_invoked"] is False
        and complete["endpoint_class"] == "bounded_engineering_terminal_package_v4"
        and complete["arm"] == arm
        and complete["sha256sums_sha256"] == manifest_sha
        and complete["file_count"] == len(listed),
        "published completion semantics drift",
    )
    run_manifest = load_json(root / "run-manifest.json", "published run manifest")
    require(
        run_manifest.get("run_id") == complete["run_id"]
        and run_manifest.get("arm") == complete["arm"],
        "published completion/run identity drift",
    )
    require(
        run_manifest.get("paper_evidence") is False
        and run_manifest.get("analyzer_eligible") is False
        and run_manifest.get("production_analyzer_invoked") is False
        and run_manifest.get("performance_values_inspected") is False,
        "published run authorization drift",
    )
    receipt_archive = root / "phase-b-receipts"
    require(
        receipt_archive.is_dir()
        and not receipt_archive.is_symlink()
        and {
            entry.name for entry in receipt_archive.iterdir()
        }
        == {"terminal.tsv", "submission.tsv", "fetch.tsv", "SHA256SUMS", "COMPLETE"},
        "published Phase-B receipt archive drift",
    )
    receipt_manifest_sha = sha256(receipt_archive / "SHA256SUMS")
    archived_listed: dict[str, str] = {}
    for number, raw in enumerate(
        (receipt_archive / "SHA256SUMS").read_text(encoding="utf-8").splitlines(), 1
    ):
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", raw)
        require(match is not None, f"unsafe Phase-B archive manifest line {number}")
        digest, name = match.groups()
        require(name not in archived_listed, "duplicate Phase-B archive manifest path")
        archived_listed[name] = digest
    require(
        set(archived_listed) == {"terminal.tsv", "submission.tsv", "fetch.tsv"}
        and all(
            sha256(receipt_archive / name) == expected
            for name, expected in archived_listed.items()
        ),
        "published Phase-B archive manifest drift",
    )
    archived_complete = load_json(
        receipt_archive / "COMPLETE", "published Phase-B archive completion"
    )
    require(
        archived_complete
        == {
            "schema": 1,
            "status": "complete",
            "sha256sums_sha256": receipt_manifest_sha,
            "file_count": 3,
        },
        "published Phase-B archive completion drift",
    )
    scheduler = load_json(root / "scheduler.json", "published scheduler")
    require(
        run_manifest.get("phase_b_receipts_manifest_sha256")
        == scheduler.get("phase_b_receipts_manifest_sha256")
        == receipt_manifest_sha,
        "published Phase-B receipt archive binding drift",
    )
    require(
        scheduler.get("terminal_sacct_sha256") == sha256(receipt_archive / "terminal.tsv")
        and scheduler.get("submission_receipt_sha256") == sha256(receipt_archive / "submission.tsv")
        and scheduler.get("fetch_receipt_sha256") == sha256(receipt_archive / "fetch.tsv"),
        "published Phase-B receipt bytes drift",
    )
    return manifest_sha


def run(cli: argparse.Namespace) -> tuple[Path, str]:
    finalizer = Path(__file__).resolve()
    require(HASH_RE.fullmatch(cli.expected_finalizer_sha256 or "") is not None, "bad finalizer hash")
    require(sha256(finalizer) == cli.expected_finalizer_sha256, "finalizer self hash drift")
    components = safe_existing(cli.components_dir, directory=True, label="components")
    protocol = safe_existing(cli.protocol, directory=False, label="protocol")
    assembler = safe_existing(cli.assembler, directory=False, label="assembler")
    terminal_path = safe_existing(cli.terminal_receipt, directory=False, label="terminal receipt")
    submission_path = safe_existing(cli.submission_receipt, directory=False, label="submission receipt")
    fetch_path = safe_existing(cli.fetch_receipt, directory=False, label="fetch receipt")
    require(cli.arm in {"frontier", "maxmc"}, "invalid arm")
    expected_job = "local-test" if cli.local_test_mode else cli.job_id
    require(cli.job_id == expected_job and (cli.local_test_mode or cli.job_id.isdigit()), "job identity drift")
    require(HASH_RE.fullmatch(cli.expected_components_manifest_sha256 or "") is not None, "bad component hash")
    require(HASH_RE.fullmatch(cli.expected_assembler_sha256 or "") is not None, "bad assembler hash")
    require(HASH_RE.fullmatch(cli.expected_sbatch_sha256 or "") is not None, "bad sbatch hash")
    require(sha256(assembler) == cli.expected_assembler_sha256, "assembler hash drift")
    require(cli.python.is_absolute() and cli.python.is_file() and os.access(cli.python, os.X_OK), "unsafe Python executable")
    require(
        cli.python.resolve() == Path(os.sys.executable).resolve(),
        "Python executable identity drift",
    )
    expected_python = "3.10.19" if cli.local_test_mode else "3.10.20"
    require(platform.python_version() == expected_python, "Phase-B Python version drift")
    cli.output_dir = safe_new_output(cli.output_dir, components)
    complete = load_json(components / "COMPONENTS_COMPLETE.json", "component completion")
    closure = load_json(components / "INPUT_CLOSURE.json", "input closure")
    require(complete.get("arm") == cli.arm and complete.get("job_id") == cli.job_id, "component arm/job drift")
    require(complete.get("paper_evidence") is False and complete.get("analyzer_eligible") is False, "component evidence drift")
    require(complete.get("actual_student_updates") == 1 and complete.get("phase_b_required") is True, "component budget/phase drift")
    require(closure.get("cost100_implemented") is False and closure.get("production_authorized") is False, "cost/production closure drift")
    require(closure.get("from_last_checkpoint") is False and closure.get("no_requeue") is True and closure.get("attempt") == 1, "resume/requeue closure drift")
    require(closure.get("finalizer_sha256") == cli.expected_finalizer_sha256, "closure finalizer drift")
    terminal_sha = sha256(terminal_path)
    terminal = parse_terminal(terminal_path, cli.job_id, local_test_mode=cli.local_test_mode)
    submission, exports = parse_submission(
        submission_path, cli.job_id, cli.expected_sbatch_sha256,
        terminal["submit_line"], local_test_mode=cli.local_test_mode,
    )
    submission_epoch = int(
        datetime.strptime(submission["utc"], "%Y-%m-%dT%H:%M:%SZ")
        .replace(tzinfo=timezone.utc).timestamp()
    )
    require(
        terminal["submit_epoch"] <= submission_epoch <= terminal["retrieved_epoch"],
        "submission receipt chronology drift",
    )
    require(
        submission["output_path"] == terminal["stdout_path"]
        and terminal["stdout_path"] == terminal["stderr_path"],
        "submission/terminal output path drift",
    )
    validate_export_bindings(exports, closure, cli.arm)
    parse_fetch(
        fetch_path, components, terminal, terminal_sha,
        arm=cli.arm, local_test_mode=cli.local_test_mode,
    )
    scheduler = {
        "schema": 1, "job_id": cli.job_id, "job_name": terminal["job_name"],
        "arm": cli.arm, "state": terminal["state"], "exit_code": terminal["exit_code"],
        "partition": terminal["partition"], "qos": terminal["qos"],
        "gpu_profile": "local-cpu" if cli.local_test_mode else "1g.10gb",
        "gpu_count": 1, "cpus": terminal["cpus"], "memory": terminal["memory"],
        "elapsed_raw": terminal["elapsed_raw"],
        "max_rss_bytes": terminal["max_rss_bytes"],
        "resource_rows": terminal["resources"],
        "terminal_sacct_sha256": terminal_sha,
        "submission_receipt_sha256": sha256(submission_path),
        "fetch_receipt_sha256": sha256(fetch_path),
        "terminal_sacct_included": True, "fetched_after_terminal": True,
        "restarts": terminal["restarts"], "array_job": False,
        "phase_b_mode": "local_fixture" if cli.local_test_mode else "post_terminal_local",
        "components_manifest_sha256": cli.expected_components_manifest_sha256,
        "bundle_manifest_sha256": closure["bundle_manifest_sha256"],
        "sbatch_sha256": cli.expected_sbatch_sha256,
        "submit_line_sha256": hashlib.sha256(terminal["submit_line"].encode("utf-8")).hexdigest(),
        "submission_export_mode": "explicit_assignments_no_all_or_none",
    }
    with tempfile.TemporaryDirectory(prefix="ued-v4-finalize-") as raw:
        phase_b_receipts = Path(raw) / "phase-b-receipts"
        phase_b_receipts_sha = create_phase_b_receipt_archive(
            phase_b_receipts, terminal_path, submission_path, fetch_path
        )
        scheduler["phase_b_receipts_manifest_sha256"] = phase_b_receipts_sha
        scheduler_path = Path(raw) / "scheduler.json"
        atomic_json(scheduler_path, scheduler)
        launcher = (
            "import runpy,sys; from pathlib import Path; script=sys.argv.pop(1); "
            "sys.path.insert(0,str(Path(script).parent)); sys.argv[0]=script; "
            "runpy.run_path(script,run_name='__main__')"
        )
        command = [
            str(cli.python), "-I", "-B", "-c", launcher, str(assembler),
            "--components-dir", str(components), "--protocol", str(protocol),
            "--expected-components-manifest-sha256", cli.expected_components_manifest_sha256,
            "--scheduler-receipt", str(scheduler_path),
            "--expected-scheduler-receipt-sha256", sha256(scheduler_path),
            "--expected-assembler-sha256", cli.expected_assembler_sha256,
            "--phase-b-receipts-dir", str(phase_b_receipts),
            "--expected-phase-b-receipts-manifest-sha256", phase_b_receipts_sha,
            "--output-dir", str(cli.output_dir),
        ]
        if cli.local_test_mode: command.append("--local-test-mode")
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "", "LC_ALL": "C", "LANG": "C",
        }
        completed = subprocess.run(command, env=environment, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False)
        require(completed.returncode == 0, f"assembler refused: {completed.stderr.strip()}")
    manifest = validate_published_package(cli.output_dir, cli.arm)
    return cli.output_dir, manifest


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--assembler", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--terminal-receipt", type=Path, required=True)
    parser.add_argument("--submission-receipt", type=Path, required=True)
    parser.add_argument("--fetch-receipt", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--arm", choices=("frontier", "maxmc"), required=True)
    parser.add_argument("--expected-components-manifest-sha256", required=True)
    parser.add_argument("--expected-assembler-sha256", required=True)
    parser.add_argument("--expected-finalizer-sha256", required=True)
    parser.add_argument("--expected-sbatch-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-test-mode", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        output, digest = run(parse_cli(argv))
    except (FinalizationError, OSError, UnicodeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"V4_TERMINAL_FINALIZATION_REFUSED: {exc}", file=os.sys.stderr)
        return 1
    print(
        "V4_TERMINAL_FINALIZATION_COMPLETE "
        f"manifest={digest} result={output} paper_evidence=false analyzer_eligible=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
