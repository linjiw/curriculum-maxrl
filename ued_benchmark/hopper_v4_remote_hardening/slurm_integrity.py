#!/usr/bin/env python3
"""Strict structural validators for v4 hardened Slurm receipts.

The launcher contract deliberately does *not* combine ``--export=NIL`` with
explicit assignments.  Slurm documents NIL as the one export mode that does
not invoke ``--get-user-env`` and also says that explicit variables cannot be
specified with NIL.  The exact UED input closure is therefore a NUL-delimited
file passed as an ordinary batch-script argument.  The batch script validates
that file before constructing the child environment.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
from typing import Any, Mapping
from zoneinfo import ZoneInfo


HASH_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
TERMINAL_HEADER = (
    "JobIDRaw|JobName|Partition|State|ExitCode|ElapsedRaw|AllocCPUS|ReqMem|"
    "NodeList|Submit|Start|End|AllocTRES|QOS|TimelimitRaw|Restarts|WorkDir|"
    "StdOut|StdErr|SubmitLine"
)
RESOURCE_HEADER = "JobIDRaw|MaxRSS|TRESUsageInMax"
SIZE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)")


class SlurmIntegrityError(RuntimeError):
    """Raised when a scheduler receipt does not prove the exact allocation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SlurmIntegrityError(message)


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"unsafe {label}")
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate {label} key: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def parse_input_envelope(path: Path, expected_keys: set[str]) -> dict[str, str]:
    """Parse the exact NUL-delimited UED runtime-input envelope."""
    require(path.is_absolute() and ".." not in path.parts, "input envelope path must be absolute")
    require(path.is_file() and not path.is_symlink(), "unsafe input envelope")
    raw = path.read_bytes()
    require(raw and raw.endswith(b"\0"), "input envelope must be nonempty NUL records")
    records = raw[:-1].split(b"\0")
    values: dict[str, str] = {}
    for encoded in records:
        require(encoded and b"=" in encoded, "malformed input-envelope record")
        key_raw, value_raw = encoded.split(b"=", 1)
        try:
            key, value = key_raw.decode("ascii"), value_raw.decode("utf-8")
        except UnicodeError as exc:
            raise SlurmIntegrityError("invalid input-envelope encoding") from exc
        require(re.fullmatch(r"UED_[A-Z0-9_]+", key) is not None, "unsafe input key")
        require(key not in values and value != "", "duplicate or empty input record")
        require("\x00" not in value and "\n" not in value and "\r" not in value, "unsafe input value")
        values[key] = value
    require(set(values) == expected_keys, "input-envelope allowlist drift")
    return values


def validate_submission(
    receipt_path: Path,
    input_envelope: Path,
    expected_input_keys: set[str],
    *,
    job_id: str,
    sbatch_path: Path,
    sbatch_sha256: str,
    receipt_sbatch_path: str | None = None,
    receipt_input_envelope_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    receipt = load_json(receipt_path, "submission receipt")
    required = {
        "schema", "status", "job_id", "created_utc", "paper_evidence",
        "production_authorized", "ambient_environment", "export_mode",
        "get_user_env", "runtime_input_mode", "sbatch_path", "sbatch_sha256",
        "input_envelope_path", "input_envelope_sha256", "input_keys", "argv",
        "work_dir", "expected_stdout_path", "remote_submission_authorized",
    }
    require(set(receipt) == required, "submission receipt keys drift")
    require(
        receipt["schema"] == 1 and receipt["status"] == "submitted"
        and receipt["job_id"] == job_id
        and UTC_RE.fullmatch(str(receipt["created_utc"])) is not None
        and receipt["paper_evidence"] is False
        and receipt["production_authorized"] is False
        and receipt["ambient_environment"] == "env-i-empty"
        and receipt["export_mode"] == "NIL"
        and receipt["get_user_env"] is False
        and receipt["runtime_input_mode"] == "NUL_argument_envelope"
        and receipt["remote_submission_authorized"] is True,
        "submission receipt semantics drift",
    )
    require(
        sbatch_path.is_absolute() and sbatch_path.is_file() and not sbatch_path.is_symlink()
        and HASH_RE.fullmatch(sbatch_sha256 or "") is not None
        and sha256(sbatch_path) == sbatch_sha256
        and receipt["sbatch_path"] == (receipt_sbatch_path or str(sbatch_path))
        and receipt["sbatch_sha256"] == sbatch_sha256,
        "submitted sbatch binding drift",
    )
    inputs = parse_input_envelope(input_envelope, expected_input_keys)
    require(
        receipt["input_envelope_path"]
        == (receipt_input_envelope_path or str(input_envelope))
        and receipt["input_envelope_sha256"] == sha256(input_envelope)
        and receipt["input_keys"] == sorted(expected_input_keys),
        "submission input-envelope binding drift",
    )
    expected_argv = [
        "/usr/bin/sbatch", "--parsable", f"--chdir={receipt['work_dir']}",
        "--export=NIL", receipt["sbatch_path"],
        f"--ued-input-envelope={receipt['input_envelope_path']}",
        f"--ued-bundle-dir={Path(receipt['sbatch_path']).parents[2]}",
        f"--ued-submitted-sbatch={receipt['sbatch_path']}",
    ]
    require(receipt["argv"] == expected_argv, "submission argv drift")
    require(
        re.fullmatch(r"/scratch/[A-Za-z0-9._-]+/maxrl", str(receipt["work_dir"])) is not None
        and receipt["expected_stdout_path"]
        == f"{receipt['work_dir']}/tests/logs/ued-v4h-terminal_{job_id}.out",
        "submission work/output binding drift",
    )
    return receipt, inputs


def _parse_multivalue(path: Path) -> dict[str, list[str]]:
    require(path.is_file() and not path.is_symlink(), "unsafe terminal receipt")
    values: dict[str, list[str]] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split("\t", 1)
        require(len(fields) == 2 and fields[0] and fields[1], f"bad terminal line {number}")
        values.setdefault(fields[0], []).append(fields[1])
    required = {
        "terminal_receipt_schema", "retrieved_utc", "retrieved_epoch",
        "terminal_end_epoch", "terminal_header", "terminal_row",
        "resource_header", "resource_row",
    }
    require(set(values) == required, "terminal receipt keys drift")
    for key in required - {"resource_row"}:
        require(len(values[key]) == 1, f"terminal receipt cardinality drift: {key}")
    return values


def slurm_size_bytes(value: str) -> int:
    match = SIZE_RE.fullmatch(value)
    require(match is not None, f"malformed Slurm size: {value}")
    exponent = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[match.group(2)]
    result = int(float(match.group(1)) * (1024 ** exponent))
    require(result > 0, "Slurm size must be positive")
    return result


def _tres(value: str, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for encoded in value.split(","):
        fields = encoded.split("=", 1)
        require(len(fields) == 2 and all(fields) and fields[0] not in result, f"{label} drift")
        result[fields[0]] = fields[1]
    return result


def validate_terminal(
    receipt_path: Path,
    runtime_receipt_path: Path,
    runtime_receipt_sha256: str,
    submission: Mapping[str, Any],
    *,
    job_id: str,
) -> dict[str, Any]:
    values = _parse_multivalue(receipt_path)
    require(values["terminal_receipt_schema"] == ["2"], "terminal schema drift")
    require(values["terminal_header"] == [TERMINAL_HEADER], "terminal header drift")
    require(values["resource_header"] == [RESOURCE_HEADER], "resource header drift")
    require(UTC_RE.fullmatch(values["retrieved_utc"][0]) is not None, "terminal UTC drift")
    require(values["retrieved_epoch"][0].isdigit() and values["terminal_end_epoch"][0].isdigit(), "terminal epoch drift")
    retrieved = int(values["retrieved_epoch"][0]); end = int(values["terminal_end_epoch"][0])
    parsed_retrieved = int(datetime.strptime(values["retrieved_utc"][0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
    require(parsed_retrieved == retrieved and retrieved >= end, "terminal retrieval chronology drift")
    row = values["terminal_row"][0].split("|")
    require(len(row) == 20 and row[0] == job_id, "terminal job identity drift")
    require(
        row[1] == "ued-v4h-terminal" and row[2] == "gpuq"
        and row[3] == "COMPLETED" and row[4] == "0:0"
        and row[6] == "2" and row[7] in {"15G", "15Gn", "15Gc"}
        and row[13] == "gpu" and row[14] == "45" and row[15] == "0",
        "terminal state/resource identity drift",
    )
    require(row[5].isdigit() and 0 < int(row[5]) <= 2700, "terminal elapsed drift")
    epochs: list[int] = []
    for encoded in row[9:12]:
        require(
            re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}", encoded)
            is not None,
            "terminal Slurm timestamp drift",
        )
        naive = datetime.strptime(encoded, "%Y-%m-%dT%H:%M:%S")
        first = naive.replace(tzinfo=ZoneInfo("America/New_York"), fold=0)
        second = naive.replace(tzinfo=ZoneInfo("America/New_York"), fold=1)
        require(first.utcoffset() == second.utcoffset(), "ambiguous terminal Slurm timestamp")
        epochs.append(int(first.timestamp()))
    require(
        epochs[0] <= epochs[1] <= epochs[2] == end
        and epochs[2] - epochs[1] == int(row[5]),
        "terminal Slurm chronology drift",
    )
    alloc = _tres(row[12], "AllocTRES")
    require(
        alloc.get("cpu") == "2"
        and alloc.get("gres/gpu") == "1"
        and alloc.get("gres/gpu:1g.10gb") == "1",
        "terminal allocation is not exact 1g.10gb MIG",
    )
    input_envelope = Path(submission["input_envelope_path"])
    expected_submit_line = [
        "/usr/bin/sbatch", "--parsable", f"--chdir={submission['work_dir']}",
        "--export=NIL",
        submission["sbatch_path"], f"--ued-input-envelope={input_envelope}",
        f"--ued-bundle-dir={Path(submission['sbatch_path']).parents[2]}",
        f"--ued-submitted-sbatch={submission['sbatch_path']}",
    ]
    require(shlex.split(row[19]) == expected_submit_line, "terminal SubmitLine drift")
    require(
        row[16] == submission["work_dir"]
        and row[17] == row[18] == submission["expected_stdout_path"],
        "terminal work/output path drift",
    )
    resources: dict[str, dict[str, Any]] = {}
    for encoded in values["resource_row"]:
        fields = encoded.split("|")
        require(len(fields) == 3 and fields[0] not in resources, "resource row drift")
        require(fields[0] in {f"{job_id}.batch", f"{job_id}.extern"}, "resource step identity drift")
        size = slurm_size_bytes(fields[1])
        usage = _tres(fields[2], "TRESUsageInMax")
        resources[fields[0]] = {
            "job_id": fields[0], "max_rss": fields[1], "max_rss_bytes": size,
            "tres_usage_in_max": fields[2], "tres": usage,
        }
    require(set(resources) == {f"{job_id}.batch", f"{job_id}.extern"}, "batch/extern resource rows missing")
    batch = resources[f"{job_id}.batch"]
    require("gres/gpumem" in batch["tres"], "batch GPU-memory accounting missing")
    gpu_memory = slurm_size_bytes(batch["tres"]["gres/gpumem"])
    require(gpu_memory <= 11_000 * 1024 * 1024, "accounted GPU memory exceeds MIG slice")
    require(HASH_RE.fullmatch(runtime_receipt_sha256 or "") is not None, "bad runtime receipt hash")
    require(sha256(runtime_receipt_path) == runtime_receipt_sha256, "runtime receipt hash drift")
    runtime = load_json(runtime_receipt_path, "GPU runtime receipt")
    require(
        runtime.get("status") == "complete"
        and runtime.get("requested_gres") == "gpu:1g.10gb:1"
        and runtime.get("rung") == "terminal"
        and runtime.get("slurm", {}).get("job_id") == job_id
        and runtime.get("slurm", {}).get("array_job") is False
        and runtime.get("slurm", {}).get("slurm_restart_count") == 0,
        "GPU runtime/terminal binding drift",
    )
    return {
        "job_id": job_id,
        "terminal_receipt_sha256": sha256(receipt_path),
        "runtime_receipt_sha256": runtime_receipt_sha256,
        "elapsed_raw": int(row[5]),
        "max_rss_bytes": max(item["max_rss_bytes"] for item in resources.values()),
        "gpu_memory_usage_bytes": gpu_memory,
        "resource_rows": [resources[name] for name in sorted(resources)],
        "terminal_end_epoch": end,
        "retrieved_epoch": retrieved,
    }
