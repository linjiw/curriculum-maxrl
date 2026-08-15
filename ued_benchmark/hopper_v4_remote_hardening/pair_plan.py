#!/usr/bin/env python3
"""Create and validate the common-prerequisite plan for both v4 arm jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence


HASH_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
BASE_COMMIT = "d053054c5290a04c1c4cd8b55704d999cad73e30"
APPLIED_SHA = "9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae"
PROTOCOL_SHA = "1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269"
FRONTIER_CONFIG_SHA = "0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2"
MAXMC_CONFIG_SHA = "a3cc3ddf387a3bb7cf3d9759c3f13e6a74f2c7e32de311ac344d54ef5e703ec6"
PURPOSE = "v4_remote_hardening_common_r1_r2_pair_plan_not_paper_evidence"

IMPORT_RECEIPT_KEYS = {
    "job_id", "utc", "host", "result_dir", "bundle_manifest_sha256",
    "upstream_commit", "upstream_tree_git_sha1", "upstream_git_bundle_sha256",
    "overlay_manifest_sha256", "applied_overlay_manifest_sha256",
    "sbatch_sha256", "environment_lock_sha256", "environment_freeze_sha256",
    "environment_manifest_sha256", "environment_setup_script_sha256",
    "conda_explicit_sha256", "environment_json_sha256", "git",
    "training_endpoint",
}
ONE_UPDATE_RECEIPT_KEYS = {
    "job_id", "utc", "run_start_utc", "run_end_utc",
    "monotonic_elapsed_seconds", "process_user_seconds",
    "process_system_seconds", "process_max_rss_kib",
    "resource_accounting_source", "external_accounting_authority",
    "resource_accounting_sha256", "run_result_sha256", "checkpoint_sha256",
    "host", "result_dir", "input_closure_sha256", "bundle_manifest_sha256",
    "upstream_commit", "upstream_tree_git_sha1", "upstream_git_bundle_sha256",
    "overlay_manifest_sha256", "applied_overlay_manifest_sha256",
    "one_update_sbatch_sha256", "config_sha256", "overlay_contract_sha256",
    "environment_lock_sha256", "environment_freeze_sha256",
    "environment_manifest_sha256", "environment_setup_script_sha256",
    "conda_explicit_sha256", "environment_json_sha256", "git",
    "import_smoke_manifest_sha256", "import_smoke_sbatch_sha256", "xpid",
    "n_parallel", "n_eval", "frontier_n_rollouts", "outer_cycles",
    "actual_ppo_updates", "n_grad_updates", "ppo_epochs", "ppo_minibatches",
    "optimizer_step_applications", "endpoint_class", "max_student_updates",
    "actual_student_updates", "total_transitions",
    "frontier_incomplete_group_count", "frontier_duplicate_new_group_count",
    "paper_evidence", "terminal_sacct_included",
}
ONE_UPDATE_COMPLETE_KEYS = {
    "complete_schema", "artifact_type", "job_id", "paper_evidence",
    "actual_ppo_updates", "n_grad_updates", "ppo_epochs", "ppo_minibatches",
    "optimizer_step_applications", "resource_accounting_source",
    "external_accounting_authority", "terminal_sacct_included",
    "input_closure_sha256", "sha256sums_sha256",
}
HARDENING_RECEIPT_KEYS = {
    "schema", "status", "purpose", "rung", "job_id", "paper_evidence",
    "analyzer_eligible", "production_authorized", "endpoint_access_authorized",
    "cost100_implemented", "max_student_updates", "bundle_manifest_sha256",
    "legacy_result_dir", "legacy_manifest_sha256",
    "environment_tree_manifest_sha256", "environment_tree_receipt_sha256",
    "environment_verified_before", "environment_verified_after",
    "gpu_runtime_receipt_path", "gpu_runtime_receipt_sha256",
    "hardening_sbatch_sha256", "legacy_sbatch_sha256", "job_guard_sha256",
    "pair_plan_manifest_sha256",
}
ENVIRONMENT_RECEIPT_KEYS = {
    "schema", "status", "installed_file_byte_closure", "environment",
    "environment_tree_sha256", "directory_count", "file_count",
    "symlink_count", "regular_file_bytes", "python_path",
    "python_resolved_path", "python_sha256", "python_version", "conda_path",
    "conda_resolved_path", "conda_sha256", "generator_path",
    "generator_sha256", "paper_evidence", "production_authorized",
}


class PairPlanError(RuntimeError):
    """Raised when the shared arm plan is ambiguous or incomplete."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PairPlanError(message)


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict[str, Any]:
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


def _loads(text: str, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate {label} key: {key}")
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=unique)
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def _canonical_directory(path: Path, label: str) -> Path:
    require(path.is_absolute() and ".." not in path.parts, f"{label} must be absolute")
    require(path.resolve(strict=True) == path and path.is_dir() and not path.is_symlink(), f"unsafe {label}")
    return path


def _parse_receipt(text: str, expected_keys: set[str], label: str) -> dict[str, str]:
    lines = text.splitlines()
    require(lines and lines[0] == "field\tvalue", f"{label} header drift")
    rows: dict[str, str] = {}
    for number, line in enumerate(lines[1:], 2):
        fields = line.split("\t", 1)
        require(
            len(fields) == 2 and fields[0] and fields[1] and fields[0] not in rows,
            f"{label} row {number} drift",
        )
        rows[fields[0]] = fields[1]
    require(set(rows) == expected_keys, f"{label} exact keyset drift")
    return rows


def _validate_result_manifest(root: Path, expected_sha256: str) -> str:
    root = _canonical_directory(root, "prerequisite result")
    require(HASH_RE.fullmatch(expected_sha256 or "") is not None, "bad prerequisite manifest hash")
    manifest = root / "SHA256SUMS"
    require(sha256(manifest) == expected_sha256, "prerequisite manifest digest drift")
    listed: dict[str, str] = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw)
        require(match is not None, "prerequisite manifest row drift")
        digest, raw_path = match.groups()
        name = raw_path.removeprefix("./")
        relative = PurePosixPath(name)
        require(
            name and not relative.is_absolute()
            and all(part not in {"", ".", ".."} for part in relative.parts)
            and relative.as_posix() not in listed,
            "prerequisite manifest path drift",
        )
        target = root.joinpath(*relative.parts)
        require(target.is_file() and not target.is_symlink(), "prerequisite payload missing")
        require(sha256(target) == digest, f"prerequisite payload drift: {name}")
        listed[relative.as_posix()] = digest
    actual: set[str] = set()
    for target in root.rglob("*"):
        require(not target.is_symlink(), "prerequisite symlink forbidden")
        if target.is_file():
            relative = target.relative_to(root).as_posix()
            if relative not in {"SHA256SUMS", "COMPLETE"}:
                actual.add(relative)
        else:
            require(target.is_dir(), "prerequisite special entry forbidden")
    require(set(listed) == actual, "prerequisite exact-tree closure drift")
    return sha256(manifest)


def _archive_result(
    rung: str,
    root: Path,
    manifest_sha256: str,
    bundle_manifest_sha256: str,
) -> dict[str, Any]:
    _validate_result_manifest(root, manifest_sha256)
    receipt_text = (root / "receipt.tsv").read_text(encoding="utf-8")
    manifest_text = (root / "SHA256SUMS").read_text(encoding="utf-8")
    complete_text = (root / "COMPLETE").read_text(encoding="utf-8")
    expected = IMPORT_RECEIPT_KEYS if rung == "import" else ONE_UPDATE_RECEIPT_KEYS
    receipt = _parse_receipt(receipt_text, expected, f"{rung} receipt")
    require(
        receipt["result_dir"] == str(root)
        and receipt["job_id"] == root.name
        and receipt["bundle_manifest_sha256"] == bundle_manifest_sha256
        and receipt["upstream_commit"] == BASE_COMMIT
        and receipt["applied_overlay_manifest_sha256"] == APPLIED_SHA,
        f"{rung} receipt provenance drift",
    )
    if rung == "import":
        require(
            receipt["training_endpoint"] == "false"
            and re.fullmatch(
                r"complete\t[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\n",
                complete_text,
            ) is not None,
            "import completion drift",
        )
    else:
        require(
            receipt["endpoint_class"] == "bounded_engineering_one_update"
            and receipt["actual_student_updates"] == "1"
            and receipt["actual_ppo_updates"] == receipt["n_grad_updates"] == "1"
            and receipt["paper_evidence"] == "false"
            and receipt["terminal_sacct_included"] == "false"
            and receipt["config_sha256"] == FRONTIER_CONFIG_SHA,
            "one-update receipt semantics drift",
        )
        complete = _loads(complete_text, "one-update completion")
        require(
            isinstance(complete, dict) and set(complete) == ONE_UPDATE_COMPLETE_KEYS
            and complete["complete_schema"] == 2
            and complete["artifact_type"] == "frontier_exact_grouped_one_update_engineering"
            and complete["job_id"] == receipt["job_id"]
            and complete["paper_evidence"] is False
            and complete["actual_ppo_updates"] == complete["n_grad_updates"] == 1
            and complete["terminal_sacct_included"] is False
            and complete["input_closure_sha256"] == receipt["input_closure_sha256"]
            and complete["sha256sums_sha256"] == manifest_sha256,
            "one-update completion drift",
        )
    receipt_sha = hashlib.sha256(receipt_text.encode("utf-8")).hexdigest()
    listed_receipt = None
    for raw in manifest_text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw)
        require(match is not None, f"{rung} archived manifest row drift")
        if match.group(2).removeprefix("./") == "receipt.tsv":
            require(listed_receipt is None, f"duplicate {rung} receipt manifest entry")
            listed_receipt = match.group(1)
    require(listed_receipt == receipt_sha, f"{rung} receipt/manifest binding drift")
    def record(text: str) -> dict[str, str]:
        return {"encoding": "utf-8", "sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text}
    return {
        "rung": rung,
        "result_dir": str(root),
        "manifest_sha256": manifest_sha256,
        "complete_sha256": sha256(root / "COMPLETE"),
        "archived": {
            "receipt.tsv": record(receipt_text),
            "SHA256SUMS": record(manifest_text),
            "COMPLETE": record(complete_text),
        },
    }


def _validate_environment_closure(
    root: Path, manifest_sha256: str, receipt_sha256: str
) -> dict[str, Any]:
    root = _canonical_directory(root, "environment closure")
    require(
        {entry.name for entry in root.iterdir()}
        == {"environment-tree.jsonl", "receipt.json", "SHA256SUMS", "COMPLETE"},
        "environment closure files drift",
    )
    manifest = root / "SHA256SUMS"
    require(sha256(manifest) == manifest_sha256, "environment closure manifest drift")
    require(sha256(root / "receipt.json") == receipt_sha256, "environment closure receipt drift")
    rows: dict[str, str] = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (environment-tree\.jsonl|receipt\.json)", raw)
        require(match is not None and match.group(2) not in rows, "environment manifest row drift")
        rows[match.group(2)] = match.group(1)
    require(set(rows) == {"environment-tree.jsonl", "receipt.json"}, "environment manifest payload drift")
    for name, digest in rows.items():
        require(sha256(root / name) == digest, f"environment closure payload drift: {name}")
    receipt = _load(root / "receipt.json", "environment receipt")
    complete = _load(root / "COMPLETE", "environment completion")
    require(
        set(receipt) == ENVIRONMENT_RECEIPT_KEYS
        and receipt["schema"] == 1 and receipt["status"] == "complete"
        and receipt["installed_file_byte_closure"] is True
        and receipt["environment_tree_sha256"] == rows["environment-tree.jsonl"]
        and receipt["python_version"] == "3.10.20"
        and receipt["paper_evidence"] is False
        and receipt["production_authorized"] is False
        and complete == {
            "file_count": 2, "schema": 1,
            "sha256sums_sha256": manifest_sha256, "status": "complete",
        },
        "environment closure semantics drift",
    )
    return receipt


def _validate_hardening(
    path: Path,
    expected_sha256: str,
    rung: str,
    legacy_result: Path,
    legacy_manifest_sha256: str,
    bundle_manifest_sha256: str,
    environment_manifest_sha256: str,
    environment_receipt_sha256: str,
) -> dict[str, Any]:
    require(rung in {"import", "one_update"}, "pair-plan hardening rung drift")
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), f"unsafe {rung} hardening receipt")
    require(sha256(path) == expected_sha256, f"{rung} hardening receipt hash drift")
    receipt = _load(path, f"{rung} hardening receipt")
    require(
        set(receipt) == HARDENING_RECEIPT_KEYS
        and receipt["schema"] == 1 and receipt["status"] == "complete"
        and receipt["purpose"] == "v4_remote_hardening_rung_integrity_not_paper_evidence"
        and receipt["rung"] == rung
        and receipt["job_id"] == legacy_result.name
        and receipt["paper_evidence"] is False
        and receipt["analyzer_eligible"] is False
        and receipt["production_authorized"] is False
        and receipt["endpoint_access_authorized"] is False
        and receipt["cost100_implemented"] is False
        and receipt["max_student_updates"] == (0 if rung == "import" else 1)
        and receipt["bundle_manifest_sha256"] == bundle_manifest_sha256
        and receipt["legacy_result_dir"] == str(legacy_result)
        and receipt["legacy_manifest_sha256"] == legacy_manifest_sha256
        and receipt["environment_tree_manifest_sha256"] == environment_manifest_sha256
        and receipt["environment_tree_receipt_sha256"] == environment_receipt_sha256
        and receipt["environment_verified_before"] is True
        and receipt["environment_verified_after"] is True
        and Path(receipt["gpu_runtime_receipt_path"]).is_absolute()
        and HASH_RE.fullmatch(str(receipt["gpu_runtime_receipt_sha256"])) is not None,
        f"{rung} hardening receipt semantics drift",
    )
    for key in ("hardening_sbatch_sha256", "legacy_sbatch_sha256", "job_guard_sha256"):
        require(HASH_RE.fullmatch(str(receipt[key])) is not None, f"bad {rung} {key}")
    require(receipt["pair_plan_manifest_sha256"] is None, f"premature {rung} pair binding")
    return receipt


def _write_text(path: Path, text: str) -> None:
    require(not path.exists(), f"refusing overwrite: {path}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _text_record(text: str) -> dict[str, str]:
    return {
        "encoding": "utf-8",
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
    }


def _archived_record(value: Any, label: str) -> str:
    require(
        isinstance(value, dict) and set(value) == {"encoding", "sha256", "text"}
        and value["encoding"] == "utf-8" and isinstance(value["text"], str)
        and value["sha256"] == hashlib.sha256(value["text"].encode("utf-8")).hexdigest(),
        f"{label} archive drift",
    )
    return value["text"]


def _validate_archived_common(common: Any) -> bool:
    require(
        isinstance(common, dict)
        and set(common) == {
            "bundle_manifest_sha256", "protocol_sha256", "environment_closure",
            "import", "import_hardening", "one_update", "one_update_hardening",
        },
        "pair-plan common keys drift",
    )
    require(
        HASH_RE.fullmatch(str(common["bundle_manifest_sha256"])) is not None
        and common["protocol_sha256"] == PROTOCOL_SHA,
        "pair-plan common identity drift",
    )
    environment = common["environment_closure"]
    require(
        isinstance(environment, dict)
        and set(environment) == {
            "path", "manifest_sha256", "receipt_sha256", "environment",
            "python_sha256", "conda_sha256",
        }
        and Path(environment["path"]).is_absolute()
        and Path(environment["environment"]).is_absolute()
        and all(HASH_RE.fullmatch(str(environment[key])) is not None for key in (
            "manifest_sha256", "receipt_sha256", "python_sha256", "conda_sha256")),
        "pair-plan environment reference drift",
    )

    parsed_receipts: dict[str, dict[str, str]] = {}
    for rung, expected_receipt_keys in (
        ("import", IMPORT_RECEIPT_KEYS), ("one_update", ONE_UPDATE_RECEIPT_KEYS)
    ):
        record = common[rung]
        require(
            isinstance(record, dict)
            and set(record) == {
                "rung", "result_dir", "manifest_sha256", "complete_sha256", "archived"
            }
            and record["rung"] == rung and Path(record["result_dir"]).is_absolute()
            and all(HASH_RE.fullmatch(str(record[key])) is not None
                    for key in ("manifest_sha256", "complete_sha256"))
            and isinstance(record["archived"], dict)
            and set(record["archived"]) == {"receipt.tsv", "SHA256SUMS", "COMPLETE"},
            f"archived {rung} structure drift",
        )
        receipt_text = _archived_record(record["archived"]["receipt.tsv"], f"{rung} receipt")
        manifest_text = _archived_record(record["archived"]["SHA256SUMS"], f"{rung} manifest")
        complete_text = _archived_record(record["archived"]["COMPLETE"], f"{rung} completion")
        require(
            hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
            == record["manifest_sha256"]
            and hashlib.sha256(complete_text.encode("utf-8")).hexdigest()
            == record["complete_sha256"],
            f"archived {rung} digest drift",
        )
        receipt = _parse_receipt(receipt_text, expected_receipt_keys, f"archived {rung} receipt")
        parsed_receipts[rung] = receipt
        receipt_digest = hashlib.sha256(receipt_text.encode("utf-8")).hexdigest()
        receipt_rows = [
            raw for raw in manifest_text.splitlines()
            if raw.endswith("  receipt.tsv") or raw.endswith("  ./receipt.tsv")
        ]
        require(
            len(receipt_rows) == 1 and receipt_rows[0].split("  ", 1)[0] == receipt_digest,
            f"archived {rung} receipt/manifest drift",
        )
        require(
            receipt["job_id"] == Path(record["result_dir"]).name
            and receipt["result_dir"] == record["result_dir"]
            and receipt["bundle_manifest_sha256"] == common["bundle_manifest_sha256"],
            f"archived {rung} semantic drift",
        )
        if rung == "import":
            require(
                re.fullmatch(
                    r"complete\t[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\n",
                    complete_text,
                ) is not None,
                "archived import completion drift",
            )
        else:
            complete = _loads(complete_text, "archived one-update completion")
            require(
                set(complete) == ONE_UPDATE_COMPLETE_KEYS
                and complete["job_id"] == receipt["job_id"]
                and complete["input_closure_sha256"] == receipt["input_closure_sha256"]
                and complete["sha256sums_sha256"] == record["manifest_sha256"],
                "archived one-update completion cross-binding drift",
            )

    require(
        parsed_receipts["one_update"]["import_smoke_manifest_sha256"]
        == common["import"]["manifest_sha256"],
        "one-update/import cross-binding drift",
    )
    for rung in ("import", "one_update"):
        entry = common[f"{rung}_hardening"]
        require(
            isinstance(entry, dict) and set(entry) == {"path", "sha256", "archived"}
            and Path(entry["path"]).is_absolute()
            and HASH_RE.fullmatch(str(entry["sha256"])) is not None,
            f"archived {rung} hardening structure drift",
        )
        text = _archived_record(entry["archived"], f"{rung} hardening")
        require(
            entry["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest(),
            f"archived {rung} hardening digest drift",
        )
        receipt = _loads(text, f"archived {rung} hardening")
        require(
            set(receipt) == HARDENING_RECEIPT_KEYS
            and receipt["rung"] == rung
            and receipt["job_id"] == parsed_receipts[rung]["job_id"]
            and receipt["bundle_manifest_sha256"] == common["bundle_manifest_sha256"]
            and receipt["legacy_result_dir"] == common[rung]["result_dir"]
            and receipt["legacy_manifest_sha256"] == common[rung]["manifest_sha256"]
            and receipt["environment_tree_manifest_sha256"] == environment["manifest_sha256"]
            and receipt["environment_tree_receipt_sha256"] == environment["receipt_sha256"]
            and receipt["environment_verified_before"] is True
            and receipt["environment_verified_after"] is True
            and receipt["pair_plan_manifest_sha256"] is None,
            f"archived {rung} hardening semantic drift",
        )
    return True


def create(cli: argparse.Namespace) -> tuple[Path, str, str]:
    tool = Path(__file__).resolve()
    require(sha256(tool) == cli.expected_tool_sha256, "pair-plan tool drift")
    for value in (
        cli.bundle_manifest_sha256, cli.import_manifest_sha256,
        cli.one_update_manifest_sha256, cli.environment_manifest_sha256,
        cli.environment_receipt_sha256, cli.import_hardening_sha256,
        cli.one_update_hardening_sha256,
    ):
        require(HASH_RE.fullmatch(value or "") is not None, "malformed input hash")
    import_root = _canonical_directory(cli.import_result_dir, "import result")
    one_root = _canonical_directory(cli.one_update_result_dir, "one-update result")
    import_archive = _archive_result(
        "import", import_root, cli.import_manifest_sha256,
        cli.bundle_manifest_sha256,
    )
    one_archive = _archive_result(
        "one_update", one_root, cli.one_update_manifest_sha256,
        cli.bundle_manifest_sha256,
    )
    environment_receipt = _validate_environment_closure(
        cli.environment_closure, cli.environment_manifest_sha256,
        cli.environment_receipt_sha256,
    )
    import_hardening = _validate_hardening(
        cli.import_hardening_receipt, cli.import_hardening_sha256,
        "import", import_root, cli.import_manifest_sha256,
        cli.bundle_manifest_sha256, cli.environment_manifest_sha256,
        cli.environment_receipt_sha256,
    )
    one_hardening = _validate_hardening(
        cli.one_update_hardening_receipt, cli.one_update_hardening_sha256,
        "one_update", one_root, cli.one_update_manifest_sha256,
        cli.bundle_manifest_sha256, cli.environment_manifest_sha256,
        cli.environment_receipt_sha256,
    )
    common = {
        "bundle_manifest_sha256": cli.bundle_manifest_sha256,
        "protocol_sha256": PROTOCOL_SHA,
        "environment_closure": {
            "path": str(cli.environment_closure),
            "manifest_sha256": cli.environment_manifest_sha256,
            "receipt_sha256": cli.environment_receipt_sha256,
            "environment": environment_receipt["environment"],
            "python_sha256": environment_receipt["python_sha256"],
            "conda_sha256": environment_receipt["conda_sha256"],
        },
        "import": import_archive,
        "import_hardening": {
            "path": str(cli.import_hardening_receipt),
            "sha256": cli.import_hardening_sha256,
            "archived": _text_record(cli.import_hardening_receipt.read_text(encoding="utf-8")),
        },
        "one_update": one_archive,
        "one_update_hardening": {
            "path": str(cli.one_update_hardening_receipt),
            "sha256": cli.one_update_hardening_sha256,
            "archived": _text_record(cli.one_update_hardening_receipt.read_text(encoding="utf-8")),
        },
    }
    pair_key = hashlib.sha256(
        json.dumps(common, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    pair_id = pair_key[:20]
    plan = {
        "schema": 1,
        "status": "frozen_before_arm_submission",
        "purpose": PURPOSE,
        "paper_evidence": False,
        "analyzer_eligible": False,
        "production_authorized": False,
        "endpoint_access_authorized": False,
        "cost100_implemented": False,
        "max_student_updates": 1,
        "pair_id": pair_id,
        "training_seed": 101,
        "arms": {
            "frontier": {"config_sha256": FRONTIER_CONFIG_SHA, "run_id": "engineering-frontier-s101"},
            "maxmc": {"config_sha256": MAXMC_CONFIG_SHA, "run_id": "engineering-maxmc-s101"},
        },
        "common": common,
        "pair_key_sha256": pair_key,
        "pair_plan_tool_sha256": cli.expected_tool_sha256,
    }
    parent = _canonical_directory(cli.output_parent, "pair-plan output parent")
    output = parent / pair_id
    require(not output.exists() and not output.is_symlink(), "pair-plan output exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{pair_id}.", dir=parent))
    try:
        _write_json(temporary / "PAIR_PLAN.json", plan)
        _write_text(temporary / "SHA256SUMS", f"{sha256(temporary / 'PAIR_PLAN.json')}  PAIR_PLAN.json\n")
        manifest_sha = sha256(temporary / "SHA256SUMS")
        _write_json(temporary / "COMPLETE", {
            "schema": 1, "status": "complete", "pair_id": pair_id,
            "sha256sums_sha256": manifest_sha, "file_count": 1,
            "paper_evidence": False, "production_authorized": False,
        })
        os.replace(temporary, output)
        validated = validate(output, manifest_sha)
        require(validated["pair_id"] == pair_id, "post-publish pair-plan drift")
        return output, manifest_sha, sha256(output / "PAIR_PLAN.json")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def validate(root: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    root = _canonical_directory(root, "pair plan")
    require({entry.name for entry in root.iterdir()} == {"PAIR_PLAN.json", "SHA256SUMS", "COMPLETE"}, "pair-plan closure drift")
    require(sha256(root / "SHA256SUMS") == expected_manifest_sha256, "pair-plan manifest drift")
    manifest_line = (root / "SHA256SUMS").read_text(encoding="utf-8")
    require(manifest_line == f"{sha256(root / 'PAIR_PLAN.json')}  PAIR_PLAN.json\n", "pair-plan payload binding drift")
    plan = _load(root / "PAIR_PLAN.json", "pair plan")
    required = {
        "schema", "status", "purpose", "paper_evidence", "analyzer_eligible",
        "production_authorized", "endpoint_access_authorized", "cost100_implemented",
        "max_student_updates", "pair_id", "training_seed", "arms", "common",
        "pair_key_sha256", "pair_plan_tool_sha256",
    }
    require(set(plan) == required, "pair-plan keys drift")
    require(
        plan["schema"] == 1 and plan["status"] == "frozen_before_arm_submission"
        and plan["purpose"] == PURPOSE and plan["training_seed"] == 101
        and plan["max_student_updates"] == 1
        and all(plan[key] is False for key in (
            "paper_evidence", "analyzer_eligible", "production_authorized",
            "endpoint_access_authorized", "cost100_implemented"))
        and set(plan["arms"]) == {"frontier", "maxmc"}
        and plan["arms"]["frontier"] == {"config_sha256": FRONTIER_CONFIG_SHA, "run_id": "engineering-frontier-s101"}
        and plan["arms"]["maxmc"] == {"config_sha256": MAXMC_CONFIG_SHA, "run_id": "engineering-maxmc-s101"},
        "pair-plan semantics drift",
    )
    pair_key = hashlib.sha256(
        json.dumps(plan["common"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    require(plan["pair_key_sha256"] == pair_key and plan["pair_id"] == pair_key[:20] == root.name, "pair-plan identity drift")
    complete = _load(root / "COMPLETE", "pair-plan completion")
    require(complete == {
        "schema": 1, "status": "complete", "pair_id": plan["pair_id"],
        "sha256sums_sha256": expected_manifest_sha256, "file_count": 1,
        "paper_evidence": False, "production_authorized": False,
    }, "pair-plan completion drift")
    common = plan["common"]
    require(
        _validate_archived_common(common),
        "pair-plan common-prerequisite cross-binding drift",
    )
    return plan


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("create")
    create_parser.add_argument("--bundle-manifest-sha256", required=True)
    create_parser.add_argument("--import-result-dir", type=Path, required=True)
    create_parser.add_argument("--import-manifest-sha256", required=True)
    create_parser.add_argument("--one-update-result-dir", type=Path, required=True)
    create_parser.add_argument("--one-update-manifest-sha256", required=True)
    create_parser.add_argument("--environment-closure", type=Path, required=True)
    create_parser.add_argument("--environment-manifest-sha256", required=True)
    create_parser.add_argument("--environment-receipt-sha256", required=True)
    create_parser.add_argument("--import-hardening-receipt", type=Path, required=True)
    create_parser.add_argument("--import-hardening-sha256", required=True)
    create_parser.add_argument("--one-update-hardening-receipt", type=Path, required=True)
    create_parser.add_argument("--one-update-hardening-sha256", required=True)
    create_parser.add_argument("--expected-tool-sha256", required=True)
    create_parser.add_argument("--output-parent", type=Path, required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--pair-plan-dir", type=Path, required=True)
    validate_parser.add_argument("--expected-manifest-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    cli = parse_cli(argv)
    try:
        if cli.command == "create":
            output, manifest, plan = create(cli)
            print(f"V4H_PAIR_PLAN_COMPLETE path={output} manifest={manifest} plan={plan}")
        else:
            plan = validate(cli.pair_plan_dir, cli.expected_manifest_sha256)
            print(f"V4H_PAIR_PLAN_VALID pair_id={plan['pair_id']}")
    except (PairPlanError, OSError, UnicodeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"V4H_PAIR_PLAN_REFUSED: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
