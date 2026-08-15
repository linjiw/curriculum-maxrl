#!/usr/bin/env python3
"""Assemble/validate a self-contained, permanently non-evidence v4h package.

The sealed Phase-A component tree is copied intact.  This structural tool
parses only value-free integrity receipts and never reads evaluation JSONL,
CSV, aggregate values, or any held endpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping, Sequence


HASH_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
FROZEN_ASSEMBLER_SHA = "b9cc64f2ed66da1ae997c1f23e175d531a6a84be97970af2e1a3d6c681936b63"
PROTOCOL_SHA = "1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269"
PURPOSE = "v4_remote_hardening_terminal_package_not_paper_evidence"
TOP_LEVEL = {
    "components", "phase-b", "RUN_MANIFEST.json", "SHA256SUMS", "COMPLETE",
}
PHASE_B_PAYLOADS = {
    "submission.json", "input-envelope.nul", "terminal.tsv", "fetch.tsv",
    "gpu-runtime.json", "hardening-receipt.json", "pair-plan",
    "environment-closure", "provenance", "SHA256SUMS", "COMPLETE",
}
PROVENANCE_FILES = {
    "development_protocol_v2_tie_aware_draft.json",
    "assemble_matched_run_v4.py", "evaluate_matched_terminal_v4.py",
    "run_matched_terminal_v4.py", "run_terminal_phase_a_v4.py",
    "assemble_remote_hardened.py", "slurm_integrity.py", "pair_plan.py",
    "environment_tree.py", "gpu_runtime_probe.py", "job_guard.py",
    "finalize_remote_hardened.py", "submitted-terminal.sbatch",
    "REMOTE_HARDENING_STATE.json",
}


class AssemblyError(RuntimeError):
    """Raised when an outer v4h package is not exactly closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssemblyError(message)


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"unsafe regular file: {path}")
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


def canonical_existing(path: Path, *, directory: bool, label: str) -> Path:
    require(path.is_absolute() and ".." not in path.parts, f"{label} must be absolute")
    require(path.resolve(strict=True) == path and not path.is_symlink(), f"unsafe {label}")
    require(path.is_dir() if directory else path.is_file(), f"missing {label}")
    return path


def canonical_new(path: Path, protected: Sequence[Path]) -> Path:
    require(path.is_absolute() and ".." not in path.parts and path.name not in {"", ".", ".."}, "output must be canonical absolute")
    require(not path.exists() and not path.is_symlink(), "output exists")
    parent = canonical_existing(path.parent, directory=True, label="output parent")
    require(parent / path.name == path, "output path drift")
    for source in protected:
        require(not path.is_relative_to(source) and not source.is_relative_to(path), "output/input overlap")
    return path


def _safe_relative(raw: str) -> PurePosixPath:
    raw = raw.removeprefix("./")
    value = PurePosixPath(raw)
    require(raw and not value.is_absolute() and all(part not in {"", ".", ".."} for part in value.parts), "unsafe manifest path")
    return value


def validate_tree(root: Path, manifest_name: str, completion_name: str) -> tuple[str, dict[str, Any]]:
    root = canonical_existing(root, directory=True, label="closed tree")
    manifest = root / manifest_name
    listed: dict[str, str] = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw)
        require(match is not None, "manifest row drift")
        relative = _safe_relative(match.group(2)).as_posix()
        require(relative not in listed, "duplicate manifest path")
        target = root.joinpath(*PurePosixPath(relative).parts)
        require(target.is_file() and not target.is_symlink() and sha256(target) == match.group(1), f"payload drift: {relative}")
        listed[relative] = match.group(1)
    actual: set[str] = set()
    for target in root.rglob("*"):
        require(not target.is_symlink(), f"symlink forbidden: {target}")
        if target.is_file():
            relative = target.relative_to(root).as_posix()
            if relative not in {manifest_name, completion_name}:
                actual.add(relative)
        else:
            require(target.is_dir(), "special entry forbidden")
    require(actual == set(listed), "exact-tree closure drift")
    return sha256(manifest), load_json(root / completion_name, "completion")


def _copy_tree(source: Path, destination: Path) -> None:
    canonical_existing(source, directory=True, label="copy source")
    require(not destination.exists(), "copy destination exists")
    for target in source.rglob("*"):
        require(not target.is_symlink(), "source symlink forbidden")
        require(target.is_file() or target.is_dir(), "source special entry forbidden")
    shutil.copytree(source, destination, symlinks=False)


def _load_module(path: Path, expected: str, name: str, *, sibling_imports: bool = False) -> ModuleType:
    require(sha256(path) == expected, f"{name} hash drift")
    if sibling_imports:
        for imported in ("evaluate_matched_terminal_v4", "run_matched_terminal_v4"):
            sys.modules.pop(imported, None)
        sys.path.insert(0, str(path.parent))
    try:
        specification = importlib.util.spec_from_file_location(f"v4h_{name}", path)
        require(specification is not None and specification.loader is not None, f"cannot load {name}")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module
    finally:
        if sibling_imports:
            sys.path.pop(0)


def _validate_components(
    components: Path, protocol: Path, manifest_sha: str,
    assembler: Path, *, local_test_mode: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    module = _load_module(assembler, FROZEN_ASSEMBLER_SHA, "frozen_assembler", sibling_imports=True)
    try:
        complete, context, actual, _closure = module._validate_components(
            components, protocol, manifest_sha, local_test_mode=local_test_mode
        )
    except Exception as exc:
        raise AssemblyError(f"frozen component validator refused: {exc}") from exc
    require(actual == manifest_sha, "component manifest drift")
    return complete, context


def tree_digest(root: Path) -> str:
    rows: list[str] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix()):
        require(not path.is_symlink(), "symlink in fetched component tree")
        rows.append(f"{sha256(path)}  ./{path.relative_to(root).as_posix()}\n")
    require(rows, "empty component tree")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _parse_tsv(path: Path, keys: set[str], label: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split("\t", 1)
        require(len(fields) == 2 and all(fields) and fields[0] not in values, f"{label} row drift")
        values[fields[0]] = fields[1]
    require(set(values) == keys, f"{label} keyset drift")
    return values


def validate_fetch(
    path: Path, components: Path, terminal: Mapping[str, Any], terminal_sha: str,
    arm: str, *, local_test_mode: bool, receipt_local_path: str | None = None,
) -> dict[str, str]:
    receipt = _parse_tsv(path, {
        "fetch_receipt_schema", "fetch_started_utc", "fetch_started_epoch",
        "retrieved_utc", "retrieved_epoch", "terminal_end_epoch",
        "terminal_receipt_sha256", "remote_path", "remote_type", "remote_digest",
        "manifest_verified", "local_path", "local_digest",
    }, "fetch receipt")
    require(receipt["fetch_receipt_schema"] == "2" and receipt["remote_type"] == "dir", "fetch schema drift")
    for utc_key, epoch_key in (("fetch_started_utc", "fetch_started_epoch"), ("retrieved_utc", "retrieved_epoch")):
        require(UTC_RE.fullmatch(receipt[utc_key]) is not None and receipt[epoch_key].isdigit(), "fetch time drift")
        parsed = int(datetime.strptime(receipt[utc_key], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
        require(parsed == int(receipt[epoch_key]), "fetch UTC/epoch drift")
    require(
        receipt["terminal_receipt_sha256"] == terminal_sha
        and receipt["terminal_end_epoch"].isdigit()
        and int(receipt["terminal_end_epoch"]) == terminal["terminal_end_epoch"]
        and int(receipt["fetch_started_epoch"]) >= terminal["retrieved_epoch"]
        and int(receipt["retrieved_epoch"]) >= int(receipt["fetch_started_epoch"])
        and receipt["manifest_verified"] == "1"
        and receipt["local_path"] == (receipt_local_path or str(components)),
        "fetch chronology/path drift",
    )
    if not local_test_mode:
        require(re.fullmatch(rf"/scratch/[A-Za-z0-9._-]+/maxrl/tests/ued-minimax-v4-terminal-components/{terminal['job_id']}-{arm}", receipt["remote_path"]) is not None, "fetch remote path drift")
    digest = tree_digest(components)
    require(receipt["remote_digest"] == receipt["local_digest"] == digest, "fetch digest drift")
    return receipt


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing overwrite: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_manifest(root: Path, name: str, excluded: set[str]) -> str:
    payloads = sorted(
        (
            path for path in root.rglob("*")
            if path.is_file() and path.relative_to(root).as_posix() not in excluded
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    text = "".join(f"{sha256(path)}  ./{path.relative_to(root).as_posix()}\n" for path in payloads)
    (root / name).write_text(text, encoding="utf-8")
    return sha256(root / name)


def _phase_b_complete(root: Path) -> str:
    manifest_sha = _write_manifest(root, "SHA256SUMS", {"SHA256SUMS", "COMPLETE"})
    _write_json(root / "COMPLETE", {
        "schema": 1, "status": "complete", "file_count": len([
            path for path in root.rglob("*") if path.is_file()
            and path.relative_to(root).as_posix() not in {"SHA256SUMS", "COMPLETE"}
        ]), "sha256sums_sha256": manifest_sha,
        "paper_evidence": False, "production_authorized": False,
    })
    return manifest_sha


def _root_complete(root: Path, run_id: str, arm: str) -> str:
    manifest_sha = _write_manifest(root, "SHA256SUMS", {"SHA256SUMS", "COMPLETE"})
    _write_json(root / "COMPLETE", {
        "schema": 1, "status": "complete", "run_id": run_id, "arm": arm,
        "file_count": len([
            path for path in root.rglob("*") if path.is_file()
            and path.relative_to(root).as_posix() not in {"SHA256SUMS", "COMPLETE"}
        ]), "sha256sums_sha256": manifest_sha, "paper_evidence": False,
        "analyzer_eligible": False, "production_authorized": False,
    })
    return manifest_sha


def _validate_hardening_receipt(
    receipt: Mapping[str, Any], context: Mapping[str, Any], components: Path,
    components_sha: str, pair_sha: str, *, source_components_dir: str | None = None,
) -> None:
    required = {
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
    require(
        set(receipt) == required and receipt["schema"] == 1 and receipt["status"] == "complete"
        and receipt["purpose"] == "v4_remote_hardening_rung_integrity_not_paper_evidence"
        and receipt["rung"] == "terminal" and receipt["job_id"] == str(context["job_id"])
        and receipt["legacy_result_dir"] == (source_components_dir or str(components))
        and receipt["legacy_manifest_sha256"] == components_sha
        and receipt["pair_plan_manifest_sha256"] == pair_sha
        and receipt["max_student_updates"] == 1
        and receipt["environment_verified_before"] is True
        and receipt["environment_verified_after"] is True
        and all(receipt[key] is False for key in (
            "paper_evidence", "analyzer_eligible", "production_authorized",
            "endpoint_access_authorized", "cost100_implemented")),
        "terminal hardening receipt drift",
    )


def _validate_sources(provenance: Path) -> None:
    require({item.name for item in provenance.iterdir()} == PROVENANCE_FILES, "Phase-B provenance file-set drift")
    require(sha256(provenance / "assemble_matched_run_v4.py") == FROZEN_ASSEMBLER_SHA, "frozen assembler drift")
    require(sha256(provenance / "development_protocol_v2_tie_aware_draft.json") == PROTOCOL_SHA, "protocol drift")


def validate_package(root: Path) -> str:
    root = canonical_existing(root, directory=True, label="remote-hardening package")
    require({item.name for item in root.iterdir()} == TOP_LEVEL, "package top-level drift")
    manifest_sha, complete = validate_tree(root, "SHA256SUMS", "COMPLETE")
    run = load_json(root / "RUN_MANIFEST.json", "run manifest")
    required = {
        "schema", "status", "purpose", "paper_evidence", "analyzer_eligible",
        "production_authorized", "endpoint_access_authorized", "cost100_implemented",
        "max_student_updates", "performance_values_inspected", "run_id", "arm",
        "job_id", "training_seed", "source_components_dir", "bundle_manifest_sha256",
        "components_manifest_sha256", "phase_b_manifest_sha256",
        "pair_plan_manifest_sha256", "environment_tree_manifest_sha256",
        "environment_tree_receipt_sha256", "submission_receipt_sha256",
        "terminal_receipt_sha256", "fetch_receipt_sha256",
        "gpu_runtime_receipt_sha256", "hardening_receipt_sha256",
        "hardening_sbatch_sha256", "legacy_sbatch_sha256", "phase_b_python_sha256",
        "phase_b_python_version", "local_test_mode",
    }
    require(
        set(run) == required and run["schema"] == 1 and run["status"] == "complete"
        and run["purpose"] == PURPOSE and run["max_student_updates"] == 1
        and run["performance_values_inspected"] is False
        and isinstance(run["local_test_mode"], bool)
        and isinstance(run["source_components_dir"], str)
        and Path(run["source_components_dir"]).is_absolute()
        and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", str(run["phase_b_python_version"])) is not None
        and all(run[key] is False for key in (
            "paper_evidence", "analyzer_eligible", "production_authorized",
            "endpoint_access_authorized", "cost100_implemented")),
        "run manifest drift",
    )
    root_payload_count = len((root / "SHA256SUMS").read_text(encoding="utf-8").splitlines())
    require(complete == {
        "schema": 1, "status": "complete", "run_id": run["run_id"], "arm": run["arm"],
        "file_count": root_payload_count, "sha256sums_sha256": manifest_sha,
        "paper_evidence": False, "analyzer_eligible": False, "production_authorized": False,
    } and root_payload_count > 0, "root completion drift")
    phase_b = root / "phase-b"
    require({item.name for item in phase_b.iterdir()} == PHASE_B_PAYLOADS, "Phase-B top-level drift")
    phase_b_sha, phase_complete = validate_tree(phase_b, "SHA256SUMS", "COMPLETE")
    phase_payload_count = len((phase_b / "SHA256SUMS").read_text(encoding="utf-8").splitlines())
    require(
        run["phase_b_manifest_sha256"] == phase_b_sha
        and phase_complete == {
            "schema": 1, "status": "complete", "file_count": phase_payload_count,
            "sha256sums_sha256": phase_b_sha, "paper_evidence": False,
            "production_authorized": False,
        },
        "Phase-B manifest drift",
    )
    provenance = phase_b / "provenance"
    _validate_sources(provenance)
    components = root / "components"
    components_sha = sha256(components / "SHA256SUMS")
    complete_component, context = _validate_components(
        components, provenance / "development_protocol_v2_tie_aware_draft.json",
        components_sha, provenance / "assemble_matched_run_v4.py",
        local_test_mode=run["local_test_mode"],
    )
    require(
        run["components_manifest_sha256"] == components_sha
        and run["run_id"] == context["run_id"]
        and run["arm"] == context["arm"]
        and run["job_id"] == str(context["job_id"])
        and run["training_seed"] == context["training_seed"]
        and run["bundle_manifest_sha256"] == complete_component["bundle_manifest_sha256"],
        "run/component binding drift",
    )
    pair_module = _load_module(provenance / "pair_plan.py", sha256(provenance / "pair_plan.py"), "pair_plan")
    pair_sha = sha256(phase_b / "pair-plan/SHA256SUMS")
    pair = pair_module.validate(phase_b / "pair-plan", pair_sha)
    require(run["pair_plan_manifest_sha256"] == pair_sha and pair["common"]["bundle_manifest_sha256"] == run["bundle_manifest_sha256"], "run/pair drift")
    hardening = load_json(phase_b / "hardening-receipt.json", "hardening receipt")
    _validate_hardening_receipt(
        hardening, context, components, components_sha, pair_sha,
        source_components_dir=run["source_components_dir"],
    )
    require(
        run["hardening_receipt_sha256"] == sha256(phase_b / "hardening-receipt.json")
        and hardening["bundle_manifest_sha256"] == run["bundle_manifest_sha256"]
        and hardening["environment_tree_manifest_sha256"] == run["environment_tree_manifest_sha256"]
        and hardening["environment_tree_receipt_sha256"] == run["environment_tree_receipt_sha256"],
        "hardening receipt binding drift",
    )
    slurm = _load_module(provenance / "slurm_integrity.py", sha256(provenance / "slurm_integrity.py"), "slurm_integrity")
    # The exact keyset is derived from the copied pair-plan/job-guard contract,
    # never from untrusted receipt contents.
    job_guard = _load_module(provenance / "job_guard.py", sha256(provenance / "job_guard.py"), "job_guard")
    submission_record = load_json(phase_b / "submission.json", "submission")
    submission, inputs = slurm.validate_submission(
        phase_b / "submission.json", phase_b / "input-envelope.nul",
        job_guard.EXPECTED_KEYS["terminal"], job_id=run["job_id"],
        sbatch_path=provenance / "submitted-terminal.sbatch",
        sbatch_sha256=run["hardening_sbatch_sha256"],
        receipt_sbatch_path=submission_record["sbatch_path"],
        receipt_input_envelope_path=submission_record["input_envelope_path"],
    )
    runtime_sha = sha256(phase_b / "gpu-runtime.json")
    terminal = slurm.validate_terminal(
        phase_b / "terminal.tsv", phase_b / "gpu-runtime.json", runtime_sha,
        submission, job_id=run["job_id"],
    )
    validate_fetch(
        phase_b / "fetch.tsv", components, terminal, sha256(phase_b / "terminal.tsv"),
        run["arm"], local_test_mode=run["local_test_mode"],
        receipt_local_path=run["source_components_dir"],
    )
    env_sha, env_complete = validate_tree(phase_b / "environment-closure", "SHA256SUMS", "COMPLETE")
    env_receipt = load_json(phase_b / "environment-closure/receipt.json", "environment receipt")
    require(
        run["environment_tree_manifest_sha256"] == env_sha
        and run["environment_tree_receipt_sha256"] == sha256(phase_b / "environment-closure/receipt.json")
        and env_complete["sha256sums_sha256"] == env_sha
        and env_receipt["installed_file_byte_closure"] is True
        and env_receipt["python_version"] == ("3.10.20" if not run["local_test_mode"] else run["phase_b_python_version"])
        and run["phase_b_python_sha256"] == env_receipt["python_sha256"],
        "Phase-B environment binding drift",
    )
    for field, path in (
        ("submission_receipt_sha256", phase_b / "submission.json"),
        ("terminal_receipt_sha256", phase_b / "terminal.tsv"),
        ("fetch_receipt_sha256", phase_b / "fetch.tsv"),
        ("gpu_runtime_receipt_sha256", phase_b / "gpu-runtime.json"),
    ):
        require(run[field] == sha256(path), f"run receipt binding drift: {field}")
    require(
        inputs["UED_PAIR_PLAN_MANIFEST_SHA256"] == pair_sha
        and inputs["UED_BUNDLE_MANIFEST_SHA256"] == run["bundle_manifest_sha256"]
        and inputs["UED_ENV_TREE_MANIFEST_SHA256"] == run["environment_tree_manifest_sha256"]
        and inputs["UED_ENV_TREE_RECEIPT_SHA256"] == run["environment_tree_receipt_sha256"]
        and inputs["UED_HARDENING_SBATCH_SHA256"] == run["hardening_sbatch_sha256"]
        and inputs["UED_LEGACY_SBATCH_SHA256"] == run["legacy_sbatch_sha256"]
        and inputs["UED_ARM"] == run["arm"],
        "input/run binding drift",
    )
    return manifest_sha


def assemble(cli: argparse.Namespace) -> tuple[Path, str]:
    tool = Path(__file__).resolve()
    require(sha256(tool) == cli.expected_assembler_sha256, "remote-hardening assembler self drift")
    components = canonical_existing(cli.components_dir, directory=True, label="components")
    protocol = canonical_existing(cli.protocol, directory=False, label="protocol")
    frozen_assembler = canonical_existing(cli.frozen_assembler, directory=False, label="frozen assembler")
    pair_plan = canonical_existing(cli.pair_plan_dir, directory=True, label="pair plan")
    environment_closure = canonical_existing(cli.environment_closure, directory=True, label="environment closure")
    bundle = canonical_existing(cli.bundle_dir, directory=True, label="bundle")
    for path, label in ((cli.submission_receipt, "submission"), (cli.input_envelope, "envelope"), (cli.terminal_receipt, "terminal"), (cli.fetch_receipt, "fetch"), (cli.hardening_receipt, "hardening")):
        canonical_existing(path, directory=False, label=label)
    components_sha = sha256(components / "SHA256SUMS")
    require(components_sha == cli.expected_components_manifest_sha256, "component hash drift")
    complete_component, context = _validate_components(
        components, protocol, components_sha, frozen_assembler,
        local_test_mode=cli.local_test_mode,
    )
    pair_module_path = bundle / "ued_benchmark/hopper_v4_remote_hardening/pair_plan.py"
    pair_module = _load_module(pair_module_path, cli.expected_pair_plan_tool_sha256, "pair_plan")
    pair = pair_module.validate(pair_plan, cli.expected_pair_plan_manifest_sha256)
    require(pair["common"]["bundle_manifest_sha256"] == complete_component["bundle_manifest_sha256"], "pair/component bundle drift")
    hardening = load_json(cli.hardening_receipt, "hardening receipt")
    _validate_hardening_receipt(hardening, context, components, components_sha, cli.expected_pair_plan_manifest_sha256)
    slurm_path = bundle / "ued_benchmark/hopper_v4_remote_hardening/slurm_integrity.py"
    slurm = _load_module(slurm_path, cli.expected_slurm_integrity_sha256, "slurm_integrity")
    job_guard_path = bundle / "ued_benchmark/hopper_v4_remote_hardening/job_guard.py"
    job_guard = _load_module(job_guard_path, cli.expected_job_guard_sha256, "job_guard")
    submitted_sbatch = bundle / "hopper/sbatch/ued_minimax_v4_remote_hardened_terminal_chain_smoke.sbatch"
    submission, _inputs = slurm.validate_submission(
        cli.submission_receipt, cli.input_envelope, job_guard.EXPECTED_KEYS["terminal"],
        job_id=str(context["job_id"]), sbatch_path=submitted_sbatch,
        sbatch_sha256=hardening["hardening_sbatch_sha256"],
    )
    runtime_path = Path(hardening["gpu_runtime_receipt_path"])
    canonical_existing(runtime_path, directory=False, label="GPU runtime receipt")
    require(sha256(runtime_path) == hardening["gpu_runtime_receipt_sha256"], "GPU runtime/hardening drift")
    terminal = slurm.validate_terminal(
        cli.terminal_receipt, runtime_path, hardening["gpu_runtime_receipt_sha256"],
        submission, job_id=str(context["job_id"]),
    )
    validate_fetch(cli.fetch_receipt, components, terminal, sha256(cli.terminal_receipt), context["arm"], local_test_mode=cli.local_test_mode)
    output = canonical_new(cli.output_dir, [components, pair_plan, environment_closure, bundle])
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _copy_tree(components, temporary / "components")
        phase_b = temporary / "phase-b"; phase_b.mkdir()
        for source, name in (
            (cli.submission_receipt, "submission.json"), (cli.input_envelope, "input-envelope.nul"),
            (cli.terminal_receipt, "terminal.tsv"), (cli.fetch_receipt, "fetch.tsv"),
            (runtime_path, "gpu-runtime.json"), (cli.hardening_receipt, "hardening-receipt.json"),
        ):
            shutil.copy2(source, phase_b / name, follow_symlinks=False)
        _copy_tree(pair_plan, phase_b / "pair-plan")
        _copy_tree(environment_closure, phase_b / "environment-closure")
        provenance = phase_b / "provenance"; provenance.mkdir()
        sources = {
            "development_protocol_v2_tie_aware_draft.json": protocol,
            "assemble_matched_run_v4.py": frozen_assembler,
            "evaluate_matched_terminal_v4.py": frozen_assembler.parent / "evaluate_matched_terminal_v4.py",
            "run_matched_terminal_v4.py": frozen_assembler.parent / "run_matched_terminal_v4.py",
            "run_terminal_phase_a_v4.py": frozen_assembler.parent / "run_terminal_phase_a_v4.py",
            "assemble_remote_hardened.py": tool,
            "slurm_integrity.py": slurm_path,
            "pair_plan.py": pair_module_path,
            "environment_tree.py": bundle / "ued_benchmark/hopper_v4_remote_hardening/environment_tree.py",
            "gpu_runtime_probe.py": bundle / "ued_benchmark/hopper_v4_remote_hardening/gpu_runtime_probe.py",
            "job_guard.py": job_guard_path,
            "finalize_remote_hardened.py": cli.finalizer,
            "submitted-terminal.sbatch": submitted_sbatch,
            "REMOTE_HARDENING_STATE.json": bundle / "REMOTE_HARDENING_STATE.json",
        }
        require(set(sources) == PROVENANCE_FILES, "provenance source-set drift")
        for name, source in sources.items():
            canonical_existing(source, directory=False, label=f"provenance {name}")
            shutil.copy2(source, provenance / name, follow_symlinks=False)
        phase_b_sha = _phase_b_complete(phase_b)
        run = {
            "schema": 1, "status": "complete", "purpose": PURPOSE,
            "paper_evidence": False, "analyzer_eligible": False,
            "production_authorized": False, "endpoint_access_authorized": False,
            "cost100_implemented": False, "max_student_updates": 1,
            "performance_values_inspected": False, "run_id": context["run_id"],
            "arm": context["arm"], "job_id": str(context["job_id"]),
            "training_seed": context["training_seed"],
            "source_components_dir": str(components),
            "bundle_manifest_sha256": complete_component["bundle_manifest_sha256"],
            "components_manifest_sha256": components_sha,
            "phase_b_manifest_sha256": phase_b_sha,
            "pair_plan_manifest_sha256": cli.expected_pair_plan_manifest_sha256,
            "environment_tree_manifest_sha256": cli.expected_environment_manifest_sha256,
            "environment_tree_receipt_sha256": cli.expected_environment_receipt_sha256,
            "submission_receipt_sha256": sha256(cli.submission_receipt),
            "terminal_receipt_sha256": sha256(cli.terminal_receipt),
            "fetch_receipt_sha256": sha256(cli.fetch_receipt),
            "gpu_runtime_receipt_sha256": sha256(runtime_path),
            "hardening_receipt_sha256": sha256(cli.hardening_receipt),
            "hardening_sbatch_sha256": hardening["hardening_sbatch_sha256"],
            "legacy_sbatch_sha256": hardening["legacy_sbatch_sha256"],
            "phase_b_python_sha256": cli.phase_b_python_sha256,
            "phase_b_python_version": cli.phase_b_python_version,
            "local_test_mode": cli.local_test_mode,
        }
        _write_json(temporary / "RUN_MANIFEST.json", run)
        manifest_sha = _root_complete(temporary, str(context["run_id"]), context["arm"])
        require(validate_package(temporary) == manifest_sha, "temporary package validation drift")
        os.replace(temporary, output)
        require(validate_package(output) == manifest_sha, "published package validation drift")
        return output, manifest_sha
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-dir", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--frozen-assembler", type=Path)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--pair-plan-dir", type=Path)
    parser.add_argument("--environment-closure", type=Path)
    parser.add_argument("--submission-receipt", type=Path)
    parser.add_argument("--input-envelope", type=Path)
    parser.add_argument("--terminal-receipt", type=Path)
    parser.add_argument("--fetch-receipt", type=Path)
    parser.add_argument("--hardening-receipt", type=Path)
    parser.add_argument("--finalizer", type=Path)
    parser.add_argument("--expected-components-manifest-sha256")
    parser.add_argument("--expected-pair-plan-manifest-sha256")
    parser.add_argument("--expected-pair-plan-tool-sha256")
    parser.add_argument("--expected-environment-manifest-sha256")
    parser.add_argument("--expected-environment-receipt-sha256")
    parser.add_argument("--expected-slurm-integrity-sha256")
    parser.add_argument("--expected-job-guard-sha256")
    parser.add_argument("--expected-assembler-sha256")
    parser.add_argument("--phase-b-python-sha256")
    parser.add_argument("--phase-b-python-version")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", type=Path)
    parser.add_argument("--local-test-mode", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    cli = parse_cli(argv)
    try:
        if cli.validate_only is not None:
            require(not any(value is not None for key, value in vars(cli).items() if key not in {"validate_only", "local_test_mode"}), "validate-only argument drift")
            digest = validate_package(cli.validate_only)
            print(f"V4H_PACKAGE_VALID manifest={digest}")
            return 0
        required = [value for key, value in vars(cli).items() if key not in {"validate_only", "local_test_mode"}]
        require(all(value is not None for value in required), "assembly arguments incomplete")
        output, digest = assemble(cli)
    except (AssemblyError, OSError, UnicodeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"V4H_ASSEMBLY_REFUSED: {exc}", file=os.sys.stderr)
        return 1
    print(f"V4H_ASSEMBLY_COMPLETE path={output} manifest={digest} paper_evidence=false analyzer_eligible=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
