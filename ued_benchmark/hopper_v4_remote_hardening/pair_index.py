#!/usr/bin/env python3
"""Bind the two separate arm packages to one exact R1/R2 pair plan."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence


PURPOSE = "v4_remote_hardening_two_arm_pair_index_not_paper_evidence"


class PairIndexError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PairIndexError(message)


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"unsafe file: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def load(path: Path, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate {label} key: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def canonical(path: Path, label: str) -> Path:
    require(path.is_absolute() and ".." not in path.parts and path.resolve(strict=True) == path and path.is_dir() and not path.is_symlink(), f"unsafe {label}")
    return path


def module(path: Path, expected: str):
    require(sha256(path) == expected, "package tool drift")
    spec = importlib.util.spec_from_file_location("v4h_pair_index_package", path)
    require(spec is not None and spec.loader is not None, "cannot load package tool")
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def validate(root: Path) -> str:
    root = canonical(root, "pair index")
    require({item.name for item in root.iterdir()} == {"PAIR_INDEX.json", "SHA256SUMS", "COMPLETE"}, "pair-index closure drift")
    expected = f"{sha256(root / 'PAIR_INDEX.json')}  PAIR_INDEX.json\n"
    require((root / "SHA256SUMS").read_text(encoding="utf-8") == expected, "pair-index manifest drift")
    manifest_sha = sha256(root / "SHA256SUMS")
    index = load(root / "PAIR_INDEX.json", "pair index")
    required = {
        "schema", "status", "purpose", "paper_evidence", "analyzer_eligible",
        "production_authorized", "endpoint_access_authorized", "cost100_implemented",
        "max_student_updates", "pair_id", "pair_plan_manifest_sha256",
        "bundle_manifest_sha256", "training_seed", "arms", "package_tool_sha256",
        "pair_index_tool_sha256",
    }
    require(
        set(index) == required and index["schema"] == 1 and index["status"] == "complete"
        and index["purpose"] == PURPOSE and index["max_student_updates"] == 1
        and index["training_seed"] == 101 and set(index["arms"]) == {"frontier", "maxmc"}
        and all(index[key] is False for key in (
            "paper_evidence", "analyzer_eligible", "production_authorized",
            "endpoint_access_authorized", "cost100_implemented")),
        "pair-index semantics drift",
    )
    for arm, record in index["arms"].items():
        require(
            isinstance(record, dict) and set(record) == {
                "package_path", "package_manifest_sha256", "run_manifest_sha256",
                "run_id", "job_id",
            }
            and Path(record["package_path"]).is_absolute()
            and isinstance(record["job_id"], str) and record["job_id"],
            f"pair-index {arm} record drift",
        )
    package_tool_path = Path(__file__).resolve().parent / "assemble_remote_hardened.py"
    package_tool = module(package_tool_path, index["package_tool_sha256"])
    pair_ids: set[str] = set()
    bundles: set[str] = set()
    for arm, record in index["arms"].items():
        package = canonical(Path(record["package_path"]), f"indexed {arm} package")
        require(
            package_tool.validate_package(package) == record["package_manifest_sha256"]
            and sha256(package / "RUN_MANIFEST.json") == record["run_manifest_sha256"],
            f"indexed {arm} package drift",
        )
        run = load(package / "RUN_MANIFEST.json", f"indexed {arm} run")
        pair = load(package / "phase-b/pair-plan/PAIR_PLAN.json", f"indexed {arm} pair")
        require(
            run["arm"] == arm and run["run_id"] == record["run_id"]
            and run["job_id"] == record["job_id"]
            and run["pair_plan_manifest_sha256"] == index["pair_plan_manifest_sha256"],
            f"indexed {arm} semantic drift",
        )
        pair_ids.add(pair["pair_id"]); bundles.add(run["bundle_manifest_sha256"])
    require(
        pair_ids == {index["pair_id"]}
        and bundles == {index["bundle_manifest_sha256"]}
        and index["arms"]["frontier"]["job_id"] != index["arms"]["maxmc"]["job_id"],
        "indexed cross-arm common provenance drift",
    )
    complete = load(root / "COMPLETE", "pair-index completion")
    require(complete == {
        "schema": 1, "status": "complete", "pair_id": index["pair_id"],
        "sha256sums_sha256": manifest_sha, "file_count": 1,
        "paper_evidence": False, "production_authorized": False,
    }, "pair-index completion drift")
    return manifest_sha


def create(cli: argparse.Namespace) -> tuple[Path, str]:
    tool = Path(__file__).resolve()
    require(sha256(tool) == cli.expected_pair_index_tool_sha256, "pair-index tool drift")
    package_tool = module(cli.package_tool, cli.expected_package_tool_sha256)
    packages = {
        "frontier": canonical(cli.frontier_package, "Frontier package"),
        "maxmc": canonical(cli.maxmc_package, "MaxMC package"),
    }
    arms: dict[str, dict[str, str]] = {}
    pair_id = None
    bundle = None
    for arm, package in packages.items():
        package_manifest = package_tool.validate_package(package)
        run = load(package / "RUN_MANIFEST.json", f"{arm} run manifest")
        require(run["arm"] == arm and run["training_seed"] == 101, f"{arm} run identity drift")
        require(run["pair_plan_manifest_sha256"] == cli.expected_pair_plan_manifest_sha256, f"{arm} pair-plan drift")
        pair_plan = load(package / "phase-b/pair-plan/PAIR_PLAN.json", f"{arm} pair plan")
        require(
            isinstance(pair_plan["pair_id"], str)
            and len(pair_plan["pair_id"]) == 20
            and all(character in "0123456789abcdef" for character in pair_plan["pair_id"]),
            "pair ID drift",
        )
        if pair_id is None:
            pair_id = pair_plan["pair_id"]; bundle = run["bundle_manifest_sha256"]
        require(pair_plan["pair_id"] == pair_id and run["bundle_manifest_sha256"] == bundle, "cross-arm common provenance drift")
        arms[arm] = {
            "package_path": str(package), "package_manifest_sha256": package_manifest,
            "run_manifest_sha256": sha256(package / "RUN_MANIFEST.json"),
            "run_id": run["run_id"], "job_id": run["job_id"],
        }
    require(arms["frontier"]["job_id"] != arms["maxmc"]["job_id"], "arm jobs must be separate non-array submissions")
    parent = canonical(cli.output_parent, "pair-index output parent")
    output = parent / f"{pair_id}-v4h-pair"
    require(not output.exists() and not output.is_symlink(), "pair-index output exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        index = {
            "schema": 1, "status": "complete", "purpose": PURPOSE,
            "paper_evidence": False, "analyzer_eligible": False,
            "production_authorized": False, "endpoint_access_authorized": False,
            "cost100_implemented": False, "max_student_updates": 1,
            "pair_id": pair_id, "pair_plan_manifest_sha256": cli.expected_pair_plan_manifest_sha256,
            "bundle_manifest_sha256": bundle, "training_seed": 101, "arms": arms,
            "package_tool_sha256": cli.expected_package_tool_sha256,
            "pair_index_tool_sha256": cli.expected_pair_index_tool_sha256,
        }
        (temporary / "PAIR_INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (temporary / "SHA256SUMS").write_text(f"{sha256(temporary / 'PAIR_INDEX.json')}  PAIR_INDEX.json\n", encoding="utf-8")
        manifest_sha = sha256(temporary / "SHA256SUMS")
        (temporary / "COMPLETE").write_text(json.dumps({
            "schema": 1, "status": "complete", "pair_id": pair_id,
            "sha256sums_sha256": manifest_sha, "file_count": 1,
            "paper_evidence": False, "production_authorized": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        require(validate(output) == manifest_sha, "published pair-index drift")
        return output, manifest_sha
    finally:
        if temporary.exists(): shutil.rmtree(temporary)


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("create")
    create_parser.add_argument("--frontier-package", type=Path, required=True)
    create_parser.add_argument("--maxmc-package", type=Path, required=True)
    create_parser.add_argument("--package-tool", type=Path, required=True)
    create_parser.add_argument("--expected-package-tool-sha256", required=True)
    create_parser.add_argument("--expected-pair-index-tool-sha256", required=True)
    create_parser.add_argument("--expected-pair-plan-manifest-sha256", required=True)
    create_parser.add_argument("--output-parent", type=Path, required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--pair-index-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    cli = parse_cli(argv)
    try:
        output, digest = create(cli) if cli.command == "create" else (cli.pair_index_dir, validate(cli.pair_index_dir))
    except (PairIndexError, OSError, UnicodeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"V4H_PAIR_INDEX_REFUSED: {exc}", file=os.sys.stderr); return 1
    print(f"V4H_PAIR_INDEX_COMPLETE path={output} manifest={digest} paper_evidence=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
