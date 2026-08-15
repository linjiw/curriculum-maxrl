#!/usr/bin/env python3
"""Post-terminal Phase B for the sibling-only v4 remote-hardening lane.

This program performs no scheduler, SSH, fetch, endpoint, analyzer, cost100,
or submission action.  It runs only under the exact byte-closed Phase-B
Python and consumes already captured/fetched receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import pwd
from types import ModuleType
from typing import Sequence


HASH_RE_LENGTH = 64


class FinalizationError(RuntimeError):
    """Raised when Phase B is not exact, isolated, and terminal-gated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizationError(message)


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"unsafe regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, expected: str, name: str) -> ModuleType:
    require(len(expected) == HASH_RE_LENGTH and sha256(path) == expected, f"{name} drift")
    specification = importlib.util.spec_from_file_location(f"v4h_phase_b_{name}", path)
    require(specification is not None and specification.loader is not None, f"cannot load {name}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def run(cli: argparse.Namespace) -> tuple[Path, str]:
    finalizer = Path(__file__).resolve()
    require(sha256(finalizer) == cli.expected_finalizer_sha256, "finalizer self drift")
    environment = cli.phase_b_environment
    python = environment / "bin/python"
    require(
        environment.is_absolute() and ".." not in environment.parts
        and environment.resolve(strict=True) == environment
        and environment.is_dir() and not environment.is_symlink()
        and python.is_file() and os.access(python, os.X_OK)
        and python.resolve(strict=True) == Path(os.sys.executable).resolve(strict=True),
        "running interpreter is not the exact Phase-B environment Python",
    )
    expected_version = platform.python_version() if cli.local_test_mode else "3.10.20"
    require(platform.python_version() == expected_version, "Phase-B Python version drift")
    user = pwd.getpwuid(os.getuid()).pw_name
    conda = cli.local_conda if cli.local_test_mode else Path(f"/home/{user}/miniconda3/bin/conda")
    require(conda is not None, "Phase-B Conda identity missing")
    bundle = cli.bundle_dir
    tree_path = bundle / "ued_benchmark/hopper_v4_remote_hardening/environment_tree.py"
    tree = load_module(tree_path, cli.expected_environment_tree_tool_sha256, "environment_tree")
    environment_receipt, environment_manifest = tree.verify(
        environment, conda, cli.environment_closure,
        cli.expected_environment_tree_tool_sha256, expected_version,
        cli.expected_environment_manifest_sha256,
        cli.expected_environment_receipt_sha256,
    )
    require(
        environment_manifest == cli.expected_environment_manifest_sha256
        and environment_receipt["python_sha256"] == sha256(python.resolve(strict=True)),
        "Phase-B installed-byte closure drift",
    )
    assembler_path = bundle / "ued_benchmark/hopper_v4_remote_hardening/assemble_remote_hardened.py"
    assembler = load_module(assembler_path, cli.expected_assembler_sha256, "assembler")
    assembly_cli = argparse.Namespace(
        components_dir=cli.components_dir,
        protocol=cli.protocol,
        frozen_assembler=cli.frozen_assembler,
        bundle_dir=bundle,
        pair_plan_dir=cli.pair_plan_dir,
        environment_closure=cli.environment_closure,
        submission_receipt=cli.submission_receipt,
        input_envelope=cli.input_envelope,
        terminal_receipt=cli.terminal_receipt,
        fetch_receipt=cli.fetch_receipt,
        hardening_receipt=cli.hardening_receipt,
        finalizer=finalizer,
        expected_components_manifest_sha256=cli.expected_components_manifest_sha256,
        expected_pair_plan_manifest_sha256=cli.expected_pair_plan_manifest_sha256,
        expected_pair_plan_tool_sha256=cli.expected_pair_plan_tool_sha256,
        expected_environment_manifest_sha256=cli.expected_environment_manifest_sha256,
        expected_environment_receipt_sha256=cli.expected_environment_receipt_sha256,
        expected_slurm_integrity_sha256=cli.expected_slurm_integrity_sha256,
        expected_job_guard_sha256=cli.expected_job_guard_sha256,
        expected_assembler_sha256=cli.expected_assembler_sha256,
        phase_b_python_sha256=environment_receipt["python_sha256"],
        phase_b_python_version=expected_version,
        output_dir=cli.output_dir,
        validate_only=None,
        local_test_mode=cli.local_test_mode,
    )
    return assembler.assemble(assembly_cli)


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--frozen-assembler", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--pair-plan-dir", type=Path, required=True)
    parser.add_argument("--phase-b-environment", type=Path, required=True)
    parser.add_argument("--environment-closure", type=Path, required=True)
    parser.add_argument("--submission-receipt", type=Path, required=True)
    parser.add_argument("--input-envelope", type=Path, required=True)
    parser.add_argument("--terminal-receipt", type=Path, required=True)
    parser.add_argument("--fetch-receipt", type=Path, required=True)
    parser.add_argument("--hardening-receipt", type=Path, required=True)
    parser.add_argument("--expected-components-manifest-sha256", required=True)
    parser.add_argument("--expected-pair-plan-manifest-sha256", required=True)
    parser.add_argument("--expected-pair-plan-tool-sha256", required=True)
    parser.add_argument("--expected-environment-manifest-sha256", required=True)
    parser.add_argument("--expected-environment-receipt-sha256", required=True)
    parser.add_argument("--expected-environment-tree-tool-sha256", required=True)
    parser.add_argument("--expected-slurm-integrity-sha256", required=True)
    parser.add_argument("--expected-job-guard-sha256", required=True)
    parser.add_argument("--expected-assembler-sha256", required=True)
    parser.add_argument("--expected-finalizer-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-test-mode", action="store_true")
    parser.add_argument("--local-conda", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        cli = parse_cli(argv)
        require((cli.local_test_mode and cli.local_conda is not None) or (not cli.local_test_mode and cli.local_conda is None), "local Conda override policy drift")
        output, digest = run(cli)
    except (FinalizationError, OSError, UnicodeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"V4H_FINALIZATION_REFUSED: {exc}", file=os.sys.stderr)
        return 1
    print(f"V4H_FINALIZATION_COMPLETE path={output} manifest={digest} paper_evidence=false analyzer_eligible=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
