#!/usr/bin/env python3
"""Create or verify a complete byte-level Conda-prefix closure.

This helper is intentionally scheduler- and network-free.  The closure lives
outside the environment so creating it cannot perturb the tree it records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence


HASH_RE = re.compile(r"^[0-9a-f]{64}$")
PAYLOADS = ("environment-tree.jsonl", "receipt.json")


class EnvironmentClosureError(RuntimeError):
    """Raised when an installed environment is not exactly closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EnvironmentClosureError(message)


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"unsafe regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_existing(path: Path, *, directory: bool, label: str) -> Path:
    require(path.is_absolute() and ".." not in path.parts, f"{label} must be absolute")
    resolved = path.resolve(strict=True)
    require(resolved == path, f"{label} must be canonical and non-symbolic")
    require((path.is_dir() if directory else path.is_file()), f"missing {label}")
    require(not path.is_symlink(), f"symbolic {label} forbidden")
    return path


def _canonical_new(path: Path, label: str) -> Path:
    require(path.is_absolute() and ".." not in path.parts, f"{label} must be absolute")
    require(path.name not in {"", ".", ".."}, f"unsafe {label} basename")
    require(not path.exists() and not path.is_symlink(), f"{label} exists")
    parent = _canonical_existing(path.parent, directory=True, label=f"{label} parent")
    require(parent / path.name == path, f"noncanonical {label}")
    return path


def _mode(path: Path, *, follow_symlinks: bool) -> str:
    return f"{stat.S_IMODE(path.stat(follow_symlinks=follow_symlinks).st_mode):04o}"


def _safe_relative(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    parsed = PurePosixPath(relative)
    require(
        relative not in {"", "."}
        and not parsed.is_absolute()
        and all(part not in {"", ".", ".."} for part in parsed.parts),
        f"unsafe environment path: {relative!r}",
    )
    return relative


def inventory(environment: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    environment = _canonical_existing(
        environment, directory=True, label="environment"
    )
    entries: list[dict[str, Any]] = []
    counts = {"directory_count": 0, "file_count": 0, "symlink_count": 0,
              "regular_file_bytes": 0}
    for path in sorted(environment.rglob("*"), key=lambda item: item.relative_to(environment).as_posix()):
        relative = _safe_relative(path, environment)
        if path.is_symlink():
            target = os.readlink(path)
            require("\x00" not in target and target != "", f"unsafe symlink target: {relative}")
            resolved = path.resolve(strict=True)
            require(
                resolved.is_relative_to(environment),
                f"environment symlink escapes prefix: {relative} -> {target}",
            )
            entries.append({
                "kind": "symlink",
                "path": relative,
                "target": target,
                "resolved_path": resolved.relative_to(environment).as_posix(),
            })
            counts["symlink_count"] += 1
        elif path.is_file():
            size = path.stat().st_size
            entries.append({
                "kind": "file",
                "mode": _mode(path, follow_symlinks=False),
                "path": relative,
                "sha256": sha256(path),
                "size": size,
            })
            counts["file_count"] += 1
            counts["regular_file_bytes"] += size
        elif path.is_dir():
            entries.append({
                "kind": "directory",
                "mode": _mode(path, follow_symlinks=False),
                "path": relative,
            })
            counts["directory_count"] += 1
        else:
            raise EnvironmentClosureError(f"special environment entry forbidden: {relative}")
    require(entries and counts["file_count"] > 0, "environment inventory is empty")
    return entries, counts


def manifest_text(environment: Path) -> tuple[str, dict[str, int]]:
    entries, counts = inventory(environment)
    header = {
        "entry_count": len(entries),
        "environment": str(environment),
        "environment_mode": _mode(environment, follow_symlinks=False),
        "hash_algorithm": "sha256",
        "schema": 1,
        "symlink_policy": "literal-target-plus-contained-resolution",
    }
    values: Iterable[Mapping[str, Any]] = (header, *entries)
    return "".join(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for value in values
    ), counts


def _python_version(python: Path) -> str:
    completed = subprocess.run(
        [str(python), "-I", "-B", "-c", "import platform;print(platform.python_version())"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    require(completed.returncode == 0, "environment Python version probe failed")
    value = completed.stdout.strip()
    require(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value) is not None, "bad Python version")
    return value


def _atomic_text(path: Path, text: str) -> None:
    require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _validate_closure_tree(root: Path) -> tuple[dict[str, Any], str]:
    root = _canonical_existing(root, directory=True, label="environment closure")
    require(
        {entry.name for entry in root.iterdir()}
        == {"environment-tree.jsonl", "receipt.json", "SHA256SUMS", "COMPLETE"},
        "environment closure top-level drift",
    )
    for path in root.rglob("*"):
        require(not path.is_symlink(), "environment closure contains a symlink")
        require(path.is_file(), "environment closure contains a non-file entry")
    manifest = root / "SHA256SUMS"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    require(len(lines) == len(PAYLOADS), "environment closure manifest cardinality drift")
    listed: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        require(match is not None, "environment closure manifest row drift")
        digest, name = match.groups()
        require(name in PAYLOADS and name not in listed, "environment closure manifest path drift")
        listed[name] = digest
    require(set(listed) == set(PAYLOADS), "environment closure manifest payload drift")
    for name, digest in listed.items():
        require(sha256(root / name) == digest, f"environment closure payload drift: {name}")
    manifest_sha = sha256(manifest)
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            require(key not in value, f"duplicate closure JSON key: {key}")
            value[key] = item
        return value
    receipt = json.loads(
        (root / "receipt.json").read_text(encoding="utf-8"),
        object_pairs_hook=unique,
    )
    complete = json.loads(
        (root / "COMPLETE").read_text(encoding="utf-8"),
        object_pairs_hook=unique,
    )
    require(
        complete == {
            "file_count": len(PAYLOADS),
            "schema": 1,
            "sha256sums_sha256": manifest_sha,
            "status": "complete",
        },
        "environment closure completion drift",
    )
    return receipt, manifest_sha


def create(
    environment: Path,
    conda: Path,
    output: Path,
    expected_tool_sha256: str,
    expected_python_version: str,
) -> tuple[Path, str, str]:
    tool = Path(__file__).resolve()
    require(HASH_RE.fullmatch(expected_tool_sha256) is not None, "bad tool hash")
    require(sha256(tool) == expected_tool_sha256, "environment-tree tool drift")
    environment = _canonical_existing(environment, directory=True, label="environment")
    require(conda.is_absolute() and ".." not in conda.parts, "Conda path must be absolute")
    require(conda.is_file() and os.access(conda, os.X_OK), "fixed Conda executable missing")
    conda_resolved = conda.resolve(strict=True)
    require(conda_resolved == conda and conda_resolved.is_file(), "Conda executable must be canonical")
    output = _canonical_new(output, "environment closure output")
    python = environment / "bin" / "python"
    require(python.is_file() and os.access(python, os.X_OK), "environment Python missing")
    python_resolved = python.resolve(strict=True)
    version = _python_version(python)
    require(version == expected_python_version, "environment Python version drift")
    text, counts = manifest_text(environment)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _atomic_text(temporary / "environment-tree.jsonl", text)
        receipt = {
            "schema": 1,
            "status": "complete",
            "installed_file_byte_closure": True,
            "environment": str(environment),
            "environment_tree_sha256": sha256(temporary / "environment-tree.jsonl"),
            **counts,
            "python_path": str(python),
            "python_resolved_path": str(python_resolved),
            "python_sha256": sha256(python_resolved),
            "python_version": version,
            "conda_path": str(conda),
            "conda_resolved_path": str(conda_resolved),
            "conda_sha256": sha256(conda_resolved),
            "generator_path": str(tool),
            "generator_sha256": expected_tool_sha256,
            "paper_evidence": False,
            "production_authorized": False,
        }
        _atomic_json(temporary / "receipt.json", receipt)
        _atomic_text(
            temporary / "SHA256SUMS",
            "".join(f"{sha256(temporary / name)}  {name}\n" for name in PAYLOADS),
        )
        manifest_sha = sha256(temporary / "SHA256SUMS")
        _atomic_json(temporary / "COMPLETE", {
            "file_count": len(PAYLOADS), "schema": 1,
            "sha256sums_sha256": manifest_sha, "status": "complete",
        })
        _validate_closure_tree(temporary)
        os.replace(temporary, output)
        verified, verified_manifest = verify(
            environment, conda, output, expected_tool_sha256,
            expected_python_version, manifest_sha, sha256(output / "receipt.json"),
        )
        require(verified_manifest == manifest_sha, "post-publish closure drift")
        return output, manifest_sha, sha256(output / "receipt.json")
    finally:
        if temporary.exists():
            import shutil
            shutil.rmtree(temporary)


def verify(
    environment: Path,
    conda: Path,
    closure: Path,
    expected_tool_sha256: str,
    expected_python_version: str,
    expected_manifest_sha256: str,
    expected_receipt_sha256: str,
) -> tuple[dict[str, Any], str]:
    tool = Path(__file__).resolve()
    for value, label in (
        (expected_tool_sha256, "tool"),
        (expected_manifest_sha256, "closure manifest"),
        (expected_receipt_sha256, "closure receipt"),
    ):
        require(HASH_RE.fullmatch(value or "") is not None, f"bad expected {label} hash")
    require(sha256(tool) == expected_tool_sha256, "environment-tree tool drift")
    environment = _canonical_existing(environment, directory=True, label="environment")
    require(
        conda.is_absolute() and ".." not in conda.parts
        and conda.resolve(strict=True) == conda
        and conda.is_file() and not conda.is_symlink()
        and os.access(conda, os.X_OK),
        "fixed canonical Conda executable missing",
    )
    receipt, manifest_sha = _validate_closure_tree(closure)
    require(manifest_sha == expected_manifest_sha256, "environment closure manifest drift")
    require(sha256(closure / "receipt.json") == expected_receipt_sha256, "environment closure receipt drift")
    required = {
        "schema", "status", "installed_file_byte_closure", "environment",
        "environment_tree_sha256", "directory_count", "file_count",
        "symlink_count", "regular_file_bytes", "python_path",
        "python_resolved_path", "python_sha256", "python_version", "conda_path",
        "conda_resolved_path", "conda_sha256", "generator_path",
        "generator_sha256", "paper_evidence", "production_authorized",
    }
    require(isinstance(receipt, dict) and set(receipt) == required, "environment receipt keys drift")
    python = environment / "bin" / "python"
    require(
        receipt["schema"] == 1 and receipt["status"] == "complete"
        and receipt["installed_file_byte_closure"] is True
        and receipt["environment"] == str(environment)
        and receipt["python_path"] == str(python)
        and receipt["python_resolved_path"] == str(python.resolve(strict=True))
        and receipt["python_sha256"] == sha256(python.resolve(strict=True))
        and receipt["python_version"] == expected_python_version
        and receipt["conda_path"] == str(conda)
        and receipt["conda_resolved_path"] == str(conda.resolve(strict=True))
        and receipt["conda_sha256"] == sha256(conda.resolve(strict=True))
        and receipt["generator_path"] == str(tool)
        and receipt["generator_sha256"] == expected_tool_sha256
        and receipt["paper_evidence"] is False
        and receipt["production_authorized"] is False,
        "environment receipt semantics drift",
    )
    text, counts = manifest_text(environment)
    require(
        hashlib.sha256(text.encode("utf-8")).hexdigest()
        == receipt["environment_tree_sha256"]
        == sha256(closure / "environment-tree.jsonl")
        and (closure / "environment-tree.jsonl").read_text(encoding="utf-8") == text,
        "installed environment byte tree drift",
    )
    for key, value in counts.items():
        require(receipt[key] == value, f"environment receipt count drift: {key}")
    require(_python_version(python) == expected_python_version, "runtime Python version drift")
    return receipt, manifest_sha


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--environment", type=Path, required=True)
        child.add_argument("--conda", type=Path, required=True)
        child.add_argument("--closure", type=Path, required=True)
        child.add_argument("--expected-tool-sha256", required=True)
        child.add_argument("--expected-python-version", required=True)
        if command == "verify":
            child.add_argument("--expected-manifest-sha256", required=True)
            child.add_argument("--expected-receipt-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    cli = parse_cli(argv)
    try:
        if cli.command == "create":
            output, manifest_sha, receipt_sha = create(
                cli.environment, cli.conda, cli.closure,
                cli.expected_tool_sha256, cli.expected_python_version,
            )
            print(
                "V4H_ENVIRONMENT_CLOSURE_COMPLETE "
                f"path={output} manifest={manifest_sha} receipt={receipt_sha}"
            )
        else:
            _receipt, manifest_sha = verify(
                cli.environment, cli.conda, cli.closure,
                cli.expected_tool_sha256, cli.expected_python_version,
                cli.expected_manifest_sha256, cli.expected_receipt_sha256,
            )
            print(f"V4H_ENVIRONMENT_CLOSURE_VALID manifest={manifest_sha}")
    except (EnvironmentClosureError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        print(f"V4H_ENVIRONMENT_CLOSURE_REFUSED: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
