"""Verify the content-addressed research and patched-runtime E2c code."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_beneath(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"manifest path is not relative and contained: {relative}")
    root = root.resolve()
    path = (root / relative_path).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"manifest path escapes root: {relative}")
    return path


def verify_code_manifest(
    manifest_path: Path,
    research_root: Path,
    runtime_root: Path,
) -> dict:
    manifest_path = manifest_path.resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported E2c code-manifest schema")
    expected_commit = "7197bbb46a2ecd866da52f6b401ff20a34fe9390"
    if manifest.get("maxrl_commit") != expected_commit:
        raise ValueError("E2c code manifest has the wrong MaxRL commit")

    roots = {
        "research": research_root.resolve(),
        "runtime_maxrl": (runtime_root / "maxrl").resolve(),
    }
    verified = []
    seen = set()
    for item in manifest.get("files", []):
        root_name = str(item.get("root"))
        if root_name not in roots:
            raise ValueError(f"unknown manifest root: {root_name}")
        relative = str(item.get("path"))
        key = (root_name, relative)
        if key in seen:
            raise ValueError(f"duplicate manifest entry: {key}")
        seen.add(key)
        path = _resolve_beneath(roots[root_name], relative)
        if not path.is_file():
            raise ValueError(f"manifest file is missing: {path}")
        size = path.stat().st_size
        digest = sha256(path)
        if size != int(item.get("bytes", -1)):
            raise ValueError(f"manifest size mismatch: {path}")
        if digest != item.get("sha256"):
            raise ValueError(f"manifest SHA-256 mismatch: {path}")
        verified.append({
            "root": root_name,
            "path": relative,
            "bytes": size,
            "sha256": digest,
        })
    if not verified:
        raise ValueError("E2c code manifest contains no files")
    import flash_attn
    import hydra
    import numpy
    import pandas
    import pyarrow
    import ray
    import torch
    import transformers
    import verl

    actual_environment = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "flash_attn": flash_attn.__version__,
        "ray": ray.__version__,
        "hydra_core": hydra.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
        "numpy": numpy.__version__,
        "verl": getattr(verl, "__version__", "unknown"),
    }
    expected_environment = manifest.get("environment", {})
    if actual_environment != expected_environment:
        differences = {
            key: {
                "expected": expected_environment.get(key),
                "actual": actual_environment.get(key),
            }
            for key in sorted(set(expected_environment) |
                              set(actual_environment))
            if expected_environment.get(key) != actual_environment.get(key)
        }
        raise ValueError(f"E2c environment drift: {differences}")
    return {
        "status": "pass",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "maxrl_commit": expected_commit,
        "files_verified": len(verified),
        "files": verified,
        "environment": actual_environment,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--research-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    args = parser.parse_args()
    report = verify_code_manifest(
        Path(args.manifest), Path(args.research_root), Path(args.runtime_root))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
