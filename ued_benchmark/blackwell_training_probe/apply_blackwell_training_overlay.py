#!/usr/bin/env python3
"""Fail-closed applicator for the isolated JAX 0.6 training compatibility layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "MODERNIZATION_CONTRACT.json"
APPLIED_MANIFEST = ".blackwell_training_overlay.json"


class OverlayError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def atomic_write(path: Path, payload: bytes) -> None:
    mode = path.stat().st_mode
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def expected_manifest(contract: dict, contract_sha: str) -> dict:
    return {
        "schema_version": 1,
        "overlay": contract["overlay"],
        "contract_sha256": contract_sha,
        "parent_overlay": contract["parent_overlay"],
        "parent_overlay_contract_sha256": contract["parent_overlay_contract_sha256"],
        "parent_applied_manifest_sha256": contract["parent_applied_manifest_sha256"],
        "upstream_commit": contract["upstream_commit"],
        "total_replacements": contract["total_replacements"],
        "file_sha256": {
            name: details["applied_sha256"]
            for name, details in contract["files"].items()
        },
        "paper_evidence": False,
    }


def inspect(source: Path, contract: dict, contract_sha: str) -> tuple[str, dict]:
    if git(source, "rev-parse", "HEAD") != contract["upstream_commit"]:
        raise OverlayError("unexpected upstream commit")
    if git(source, "rev-parse", "HEAD^{tree}") != contract["upstream_tree"]:
        raise OverlayError("unexpected upstream tree")
    parent_manifest = source / ".frontierrl_overlay.json"
    if not parent_manifest.is_file() or parent_manifest.is_symlink():
        raise OverlayError("missing or unsafe parent overlay manifest")
    if sha256(parent_manifest) != contract["parent_applied_manifest_sha256"]:
        raise OverlayError("parent overlay manifest digest mismatch")
    parent = json.loads(parent_manifest.read_text())
    if parent.get("overlay_contract_sha256") != contract["parent_overlay_contract_sha256"]:
        raise OverlayError("parent overlay contract mismatch")

    states = {}
    removed_total = 0
    for relative, details in contract["files"].items():
        path = source / relative
        if not path.is_file() or path.is_symlink():
            raise OverlayError(f"missing or unsafe source file: {relative}")
        digest = sha256(path)
        text = path.read_text()
        removed_count = text.count(contract["removed_api"])
        removed_total += removed_count
        if digest == details["parent_sha256"] and removed_count == details["replacements"]:
            states[relative] = "parent"
        elif digest == details["applied_sha256"] and removed_count == 0:
            states[relative] = "applied"
        else:
            raise OverlayError(f"source drift or partial modernization: {relative}")

    unique_states = set(states.values())
    if unique_states == {"parent"} and removed_total == contract["total_replacements"]:
        state = "applicable"
    elif unique_states == {"applied"} and removed_total == 0:
        state = "applied"
    else:
        raise OverlayError("mixed parent/applied modernization state")

    applied_path = source / APPLIED_MANIFEST
    if state == "applied":
        if not applied_path.is_file() or applied_path.is_symlink():
            raise OverlayError("missing or unsafe applied modernization manifest")
        actual_manifest = json.loads(applied_path.read_text())
        if actual_manifest != expected_manifest(contract, contract_sha):
            raise OverlayError("applied modernization manifest mismatch")
    elif applied_path.exists():
        raise OverlayError("unexpected applied modernization manifest")
    return state, states


def apply(source: Path, contract: dict, contract_sha: str) -> None:
    state, _ = inspect(source, contract, contract_sha)
    if state != "applicable":
        raise OverlayError("overlay is already applied")
    old = contract["removed_api"].encode()
    new = contract["replacement_api"].encode()
    for relative, details in contract["files"].items():
        path = source / relative
        payload = path.read_bytes()
        if payload.count(old) != details["replacements"]:
            raise OverlayError(f"replacement count drift: {relative}")
        updated = payload.replace(old, new)
        for cleanup in details.get("post_replacements", []):
            cleanup_old = cleanup["old"].encode()
            cleanup_new = cleanup["new"].encode()
            if updated.count(cleanup_old) != cleanup["count"]:
                raise OverlayError(f"post-replacement cleanup drift: {relative}")
            updated = updated.replace(cleanup_old, cleanup_new)
        atomic_write(path, updated)
        if sha256(path) != details["applied_sha256"]:
            raise OverlayError(f"post-application digest mismatch: {relative}")

    manifest_path = source / APPLIED_MANIFEST
    payload = (json.dumps(expected_manifest(contract, contract_sha), indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{APPLIED_MANIFEST}.", dir=source)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    final_state, _ = inspect(source, contract, contract_sha)
    if final_state != "applied":
        raise OverlayError("post-application validation failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    contract = json.loads(CONTRACT_PATH.read_text())
    contract_sha = sha256(CONTRACT_PATH)
    target = args.target.resolve()
    try:
        if args.apply:
            apply(target, contract, contract_sha)
        state, states = inspect(target, contract, contract_sha)
    except (OverlayError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({
        "status": state,
        "overlay": contract["overlay"],
        "contract_sha256": contract_sha,
        "files": len(states),
        "replacements": contract["total_replacements"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
