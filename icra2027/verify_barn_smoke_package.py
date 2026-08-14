"""Verify prepared-package controls without reading BARN course assets.

The engineering timing smoke must remain blind to prospective held-out course
files.  Dataset preparation has already verified every archive member.  This
module therefore authenticates the preparation controls and confirms that the
checksum manifest declares the frozen hashes for one selected training course,
but it deliberately never resolves, stats, opens, or hashes any dataset asset.
The campaign runner subsequently verifies only the selected training course's
real assets against the same frozen manifest row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Sequence


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CHECKSUM_LINE_RE = re.compile(r"([0-9a-f]{64}) ([ *])(.+)\Z")
_COMPLETE_FIELDS = {
    "artifact_type", "completed_utc", "sha256sums_sha256"}
_COMPLETE_TYPE = "barn_hopper_dataset_complete"


class SmokePackageError(ValueError):
    """Prepared package controls do not satisfy the smoke contract."""


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _require_regular(path: Path, label: str) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError as error:
        raise SmokePackageError(f"missing {label}") from error
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise SmokePackageError(
            f"{label} is not a regular non-symbolic file")


def _safe_relative(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SmokePackageError(f"{label} must be a non-empty relative path")
    name = value[2:] if value.startswith("./") else value
    if (not name or name.startswith("./") or name.endswith("/")
            or "//" in name or "\\" in name):
        raise SmokePackageError(f"unsafe {label}: {value!r}")
    relative = PurePosixPath(name)
    if (relative.is_absolute()
            or any(part in ("", ".", "..") for part in relative.parts)):
        raise SmokePackageError(f"unsafe {label}: {value!r}")
    return relative.as_posix()


def _parse_complete(path: Path) -> dict[str, str]:
    complete: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise SmokePackageError(
            "prepared-dataset COMPLETE is not UTF-8") from error
    for line in lines:
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0] or fields[0] in complete:
            raise SmokePackageError("malformed prepared-dataset COMPLETE")
        complete[fields[0]] = fields[1]
    if set(complete) != _COMPLETE_FIELDS:
        raise SmokePackageError(
            "prepared-dataset COMPLETE fields are not exact")
    if complete["artifact_type"] != _COMPLETE_TYPE:
        raise SmokePackageError("prepared-dataset COMPLETE type mismatch")
    if not complete["completed_utc"]:
        raise SmokePackageError(
            "prepared-dataset COMPLETE timestamp is empty")
    if _SHA256_RE.fullmatch(complete["sha256sums_sha256"]) is None:
        raise SmokePackageError(
            "prepared-dataset COMPLETE checksum digest is invalid")
    return complete


def _parse_checksums(path: Path) -> dict[str, str]:
    listed: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise SmokePackageError(
            "prepared-dataset SHA256SUMS is not UTF-8") from error
    if not lines:
        raise SmokePackageError("prepared-dataset SHA256SUMS is empty")
    for raw in lines:
        match = _CHECKSUM_LINE_RE.fullmatch(raw)
        if match is None:
            raise SmokePackageError(
                "unsafe prepared-dataset SHA256SUMS format")
        name = _safe_relative(
            match.group(3), label="prepared-dataset SHA256SUMS path")
        if name in listed or name in {"SHA256SUMS", "COMPLETE"}:
            raise SmokePackageError(
                "duplicate/unsafe prepared-dataset checksum entry")
        listed[name] = match.group(1)
    return listed


def _read_manifest_row(path: Path, course_id: str) -> dict:
    _require_regular(path, "frozen BARN manifest")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise SmokePackageError("frozen BARN manifest is not UTF-8") from error
    selected = None
    seen: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise SmokePackageError(
                f"blank frozen-manifest line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SmokePackageError(
                f"invalid frozen-manifest line {line_number}") from error
        if not isinstance(row, dict) or not isinstance(row.get("env_id"), str):
            raise SmokePackageError(
                f"invalid frozen-manifest row {line_number}")
        env_id = row["env_id"]
        if env_id in seen:
            raise SmokePackageError(
                f"duplicate frozen-manifest env_id: {env_id!r}")
        seen.add(env_id)
        if env_id == course_id:
            selected = row
    if selected is None:
        raise SmokePackageError(
            "engineering course is absent from frozen manifest")
    return selected


def verify_smoke_package_controls(
    package: Path, manifest_path: Path, course_id: str
) -> dict[str, object]:
    """Authenticate controls and selected declarations without asset access."""

    package = Path(package)
    manifest_path = Path(manifest_path)
    if package.is_symlink() or not package.is_dir():
        raise SmokePackageError(
            "prepared dataset package is not a non-symbolic directory")
    if not isinstance(course_id, str) or not course_id:
        raise SmokePackageError("engineering course id must be non-empty")

    checksum_path = package / "SHA256SUMS"
    complete_path = package / "COMPLETE"
    receipt_path = package / "PREPARE_RECEIPT.json"
    for path, label in (
        (checksum_path, "SHA256SUMS"),
        (complete_path, "COMPLETE"),
        (receipt_path, "PREPARE_RECEIPT.json"),
    ):
        _require_regular(path, label)

    complete = _parse_complete(complete_path)
    checksum_digest = _digest(checksum_path)
    if checksum_digest != complete["sha256sums_sha256"]:
        raise SmokePackageError(
            "prepared-dataset checksum manifest digest mismatch")
    listed = _parse_checksums(checksum_path)
    receipt_digest = _digest(receipt_path)
    if listed.get("PREPARE_RECEIPT.json") != receipt_digest:
        raise SmokePackageError(
            "prepared-dataset receipt is not checksum-bound")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SmokePackageError(
            "prepared-dataset receipt is not valid UTF-8 JSON") from error
    if not isinstance(receipt, dict):
        raise SmokePackageError("prepared-dataset receipt must be an object")

    row = _read_manifest_row(manifest_path, course_id)
    checks = [("asset", "asset_sha256"),
              ("path_asset", "path_sha256")]
    has_grid_path = "grid_asset" in row
    has_grid_hash = "grid_sha256" in row
    if has_grid_path != has_grid_hash:
        raise SmokePackageError(
            "selected train row has incomplete grid checksum fields")
    if has_grid_path:
        checks.append(("grid_asset", "grid_sha256"))
    selected_declarations = {}
    for path_field, hash_field in checks:
        relative = _safe_relative(
            row.get(path_field), label=f"selected train {path_field}")
        expected = row.get(hash_field)
        if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
            raise SmokePackageError(
                f"selected train {hash_field} is not a SHA-256 digest")
        name = f"BARN_dataset/{relative}"
        if listed.get(name) != expected:
            raise SmokePackageError(
                f"prepared package does not bind selected train {path_field}")
        selected_declarations[path_field] = expected

    return {
        "course_id": course_id,
        "checksum_manifest_sha256": checksum_digest,
        "receipt_sha256": receipt_digest,
        "selected_asset_declarations": selected_declarations,
        "dataset_assets_read": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Verify prepared BARN controls and one train course's "
                     "checksum declarations without reading dataset assets"))
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--course-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_smoke_package_controls(
        args.package, args.manifest, args.course_id)
    print(
        "BARN_SMOKE_PACKAGE_CONTROL_PASS"
        f"\tcourse={result['course_id']}\tdataset_assets_read=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
