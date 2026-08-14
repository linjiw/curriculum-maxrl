import hashlib
import json
from pathlib import Path

import pytest

from icra2027.verify_barn_smoke_package import (
    SmokePackageError,
    verify_smoke_package_controls,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(env_id: str, index: int, digit: str) -> dict:
    return {
        "env_id": env_id,
        "barn_index": index,
        "asset": f"world_files/world_{index}.world",
        "asset_sha256": digit * 64,
        "path_asset": f"path_files/path_{index}.npy",
        "path_sha256": chr(ord(digit) + 1) * 64,
        "grid_asset": f"grid_files/grid_{index}.npy",
        "grid_sha256": chr(ord(digit) + 2) * 64,
    }


def _write_complete(package: Path) -> None:
    checksum_digest = _sha256(package / "SHA256SUMS")
    (package / "COMPLETE").write_text(
        "artifact_type\tbarn_hopper_dataset_complete\n"
        "completed_utc\t2026-08-14T00:00:00Z\n"
        f"sha256sums_sha256\t{checksum_digest}\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    package = tmp_path / "package"
    package.mkdir()
    selected = _row("barn-299", 299, "1")
    heldout = _row("barn-001", 1, "4")
    manifest = tmp_path / "barn_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n"
                for row in (selected, heldout)),
        encoding="utf-8",
    )
    receipt = package / "PREPARE_RECEIPT.json"
    receipt.write_text(
        json.dumps({"artifact_type": "barn_hopper_dataset_prepare_receipt"})
        + "\n",
        encoding="utf-8",
    )
    lines = [f"{_sha256(receipt)}  ./PREPARE_RECEIPT.json"]
    for row in (selected, heldout):
        for path_field, hash_field in (
            ("asset", "asset_sha256"),
            ("path_asset", "path_sha256"),
            ("grid_asset", "grid_sha256"),
        ):
            lines.append(
                f"{row[hash_field]}  ./BARN_dataset/{row[path_field]}")
    (package / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    _write_complete(package)
    return package, manifest, selected, heldout


def test_control_verifier_reads_no_dataset_assets(tmp_path):
    package, manifest, selected, heldout = _fixture(tmp_path)

    # No BARN_dataset directory or course asset exists.  Both selected and
    # held-out declarations are present only in SHA256SUMS.  Success therefore
    # proves that the control verifier did not require filesystem access to any
    # course asset; the runner owns selected-train asset verification.
    assert not (package / "BARN_dataset").exists()
    result = verify_smoke_package_controls(package, manifest, "barn-299")

    assert result["course_id"] == selected["env_id"]
    assert result["dataset_assets_read"] is False
    assert set(result["selected_asset_declarations"]) == {
        "asset", "path_asset", "grid_asset"}
    assert heldout["env_id"] != result["course_id"]


def test_control_verifier_rejects_tampered_complete(tmp_path):
    package, manifest, _, _ = _fixture(tmp_path)
    (package / "COMPLETE").write_text(
        "artifact_type\tbarn_hopper_dataset_complete\n"
        "completed_utc\t2026-08-14T00:00:00Z\n"
        f"sha256sums_sha256\t{'0' * 64}\n",
        encoding="utf-8",
    )

    with pytest.raises(
            SmokePackageError, match="checksum manifest digest mismatch"):
        verify_smoke_package_controls(package, manifest, "barn-299")


def test_control_verifier_rejects_tampered_receipt(tmp_path):
    package, manifest, _, _ = _fixture(tmp_path)
    (package / "PREPARE_RECEIPT.json").write_text(
        '{"tampered": true}\n', encoding="utf-8")

    with pytest.raises(SmokePackageError, match="receipt is not checksum-bound"):
        verify_smoke_package_controls(package, manifest, "barn-299")


def test_control_verifier_rejects_tampered_selected_declaration(tmp_path):
    package, manifest, selected, _ = _fixture(tmp_path)
    checksums = package / "SHA256SUMS"
    checksums.write_text(
        checksums.read_text(encoding="utf-8").replace(
            selected["asset_sha256"], "f" * 64, 1),
        encoding="utf-8",
    )
    # Rebind COMPLETE so the failure is specifically the selected-row binding,
    # not the outer COMPLETE -> SHA256SUMS check.
    _write_complete(package)

    with pytest.raises(
            SmokePackageError, match="does not bind selected train asset"):
        verify_smoke_package_controls(package, manifest, "barn-299")
