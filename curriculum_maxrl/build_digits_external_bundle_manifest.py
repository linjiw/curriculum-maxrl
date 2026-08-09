"""Build or validate the content manifest for unshipped Digits replay payloads.

The public repository keeps the compact analysis/receipt chain.  The large
per-run ledgers and recovery checkpoints remain on the execution machine until
they can be deposited in content-addressed storage.  This manifest records
their relative names, byte sizes, and SHA-256 digests without embedding a
machine-specific path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "curriculum_maxrl" / "digits_factorial"
DEFAULT_OUTPUT = PACKAGE_ROOT / "EXTERNAL_REPLAY_BUNDLE_MANIFEST.json"

CELLS = (
    "practical_maxrl__uniform",
    "practical_maxrl__p1mp",
    "practical_maxrl__u8",
    "rloo__uniform",
    "rloo__p1mp",
    "rloo__u8",
)
SCIENTIFIC_FILES = (
    "summary.json",
    "ledger.npz",
    "checkpoint_step0000.pt",
    "checkpoint_step0128.pt",
    "checkpoint_step0256.pt",
    "checkpoint_step0384.pt",
    "checkpoint_step0512.pt",
)
ENGINEERING_SCIENTIFIC_FILES = (
    "summary.json",
    "ledger.npz",
    "checkpoint_step0000.pt",
    "checkpoint_step0032.pt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def expected_runs() -> dict[str, list[Path]]:
    roots: dict[str, list[Path]] = {
        "development": [],
        "confirmation_tuned": [],
        "confirmation_common": [],
        "engineering_reseal_v3": [],
    }
    for rate in ("0.03", "0.1", "0.3", "1", "3"):
        for seed in range(31_000, 31_004):
            for cell in CELLS:
                roots["development"].append(
                    PACKAGE_ROOT
                    / "development"
                    / "registered_v1"
                    / f"lr_{rate}"
                    / f"seed_{seed}"
                    / cell
                )
    for schedule in ("tuned", "common"):
        key = f"confirmation_{schedule}"
        for seed in range(32_000, 32_024):
            for cell in CELLS:
                roots[key].append(
                    PACKAGE_ROOT
                    / "confirmation"
                    / "registered_v1"
                    / schedule
                    / f"seed_{seed}"
                    / cell
                )
    for execution in ("serial", "parallel"):
        for cell in CELLS:
            roots["engineering_reseal_v3"].append(
                PACKAGE_ROOT / "engineering" / "reseal_v3" / execution / cell
            )
    return roots


def build_manifest() -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    sections: dict[str, dict[str, int]] = {}
    for section, run_directories in expected_runs().items():
        scientific_files = (
            ENGINEERING_SCIENTIFIC_FILES
            if section == "engineering_reseal_v3"
            else SCIENTIFIC_FILES
        )
        section_bytes = 0
        for run_directory in run_directories:
            if not run_directory.is_dir():
                raise FileNotFoundError(f"missing replay run directory: {run_directory}")
            observed = {path.name for path in run_directory.iterdir() if path.is_file()}
            required = {*scientific_files, "timing.json"}
            if observed != required:
                raise ValueError(
                    f"unexpected run files in {run_directory}: "
                    f"missing={sorted(required - observed)}, extra={sorted(observed - required)}"
                )
            for filename in scientific_files:
                path = run_directory / filename
                relative = path.relative_to(PROJECT_ROOT).as_posix()
                size = path.stat().st_size
                section_bytes += size
                files[relative] = {"bytes": size, "sha256": sha256_file(path)}
        sections[section] = {
            "run_directories": len(run_directories),
            "scientific_files": len(run_directories) * len(scientific_files),
            "bytes": section_bytes,
        }

    file_manifest_sha = hashlib.sha256(canonical_bytes(files)).hexdigest()
    return {
        "schema": "curriculum-maxrl/digits-external-replay-bundle/v1",
        "status": "retained_locally_not_shipped",
        "content_addressed_download_uri": None,
        "scope": (
            "Full ledgers plus five recovery checkpoints for each development/"
            "confirmation run and two for each 32-step engineering run. Unbound "
            "timing sidecars and the private Python environment are excluded."
        ),
        "source_lock_sha256": sha256_file(PACKAGE_ROOT / "SOURCE_LOCK.json"),
        "sections": sections,
        "total_run_directories": sum(item["run_directories"] for item in sections.values()),
        "total_scientific_files": len(files),
        "total_bytes": sum(item["bytes"] for item in sections.values()),
        "file_manifest_sha256": file_manifest_sha,
        "files": files,
    }


def validate_stored_structure(payload: dict[str, object]) -> None:
    required = {
        "schema",
        "status",
        "content_addressed_download_uri",
        "scope",
        "source_lock_sha256",
        "sections",
        "total_run_directories",
        "total_scientific_files",
        "total_bytes",
        "file_manifest_sha256",
        "files",
    }
    if set(payload) != required:
        raise ValueError("external replay manifest top-level schema differs")
    if payload["schema"] != "curriculum-maxrl/digits-external-replay-bundle/v1":
        raise ValueError("external replay manifest schema differs")
    if payload["status"] != "retained_locally_not_shipped":
        raise ValueError("external replay manifest must disclose that payloads are unshipped")
    if payload["content_addressed_download_uri"] is not None:
        raise ValueError("download URI must remain null until a verified deposit exists")
    if payload["source_lock_sha256"] != sha256_file(PACKAGE_ROOT / "SOURCE_LOCK.json"):
        raise ValueError("external replay manifest source-lock digest differs")
    expected_section_counts = {
        "development": {"run_directories": 120, "scientific_files": 840},
        "confirmation_tuned": {"run_directories": 144, "scientific_files": 1008},
        "confirmation_common": {"run_directories": 144, "scientific_files": 1008},
        "engineering_reseal_v3": {"run_directories": 12, "scientific_files": 48},
    }
    sections = payload["sections"]
    if not isinstance(sections, dict) or set(sections) != set(expected_section_counts):
        raise ValueError("external replay section set differs")
    for section, counts in expected_section_counts.items():
        record = sections[section]
        if not isinstance(record, dict) or set(record) != {
            "run_directories",
            "scientific_files",
            "bytes",
        }:
            raise ValueError(f"external replay section schema differs: {section}")
        if any(record[key] != value for key, value in counts.items()):
            raise ValueError(f"external replay section counts differ: {section}")
        if type(record["bytes"]) is not int or record["bytes"] <= 0:
            raise ValueError(f"external replay section byte total is invalid: {section}")
    if payload["total_run_directories"] != 420 or payload["total_scientific_files"] != 2904:
        raise ValueError("external replay global counts differ")
    files = payload["files"]
    if not isinstance(files, dict) or len(files) != payload["total_scientific_files"]:
        raise ValueError("external replay file count differs")
    if hashlib.sha256(canonical_bytes(files)).hexdigest() != payload["file_manifest_sha256"]:
        raise ValueError("external replay file-manifest digest differs")
    total_bytes = 0
    for relative, record in files.items():
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("external replay manifest contains a non-relative path")
        if "/Users/" in relative or "/home/" in relative or "/tmp/" in relative:
            raise ValueError("external replay manifest leaks a machine-specific path")
        if not isinstance(record, dict) or set(record) != {"bytes", "sha256"}:
            raise ValueError("external replay file record schema differs")
        if type(record["bytes"]) is not int or record["bytes"] <= 0:
            raise ValueError("external replay file size is invalid")
        digest = record["sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("external replay file digest is invalid")
        int(digest, 16)
        total_bytes += record["bytes"]
    if total_bytes != payload["total_bytes"]:
        raise ValueError("external replay byte total differs")
    if total_bytes != sum(record["bytes"] for record in sections.values()):
        raise ValueError("external replay section/global byte totals disagree")

    expected_paths: set[str] = set()
    for section, run_directories in expected_runs().items():
        scientific_files = (
            ENGINEERING_SCIENTIFIC_FILES
            if section == "engineering_reseal_v3"
            else SCIENTIFIC_FILES
        )
        for run_directory in run_directories:
            for filename in scientific_files:
                expected_paths.add(
                    (run_directory / filename).relative_to(PROJECT_ROOT).as_posix()
                )
    if set(files) != expected_paths:
        raise ValueError("external replay logical path set differs from frozen schedules")

    common_tuned_pairs = 0
    for seed in range(32_000, 32_024):
        for cell in CELLS:
            for filename in ("ledger.npz", *SCIENTIFIC_FILES[2:]):
                common = (
                    PACKAGE_ROOT
                    / "confirmation"
                    / "registered_v1"
                    / "common"
                    / f"seed_{seed}"
                    / cell
                    / filename
                ).relative_to(PROJECT_ROOT).as_posix()
                tuned = common.replace("/common/", "/tuned/", 1)
                if files[common] != files[tuned]:
                    raise ValueError("common/tuned paired scientific hashes differ")
                common_tuned_pairs += 1
    if common_tuned_pairs != 864:
        raise ValueError("common/tuned paired-file count differs")

    engineering_pairs = 0
    for cell in CELLS:
        for filename in ENGINEERING_SCIENTIFIC_FILES:
            serial = (
                PACKAGE_ROOT / "engineering" / "reseal_v3" / "serial" / cell / filename
            ).relative_to(PROJECT_ROOT).as_posix()
            parallel = serial.replace("/serial/", "/parallel/", 1)
            if files[serial] != files[parallel]:
                raise ValueError("engineering serial/parallel scientific hashes differ")
            engineering_pairs += 1
    if engineering_pairs != 24:
        raise ValueError("engineering paired-file count differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the stored compact manifest without requiring the unshipped payload",
    )
    parser.add_argument(
        "--verify-local-payload",
        action="store_true",
        help="re-hash the retained local payload and compare it byte-for-byte with the manifest",
    )
    args = parser.parse_args()

    if args.check or args.verify_local_payload:
        stored = json.loads(args.output.read_text(encoding="utf-8"))
        validate_stored_structure(stored)
        if args.verify_local_payload:
            observed = build_manifest()
            if canonical_bytes(observed) != canonical_bytes(stored):
                raise SystemExit("stored Digits external replay manifest differs from local payload")
        print(
            "Digits external replay manifest passes"
            + (" and matches the retained local payload" if args.verify_local_payload else "")
        )
        return

    payload = build_manifest()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing manifest: {args.output}")
    args.output.write_bytes(json.dumps(payload, indent=1, sort_keys=True).encode() + b"\n")
    print(args.output)


if __name__ == "__main__":
    main()
