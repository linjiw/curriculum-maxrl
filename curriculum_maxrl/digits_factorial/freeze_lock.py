"""Create the canonical source/runtime lock after source review."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from .core import (
    DATA_MANIFEST_PATH,
    EXPECTED_RUNTIME,
    LOCK_SCHEMA,
    SOURCE_LOCK_PATH,
    frozen_schedule,
    sha256_file,
    strict_json_load,
    write_json,
)
from .locking import live_source_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replace-preexecution-lock",
        action="store_true",
        help="replace only a lock that has never authorized an evidence run",
    )
    args = parser.parse_args()
    if SOURCE_LOCK_PATH.exists() and not args.replace_preexecution_lock:
        raise SystemExit(f"refusing to overwrite source lock: {SOURCE_LOCK_PATH}")
    manifest = strict_json_load(DATA_MANIFEST_PATH)
    payload = {
        "schema": LOCK_SCHEMA,
        "status": "frozen_before_lr_development_or_confirmation",
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Pre-development/pre-confirmation source, data, schedule, and runtime "
            "lock for the Digits exact-probability estimator-by-sampler factorial."
        ),
        "public_preexecution_commit": None,
        "public_preexecution_commit_disclosure": (
            "No public pre-execution commit existed when this local lock was sealed."
        ),
        "protocol_relative_path": "curriculum_maxrl/digits_factorial/PROTOCOL.md",
        "data_manifest_relative_path": (
            "curriculum_maxrl/digits_factorial/digits_split_manifest.json"
        ),
        "data_manifest_sha256": sha256_file(DATA_MANIFEST_PATH),
        "expected_runtime": EXPECTED_RUNTIME,
        "schedule": frozen_schedule(),
        "source_sha256": live_source_manifest(),
    }
    write_json(SOURCE_LOCK_PATH, payload)
    print(SOURCE_LOCK_PATH)


if __name__ == "__main__":
    main()
