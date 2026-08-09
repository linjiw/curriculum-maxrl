"""Build the canonical ProCuRL-selection source/runtime lock without overwriting.

This command records no outcome and launches no experiment.  It is intended to
be called exactly once *after* a passing independent pre-seal review.  The
writer uses an fsynced temporary file followed by a hard-link create, so the
canonical target is installed atomically and an existing target is never
replaced.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from frontier_rl.examples import run_acrobot_procurl_selection as runner


def build_lock_payload() -> dict:
    runtime = runner._runtime()
    if runner._runtime_versions(runtime) != runner.PINNED_RUNTIME_VERSIONS:
        raise RuntimeError(
            "lock creation requires the exact pinned runtime: "
            f"expected={runner.PINNED_RUNTIME_VERSIONS!r}, observed={runtime!r}"
        )
    seed_audit = runner.seed_collision_audit()
    if seed_audit.get("passed") is not True:
        raise RuntimeError("seed/RNG audit did not pass")
    v2_audit = runner._v2_dependency_audit()
    if v2_audit.get("passed") is not True:
        raise RuntimeError("V2 transitive dependency audit did not pass")
    source_hashes = runner._source_hashes(require_all=True)
    payload = {
        "schema": runner.LOCK_SCHEMA,
        "status": "sealed_before_any_quick_development_or_confirmation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Canonical pre-execution source/runtime lock for the Acrobot "
            "ProCuRL selection-semantic study."
        ),
        "runtime": runtime,
        "schedule": runner._locked_schedule(),
        "seed_collision_audit": seed_audit,
        "source_sha256": source_hashes,
        "v2_dependency_audit": v2_audit,
    }
    if set(payload) != runner.LOCK_KEYS:
        raise RuntimeError("lock builder top-level schema drifted")
    return payload


def write_lock_atomic_refuse_overwrite(
    output: Path = runner.LOCK_PATH,
) -> dict:
    output = output.resolve()
    if output != runner.LOCK_PATH.resolve():
        raise RuntimeError(
            "the lock builder writes only the canonical path: "
            f"{runner.LOCK_PATH.resolve()}"
        )
    payload = build_lock_payload()
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite existing lock: {output}"
            ) from error
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=runner.LOCK_PATH)
    args = parser.parse_args()
    try:
        payload = write_lock_atomic_refuse_overwrite(args.output)
    except (RuntimeError, FileExistsError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "written": str(args.output.resolve()),
                "schema": payload["schema"],
                "status": payload["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
