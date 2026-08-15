"""Verify the vendored Acrobot sources against the sealed V2 tournament lock.

The V2 lock (`ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json`, sealed
2026-08-08T07:00:25Z) records a SHA-256 for every source file that produced the
paper's Acrobot result.  This script proves the hermetic tree under
`acrobot_u64/vendor/` is byte-identical to those files.

It deliberately reports the RUNTIME mismatch as well.  The lock's equality rule
requires the runtime to match too, and this host does not match the macOS
runtime the V2 tournament ran on.  That is why the U64 campaign runs every arm
fresh under its own preregistration rather than extending the sealed V2 result.

Exit 0 iff every locked source file is present and byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import sys

HERE = pathlib.Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
LOCK = VENDOR / "frontier_rl" / "examples" / "ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    expected: dict[str, str] = lock["source_sha256"]

    print(f"lock schema : {lock['schema']}")
    print(f"sealed_utc  : {lock['sealed_utc']}")
    print(f"locked files: {len(expected)}")
    print()

    ok = True
    for rel, want in sorted(expected.items()):
        path = VENDOR / rel
        if not path.is_file():
            print(f"MISSING   {rel}")
            ok = False
            continue
        got = sha256(path)
        if got == want:
            print(f"OK        {rel}  {got[:16]}")
        else:
            print(f"MISMATCH  {rel}\n            want {want}\n            got  {got}")
            ok = False

    print()
    print("Runtime comparison (informational; see module docstring):")
    try:
        import numpy
        numpy_v = numpy.__version__
    except Exception:
        numpy_v = "absent"
    try:
        import gymnasium
        gym_v = gymnasium.__version__
    except Exception:
        gym_v = "absent"

    locked_rt = lock["runtime"]
    here_rt = {
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy_v,
        "gymnasium": gym_v,
    }
    for key in locked_rt:
        mark = "same" if str(locked_rt[key]) == str(here_rt.get(key)) else "DIFFERS"
        print(f"  {key:22s} lock={locked_rt[key]!s:34s} here={here_rt.get(key)!s:28s} {mark}")

    print()
    print("SOURCE LOCK VERIFIED" if ok else "SOURCE LOCK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
