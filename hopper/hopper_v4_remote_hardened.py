#!/usr/bin/env python3
"""Build or render the exact NIL-bootstrap v4 Slurm submission closure.

No command in this file calls Slurm.  ``submit`` is an explicit refusal.  A
future, separately authorized Hopper audit may execute the rendered argv and
must create the strict submission receipt validated by ``slurm_integrity.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import pwd
import re
import shlex
import tempfile
from typing import Mapping, Sequence


HASH_RE = re.compile(r"^[0-9a-f]{64}$")
BASE_COMMIT = "d053054c5290a04c1c4cd8b55704d999cad73e30"
BASE_TREE = "b0cace1fc54984e21a842f12d15d0b899e33d270"
CONTRACT_SHA = "3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b"
PROTOCOL_SHA = "1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269"
FRONTIER_CONFIG_SHA = "0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2"
MAXMC_CONFIG_SHA = "a3cc3ddf387a3bb7cf3d9759c3f13e6a74f2c7e32de311ac344d54ef5e703ec6"

LEGACY_COMMON = {
    "UED_BUNDLE_DIR", "UED_BUNDLE_MANIFEST_SHA256", "UED_UPSTREAM_COMMIT",
    "UED_UPSTREAM_TREE", "UED_UPSTREAM_BUNDLE_SHA256",
    "UED_OVERLAY_MANIFEST_SHA256", "UED_SBATCH_SHA256", "UED_ENV_DIR",
    "UED_ENV_LOCK_SHA256", "UED_ENV_FREEZE_SHA256", "UED_ENV_MANIFEST_SHA256",
}
LEGACY_KEYS = {
    "import": LEGACY_COMMON,
    "one_update": LEGACY_COMMON | {
        "UED_CONFIG_SHA256", "UED_CONTRACT_SHA256",
        "UED_IMPORT_SMOKE_RESULT_DIR", "UED_IMPORT_SMOKE_MANIFEST_SHA256",
    },
    "terminal": LEGACY_COMMON | {
        "UED_IMPORT_SMOKE_RESULT_DIR", "UED_IMPORT_SMOKE_MANIFEST_SHA256",
        "UED_ONE_UPDATE_RESULT_DIR", "UED_ONE_UPDATE_MANIFEST_SHA256", "UED_ARM",
        "UED_CONFIG_SHA256", "UED_CONTRACT_SHA256", "UED_PROTOCOL_SHA256",
        "UED_PHASE_A_DRIVER_SHA256", "UED_TRAINING_DRIVER_SHA256",
        "UED_EVALUATION_DRIVER_SHA256", "UED_ASSEMBLER_SHA256",
        "UED_FINALIZER_SHA256",
    },
}
HARDENING_COMMON = {
    "UED_REMOTE_HARDENING_STATE_SHA256", "UED_HARDENING_SBATCH_SHA256",
    "UED_LEGACY_SBATCH_SHA256", "UED_ENV_TREE_DIR",
    "UED_ENV_TREE_MANIFEST_SHA256", "UED_ENV_TREE_RECEIPT_SHA256",
    "UED_ENV_TREE_TOOL_SHA256", "UED_GPU_PROBE_TOOL_SHA256",
    "UED_JOB_GUARD_SHA256",
}
PAIR_KEYS = {
    "UED_PAIR_PLAN_DIR", "UED_PAIR_PLAN_MANIFEST_SHA256",
    "UED_PAIR_PLAN_TOOL_SHA256",
}
EXPECTED_KEYS = {
    rung: LEGACY_KEYS[rung] | HARDENING_COMMON | (PAIR_KEYS if rung == "terminal" else set())
    for rung in ("import", "one_update", "terminal")
}
SBATCH_NAMES = {
    "ued_minimax_v4_remote_hardened_gpu_smoke.sbatch": "import",
    "ued_minimax_v4_remote_hardened_one_update_smoke.sbatch": "one_update",
    "ued_minimax_v4_remote_hardened_terminal_chain_smoke.sbatch": "terminal",
}
LEGACY_NAMES = {
    "import": "ued_minimax_v4_gpu_smoke.sbatch",
    "one_update": "ued_minimax_v4_one_update_smoke.sbatch",
    "terminal": "ued_minimax_v4_terminal_chain_smoke.sbatch",
}


class LauncherError(RuntimeError):
    """Raised when a prospective submission is not exact and hermetic."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LauncherError(message)


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"unsafe regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_assignments(assignments: Sequence[str], expected: set[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for encoded in assignments:
        require("=" in encoded, f"input lacks '=': {encoded}")
        key, value = encoded.split("=", 1)
        require(re.fullmatch(r"UED_[A-Z0-9_]+", key) is not None, f"unsafe input key: {key}")
        require(key in expected, f"extra input forbidden: {key}")
        require(key not in values and value != "", f"duplicate or empty input: {key}")
        require(not any(character in value for character in "\x00\n\r\t"), f"unsafe input value: {key}")
        values[key] = value
    require(set(values) == expected, "input-envelope allowlist drift")
    return values


def parse_envelope(path: Path, expected: set[str]) -> dict[str, str]:
    require(path.is_absolute() and ".." not in path.parts, "envelope path must be absolute")
    require(path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(), "unsafe envelope")
    raw = path.read_bytes()
    require(raw and raw.endswith(b"\0"), "envelope is not NUL-delimited")
    assignments: list[str] = []
    for value in raw[:-1].split(b"\0"):
        try:
            assignments.append(value.decode("utf-8"))
        except UnicodeError as exc:
            raise LauncherError("envelope is not UTF-8") from exc
    return parse_assignments(assignments, expected)


def _user() -> str:
    value = pwd.getpwuid(os.getuid()).pw_name
    require(re.fullmatch(r"[A-Za-z0-9._-]+", value) is not None and value not in {".", ".."}, "unsafe user")
    return value


def _canonical_existing(path: Path, label: str) -> Path:
    require(path.is_absolute() and ".." not in path.parts, f"{label} must be absolute")
    require(path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(), f"unsafe {label}")
    return path


def validate_values(
    sbatch: Path, rung: str, values: Mapping[str, str], *, local_test_mode: bool
) -> None:
    for key, value in values.items():
        if key.endswith("_SHA256"):
            require(HASH_RE.fullmatch(value) is not None, f"malformed SHA-256: {key}")
    require(values["UED_UPSTREAM_COMMIT"] == BASE_COMMIT, "upstream commit drift")
    require(values["UED_UPSTREAM_TREE"] == BASE_TREE, "upstream tree drift")
    bundle = Path(values["UED_BUNDLE_DIR"])
    require(bundle.is_absolute() and ".." not in bundle.parts, "bundle path must be absolute")
    require(bundle.resolve(strict=True) == bundle and bundle.is_dir() and not bundle.is_symlink(), "unsafe bundle")
    user = _user()
    if not local_test_mode:
        require(bundle.is_relative_to(Path(f"/scratch/{user}/maxrl/bundles/ued_minimax_v4_engineering")), "bundle namespace drift")
        require(Path(values["UED_ENV_DIR"]).is_relative_to(Path(f"/scratch/{user}/envs")), "environment namespace drift")
        require(Path(values["UED_ENV_TREE_DIR"]).is_relative_to(Path(f"/scratch/{user}/maxrl/provenance")), "environment closure namespace drift")
    require(bundle.name == values["UED_BUNDLE_MANIFEST_SHA256"][:20], "bundle content-address drift")
    require(bundle.name != "06ffeeeb6998e8ddb1ce", "protected v3 bundle forbidden")
    require(sha256(bundle / "SHA256SUMS") == values["UED_BUNDLE_MANIFEST_SHA256"], "bundle manifest drift")
    require(sha256(bundle / "REMOTE_HARDENING_STATE.json") == values["UED_REMOTE_HARDENING_STATE_SHA256"], "remote state drift")
    require(sbatch == bundle / "hopper/sbatch" / sbatch.name, "sbatch/bundle path drift")
    require(sha256(sbatch) == values["UED_HARDENING_SBATCH_SHA256"], "hardening sbatch drift")
    legacy = bundle / "hopper/sbatch" / LEGACY_NAMES[rung]
    require(sha256(legacy) == values["UED_LEGACY_SBATCH_SHA256"], "legacy sbatch drift")
    require(values["UED_SBATCH_SHA256"] == values["UED_LEGACY_SBATCH_SHA256"], "legacy self-binding drift")
    require(values.get("UED_CONTRACT_SHA256", CONTRACT_SHA) == CONTRACT_SHA, "contract drift")
    require(values.get("UED_PROTOCOL_SHA256", PROTOCOL_SHA) == PROTOCOL_SHA, "protocol drift")
    if rung == "one_update":
        require(values["UED_CONFIG_SHA256"] == FRONTIER_CONFIG_SHA, "one-update config drift")
    if rung == "terminal":
        arm = values["UED_ARM"]
        require(arm in {"frontier", "maxmc"}, "terminal arm drift")
        require(values["UED_CONFIG_SHA256"] == (
            FRONTIER_CONFIG_SHA if arm == "frontier" else MAXMC_CONFIG_SHA
        ), "terminal arm/config drift")
        pair = Path(values["UED_PAIR_PLAN_DIR"])
        require(pair.is_absolute() and pair.resolve(strict=True) == pair and pair.is_dir() and not pair.is_symlink(), "unsafe pair plan")
        require(sha256(pair / "SHA256SUMS") == values["UED_PAIR_PLAN_MANIFEST_SHA256"], "pair-plan manifest drift")


def build_envelope(cli: argparse.Namespace) -> tuple[Path, str]:
    sbatch = _canonical_existing(cli.sbatch, "sbatch")
    require(sbatch.name in SBATCH_NAMES, "unrecognized hardened sbatch")
    rung = SBATCH_NAMES[sbatch.name]
    values = parse_assignments(cli.assignments, EXPECTED_KEYS[rung])
    validate_values(sbatch, rung, values, local_test_mode=cli.local_test_mode)
    output = cli.output
    require(output.is_absolute() and ".." not in output.parts and output.name not in {"", ".", ".."}, "output must be absolute")
    require(not output.exists() and not output.is_symlink(), "output exists")
    require(output.parent.resolve(strict=True) == output.parent and output.parent.is_dir() and not output.parent.is_symlink(), "unsafe output parent")
    payload = b"".join(
        key.encode("ascii") + b"=" + values[key].encode("utf-8") + b"\0"
        for key in sorted(values)
    )
    descriptor, raw = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(raw)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    require(parse_envelope(output, EXPECTED_KEYS[rung]) == values, "post-write envelope drift")
    return output, sha256(output)


def render(cli: argparse.Namespace) -> str:
    ambient = sorted(key for key in os.environ if key.startswith("SBATCH_") or key.startswith("UED_"))
    require(not ambient, f"ambient submission controls forbidden: {','.join(ambient)}")
    sbatch = _canonical_existing(cli.sbatch, "sbatch")
    require(sbatch.name in SBATCH_NAMES, "unrecognized hardened sbatch")
    rung = SBATCH_NAMES[sbatch.name]
    values = parse_envelope(cli.envelope, EXPECTED_KEYS[rung])
    validate_values(sbatch, rung, values, local_test_mode=cli.local_test_mode)
    argv = [
        "/usr/bin/env", "-i", "/usr/bin/sbatch", "--parsable",
        f"--chdir=/scratch/{_user()}/maxrl", "--export=NIL",
        str(sbatch), f"--ued-input-envelope={cli.envelope}",
        f"--ued-bundle-dir={values['UED_BUNDLE_DIR']}",
        f"--ued-submitted-sbatch={sbatch}",
    ]
    return shlex.join(argv)


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-envelope")
    build.add_argument("--sbatch", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--local-test-mode", action="store_true")
    build.add_argument("assignments", nargs="+")
    rendered = subparsers.add_parser("render")
    rendered.add_argument("--sbatch", type=Path, required=True)
    rendered.add_argument("--envelope", type=Path, required=True)
    rendered.add_argument("--local-test-mode", action="store_true")
    subparsers.add_parser("submit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    cli = parse_cli(argv)
    try:
        if cli.command == "submit":
            raise LauncherError(
                "remote submission is HOLD pending a subsequent explicit exact-Hopper ladder audit"
            )
        if cli.command == "build-envelope":
            path, digest = build_envelope(cli)
            print(f"V4H_INPUT_ENVELOPE_COMPLETE path={path} sha256={digest}")
        else:
            print(render(cli))
    except (LauncherError, OSError, UnicodeError, ValueError) as exc:
        print(f"V4H_LAUNCH_REFUSED: {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
