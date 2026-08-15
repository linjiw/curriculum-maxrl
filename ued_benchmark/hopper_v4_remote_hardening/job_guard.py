#!/usr/bin/env python3
"""Hermetic pre/post guard for the v4 remote-hardening engineering rungs.

This tool never submits a job and never reads a performance endpoint.  The
batch wrappers invoke it before and after the frozen d602 rung implementation.
It proves the exact bundle, installed environment bytes, allocation identity,
GPU/MIG runtime, shared pair plan, and bounded legacy result closure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import re
import shutil
import tempfile
from types import ModuleType
from typing import Any, Mapping, Sequence


HASH_RE = re.compile(r"^[0-9a-f]{64}$")
BASE_BUNDLE_SHA256 = "d602ce7854f8f3e99352025b97eed2fde32733c0dd23297d5c28b1051e7aeaf0"
BASE_COMMIT = "d053054c5290a04c1c4cd8b55704d999cad73e30"
BASE_TREE = "b0cace1fc54984e21a842f12d15d0b899e33d270"
PURPOSE = "v4_remote_hardening_rung_integrity_not_paper_evidence"

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
EXPECTED_LEGACY_SBATCH = {
    "import": "hopper/sbatch/ued_minimax_v4_gpu_smoke.sbatch",
    "one_update": "hopper/sbatch/ued_minimax_v4_one_update_smoke.sbatch",
    "terminal": "hopper/sbatch/ued_minimax_v4_terminal_chain_smoke.sbatch",
}
EXPECTED_HARDENING_SBATCH = {
    "import": "hopper/sbatch/ued_minimax_v4_remote_hardened_gpu_smoke.sbatch",
    "one_update": "hopper/sbatch/ued_minimax_v4_remote_hardened_one_update_smoke.sbatch",
    "terminal": "hopper/sbatch/ued_minimax_v4_remote_hardened_terminal_chain_smoke.sbatch",
}


class JobGuardError(RuntimeError):
    """Raised when a hardened rung does not have an exact closed identity."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise JobGuardError(message)


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"unsafe regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"unsafe {label}")
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate {label} key: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def _module(path: Path, expected_sha256: str, name: str) -> ModuleType:
    require(HASH_RE.fullmatch(expected_sha256 or "") is not None, f"bad {name} hash")
    require(sha256(path) == expected_sha256, f"{name} tool drift")
    specification = importlib.util.spec_from_file_location(f"v4h_{name}", path)
    require(specification is not None and specification.loader is not None, f"cannot load {name}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def parse_envelope(path: Path, rung: str) -> dict[str, str]:
    require(rung in EXPECTED_KEYS, "unknown rung")
    require(path.is_absolute() and ".." not in path.parts, "input envelope must be absolute")
    require(path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(), "unsafe input envelope")
    raw = path.read_bytes()
    require(raw and raw.endswith(b"\0"), "input envelope must contain NUL records")
    values: dict[str, str] = {}
    for encoded in raw[:-1].split(b"\0"):
        require(encoded and b"=" in encoded, "malformed input-envelope record")
        key_raw, value_raw = encoded.split(b"=", 1)
        try:
            key = key_raw.decode("ascii")
            value = value_raw.decode("utf-8")
        except UnicodeError as exc:
            raise JobGuardError("invalid input-envelope encoding") from exc
        require(re.fullmatch(r"UED_[A-Z0-9_]+", key) is not None, "unsafe input-envelope key")
        require(key not in values and value != "", "duplicate or empty input-envelope record")
        require(not any(character in value for character in "\x00\n\r\t"), "unsafe input-envelope value")
        values[key] = value
    require(set(values) == EXPECTED_KEYS[rung], f"{rung} input-envelope keyset drift")
    for key, value in values.items():
        if key.endswith("_SHA256"):
            require(HASH_RE.fullmatch(value) is not None, f"malformed hash: {key}")
    return values


def _canonical_directory(path: Path, prefix: Path, label: str) -> Path:
    require(path.is_absolute() and ".." not in path.parts, f"{label} must be absolute")
    require(path.resolve(strict=True) == path and path.is_dir() and not path.is_symlink(), f"unsafe {label}")
    require(path.is_relative_to(prefix), f"{label} outside fixed namespace")
    return path


def _validate_manifest(
    root: Path, name: str, expected_sha256: str, mode: str,
    *, completion_name: str = "COMPLETE",
) -> None:
    require(mode in {"exact", "result"}, "manifest closure mode drift")
    manifest = root / name
    require(HASH_RE.fullmatch(expected_sha256 or "") is not None, "bad manifest hash")
    require(sha256(manifest) == expected_sha256, f"{name} digest drift")
    listed: dict[str, str] = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw)
        require(match is not None, f"{name} row drift")
        digest, encoded = match.groups()
        encoded = encoded.removeprefix("./")
        relative = PurePosixPath(encoded)
        require(
            encoded and not relative.is_absolute()
            and all(part not in {"", ".", ".."} for part in relative.parts)
            and relative.as_posix() not in listed,
            f"unsafe {name} path",
        )
        target = root.joinpath(*relative.parts)
        require(target.is_file() and not target.is_symlink(), f"missing {name} payload")
        require(sha256(target) == digest, f"{name} payload drift: {encoded}")
        listed[relative.as_posix()] = digest
    actual: set[str] = set()
    for target in root.rglob("*"):
        require(not target.is_symlink(), f"symlink in closed tree: {target}")
        if target.is_file():
            relative = target.relative_to(root).as_posix()
            if relative != name and not (mode == "result" and relative == completion_name):
                actual.add(relative)
        else:
            require(target.is_dir(), f"special entry in closed tree: {target}")
    require(actual == set(listed), f"{name} exact-tree closure drift")


def _runtime_user() -> str:
    user = pwd.getpwuid(os.getuid()).pw_name
    require(re.fullmatch(r"[A-Za-z0-9._-]+", user) is not None and user not in {".", ".."}, "unsafe user")
    return user


def validate_bundle(
    values: Mapping[str, str], rung: str, user: str, *, local_test_mode: bool
) -> Path:
    bundle = Path(values["UED_BUNDLE_DIR"])
    namespace = (
        bundle.parent if local_test_mode
        else Path(f"/scratch/{user}/maxrl/bundles/ued_minimax_v4_engineering")
    )
    _canonical_directory(bundle, namespace, "bundle")
    require(bundle.name == values["UED_BUNDLE_MANIFEST_SHA256"][:20], "bundle content-address drift")
    require(bundle.name != "06ffeeeb6998e8ddb1ce", "protected v3 bundle forbidden")
    _validate_manifest(bundle, "SHA256SUMS", values["UED_BUNDLE_MANIFEST_SHA256"], "exact")
    require(values["UED_UPSTREAM_COMMIT"] == BASE_COMMIT, "upstream commit drift")
    require(values["UED_UPSTREAM_TREE"] == BASE_TREE, "upstream tree drift")
    remote_state = bundle / "REMOTE_HARDENING_STATE.json"
    require(sha256(remote_state) == values["UED_REMOTE_HARDENING_STATE_SHA256"], "remote state drift")
    state = _load(remote_state, "remote-hardening state")
    require(
        state.get("schema") == 1
        and state.get("status") == "local_candidate_remote_submission_hold"
        and state.get("historical_base_bundle_manifest_sha256") == BASE_BUNDLE_SHA256
        and state.get("paper_evidence") is False
        and state.get("analyzer_eligible") is False
        and state.get("production_authorized") is False
        and state.get("endpoint_access_authorized") is False
        and state.get("cost100_implemented") is False
        and state.get("remote_submission_authorized") is False
        and state.get("max_student_updates") == 1
        and state.get("protected_v3_job", {}).get("job_id") == "9367063"
        and state.get("protected_v3_job", {}).get("mutation_forbidden") is True,
        "remote-hardening state semantics drift",
    )
    hardening = bundle / EXPECTED_HARDENING_SBATCH[rung]
    legacy = bundle / EXPECTED_LEGACY_SBATCH[rung]
    require(sha256(hardening) == values["UED_HARDENING_SBATCH_SHA256"], "hardening sbatch drift")
    require(sha256(legacy) == values["UED_LEGACY_SBATCH_SHA256"], "legacy sbatch drift")
    require(values["UED_SBATCH_SHA256"] == values["UED_LEGACY_SBATCH_SHA256"], "legacy environment binding drift")
    require(sha256(Path(__file__).resolve()) == values["UED_JOB_GUARD_SHA256"], "job-guard self drift")
    return bundle


def _verify_environment(
    values: Mapping[str, str], bundle: Path, user: str, *,
    local_test_mode: bool, local_conda: Path | None,
) -> dict[str, Any]:
    environment = Path(values["UED_ENV_DIR"])
    closure = Path(values["UED_ENV_TREE_DIR"])
    if local_test_mode:
        require(local_conda is not None, "local test requires fixed Conda path")
        conda = local_conda
    else:
        require(local_conda is None, "Conda override forbidden")
        conda = Path(f"/home/{user}/miniconda3/bin/conda")
        _canonical_directory(environment, Path(f"/scratch/{user}"), "environment")
        _canonical_directory(closure, Path(f"/scratch/{user}"), "environment closure")
    tool_path = bundle / "ued_benchmark/hopper_v4_remote_hardening/environment_tree.py"
    tool = _module(tool_path, values["UED_ENV_TREE_TOOL_SHA256"], "environment_tree")
    receipt, manifest_sha = tool.verify(
        environment, conda, closure, values["UED_ENV_TREE_TOOL_SHA256"],
        "3.10.20", values["UED_ENV_TREE_MANIFEST_SHA256"],
        values["UED_ENV_TREE_RECEIPT_SHA256"],
    )
    require(manifest_sha == values["UED_ENV_TREE_MANIFEST_SHA256"], "environment closure drift")
    return receipt


def _validate_pair(values: Mapping[str, str], bundle: Path, rung: str) -> dict[str, Any] | None:
    if rung != "terminal":
        return None
    path = bundle / "ued_benchmark/hopper_v4_remote_hardening/pair_plan.py"
    tool = _module(path, values["UED_PAIR_PLAN_TOOL_SHA256"], "pair_plan")
    plan = tool.validate(Path(values["UED_PAIR_PLAN_DIR"]), values["UED_PAIR_PLAN_MANIFEST_SHA256"])
    arm = values["UED_ARM"]
    require(arm in {"frontier", "maxmc"}, "terminal arm drift")
    require(plan["common"]["bundle_manifest_sha256"] == values["UED_BUNDLE_MANIFEST_SHA256"], "pair/bundle drift")
    require(plan["common"]["import"]["result_dir"] == values["UED_IMPORT_SMOKE_RESULT_DIR"], "pair/import path drift")
    require(plan["common"]["import"]["manifest_sha256"] == values["UED_IMPORT_SMOKE_MANIFEST_SHA256"], "pair/import hash drift")
    require(plan["common"]["one_update"]["result_dir"] == values["UED_ONE_UPDATE_RESULT_DIR"], "pair/one-update path drift")
    require(plan["common"]["one_update"]["manifest_sha256"] == values["UED_ONE_UPDATE_MANIFEST_SHA256"], "pair/one-update hash drift")
    require(plan["arms"][arm]["config_sha256"] == values["UED_CONFIG_SHA256"], "pair arm/config drift")
    return plan


def _runtime_parent(rung: str, user: str, local_parent: Path | None) -> Path:
    if local_parent is not None:
        require(local_parent.is_absolute() and local_parent.resolve(strict=True) == local_parent, "unsafe local runtime parent")
        return local_parent
    parent = Path(f"/scratch/{user}/maxrl/tests/ued-minimax-v4-remote-hardening/{rung}")
    parent.mkdir(parents=True, exist_ok=True)
    return _canonical_directory(parent, Path(f"/scratch/{user}"), "runtime parent")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def preflight(cli: argparse.Namespace) -> tuple[Path, str]:
    values = parse_envelope(cli.input_envelope, cli.rung)
    require(
        all(os.environ.get(key) == value for key, value in values.items())
        and not any(key.startswith("UED_") and key not in values for key in os.environ),
        "runtime UED environment differs from exact input envelope",
    )
    user = _runtime_user()
    bundle = validate_bundle(values, cli.rung, user, local_test_mode=cli.local_test_mode)
    environment_receipt = _verify_environment(
        values, bundle, user, local_test_mode=cli.local_test_mode,
        local_conda=cli.local_conda,
    )
    plan = _validate_pair(values, bundle, cli.rung)
    job_id = "local-test" if cli.local_test_mode else os.environ.get("SLURM_JOB_ID", "")
    require(job_id == "local-test" or re.fullmatch(r"[0-9]+", job_id) is not None, "job identity drift")
    parent = _runtime_parent(cli.rung, user, cli.local_runtime_parent)
    runtime = parent / f"{job_id}-{values.get('UED_ARM', cli.rung)}"
    require(not runtime.exists() and not runtime.is_symlink(), "hardening runtime already exists")
    runtime.mkdir(mode=0o750)
    try:
        gpu_path = runtime / "gpu-runtime.json"
        gpu_tool_path = bundle / "ued_benchmark/hopper_v4_remote_hardening/gpu_runtime_probe.py"
        gpu_tool = _module(gpu_tool_path, values["UED_GPU_PROBE_TOOL_SHA256"], "gpu_runtime_probe")
        _, gpu_sha = gpu_tool.probe(
            gpu_path, cli.rung, values["UED_GPU_PROBE_TOOL_SHA256"],
            values["UED_ENV_TREE_MANIFEST_SHA256"],
            local_test_mode=cli.local_test_mode, mock_input=cli.gpu_mock,
            environment=os.environ,
        )
        receipt = {
            "schema": 1, "status": "preflight_complete", "purpose": PURPOSE,
            "rung": cli.rung, "job_id": job_id, "paper_evidence": False,
            "analyzer_eligible": False, "production_authorized": False,
            "endpoint_access_authorized": False, "cost100_implemented": False,
            "max_student_updates": 0 if cli.rung == "import" else 1,
            "input_envelope_path": str(cli.input_envelope),
            "input_envelope_sha256": sha256(cli.input_envelope),
            "bundle_manifest_sha256": values["UED_BUNDLE_MANIFEST_SHA256"],
            "environment_tree_manifest_sha256": values["UED_ENV_TREE_MANIFEST_SHA256"],
            "environment_tree_receipt_sha256": values["UED_ENV_TREE_RECEIPT_SHA256"],
            "environment_python_sha256": environment_receipt["python_sha256"],
            "gpu_runtime_receipt_path": str(gpu_path),
            "gpu_runtime_receipt_sha256": gpu_sha,
            "pair_plan_manifest_sha256": (
                values["UED_PAIR_PLAN_MANIFEST_SHA256"] if plan is not None else None
            ),
        }
        _write_json(runtime / "PRECHECK.json", receipt)
        return runtime, sha256(runtime / "PRECHECK.json")
    except Exception:
        if runtime.exists():
            shutil.rmtree(runtime)
        raise


def _discover_result(values: Mapping[str, str], rung: str, user: str, job_id: str) -> tuple[Path, str, str]:
    scratch = Path(f"/scratch/{user}")
    if rung == "import":
        root = scratch / f"maxrl/tests/ued-minimax-v4-import/{job_id}"
        manifest_name, complete_name = "SHA256SUMS", "COMPLETE"
    elif rung == "one_update":
        parent = scratch / "maxrl/tests/ued-minimax-v4-one-update"
        matches = sorted(parent.glob(f"*/job-{job_id}"))
        require(len(matches) == 1, "one-update result identity is not unique")
        root = matches[0]
        manifest_name, complete_name = "SHA256SUMS", "COMPLETE"
    else:
        root = scratch / f"maxrl/tests/ued-minimax-v4-terminal-components/{job_id}-{values['UED_ARM']}"
        manifest_name, complete_name = "SHA256SUMS", "COMPONENTS_COMPLETE.json"
    _canonical_directory(root, scratch, "legacy result")
    require((root / complete_name).is_file() and not (root / complete_name).is_symlink(), "legacy result incomplete")
    manifest_sha = sha256(root / manifest_name)
    _validate_manifest(
        root, manifest_name, manifest_sha, "result", completion_name=complete_name
    )
    return root, manifest_name, manifest_sha


def postflight(cli: argparse.Namespace) -> tuple[Path, str]:
    values = parse_envelope(cli.input_envelope, cli.rung)
    require(
        all(os.environ.get(key) == value for key, value in values.items())
        and not any(key.startswith("UED_") and key not in values for key in os.environ),
        "postflight UED environment drift",
    )
    user = _runtime_user()
    bundle = validate_bundle(values, cli.rung, user, local_test_mode=cli.local_test_mode)
    _verify_environment(
        values, bundle, user, local_test_mode=cli.local_test_mode,
        local_conda=cli.local_conda,
    )
    _validate_pair(values, bundle, cli.rung)
    job_id = "local-test" if cli.local_test_mode else os.environ.get("SLURM_JOB_ID", "")
    parent = _runtime_parent(cli.rung, user, cli.local_runtime_parent)
    runtime = parent / f"{job_id}-{values.get('UED_ARM', cli.rung)}"
    _canonical_directory(runtime, parent, "hardening runtime")
    precheck_path = runtime / "PRECHECK.json"
    precheck = _load(precheck_path, "hardening precheck")
    require(
        precheck.get("status") == "preflight_complete"
        and precheck.get("input_envelope_sha256") == sha256(cli.input_envelope)
        and precheck.get("job_id") == job_id
        and precheck.get("rung") == cli.rung,
        "hardening precheck drift",
    )
    if cli.local_test_mode:
        require(cli.local_result_dir is not None, "local postflight requires explicit mock result")
        result = cli.local_result_dir
        _canonical_directory(result, result.parent, "local legacy result")
        manifest_name = cli.local_manifest_name
        manifest_sha = sha256(result / manifest_name)
        _validate_manifest(result, manifest_name, manifest_sha, "result")
    else:
        require(cli.local_result_dir is None, "legacy result override forbidden")
        result, manifest_name, manifest_sha = _discover_result(values, cli.rung, user, job_id)
    receipt = {
        "schema": 1, "status": "complete", "purpose": PURPOSE,
        "rung": cli.rung, "job_id": job_id, "paper_evidence": False,
        "analyzer_eligible": False, "production_authorized": False,
        "endpoint_access_authorized": False, "cost100_implemented": False,
        "max_student_updates": 0 if cli.rung == "import" else 1,
        "bundle_manifest_sha256": values["UED_BUNDLE_MANIFEST_SHA256"],
        "legacy_result_dir": str(result), "legacy_manifest_sha256": manifest_sha,
        "environment_tree_manifest_sha256": values["UED_ENV_TREE_MANIFEST_SHA256"],
        "environment_tree_receipt_sha256": values["UED_ENV_TREE_RECEIPT_SHA256"],
        "environment_verified_before": True, "environment_verified_after": True,
        "gpu_runtime_receipt_path": precheck["gpu_runtime_receipt_path"],
        "gpu_runtime_receipt_sha256": precheck["gpu_runtime_receipt_sha256"],
        "hardening_sbatch_sha256": values["UED_HARDENING_SBATCH_SHA256"],
        "legacy_sbatch_sha256": values["UED_LEGACY_SBATCH_SHA256"],
        "job_guard_sha256": values["UED_JOB_GUARD_SHA256"],
        "pair_plan_manifest_sha256": values.get("UED_PAIR_PLAN_MANIFEST_SHA256"),
    }
    output = runtime / "HARDENING_RECEIPT.json"
    _write_json(output, receipt)
    _write_json(runtime / "COMPLETE", {
        "schema": 1, "status": "complete", "rung": cli.rung,
        "job_id": job_id, "hardening_receipt_sha256": sha256(output),
        "paper_evidence": False, "production_authorized": False,
    })
    return output, sha256(output)


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "postflight"):
        child = subparsers.add_parser(command)
        child.add_argument("--rung", choices=("import", "one_update", "terminal"), required=True)
        child.add_argument("--input-envelope", type=Path, required=True)
        child.add_argument("--local-test-mode", action="store_true")
        child.add_argument("--local-runtime-parent", type=Path)
        child.add_argument("--local-conda", type=Path)
        if command == "preflight":
            child.add_argument("--gpu-mock", type=Path)
        else:
            child.add_argument("--local-result-dir", type=Path)
            child.add_argument("--local-manifest-name", default="SHA256SUMS")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    cli = parse_cli(argv)
    try:
        if cli.local_test_mode:
            require(cli.local_runtime_parent is not None and cli.local_conda is not None, "incomplete local-test closure")
        else:
            require(cli.local_runtime_parent is None and cli.local_conda is None, "local override forbidden")
        if cli.command == "preflight":
            runtime, digest = preflight(cli)
            print(f"V4H_PREFLIGHT_COMPLETE path={runtime} receipt_sha256={digest}")
        else:
            receipt, digest = postflight(cli)
            print(f"V4H_POSTFLIGHT_COMPLETE path={receipt} sha256={digest}")
    except (JobGuardError, OSError, UnicodeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"V4H_JOB_GUARD_REFUSED: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
