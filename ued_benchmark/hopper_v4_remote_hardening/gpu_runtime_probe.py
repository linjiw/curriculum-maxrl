#!/usr/bin/env python3
"""Fail-closed runtime proof for one Hopper A100 MIG 1g.10gb allocation."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence


HASH_RE = re.compile(r"^[0-9a-f]{64}$")
MIB = 1024 * 1024
MIN_MIG_BYTES = 9_000 * MIB
MAX_MIG_BYTES = 11_000 * MIB


class GPUProbeError(RuntimeError):
    """Raised when the allocation is not the exact bounded MIG profile."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GPUProbeError(message)


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slurm_identity(
    environment: Mapping[str, str], rung: str, local_test_mode: bool
) -> dict[str, Any]:
    expected_job = "local-test" if local_test_mode else environment.get("SLURM_JOB_ID")
    expected_name = {
        "import": "ued-v4h-import",
        "one_update": "ued-v4h-one-update",
        "terminal": "ued-v4h-terminal",
    }[rung]
    require(
        expected_job == ("local-test" if local_test_mode else environment.get("SLURM_JOB_ID"))
        and (local_test_mode or re.fullmatch(r"[0-9]+", expected_job or "") is not None),
        "Slurm job identity drift",
    )
    require(
        not environment.get("SLURM_ARRAY_JOB_ID")
        and not environment.get("SLURM_ARRAY_TASK_ID"),
        "arrays forbidden",
    )
    require(environment.get("SLURM_RESTART_COUNT", "0") == "0", "restarted job forbidden")
    require(environment.get("SLURM_JOB_NAME") == expected_name, "Slurm job-name drift")
    require(environment.get("SLURM_JOB_PARTITION") == "gpuq", "partition drift")
    require(environment.get("SLURM_JOB_QOS") == "gpu", "QOS drift")
    require(environment.get("SLURM_CPUS_PER_TASK") == "2", "CPU allocation drift")
    require(environment.get("SLURM_JOB_NUM_NODES", "1") == "1", "node allocation drift")
    require(environment.get("SLURM_NTASKS") == "1", "task allocation drift")
    require(environment.get("SLURM_MEM_PER_NODE") == "15360", "memory allocation drift")
    require(environment.get("SLURM_GPUS_ON_NODE") == "1", "GPU count allocation drift")
    require(
        environment.get("SLURM_TRES_PER_NODE") == "gres/gpu:1g.10gb:1",
        "typed MIG allocation drift",
    )
    require(environment.get("SLURM_EXPORT_ENV") == "NIL", "Slurm export mode drift")
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    job_gpus = environment.get("SLURM_JOB_GPUS", "")
    for value, label in ((visible, "CUDA_VISIBLE_DEVICES"), (job_gpus, "SLURM_JOB_GPUS")):
        require(
            value and "," not in value and not any(character.isspace() for character in value),
            f"{label} must identify exactly one device",
        )
    return {
        "job_id": expected_job,
        "job_name": expected_name,
        "partition": "gpuq",
        "qos": "gpu",
        "cuda_visible_devices": visible,
        "slurm_job_gpus": job_gpus,
        "slurm_gpus_on_node": 1,
        "slurm_cpus_per_task": 2,
        "slurm_job_num_nodes": 1,
        "slurm_ntasks": 1,
        "slurm_mem_per_node_mib": 15360,
        "slurm_tres_per_node": "gres/gpu:1g.10gb:1",
        "slurm_restart_count": 0,
        "array_job": False,
        "export_mode": "NIL",
    }


def _cuda_driver_probe() -> dict[str, Any]:
    try:
        library = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        raise GPUProbeError("CUDA driver library unavailable") from exc
    require(library.cuInit(0) == 0, "cuInit failed")
    count = ctypes.c_int()
    require(library.cuDeviceGetCount(ctypes.byref(count)) == 0, "cuDeviceGetCount failed")
    require(count.value == 1, "CUDA driver sees other than one device")
    device = ctypes.c_int()
    require(library.cuDeviceGet(ctypes.byref(device), 0) == 0, "cuDeviceGet failed")
    name_buffer = ctypes.create_string_buffer(256)
    require(
        library.cuDeviceGetName(name_buffer, len(name_buffer), device) == 0,
        "cuDeviceGetName failed",
    )
    total = ctypes.c_size_t()
    total_mem = getattr(library, "cuDeviceTotalMem_v2", None) or getattr(
        library, "cuDeviceTotalMem", None
    )
    require(total_mem is not None, "CUDA total-memory API unavailable")
    require(total_mem(ctypes.byref(total), device) == 0, "cuDeviceTotalMem failed")
    name = name_buffer.value.decode("utf-8", errors="strict")
    return {"device_count": count.value, "device_name": name,
            "total_memory_bytes": int(total.value)}


def _jax_probe() -> dict[str, Any]:
    try:
        import jax
    except Exception as exc:  # pragma: no cover - exercised only on Hopper
        raise GPUProbeError("JAX import failed") from exc
    require(jax.default_backend() == "gpu", "JAX backend is not GPU")
    devices = jax.devices("gpu")
    require(len(devices) == 1, "JAX sees other than one GPU device")
    device = devices[0]
    return {
        "backend": "gpu",
        "device_count": 1,
        "device_id": int(device.id),
        "device_kind": str(device.device_kind),
        "platform": str(device.platform),
    }


def _load_mock(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), "unsafe GPU mock")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(
        isinstance(value, dict) and set(value) == {"cuda_driver", "jax"},
        "GPU mock keys drift",
    )
    return value["cuda_driver"], value["jax"]


def validate_measurements(cuda: Mapping[str, Any], jax: Mapping[str, Any]) -> None:
    require(
        set(cuda) == {"device_count", "device_name", "total_memory_bytes"}
        and cuda.get("device_count") == 1
        and isinstance(cuda.get("device_name"), str)
        and "A100" in cuda["device_name"]
        and isinstance(cuda.get("total_memory_bytes"), int)
        and MIN_MIG_BYTES <= cuda["total_memory_bytes"] <= MAX_MIG_BYTES,
        "CUDA device is not exactly one A100 1g.10gb-class MIG slice",
    )
    require(
        set(jax) == {"backend", "device_count", "device_id", "device_kind", "platform"}
        and jax.get("backend") == "gpu"
        and jax.get("platform") == "gpu"
        and jax.get("device_count") == 1
        and jax.get("device_id") == 0
        and isinstance(jax.get("device_kind"), str)
        and "A100" in jax["device_kind"],
        "JAX device is not exactly one A100 GPU",
    )


def probe(
    output: Path,
    rung: str,
    expected_tool_sha256: str,
    environment_tree_manifest_sha256: str,
    *,
    local_test_mode: bool,
    mock_input: Path | None,
    environment: Mapping[str, str],
) -> tuple[Path, str]:
    tool = Path(__file__).resolve()
    require(rung in {"import", "one_update", "terminal"}, "invalid rung")
    require(HASH_RE.fullmatch(expected_tool_sha256 or "") is not None, "bad tool hash")
    require(sha256(tool) == expected_tool_sha256, "GPU probe tool drift")
    require(
        HASH_RE.fullmatch(environment_tree_manifest_sha256 or "") is not None,
        "bad environment-tree hash",
    )
    require(output.is_absolute() and ".." not in output.parts, "output must be absolute")
    require(not output.exists() and not output.is_symlink(), "GPU receipt output exists")
    require(output.parent.is_dir() and output.parent.resolve(strict=True) == output.parent, "unsafe output parent")
    slurm = _slurm_identity(environment, rung, local_test_mode)
    if local_test_mode:
        require(mock_input is not None, "local GPU probe requires an explicit mock")
        cuda, jax = _load_mock(mock_input)
    else:
        require(mock_input is None, "GPU mock forbidden outside local test mode")
        cuda, jax = _cuda_driver_probe(), _jax_probe()
    validate_measurements(cuda, jax)
    receipt = {
        "schema": 1,
        "status": "complete",
        "purpose": "v4_remote_hardening_a100_mig_runtime_integrity",
        "paper_evidence": False,
        "production_authorized": False,
        "requested_gres": "gpu:1g.10gb:1",
        "rung": rung,
        "environment_tree_manifest_sha256": environment_tree_manifest_sha256,
        "probe_tool_sha256": expected_tool_sha256,
        "slurm": slurm,
        "cuda_driver": dict(cuda),
        "jax": dict(jax),
    }
    descriptor, raw = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output, sha256(output)


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rung", choices=("import", "one_update", "terminal"), required=True)
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--environment-tree-manifest-sha256", required=True)
    parser.add_argument("--local-test-mode", action="store_true")
    parser.add_argument("--mock-input", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    cli = parse_cli(argv)
    try:
        output, digest = probe(
            cli.output, cli.rung, cli.expected_tool_sha256,
            cli.environment_tree_manifest_sha256,
            local_test_mode=cli.local_test_mode, mock_input=cli.mock_input,
            environment=os.environ,
        )
    except (GPUProbeError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        print(f"V4H_GPU_RUNTIME_REFUSED: {exc}", file=os.sys.stderr)
        return 1
    print(f"V4H_GPU_RUNTIME_COMPLETE path={output} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
