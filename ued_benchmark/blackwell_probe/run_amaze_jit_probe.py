#!/usr/bin/env python3
"""Run one bounded minimax AMaze reset/step JIT on the Blackwell probe lane."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("manifest.json"),
    )
    args = parser.parse_args()

    if os.environ.get("PYTHONPATH"):
        raise RuntimeError("PYTHONPATH must be unset for this isolated probe")
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise RuntimeError("XLA_PYTHON_CLIENT_PREALLOCATE=false is required")

    source = args.source.resolve()
    manifest = json.loads(args.manifest.read_text())
    if _git(source, "rev-parse", "HEAD") != manifest["source"]["upstream_commit"]:
        raise RuntimeError("unexpected minimax Git commit")
    overlay_path = source / ".frontierrl_overlay.json"
    overlay = json.loads(overlay_path.read_text())
    if overlay["overlay_contract_sha256"] != manifest["source"]["overlay_contract_sha256"]:
        raise RuntimeError("unexpected Frontier overlay contract")
    environment_path = source / manifest["modernization_patch"]["touched_file"]
    if _sha256(environment_path) != manifest["modernization_patch"]["patched_file_sha256"]:
        raise RuntimeError("modernized source does not match manifest")

    sys.path.insert(0, str(source / "src"))
    import jax
    import jaxlib
    import minimax
    from minimax import envs

    if not Path(minimax.__file__).resolve().is_relative_to(source):
        raise RuntimeError(f"minimax imported from wrong tree: {minimax.__file__}")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"expected gpu backend, got {jax.default_backend()}")

    env, _ = envs.make(
        "Maze",
        env_kwargs={
            "max_episode_steps": 32,
            "height": 13,
            "width": 13,
            "n_walls": 20,
            "agent_view_size": 5,
            "see_through_walls": False,
        },
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    reset_key, step_key = jax.random.split(jax.random.PRNGKey(0))

    started = time.monotonic()
    obs, state = reset(reset_key)
    jax.block_until_ready(obs)
    reset_seconds = time.monotonic() - started

    started = time.monotonic()
    next_obs, next_state, reward, done, info = step(step_key, state, 0)
    jax.block_until_ready((next_obs, next_state, reward, done, info))
    step_seconds = time.monotonic() - started

    if tuple(obs["image"].shape) != (5, 5, 3):
        raise RuntimeError(f"unexpected reset observation shape: {obs['image'].shape}")
    if tuple(next_obs["image"].shape) != (5, 5, 3):
        raise RuntimeError(f"unexpected step observation shape: {next_obs['image'].shape}")
    if int(next_state.time) != 1:
        raise RuntimeError(f"one step should produce time=1, got {next_state.time}")

    device = jax.devices()[0]
    result = {
        "action": 0,
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": getattr(device, "device_kind", "unknown"),
        "done": bool(done),
        "flax": importlib.metadata.version("flax"),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "minimax_import": str(Path(minimax.__file__).resolve()),
        "next_obs_image_shape": list(next_obs["image"].shape),
        "numpy": importlib.metadata.version("numpy"),
        "obs_image_shape": list(obs["image"].shape),
        "optax": importlib.metadata.version("optax"),
        "overlay_contract_sha256": overlay["overlay_contract_sha256"],
        "reset_compile_execute_seconds": round(reset_seconds, 6),
        "reward": float(reward),
        "source_commit": _git(source, "rev-parse", "HEAD"),
        "state_time": int(next_state.time),
        "step_compile_execute_seconds": round(step_seconds, 6),
    }
    print("AMAZE_JIT_OK " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
