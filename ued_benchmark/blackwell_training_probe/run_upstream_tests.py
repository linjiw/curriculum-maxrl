#!/usr/bin/env python3
"""Run minimax's legacy upstream tests without polluting PYTHONPATH."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pytest-target", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PYTHONPATH"):
        raise RuntimeError("PYTHONPATH must be unset")
    required = {
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": "cpu",
        "JAX_THREEFRY_PARTITIONABLE": "false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    for name, expected in required.items():
        if os.environ.get(name, "").lower() != expected:
            raise RuntimeError(f"{name}={expected} is required")

    source = args.source.resolve()
    pytest_target = args.pytest_target.resolve()
    sys.path[:0] = [str(pytest_target), str(source / "src")]
    import minimax
    import jax

    if not Path(minimax.__file__).resolve().is_relative_to(source):
        raise RuntimeError("minimax imported outside declared source")
    if jax.default_backend() != "cpu" or jax.devices()[0].platform != "cpu":
        raise RuntimeError("upstream tests are not isolated to CPU")
    if jax.config.jax_threefry_partitionable is not False:
        raise RuntimeError("source-era Threefry mode is not active")

    # The archived upstream tests use `envs`, `models`, `agents`, `util`, and
    # `tests` as top-level packages even though production code imports them as
    # `minimax.*`. Alias already-loaded canonical modules so collection does not
    # register environments/models twice. Test source is not modified.
    for name, module in list(sys.modules.items()):
        if name.startswith("minimax.") and name.split(".")[1] in {
            "envs",
            "models",
            "agents",
            "util",
        }:
            sys.modules.setdefault(name[len("minimax.") :], module)
    import minimax.tests

    sys.modules["tests"] = minimax.tests
    import minimax.tests.base_req_rollout_storage as base
    import minimax.tests.dummy_test_envs as dummy

    sys.modules["tests.base_req_rollout_storage"] = base
    sys.modules["tests.dummy_test_envs"] = dummy
    import pytest

    return int(pytest.main(["-q", str(source / "src/minimax/tests")]))


if __name__ == "__main__":
    raise SystemExit(main())
