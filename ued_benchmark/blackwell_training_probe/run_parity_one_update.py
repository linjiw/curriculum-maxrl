#!/usr/bin/env python3
"""Run one exact grouped Frontier engineering update and compare lane parity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "ued_benchmark/configs/maze_frontier_exact_grouped_n8.json"
FRONTIER_CONTRACT = REPO_ROOT / "ued_benchmark/OVERLAY_CONTRACT.json"
MODERN_CONTRACT = Path(__file__).resolve().with_name("MODERNIZATION_CONTRACT.json")
PARITY_PROTOCOL = Path(__file__).resolve().with_name("PARITY_PROTOCOL.json")
UPSTREAM_COMMIT = "d053054c5290a04c1c4cd8b55704d999cad73e30"
UPSTREAM_TREE = "b0cace1fc54984e21a842f12d15d0b899e33d270"
PARENT_MANIFEST_SHA256 = "d929efa2f059a93125e217ec4713ae81670c769d979c67abd2b10efc64268af3"
CONFIG_SHA256 = "b49168142a9d5a5d8edce88634975ac52d7615dca8bad9fff1cbcaf29ec43508"
FRONTIER_CONTRACT_SHA256 = "5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000"
PARITY_PROTOCOL_SHA256 = "bcd1ba38435b43312e8f4559fad4efdae96169a9a806cb41d551dc76bb8420aa"


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def expected_modern_manifest(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "overlay": contract["overlay"],
        "contract_sha256": sha256(MODERN_CONTRACT),
        "parent_overlay": contract["parent_overlay"],
        "parent_overlay_contract_sha256": contract["parent_overlay_contract_sha256"],
        "parent_applied_manifest_sha256": contract["parent_applied_manifest_sha256"],
        "upstream_commit": contract["upstream_commit"],
        "total_replacements": contract["total_replacements"],
        "file_sha256": {
            name: details["applied_sha256"]
            for name, details in contract["files"].items()
        },
        "paper_evidence": False,
    }


def validate_source(source: Path, lane: str) -> dict[str, Any]:
    require(source.is_dir() and not source.is_symlink(), "unsafe source directory")
    require(git(source, "rev-parse", "HEAD") == UPSTREAM_COMMIT, "source commit drift")
    require(git(source, "rev-parse", "HEAD^{tree}") == UPSTREAM_TREE, "source tree drift")
    parent_path = source / ".frontierrl_overlay.json"
    require(parent_path.is_file() and not parent_path.is_symlink(), "unsafe parent manifest")
    require(sha256(parent_path) == PARENT_MANIFEST_SHA256, "parent manifest digest drift")
    parent = json.loads(parent_path.read_text())
    require(
        parent.get("overlay_contract_sha256") == FRONTIER_CONTRACT_SHA256,
        "Frontier contract drift",
    )

    modern_contract = json.loads(MODERN_CONTRACT.read_text())
    removed_count = 0
    for path in (source / "src/minimax").rglob("*.py"):
        removed_count += path.read_text().count("jax.tree_map")

    if lane == "reference":
        require(removed_count == 35, f"reference removed-API count drift: {removed_count}")
        require(
            not (source / ".blackwell_training_overlay.json").exists(),
            "reference source unexpectedly has modernization manifest",
        )
        for relative, digest in parent["overlay_file_sha256"].items():
            require(sha256(source / relative) == digest, f"reference overlay drift: {relative}")
        modern_manifest_sha = None
    else:
        require(removed_count == 0, f"modern source retains removed API: {removed_count}")
        modern_path = source / ".blackwell_training_overlay.json"
        require(modern_path.is_file() and not modern_path.is_symlink(), "unsafe modern manifest")
        modern = json.loads(modern_path.read_text())
        require(modern == expected_modern_manifest(modern_contract), "modern manifest drift")
        for relative, details in modern_contract["files"].items():
            require(
                sha256(source / relative) == details["applied_sha256"],
                f"modern source digest drift: {relative}",
            )
        modern_manifest_sha = sha256(modern_path)
    return {
        "parent_manifest_sha256": sha256(parent_path),
        "modernization_contract_sha256": sha256(MODERN_CONTRACT),
        "modernization_manifest_sha256": modern_manifest_sha,
        "removed_api_occurrences": removed_count,
    }


def scalar_stats(np: Any, stats: Mapping[str, Any]) -> dict[str, float | int | bool | str]:
    keep = (
        "actor_loss",
        "entropy",
        "grad_norm",
        "mean_gae",
        "mean_target",
        "mean_value",
        "n_updates",
        "plr/frontier_duplicate_new_group_count",
        "plr/frontier_group_size_match",
        "plr/frontier_incomplete_group_count",
        "plr/frontier_n_eval",
        "plr/frontier_n_rollouts",
        "plr/frontier_total_successes",
        "plr/frontier_total_trials",
        "return",
        "total_loss",
        "value_loss",
    )
    result: dict[str, float | int | bool | str] = {}
    for key in keep:
        if key not in stats:
            continue
        array = np.asarray(stats[key])
        if array.size != 1:
            continue
        value = array.reshape(-1)[0]
        if np.issubdtype(array.dtype, np.bool_):
            result[key] = bool(value)
        elif np.issubdtype(array.dtype, np.integer):
            result[key] = int(value)
        elif np.issubdtype(array.dtype, np.floating):
            number = float(value)
            if np.isnan(number):
                result[key] = "nan"
            elif np.isposinf(number):
                result[key] = "+inf"
            elif np.isneginf(number):
                result[key] = "-inf"
            else:
                result[key] = number
    return result


def leaf_signature(jax: Any, np: Any, tree: Any) -> dict[str, Any]:
    entries = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.asarray(leaf)
        contiguous = np.ascontiguousarray(array)
        entry: dict[str, Any] = {
            "path": jax.tree_util.keystr(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "size": int(array.size),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
        if np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
            numeric = array.astype(np.complex128, copy=False)
            finite_mask = np.isfinite(numeric)
            finite_values = np.abs(numeric[finite_mask])
            entry.update(
                {
                    "nan_count": int(np.isnan(numeric).sum()),
                    "infinite_count": int(np.isinf(numeric).sum()),
                    "positive_inf_count": (
                        None if np.issubdtype(array.dtype, np.complexfloating)
                        else int(np.isposinf(array).sum())
                    ),
                    "negative_inf_count": (
                        None if np.issubdtype(array.dtype, np.complexfloating)
                        else int(np.isneginf(array).sum())
                    ),
                    "abs_sum": float(finite_values.sum(dtype=np.float64)),
                    "squared_l2": float(
                        np.square(finite_values).sum(dtype=np.float64)
                    ),
                    "max_abs": float(finite_values.max(initial=0.0)),
                }
            )
        entries.append(entry)
    structure = [
        {key: entry[key] for key in ("path", "shape", "dtype", "size")}
        for entry in entries
    ]
    return {
        "leaf_count": len(entries),
        "structure_sha256": hashlib.sha256(
            json.dumps(structure, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "leaves": entries,
    }


def compare_leaf_signatures(
    np: Any,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    rtol: float,
    atol: float,
    label: str,
) -> dict[str, Any]:
    require(reference["leaf_count"] == candidate["leaf_count"], f"{label} leaf count drift")
    require(
        reference["structure_sha256"] == candidate["structure_sha256"],
        f"{label} checkpoint structure drift",
    )
    max_relative = 0.0
    max_absolute = 0.0
    exact_leaf_hashes = 0
    for before, after in zip(reference["leaves"], candidate["leaves"]):
        require(
            (before["path"], before["shape"], before["dtype"], before["size"])
            == (after["path"], after["shape"], after["dtype"], after["size"]),
            f"{label} leaf structure drift: {before['path']}",
        )
        if before["sha256"] == after["sha256"]:
            exact_leaf_hashes += 1
        for metric in (
            "nan_count",
            "infinite_count",
            "positive_inf_count",
            "negative_inf_count",
        ):
            if metric in before or metric in after:
                require(
                    before.get(metric) == after.get(metric),
                    f"{label} non-finite sentinel drift at {before['path']}:{metric}",
                )
        for metric in ("abs_sum", "squared_l2", "max_abs"):
            if metric not in before and metric not in after:
                continue
            require(metric in before and metric in after, f"{label} numeric type drift")
            left = float(before[metric])
            right = float(after[metric])
            absolute = abs(left - right)
            relative = absolute / max(abs(left), atol)
            max_absolute = max(max_absolute, absolute)
            max_relative = max(max_relative, relative)
            require(
                bool(np.isclose(left, right, rtol=rtol, atol=atol)),
                f"{label} numeric drift at {before['path']}:{metric}: {left} != {right}",
            )
    return {
        "exact_leaf_hashes": exact_leaf_hashes,
        "leaf_count": candidate["leaf_count"],
        "max_aggregate_absolute_error": max_absolute,
        "max_aggregate_relative_error": max_relative,
    }


def compare_reference(
    np: Any,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    backend: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    tolerances = protocol["tolerances"][backend]
    stat_rtol = float(tolerances["stat_rtol"])
    stat_atol = float(tolerances["stat_atol"])
    leaf_rtol = float(tolerances["leaf_aggregate_rtol"])
    leaf_atol = float(tolerances["leaf_aggregate_atol"])
    require(reference["schedule"] == candidate["schedule"], "engineering schedule drift")
    require(reference["final_state"] == candidate["final_state"], "final counter drift")
    require(
        reference["checkpoint"]["structure_sha256"]
        == candidate["checkpoint"]["structure_sha256"],
        "serialized checkpoint structure drift",
    )
    require(
        reference["checkpoint"]["serialized_leaf_count"]
        == candidate["checkpoint"]["serialized_leaf_count"],
        "serialized checkpoint leaf-count drift",
    )
    for ref_cycle, cand_cycle in zip(reference["cycles"], candidate["cycles"]):
        require(ref_cycle["state"] == cand_cycle["state"], "cycle counter drift")
        require(set(ref_cycle["stats"]) == set(cand_cycle["stats"]), "stats key drift")
        for key, left in ref_cycle["stats"].items():
            right = cand_cycle["stats"][key]
            if isinstance(left, (bool, int, str)):
                require(left == right, f"exact stat drift: {key}")
            else:
                require(
                    bool(
                        np.isclose(
                            left,
                            right,
                            rtol=stat_rtol,
                            atol=stat_atol,
                        )
                    ),
                    f"numeric stat drift: {key}: {left} != {right}",
                )
    initial = compare_leaf_signatures(
        np,
        reference["numerical"]["initial"],
        candidate["numerical"]["initial"],
        rtol=leaf_rtol,
        atol=leaf_atol,
        label="initial train state",
    )
    require(
        initial["exact_leaf_hashes"] == initial["leaf_count"],
        "initial train state is not byte-exact",
    )
    return {
        "status": "passed",
        "reference_receipt_sha256": reference["receipt_sha256"],
        "tolerances": tolerances,
        "initial": initial,
        "final": compare_leaf_signatures(
            np, reference["numerical"]["final"], candidate["numerical"]["final"],
            rtol=leaf_rtol, atol=leaf_atol, label="final train state",
        ),
    }


def run(cli: argparse.Namespace) -> dict[str, Any]:
    if cli.lane == "reference":
        require(cli.initial_checkpoint is None, "reference lane cannot import initialization")
        require(cli.reference_receipt is None, "reference lane cannot compare to itself")
    if cli.reference_receipt is not None:
        require(cli.lane == "modern", "only the modern lane may run parity")
    require(not os.environ.get("PYTHONPATH"), "PYTHONPATH must be unset")
    require(
        os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() == "false",
        "XLA_PYTHON_CLIENT_PREALLOCATE=false is required",
    )
    required_platform = "cpu" if cli.backend == "cpu" else "cuda"
    require(
        os.environ.get("JAX_PLATFORMS") == required_platform,
        f"JAX_PLATFORMS={required_platform} must be set before import",
    )
    require(
        os.environ.get("JAX_PLATFORM_NAME") == cli.backend,
        f"JAX_PLATFORM_NAME={cli.backend} must be set before import",
    )
    require(
        os.environ.get("JAX_THREEFRY_PARTITIONABLE", "").lower() == "false",
        "JAX_THREEFRY_PARTITIONABLE=false is required for source-era PRNG parity",
    )
    require(sha256(CONFIG) == CONFIG_SHA256, "authored config digest drift")
    require(
        sha256(FRONTIER_CONTRACT) == FRONTIER_CONTRACT_SHA256,
        "Frontier contract digest drift",
    )
    require(sha256(PARITY_PROTOCOL) == PARITY_PROTOCOL_SHA256, "parity protocol drift")
    parity_protocol = json.loads(PARITY_PROTOCOL.read_text())
    require(parity_protocol.get("schema_version") == 1, "parity protocol schema drift")
    require(parity_protocol.get("paper_evidence") is False, "evidence protocol forbidden")
    require(
        parity_protocol["source_era_prng"]["jax_threefry_partitionable"] is False,
        "parity protocol PRNG drift",
    )
    source = cli.source.resolve()
    source_receipt = validate_source(source, cli.lane)
    output = cli.output.resolve()
    require(not output.exists(), "output directory already exists")
    output.mkdir(parents=True, mode=0o700)

    sys.path[:0] = [str(source / "src"), str(REPO_ROOT)]
    import jax
    import numpy as np
    from minimax.util.checkpoint import load_pkl_object, safe_checkpoint
    from ued_benchmark.scripts import run_grouped_one_update as base

    require(jax.default_backend() == cli.backend, "requested backend is not active")
    require(
        jax.config.jax_threefry_partitionable is False,
        "active JAX Threefry partitioning mode drift",
    )
    devices = jax.devices(cli.backend)
    require(len(devices) == 1, "exactly one backend device is required")
    require(devices[0].platform == cli.backend, "active device platform drift")
    minimax_path = Path(sys.modules["minimax"].__file__).resolve()
    require(minimax_path.is_relative_to(source), "minimax imported outside declared source")

    config_document = base.load_frozen_config(CONFIG)
    args = base.configure_engineering_run(config_document, local_test_mode=True)
    runner_args = args.train_runner_args
    require(runner_args.n_parallel == 4, "n_parallel drift")
    require(runner_args.n_eval == 8, "n_eval drift")
    require(runner_args.frontier_n_rollouts == 8, "Frontier N drift")
    require(runner_args.frontier_require_n_eval_match is True, "strict grouping disabled")
    require(runner_args.n_rollout_steps == 2, "bounded rollout horizon drift")
    require(args.student_model_args.recurrent_arch == "lstm", "recurrent architecture drift")
    active_protocol_schedule = {
        "n_parallel": runner_args.n_parallel,
        "n_eval": runner_args.n_eval,
        "frontier_n_rollouts": runner_args.frontier_n_rollouts,
        "rollout_steps": runner_args.n_rollout_steps,
        "outer_cycles": 2,
        "student_updates": 1,
        "total_transitions": 128,
        "buffer_size": runner_args.buffer_size,
        "replay_probability": runner_args.replay_prob,
        "student_recurrent_arch": args.student_model_args.recurrent_arch,
        "student_recurrent_hidden_dim": args.student_model_args.recurrent_hidden_dim,
    }
    require(
        active_protocol_schedule == parity_protocol["schedule"],
        "active schedule does not match frozen parity protocol",
    )

    experiment = base._make_experiment(args)
    wall_start = time.monotonic()
    state = experiment.runner.reset(jax.random.PRNGKey(args.seed))
    base._block(state)
    native_initial_signature = leaf_signature(jax, np, state[1].state_dict)
    initial_source = "native_seed"
    initial_source_sha256 = None
    if cli.initial_checkpoint is not None:
        initial_checkpoint_source = cli.initial_checkpoint.resolve()
        require(
            initial_checkpoint_source.is_file()
            and not initial_checkpoint_source.is_symlink(),
            "unsafe reference initial checkpoint",
        )
        loaded_initial = load_pkl_object(str(initial_checkpoint_source))
        state = experiment.runner.load_checkpoint_state(state, loaded_initial)
        base._block(state)
        initial_source = "cross_version_reference_checkpoint"
        initial_source_sha256 = sha256(initial_checkpoint_source)
    initial_state = base._state_summary(state)
    require(initial_state["n_iters"] == 0, "initial checkpoint iteration drift")
    require(initial_state["n_updates"] == 0, "initial checkpoint update drift")
    require(initial_state["n_grad_updates"] == 0, "initial gradient counter drift")
    require(initial_state["buffer_filled_count"] == 0, "initial buffer is not empty")
    require(initial_state["frontier_total_trials"] == 0, "initial posterior is not empty")
    initial_signature = leaf_signature(jax, np, state[1].state_dict)
    initial_checkpoint_state = experiment.runner.get_checkpoint_state(state)
    initial_checkpoint_path = output / "initial-checkpoint.pkl"
    safe_checkpoint(initial_checkpoint_state, str(output), "initial-checkpoint")
    require(initial_checkpoint_path.is_file(), "initial checkpoint missing")

    cycles = []
    for cycle in (1, 2):
        started = time.monotonic()
        stats, eval_stats, *state = experiment.step(state, False)
        require(not eval_stats, "OOD or periodic evaluation unexpectedly executed")
        base._block((stats, state))
        summary = base._state_summary(state)
        base._assert_cycle(summary, cycle=cycle)
        selected_stats = scalar_stats(np, stats)
        if cycle == 2:
            for key in (
                "actor_loss",
                "entropy",
                "grad_norm",
                "mean_gae",
                "mean_target",
                "mean_value",
                "total_loss",
                "value_loss",
            ):
                require(
                    isinstance(selected_stats.get(key), float),
                    f"cycle-2 optimizer stat is not finite: {key}",
                )
        cycles.append(
            {
                "cycle": cycle,
                "elapsed_seconds": time.monotonic() - started,
                "state": summary,
                "stats": selected_stats,
            }
        )

    final_state = base._state_summary(state)
    require(final_state["n_updates"] == 1, "exactly one PPO update was not executed")
    final_signature = leaf_signature(jax, np, state[1].state_dict)
    checkpoint_state = experiment.runner.get_checkpoint_state(state)
    checkpoint_signature = leaf_signature(jax, np, checkpoint_state)
    checkpoint_path = output / "checkpoint.pkl"
    safe_checkpoint(checkpoint_state, str(output), "checkpoint")
    reloaded = load_pkl_object(str(checkpoint_path))
    serialized_leaf_count = base._assert_same_pytree(checkpoint_state, reloaded, "pickle")
    resumed_experiment = base._make_experiment(args)
    fresh_state = resumed_experiment.runner.reset(jax.random.PRNGKey(args.seed))
    resumed = resumed_experiment.runner.load_checkpoint_state(fresh_state, reloaded)
    base._block(resumed)
    require(base._state_summary(resumed) == final_state, "checkpoint counter continuity drift")
    resumed_leaf_count = base._assert_same_pytree(
        state[1].state_dict, resumed[1].state_dict, "resumed train state"
    )

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "bounded Blackwell training compatibility and parity probe",
        "paper_evidence": False,
        "ood_evaluation": False,
        "max_student_updates": 1,
        "actual_student_updates": 1,
        "status": "passed",
        "lane": cli.lane,
        "backend": cli.backend,
        "seed": args.seed,
        "runtime": {
            "python": platform.python_version(),
            "jax": importlib.metadata.version("jax"),
            "jaxlib": importlib.metadata.version("jaxlib"),
            "flax": importlib.metadata.version("flax"),
            "optax": importlib.metadata.version("optax"),
            "tensorflow_probability": importlib.metadata.version("tensorflow-probability"),
            "device_kind": devices[0].device_kind,
            "device_platform": devices[0].platform,
            "minimax_module": str(minimax_path),
            "jax_platforms_env": os.environ["JAX_PLATFORMS"],
            "jax_platform_name_env": os.environ["JAX_PLATFORM_NAME"],
            "jax_threefry_partitionable_env": os.environ[
                "JAX_THREEFRY_PARTITIONABLE"
            ],
            "jax_threefry_partitionable_active": bool(
                jax.config.jax_threefry_partitionable
            ),
        },
        "source": source_receipt,
        "schedule": {
            "outer_cycles": 2,
            "n_parallel": 4,
            "n_eval": 8,
            "frontier_n_rollouts": 8,
            "rollout_steps": 2,
            "total_transitions": 128,
            "buffer_size": 8,
            "minimum_fill_ratio": 0.5,
            "replay_probability": 1.0,
            "student_recurrent_arch": "lstm",
            "student_recurrent_hidden_dim": 16,
            "ppo_epochs": 1,
            "ppo_unroll_update": 1,
        },
        "cycles": cycles,
        "final_state": final_state,
        "numerical": {
            "native_initial": native_initial_signature,
            "initial": initial_signature,
            "final": final_signature,
        },
        "initial_checkpoint": {
            "sha256": sha256(initial_checkpoint_path),
            "source": initial_source,
            "source_sha256": initial_source_sha256,
            "structure_sha256": leaf_signature(
                jax, np, initial_checkpoint_state
            )["structure_sha256"],
            "state": initial_state,
        },
        "checkpoint": {
            "sha256": sha256(checkpoint_path),
            "serialized_leaf_count": serialized_leaf_count,
            "resumed_leaf_count": resumed_leaf_count,
            "structure_sha256": checkpoint_signature["structure_sha256"],
            "exact_pickle_round_trip": True,
            "exact_resume_round_trip": True,
            "post_resume_update_executed": False,
        },
        "wall_seconds": time.monotonic() - wall_start,
        "hashes": {
            "script_sha256": sha256(Path(__file__).resolve()),
            "config_sha256": CONFIG_SHA256,
            "frontier_contract_sha256": FRONTIER_CONTRACT_SHA256,
            "parity_protocol_sha256": PARITY_PROTOCOL_SHA256,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_tree": UPSTREAM_TREE,
        },
    }
    receipt["parity"] = {"status": "reference"}
    if cli.reference_receipt is not None:
        reference_path = cli.reference_receipt.resolve()
        reference = json.loads(reference_path.read_text())
        reference["receipt_sha256"] = sha256(reference_path)
        receipt["parity"] = compare_reference(
            np, reference, receipt, cli.backend, parity_protocol
        )
    receipt_path = output / "receipt.json"
    base._atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", choices=("reference", "modern"), required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--reference-receipt", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    cli = parser.parse_args()
    try:
        receipt = run(cli)
    except (GateError, AssertionError, ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"BLACKWELL_PARITY_FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "BLACKWELL_PARITY_PASS "
        f"lane={receipt['lane']} backend={receipt['backend']} "
        f"updates={receipt['actual_student_updates']} "
        f"trials={receipt['final_state']['frontier_total_trials']} "
        f"parity={receipt['parity']['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
