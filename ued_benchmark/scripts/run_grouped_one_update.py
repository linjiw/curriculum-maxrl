#!/usr/bin/env python3
"""Run the bounded, non-evidence Frontier 4x8 one-update validation.

This is deliberately not the minimax training entry point.  It executes two
outer PLR cycles: one new-level warmup cycle and one forced-replay cycle that
must perform exactly one PPO update.  The authored benchmark configuration is
content-addressed before the engineering-only buffer/sampling overrides are
applied.  A checkpoint is saved atomically and loaded into a fresh runner, but
no post-resume optimization step is taken.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Mapping, Sequence

import jax
import numpy as np

from minimax.arguments import parser as minimax_parser
from minimax.runners import ExperimentRunner
from minimax.util.checkpoint import load_pkl_object, safe_checkpoint


AUTHORED_CONFIG_SHA256 = (
    "b49168142a9d5a5d8edce88634975ac52d7615dca8bad9fff1cbcaf29ec43508"
)
CONTRACT_SHA256 = (
    "5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000"
)
UPSTREAM_COMMIT = "d053054c5290a04c1c4cd8b55704d999cad73e30"
OVERLAY_VERSION = "frontier-activity-v3"
ENGINEERING_SEED = 1
ENGINEERING_XPID = (
    "eng1-ca-ovv3ch5868d346_N8ne8a1.0b1.0th0.0eastrict-4p-b8-rp1-mf0.5-seed1"
)

_SHA256_FIELDS = (
    "bundle_manifest_sha256",
    "upstream_git_bundle_sha256",
    "overlay_manifest_sha256",
    "applied_overlay_manifest_sha256",
    "sbatch_sha256",
    "config_sha256",
    "overlay_contract_sha256",
    "environment_lock_sha256",
    "environment_freeze_sha256",
    "environment_manifest_sha256",
    "environment_setup_script_sha256",
    "conda_explicit_sha256",
    "environment_json_sha256",
    "import_smoke_manifest_sha256",
    "import_smoke_bundle_manifest_sha256",
    "import_smoke_sbatch_sha256",
)

_EXPECTED_RESOURCES = {
    "partition": "gpuq",
    "qos": "gpu",
    "gres": "gpu:1g.10gb:1",
    "cpus_per_task": 2,
    "memory": "15G",
    "walltime": "00:30:00",
}

_AUTHORED_SEMANTICS = {
    "seed": 1,
    "train_runner": "plr",
    "n_devices": 1,
    "n_parallel": 4,
    "n_eval": 8,
    "n_rollout_steps": 256,
    "ued_score": "coefficient_activity",
    "plr_buffer_size": 500,
    "plr_replay_prob": 0.5,
    "plr_min_fill_ratio": 0.5,
    "plr_use_robust_plr": True,
    "plr_force_unique": True,
    "plr_frontier_n_rollouts": 8,
    "plr_frontier_require_n_eval_match": True,
    "plr_frontier_posterior_mode": "expected_activity",
    "plr_frontier_overlay_version": OVERLAY_VERSION,
    "plr_frontier_overlay_contract_sha256": CONTRACT_SHA256,
    "student_recurrent_arch": "lstm",
    "student_recurrent_hidden_dim": 256,
    "student_ppo_n_epochs": 5,
    "student_ppo_n_minibatches": 1,
    "maze_max_episode_steps": 250,
}


class ValidationError(RuntimeError):
    """Raised when an input or runtime invariant fails closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _single(config_args: Mapping[str, Any], key: str) -> Any:
    _require(key in config_args, f"authored config is missing {key!r}")
    value = config_args[key]
    _require(
        isinstance(value, list) and len(value) == 1,
        f"authored config field {key!r} must be a one-element list",
    )
    return value[0]


def load_frozen_config(path: Path) -> dict[str, Any]:
    """Load and semantically validate the exact authored Frontier config."""
    _require(path.is_file() and not path.is_symlink(), "unsafe config path")
    actual_sha = _sha256(path)
    _require(
        actual_sha == AUTHORED_CONFIG_SHA256,
        f"authored config SHA-256 mismatch: {actual_sha}",
    )
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    _require(set(document) == {"args"}, "unexpected authored config structure")
    config_args = document["args"]
    _require(isinstance(config_args, dict), "authored config args must be an object")
    for key, expected in _AUTHORED_SEMANTICS.items():
        actual = _single(config_args, key)
        _require(
            type(actual) is type(expected) and actual == expected,
            f"authored config mismatch for {key}: {actual!r} != {expected!r}",
        )
    return document


def load_provenance(path: Path, *, local_test_mode: bool) -> dict[str, Any]:
    """Validate the content-address and bounded-job declaration."""
    _require(path.is_file() and not path.is_symlink(), "unsafe provenance path")
    with path.open(encoding="utf-8") as stream:
        provenance = json.load(stream)
    _require(provenance.get("provenance_schema") == 1, "bad provenance schema")
    _require(
        provenance.get("purpose")
        == "bounded Frontier grouped one-update engineering validation",
        "bad provenance purpose",
    )
    _require(provenance.get("paper_evidence") is False, "paper evidence provenance forbidden")
    _require("training_endpoint" not in provenance, "ambiguous training_endpoint field forbidden")
    _require(
        provenance.get("endpoint_class") == "bounded_engineering_one_update",
        "bad endpoint class",
    )
    _require(provenance.get("max_student_updates") == 1, "bad update ceiling")
    _require(provenance.get("git") == "git version 2.45.2", "environment Git drift")
    _require(provenance.get("xpid") == ENGINEERING_XPID, "unexpected engineering xpid")
    _require(provenance.get("resources") == _EXPECTED_RESOURCES, "resource declaration drift")
    hashes = provenance.get("hashes")
    _require(isinstance(hashes, dict), "missing provenance hashes")
    for field in _SHA256_FIELDS:
        value = hashes.get(field)
        _require(
            isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value),
            f"invalid SHA-256 provenance field: {field}",
        )
    _require(hashes["config_sha256"] == AUTHORED_CONFIG_SHA256, "wrong config hash")
    _require(hashes["overlay_contract_sha256"] == CONTRACT_SHA256, "wrong contract hash")
    _require(
        hashes["import_smoke_bundle_manifest_sha256"]
        == hashes["bundle_manifest_sha256"],
        "import/JIT gate did not validate this exact bundle",
    )
    _require(hashes.get("upstream_commit") == UPSTREAM_COMMIT, "wrong upstream commit")
    tree = hashes.get("upstream_tree_git_sha1")
    _require(
        isinstance(tree, str)
        and len(tree) == 40
        and all(c in "0123456789abcdef" for c in tree),
        "invalid upstream tree",
    )
    job_id = provenance.get("job_id")
    if local_test_mode:
        _require(job_id == "local-test", "local test must use the local-test job id")
    else:
        _require(
            isinstance(job_id, str) and job_id.isdigit(),
            "production provenance requires a numeric Slurm job id",
        )
        _require(os.environ.get("SLURM_JOB_ID") == job_id, "Slurm job id mismatch")
    return provenance


def _parse_authored_args(config_document: Mapping[str, Any]) -> Any:
    argv = ["run-grouped-one-update"]
    for key, values in config_document["args"].items():
        value = values[0]
        argv.append(f"--{key}={value}")
    previous_argv = sys.argv
    try:
        sys.argv = argv
        return minimax_parser.parse_args()
    finally:
        sys.argv = previous_argv


def configure_engineering_run(config_document: Mapping[str, Any], local_test_mode: bool) -> Any:
    """Parse the frozen config and apply the declared non-evidence schedule."""
    args = _parse_authored_args(config_document)
    _require(args.seed == ENGINEERING_SEED, "parsed seed drift")
    args.xpid = ENGINEERING_XPID
    args.n_total_updates = 1
    args.test_interval = 0
    args.checkpoint_interval = 0
    args.archive_interval = 0
    args.archive_init_checkpoint = False
    args.from_last_checkpoint = False
    args.eval_args.env_names = None

    # Four new levels fill exactly half of this engineering-only buffer on the
    # first cycle.  Forced replay then makes the second cycle the sole update.
    args.train_runner_args.buffer_size = 8
    args.train_runner_args.min_fill_ratio = 0.5
    args.train_runner_args.replay_prob = 1.0

    if local_test_mode:
        # Preserve the exact 4x8/N=8 grouped estimator and two-cycle control
        # flow while shrinking only horizon, model, and PPO work for CPU CI.
        args.train_runner_args.n_rollout_steps = 2
        args.train_runner_args.n_unroll_rollout = 1
        args.env_args.max_episode_steps = 2
        args.student_rl_args.n_unroll_update = 1
        args.student_rl_args.n_epochs = 1
        args.student_model_args.hidden_dim = 16
        args.student_model_args.recurrent_hidden_dim = 16
        args.student_model_args.n_conv_filters = 4

    return args


def _make_experiment(args: Any) -> ExperimentRunner:
    p = copy.deepcopy(args)
    return ExperimentRunner(
        train_runner=p.train_runner,
        env_name=p.env_name,
        agent_rl_algo=p.agent_rl_algo,
        student_model_name=p.student_model_name,
        teacher_model_name=p.teacher_model_name,
        train_runner_kwargs=p.train_runner_args,
        env_kwargs=p.env_args,
        ued_env_kwargs=p.ued_env_args,
        student_rl_kwargs=p.student_rl_args,
        teacher_rl_kwargs=p.teacher_rl_args,
        student_model_kwargs=p.student_model_args,
        teacher_model_kwargs=p.teacher_model_args,
        eval_kwargs=p.eval_args,
        eval_env_kwargs=p.eval_env_args,
        n_devices=p.n_devices,
    )


def _block(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready")
        else value,
        tree,
    )


def _scalar(value: Any) -> int:
    array = np.asarray(value).reshape(-1)
    _require(array.size == 1, "expected a scalar array")
    return int(array[0])


def _optimizer_step_applications(train_state: Any) -> int:
    """Read the sole Adam step counter from the pinned Optax state."""
    counts = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(train_state.opt_state)[0]:
        if path and getattr(path[-1], "name", None) == "count":
            counts.append(_scalar(leaf))
    _require(
        len(counts) == 1,
        f"expected one optimizer step counter in pinned Optax state; got {counts}",
    )
    return counts[0]


def _state_summary(state: Sequence[Any]) -> dict[str, Any]:
    train_state = state[1]
    buffer = train_state.plr_buffer
    return {
        "n_iters": _scalar(train_state.n_iters),
        "n_updates": _scalar(train_state.n_updates),
        "n_grad_updates": _scalar(train_state.n_grad_updates),
        "optimizer_step_applications": _optimizer_step_applications(train_state),
        "buffer_filled_count": _scalar(buffer.filled_count),
        "frontier_total_trials": int(np.asarray(buffer.trial_counts).sum()),
        "frontier_total_successes": int(np.asarray(buffer.success_counts).sum()),
        "frontier_incomplete_group_count": _scalar(buffer.incomplete_group_count),
        "frontier_duplicate_new_group_count": _scalar(buffer.duplicate_new_group_count),
        "frontier_n_rollouts": int(buffer.frontier_n_rollouts),
        "frontier_n_eval": int(buffer.frontier_n_eval),
        "frontier_group_size_match": bool(
            buffer.frontier_n_rollouts == buffer.frontier_n_eval
        ),
    }


def _assert_cycle(
    summary: Mapping[str, Any],
    *,
    cycle: int,
    expected_optimizer_step_applications: int,
) -> None:
    expected = {
        1: {
            "n_iters": 1,
            "n_updates": 0,
            "n_grad_updates": 0,
            "optimizer_step_applications": 0,
            "buffer_filled_count": 4,
            "frontier_total_trials": 32,
        },
        2: {
            "n_iters": 2,
            "n_updates": 1,
            "n_grad_updates": 1,
            "optimizer_step_applications": expected_optimizer_step_applications,
            "buffer_filled_count": 4,
            "frontier_total_trials": 64,
        },
    }[cycle]
    for key, value in expected.items():
        _require(summary[key] == value, f"cycle {cycle} invariant failed: {key}")
    _require(summary["frontier_n_rollouts"] == 8, "Frontier N drift")
    _require(summary["frontier_n_eval"] == 8, "n_eval drift")
    _require(summary["frontier_group_size_match"] is True, "N != n_eval")
    _require(summary["frontier_incomplete_group_count"] == 0, "incomplete group")
    _require(summary["frontier_duplicate_new_group_count"] == 0, "duplicate new group")
    _require(
        0 <= summary["frontier_total_successes"] <= summary["frontier_total_trials"],
        "invalid posterior counts",
    )


def _assert_same_pytree(left: Any, right: Any, label: str) -> int:
    left_def = jax.tree_util.tree_structure(left)
    right_def = jax.tree_util.tree_structure(right)
    _require(left_def == right_def, f"{label} pytree structure changed")
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    _require(len(left_leaves) == len(right_leaves), f"{label} leaf count changed")
    for index, (before, after) in enumerate(zip(left_leaves, right_leaves)):
        _require(
            np.array_equal(np.asarray(before), np.asarray(after)),
            f"{label} leaf {index} changed across checkpoint round-trip",
        )
    return len(left_leaves)


def _json_scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.size != 1:
        return None
    scalar = array.reshape(-1)[0]
    if np.issubdtype(array.dtype, np.bool_):
        return bool(scalar)
    if np.issubdtype(array.dtype, np.integer):
        return int(scalar)
    if np.issubdtype(array.dtype, np.floating):
        number = float(scalar)
        return number if np.isfinite(number) else None
    return None


def _selected_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "n_updates",
        "plr/frontier_n_rollouts",
        "plr/frontier_n_eval",
        "plr/frontier_group_size_match",
        "plr/frontier_total_trials",
        "plr/frontier_total_successes",
        "plr/frontier_incomplete_group_count",
        "plr/frontier_duplicate_new_group_count",
    )
    return {key: _json_scalar(stats[key]) for key in keep if key in stats}


def _atomic_json(path: Path, record: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _utc_now() -> str:
    """Return a timezone-explicit timestamp from Python's runtime clock."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def run(args_cli: argparse.Namespace) -> dict[str, Any]:
    local_test_mode = bool(args_cli.local_test_mode)
    if local_test_mode:
        _require("SLURM_JOB_ID" not in os.environ, "local-test mode forbidden under Slurm")
    else:
        _require("SLURM_JOB_ID" in os.environ, "production run requires Slurm")

    output_dir = args_cli.output_dir.resolve()
    _require(output_dir.is_dir() and not output_dir.is_symlink(), "unsafe output directory")
    config_document = load_frozen_config(args_cli.config.resolve())
    _require(
        args_cli.contract.is_file() and not args_cli.contract.is_symlink(),
        "unsafe overlay contract path",
    )
    _require(_sha256(args_cli.contract) == CONTRACT_SHA256, "overlay contract drift")
    provenance = load_provenance(
        args_cli.provenance.resolve(), local_test_mode=local_test_mode
    )

    parsed = configure_engineering_run(config_document, local_test_mode)
    _require(parsed.xpid == ENGINEERING_XPID, "xpid drift")
    runner_args = parsed.train_runner_args
    _require(runner_args.n_parallel == 4, "n_parallel drift")
    _require(runner_args.n_eval == 8, "n_eval drift")
    _require(runner_args.frontier_n_rollouts == 8, "Frontier N drift")
    _require(runner_args.frontier_require_n_eval_match is True, "strict mode disabled")
    _require(runner_args.buffer_size == 8, "engineering buffer drift")
    _require(runner_args.replay_prob == 1.0, "forced replay drift")
    _require(runner_args.min_fill_ratio == 0.5, "warmup ratio drift")
    _require(runner_args.use_robust_plr is True, "robust PLR drift")
    expected_ppo_epochs = 1 if local_test_mode else 5
    _require(
        parsed.student_rl_args.n_epochs == expected_ppo_epochs,
        "PPO epoch count drift",
    )
    _require(parsed.student_rl_args.n_minibatches == 1, "PPO minibatch count drift")
    expected_optimizer_step_applications = (
        parsed.student_rl_args.n_epochs * parsed.student_rl_args.n_minibatches
    )

    import minimax  # Imported only after input validation.

    patched_source = args_cli.patched_source_dir.resolve()
    minimax_module = Path(minimax.__file__).resolve()
    _require(
        minimax_module.is_relative_to(patched_source),
        f"minimax import is outside patched source: {minimax_module}",
    )
    if local_test_mode:
        _require(jax.default_backend() == "cpu", "local test must use the CPU backend")
        devices = jax.devices("cpu")
    else:
        devices = jax.devices("gpu")
        _require(jax.default_backend() == "gpu", "production backend is not GPU")
        _require(len(devices) == 1, "production run requires exactly one GPU device")

    transitions_per_cycle = (
        runner_args.n_parallel * runner_args.n_eval * runner_args.n_rollout_steps
    )
    expected_transitions = 2 * transitions_per_cycle
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    run_start_utc = _utc_now()
    monotonic_start_ns = time.monotonic_ns()
    experiment = _make_experiment(parsed)
    state = experiment.runner.reset(jax.random.PRNGKey(parsed.seed))

    cycle_records = []
    for cycle in (1, 2):
        cycle_start = time.monotonic()
        stats, eval_stats, *state = experiment.step(state, False)
        _require(not eval_stats, "evaluation unexpectedly executed")
        _block((stats, state))
        summary = _state_summary(state)
        _assert_cycle(
            summary,
            cycle=cycle,
            expected_optimizer_step_applications=(
                expected_optimizer_step_applications if cycle == 2 else 0
            ),
        )
        cycle_records.append(
            {
                "cycle": cycle,
                "role": "new_level_warmup" if cycle == 1 else "forced_replay_update",
                "elapsed_seconds": time.monotonic() - cycle_start,
                "transitions": transitions_per_cycle,
                "state": summary,
                "selected_stats": _selected_stats(stats),
            }
        )

    final_summary = _state_summary(state)
    _require(final_summary["n_updates"] == 1, "run did not end at one PPO update")
    _require(final_summary["n_grad_updates"] == 1, "gradient-update counter drift")
    _require(
        final_summary["optimizer_step_applications"]
        == expected_optimizer_step_applications,
        "optimizer step-application count drift",
    )

    checkpoint_state = experiment.runner.get_checkpoint_state(state)
    checkpoint_path = output_dir / "checkpoint.pkl"
    _require(not checkpoint_path.exists(), "checkpoint output already exists")
    safe_checkpoint(checkpoint_state, str(output_dir), "checkpoint")
    _require(checkpoint_path.is_file() and not checkpoint_path.is_symlink(), "checkpoint missing")
    checkpoint_sha256 = _sha256(checkpoint_path)
    reloaded_checkpoint = load_pkl_object(str(checkpoint_path))
    serialized_leaf_count = _assert_same_pytree(
        checkpoint_state, reloaded_checkpoint, "serialized runner state"
    )

    # Exercise the overlay's fail-closed static PLR signature checks on a
    # freshly constructed runner.  No optimization step follows this reload.
    resumed_experiment = _make_experiment(parsed)
    fresh_state = resumed_experiment.runner.reset(jax.random.PRNGKey(parsed.seed))
    resumed_state = resumed_experiment.runner.load_checkpoint_state(
        fresh_state, reloaded_checkpoint
    )
    _block(resumed_state[1])
    resumed_summary = _state_summary(resumed_state)
    _require(resumed_summary == final_summary, "resume counter/buffer continuity failed")
    train_state_leaf_count = _assert_same_pytree(
        state[1].state_dict, resumed_state[1].state_dict, "resumed train state"
    )

    monotonic_end_ns = time.monotonic_ns()
    run_end_utc = _utc_now()
    monotonic_elapsed_seconds = (
        monotonic_end_ns - monotonic_start_ns
    ) / 1_000_000_000.0
    _require(monotonic_elapsed_seconds > 0.0, "non-positive monotonic runtime")
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    record = {
        "result_schema": 1,
        "purpose": "bounded Frontier grouped one-update engineering validation",
        "paper_evidence": False,
        "endpoint_class": "bounded_engineering_one_update",
        "max_student_updates": 1,
        "actual_student_updates": 1,
        "ood_evaluation": False,
        "status": "passed",
        "job_id": provenance["job_id"],
        "xpid": parsed.xpid,
        "seed": parsed.seed,
        "backend": jax.default_backend(),
        "devices": [
            {
                "id": int(device.id),
                "platform": device.platform,
                "device_kind": device.device_kind,
            }
            for device in devices
        ],
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "jax": metadata.version("jax"),
            "jaxlib": metadata.version("jaxlib"),
            "git": provenance["git"],
            "minimax_module": str(minimax_module),
            "local_test_mode": local_test_mode,
        },
        "authored_config": {
            "sha256": AUTHORED_CONFIG_SHA256,
            "n_parallel": 4,
            "n_eval": 8,
            "frontier_n_rollouts": 8,
            "buffer_size": 500,
            "replay_probability": 0.5,
            "rollout_steps": 256,
            "maze_max_episode_steps": 250,
            "student_recurrent_arch": "lstm",
            "student_recurrent_hidden_dim": 256,
        },
        "engineering_schedule": {
            "outer_cycles": 2,
            "actual_ppo_updates": 1,
            "ppo_epochs": parsed.student_rl_args.n_epochs,
            "ppo_minibatches": parsed.student_rl_args.n_minibatches,
            "expected_optimizer_step_applications": (
                expected_optimizer_step_applications
            ),
            "optimizer_step_applications": final_summary[
                "optimizer_step_applications"
            ],
            "buffer_size": 8,
            "minimum_fill_ratio": 0.5,
            "replay_probability": 1.0,
            "n_parallel": 4,
            "n_eval": 8,
            "frontier_n_rollouts": 8,
            "rollout_steps": runner_args.n_rollout_steps,
            "maze_max_episode_steps": parsed.env_args.max_episode_steps,
            "transitions_per_cycle": transitions_per_cycle,
            "total_transitions": expected_transitions,
            "local_test_shrink": local_test_mode,
        },
        "cycles": cycle_records,
        "final_state": final_summary,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": checkpoint_sha256,
            "serialized_runner_state_leaf_count": serialized_leaf_count,
            "resumed_train_state_leaf_count": train_state_leaf_count,
            "fresh_runner_static_signature_validated": True,
            "counter_and_buffer_continuity": True,
            "train_state_exact_leaf_continuity": True,
            "post_resume_update_executed": False,
        },
        "resource_accounting": {
            "resource_schema": 2,
            "scope": "in-process engineering diagnostics",
            "accounting_source": "python_resource_getrusage_self_and_monotonic_ns",
            "run_start_utc": run_start_utc,
            "run_end_utc": run_end_utc,
            "monotonic_elapsed_seconds": monotonic_elapsed_seconds,
            "process_user_seconds": float(usage_end.ru_utime - usage_start.ru_utime),
            "process_system_seconds": float(usage_end.ru_stime - usage_start.ru_stime),
            "process_max_rss_kib": int(usage_end.ru_maxrss),
            "process_minor_page_faults": int(
                usage_end.ru_minflt - usage_start.ru_minflt
            ),
            "process_major_page_faults": int(
                usage_end.ru_majflt - usage_start.ru_majflt
            ),
            "process_block_input_operations": int(
                usage_end.ru_inblock - usage_start.ru_inblock
            ),
            "process_block_output_operations": int(
                usage_end.ru_oublock - usage_start.ru_oublock
            ),
            "process_voluntary_context_switches": int(
                usage_end.ru_nvcsw - usage_start.ru_nvcsw
            ),
            "process_involuntary_context_switches": int(
                usage_end.ru_nivcsw - usage_start.ru_nivcsw
            ),
            "transitions": expected_transitions,
            "transitions_per_wall_second": (
                expected_transitions / monotonic_elapsed_seconds
            ),
            "slurm_request": provenance["resources"],
            "external_accounting_authority": "terminal_slurm_sacct",
            "terminal_sacct_included": False,
        },
        "hashes": provenance["hashes"],
    }
    result_path = output_dir / "run-result.json"
    _require(not result_path.exists(), "run result already exists")
    _atomic_json(result_path, record)
    return record


def _parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--contract", type=Path, required=True)
    cli.add_argument("--provenance", type=Path, required=True)
    cli.add_argument("--patched-source-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--local-test-mode", action="store_true")
    return cli.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        record = run(_parse_cli(argv))
    except (ValidationError, AssertionError, ValueError, KeyError) as error:
        print(f"GROUPED_ONE_UPDATE_FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "GROUPED_ONE_UPDATE_PASS "
        f"updates={record['final_state']['n_updates']} "
        f"trials={record['final_state']['frontier_total_trials']} "
        f"checkpoint={record['checkpoint']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
