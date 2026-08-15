#!/usr/bin/env python3
"""Capture Frontier PPO components without applying an optimizer update."""

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
import tempfile
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = Path(__file__).resolve().with_name("COMPONENT_PARITY_PROTOCOL.json")
PROTOCOL_SHA256 = "0f8c083202a189ec234f32c0e1c15e7c09753892fb05af0d6262b9ff0bf9f1a5"
CONFIG = REPO_ROOT / "ued_benchmark/configs/maze_frontier_exact_grouped_n8.json"
FRONTIER_CONTRACT = REPO_ROOT / "ued_benchmark/OVERLAY_CONTRACT.json"
MODERN_CONTRACT = REPO_ROOT / "ued_benchmark/blackwell_training_probe/MODERNIZATION_CONTRACT.json"
RUN_ROOT = Path("/data/robotixx/ued_bench/runs")
FROZEN_INITIAL_CHECKPOINT = Path(
    "/data/robotixx/ued_bench/runs/blackwell_training_probe_b7c865/"
    "reference-jax0431-cpu-protocol-v6/initial-checkpoint.pkl"
)
UPSTREAM_COMMIT = "d053054c5290a04c1c4cd8b55704d999cad73e30"
UPSTREAM_TREE = "b0cace1fc54984e21a842f12d15d0b899e33d270"
FRONTIER_CONTRACT_SHA256 = "5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000"
PARENT_MANIFEST_SHA256 = "d929efa2f059a93125e217ec4713ae81670c769d979c67abd2b10efc64268af3"
MODERN_CONTRACT_SHA256 = "b7c865e007634c5a20e2b942ff98f24d6ac9ff624d5b17b62e5e9fa2124e5c00"
MODERN_MANIFEST_SHA256 = "ea5fb73c0072cd95829630344e559f02a83f65b0f8b479845ef4dff8921ff65c"
INITIAL_CHECKPOINT_SHA256 = "4dd07bf02eeb7ec072e4ec72b3aa02180c3ae84284ba20b27174f3dfa9886187"
CONFIG_SHA256 = "b49168142a9d5a5d8edce88634975ac52d7615dca8bad9fff1cbcaf29ec43508"
PRESERVED_GPU_PID = 2786996


class CaptureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CaptureError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(source: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_environment(backend: str) -> None:
    require(not os.environ.get("PYTHONPATH"), "PYTHONPATH must be unset")
    required = {
        "JAX_PLATFORMS": "cpu" if backend == "cpu" else "cuda",
        "JAX_PLATFORM_NAME": backend,
        "JAX_THREEFRY_PARTITIONABLE": "false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    for name, expected in required.items():
        require(os.environ.get(name, "").lower() == expected, f"{name} drift")
    require(
        os.environ.get("TMPDIR") == "/data/robotixx/ued_bench/tmp",
        "TMPDIR must remain on /data",
    )
    require(
        os.environ.get("PIP_CACHE_DIR") == "/data/robotixx/ued_bench/cache/pip",
        "PIP_CACHE_DIR must remain on /data",
    )


def validate_source(source: Path, lane: str) -> dict[str, Any]:
    require(source.is_dir() and not source.is_symlink(), "unsafe source directory")
    require(git(source, "rev-parse", "HEAD") == UPSTREAM_COMMIT, "source commit drift")
    require(git(source, "rev-parse", "HEAD^{tree}") == UPSTREAM_TREE, "source tree drift")
    parent_manifest = source / ".frontierrl_overlay.json"
    require(parent_manifest.is_file() and not parent_manifest.is_symlink(), "bad parent manifest")
    require(sha256(parent_manifest) == PARENT_MANIFEST_SHA256, "parent manifest drift")
    require(sha256(FRONTIER_CONTRACT) == FRONTIER_CONTRACT_SHA256, "Frontier contract drift")
    require(sha256(MODERN_CONTRACT) == MODERN_CONTRACT_SHA256, "modern contract drift")
    removed_count = sum(
        path.read_text().count("jax.tree_map")
        for path in (source / "src/minimax").rglob("*.py")
    )
    modern_manifest = source / ".blackwell_training_overlay.json"
    if lane == "reference":
        require(removed_count == 35, "reference removed-API count drift")
        require(not modern_manifest.exists(), "reference source unexpectedly modernized")
        modern_sha = None
    else:
        require(removed_count == 0, "modern source retains removed API")
        require(modern_manifest.is_file() and not modern_manifest.is_symlink(), "bad modern manifest")
        require(sha256(modern_manifest) == MODERN_MANIFEST_SHA256, "modern manifest drift")
        modern_sha = sha256(modern_manifest)
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tree": UPSTREAM_TREE,
        "frontier_manifest_sha256": sha256(parent_manifest),
        "modernization_manifest_sha256": modern_sha,
        "removed_api_occurrences": removed_count,
    }


def tree_digest(jax: Any, np: Any, tree: Any) -> str:
    digest = hashlib.sha256()
    flattened, _ = jax.tree_util.tree_flatten_with_path(tree)
    for path, leaf in flattened:
        array = np.ascontiguousarray(np.asarray(leaf))
        descriptor = json.dumps(
            {
                "path": jax.tree_util.keystr(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(array.tobytes())
    return digest.hexdigest()


class ArrayBundle:
    def __init__(self, np: Any, jax: Any, output: Path):
        self.np = np
        self.jax = jax
        self.directory = output / "arrays"
        self.directory.mkdir(mode=0o700)
        self.records: list[dict[str, Any]] = []
        self.groups: dict[str, list[dict[str, Any]]] = {}

    def add(self, stage: str, label: str, tree: Any) -> None:
        flattened, _ = self.jax.tree_util.tree_flatten_with_path(tree)
        structure = []
        first = len(self.records)
        for path, leaf in flattened:
            array = self.np.ascontiguousarray(self.np.asarray(leaf))
            require(array.dtype.kind != "O", f"object array forbidden: {stage}/{label}")
            index = len(self.records)
            filename = f"{index:05d}.npy"
            target = self.directory / filename
            with target.open("wb") as handle:
                self.np.save(handle, array, allow_pickle=False)
            finite = self.np.isfinite(array)
            finite_abs = self.np.abs(array[finite].astype(self.np.float64, copy=False))
            path_text = self.jax.tree_util.keystr(path)
            record = {
                "index": index,
                "stage": stage,
                "label": label,
                "path": path_text,
                "file": f"arrays/{filename}",
                "file_sha256": sha256(target),
                "raw_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "size": int(array.size),
                "nan_count": int(self.np.isnan(array).sum()),
                "infinite_count": int(self.np.isinf(array).sum()),
                "positive_inf_count": int(self.np.isposinf(array).sum()),
                "negative_inf_count": int(self.np.isneginf(array).sum()),
                "abs_sum": float(finite_abs.sum(dtype=self.np.float64)),
                "squared_l2": float(
                    self.np.square(finite_abs).sum(dtype=self.np.float64)
                ),
                "max_abs": float(finite_abs.max(initial=0.0)),
            }
            self.records.append(record)
            structure.append(
                {
                    "path": path_text,
                    "shape": record["shape"],
                    "dtype": record["dtype"],
                    "size": record["size"],
                }
            )
        structure_sha = hashlib.sha256(
            json.dumps(structure, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.groups.setdefault(stage, []).append(
            {
                "label": label,
                "first_record": first,
                "record_count": len(flattened),
                "structure_sha256": structure_sha,
            }
        )


def state_summary(np: Any, state: Any) -> dict[str, Any]:
    train_state = state[1]
    buffer = train_state.plr_buffer
    return {
        "n_iters": int(np.asarray(train_state.n_iters).reshape(-1)[0]),
        "n_updates": int(np.asarray(train_state.n_updates).reshape(-1)[0]),
        "n_grad_updates": int(np.asarray(train_state.n_grad_updates).reshape(-1)[0]),
        "buffer_filled_count": int(np.asarray(buffer.filled_count).reshape(-1)[0]),
        "frontier_total_trials": int(np.asarray(buffer.trial_counts).sum()),
        "frontier_total_successes": int(np.asarray(buffer.success_counts).sum()),
        "frontier_incomplete_group_count": int(
            np.asarray(buffer.incomplete_group_count).reshape(-1)[0]
        ),
        "frontier_duplicate_new_group_count": int(
            np.asarray(buffer.duplicate_new_group_count).reshape(-1)[0]
        ),
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    validate_environment(args.backend)
    require(sha256(PROTOCOL) == PROTOCOL_SHA256, "component protocol drift")
    protocol = json.loads(PROTOCOL.read_text())
    require(protocol["execution_limits"]["optimizer_applications"] == 0, "bad protocol")
    require(protocol["execution_limits"]["gpu_ppo_updates"] == 0, "bad GPU budget")
    require(sha256(CONFIG) == CONFIG_SHA256, "configuration drift")
    require(args.initial_checkpoint.resolve() == FROZEN_INITIAL_CHECKPOINT, "input path drift")
    require(sha256(args.initial_checkpoint.resolve()) == INITIAL_CHECKPOINT_SHA256, "input digest drift")
    require(args.lane == "modern" or args.backend == "cpu", "reference lane is CPU only")
    source = args.source.resolve()
    source_record = validate_source(source, args.lane)
    output = args.output.resolve()
    run_root = RUN_ROOT.resolve()
    require(output.is_relative_to(run_root), "output must remain under declared /data run root")
    require(not output.exists(), "output already exists")
    require(output.parent.is_dir() and not output.parent.is_symlink(), "unsafe output parent")
    gpu_pid_present_before = Path(f"/proc/{PRESERVED_GPU_PID}").is_dir()
    if args.backend == "gpu":
        require(gpu_pid_present_before, "preserved GPU process is absent before capture")

    sys.path[:0] = [str(source / "src"), str(REPO_ROOT)]
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from minimax.util.checkpoint import load_pkl_object
    from ued_benchmark.scripts import run_grouped_one_update as base

    require(jax.default_backend() == args.backend, "requested backend is not active")
    devices = jax.devices(args.backend)
    require(len(devices) == 1 and devices[0].platform == args.backend, "device gate failed")
    if args.backend == "gpu":
        require(devices[0].device_kind == "NVIDIA GeForce RTX 5090", "GPU kind drift")
    require(jax.config.jax_threefry_partitionable is False, "Threefry mode drift")
    minimax_path = Path(sys.modules["minimax"].__file__).resolve()
    require(minimax_path.is_relative_to(source), "minimax imported outside source")

    config = base.load_frozen_config(CONFIG)
    parsed = base.configure_engineering_run(config, local_test_mode=True)
    require(parsed.student_rl_args.n_epochs == 1, "PPO epoch drift")
    require(parsed.student_rl_args.n_minibatches == 1, "PPO minibatch drift")
    require(parsed.train_runner_args.n_parallel == 4, "parallel count drift")
    require(parsed.train_runner_args.n_eval == 8, "evaluation count drift")
    require(parsed.train_runner_args.frontier_n_rollouts == 8, "Frontier N drift")
    require(parsed.train_runner_args.n_rollout_steps == 2, "rollout horizon drift")
    require(parsed.train_runner_args.max_grad_norm == 0.5, "clip threshold drift")
    require(parsed.train_runner_args.lr == 0.0003, "learning-rate drift")
    require(parsed.train_runner_args.adam_eps == 1e-5, "Adam epsilon drift")
    experiment = base._make_experiment(parsed)
    runner = experiment.runner
    require(runner.n_devices == 1, "multi-device capture forbidden")
    require(runner.n_students == 1, "student population drift")
    require(not runner.use_parallel_eval, "parallel evaluation forbidden")
    require(not runner.use_mutations, "mutation path forbidden")

    output.mkdir(mode=0o700)
    bundle = ArrayBundle(np, jax, output)
    reset_state = runner.reset(jax.random.PRNGKey(parsed.seed))
    checkpoint = load_pkl_object(str(args.initial_checkpoint.resolve()))
    initial_state = runner.load_checkpoint_state(reset_state, checkpoint)
    base._block(initial_state)
    initial_summary = state_summary(np, initial_state)
    require(initial_summary["n_iters"] == 0, "initial iteration drift")
    require(initial_summary["n_updates"] == 0, "initial update drift")
    require(initial_summary["n_grad_updates"] == 0, "initial gradient-update drift")
    initial_param_digest = tree_digest(jax, np, initial_state[1].params)
    bundle.add("initial_state", "runner_rng", initial_state[0])
    bundle.add("initial_state", "train_state", initial_state[1].state_dict)

    # The only ExperimentRunner.step call is cycle one. Robust PLR must select
    # its fake-update branch, leaving parameters and optimizer counters intact.
    cycle_one_stats, eval_stats, *cycle_one_state = experiment.step(initial_state, False)
    base._block((cycle_one_stats, cycle_one_state))
    require(not eval_stats, "evaluation unexpectedly executed")
    cycle_one_summary = state_summary(np, cycle_one_state)
    require(cycle_one_summary["n_iters"] == 1, "cycle-one iteration drift")
    require(cycle_one_summary["n_updates"] == 0, "cycle one applied an update")
    require(cycle_one_summary["n_grad_updates"] == 0, "cycle one applied gradients")
    require(cycle_one_summary["buffer_filled_count"] == 4, "cycle-one buffer drift")
    require(cycle_one_summary["frontier_total_trials"] == 32, "cycle-one trial drift")
    require(
        tree_digest(jax, np, cycle_one_state[1].params) == initial_param_digest,
        "cycle one mutated parameters",
    )
    bundle.add("cycle_one_control", "runner_rng", cycle_one_state[0])
    bundle.add("cycle_one_control", "train_state", cycle_one_state[1].state_dict)
    bundle.add("cycle_one_control", "environment_state", tuple(cycle_one_state[2:]))
    bundle.add("cycle_one_control", "stats", cycle_one_stats)

    require(len(cycle_one_state) == 9, "unexpected PLR runner-state arity")
    (
        entry_rng,
        train_state,
        _previous_state,
        _previous_start_state,
        _previous_obs,
        _previous_carry,
        _previous_extra,
        _previous_ep_stats,
        _previous_plr_buffer,
    ) = cycle_one_state
    rng_after_reset_split, reset_vrng = jax.random.split(entry_rng, 2)
    obs, environment_state, extra = runner.benv.reset(
        jnp.asarray([reset_vrng]), runner.n_parallel, 1
    )
    new_levels = environment_state
    sample_parent_rng, sample_rng = jax.random.split(rng_after_reset_split)
    levels, level_idxs, is_replay, next_plr_buffer = runner.plr_mgr.sample(
        sample_rng, train_state.plr_buffer, new_levels, runner.n_parallel
    )
    train_state = train_state.replace(plr_buffer=next_plr_buffer)
    parent_idxs = jnp.full((runner.n_students, runner.n_parallel), -1)
    if runner.force_unique:
        level_idxs, dupe_mask = runner.plr_mgr.dedupe_levels(
            next_plr_buffer, levels, level_idxs
        )
    else:
        dupe_mask = None
    base._block((levels, level_idxs, is_replay, train_state))
    require(bool(np.asarray(is_replay).reshape(-1)[0]), "cycle two is not replay")
    bundle.add(
        "task_stream",
        "rng_keys",
        {
            "cycle_two_entry": entry_rng,
            "after_reset_split": rng_after_reset_split,
            "reset": reset_vrng,
            "sample_parent": sample_parent_rng,
            "sample": sample_rng,
        },
    )
    bundle.add("task_stream", "new_levels", new_levels)
    bundle.add("task_stream", "selected_levels", levels)
    bundle.add("task_stream", "level_idxs", level_idxs)
    bundle.add("task_stream", "is_replay", is_replay)

    result = runner._eval_and_update_plr(
        sample_parent_rng,
        levels,
        level_idxs,
        train_state,
        update_plr=jnp.asarray([True]),
        parent_idxs=parent_idxs,
        dupe_mask=dupe_mask,
    )
    base._block(result)
    (
        rollout_return_rng,
        post_rollout_train_state,
        final_environment_state,
        final_start_state,
        final_obs,
        final_carry,
        final_extra,
        final_ep_stats,
        rollout_start_state,
        train_batch,
        ued_scores,
    ) = result
    require(
        int(np.asarray(post_rollout_train_state.n_updates).reshape(-1)[0]) == 0,
        "rollout path changed update counter",
    )
    require(
        int(np.asarray(post_rollout_train_state.n_grad_updates).reshape(-1)[0]) == 0,
        "rollout path changed gradient counter",
    )
    require(
        tree_digest(jax, np, post_rollout_train_state.params) == initial_param_digest,
        "rollout path mutated parameters",
    )
    bundle.add("task_stream", "rollout_start_state", rollout_start_state)
    bundle.add("rollout_observation_stream", "observations", train_batch.obs)
    bundle.add("rollout_observation_stream", "rewards", train_batch.rewards)
    bundle.add("rollout_observation_stream", "dones", train_batch.dones)
    bundle.add("rollout_action_stream", "actions", train_batch.actions)
    bundle.add("rollout_forward_stream", "old_log_probabilities", train_batch.log_pis)
    bundle.add("rollout_forward_stream", "old_values", train_batch.values)
    bundle.add("rollout_return_batch", "targets", train_batch.targets)
    bundle.add("rollout_return_batch", "advantages", train_batch.advantages)
    bundle.add("rollout_return_batch", "carry", train_batch.carry)

    update_parent_rng, population_update_rng = jax.random.split(rollout_return_rng)
    _discarded_pop_rng, agent_update_rng = jax.random.split(population_update_rng, 2)
    epoch_rng = jax.random.split(agent_update_rng, 1)[0]
    permutation_rng, epoch_update_rng = jax.random.split(epoch_rng)
    loss_rng = jax.random.split(epoch_update_rng, 1)[0]
    single_train_state = jax.tree_util.tree_map(lambda value: value[0], post_rollout_train_state)
    single_batch = jax.tree_util.tree_map(lambda value: value[0], train_batch)
    proposal_input_param_digest = tree_digest(jax, np, single_train_state.params)
    agent = runner.student_pop.agent
    permutation = jax.random.permutation(
        permutation_rng, jnp.arange(single_batch.dones.shape[1])
    )
    minibatches = agent._get_minibatches(permutation_rng, single_batch)
    minibatch = jax.tree_util.tree_map(lambda value: value[0], minibatches)
    base._block((permutation, minibatch))
    bundle.add(
        "minibatch_stream",
        "rng_keys",
        {
            "update_parent": update_parent_rng,
            "population_update": population_update_rng,
            "agent_update": agent_update_rng,
            "epoch": epoch_rng,
            "permutation": permutation_rng,
            "epoch_update": epoch_update_rng,
            "loss": loss_rng,
        },
    )
    bundle.add("minibatch_stream", "permutation", permutation)
    bundle.add("minibatch_stream", "minibatch", minibatch)

    recurrent_carry = jax.tree_util.tree_map(lambda value: value[0, :], minibatch.carry)
    shifted_dones = minibatch.dones.at[1:, :].set(minibatch.dones[:-1, :])
    shifted_dones = shifted_dones.at[0, :].set(False)
    model_value, logits, model_carry = agent.model.apply(
        single_train_state.params,
        minibatch.obs,
        recurrent_carry,
        shifted_dones,
    )
    value, log_probability, entropy, evaluated_carry = single_train_state.apply_fn(
        single_train_state.params,
        minibatch.actions,
        minibatch.obs,
        recurrent_carry,
        shifted_dones,
    )
    base._block((model_value, logits, model_carry, value, log_probability, entropy, evaluated_carry))
    bundle.add(
        "ppo_forward",
        "model",
        {"value": model_value, "logits": logits, "carry": model_carry},
    )
    bundle.add(
        "ppo_forward",
        "evaluated_action",
        {
            "value": value,
            "log_probability": log_probability,
            "entropy": entropy,
            "carry": evaluated_carry,
        },
    )

    value_old = minibatch.values
    target = minibatch.targets
    advantage = minibatch.advantages
    value_pred_clipped = value_old + (value - value_old).clip(
        -agent.clip_eps, agent.clip_eps
    )
    value_losses = jnp.square(value - target)
    value_losses_clipped = jnp.square(value_pred_clipped - target)
    ratio = jnp.exp(log_probability - minibatch.log_pis)
    normalized_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
    actor_term_one = ratio * normalized_advantage
    actor_term_two = jnp.clip(
        ratio, 1.0 - agent.clip_eps, 1.0 + agent.clip_eps
    ) * normalized_advantage
    actor_minimum = jnp.minimum(actor_term_one, actor_term_two)
    bundle.add(
        "ppo_loss_elements",
        "elements",
        {
            "ratio": ratio,
            "normalized_advantage": normalized_advantage,
            "actor_term_one": actor_term_one,
            "actor_term_two": actor_term_two,
            "actor_minimum": actor_minimum,
            "value_pred_clipped": value_pred_clipped,
            "value_losses": value_losses,
            "value_losses_clipped": value_losses_clipped,
            "entropy": entropy,
        },
    )

    def loss_and_gradient(params: Any, batch: Any, rng: Any) -> Any:
        return agent.grad_fn(params, single_train_state.apply_fn, batch, rng)

    (loss_and_aux, gradients) = jax.jit(loss_and_gradient)(
        single_train_state.params, minibatch, loss_rng
    )
    base._block((loss_and_aux, gradients))
    total_loss, auxiliary = loss_and_aux
    actor_loss, value_loss, mean_entropy, mean_value, mean_target, mean_advantage = auxiliary
    bundle.add(
        "ppo_loss_terms",
        "canonical_loss",
        {
            "total": total_loss,
            "actor": actor_loss,
            "value": value_loss,
            "entropy": mean_entropy,
            "mean_value": mean_value,
            "mean_target": mean_target,
            "mean_advantage": mean_advantage,
        },
    )
    bundle.add("unclipped_gradients", "gradient_tree", gradients)
    bundle.add(
        "unclipped_gradients",
        "per_leaf_norms",
        jax.tree_util.tree_map(lambda value: jnp.linalg.norm(value), gradients),
    )

    global_norm = optax.global_norm(gradients)
    clip_transformation = optax.clip_by_global_norm(runner.max_grad_norm)
    clipped_gradients, _clip_state = jax.jit(
        lambda grad: clip_transformation.update(
            grad, clip_transformation.init(single_train_state.params), single_train_state.params
        )
    )(gradients)
    clipped_global_norm = optax.global_norm(clipped_gradients)
    clip_factor = jnp.where(global_norm == 0, 1.0, clipped_global_norm / global_norm)
    base._block((global_norm, clipped_gradients, clipped_global_norm, clip_factor))
    bundle.add(
        "clipping_and_global_norm",
        "scalars",
        {
            "global_norm": global_norm,
            "clipped_global_norm": clipped_global_norm,
            "clip_factor": clip_factor,
            "threshold": jnp.asarray(runner.max_grad_norm),
        },
    )
    bundle.add("clipping_and_global_norm", "clipped_gradient_tree", clipped_gradients)

    # GradientTransformation.update computes proposed tensors and state only.
    # They are never attached to the TrainState and parameters are never changed.
    proposed_updates, proposed_opt_state = jax.jit(
        lambda grad, opt_state, params: single_train_state.tx.update(
            grad, opt_state, params
        )
    )(gradients, single_train_state.opt_state, single_train_state.params)
    base._block((proposed_updates, proposed_opt_state))
    bundle.add("adam_proposal", "parameter_update_tree", proposed_updates)
    bundle.add("adam_proposal", "proposed_optimizer_state", proposed_opt_state)
    proposal_output_param_digest = tree_digest(jax, np, single_train_state.params)
    require(
        proposal_output_param_digest == proposal_input_param_digest,
        "diagnostic mutated parameters",
    )
    require(
        int(np.asarray(single_train_state.n_updates).reshape(-1)[0]) == 0,
        "diagnostic changed update counter",
    )
    require(
        int(np.asarray(single_train_state.n_grad_updates).reshape(-1)[0]) == 0,
        "diagnostic changed gradient-update counter",
    )
    gpu_pid_present_after = Path(f"/proc/{PRESERVED_GPU_PID}").is_dir()
    if args.backend == "gpu":
        require(gpu_pid_present_after, "preserved GPU process disappeared during capture")

    receipt = {
        "schema_version": 1,
        "status": "captured_without_optimizer_application",
        "paper_evidence": False,
        "performance_endpoint": False,
        "ood_evaluation": False,
        "seed_count": 1,
        "optimizer_applications": 0,
        "parameter_mutations": 0,
        "cycle_two_experiment_steps": 0,
        "cycle_two_agent_updates": 0,
        "gradient_transformation_proposals": 1,
        "lane": args.lane,
        "backend": args.backend,
        "runtime": {
            "python": platform.python_version(),
            "jax": importlib.metadata.version("jax"),
            "jaxlib": importlib.metadata.version("jaxlib"),
            "flax": importlib.metadata.version("flax"),
            "optax": importlib.metadata.version("optax"),
            "tensorflow_probability": importlib.metadata.version("tensorflow-probability"),
            "device_platform": devices[0].platform,
            "device_kind": devices[0].device_kind,
            "jax_platforms_env": os.environ["JAX_PLATFORMS"],
            "jax_platform_name_env": os.environ["JAX_PLATFORM_NAME"],
            "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
            "xla_python_client_preallocate_env": os.environ[
                "XLA_PYTHON_CLIENT_PREALLOCATE"
            ],
            "minimax_module": str(minimax_path),
        },
        "source": source_record,
        "hashes": {
            "protocol_sha256": PROTOCOL_SHA256,
            "capture_script_sha256": sha256(Path(__file__).resolve()),
            "config_sha256": CONFIG_SHA256,
            "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA256,
        },
        "initial_summary": initial_summary,
        "cycle_one_summary": cycle_one_summary,
        "cycle_two_partial_summary": {
            "n_iters": int(np.asarray(post_rollout_train_state.n_iters).reshape(-1)[0]),
            "n_updates": int(np.asarray(post_rollout_train_state.n_updates).reshape(-1)[0]),
            "n_grad_updates": int(
                np.asarray(post_rollout_train_state.n_grad_updates).reshape(-1)[0]
            ),
            "frontier_total_trials": int(
                np.asarray(post_rollout_train_state.plr_buffer.trial_counts).sum()
            ),
            "is_replay": bool(np.asarray(is_replay).reshape(-1)[0]),
        },
        "parameters": {
            "initial_population_sha256": initial_param_digest,
            "proposal_input_sha256": proposal_input_param_digest,
            "proposal_output_sha256": proposal_output_param_digest,
            "unchanged": True,
        },
        "preserved_gpu_process": {
            "pid": PRESERVED_GPU_PID,
            "present_before": gpu_pid_present_before,
            "present_after": gpu_pid_present_after,
        },
        "groups": bundle.groups,
        "records": bundle.records,
        "record_count": len(bundle.records),
    }
    atomic_json(output / "capture.json", receipt)
    print(
        "COMPONENT_CAPTURE_OK "
        f"lane={args.lane} backend={args.backend} "
        f"records={len(bundle.records)} optimizer_applications=0"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", choices=("reference", "modern"), required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    args = parser.parse_args()
    try:
        capture(args)
    except (
        CaptureError,
        AssertionError,
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"COMPONENT_CAPTURE_ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
