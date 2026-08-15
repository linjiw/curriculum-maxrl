#!/usr/bin/env python3
"""Externally evaluate one matched-development terminal checkpoint.

The evaluation seed is always ``100000 + training_seed``.  The evaluator runs
exactly ten policy episodes, in order, on ``Maze-SixteenRooms``,
``Maze-Labyrinth``, and ``Maze-StandardMaze``.  It emits a closed atomic
package containing all 30 raw per-episode solved outcomes, the aggregate CSV
expected by the frozen analyzer, and a receipt binding the terminal checkpoint
and all provenance.

A deterministic synthetic path exists only for bounded CPU package tests.  It
still consumes and hash-binds a self-generated terminal checkpoint, is visibly
marked non-evidence, and is refused outside engineering test mode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

import run_matched_terminal as training


ENVIRONMENTS = (
    "Maze-SixteenRooms",
    "Maze-Labyrinth",
    "Maze-StandardMaze",
)
N_EPISODES = 10
EVALUATION_SEED_OFFSET = 100000
EVALUATION_HORIZON = 450
FLOAT32_AGGREGATE_TOLERANCE = 2e-6
PAYLOADS = (
    "evaluation-episodes.jsonl",
    "evaluation.csv",
    "evaluation-receipt.json",
)
MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$")


class EvaluationError(RuntimeError):
    """Raised when the external evaluation contract is not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def _sha256(path: Path) -> str:
    try:
        return training.sha256(path)
    except training.DriverError as exc:
        raise EvaluationError(str(exc)) from exc


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return training.load_json(path, label)
    except training.DriverError as exc:
        raise EvaluationError(str(exc)) from exc


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        training.atomic_json(path, value)
    except training.DriverError as exc:
        raise EvaluationError(str(exc)) from exc


def _atomic_text(path: Path, value: str) -> None:
    require(path.parent.is_dir() and not path.parent.is_symlink(), "unsafe output parent")
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_training_inputs(
    checkpoint: Path,
    endpoint: Path,
    training_receipt_path: Path,
    meta_path: Path,
    context: Mapping[str, Any],
    context_sha: str,
    protocol_sha: str,
    expected_training_driver_sha: str,
    fresh_source_receipt: Mapping[str, Any],
    engineering_test_mode: bool,
    slurm_engineering_test_mode: bool,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    receipt = _load_json(training_receipt_path, "training receipt")
    require(receipt.get("schema") == 1 and receipt.get("status") == "completed", "training is incomplete")
    require(receipt.get("protocol_id") == training.PROTOCOL_ID, "training protocol drift")
    require(receipt.get("purpose") == training.PURPOSE, "training purpose drift")
    require(receipt.get("paper_evidence") is False, "training evidence label drift")
    expected_execution_mode = (
        "slurm"
        if slurm_engineering_test_mode
        else "local"
        if engineering_test_mode
        else "production"
    )
    expected_endpoint_class = (
        "bounded_engineering_test"
        if engineering_test_mode or slurm_engineering_test_mode
        else "matched_development"
    )
    require(
        receipt.get("endpoint_class") == expected_endpoint_class,
        "training/evaluation endpoint-class drift",
    )
    engineering = receipt.get("engineering_test")
    require(
        isinstance(engineering, dict)
        and set(engineering) == {"enabled", "execution_mode", "overrides"}
        and engineering["enabled"]
        is (engineering_test_mode or slurm_engineering_test_mode)
        and engineering["execution_mode"] == expected_execution_mode
        and isinstance(engineering["overrides"], list)
        and (
            len(engineering["overrides"]) >= 1
            if engineering_test_mode or slurm_engineering_test_mode
            else engineering["overrides"] == []
        ),
        "training/evaluation execution-mode drift",
    )
    for key in ("run_id", "arm", "training_seed", "job_id"):
        require(receipt.get(key) == context[key], f"training/context identity drift: {key}")
    try:
        training.validate_training_sidecar(
            training_receipt_path.parent, context["run_id"], context["arm"]
        )
    except training.DriverError as exc:
        raise EvaluationError(str(exc)) from exc
    provenance = receipt.get("provenance")
    require(isinstance(provenance, dict), "training provenance is missing")
    require(provenance.get("run_context") == context, "training run-context contents drift")
    require(provenance.get("run_context_sha256") == context_sha, "training run-context hash drift")
    require(
        provenance.get("protocol_sha256") == protocol_sha,
        "training/evaluation protocol hash drift",
    )
    require(
        provenance.get("training_driver_sha256") == expected_training_driver_sha,
        "training receipt driver hash drift",
    )
    require(
        provenance.get("source") == fresh_source_receipt,
        "training/evaluation source receipt drift",
    )
    terminal = receipt.get("terminal_checkpoint")
    require(isinstance(terminal, dict), "terminal checkpoint receipt is missing")
    checkpoint_sha = _sha256(checkpoint)
    require(terminal.get("sha256") == checkpoint_sha, "terminal checkpoint hash drift")
    require(terminal.get("saved_after_loop_termination") is True, "checkpoint is not terminal")
    require(terminal.get("periodic_checkpoint_used") is False, "periodic checkpoint is inadmissible")
    require(receipt.get("resumed") is False, "resumed training is inadmissible")
    require(
        _sha256(endpoint) == receipt.get("endpoint", {}).get("sha256"),
        "endpoint receipt hash drift",
    )
    meta = _load_json(meta_path, "training metadata")
    require(_sha256(meta_path) == receipt.get("config", {}).get("meta_sha256"), "metadata hash drift")
    require(meta.get("xpid") == context["run_id"], "metadata run ID drift")
    config = meta.get("config")
    require(isinstance(config, dict), "metadata config missing")
    require(config.get("seed") == context["training_seed"], "metadata seed drift")
    require(config.get("xpid") == context["run_id"], "metadata config run ID drift")
    require(config.get("from_last_checkpoint") is False, "metadata permits resume")
    require(
        config.get("eval_args", {}).get("env_names") == ",".join(ENVIRONMENTS)
        and config.get("eval_args", {}).get("n_episodes") == N_EPISODES,
        "metadata evaluation target drift",
    )
    require(
        training.canonical_sha256(config)
        == receipt.get("config", {}).get("resolved_canonical_sha256"),
        "resolved config hash drift",
    )
    return receipt, meta, checkpoint_sha


def _synthetic_records(checkpoint_sha: str, evaluation_seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        for episode in range(N_EPISODES):
            key = f"{checkpoint_sha}|{evaluation_seed}|{environment}|{episode}".encode("utf-8")
            digest = hashlib.sha256(key).digest()
            solved = bool(digest[0] & 1)
            episode_return = ((digest[1] % 10) + 1) / 10.0 if solved else 0.0
            records.append(
                {
                    "environment": environment,
                    "episode": episode,
                    "agent_index": 0,
                    "solved": solved,
                    "return": episode_return,
                }
            )
    return records


def validate_backend(
    jax: Any,
    *,
    engineering_test_mode: bool,
    slurm_engineering_test_mode: bool = False,
) -> list[dict[str, Any]]:
    require(
        not (engineering_test_mode and slurm_engineering_test_mode),
        "local and Slurm engineering modes are mutually exclusive",
    )
    expected = "cpu" if engineering_test_mode else "gpu"
    require(jax.default_backend() == expected, f"expected {expected} evaluation backend")
    devices = jax.devices(expected)
    require(len(devices) == 1, "evaluation requires exactly one visible device")
    return [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
        }
        for device in devices
    ]


def _actual_records(
    checkpoint: Path,
    meta_path: Path,
    evaluation_seed: int,
    source_dir: Path,
    engineering_test_mode: bool,
    slurm_engineering_test_mode: bool,
    verify_independent_aggregate: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_module_dir = source_dir / "src"
    require(source_module_dir.is_dir(), "patched source lacks src directory")
    require("minimax" not in sys.modules, "minimax was imported before evaluator source validation")
    sys.path.insert(0, str(source_module_dir))
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
    import minimax  # type: ignore
    import minimax.agents as agents  # type: ignore
    import minimax.models as models  # type: ignore
    from minimax.runners import EvalRunner  # type: ignore
    from minimax.util.checkpoint import load_config, load_pkl_object  # type: ignore
    from minimax.util.rl import AgentPop  # type: ignore

    minimax_path = Path(minimax.__file__).resolve()
    require(minimax_path.is_relative_to(source_dir), "minimax import escaped patched source")
    device_receipt = validate_backend(
        jax,
        engineering_test_mode=engineering_test_mode,
        slurm_engineering_test_mode=slurm_engineering_test_mode,
    )
    checkpoint_state = load_pkl_object(str(checkpoint))
    require(isinstance(checkpoint_state, (list, tuple)) and len(checkpoint_state) >= 2, "checkpoint shape drift")
    train_state = checkpoint_state[1]
    require(isinstance(train_state, Mapping) and "params" in train_state, "checkpoint params missing")
    agent_indices = jnp.asarray([0], dtype=jnp.int32)
    params = jax.tree_util.tree_map(
        lambda value: jnp.take(value, indices=agent_indices, axis=0),
        train_state["params"],
    )

    xp_args = load_config(str(meta_path))
    student_model = models.make(
        env_name=xp_args.env_name,
        model_name=xp_args.student_model_name,
        **xp_args.student_model_args,
    )
    pop = AgentPop(agent=agents.PPOAgent(model=student_model), n_agents=1)
    runner = EvalRunner(
        pop=pop,
        env_names=",".join(ENVIRONMENTS),
        env_kwargs=xp_args.eval_env_args,
        n_episodes=N_EPISODES,
        render_mode=None,
        agent_idxs="*",
    )
    require(tuple(runner.ext_env_names) == ENVIRONMENTS, "resolved evaluation environment order drift")
    require(runner.n_episodes == N_EPISODES and runner.n_envs == len(ENVIRONMENTS), "evaluation shape drift")
    evaluation_horizons = tuple(
        int(benv.env.max_episode_steps()) for benv in runner.benvs
    )
    require(
        evaluation_horizons == (EVALUATION_HORIZON,) * len(ENVIRONMENTS),
        "resolved evaluation horizon drift",
    )

    rng = jax.random.PRNGKey(evaluation_seed)
    rng, *rollout_rngs = jax.random.split(rng, runner.n_envs + 1)
    records: list[dict[str, Any]] = []
    for index, (benv, env_params) in enumerate(zip(runner.benvs, runner.env_params)):
        rng, *reset_rngs = jax.random.split(rng, pop.n_agents + 1)
        obs, state, extra = benv.reset(jnp.array(reset_rngs))
        if pop.agent.is_recurrent:
            rng, carry_rng = jax.random.split(rng)
            zero_carry = pop.init_carry(carry_rng, obs)
        else:
            zero_carry = None
        ep_stats = runner.rolling_stats.reset_stats(batch_shape=(pop.n_agents, N_EPISODES))
        ep_stats = runner._rollout_benv(
            rollout_rngs[index],
            benv,
            jax.lax.stop_gradient(params),
            env_params,
            state,
            obs,
            zero_carry,
            zero_carry,
            extra,
            ep_stats,
        )
        jax.tree_util.tree_map(
            lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
            ep_stats,
        )
        episode_counts = np.asarray(ep_stats["n_episodes"])[0, :, 0]
        returns = np.asarray(ep_stats["return"])[0, :, 0]
        solved = np.asarray(jax.vmap(jax.vmap(benv.env.eval_solved_rate))(ep_stats))[0, :, 0]
        require(
            episode_counts.shape == returns.shape == solved.shape == (N_EPISODES,),
            "raw episode array shape drift",
        )
        require((episode_counts == 1).all(), "an evaluation lane did not produce exactly one episode")
        for episode in range(N_EPISODES):
            episode_return = float(returns[episode])
            require(math.isfinite(episode_return), "non-finite episode return")
            records.append(
                {
                    "environment": ENVIRONMENTS[index],
                    "episode": episode,
                    "agent_index": 0,
                    "solved": bool(solved[episode]),
                    "return": episode_return,
                }
            )
    aggregate_parity: dict[str, Any] = {
        "checked": False,
        "all_six_fields_checked": False,
        "max_abs_error": None,
        "per_field_abs_error": None,
        "float32_tolerance": FLOAT32_AGGREGATE_TOLERANCE,
    }
    if verify_independent_aggregate:
        validate_records(records)
        raw_aggregate = _aggregate(records)
        independent = runner.run(jax.random.PRNGKey(evaluation_seed), params)
        jax.tree_util.tree_map(
            lambda value: (
                value.block_until_ready() if hasattr(value, "block_until_ready") else value
            ),
            independent,
        )
        require(set(independent) == set(raw_aggregate), "independent EvalRunner aggregate keys drift")
        aggregate_errors = {
            key: abs(float(np.asarray(independent[key])) - raw_aggregate[key])
            for key in sorted(raw_aggregate)
        }
        aggregate_max_abs_error = max(aggregate_errors.values(), default=0.0)
        require(
            aggregate_max_abs_error <= FLOAT32_AGGREGATE_TOLERANCE,
            "raw episode aggregation disagrees with independent EvalRunner.run",
        )
        aggregate_parity = {
            "checked": True,
            "all_six_fields_checked": True,
            "max_abs_error": aggregate_max_abs_error,
            "per_field_abs_error": aggregate_errors,
            "float32_tolerance": FLOAT32_AGGREGATE_TOLERANCE,
        }
    return records, {
        "backend": jax.default_backend(),
        "device_count": len(device_receipt),
        "devices": device_receipt,
        "minimax_module": str(minimax_path),
        "per_environment_max_episode_horizons": list(evaluation_horizons),
        "raw_vs_independent_evalrunner": aggregate_parity,
    }


def validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    require(len(records) == len(ENVIRONMENTS) * N_EPISODES, "raw record count drift")
    expected = [
        (environment, episode)
        for environment in ENVIRONMENTS
        for episode in range(N_EPISODES)
    ]
    actual: list[tuple[str, int]] = []
    for record in records:
        require(set(record) == {"environment", "episode", "agent_index", "solved", "return"}, "raw record keys drift")
        require(record["agent_index"] == 0, "raw record agent drift")
        require(type(record["solved"]) is bool, "raw solved outcome is not boolean")
        require(
            isinstance(record["episode"], int) and not isinstance(record["episode"], bool),
            "episode index is not integral",
        )
        episode_return = record["return"]
        require(
            isinstance(episode_return, (int, float))
            and not isinstance(episode_return, bool)
            and math.isfinite(float(episode_return)),
            "raw episode return is invalid",
        )
        require(0.0 <= float(episode_return) <= 1.0, "raw AMaze return is out of range")
        require(
            bool(float(episode_return) > 0.0) is record["solved"],
            "raw solved/return semantics drift",
        )
        actual.append((record["environment"], record["episode"]))
    require(actual == expected, "raw record environment/episode order drift")


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for environment in ENVIRONMENTS:
        rows = [row for row in records if row["environment"] == environment]
        require(len(rows) == N_EPISODES, "environment episode count drift")
        values[f"eval/a0:test_return:{environment}"] = float(
            np.mean([float(row["return"]) for row in rows])
        )
        values[f"eval/a0:test_solved_rate:{environment}"] = float(
            np.mean([int(row["solved"]) for row in rows])
        )
    return values


def _evaluation_csv(aggregate: Mapping[str, float]) -> str:
    fieldnames = [
        key
        for environment in ENVIRONMENTS
        for key in (
            f"eval/a0:test_return:{environment}",
            f"eval/a0:test_solved_rate:{environment}",
        )
    ]
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerow({key: format(aggregate[key], ".17g") for key in fieldnames})
    return stream.getvalue()


def validate_package(root: Path, run_id: str) -> str:
    require(root.is_dir() and not root.is_symlink(), "evaluation package is missing")
    expected_names = set(PAYLOADS) | {"SHA256SUMS", "COMPLETE"}
    actual_names: set[str] = set()
    for entry in root.iterdir():
        require(entry.is_file() and not entry.is_symlink(), f"unsafe package entry: {entry.name}")
        actual_names.add(entry.name)
    require(actual_names == expected_names, "evaluation package closure drift")
    listed: dict[str, str] = {}
    lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in lines:
        match = MANIFEST_LINE.fullmatch(line)
        require(match is not None, "unsafe SHA256SUMS line")
        digest, name = match.groups()
        require(name not in listed, "duplicate SHA256SUMS path")
        listed[name] = digest
    require(set(listed) == set(PAYLOADS), "SHA256SUMS payload closure drift")
    for name, expected in listed.items():
        require(_sha256(root / name) == expected, f"evaluation payload hash drift: {name}")
    manifest_sha = _sha256(root / "SHA256SUMS")
    complete = _load_json(root / "COMPLETE", "evaluation COMPLETE")
    require(
        complete
        == {
            "schema": 1,
            "status": "complete",
            "run_id": run_id,
            "sha256sums_sha256": manifest_sha,
            "file_count": len(PAYLOADS),
        },
        "evaluation COMPLETE binding drift",
    )
    return manifest_sha


def _write_package(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    receipt_base: Mapping[str, Any],
) -> dict[str, Any]:
    require(output_dir.is_absolute(), "evaluation output directory must be absolute")
    require(not output_dir.exists() and not output_dir.is_symlink(), "evaluation output exists")
    require(output_dir.parent.is_dir() and not output_dir.parent.is_symlink(), "unsafe output parent")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        raw_text = "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for record in records
        )
        raw_path = temporary / "evaluation-episodes.jsonl"
        _atomic_text(raw_path, raw_text)
        aggregate = _aggregate(records)
        csv_path = temporary / "evaluation.csv"
        _atomic_text(csv_path, _evaluation_csv(aggregate))
        receipt = dict(receipt_base)
        receipt["raw_results"] = {
            "path": raw_path.name,
            "sha256": _sha256(raw_path),
            "record_count": len(records),
        }
        receipt["aggregate_results"] = {
            "path": csv_path.name,
            "sha256": _sha256(csv_path),
            "values": aggregate,
        }
        receipt_path = temporary / "evaluation-receipt.json"
        _atomic_json(receipt_path, receipt)
        manifest_text = "".join(f"{_sha256(temporary / name)}  {name}\n" for name in PAYLOADS)
        manifest_path = temporary / "SHA256SUMS"
        _atomic_text(manifest_path, manifest_text)
        complete = {
            "schema": 1,
            "status": "complete",
            "run_id": receipt["run_id"],
            "sha256sums_sha256": _sha256(manifest_path),
            "file_count": len(PAYLOADS),
        }
        _atomic_json(temporary / "COMPLETE", complete)
        validate_package(temporary, receipt["run_id"])
        os.replace(temporary, output_dir)
        validate_package(output_dir, receipt["run_id"])
        return receipt
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run(cli: argparse.Namespace) -> dict[str, Any]:
    driver_path = Path(__file__).resolve()
    driver_sha = _sha256(driver_path)
    require(training._is_hash(cli.expected_driver_sha256), "expected evaluator SHA-256 is malformed")
    require(driver_sha == cli.expected_driver_sha256, "evaluation driver SHA-256 mismatch")
    require(
        not (cli.engineering_test_mode and cli.slurm_engineering_test_mode),
        "local and Slurm engineering modes are mutually exclusive",
    )
    context_path = cli.run_context.resolve()
    try:
        context = training.validate_run_context(
            context_path,
            cli.expected_run_context_sha256,
            arm=cli.arm,
            engineering_test_mode=cli.engineering_test_mode,
            slurm_engineering_test_mode=cli.slurm_engineering_test_mode,
        )
    except training.DriverError as exc:
        raise EvaluationError(str(exc)) from exc
    require(
        context["provenance"]["evaluation_driver_sha256"] == driver_sha,
        "run context binds another evaluation driver",
    )
    training_driver_path = Path(training.__file__).resolve()
    require(
        context["provenance"]["training_driver_sha256"]
        == _sha256(training_driver_path),
        "run context binds a drifted training driver",
    )
    protocol, protocol_sha = training.load_protocol(cli.protocol.resolve())
    try:
        training.validate_campaign_binding(
            cli.campaign_manifest,
            cli.expected_campaign_manifest_sha256,
            context=context,
            protocol=protocol,
            protocol_sha256=protocol_sha,
            engineering_test_mode=cli.engineering_test_mode,
            slurm_engineering_test_mode=cli.slurm_engineering_test_mode,
        )
    except training.DriverError as exc:
        raise EvaluationError(str(exc)) from exc
    require(tuple(protocol["evaluation"]["environments"]) == ENVIRONMENTS, "protocol env order drift")
    require(protocol["evaluation"]["n_episodes_per_environment"] == N_EPISODES, "protocol episode drift")
    require(protocol["evaluation"]["evaluation_seed_offset"] == EVALUATION_SEED_OFFSET, "protocol seed offset drift")
    require(
        protocol["evaluation"]["max_episode_horizon"] == EVALUATION_HORIZON,
        "protocol evaluation horizon drift",
    )
    source_dir = cli.patched_source_dir.resolve()
    try:
        source_receipt = training.validate_source(
            source_dir,
            context,
            git_executable=cli.git_executable,
            require_pinned_git=not cli.engineering_test_mode,
        )
    except training.DriverError as exc:
        raise EvaluationError(str(exc)) from exc

    checkpoint = cli.checkpoint.resolve()
    endpoint = cli.endpoint.resolve()
    training_receipt_path = cli.training_receipt.resolve()
    meta_path = cli.meta.resolve()
    training_receipt, _meta, checkpoint_sha = _validate_training_inputs(
        checkpoint,
        endpoint,
        training_receipt_path,
        meta_path,
        context,
        cli.expected_run_context_sha256,
        protocol_sha,
        context["provenance"]["training_driver_sha256"],
        source_receipt,
        cli.engineering_test_mode,
        cli.slurm_engineering_test_mode,
    )
    if cli.synthetic_test_mode:
        require(cli.engineering_test_mode, "synthetic evaluator is local-test-only")
        require(
            training_receipt.get("endpoint_class") == "bounded_engineering_test",
            "synthetic evaluator requires a bounded training endpoint",
        )
    if cli.engineering_verify_independent_aggregate:
        require(
            cli.engineering_test_mode,
            "independent aggregate verification is local-test-only",
        )
        require(not cli.synthetic_test_mode, "synthetic evaluator cannot verify EvalRunner parity")
    evaluation_seed = EVALUATION_SEED_OFFSET + context["training_seed"]
    if cli.synthetic_test_mode:
        records = _synthetic_records(checkpoint_sha, evaluation_seed)
        runtime = {
            "backend": "deterministic_synthetic",
            "device_count": 0,
            "devices": [],
            "minimax_module": None,
        }
    else:
        records, runtime = _actual_records(
            checkpoint,
            meta_path,
            evaluation_seed,
            source_dir,
            cli.engineering_test_mode,
            cli.slurm_engineering_test_mode,
            cli.engineering_verify_independent_aggregate,
        )
    validate_records(records)
    resolved_horizons = runtime.get("per_environment_max_episode_horizons")
    if cli.synthetic_test_mode:
        resolved_horizons = [EVALUATION_HORIZON] * len(ENVIRONMENTS)
    require(
        resolved_horizons == [EVALUATION_HORIZON] * len(ENVIRONMENTS),
        "runtime evaluation horizon receipt drift",
    )
    primary_transition_budget = N_EPISODES * sum(resolved_horizons)

    receipt_base = {
        "schema": 1,
        "status": "completed",
        "protocol_id": training.PROTOCOL_ID,
        "purpose": training.PURPOSE,
        "paper_evidence": False,
        "run_id": context["run_id"],
        "arm": context["arm"],
        "training_seed": context["training_seed"],
        "evaluation_seed": evaluation_seed,
        "environments": list(ENVIRONMENTS),
        "n_episodes_per_environment": N_EPISODES,
        "agent_indices": [0],
        "synthetic_test_mode": bool(cli.synthetic_test_mode),
        "evaluation_transition_accounting": {
            "environment_count": len(ENVIRONMENTS),
            "episodes_per_environment": N_EPISODES,
            "max_episode_horizon": EVALUATION_HORIZON,
            "per_environment_max_episode_horizons": resolved_horizons,
            "budgeted_primary_max_transitions": primary_transition_budget,
            "effective_primary_transitions": (
                0
                if cli.synthetic_test_mode
                else primary_transition_budget
            ),
            "primary_runner_scans_full_horizon": not cli.synthetic_test_mode,
            "engineering_independent_verification_transitions": (
                primary_transition_budget
                if cli.engineering_verify_independent_aggregate
                else 0
            ),
            "total_runtime_transitions": (
                0
                if cli.synthetic_test_mode
                else (
                    primary_transition_budget
                    * (2 if cli.engineering_verify_independent_aggregate else 1)
                )
            ),
            "excluded_from_student_training_transitions": True,
        },
        "terminal_checkpoint": {"sha256": checkpoint_sha},
        "training_receipt_sha256": _sha256(training_receipt_path),
        "meta_sha256": _sha256(meta_path),
        "provenance": {
            "run_context": context,
            "run_context_sha256": cli.expected_run_context_sha256,
            "protocol_sha256": protocol_sha,
            "evaluation_driver_sha256": driver_sha,
            "source": source_receipt,
            "runtime": runtime,
        },
    }
    return _write_package(cli.output_dir, records, receipt_base)


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=training.ARMS, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--campaign-manifest",
        type=Path,
        help="frozen campaign manifest; required under Slurm/production",
    )
    parser.add_argument(
        "--expected-campaign-manifest-sha256",
        help="exact digest of --campaign-manifest; required under Slurm/production",
    )
    parser.add_argument("--run-context", type=Path, required=True)
    parser.add_argument("--expected-run-context-sha256", required=True)
    parser.add_argument("--expected-driver-sha256", required=True)
    parser.add_argument("--patched-source-dir", type=Path, required=True)
    parser.add_argument(
        "--git-executable",
        type=Path,
        help="absolute Git executable; required from the active pinned environment under Slurm/production",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engineering-test-mode", action="store_true")
    parser.add_argument(
        "--slurm-engineering-test-mode",
        action="store_true",
        help="bounded, permanently non-evidence engineering mode under Slurm",
    )
    parser.add_argument("--synthetic-test-mode", action="store_true")
    parser.add_argument("--engineering-verify-independent-aggregate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        receipt = run(parse_cli(argv))
    except (EvaluationError, training.DriverError, AssertionError, KeyError, TypeError, ValueError) as exc:
        print(f"MATCHED_EVALUATION_REFUSED: {exc}", file=sys.stderr)
        return 1
    print(
        "MATCHED_EVALUATION_COMPLETE "
        f"run_id={receipt['run_id']} seed={receipt['evaluation_seed']} "
        f"records={receipt['raw_results']['record_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
