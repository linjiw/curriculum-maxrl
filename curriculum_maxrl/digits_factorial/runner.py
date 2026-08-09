"""Independent, source-locked runner for the Digits factorial."""

from __future__ import annotations

import argparse
import copy
import math
import multiprocessing
import os
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .core import (
    ACTION_BUDGET,
    ARTIFACT_SCHEMA,
    CELLS,
    CHECKPOINT_SCHEMA,
    CONFIRMATION_SEEDS,
    DATA_MANIFEST_PATH,
    DEVELOPMENT_LRS,
    DEVELOPMENT_SEEDS,
    DEV_SIZE,
    ENGINEERING_SEED,
    EVAL_INTERVAL_STEPS,
    EXPOSURE_BIN_EDGES,
    GROUP_SIZE,
    GROUPS_PER_STEP,
    MOMENTUM,
    N_CLASSES,
    N_STEPS,
    PARAMETER_COUNT,
    PROJECT_ROOT,
    RECOVERY_STEPS,
    SAMPLERS,
    SOURCE_LOCK_PATH,
    TEST_SIZE,
    TRAIN_SIZE,
    Cell,
    assert_pinned_runtime,
    cell_from_name,
    configure_deterministic_cpu,
    domain_seed,
    estimator_weights,
    evaluate_probabilities,
    generate_rng_tapes,
    initialize_model,
    inverse_cdf_actions,
    inverse_cdf_indices,
    load_stored_digits,
    model_state_sha256,
    normalized_auc,
    sampler_probabilities,
    sha256_array,
    sha256_file,
    strict_json_load,
    torch_state_sha256,
    write_json,
)
from .locking import (
    load_and_verify_source_lock,
    validate_lr_selection_document,
    verify_execution_authorization,
)


CANONICAL_PHASES = {"development", "confirmation_tuned", "confirmation_common"}


@dataclass(frozen=True)
class RunSpec:
    phase: str
    cell: Cell
    seed: int
    learning_rate: float
    steps: int
    include_test: bool
    output_dir: Path
    authorization_path: Path | None = None
    lr_selection_path: Path | None = None
    worker_mode: str = "direct"
    requested_workers: int = 1

    @property
    def canonical(self) -> bool:
        return self.phase in CANONICAL_PHASES


def _validate_spec(spec: RunSpec) -> None:
    if spec.phase not in CANONICAL_PHASES | {"engineering"}:
        raise ValueError(f"unrecognized phase: {spec.phase}")
    if not math.isfinite(spec.learning_rate) or spec.learning_rate < 0.0:
        raise ValueError("learning rate must be finite and nonnegative")
    if spec.worker_mode not in {"direct", "serial", "process_pool_worker"}:
        raise ValueError("unregistered worker mode")
    if type(spec.requested_workers) is not int or spec.requested_workers < 1:
        raise ValueError("requested worker count must be a positive exact integer")
    if spec.worker_mode in {"direct", "serial"} and spec.requested_workers != 1:
        raise ValueError("direct/serial execution must request exactly one worker")
    if spec.worker_mode == "process_pool_worker" and spec.requested_workers < 2:
        raise ValueError("process-pool execution must request at least two workers")
    if spec.phase == "engineering":
        if spec.seed != ENGINEERING_SEED:
            raise ValueError(f"engineering must use reserved seed {ENGINEERING_SEED}")
        if not 1 <= spec.steps < N_STEPS:
            raise ValueError("engineering must use a positive truncated schedule")
        if spec.include_test:
            raise ValueError("engineering may not inspect sealed-test outcomes")
        if "engineering" not in spec.output_dir.parts:
            raise ValueError("engineering outputs must live under an engineering directory")
        return

    if spec.steps != N_STEPS:
        raise ValueError("evidence-bearing phases require the exact 512-step schedule")
    if spec.authorization_path is None:
        raise ValueError("evidence-bearing phase lacks execution authorization")
    if spec.phase == "development":
        if spec.seed not in DEVELOPMENT_SEEDS:
            raise ValueError("unregistered development seed")
        if spec.learning_rate not in DEVELOPMENT_LRS:
            raise ValueError("unregistered development learning rate")
        if spec.include_test:
            raise ValueError("development artifacts must omit sealed-test outcomes")
    else:
        if spec.seed not in CONFIRMATION_SEEDS:
            raise ValueError("unregistered confirmation seed")
        if not spec.include_test:
            raise ValueError("confirmation must store sealed-test outcomes")
        if spec.lr_selection_path is None:
            raise ValueError("confirmation requires the frozen LR selection artifact")


def _read_lr_selection(path: Path) -> dict[str, Any]:
    _, lock_sha = load_and_verify_source_lock(check_runtime=False)
    payload = strict_json_load(path)
    validate_lr_selection_document(payload, lock_sha256=lock_sha)
    return payload


def _expected_confirmation_lr(phase: str, cell: Cell, selection: Mapping[str, Any]) -> float:
    if phase == "confirmation_tuned":
        return float(selection["selected_learning_rates_by_estimator"][cell.estimator])
    if phase == "confirmation_common":
        return float(selection["selected_common_learning_rate"])
    raise ValueError("not a confirmation phase")


def _validate_confirmation_lr(
    phase: str,
    cell: Cell,
    learning_rate: float,
    selection: Mapping[str, Any],
) -> None:
    expected = _expected_confirmation_lr(phase, cell, selection)
    if type(learning_rate) is not float or learning_rate != expected:
        raise ValueError(f"confirmation LR {learning_rate} != frozen {expected}")


def _eval_steps(steps: int) -> tuple[int, ...]:
    checkpoints = [0]
    checkpoints.extend(range(EVAL_INTERVAL_STEPS, steps + 1, EVAL_INTERVAL_STEPS))
    if checkpoints[-1] != steps:
        checkpoints.append(steps)
    return tuple(checkpoints)


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    seed: int,
    tape_metadata: Mapping[str, Any],
) -> None:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "step": step,
        "action_budget": step * GROUPS_PER_STEP * GROUP_SIZE,
        "logical_seed": seed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "python_random_state": random.getstate(),
        "numpy_tape_terminal_states": copy.deepcopy(tape_metadata["terminal_states"]),
        "numpy_tape_sha256": copy.deepcopy(tape_metadata["full_tape_sha256"]),
    }
    torch.save(payload, path)


@torch.no_grad()
def _probabilities(model: torch.nn.Module, inputs: torch.Tensor) -> np.ndarray:
    return torch.softmax(model(inputs), dim=1).detach().cpu().numpy()


def _evaluation_record(
    model: torch.nn.Module,
    arrays: Mapping[str, tuple[torch.Tensor, np.ndarray]],
    *,
    step: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics: dict[str, Any] = {
        "step": step,
        "action_budget": step * GROUPS_PER_STEP * GROUP_SIZE,
        "splits": {},
    }
    probability_arrays: dict[str, np.ndarray] = {}
    for name, (inputs, targets) in arrays.items():
        probs = _probabilities(model, inputs)
        probability_arrays[name] = probs
        metrics["splits"][name] = evaluate_probabilities(probs, targets)
    return metrics, probability_arrays


def _gradient_norm(model: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            value = float(torch.square(parameter.grad.detach()).sum().item())
            squared += value
    return math.sqrt(squared)


def _npz_payload_hashes(payload: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {name: sha256_array(value) for name, value in sorted(payload.items())}


def run_one(spec: RunSpec) -> Path:
    """Execute one cell and emit a complete ledger plus recovery checkpoints."""

    execution_settings = configure_deterministic_cpu()
    _validate_spec(spec)
    lock, lock_sha = load_and_verify_source_lock(check_runtime=True)
    if spec.canonical:
        assert spec.authorization_path is not None
        verify_execution_authorization(
            spec.authorization_path,
            phase=spec.phase,
            lock_sha256=lock_sha,
            lr_selection_path=spec.lr_selection_path,
        )
    selection: dict[str, Any] | None = None
    if spec.phase.startswith("confirmation"):
        assert spec.lr_selection_path is not None
        selection = _read_lr_selection(spec.lr_selection_path)
        _validate_confirmation_lr(
            spec.phase, spec.cell, spec.learning_rate, selection
        )

    if spec.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {spec.output_dir}")
    spec.output_dir.mkdir(parents=True)
    failure_path = spec.output_dir / "failure.json"
    started = time.perf_counter()

    try:
        runtime = assert_pinned_runtime()
        torch.manual_seed(domain_seed(spec.seed, "torch-global") % (2**63 - 1))
        random.seed(domain_seed(spec.seed, "python-global"))
        x, y, split_indices, data_manifest = load_stored_digits()
        train_indices = split_indices["train"]
        dev_indices = split_indices["dev"]
        test_indices = split_indices["test"]
        x_train_np = x[train_indices]
        y_train = y[train_indices]
        x_train = torch.from_numpy(x_train_np)
        x_dev = torch.from_numpy(x[dev_indices])
        x_test = torch.from_numpy(x[test_indices])
        evaluation_arrays: dict[str, tuple[torch.Tensor, np.ndarray]] = {
            "train": (x_train, y_train),
            "dev": (x_dev, y[dev_indices]),
        }
        if spec.include_test:
            evaluation_arrays["test"] = (x_test, y[test_indices])

        model = initialize_model(spec.seed)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=spec.learning_rate, momentum=MOMENTUM, weight_decay=0.0
        )
        initial_model_sha = model_state_sha256(model)
        task_tape, action_tape, tape_metadata = generate_rng_tapes(
            spec.seed, steps=spec.steps
        )

        ledger: dict[str, np.ndarray] = {
            "p_train_by_step": np.empty((spec.steps, TRAIN_SIZE), dtype=np.float64),
            "q_by_step": np.empty((spec.steps, TRAIN_SIZE), dtype=np.float64),
            "selected_train_positions": np.empty(
                (spec.steps, GROUPS_PER_STEP), dtype=np.int32
            ),
            "selected_original_indices": np.empty(
                (spec.steps, GROUPS_PER_STEP), dtype=np.int32
            ),
            "selected_action_probabilities": np.empty(
                (spec.steps, GROUPS_PER_STEP, N_CLASSES), dtype=np.float64
            ),
            "actions": np.empty(
                (spec.steps, GROUPS_PER_STEP, GROUP_SIZE), dtype=np.uint8
            ),
            "rewards": np.empty(
                (spec.steps, GROUPS_PER_STEP, GROUP_SIZE), dtype=np.uint8
            ),
            "weights": np.empty(
                (spec.steps, GROUPS_PER_STEP, GROUP_SIZE), dtype=np.float64
            ),
            "group_success_count": np.empty(
                (spec.steps, GROUPS_PER_STEP), dtype=np.uint8
            ),
            "group_absolute_mass": np.empty(
                (spec.steps, GROUPS_PER_STEP), dtype=np.float64
            ),
            "loss": np.empty(spec.steps, dtype=np.float64),
            "gradient_norm": np.empty(spec.steps, dtype=np.float64),
        }

        eval_steps = _eval_steps(spec.steps)
        eval_records: list[dict[str, Any]] = []
        eval_probability_lists: dict[str, list[np.ndarray]] = {
            name: [] for name in evaluation_arrays
        }
        checkpoint_records: list[dict[str, Any]] = []

        initial_eval, initial_probs = _evaluation_record(
            model, evaluation_arrays, step=0
        )
        eval_records.append(initial_eval)
        for name, probs in initial_probs.items():
            eval_probability_lists[name].append(probs)

        checkpoint_steps = (
            RECOVERY_STEPS if spec.steps == N_STEPS else tuple(sorted({0, spec.steps}))
        )
        checkpoint_path = spec.output_dir / "checkpoint_step0000.pt"
        _save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            step=0,
            seed=spec.seed,
            tape_metadata=tape_metadata,
        )
        checkpoint_records.append(
            {
                "step": 0,
                "relative_path": checkpoint_path.name,
                "sha256": sha256_file(checkpoint_path),
                "model_state_sha256": model_state_sha256(model),
                "optimizer_state_sha256": torch_state_sha256(
                    optimizer.state_dict()
                ),
                "torch_rng_state_sha256": torch_state_sha256(
                    torch.get_rng_state()
                ),
                "python_rng_state_sha256": torch_state_sha256(random.getstate()),
            }
        )

        target_tensor = torch.from_numpy(y_train)
        eval_set = set(eval_steps[1:])
        recovery_set = set(checkpoint_steps[1:])
        for step_index in range(spec.steps):
            with torch.no_grad():
                full_probs = torch.softmax(model(x_train), dim=1)
                p_train = full_probs[
                    torch.arange(TRAIN_SIZE, dtype=torch.long), target_tensor
                ].cpu().numpy()
            q = sampler_probabilities(spec.cell.sampler, p_train)
            selected = inverse_cdf_indices(q, task_tape[step_index])

            selected_inputs = x_train[selected]
            logits = model(selected_inputs)
            action_probs_tensor = torch.softmax(logits, dim=1)
            action_probs = action_probs_tensor.detach().cpu().numpy()
            actions = inverse_cdf_actions(action_probs, action_tape[step_index])
            rewards = (actions == y_train[selected, None]).astype(np.float64)
            weights = estimator_weights(spec.cell.estimator, rewards)

            log_probs = torch.log_softmax(logits, dim=1)
            action_tensor = torch.from_numpy(actions.astype(np.int64, copy=False))
            chosen_log_probs = log_probs.gather(1, action_tensor)
            weight_tensor = torch.from_numpy(weights)
            loss = -(weight_tensor.detach() * chosen_log_probs).sum() / GROUPS_PER_STEP
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = _gradient_norm(model)
            optimizer.step()

            ledger["p_train_by_step"][step_index] = p_train
            ledger["q_by_step"][step_index] = q
            ledger["selected_train_positions"][step_index] = selected
            ledger["selected_original_indices"][step_index] = train_indices[selected]
            ledger["selected_action_probabilities"][step_index] = action_probs
            ledger["actions"][step_index] = actions
            ledger["rewards"][step_index] = rewards.astype(np.uint8)
            ledger["weights"][step_index] = weights
            ledger["group_success_count"][step_index] = rewards.sum(axis=1).astype(np.uint8)
            ledger["group_absolute_mass"][step_index] = np.abs(weights).sum(axis=1)
            ledger["loss"][step_index] = float(loss.detach().item())
            ledger["gradient_norm"][step_index] = grad_norm

            completed_step = step_index + 1
            if completed_step in eval_set:
                record, probabilities = _evaluation_record(
                    model, evaluation_arrays, step=completed_step
                )
                eval_records.append(record)
                for name, probs in probabilities.items():
                    eval_probability_lists[name].append(probs)
            if completed_step in recovery_set:
                checkpoint_path = (
                    spec.output_dir / f"checkpoint_step{completed_step:04d}.pt"
                )
                _save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=completed_step,
                    seed=spec.seed,
                    tape_metadata=tape_metadata,
                )
                checkpoint_records.append(
                    {
                        "step": completed_step,
                        "relative_path": checkpoint_path.name,
                        "sha256": sha256_file(checkpoint_path),
                        "model_state_sha256": model_state_sha256(model),
                        "optimizer_state_sha256": torch_state_sha256(
                            optimizer.state_dict()
                        ),
                        "torch_rng_state_sha256": torch_state_sha256(
                            torch.get_rng_state()
                        ),
                        "python_rng_state_sha256": torch_state_sha256(
                            random.getstate()
                        ),
                    }
                )

        for name, values in eval_probability_lists.items():
            ledger[f"eval_probabilities_{name}"] = np.stack(values, axis=0)
        ledger["eval_steps"] = np.asarray(eval_steps, dtype=np.int32)
        ledger["eval_action_budgets"] = (
            ledger["eval_steps"].astype(np.int64) * GROUPS_PER_STEP * GROUP_SIZE
        )

        ledger_path = spec.output_dir / "ledger.npz"
        np.savez_compressed(ledger_path, **ledger)
        ledger_array_hashes = _npz_payload_hashes(ledger)
        k_values = ledger["group_success_count"]
        regimes = {
            "dead": int(np.count_nonzero(k_values == 0)),
            "mixed": int(np.count_nonzero((k_values > 0) & (k_values < GROUP_SIZE))),
            "all_pass": int(np.count_nonzero(k_values == GROUP_SIZE)),
        }
        selected_p = np.take_along_axis(
            ledger["p_train_by_step"],
            ledger["selected_train_positions"].astype(np.int64),
            axis=1,
        )
        exposure_counts, _ = np.histogram(selected_p, bins=EXPOSURE_BIN_EDGES)
        pool_counts, _ = np.histogram(
            ledger["p_train_by_step"].reshape(-1), bins=EXPOSURE_BIN_EDGES
        )
        primary_split = "test" if spec.include_test else "dev"
        primary_curve = [
            record["splits"][primary_split]["c_k"]["8"] for record in eval_records
        ]
        primary_auc = normalized_auc(ledger["eval_action_budgets"], primary_curve)
        summary = {
            "schema": ARTIFACT_SCHEMA,
            "artifact_state": "complete",
            "phase": spec.phase,
            "canonical_evidence": spec.canonical,
            "cell": {
                "name": spec.cell.name,
                "estimator": spec.cell.estimator,
                "sampler": spec.cell.sampler,
            },
            "logical_seed": spec.seed,
            "learning_rate": spec.learning_rate,
            "steps": spec.steps,
            "primary_outcome": {
                "split": primary_split,
                "c8_normalized_action_auc": primary_auc,
            },
            "provenance": {
                "runtime": runtime,
                "source_lock_relative_path": (
                    "curriculum_maxrl/digits_factorial/SOURCE_LOCK.json"
                ),
                "source_lock_sha256": lock_sha,
                "source_sha256": lock["source_sha256"],
                "data_manifest_relative_path": (
                    "curriculum_maxrl/digits_factorial/digits_split_manifest.json"
                ),
                "data_manifest_sha256": sha256_file(DATA_MANIFEST_PATH),
                "data_array_sha256": data_manifest["array_sha256"],
                "split_index_sha256": data_manifest["index_sha256"],
                "authorization_relative_path": (
                    spec.authorization_path.resolve().relative_to(
                        PROJECT_ROOT.resolve()
                    ).as_posix()
                    if spec.authorization_path is not None
                    else None
                ),
                "authorization_sha256": (
                    sha256_file(spec.authorization_path)
                    if spec.authorization_path is not None
                    else None
                ),
                "lr_selection_relative_path": (
                    spec.lr_selection_path.resolve().relative_to(
                        PROJECT_ROOT.resolve()
                    ).as_posix()
                    if spec.lr_selection_path is not None
                    else None
                ),
                "lr_selection_sha256": (
                    sha256_file(spec.lr_selection_path)
                    if spec.lr_selection_path is not None
                    else None
                ),
            },
            "execution": execution_settings,
            "rng_tapes": tape_metadata,
            "initial_model_state_sha256": initial_model_sha,
            "final_model_state_sha256": model_state_sha256(model),
            "ledger": {
                "relative_path": ledger_path.name,
                "sha256": sha256_file(ledger_path),
                "array_sha256": ledger_array_hashes,
            },
            "recovery_checkpoints": checkpoint_records,
            "evaluation_records": eval_records,
            "accounting": {
                "paid_actions": spec.steps * GROUPS_PER_STEP * GROUP_SIZE,
                "groups": spec.steps * GROUPS_PER_STEP,
                "updates": spec.steps,
                "training_scoring_forward_examples": spec.steps * TRAIN_SIZE,
                "training_action_forward_examples": spec.steps * GROUPS_PER_STEP,
                "evaluation_forward_examples": sum(
                    len(indices) * len(eval_steps)
                    for name, indices in split_indices.items()
                    if name in evaluation_arrays
                ),
                "group_regime_counts": regimes,
                "coefficient_absolute_mass_total": float(
                    ledger["group_absolute_mass"].sum()
                ),
                "nonzero_mass_group_count": int(
                    np.count_nonzero(ledger["group_absolute_mass"] > 0.0)
                ),
            },
            "sampler_exposure": {
                "bin_edges": EXPOSURE_BIN_EDGES.tolist(),
                "selected_group_counts": exposure_counts.tolist(),
                "full_pool_step_counts": pool_counts.tolist(),
                "selected_mean_p": float(selected_p.mean()),
            },
            "failure_ledger": [],
        }
        summary_path = spec.output_dir / "summary.json"
        write_json(summary_path, summary)
        write_json(
            spec.output_dir / "timing.json",
            {
                "schema": "curriculum-maxrl/digits-factorial-unbound-timing/v1",
                "non_evidentiary_unbound_metadata": True,
                "worker_mode": spec.worker_mode,
                "requested_workers": spec.requested_workers,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        return summary_path
    except Exception as error:
        write_json(
            failure_path,
            {
                "schema": "curriculum-maxrl/digits-factorial-failure/v1",
                "artifact_state": "failed",
                "phase": spec.phase,
                "cell": spec.cell.name,
                "seed": spec.seed,
                "learning_rate": spec.learning_rate,
                "steps": spec.steps,
                "exception_type": type(error).__name__,
                "exception": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


def _run_spec_worker(payload: dict[str, Any]) -> str:
    configure_deterministic_cpu()
    spec = RunSpec(
        phase=payload["phase"],
        cell=cell_from_name(payload["cell"]),
        seed=int(payload["seed"]),
        learning_rate=float(payload["learning_rate"]),
        steps=int(payload["steps"]),
        include_test=bool(payload["include_test"]),
        output_dir=Path(payload["output_dir"]),
        authorization_path=(
            Path(payload["authorization_path"])
            if payload.get("authorization_path")
            else None
        ),
        lr_selection_path=(
            Path(payload["lr_selection_path"])
            if payload.get("lr_selection_path")
            else None
        ),
        worker_mode=payload["worker_mode"],
        requested_workers=payload["requested_workers"],
    )
    return str(run_one(spec))


def _payload(spec: RunSpec) -> dict[str, Any]:
    return {
        "phase": spec.phase,
        "cell": spec.cell.name,
        "seed": spec.seed,
        "learning_rate": spec.learning_rate,
        "steps": spec.steps,
        "include_test": spec.include_test,
        "output_dir": str(spec.output_dir),
        "authorization_path": str(spec.authorization_path) if spec.authorization_path else None,
        "lr_selection_path": str(spec.lr_selection_path) if spec.lr_selection_path else None,
        "worker_mode": spec.worker_mode,
        "requested_workers": spec.requested_workers,
    }


def run_specs(specs: Iterable[RunSpec], *, workers: int) -> list[Path]:
    configure_deterministic_cpu()
    if workers < 1:
        raise ValueError("workers must be positive")
    worker_mode = "serial" if workers == 1 else "process_pool_worker"
    materialized = [
        replace(spec, worker_mode=worker_mode, requested_workers=workers)
        for spec in specs
    ]
    if workers <= 1:
        return [run_one(spec) for spec in materialized]
    outputs: list[Path] = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = {executor.submit(_run_spec_worker, _payload(spec)): spec for spec in materialized}
        for future in as_completed(futures):
            outputs.append(Path(future.result()))
    return sorted(outputs)


def development_specs(output_root: Path, authorization_path: Path) -> list[RunSpec]:
    return [
        RunSpec(
            phase="development",
            cell=Cell(estimator, sampler),
            seed=seed,
            learning_rate=learning_rate,
            steps=N_STEPS,
            include_test=False,
            output_dir=(
                output_root
                / f"lr_{learning_rate:g}"
                / f"seed_{seed}"
                / Cell(estimator, sampler).name
            ),
            authorization_path=authorization_path,
        )
        for learning_rate in DEVELOPMENT_LRS
        for seed in DEVELOPMENT_SEEDS
        for estimator, sampler in CELLS
    ]


def confirmation_specs(
    output_root: Path,
    authorization_path: Path,
    lr_selection_path: Path,
    *,
    phase: str,
) -> list[RunSpec]:
    selection = _read_lr_selection(lr_selection_path)
    return [
        RunSpec(
            phase=phase,
            cell=Cell(estimator, sampler),
            seed=seed,
            learning_rate=_expected_confirmation_lr(
                phase, Cell(estimator, sampler), selection
            ),
            steps=N_STEPS,
            include_test=True,
            output_dir=output_root / f"seed_{seed}" / Cell(estimator, sampler).name,
            authorization_path=authorization_path,
            lr_selection_path=lr_selection_path,
        )
        for seed in CONFIRMATION_SEEDS
        for estimator, sampler in CELLS
    ]


def _main_run_cell(args: argparse.Namespace) -> None:
    spec = RunSpec(
        phase=args.phase,
        cell=cell_from_name(args.cell),
        seed=args.seed,
        learning_rate=args.learning_rate,
        steps=args.steps,
        include_test=args.include_test,
        output_dir=args.output,
        authorization_path=args.authorization,
        lr_selection_path=args.lr_selection,
    )
    print(run_one(spec))


def _main_engineering_suite(args: argparse.Namespace) -> None:
    specs = [
        RunSpec(
            phase="engineering",
            cell=Cell(estimator, sampler),
            seed=ENGINEERING_SEED,
            learning_rate=args.learning_rate,
            steps=args.steps,
            include_test=False,
            output_dir=args.output / Cell(estimator, sampler).name,
        )
        for estimator, sampler in CELLS
    ]
    for output in run_specs(specs, workers=args.workers):
        print(output)


def _main_schedule(args: argparse.Namespace) -> None:
    if args.phase == "development":
        specs = development_specs(args.output, args.authorization)
    else:
        specs = confirmation_specs(
            args.output,
            args.authorization,
            args.lr_selection,
            phase=args.phase,
        )
    for output in run_specs(specs, workers=args.workers):
        print(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    cell = subparsers.add_parser("run-cell")
    cell.add_argument("--phase", required=True)
    cell.add_argument("--cell", required=True)
    cell.add_argument("--seed", type=int, required=True)
    cell.add_argument("--learning-rate", type=float, required=True)
    cell.add_argument("--steps", type=int, required=True)
    cell.add_argument("--include-test", action="store_true")
    cell.add_argument("--output", type=Path, required=True)
    cell.add_argument("--authorization", type=Path)
    cell.add_argument("--lr-selection", type=Path)
    cell.set_defaults(function=_main_run_cell)

    engineering = subparsers.add_parser("engineering-suite")
    engineering.add_argument("--steps", type=int, default=4)
    engineering.add_argument("--learning-rate", type=float, default=0.0)
    engineering.add_argument("--workers", type=int, default=1)
    engineering.add_argument("--output", type=Path, required=True)
    engineering.set_defaults(function=_main_engineering_suite)

    schedule = subparsers.add_parser("run-schedule")
    schedule.add_argument(
        "--phase",
        choices=("development", "confirmation_tuned", "confirmation_common"),
        required=True,
    )
    schedule.add_argument("--authorization", type=Path, required=True)
    schedule.add_argument("--lr-selection", type=Path)
    schedule.add_argument("--workers", type=int, default=1)
    schedule.add_argument("--output", type=Path, required=True)
    schedule.set_defaults(function=_main_schedule)
    return parser


def main() -> None:
    configure_deterministic_cpu()
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
