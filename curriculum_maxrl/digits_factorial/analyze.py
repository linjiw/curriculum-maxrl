"""Independent ledger reconstruction and registered Digits-factorial analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .core import (
    ACTION_BUDGET,
    ANALYSIS_SCHEMA,
    ARTIFACT_SCHEMA,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CELLS,
    CHECKPOINT_SCHEMA,
    CONFIRMATION_SEEDS,
    DATA_MANIFEST_PATH,
    DEVELOPMENT_LRS,
    DEVELOPMENT_SEEDS,
    EVAL_INTERVAL_STEPS,
    ENGINEERING_AUDIT_SCHEMA,
    EXPECTED_RUNTIME,
    EXPOSURE_BIN_EDGES,
    GROUP_SIZE,
    GROUPS_PER_STEP,
    N_CLASSES,
    N_STEPS,
    PARAMETER_COUNT,
    PROJECT_ROOT,
    RECOVERY_STEPS,
    SOURCE_LOCK_PATH,
    TRAIN_SIZE,
    Cell,
    configure_deterministic_cpu,
    domain_seed,
    generate_rng_tapes,
    initialize_model,
    json_ready,
    load_stored_digits,
    normalized_auc,
    sha256_array,
    sha256_file,
    strict_json_load,
    torch_state_sha256,
    write_json,
)
from .locking import (
    load_and_verify_source_lock,
    validate_engineering_audit_binding,
    validate_lr_selection_document,
    verify_execution_authorization,
)


from .core import LR_SELECTION_SCHEMA


VALIDATION_SCHEMA = "curriculum-maxrl/digits-factorial-run-validation/v2"


def _assert_close(observed: Any, expected: Any, *, label: str, atol: float = 1e-12) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise ValueError(f"{label}: mapping keys differ")
        for key in expected:
            _assert_close(observed[key], expected[key], label=f"{label}.{key}", atol=atol)
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"{label}: list shapes differ")
        for index, (got, want) in enumerate(zip(observed, expected)):
            _assert_close(got, want, label=f"{label}[{index}]", atol=atol)
        return
    if type(expected) is float:
        if type(observed) is not float or not math.isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=atol
        ):
            raise ValueError(f"{label}: {observed} != {expected}")
        return
    if type(expected) is int:
        if type(observed) is not int or observed != expected:
            raise ValueError(f"{label}: {observed} != {expected}")
        return
    if type(expected) is bool:
        if type(observed) is not bool or observed is not expected:
            raise ValueError(f"{label}: {observed} != {expected}")
        return
    if observed != expected:
        raise ValueError(f"{label}: {observed} != {expected}")


def _independent_score(sampler: str, p: np.ndarray) -> np.ndarray:
    if sampler == "uniform":
        return np.ones_like(p)
    if sampler == "p1mp":
        return p * (1.0 - p)
    if sampler == "u8":
        return np.maximum(0.0, 1.0 - np.power(1.0 - p, 8) - p)
    raise ValueError(f"unknown sampler: {sampler}")


def _independent_q(sampler: str, p: np.ndarray) -> np.ndarray:
    score = _independent_score(sampler, p)
    if sampler == "uniform":
        return np.full_like(score, 1.0 / score.shape[1])
    total = score.sum(axis=1, dtype=np.float64)
    q = np.empty_like(score)
    uniform = np.full(score.shape[1], 1.0 / score.shape[1], dtype=np.float64)
    for step in range(score.shape[0]):
        if not math.isfinite(float(total[step])) or total[step] <= 0.0:
            q[step] = uniform
        else:
            q[step] = 0.1 / score.shape[1] + 0.9 * score[step] / total[step]
            q[step] /= q[step].sum(dtype=np.float64)
    return q


def _independent_weights(estimator: str, rewards: np.ndarray) -> np.ndarray:
    r = rewards.astype(np.float64, copy=False)
    k = r.sum(axis=2, keepdims=True)
    if estimator == "practical_maxrl":
        safe = np.where(k > 0.0, k, 1.0)
        return np.where(k > 0.0, r / safe - 1.0 / 8.0, 0.0)
    if estimator == "rloo":
        return (r - (k - r) / 7.0) / 8.0
    raise ValueError(f"unknown estimator: {estimator}")


def _independent_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    p_y = p[np.arange(len(y)), y]
    predictions = p.argmax(axis=1)
    correct = predictions == y
    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(y)), y] = 1.0
    class_accuracy = [float(correct[y == label].mean()) for label in range(10)]
    class_mean_p = [float(p_y[y == label].mean()) for label in range(10)]
    return {
        "mean_p_y": float(p_y.mean()),
        "c_k": {
            str(k): float(np.mean(1.0 - np.power(1.0 - p_y, k), dtype=np.float64))
            for k in (1, 2, 4, 8, 16, 32)
        },
        "nll": float(-np.log(np.clip(p_y, np.finfo(np.float64).tiny, 1.0)).mean()),
        "brier": float(np.square(p - one_hot).sum(axis=1).mean()),
        "top1_accuracy": float(correct.mean()),
        "macro_class_accuracy": float(np.mean(class_accuracy)),
        "macro_class_mean_p_y": float(np.mean(class_mean_p)),
        "per_class_accuracy": class_accuracy,
        "per_class_mean_p_y": class_mean_p,
    }


def _inverse_rows(q: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    result = np.empty(uniforms.shape, dtype=np.int64)
    for step in range(len(q)):
        cdf = np.cumsum(q[step], dtype=np.float64)
        cdf[-1] = 1.0
        result[step] = np.searchsorted(cdf, uniforms[step], side="right")
    return result


def _independent_actions(probabilities: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(probabilities, axis=2, dtype=np.float64)
    cdf[:, :, -1] = 1.0
    actions = (uniforms[:, :, :, None] >= cdf[:, :, None, :]).sum(axis=3)
    return np.minimum(actions, N_CLASSES - 1).astype(np.uint8)


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_array(tensor.detach().cpu().numpy())))
    return digest.hexdigest()


def _assert_finite_json_tree(value: Any, *, location: str = "root") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{location}: non-finite JSON number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json_tree(item, location=f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location}: non-string JSON key")
            _assert_finite_json_tree(item, location=f"{location}.{key}")
        return
    raise ValueError(f"{location}: unsupported JSON value type {type(value)}")


def _exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} has an unexpected field set")
    return value


def _require_exact_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an exact JSON integer")
    return value


def _require_exact_float(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be an exact finite JSON float")
    return value


def _safe_run_member(run_dir: Path, relative: object, *, expected: str, label: str) -> Path:
    if not isinstance(relative, str) or relative != expected or Path(relative).name != relative:
        raise ValueError(f"{label} path is not the exact registered filename")
    path = (run_dir / relative).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as error:
        raise ValueError(f"{label} path escapes run directory") from error
    if not path.is_file():
        raise ValueError(f"{label} is missing")
    return path


def _safe_project_provenance_path(relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{label} is not a canonical project-relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or any(part in ("", ".", "..") for part in candidate.parts):
        raise ValueError(f"{label} is not a canonical project-relative path")
    path = (PROJECT_ROOT / candidate).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes project root") from error
    if not path.is_file():
        raise ValueError(f"{label} is missing")
    return path


def _strict_npz_load(path: Path, expected_keys: set[str]) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = [item.filename for item in archive.infolist()]
    except zipfile.BadZipFile as error:
        raise ValueError("ledger is not a valid NPZ archive") from error
    if len(members) != len(set(members)):
        raise ValueError("ledger NPZ contains duplicate member names")
    expected_members = {f"{key}.npy" for key in expected_keys}
    if set(members) != expected_members or any("/" in member or "\\" in member for member in members):
        raise ValueError("ledger NPZ member set is not exact")
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != expected_keys:
            raise ValueError("ledger array key set is not exact and unique")
        arrays = {name: archive[name] for name in archive.files}
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise ValueError(f"ledger array uses object dtype: {name}")
        if np.issubdtype(array.dtype, np.floating) and np.any(~np.isfinite(array)):
            raise ValueError(f"ledger array contains non-finite values: {name}")
    return arrays


MODEL_STATE_SPECS = {
    "linear1.weight": ((64, 64), torch.float64),
    "linear1.bias": ((64,), torch.float64),
    "linear2.weight": ((10, 64), torch.float64),
    "linear2.bias": ((10,), torch.float64),
}


def _validate_model_state(state: object, *, label: str) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict) or set(state) != set(MODEL_STATE_SPECS):
        raise ValueError(f"{label} model state has wrong parameter keys")
    for name, (shape, dtype) in MODEL_STATE_SPECS.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"{label} model tensor shape/dtype mismatch: {name}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{label} model tensor contains non-finite values: {name}")
    return state


def _validate_optimizer_state(
    state: object, *, learning_rate: float, step: int, label: str
) -> dict[str, Any]:
    top = _exact_keys(state, {"state", "param_groups"}, label=f"{label} optimizer")
    groups = top["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError(f"{label} optimizer must have one parameter group")
    group = _exact_keys(
        groups[0],
        {
            "lr",
            "momentum",
            "dampening",
            "weight_decay",
            "nesterov",
            "maximize",
            "foreach",
            "differentiable",
            "fused",
            "params",
        },
        label=f"{label} optimizer parameter group",
    )
    expected_scalars = {
        "lr": learning_rate,
        "momentum": 0.9,
        "dampening": 0,
        "weight_decay": 0.0,
        "nesterov": False,
        "maximize": False,
        "foreach": None,
        "differentiable": False,
        "fused": None,
        "params": [0, 1, 2, 3],
    }
    if group != expected_scalars or any(
        type(group[key]) is not type(expected_scalars[key])
        for key in expected_scalars
        if key != "params"
    ) or type(group["params"]) is not list or any(
        type(value) is not int for value in group["params"]
    ):
        raise ValueError(f"{label} optimizer hyperparameters/parameter IDs differ")
    optimizer_state = top["state"]
    if not isinstance(optimizer_state, dict):
        raise ValueError(f"{label} optimizer state is not a mapping")
    expected_ids = set() if step == 0 else {0, 1, 2, 3}
    if set(optimizer_state) != expected_ids:
        raise ValueError(f"{label} optimizer state IDs differ")
    shapes = ((64, 64), (64,), (10, 64), (10,))
    for parameter_id, shape in enumerate(shapes):
        if step == 0:
            break
        item = optimizer_state[parameter_id]
        if not isinstance(item, dict) or set(item) != {"momentum_buffer"}:
            raise ValueError(f"{label} optimizer buffer schema differs")
        buffer = item["momentum_buffer"]
        if not isinstance(buffer, torch.Tensor) or tuple(buffer.shape) != shape or buffer.dtype != torch.float64:
            raise ValueError(f"{label} optimizer buffer shape/dtype differs")
        if not bool(torch.isfinite(buffer).all()):
            raise ValueError(f"{label} optimizer buffer is non-finite")
    return top


def _exact_tensor_tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(_exact_tensor_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_exact_tensor_tree_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _load_safe_checkpoint(
    path: Path,
    *,
    expected_step: int,
    seed: int,
    learning_rate: float,
    tape_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema",
        "step",
        "action_budget",
        "logical_seed",
        "model_state_dict",
        "optimizer_state_dict",
        "torch_cpu_rng_state",
        "python_random_state",
        "numpy_tape_terminal_states",
        "numpy_tape_sha256",
    }
    checkpoint = _exact_keys(checkpoint, required, label="recovery checkpoint")
    if checkpoint["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("recovery checkpoint schema mismatch")
    if type(checkpoint["step"]) is not int or checkpoint["step"] != expected_step:
        raise ValueError("recovery checkpoint step mismatch")
    if type(checkpoint["action_budget"]) is not int or checkpoint["action_budget"] != expected_step * 512:
        raise ValueError("recovery checkpoint action budget mismatch")
    if type(checkpoint["logical_seed"]) is not int or checkpoint["logical_seed"] != seed:
        raise ValueError("recovery checkpoint logical seed mismatch")
    _validate_model_state(checkpoint["model_state_dict"], label="checkpoint")
    _validate_optimizer_state(
        checkpoint["optimizer_state_dict"],
        learning_rate=learning_rate,
        step=expected_step,
        label="checkpoint",
    )
    rng = checkpoint["torch_cpu_rng_state"]
    if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1 or rng.numel() != 5056:
        raise ValueError("recovery checkpoint torch RNG shape/dtype mismatch")
    python_state = checkpoint["python_random_state"]
    if (
        not isinstance(python_state, tuple)
        or len(python_state) != 3
        or type(python_state[0]) is not int
        or python_state[0] != 3
        or not isinstance(python_state[1], tuple)
        or len(python_state[1]) != 625
        or any(type(value) is not int for value in python_state[1])
        or python_state[2] is not None
    ):
        raise ValueError("recovery checkpoint Python RNG schema differs")
    if checkpoint["numpy_tape_terminal_states"] != tape_metadata["terminal_states"]:
        raise ValueError("recovery checkpoint NumPy tape terminal states mismatch")
    if checkpoint["numpy_tape_sha256"] != tape_metadata["full_tape_sha256"]:
        raise ValueError("recovery checkpoint NumPy tape hashes mismatch")
    return checkpoint


def validate_run(summary_path: Path, *, check_live_lock: bool = True) -> dict[str, Any]:
    """Fail-closed validation by deterministic learner/optimizer replay."""

    execution_expected = configure_deterministic_cpu()
    summary = strict_json_load(summary_path)
    _assert_finite_json_tree(summary)
    summary = _exact_keys(
        summary,
        {
            "schema",
            "artifact_state",
            "phase",
            "canonical_evidence",
            "cell",
            "logical_seed",
            "learning_rate",
            "steps",
            "primary_outcome",
            "provenance",
            "execution",
            "rng_tapes",
            "initial_model_state_sha256",
            "final_model_state_sha256",
            "ledger",
            "recovery_checkpoints",
            "evaluation_records",
            "accounting",
            "sampler_exposure",
            "failure_ledger",
        },
        label="run summary",
    )
    if summary["schema"] != ARTIFACT_SCHEMA or summary["artifact_state"] != "complete":
        raise ValueError("incomplete or wrong-schema run")
    if summary["failure_ledger"] != []:
        raise ValueError("completed artifact contains a failure entry")
    phase = summary["phase"]
    if phase not in {"engineering", "development", "confirmation_tuned", "confirmation_common"}:
        raise ValueError("unregistered run phase")
    canonical = phase != "engineering"
    if type(summary["canonical_evidence"]) is not bool or summary["canonical_evidence"] != canonical:
        raise ValueError("canonical-evidence flag mismatch")
    steps = _require_exact_int(summary["steps"], label="steps")
    seed = _require_exact_int(summary["logical_seed"], label="logical_seed")
    learning_rate = _require_exact_float(summary["learning_rate"], label="learning_rate")
    cell_payload = _exact_keys(summary["cell"], {"name", "estimator", "sampler"}, label="cell")
    cell = Cell(cell_payload["estimator"], cell_payload["sampler"])
    if cell_payload["name"] != cell.name:
        raise ValueError("cell name does not match estimator/sampler")
    if phase == "engineering":
        if seed != 33000 or not 1 <= steps < N_STEPS or learning_rate != 0.0:
            raise ValueError("engineering schedule must be reserved-seed truncated zero-LR")
    elif phase == "development":
        if seed not in DEVELOPMENT_SEEDS or steps != N_STEPS or learning_rate not in DEVELOPMENT_LRS:
            raise ValueError("development schedule mismatch")
    elif seed not in CONFIRMATION_SEEDS or steps != N_STEPS:
        raise ValueError("confirmation schedule mismatch")

    lock, lock_sha = load_and_verify_source_lock(check_runtime=check_live_lock)
    provenance = _exact_keys(
        summary["provenance"],
        {
            "runtime",
            "source_lock_relative_path",
            "source_lock_sha256",
            "source_sha256",
            "data_manifest_relative_path",
            "data_manifest_sha256",
            "data_array_sha256",
            "split_index_sha256",
            "authorization_relative_path",
            "authorization_sha256",
            "lr_selection_relative_path",
            "lr_selection_sha256",
        },
        label="provenance",
    )
    runtime = _exact_keys(
        provenance["runtime"],
        {
            "python_implementation",
            "python",
            "numpy",
            "scipy",
            "torch",
            "scikit_learn",
            "platform",
            "machine",
            "executable",
        },
        label="recorded runtime",
    )
    if {key: runtime[key] for key in EXPECTED_RUNTIME} != EXPECTED_RUNTIME or not all(
        isinstance(value, str) for value in runtime.values()
    ):
        raise ValueError("recorded execution runtime differs from pinned runtime")
    if provenance["source_lock_relative_path"] != "curriculum_maxrl/digits_factorial/SOURCE_LOCK.json":
        raise ValueError("noncanonical source-lock provenance path")
    if provenance["source_lock_sha256"] != lock_sha or provenance["source_sha256"] != lock["source_sha256"]:
        raise ValueError("run source provenance differs from live lock")
    if provenance["data_manifest_relative_path"] != "curriculum_maxrl/digits_factorial/digits_split_manifest.json":
        raise ValueError("noncanonical data-manifest path")
    if provenance["data_manifest_sha256"] != sha256_file(DATA_MANIFEST_PATH):
        raise ValueError("data-manifest SHA mismatch")
    if summary["execution"] != execution_expected:
        raise ValueError("thread/deterministic execution provenance mismatch")

    if phase == "engineering":
        if any(
            provenance[key] is not None
            for key in (
                "authorization_relative_path",
                "authorization_sha256",
                "lr_selection_relative_path",
                "lr_selection_sha256",
            )
        ):
            raise ValueError("engineering run contains authorization/LR-selection provenance")
    else:
        auth_relative = provenance["authorization_relative_path"]
        if not isinstance(auth_relative, str):
            raise ValueError("canonical run lacks authorization path")
        auth_path = _safe_project_provenance_path(
            auth_relative, label="authorization provenance path"
        )
        if sha256_file(auth_path) != provenance["authorization_sha256"]:
            raise ValueError("canonical run authorization SHA mismatch")
        selection_path: Path | None = None
        if phase.startswith("confirmation"):
            lr_relative = provenance["lr_selection_relative_path"]
            if not isinstance(lr_relative, str):
                raise ValueError("confirmation lacks LR-selection path")
            selection_path = _safe_project_provenance_path(
                lr_relative, label="LR-selection provenance path"
            )
            if sha256_file(selection_path) != provenance["lr_selection_sha256"]:
                raise ValueError("confirmation LR-selection SHA mismatch")
        elif provenance["lr_selection_relative_path"] is not None or provenance["lr_selection_sha256"] is not None:
            raise ValueError("development run contains LR-selection provenance")
        verify_execution_authorization(
            auth_path,
            phase=phase,
            lock_sha256=lock_sha,
            lr_selection_path=selection_path,
        )
        if selection_path is not None:
            selection = strict_json_load(selection_path)
            validate_lr_selection_document(selection, lock_sha256=lock_sha)
            expected_lr = (
                float(selection["selected_learning_rates_by_estimator"][cell.estimator])
                if phase == "confirmation_tuned"
                else float(selection["selected_common_learning_rate"])
            )
            if learning_rate != expected_lr:
                raise ValueError("confirmation learning rate differs from frozen selection")

    x, y, splits, data_manifest = load_stored_digits()
    if provenance["data_array_sha256"] != data_manifest["array_sha256"] or provenance[
        "split_index_sha256"
    ] != data_manifest["index_sha256"]:
        raise ValueError("data/split hashes differ from frozen manifest")
    train_targets = y[splits["train"]]
    expected_eval_splits = ("train", "dev", "test") if phase.startswith("confirmation") else ("train", "dev")
    primary_split = "test" if phase.startswith("confirmation") else "dev"
    primary = _exact_keys(summary["primary_outcome"], {"split", "c8_normalized_action_auc"}, label="primary outcome")
    if primary["split"] != primary_split:
        raise ValueError("primary outcome split mismatch")
    _require_exact_float(primary["c8_normalized_action_auc"], label="primary C8 AUC")

    expected_eval_steps = [0, *range(EVAL_INTERVAL_STEPS, steps + 1, EVAL_INTERVAL_STEPS)]
    if expected_eval_steps[-1] != steps:
        expected_eval_steps.append(steps)
    n_eval = len(expected_eval_steps)
    base_specs: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
        "p_train_by_step": ((steps, TRAIN_SIZE), np.dtype("float64")),
        "q_by_step": ((steps, TRAIN_SIZE), np.dtype("float64")),
        "selected_train_positions": ((steps, 64), np.dtype("int32")),
        "selected_original_indices": ((steps, 64), np.dtype("int32")),
        "selected_action_probabilities": ((steps, 64, 10), np.dtype("float64")),
        "actions": ((steps, 64, 8), np.dtype("uint8")),
        "rewards": ((steps, 64, 8), np.dtype("uint8")),
        "weights": ((steps, 64, 8), np.dtype("float64")),
        "group_success_count": ((steps, 64), np.dtype("uint8")),
        "group_absolute_mass": ((steps, 64), np.dtype("float64")),
        "loss": ((steps,), np.dtype("float64")),
        "gradient_norm": ((steps,), np.dtype("float64")),
        "eval_steps": ((n_eval,), np.dtype("int32")),
        "eval_action_budgets": ((n_eval,), np.dtype("int64")),
    }
    for split in expected_eval_splits:
        base_specs[f"eval_probabilities_{split}"] = (
            (n_eval, len(splits[split]), 10),
            np.dtype("float64"),
        )
    ledger_record = _exact_keys(summary["ledger"], {"relative_path", "sha256", "array_sha256"}, label="ledger record")
    ledger_path = _safe_run_member(
        summary_path.parent,
        ledger_record["relative_path"],
        expected="ledger.npz",
        label="ledger",
    )
    if sha256_file(ledger_path) != ledger_record["sha256"]:
        raise ValueError("ledger file SHA mismatch")
    ledger = _strict_npz_load(ledger_path, set(base_specs))
    if ledger_record["array_sha256"] != {
        name: sha256_array(array) for name, array in sorted(ledger.items())
    }:
        raise ValueError("ledger array SHA manifest mismatch")
    for name, (shape, dtype) in base_specs.items():
        if ledger[name].shape != shape or ledger[name].dtype != dtype:
            raise ValueError(f"ledger shape/dtype mismatch: {name}")
    if np.any((ledger["p_train_by_step"] < 0.0) | (ledger["p_train_by_step"] > 1.0)):
        raise ValueError("p_train contains values outside [0,1]")
    selected_probs = ledger["selected_action_probabilities"]
    if np.any(selected_probs < 0.0) or np.any(selected_probs > 1.0) or not np.allclose(
        selected_probs.sum(axis=2), 1.0, rtol=0.0, atol=2e-15
    ):
        raise ValueError("selected categorical probabilities are not valid simplex rows")

    task_tape, action_tape, tape_metadata = generate_rng_tapes(seed, steps=steps)
    if summary["rng_tapes"] != tape_metadata:
        raise ValueError("RNG tape metadata differs from regeneration")
    expected_recovery = list(RECOVERY_STEPS) if steps == N_STEPS else [0, steps]
    records = summary["recovery_checkpoints"]
    if not isinstance(records, list) or len(records) != len(expected_recovery):
        raise ValueError("recovery checkpoint record count mismatch")
    checkpoint_payloads: dict[int, dict[str, Any]] = {}
    checkpoint_files: set[str] = set()
    for expected_step, record_value in zip(expected_recovery, records):
        record = _exact_keys(
            record_value,
            {
                "step",
                "relative_path",
                "sha256",
                "model_state_sha256",
                "optimizer_state_sha256",
                "torch_rng_state_sha256",
                "python_rng_state_sha256",
            },
            label="checkpoint record",
        )
        if type(record["step"]) is not int or record["step"] != expected_step:
            raise ValueError("checkpoint record step mismatch")
        filename = f"checkpoint_step{expected_step:04d}.pt"
        checkpoint_path = _safe_run_member(
            summary_path.parent, record["relative_path"], expected=filename, label="checkpoint"
        )
        checkpoint_files.add(filename)
        if sha256_file(checkpoint_path) != record["sha256"]:
            raise ValueError("checkpoint file SHA mismatch")
        checkpoint = _load_safe_checkpoint(
            checkpoint_path,
            expected_step=expected_step,
            seed=seed,
            learning_rate=learning_rate,
            tape_metadata=tape_metadata,
        )
        if _state_dict_sha256(checkpoint["model_state_dict"]) != record["model_state_sha256"]:
            raise ValueError("checkpoint model digest mismatch")
        if torch_state_sha256(checkpoint["optimizer_state_dict"]) != record[
            "optimizer_state_sha256"
        ]:
            raise ValueError("checkpoint optimizer digest mismatch")
        if torch_state_sha256(checkpoint["torch_cpu_rng_state"]) != record[
            "torch_rng_state_sha256"
        ]:
            raise ValueError("checkpoint RNG digest mismatch")
        if torch_state_sha256(checkpoint["python_random_state"]) != record[
            "python_rng_state_sha256"
        ]:
            raise ValueError("checkpoint Python RNG digest mismatch")
        checkpoint_payloads[expected_step] = checkpoint

    expected_run_files = {"summary.json", "ledger.npz", "timing.json", *checkpoint_files}
    observed_run_files = {path.name for path in summary_path.parent.iterdir() if path.is_file()}
    observed_subdirs = [path for path in summary_path.parent.iterdir() if path.is_dir()]
    if observed_run_files != expected_run_files or observed_subdirs:
        raise ValueError("completed run directory has extra/missing artifacts")
    timing = strict_json_load(summary_path.parent / "timing.json")
    _assert_finite_json_tree(timing, location="timing")
    timing = _exact_keys(
        timing,
        {
            "schema",
            "non_evidentiary_unbound_metadata",
            "worker_mode",
            "requested_workers",
            "wall_seconds",
        },
        label="timing record",
    )
    if timing["schema"] != "curriculum-maxrl/digits-factorial-unbound-timing/v1" or timing[
        "non_evidentiary_unbound_metadata"
    ] is not True or type(timing["wall_seconds"]) is not float or timing["wall_seconds"] <= 0.0:
        raise ValueError("invalid unbound timing record")
    if timing["worker_mode"] not in {"direct", "serial", "process_pool_worker"}:
        raise ValueError("invalid timing worker mode")
    if type(timing["requested_workers"]) is not int or timing["requested_workers"] < 1:
        raise ValueError("invalid timing worker count")
    if timing["worker_mode"] in {"direct", "serial"} and timing["requested_workers"] != 1:
        raise ValueError("invalid direct/serial worker provenance")
    if timing["worker_mode"] == "process_pool_worker" and timing["requested_workers"] < 2:
        raise ValueError("invalid process-pool worker provenance")

    # Deterministic, full learner/SGD replay. Nothing scientific is trusted.
    torch.manual_seed(domain_seed(seed, "torch-global") % (2**63 - 1))
    random.seed(domain_seed(seed, "python-global"))
    model = initialize_model(seed)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0
    )
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise ValueError("replay model parameter count mismatch")
    initial_hash = _state_dict_sha256(model.state_dict())
    if initial_hash != summary["initial_model_state_sha256"]:
        raise ValueError("stored initialization differs from deterministic initialization")

    x_train = torch.from_numpy(x[splits["train"]])
    train_target_tensor = torch.from_numpy(train_targets)
    eval_inputs = {split: torch.from_numpy(x[splits[split]]) for split in expected_eval_splits}

    def compare_checkpoint(step: int) -> None:
        checkpoint = checkpoint_payloads[step]
        if not _exact_tensor_tree_equal(checkpoint["model_state_dict"], model.state_dict()):
            raise ValueError(f"checkpoint model differs from replay at step {step}")
        if not _exact_tensor_tree_equal(checkpoint["optimizer_state_dict"], optimizer.state_dict()):
            raise ValueError(f"checkpoint optimizer differs from replay at step {step}")
        if not torch.equal(checkpoint["torch_cpu_rng_state"], torch.get_rng_state()):
            raise ValueError(f"checkpoint torch RNG differs from replay at step {step}")
        if checkpoint["python_random_state"] != random.getstate():
            raise ValueError(f"checkpoint Python RNG differs from replay at step {step}")

    def replay_eval(eval_index: int, step: int) -> dict[str, Any]:
        record: dict[str, Any] = {"step": step, "action_budget": step * 512, "splits": {}}
        with torch.no_grad():
            for split in expected_eval_splits:
                probabilities = torch.softmax(model(eval_inputs[split]), dim=1).cpu().numpy()
                if not np.array_equal(probabilities, ledger[f"eval_probabilities_{split}"][eval_index]):
                    raise ValueError(f"evaluation probabilities differ from replay: {split}, step {step}")
                record["splits"][split] = _independent_metrics(probabilities, y[splits[split]])
        return record

    compare_checkpoint(0)
    reconstructed_records = [replay_eval(0, 0)]
    eval_index = 1
    recovery_set = set(expected_recovery[1:])
    eval_set = set(expected_eval_steps[1:])
    for step_index in range(steps):
        with torch.no_grad():
            full_probs = torch.softmax(model(x_train), dim=1)
            replay_p = full_probs[
                torch.arange(TRAIN_SIZE, dtype=torch.long), train_target_tensor
            ].cpu().numpy()
        if not np.array_equal(replay_p, ledger["p_train_by_step"][step_index]):
            raise ValueError(f"p_train differs from learner replay at step {step_index}")
        replay_q = _independent_q(cell.sampler, replay_p[None, :])[0]
        if not np.array_equal(replay_q, ledger["q_by_step"][step_index]):
            raise ValueError(f"q differs from learner replay at step {step_index}")
        replay_selected = _inverse_rows(replay_q[None, :], task_tape[step_index : step_index + 1])[0]
        if not np.array_equal(replay_selected, ledger["selected_train_positions"][step_index]):
            raise ValueError(f"selected IDs differ from learner replay at step {step_index}")
        if not np.array_equal(
            splits["train"][replay_selected], ledger["selected_original_indices"][step_index]
        ):
            raise ValueError("selected original IDs differ from replay")
        logits = model(x_train[replay_selected])
        replay_action_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        if not np.array_equal(
            replay_action_probs, ledger["selected_action_probabilities"][step_index]
        ):
            raise ValueError(f"selected action probabilities differ from replay at step {step_index}")
        selected_correct = replay_action_probs[
            np.arange(64), train_targets[replay_selected]
        ]
        if not np.allclose(selected_correct, replay_p[replay_selected], rtol=0.0, atol=2e-15):
            raise ValueError("selected correct-class probabilities are not tied to p_train")
        replay_actions = _independent_actions(
            replay_action_probs[None, :, :], action_tape[step_index : step_index + 1]
        )[0]
        if not np.array_equal(replay_actions, ledger["actions"][step_index]):
            raise ValueError(f"actions differ from learner replay at step {step_index}")
        replay_rewards = (
            replay_actions == train_targets[replay_selected, None]
        ).astype(np.uint8)
        if not np.array_equal(replay_rewards, ledger["rewards"][step_index]):
            raise ValueError(f"rewards differ from learner replay at step {step_index}")
        replay_weights = _independent_weights(
            cell.estimator, replay_rewards[None, :, :]
        )[0]
        if not np.array_equal(replay_weights, ledger["weights"][step_index]):
            raise ValueError(f"weights differ from learner replay at step {step_index}")
        replay_k = replay_rewards.sum(axis=1).astype(np.uint8)
        replay_mass = np.abs(replay_weights).sum(axis=1)
        if not np.array_equal(replay_k, ledger["group_success_count"][step_index]) or not np.array_equal(
            replay_mass, ledger["group_absolute_mass"][step_index]
        ):
            raise ValueError("success-count/mass ledger differs from replay")
        log_probs = torch.log_softmax(logits, dim=1)
        chosen = log_probs.gather(1, torch.from_numpy(replay_actions.astype(np.int64)))
        loss = -(torch.from_numpy(replay_weights).detach() * chosen).sum() / 64
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_squared = sum(
            float(torch.square(parameter.grad.detach()).sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        grad_norm = math.sqrt(grad_squared)
        if float(loss.detach().item()) != ledger["loss"][step_index] or grad_norm != ledger[
            "gradient_norm"
        ][step_index]:
            raise ValueError(f"loss/gradient norm differs from replay at step {step_index}")
        optimizer.step()
        completed = step_index + 1
        if completed in eval_set:
            reconstructed_records.append(replay_eval(eval_index, completed))
            eval_index += 1
        if completed in recovery_set:
            compare_checkpoint(completed)

    if _state_dict_sha256(model.state_dict()) != summary["final_model_state_sha256"]:
        raise ValueError("final model state differs from replay")
    if not np.array_equal(ledger["eval_steps"], np.asarray(expected_eval_steps, dtype=np.int32)):
        raise ValueError("evaluation step grid mismatch")
    expected_budgets = np.asarray(expected_eval_steps, dtype=np.int64) * 512
    if not np.array_equal(ledger["eval_action_budgets"], expected_budgets):
        raise ValueError("evaluation action-budget grid mismatch")
    if len(summary["evaluation_records"]) != n_eval:
        raise ValueError("evaluation record count mismatch")
    for index, record in enumerate(reconstructed_records):
        _assert_close(summary["evaluation_records"][index], record, label=f"eval[{index}]", atol=2e-14)
    primary_curve = [record["splits"][primary_split]["c_k"]["8"] for record in reconstructed_records]
    primary_auc = normalized_auc(expected_budgets, primary_curve)
    if primary_auc != primary["c8_normalized_action_auc"]:
        raise ValueError("primary C8 AUC differs from replay")

    expected_k = ledger["group_success_count"]
    expected_mass = ledger["group_absolute_mass"]
    selected_p = np.take_along_axis(
        ledger["p_train_by_step"], ledger["selected_train_positions"], axis=1
    )
    selected_counts, _ = np.histogram(selected_p, bins=EXPOSURE_BIN_EDGES)
    pool_counts, _ = np.histogram(ledger["p_train_by_step"].reshape(-1), bins=EXPOSURE_BIN_EDGES)
    expected_exposure = {
        "bin_edges": EXPOSURE_BIN_EDGES.tolist(),
        "selected_group_counts": selected_counts.tolist(),
        "full_pool_step_counts": pool_counts.tolist(),
        "selected_mean_p": float(selected_p.mean()),
    }
    _assert_close(summary["sampler_exposure"], expected_exposure, label="sampler exposure")
    accounting = _exact_keys(
        summary["accounting"],
        {
            "paid_actions",
            "groups",
            "updates",
            "training_scoring_forward_examples",
            "training_action_forward_examples",
            "evaluation_forward_examples",
            "group_regime_counts",
            "coefficient_absolute_mass_total",
            "nonzero_mass_group_count",
        },
        label="accounting",
    )
    expected_accounting = {
        "paid_actions": steps * 512,
        "groups": steps * 64,
        "updates": steps,
        "training_scoring_forward_examples": steps * TRAIN_SIZE,
        "training_action_forward_examples": steps * 64,
        "evaluation_forward_examples": sum(len(splits[name]) * n_eval for name in expected_eval_splits),
        "group_regime_counts": {
            "dead": int(np.count_nonzero(expected_k == 0)),
            "mixed": int(np.count_nonzero((expected_k > 0) & (expected_k < 8))),
            "all_pass": int(np.count_nonzero(expected_k == 8)),
        },
        "coefficient_absolute_mass_total": float(expected_mass.sum()),
        "nonzero_mass_group_count": int(np.count_nonzero(expected_mass > 0.0)),
    }
    _assert_close(accounting, expected_accounting, label="accounting", atol=0.0)
    if canonical and accounting["paid_actions"] != ACTION_BUDGET:
        raise ValueError("canonical action budget mismatch")

    result: dict[str, Any] = {
        "schema": VALIDATION_SCHEMA,
        "passed": True,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "phase": phase,
        "cell": cell.name,
        "estimator": cell.estimator,
        "sampler": cell.sampler,
        "seed": seed,
        "learning_rate": learning_rate,
        "initial_model_state_sha256": initial_hash,
        "rng_tape_sha256": tape_metadata["full_tape_sha256"],
        "authorization_sha256": provenance["authorization_sha256"],
        "lr_selection_sha256": provenance["lr_selection_sha256"],
        "primary_c8_auc": primary_auc,
        "initial_dev_c8": reconstructed_records[0]["splits"]["dev"]["c_k"]["8"],
        "final_dev_c8": reconstructed_records[-1]["splits"]["dev"]["c_k"]["8"],
        "q_by_step": ledger["q_by_step"],
        "primary_curve": primary_curve,
        "eval_action_budgets": expected_budgets,
        "execution": summary["execution"],
        "orchestration": {
            "worker_mode": timing["worker_mode"],
            "requested_workers": timing["requested_workers"],
        },
    }
    return result


def formula_audit() -> dict[str, Any]:
    from curriculum_maxrl.estimators import weights_maxrl, weights_rloo

    checked = 0
    for mask in range(1 << GROUP_SIZE):
        rewards = np.asarray([(mask >> bit) & 1 for bit in range(GROUP_SIZE)], dtype=np.float64)
        for estimator, canonical in (
            ("practical_maxrl", weights_maxrl),
            ("rloo", weights_rloo),
        ):
            got = _independent_weights(estimator, rewards[None, None, :])[0, 0]
            expected = canonical(rewards)
            if not np.array_equal(got, expected):
                raise ValueError(f"formula mismatch at mask={mask}, estimator={estimator}")
            k = int(rewards.sum())
            mass = float(np.abs(got).sum())
            analytic_mass = (
                (0.0 if k == 0 else 2.0 * (GROUP_SIZE - k) / GROUP_SIZE)
                if estimator == "practical_maxrl"
                else 2.0 * k * (GROUP_SIZE - k) / (GROUP_SIZE * (GROUP_SIZE - 1))
            )
            if not math.isclose(mass, analytic_mass, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError("coefficient-mass identity mismatch")
            checked += 1
    for mask in range(4):
        rewards = np.asarray([(mask >> bit) & 1 for bit in range(2)], dtype=np.float64)
        k = rewards.sum()
        maxrl = np.zeros(2) if k == 0 else rewards / k - 0.5
        rloo = 0.5 * (rewards - (k - rewards))
        if not np.array_equal(maxrl, rloo):
            raise ValueError("N=2 MaxRL/RLOO identity failed")
    grid = np.linspace(0.0, 1.0, 1001)
    if not np.allclose(1.0 - (1.0 - grid) ** 2 - grid, grid * (1.0 - grid), atol=2e-16):
        raise ValueError("u2/p(1-p) identity failed")
    return {
        "passed": True,
        "n8_binary_vectors": 1 << GROUP_SIZE,
        "estimator_vector_checks": checked,
        "n2_estimator_identity": True,
        "n2_sampler_identity": True,
    }


def _assert_exact_engineering_tree(root: Path) -> list[Path]:
    expected_cells = {Cell(*cell).name for cell in CELLS}
    if not root.is_dir() or {path.name for path in root.iterdir() if path.is_dir()} != expected_cells:
        raise ValueError("engineering root must contain exactly the six registered cell directories")
    if any(path.is_file() for path in root.iterdir()):
        raise ValueError("engineering root contains an unregistered file")
    return [root / cell / "summary.json" for cell in sorted(expected_cells)]


def analyze_engineering(
    root: Path,
    *,
    parallel_root: Path,
    check_runtime: bool = True,
) -> dict[str, Any]:
    _, lock_sha = load_and_verify_source_lock(check_runtime=check_runtime)
    serial_paths = _assert_exact_engineering_tree(root)
    parallel_paths = _assert_exact_engineering_tree(parallel_root)
    validations = [
        validate_run(path, check_live_lock=check_runtime) for path in serial_paths
    ]
    parallel_validations = [
        validate_run(path, check_live_lock=check_runtime) for path in parallel_paths
    ]
    for item in validations + parallel_validations:
        directory_cell = Path(item["summary_path"]).parent.name
        if item["cell"] != directory_cell:
            raise ValueError("engineering artifact identity differs from its cell directory")
    expected = {Cell(*cell).name for cell in CELLS}
    if {item["cell"] for item in validations} != expected or len(validations) != 6:
        raise ValueError("engineering suite does not contain exactly six cells")
    if {item["seed"] for item in validations} != {33000}:
        raise ValueError("engineering suite used a non-reserved seed")
    summaries = {
        item["cell"]: strict_json_load(Path(item["summary_path"])) for item in validations
    }
    zero_lr = all(summary["learning_rate"] == 0.0 for summary in summaries.values())
    if not zero_lr:
        raise ValueError("zero-LR engineering gate received a nonzero learning rate")
    trajectory_hashes: dict[str, dict[str, str]] = {}
    for cell_name, summary in summaries.items():
        summary_path = Path(next(v["summary_path"] for v in validations if v["cell"] == cell_name))
        with np.load(summary_path.parent / "ledger.npz", allow_pickle=False) as archive:
            trajectory_hashes[cell_name] = {
                "eval_train": sha256_array(archive["eval_probabilities_train"]),
                "eval_dev": sha256_array(archive["eval_probabilities_dev"]),
                "selected": sha256_array(archive["selected_train_positions"]),
                "actions": sha256_array(archive["actions"]),
            }
    comparisons: dict[str, Any] = {}
    for sampler in ("uniform", "p1mp", "u8"):
        left = trajectory_hashes[f"practical_maxrl__{sampler}"]
        right = trajectory_hashes[f"rloo__{sampler}"]
        comparisons[sampler] = {
            "model_eval_trajectories_identical": (
                left["eval_train"] == right["eval_train"]
                and left["eval_dev"] == right["eval_dev"]
            ),
            "selected_examples_identical": left["selected"] == right["selected"],
            "actions_identical": left["actions"] == right["actions"],
        }
        if not all(comparisons[sampler].values()):
            raise ValueError(f"zero-LR paired trajectory gate failed for {sampler}")
    init_hashes = {summary["initial_model_state_sha256"] for summary in summaries.values()}
    tape_hashes = {
        json.dumps(summary["rng_tapes"]["full_tape_sha256"], sort_keys=True)
        for summary in summaries.values()
    }
    if len(init_hashes) != 1 or len(tape_hashes) != 1:
        raise ValueError("engineering initialization or tape hashes differ across cells")
    compared_hashes: dict[str, dict[str, str]] = {}
    for cell in sorted(expected):
        serial_dir = root / cell
        parallel_dir = parallel_root / cell
        serial_files = {
            path.name for path in serial_dir.iterdir() if path.is_file() and path.name != "timing.json"
        }
        parallel_files = {
            path.name for path in parallel_dir.iterdir() if path.is_file() and path.name != "timing.json"
        }
        if serial_files != parallel_files:
            raise ValueError(f"serial/parallel scientific file sets differ for {cell}")
        compared_hashes[cell] = {}
        for filename in sorted(serial_files):
            serial_hash = sha256_file(serial_dir / filename)
            parallel_hash = sha256_file(parallel_dir / filename)
            if serial_hash != parallel_hash:
                raise ValueError(f"serial/parallel byte mismatch: {cell}/{filename}")
            compared_hashes[cell][filename] = serial_hash
    execution_values = {json.dumps(item["execution"], sort_keys=True) for item in validations + parallel_validations}
    if len(execution_values) != 1:
        raise ValueError("serial/parallel thread provenance differs")
    serial_orchestration = {
        json.dumps(item["orchestration"], sort_keys=True) for item in validations
    }
    parallel_orchestration = {
        json.dumps(item["orchestration"], sort_keys=True)
        for item in parallel_validations
    }
    if serial_orchestration != {
        json.dumps({"worker_mode": "serial", "requested_workers": 1}, sort_keys=True)
    }:
        raise ValueError("serial engineering root lacks exact worker provenance")
    if len(parallel_orchestration) != 1:
        raise ValueError("parallel engineering worker provenance differs across cells")
    parallel_record = json.loads(next(iter(parallel_orchestration)))
    if (
        parallel_record.get("worker_mode") != "process_pool_worker"
        or type(parallel_record.get("requested_workers")) is not int
        or parallel_record["requested_workers"] < 2
    ):
        raise ValueError("parallel engineering root lacks exact worker provenance")
    return {
        "schema": ENGINEERING_AUDIT_SCHEMA,
        "canonical_evidence": False,
        "passed": True,
        "source_lock_sha256": lock_sha,
        "formula_audit": formula_audit(),
        "zero_lr_sampler_pair_checks": comparisons,
        "shared_initialization": True,
        "shared_rng_tapes": True,
        "worker_execution": {
            "thread_provenance": validations[0]["execution"],
            "orchestration_provenance": {
                "serial": {"worker_mode": "serial", "requested_workers": 1},
                "parallel": parallel_record,
            },
            "serial_parallel_scientific_files_byte_identical": True,
            "timing_files_excluded_as_unbound_metadata": True,
            "compared_file_sha256": compared_hashes,
        },
        "validated_run_sha256": {
            "serial": {item["cell"]: item["summary_sha256"] for item in validations},
            "parallel": {
                item["cell"]: item["summary_sha256"] for item in parallel_validations
            },
        },
    }


def _discover_exact_runs(root: Path) -> list[Path]:
    return sorted(root.rglob("summary.json"))


def select_learning_rate_on_literal_exact_tie(
    rate_scores: Mapping[str, float],
) -> float:
    expected = {f"{rate:g}" for rate in DEVELOPMENT_LRS}
    if set(rate_scores) != expected:
        raise ValueError("learning-rate score table is incomplete")
    best = max(rate_scores.values())
    return min(rate for rate in DEVELOPMENT_LRS if rate_scores[f"{rate:g}"] == best)


def _exact_development_paths(root: Path) -> list[Path]:
    rate_names = {f"lr_{rate:g}" for rate in DEVELOPMENT_LRS}
    if not root.is_dir() or {path.name for path in root.iterdir() if path.is_dir()} != rate_names:
        raise ValueError("development root has missing/extra LR directories")
    if any(path.is_file() for path in root.iterdir()):
        raise ValueError("development root contains an extra file")
    expected_seeds = {f"seed_{seed}" for seed in DEVELOPMENT_SEEDS}
    expected_cells = {Cell(*cell).name for cell in CELLS}
    paths: list[Path] = []
    for rate_name in sorted(rate_names):
        rate_dir = root / rate_name
        if {path.name for path in rate_dir.iterdir() if path.is_dir()} != expected_seeds or any(
            path.is_file() for path in rate_dir.iterdir()
        ):
            raise ValueError(f"development LR directory tree is not exact: {rate_name}")
        for seed_name in sorted(expected_seeds):
            seed_dir = rate_dir / seed_name
            if {path.name for path in seed_dir.iterdir() if path.is_dir()} != expected_cells or any(
                path.is_file() for path in seed_dir.iterdir()
            ):
                raise ValueError(f"development seed directory tree is not exact: {seed_dir}")
            paths.extend(seed_dir / cell / "summary.json" for cell in sorted(expected_cells))
    return sorted(paths)


def analyze_development(
    root: Path, *, output: Path, check_runtime: bool = True
) -> dict[str, Any]:
    _, lock_sha = load_and_verify_source_lock(check_runtime=check_runtime)
    paths = _exact_development_paths(root)
    validations = [
        validate_run(path, check_live_lock=check_runtime) for path in paths
    ]
    expected_keys = {
        (lr, seed, Cell(*cell).name)
        for lr in DEVELOPMENT_LRS
        for seed in DEVELOPMENT_SEEDS
        for cell in CELLS
    }
    observed: dict[tuple[float, int, str], dict[str, Any]] = {}
    for item in validations:
        if item["phase"] != "development":
            raise ValueError("non-development run found in development root")
        artifact_path = Path(item["summary_path"])
        directory_cell = artifact_path.parent.name
        seed_name = artifact_path.parent.parent.name
        rate_name = artifact_path.parent.parent.parent.name
        if (
            item["cell"] != directory_cell
            or seed_name != f"seed_{item['seed']}"
            or rate_name != f"lr_{item['learning_rate']:g}"
        ):
            raise ValueError(
                "development artifact identity differs from its LR/seed/cell directories"
            )
        key = (item["learning_rate"], item["seed"], item["cell"])
        if key in observed:
            raise ValueError(f"duplicate development run: {key}")
        observed[key] = item
    if set(observed) != expected_keys:
        missing = sorted(expected_keys - set(observed))
        extra = sorted(set(observed) - expected_keys)
        raise ValueError(f"development schedule incomplete; missing={missing}, extra={extra}")

    estimator_scores: dict[str, dict[str, float]] = {}
    selected: dict[str, float] = {}
    for estimator in ("practical_maxrl", "rloo"):
        rate_scores: dict[str, float] = {}
        for lr in DEVELOPMENT_LRS:
            values = [
                observed[(lr, seed, Cell(estimator, sampler).name)]["primary_c8_auc"]
                for seed in DEVELOPMENT_SEEDS
                for sampler in ("uniform", "p1mp", "u8")
            ]
            rate_scores[f"{lr:g}"] = float(np.mean(values))
        selected_lr = select_learning_rate_on_literal_exact_tie(rate_scores)
        estimator_scores[estimator] = rate_scores
        selected[estimator] = selected_lr

    common_scores = {
        f"{lr:g}": float(
            np.mean(
                [
                    observed[(lr, seed, Cell(*cell).name)]["primary_c8_auc"]
                    for seed in DEVELOPMENT_SEEDS
                    for cell in CELLS
                ]
            )
        )
        for lr in DEVELOPMENT_LRS
    }
    selected_common = select_learning_rate_on_literal_exact_tie(common_scores)
    uniform_improvements = [
        observed[(selected[estimator], seed, Cell(estimator, "uniform").name)][
            "final_dev_c8"
        ]
        - observed[(selected[estimator], seed, Cell(estimator, "uniform").name)][
            "initial_dev_c8"
        ]
        for estimator in ("practical_maxrl", "rloo")
        for seed in DEVELOPMENT_SEEDS
    ]
    median_improvement = float(np.median(uniform_improvements))
    init_groups: dict[int, set[str]] = {}
    tape_groups: dict[int, set[str]] = {}
    for item in validations:
        init_groups.setdefault(item["seed"], set()).add(item["initial_model_state_sha256"])
        tape_groups.setdefault(item["seed"], set()).add(
            json.dumps(item["rng_tape_sha256"], sort_keys=True)
        )
    cross_cell = all(len(values) == 1 for values in init_groups.values()) and all(
        len(values) == 1 for values in tape_groups.values()
    )
    authorization_hashes = {item["authorization_sha256"] for item in validations}
    if len(authorization_hashes) != 1 or None in authorization_hashes:
        raise ValueError("development runs do not share one authorization")
    first_summary = strict_json_load(paths[0])
    auth_relative = first_summary["provenance"]["authorization_relative_path"]
    auth_path = _safe_project_provenance_path(
        auth_relative, label="development authorization provenance path"
    )
    authorization = verify_execution_authorization(
        auth_path, phase="development", lock_sha256=lock_sha
    )
    engineering_binding = authorization["zero_lr_engineering_audit"]
    validate_engineering_audit_binding(engineering_binding, lock_sha256=lock_sha)
    formula = formula_audit()
    gates = {
        "all_120_runs_complete_finite_and_valid": len(validations) == 120,
        "stored_split_and_hashes_valid": True,
        "formula_and_mass_audit_passed": formula["passed"] is True,
        "zero_lr_engineering_audit_bound_and_passing": True,
        "cross_cell_initialization_and_tapes_identical": cross_cell,
        "thread_provenance_valid": len(
            {json.dumps(item["execution"], sort_keys=True) for item in validations}
        )
        == 1,
        "exact_budgets_and_checkpoints": True,
        "valid_learning_rate_each_estimator": set(selected)
        == {"practical_maxrl", "rloo"},
        "uniform_arms_median_dev_c8_improvement": median_improvement,
        "uniform_arms_median_dev_c8_improvement_at_least_0p02": median_improvement
        >= 0.02,
        "sealed_test_outcomes_absent_from_development": True,
    }
    all_gates = all(
        value is True
        for key, value in gates.items()
        if key != "uniform_arms_median_dev_c8_improvement"
    )
    payload = {
        "schema": LR_SELECTION_SCHEMA,
        "status": "frozen_after_development_before_test_materialization",
        "all_development_gates_passed": all_gates,
        "source_lock_sha256": lock_sha,
        "development_authorization": {
            "relative_path": auth_relative,
            "sha256": sha256_file(auth_path),
        },
        "zero_lr_engineering_audit": engineering_binding,
        "development_run_manifest": {
            str(path.relative_to(root)): sha256_file(path) for path in paths
        },
        "selection_metric": "development C8 normalized action-budget AUC",
        "tie_break": "smaller learning rate on literal exact score equality",
        "estimator_rate_scores": estimator_scores,
        "common_rate_scores": common_scores,
        "selected_learning_rates_by_estimator": selected,
        "selected_common_learning_rate": selected_common,
        "gates": gates,
    }
    if all_gates:
        validate_lr_selection_document(payload, lock_sha256=lock_sha)
    write_json(output, payload)
    return payload


def exact_sign_flip_pvalue(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0 or np.any(~np.isfinite(x)):
        raise ValueError("invalid paired values for sign-flip test")
    split = len(x) // 2
    if float(x.sum()) == 0.0:
        return 1.0

    def signed_sums(part: np.ndarray) -> np.ndarray:
        sums = np.zeros(1 << len(part), dtype=np.float64)
        total = float(part.sum())
        for mask in range(1 << len(part)):
            selected_sum = 0.0
            for index, value in enumerate(part):
                if mask & (1 << index):
                    selected_sum += float(value)
            sums[mask] = total - 2.0 * selected_sum
        return sums

    left = signed_sums(x[:split])
    right = np.sort(signed_sums(x[split:]))
    threshold = abs(float(x.sum()))
    extreme = 0
    for value in left:
        lower = np.searchsorted(right, -threshold - value, side="right")
        upper = len(right) - np.searchsorted(right, threshold - value, side="left")
        extreme += int(lower + upper)
    return min(1.0, extreme / float((1 << len(x))))


def bootstrap_interval(values: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64DXSM(BOOTSTRAP_SEED))
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    chunk = 10_000
    for start in range(0, BOOTSTRAP_REPLICATES, chunk):
        stop = min(start + chunk, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, len(x), size=(stop - start, len(x)))
        means[start:stop] = x[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def holm_rejections(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, bool]:
    ordered = sorted(pvalues, key=lambda name: (pvalues[name], name))
    rejected = {name: False for name in pvalues}
    continue_rejecting = True
    count = len(ordered)
    for rank, name in enumerate(ordered):
        threshold = alpha / (count - rank)
        if continue_rejecting and pvalues[name] <= threshold:
            rejected[name] = True
        else:
            continue_rejecting = False
    return rejected


def _collect_confirmation(
    root: Path, phase: str, *, check_runtime: bool = True
) -> tuple[dict[int, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    expected_seed_names = {f"seed_{seed}" for seed in CONFIRMATION_SEEDS}
    expected_cells = {Cell(*cell).name for cell in CELLS}
    if not root.is_dir():
        raise ValueError("confirmation result root is missing")
    observed_root_dirs = {path.name for path in root.iterdir() if path.is_dir()}
    if not observed_root_dirs <= expected_seed_names or any(path.is_file() for path in root.iterdir()):
        raise ValueError("confirmation root contains an extra result directory/file")
    for seed in CONFIRMATION_SEEDS:
        seed_dir = root / f"seed_{seed}"
        if seed_dir.exists():
            observed_cells = {path.name for path in seed_dir.iterdir() if path.is_dir()}
            if not observed_cells <= expected_cells or any(path.is_file() for path in seed_dir.iterdir()):
                raise ValueError(f"confirmation seed directory contains extra results: {seed_dir}")
        for estimator, sampler in CELLS:
            cell = Cell(estimator, sampler).name
            cell_dir = seed_dir / cell
            summary_path = cell_dir / "summary.json"
            failure_path = cell_dir / "failure.json"
            if cell_dir.exists() and any(path.is_dir() for path in cell_dir.iterdir()):
                raise ValueError(f"confirmation cell contains an extra nested directory: {cell_dir}")
            cell_files = (
                {path.name for path in cell_dir.iterdir() if path.is_file()}
                if cell_dir.exists()
                else set()
            )
            if summary_path.exists() and failure_path.exists():
                raise ValueError(f"confirmation cell contains both summary and failure: {cell_dir}")
            if not summary_path.exists():
                if failure_path.exists():
                    if cell_files != {"failure.json"}:
                        raise ValueError(
                            f"failed confirmation cell contains extra artifacts: {cell_dir}"
                        )
                    failure = strict_json_load(failure_path)
                    _assert_finite_json_tree(failure, location="confirmation failure")
                    failure = _exact_keys(
                        failure,
                        {
                            "schema",
                            "artifact_state",
                            "phase",
                            "cell",
                            "seed",
                            "learning_rate",
                            "steps",
                            "exception_type",
                            "exception",
                            "traceback",
                            "wall_seconds",
                        },
                        label="confirmation failure ledger",
                    )
                    if (
                        failure["schema"]
                        != "curriculum-maxrl/digits-factorial-failure/v1"
                        or failure["artifact_state"] != "failed"
                        or failure["phase"] != phase
                        or failure["cell"] != cell
                        or type(failure["seed"]) is not int
                        or failure["seed"] != seed
                        or type(failure["learning_rate"]) is not float
                        or not math.isfinite(failure["learning_rate"])
                        or type(failure["steps"]) is not int
                        or failure["steps"] != N_STEPS
                        or not all(
                            isinstance(failure[name], str)
                            for name in ("exception_type", "exception", "traceback")
                        )
                        or type(failure["wall_seconds"]) is not float
                        or failure["wall_seconds"] <= 0.0
                    ):
                        raise ValueError(
                            f"confirmation failure ledger identity/schema differs: {cell_dir}"
                        )
                elif cell_files:
                    raise ValueError(
                        f"confirmation cell without summary/failure contains files: {cell_dir}"
                    )
                failures.append(
                    {
                        "seed": seed,
                        "cell": cell,
                        "reason": "failed" if failure_path.exists() else "missing",
                        "failure_sha256": sha256_file(failure_path) if failure_path.exists() else None,
                    }
                )
                continue
            try:
                item = validate_run(summary_path, check_live_lock=check_runtime)
                if item["phase"] != phase:
                    raise ValueError(f"phase {item['phase']} != {phase}")
                if item["seed"] != seed or item["cell"] != cell:
                    raise ValueError(
                        "confirmation artifact identity differs from its seed/cell directory"
                    )
                by_seed.setdefault(seed, {})[cell] = item
            except Exception as error:
                failures.append({"seed": seed, "cell": cell, "reason": str(error)})
    return by_seed, failures


def _contrast_report(values: Sequence[float]) -> dict[str, Any]:
    low, high = bootstrap_interval(values)
    return {
        "values": list(map(float, values)),
        "mean": float(np.mean(values)),
        "bootstrap_percentile_95": [low, high],
        "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(values),
        "positive_blocks": int(np.count_nonzero(np.asarray(values) > 0.0)),
        "zero_blocks": int(np.count_nonzero(np.asarray(values) == 0.0)),
        "n_blocks": len(values),
    }


def _analyze_confirmation_collected(
    root: Path,
    phase: str,
    by_seed: Mapping[int, Mapping[str, dict[str, Any]]],
    failures: list[dict[str, Any]],
    *,
    seed_subset: Sequence[int] | None = None,
) -> dict[str, Any]:
    expected_cells = {Cell(*cell).name for cell in CELLS}
    all_complete_seeds = [
        seed for seed in CONFIRMATION_SEEDS if set(by_seed.get(seed, {})) == expected_cells
    ]
    complete_seeds = (
        all_complete_seeds
        if seed_subset is None
        else [seed for seed in seed_subset if seed in all_complete_seeds]
    )
    removed_blocks = [seed for seed in CONFIRMATION_SEEDS if seed not in all_complete_seeds]
    result: dict[str, Any] = {
        "phase": phase,
        "root": str(root),
        "complete_block_seeds": complete_seeds,
        "removed_block_seeds": removed_blocks,
        "cell_failures": failures,
        "n_complete_blocks": len(complete_seeds),
        "minimum_20_blocks_gate": len(complete_seeds) >= 20,
        "analysis_seed_subset": None if seed_subset is None else list(seed_subset),
    }
    if len(complete_seeds) < 20:
        result.update({"status": "inconclusive_fewer_than_20_complete_blocks", "inference": None})
        return result

    def outcomes(cell: str) -> np.ndarray:
        return np.asarray(
            [by_seed[seed][cell]["primary_c8_auc"] for seed in complete_seeds],
            dtype=np.float64,
        )

    for seed in complete_seeds:
        initial_hashes = {by_seed[seed][cell]["initial_model_state_sha256"] for cell in expected_cells}
        tape_hashes = {
            json.dumps(by_seed[seed][cell]["rng_tape_sha256"], sort_keys=True)
            for cell in expected_cells
        }
        if len(initial_hashes) != 1 or len(tape_hashes) != 1:
            raise ValueError(f"cross-cell initialization/tapes differ for seed {seed}")

    m_u8 = outcomes("practical_maxrl__u8")
    m_p = outcomes("practical_maxrl__p1mp")
    m_uniform = outcomes("practical_maxrl__uniform")
    r_u8 = outcomes("rloo__u8")
    r_p = outcomes("rloo__p1mp")
    r_uniform = outcomes("rloo__uniform")
    interaction = 0.5 * ((m_u8 + r_p) - (m_p + r_u8))
    contrasts = {
        "interaction": _contrast_report(interaction),
        "maxrl_u8_minus_p1mp": _contrast_report(m_u8 - m_p),
        "rloo_p1mp_minus_u8": _contrast_report(r_p - r_u8),
        "maxrl_u8_minus_uniform": _contrast_report(m_u8 - m_uniform),
        "rloo_p1mp_minus_uniform": _contrast_report(r_p - r_uniform),
    }
    family_names = (
        "maxrl_u8_minus_p1mp",
        "rloo_p1mp_minus_u8",
        "maxrl_u8_minus_uniform",
        "rloo_p1mp_minus_uniform",
    )
    raw_p = {name: contrasts[name]["exact_two_sided_sign_flip_p"] for name in family_names}
    rejected = holm_rejections(raw_p)
    for name in family_names:
        contrasts[name]["holm_family"] = "four predeclared simple effects and anchors"
        contrasts[name]["holm_rejected_at_0p05"] = rejected[name]

    interaction_report = contrasts["interaction"]
    matched_names = ("maxrl_u8_minus_p1mp", "rloo_p1mp_minus_u8")
    both_matched = all(
        contrasts[name]["mean"] > 0.0
        and contrasts[name]["bootstrap_percentile_95"][0] > 0.0
        and contrasts[name]["holm_rejected_at_0p05"]
        for name in matched_names
    )

    tv_values: list[float] = []
    for seed in complete_seeds:
        for estimator in ("practical_maxrl", "rloo"):
            q_u8 = by_seed[seed][f"{estimator}__u8"]["q_by_step"]
            q_p = by_seed[seed][f"{estimator}__p1mp"]["q_by_step"]
            tv_values.extend((0.5 * np.abs(q_u8 - q_p).sum(axis=1)).tolist())
    mean_tv = float(np.mean(tv_values))
    treatment_passed = mean_tv >= 0.02
    primary_supported = (
        interaction_report["mean"] >= 0.01
        and interaction_report["bootstrap_percentile_95"][0] > 0.0
        and interaction_report["exact_two_sided_sign_flip_p"] <= 0.05
        and treatment_passed
    )
    result.update(
        {
            "status": "analyzed",
            "contrasts": contrasts,
            "primary_supported": primary_supported,
            "both_estimators_match_registered_sampler": both_matched,
            "treatment_delivery": {
                "action_budget_weighted_mean_tv_q_u8_vs_p1mp": mean_tv,
                "threshold": 0.02,
                "passed": treatment_passed,
            },
        }
    )
    return result


def _analyze_confirmation_root(
    root: Path, phase: str, *, check_runtime: bool = True
) -> dict[str, Any]:
    by_seed, failures = _collect_confirmation(root, phase, check_runtime=check_runtime)
    return _analyze_confirmation_collected(root, phase, by_seed, failures)


def shared_complete_confirmation_seeds(
    tuned_map: Mapping[int, Mapping[str, Any]],
    common_map: Mapping[int, Mapping[str, Any]],
) -> list[int]:
    expected_cells = {Cell(*cell).name for cell in CELLS}
    return [
        seed
        for seed in CONFIRMATION_SEEDS
        if set(tuned_map.get(seed, {})) == expected_cells
        and set(common_map.get(seed, {})) == expected_cells
    ]


def analyze_confirmation(
    tuned_root: Path,
    common_root: Path,
    *,
    lr_selection_path: Path,
    output: Path,
    check_runtime: bool = True,
) -> dict[str, Any]:
    _, lock_sha = load_and_verify_source_lock(check_runtime=check_runtime)
    selection = strict_json_load(lr_selection_path)
    validate_lr_selection_document(selection, lock_sha256=lock_sha)
    tuned_map, tuned_failures = _collect_confirmation(
        tuned_root, "confirmation_tuned", check_runtime=check_runtime
    )
    common_map, common_failures = _collect_confirmation(
        common_root, "confirmation_common", check_runtime=check_runtime
    )
    tuned = _analyze_confirmation_collected(
        tuned_root, "confirmation_tuned", tuned_map, tuned_failures
    )
    common = _analyze_confirmation_collected(
        common_root, "confirmation_common", common_map, common_failures
    )
    intersection = shared_complete_confirmation_seeds(tuned_map, common_map)
    optimizer_sensitivity: dict[str, Any]
    if len(intersection) < 20:
        optimizer_sensitivity = {
            "status": "inconclusive_fewer_than_20_shared_complete_blocks",
            "shared_complete_block_seeds": intersection,
            "n_shared_complete_blocks": len(intersection),
        }
    else:
        tuned_shared = _analyze_confirmation_collected(
            tuned_root,
            "confirmation_tuned",
            tuned_map,
            tuned_failures,
            seed_subset=intersection,
        )
        common_shared = _analyze_confirmation_collected(
            common_root,
            "confirmation_common",
            common_map,
            common_failures,
            seed_subset=intersection,
        )
        tuned_mean = tuned_shared["contrasts"]["interaction"]["mean"]
        common_mean = common_shared["contrasts"]["interaction"]["mean"]
        reversal = tuned_mean > 0.0 and common_mean < 0.0
        optimizer_sensitivity = {
            "status": "analyzed_on_shared_complete_blocks",
            "shared_complete_block_seeds": intersection,
            "n_shared_complete_blocks": len(intersection),
            "tuned_interaction_mean": tuned_mean,
            "common_rate_interaction_mean": common_mean,
            "common_rate_sign_reversal": reversal,
            "required_label": (
                "optimizer-sensitive" if tuned_shared["primary_supported"] and reversal else None
            ),
        }
    payload = {
        "schema": ANALYSIS_SCHEMA,
        "source_lock_sha256": lock_sha,
        "lr_selection_relative_path": lr_selection_path.resolve().relative_to(
            PROJECT_ROOT.resolve()
        ).as_posix(),
        "lr_selection_sha256": sha256_file(lr_selection_path),
        "tuned": tuned,
        "common_rate_robustness": common,
        "optimizer_sensitivity": optimizer_sensitivity,
        "claim_boundary": (
            "Controlled contextual bandit on sklearn Digits; not trajectory RL, "
            "GRPO, ImageNet, or native ProCuRL."
        ),
    }
    write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-run")
    validate.add_argument("summary", type=Path)
    engineering = subparsers.add_parser("analyze-engineering")
    engineering.add_argument("root", type=Path)
    engineering.add_argument("--parallel-root", type=Path, required=True)
    engineering.add_argument("--output", type=Path, required=True)
    development = subparsers.add_parser("analyze-development")
    development.add_argument("root", type=Path)
    development.add_argument("--output", type=Path, required=True)
    confirmation = subparsers.add_parser("analyze-confirmation")
    confirmation.add_argument("--tuned-root", type=Path, required=True)
    confirmation.add_argument("--common-root", type=Path, required=True)
    confirmation.add_argument("--lr-selection", type=Path, required=True)
    confirmation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate-run":
        payload = validate_run(args.summary)
        payload.pop("q_by_step", None)
        payload.pop("eval_action_budgets", None)
        print(json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False))
    elif args.command == "analyze-engineering":
        payload = analyze_engineering(args.root, parallel_root=args.parallel_root)
        write_json(args.output, payload)
        print(args.output)
    elif args.command == "analyze-development":
        analyze_development(args.root, output=args.output)
        print(args.output)
    else:
        analyze_confirmation(
            args.tuned_root,
            args.common_root,
            lr_selection_path=args.lr_selection,
            output=args.output,
        )
        print(args.output)


if __name__ == "__main__":
    main()
