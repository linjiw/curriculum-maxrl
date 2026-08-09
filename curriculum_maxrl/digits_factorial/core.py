"""Frozen mechanics for the Digits exact-probability factorial.

This module is intentionally self-contained.  The training implementation does
not call :mod:`curriculum_maxrl.estimators`; the canonical repository functions
are imported only by regression tests.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from torch import nn


SCHEMA_VERSION = "v1"
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DATA_MANIFEST_PATH = PACKAGE_DIR / "digits_split_manifest.json"
SOURCE_LOCK_PATH = PACKAGE_DIR / "SOURCE_LOCK.json"

EXPECTED_RUNTIME = {
    "python_implementation": "CPython",
    "python": "3.11.14",
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "torch": "2.8.0",
    "scikit_learn": "1.5.2",
}

TRAIN_SIZE = 1077
DEV_SIZE = 360
TEST_SIZE = 360
N_CLASSES = 10
INPUT_DIM = 64
HIDDEN_DIM = 64
PARAMETER_COUNT = 4810

GROUP_SIZE = 8
GROUPS_PER_STEP = 64
N_STEPS = 512
ACTION_BUDGET = N_STEPS * GROUPS_PER_STEP * GROUP_SIZE
EVAL_INTERVAL_ACTIONS = 8192
EVAL_INTERVAL_STEPS = EVAL_INTERVAL_ACTIONS // (GROUPS_PER_STEP * GROUP_SIZE)
EVAL_STEPS = tuple(range(0, N_STEPS + 1, EVAL_INTERVAL_STEPS))
RECOVERY_STEPS = (0, 128, 256, 384, 512)
TRAIN_SCORING_FORWARD_COUNT = N_STEPS * TRAIN_SIZE
UNIFORM_FLOOR = 0.10
MOMENTUM = 0.9
EXPOSURE_BIN_EDGES = np.linspace(0.0, 1.0, 11, dtype=np.float64)

ESTIMATORS = ("practical_maxrl", "rloo")
SAMPLERS = ("uniform", "p1mp", "u8")
CELLS = tuple((estimator, sampler) for estimator in ESTIMATORS for sampler in SAMPLERS)
DEVELOPMENT_LRS = (0.03, 0.1, 0.3, 1.0, 3.0)
DEVELOPMENT_SEEDS = tuple(range(31000, 31004))
CONFIRMATION_SEEDS = tuple(range(32000, 32024))
ENGINEERING_SEED = 33000
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20260808

ARTIFACT_SCHEMA = "curriculum-maxrl/digits-factorial-run/v2"
ANALYSIS_SCHEMA = "curriculum-maxrl/digits-factorial-analysis/v2"
LOCK_SCHEMA = "curriculum-maxrl/digits-factorial-source-lock/v2"
AUTHORIZATION_SCHEMA = "curriculum-maxrl/digits-factorial-execution-authorization/v2"
CHECKPOINT_SCHEMA = "curriculum-maxrl/digits-factorial-recovery-checkpoint/v2"
ENGINEERING_AUDIT_SCHEMA = "curriculum-maxrl/digits-factorial-engineering-audit/v2"
LR_SELECTION_SCHEMA = "curriculum-maxrl/digits-factorial-lr-selection/v2"
THREAD_COUNT = 1
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def strict_json_load(path: Path) -> dict[str, Any]:
    """Read a JSON object while rejecting duplicate keys and non-finite values."""

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number in {path}: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object in {path}")
    return payload


def json_ready(value: Any) -> Any:
    """Convert nested NumPy/PyTorch-adjacent values to strict-JSON values."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite value cannot be serialized")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
    path.write_text(encoded + "\n", encoding="utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_array(array: np.ndarray) -> str:
    """Hash an array through NumPy's canonical, pickle-free NPY encoding."""

    handle = io.BytesIO()
    np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
    return sha256_bytes(handle.getvalue())


def domain_seed(logical_seed: int, domain: str) -> int:
    payload = f"digits-factorial-v1|{logical_seed}|{domain}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def make_rng(logical_seed: int, domain: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(domain_seed(logical_seed, domain)))


def generate_rng_tapes(
    logical_seed: int, *, steps: int = N_STEPS
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not (1 <= steps <= N_STEPS):
        raise ValueError(f"steps must be in [1, {N_STEPS}]")
    task_rng = make_rng(logical_seed, "task-selection-uniforms")
    action_rng = make_rng(logical_seed, "action-uniforms")
    task_uniforms = task_rng.random((N_STEPS, GROUPS_PER_STEP), dtype=np.float64)
    action_uniforms = action_rng.random(
        (N_STEPS, GROUPS_PER_STEP, GROUP_SIZE), dtype=np.float64
    )
    metadata = {
        "generator": "NumPy PCG64DXSM",
        "logical_seed": logical_seed,
        "domain_seeds": {
            "task_selection": domain_seed(logical_seed, "task-selection-uniforms"),
            "actions": domain_seed(logical_seed, "action-uniforms"),
        },
        "full_tape_shapes": {
            "task_selection": list(task_uniforms.shape),
            "actions": list(action_uniforms.shape),
        },
        "full_tape_sha256": {
            "task_selection": sha256_array(task_uniforms),
            "actions": sha256_array(action_uniforms),
        },
        "terminal_states": {
            "task_selection": json_ready(task_rng.bit_generator.state),
            "actions": json_ready(action_rng.bit_generator.state),
        },
    }
    return task_uniforms[:steps].copy(), action_uniforms[:steps].copy(), metadata


def load_stored_digits(
    manifest_path: Path = DATA_MANIFEST_PATH,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Load Digits and use only the stored original-dataset split indices."""

    manifest = strict_json_load(manifest_path)
    if set(manifest) != {
        "schema",
        "dataset",
        "construction",
        "array_sha256",
        "index_sha256",
        "class_counts",
        "indices",
    } or manifest.get("schema") != "curriculum-maxrl/digits-split/v1":
        raise ValueError("unexpected Digits split manifest schema")
    expected_dataset = {
        "loader": "sklearn.datasets.load_digits",
        "n_examples": 1797,
        "raw_data_shape": [1797, 64],
        "target_shape": [1797],
        "normalization": "astype(float64) / 16.0",
        "sklearn_version": EXPECTED_RUNTIME["scikit_learn"],
    }
    expected_construction = {
        "stage_1": {
            "algorithm": "StratifiedShuffleSplit(n_splits=1,test_size=360)",
            "random_state": 20260808,
            "output": "sealed test plus remaining 1437",
        },
        "stage_2": {
            "algorithm": "StratifiedShuffleSplit(n_splits=1,test_size=360) on remaining",
            "random_state": 20260809,
            "output": "development 360 plus training 1077",
        },
        "index_semantics": "original load_digits row indices; stored order is authoritative",
        "implicit_regeneration_forbidden": True,
    }
    if manifest["dataset"] != expected_dataset or manifest["construction"] != expected_construction:
        raise ValueError("Digits manifest dataset/construction declaration differs")
    if not isinstance(manifest["array_sha256"], dict) or set(
        manifest["array_sha256"]
    ) != {"raw_data", "target", "normalized_data"}:
        raise ValueError("Digits manifest array-hash schema differs")
    if not isinstance(manifest["index_sha256"], dict) or set(
        manifest["index_sha256"]
    ) != {"train", "dev", "test"}:
        raise ValueError("Digits manifest index-hash schema differs")
    if not isinstance(manifest["indices"], dict) or set(manifest["indices"]) != {
        "train",
        "dev",
        "test",
    }:
        raise ValueError("Digits manifest index schema differs")
    bunch = load_digits()
    raw_x = np.asarray(bunch.data)
    y = np.asarray(bunch.target)
    expected = manifest["array_sha256"]
    observed = {
        "raw_data": sha256_array(raw_x),
        "target": sha256_array(y),
        "normalized_data": sha256_array(raw_x.astype(np.float64) / 16.0),
    }
    if observed != expected:
        raise ValueError(f"Digits dataset hash mismatch: {observed} != {expected}")
    splits = {
        name: np.asarray(manifest["indices"][name], dtype=np.int64)
        for name in ("train", "dev", "test")
    }
    for name, expected_size in (("train", TRAIN_SIZE), ("dev", DEV_SIZE), ("test", TEST_SIZE)):
        stored_indices = manifest["indices"][name]
        if not isinstance(stored_indices, list) or any(type(value) is not int for value in stored_indices):
            raise ValueError(f"stored {name} indices are not exact JSON integers")
        if splits[name].shape != (expected_size,):
            raise ValueError(f"stored {name} split has wrong shape")
        if sha256_array(splits[name]) != manifest["index_sha256"][name]:
            raise ValueError(f"stored {name} index hash mismatch")
    concatenated = np.concatenate([splits["train"], splits["dev"], splits["test"]])
    if len(np.unique(concatenated)) != len(y) or set(concatenated.tolist()) != set(range(len(y))):
        raise ValueError("stored splits are not a disjoint exhaustive partition")
    expected_count_keys = {"all", "train", "dev", "test"}
    if not isinstance(manifest["class_counts"], dict) or set(
        manifest["class_counts"]
    ) != expected_count_keys:
        raise ValueError("Digits manifest class-count schema differs")
    for name, indices in {"all": np.arange(len(y), dtype=np.int64), **splits}.items():
        observed_counts = {
            str(label): int(np.count_nonzero(y[indices] == label))
            for label in range(N_CLASSES)
        }
        if manifest["class_counts"][name] != observed_counts:
            raise ValueError(f"stored {name} class counts differ from indices/targets")
    x = raw_x.astype(np.float64) / 16.0
    return x, y.astype(np.int64, copy=False), splits, manifest


class DigitsMLP(nn.Module):
    """Frozen 64-64-10 ReLU policy (4,810 trainable parameters)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(INPUT_DIM, HIDDEN_DIM, bias=True, dtype=torch.float64)
        self.linear2 = nn.Linear(HIDDEN_DIM, N_CLASSES, bias=True, dtype=torch.float64)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(inputs)))


def initialize_model(logical_seed: int) -> DigitsMLP:
    """Initialize all cells from an identical NumPy-generated checkpoint."""

    model = DigitsMLP()
    rng = make_rng(logical_seed, "model-initialization")
    with torch.no_grad():
        for layer in (model.linear1, model.linear2):
            bound = 1.0 / math.sqrt(layer.in_features)
            weight = rng.uniform(-bound, bound, size=tuple(layer.weight.shape))
            bias = rng.uniform(-bound, bound, size=tuple(layer.bias.shape))
            layer.weight.copy_(torch.from_numpy(weight))
            layer.bias.copy_(torch.from_numpy(bias))
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count != PARAMETER_COUNT:
        raise AssertionError(f"model has {count} parameters, expected {PARAMETER_COUNT}")
    return model


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_array(array)))
    return digest.hexdigest()


def torch_state_sha256(value: Any) -> str:
    """Canonical digest for nested tensor/optimizer/RNG state."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor:")
            digest.update(str(item.dtype).encode("ascii"))
            digest.update(bytes.fromhex(sha256_array(item.detach().cpu().numpy())))
        elif isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                update(key)
                update(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(b"sequence[")
            for child in item:
                update(child)
            digest.update(b"]")
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(
                json.dumps(item, sort_keys=True, allow_nan=False).encode("utf-8")
            )
            digest.update(b";")
        else:
            raise TypeError(f"unsupported state value for canonical hash: {type(item)}")

    update(value)
    return digest.hexdigest()


def practical_maxrl_weights(rewards: np.ndarray) -> np.ndarray:
    """Independent exact implementation of dropped-group practical MaxRL."""

    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.shape[-1] != GROUP_SIZE:
        raise ValueError(f"last reward dimension must be {GROUP_SIZE}")
    counts = rewards.sum(axis=-1, keepdims=True)
    safe_counts = np.where(counts > 0.0, counts, 1.0)
    values = rewards / safe_counts - 1.0 / GROUP_SIZE
    return np.where(counts > 0.0, values, 0.0)


def rloo_weights(rewards: np.ndarray) -> np.ndarray:
    """Independent exact N=8 leave-one-out baseline implementation."""

    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.shape[-1] != GROUP_SIZE:
        raise ValueError(f"last reward dimension must be {GROUP_SIZE}")
    counts = rewards.sum(axis=-1, keepdims=True)
    return (rewards - (counts - rewards) / (GROUP_SIZE - 1)) / GROUP_SIZE


def estimator_weights(estimator: str, rewards: np.ndarray) -> np.ndarray:
    if estimator == "practical_maxrl":
        return practical_maxrl_weights(rewards)
    if estimator == "rloo":
        return rloo_weights(rewards)
    raise ValueError(f"unknown estimator: {estimator}")


def expected_binary_mass(estimator: str, success_count: int) -> float:
    k = int(success_count)
    if not 0 <= k <= GROUP_SIZE:
        raise ValueError("success_count outside group")
    if estimator == "practical_maxrl":
        return 0.0 if k == 0 else 2.0 * (GROUP_SIZE - k) / GROUP_SIZE
    if estimator == "rloo":
        return 2.0 * k * (GROUP_SIZE - k) / (GROUP_SIZE * (GROUP_SIZE - 1))
    raise ValueError(f"unknown estimator: {estimator}")


def sampler_score(sampler: str, correct_probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(correct_probabilities, dtype=np.float64)
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("correct-class probabilities must be finite and in [0,1]")
    if sampler == "uniform":
        return np.ones_like(p)
    if sampler == "p1mp":
        return p * (1.0 - p)
    if sampler == "u8":
        return np.maximum(0.0, 1.0 - np.power(1.0 - p, GROUP_SIZE) - p)
    raise ValueError(f"unknown sampler: {sampler}")


def sampler_probabilities(sampler: str, correct_probabilities: np.ndarray) -> np.ndarray:
    scores = sampler_score(sampler, correct_probabilities)
    size = len(scores)
    if sampler == "uniform":
        return np.full(size, 1.0 / size, dtype=np.float64)
    total = float(scores.sum(dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        return np.full(size, 1.0 / size, dtype=np.float64)
    q = UNIFORM_FLOOR / size + (1.0 - UNIFORM_FLOOR) * scores / total
    q /= q.sum(dtype=np.float64)
    return q


def inverse_cdf_indices(probabilities: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    q = np.asarray(probabilities, dtype=np.float64)
    u = np.asarray(uniforms, dtype=np.float64)
    if q.ndim != 1 or np.any(q < 0.0) or not np.isclose(q.sum(), 1.0, atol=1e-12):
        raise ValueError("invalid categorical probability vector")
    cdf = np.cumsum(q, dtype=np.float64)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, u, side="right").astype(np.int64)


def inverse_cdf_actions(probabilities: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    u = np.asarray(uniforms, dtype=np.float64)
    if probs.shape != (GROUPS_PER_STEP, N_CLASSES):
        raise ValueError("action probabilities have wrong shape")
    if u.shape != (GROUPS_PER_STEP, GROUP_SIZE):
        raise ValueError("action uniforms have wrong shape")
    cdf = np.cumsum(probs, axis=1, dtype=np.float64)
    cdf[:, -1] = 1.0
    actions = (u[:, :, None] >= cdf[:, None, :]).sum(axis=2)
    return np.minimum(actions, N_CLASSES - 1).astype(np.int64)


def analytic_ck(correct_probabilities: np.ndarray, k: int) -> float:
    if k not in (1, 2, 4, 8, 16, 32):
        raise ValueError("k is not in the frozen analytic C_k grid")
    p = np.asarray(correct_probabilities, dtype=np.float64)
    return float(np.mean(1.0 - np.power(1.0 - p, k), dtype=np.float64))


def evaluate_probabilities(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    if probs.shape != (len(y), N_CLASSES):
        raise ValueError("evaluation probability matrix has wrong shape")
    p_y = probs[np.arange(len(y)), y]
    predictions = probs.argmax(axis=1)
    correctness = predictions == y
    one_hot = np.eye(N_CLASSES, dtype=np.float64)[y]
    per_class_accuracy = [
        float(correctness[y == label].mean()) for label in range(N_CLASSES)
    ]
    per_class_mean_p = [float(p_y[y == label].mean()) for label in range(N_CLASSES)]
    return {
        "mean_p_y": float(p_y.mean()),
        "c_k": {str(k): analytic_ck(p_y, k) for k in (1, 2, 4, 8, 16, 32)},
        "nll": float(-np.log(np.clip(p_y, np.finfo(np.float64).tiny, 1.0)).mean()),
        "brier": float(np.square(probs - one_hot).sum(axis=1).mean()),
        "top1_accuracy": float(correctness.mean()),
        "macro_class_accuracy": float(np.mean(per_class_accuracy)),
        "macro_class_mean_p_y": float(np.mean(per_class_mean_p)),
        "per_class_accuracy": per_class_accuracy,
        "per_class_mean_p_y": per_class_mean_p,
    }


def normalized_auc(action_budgets: Sequence[int], values: Sequence[float]) -> float:
    x = np.asarray(action_budgets, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
        raise ValueError("AUC arrays have incompatible shapes")
    if x[0] != 0 or np.any(np.diff(x) <= 0) or x[-1] <= 0:
        raise ValueError("AUC budget axis must start at zero and strictly increase")
    return float(np.trapz(y, x=x) / x[-1])


def runtime_record() -> dict[str, str]:
    import scipy

    return {
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__.split("+")[0],
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
    }


def assert_pinned_runtime(*, allow_platform_fields: bool = True) -> dict[str, str]:
    observed = runtime_record()
    projected = {key: observed[key] for key in EXPECTED_RUNTIME}
    if projected != EXPECTED_RUNTIME:
        raise RuntimeError(f"runtime mismatch: {projected} != {EXPECTED_RUNTIME}")
    return observed


def configure_deterministic_cpu() -> dict[str, Any]:
    """Force and verify the frozen single-thread deterministic CPU execution."""

    import os

    for variable in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[variable] = str(THREAD_COUNT)
    torch.set_num_threads(THREAD_COUNT)
    try:
        torch.set_num_interop_threads(THREAD_COUNT)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before parallel work.
        pass
    torch.use_deterministic_algorithms(True)
    observed = {
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "thread_environment": {
            variable: os.environ.get(variable) for variable in THREAD_ENVIRONMENT_VARIABLES
        },
    }
    expected = {
        "device": "cpu",
        "torch_num_threads": THREAD_COUNT,
        "torch_num_interop_threads": THREAD_COUNT,
        "deterministic_algorithms": True,
        "thread_environment": {
            variable: str(THREAD_COUNT) for variable in THREAD_ENVIRONMENT_VARIABLES
        },
    }
    if observed != expected:
        raise RuntimeError(f"deterministic CPU settings mismatch: {observed} != {expected}")
    return observed


def stable_torch_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {name: tensor.detach().cpu().numpy().copy() for name, tensor in state.items()}


@dataclass(frozen=True)
class Cell:
    estimator: str
    sampler: str

    def __post_init__(self) -> None:
        if (self.estimator, self.sampler) not in CELLS:
            raise ValueError(f"unregistered cell: {(self.estimator, self.sampler)}")

    @property
    def name(self) -> str:
        return f"{self.estimator}__{self.sampler}"


def cell_from_name(name: str) -> Cell:
    pieces = name.split("__")
    if len(pieces) != 2:
        raise ValueError(f"invalid cell name: {name}")
    return Cell(pieces[0], pieces[1])


def frozen_schedule() -> dict[str, Any]:
    return {
        "cells": [Cell(*cell).name for cell in CELLS],
        "development_learning_rates": list(DEVELOPMENT_LRS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "engineering_seed": ENGINEERING_SEED,
        "group_size": GROUP_SIZE,
        "groups_per_step": GROUPS_PER_STEP,
        "steps": N_STEPS,
        "action_budget": ACTION_BUDGET,
        "eval_interval_actions": EVAL_INTERVAL_ACTIONS,
        "eval_steps": list(EVAL_STEPS),
        "recovery_steps": list(RECOVERY_STEPS),
        "uniform_floor": UNIFORM_FLOOR,
        "optimizer": {"name": "SGD", "momentum": MOMENTUM, "weight_decay": 0.0},
        "execution": {
            "device": "cpu",
            "torch_num_threads": THREAD_COUNT,
            "torch_num_interop_threads": THREAD_COUNT,
            "deterministic_algorithms": True,
            "thread_environment_variables": list(THREAD_ENVIRONMENT_VARIABLES),
        },
        "model": "Linear(64,64)-ReLU-Linear(64,10), float64",
        "parameter_count": PARAMETER_COUNT,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
