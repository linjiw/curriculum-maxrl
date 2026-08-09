"""Independent read-only verifier for the V3 Acrobot confirmation artifact."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import tempfile
from pathlib import Path

import gymnasium
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CASES = ("uniform_shared_h64", "teacher_shared_h64")
EXPECTED_SEEDS = list(range(12_000, 12_020))
PRIMARY_METRIC = "auc_mean_pass_by_transitions"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_auc(y: list[float], x: list[int]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if len(x_arr) != len(y_arr) or len(x_arr) < 2:
        raise ValueError("AUC curves must have equal length of at least two")
    if x_arr[0] != 0 or x_arr[-1] < 2_000_000:
        raise ValueError("AUC curve lacks initialization or nominal terminal point")
    if np.any(np.diff(x_arr) <= 0):
        raise ValueError("transition coordinates must be strictly increasing")
    if not (np.isfinite(x_arr).all() and np.isfinite(y_arr).all()):
        raise ValueError("AUC inputs must be finite")
    return float(np.trapezoid(y_arr, x_arr) / (x_arr[-1] - x_arr[0]))


def _exact_sign_flip_p(values: np.ndarray) -> float:
    if values.shape != (20,) or not np.isfinite(values).all():
        raise ValueError("the V3 sign-flip test requires 20 finite pairs")
    observed = abs(float(values.mean()))
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=20):
        statistic = abs(float(np.dot(signs, values) / 20.0))
        extreme += statistic >= observed - 1e-15
    return float(extreme / (2**20))


def _bootstrap_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(25_000)
    draws = values[rng.integers(0, 20, size=(20_000, 20))].mean(axis=1)
    return [float(value) for value in np.quantile(draws, (0.025, 0.975))]


def _assert_close(actual, expected, label: str, atol: float = 1e-12) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _verify_lock(artifact: dict, lock: dict) -> dict:
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
    }
    if runtime != lock["runtime"]:
        raise ValueError(f"analysis runtime differs from source lock: {runtime}")
    artifact_runtime = {
        key: artifact["provenance"][key] for key in runtime
    }
    if artifact_runtime != lock["runtime"]:
        raise ValueError("artifact runtime differs from source lock")
    recorded = artifact["provenance"]["source_sha256"]
    for relative, expected_hash in lock["source_sha256"].items():
        if recorded.get(relative) != expected_hash:
            raise ValueError(f"artifact source-lock mismatch for {relative}")
        if _sha256(PROJECT_ROOT / relative) != expected_hash:
            raise ValueError(f"current source-lock mismatch for {relative}")
    return {
        "passed": True,
        "runtime": runtime,
        "checked_source_files": sorted(lock["source_sha256"]),
    }


def verify(artifact: dict, lock: dict) -> dict:
    source_lock = _verify_lock(artifact, lock)
    protocol = artifact["protocol"]
    expected_protocol = {
        "stage": "core",
        "status": "confirmatory",
        "exploratory": False,
        "explicit_exploratory": False,
        "paired_seeds": EXPECTED_SEEDS,
        "n_rollouts": 16,
        "core_architectures": ["shared"],
        "core_analysis_mode": "two_cell_shared_efficacy_only",
        "transfer_claim_evaluated": False,
        "condition_names": list(EXPECTED_CASES),
        "eval_interval_transitions": 100_000,
        "eval_n_per_task": 32,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(
                f"protocol mismatch for {key}: {protocol.get(key)!r} != {expected!r}"
            )
    if protocol["budget"] != {
        "transition_budget": 2_000_000,
        "optimizer_update_budget": None,
        "transition_safety_cap": None,
    }:
        raise ValueError("unexpected V3 budget")
    if artifact.get("artifact_state") != "complete":
        raise ValueError("artifact is not complete")
    if artifact.get("run_failures"):
        raise ValueError("artifact contains failed runs")
    if tuple(artifact["cases"]) != EXPECTED_CASES:
        raise ValueError("artifact does not contain exactly the ordered V3 cases")

    recomputed: dict[str, list[float]] = {}
    per_case_checks = {}
    for case_name in EXPECTED_CASES:
        record = artifact["cases"][case_name]
        config = record["config"]
        expected_sampling = "uniform" if case_name.startswith("uniform") else "teacher"
        if (
            config["sampling"] != expected_sampling
            or config["architecture"] != "shared"
            or config["hidden_size"] != 64
            or config["learning_rate"] != 3e-4
            or config["hindsight_scale"] != 0.0
        ):
            raise ValueError(f"condition configuration mismatch for {case_name}")
        runs = record["runs"]
        if [run.get("seed") for run in runs] != EXPECTED_SEEDS:
            raise ValueError(f"seed order mismatch for {case_name}")
        aucs = []
        for run in runs:
            if not all(
                run.get(key) is True
                for key in (
                    "numeric_valid",
                    "accounting_valid",
                    "verifier_relabel_checks_valid",
                    "evaluation_cadence_invariant",
                )
            ):
                raise ValueError(f"invalid run in {case_name}, seed {run.get('seed')}")
            if run["total_parameters"] != 640 or run["active_parameters_per_task"] != 640:
                raise ValueError("shared H64 parameter-count invariant failed")
            auc = _normalized_auc(run["mean_pass_curve"], run["x_transitions"])
            _assert_close(auc, run[PRIMARY_METRIC], f"run AUC {case_name}")
            aucs.append(auc)
        recomputed[case_name] = aucs
        per_case_checks[case_name] = {
            "n_runs": len(runs),
            "all_valid": True,
            "mean_recomputed_auc": float(np.mean(aucs)),
        }

    differences = np.asarray(recomputed["teacher_shared_h64"]) - np.asarray(
        recomputed["uniform_shared_h64"]
    )
    mean_difference = float(differences.mean())
    p_value = _exact_sign_flip_p(differences)
    interval = _bootstrap_ci(differences)
    supported = bool(mean_difference >= 0.03 and p_value <= 0.05)

    saved = artifact["paired_core_contrasts"]["curriculum_efficacy_shared"]
    if saved["metric"] != PRIMARY_METRIC:
        raise ValueError("saved primary metric mismatch")
    if not np.allclose(
        saved["per_seed_contrast"], differences, rtol=0.0, atol=1e-12
    ):
        raise ValueError("saved paired differences do not reproduce")
    _assert_close(saved["mean_contrast"], mean_difference, "mean contrast")
    _assert_close(
        saved["exact_paired_sign_flip_p_two_sided"], p_value, "exact p-value"
    )
    if not np.allclose(
        saved["mean_ci95_paired_seed_bootstrap"], interval, rtol=0.0, atol=1e-12
    ):
        raise ValueError("saved bootstrap interval does not reproduce")
    decision = artifact["predeclared_core_decision"]
    if decision.get("efficacy_supported") is not supported:
        raise ValueError("saved efficacy decision does not reproduce")
    if decision.get("transfer_claim_evaluated") is not False:
        raise ValueError("artifact improperly evaluates a transfer claim")

    return {
        "schema": "curriculum-maxrl/acrobot-neural-v3-verification/v1",
        "all_checks_passed": True,
        "source_lock": source_lock,
        "protocol_checks_passed": True,
        "per_case": per_case_checks,
        "primary": {
            "metric": PRIMARY_METRIC,
            "estimand": "teacher_shared_h64 minus uniform_shared_h64",
            "n_pairs": 20,
            "per_seed_contrast": differences.tolist(),
            "mean_contrast": mean_difference,
            "bootstrap_ci95": interval,
            "exact_two_sided_sign_flip_p": p_value,
            "minimum_effect_threshold": 0.03,
            "efficacy_supported": supported,
            "transfer_claim_evaluated": False,
        },
    }


def _write_exclusive(path: Path, payload: dict, overwrite: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).with_name("ACROBOT_NEURAL_V3_LOCK.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    report = verify(artifact, lock)
    report["artifact"] = str(args.artifact.resolve())
    report["artifact_sha256"] = _sha256(args.artifact)
    report["lock"] = str(args.lock.resolve())
    report["lock_sha256"] = _sha256(args.lock)
    _write_exclusive(args.output, report, args.overwrite)
    primary = report["primary"]
    print(
        "V3 verification passed: "
        f"mean={primary['mean_contrast']:.6f}, "
        f"p={primary['exact_two_sided_sign_flip_p']:.6g}, "
        f"supported={primary['efficacy_supported']}"
    )


if __name__ == "__main__":
    main()
