"""Post-hoc mechanism and native-task audit of the frozen Acrobot V3 runs.

This script performs no training.  It derives realized practical-MaxRL
coefficient mass from the saved group success counts and recomputes standard
Acrobot success/return AUCs from the frozen evaluation curves.  Every
inferential quantity added here is explicitly post hoc and descriptive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = HERE / "acrobot_neural_v3_shared_confirmatory.json"
DEFAULT_LOCK = HERE / "ACROBOT_NEURAL_V3_LOCK.json"
EXPECTED_CASES = ("uniform_shared_h64", "teacher_shared_h64")
EXPECTED_SEEDS = list(range(12_000, 12_020))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_trapezoid(values: list[float], coordinates: list[int]) -> float:
    x = np.asarray(coordinates, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("AUC inputs must be equal-length one-dimensional curves")
    if x[0] != 0 or np.any(np.diff(x) <= 0):
        raise ValueError("AUC coordinates must start at zero and strictly increase")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("AUC inputs must be finite")
    area = np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0)
    return float(area / (x[-1] - x[0]))


def practical_maxrl_mass(success_count: int, group_size: int) -> float:
    """Observed sum_i |r_i/K - 1/N| with both extreme groups dropped."""
    if not 0 <= success_count <= group_size or group_size < 1:
        raise ValueError("success count must lie in [0, group_size]")
    if success_count in (0, group_size):
        return 0.0
    return 2.0 * (group_size - success_count) / group_size


def exact_two_sided_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not 1 <= len(values) <= 20 or not np.isfinite(values).all():
        raise ValueError("the Acrobot audit requires 1..20 finite pairs")
    signed_sums = np.zeros(1, dtype=np.float64)
    for value in values:
        signed_sums = np.concatenate((signed_sums - value, signed_sums + value))
    observed = abs(float(values.mean()))
    statistics = np.abs(signed_sums / len(values))
    return float(np.mean(statistics >= observed - 1e-15))


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[
        rng.integers(0, len(values), size=(20_000, len(values)))
    ].mean(axis=1)
    return [float(v) for v in np.quantile(draws, (0.025, 0.975))]


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _run_metrics(run: dict, group_size: int) -> dict[str, float]:
    groups = run["group_diagnostics"]
    if len(groups) != run["sampled_groups"]:
        raise ValueError("saved group count does not match sampled_groups")
    if not groups or groups[0]["transition_start"] != 0:
        raise ValueError("group transition ledger lacks its zero origin")
    previous_end = 0
    masses = []
    for index, group in enumerate(groups, start=1):
        if group["group"] != index or group["transition_start"] != previous_end:
            raise ValueError("group transition ledger is not contiguous")
        if group["transition_end"] - group["transition_start"] != group["n_transitions"]:
            raise ValueError("group transition count is inconsistent")
        previous_end = group["transition_end"]
        count = int(group["success_count"])
        expected_regime = (
            "dead" if count == 0 else "all_pass" if count == group_size else "mixed"
        )
        if group["regime"] != expected_regime:
            raise ValueError("saved group regime disagrees with its success count")
        masses.append(practical_maxrl_mass(count, group_size))
    if previous_end != run["transitions"]:
        raise ValueError("group transition ledger does not reach the run total")
    mass = np.asarray(masses, dtype=np.float64)
    nonzero_groups = int(np.count_nonzero(mass))
    if nonzero_groups != run["live_groups"]:
        raise ValueError("nonzero coefficient-mass groups disagree with live_groups")
    if run["relabeled_groups"] != 0:
        raise ValueError("V3 mechanism audit requires hindsight-free runs")

    return {
        "coefficient_mass_per_group": float(mass.mean()),
        "coefficient_mass_per_million_transitions": float(
            mass.sum() * 1_000_000.0 / run["transitions"]
        ),
        "nonzero_group_fraction": float(nonzero_groups / len(groups)),
        "live_groups": float(run["live_groups"]),
        "all_pass_groups": float(run["all_pass_groups"]),
        "optimizer_updates_per_million_transitions": float(
            run["optimizer_updates"] * 1_000_000.0 / run["transitions"]
        ),
        "mean_transitions_per_episode": float(
            run["transitions"] / (len(groups) * group_size)
        ),
        "native_success_auc": normalized_trapezoid(
            run["native_success_rate_curve"], run["x_transitions"]
        ),
        "native_return_auc": normalized_trapezoid(
            run["mean_native_return_curve"], run["x_transitions"]
        ),
        "final_native_success_rate": float(run["final_native_success_rate"]),
        "target_uniform_auc": normalized_trapezoid(
            run["mean_pass_curve"], run["x_transitions"]
        ),
    }


def _paired_summary(
    uniform: dict[int, dict[str, float]],
    teacher: dict[int, dict[str, float]],
    metric: str,
    seed: int,
) -> dict:
    uniform_values = np.asarray([uniform[s][metric] for s in EXPECTED_SEEDS])
    teacher_values = np.asarray([teacher[s][metric] for s in EXPECTED_SEEDS])
    differences = teacher_values - uniform_values
    return {
        "estimand": "u16 curriculum minus uniform",
        "n_paired_seeds": 20,
        "uniform_mean": float(uniform_values.mean()),
        "u16_mean": float(teacher_values.mean()),
        "mean_paired_difference": float(differences.mean()),
        "positive_pairs": int(np.sum(differences > 0)),
        "ties": int(np.sum(differences == 0)),
        "negative_pairs": int(np.sum(differences < 0)),
        "paired_differences": differences.tolist(),
        "posthoc_bootstrap_ci95": bootstrap_ci(differences, seed),
        "posthoc_exact_two_sided_sign_flip_p": exact_two_sided_sign_flip_p(
            differences
        ),
    }


def analyze(artifact: dict, lock: dict, artifact_path: Path, lock_path: Path) -> dict:
    protocol = artifact["protocol"]
    if artifact.get("artifact_state") != "complete" or artifact.get("run_failures"):
        raise ValueError("Acrobot V3 artifact is not a complete 40-run record")
    if tuple(artifact["cases"]) != EXPECTED_CASES:
        raise ValueError("unexpected Acrobot V3 case set or ordering")
    if protocol.get("paired_seeds") != EXPECTED_SEEDS:
        raise ValueError("unexpected Acrobot V3 paired seeds")
    if protocol.get("n_rollouts") != 16:
        raise ValueError("unexpected Acrobot V3 group size")
    if protocol.get("budget") != {
        "transition_budget": 2_000_000,
        "optimizer_update_budget": None,
        "transition_safety_cap": None,
    }:
        raise ValueError("unexpected Acrobot V3 budget")

    recorded_hashes = artifact["provenance"]["source_sha256"]
    if recorded_hashes != lock["source_sha256"]:
        raise ValueError("artifact hashes do not match the frozen V3 source lock")

    by_case: dict[str, dict[int, dict[str, float]]] = {}
    for case_name in EXPECTED_CASES:
        case = artifact["cases"][case_name]
        if case["config"]["hindsight_scale"] != 0.0:
            raise ValueError("Acrobot V3 is expected to be hindsight-free")
        if [run["seed"] for run in case["runs"]] != EXPECTED_SEEDS:
            raise ValueError(f"seed order mismatch in {case_name}")
        by_case[case_name] = {}
        for run in case["runs"]:
            if not all(
                run.get(key) is True
                for key in (
                    "numeric_valid",
                    "accounting_valid",
                    "verifier_relabel_checks_valid",
                    "evaluation_cadence_invariant",
                )
            ):
                raise ValueError(f"invalid saved run in {case_name}")
            metrics = _run_metrics(run, protocol["n_rollouts"])
            _assert_close(
                metrics["target_uniform_auc"],
                run["auc_mean_pass_by_transitions"],
                f"target-uniform AUC for {case_name}, seed {run['seed']}",
            )
            by_case[case_name][run["seed"]] = metrics

    uniform = by_case["uniform_shared_h64"]
    teacher = by_case["teacher_shared_h64"]
    primary = _paired_summary(uniform, teacher, "target_uniform_auc", 26_080_700)
    saved_primary = artifact["paired_core_contrasts"]["curriculum_efficacy_shared"]
    _assert_close(
        primary["mean_paired_difference"], saved_primary["mean_contrast"],
        "registered primary contrast",
    )
    if not np.allclose(
        primary["paired_differences"], saved_primary["per_seed_contrast"],
        rtol=0.0, atol=1e-12,
    ):
        raise ValueError("registered primary paired differences do not reproduce")

    primary_differences = np.asarray(
        primary["paired_differences"], dtype=np.float64
    )
    alternating_seed_sensitivity = {}
    for label, indices in {
        "even_logical_seeds": np.arange(0, 20, 2),
        "odd_logical_seeds": np.arange(1, 20, 2),
    }.items():
        values = primary_differences[indices]
        alternating_seed_sensitivity[label] = {
            "seeds": [EXPECTED_SEEDS[index] for index in indices],
            "n": len(values),
            "mean_paired_difference": float(values.mean()),
            "positive_pairs": int(np.sum(values > 0.0)),
            "exact_two_sided_sign_flip_p_descriptive": (
                exact_two_sided_sign_flip_p(values)
            ),
        }

    posthoc_metrics = (
        "coefficient_mass_per_group",
        "coefficient_mass_per_million_transitions",
        "nonzero_group_fraction",
        "optimizer_updates_per_million_transitions",
        "mean_transitions_per_episode",
        "native_success_auc",
        "native_return_auc",
        "final_native_success_rate",
    )
    posthoc = {
        metric: _paired_summary(uniform, teacher, metric, 26_080_701 + index)
        for index, metric in enumerate(posthoc_metrics)
    }
    return {
        "schema": "curriculum-maxrl/acrobot-v3-posthoc-mechanism/v2",
        "status": "historical_descriptive_after_rng_domain_audit",
        "new_training_performed": False,
        "claim_boundary": (
            "A later audit found cross-seed reuse of one numeric RNG root, so "
            "the original paired inference is not treated as clean confirmation. "
            "All quantities in this file are historical or post-hoc descriptive "
            "summaries and cannot establish a causal mediator."
        ),
        "input": {
            "artifact": artifact_path.name,
            "artifact_sha256": sha256(artifact_path),
            "source_lock": lock_path.name,
            "source_lock_sha256": sha256(lock_path),
            "embedded_source_hashes_match_lock": True,
            "frozen_runtime": lock["runtime"],
        },
        "protocol": {
            "environment": protocol["gymnasium_environment"],
            "task_thresholds": protocol["thresholds"],
            "hardest_threshold_is_native_success": True,
            "group_size": protocol["n_rollouts"],
            "nominal_transitions_per_run": 2_000_000,
            "paired_seeds": EXPECTED_SEEDS,
            "hindsight": False,
            "shared_neural_policy_parameters": 640,
        },
        "registered_primary_anchor": primary,
        "rng_domain_audit": {
            "finding": (
                "For logical seed s, the historical actor used parameter root s "
                "and action root s+1; therefore action root s+1 equals the "
                "parameter root of neighboring logical seed s+1."
            ),
            "consequence": (
                "The 20 neighboring paired units are not cleanly domain-separated, "
                "so the original sign-exchangeability argument is not retained."
            ),
            "alternating_seed_sensitivity_descriptive": alternating_seed_sensitivity,
            "repair": (
                "Use globally unique, independently reconstructed domain roots and "
                "fresh logical seeds in the replacement three-arm tournament."
            ),
        },
        "posthoc": posthoc,
        "interpretation": {
            "mechanism": (
                "The u16 curriculum exposed the practical MaxRL learner to more "
                "nonzero coefficient mass both per sampled group and per paid "
                "environment transition."
            ),
            "standard_task": (
                "The same paired runs improved native Acrobot success and return "
                "AUC; these were protocol-listed secondary outcomes."
            ),
            "not_established": [
                "causal mediation by coefficient mass",
                "superiority to p(1-p), PLR, ALP, PAIRED, or ACCEL",
                "generalization beyond this fixed Acrobot threshold family",
                "estimator robustness beyond practical MaxRL at the fixed learning rate",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    result = analyze(artifact, lock, args.artifact, args.lock)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.check is not None:
        if args.check.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"derived audit differs from {args.check}")
        print(f"Acrobot V3 mechanism audit matches {args.check}")
        return
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
