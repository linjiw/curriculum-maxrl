#!/usr/bin/env python3
"""Aggregate paired development seeds for the exact UniLab Stewart pilot."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = tuple(
    ROOT / "frontier_rl" / "examples" / f"unilab_stewart_grouped_fixed_radius_seed{seed}_dev.json"
    for seed in range(3)
)
DEFAULT_OUTPUT = (
    ROOT / "frontier_rl" / "examples" / "unilab_stewart_grouped_fixed_radius_analysis_dev.json"
)
ARMS = ("uniform", "learnability", "advmass")


def exact_sign_flip_p(differences: np.ndarray) -> float:
    """Exact two-sided paired sign-flip p-value for a small seed family."""
    observed = abs(float(differences.mean()))
    null = [
        abs(float(np.mean(differences * np.asarray(signs, dtype=float))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1.0e-15))


def bootstrap_interval(
    differences: np.ndarray, *, samples: int = 20_000, seed: int = 202_607_23
) -> list[float]:
    """Descriptive paired-seed percentile interval; not confirmatory inference."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def _mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def analyze(paths: list[Path]) -> dict:
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(artifacts) < 2:
        raise ValueError("analysis requires at least two paired development seeds")

    reference_config = dict(artifacts[0]["config"])
    reference_config.pop("seed")
    warm_sha = artifacts[0]["warm_start"]["sha256"]
    rows: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    seeds = []
    for path, artifact in zip(paths, artifacts, strict=True):
        config = dict(artifact["config"])
        seed = int(config.pop("seed"))
        if config != reference_config:
            raise ValueError(f"non-seed config mismatch in {path}")
        if artifact["warm_start"]["sha256"] != warm_sha:
            raise ValueError(f"warm-start mismatch in {path}")
        by_arm = {run["arm"]: run for run in artifact["runs"]}
        if set(by_arm) != set(ARMS):
            raise ValueError(f"arm mismatch in {path}")
        seeds.append(seed)
        for arm in ARMS:
            summary = by_arm[arm]["summary"]
            rows[arm].append(
                {
                    "seed": seed,
                    "transition_auc": float(summary["normalized_transition_auc"]),
                    "final_mean_pass_rate": float(summary["final_mean_pass_rate"]),
                    "policy_updates": int(summary["policy_updates"]),
                    "mixed_groups": int(summary["mixed_groups"]),
                    "all_fail_groups": int(summary["all_fail_groups"]),
                    "all_pass_groups": int(summary["all_pass_groups"]),
                    "realized_coefficient_mass": float(summary["realized_coefficient_mass"]),
                    "backend_env_steps": int(summary["backend_env_steps"]),
                }
            )

    seeds = sorted(seeds)
    if seeds != list(range(min(seeds), min(seeds) + len(seeds))):
        raise ValueError(f"expected consecutive paired seeds, got {seeds}")
    per_arm = {}
    metric_keys = (
        "transition_auc",
        "final_mean_pass_rate",
        "policy_updates",
        "mixed_groups",
        "all_fail_groups",
        "realized_coefficient_mass",
    )
    for arm in ARMS:
        ordered = sorted(rows[arm], key=lambda row: row["seed"])
        per_arm[arm] = {
            "per_seed": ordered,
            "aggregate": {
                key: _mean_sd([float(row[key]) for row in ordered]) for key in metric_keys
            },
        }

    contrasts = {}
    for treatment, control in (
        ("learnability", "uniform"),
        ("advmass", "uniform"),
        ("advmass", "learnability"),
    ):
        key = f"{treatment}_minus_{control}"
        differences = np.asarray(
            [
                per_arm[treatment]["per_seed"][index]["transition_auc"]
                - per_arm[control]["per_seed"][index]["transition_auc"]
                for index in range(len(seeds))
            ],
            dtype=float,
        )
        contrasts[key] = {
            "per_seed_auc_difference": differences.tolist(),
            "mean_auc_difference": float(differences.mean()),
            "sample_sd": float(differences.std(ddof=1)),
            "paired_bootstrap_95_interval_descriptive": bootstrap_interval(differences),
            "exact_two_sided_sign_flip_p": exact_sign_flip_p(differences),
        }

    return {
        "status": "three-paired-seed development analysis; not confirmatory",
        "claim_boundary": (
            "pipeline learning and estimator-mechanics diagnostic only; seed count is "
            "insufficient to rank teachers"
        ),
        "input_artifacts": [str(path.relative_to(ROOT)) for path in paths],
        "paired_seeds": seeds,
        "warm_start_sha256": warm_sha,
        "shared_config_excluding_seed": reference_config,
        "arms": per_arm,
        "auc_contrasts": contrasts,
        "interpretation": {
            "pipeline": "all arms improved substantially from the common warm start",
            "mechanism": (
                "advmass produced more mixed/update-bearing groups, fewer all-fail groups, "
                "and more realized scalar coefficient mass than uniform"
            ),
            "performance": (
                "learnability had the highest mean transition AUC; teacher ordering varied "
                "by seed, so no performance superiority claim is supported"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args.inputs)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
