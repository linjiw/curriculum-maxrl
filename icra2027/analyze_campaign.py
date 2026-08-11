"""Pre-specified paired analysis for ICRA navigation campaign artifacts."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


COMPARATORS = ("uniform", "learnability", "staged")


def exact_sign_flip_p(differences: np.ndarray) -> float | None:
    differences = np.asarray(differences, dtype=float)
    if len(differences) < 2:
        return None
    observed = abs(float(differences.mean()))
    exceed = 0
    total = 2 ** len(differences)
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistic = abs(float(np.mean(differences * np.asarray(signs))))
        exceed += statistic >= observed - 1e-15
    return exceed / total


def paired_bootstrap_ci(differences: np.ndarray, *, draws: int = 20_000,
                        seed: int = 20270811) -> list[float | None]:
    differences = np.asarray(differences, dtype=float)
    if len(differences) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    sample = rng.choice(differences, size=(draws, len(differences)),
                        replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(sample, [0.025, 0.975])]


def _seed_map(artifact: dict, arm: str) -> dict[int, dict]:
    return {int(row["seed"]): row for row in artifact["results"].get(arm, [])}


def auc_at_budget(run: dict, currency: str, budget: float) -> float:
    """Linearly interpolate a run's success curve to one common budget."""
    x = np.array([row[currency] for row in run["history"]], dtype=float)
    y = np.array([row["eval"]["mean_success"]
                  for row in run["history"]], dtype=float)
    if budget <= 0.0:
        return float(y[0])
    if budget > x[-1] + 1e-12:
        raise ValueError(f"run ends at {x[-1]} {currency}, below {budget}")
    keep = x < budget
    clipped_x = np.append(x[keep], budget)
    clipped_y = np.append(y[keep], np.interp(budget, x, y))
    return float(np.trapz(clipped_y, clipped_x) / budget)


def _paired_summary(ours: dict[int, dict], other: dict[int, dict],
                    currency: str) -> dict:
    paired_seeds = sorted(set(ours) & set(other))
    delta = []
    common_budgets = []
    for seed in paired_seeds:
        budget = min(float(ours[seed]["history"][-1][currency]),
                     float(other[seed]["history"][-1][currency]))
        common_budgets.append(budget)
        delta.append(auc_at_budget(ours[seed], currency, budget)
                     - auc_at_budget(other[seed], currency, budget))
    delta = np.asarray(delta, dtype=float)
    return {
        "currency": currency,
        "paired_seeds": paired_seeds,
        "common_budget_per_seed": common_budgets,
        "n": len(delta),
        "mean_delta": float(delta.mean()) if len(delta) else None,
        "positive": int(np.sum(delta > 0)),
        "ties": int(np.sum(delta == 0)),
        "paired_bootstrap_95_ci": paired_bootstrap_ci(delta),
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(delta),
        "per_seed_delta": delta.tolist(),
    }


def analyze(artifact: dict) -> dict:
    ours = _seed_map(artifact, "ours_uN")
    arms = {}
    for arm, rows in artifact["results"].items():
        episode_auc = np.array([
            row["target_uniform_auc_by_episode"] for row in rows], dtype=float)
        wall_auc = np.array([
            row["target_uniform_auc_by_own_training_wall"]
            for row in rows], dtype=float)
        arms[arm] = {
            "n": len(episode_auc),
            "episode_auc_mean": (float(episode_auc.mean())
                                 if len(episode_auc) else None),
            "episode_auc_sd": (float(episode_auc.std(ddof=1))
                               if len(episode_auc) > 1 else None),
            "own_horizon_wall_auc_mean": (float(wall_auc.mean())
                                          if len(wall_auc) else None),
            "final_mean_success": (float(np.mean([
                row["final"]["eval"]["mean_success"] for row in rows]))
                if rows else None),
            "final_dead_group_rate": (float(np.mean([
                row["final"]["dead_group_rate"] for row in rows]))
                if rows else None),
        }

    contrasts = {}
    for comparator in COMPARATORS:
        other = _seed_map(artifact, comparator)
        contrasts[f"ours_uN_minus_{comparator}"] = {
            "primary_matched_wall": _paired_summary(
                ours, other, "training_wall_seconds"),
            "co_primary_matched_sim_steps": _paired_summary(
                ours, other, "sim_steps"),
        }

    primary_rows = [row["primary_matched_wall"]
                    for row in contrasts.values()]
    enough = min((row["n"] for row in primary_rows), default=0) >= 5
    directional = all(
        row["mean_delta"] is not None and row["mean_delta"] >= 0.0
        for row in primary_rows)
    return {
        "analysis_schema_version": 1,
        "input_evidence_status": artifact.get("evidence_status"),
        "primary_metric": (
            "target-uniform mean-success AUC at paired common training wall time"),
        "co_primary_accounting_metric": (
            "target-uniform mean-success AUC at paired common simulator steps"),
        "arms": arms,
        "paired_contrasts": contrasts,
        "aug24_checkpoint": {
            "minimum_seed_requirement_met": enough,
            "directional_bar_met": directional,
            "decision_ready": (enough and artifact.get("evidence_status")
                               == "full_barn_campaign"),
            "note": ("Smoke runs validate plumbing only. A go/no-go decision "
                     "requires at least five full BARN seeds per arm."),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text())
    report = analyze(artifact)
    output = args.output or args.artifact.with_name(
        args.artifact.stem + "_analysis.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["aug24_checkpoint"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
