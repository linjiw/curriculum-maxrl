"""Schedule-matched estimator comparison with held-out learning-rate tuning.

This CPU control isolates estimator behavior on the balanced exact-score
skill-chain pool. Practical MaxRL, GRPO, and RLOO receive the same uniform
task schedule, rollout uniforms, group size, and generation budget. Because
their coefficient scales differ, learning rates are selected on disjoint
tuning seeds and evaluated on paired held-out seeds.

The experiment does not identify the adaptive neural-maze interaction. It is
a local, schedule-matched control with a transparent tuning split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np

from run_estimator_variants import paired_summary, run, summarize


LR_GRIDS = {
    "practical": [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0,
                  48.0, 64.0, 96.0],
    "grpo": [2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0,
             96.0, 128.0, 192.0, 256.0, 384.0],
    "rloo": [4.0, 8.0, 16.0, 32.0, 48.0, 64.0, 96.0, 128.0,
             192.0, 256.0, 384.0, 512.0, 768.0, 1024.0, 1536.0],
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-seeds", type=int, default=10)
    parser.add_argument("--test-seeds", type=int, default=20)
    parser.add_argument("--total-groups", type=int, default=3200)
    parser.add_argument("--n-rollouts", type=int, default=16)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).with_name(
            "results_schedule_matched_estimators.json"))
    args = parser.parse_args()
    if args.tune_seeds < 2 or args.test_seeds < 2:
        parser.error("tuning and test seed counts must each be at least two")
    if args.total_groups < 1 or args.n_rollouts < 1:
        parser.error("total groups and rollout count must be positive")

    tune_ids = list(range(1000, 1000 + args.tune_seeds))
    test_ids = list(range(2000, 2000 + args.test_seeds))
    level_range = (1, 12)
    tuning = {}
    selected = {}

    for estimator, grid in LR_GRIDS.items():
        tuning[estimator] = {}
        for lr in grid:
            rows = [
                run(estimator, seed, level_range,
                    total_groups=args.total_groups,
                    n_rollouts=args.n_rollouts, lr=lr)
                for seed in tune_ids
            ]
            auc = np.asarray([row["auc"] for row in rows], float)
            tuning[estimator][str(lr)] = {
                "auc_mean": float(auc.mean()),
                "auc_sd": float(auc.std(ddof=1)),
                "auc_per_seed": auc.tolist(),
            }
        # One-standard-error rule: among rates whose tuning mean is within
        # one SE of the best observed mean, choose the smallest. This avoids
        # selecting an unnecessarily saturating step merely because a toy
        # exact-score trajectory differs in its last decimal place.
        best_lr = max(
            grid, key=lambda lr: (tuning[estimator][str(lr)]["auc_mean"], -lr))
        best_row = tuning[estimator][str(best_lr)]
        cutoff = (best_row["auc_mean"]
                  - best_row["auc_sd"] / np.sqrt(args.tune_seeds))
        selected[estimator] = min(
            lr for lr in grid
            if tuning[estimator][str(lr)]["auc_mean"] >= cutoff)
        tuning[estimator]["selection"] = {
            "best_observed_learning_rate": best_lr,
            "best_mean_auc": best_row["auc_mean"],
            "one_standard_error_cutoff": float(cutoff),
            "selected_learning_rate": selected[estimator],
            "selected_is_grid_boundary": selected[estimator] in (grid[0], grid[-1]),
        }

    test_rows = {}
    test_summary = {}
    for estimator in LR_GRIDS:
        lr = selected[estimator]
        rows = [
            run(estimator, seed, level_range,
                total_groups=args.total_groups,
                n_rollouts=args.n_rollouts, lr=lr)
            for seed in test_ids
        ]
        test_rows[estimator] = rows
        test_summary[estimator] = summarize(rows)
        test_summary[estimator]["selected_learning_rate"] = lr

    here = Path(__file__).resolve().parent
    sources = [
        here / "run_schedule_matched_estimators.py",
        here / "run_estimator_variants.py",
        here / "estimators.py",
        here / "testbed.py",
    ]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"

    payload = {
        "status": "schedule_matched_exact_score_control",
        "scope": (
            "Balanced tabular skill-chain pool; does not identify the "
            "adaptive neural-maze curriculum-by-estimator interaction."),
        "protocol": {
            "tuning_seed_ids": tune_ids,
            "held_out_test_seed_ids": test_ids,
            "total_groups": args.total_groups,
            "n_rollouts": args.n_rollouts,
            "task_schedule": "uniform; identical seed aligns task IDs across estimators",
            "rollout_randomness": "common random numbers within each paired seed",
            "rng_streams": (
                "SeedSequence([replicate_id, 20260804]).spawn(2) creates "
                "separately spawned rollout and task-schedule streams"),
            "evaluation": "exact pool mean every 400 groups; AUC is the unweighted grid mean",
            "uncertainty": "sample SD across held-out training seeds (ddof=1)",
            "selection": (
                "smallest learning rate within one standard error of the "
                "best tuning-seed mean AUC"),
            "selection_caveat": (
                "the smaller-raw-rate preference is parameterization-"
                "dependent; estimator ordering is not interpreted as "
                "intrinsic"),
            "command": " ".join(sys.argv),
            "source_commit": commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "source_sha256": {p.name: file_hash(p) for p in sources},
        },
        "learning_rate_grids": LR_GRIDS,
        "tuning": tuning,
        "held_out_test": test_summary,
        "paired_contrasts": {
            "practical_minus_grpo_auc": paired_summary(
                test_rows["practical"], test_rows["grpo"], "auc"),
            "practical_minus_rloo_auc": paired_summary(
                test_rows["practical"], test_rows["rloo"], "auc"),
            "grpo_minus_rloo_auc": paired_summary(
                test_rows["grpo"], test_rows["rloo"], "auc"),
        },
    }
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    print("selected learning rates:", selected)
    for estimator, row in test_summary.items():
        print(
            f"{estimator:10s} AUC {row['auc_mean']:.4f}±{row['auc_sd']:.4f}; "
            f"final {row['final_mean_mean']:.4f}±{row['final_mean_sd']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
