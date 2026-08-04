"""Post-hoc learning-rate sensitivity for the frontier full-CV control.

This is a robustness diagnostic, not a confirmatory estimator comparison.
It tests whether the common learning rate in ``run_estimator_variants.py``
conceals a reliable full-control-variate bootstrap regime.
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

from run_estimator_variants import run


DEFAULT_LRS = [0.01, 0.1, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--total-groups", type=int, default=3200)
    parser.add_argument("--n-rollouts", type=int, default=16)
    parser.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS)
    parser.add_argument("--unlock-threshold", type=float, default=0.1)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).with_name("results_fullcv_lr_sensitivity.json"))
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("--seeds must be at least 2 for sample-SD summaries")
    if args.total_groups < 1 or args.n_rollouts < 1:
        parser.error("--total-groups and --n-rollouts must be positive")

    here = Path(__file__).resolve().parent
    sources = [
        here / "run_fullcv_lr_sensitivity.py",
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
        "status": "post_hoc_sensitivity_not_confirmatory",
        "protocol": {
            "seeds": args.seeds,
            "seed_ids": list(range(args.seeds)),
            "regime": "frontier_heavy_levels_5_to_12",
            "maximum_initial_pass_rate": 1e-5,
            "total_groups": args.total_groups,
            "n_rollouts": args.n_rollouts,
            "estimator": "full_cv",
            "unlock_threshold_final_mean": args.unlock_threshold,
            "uncertainty": "sample SD across seeds (ddof=1)",
            "rng_streams": (
                "SeedSequence([replicate_id, 20260804]).spawn(2) creates "
                "separately spawned rollout and task-schedule streams"),
            "command": " ".join(sys.argv),
            "source_commit": commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "source_sha256": {p.name: file_hash(p) for p in sources},
        },
        "learning_rates": {},
    }

    for lr in args.lrs:
        rows = [
            run("full_cv", seed, (5, 12), total_groups=args.total_groups,
                n_rollouts=args.n_rollouts, lr=lr)
            for seed in range(args.seeds)
        ]
        auc = np.asarray([row["auc"] for row in rows])
        final = np.asarray([row["final_mean"] for row in rows])
        payload["learning_rates"][str(lr)] = {
            "auc_mean": float(auc.mean()),
            "auc_sd": float(auc.std(ddof=1)),
            "auc_median": float(np.median(auc)),
            "final_mean": float(final.mean()),
            "final_sd": float(final.std(ddof=1)),
            "final_median": float(np.median(final)),
            "unlocked_seeds": int((final > args.unlock_threshold).sum()),
            "best_final": float(final.max()),
            "auc_per_seed": auc.tolist(),
            "final_per_seed": final.tolist(),
        }
        row = payload["learning_rates"][str(lr)]
        print(
            f"lr={lr:>4g} AUC {row['auc_mean']:.6f}±{row['auc_sd']:.6f} "
            f"median {row['auc_median']:.6f}; unlocked "
            f"{row['unlocked_seeds']}/{args.seeds}")

    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
