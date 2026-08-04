"""Matched-budget control for the three MaxRL finite-group estimators.

This CPU experiment directly tests whether retaining Eq. (10)'s all-fail
control variate substitutes for verified hindsight creation. It compares:

* raw: success average, zero when K=0;
* full_cv: centered estimator retaining -1/N when K=0;
* practical: centered estimator dropping K=0 groups;
* GRPO and RLOO under the same realized task schedule; and
* practical+hindsight: the deployed estimator with exact prefix relabeling.

Every arm receives the same uniform task draws and generation budget. The
balanced pool contains levels 1--12; the frontier-heavy pool contains levels
5--12 and has maximum initial pass rate 1e-5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np

from estimators import (
    weights_grpo,
    weights_maxrl,
    weights_maxrl_full_cv,
    weights_maxrl_raw,
    weights_rloo,
)
from run_hindsight import correct_prefix_len
from testbed import SkillChainEnv


ESTIMATORS = {
    "raw": weights_maxrl_raw,
    "full_cv": weights_maxrl_full_cv,
    "practical": weights_maxrl,
    "grpo": weights_grpo,
    "rloo": weights_rloo,
}
REGIMES = {"balanced": (1, 12), "frontier_heavy": (5, 12)}
RNG_NAMESPACE = 20260804


def independent_streams(seed: int):
    """Return separately spawned rollout and task-schedule streams.

    A structured SeedSequence avoids the cross-replicate collisions created
    by additive offsets (for example, replicate ``s+5`` reusing replicate
    ``s``'s schedule stream as its rollout stream). Reusing the same replicate
    ID across arms still supplies common random numbers for paired contrasts.
    """
    root = np.random.SeedSequence([int(seed), RNG_NAMESPACE])
    return root.spawn(2)


def run(arm: str, seed: int, level_range: tuple[int, int], *,
        total_groups: int = 3200, n_rollouts: int = 16, lr: float = 0.5,
        eval_every: int = 400) -> dict:
    rollout_seed, schedule_seed = independent_streams(seed)
    env = SkillChainEnv(seed=rollout_seed)
    levels = np.asarray(env.task_level)
    lo, hi = level_range
    pool = np.asarray([t for t in range(env.n_tasks)
                       if lo <= levels[t] <= hi])
    rng = np.random.default_rng(schedule_seed)
    estimator = (weights_maxrl if arm == "practical+hindsight"
                 else ESTIMATORS[arm])
    history = []
    all_fail_groups = 0
    nonzero_all_fail_updates = 0
    relabeled_updates = 0

    for used in range(1, total_groups + 1):
        task_id = int(pool[rng.integers(len(pool))])
        actions, rewards = env.rollout(task_id, n_rollouts)
        all_fail = rewards.sum() == 0
        all_fail_groups += int(all_fail)
        weights = estimator(rewards)

        if np.any(weights != 0):
            nonzero_all_fail_updates += int(all_fail)
            env.apply_gradient(task_id, actions, weights, lr)
        elif arm == "practical+hindsight" and all_fail:
            prefixes = np.asarray([correct_prefix_len(a) for a in actions])
            prefix_len = int(prefixes.max())
            if prefix_len >= 1:
                target = ((task_id // env.n_levels) * env.n_levels
                          + prefix_len - 1)
                relabeled_rewards = (prefixes >= prefix_len).astype(float)
                relabeled_weights = weights_maxrl(relabeled_rewards)
                if np.any(relabeled_weights != 0):
                    env.apply_gradient(
                        target, actions[:, :prefix_len], relabeled_weights, lr)
                    relabeled_updates += 1

        if used % eval_every == 0:
            history.append(float(env.true_pass_rates()[pool].mean()))

    if total_groups % eval_every:
        history.append(float(env.true_pass_rates()[pool].mean()))

    return {
        "auc": float(np.mean(history)),
        "final_mean": history[-1],
        "trajectory": history,
        "all_fail_groups": all_fail_groups,
        "nonzero_all_fail_updates": nonzero_all_fail_updates,
        "relabeled_updates": relabeled_updates,
    }


def summarize(rows: list[dict]) -> dict:
    out = {}
    for key in ("auc", "final_mean", "all_fail_groups",
                "nonzero_all_fail_updates", "relabeled_updates"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_sd"] = float(values.std(ddof=1))
    out["auc_per_seed"] = [row["auc"] for row in rows]
    out["final_per_seed"] = [row["final_mean"] for row in rows]
    out["trajectories"] = [row["trajectory"] for row in rows]
    return out


def paired_summary(left: list[dict], right: list[dict], key: str) -> dict:
    """Paired left-minus-right summary plus an exact two-sided sign test."""
    delta = np.asarray([a[key] - b[key] for a, b in zip(left, right)], float)
    nonzero = delta[delta != 0]
    wins = int((nonzero > 0).sum())
    n = int(len(nonzero))
    tail = min(wins, n - wins)
    p_sign = (min(1.0, 2.0 * sum(math.comb(n, k) for k in range(tail + 1))
                  / (2**n)) if n else 1.0)
    return {
        "mean_delta": float(delta.mean()),
        "sd_delta": float(delta.std(ddof=1)),
        "positive_pairs": wins,
        "nonzero_pairs": n,
        "two_sided_exact_sign_p": p_sign,
        "delta_per_seed": delta.tolist(),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--total-groups", type=int, default=3200)
    parser.add_argument("--n-rollouts", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).with_name("results_estimator_variants.json"))
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("--seeds must be at least 2 for sample-SD summaries")
    if args.total_groups < 1:
        parser.error("--total-groups must be at least 1")

    here = Path(__file__).resolve().parent
    source_files = [
        here / "run_estimator_variants.py",
        here / "estimators.py",
        here / "testbed.py",
        here / "run_hindsight.py",
    ]
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "unavailable"

    payload = {
        "protocol": {
            "seeds": args.seeds,
            "total_groups": args.total_groups,
            "n_rollouts": args.n_rollouts,
            "learning_rate": args.lr,
            "sampling": "uniform with identical generation budget",
            "evaluation": "pool mean exact pass rate every 400 groups and at final group",
            "auc_definition": "unweighted mean of fixed-grid evaluations",
            "uncertainty": "sample SD across training seeds (ddof=1)",
            "seed_ids": list(range(args.seeds)),
            "common_random_numbers": "same seed aligns task draws and rollout uniforms across arms",
            "rng_streams": (
                "SeedSequence([replicate_id, 20260804]).spawn(2) creates "
                "separately spawned rollout and task-schedule streams"),
            "estimator_specific_tuning": False,
            "command": " ".join(sys.argv),
            "source_commit": source_commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "source_sha256": {p.name: sha256(p) for p in source_files},
        },
        "regimes": {},
    }

    for regime, level_range in REGIMES.items():
        payload["regimes"][regime] = {}
        print(f"\n--- {regime} (levels {level_range[0]}--{level_range[1]}) ---")
        rows_by_arm = {}
        for arm in [
                "raw", "full_cv", "practical", "grpo", "rloo",
                "practical+hindsight"]:
            rows = [
                run(arm, seed, level_range, total_groups=args.total_groups,
                    n_rollouts=args.n_rollouts, lr=args.lr)
                for seed in range(args.seeds)
            ]
            rows_by_arm[arm] = rows
            summary = summarize(rows)
            payload["regimes"][regime][arm] = summary
            print(
                f"{arm:21s} AUC {summary['auc_mean']:.4f}"
                f"±{summary['auc_sd']:.4f}  final "
                f"{summary['final_mean_mean']:.4f}"
                f"±{summary['final_mean_sd']:.4f}  nonzero K=0 "
                f"{summary['nonzero_all_fail_updates_mean']:.0f}")

        payload["regimes"][regime]["paired_contrasts"] = {
            "hindsight_minus_full_cv_auc": paired_summary(
                rows_by_arm["practical+hindsight"], rows_by_arm["full_cv"],
                "auc"),
            "hindsight_minus_full_cv_final": paired_summary(
                rows_by_arm["practical+hindsight"], rows_by_arm["full_cv"],
                "final_mean"),
            "full_cv_minus_practical_auc": paired_summary(
                rows_by_arm["full_cv"], rows_by_arm["practical"], "auc"),
            "practical_minus_grpo_auc": paired_summary(
                rows_by_arm["practical"], rows_by_arm["grpo"], "auc"),
            "practical_minus_rloo_auc": paired_summary(
                rows_by_arm["practical"], rows_by_arm["rloo"], "auc"),
        }

    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
