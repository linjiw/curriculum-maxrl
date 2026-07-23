"""Transition-matched Gymnasium validation of curriculum transfer.

This is the reproducible counterpart to the MountainCar shared-policy claim.
The tasks are nested binary predicates (reach x >= target), while one
task-agnostic policy is shared across every predicate.  We compare:

* flag-only sparse training;
* uniform curriculum;
* exact practical-MaxRL coefficient-mass priority;
* that teacher plus centered hindsight;
* that teacher plus success-only hindsight; and
* a per-bin-parameter negative control.

Evaluation preserves the training RNG state, and runs are matched by actual
Gymnasium transitions rather than rollout-group count.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
import gymnasium

from frontier_rl import FrontierTeacher, FrontierTrainer, TrainerConfig
from frontier_rl.adapters.gym_classic import MountainCarSpace


@dataclass(frozen=True)
class Case:
    name: str
    sampling: str
    shared: bool
    hindsight: bool = False
    hindsight_estimator: str = "maxrl"
    gamma: float = 1.0


CASES = (
    Case("flag_only_shared", "flag_only", True),
    Case("uniform_shared", "uniform", True),
    Case("advmass_gamma1_shared", "advmass", True),
    Case("legacy_frontier_gamma1_shared", "legacy_frontier", True),
    Case("learnability_gamma1_shared", "learnability", True),
    Case("advmass_shared", "advmass", True, gamma=4.0),
    Case("advmass_shared_hindsight", "advmass", True, True, "maxrl", 4.0),
    Case("advmass_shared_success_only_hindsight", "advmass", True, True,
         "success_only", 4.0),
    Case("advmass_per_bin_hindsight_control", "advmass", False, True,
         "maxrl", 4.0),
)


def normalized_trapezoid(y: list[float], x: list[int]) -> float:
    if len(x) < 2 or x[-1] <= x[0]:
        return float(y[-1])
    area = sum((x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2
               for i in range(1, len(x)))
    return float(area / (x[-1] - x[0]))


def run_case(case: Case, seed: int, transition_budget: int,
             eval_interval: int, eval_n: int, n_rollouts: int = 16,
             eval_seed_base: int = 1_000_000) -> dict:
    env = MountainCarSpace(seed=seed, share_policy_across_tasks=case.shared)
    cfg = TrainerConfig(
        n_rollouts=n_rollouts,
        tasks_per_step=1,
        hindsight=case.hindsight,
        hindsight_estimator=case.hindsight_estimator,
        teacher_gamma=case.gamma,
        seed=seed,
    )
    teacher = FrontierTeacher(
        env.n_tasks, n_rollouts, decay=cfg.teacher_decay,
        floor=cfg.teacher_floor, gamma=case.gamma, seed=seed + 1000,
    )
    if case.sampling == "uniform":
        teacher.distribution = lambda: np.full(env.n_tasks, 1.0 / env.n_tasks)
    elif case.sampling == "flag_only":
        flag = np.zeros(env.n_tasks)
        flag[-1] = 1.0
        teacher.distribution = lambda: flag
    elif case.sampling == "legacy_frontier":
        teacher.utility = lambda p: (
            (1.0 - (1.0 - p) ** n_rollouts) * (1.0 - p)
        )
    elif case.sampling == "learnability":
        teacher.utility = lambda p: p * (1.0 - p)
    elif case.sampling != "advmass":
        raise ValueError(f"unknown sampling rule: {case.sampling}")

    trainer = FrontierTrainer(env, env, cfg, teacher=teacher)
    transitions = 0
    groups = 0
    dead = 0
    all_pass = 0
    relabeled = 0
    xs = [0]
    eval_seed = eval_seed_base + seed
    initial = env.eval_pass_rates(n=eval_n, seed=eval_seed)
    means = [float(initial.mean())]
    flags = [float(initial[-1])]
    next_eval = eval_interval

    try:
        while transitions < transition_budget:
            stats = trainer.step()
            transitions += stats.env_steps
            groups += stats.live_groups + stats.dead_groups + stats.all_pass_groups
            dead += stats.dead_groups
            all_pass += stats.all_pass_groups
            relabeled += stats.relabeled_groups
            if transitions >= next_eval or transitions >= transition_budget:
                p = env.eval_pass_rates(n=eval_n, seed=eval_seed)
                xs.append(transitions)
                means.append(float(p.mean()))
                flags.append(float(p[-1]))
                while next_eval <= transitions:
                    next_eval += eval_interval
    finally:
        env.close()

    return {
        "seed": seed,
        "transitions": transitions,
        "groups": groups,
        "dead_groups": dead,
        "all_pass_groups": all_pass,
        "relabeled_groups": relabeled,
        "x_transitions": xs,
        "mean_pass_curve": means,
        "flag_pass_curve": flags,
        "auc_mean_pass": normalized_trapezoid(means, xs),
        "final_mean_pass": means[-1],
        "final_flag_pass": flags[-1],
    }


def bootstrap_mean_ci(values, seed: int = 0,
                      n_boot: int = 20_000) -> list[float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return [float(values.mean()), float(values.mean())]
    rng = np.random.default_rng(seed)
    samples = values[
        rng.integers(0, len(values), size=(n_boot, len(values)))
    ].mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def summarize(runs: list[dict]) -> dict:
    keys = ("auc_mean_pass", "final_mean_pass", "final_flag_pass",
            "transitions", "dead_groups", "all_pass_groups", "relabeled_groups")
    out = {"n_seeds": len(runs)}
    for key in keys:
        values = np.asarray([run[key] for run in runs], dtype=float)
        out[key] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "mean_ci95_bootstrap": bootstrap_mean_ci(values),
        }
    return out


def paired_comparison(a: list[dict], b: list[dict], key: str) -> dict:
    """Summarize paired ``a-b`` effects for identically ordered seeds."""
    seeds_a = [run["seed"] for run in a]
    seeds_b = [run["seed"] for run in b]
    if len(a) != len(b) or seeds_a != seeds_b:
        raise ValueError(
            f"paired runs must have identical ordered seeds: {seeds_a} != {seeds_b}"
        )
    delta = np.asarray([x[key] - y[key] for x, y in zip(a, b)], dtype=float)
    observed = abs(float(delta.mean()))
    null_means = (
        abs(sum(sign * value for sign, value in zip(signs, delta)) / len(delta))
        for signs in itertools.product((-1.0, 1.0), repeat=len(delta))
    )
    sign_flip_p = sum(value >= observed - 1e-15 for value in null_means) / (
        2 ** len(delta)
    )
    return {
        "metric": key,
        "mean_delta": float(delta.mean()),
        "sample_std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
        "mean_delta_ci95_bootstrap": bootstrap_mean_ci(delta, seed=17),
        "exact_sign_flip_p_two_sided": float(sign_flip_p),
        "per_seed_delta": delta.tolist(),
    }


PAIR_SPECS = (
    ("advmass_gamma1_minus_uniform", "advmass_gamma1_shared",
     "uniform_shared"),
    ("advmass_gamma4_minus_uniform", "advmass_shared", "uniform_shared"),
    ("gamma4_minus_gamma1", "advmass_shared", "advmass_gamma1_shared"),
    ("exact_minus_legacy_gamma1", "advmass_gamma1_shared",
     "legacy_frontier_gamma1_shared"),
    ("exact_minus_learnability_gamma1", "advmass_gamma1_shared",
     "learnability_gamma1_shared"),
    ("centered_hindsight_minus_no_hindsight",
     "advmass_shared_hindsight", "advmass_shared"),
    ("success_only_hindsight_minus_no_hindsight",
     "advmass_shared_success_only_hindsight", "advmass_shared"),
    ("centered_minus_success_only", "advmass_shared_hindsight",
     "advmass_shared_success_only_hindsight"),
    ("shared_minus_per_bin_centered", "advmass_shared_hindsight",
     "advmass_per_bin_hindsight_control"),
)


def attach_paired_analysis(result: dict) -> None:
    """Attach paired effects and a Holm correction over the AUC family."""
    result["paired_comparisons"] = {}
    for label, a_name, b_name in PAIR_SPECS:
        a = result["cases"][a_name]["runs"]
        b = result["cases"][b_name]["runs"]
        result["paired_comparisons"][label] = {
            key: paired_comparison(a, b, key)
            for key in ("auc_mean_pass", "final_mean_pass", "final_flag_pass")
        }

    ordered = sorted(
        (metrics["auc_mean_pass"]["exact_sign_flip_p_two_sided"], label)
        for label, metrics in result["paired_comparisons"].items()
    )
    m = len(ordered)
    running_adjusted = 0.0
    still_rejecting = True
    holm = {}
    for rank, (p_value, label) in enumerate(ordered, start=1):
        running_adjusted = max(running_adjusted, (m - rank + 1) * p_value)
        threshold = 0.05 / (m - rank + 1)
        reject = still_rejecting and p_value <= threshold
        if not reject:
            still_rejecting = False
        holm[label] = {
            "raw_p": float(p_value),
            "adjusted_p": float(min(running_adjusted, 1.0)),
            "reject_familywise_0.05": reject,
        }
    result["auc_holm_bonferroni"] = holm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transition-budget", type=int, default=500_000)
    ap.add_argument("--eval-interval", type=int, default=100_000)
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--reanalyze-existing", action="store_true",
        help="recompute paired statistics in --output without retraining",
    )
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.quick:
        args.transition_budget = min(args.transition_budget, 100_000)
        args.eval_interval = min(args.eval_interval, 25_000)
        args.eval_n = min(args.eval_n, 12)
        args.seeds = min(args.seeds, 2)
    if args.output is None:
        filename = ("mountaincar_shared_quick.json" if args.quick else
                    "mountaincar_shared_validation.json")
        args.output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), filename
        )
    if args.reanalyze_existing:
        with open(args.output) as f:
            result = json.load(f)
        attach_paired_analysis(result)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"reanalyzed {args.output}")
        return

    source_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(source_dir, "..", ".."))
    source_files = [
        os.path.abspath(__file__),
        os.path.join(repo_dir, "frontier_rl", "adapters", "gym_classic.py"),
        os.path.join(repo_dir, "frontier_rl", "trainer.py"),
        os.path.join(repo_dir, "frontier_rl", "teacher.py"),
        os.path.join(repo_dir, "frontier_rl", "estimators.py"),
        os.path.join(repo_dir, "frontier_rl", "interfaces.py"),
    ]
    source_sha256 = {}
    for path in source_files:
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        source_sha256[os.path.relpath(path, repo_dir)] = digest

    result = {
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True,
                text=True, check=False,
            ).stdout.strip() or None,
            "git_worktree_dirty": bool(subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True,
                text=True, check=False,
            ).stdout.strip()),
            "source_sha256": source_sha256,
        },
        "protocol": {
            "transition_budget": args.transition_budget,
            "eval_interval": args.eval_interval,
            "eval_n": args.eval_n,
            "seeds": args.seeds,
            "n_rollouts": 16,
            "tasks_per_step": 1,
            "teacher_decay": 0.7,
            "teacher_floor": 0.1,
            "hindsight_scale": 1.0,
            "max_episode_steps": 200,
            "targets": np.linspace(-0.35, 0.5, 10).tolist(),
            "tile_bins": [12, 12],
            "policy_learning_rate": 0.15,
            "evaluation": (
                "fixed per-seed common random numbers; training RNG restored"
            ),
            "summary_std": "sample standard deviation (ddof=1)",
            "auc": (
                "normalized trapezoid over actual transition checkpoints; "
                "the final complete group may overshoot the nominal budget"
            ),
            "gymnasium_task": "MountainCar-v0 dynamics; custom nested binary thresholds",
            "relabel_stopping": (
                "successful relabeled traces stop at first credited threshold crossing"
            ),
        },
        "cases": {},
    }

    for case in CASES:
        runs = []
        for seed in range(args.seeds):
            run = run_case(case, seed, args.transition_budget,
                           args.eval_interval, args.eval_n)
            runs.append(run)
        summary = summarize(runs)
        result["cases"][case.name] = {
            "config": asdict(case), "summary": summary, "runs": runs,
        }
        print(
            f"{case.name:42s} "
            f"AUC={summary['auc_mean_pass']['mean']:.3f} "
            f"final={summary['final_mean_pass']['mean']:.3f} "
            f"flag={summary['final_flag_pass']['mean']:.3f} "
            f"transitions={summary['transitions']['mean']:.0f}",
            flush=True,
        )

    attach_paired_analysis(result)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
