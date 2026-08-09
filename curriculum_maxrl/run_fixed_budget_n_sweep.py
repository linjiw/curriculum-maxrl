"""Fixed-completion SkillChain sweep over MaxRL rollout-group size ``N``.

Scientific question
-------------------
Does the rollout-count-aware coefficient-mass score

    u_N(p) = 1 - (1 - p)**N - p

become a better task-sampling priority than the N-agnostic learnability score
``p(1-p)`` as MaxRL's rollout-group size grows?

The experiment compares three *sampling rules* under the same practical
dropped-group MaxRL estimator at each N in {2, 4, 8, 16, 32}:

* ``uniform``: uniform task sampling;
* ``learnability``: Thompson draw followed by p(1-p), a score-level
  ProCuRL/SFL-style comparator (not a faithful reproduction of either full
  algorithm); and
* ``u_n``: the paper's exact half coefficient-mass score for the deployed
  N-rollout estimator.

Every (N, sampler, seed) cell receives exactly the same number of sampled
completions.  The default 51,200-completion budget is inherited from the
existing ``results_fixed_n.json`` protocol: 400 steps x 8 tasks x N=16.
Checkpoints are aligned by completions rather than optimizer steps.  Because
N necessarily changes the number of groups/updates at fixed completion
budget, causal comparisons are paired *within N*; raw performance across N
also includes that real compute-allocation tradeoff.

The teacher uses the existing discounted-Beta/Thompson design.  To avoid a
hidden N-dependent memory horizon, its reference group decay d_ref at N_ref
is converted to d_N = d_ref**(N/N_ref).  Thus evidence has the same decay per
sampled completion for every N.  A 10% uniform floor and utility power 1 are
held fixed to isolate the utility form.

Run from the repository root:

    python3 curriculum_maxrl/run_fixed_budget_n_sweep.py

The output JSON includes the full paired seed curves, exact budget counters,
aggregate summaries, paired contrasts, and invariant/legacy-anchor checks.
This is a deterministic NumPy experiment; ``--workers`` affects only runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from curriculum_maxrl.estimators import weights_maxrl
from curriculum_maxrl.teachers import Teacher
from curriculum_maxrl.testbed import SkillChainEnv


N_VALUES = (2, 4, 8, 16, 32)
SAMPLERS = ("uniform", "learnability", "u_n")
CONTRASTS = (
    ("u_n_minus_uniform", "u_n", "uniform"),
    ("u_n_minus_learnability", "u_n", "learnability"),
    ("learnability_minus_uniform", "learnability", "uniform"),
)
PRIMARY_METRIC = "normalized_auc_mean_pass"
CONTRAST_METRICS = (
    PRIMARY_METRIC,
    "normalized_auc_pass_at_8",
    "final_mean_pass",
    "final_pass_at_8",
)


@dataclass(frozen=True)
class SweepConfig:
    total_completions: int = 51_200
    checkpoint_every: int = 2_560
    tasks_per_step: int = 8
    learning_rate: float = 0.5
    uniform_floor: float = 0.1
    reference_group_decay: float = 0.9
    reference_n: int = 16
    eval_k: int = 8


def utility_values(sampler: str, p: np.ndarray, n_rollouts: int) -> np.ndarray:
    """Return non-negative priority scores for posterior draws ``p``.

    ``u_2(p)`` is evaluated through its algebraically identical p(1-p) form.
    This removes irrelevant floating-point differences and makes the N=2
    identity a deterministic paired implementation check.
    """
    p = np.asarray(p, dtype=float)
    if sampler == "uniform":
        return np.ones_like(p)
    if sampler == "learnability" or (sampler == "u_n" and n_rollouts == 2):
        return p * (1.0 - p)
    if sampler == "u_n":
        return np.maximum(1.0 - (1.0 - p) ** n_rollouts - p, 0.0)
    raise ValueError(f"unknown sampler {sampler!r}")


class FixedBudgetUtilityTeacher(Teacher):
    """Existing TaskStats/Thompson teacher with an N-normalized decay clock."""

    def __init__(
        self,
        n_tasks: int,
        *,
        sampler: str,
        n_rollouts: int,
        seed: int,
        uniform_floor: float,
        reference_group_decay: float,
        reference_n: int,
    ) -> None:
        super().__init__(n_tasks, seed=seed)
        if sampler not in SAMPLERS:
            raise ValueError(f"unknown sampler {sampler!r}")
        self.sampler = sampler
        self.n_rollouts = int(n_rollouts)
        self.uniform_floor = float(uniform_floor)
        self.group_decay = float(
            reference_group_decay ** (self.n_rollouts / reference_n)
        )

    def observe(self, task_id: int, rewards: np.ndarray) -> None:
        """Update discounted pseudo-counts using a completion-normalized clock."""
        rewards = np.asarray(rewards, dtype=float)
        st = self.stats[task_id]
        a, b = st.alpha_beta
        successes = float(rewards.sum())
        st.alpha_beta = (
            1.0 + (a - 1.0) * self.group_decay + successes,
            1.0 + (b - 1.0) * self.group_decay + len(rewards) - successes,
        )
        st.visits += 1

    def distribution(self) -> np.ndarray:
        if self.sampler == "uniform":
            return np.full(self.n_tasks, 1.0 / self.n_tasks)

        posterior_draws = np.array(
            [self.rng.beta(*st.alpha_beta) for st in self.stats], dtype=float
        )
        scores = utility_values(self.sampler, posterior_draws, self.n_rollouts)
        if scores.sum() <= 1e-12:
            scores = np.ones(self.n_tasks)
        priority = scores / scores.sum()
        uniform = np.full(self.n_tasks, 1.0 / self.n_tasks)
        return (
            (1.0 - self.uniform_floor) * priority
            + self.uniform_floor * uniform
        )


def validate_design(config: SweepConfig, n_values: Sequence[int]) -> None:
    if config.total_completions <= 0 or config.checkpoint_every <= 0:
        raise ValueError("completion budget and checkpoint interval must be positive")
    if config.tasks_per_step <= 0 or config.eval_k <= 0:
        raise ValueError("tasks_per_step and eval_k must be positive")
    if not (0.0 <= config.uniform_floor < 1.0):
        raise ValueError("uniform_floor must lie in [0, 1)")
    if not (0.0 < config.reference_group_decay <= 1.0):
        raise ValueError("reference_group_decay must lie in (0, 1]")
    if config.reference_n <= 0:
        raise ValueError("reference_n must be positive")
    if not n_values or any(int(n) != n or n < 2 for n in n_values):
        raise ValueError("all N values must be integers >= 2")

    for n_rollouts in n_values:
        completions_per_step = config.tasks_per_step * int(n_rollouts)
        if config.total_completions % completions_per_step:
            raise ValueError(
                f"total_completions={config.total_completions} is not divisible "
                f"by tasks_per_step*N={completions_per_step} for N={n_rollouts}"
            )
        if config.checkpoint_every % completions_per_step:
            raise ValueError(
                f"checkpoint_every={config.checkpoint_every} is not divisible "
                f"by tasks_per_step*N={completions_per_step} for N={n_rollouts}"
            )
    if config.total_completions % config.checkpoint_every:
        raise ValueError("total_completions must be divisible by checkpoint_every")


def policy_metrics(env: SkillChainEnv, *, eval_k: int) -> dict[str, float]:
    p = env.true_pass_rates()
    pass_at_k = 1.0 - (1.0 - p) ** eval_k
    frontier_by_chain = []
    for chain in range(env.n_chains):
        start = chain * env.n_levels
        chain_p = p[start : start + env.n_levels]
        solved = np.flatnonzero(chain_p > 0.5)
        frontier_by_chain.append(int(solved[-1] + 1) if len(solved) else 0)
    return {
        "mean_pass": float(p.mean()),
        "pass_at_8": float(pass_at_k.mean()),
        "solved_frac_p_gt_0p9": float((p > 0.9).mean()),
        "frontier_level_mean": float(np.mean(frontier_by_chain)),
    }


def normalized_auc(checkpoints: list[dict], metric: str, budget: int) -> float:
    x = np.asarray([row["completions"] for row in checkpoints], dtype=float)
    y = np.asarray([row[metric] for row in checkpoints], dtype=float)
    return float(np.trapz(y, x) / budget)


def run_cell(
    sampler: str,
    n_rollouts: int,
    seed: int,
    config: SweepConfig,
) -> dict:
    """Run one deterministic paired cell and return all accounting/metrics."""
    validate_design(config, (n_rollouts,))
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}")

    env = SkillChainEnv(seed=seed)
    teacher = FixedBudgetUtilityTeacher(
        env.n_tasks,
        sampler=sampler,
        n_rollouts=n_rollouts,
        seed=seed + 1000,
        uniform_floor=config.uniform_floor,
        reference_group_decay=config.reference_group_decay,
        reference_n=config.reference_n,
    )
    completions_per_step = config.tasks_per_step * n_rollouts
    steps = config.total_completions // completions_per_step
    checkpoints = [
        {"completions": 0, **policy_metrics(env, eval_k=config.eval_k)}
    ]
    requested_task_counts = np.zeros(env.n_tasks, dtype=np.int64)
    dead_groups = 0
    mixed_groups = 0
    all_pass_groups = 0
    coefficient_l1_mass = 0.0
    sampled_successes = 0

    for step in range(steps):
        task_ids = teacher.sample_tasks(config.tasks_per_step)
        for task_id_raw in task_ids:
            task_id = int(task_id_raw)
            requested_task_counts[task_id] += 1
            actions, rewards = env.rollout(task_id, n_rollouts)
            sampled_successes += int(rewards.sum())
            teacher.observe(task_id, rewards)
            weights = weights_maxrl(rewards)
            coefficient_l1_mass += float(np.abs(weights).sum())
            k = int(rewards.sum())
            if k == 0:
                dead_groups += 1
            elif k == n_rollouts:
                all_pass_groups += 1
            else:
                mixed_groups += 1
            if np.any(weights != 0.0):
                env.apply_gradient(task_id, actions, weights, config.learning_rate)

        completions = (step + 1) * completions_per_step
        if completions % config.checkpoint_every == 0:
            checkpoints.append(
                {
                    "completions": completions,
                    **policy_metrics(env, eval_k=config.eval_k),
                }
            )

    groups = steps * config.tasks_per_step
    if groups != dead_groups + mixed_groups + all_pass_groups:
        raise AssertionError("group outcome accounting does not sum to total groups")
    if groups * n_rollouts != config.total_completions:
        raise AssertionError("cell did not consume the requested completion budget")
    expected_checkpoints = list(
        range(0, config.total_completions + 1, config.checkpoint_every)
    )
    if [row["completions"] for row in checkpoints] != expected_checkpoints:
        raise AssertionError("completion-aligned checkpoint schedule drifted")

    final = checkpoints[-1]
    return {
        "sampler": sampler,
        "n_rollouts": n_rollouts,
        "seed": seed,
        "steps": steps,
        "groups": groups,
        "completions": groups * n_rollouts,
        "teacher_group_decay": teacher.group_decay,
        "sampled_successes": sampled_successes,
        "dead_groups": dead_groups,
        "mixed_groups": mixed_groups,
        "all_pass_groups": all_pass_groups,
        "coefficient_l1_mass": coefficient_l1_mass,
        "requested_task_counts": requested_task_counts.tolist(),
        "normalized_auc_mean_pass": normalized_auc(
            checkpoints, "mean_pass", config.total_completions
        ),
        "normalized_auc_pass_at_8": normalized_auc(
            checkpoints, "pass_at_8", config.total_completions
        ),
        "final_mean_pass": final["mean_pass"],
        "final_pass_at_8": final["pass_at_8"],
        "final_solved_frac_p_gt_0p9": final["solved_frac_p_gt_0p9"],
        "final_frontier_level_mean": final["frontier_level_mean"],
        "checkpoints": checkpoints,
    }


def _run_cell_tuple(args: tuple[str, int, int, SweepConfig]) -> dict:
    return run_cell(*args)


def summarize(values: Iterable[float]) -> dict[str, object]:
    x = np.asarray(list(values), dtype=float)
    if len(x) == 0:
        raise ValueError("cannot summarize an empty collection")
    sample_sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return {
        "mean": float(x.mean()),
        "sample_sd": sample_sd,
        "standard_error": float(sample_sd / math.sqrt(len(x))),
        "min": float(x.min()),
        "max": float(x.max()),
        "values_by_seed": x.tolist(),
    }


def exact_two_sided_sign_p(positive: int, negative: int) -> float | None:
    """Exact paired sign-test p-value after dropping zero differences."""
    nonzero = positive + negative
    if nonzero == 0:
        return None
    tail = min(positive, negative)
    probability = 2.0 * sum(
        math.comb(nonzero, k) for k in range(tail + 1)
    ) / (2.0 ** nonzero)
    return min(1.0, probability)


def build_results(
    runs: list[dict],
    config: SweepConfig,
    n_values: Sequence[int],
    seeds: Sequence[int],
) -> dict:
    lookup = {
        (run["n_rollouts"], run["sampler"], run["seed"]): run for run in runs
    }
    by_n: dict[str, object] = {}
    for n_rollouts in n_values:
        arms: dict[str, object] = {}
        for sampler in SAMPLERS:
            seed_runs = [lookup[(n_rollouts, sampler, seed)] for seed in seeds]
            arms[sampler] = {
                "summary": {
                    metric: summarize(run[metric] for run in seed_runs)
                    for metric in CONTRAST_METRICS
                },
                "seed_runs": seed_runs,
            }

        paired_contrasts: dict[str, object] = {}
        for contrast_name, lhs, rhs in CONTRASTS:
            metric_rows = {}
            for metric in CONTRAST_METRICS:
                differences = [
                    lookup[(n_rollouts, lhs, seed)][metric]
                    - lookup[(n_rollouts, rhs, seed)][metric]
                    for seed in seeds
                ]
                row = summarize(differences)
                row["positive_seeds"] = int(np.sum(np.asarray(differences) > 0.0))
                row["zero_seeds"] = int(np.sum(np.asarray(differences) == 0.0))
                row["negative_seeds"] = int(np.sum(np.asarray(differences) < 0.0))
                row["two_sided_exact_sign_p"] = exact_two_sided_sign_p(
                    row["positive_seeds"], row["negative_seeds"]
                )
                metric_rows[metric] = row
            paired_contrasts[contrast_name] = {
                "lhs": lhs,
                "rhs": rhs,
                "metrics": metric_rows,
            }
        by_n[str(n_rollouts)] = {
            "steps_per_cell": config.total_completions
            // (config.tasks_per_step * n_rollouts),
            "groups_per_cell": config.total_completions // n_rollouts,
            "arms": arms,
            "paired_contrasts": paired_contrasts,
        }

    n2_identity = all(
        lookup[(2, "learnability", seed)]["checkpoints"]
        == lookup[(2, "u_n", seed)]["checkpoints"]
        and lookup[(2, "learnability", seed)]["requested_task_counts"]
        == lookup[(2, "u_n", seed)]["requested_task_counts"]
        for seed in seeds
    ) if 2 in n_values else None
    exact_budgets = all(
        run["completions"] == config.total_completions for run in runs
    )
    common_checkpoints = len(
        {
            tuple(row["completions"] for row in run["checkpoints"])
            for run in runs
        }
    ) == 1

    legacy_anchor = legacy_uniform_n16_anchor(lookup, seeds)
    return {
        "schema_version": 1,
        "experiment": "skillchain_fixed_completion_n_sweep",
        "research_question": (
            "Within each MaxRL rollout-group size N, does u_N task sampling "
            "improve completion-indexed learning over uniform and p(1-p)?"
        ),
        "primary_metric": PRIMARY_METRIC,
        "interpretation_guardrail": (
            "Primary causal comparisons are paired sampler contrasts within N. "
            "Across-N absolute performance also changes the number of groups and "
            "updates at the fixed completion budget."
        ),
        "learnability_scope": (
            "p(1-p) is a score-level ProCuRL/SFL-style comparator, not a "
            "faithful reproduction of either complete algorithm."
        ),
        "analysis_status": (
            "Post-guidance CPU follow-up; paired seed summaries and sign tests "
            "are descriptive and were not preregistered."
        ),
        "protocol": {
            "environment": "SkillChainEnv: 3 nested chains x 12 levels x 10 actions",
            "estimator": "practical dropped-group MaxRL weights",
            "hindsight": False,
            "budget_derivation": "400 steps x 8 tasks x 16 rollouts = 51,200",
            "checkpoint_clock": "sampled completions",
            "teacher_evidence": (
                "discounted Beta pseudo-counts with Thompson draws; decay "
                "normalized to a common per-completion rate"
            ),
        },
        "config": {**asdict(config), "n_values": list(n_values), "seeds": list(seeds)},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "git_commit": git_commit(),
            "source_sha256": source_checksums(),
        },
        "checks": {
            "all_cells_exact_completion_budget": exact_budgets,
            "all_cells_share_completion_checkpoints": common_checkpoints,
            "u_2_equals_p_times_one_minus_p_pairwise": n2_identity,
            "legacy_uniform_n16_final_anchor": legacy_anchor,
        },
        "by_n": by_n,
    }


def legacy_uniform_n16_anchor(
    lookup: dict[tuple[int, str, int], dict], seeds: Sequence[int]
) -> dict[str, object] | None:
    """Compare the overlapping five-seed cell to results_fixed_n.json."""
    anchor_path = Path(__file__).with_name("results_fixed_n.json")
    anchor_seeds = [seed for seed in range(5) if seed in seeds]
    if (16, "uniform", 0) not in lookup or not anchor_path.exists() or len(anchor_seeds) < 5:
        return None
    with anchor_path.open() as handle:
        old = json.load(handle)
    expected = float(old["uniform+maxrl"]["mean_pass"])
    observed = float(
        np.mean([lookup[(16, "uniform", seed)]["final_mean_pass"] for seed in range(5)])
    )
    return {
        "source": str(anchor_path.relative_to(PROJECT_ROOT)),
        "expected_five_seed_mean": expected,
        "observed_five_seed_mean": observed,
        "absolute_error": abs(observed - expected),
        "passed_at_1e_minus_12": abs(observed - expected) <= 1e-12,
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_checksums() -> dict[str, str]:
    relative_paths = (
        "curriculum_maxrl/run_fixed_budget_n_sweep.py",
        "curriculum_maxrl/testbed.py",
        "curriculum_maxrl/estimators.py",
        "curriculum_maxrl/teachers.py",
        "curriculum_maxrl/results_fixed_n.json",
    )
    checksums = {}
    for relative in relative_paths:
        path = PROJECT_ROOT / relative
        if path.exists():
            checksums[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return checksums


def run_sweep(
    config: SweepConfig,
    *,
    n_values: Sequence[int] = N_VALUES,
    seeds: Sequence[int] = tuple(range(8)),
    workers: int = 1,
) -> dict:
    validate_design(config, n_values)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty sequence of unique integers")
    jobs = [
        (sampler, int(n_rollouts), int(seed), config)
        for n_rollouts in n_values
        for sampler in SAMPLERS
        for seed in seeds
    ]
    if workers == 1:
        runs = [_run_cell_tuple(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            runs = list(pool.map(_run_cell_tuple, jobs))
    return build_results(runs, config, n_values, seeds)


def parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def print_summary(results: dict) -> None:
    print("\nPrimary metric: normalized AUC of exact mean pass rate vs completions")
    print("N    uniform     p(1-p)       u_N      u_N-p(1-p)   positive seeds")
    for n_text, row in results["by_n"].items():
        arms = row["arms"]
        uniform = arms["uniform"]["summary"][PRIMARY_METRIC]["mean"]
        learnability = arms["learnability"]["summary"][PRIMARY_METRIC]["mean"]
        u_n = arms["u_n"]["summary"][PRIMARY_METRIC]["mean"]
        contrast = row["paired_contrasts"]["u_n_minus_learnability"]["metrics"][
            PRIMARY_METRIC
        ]
        print(
            f"{int(n_text):2d}   {uniform:8.4f}   {learnability:8.4f}   "
            f"{u_n:8.4f}      {contrast['mean']:+8.4f}       "
            f"{contrast['positive_seeds']}/{len(results['config']['seeds'])}"
        )
    print("\nChecks:")
    for name, value in results["checks"].items():
        print(f"  {name}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", type=parse_int_csv, default=N_VALUES)
    parser.add_argument("--seeds", type=parse_int_csv, default=tuple(range(8)))
    parser.add_argument("--total-completions", type=int, default=51_200)
    parser.add_argument("--checkpoint-every", type=int, default=2_560)
    parser.add_argument("--tasks-per-step", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--uniform-floor", type=float, default=0.1)
    parser.add_argument("--reference-group-decay", type=float, default=0.9)
    parser.add_argument("--reference-n", type=int, default=16)
    parser.add_argument("--eval-k", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="parallel CPU processes (results are deterministic across values)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("results_fixed_budget_n_sweep.json"),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    config = SweepConfig(
        total_completions=args.total_completions,
        checkpoint_every=args.checkpoint_every,
        tasks_per_step=args.tasks_per_step,
        learning_rate=args.learning_rate,
        uniform_floor=args.uniform_floor,
        reference_group_decay=args.reference_group_decay,
        reference_n=args.reference_n,
        eval_k=args.eval_k,
    )
    results = run_sweep(
        config,
        n_values=args.n_values,
        seeds=args.seeds,
        workers=args.workers,
    )
    print_summary(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
