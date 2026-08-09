"""Paired component ablation on the retained synthetic skill chain.

This driver isolates five decisions in Curriculum MaxRL while reusing the
canonical ``FrontierTrainer``, ``FrontierTeacher``, and ``SkillChainSpace``:

* adaptive teacher versus uniform task sampling;
* proportional (gamma=1) versus concentrated (gamma=4) priority;
* no hindsight versus centered hindsight;
* centered versus success-only hindsight; and
* the scale of the hindsight update.

Every condition receives the same number of rollout groups and attempts for
the same paired seed set.  The primary metric is deliberately called a
``checkpoint_mean``: it is the arithmetic mean of the exact, deterministic
mean pass rate at equally spaced checkpoints including step zero.  It is not
an AUC and performs no interpolation.

Run:

    python3 -m frontier_rl.examples.run_skill_chain_ablation
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontier_rl import FrontierTeacher, FrontierTrainer, TrainerConfig
from frontier_rl.adapters.skill_chain import SkillChainSpace


@dataclass(frozen=True)
class Case:
    name: str
    label: str
    teacher: str
    gamma: float
    hindsight: bool
    hindsight_estimator: str = "maxrl"
    hindsight_scale: float = 1.0


@dataclass(frozen=True)
class Contrast:
    name: str
    description: str
    coefficients: dict[str, float]


CASES = (
    Case(
        "uniform_no_hindsight",
        "Uniform MaxRL — no hindsight",
        "uniform",
        1.0,
        False,
    ),
    Case(
        "teacher_g1_no_hindsight",
        "Curriculum-MaxRL teacher (gamma=1) — no hindsight",
        "teacher",
        1.0,
        False,
    ),
    Case(
        "teacher_g4_no_hindsight",
        "Curriculum-MaxRL teacher (gamma=4) — no hindsight",
        "teacher",
        4.0,
        False,
    ),
    Case(
        "uniform_centered_hindsight",
        "Uniform MaxRL + centered hindsight",
        "uniform",
        1.0,
        True,
    ),
    Case(
        "teacher_g1_centered_hindsight",
        "Curriculum-MaxRL (gamma=1) + centered hindsight",
        "teacher",
        1.0,
        True,
    ),
    Case(
        "teacher_g4_centered_hindsight",
        "Curriculum-MaxRL full stack (gamma=4, centered hindsight, scale=1)",
        "teacher",
        4.0,
        True,
    ),
    Case(
        "teacher_g4_success_only_hindsight",
        "Curriculum-MaxRL (gamma=4) + success-only hindsight",
        "teacher",
        4.0,
        True,
        "success_only",
    ),
    Case(
        "teacher_g4_centered_scale_0p25",
        "Curriculum-MaxRL full stack with centered scale=0.25",
        "teacher",
        4.0,
        True,
        "maxrl",
        0.25,
    ),
    Case(
        "teacher_g4_centered_scale_0p5",
        "Curriculum-MaxRL full stack with centered scale=0.5",
        "teacher",
        4.0,
        True,
        "maxrl",
        0.5,
    ),
    Case(
        "teacher_g4_centered_scale_2p0",
        "Curriculum-MaxRL full stack with centered scale=2",
        "teacher",
        4.0,
        True,
        "maxrl",
        2.0,
    ),
    Case(
        "teacher_g4_centered_scale_4p0",
        "Curriculum-MaxRL full stack with centered scale=4",
        "teacher",
        4.0,
        True,
        "maxrl",
        4.0,
    ),
    Case(
        "teacher_g4_centered_scale_8p0",
        "Curriculum-MaxRL full stack with centered scale=8",
        "teacher",
        4.0,
        True,
        "maxrl",
        8.0,
    ),
)


CONTRASTS = (
    Contrast(
        "teacher_main_without_hindsight",
        "adaptive gamma=1 teacher minus uniform, both without hindsight",
        {"teacher_g1_no_hindsight": 1.0, "uniform_no_hindsight": -1.0},
    ),
    Contrast(
        "gamma4_minus_gamma1_without_hindsight",
        "concentrated minus proportional teacher, both without hindsight",
        {"teacher_g4_no_hindsight": 1.0, "teacher_g1_no_hindsight": -1.0},
    ),
    Contrast(
        "centered_hindsight_under_uniform",
        "centered hindsight minus no hindsight under uniform sampling",
        {"uniform_centered_hindsight": 1.0, "uniform_no_hindsight": -1.0},
    ),
    Contrast(
        "centered_hindsight_under_teacher_g1",
        "centered hindsight minus no hindsight under the gamma=1 teacher",
        {
            "teacher_g1_centered_hindsight": 1.0,
            "teacher_g1_no_hindsight": -1.0,
        },
    ),
    Contrast(
        "teacher_by_centered_hindsight_interaction",
        "difference-in-differences: teacher effect with hindsight minus without",
        {
            "teacher_g1_centered_hindsight": 1.0,
            "teacher_g1_no_hindsight": -1.0,
            "uniform_centered_hindsight": -1.0,
            "uniform_no_hindsight": 1.0,
        },
    ),
    Contrast(
        "gamma4_minus_gamma1_with_centered_hindsight",
        "concentrated minus proportional teacher with centered hindsight",
        {
            "teacher_g4_centered_hindsight": 1.0,
            "teacher_g1_centered_hindsight": -1.0,
        },
    ),
    Contrast(
        "centered_hindsight_under_teacher_g4",
        "centered hindsight minus no hindsight under the gamma=4 teacher",
        {
            "teacher_g4_centered_hindsight": 1.0,
            "teacher_g4_no_hindsight": -1.0,
        },
    ),
    Contrast(
        "full_stack_minus_uniform_centered_hindsight",
        "Curriculum-MaxRL full stack minus uniform centered hindsight",
        {
            "teacher_g4_centered_hindsight": 1.0,
            "uniform_centered_hindsight": -1.0,
        },
    ),
    Contrast(
        "gamma_by_centered_hindsight_interaction",
        "difference-in-differences: gamma effect with hindsight minus without",
        {
            "teacher_g4_centered_hindsight": 1.0,
            "teacher_g1_centered_hindsight": -1.0,
            "teacher_g4_no_hindsight": -1.0,
            "teacher_g1_no_hindsight": 1.0,
        },
    ),
    Contrast(
        "success_only_minus_centered_hindsight",
        "success-only minus centered hindsight under the gamma=4 teacher",
        {
            "teacher_g4_success_only_hindsight": 1.0,
            "teacher_g4_centered_hindsight": -1.0,
        },
    ),
    Contrast(
        "centered_scale_0p25_minus_1p0",
        "centered hindsight scale 0.25 minus scale 1.0",
        {
            "teacher_g4_centered_scale_0p25": 1.0,
            "teacher_g4_centered_hindsight": -1.0,
        },
    ),
    Contrast(
        "centered_scale_0p5_minus_1p0",
        "centered hindsight scale 0.5 minus scale 1.0",
        {
            "teacher_g4_centered_scale_0p5": 1.0,
            "teacher_g4_centered_hindsight": -1.0,
        },
    ),
    Contrast(
        "centered_scale_2p0_minus_1p0",
        "centered hindsight scale 2.0 minus scale 1.0",
        {
            "teacher_g4_centered_scale_2p0": 1.0,
            "teacher_g4_centered_hindsight": -1.0,
        },
    ),
    Contrast(
        "centered_scale_4p0_minus_1p0",
        "centered hindsight scale 4.0 minus scale 1.0",
        {
            "teacher_g4_centered_scale_4p0": 1.0,
            "teacher_g4_centered_hindsight": -1.0,
        },
    ),
    Contrast(
        "centered_scale_8p0_minus_1p0",
        "centered hindsight scale 8.0 minus scale 1.0",
        {
            "teacher_g4_centered_scale_8p0": 1.0,
            "teacher_g4_centered_hindsight": -1.0,
        },
    ),
)


def run_case(
    case: Case,
    seed: int,
    *,
    steps: int = 400,
    checkpoint_every: int = 10,
    n_rollouts: int = 16,
    tasks_per_step: int = 8,
) -> dict:
    """Run one paired condition and return its complete checkpoint curve."""
    if steps < 1 or checkpoint_every < 1 or steps % checkpoint_every != 0:
        raise ValueError("steps must be positive and divisible by checkpoint_every")
    if case.teacher not in {"uniform", "teacher"}:
        raise ValueError(f"unknown teacher mode {case.teacher!r}")

    env = SkillChainSpace(seed=seed)
    cfg = TrainerConfig(
        n_rollouts=n_rollouts,
        tasks_per_step=tasks_per_step,
        hindsight=case.hindsight,
        hindsight_scale=case.hindsight_scale,
        hindsight_estimator=case.hindsight_estimator,
        teacher_gamma=case.gamma,
        seed=seed,
    )
    teacher = FrontierTeacher(
        env.n_tasks,
        n_rollouts,
        decay=cfg.teacher_decay,
        floor=cfg.teacher_floor,
        gamma=case.gamma,
        seed=seed + 1000,
    )
    if case.teacher == "uniform":
        uniform = np.full(env.n_tasks, 1.0 / env.n_tasks)
        teacher.distribution = lambda: uniform

    trainer = FrontierTrainer(env, env, cfg, teacher=teacher)
    checkpoint_steps = [0]
    pass_curve = [float(env.true_pass_rates().mean())]
    hardest_curve = [
        float(env.true_pass_rates().reshape(env.n_chains, env.n_levels)[:, -1].mean())
    ]
    totals = {
        "live_groups": 0,
        "dead_groups": 0,
        "all_pass_groups": 0,
        "relabeled_groups": 0,
        "skill_decisions": 0,
    }

    for step in range(1, steps + 1):
        stats = trainer.step()
        totals["live_groups"] += stats.live_groups
        totals["dead_groups"] += stats.dead_groups
        totals["all_pass_groups"] += stats.all_pass_groups
        totals["relabeled_groups"] += stats.relabeled_groups
        totals["skill_decisions"] += stats.env_steps
        if step % checkpoint_every == 0:
            rates = env.true_pass_rates()
            checkpoint_steps.append(step)
            pass_curve.append(float(rates.mean()))
            hardest_curve.append(
                float(rates.reshape(env.n_chains, env.n_levels)[:, -1].mean())
            )

    sampled_groups = steps * tasks_per_step
    if (
        totals["live_groups"]
        + totals["dead_groups"]
        + totals["all_pass_groups"]
        != sampled_groups
    ):
        raise RuntimeError("group accounting mismatch")

    return {
        "seed": seed,
        "checkpoint_steps": checkpoint_steps,
        "mean_pass_curve": pass_curve,
        "hardest_level_mean_curve": hardest_curve,
        "checkpoint_mean": float(np.mean(pass_curve)),
        "final_mean_pass": pass_curve[-1],
        "final_hardest_level_mean_pass": hardest_curve[-1],
        "sampled_groups": sampled_groups,
        "rollout_attempts": sampled_groups * n_rollouts,
        **totals,
    }


def bootstrap_mean_ci(
    values: np.ndarray | list[float], *, seed: int, n_boot: int = 20_000
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("bootstrap input must be a non-empty vector")
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    samples = values[
        rng.integers(0, len(values), size=(n_boot, len(values)))
    ].mean(axis=1)
    return [float(x) for x in np.quantile(samples, (0.025, 0.975))]


def exact_sign_flip_p(values: np.ndarray | list[float]) -> float:
    """Exact two-sided paired sign-flip randomization p-value."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not (1 <= len(values) <= 20):
        raise ValueError("exact sign-flip test supports between 1 and 20 pairs")
    observed = abs(float(values.mean()))
    extreme = 0
    total = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        null_value = abs(float(np.dot(signs, values) / len(values)))
        extreme += null_value >= observed - 1e-15
    return float(extreme / total)


def summarize_metric(runs: list[dict], key: str, *, bootstrap_seed: int) -> dict:
    values = np.asarray([run[key] for run in runs], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "mean_ci95_paired_seed_bootstrap": bootstrap_mean_ci(
            values, seed=bootstrap_seed
        ),
        "per_seed": values.tolist(),
    }


def contrast_values(cases: dict, contrast: Contrast, metric: str) -> np.ndarray:
    """Evaluate a pre-specified linear contrast for every paired seed."""
    seeds = None
    result = None
    for case_name, coefficient in contrast.coefficients.items():
        runs = cases[case_name]["runs"]
        case_seeds = [run["seed"] for run in runs]
        if seeds is None:
            seeds = case_seeds
            result = np.zeros(len(runs), dtype=np.float64)
        elif case_seeds != seeds:
            raise ValueError(f"seed mismatch in contrast {contrast.name}")
        result += coefficient * np.asarray(
            [run[metric] for run in runs], dtype=np.float64
        )
    return result


def holm_adjust(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down correction with monotone adjusted p-values."""
    ordered = sorted((float(p), name) for name, p in p_values.items())
    m = len(ordered)
    running_adjusted = 0.0
    still_rejecting = True
    out = {}
    for rank, (p_value, name) in enumerate(ordered, start=1):
        multiplier = m - rank + 1
        running_adjusted = max(running_adjusted, multiplier * p_value)
        reject = still_rejecting and p_value <= alpha / multiplier
        if not reject:
            still_rejecting = False
        out[name] = {
            "raw_p": p_value,
            "holm_adjusted_p": float(min(running_adjusted, 1.0)),
            "reject_familywise_0.05": bool(reject),
        }
    return out


def attach_contrast_analysis(result: dict) -> None:
    """Attach checkpoint-mean paired inference and one Holm family."""
    analyses = {}
    p_values = {}
    for index, contrast in enumerate(CONTRASTS):
        values = contrast_values(result["cases"], contrast, "checkpoint_mean")
        p_value = exact_sign_flip_p(values)
        analyses[contrast.name] = {
            "description": contrast.description,
            "coefficients": contrast.coefficients,
            "metric": "checkpoint_mean",
            "mean_contrast": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "mean_ci95_paired_seed_bootstrap": bootstrap_mean_ci(
                values, seed=10_000 + index
            ),
            "exact_paired_sign_flip_p_two_sided": p_value,
            "per_seed_contrast": values.tolist(),
        }
        p_values[contrast.name] = p_value

    adjusted = holm_adjust(p_values)
    for name, correction in adjusted.items():
        analyses[name].update(correction)
    result["paired_checkpoint_mean_contrasts"] = analyses
    result["multiplicity"] = {
        "family": list(p_values),
        "method": "Holm step-down correction",
        "familywise_alpha": 0.05,
        "test": "exact two-sided paired sign-flip randomization",
        "assumption": "paired seed contrasts are exchangeable under sign reversal",
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict:
    relevant = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "frontier_rl" / "trainer.py",
        PROJECT_ROOT / "frontier_rl" / "teacher.py",
        PROJECT_ROOT / "frontier_rl" / "estimators.py",
        PROJECT_ROOT / "frontier_rl" / "adapters" / "skill_chain.py",
    )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or None,
        "git_worktree_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        ),
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in relevant
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.quick:
        args.seeds = min(args.seeds, 4)
        args.steps = min(args.steps, 100)
        args.checkpoint_every = 10
    if not (2 <= args.seeds <= 20):
        parser.error("--seeds must lie in [2, 20] for exact paired inference")
    if args.steps < 1 or args.checkpoint_every < 1:
        parser.error("steps and checkpoint interval must be positive")
    if args.steps % args.checkpoint_every != 0:
        parser.error("--steps must be divisible by --checkpoint-every")
    if args.output is None:
        filename = (
            "skill_chain_component_ablation_quick.json"
            if args.quick
            else "skill_chain_component_ablation.json"
        )
        args.output = str(Path(__file__).resolve().with_name(filename))

    result = {
        "provenance": provenance(),
        "protocol": {
            "paired_seeds": list(range(args.seeds)),
            "steps": args.steps,
            "checkpoint_every": args.checkpoint_every,
            "checkpoint_steps": list(
                range(0, args.steps + 1, args.checkpoint_every)
            ),
            "n_rollouts": 16,
            "tasks_per_step": 8,
            "n_chains": 3,
            "n_levels_per_chain": 12,
            "n_actions": 10,
            "teacher_decay": 0.7,
            "teacher_floor": 0.1,
            "matched_budget": (
                "all conditions receive the same trainer steps, sampled groups, "
                "and rollout attempts; hindsight reuses failed rollouts"
            ),
            "compute_caveat": (
                "primitive skill-decision counts vary with sampled task depth, and "
                "hindsight can add one reused-data policy update for an all-fail "
                "group; rollout sampling is matched but optimizer compute is not"
            ),
            "primary_metric": (
                "arithmetic mean of exact mean pass rate at every listed, equally "
                "spaced checkpoint including step zero; no interpolation and not AUC"
            ),
            "evaluation": (
                "analytic pass rates from current softmax parameters; deterministic "
                "and does not consume training randomness"
            ),
            "uncertainty": "paired-seed percentile bootstrap with 20,000 resamples",
            "inference": (
                "pre-specified exact paired sign-flip tests on checkpoint_mean, "
                "Holm-corrected as one family"
            ),
        },
        "cases": {},
    }

    for case_index, case in enumerate(CASES):
        runs = [
            run_case(
                case,
                seed,
                steps=args.steps,
                checkpoint_every=args.checkpoint_every,
            )
            for seed in range(args.seeds)
        ]
        result["cases"][case.name] = {
            "config": asdict(case),
            "summary": {
                "n_seeds": args.seeds,
                "checkpoint_mean": summarize_metric(
                    runs, "checkpoint_mean", bootstrap_seed=100 + case_index
                ),
                "final_mean_pass": summarize_metric(
                    runs, "final_mean_pass", bootstrap_seed=200 + case_index
                ),
                "final_hardest_level_mean_pass": summarize_metric(
                    runs,
                    "final_hardest_level_mean_pass",
                    bootstrap_seed=300 + case_index,
                ),
                "skill_decisions": summarize_metric(
                    runs, "skill_decisions", bootstrap_seed=400 + case_index
                ),
                "relabeled_groups": summarize_metric(
                    runs, "relabeled_groups", bootstrap_seed=500 + case_index
                ),
            },
            "runs": runs,
        }
        summary = result["cases"][case.name]["summary"]
        print(
            f"{case.name:43s} "
            f"checkpoint_mean={summary['checkpoint_mean']['mean']:.3f} "
            f"final={summary['final_mean_pass']['mean']:.3f} "
            f"hardest={summary['final_hardest_level_mean_pass']['mean']:.3f}",
            flush=True,
        )

    attach_contrast_analysis(result)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
