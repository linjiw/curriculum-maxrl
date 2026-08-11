"""Outcome-blind engineering smoke test for the ICRA navigation campaign.

This runner exercises the exact comparison shape intended for the BARN/Jackal
backend on the repository's small goal-conditioned grid navigation adapter:

* estimator-derived u_N teacher (ours),
* uniform sampling,
* compute-blind p(1-p) learnability, and
* a hand-ordered easy-to-hard promotion schedule.

The grid result is an integration check, not paper evidence.  It verifies that
all arms share fixed held-out evaluation streams and emit the budget and
difficulty-resolved fields required by ``prereg_icra.md``.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from frontier_rl import (FrontierTeacher, FrontierTrainer,
                         LearnabilityTeacher, StagedDifficultyTeacher,
                         TrainerConfig, UniformTeacher)
from frontier_rl.adapters.grid_reach import GridReachSpace
from frontier_rl.evaluation import TaskEval, summarize, teacher_calibration


ARM_NAMES = ("ours_uN", "uniform", "learnability", "staged")


def make_teacher(arm: str, n_tasks: int, n_rollouts: int, seed: int):
    common = dict(n_tasks=n_tasks, n_rollouts=n_rollouts, floor=0.1,
                  decay=0.7, seed=seed)
    if arm == "ours_uN":
        return FrontierTeacher(**common, gamma=1.0)
    if arm == "uniform":
        return UniformTeacher(**common, gamma=1.0)
    if arm == "learnability":
        return LearnabilityTeacher(**common, gamma=1.0)
    if arm == "staged":
        return StagedDifficultyTeacher(
            **common, difficulty_order=np.arange(n_tasks), initial_tasks=2,
            promotion_threshold=0.7, min_frontier_groups=5)
    raise ValueError(f"unknown arm {arm!r}; choose from {ARM_NAMES}")


def evaluate(env: GridReachSpace, teacher, *, n_episodes: int,
             eval_seed: int) -> tuple[dict, list[TaskEval]]:
    evals = []
    for task_id in range(env.n_tasks):
        successes, sim_steps = env.evaluate_task(
            task_id, n_episodes, seed=eval_seed)
        evals.append(TaskEval(task_id, n_episodes, successes, sim_steps))

    easy_count = max(1, int(np.ceil(env.n_tasks / 10)))
    summary = summarize(evals, ks=(1, 4, 8),
                        easy_set=list(range(easy_count)))
    rates = np.array([row.rate() for row in evals])
    deciles = np.array_split(np.arange(env.n_tasks), min(10, env.n_tasks))
    summary["per_task_success"] = rates.tolist()
    summary["success_by_difficulty_bin"] = [
        float(rates[idx].mean()) for idx in deciles]
    summary.update(teacher_calibration(teacher, evals, min_visits=1))
    return summary, evals


def _teacher_diagnostics(teacher) -> dict:
    p_hat = teacher.pass_rate_estimates()
    if isinstance(teacher, StagedDifficultyTeacher):
        weights = teacher.distribution()
    elif isinstance(teacher, UniformTeacher):
        weights = teacher.distribution()
    else:
        utility = teacher.utility(p_hat) ** teacher.gamma
        weights = (utility / utility.sum() if utility.sum() > 1e-12
                   else np.full(teacher.n_tasks, 1.0 / teacher.n_tasks))
        weights = ((1.0 - teacher.floor) * weights
                   + teacher.floor / teacher.n_tasks)
    return {
        "posterior_mean": p_hat.tolist(),
        "sampling_weights_at_posterior_mean": weights.tolist(),
        "visits": teacher.visits.tolist(),
        **teacher.metrics(),
    }


def run_one(arm: str, seed: int, *, steps: int = 60, radius: int = 8,
            n_rollouts: int = 16, tasks_per_step: int = 4,
            eval_every: int = 10, eval_episodes: int = 32) -> dict:
    env = GridReachSpace(radius=radius, seed=seed)
    teacher = make_teacher(arm, env.n_tasks, n_rollouts, seed + 1000)
    config = TrainerConfig(
        n_rollouts=n_rollouts,
        tasks_per_step=tasks_per_step,
        hindsight=False,
        estimator="maxrl",
        teacher_gamma=1.0,
        seed=seed,
    )
    trainer = FrontierTrainer(env, env, config, teacher=teacher)
    eval_seed = 100_000 + seed
    history = []
    totals = {"live_groups": 0, "dead_groups": 0,
              "relabeled_groups": 0, "training_wall_seconds": 0.0}

    def checkpoint(step: int) -> None:
        result, _ = evaluate(env, teacher, n_episodes=eval_episodes,
                             eval_seed=eval_seed)
        groups = totals["live_groups"] + totals["dead_groups"]
        history.append({
            "step": step,
            "episodes": env.training_episodes,
            "sim_steps": env.training_sim_steps,
            "training_wall_seconds": totals["training_wall_seconds"],
            "dead_group_rate": totals["dead_groups"] / max(groups, 1),
            "live_groups": totals["live_groups"],
            "dead_groups": totals["dead_groups"],
            "relabeled_groups": totals["relabeled_groups"],
            "eval": result,
            "teacher": _teacher_diagnostics(teacher),
        })

    checkpoint(0)
    for step in range(1, steps + 1):
        started = time.perf_counter()
        stats = trainer.step()
        totals["training_wall_seconds"] += time.perf_counter() - started
        totals["live_groups"] += stats.live_groups
        totals["dead_groups"] += stats.dead_groups
        totals["relabeled_groups"] += stats.relabeled_groups
        if step % eval_every == 0 or step == steps:
            checkpoint(step)

    x = np.array([row["episodes"] for row in history], dtype=float)
    y = np.array([row["eval"]["mean_success"] for row in history], dtype=float)
    auc = float(np.trapz(y, x) / x[-1]) if x[-1] > 0 else float(y[-1])
    wall = np.array([row["training_wall_seconds"] for row in history],
                    dtype=float)
    wall_auc = (float(np.trapz(y, wall) / wall[-1])
                if wall[-1] > 0 else float(y[-1]))
    return {
        "arm": arm,
        "seed": seed,
        "eval_seed": eval_seed,
        "target_uniform_auc_by_episode": auc,
        "target_uniform_auc_by_own_training_wall": wall_auc,
        "final": history[-1],
        "history": history,
    }


def run_campaign(*, arms=ARM_NAMES, seeds=1, seed_start=0, steps=60,
                 radius=8, n_rollouts=16, tasks_per_step=4, eval_every=10,
                 eval_episodes=32) -> dict:
    results = {arm: [] for arm in arms}
    for arm in arms:
        for seed in range(seed_start, seed_start + seeds):
            results[arm].append(run_one(
                arm, seed, steps=steps, radius=radius,
                n_rollouts=n_rollouts, tasks_per_step=tasks_per_step,
                eval_every=eval_every, eval_episodes=eval_episodes))
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "engineering_smoke_not_paper_evidence",
        "domain": "goal_conditioned_grid_navigation_smoke",
        "heldout_protocol": (
            "fixed per-task/per-episode seeds; evaluation does not mutate "
            "training RNG or budget counters"),
        "config": {
            "arms": list(arms),
            "seeds": seeds,
            "seed_start": seed_start,
            "steps": steps,
            "radius": radius,
            "n_rollouts": n_rollouts,
            "tasks_per_step": tasks_per_step,
            "eval_every": eval_every,
            "eval_episodes": eval_episodes,
            "hindsight": False,
            "estimator": "maxrl",
            "teacher_gamma": 1.0,
            "difficulty_metadata": "Chebyshev goal-distance ring",
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default=",".join(ARM_NAMES))
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--n-rollouts", type=int, default=16)
    parser.add_argument("--tasks-per-step", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument(
        "--output", type=Path,
        default=Path("icra2027/results/navigation_smoke.json"))
    args = parser.parse_args()
    arms = tuple(part.strip() for part in args.arms.split(",") if part.strip())
    unknown = set(arms) - set(ARM_NAMES)
    if unknown:
        parser.error(f"unknown arms: {sorted(unknown)}")
    artifact = run_campaign(
        arms=arms, seeds=args.seeds, seed_start=args.seed_start,
        steps=args.steps, radius=args.radius, n_rollouts=args.n_rollouts,
        tasks_per_step=args.tasks_per_step, eval_every=args.eval_every,
        eval_episodes=args.eval_episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    for arm in arms:
        aucs = [row["target_uniform_auc_by_episode"]
                for row in artifact["results"][arm]]
        finals = [row["final"]["eval"]["mean_success"]
                  for row in artifact["results"][arm]]
        print(f"{arm:14s} AUC={np.mean(aucs):.4f} "
              f"final={np.mean(finals):.4f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
