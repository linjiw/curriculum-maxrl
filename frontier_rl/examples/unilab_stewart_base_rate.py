#!/usr/bin/env python3
"""Fixed-policy feasibility probe for UniLab's StewartBalance task.

This is deliberately not a training result.  It measures whether short,
complete CPU episodes expose a useful range of binary success probabilities
before we build a grouped Curriculum-MaxRL learner.  Run it from a UniLab
checkout so UniLab's locked environment and optional simulator dependency are
used, for example::

    uv run --extra motrix python \
      ../curriculum-maxrl/frontier_rl/examples/unilab_stewart_base_rate.py \
      --output /tmp/unilab_stewart_base_rate.json

The task's terminal semantics are unambiguous in the pinned UniLab revision:
successful stillness terminates with positive reward, while a fall terminates
with the configured negative fall penalty.  Time-limit truncation is a failed
attempt for this sparse verifier.  The policy is the deterministic zero-action
policy, so the result is an axis/lifecycle diagnostic rather than evidence for
any curriculum or learner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from unilab.envs.manipulation.stewart.balance import StewartBalanceCfg, StewartBalanceEnv


DEFAULT_RATIOS = (0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def _run_cell(
    *,
    ratio: float,
    seed: int,
    num_envs: int,
    steps: int,
    horizon_seconds: float,
    backend: str,
) -> dict[str, int | float]:
    np.random.seed(seed)
    cfg = StewartBalanceCfg(
        max_episode_seconds=horizon_seconds,
        init_ball_radius_ratio=ratio,
    )
    env = StewartBalanceEnv(cfg, num_envs=num_envs, backend_type=backend)
    try:
        env.init_state()
        assert env.state is not None
        # UniLab normally clears these counters during the initial autoreset.
        # Set them explicitly so this diagnostic never counts partial initial
        # episodes if that lifecycle implementation changes later.
        env.state.info["steps"].fill(0)

        success = 0
        fall = 0
        timeout = 0
        other_terminal = 0
        zero_actions = np.zeros((num_envs, 2), dtype=np.float32)
        for _ in range(steps):
            state = env.step(zero_actions)
            success_mask = state.terminated & (state.reward > 0)
            fall_mask = state.terminated & (state.reward < 0)
            other_mask = state.terminated & ~(success_mask | fall_mask)
            timeout_mask = state.truncated & ~state.terminated
            success += int(success_mask.sum())
            fall += int(fall_mask.sum())
            other_terminal += int(other_mask.sum())
            timeout += int(timeout_mask.sum())
    finally:
        env.close()

    completed = success + fall + timeout + other_terminal
    if completed == 0:
        raise RuntimeError("probe produced no complete episodes")
    return {
        "ratio": ratio,
        "seed": seed,
        "success": success,
        "fall": fall,
        "timeout": timeout,
        "other_terminal": other_terminal,
        "completed": completed,
        "pass_rate": success / completed,
    }


def run_probe(args: argparse.Namespace) -> dict:
    cells = []
    for seed in args.seeds:
        for ratio_index, ratio in enumerate(args.ratios):
            # Give each cell a stable independent NumPy stream while preserving
            # the user-visible replicate id in the artifact.
            cell_seed = int(seed) * 10_000 + ratio_index
            cell = _run_cell(
                ratio=float(ratio),
                seed=cell_seed,
                num_envs=args.num_envs,
                steps=args.steps,
                horizon_seconds=args.horizon_seconds,
                backend=args.backend,
            )
            cell["replicate"] = int(seed)
            cells.append(cell)

    summary = []
    for ratio in args.ratios:
        selected = [cell for cell in cells if cell["ratio"] == float(ratio)]
        rates = np.asarray([cell["pass_rate"] for cell in selected], dtype=float)
        total_success = sum(int(cell["success"]) for cell in selected)
        total_completed = sum(int(cell["completed"]) for cell in selected)
        summary.append(
            {
                "ratio": float(ratio),
                "replicates": len(selected),
                "total_success": total_success,
                "total_completed": total_completed,
                "pooled_pass_rate": total_success / total_completed,
                "mean_replicate_pass_rate": float(rates.mean()),
                "min_replicate_pass_rate": float(rates.min()),
                "max_replicate_pass_rate": float(rates.max()),
            }
        )

    return {
        "status": "fixed-policy feasibility diagnostic; not a training result",
        "task": "UniLab StewartBalance",
        "backend": args.backend,
        "policy": "zero action",
        "success_verifier": "positive-reward task termination; timeout/fall are failures",
        "num_envs": args.num_envs,
        "simulator_steps": args.steps,
        "horizon_seconds": args.horizon_seconds,
        "control_steps_per_horizon": int(round(args.horizon_seconds / 0.02)),
        "ratios": [float(ratio) for ratio in args.ratios],
        "replicates": [int(seed) for seed in args.seeds],
        "cells": cells,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="motrix", choices=("motrix", "mujoco"))
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--horizon-seconds", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.steps <= 0 or args.horizon_seconds <= 0:
        parser.error("num-envs, steps, and horizon-seconds must be positive")
    if not args.seeds or not args.ratios:
        parser.error("at least one seed and ratio are required")
    if any(not 0.0 <= ratio <= 1.0 for ratio in args.ratios):
        parser.error("ratios must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    result = run_probe(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
