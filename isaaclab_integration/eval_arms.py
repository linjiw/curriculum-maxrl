# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate pilot-arm checkpoints on a FIXED per-level grid (the honest P-B/P-C readout).

Training telemetry is biased by each arm's own sampling distribution — an arm that
never visits level 9 reports nothing about level 9. This script loads each run's
checkpoints and rolls them out under the TRAINING task cfg with the curriculum term
swapped for ``FixedLevelProbe`` (env i pinned to level i % n_bins), producing
per-level pass rates on the same grid for every arm. Success predicate defaults to
``tile`` — the same signal the teacher trained on.

Writes ``eval_frontier_<ckpt>.json`` into each run dir. Run in-container:

  docker exec isaac-lab-base bash -c "cd /workspace/isaaclab && /isaac-sim/python.sh \
    scripts/curriculum-maxrl/isaaclab_integration/eval_arms.py --headless \
    --log_root logs/rsl_rl/anymal_c_rough --num_envs 200 --episodes_per_env 4"
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import glob
import json
import os
import re
import sys

from isaaclab.app import AppLauncher

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_HERE))  # scripts/curriculum-maxrl

parser = argparse.ArgumentParser(description="Fixed-grid eval of frontier-pilot checkpoints.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Anymal-C-v0")
parser.add_argument("--log_root", type=str, default="logs/rsl_rl/anymal_c_rough")
parser.add_argument("--run_glob", type=str, default="*", help="Subset of run dirs to evaluate.")
parser.add_argument("--checkpoints", type=str, default="mid,final",
                    help="'mid,final' | 'final' | comma list of iteration numbers.")
parser.add_argument("--num_envs", type=int, default=200)
parser.add_argument("--episodes_per_env", type=int, default=4)
parser.add_argument("--success_fn", type=str, default="tile", choices=["survival", "distance", "tile"])
parser.add_argument("--seed", type=int, default=12345,
                    help="Fixed evaluation seed reused for every arm/checkpoint.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.managers import CurriculumTermCfg as CurrTerm

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

from isaaclab_integration.frontier_terms import FixedLevelProbe


def pick_checkpoints(run_dir: str, spec: str) -> list[str]:
    ckpts = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")),
                   key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))
    if not ckpts:
        return []
    by_iter = {int(re.search(r"model_(\d+)\.pt", p).group(1)): p for p in ckpts}
    iters = sorted(by_iter)
    picks = []
    for token in spec.split(","):
        token = token.strip()
        if token == "final":
            picks.append(by_iter[iters[-1]])
        elif token == "mid":
            picks.append(by_iter[min(iters, key=lambda i: abs(i - iters[-1] // 2))])
        elif token.isdigit() and int(token) in by_iter:
            picks.append(by_iter[int(token)])
    return list(dict.fromkeys(picks))


def evaluate(env, policy, probe: FixedLevelProbe, max_steps: int) -> dict:
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(max_steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)
    return probe.results()


def main():
    # one env instance reused across all runs/checkpoints (same task cfg for all arms)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.curriculum.terrain_levels = CurrTerm(
        func=FixedLevelProbe, params={"success_fn": args_cli.success_fn})
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env = gym.make(args_cli.task, cfg=env_cfg)
    # fetch the live probe instance from the manager's own (deep-copied) term cfg;
    # CurriculumManager has no get_term_cfg and the class is instantiated at play
    mgr = env.unwrapped.curriculum_manager
    term_cfg = dict(zip(mgr._term_names, mgr._term_cfgs))["terrain_levels"]
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # triggers play + term init
    probe: FixedLevelProbe = term_cfg.func
    assert isinstance(probe, FixedLevelProbe), f"probe not initialized: {type(probe)}"
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    # steps for ~episodes_per_env full episodes (episodes may end early — that's data)
    steps_per_episode = int(env.unwrapped.max_episode_length)
    max_steps = steps_per_episode * args_cli.episodes_per_env

    run_dirs = sorted(
        d for d in glob.glob(os.path.join(args_cli.log_root, args_cli.run_glob))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "params", "arm.yaml"))
    )
    print(f"[eval] {len(run_dirs)} runs under {args_cli.log_root}")
    for run_dir in run_dirs:
        for ckpt in pick_checkpoints(run_dir, args_cli.checkpoints):
            it = re.search(r"model_(\d+)\.pt", ckpt).group(1)
            out_path = os.path.join(run_dir, f"eval_frontier_{it}.json")
            if os.path.exists(out_path):
                print(f"[eval] SKIP (exists) {out_path}")
                continue
            runner.load(ckpt)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            # fresh episode boundaries, THEN fresh tallies: env.reset() fires the
            # probe on the previous policy's mid-flight episodes — those partial
            # outcomes must not leak into this checkpoint's counts.
            env.seed(args_cli.seed)
            env.reset()
            probe.succ = None
            results = evaluate(env, policy, probe, max_steps)
            results["checkpoint"] = os.path.basename(ckpt)
            results["success_fn"] = args_cli.success_fn
            results["eval_seed"] = args_cli.seed
            with open(out_path, "w") as f:
                json.dump(results, f, indent=1)
            print(f"[eval] {os.path.basename(run_dir)} @ iter {it}: "
                  f"macro_pass={results['mean_pass']:.3f} "
                  f"micro_pass={results['micro_pass']:.3f} "
                  f"per_level={results['per_level_pass']}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
