# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train an RL agent with RSL-RL under one arm of the frontier-curriculum experiment.

A thin fork of scripts/reinforcement_learning/rsl_rl/train.py that adds:
  --arm {control,greedy,scripted,uniform,teacher}   which curriculum term to run
  --success_fn {survival,distance,tile}             teacher's binary evidence signal
  --scripted_total_steps N                          scripted arm's ramp length
  --teacher_param key=value (repeatable)            teacher knob overrides

The arm swap happens on the parsed env cfg (after Hydra, before gym.make), under
the same `curriculum.terrain_levels` term name, so the terrain grid and everything
else stay identical across arms. Run inside the isaac-lab-base container:

  /isaac-sim/python.sh scripts/curriculum-maxrl/isaaclab_integration/train_frontier.py \
      --task Isaac-Velocity-Rough-Anymal-C-v0 --headless --num_envs 1024 \
      --max_iterations 600 --seed 42 --arm teacher --success_fn tile \
      agent.run_name=teacher_s42
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports: rsl_rl cli_args from the stock script dir + our integration pkg
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # IsaacLab root
sys.path.append(os.path.join(_REPO, "scripts", "reinforcement_learning", "rsl_rl"))
sys.path.append(os.path.dirname(_HERE))  # scripts/curriculum-maxrl
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Frontier-curriculum RSL-RL training.")
parser.add_argument("--arm", type=str, default="teacher",
                    choices=["control", "greedy", "scripted", "uniform", "teacher",
                             "teacher_g4", "hybrid"],
                    help="Curriculum arm to run (greedy = stock terrain_levels_vel; "
                         "teacher_g4 = concentration-fixed teacher; hybrid = "
                         "per-env walk + posterior masks).")
parser.add_argument("--success_fn", type=str, default="survival",
                    choices=["survival", "distance", "tile"],
                    help="Teacher's binary success predicate.")
parser.add_argument("--scripted_total_steps", type=int, default=36000,
                    help="Scripted arm: policy steps for the level ramp "
                         "(default 1500 iters * 24 steps).")
parser.add_argument("--teacher_param", action="append", default=[],
                    help="Teacher knob override key=value (float/int/bool parsed). "
                         "Keys: decay_half_life, floor, gamma, optimism_k, thompson, "
                         "utility, advmass_n, max_prob, load_state, save_every_calls, "
                         "distance_fraction.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Anymal-C-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_integration.frontier_terms import FrontierTerrainTeacher, apply_arm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_CURRICULUM_STATE_KEY = "frontier_curriculum_state"


def _parse_teacher_params(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, separator, raw = pair.partition("=")
        if not separator or not key or not raw:
            raise ValueError(f"teacher parameters must use non-empty key=value syntax, got {pair!r}")
        if raw.lower() in ("true", "false"):
            out[key] = raw.lower() == "true"
        else:
            try:
                out[key] = int(raw) if raw.lstrip("-").isdigit() else float(raw)
            except ValueError:
                out[key] = raw
    return out


def _live_terrain_term(env):
    """Return the instantiated terrain-level term from CurriculumManager."""
    manager = env.unwrapped.curriculum_manager
    for name, term_cfg in zip(manager._term_names, manager._term_cfgs):
        if name == "terrain_levels":
            return term_cfg.func
    return None


class FrontierOnPolicyRunner(OnPolicyRunner):
    """RSL-RL runner that couples teacher state to every model checkpoint."""

    def __init__(self, *args, curriculum_term=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_term = curriculum_term
        self.curriculum_state_restored = False

    def save(self, path: str, infos: dict | None = None) -> None:
        if self.curriculum_term is None:
            return super().save(path, infos)
        payload = dict(infos or {})
        payload[_CURRICULUM_STATE_KEY] = self.curriculum_term.state_dict()
        super().save(path, payload)

    def load(self, path: str, *args, **kwargs) -> dict:
        infos = super().load(path, *args, **kwargs)
        state = infos.get(_CURRICULUM_STATE_KEY) if isinstance(infos, dict) else None
        if state is not None and self.curriculum_term is not None:
            self.curriculum_term.load_state_dict(state)
            self.curriculum_state_restored = True
        return infos


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL under the selected curriculum arm."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ---- the arm swap (after Hydra, before gym.make) ----
    teacher_params = _parse_teacher_params(args_cli.teacher_param)
    apply_arm(
        env_cfg, args_cli.arm,
        success_fn=args_cli.success_fn,
        scripted_total_steps=args_cli.scripted_total_steps,
        teacher_params=teacher_params,
    )

    # experiment/log naming: arm goes into the run name for clean TB comparison
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.arm}"
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    start_time = time.time()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    terrain_term = _live_terrain_term(env)
    curriculum_term = terrain_term if isinstance(terrain_term, FrontierTerrainTeacher) else None
    runner = FrontierOnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
        curriculum_term=curriculum_term,
    )
    try:  # the container mounts scripts/ without the repo's .git — non-fatal
        runner.add_git_repo_to_log(__file__)
    except Exception as e:
        print(f"[WARN] Skipping git state logging: {e}")
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        if curriculum_term is not None and runner.curriculum_state_restored:
            # The wrapper's construction reset happened before runner.load().
            # Reset once more so the first resumed episodes are assigned from
            # the restored teacher distribution; zero episode lengths keep
            # this reset out of the evidence stream.
            env.reset()
        if (
            curriculum_term is not None
            and not runner.curriculum_state_restored
            and not teacher_params.get("load_state")
        ):
            raise RuntimeError(
                "Teacher-arm checkpoint has no embedded curriculum state. "
                "Pass --teacher_param load_state=<legacy teacher_state.json> "
                "or resume from a checkpoint produced by this launcher."
            )

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "arm.yaml"), {
        "arm": args_cli.arm, "success_fn": args_cli.success_fn,
        "scripted_total_steps": args_cli.scripted_total_steps,
        "teacher_params": teacher_params,
    })

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    if curriculum_term is not None:
        curriculum_term.save_state(env.unwrapped)
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
