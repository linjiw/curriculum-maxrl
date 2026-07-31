"""Dense-native-reward REINFORCE baseline (user challenge: standard RL
on CartPole should NOT be zero).

What standard libraries (SB3 etc.) train on is the env's NATIVE reward:
  MountainCar-v0: -1 per step (all returns -200 unless the flag is hit
                  within truncation -> zero return variance -> REINFORCE
                  has no gradient; the known hard-exploration flatline)
  CartPole-v1:    +1 per step (returns vary immediately -> dense signal;
                  vanilla REINFORCE solves it)

This arm: same TilePolicy (shared), same episode budget as the
convergence study, REINFORCE with batch-mean-baselined, std-normalized
trajectory returns on the NATIVE reward. Report in the same currencies:
MC flag rate (x>=0.5 within 200 steps), CP P(survive>=400) + mean length.

Prediction (registered here before running): CartPole dense SOLVES
(>=0.9 survive-400), MountainCar dense flatlines (~0.0 flag) — the
'standard RL is zero' claim is honest for MountainCar only.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np

from frontier_rl.adapters.gym_classic import TilePolicy

try:
    import gymnasium as gym
except ImportError as e:
    raise ImportError("pip install gymnasium") from e

# match the convergence study budget: episodes = steps * tasks_per_step *
# n_rollouts at cap (mc 2400*6*10... that is per-group; convergence study
# trains 6 tasks/step x 10 rollouts = 60 episodes per step, cap 2400 steps
# -> 144k episodes at cap; it converged by ~500 steps = 30k episodes.
# Dense baseline gets the SAME max episode budget as the cap.
BUDGET = {"mc": 2400 * 60, "cp": 1600 * 60}
BATCH = 60          # episodes per update, matching one trainer step
EVAL_EVERY = 40 * 60  # episodes between evals (= 40 trainer steps)


def run_env(env_name, seed):
    if env_name == "mc":
        env = gym.make("MountainCar-v0").unwrapped
        pol = TilePolicy([12, 12], [-1.2, -0.07], [0.6, 0.07],
                         n_tasks=1, n_actions=3, lr=0.15, seed=seed,
                         shared=True)
        max_steps = 200
    else:
        env = gym.make("CartPole-v1").unwrapped
        pol = TilePolicy([6, 6, 8, 8], [-2.4, -3.0, -0.21, -3.0],
                         [2.4, 3.0, 0.21, 3.0],
                         n_tasks=1, n_actions=2, lr=0.1, seed=seed,
                         shared=True)
        max_steps = 500
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)

    def episode(record=True):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        traj, ret, steps_alive, max_x = [], 0.0, 0, -1.2
        for _ in range(max_steps):
            a, tile = pol.act(obs, 0)
            if record:
                traj.append((tile, a))
            obs, r, term, trunc, _ = env.step(a)
            ret += float(r)
            steps_alive += 1
            max_x = max(max_x, float(obs[0]))
            if term or trunc:
                break
        return traj, ret, steps_alive, max_x

    def evaluate(n=24):
        flags, survived, lens = 0, 0, []
        for _ in range(n):
            _, ret, alive, max_x = episode(record=False)
            if env_name == "mc" and max_x >= 0.5:
                flags += 1
            if env_name == "cp":
                lens.append(alive)
                if alive >= 400:
                    survived += 1
        if env_name == "mc":
            return {"flag": flags / n}
        return {"survive400": survived / n, "mean_len": float(np.mean(lens))}

    curve = []
    episodes_done = 0
    while episodes_done < BUDGET[env_name]:
        trajs, rets = [], []
        for _ in range(BATCH):
            traj, ret, _, _ = episode()
            trajs.append(traj)
            rets.append(ret)
        episodes_done += BATCH
        rets = np.asarray(rets, float)
        sd = rets.std()
        w = (rets - rets.mean()) / (sd + 1e-8) if sd > 1e-8 else np.zeros_like(rets)
        # zero-variance batch (MountainCar all -200): no gradient — exactly
        # the standard-REINFORCE behavior we are measuring
        pol.update(0, trajs, w / BATCH)
        if episodes_done % EVAL_EVERY == 0:
            m = evaluate()
            curve.append({"episodes": episodes_done, **m})
            print(f"  {env_name} seed{seed} ep{episodes_done}: {m}", flush=True)
            # early exit when solved
            if env_name == "cp" and m["survive400"] >= 0.99:
                break
            if env_name == "mc" and m["flag"] >= 0.99:
                break
    final = evaluate(n=100)
    return {"curve": curve, "final": final}


if __name__ == "__main__":
    out = {}
    for env_name in ("cp", "mc"):
        runs = [run_env(env_name, s) for s in range(3)]
        out[env_name] = runs
        key = "survive400" if env_name == "cp" else "flag"
        vals = [r["final"][key] for r in runs]
        print(f"{env_name} dense-native REINFORCE final {key}: "
              f"{np.mean(vals):.3f}±{np.std(vals):.3f}  {vals}", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "gym_dense_baseline.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
