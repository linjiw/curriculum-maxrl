"""Live glue: CosmosLiberoSpace against the real cosmos-framework stack.

Verified against the checkout at
`cosmos3edge/cosmos-framework` (2026-07-23), targeting these real surfaces:

  - `cosmos_framework/simulation/libero/closed_loop_eval.py`:
      ActionEnvironmentClient(server_url, domain_name, prompt, image_size,
      timeout) — `.prompt` is a plain attribute sent with every request, so
      per-task conditioning = set it before each group's wave;
      `.predict_batch(list-of-multiview-obs) -> action chunks` (ONE diffusion
      forward for the whole wave);
      `_run_task_vectorized(...)` — the wave loop we mirror (SubprocVectorEnv
      with the spawn-safe `_LiberoEnvFactory`, `venv.set_init_state(states,
      id=slots)`, per-step active-mask, success from `info["success"]`);
      `_augment_task_prompt_with_viewpoint`, `_get_libero_images`,
      `_format_action`, `_framewise_action_to_delta`, `_remap_gripper`,
      `TASK_MAX_STEPS`.
  - LIBERO: `task_suite.get_task_init_states(task_id)` (init-state control),
      `env.parsed_problem["goal_state"]` (the BDDL goal conjunction, e.g.
      [["Open", "microwave_1"], ["In", "bowl_1", "microwave_1"]]) — the
      oracle predicate vocabulary.
  - `cosmos_framework/model/generator/algorithm/loss/flow_matching.py`:
      `compute_flow_matching_loss` returns `(weighted_mean, per_instance
      [B])` — the per-instance vector is the hook: Phase-1's weighted-CFM
      step is `(weights_B * per_instance_loss_B).sum() / weights_B.sum()`
      instead of `.mean()`; no loss-math change.

This module imports NONE of that at module level (same lazy pattern as the
isaaclab adapter) — every cosmos/libero touchpoint arrives via constructor
injection, so the file unit-tests on CPU with fakes and imports cleanly in
the curriculumrl venv.  What it provides:

  LiveRolloutBackend   — rollout_fn for CosmosLiberoSpace: one group = one
                         wave of N episodes of ONE (task, init-state) pair
  make_oracle_verifier — achieved-predicate extraction recorded AT EPISODE
                         END while the sim state is alive (the verifier_fn
                         then just reads info; matches the mock's contract)
  WeightedCFMBuffer    — the Policy side of Phase 1: update() appends
                         (trajectory ref, REWRITTEN goal, weight) rows to a
                         JSONL manifest the 24 GB weighted-SFT trainer
                         consumes; one manifest per training round
  Phase1Round          — the round loop skeleton: collect -> write manifest
                         -> (external) weighted SFT -> redeploy -> repeat

Design rule carried over from the response doc: live groups are verified
ONLY by the sim's binary success; predicate extraction exists for DEAD
groups (relabeling) and for eval-time poison measurement.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# predicate extraction (oracle arm)
# ---------------------------------------------------------------------------
def goal_predicates_of(parsed_problem: dict) -> list[str]:
    """BDDL goal conjunction -> canonical predicate strings.

    LIBERO's parsed_problem["goal_state"] is a list of s-expressions like
    ["Open", "microwave_1"] or ["In", "akita_black_bowl_1", "microwave_1"].
    Canonical form: lowercase head + comma-joined args — the same strings
    CosmosLiberoSpace uses as its Predicate type and PoisonRateMeter
    classes by head symbol.
    """
    out = []
    for expr in parsed_problem.get("goal_state", []):
        head, *args = expr
        out.append(f"{str(head).lower()}({','.join(str(a) for a in args)})")
    return out


def make_oracle_verifier(eval_predicate: Callable[[Any, list], bool]):
    """Build the end-of-episode predicate snapshotter + the verifier_fn pair.

    eval_predicate(env, expr) -> bool: evaluates ONE goal s-expression on a
    live env (LIBERO exposes this through its predicate registry — inject
    the suite's own evaluator so this module stays sim-free).  The
    snapshotter runs inside the wave loop while envs are alive; verifier_fn
    is pure info-reading afterwards, matching CosmosLiberoSpace's contract.
    """
    def snapshot(env, parsed_problem: dict) -> set:
        achieved = set()
        for expr in parsed_problem.get("goal_state", []):
            try:
                if eval_predicate(env, expr):
                    head, *args = expr
                    achieved.add(
                        f"{str(head).lower()}({','.join(str(a) for a in args)})")
            except Exception:      # noqa: BLE001 — a predicate that cannot
                pass               # be evaluated is simply not achieved
        return achieved

    def verifier_fn(info: dict) -> set:
        return set(info.get("achieved_predicates", set()))

    return snapshot, verifier_fn


# ---------------------------------------------------------------------------
# rollout backend (one group = one wave)
# ---------------------------------------------------------------------------
@dataclass
class LiveRolloutBackend:
    """rollout_fn for CosmosLiberoSpace over the policy server + vector envs.

    Injected pieces (all from cosmos-framework / LIBERO, resolved by the
    launch script that lives in that repo's venv):

      client: ActionEnvironmentClient — .prompt mutated per group (that is
        the documented conditioning path; predict/predict_batch read it).
      venv_of_task(task_id) -> (venv, parsed_problem, max_steps): a cached
        SubprocVectorEnv built from the task's BDDL via _LiberoEnvFactory,
        plus its parsed problem and the suite's TASK_MAX_STEPS cap.
      init_states_of(task_id, init_bin, n) -> np.ndarray [n, ...]: init
        states for the wave — task_suite.get_task_init_states(task_id)
        sliced by bin (or the JSON custom-state path).  A fixed bin must
        return the SAME states each call (Q2.4: groups are N episodes of
        one arm).
      get_images(obs) -> list[np.ndarray]: per-env multi-view frames
        (_get_libero_images bound with cameras/flip/rotate args).
      action_to_env(raw_chunk_row) -> list[float]: _format_action +
        _framewise_action_to_delta + _remap_gripper bound with the run's
        action_dim/rotation/gripper config.
      predicate_snapshot(env_handle, parsed_problem) -> set | None: the
        oracle snapshotter (None disables predicate capture — self-verified
        arms extract from recorded frames instead).

    Telemetry lands on `ledger` (frontier_rl.evaluation.RunLedger) if given.
    """
    client: Any
    venv_of_task: Callable
    init_states_of: Callable
    get_images: Callable
    action_to_env: Callable
    prompt_of_template: Callable[[str], str] = lambda t: t
    predicate_snapshot: Optional[Callable] = None
    action_horizon: int = 16
    warmup_steps: int = 10
    dummy_action: Sequence[float] = (0., 0., 0., 0., 0., 0., -1.)
    record_frames: bool = True
    ledger: Any = None

    def __call__(self, template: str, init_bin: Optional[int], n: int,
                 *, task_id: int) -> tuple:
        """One group: n parallel episodes of one (task, init-bin) pair."""
        self.client.prompt = self.prompt_of_template(template)
        venv, parsed_problem, max_steps = self.venv_of_task(task_id)
        states = self.init_states_of(task_id, init_bin, n)
        slots = list(range(n))
        t0 = time.perf_counter()

        venv.reset(id=slots)
        obs_arr = venv.set_init_state(np.asarray(states, np.float64), id=slots)
        obs = {s: obs_arr[i] for i, s in enumerate(slots)}
        done = {s: False for s in slots}
        succ = {s: False for s in slots}
        frames: dict[int, list] = {s: [] for s in slots}
        actions: dict[int, list] = {s: [] for s in slots}
        step, batches = 0, 0

        for _ in range(self.warmup_steps):
            act = np.stack([list(self.dummy_action) for _ in slots])
            obs_arr, _, _, _ = venv.step(act, id=slots)
            for i, s in enumerate(slots):
                obs[s] = obs_arr[i]
            step += 1

        while step < max_steps:
            active = [s for s in slots if not done[s]]
            if not active:
                break
            imgs = [self.get_images(obs[s]) for s in active]
            if self.record_frames:
                for s, im in zip(active, imgs):
                    frames[s].append(im[0])
            chunks = self.client.predict_batch(imgs)     # ONE diffusion fwd
            batches += 1
            by_slot = {s: chunks[k] for k, s in enumerate(active)}
            horizon = (self.action_horizon if self.action_horizon > 0
                       else len(chunks[0]))
            for h in range(horizon):
                cur = [s for s in slots if not done[s]]
                if not cur or step >= max_steps:
                    break
                env_actions = [self.action_to_env(by_slot[s][h]) for s in cur]
                obs_arr, _, d, info = venv.step(np.stack(env_actions), id=cur)
                step += 1
                for i, s in enumerate(cur):
                    obs[s] = obs_arr[i]
                    actions[s].append(env_actions[i])
                    ii = info[i] if isinstance(info, (list, np.ndarray)) else info
                    if isinstance(ii, dict) and ii.get("success"):
                        done[s], succ[s] = True, True
                    elif bool(d[i]):
                        done[s] = True
                        succ[s] = (ii.get("success", True)
                                   if isinstance(ii, dict) else True)

        # predicate snapshot while envs are alive (oracle arm; dead-group use)
        achieved: dict[int, set] = {s: set() for s in slots}
        if self.predicate_snapshot is not None:
            for s in slots:
                if not succ[s]:      # only failures ever need relabel evidence
                    achieved[s] = self.predicate_snapshot(
                        _worker_env(venv, s), parsed_problem)

        rewards = np.array([float(succ[s]) for s in slots])
        trajs = [{"language_goal": template, "task_id": task_id,
                  "init_bin": init_bin, "frames": frames[s],
                  "actions": actions[s]} for s in slots]
        infos = [{"success": bool(succ[s]),
                  "achieved_predicates": achieved[s],
                  "steps": len(actions[s])} for s in slots]
        if self.ledger is not None:
            self.ledger.observe_group(
                n, sum(len(a) for a in actions.values()),
                live=bool(rewards.sum()), server_batches=batches)
            self.ledger.wall_seconds += time.perf_counter() - t0
        return rewards, trajs, infos


def _worker_env(venv, slot: int):
    """Best-effort handle to one worker's env for predicate evaluation.

    LIBERO's SubprocVectorEnv keeps workers in subprocesses; predicate
    evaluation needs sim state.  Two supported routes: (a) venvs that expose
    a `get_env_state`/attr passthrough — pass a predicate_snapshot that uses
    it; (b) run predicate evaluation via `venv.env_method` if available.
    This helper centralizes the fallback so backends can override one place.
    """
    for attr in ("workers", "envs"):
        ws = getattr(venv, attr, None)
        if ws is not None:
            try:
                return ws[slot]
            except Exception:  # noqa: BLE001
                pass
    return venv  # snapshot fn must then use venv-level APIs (env_method)


# ---------------------------------------------------------------------------
# weighted-CFM buffer (the Policy side of Phase 1)
# ---------------------------------------------------------------------------
class WeightedCFMBuffer:
    """Policy.update -> JSONL manifest for the weighted flow-matching SFT.

    Phase 1 is round-based RFT: the trainer loop calls update() with the
    estimator's weights; rows with weight 0 are dropped (failures under
    positive-part weights, all-pass groups); each row carries the REWRITTEN
    language goal (already rebuilt by CosmosLiberoSpace.relabel for
    relabeled groups — this class must never see a raw failed-task goal on
    a relabeled row).  The cosmos side consumes the manifest by multiplying
    `compute_flow_matching_loss`'s per-instance vector:

        loss = (w * per_instance_loss).sum() / w.sum()

    save_traj(traj) -> str: persists frames/actions (e.g. mp4 + npz under
    the round dir) and returns the reference the manifest stores.  Kept
    injectable so unit tests run without disk-heavy video encoding.
    """

    def __init__(self, manifest_path: str | Path,
                 save_traj: Optional[Callable] = None):
        self.path = Path(manifest_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.save_traj = save_traj or (lambda traj: "")
        self.rows = 0

    def update(self, task_id: int, trajectories: Sequence, weights) -> None:
        weights = np.asarray(weights, dtype=float)
        with self.path.open("a", encoding="utf-8") as f:
            for traj, w in zip(trajectories, weights):
                if w == 0.0:
                    continue
                row = {
                    "task_id": int(task_id),
                    "weight": float(w),
                    "language_goal": traj["language_goal"],
                    "init_bin": traj.get("init_bin"),
                    "relabeled": int(task_id != traj.get("task_id", task_id)),
                    "n_actions": len(traj.get("actions", [])),
                    "data_ref": self.save_traj(traj),
                }
                f.write(json.dumps(row) + "\n")
                self.rows += 1


# ---------------------------------------------------------------------------
# round loop skeleton
# ---------------------------------------------------------------------------
@dataclass
class Phase1Round:
    """One collect->train round of the Phase-1 RFT loop.

    The trainer (FrontierTrainer with positive_weights=True) runs
    `groups_per_round` steps against the live backend, filling the round's
    manifest; `launch_sft(manifest_path, round_idx) -> new_checkpoint` is
    the external weighted-SFT call (the 24 GB stack); `redeploy(ckpt)`
    points the policy server at it.  Teacher state persists across rounds
    (`teacher.state_dict()` into the round dir) — the posterior IS the
    curriculum's memory; losing it resets the frontier walk.
    """
    trainer: Any
    buffer: WeightedCFMBuffer
    launch_sft: Callable[[Path, int], str]
    redeploy: Callable[[str], None]
    round_dir: Path
    groups_per_round: int = 40

    def run(self, round_idx: int) -> dict:
        steps = max(1, self.groups_per_round // self.trainer.cfg.tasks_per_step)
        stats = self.trainer.train(steps=steps)
        state_path = Path(self.round_dir) / f"teacher_round{round_idx}.json"
        state = {k: np.asarray(v).tolist()
                 for k, v in self.trainer.teacher.state_dict().items()}
        state_path.write_text(json.dumps(state))
        ckpt = self.launch_sft(self.buffer.path, round_idx)
        self.redeploy(ckpt)
        return {
            "round": round_idx,
            "manifest_rows": self.buffer.rows,
            "live_groups": sum(s.live_groups for s in stats),
            "dead_groups": sum(s.dead_groups for s in stats),
            "relabeled_groups": sum(s.relabeled_groups for s in stats),
            "checkpoint": ckpt,
        }
