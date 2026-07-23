"""Curriculum terms for the frontier-teacher × Isaac Lab integration (ladder step 1:
terrain-level axis on Isaac-Velocity-Rough-Anymal-C-v0).

Every arm of the pre-registered experiment is a ``ManagerTermBase`` subclass that
replaces the stock ``terrain_levels`` curriculum term *under the same attribute
name* — keeping ``terrain_generator.curriculum=True`` so all arms train on an
identical row-graded terrain grid (a ``terrain_levels=null`` control would silently
switch the generator to per-tile random difficulty, changing the task itself).

Arms:
  StaticTerrainLevels     control — graded terrain, levels frozen at their init draw
  (stock terrain_levels_vel)  greedy adaptive baseline, unchanged
  ScriptedTerrainLevels   open-loop schedule — level cap ramps linearly over training
  UniformTerrainLevels    uniform resampling over all levels every reset
  FrontierTerrainTeacher  the frontier teacher (FrontierBinTeacher posterior)

Registration idiom (verified against this fork's manager code): pass the CLASS in
``CurrTerm(func=<Class>, params={...})``. The manager initializes it with
``func(cfg=term_cfg, env=env)`` once the sim plays, and ``class_to_dict`` /
Hydra round-trip the func as a ``module:ClassName`` string — a pre-built stateful
instance also works at call time but is the fragile path; the class is the contract.

Success signal is selected by the ``success_fn`` param as a STRING key into
``SUCCESS_FNS`` (strings survive the Hydra/env.yaml round-trip; closures do not):
  "survival"  — reached time_out without early termination
  "distance"  — walked ≥ ``distance_fraction`` of the commanded distance (the
                demote predicate of terrain_levels_vel, made binary)
  "tile"      — walked ≥ half a terrain tile (the promote predicate of
                terrain_levels_vel — greedy's move_up signal)

This module imports isaaclab lazily/optionally so it unit-tests on CPU without
Isaac Sim (mirrors frontier_rl's duck-typed adapter design).
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

# frontier_rl lives one directory up (scripts/curriculum-maxrl); make it importable
# when this module is loaded by in-container training scripts.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from frontier_rl.adapters.isaaclab_curriculum import FrontierBinTeacher  # noqa: E402

try:  # real base class in-container; tiny stub for CPU unit tests
    from isaaclab.managers import ManagerTermBase
except ImportError:

    class ManagerTermBase:  # type: ignore[no-redef]
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env

        def reset(self, env_ids=None):
            pass


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg


# ---------------------------------------------------------------------------
# Success predicates (module-level, selected by string key)
# ---------------------------------------------------------------------------

def _success_survival(env, env_ids) -> torch.Tensor:
    """Reached the episode time-out without terminating early."""
    return env.termination_manager.time_outs[env_ids]


def _walked_distance(env, env_ids) -> torch.Tensor:
    asset = env.scene["robot"]
    return torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)


def _success_distance(env, env_ids, fraction: float = 0.5,
                      command_name: str = "base_velocity") -> torch.Tensor:
    """Stock-compatible endpoint approximation of commanded-distance success.

    This uses the command visible at reset, as ``terrain_levels_vel`` does. It
    is not an integral over commands that may have been resampled mid-episode;
    use ``tile`` for the signal-identical promotion predicate.
    """
    command = env.command_manager.get_command(command_name)
    required = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * fraction
    return _walked_distance(env, env_ids) > required


def _success_tile(env, env_ids) -> torch.Tensor:
    """Walked at least half a terrain tile — greedy terrain_levels_vel's move_up signal."""
    tile = env.scene.terrain.cfg.terrain_generator.size[0] / 2
    return _walked_distance(env, env_ids) > tile


SUCCESS_FNS = {
    "survival": _success_survival,
    "distance": _success_distance,
    "tile": _success_tile,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _write_levels(terrain, env_ids, new_levels: torch.Tensor) -> None:
    """Write per-env terrain levels + spawn origins (the actuation contract of
    terrain_levels_vel / TerrainImporter.update_env_origins)."""
    terrain.terrain_levels[env_ids] = new_levels
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]


def _env_id_tensor(env_ids, device: torch.device) -> torch.Tensor:
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def _completed_env_ids(env, env_ids, device: torch.device) -> torch.Tensor:
    """Return only reset envs that actually completed at least one step."""
    episode_lengths = getattr(env, "episode_length_buf", None)
    if episode_lengths is None:
        raise RuntimeError("episode_length_buf is required to distinguish construction resets")
    ids = _env_id_tensor(env_ids, device)
    return ids[episode_lengths[ids] > 0]


class _TerrainCurriculumBase(ManagerTermBase):
    """Common plumbing: n_bins discovery, seeded generator, mean-level telemetry."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._n_bins: int | None = None
        self._torch_gen: torch.Generator | None = None

    def _bins(self, env) -> int:
        if self._n_bins is None:
            self._n_bins = int(env.scene.terrain.max_terrain_level)
            if self._n_bins <= 0:
                raise ValueError(f"terrain must expose at least one level, got {self._n_bins}")
        return self._n_bins

    def _gen(self, env) -> torch.Generator:
        if self._torch_gen is None:
            seed = getattr(env.cfg, "seed", None)
            self._torch_gen = torch.Generator(device=env.scene.terrain.terrain_levels.device)
            self._torch_gen.manual_seed(int(seed) if seed is not None else 0)
        return self._torch_gen


# ---------------------------------------------------------------------------
# Arm 1 — control: identical graded terrain, no curriculum motion
# ---------------------------------------------------------------------------

class StaticTerrainLevels(_TerrainCurriculumBase):
    """Do-nothing term. Keeps terrain_generator.curriculum=True (same terrain grid
    as every other arm) but never moves an env's level: the honest no-curriculum
    control. Levels stay at their init draw from [0, max_init_terrain_level]."""

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
        return torch.mean(env.scene.terrain.terrain_levels.float())


# ---------------------------------------------------------------------------
# Arm 3 — scripted: open-loop linear ladder
# ---------------------------------------------------------------------------

class ScriptedTerrainLevels(_TerrainCurriculumBase):
    """Open-loop schedule: the highest allowed level ramps linearly from 1 to
    n_bins over ``total_steps`` policy steps; resetting envs draw uniformly from
    the currently allowed range. The scripted-arm competitor from the D9 design."""

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int],
                 total_steps: int = 36000) -> dict[str, torch.Tensor | float]:
        if not isinstance(total_steps, (int, np.integer)) or total_steps <= 0:
            raise ValueError(f"total_steps must be a positive integer, got {total_steps!r}")
        terrain = env.scene.terrain
        n_bins = self._bins(env)
        progress = min(1.0, env.common_step_counter / total_steps)
        allowed_max = max(1, math.ceil(progress * n_bins))  # levels [0, allowed_max)
        new_levels = torch.randint(
            0, allowed_max, (len(env_ids),), generator=self._gen(env),
            device=terrain.terrain_levels.device, dtype=terrain.terrain_levels.dtype,
        )
        _write_levels(terrain, env_ids, new_levels)
        return {"mean_bin": torch.mean(terrain.terrain_levels.float()),
                "allowed_max": float(allowed_max)}


# ---------------------------------------------------------------------------
# Arm 4 — uniform: the frontier_rl "uniform" baseline
# ---------------------------------------------------------------------------

class UniformTerrainLevels(_TerrainCurriculumBase):
    """Uniform resampling over all levels at every reset — the uniform-sampling
    baseline of the frontier_rl REPORT (expected to waste resets at both ends)."""

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
        terrain = env.scene.terrain
        new_levels = torch.randint(
            0, self._bins(env), (len(env_ids),), generator=self._gen(env),
            device=terrain.terrain_levels.device, dtype=terrain.terrain_levels.dtype,
        )
        _write_levels(terrain, env_ids, new_levels)
        return torch.mean(terrain.terrain_levels.float())


# ---------------------------------------------------------------------------
# Arm 5 — the frontier teacher
# ---------------------------------------------------------------------------

class FrontierTerrainTeacher(_TerrainCurriculumBase):
    """Frontier-teacher curriculum over terrain levels.

    Per reset batch (fires in _reset_idx BEFORE scene.reset — verified contract):
      1. observe: (level, success?) of each ending episode → Beta posterior
         (evidence-scaled decay, half-life in episode-equivalents)
      2. sample:  next-episode levels ∝ learnability u(p̃)^γ + uniform floor
      3. actuate: write terrain_levels + env_origins for exactly these envs
      4. telemetry: dict → Curriculum/terrain_levels/* TensorBoard tags

    Teacher state is periodically checkpointed to
    ``<log_dir>/curriculum_teacher/teacher_state.json`` (their §4 item 3); pass
    ``load_state`` to resume. Eval uses the *_PLAY cfgs (curriculum disabled), so
    teacher state is never restored at eval — the SONIC rule holds by construction.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.teacher: FrontierBinTeacher | None = None
        self._calls = 0

    # -- lazy init (terrain grid only exists once the scene is built) --------
    def _ensure_teacher(self, env, params: dict) -> FrontierBinTeacher:
        if self.teacher is None:
            seed = getattr(env.cfg, "seed", None)
            self.teacher = FrontierBinTeacher(
                self._bins(env),
                utility=params.get("utility", "learnability"),
                advmass_n=params.get("advmass_n", 16),
                decay_half_life=params.get("decay_half_life", 2048.0),
                floor=params.get("floor", 0.1),
                gamma=params.get("gamma", 1.0),
                optimism_k=params.get("optimism_k", 1.0),
                thompson=params.get("thompson", False),
                max_prob=params.get("max_prob"),
                seed=int(seed) if seed is not None else params.get("seed", 0),
            )
            load_state = params.get("load_state")
            if load_state:
                with open(load_state) as f:
                    self.load_state_dict(json.load(f))
        return self.teacher

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int],
                 success_fn: str = "survival", distance_fraction: float = 0.5,
                 command_name: str = "base_velocity",
                 utility: str = "learnability", advmass_n: int = 16,
                 decay_half_life: float = 2048.0,
                 floor: float = 0.1, gamma: float = 1.0, optimism_k: float = 1.0,
                 thompson: bool = False, max_prob: float | None = None, seed: int = 0,
                 load_state: str | None = None,
                 save_every_calls: int = 200) -> dict[str, torch.Tensor | float]:
        if success_fn not in SUCCESS_FNS:
            raise ValueError(f"unknown success_fn {success_fn!r}; choose from {sorted(SUCCESS_FNS)}")
        if not np.isfinite(distance_fraction) or distance_fraction < 0:
            raise ValueError(f"distance_fraction must be non-negative and finite, got {distance_fraction}")
        if (
            not isinstance(save_every_calls, (int, np.integer))
            or save_every_calls < 0
        ):
            raise ValueError(
                f"save_every_calls must be a non-negative integer, got {save_every_calls!r}"
            )
        teacher = self._ensure_teacher(env, dict(
            utility=utility, advmass_n=advmass_n,
            decay_half_life=decay_half_life, floor=floor, gamma=gamma,
            optimism_k=optimism_k, thompson=thompson, max_prob=max_prob,
            seed=seed, load_state=load_state,
        ))
        # Q9 tripwire (SONIC_RESPONSE): probability ceiling as a safety assertion,
        # not shaping. Inactive by default at 10 bins (floor + effective_bins
        # telemetry are the auditable guards there); exposed for larger bin counts.
        teacher.max_prob = max_prob
        terrain = env.scene.terrain
        ids = _env_id_tensor(env_ids, terrain.terrain_levels.device)

        # 1) OBSERVE the episodes that just ended (origins/commands still theirs).
        # Guard: the RslRlVecEnvWrapper triggers one full reset at construction,
        # BEFORE any episode has run — observing it would inject num_envs fake
        # failures into the posterior. Filter per env rather than gating the
        # whole batch so mixed construction/completed reset batches stay valid.
        completed_ids = _completed_env_ids(
            env, ids, terrain.terrain_levels.device
        )
        if len(completed_ids) > 0:
            levels = terrain.terrain_levels[completed_ids]
            if success_fn == "distance":
                success = _success_distance(
                    env, completed_ids, distance_fraction, command_name
                )
            else:
                success = SUCCESS_FNS[success_fn](env, completed_ids)
            teacher.observe_resets(levels, ~success)

        # 2) SAMPLE + 3) ACTUATE next-episode difficulty for exactly these envs
        new_levels = torch.as_tensor(
            teacher.sample_bins(len(ids)),
            device=terrain.terrain_levels.device, dtype=terrain.terrain_levels.dtype,
        ).clamp_(0, self._bins(env) - 1)
        _write_levels(terrain, ids, new_levels)

        # 4) TELEMETRY (+ periodic state checkpoint)
        self._calls += 1
        if save_every_calls and self._calls % save_every_calls == 0:
            self._save_state(env)
        m = teacher.metrics()
        # P-A gate quantity (ISAACLAB_DESIGN §2): sampling mass on ZPD bins
        # (p̂ ∈ [0.2, 0.8]) — must exceed the uniform share for the mechanism
        # gate to pass. Computed here from the teacher's own posterior + probs
        # so the analyzer can read it as a TB series, not a final snapshot.
        p_hat = teacher.pass_rate_estimates()
        probs = teacher.sampling_probs()
        zpd = (p_hat > 0.2) & (p_hat < 0.8)
        return {
            "mean_bin": torch.mean(terrain.terrain_levels.float()),
            "n_bins": float(self._bins(env)),
            "frontier_bin": m["teacher/frontier_bin"],
            "dead_frac": m["teacher/frac_dead"],
            "mastered_frac": m["teacher/frac_mastered"],
            "effective_bins": m["teacher/effective_bins"],
            "max_prob": m["teacher/max_prob"],
            "seen_frac": m["teacher/seen_frac"],
            "zpd_mass": float(probs[zpd].sum()),
            "zpd_bins": float(zpd.sum()),
        }

    def state_dict(self) -> dict:
        return {
            "version": 3,
            "calls": self._calls,
            "teacher": self.teacher.state_dict() if self.teacher is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        if self.teacher is None:
            raise RuntimeError("teacher must be initialized before loading state")
        # Backward compatibility: original JSON files stored succ/fail at the
        # top level and calls as an auxiliary field.
        teacher_state = state.get("teacher") if "teacher" in state else state
        if teacher_state:
            self.teacher.load_state_dict(teacher_state)
        calls = state.get("calls", 0)
        if not isinstance(calls, (int, np.integer)) or calls < 0:
            raise ValueError(f"state calls must be a non-negative integer, got {calls!r}")
        self._calls = int(calls)

    def _save_state(self, env) -> None:
        log_dir = getattr(env.cfg, "log_dir", None)
        if not log_dir:
            return
        out_dir = os.path.join(log_dir, "curriculum_teacher")
        os.makedirs(out_dir, exist_ok=True)
        state = self.state_dict()
        state["pass_rates"] = self.teacher.pass_rate_estimates()
        state["sampling_probs"] = self.teacher.sampling_probs()

        def jsonable(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {key: jsonable(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [jsonable(item) for item in value]
            return value

        path = os.path.join(out_dir, "teacher_state.json")
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(jsonable(state), f)
        os.replace(tmp_path, path)

    def save_state(self, env) -> None:
        """Persist the latest human-readable state, including at run end."""
        self._save_state(env)


# ---------------------------------------------------------------------------
# Eval probe — per-level success measurement on a FIXED level grid
# ---------------------------------------------------------------------------

class FixedLevelProbe(_TerrainCurriculumBase):
    """Measurement term for eval: pins env i to level ``i % n_bins`` forever and
    tallies (level, success) per finished episode.

    This is the off-distribution readout P-B/P-C need: training telemetry is
    biased by each arm's own sampling distribution (an arm that never visits
    level 9 reports nothing about level 9). Running the SAME fixed grid against
    every arm's checkpoints makes the arms comparable. Reuses the verified
    curriculum-term seam: fires in _reset_idx before scene.reset, so the ending
    episode's final state is still readable (same contract as the teacher).

    Read ``succ``/``fail`` from the live term instance after rollout.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.succ: np.ndarray | None = None
        self.fail: np.ndarray | None = None

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int],
                 success_fn: str = "tile", distance_fraction: float = 0.5,
                 command_name: str = "base_velocity") -> dict[str, float]:
        if success_fn not in SUCCESS_FNS:
            raise ValueError(f"unknown success_fn {success_fn!r}; choose from {sorted(SUCCESS_FNS)}")
        if not np.isfinite(distance_fraction) or distance_fraction < 0:
            raise ValueError(f"distance_fraction must be non-negative and finite, got {distance_fraction}")
        n_bins = self._bins(env)
        terrain = env.scene.terrain
        ids = _env_id_tensor(env_ids, terrain.terrain_levels.device)
        if self.succ is None:
            self.succ = np.zeros(n_bins)
            self.fail = np.zeros(n_bins)
        # Observe only envs with a completed episode. This also skips the
        # construction-time full reset without misclassifying a mixed batch.
        completed_ids = _completed_env_ids(
            env, ids, terrain.terrain_levels.device
        )
        if len(completed_ids) > 0:
            levels = terrain.terrain_levels[completed_ids].cpu().numpy()
            if success_fn == "distance":
                success = _success_distance(
                    env, completed_ids, distance_fraction, command_name
                )
            else:
                success = SUCCESS_FNS[success_fn](env, completed_ids)
            success = success.cpu().numpy().astype(bool)
            np.add.at(self.succ, levels[success], 1.0)
            np.add.at(self.fail, levels[~success], 1.0)
        # pin: env i always runs level i % n_bins
        assigned = (ids % n_bins).to(dtype=terrain.terrain_levels.dtype)
        _write_levels(terrain, ids, assigned)
        total = self.succ + self.fail
        with np.errstate(invalid="ignore", divide="ignore"):
            per_level = np.where(total > 0, self.succ / np.maximum(total, 1), np.nan)
        macro_pass = float(np.nanmean(per_level)) if np.any(total > 0) else 0.0
        return {"episodes": float(total.sum()),
                "mean_pass": macro_pass,
                "micro_pass": float((self.succ.sum() / total.sum()) if total.sum() else 0.0)}

    def results(self) -> dict:
        if self.succ is None or self.fail is None:
            return {
                "per_level_pass": [],
                "episodes_per_level": [],
                "mean_pass": 0.0,
                "micro_pass": 0.0,
            }
        total = self.succ + self.fail
        with np.errstate(invalid="ignore", divide="ignore"):
            per_level = np.where(total > 0, self.succ / np.maximum(total, 1), np.nan)
        macro_pass = float(np.nanmean(per_level)) if np.any(total > 0) else 0.0
        return {
            "per_level_pass": [None if np.isnan(x) else round(float(x), 4) for x in per_level],
            "episodes_per_level": total.tolist(),
            "mean_pass": macro_pass,
            "micro_pass": float(self.succ.sum() / total.sum()) if total.sum() else 0.0,
        }


# ---------------------------------------------------------------------------
# Arm wiring helper (used by the launcher; usable from any custom cfg too)
# ---------------------------------------------------------------------------

ARM_TERMS = {
    "control": StaticTerrainLevels,
    "greedy": None,  # stock terrain_levels_vel — leave the task cfg untouched
    "scripted": ScriptedTerrainLevels,
    "uniform": UniformTerrainLevels,
    "teacher": FrontierTerrainTeacher,
}


def apply_arm(env_cfg, arm: str, *, success_fn: str = "survival",
              scripted_total_steps: int = 36000, teacher_params: dict | None = None):
    """Mutate a LocomotionVelocityRoughEnvCfg-style cfg in place to run one arm.

    Replaces ``env_cfg.curriculum.terrain_levels.func`` under the SAME term name so
    ``terrain_generator.curriculum`` stays True and the terrain grid is identical
    across arms. Must be called after Hydra parsing / before gym.make (new params
    cannot be injected via CLI overrides in this fork — update_class_from_dict
    rejects unknown keys).
    """
    if arm not in ARM_TERMS:
        raise ValueError(f"Unknown arm '{arm}'. Choose from {sorted(ARM_TERMS)}.")
    if arm == "greedy":
        return env_cfg
    term = env_cfg.curriculum.terrain_levels
    term.func = ARM_TERMS[arm]
    if arm == "control" or arm == "uniform":
        term.params = {}
    elif arm == "scripted":
        term.params = {"total_steps": scripted_total_steps}
    elif arm == "teacher":
        term.params = {"success_fn": success_fn, **(teacher_params or {})}
    return env_cfg
