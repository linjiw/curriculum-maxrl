"""IsaacLab adapter: the frontier teacher as a curriculum/command term for
ManagerBasedRLEnv workflows (massively parallel GPU RL, e.g. GR00T/GEAR-SONIC
humanoid tracking, locomotion terrains, manipulation goal spaces).

IsaacLab's native pattern (verified against isaaclab.managers and a production
fork, gear_sonic/envs/manager_env/mdp/curriculum.py):

  - a *curriculum term* is a function `(env, env_ids, ...) -> tensor` invoked
    by the CurriculumManager on episode resets;
  - task assignment lives in the *command manager* (which reference motion /
    goal each env tracks);
  - success/failure lives in the *termination manager* (`reset_terminated`).

frontier_rl maps onto this WITHOUT the episodic-group loop of FrontierTrainer:
in a 4096-env sim there are no discrete "groups"; instead every reset event is
one Bernoulli observation for its env's task bin.  The teacher runs in
aggregate:

  observe:  on reset, (bin, terminated-early?) pairs stream into the posterior
  sample:   resampled envs draw their next bin from the teacher distribution

This file has NO isaaclab import at module level — the term functions receive
`env` duck-typed, so the module is unit-testable on CPU (see
isaaclab_integration/test_frontier_terms.py) and imports cleanly in a sim-less
venv. Dense-reward PPO note (per SONIC_RESPONSE.md): with hazard-style
per-reset evidence and no natural group size, `utility="learnability"` is the
recommended default; advmass needs an explicit N with visits-per-recompute
semantics.

Usage sketch inside an IsaacLab task cfg:

    from frontier_rl.adapters.isaaclab_curriculum import FrontierBinTeacher

    teacher = FrontierBinTeacher(n_bins=len(bins), utility="learnability",
                                 decay_half_life=2048)   # episode-equivalents

    # 1. termination hook (call each step or on resets):
    #    teacher.observe_resets(bin_ids=env_bins[reset_ids],
    #                           failed=env.termination_manager.terminated[reset_ids])
    # 2. command/reset hook (when assigning new tasks to reset envs):
    #    new_bins = teacher.sample_bins(len(reset_ids))
    # 3. optional telemetry:  logger.log(teacher.metrics(), step)
"""

from __future__ import annotations

import copy

import numpy as np


class FrontierBinTeacher:
    """Vectorized frontier teacher for massively parallel reset streams.

    Differences from frontier_rl.teacher.FrontierTeacher, each forced by the
    parallel-sim setting and consistent with SONIC_RESPONSE.md:

    - evidence-scaled decay: a half-life in *episode-equivalents* rather than
      per-observe decay, so behavior is invariant to env count (Q4);
    - deterministic utility-space optimism over a mean ± k*std interval by
      default instead of Thompson — reproducibility guardrails in sim RL favor
      determinism (Q3); pass thompson=True to restore sampling;
    - utility="learnability" default (no natural N in a reset stream, Q2);
    - a max_prob tripwire instead of a shaping cap (Q9).
    """

    def __init__(self, n_bins: int, *, utility: str = "learnability",
                 advmass_n: int = 16, decay_half_life: float = 2048.0,
                 floor: float = 0.1, gamma: float = 1.0,
                 optimism_k: float = 1.0, thompson: bool = False,
                 max_prob: float | None = None, seed: int = 0):
        if not isinstance(n_bins, (int, np.integer)) or n_bins <= 0:
            raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
        if utility not in ("learnability", "advmass"):
            raise ValueError(f"utility must be 'learnability' or 'advmass', got {utility!r}")
        if not isinstance(advmass_n, (int, np.integer)) or advmass_n < 2:
            raise ValueError(f"advmass_n must be at least 2, got {advmass_n}")
        if not np.isfinite(decay_half_life) or decay_half_life <= 0:
            raise ValueError(f"decay_half_life must be positive and finite, got {decay_half_life}")
        if not 0.0 <= floor <= 1.0:
            raise ValueError(f"floor must be in [0, 1], got {floor}")
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError(f"gamma must be positive and finite, got {gamma}")
        if not np.isfinite(optimism_k) or optimism_k < 0:
            raise ValueError(f"optimism_k must be non-negative and finite, got {optimism_k}")
        self.n_bins = n_bins
        self.utility_kind = utility
        self.advmass_n = int(advmass_n)
        self.half_life = float(decay_half_life)
        self.floor = float(floor)
        self.gamma = float(gamma)
        self.optimism_k = float(optimism_k)
        self.thompson = bool(thompson)
        self._max_prob: float | None = None
        self.max_prob = max_prob
        self.rng = np.random.default_rng(seed)
        self.succ = np.zeros(n_bins)
        self.fail = np.zeros(n_bins)
        self._probs = np.full(n_bins, 1.0 / n_bins)
        self._dirty = True

    @property
    def max_prob(self) -> float | None:
        return self._max_prob

    @max_prob.setter
    def max_prob(self, value: float | None) -> None:
        if value is not None and (not np.isfinite(value) or not 0.0 < value <= 1.0):
            raise ValueError(f"max_prob must be in (0, 1], got {value}")
        self._max_prob = None if value is None else float(value)

    # -- evidence (vectorized) ---------------------------------------------
    def observe_resets(self, bin_ids, failed) -> None:
        """Stream one batch of reset events.

        Args:
          bin_ids: int array/tensor (M,) — the bin each resetting env was on.
          failed:  bool array/tensor (M,) — early-termination flag per env.

        Accepts torch tensors (any device) or numpy — IsaacLab curriculum
        terms hand us CUDA tensors; the teacher's own state is tiny, so a
        single small D2H copy per reset batch is the cheap direction.
        """
        if hasattr(bin_ids, "cpu"):
            bin_ids = bin_ids.detach().cpu().numpy()
        if hasattr(failed, "cpu"):
            failed = failed.detach().cpu().numpy()
        bin_ids = np.asarray(bin_ids, dtype=int)
        failed = np.asarray(failed, dtype=bool)
        if bin_ids.ndim != 1 or failed.ndim != 1:
            raise ValueError(
                f"bin_ids and failed must be one-dimensional, got {bin_ids.shape} and {failed.shape}"
            )
        if bin_ids.shape != failed.shape:
            raise ValueError(f"bin_ids and failed must have the same shape, got {bin_ids.shape} and {failed.shape}")
        if np.any((bin_ids < 0) | (bin_ids >= self.n_bins)):
            raise ValueError(f"bin_ids must be in [0, {self.n_bins}), got {bin_ids}")
        n_events = bin_ids.size
        if n_events == 0:
            return
        # evidence-scaled decay: half-life in episode-equivalents
        decay = 0.5 ** (n_events / self.half_life)
        self.succ *= decay
        self.fail *= decay
        np.add.at(self.succ, bin_ids[~failed], 1.0)
        np.add.at(self.fail, bin_ids[failed], 1.0)
        self._dirty = True

    # -- posterior + utility -------------------------------------------------
    def _posterior(self):
        a = 1.0 + self.succ
        b = 1.0 + self.fail
        return a, b

    def _utility_values(self, p):
        p = np.asarray(p, dtype=float)
        if self.utility_kind == "learnability":
            return p * (1.0 - p)
        return np.maximum((1.0 - (1.0 - p) ** self.advmass_n) - p, 0.0)

    def _utility_peak(self) -> float:
        if self.utility_kind == "learnability":
            return 0.5
        return float(1.0 - self.advmass_n ** (-1.0 / (self.advmass_n - 1)))

    def _utility(self):
        a, b = self._posterior()
        if self.thompson:
            p = self.rng.beta(a, b)
        else:
            mean = a / (a + b)
            std = np.sqrt(mean * (1 - mean) / (a + b + 1))
            lower = np.clip(mean - self.optimism_k * std, 1e-4, 1 - 1e-4)
            upper = np.clip(mean + self.optimism_k * std, 1e-4, 1 - 1e-4)
            # Both utilities are concave with one interior maximum. Optimism
            # must maximize utility over the confidence interval; applying
            # mean+k*std directly to p can lower a non-monotone utility.
            p = np.clip(self._utility_peak(), lower, upper)
        u = self._utility_values(p)
        return u ** self.gamma

    def _recompute(self):
        u = self._utility()
        if u.sum() <= 1e-12:
            u = np.ones(self.n_bins)
        probs = u / u.sum()
        probs = (1 - self.floor) * probs + self.floor / self.n_bins
        if self.max_prob is not None and probs.max() > self.max_prob + 1e-12:
            raise RuntimeError(
                f"curriculum probability tripwire fired: max={probs.max():.6f} "
                f"> configured max_prob={self.max_prob:.6f}"
            )
        self._probs = probs
        self._dirty = False

    # -- sampling -------------------------------------------------------------
    def sample_bins(self, m: int) -> np.ndarray:
        if not isinstance(m, (int, np.integer)) or m < 0:
            raise ValueError(f"sample size must be a non-negative integer, got {m!r}")
        if self._dirty:
            self._recompute()
        elif self.max_prob is not None and self._probs.max() > self.max_prob + 1e-12:
            raise RuntimeError(
                f"curriculum probability tripwire fired: max={self._probs.max():.6f} "
                f"> configured max_prob={self.max_prob:.6f}"
            )
        return self.rng.choice(self.n_bins, size=m, p=self._probs)

    def sampling_probs(self) -> np.ndarray:
        if self._dirty:
            self._recompute()
        return self._probs.copy()

    # -- telemetry / persistence ----------------------------------------------
    def pass_rate_estimates(self) -> np.ndarray:
        a, b = self._posterior()
        return a / (a + b)

    def argmax_utility(self) -> int:
        """The frontier bin: where the (mean-posterior) utility peaks."""
        a, b = self._posterior()
        p = a / (a + b)
        u = self._utility_values(p)
        return int(np.argmax(u))

    def dead_fraction(self, threshold: float = 0.05) -> float:
        """Fraction of *seen* bins the posterior currently rates unlearnable."""
        p = self.pass_rate_estimates()
        seen = (self.succ + self.fail) > 1.0
        if not seen.any():
            return 0.0
        return float((p[seen] < threshold).mean())

    def mastered_fraction(self, threshold: float = 0.9) -> float:
        p = self.pass_rate_estimates()
        seen = (self.succ + self.fail) > 1.0
        if not seen.any():
            return 0.0
        return float((p[seen] > threshold).mean())

    def metrics(self) -> dict:
        seen = (self.succ + self.fail) > 1.0
        probs = self.sampling_probs()
        return {"teacher/effective_bins": float(1.0 / (probs ** 2).sum()),
                "teacher/max_prob": float(probs.max()),
                "teacher/seen_frac": float(seen.mean()),
                "teacher/frontier_bin": float(self.argmax_utility()),
                "teacher/frac_dead": self.dead_fraction(),
                "teacher/frac_mastered": self.mastered_fraction()}

    def state_dict(self) -> dict:
        return {
            "version": 3,
            "n_bins": self.n_bins,
            "config": {
                "utility": self.utility_kind,
                "advmass_n": self.advmass_n,
                "decay_half_life": self.half_life,
                "floor": self.floor,
                "gamma": self.gamma,
                "optimism_k": self.optimism_k,
                "thompson": self.thompson,
                "max_prob": self.max_prob,
            },
            "succ": self.succ.copy(),
            "fail": self.fail.copy(),
            "sampling_probs": self._probs.copy(),
            "dirty": self._dirty,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict) -> None:
        if "n_bins" in state and int(state["n_bins"]) != self.n_bins:
            raise ValueError(f"state has {state['n_bins']} bins, teacher expects {self.n_bins}")
        config = state.get("config")
        if config is not None:
            expected = self.state_dict()["config"]
            mismatches = [
                key for key, value in expected.items()
                if key not in config or config[key] != value
            ]
            if mismatches:
                detail = ", ".join(
                    f"{key}: checkpoint={config.get(key)!r}, current={expected[key]!r}"
                    for key in mismatches
                )
                raise ValueError(f"teacher configuration mismatch on resume ({detail})")
        succ = np.asarray(state["succ"], dtype=float)
        fail = np.asarray(state["fail"], dtype=float)
        if succ.shape != (self.n_bins,) or fail.shape != (self.n_bins,):
            raise ValueError(
                f"state counts must have shape ({self.n_bins},), got {succ.shape} and {fail.shape}"
            )
        if not np.all(np.isfinite(succ)) or not np.all(np.isfinite(fail)) or np.any(succ < 0) or np.any(fail < 0):
            raise ValueError("state counts must be finite and non-negative")
        self.succ = succ.copy()
        self.fail = fail.copy()
        if "rng_state" in state:
            self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        probs = state.get("sampling_probs")
        if probs is not None:
            probs = np.asarray(probs, dtype=float)
            if (
                probs.shape != (self.n_bins,)
                or not np.all(np.isfinite(probs))
                or np.any(probs < 0)
                or not np.isclose(probs.sum(), 1.0)
            ):
                raise ValueError("state sampling_probs must be a finite probability vector")
            self._probs = probs.copy()
            self._dirty = bool(state.get("dirty", False))
        else:
            # Backward-compatible load of the original count-only state.
            self._dirty = True


# --------------------------------------------------------------------------
# IsaacLab curriculum term (the §9.2 integration from the infra guide)
# --------------------------------------------------------------------------
class FrontierTerrainTeacherTerm:
    """A stateful curriculum term for ManagerBasedRLEnv, terrain-level axis.

    Drop-in for the stock ``terrain_levels_vel``: observe the ending episodes'
    outcomes, Thompson/optimism-sample next-episode terrain levels from the
    frontier teacher, write them to ``terrain.terrain_levels`` + env origins.
    Runs inside ``CurriculumManager.compute(env_ids)`` — i.e. in
    ``_reset_idx`` BEFORE ``scene.reset``, so written origins take effect for
    the episode about to start (the contract ``terrain_levels_vel`` uses).

    IsaacLab-independent by construction: `env` is duck-typed
    (needs .scene.terrain{.terrain_levels,.max_terrain_level,.env_origins,
    .terrain_origins,.terrain_types} and .termination_manager.time_outs), so
    the class is unit-testable on CPU with a stub env.  To register in a task
    cfg, subclass isaaclab.managers.ManagerTermBase and delegate, or use it
    directly as ``CurrTerm(func=FrontierTerrainTeacherTerm(cfg_dict), ...)``
    if your fork accepts callables with state (most do).

    Success signal: ``time_outs`` (survived to timeout) by default — swap in a
    task predicate via ``success_fn(env, env_ids) -> bool tensor`` for tasks
    where survival saturates (see the infra guide §9.3).
    """

    def __init__(self, n_bins: int = None, *, utility: str = "learnability",
                 advmass_n: int = 16, decay_half_life: float = 2048.0,
                 floor: float = 0.1, gamma: float = 1.0,
                 optimism_k: float = 1.0, thompson: bool = False,
                 max_prob: float | None = None, success_fn=None, seed: int = 0):
        self._pending = dict(
            utility=utility, advmass_n=advmass_n,
            decay_half_life=decay_half_life, floor=floor, gamma=gamma,
            optimism_k=optimism_k, thompson=thompson,
            max_prob=max_prob, seed=seed,
        )
        self.teacher = (
            FrontierBinTeacher(n_bins, **self._pending)
            if n_bins is not None else None
        )
        self._pending_state: dict | None = None
        self.success_fn = success_fn

    def _ensure_teacher(self, env, floor_override: float | None = None):
        if self.teacher is None:
            n_bins = int(env.scene.terrain.max_terrain_level)
            kw = dict(self._pending)
            if floor_override is not None:
                kw["floor"] = floor_override
            self.teacher = FrontierBinTeacher(n_bins, **kw)
            if self._pending_state:
                self.teacher.load_state_dict(self._pending_state)
                self._pending_state = None
        elif floor_override is not None and floor_override != self.teacher.floor:
            if not 0.0 <= floor_override <= 1.0:
                raise ValueError(f"uniform_floor must be in [0, 1], got {floor_override}")
            self.teacher.floor = float(floor_override)
            self.teacher._dirty = True

    def __call__(self, env, env_ids, uniform_floor: float = None):
        import torch
        self._ensure_teacher(env, floor_override=uniform_floor)
        terrain = env.scene.terrain
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(
            env_ids, device=terrain.terrain_levels.device, dtype=torch.long
        )
        # 1) OBSERVE the episodes that just ended
        ep_len = getattr(env, "episode_length_buf", None)
        if ep_len is None:
            raise RuntimeError("episode_length_buf is required to distinguish construction resets")
        completed_ids = ids[ep_len[ids] > 0]
        if len(completed_ids) > 0:
            bins = terrain.terrain_levels[completed_ids]
            if self.success_fn is not None:
                success = self.success_fn(env, completed_ids)
            else:
                success = env.termination_manager.time_outs[completed_ids]
            failed = ~success if hasattr(success, "__invert__") else np.logical_not(success)
            self.teacher.observe_resets(bins, failed)
        # 2) SAMPLE next-episode difficulty for exactly these envs
        new_bins = torch.as_tensor(self.teacher.sample_bins(len(ids)),
                                   device=terrain.terrain_levels.device,
                                   dtype=terrain.terrain_levels.dtype)
        terrain.terrain_levels[ids] = new_bins.clamp(
            0, int(terrain.max_terrain_level) - 1)
        terrain.env_origins[ids] = terrain.terrain_origins[
            terrain.terrain_levels[ids], terrain.terrain_types[ids]]
        # 3) TELEMETRY -> Curriculum/<term>/* in TensorBoard
        m = self.teacher.metrics()
        return {"mean_bin": terrain.terrain_levels.float().mean(),
                "frontier_bin": m["teacher/frontier_bin"],
                "dead_frac": m["teacher/frac_dead"],
                "mastered_frac": m["teacher/frac_mastered"],
                "effective_bins": m["teacher/effective_bins"]}

    # persistence (call from your checkpoint hooks)
    def state_dict(self):
        if self.teacher:
            return self.teacher.state_dict()
        return copy.deepcopy(self._pending_state) if self._pending_state else {}

    def load_state_dict(self, state):
        if not state:
            return
        if self.teacher:
            self.teacher.load_state_dict(state)
        else:
            self._pending_state = copy.deepcopy(state)


def make_curriculum_term(teacher: FrontierBinTeacher, bin_of_env_fn,
                         command_name: str = None):
    """Minimal function-style wrapper (kept for command-manager-side axes,
    where task ASSIGNMENT lives in a custom command term and this term only
    feeds evidence).  See FrontierTerrainTeacherTerm for the full pattern."""
    def frontier_curriculum(env, env_ids, **kwargs):
        import torch
        ep_len = getattr(env, "episode_length_buf", None)
        if ep_len is None:
            raise RuntimeError("episode_length_buf is required to distinguish construction resets")
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(
            env_ids, device=ep_len.device, dtype=torch.long
        )
        completed_ids = ids[ep_len[ids] > 0]
        if len(completed_ids) == 0:
            return torch.tensor(
                teacher.metrics()["teacher/effective_bins"], device=ep_len.device
            )
        cpu_ids = completed_ids.detach().cpu().numpy()
        bins = bin_of_env_fn(env, cpu_ids)
        failed = env.termination_manager.terminated[completed_ids]
        teacher.observe_resets(bins, failed)
        return torch.tensor(
            teacher.metrics()["teacher/effective_bins"], device=ep_len.device
        )
    return frontier_curriculum
