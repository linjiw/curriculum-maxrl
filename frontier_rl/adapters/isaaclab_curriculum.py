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
test_framework.py::test_isaaclab_adapter) and imports cleanly in a sim-less
venv.  Dense-reward PPO note (per SONIC_RESPONSE.md): with hazard-style
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

import numpy as np


class FrontierBinTeacher:
    """Vectorized frontier teacher for massively parallel reset streams.

    Differences from frontier_rl.teacher.FrontierTeacher, each forced by the
    parallel-sim setting and consistent with SONIC_RESPONSE.md:

    - evidence-scaled decay: a half-life in *episode-equivalents* rather than
      per-observe decay, so behavior is invariant to env count (Q4);
    - deterministic optimism (mean + k*std) by default instead of Thompson —
      reproducibility guardrails in sim RL favor determinism (Q3); pass
      thompson=True to restore sampling;
    - utility="learnability" default (no natural N in a reset stream, Q2);
    - a max_prob tripwire instead of a shaping cap (Q9).
    """

    def __init__(self, n_bins: int, *, utility: str = "learnability",
                 advmass_n: int = 16, decay_half_life: float = 2048.0,
                 floor: float = 0.1, gamma: float = 1.0,
                 optimism_k: float = 1.0, thompson: bool = False,
                 max_prob: float | None = None, seed: int = 0):
        assert utility in ("learnability", "advmass")
        self.n_bins = n_bins
        self.utility_kind = utility
        self.advmass_n = advmass_n
        self.half_life = decay_half_life
        self.floor = floor
        self.gamma = gamma
        self.optimism_k = optimism_k
        self.thompson = thompson
        self.max_prob = max_prob
        self.rng = np.random.default_rng(seed)
        self.succ = np.zeros(n_bins)
        self.fail = np.zeros(n_bins)
        self._probs = np.full(n_bins, 1.0 / n_bins)
        self._dirty = True

    # -- evidence (vectorized) ---------------------------------------------
    def observe_resets(self, bin_ids, failed) -> None:
        """Stream one batch of reset events.

        Args:
          bin_ids: int array (M,) — the bin each resetting env was on.
          failed:  bool array (M,) — early-termination flag per env.
        """
        bin_ids = np.asarray(bin_ids, dtype=int)
        failed = np.asarray(failed, dtype=bool)
        n_events = len(bin_ids)
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

    def _utility(self):
        a, b = self._posterior()
        if self.thompson:
            p = self.rng.beta(a, b)
        else:
            mean = a / (a + b)
            std = np.sqrt(mean * (1 - mean) / (a + b + 1))
            p = np.clip(mean + self.optimism_k * std, 1e-4, 1 - 1e-4)
        if self.utility_kind == "learnability":
            u = p * (1.0 - p)
        else:
            u = np.maximum((1.0 - (1.0 - p) ** self.advmass_n) - p, 0.0)
        return u ** self.gamma

    def _recompute(self):
        u = self._utility()
        if u.sum() <= 1e-12:
            u = np.ones(self.n_bins)
        probs = u / u.sum()
        probs = (1 - self.floor) * probs + self.floor / self.n_bins
        if self.max_prob is not None:   # tripwire, not shaping
            probs = np.minimum(probs, self.max_prob)
            probs = probs / probs.sum()
        self._probs = probs
        self._dirty = False

    # -- sampling -------------------------------------------------------------
    def sample_bins(self, m: int) -> np.ndarray:
        if self._dirty:
            self._recompute()
        return self.rng.choice(self.n_bins, size=m, p=self._probs)

    def sampling_probs(self) -> np.ndarray:
        if self._dirty:
            self._recompute()
        return self._probs.copy()

    # -- telemetry / persistence ----------------------------------------------
    def metrics(self) -> dict:
        a, b = self._posterior()
        p = a / (a + b)
        seen = (self.succ + self.fail) > 1.0
        out = {"teacher/effective_bins": float(1.0 / (self.sampling_probs() ** 2).sum()),
               "teacher/seen_frac": float(seen.mean())}
        if seen.any():
            out["teacher/frac_dead"] = float((p[seen] < 0.05).mean())
            out["teacher/frac_mastered"] = float((p[seen] > 0.9).mean())
        return out

    def state_dict(self) -> dict:
        return {"succ": self.succ.copy(), "fail": self.fail.copy()}

    def load_state_dict(self, state: dict) -> None:
        self.succ = np.asarray(state["succ"], dtype=float)
        self.fail = np.asarray(state["fail"], dtype=float)
        self._dirty = True


# --------------------------------------------------------------------------
# IsaacLab curriculum-term wrappers (import isaaclab lazily; these are thin)
# --------------------------------------------------------------------------
def make_curriculum_term(teacher: FrontierBinTeacher, bin_of_env_fn,
                         command_name: str = None):
    """Build a function usable as isaaclab.managers.CurriculumTermCfg(func=...).

    bin_of_env_fn(env, env_ids) -> int array: which bin each env was running.
    The returned term observes terminations for the resetting envs and returns
    the teacher's effective-bin count (a scalar the CurriculumManager logs).
    Task *assignment* stays in your command term — call
    `teacher.sample_bins(...)` there when resampling.
    """
    def frontier_curriculum(env, env_ids, **kwargs):
        import torch
        ids = env_ids if not hasattr(env_ids, "cpu") else env_ids.cpu().numpy()
        bins = bin_of_env_fn(env, ids)
        failed = env.termination_manager.terminated[env_ids]
        failed = failed.cpu().numpy() if hasattr(failed, "cpu") else np.asarray(failed)
        teacher.observe_resets(bins, failed)
        return torch.tensor(teacher.metrics()["teacher/effective_bins"])
    return frontier_curriculum
