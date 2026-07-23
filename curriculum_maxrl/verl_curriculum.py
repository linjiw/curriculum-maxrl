"""Legacy heuristic-frontier sampler prototype for the verl-based MaxRL trainer.

The maintained derived-utility integration is ``../verl_integration``. This
module remains as a CPU-tested research baseline and budget-allocation fixture.

Drop-in integration (see DESIGN.md section 7):

1. In ``verl/trainer/main_ppo.py``, when ``config.data.get("curriculum", {}).get("enable")``:

       from curriculum_maxrl.verl_curriculum import FrontierTeacher, CurriculumSampler
       teacher = FrontierTeacher(len(train_dataset),
                                 n_rollouts=config.actor_rollout_ref.rollout.n)
       train_sampler = CurriculumSampler(train_dataset, teacher,
                                         seed=config.data.get("seed", 1))

2. In ``RayPPOTrainer.fit()``, after rewards are computed for a batch (where
   ``batch.non_tensor_batch["uid"]`` groups rollouts by prompt and dataset
   indices travel in ``non_tensor_batch["index"]`` — verl keeps the original
   dataset row index in ``extra_info`` for most preprocessors; otherwise add
   it in the dataset ``__getitem__``):

       scores = batch.batch["token_level_scores"].sum(-1).cpu().numpy()
       teacher.observe_batch(dataset_indices, uids, scores)

   The sampler reads teacher weights lazily on each epoch, so no further
   plumbing is required.

Only numpy + torch Sampler API; no other verl dependency, so it is unit-testable
on CPU (see test_verl_curriculum.py).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

try:  # torch only needed for the Sampler base class inside verl
    from torch.utils.data import Sampler
except ImportError:  # pragma: no cover - CPU test environment without torch
    class Sampler:  # type: ignore
        def __init__(self, data_source=None):
            pass


class FrontierTeacher:
    """MaxRL-native curriculum teacher over a fixed prompt dataset.

    Maintains a decayed Beta posterior over each prompt's pass rate and scores
    prompts by the frontier utility

        u(p) = (1 - (1-p)^N) * (1 - p),

    where N is the rollout group size: ``1-(1-p)^N`` is precisely the
    probability that the MaxRL group is *not* dropped (K >= 1), i.e. the
    estimator's effective signal weight, and ``1-p`` is the remaining
    headroom.  Thompson sampling over the posterior gives optimism on
    unvisited prompts.

    ``floor`` mixes in a uniform distribution to preserve coverage (replay
    against forgetting + keeps posterior fresh on retired prompts).
    """

    def __init__(self, n_prompts: int, n_rollouts: int = 16, decay: float = 0.9,
                 floor: float = 0.1, seed: int = 0):
        self.n_prompts = n_prompts
        self.n_rollouts = n_rollouts
        self.decay = decay
        self.floor = floor
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones(n_prompts, dtype=np.float64)
        self.beta = np.ones(n_prompts, dtype=np.float64)
        self.visits = np.zeros(n_prompts, dtype=np.int64)

    def observe_batch(self, dataset_indices: np.ndarray, uids: np.ndarray,
                      scores: np.ndarray, success_threshold: float = 0.5):
        """Update posteriors from one training batch.

        Args:
          dataset_indices: (bs,) original dataset row per rollout
          uids: (bs,) prompt-group id per rollout (verl's ``uid``)
          scores: (bs,) scalar reward per rollout
        """
        by_uid: dict = defaultdict(list)
        uid_to_idx: dict = {}
        for di, u, s in zip(dataset_indices, uids, scores):
            by_uid[u].append(float(s) > success_threshold)
            uid_to_idx[u] = int(di)
        for u, successes in by_uid.items():
            idx = uid_to_idx[u]
            k = sum(successes)
            n = len(successes)
            self.alpha[idx] = 1.0 + (self.alpha[idx] - 1.0) * self.decay + k
            self.beta[idx] = 1.0 + (self.beta[idx] - 1.0) * self.decay + (n - k)
            self.visits[idx] += 1

    def utility(self, p: np.ndarray) -> np.ndarray:
        return (1.0 - (1.0 - p) ** self.n_rollouts) * (1.0 - p)

    def sampling_weights(self) -> np.ndarray:
        p = self.rng.beta(self.alpha, self.beta)
        u = self.utility(p)
        total = u.sum()
        if total <= 1e-12:
            return np.full(self.n_prompts, 1.0 / self.n_prompts)
        probs = u / total
        uniform = np.full(self.n_prompts, 1.0 / self.n_prompts)
        return (1.0 - self.floor) * probs + self.floor * uniform

    def pass_rate_estimates(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    # -- checkpointing --------------------------------------------------
    def state_dict(self) -> dict:
        return {"alpha": self.alpha.copy(), "beta": self.beta.copy(),
                "visits": self.visits.copy()}

    def load_state_dict(self, state: dict):
        self.alpha = np.asarray(state["alpha"], dtype=np.float64)
        self.beta = np.asarray(state["beta"], dtype=np.float64)
        self.visits = np.asarray(state["visits"], dtype=np.int64)


class CurriculumSampler(Sampler):
    """Infinite-horizon weighted sampler driven by a FrontierTeacher.

    Re-draws teacher weights at the start of every epoch (verl iterates the
    dataloader once per epoch), so curriculum updates take effect without
    rebuilding the dataloader.  Sampling is with replacement — standard for
    curriculum methods; epoch length matches the dataset size so trainer
    bookkeeping (total_training_steps) is unchanged.
    """

    def __init__(self, data_source, teacher: FrontierTeacher, seed: int = 1):
        super().__init__(data_source)
        self.n = len(data_source)
        self.teacher = teacher
        self.rng = np.random.default_rng(seed)
        self.epoch = 0

    def __len__(self):
        return self.n

    def __iter__(self):
        self.epoch += 1
        w = self.teacher.sampling_weights()
        idx = self.rng.choice(self.n, size=self.n, replace=True, p=w)
        return iter(idx.tolist())


def allocate_rollout_budget(p_hat: np.ndarray, total_budget: int,
                            n_min: int = 4, n_max: int = 64) -> np.ndarray:
    """Per-prompt rollout counts N_i ∝ 1/p̂_i within [n_min, n_max], summing to
    the batch budget. Under practical dropped-group Algorithm 1, the exact
    objective order is N_i-1. This legacy inverse-probability rule predates
    the optimal coefficient-mass water-filling allocator in ``teachers.py``.
    (Phase-2 feature: requires per-sample ``n`` support in the rollout worker.)
    """
    p_hat = np.asarray(p_hat, dtype=float)
    if p_hat.ndim != 1:
        raise ValueError(f"p_hat must be one-dimensional, got shape {p_hat.shape}")
    if not np.all(np.isfinite(p_hat)):
        raise ValueError("p_hat must contain only finite values")
    if np.any((p_hat < 0.0) | (p_hat > 1.0)):
        raise ValueError("p_hat values must lie in [0, 1]")
    if (isinstance(total_budget, (bool, np.bool_))
            or not isinstance(total_budget, (int, np.integer))
            or total_budget < 0):
        raise ValueError(f"total_budget must be a non-negative integer, got {total_budget!r}")
    if (isinstance(n_min, (bool, np.bool_))
            or isinstance(n_max, (bool, np.bool_))
            or not isinstance(n_min, (int, np.integer))
            or not isinstance(n_max, (int, np.integer))):
        raise ValueError("n_min and n_max must be integers")
    if n_min < 1 or n_max < n_min:
        raise ValueError(f"require 1 <= n_min <= n_max, got {n_min} and {n_max}")

    n_prompts = len(p_hat)
    if n_prompts == 0:
        if total_budget == 0:
            return np.empty(0, dtype=int)
        raise ValueError("a non-zero budget cannot be allocated to an empty prompt set")
    min_budget, max_budget = n_min * n_prompts, n_max * n_prompts
    if not min_budget <= total_budget <= max_budget:
        raise ValueError(
            f"total_budget={total_budget} is infeasible for {n_prompts} prompts with "
            f"bounds [{n_min}, {n_max}]; expected [{min_budget}, {max_budget}]"
        )

    raw = 1.0 / np.maximum(p_hat, 1.0 / n_max)
    scaled = raw / raw.sum() * total_budget
    n = np.clip(np.round(scaled), n_min, n_max).astype(int)
    # settle rounding drift on the hardest prompts still inside [n_min, n_max]
    order = np.argsort(-raw, kind="stable")  # hardest first
    while n.sum() > total_budget:
        movable = [i for i in order[::-1] if n[i] > n_min]  # easiest first
        n[movable[0]] -= 1
    while n.sum() < total_budget:
        movable = [i for i in order if n[i] < n_max]  # hardest first
        n[movable[0]] += 1
    return n
