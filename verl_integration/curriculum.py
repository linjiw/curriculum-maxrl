# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Curriculum teacher + weighted sampler for MaxRL-style RL training.

The teacher tracks discounted Beta pseudo-counts for each prompt's pass rate
and prioritizes prompts by the *expected scalar coefficient mass* of the
practical dropped-group MaxRL estimator: for
a group of N rollouts on a prompt with pass rate p, the expected total
|advantage| emitted (Algorithm 1 of arXiv:2602.02710, w_succ = 1/K - 1/N,
w_fail = -1/N, K=0 groups dropped) is exactly

    E[sum_j |w_j|] = 2 * (pass@N(p) - pass@1(p)) = 2 * ((1-(1-p)^N) - p),

i.e. twice the probability the prompt is solvable within N attempts but not
within one. This identity is exact for binary rewards, fixed N, and the
normalized theoretical coefficients above. The official MaxRL code uses an
N-scaled epsilon-normalized form; at fixed N it has the same relative priority
away from the epsilon boundary. Thresholding non-binary rewards makes this a
proxy rather than a literal emitted-magnitude identity.

This is a smooth stochastic priority score, not the gradient norm
or the unconstrained maximizer of one-step utility (which would hard-select an
argmax). It peaks at p* ~ ln(N)/N. At N=2 it matches the "learnability"
objective p(1-p) of Rutherford et al. (2024), up to a constant. The
older heuristic frontier utility (1-(1-p)^N)(1-p) is available via
utility="frontier". It equals the exact family at index N+1 because
`(1-(1-p)^N)(1-p) = 1-(1-p)^(N+1)-p = u_(N+1)(p)`. It is close for the
configured moderate/large group sizes, but is not identical at fixed N.

Enable via config:

    data.curriculum.enable=true
    data.curriculum.floor=0.1          # uniform replay floor
    data.curriculum.decay=0.7          # pseudo-count decay per observation
    data.curriculum.success_threshold=0.5
    data.curriculum.utility=advmass    # or "frontier"
    data.curriculum.power=1.0          # concentration; keep 1 on flat pools

The sampler re-draws Thompson weights each epoch; the trainer feeds
observations back after every reward computation (see ray_trainer.fit).
Sampling intentionally changes the training-task distribution; this patch
does not importance-correct back to the original dataset distribution.
"""

import copy
import warnings
from collections import defaultdict

import numpy as np
try:
    from torch.utils.data import Dataset, Sampler
except ImportError:  # pragma: no cover - algebra/unit use without torch
    class Dataset:  # type: ignore
        pass

    class Sampler:  # type: ignore
        pass


class CurriculumIndexedDataset(Dataset):
    """Attach the post-filter sampler position to every training row.

    Source ``extra_info.index`` values are not reliable sampler positions:
    they can be non-contiguous after filtering or collide across concatenated
    files. This wrapper supplies a separate, verified position while
    delegating dataset-specific methods such as ``resume_dataset_state``.
    """

    def __init__(self, dataset):
        if len(dataset) < 1:
            raise ValueError("curriculum training dataset must be non-empty")
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, position):
        position = int(position)
        row = self.dataset[position]
        if not isinstance(row, dict):
            raise TypeError("curriculum dataset rows must be dictionaries")
        row = dict(row)
        row["curriculum_index"] = position
        return row

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


class FrontierTeacher:
    STATE_VERSION = 2

    def __init__(self, n_prompts, n_rollouts=16, decay=0.7, floor=0.1, seed=0,
                 success_threshold=0.5, utility="advmass", power=1.0):
        # decay=0.7 validated in VALIDATION.md V2b: the true-p-priority gap
        # is a tracking problem; faster forgetting closes ~19% of it.
        # power: sample ∝ u^power (V6) — sharper concentration compounds on
        # chain-structured pools (γ≈4); keep 1.0 for flat prompt sets (GSM8K).
        if utility not in ("advmass", "frontier"):
            raise ValueError("utility must be 'advmass' or 'frontier'")
        if int(n_prompts) != n_prompts or n_prompts < 1:
            raise ValueError("n_prompts must be a positive integer")
        if int(n_rollouts) != n_rollouts or n_rollouts < 2:
            raise ValueError("n_rollouts must be an integer >= 2")
        if not (0.0 <= float(decay) <= 1.0):
            raise ValueError("decay must lie in [0, 1]")
        if not (0.0 <= float(floor) <= 1.0):
            raise ValueError("floor must lie in [0, 1]")
        if not np.isfinite(power) or power <= 0.0:
            raise ValueError("power must be positive and finite")
        if not np.isfinite(success_threshold):
            raise ValueError("success_threshold must be finite")
        self.n_prompts = n_prompts
        self.n_rollouts = n_rollouts
        self.decay = decay
        self.floor = floor
        self.success_threshold = success_threshold
        self.utility_kind = utility
        self.power = power
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones(n_prompts, dtype=np.float64)
        self.beta = np.ones(n_prompts, dtype=np.float64)
        self.visits = np.zeros(n_prompts, dtype=np.int64)

    def observe_batch(self, dataset_indices, uids, scores):
        """Update discounted pseudo-counts from one training batch.

        Args:
          dataset_indices: (bs,) post-filter sampler position per rollout,
            supplied by ``CurriculumIndexedDataset``
          uids: (bs,) prompt-group id per rollout
          scores: (bs,) scalar reward per rollout
        """
        dataset_indices = list(dataset_indices)
        uids = list(uids)
        scores = list(scores)
        if not (len(dataset_indices) == len(uids) == len(scores)):
            raise ValueError("curriculum feedback arrays must have equal length")
        if not dataset_indices:
            raise ValueError("curriculum feedback batch must be non-empty")

        by_uid = defaultdict(list)
        uid_to_idx = {}
        for di, u, s in zip(dataset_indices, uids, scores):
            try:
                idx = int(di)
            except (TypeError, ValueError):
                raise ValueError(f"invalid curriculum_index {di!r}") from None
            if isinstance(di, (float, np.floating)) and not float(di).is_integer():
                raise ValueError(f"non-integral curriculum_index {di!r}")
            if not (0 <= idx < self.n_prompts):
                raise IndexError(
                    f"curriculum_index {idx} outside [0, {self.n_prompts})"
                )
            score = float(s)
            if not np.isfinite(score):
                raise ValueError(f"non-finite curriculum reward {s!r}")
            try:
                previous = uid_to_idx.get(u)
            except TypeError:
                raise TypeError("curriculum uid values must be hashable") from None
            if previous is not None and previous != idx:
                raise ValueError(
                    f"rollout uid {u!r} maps to positions {previous} and {idx}"
                )
            by_uid[u].append(score > self.success_threshold)
            uid_to_idx[u] = idx
        for u, successes in by_uid.items():
            idx = uid_to_idx[u]
            k = sum(successes)
            n = len(successes)
            self.alpha[idx] = 1.0 + (self.alpha[idx] - 1.0) * self.decay + k
            self.beta[idx] = 1.0 + (self.beta[idx] - 1.0) * self.decay + (n - k)
            self.visits[idx] += 1

    def utility(self, p):
        """Return the configured nonnegative priority for supplied pass rates."""
        p = np.asarray(p, dtype=np.float64)
        pass_at_n = 1.0 - (1.0 - p) ** self.n_rollouts
        if self.utility_kind == "advmass":
            u = np.maximum(pass_at_n - p, 0.0)
        else:  # frontier
            u = pass_at_n * (1.0 - p)
        if self.power != 1.0:
            u = u ** self.power
        return u

    def sampling_weights(self):
        p = self.rng.beta(self.alpha, self.beta)
        u = self.utility(p)
        total = u.sum()
        if total <= 1e-12:
            return np.full(self.n_prompts, 1.0 / self.n_prompts)
        probs = u / total
        uniform = np.full(self.n_prompts, 1.0 / self.n_prompts)
        return (1.0 - self.floor) * probs + self.floor * uniform

    def pass_rate_estimates(self):
        return self.alpha / (self.alpha + self.beta)

    def metrics(self):
        """Scalars for logging."""
        p = self.pass_rate_estimates()
        visited = self.visits > 0
        out = {
            "curriculum/visited_frac": float(visited.mean()),
            "curriculum/mean_p_hat_visited": float(p[visited].mean()) if visited.any() else 0.0,
        }
        if visited.any():
            pv = p[visited]
            out["curriculum/frac_dead_p_lt_0.05"] = float((pv < 0.05).mean())
            out["curriculum/frac_mastered_p_gt_0.9"] = float((pv > 0.9).mean())
        return out

    def state_dict(self):
        return {"version": self.STATE_VERSION,
                "config": {"n_prompts": self.n_prompts,
                           "n_rollouts": self.n_rollouts,
                           "decay": self.decay, "floor": self.floor,
                           "success_threshold": self.success_threshold,
                           "utility": self.utility_kind,
                           "power": self.power},
                "alpha": self.alpha.copy(), "beta": self.beta.copy(),
                "visits": self.visits.copy(),
                "rng_state": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        if state.get("version") not in (None, self.STATE_VERSION):
            raise ValueError(f"unsupported curriculum state version {state.get('version')}")
        expected = self.state_dict()["config"]
        saved = state.get("config")
        if saved is None:
            warnings.warn(
                "loading legacy curriculum state without configuration metadata",
                RuntimeWarning,
            )
        elif saved != expected:
            raise ValueError(
                f"curriculum checkpoint config mismatch: saved={saved}, current={expected}"
            )
        alpha = np.asarray(state["alpha"], dtype=np.float64)
        beta = np.asarray(state["beta"], dtype=np.float64)
        visits = np.asarray(state["visits"], dtype=np.int64)
        expected_shape = (self.n_prompts,)
        if (alpha.shape != expected_shape or beta.shape != expected_shape
                or visits.shape != expected_shape):
            raise ValueError(
                f"curriculum checkpoint arrays must have shape {expected_shape}"
            )
        if (not np.isfinite(alpha).all() or not np.isfinite(beta).all()
                or (alpha <= 0).any() or (beta <= 0).any()
                or (visits < 0).any()):
            raise ValueError("curriculum checkpoint contains invalid pseudo-counts")
        self.alpha = alpha
        self.beta = beta
        self.visits = visits
        if "rng_state" in state:
            self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])


class CurriculumSampler(Sampler):
    """Weighted sampler (with replacement) driven by a FrontierTeacher.

    Weights are re-drawn from the teacher at the start of every epoch, so the
    curriculum adapts without rebuilding the dataloader.  Epoch length equals
    the dataset size, keeping total_training_steps bookkeeping unchanged.
    """

    def __init__(self, data_source, teacher, seed=1):
        self.n = len(data_source)
        if self.n != teacher.n_prompts:
            raise ValueError(
                "teacher prompt count must equal the wrapped dataset length"
            )
        self.teacher = teacher
        # One checkpointed generator drives both Thompson draws and sampled
        # indices.  A separate sampler generator would make resumed prompt
        # order diverge even when the teacher state is restored.  ``seed`` is
        # retained for API compatibility; the teacher seed is authoritative.
        self.rng = teacher.rng
        self.epoch = 0
        self._indices = None
        self._cursor = 0

    def __len__(self):
        return self.n

    def __iter__(self):
        if self._indices is None or self._cursor >= self.n:
            self.epoch += 1
            w = self.teacher.sampling_weights()
            self._indices = self.rng.choice(
                self.n, size=self.n, replace=True, p=w
            ).astype(np.int64)
            self._cursor = 0
        while self._cursor < self.n:
            idx = int(self._indices[self._cursor])
            self._cursor += 1
            yield idx

    def state_dict(self):
        return {
            "version": 1,
            "n": self.n,
            "epoch": self.epoch,
            "cursor": self._cursor,
            "indices": (None if self._indices is None else
                        self._indices.copy()),
        }

    def load_state_dict(self, state):
        if state.get("version") != 1 or int(state.get("n", -1)) != self.n:
            raise ValueError("incompatible curriculum sampler checkpoint")
        indices = state.get("indices")
        if indices is not None:
            indices = np.asarray(indices, dtype=np.int64)
            if (indices.shape != (self.n,) or (indices < 0).any()
                    or (indices >= self.n).any()):
                raise ValueError("invalid curriculum sampler index vector")
        cursor = int(state.get("cursor", -1))
        if not (0 <= cursor <= self.n):
            raise ValueError("invalid curriculum sampler cursor")
        self.epoch = int(state.get("epoch", 0))
        self._cursor = cursor
        self._indices = indices
