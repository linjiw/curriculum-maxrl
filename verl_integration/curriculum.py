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

The teacher tracks a decayed Beta posterior over each prompt's pass rate and
samples prompts by the *expected coefficient L1 mass* of the MaxRL estimator: for
a group of N rollouts on a prompt with pass rate p, the expected coefficient
mass emitted (practical Algorithm 1 of arXiv:2602.02710,
w_succ = 1/K - 1/N,
w_fail = -1/N, K=0 groups dropped) is exactly

    E[sum_j |w_j|] = 2 * (pass@N(p) - pass@1(p)) = 2 * ((1-(1-p)^N) - p),

i.e. twice the probability the prompt is solvable within N attempts but not
within one. Sampling proportional to the normalized half-mass utility is a
soft exploration/coverage policy, not the one-step mass optimizer. It peaks
at p* ~ ln(N)/N, so larger group sizes target harder prompts. At N=2 it equals
the "learnability" objective p(1-p) of Rutherford et al. (2024); at N=1
the exact advantage-mass utility is zero.  The
older heuristic frontier utility (1-(1-p)^N)(1-p) is available via
utility="frontier" and is numerically near-identical.

Scope: this identity is for practical Algorithm 1, which drops the K=0
control term and has exact expected objective order N-1. Paper Eq. (9) and
full Eq. (10) have order N but a different coefficient-mass contract.

Enable via config:

    data.curriculum.enable=true
    data.curriculum.floor=0.1          # uniform replay floor
    data.curriculum.decay=0.9          # posterior decay per observation
    data.curriculum.success_threshold=0.5
    data.curriculum.utility=advmass    # or "frontier"

The sampler re-draws Thompson weights each epoch; the trainer feeds
observations back after every reward computation (see ray_trainer.fit).
"""

import copy
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class FrontierTeacher:
    def __init__(self, n_prompts, n_rollouts=16, decay=0.7, floor=0.1, seed=0,
                 success_threshold=0.5, utility="advmass", power=1.0):
        # decay=0.7 validated in VALIDATION.md V2b: the oracle-vs-Thompson gap
        # is a tracking problem; faster forgetting closes ~19% of it.
        # power: sample ∝ u^power (V6) — sharper concentration compounds on
        # chain-structured pools (γ≈4); keep 1.0 for flat prompt sets (GSM8K).
        if not isinstance(n_prompts, (int, np.integer)) or n_prompts <= 0:
            raise ValueError(f"n_prompts must be a positive integer, got {n_prompts!r}")
        if not isinstance(n_rollouts, (int, np.integer)) or n_rollouts < 2:
            raise ValueError(f"n_rollouts must be an integer >= 2, got {n_rollouts!r}")
        if not np.isfinite(decay) or not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be finite and in [0, 1], got {decay!r}")
        if not np.isfinite(floor) or not 0.0 <= floor <= 1.0:
            raise ValueError(f"floor must be finite and in [0, 1], got {floor!r}")
        if not np.isfinite(success_threshold):
            raise ValueError(f"success_threshold must be finite, got {success_threshold!r}")
        if utility not in ("advmass", "frontier"):
            raise ValueError(f"utility must be 'advmass' or 'frontier', got {utility!r}")
        if not np.isfinite(power) or power <= 0:
            raise ValueError(f"power must be positive and finite, got {power!r}")
        self.n_prompts = int(n_prompts)
        self.n_rollouts = int(n_rollouts)
        self.decay = float(decay)
        self.floor = float(floor)
        self.success_threshold = float(success_threshold)
        self.utility_kind = utility
        self.power = float(power)
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones(n_prompts, dtype=np.float64)
        self.beta = np.ones(n_prompts, dtype=np.float64)
        self.visits = np.zeros(n_prompts, dtype=np.int64)

    def observe_batch(self, dataset_indices, uids, scores):
        """Update posteriors from one training batch.

        Args:
          dataset_indices: (bs,) original dataset row per rollout
          uids: (bs,) prompt-group id per rollout
          scores: (bs,) scalar reward per rollout
        """
        dataset_indices = np.asarray(dataset_indices)
        uids = np.asarray(uids)
        scores = np.asarray(scores, dtype=float)
        if dataset_indices.ndim != 1 or uids.ndim != 1 or scores.ndim != 1:
            raise ValueError(
                "dataset_indices, uids, and scores must all be one-dimensional"
            )
        if not (len(dataset_indices) == len(uids) == len(scores)):
            raise ValueError(
                "dataset_indices, uids, and scores must have equal lengths"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("scores must contain only finite values")

        by_uid = defaultdict(list)
        uid_to_idx = {}
        for di, u, s in zip(dataset_indices, uids, scores):
            try:
                idx = int(di)
                if float(di) != idx:
                    raise ValueError(f"dataset index must be integral, got {di!r}")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid dataset index {di!r}") from exc
            if not (0 <= idx < self.n_prompts):
                raise ValueError(
                    f"dataset index {idx} is outside [0, {self.n_prompts})"
                )
            if u in uid_to_idx and uid_to_idx[u] != idx:
                raise ValueError(
                    f"uid {u!r} maps to multiple dataset indices: "
                    f"{uid_to_idx[u]} and {idx}"
                )
            by_uid[u].append(float(s) > self.success_threshold)
            uid_to_idx[u] = idx
        for u, successes in by_uid.items():
            idx = uid_to_idx[u]
            k = sum(successes)
            n = len(successes)
            self.alpha[idx] = 1.0 + (self.alpha[idx] - 1.0) * self.decay + k
            self.beta[idx] = 1.0 + (self.beta[idx] - 1.0) * self.decay + (n - k)
            self.visits[idx] += 1

    def sampling_weights(self):
        p = self.rng.beta(self.alpha, self.beta)
        pass_at_n = 1.0 - (1.0 - p) ** self.n_rollouts
        if self.utility_kind == "advmass":
            u = np.maximum(pass_at_n - p, 0.0)
        else:  # frontier
            u = pass_at_n * (1.0 - p)
        if self.power != 1.0:
            u = u ** self.power
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
        return {
            "version": 2,
            "n_prompts": self.n_prompts,
            "config": {
                "n_rollouts": self.n_rollouts,
                "decay": self.decay,
                "floor": self.floor,
                "success_threshold": self.success_threshold,
                "utility": self.utility_kind,
                "power": self.power,
            },
            "alpha": self.alpha.copy(),
            "beta": self.beta.copy(),
            "visits": self.visits.copy(),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        if "n_prompts" in state and int(state["n_prompts"]) != self.n_prompts:
            raise ValueError(
                f"state has {state['n_prompts']} prompts, teacher expects {self.n_prompts}"
            )
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
                raise ValueError(
                    f"teacher configuration mismatch on resume ({detail})"
                )

        alpha = np.asarray(state["alpha"], dtype=np.float64)
        beta = np.asarray(state["beta"], dtype=np.float64)
        visits = np.asarray(state["visits"], dtype=np.int64)
        expected_shape = (self.n_prompts,)
        if (
            alpha.shape != expected_shape
            or beta.shape != expected_shape
            or visits.shape != expected_shape
        ):
            raise ValueError(
                f"teacher arrays must have shape {expected_shape}, got "
                f"{alpha.shape}, {beta.shape}, and {visits.shape}"
            )
        if (
            not np.all(np.isfinite(alpha))
            or not np.all(np.isfinite(beta))
            or np.any(alpha <= 0)
            or np.any(beta <= 0)
            or np.any(visits < 0)
        ):
            raise ValueError("teacher state contains invalid posterior parameters")
        self.alpha = alpha.copy()
        self.beta = beta.copy()
        self.visits = visits.copy()
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
        if self.n <= 0:
            raise ValueError("curriculum sampling requires a non-empty dataset")
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

    def state_dict(self):
        return {
            "version": 1,
            "n": self.n,
            "epoch": self.epoch,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        if int(state.get("n", self.n)) != self.n:
            raise ValueError(
                f"sampler state has dataset size {state.get('n')}, expected {self.n}"
            )
        epoch = state.get("epoch", 0)
        if not isinstance(epoch, (int, np.integer)) or epoch < 0:
            raise ValueError(f"sampler epoch must be a non-negative integer, got {epoch!r}")
        self.epoch = int(epoch)
        if "rng_state" in state:
            self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
