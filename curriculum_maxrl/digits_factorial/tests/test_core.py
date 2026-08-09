from __future__ import annotations

import itertools
import math

import numpy as np
import torch

from curriculum_maxrl.digits_factorial import core
from curriculum_maxrl.estimators import weights_maxrl, weights_rloo


def test_frozen_digits_manifest_is_disjoint_complete_and_hashed() -> None:
    x, y, splits, manifest = core.load_stored_digits()
    assert x.shape == (1797, 64)
    assert y.shape == (1797,)
    assert {name: len(indices) for name, indices in splits.items()} == {
        "train": 1077,
        "dev": 360,
        "test": 360,
    }
    joined = np.concatenate(list(splits.values()))
    assert np.array_equal(np.sort(joined), np.arange(1797))
    assert all(sum(manifest["class_counts"][name].values()) == len(splits[name]) for name in splits)


def test_model_has_exactly_4810_float64_parameters_and_shared_initialization() -> None:
    left = core.initialize_model(33000)
    right = core.initialize_model(33000)
    assert sum(parameter.numel() for parameter in left.parameters()) == 4810
    assert all(parameter.dtype is torch.float64 for parameter in left.parameters())
    assert core.model_state_sha256(left) == core.model_state_sha256(right)


def test_independent_n8_estimators_match_canonical_functions_and_mass_identities() -> None:
    for bits in itertools.product((0.0, 1.0), repeat=8):
        rewards = np.asarray(bits, dtype=np.float64)
        maxrl = core.practical_maxrl_weights(rewards)
        rloo = core.rloo_weights(rewards)
        assert np.array_equal(maxrl, weights_maxrl(rewards))
        assert np.array_equal(rloo, weights_rloo(rewards))
        k = int(rewards.sum())
        assert math.isclose(
            float(np.abs(maxrl).sum()),
            core.expected_binary_mass("practical_maxrl", k),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        assert math.isclose(
            float(np.abs(rloo).sum()),
            core.expected_binary_mass("rloo", k),
            rel_tol=0.0,
            abs_tol=1e-15,
        )


def test_n2_estimator_and_sampler_identities() -> None:
    for bits in itertools.product((0.0, 1.0), repeat=2):
        rewards = np.asarray(bits)
        k = rewards.sum()
        maxrl = np.zeros(2) if k == 0 else rewards / k - 0.5
        rloo = 0.5 * (rewards - (k - rewards))
        assert np.array_equal(maxrl, rloo)
    p = np.linspace(0.0, 1.0, 1001)
    assert np.allclose(1.0 - (1.0 - p) ** 2 - p, p * (1.0 - p), atol=2e-16)


def test_sampler_probabilities_are_exactly_normalized_and_respect_floor() -> None:
    p = np.linspace(0.0, 1.0, 1077)
    for sampler in core.SAMPLERS:
        q = core.sampler_probabilities(sampler, p)
        assert q.shape == p.shape
        assert np.isclose(q.sum(), 1.0, atol=1e-15)
        assert np.all(q >= 0.1 / len(p) - 2e-18)
    assert np.array_equal(
        core.sampler_probabilities("uniform", p),
        np.full(len(p), 1.0 / len(p)),
    )


def test_rng_tapes_are_deterministic_and_domain_separated() -> None:
    task_a, action_a, metadata_a = core.generate_rng_tapes(33000, steps=3)
    task_b, action_b, metadata_b = core.generate_rng_tapes(33000, steps=3)
    assert np.array_equal(task_a, task_b)
    assert np.array_equal(action_a, action_b)
    assert metadata_a == metadata_b
    assert metadata_a["domain_seeds"]["task_selection"] != metadata_a["domain_seeds"]["actions"]


def test_exact_budget_and_evaluation_grid() -> None:
    assert core.ACTION_BUDGET == 262_144
    assert core.TRAIN_SCORING_FORWARD_COUNT == 551_424
    assert core.EVAL_STEPS == tuple(range(0, 513, 16))
    assert core.RECOVERY_STEPS == (0, 128, 256, 384, 512)


def test_single_thread_deterministic_cpu_settings_are_enforced() -> None:
    observed = core.configure_deterministic_cpu()
    assert observed["torch_num_threads"] == 1
    assert observed["torch_num_interop_threads"] == 1
    assert observed["deterministic_algorithms"] is True
    assert set(observed["thread_environment"].values()) == {"1"}
