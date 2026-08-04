"""Exact endpoint and expectation checks for finite-group estimators."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from curriculum_maxrl.estimators import (
    _c_TN,
    weights_maxrl_full_cv,
    weights_maxrl_t,
)


def test_subset_estimator_endpoints() -> None:
    for rewards in (
        np.zeros(4),
        np.ones(4),
        np.array([1.0, 0.0, 1.0, 0.0]),
    ):
        assert np.allclose(
            weights_maxrl_t(rewards, 4),
            weights_maxrl_full_cv(rewards),
        )
        assert np.allclose(
            weights_maxrl_t(rewards, 1),
            rewards / 4.0 - 0.25,
        )


@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("truncation", [1, 2])
@pytest.mark.parametrize("p", [0.05, 0.3, 0.8])
def test_subset_estimator_exact_expected_gradient(
    n: int, truncation: int, p: float
) -> None:
    """Exhaustively recover the T-truncated Bernoulli-logit gradient."""
    if truncation > n:
        pytest.skip("truncation exceeds group size")
    q = 1.0 - p
    expected = 0.0
    for successes in range(n + 1):
        probability = comb(n, successes) * p**successes * q ** (n - successes)
        coefficient = _c_TN(successes, n, truncation)
        # Conditional score sums for a Bernoulli logit: q per success and
        # -p per failure. The -1/N control variate is included at every K.
        update = (
            coefficient * successes * q
            - (successes * q - (n - successes) * p) / n
        )
        expected += probability * update
    target = q * (1.0 - q**truncation)
    assert np.isclose(expected, target, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("truncation", [0, 5])
def test_subset_estimator_rejects_invalid_order(truncation: int) -> None:
    with pytest.raises(ValueError):
        weights_maxrl_t(np.zeros(4), truncation)
