from __future__ import annotations

import itertools

import numpy as np

from curriculum_maxrl.digits_factorial.analyze import (
    exact_sign_flip_pvalue,
    formula_audit,
    holm_rejections,
    select_learning_rate_on_literal_exact_tie,
)


def brute_force_sign_flip(values: np.ndarray) -> float:
    observed = abs(values.sum())
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        if abs(float(values @ np.asarray(signs))) >= observed:
            extreme += 1
    return extreme / (2 ** len(values))


def test_exact_sign_flip_matches_brute_force_small_vectors() -> None:
    for values in (
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([-0.3, 0.2, 0.9, 1.1]),
        np.asarray([0.0, 0.0, 1.0, -1.0, 2.0]),
    ):
        assert exact_sign_flip_pvalue(values) == brute_force_sign_flip(values)


def test_zero_sum_sign_flip_pvalues_are_exactly_one_and_bounded() -> None:
    for values in (
        np.asarray([0.0]),
        np.asarray([1.0, -1.0]),
        np.asarray([0.0, 1.0, -1.0, 2.0, -2.0]),
    ):
        assert exact_sign_flip_pvalue(values) == 1.0
        assert 0.0 <= exact_sign_flip_pvalue(values) <= 1.0


def test_holm_step_down_stops_after_first_failure() -> None:
    got = holm_rejections({"a": 0.001, "b": 0.02, "c": 0.021, "d": 0.9})
    assert got == {"a": True, "b": False, "c": False, "d": False}


def test_formula_audit_is_exhaustive() -> None:
    report = formula_audit()
    assert report["passed"] is True
    assert report["n8_binary_vectors"] == 256
    assert report["estimator_vector_checks"] == 512


def test_lr_selection_does_not_treat_near_tie_as_exact_tie() -> None:
    scores = {"0.03": 0.5, "0.1": 0.5 + 5e-16, "0.3": 0.4, "1": 0.3, "3": 0.2}
    assert select_learning_rate_on_literal_exact_tie(scores) == 0.1
    scores["0.03"] = scores["0.1"]
    assert select_learning_rate_on_literal_exact_tie(scores) == 0.03
