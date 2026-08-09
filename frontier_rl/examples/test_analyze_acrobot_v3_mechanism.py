"""Regression tests for the Acrobot V3 post-hoc mechanism audit."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from analyze_acrobot_v3_mechanism import (
    DEFAULT_ARTIFACT,
    DEFAULT_LOCK,
    analyze,
    exact_two_sided_sign_flip_p,
    normalized_trapezoid,
    practical_maxrl_mass,
)


def _inputs():
    artifact = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    lock = json.loads(DEFAULT_LOCK.read_text(encoding="utf-8"))
    return artifact, lock


def test_closed_form_realized_mass() -> None:
    assert practical_maxrl_mass(0, 16) == 0.0
    assert practical_maxrl_mass(16, 16) == 0.0
    assert practical_maxrl_mass(1, 16) == 1.875
    assert practical_maxrl_mass(8, 16) == 1.0


def test_auc_and_exact_sign_flip_helpers() -> None:
    assert normalized_trapezoid([0.0, 1.0], [0, 2]) == 0.5
    values = np.ones(20)
    assert exact_two_sided_sign_flip_p(values) == 2.0 / (2**20)


def test_frozen_artifact_anchors() -> None:
    artifact, lock = _inputs()
    result = analyze(artifact, lock, DEFAULT_ARTIFACT, DEFAULT_LOCK)
    primary = result["registered_primary_anchor"]
    assert abs(primary["mean_paired_difference"] - 0.03635237545778361) < 1e-12
    mass = result["posthoc"]["coefficient_mass_per_group"]
    assert mass["positive_pairs"] == 20
    assert abs(mass["mean_paired_difference"] - 0.13313967902899865) < 1e-12
    native = result["posthoc"]["native_success_auc"]
    assert native["positive_pairs"] == 15
    assert abs(native["mean_paired_difference"] - 0.06406778851699645) < 1e-12
    assert result["status"] == "historical_descriptive_after_rng_domain_audit"
    sensitivity = result["rng_domain_audit"][
        "alternating_seed_sensitivity_descriptive"
    ]
    assert sensitivity["even_logical_seeds"]["n"] == 10
    assert sensitivity["odd_logical_seeds"]["n"] == 10
    assert sensitivity["even_logical_seeds"]["positive_pairs"] == 6
    assert sensitivity["odd_logical_seeds"]["positive_pairs"] == 8


def test_tampered_group_ledger_is_rejected() -> None:
    artifact, lock = _inputs()
    damaged = copy.deepcopy(artifact)
    damaged["cases"]["uniform_shared_h64"]["runs"][0]["group_diagnostics"][0][
        "success_count"
    ] = 16
    try:
        analyze(damaged, lock, Path("damaged.json"), DEFAULT_LOCK)
        raise AssertionError("tampered regime should be rejected")
    except ValueError as error:
        assert "regime" in str(error)


if __name__ == "__main__":
    test_closed_form_realized_mass()
    test_auc_and_exact_sign_flip_helpers()
    test_frozen_artifact_anchors()
    test_tampered_group_ledger_is_rejected()
    print("Acrobot V3 mechanism-audit tests passed")
