"""Regression tests for the independent Acrobot V3 artifact verifier."""

from __future__ import annotations

import platform

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from frontier_rl.examples import analyze_acrobot_v3_confirmatory as verifier


def _run(seed: int, terminal_pass: float) -> dict:
    auc = terminal_pass / 2.0
    return {
        "seed": seed,
        "numeric_valid": True,
        "accounting_valid": True,
        "verifier_relabel_checks_valid": True,
        "evaluation_cadence_invariant": True,
        "total_parameters": 640,
        "active_parameters_per_task": 640,
        "x_transitions": [0, 2_000_000],
        "mean_pass_curve": [0.0, terminal_pass],
        "auc_mean_pass_by_transitions": auc,
    }


def _artifact(effect: float = 0.04) -> dict:
    seeds = verifier.EXPECTED_SEEDS
    uniform = [_run(seed, 0.0) for seed in seeds]
    teacher = [_run(seed, 2.0 * effect) for seed in seeds]
    p_value = 2.0 / (2**20)
    return {
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
            "source_sha256": {},
        },
        "protocol": {
            "stage": "core",
            "status": "confirmatory",
            "exploratory": False,
            "explicit_exploratory": False,
            "paired_seeds": seeds,
            "n_rollouts": 16,
            "core_architectures": ["shared"],
            "core_analysis_mode": "two_cell_shared_efficacy_only",
            "transfer_claim_evaluated": False,
            "condition_names": list(verifier.EXPECTED_CASES),
            "eval_interval_transitions": 100_000,
            "eval_n_per_task": 32,
            "budget": {
                "transition_budget": 2_000_000,
                "optimizer_update_budget": None,
                "transition_safety_cap": None,
            },
        },
        "artifact_state": "complete",
        "run_failures": [],
        "cases": {
            "uniform_shared_h64": {
                "config": {
                    "sampling": "uniform",
                    "architecture": "shared",
                    "hidden_size": 64,
                    "learning_rate": 3e-4,
                    "hindsight_scale": 0.0,
                },
                "runs": uniform,
            },
            "teacher_shared_h64": {
                "config": {
                    "sampling": "teacher",
                    "architecture": "shared",
                    "hidden_size": 64,
                    "learning_rate": 3e-4,
                    "hindsight_scale": 0.0,
                },
                "runs": teacher,
            },
        },
        "paired_core_contrasts": {
            "curriculum_efficacy_shared": {
                "metric": verifier.PRIMARY_METRIC,
                "per_seed_contrast": [effect] * 20,
                "mean_contrast": effect,
                "exact_paired_sign_flip_p_two_sided": p_value,
                "mean_ci95_paired_seed_bootstrap": [effect, effect],
            }
        },
        "predeclared_core_decision": {
            "efficacy_supported": effect >= 0.03 and p_value <= 0.05,
            "transfer_claim_evaluated": False,
        },
    }


def _lock() -> dict:
    return {
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
        },
        "source_sha256": {},
    }


def test_full_synthetic_confirmation_reproduces():
    report = verifier.verify(_artifact(), _lock())
    assert report["all_checks_passed"]
    assert report["primary"]["mean_contrast"] == pytest.approx(0.04)
    assert report["primary"]["efficacy_supported"]
    assert report["primary"]["transfer_claim_evaluated"] is False


def test_tampered_auc_is_rejected():
    artifact = _artifact()
    artifact["cases"]["teacher_shared_h64"]["runs"][0][
        "auc_mean_pass_by_transitions"
    ] = 0.2
    with pytest.raises(ValueError, match="run AUC"):
        verifier.verify(artifact, _lock())
