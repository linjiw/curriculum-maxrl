"""Focused regression tests for the retained skill-chain component ablation."""

from __future__ import annotations

import numpy as np
import pytest

from frontier_rl.examples.run_skill_chain_ablation import (
    CASES,
    CONTRASTS,
    Contrast,
    contrast_values,
    exact_sign_flip_p,
    holm_adjust,
    run_case,
)


def test_checkpoint_mean_and_rollout_budget_are_explicit():
    run = run_case(
        CASES[0],
        seed=3,
        steps=20,
        checkpoint_every=5,
        n_rollouts=4,
        tasks_per_step=2,
    )
    assert run["checkpoint_steps"] == [0, 5, 10, 15, 20]
    assert run["checkpoint_mean"] == pytest.approx(
        np.mean(run["mean_pass_curve"])
    )
    assert run["sampled_groups"] == 40
    assert run["rollout_attempts"] == 160
    assert (
        run["live_groups"] + run["dead_groups"] + run["all_pass_groups"]
        == run["sampled_groups"]
    )


def test_full_stack_has_direct_hindsight_and_uniform_hindsight_controls():
    names = {contrast.name for contrast in CONTRASTS}
    assert "centered_hindsight_under_teacher_g4" in names
    assert "full_stack_minus_uniform_centered_hindsight" in names
    assert "centered_scale_4p0_minus_1p0" in names
    assert "centered_scale_8p0_minus_1p0" in names


def test_linear_contrast_uses_identically_ordered_paired_seeds():
    cases = {
        "a": {
            "runs": [
                {"seed": 0, "checkpoint_mean": 0.5},
                {"seed": 1, "checkpoint_mean": 0.8},
            ]
        },
        "b": {
            "runs": [
                {"seed": 0, "checkpoint_mean": 0.2},
                {"seed": 1, "checkpoint_mean": 0.3},
            ]
        },
    }
    contrast = Contrast("a_minus_b", "test", {"a": 1.0, "b": -1.0})
    assert np.allclose(
        contrast_values(cases, contrast, "checkpoint_mean"), [0.3, 0.5]
    )

    cases["b"]["runs"][1]["seed"] = 2
    with pytest.raises(ValueError, match="seed mismatch"):
        contrast_values(cases, contrast, "checkpoint_mean")


def test_exact_sign_flip_and_holm_correction():
    # For two identical positive effects, two of the four sign assignments are
    # at least as extreme as the observed absolute mean.
    assert exact_sign_flip_p([1.0, 1.0]) == pytest.approx(0.5)
    assert exact_sign_flip_p([0.0, 0.0, 0.0]) == pytest.approx(1.0)

    adjusted = holm_adjust({"small": 0.01, "middle": 0.03, "large": 0.5})
    assert adjusted["small"]["holm_adjusted_p"] == pytest.approx(0.03)
    assert adjusted["middle"]["holm_adjusted_p"] == pytest.approx(0.06)
    assert adjusted["large"]["holm_adjusted_p"] == pytest.approx(0.5)
    assert adjusted["small"]["reject_familywise_0.05"]
    assert not adjusted["middle"]["reject_familywise_0.05"]
