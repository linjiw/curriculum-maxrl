"""Regression tests for the production verl drop-in curriculum module."""

from __future__ import annotations

import numpy as np
import pytest

from verl_integration.curriculum import (
    CurriculumIndexedDataset,
    CurriculumSampler,
    FrontierTeacher,
)


def test_dataset_wrapper_uses_post_filter_position_not_source_id():
    source = [
        {"index": 100},
        {"index": 7},
        {"index": 100},  # collision is harmless for the separate position
    ]
    wrapped = CurriculumIndexedDataset(source)
    assert [wrapped[i]["curriculum_index"] for i in range(3)] == [0, 1, 2]
    assert [wrapped[i]["index"] for i in range(3)] == [100, 7, 100]


def test_production_teacher_and_sampler_resume_mid_epoch():
    data = list(range(12))
    teacher_a = FrontierTeacher(len(data), n_rollouts=8, seed=11)
    teacher_a.observe_batch(
        np.array([3] * 8), np.array(["group"] * 8),
        np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    sampler_a = CurriculumSampler(data, teacher_a)
    iterator_a = iter(sampler_a)
    assert len([next(iterator_a) for _ in range(5)]) == 5
    teacher_state = teacher_a.state_dict()
    sampler_state = sampler_a.state_dict()
    expected_remainder = list(iterator_a)
    expected_next_epoch = list(iter(sampler_a))

    teacher_b = FrontierTeacher(len(data), n_rollouts=8, seed=999)
    teacher_b.load_state_dict(teacher_state)
    sampler_b = CurriculumSampler(data, teacher_b)
    sampler_b.load_state_dict(sampler_state)
    assert list(iter(sampler_b)) == expected_remainder
    assert list(iter(sampler_b)) == expected_next_epoch


def test_production_advmass_is_exact_fixed_n_family():
    teacher = FrontierTeacher(3, n_rollouts=4, seed=0, utility="advmass")
    p = np.array([0.1, 0.3, 0.8])
    expected = 1.0 - (1.0 - p) ** 4 - p
    got = teacher.utility(p)
    assert np.allclose(got, expected)


def test_feedback_contract_and_checkpoint_config_fail_loudly():
    teacher = FrontierTeacher(4, n_rollouts=8, seed=0)
    with pytest.raises(ValueError, match="equal length"):
        teacher.observe_batch([0, 0], ["u"], [1.0, 0.0])
    with pytest.raises(ValueError, match="maps to positions"):
        teacher.observe_batch([0, 1], ["u", "u"], [1.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        teacher.observe_batch([0], ["u"], [np.nan])

    state = teacher.state_dict()
    incompatible = FrontierTeacher(4, n_rollouts=4, seed=0)
    with pytest.raises(ValueError, match="config mismatch"):
        incompatible.load_state_dict(state)
