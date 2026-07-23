"""CPU tests for the production verl curriculum module."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from curriculum import CurriculumSampler, FrontierTeacher  # noqa: E402


def observed_teacher(seed=0):
    teacher = FrontierTeacher(4, n_rollouts=8, seed=seed)
    teacher.observe_batch(
        np.repeat(np.arange(4), 4),
        np.repeat(np.array(["a", "b", "c", "d"]), 4),
        np.array([
            1, 1, 1, 1,
            1, 0, 0, 0,
            0, 0, 0, 0,
            1, 1, 0, 0,
        ]),
    )
    return teacher


def test_teacher_checkpoint_restores_exact_rng_stream():
    teacher = observed_teacher(seed=7)
    state = teacher.state_dict()
    expected = teacher.sampling_weights()

    restored = FrontierTeacher(4, n_rollouts=8, seed=999)
    restored.load_state_dict(state)
    assert np.array_equal(restored.sampling_weights(), expected)


def test_teacher_checkpoint_rejects_configuration_and_shape_mismatch():
    state = observed_teacher().state_dict()
    with pytest.raises(ValueError, match="configuration mismatch"):
        FrontierTeacher(4, n_rollouts=4).load_state_dict(state)
    with pytest.raises(ValueError, match="prompts"):
        FrontierTeacher(5, n_rollouts=8).load_state_dict(state)


def test_sampler_checkpoint_restores_exact_prompt_stream():
    teacher = observed_teacher(seed=3)
    sampler = CurriculumSampler(list(range(4)), teacher, seed=4)
    teacher_state = teacher.state_dict()
    sampler_state = sampler.state_dict()
    expected = list(sampler)

    restored_teacher = FrontierTeacher(4, n_rollouts=8, seed=30)
    restored_sampler = CurriculumSampler(
        list(range(4)), restored_teacher, seed=40
    )
    restored_teacher.load_state_dict(teacher_state)
    restored_sampler.load_state_dict(sampler_state)
    assert list(restored_sampler) == expected


def test_observe_batch_rejects_inconsistent_or_malformed_groups():
    teacher = FrontierTeacher(3)
    with pytest.raises(ValueError, match="equal lengths"):
        teacher.observe_batch([0, 0], ["a"], [1, 0])
    with pytest.raises(ValueError, match="multiple dataset indices"):
        teacher.observe_batch([0, 1], ["a", "a"], [1, 0])
    with pytest.raises(ValueError, match="outside"):
        teacher.observe_batch([3], ["a"], [1])
    with pytest.raises(ValueError, match="finite"):
        teacher.observe_batch([0], ["a"], [np.nan])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_prompts": 0},
        {"n_prompts": 2, "n_rollouts": 1},
        {"n_prompts": 2, "decay": 1.1},
        {"n_prompts": 2, "floor": -0.1},
        {"n_prompts": 2, "utility": "unknown"},
        {"n_prompts": 2, "power": 0},
    ],
)
def test_teacher_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        FrontierTeacher(**kwargs)
