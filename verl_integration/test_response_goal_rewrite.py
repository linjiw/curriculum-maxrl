"""CPU-only regression and property tests for response goal rewriting."""

from __future__ import annotations

import random

import pytest

from verl_integration.hindsight import CountdownHindsight, JugsHindsight
from verl_integration.response_goal_rewrite import rewrite_response_goal


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>I need to reach 12.</think><answer>4 * 3</answer>",
         "<think>I need to reach 99.</think><answer>4 * 3</answer>"),
        ("<think>The target is 12. Find a solution.</think>",
         "<think>The target is 99. Find a solution.</think>"),
        ("<think>Our goal value: 12. We want to make 12.</think>",
         "<think>Our goal value: 99. We want to make 99.</think>"),
        ("Need to create an equation that equals 12.\n<answer>6 + 6</answer>",
         "Need to create an equation that equals 99.\n<answer>6 + 6</answer>"),
        ("<think>We need to find an expression that evaluates to 12.</think>",
         "<think>We need to find an expression that evaluates to 99.</think>"),
        ("<think>The goal is to make 12. We must reach 12.</think>",
         "<think>The goal is to make 99. We must reach 99.</think>"),
        ("<think>We are trying to get 12.</think>",
         "<think>We are trying to get 99.</think>"),
        ("<THINK>Goal = 12.</THINK><ANSWER>12</ANSWER>",
         "<THINK>Goal = 99.</THINK><ANSWER>12</ANSWER>"),
    ],
)
def test_supported_countdown_goal_statements(text: str, expected: str) -> None:
    assert rewrite_response_goal(text, 12, 99) == expected


@pytest.mark.parametrize(
    "text",
    [
        "12.5",
        "A measurement changed from 12.5 to 99.5.",
        "4 * 3 = 12",
        "First compute 4 * 3 = 12, then continue.",
        "Values are -12, +12, - 12, and + 12.",
        "Identifiers v12 and 12th, plus integers 112 and 120.",
        "The target is -12, not 12 / 3.",
        "The target is 12 - x.",
        "The target is 12 -> 99.",
        "The goal is 12 % 5.",
        "We can make 12 with 4 * 3.",
        "<answer>The target is 12; 4 * 3 = 12; 12.5</answer>",
    ],
)
def test_numeric_work_and_answer_blocks_are_never_rewritten(text: str) -> None:
    # In particular, reject the old corruptions 12.5 -> 99.5 and
    # 4 * 3 = 12 -> 4 * 3 = 99.
    assert rewrite_response_goal(text, 12, 99) == text


def test_only_goal_mentions_change_in_mixed_countdown_trace() -> None:
    text = (
        "<think>We need to reach 12. Keep 12.5 as a decimal. "
        "An intermediate check is 4 * 3 = 12. The target is 12.</think>"
        "<answer>(4 * 3) + 87</answer>"
    )
    assert rewrite_response_goal(text, 12, 99) == (
        "<think>We need to reach 99. Keep 12.5 as a decimal. "
        "An intermediate check is 4 * 3 = 12. The target is 99.</think>"
        "<answer>(4 * 3) + 87</answer>"
    )


def test_release_classes_route_through_safe_rewriter() -> None:
    countdown = object.__new__(CountdownHindsight)
    jugs = object.__new__(JugsHindsight)
    assert countdown._rewrite_response_text(
        "Goal: 12. 4 * 3 = 12.", 12, 99
    ) == "Goal: 99. 4 * 3 = 12."
    assert jugs._rewrite_response_text(
        "We need to have a jug contain exactly 4 litres. "
        "Jug A contains exactly 4 litres.",
        4,
        7,
    ) == (
        "We need to have a jug contain exactly 7 litres. "
        "Jug A contains exactly 4 litres."
    )


def test_rewrite_is_idempotent_after_goal_change() -> None:
    once = rewrite_response_goal("The goal is 12.", 12, 99)
    assert rewrite_response_goal(once, 12, 99) == once


def test_unknown_domain_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported relabel domain"):
        rewrite_response_goal("The goal is 12.", 12, 99, domain="other")


def test_fuzz_non_goal_numeric_context_is_byte_stable() -> None:
    """Property-style corpus: unrelated numeric contexts never change."""
    rng = random.Random(20260807)
    templates = [
        "{a} * {b} = 12",
        "{a} + 12 = {b}",
        "decimal 12.{d}",
        "signed -12 and +12",
        "range {a}-12-{b}",
        "token x12_{d}",
        "<answer>{a} * {b} = 12</answer>",
        "we can make 12 with {a} and {b}",
        "the target is 12 - {a}",
        "the goal is 12 / {b}",
    ]
    for _ in range(500):
        text = rng.choice(templates).format(
            a=rng.randint(1, 999),
            b=rng.randint(1, 999),
            d=rng.randint(0, 9),
        )
        assert rewrite_response_goal(text, 12, 99) == text


def test_fuzz_explicit_goal_changes_once_and_preserves_suffix() -> None:
    """Property-style corpus: goal changes; generated arithmetic does not."""
    rng = random.Random(20260808)
    prefixes = [
        "The target is ",
        "our goal: ",
        "We need to reach ",
        "I want to make ",
        "create an equation that equals ",
    ]
    for _ in range(500):
        a, b = rng.randint(1, 999), rng.randint(1, 999)
        suffix = f". Intermediate: {a} * {b} = 12; decimal 12.5."
        prefix = rng.choice(prefixes)
        text = prefix + "12" + suffix
        rewritten = rewrite_response_goal(text, 12, 99)
        assert rewritten == prefix + "99" + suffix
