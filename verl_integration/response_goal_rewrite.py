"""Conservative response rewriting for exact hindsight relabels.

Hindsight changes the task's target, so a response that explicitly repeats
the old *goal* should repeat the new goal instead.  Numeric replacement is
otherwise unsafe: a bare ``\b12\b`` also matches the ``12`` in ``12.5`` and
the result of an intermediate equation such as ``4 * 3 = 12``.

This module therefore rewrites only a small, explicit grammar of goal
statements.  It never rewrites ``<answer>...</answer>`` blocks, signed or
decimal numbers, digit substrings, or arithmetic expressions.  Unknown
wording is deliberately left unchanged; leaving an unrecognised statement
alone is safer than silently changing the reasoning that certifies the
relabeled answer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_ANSWER_BLOCK_RE = re.compile(r"(<answer\b[^>]*>.*?</answer\s*>)",
                              re.IGNORECASE | re.DOTALL)


def _target_token(old_target: int) -> str:
    """An exact, unsigned integer token that is not part of arithmetic."""
    old = re.escape(str(int(old_target)))
    # The lookarounds protect decimals, signed values, identifiers, and
    # longer integers.  The final lookahead protects a value that begins an
    # arithmetic continuation (for example ``target is 12 / 3``).
    return (
        rf"(?<![\w.+-]){old}(?![\w]|\.\d)"
        rf"(?!\s*[+\-*/%=^])"
    )


def _goal_patterns(old_target: int, domain: str) -> Iterable[re.Pattern[str]]:
    target = _target_token(old_target)
    flags = re.IGNORECASE

    # Named declarations: "the target is 42", "goal: 42", and variants.
    yield re.compile(
        rf"(?P<prefix>\b(?:the\s+|our\s+|my\s+)?"
        rf"(?:target|goal)(?:\s+(?:number|value))?\s*"
        rf"(?:is|equals?|=|:)\s*)(?P<target>{target})",
        flags,
    )

    # Intent declarations: "we need to reach 42", "trying to make 42".
    # Requiring an intent verb prevents a later arithmetic observation such
    # as "we can make 12 with 4 * 3" from being treated as the task goal.
    yield re.compile(
        rf"(?P<prefix>\b(?:(?:i|we)\s+)?(?:"
        rf"(?:need|want|aim|have|try|trying|are\s+trying)\s+to|must)\s+"
        rf"(?:reach|hit|make|produce|obtain|get)\s+)"
        rf"(?P<target>{target})",
        flags,
    )

    # "The goal is to reach 42" has both a named goal and an action verb.
    yield re.compile(
        rf"(?P<prefix>\b(?:the\s+|our\s+|my\s+)?(?:target|goal)\s+"
        rf"(?:is\s+)?to\s+(?:reach|hit|make|produce|obtain|get)\s+)"
        rf"(?P<target>{target})",
        flags,
    )

    # Common Countdown planning declaration from instruction-tuned models:
    # "create an equation that equals 42".  The deliberately closed grammar
    # between the intent and target excludes equations and numeric work.
    yield re.compile(
        rf"(?P<prefix>\b(?:(?:(?:i|we)\s+)?"
        rf"(?:need|want|try|trying|are\s+trying)\s+to\s+)?"
        rf"(?:find|create|construct|build)\s+"
        rf"(?:an?\s+)?(?:expression|equation|solution)\s+"
        rf"(?:that\s+)?(?:equals?|evaluates?\s+to)\s+)"
        rf"(?P<target>{target})",
        flags,
    )

    if domain == "jugs":
        # Jugs answers are move names, but its think text can restate the
        # goal.  Require an intent phrase so intermediate state descriptions
        # ("jug A contains exactly 4 litres") remain untouched.
        yield re.compile(
            rf"(?P<prefix>\b(?:(?:i|we)\s+)?(?:need|want|aim|have|try|trying)"
            rf"\s+to\s+(?:(?:make|have)\s+)?(?:a\s+jug\s+)?"
            rf"contain\s+exactly\s+)(?P<target>{target})"
            rf"(?P<suffix>\s+lit(?:re|er)s?\b)",
            flags,
        )


def rewrite_response_goal(
    response_text: str,
    old_target: int,
    new_target: int,
    *,
    domain: str = "countdown",
) -> str:
    """Rewrite explicit old-goal declarations in a model response.

    Supported domains are ``countdown`` and ``jugs``.  Every other number,
    including values in the final answer, remains byte-for-byte unchanged.
    The function is intentionally idempotent after the first rewrite because
    it searches only for ``old_target``.
    """
    if domain not in {"countdown", "jugs"}:
        raise ValueError(f"unsupported relabel domain: {domain!r}")
    old_target = int(old_target)
    new_target = int(new_target)
    if old_target == new_target:
        return response_text

    def rewrite_unprotected(segment: str) -> str:
        for pattern in _goal_patterns(old_target, domain):
            def replace(match: re.Match[str]) -> str:
                suffix = match.groupdict().get("suffix") or ""
                return f"{match.group('prefix')}{new_target}{suffix}"

            segment = pattern.sub(replace, segment)
        return segment

    # Capturing split keeps answer blocks in the output.  Only even-indexed
    # text segments are candidates for rewriting.
    pieces = _ANSWER_BLOCK_RE.split(response_text)
    for index in range(0, len(pieces), 2):
        pieces[index] = rewrite_unprotected(pieces[index])
    return "".join(pieces)
