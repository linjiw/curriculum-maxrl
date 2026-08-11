"""Strict binary Countdown verifier used by the verl reward hook.

The parquet ground truth is ``{"target": int, "numbers": list[int]}``.
Only an arithmetic expression inside the last ``<answer>`` block is scored.
The expression must use every supplied integer exactly once. Evaluation uses
an AST whitelist and exact rational arithmetic; Python ``eval`` is never used.
"""

from __future__ import annotations

import ast
import re
from fractions import Fraction
from typing import Optional

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_INTEGER_RE = re.compile(r"(?<![\d.])-?\d+(?![\d.])")
_TARGET_SLOT_RE = re.compile(r"(create an equation that equals\s+)(-?\d+)(\s*[.])", re.IGNORECASE)


def extract_equation(solution_str: str) -> Optional[str]:
    matches = _ANSWER_RE.findall(solution_str or "")
    return matches[-1].strip() if matches else None


def _eval_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ZeroDivisionError
        return left / right
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def evaluate_equation(equation: str) -> Optional[Fraction]:
    if len(equation) > 512:
        return None
    try:
        return _eval_node(ast.parse(equation, mode="eval"))
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def uses_numbers_once(equation: str, numbers: list[int]) -> bool:
    """Check leaf integer literals, treating unary minus as sign syntax."""
    try:
        tree = ast.parse(equation, mode="eval")
    except SyntaxError:
        return False

    leaves: list[int] = []

    def visit(node: ast.AST, sign: int = 1) -> bool:
        if isinstance(node, ast.Expression):
            return visit(node.body, sign)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            leaves.append(sign * node.value)
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand, sign if isinstance(node.op, ast.UAdd) else -sign)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            return visit(node.left) and visit(node.right)
        return False

    return visit(tree) and sorted(leaves) == sorted(int(n) for n in numbers)


def achieved_value(solution_str: str, numbers: list[int]) -> Optional[int]:
    """Return the positive integer reached by a verifier-valid expression."""
    equation = extract_equation(solution_str)
    if equation is None or not uses_numbers_once(equation, numbers):
        return None
    value = evaluate_equation(equation)
    if value is None or value.denominator != 1:
        return None
    integer = int(value)
    return integer if 0 < integer <= 100_000 else None


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs) -> float:
    if not isinstance(ground_truth, dict):
        return 0.0
    value = achieved_value(solution_str or "", list(ground_truth.get("numbers", [])))
    return float(value == int(ground_truth.get("target", -1)))


def rewrite_prompt_text(text: str, old_target: int, new_target: int) -> Optional[str]:
    """Rewrite only the explicit target slot, never decimals or examples."""
    matches = list(_TARGET_SLOT_RE.finditer(text or ""))
    valid = [m for m in matches if int(m.group(2)) == int(old_target)]
    if len(valid) != 1:
        return None
    match = valid[0]
    return text[: match.start(2)] + str(int(new_target)) + text[match.end(2) :]

