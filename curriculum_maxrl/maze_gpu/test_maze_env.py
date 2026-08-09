"""Regression tests for verifier-aligned maze hindsight depth."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maze_env import EOS, T_L, T_R, deepest_prefix, simulate_prefix, verify


def test_deepest_prefix_ignores_loop_length():
    grid = np.ones((5, 5), dtype=np.int8)
    grid[1, 1:4] = 0
    response = [T_R, T_L, T_R, T_L]

    legal_n, legal_end = simulate_prefix(grid, response)
    prefix_n, achieved, depth = deepest_prefix(grid, response)

    assert legal_n == 4 and legal_end == (1, 1)
    assert prefix_n == 1 and achieved == (1, 2) and depth == 1
    assert verify(grid, achieved, response[:prefix_n] + [EOS])


def test_verifier_enforces_level_move_budget():
    grid = np.ones((5, 5), dtype=np.int8)
    grid[1, 1:4] = 0
    goal = (1, 3)
    assert verify(grid, goal, [T_R, T_R, EOS], max_moves=2)
    # This reaches the same goal only after wasting two extra legal moves.
    late = [T_R, T_L, T_R, T_R, EOS]
    assert verify(grid, goal, late)
    assert not verify(grid, goal, late, max_moves=2)
