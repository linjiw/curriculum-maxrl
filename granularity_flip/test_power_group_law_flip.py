from __future__ import annotations

import numpy as np

from .power_group_law_flip import SESOI, simulate_cell


def test_point_estimate_clause_caps_power_at_the_sesoi():
    rng = np.random.default_rng(4)
    support, _, _ = simulate_cell(rng, 48, SESOI, 0.0077, 50_000)
    assert 0.48 < support < 0.52


def test_48_blocks_are_powered_for_1p5x_sesoi_at_pessimistic_sd():
    rng = np.random.default_rng(5)
    support, _, _ = simulate_cell(rng, 48, 0.0075, 0.0135, 50_000)
    assert 0.89 < support < 0.92
