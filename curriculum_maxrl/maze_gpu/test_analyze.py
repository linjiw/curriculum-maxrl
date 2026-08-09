"""Regression tests for anchored maze AUC accounting."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze import summarize


def test_auc_includes_post_sft_anchor(tmp_path):
    path = tmp_path / "run.jsonl"
    records = [
        {"step": -1, "eval": {"0": 0.0}},
        {"step": 0, "elapsed": 10.0, "eval": {"0": 1.0}},
        {"step": 1, "elapsed": 20.0, "eval": {"0": 1.0}},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    got = summarize(str(path))
    assert np.isclose(got["auc_steps"], 0.75)
    assert np.isclose(got["auc_wall_seconds"], 0.75)
    assert np.isclose(got["historical_auc_steps_unanchored"], 1.0)
