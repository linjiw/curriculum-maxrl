from __future__ import annotations

import json
from pathlib import Path

import pytest

from granularity_flip import replay_delivery as replay


def _write_stream(path: Path, count: int) -> None:
    rows = []
    for update in range(1, replay.N_UPDATES + 1):
        rows.append(
            {
                "record_type": "telemetry",
                "protocol": "maze_score_v2",
                "completed_updates": update,
                "selected_levels": [0] * replay.TASKS_PER_UPDATE,
                "group_k": [count] * replay.TASKS_PER_UPDATE,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_replay_is_zero_for_iid_prior_preserving_counts(tmp_path: Path) -> None:
    path = tmp_path / "mazescore_un_s20.telemetry.jsonl"
    _write_stream(path, 16)
    result = replay.replay_paths([path])
    assert result["outcome_blind"] is True
    assert result["n_seed_blocks"] == 1
    # A degenerate K=16 stream is not binomial and therefore separates after
    # the first observation; initialization itself agrees exactly.
    assert result["blocks"]["20"]["min_tv"] == pytest.approx(0.0)
    assert result["mean_update_tv_across_blocks"] > 0.0


def test_replay_rejects_partial_stream_and_duplicate_seed(tmp_path: Path) -> None:
    path = tmp_path / "mazescore_un_s20.telemetry.jsonl"
    _write_stream(path, 0)
    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(replay.ReplayError, match="updates"):
        replay.replay_paths([path])

    _write_stream(path, 0)
    with pytest.raises(replay.ReplayError, match="duplicate"):
        replay.replay_paths([path, path])
