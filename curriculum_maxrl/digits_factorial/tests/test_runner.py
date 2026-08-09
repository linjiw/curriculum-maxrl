from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from curriculum_maxrl.digits_factorial.analyze import analyze_engineering, validate_run
from curriculum_maxrl.digits_factorial.core import Cell
from curriculum_maxrl.digits_factorial.runner import RunSpec, run_one, run_specs


def engineering_spec(tmp_path: Path, estimator: str, sampler: str) -> RunSpec:
    return RunSpec(
        phase="engineering",
        cell=Cell(estimator, sampler),
        seed=33000,
        learning_rate=0.0,
        steps=2,
        include_test=False,
        output_dir=tmp_path / "engineering" / f"{estimator}__{sampler}",
    )


def test_evidence_phase_fails_closed_without_authorization(tmp_path: Path) -> None:
    spec = RunSpec(
        phase="development",
        cell=Cell("practical_maxrl", "uniform"),
        seed=31000,
        learning_rate=0.1,
        steps=512,
        include_test=False,
        output_dir=tmp_path / "development",
    )
    with pytest.raises(ValueError, match="authorization"):
        run_one(spec)


def test_truncated_engineering_run_round_trips_through_independent_analyzer(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    summary_path = run_one(engineering_spec(tmp_path, "practical_maxrl", "uniform"))
    result = validate_run(summary_path)
    assert result["passed"] is True
    assert result["phase"] == "engineering"
    summary = __import__("json").loads(summary_path.read_text())
    assert summary["accounting"]["paid_actions"] == 1024
    assert "include_sealed_test" not in summary
    assert summary["primary_outcome"]["split"] == "dev"
    assert summary["execution"]["torch_num_threads"] == 1


def test_zero_lr_shares_selected_examples_actions_and_evaluation_within_sampler(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    left = run_one(engineering_spec(tmp_path, "practical_maxrl", "p1mp"))
    right = run_one(engineering_spec(tmp_path, "rloo", "p1mp"))
    with np.load(left.parent / "ledger.npz", allow_pickle=False) as a, np.load(
        right.parent / "ledger.npz", allow_pickle=False
    ) as b:
        for name in (
            "selected_train_positions",
            "actions",
            "eval_probabilities_train",
            "eval_probabilities_dev",
        ):
            assert np.array_equal(a[name], b[name])
        assert not np.array_equal(a["weights"], b["weights"])


def test_serial_and_spawn_parallel_engineering_scientific_artifacts_are_byte_equal(
    tmp_path: Path,
) -> None:
    serial_root = tmp_path / "serial" / "engineering"
    parallel_root = tmp_path / "parallel" / "engineering"

    def specs(root: Path) -> list[RunSpec]:
        return [
            RunSpec(
                phase="engineering",
                cell=Cell(estimator, sampler),
                seed=33000,
                learning_rate=0.0,
                steps=1,
                include_test=False,
                output_dir=root / f"{estimator}__{sampler}",
            )
            for estimator in ("practical_maxrl", "rloo")
            for sampler in ("uniform", "p1mp", "u8")
        ]

    run_specs(specs(serial_root), workers=1)
    run_specs(specs(parallel_root), workers=2)
    report = analyze_engineering(serial_root, parallel_root=parallel_root)
    assert report["passed"] is True
    assert report["worker_execution"][
        "serial_parallel_scientific_files_byte_identical"
    ] is True
    assert report["worker_execution"]["orchestration_provenance"] == {
        "serial": {"worker_mode": "serial", "requested_workers": 1},
        "parallel": {
            "worker_mode": "process_pool_worker",
            "requested_workers": 2,
        },
    }
