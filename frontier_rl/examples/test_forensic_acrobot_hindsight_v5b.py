from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from frontier_rl.examples import forensic_acrobot_hindsight_v5b as forensic


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "frontier_rl/examples/acrobot_hindsight_v5b_factorial.json"
LOCK = ROOT / "frontier_rl/examples/ACROBOT_HINDSIGHT_V5B_LOCK.json"


def _artifact_is_materialized() -> bool:
    if not ARTIFACT.exists():
        return False
    with ARTIFACT.open("rb") as handle:
        return not handle.read(128).startswith(
            b"version https://git-lfs.github.com/spec/v1\n"
        )


def _on_sealed_runtime() -> bool:
    """The forensic chain refuses to replay off the sealed runtime — the lock
    pins python/platform/numpy exactly, so on any other machine verify()
    correctly raises rather than authorizing.  Skip (like the LFS guard)
    instead of red-failing every environment that isn't the original one."""
    import platform

    if not LOCK.exists():
        return False
    runtime = json.loads(LOCK.read_text()).get("runtime", {})
    return (
        platform.python_version() == runtime.get("python")
        and platform.platform() == runtime.get("platform")
        and np.__version__ == runtime.get("numpy")
    )


def test_numpy_reduction_and_ulp_helpers_are_exact():
    run = {
        "update_diagnostics": [
            {"source": "requested_live", "update_norm": value}
            for value in (0.1, 0.2, 0.3)
        ]
    }
    totals = forensic._numpy_source_norms(run)
    values = np.asarray((0.1, 0.2, 0.3), dtype=np.float64)
    assert totals["requested_live"] == {
        "count": 3,
        "M": float(values.sum()),
        "Q": float(np.dot(values, values)),
    }
    adjacent = float(np.nextafter(1.0, 2.0))
    assert forensic._ulp_distance(1.0, adjacent) == 1
    assert forensic._ulp_distance(adjacent, 1.0) == 1


@pytest.mark.skipif(
    not _artifact_is_materialized(),
    reason="V5B raw artifact is not materialized; run git lfs pull",
)
def test_verifier_rejects_every_artifact_other_than_the_sealed_original(monkeypatch):
    monkeypatch.setattr(forensic, "EXPECTED_ARTIFACT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="forensic verifier is bound"):
        forensic.verify(ARTIFACT, LOCK)


@pytest.mark.skipif(
    not _artifact_is_materialized(),
    reason="V5B raw artifact is not materialized; run git lfs pull",
)
@pytest.mark.skipif(
    not _on_sealed_runtime(),
    reason="forensic replay is bound to the sealed runtime "
    "(python/platform/numpy pinned in the V5B lock)",
)
def test_current_artifact_forensics_are_non_authorizing_and_outcome_free():
    report = forensic.verify(ARTIFACT, LOCK)
    assert report["artifact_sha256"] == forensic.EXPECTED_ARTIFACT_SHA256
    assert report["registered_primary_family_authorized"] is False
    assert report["original_locked_analyzer_passed"] is False
    assert report["mechanical_integrity"]["registered_runs_validated"] == 180
    assert report["mechanical_integrity"]["raw_group_records_validated"] == 53_510
    assert report["mechanical_integrity"]["raw_update_records_validated"] == 45_000
    assert report["mechanical_integrity"]["evaluation_checkpoints_validated"] == 1_080
    reductions = report["reduction_forensics"]
    assert reductions["reduction_fields_checked"] == 720
    assert reductions["saved_runner_numpy_exact_mismatches"] == 0
    assert reductions["analyzer_sequential_python_exact_mismatches"] == 377
    assert reductions["maximum_absolute_difference"] == 1.9984014443252818e-15
    assert reductions["maximum_ulp_distance"] == 11
    assert report["compatibility_diagnostic"]["all_remaining_frozen_checks_passed"]

    serialized = json.dumps(report, sort_keys=True)
    for forbidden in (
        '"independent_contrasts"',
        '"independent_case_summaries"',
        '"independent_predeclared_scale_decision"',
        '"paired_scale_contrasts"',
        '"stage_b_case_summaries"',
        '"predeclared_scale_decision"',
        '"mean_contrast"',
        '"per_seed_contrast"',
    ):
        assert forbidden not in serialized
