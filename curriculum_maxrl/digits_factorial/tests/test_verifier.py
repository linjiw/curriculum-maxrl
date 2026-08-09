from __future__ import annotations

from pathlib import Path

import pytest

from curriculum_maxrl.digits_factorial import locking, verify_portable
from curriculum_maxrl.digits_factorial.verify_portable import verify_source_only


def test_portable_source_verifier_checks_exact_locked_file_set() -> None:
    report = verify_source_only(check_runtime=True)
    assert report["passed"] is True
    assert report["source_file_count"] == len(report["checked_source_files"])


def test_skip_runtime_check_does_not_call_host_runtime_assertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        locking,
        "assert_pinned_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("host runtime was checked")),
    )
    assert verify_source_only(check_runtime=False)["passed"] is True


def test_portable_engineering_threads_skip_runtime_flag_into_reanalysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake(root: Path, *, parallel_root: Path, check_runtime: bool):
        captured.update(
            root=root, parallel_root=parallel_root, check_runtime=check_runtime
        )
        return {"passed": True}

    monkeypatch.setattr(verify_portable.analyze, "analyze_engineering", fake)
    report = verify_portable.verify_engineering(
        Path("serial"), Path("parallel"), None, check_runtime=False
    )
    assert report["passed"] is True
    assert captured["check_runtime"] is False
