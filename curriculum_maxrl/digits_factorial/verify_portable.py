"""Read-only integrity and reanalysis verifier for the Digits evidence bundle.

The verifier never launches training.  It checks the frozen source/data/runtime
declaration and can independently validate engineering, development, and
confirmation artifacts.  Fresh analyses are written only to an OS temporary
directory and compared with stored reports.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from . import analyze
from .core import (
    LOCK_SCHEMA,
    PROJECT_ROOT,
    SOURCE_LOCK_PATH,
    sha256_file,
    strict_json_load,
)
from .locking import LOCKED_RELATIVE_PATHS, load_and_verify_source_lock


VERIFICATION_SCHEMA = "curriculum-maxrl/digits-factorial-portable-verification/v1"


def _compare_payloads(
    observed: Any,
    expected: Any,
    *,
    ignored_keys: frozenset[str] = frozenset(),
    location: str = "root",
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping):
            raise ValueError(f"{location}: stored/fresh types differ")
        expected_keys = set(expected) - ignored_keys
        observed_keys = set(observed) - ignored_keys
        if expected_keys != observed_keys:
            raise ValueError(f"{location}: stored/fresh keys differ")
        for key in expected_keys:
            _compare_payloads(
                observed[key],
                expected[key],
                ignored_keys=ignored_keys,
                location=f"{location}.{key}",
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"{location}: stored/fresh list shapes differ")
        for index, (got, want) in enumerate(zip(observed, expected)):
            _compare_payloads(
                got,
                want,
                ignored_keys=ignored_keys,
                location=f"{location}[{index}]",
            )
        return
    if observed != expected:
        raise ValueError(f"{location}: stored value differs from fresh reanalysis")


def verify_source_only(*, check_runtime: bool) -> dict[str, Any]:
    lock, lock_sha = load_and_verify_source_lock(check_runtime=check_runtime)
    if lock.get("schema") != LOCK_SCHEMA:
        raise ValueError("unexpected source lock schema")
    return {
        "passed": True,
        "source_lock_sha256": lock_sha,
        "checked_source_files": sorted(LOCKED_RELATIVE_PATHS),
        "source_file_count": len(LOCKED_RELATIVE_PATHS),
        "data_manifest_sha256": lock["data_manifest_sha256"],
        "runtime_checked_on_this_machine": check_runtime,
    }


def verify_engineering(
    root: Path,
    parallel_root: Path,
    stored_analysis: Path | None,
    *,
    check_runtime: bool,
) -> dict[str, Any]:
    fresh = analyze.analyze_engineering(
        root, parallel_root=parallel_root, check_runtime=check_runtime
    )
    if stored_analysis is not None:
        stored = strict_json_load(stored_analysis)
        _compare_payloads(stored, fresh)
    return {
        "passed": True,
        "root": str(root),
        "parallel_root": str(parallel_root),
        "stored_analysis_sha256": (
            sha256_file(stored_analysis) if stored_analysis is not None else None
        ),
        "fresh_analysis": fresh,
    }


def verify_development(
    root: Path, stored_selection: Path, *, check_runtime: bool
) -> dict[str, Any]:
    stored = strict_json_load(stored_selection)
    with tempfile.TemporaryDirectory(prefix="digits-factorial-verify-") as directory:
        fresh_path = Path(directory) / "lr_selection.json"
        fresh = analyze.analyze_development(
            root, output=fresh_path, check_runtime=check_runtime
        )
    # The caller may relocate a complete evidence bundle; absolute root labels
    # are provenance metadata rather than statistical content.
    _compare_payloads(stored, fresh, ignored_keys=frozenset({"development_root"}))
    return {
        "passed": True,
        "development_root": str(root),
        "stored_selection_sha256": sha256_file(stored_selection),
        "selected_learning_rates_by_estimator": fresh[
            "selected_learning_rates_by_estimator"
        ],
        "selected_common_learning_rate": fresh["selected_common_learning_rate"],
    }


def verify_confirmation(
    tuned_root: Path,
    common_root: Path,
    selection: Path,
    stored_analysis: Path,
    *,
    check_runtime: bool,
) -> dict[str, Any]:
    stored = strict_json_load(stored_analysis)
    with tempfile.TemporaryDirectory(prefix="digits-factorial-verify-") as directory:
        fresh_path = Path(directory) / "confirmation.json"
        fresh = analyze.analyze_confirmation(
            tuned_root,
            common_root,
            lr_selection_path=selection,
            output=fresh_path,
            check_runtime=check_runtime,
        )
    _compare_payloads(
        stored,
        fresh,
        ignored_keys=frozenset(
            {
                "root",
                "lr_selection_relative_path",
            }
        ),
    )
    return {
        "passed": True,
        "tuned_root": str(tuned_root),
        "common_root": str(common_root),
        "stored_analysis_sha256": sha256_file(stored_analysis),
        "tuned_status": fresh["tuned"]["status"],
        "common_rate_status": fresh["common_rate_robustness"]["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument(
        "--skip-runtime-check",
        action="store_true",
        help="verify bundle integrity on a machine other than the execution runtime",
    )
    parser.add_argument("--engineering-root", type=Path)
    parser.add_argument("--engineering-parallel-root", type=Path)
    parser.add_argument("--engineering-analysis", type=Path)
    parser.add_argument("--development-root", type=Path)
    parser.add_argument("--lr-selection", type=Path)
    parser.add_argument("--tuned-root", type=Path)
    parser.add_argument("--common-root", type=Path)
    parser.add_argument("--confirmation-analysis", type=Path)
    args = parser.parse_args()
    check_runtime = not args.skip_runtime_check

    report: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "source": verify_source_only(check_runtime=check_runtime),
    }
    if args.engineering_root is not None:
        if args.engineering_parallel_root is None:
            raise SystemExit("--engineering-root requires --engineering-parallel-root")
        report["engineering"] = verify_engineering(
            args.engineering_root,
            args.engineering_parallel_root,
            args.engineering_analysis,
            check_runtime=check_runtime,
        )
    if args.development_root is not None:
        if args.lr_selection is None:
            raise SystemExit("--development-root requires --lr-selection")
        report["development"] = verify_development(
            args.development_root, args.lr_selection, check_runtime=check_runtime
        )
    if args.tuned_root is not None or args.common_root is not None:
        required = (args.tuned_root, args.common_root, args.lr_selection, args.confirmation_analysis)
        if any(item is None for item in required):
            raise SystemExit(
                "confirmation verification requires --tuned-root, --common-root, "
                "--lr-selection, and --confirmation-analysis"
            )
        report["confirmation"] = verify_confirmation(
            args.tuned_root,
            args.common_root,
            args.lr_selection,
            args.confirmation_analysis,
            check_runtime=check_runtime,
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
