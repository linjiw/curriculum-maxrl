from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from curriculum_maxrl.analysis import maze_wave2_auc_multiverse as auc


def _passk(coverage: float) -> dict[str, dict[str, float]]:
    return {
        str(level): {"1": coverage / 2.0, "8": coverage}
        for level in range(13)
    }


def _write_synthetic_inputs(root: Path) -> tuple[Path, Path]:
    raw_dir = root / "raw"
    raw_dir.mkdir()
    cells = {}
    for sampler_index, sampler in enumerate(auc.SAMPLERS):
        for estimator in auc.ESTIMATORS:
            for seed in auc.SEEDS:
                init = 0.2
                values = []
                rows = [{"step": -1, "passk": _passk(init)}]
                for step in auc.CHECKPOINT_STEPS:
                    progress = step / 250.0
                    common = 0.01 * progress
                    advantage = 0.0
                    if estimator == "maxrl":
                        advantage = (
                            0.012 + 0.004 * sampler_index
                            + 0.0005 * (seed - min(auc.SEEDS))
                        ) * progress ** 2
                    coverage = init + common + advantage
                    values.append(coverage)
                    rows.append({
                        "step": step,
                        "final": step == 250,
                        "passk": _passk(coverage),
                    })
                filename = auc._raw_filename(sampler, estimator, seed)
                (raw_dir / filename).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                cells[auc._cell_key(sampler, estimator, seed)] = {
                    "cov_auc_delta": sum(values) / len(values) - init,
                }
    summary = root / "summary.json"
    summary.write_text(json.dumps({"cells": cells}, indent=2) + "\n")
    return summary, raw_dir


class MazeWave2AucMultiverseTest(unittest.TestCase):
    def test_missing_raw_data_is_structured_explicit_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty"
            empty.mkdir()
            result = auc.build_analysis(auc.DEFAULT_SUMMARY, empty)
            self.assertEqual(result["status"], "insufficient_raw_trajectories")
            self.assertEqual(result["raw_data_audit"]["n_missing_files"], 24)
            self.assertIsNone(result["multiverse"])
            self.assertIsNone(
                result["robustness_summary"]["minimum_positive_blocks"]
            )
            self.assertEqual(
                result["legacy_summary_anchor"]["per_sampler"]["uniform"][
                    "n_positive"
                ],
                6,
            )
            self.assertIn("not estimable", auc.render_report(result))

    def test_complete_synthetic_multiverse_has_frozen_variant_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary, raw_dir = _write_synthetic_inputs(Path(directory))
            result = auc.build_analysis(summary, raw_dir)
            self.assertEqual(result["status"], "complete")
            robustness = result["robustness_summary"]
            self.assertEqual(robustness["n_base_variants"], 12)
            self.assertEqual(
                robustness["n_leave_one_checkpoint_out_variants"], 88
            )
            self.assertEqual(robustness["n_total_variants"], 100)
            for scope in (
                "base_variants", "base_plus_leave_one_checkpoint_out"
            ):
                for view in (
                    "uniform", "frontier_un", "sampler_average_within_block"
                ):
                    self.assertEqual(
                        robustness[scope][view]["minimum_positive_blocks"], 6
                    )
            self.assertIn("100 total", auc.render_report(result))

    def test_raw_logs_must_reproduce_frozen_legacy_auc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary, raw_dir = _write_synthetic_inputs(Path(directory))
            source = json.loads(summary.read_text())
            key = auc._cell_key("uniform", "maxrl", min(auc.SEEDS))
            source["cells"][key]["cov_auc_delta"] += 0.01
            summary.write_text(json.dumps(source, indent=2) + "\n")
            result = auc.build_analysis(summary, raw_dir)
            self.assertEqual(result["status"], "invalid_raw_trajectories")
            errors = result["raw_data_audit"]["validation_errors"]
            self.assertEqual(len(errors), 1)
            self.assertIn("does not match frozen summary", errors[0])

    def test_cli_writes_audit_but_returns_nonzero_when_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty"
            empty.mkdir()
            json_out = root / "result.json"
            report_out = root / "report.md"
            args = [
                "--summary", str(auc.DEFAULT_SUMMARY),
                "--raw-dir", str(empty),
                "--json-out", str(json_out),
                "--report-out", str(report_out),
            ]
            self.assertEqual(auc.main(args), 2)
            self.assertTrue(json_out.is_file())
            self.assertTrue(report_out.is_file())
            self.assertEqual(auc.main(args + ["--check"]), 2)


if __name__ == "__main__":
    unittest.main()
