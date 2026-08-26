from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from llm_calibration.analyze_count_law_calibration import (
    AnalysisError,
    EXPECTED_GROUP_SIZE,
    EXPECTED_TASKS_PER_TIER,
    EXPECTED_TIERS,
    analyze_rows,
)


def synthetic_rows() -> list[dict]:
    rows = []
    for tier_index, (tier, operands) in enumerate(EXPECTED_TIERS.items()):
        for task_index in range(EXPECTED_TASKS_PER_TIER):
            # Same pooled mean in every tier, with deliberately heterogeneous
            # all-fail/all-pass task groups and hence a large positive gap.
            success = int(task_index % 2 == 1)
            rewards = [success] * EXPECTED_GROUP_SIZE
            numbers = [tier_index * 1000 + task_index * 10 + i + 1 for i in range(operands)]
            rows.append({
                "data_source": tier,
                "ground_truth": {"target": 50_000 + tier_index * 1000 + task_index, "numbers": numbers},
                "rewards": rewards,
                "completions": ["<answer>1</answer>"] * EXPECTED_GROUP_SIZE,
                "achieved_values": [1] * EXPECTED_GROUP_SIZE,
                "new_tokens": [4] * EXPECTED_GROUP_SIZE,
            })
    return rows


class AnalyzerTests(unittest.TestCase):
    def test_same_mean_different_count_law_gap(self):
        result = analyze_rows(synthetic_rows())
        self.assertIsNone(result["inference"]["hypothesis_test"])
        self.assertIsNone(result["inference"]["decision_rule"])
        for bucket in result["buckets"].values():
            self.assertEqual(bucket["mean_pass_rate"], 0.5)
            self.assertEqual(bucket["empirical_all_fail_probability"], 0.5)
            self.assertGreater(bucket["all_fail_gap"], 0.49)
            self.assertAlmostEqual(
                bucket["plugin_minus_count_law_activity"],
                2 * bucket["all_fail_gap"],
            )
            self.assertEqual(sum(bucket["count_histogram_k_0_to_16"]), 128)

    def test_duplicate_task_fails_closed(self):
        rows = synthetic_rows()
        rows[1]["ground_truth"] = copy.deepcopy(rows[0]["ground_truth"])
        with self.assertRaisesRegex(AnalysisError, "duplicate task identity"):
            analyze_rows(rows)

    def test_nonbinary_reward_fails_closed(self):
        rows = synthetic_rows()
        rows[0]["rewards"][0] = 0.0
        with self.assertRaisesRegex(AnalysisError, "binary integers"):
            analyze_rows(rows)

    def test_missing_row_fails_closed(self):
        with self.assertRaisesRegex(AnalysisError, "expected 384 rows"):
            analyze_rows(synthetic_rows()[:-1])

    def test_cli_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.jsonl"
            raw.write_text(
                "".join(json.dumps(row) + "\n" for row in synthetic_rows()),
                encoding="utf-8",
            )
            output = root / "analysis.json"
            output.write_text("already here\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "llm_calibration.analyze_count_law_calibration",
                    "--raw",
                    str(raw),
                    "--output",
                    str(output),
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("refusing to overwrite", completed.stderr)


if __name__ == "__main__":
    unittest.main()
