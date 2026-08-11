import json
import tempfile
import unittest
from pathlib import Path

from curriculum_maxrl.audit_countdown_sft_overlap import audit, extract_task


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class CountdownSftOverlapAuditTest(unittest.TestCase):
    def test_saved_summary_matches_quoted_overlap_and_blocker(self) -> None:
        path = Path(__file__).with_name("data_integrity_check.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 2)
        overlap = payload["countdown_v2"]["sft_evaluation_overlap"]
        self.assertFalse(overlap["source_inputs_available_in_this_checkout"])
        self.assertEqual(
            overlap["verification_status"],
            "verified_in_original_source_audit_inputs_not_vendored_here",
        )
        tiers = overlap["tiers"]
        self.assertEqual(tiers["countdown_tier0"]["eval_unique_tasks"], 128)
        self.assertEqual(tiers["countdown_tier0"]["sft_overlap_unique_tasks"], 27)
        self.assertEqual(tiers["countdown_tier0"]["clean_unique_tasks"], 101)
        self.assertEqual(tiers["countdown_tier1"]["sft_overlap_unique_tasks"], 0)
        self.assertEqual(tiers["countdown_tier2"]["sft_overlap_unique_tasks"], 0)
        self.assertEqual(
            payload["countdown_v2"]["clean_tier0_reanalysis"]["status"],
            "blocked_missing_per_task_outcomes",
        )

    def test_nested_and_prompt_task_extraction(self) -> None:
        nested = {
            "reward_model": {"ground_truth": {"target": 24, "numbers": [6, 1, 4, 3]}}
        }
        prompt = {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Using the numbers [3, 6, 1, 4], create an equation that equals 24."
                    ),
                }
            ]
        }
        self.assertEqual(extract_task(nested), (24, (1, 3, 4, 6)))
        self.assertEqual(extract_task(prompt), (24, (1, 3, 4, 6)))

    def test_overlap_and_clean_tier0_reanalysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sft_path = root / "sft.jsonl"
            eval_path = root / "eval.jsonl"
            outcomes_path = root / "outcomes.jsonl"
            write_jsonl(
                sft_path,
                [
                    {"numbers": [2, 3], "target": 5},
                    {"numbers": [2, 3], "target": 5},  # duplicate row, one task
                    {"numbers": [1, 2, 3], "target": 6},
                ],
            )
            write_jsonl(
                eval_path,
                [
                    {"numbers": [3, 2], "target": 5, "data_source": "countdown_tier0"},
                    {"numbers": [4, 5], "target": 9, "data_source": "countdown_tier0"},
                    {"numbers": [1, 2, 4], "target": 7, "data_source": "countdown_tier1"},
                    {"numbers": [1, 2, 3, 4], "target": 10, "data_source": "countdown_tier2"},
                ],
            )
            write_jsonl(
                outcomes_path,
                [
                    {
                        "arm": "B1",
                        "seed": 1,
                        "step": 60,
                        "numbers": [2, 3],
                        "target": 5,
                        "tier": 0,
                        "successes": [1, 1, 1, 1],
                    },
                    {
                        "arm": "B1",
                        "seed": 1,
                        "step": 60,
                        "numbers": [4, 5],
                        "target": 9,
                        "tier": 0,
                        "successes": [0, 1, 0, 0],
                    },
                ],
            )

            report = audit([sft_path], [eval_path], [outcomes_path])

        tiers = report["sft_evaluation_overlap"]["tiers"]
        self.assertEqual(tiers["countdown_tier0"]["sft_overlap_unique_tasks"], 1)
        self.assertEqual(tiers["countdown_tier0"]["clean_unique_tasks"], 1)
        self.assertEqual(tiers["countdown_tier1"]["sft_overlap_unique_tasks"], 0)
        self.assertEqual(tiers["countdown_tier2"]["sft_overlap_unique_tasks"], 0)
        self.assertEqual(report["sft"]["duplicate_task_rows"], 1)

        clean = report["clean_tier0_reanalysis"]
        self.assertEqual(clean["status"], "complete")
        self.assertEqual(len(clean["summaries"]), 1)
        summary = clean["summaries"][0]
        self.assertEqual(summary["clean_tasks_observed"], 1)
        self.assertEqual(summary["mean@4"], 0.25)
        self.assertEqual(summary["pass@4"], 1.0)

    def test_without_outcomes_reports_exact_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sft_path = root / "sft.jsonl"
            eval_path = root / "eval.jsonl"
            write_jsonl(sft_path, [{"numbers": [2, 3], "target": 5}])
            write_jsonl(
                eval_path,
                [
                    {"numbers": [2, 4], "target": 6, "data_source": "countdown_tier0"},
                    {
                        "numbers": [1, 2, 4],
                        "target": 7,
                        "data_source": "countdown_tier1",
                    },
                    {
                        "numbers": [1, 2, 3, 4],
                        "target": 10,
                        "data_source": "countdown_tier2",
                    },
                ],
            )
            report = audit([sft_path], [eval_path])
        self.assertEqual(
            report["clean_tier0_reanalysis"]["status"],
            "blocked_missing_per_task_outcomes",
        )


if __name__ == "__main__":
    unittest.main()
