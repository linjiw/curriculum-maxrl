from __future__ import annotations

import contextlib
import io
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from curriculum_maxrl.maze_score import analyze_maze_score as analyzer


def _passk(value: float) -> dict[str, dict[str, float]]:
    return {str(level): {"8": value} for level in range(13)}


class CampaignFixture:
    def __init__(
        self,
        root: Path,
        *,
        un_delta: float = 0.020,
        learn_delta: float = 0.010,
        unif_delta: float = 0.000,
    ) -> None:
        self.root = root
        self.deltas = {"un": un_delta, "learn": learn_delta, "unif": unif_delta}
        self.paths: list[Path] = []
        for seed in analyzer.EXPECTED_SEEDS:
            for arm in analyzer.ARMS:
                path = root / f"mazescore_{arm}_s{seed}.jsonl"
                self.write_cell(path, arm, seed)
                self.paths.append(path)

    def records(self, arm: str, seed: int) -> list[dict]:
        exponent = {"un": analyzer.EXPECTED_ROLLOUTS, "learn": 2, "unif": None}[arm]
        records: list[dict] = [
            {
                "record_type": "config",
                "protocol": analyzer.PROTOCOL,
                "campaign": "unit-test-campaign",
                "source_manifest": {
                    "train.py": "a" * 64,
                    "analyze_maze_score.py": "b" * 64,
                },
                "arm": arm,
                "seed": seed,
                "seeds": {
                    "base": seed,
                    "sft": seed,
                    "rl": seed,
                    "teacher": seed + 77,
                    "eval_tasks": 202_608_130 + seed,
                    "eval_samples": 302_608_130 + seed,
                },
                "estimator": "maxrl",
                "teacher": {
                    "un": "frontier_un",
                    "learn": "learnability",
                    "unif": "uniform",
                }[arm],
                "rollouts": analyzer.EXPECTED_ROLLOUTS,
                "score_family": (
                    "uniform" if arm == "unif" else "coefficient_activity"
                ),
                "effective_exponent": exponent,
                "steps": analyzer.EXPECTED_STEPS,
                "lr": analyzer.EXPECTED_LR,
                "d_model": analyzer.EXPECTED_D_MODEL,
                "n_layers": analyzer.EXPECTED_N_LAYERS,
                "hindsight": False,
                "hindsight_dense": False,
                "hindsight_to_teacher": False,
                "teacher_power": analyzer.EXPECTED_TEACHER_POWER,
                "tasks_per_step": analyzer.EXPECTED_TASKS_PER_STEP,
                "sft_steps": analyzer.EXPECTED_SFT_STEPS,
                "eval_every": analyzer.EXPECTED_EVAL_EVERY,
                "eval_tasks_per_level": analyzer.EXPECTED_EVAL_TASKS_PER_LEVEL,
                "eval_samples": analyzer.EXPECTED_EVAL_SAMPLES,
                "planned_rl_eval_count": len(analyzer.EXPECTED_UPDATES),
                "sft_checkpoint_sha256": f"{seed:064x}",
            },
            {
                "record_type": "evaluation",
                "protocol": analyzer.PROTOCOL,
                "phase": "post_sft",
                "completed_updates": 0,
                "passk": _passk(0.10),
            },
        ]
        for update in analyzer.EXPECTED_UPDATES:
            record = {
                "record_type": "evaluation",
                "protocol": analyzer.PROTOCOL,
                "phase": "rl",
                "completed_updates": update,
                "passk": _passk(0.10 + self.deltas[arm]),
            }
            if update == analyzer.EXPECTED_STEPS:
                record["final"] = True
            records.append(record)
        return records

    def write_cell(
        self,
        path: Path,
        arm: str,
        seed: int,
        mutate=None,
    ) -> None:
        records = self.records(arm, seed)
        if mutate is not None:
            mutate(records)
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )


class AnalyzeMazeScoreTests(unittest.TestCase):
    def test_complete_campaign_supported_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            first = analyzer.analyze_paths(fixture.paths)
            second = analyzer.analyze_paths(list(reversed(fixture.paths)))

        self.assertTrue(first["complete"])
        self.assertEqual(first["contrasts"], second["contrasts"])
        primary = first["contrasts"]["primary_un_minus_learn"]
        secondary = first["contrasts"]["secondary_un_minus_unif"]
        self.assertAlmostEqual(primary["mean"], 0.010)
        self.assertAlmostEqual(secondary["mean"], 0.020)
        self.assertEqual(primary["decision"], "supported")
        self.assertEqual(secondary["decision"], "supported")
        self.assertAlmostEqual(
            primary["sign_flip_p_two_sided_exact"], 2.0 / (2 ** len(analyzer.EXPECTED_SEEDS))
        )
        self.assertLess(primary["holm_adjusted_p"], 0.05)

    def test_small_precise_effect_is_practically_ruled_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(
                Path(temporary), un_delta=0.020, learn_delta=0.019, unif_delta=0.019
            )
            result = analyzer.analyze_paths(fixture.paths)

        for contrast in result["contrasts"].values():
            self.assertAlmostEqual(contrast["mean"], 0.001)
            self.assertLess(contrast["bootstrap_ci_95"][1], analyzer.SESOI)
            self.assertEqual(contrast["decision"], "practically_ruled_out")

    def test_exact_sign_flip_matches_brute_force(self) -> None:
        differences = np.asarray([0.5, -0.25, 0.125, 0.375, -0.0625])
        observed = abs(float(np.mean(differences)))
        brute_statistics = [
            abs(float(np.mean(differences * np.asarray(signs))))
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        ]
        brute = sum(value >= observed - 1e-15 for value in brute_statistics) / len(
            brute_statistics
        )
        self.assertEqual(analyzer.exact_sign_flip_p(differences), brute)
        with self.assertRaisesRegex(analyzer.AnalysisError, "at most"):
            analyzer.exact_sign_flip_p([1.0] * (analyzer.MAX_EXACT_SIGN_FLIP_N + 1))

    def test_rejects_missing_duplicate_and_extra_timepoints(self) -> None:
        mutations = {
            "missing": lambda rows: rows.pop(3),
            "duplicate": lambda rows: rows.append(dict(rows[2])),
            "extra": lambda rows: rows.append(
                {
                    "record_type": "evaluation",
                    "protocol": analyzer.PROTOCOL,
                    "phase": "rl",
                    "completed_updates": 275,
                    "passk": _passk(0.2),
                }
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = CampaignFixture(root)
                target = root / "mazescore_un_s20.jsonl"
                fixture.write_cell(target, "un", 20, mutation)
                with self.assertRaisesRegex(analyzer.AnalysisError, "timepoint|duplicate"):
                    analyzer.analyze_paths(fixture.paths)

    def test_rejects_nonterminal_final_and_partial_matrix_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = CampaignFixture(root)
            target = root / "mazescore_un_s20.jsonl"

            def remove_final(records: list[dict]) -> None:
                records[-1].pop("final")

            fixture.write_cell(target, "un", 20, remove_final)
            with self.assertRaisesRegex(analyzer.AnalysisError, "final=true"):
                analyzer.analyze_paths(fixture.paths)

            output = root / "forbidden_partial_output.json"
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = analyzer.main(
                    [*[str(path) for path in fixture.paths[:-1]], "--output", str(output)]
                )
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())
            self.assertEqual(stdout.getvalue(), "")
            self.assertNotIn("cov_auc", stderr.getvalue())

    def test_rejects_provenance_exponent_and_checkpoint_mismatches(self) -> None:
        cases = {
            "manifest": (
                "mazescore_un_s20.jsonl",
                lambda rows: rows[0].__setitem__("source_manifest", {"train.py": "c" * 64}),
                "source_manifest mismatch",
            ),
            "exponent": (
                "mazescore_un_s20.jsonl",
                lambda rows: rows[0].__setitem__("effective_exponent", 33),
                "effective_exponent",
            ),
            "checkpoint": (
                "mazescore_learn_s20.jsonl",
                lambda rows: rows[0].__setitem__("sft_checkpoint_sha256", "f" * 64),
                "SFT checkpoint",
            ),
            "lr": (
                "mazescore_unif_s20.jsonl",
                lambda rows: rows[0].__setitem__("lr", 2e-4),
                "lr",
            ),
            "hindsight": (
                "mazescore_unif_s20.jsonl",
                lambda rows: rows[0].__setitem__("hindsight", True),
                "hindsight",
            ),
            "seed_mapping": (
                "mazescore_un_s20.jsonl",
                lambda rows: rows[0]["seeds"].__setitem__("eval_tasks", 12345),
                "seeds.eval_tasks",
            ),
        }
        for name, (filename, mutation, message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = CampaignFixture(root)
                match = analyzer._FILENAME_RE.fullmatch(filename)
                assert match is not None
                fixture.write_cell(
                    root / filename, match.group(1), int(match.group(2)), mutation
                )
                with self.assertRaisesRegex(analyzer.AnalysisError, message):
                    analyzer.analyze_paths(fixture.paths)

    def test_holm_adjustment(self) -> None:
        adjusted = analyzer.holm_adjust({"larger": 0.04, "smaller": 0.01})
        self.assertEqual(adjusted, {"larger": 0.04, "smaller": 0.02})


if __name__ == "__main__":
    unittest.main()
