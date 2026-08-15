"""Endpoint-blind contract tests for the DRAFT v4 terminal driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT/"ued_benchmark/scripts"
CPU_PYTHON = Path("/data/robotixx/ued_bench/envs/minimax-jax0431-cpu/bin/python")
SOURCE = Path(os.environ.get("TIE_AWARE_MINIMAX_SOURCE", "/nonexistent"))
PROTOCOL = ROOT/"ued_benchmark/analysis/development_protocol_v2_tie_aware_draft.json"
CONFIGS = {
	"frontier": ROOT/"ued_benchmark/configs/maze_frontier_exact_grouped_n8_tie_aware_v4.json",
	"maxmc": ROOT/"ued_benchmark/configs/maze_maxmc_group_matched_4x8_b500_tie_aware_v4.json",
}
sys.path.insert(0, str(SCRIPTS))

import run_matched_terminal_v4 as terminal  # noqa: E402


def _sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
	path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n", encoding="utf-8")


def _context(arm: str, training_driver_sha: str) -> dict[str, object]:
	return {
		"schema": 1,
		"protocol_id": terminal.PROTOCOL_ID,
		"purpose": terminal.PURPOSE,
		"run_id": f"engineering-{arm}-s101",
		"arm": arm,
		"training_seed": 101,
		"job_id": "local-test",
		"campaign_manifest_sha256": "c"*64,
		"provenance": {
			"base_commit": terminal.BASE_COMMIT,
			"base_tree": terminal.BASE_TREE,
			"overlay_contract_sha256": terminal.OVERLAY_CONTRACT_SHA256,
			"bundle_manifest_sha256": "b"*64,
			"overlay_manifest_sha256": "d"*64,
			"applied_overlay_manifest_sha256": _sha256(SOURCE/".frontierrl_overlay.json"),
			"environment_manifest_sha256": "e"*64,
			"training_driver_sha256": training_driver_sha,
			"evaluation_driver_sha256": _sha256(SCRIPTS/"evaluate_matched_terminal.py"),
			"sbatch_sha256": "f"*64,
		},
	}


class TieAwareTerminalContractTest(unittest.TestCase):
	def test_independent_reconstruction_equalizes_transformed_tie_mass(self):
		scores = np.asarray([9.0, 9.0, 9.0, 2.0, 100.0], dtype=np.float32)
		ages = np.asarray([0, 3, 7, 2, 999], dtype=np.uint32)
		filled = np.asarray([True, True, True, True, False])
		replay, score_dist, blocks = terminal.replay_distribution(
			scores,
			ages,
			filled,
			temperature=0.3,
			staleness_coef=0.3,
			use_score_ranks=True,
			tie_aware_score_ranks=True,
		)
		self.assertEqual(blocks, [3, 1])
		np.testing.assert_allclose(score_dist[:3], score_dist[0], rtol=0, atol=0)
		self.assertEqual(float(score_dist[4]), 0.0)
		self.assertEqual(float(replay[4]), 0.0)
		self.assertAlmostEqual(float(replay.sum()), 1.0, places=12)
		permutation = np.asarray([2, 4, 0, 3, 1])
		permuted_replay, permuted_score, permuted_blocks = terminal.replay_distribution(
			scores[permutation],
			ages[permutation],
			filled[permutation],
			temperature=0.3,
			staleness_coef=0.3,
			use_score_ranks=True,
			tie_aware_score_ranks=True,
		)
		self.assertEqual(permuted_blocks, blocks)
		np.testing.assert_array_equal(permuted_score, score_dist[permutation])
		np.testing.assert_array_equal(permuted_replay, replay[permutation])
		self.assertEqual(
			float(1.0/np.square(np.sort(permuted_score)).sum()),
			float(1.0/np.square(np.sort(score_dist)).sum()),
		)
		self.assertEqual(
			float(1.0/np.square(np.sort(permuted_replay)).sum()),
			float(1.0/np.square(np.sort(replay)).sum()),
		)

		stable_replay, stable_score, stable_blocks = terminal.replay_distribution(
			scores,
			ages,
			filled,
			temperature=0.3,
			staleness_coef=0.3,
			use_score_ranks=True,
			tie_aware_score_ranks=False,
		)
		self.assertEqual(stable_blocks, [3, 1])
		self.assertGreater(stable_score[0], stable_score[1])
		self.assertAlmostEqual(float(stable_replay.sum()), 1.0, places=12)

	def test_independent_reconstruction_rejects_nonfinite_filled_score(self):
		with self.assertRaisesRegex(terminal.DriverError, "non-finite"):
			terminal.replay_distribution(
				np.asarray([1.0, np.nan]),
				np.asarray([0, 1]),
				np.asarray([True, True]),
				temperature=0.3,
				staleness_coef=0.3,
				use_score_ranks=True,
				tie_aware_score_ranks=True,
			)

	def test_replay_integrity_checks_last_and_cumulative_accounting(self):
		valid = {
			"replay_integrity": {
				"tie_aware_score_ranks": True,
				"nonfinite_filled_score_count": 0,
				"nonfinite_score_rejection_count": 0,
				"replay_group_draw_count": 12,
				"replay_distinct_group_count": 8,
				"replay_duplicate_group_count": 4,
				"last_replay_group_count": 4,
				"last_replay_distinct_group_count": 3,
				"last_replay_duplicate_group_count": 1,
				"force_unique_resamples_replay": False,
				"sample_identity": "replay buffer slot index",
			}
		}
		terminal.validate_replay_integrity(valid)
		invalid = json.loads(json.dumps(valid))
		invalid["replay_integrity"]["replay_duplicate_group_count"] = 3
		with self.assertRaisesRegex(terminal.DriverError, "cumulative replay"):
			terminal.validate_replay_integrity(invalid)

	def test_draft_driver_rejects_production_before_input_access(self):
		args = argparse.Namespace(
			expected_driver_sha256=_sha256(SCRIPTS/"run_matched_terminal_v4.py"),
			arm="frontier",
			engineering_test_mode=False,
			slurm_engineering_test_mode=False,
		)
		with self.assertRaisesRegex(terminal.DriverError, "forbids matched-development"):
			terminal.run(args)

	def test_both_arms_require_common_replay_snapshot_sidecar(self):
		for arm in terminal.ARMS:
			with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
				root = Path(directory)
				output = root/f"{arm}-sidecar"
				receipt = {"run_id": f"engineering-{arm}", "arm": arm}
				snapshot = {
					"schema": 1,
					"arm": arm,
					"kind": "tie_aware_plr_buffer_safe_snapshot",
				}
				terminal.write_training_sidecar(output, receipt, snapshot)
				self.assertTrue((output/"plr-replay-snapshot.json").is_file())
				terminal.validate_training_sidecar(output, receipt["run_id"], arm)


@unittest.skipUnless(
	SOURCE.is_dir() and CPU_PYTHON.is_file(),
	"set TIE_AWARE_MINIMAX_SOURCE to a v4 applied clone for bounded runtime tests",
)
class TieAwareTerminalBoundedE2E(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.temporary = tempfile.TemporaryDirectory()
		cls.root = Path(cls.temporary.name)
		for name in ("tmp", "cache", "jax-cache"):
			(cls.root/name).mkdir()
		cls.environment = os.environ.copy()
		cls.environment.update({
			"JAX_PLATFORMS": "cpu",
			"PYTHONNOUSERSITE": "1",
			"PYTHONPATH": "",
			"TMPDIR": str(cls.root/"tmp"),
			"XDG_CACHE_HOME": str(cls.root/"cache"),
			"JAX_COMPILATION_CACHE_DIR": str(cls.root/"jax-cache"),
			"XLA_PYTHON_CLIENT_PREALLOCATE": "false",
		})
		cls.driver_sha = _sha256(SCRIPTS/"run_matched_terminal_v4.py")
		cls.cells: dict[str, dict[str, Path]] = {}
		overrides = [
			"n_total_updates=1",
			"test_interval=0",
			"log_interval=1",
			"train_runner_args.buffer_size=8",
			"train_runner_args.replay_prob=1.0",
			"train_runner_args.min_fill_ratio=0.5",
			"train_runner_args.n_rollout_steps=2",
			"train_runner_args.n_unroll_rollout=1",
			"env_args.max_episode_steps=2",
			"student_rl_args.n_unroll_update=1",
			"student_rl_args.n_epochs=1",
			"student_model_args.hidden_dim=16",
			"student_model_args.recurrent_hidden_dim=16",
			"student_model_args.n_conv_filters=4",
			"driver.max_outer_cycles=4",
		]
		for arm in terminal.ARMS:
			context_path = cls.root/f"{arm}-context.json"
			_write_json(context_path, _context(arm, cls.driver_sha))
			output = cls.root/f"engineering-{arm}-s101"
			sidecar = cls.root/f"engineering-{arm}-s101-driver-sidecar"
			command = [
				str(CPU_PYTHON),
				str(SCRIPTS/"run_matched_terminal_v4.py"),
				"--arm", arm,
				"--config", str(CONFIGS[arm]),
				"--protocol", str(PROTOCOL),
				"--run-context", str(context_path),
				"--expected-run-context-sha256", _sha256(context_path),
				"--expected-driver-sha256", cls.driver_sha,
				"--patched-source-dir", str(SOURCE),
				"--output-dir", str(output),
				"--sidecar-dir", str(sidecar),
				"--engineering-test-mode",
			]
			for override in overrides:
				command.extend(("--engineering-override", override))
			completed = subprocess.run(
				command,
				cwd=ROOT,
				env=cls.environment,
				text=True,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				timeout=300,
			)
			if completed.returncode:
				raise AssertionError(
					f"bounded {arm} driver failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
				)
			cls.cells[arm] = {"output": output, "sidecar": sidecar}

	@classmethod
	def tearDownClass(cls) -> None:
		cls.temporary.cleanup()

	def test_both_arms_emit_fresh_common_replay_snapshot(self) -> None:
		for arm, paths in self.cells.items():
			with self.subTest(arm=arm):
				endpoint = json.loads((paths["output"]/"endpoint.json").read_text())
				receipt = json.loads((paths["sidecar"]/"training-receipt.json").read_text())
				snapshot = json.loads((paths["sidecar"]/"plr-replay-snapshot.json").read_text())
				self.assertFalse(endpoint["paper_evidence"])
				self.assertEqual(endpoint["replay_integrity"], receipt["integrity"]["terminal"]["replay_integrity"])
				self.assertEqual(endpoint["replay_integrity"], receipt["integrity"]["checkpoint_round_trip"]["replay_integrity"])
				self.assertTrue(endpoint["replay_integrity"]["tie_aware_score_ranks"])
				self.assertEqual(snapshot["arm"], arm)
				self.assertEqual(snapshot["checkpoint_sha256"], receipt["terminal_checkpoint"]["sha256"])
				self.assertTrue(snapshot["replay_distribution"]["tie_aware_score_ranks"])
				self.assertEqual(
					snapshot["replay_distribution"]["score_normalization_order"],
					"canonical_ascending_unnormalized_mass",
				)
				self.assertEqual(
					snapshot["replay_distribution"][
						"distinct_score_stable_equivalence_float32_abs_tolerance"
					],
					terminal.FLOAT32_DISTINCT_SCORE_EQUIVALENCE_TOLERANCE,
				)
				self.assertTrue(np.isfinite(snapshot["replay_distribution"]["score_effective_support"]))
				self.assertTrue(np.isfinite(snapshot["replay_distribution"]["replay_effective_support"]))
				self.assertEqual(
					snapshot["sampling_diagnostics"]["replay_group_draw_count"],
					snapshot["sampling_diagnostics"]["replay_distinct_group_count"]
					+snapshot["sampling_diagnostics"]["replay_duplicate_group_count"],
				)


if __name__ == "__main__":
	unittest.main()
