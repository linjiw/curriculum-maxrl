"""CPU-only protocol checks for the MAZE-SCORE training entry point."""

from __future__ import annotations

import contextlib
import io
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from curriculum_maxrl.estimators import (
    coefficient_activity,
    legacy_frontier_activity,
)
from curriculum_maxrl.maze_gpu import train


class CoefficientActivityTest(unittest.TestCase):
    def test_n2_is_learnability(self):
        p = np.linspace(0.0, 1.0, 101)
        np.testing.assert_allclose(coefficient_activity(p, 2), p * (1.0 - p))

    def test_endpoints_scalar_and_vector(self):
        for n in (1, 2, 32):
            self.assertAlmostEqual(float(coefficient_activity(0.0, n)), 0.0)
            self.assertAlmostEqual(float(coefficient_activity(1.0, n)), 0.0)
        scalar = coefficient_activity(0.25, 8)
        vector = coefficient_activity(np.array([0.25, 0.5]), 8)
        self.assertEqual(np.ndim(scalar), 0)
        self.assertEqual(vector.shape, (2,))
        self.assertAlmostEqual(float(vector[0]), float(scalar))

    def test_legacy_is_exact_off_by_one(self):
        p = np.linspace(0.01, 0.99, 99)
        np.testing.assert_allclose(
            legacy_frontier_activity(p, 32), coefficient_activity(p, 33))
        self.assertFalse(np.allclose(
            legacy_frontier_activity(p, 32), coefficient_activity(p, 32)))

    def test_invalid_rollout_count(self):
        with self.assertRaises(ValueError):
            coefficient_activity(0.5, 0)
        with self.assertRaises(ValueError):
            legacy_frontier_activity(0.5, 0)


class ProtocolDispatchTest(unittest.TestCase):
    def test_exact_alias_dispatch(self):
        self.assertIs(train.TEACHERS["coefficient_activity"],
                      train.FrontierUNTeacher)
        self.assertIs(train.TEACHERS["frontier_un"], train.FrontierUNTeacher)
        self.assertIs(train.TEACHERS["frontier"], train.FrontierTeacher)
        self.assertIsNot(train.TEACHERS["frontier"],
                         train.TEACHERS["coefficient_activity"])
        self.assertEqual(train.score_metadata("frontier_un", 32),
                         ("coefficient_activity", 32))
        self.assertEqual(train.score_metadata("frontier", 32),
                         ("legacy_frontier_activity", 33))

    def test_v2_checkpoint_is_absolute_and_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shared.pt"
            self.assertEqual(
                train.resolve_sft_checkpoint("maze_score_v2", str(path), 7),
                path)
            with self.assertRaises(ValueError):
                train.resolve_sft_checkpoint("maze_score_v2", "shared.pt", 7)
            missing = Path(tmp) / "missing" / "shared.pt"
            with self.assertRaises(FileNotFoundError):
                train.resolve_sft_checkpoint(
                    "maze_score_v2", str(missing), 7)

    def test_prepare_sft_only_creates_and_hashes_exact_path_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "shared.pt"
            old_device = train.DEVICE
            old_argv = sys.argv
            output = io.StringIO()
            try:
                train.DEVICE = "cpu"
                sys.argv = [
                    "train.py", "--protocol", "maze_score_v2",
                    "--prepare-sft-only", "--sft-ckpt", str(checkpoint),
                    "--sft-steps", "0", "--d-model", "8",
                    "--n-layers", "1",
                ]
                with contextlib.redirect_stdout(output):
                    train.main()
            finally:
                train.DEVICE = old_device
                sys.argv = old_argv
            self.assertTrue(checkpoint.is_file())
            self.assertIn(train.sha256_file(checkpoint), output.getvalue())

    def test_legacy_checkpoint_resolution_is_unchanged(self):
        path = train.resolve_sft_checkpoint(
            "legacy_v1", "warm.pt", 7, script_dir="/tmp/maze")
        self.assertEqual(path, Path("/tmp/maze/seed7_warm.pt"))

    def test_exact_v2_eval_schedule(self):
        expected = tuple(range(25, 251, 25))
        self.assertEqual(train.maze_score_v2_eval_schedule(250, 25), expected)
        self.assertEqual(expected[-1], 250)
        self.assertNotIn(0, expected)
        with self.assertRaises(ValueError):
            train.maze_score_v2_eval_schedule(300, 25)
        with self.assertRaises(ValueError):
            train.maze_score_v2_eval_schedule(250, 50)

    def test_teacher_distribution_snapshot_does_not_advance_rng(self):
        inspected = train.FrontierUNTeacher(32, seed=91)
        control = train.FrontierUNTeacher(32, seed=91)
        dist = train.teacher_distribution_snapshot(inspected)
        self.assertAlmostEqual(float(dist.sum()), 1.0)
        np.testing.assert_array_equal(
            inspected.sample_levels(64), control.sample_levels(64))

    def test_per_level_task_stream_is_arm_order_independent(self):
        streams_a = train.make_level_task_rngs(73)
        streams_b = train.make_level_task_rngs(73)
        expected = [train.sample_task(4, streams_a[4]).prompt for _ in range(3)]
        for _ in range(20):
            train.sample_task(0, streams_b[0])
        observed = [train.sample_task(4, streams_b[4]).prompt for _ in range(3)]
        self.assertEqual(observed, expected)

    def test_rl_phase_reset_erases_create_load_rng_consumption(self):
        train.seed_global_rngs(19)
        for _ in range(17):
            random.random()
            np.random.random()
            torch.rand(3)
        train.seed_global_rngs(101)
        after_create = (
            random.random(), float(np.random.random()), torch.rand(4))

        train.seed_global_rngs(19)
        random.random()
        np.random.random()
        torch.rand(1)
        train.seed_global_rngs(101)
        after_load = (
            random.random(), float(np.random.random()), torch.rand(4))

        self.assertEqual(after_create[0], after_load[0])
        self.assertEqual(after_create[1], after_load[1])
        torch.testing.assert_close(after_create[2], after_load[2])

    def test_generate_accepts_dedicated_generator(self):
        torch.manual_seed(3)
        model = train.TinyTransformer(d_model=8, n_layers=1, n_heads=8).cpu()
        prompts = torch.tensor([[1, 2]], dtype=torch.long)
        lengths = torch.tensor([2], dtype=torch.long)
        gen_a = torch.Generator(device="cpu").manual_seed(44)
        gen_b = torch.Generator(device="cpu").manual_seed(44)
        global_state = torch.random.get_rng_state().clone()
        out_a = model.generate(prompts, lengths, 3, eos=-1, generator=gen_a)
        torch.testing.assert_close(torch.random.get_rng_state(), global_state)
        out_b = model.generate(prompts, lengths, 3, eos=-1, generator=gen_b)
        torch.testing.assert_close(out_a, out_b)


if __name__ == "__main__":
    unittest.main()
