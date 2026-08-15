"""Bounded, endpoint-blind tests for the matched terminal UED drivers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "ued_benchmark" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_matched_terminal as evaluator  # noqa: E402
import run_matched_terminal as terminal  # noqa: E402


SOURCE = Path(
    "/data/robotixx/ued_bench/src/minimax-frontier-v3-final-5868d346-d053054"
)
CPU_PYTHON = Path("/data/robotixx/ued_bench/envs/minimax-jax0431-cpu/bin/python")
PROTOCOL = ROOT / "ued_benchmark" / "analysis" / "development_protocol_v1.json"
CONFIGS = {
    "frontier": ROOT / "ued_benchmark" / "configs" / "maze_frontier_exact_grouped_n8.json",
    "maxmc": ROOT / "ued_benchmark" / "configs" / "maze_maxmc_group_matched_4x8_b500.json",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _context(arm: str, training_driver_sha: str, evaluation_driver_sha: str) -> dict:
    return {
        "schema": 1,
        "protocol_id": terminal.PROTOCOL_ID,
        "purpose": terminal.PURPOSE,
        "run_id": f"engineering-{arm}-s101",
        "arm": arm,
        "training_seed": 101,
        "job_id": "local-test",
        "campaign_manifest_sha256": "c" * 64,
        "provenance": {
            "base_commit": terminal.BASE_COMMIT,
            "base_tree": terminal.BASE_TREE,
            "overlay_contract_sha256": terminal.OVERLAY_CONTRACT_SHA256,
            "bundle_manifest_sha256": "b" * 64,
            "overlay_manifest_sha256": "d" * 64,
            "applied_overlay_manifest_sha256": _sha(SOURCE / ".frontierrl_overlay.json"),
            "environment_manifest_sha256": "e" * 64,
            "training_driver_sha256": training_driver_sha,
            "evaluation_driver_sha256": evaluation_driver_sha,
            "sbatch_sha256": "f" * 64,
        },
    }


def _engineering_campaign(context: dict) -> dict:
    provenance = dict(context["provenance"])
    provenance["assembler_driver_sha256"] = _sha(
        SCRIPTS / "assemble_matched_run.py"
    )
    return {
        "schema": 1,
        "protocol_id": terminal.PROTOCOL_ID,
        "purpose": terminal.PURPOSE,
        "created_utc": "2026-08-14T10:00:00Z",
        "frozen_before_endpoint_access": True,
        "protocol_sha256": _sha(PROTOCOL),
        "analyzer_sha256": _sha(
            ROOT / "ued_benchmark" / "analysis" / "preregistered_dev_analysis.py"
        ),
        "provenance": provenance,
        "hardware": {
            "partition": "gpuq",
            "gpu_model": "NVIDIA A100",
            "gpu_profile": "1g.10gb",
            "gpu_count": 1,
            "n_devices": 1,
        },
        "submissions": [
            {
                "arm": context["arm"],
                "training_seed": context["training_seed"],
                "evaluation_seed": 100000 + context["training_seed"],
                "run_id": context["run_id"],
                "job_id": context["job_id"],
                "attempt": 1,
            }
        ],
    }


class MatchedTerminalContractTests(unittest.TestCase):
    def test_slurm_campaign_binding_is_exact_and_pre_endpoint(self) -> None:
        training_sha = _sha(SCRIPTS / "run_matched_terminal.py")
        evaluation_sha = _sha(SCRIPTS / "evaluate_matched_terminal.py")
        context = _context("frontier", training_sha, evaluation_sha)
        context["job_id"] = "12345"
        context["run_id"] = "engineering-slurm-12345-frontier-s101"
        protocol, protocol_sha = terminal.load_protocol(PROTOCOL)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign_path = root / "campaign.json"
            _write_json(campaign_path, _engineering_campaign(context))
            campaign_sha = _sha(campaign_path)
            context["campaign_manifest_sha256"] = campaign_sha
            validated = terminal.validate_campaign_binding(
                campaign_path,
                campaign_sha,
                context=context,
                protocol=protocol,
                protocol_sha256=protocol_sha,
                engineering_test_mode=False,
                slurm_engineering_test_mode=True,
            )
            self.assertEqual(validated["analyzer_sha256"], _sha(
                ROOT / "ued_benchmark" / "analysis" / "preregistered_dev_analysis.py"
            ))

            drifted = _engineering_campaign(context)
            drifted["analyzer_sha256"] = "0" * 64
            drifted_path = root / "drifted-campaign.json"
            _write_json(drifted_path, drifted)
            context["campaign_manifest_sha256"] = _sha(drifted_path)
            with self.assertRaisesRegex(terminal.DriverError, "analyzer hash drift"):
                terminal.validate_campaign_binding(
                    drifted_path,
                    _sha(drifted_path),
                    context=context,
                    protocol=protocol,
                    protocol_sha256=protocol_sha,
                    engineering_test_mode=False,
                    slurm_engineering_test_mode=True,
                )

    def test_campaign_binding_is_required_outside_local_fixtures(self) -> None:
        context = _context(
            "frontier",
            _sha(SCRIPTS / "run_matched_terminal.py"),
            _sha(SCRIPTS / "evaluate_matched_terminal.py"),
        )
        protocol, protocol_sha = terminal.load_protocol(PROTOCOL)
        with self.assertRaisesRegex(terminal.DriverError, "required under Slurm/production"):
            terminal.validate_campaign_binding(
                None,
                None,
                context=context,
                protocol=protocol,
                protocol_sha256=protocol_sha,
                engineering_test_mode=False,
                slurm_engineering_test_mode=True,
            )

    def test_json_and_campaign_paths_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"schema": 1, "schema": 1}\n', encoding="utf-8")
            with self.assertRaisesRegex(terminal.DriverError, "duplicate JSON key"):
                terminal.load_json(duplicate, "duplicate fixture")

            target = root / "campaign.json"
            target.write_text("{}\n", encoding="utf-8")
            link = root / "campaign-link.json"
            link.symlink_to(target)
            context = _context(
                "frontier",
                _sha(SCRIPTS / "run_matched_terminal.py"),
                _sha(SCRIPTS / "evaluate_matched_terminal.py"),
            )
            context["campaign_manifest_sha256"] = _sha(target)
            protocol, protocol_sha = terminal.load_protocol(PROTOCOL)
            with self.assertRaisesRegex(terminal.DriverError, "unsafe or missing file"):
                terminal.validate_campaign_binding(
                    link,
                    _sha(target),
                    context=context,
                    protocol=protocol,
                    protocol_sha256=protocol_sha,
                    engineering_test_mode=False,
                    slurm_engineering_test_mode=True,
                )

    def test_slurm_engineering_context_is_explicit_and_job_bound(self) -> None:
        training_sha = _sha(SCRIPTS / "run_matched_terminal.py")
        evaluation_sha = _sha(SCRIPTS / "evaluate_matched_terminal.py")
        context = _context("frontier", training_sha, evaluation_sha)
        context["job_id"] = "12345"
        context["run_id"] = "engineering-slurm-12345-frontier-s101"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.json"
            _write_json(path, context)
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}, clear=False):
                validated = terminal.validate_run_context(
                    path,
                    _sha(path),
                    arm="frontier",
                    engineering_test_mode=False,
                    slurm_engineering_test_mode=True,
                )
            self.assertEqual(validated["job_id"], "12345")
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "54321"}, clear=False):
                with self.assertRaisesRegex(terminal.DriverError, "job ID/context mismatch"):
                    terminal.validate_run_context(
                        path,
                        _sha(path),
                        arm="frontier",
                        engineering_test_mode=False,
                        slurm_engineering_test_mode=True,
                    )

            context["job_id"] = "12345_7"
            context["run_id"] = "engineering-slurm-12345_7-frontier-s101"
            _write_json(path, context)
            with mock.patch.dict(
                os.environ,
                {
                    "SLURM_JOB_ID": "12352",
                    "SLURM_ARRAY_JOB_ID": "12345",
                    "SLURM_ARRAY_TASK_ID": "7",
                },
                clear=False,
            ):
                validated = terminal.validate_run_context(
                    path,
                    _sha(path),
                    arm="frontier",
                    engineering_test_mode=False,
                    slurm_engineering_test_mode=True,
                )
            self.assertEqual(validated["job_id"], "12345_7")

    def test_nonlocal_source_validation_requires_explicit_pinned_git(self) -> None:
        context = _context(
            "frontier",
            _sha(SCRIPTS / "run_matched_terminal.py"),
            _sha(SCRIPTS / "evaluate_matched_terminal.py"),
        )
        with self.assertRaisesRegex(terminal.DriverError, "pinned environment Git is required"):
            terminal.validate_source(SOURCE, context, require_pinned_git=True)

    def test_level_identity_materializes_each_leaf_once_and_preserves_hash(self) -> None:
        class CountedLeaf:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __array__(self, dtype=None):
                self.calls += 1
                return np.asarray(self.value, dtype=dtype)

        leaves = [
            CountedLeaf(np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3)),
            CountedLeaf(np.arange(4, dtype=np.int32).reshape(1, 2, 2)),
        ]
        structure = "PyTreeDef({'a': *, 'b': *})"

        class TreeUtil:
            @staticmethod
            def tree_flatten(_levels):
                return leaves, structure

        class FakeJax:
            tree_util = TreeUtil()

        materialized_structure, host_leaves = terminal._materialize_level_identity_source(
            object(), FakeJax()
        )
        first = terminal._level_identity(materialized_structure, host_leaves, 0, 0)
        second = terminal._level_identity(materialized_structure, host_leaves, 0, 1)
        self.assertNotEqual(first, second)
        self.assertEqual([leaf.calls for leaf in leaves], [1, 1])

        digest = hashlib.sha256()
        digest.update(structure.encode("utf-8"))
        for index, leaf in enumerate(host_leaves):
            array = np.ascontiguousarray(leaf[0, 0])
            descriptor = {
                "index": index,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
            digest.update(
                json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            digest.update(b"\0")
            digest.update(array.tobytes(order="C"))
        self.assertEqual(first, digest.hexdigest())

    def test_full_500_slot_float32_probability_mass_regression(self) -> None:
        weights = np.linspace(1.0, 500.0, 500, dtype=np.float32)
        probabilities = (weights / weights.sum(dtype=np.float32)).astype(np.float32)
        total = terminal.validate_probability_mass(
            probabilities,
            label="synthetic 500-slot replay mass",
            tolerance=terminal.FLOAT32_DIAGNOSTIC_TOLERANCE,
        )
        self.assertLessEqual(
            abs(total - 1.0), terminal.FLOAT32_DIAGNOSTIC_TOLERANCE
        )
        with self.assertRaisesRegex(terminal.DriverError, "negative"):
            terminal.validate_probability_mass(
                np.asarray([1.1, -0.1], dtype=np.float32),
                label="negative fixture",
                tolerance=terminal.FLOAT32_DIAGNOSTIC_TOLERANCE,
            )
        with self.assertRaisesRegex(terminal.DriverError, "non-finite"):
            terminal.validate_probability_mass(
                np.asarray([np.nan, 1.0], dtype=np.float32),
                label="nonfinite fixture",
                tolerance=terminal.FLOAT32_DIAGNOSTIC_TOLERANCE,
            )

    def test_engineering_override_parser_is_typed_bounded_and_closed(self) -> None:
        parsed = terminal.parse_engineering_overrides(
            ["n_total_updates=1", "train_runner_args.replay_prob=1.0"]
        )
        self.assertEqual(
            parsed,
            {"n_total_updates": 1, "train_runner_args.replay_prob": 1.0},
        )
        with self.assertRaisesRegex(terminal.DriverError, "not allowed"):
            terminal.parse_engineering_overrides(["train_runner_args.n_eval=1"])
        with self.assertRaisesRegex(terminal.DriverError, "out of bounds"):
            terminal.parse_engineering_overrides(["n_total_updates=3"])
        with self.assertRaisesRegex(terminal.DriverError, "must be an integer"):
            terminal.parse_engineering_overrides(["n_total_updates=1.0"])
        with self.assertRaisesRegex(terminal.DriverError, "duplicate"):
            terminal.parse_engineering_overrides(["log_interval=1", "log_interval=2"])

    def test_evaluator_backend_contract_fails_closed(self) -> None:
        class Device:
            id = 0
            platform = "gpu"
            device_kind = "synthetic-gpu"

        class FakeJax:
            def __init__(self, backend: str, count: int = 1):
                self.backend = backend
                self.count = count

            def default_backend(self) -> str:
                return self.backend

            def devices(self, backend: str):
                self.requested = backend
                return [Device() for _ in range(self.count)]

        receipt = evaluator.validate_backend(FakeJax("gpu"), engineering_test_mode=False)
        self.assertEqual(receipt[0]["device_kind"], "synthetic-gpu")
        with self.assertRaisesRegex(evaluator.EvaluationError, "expected gpu"):
            evaluator.validate_backend(FakeJax("cpu"), engineering_test_mode=False)
        with self.assertRaisesRegex(evaluator.EvaluationError, "exactly one"):
            evaluator.validate_backend(FakeJax("gpu", count=2), engineering_test_mode=False)

    @unittest.skipUnless(SOURCE.is_dir(), "pinned applied source clone is unavailable")
    def test_source_closure_rejects_unpatched_worktree_edit(self) -> None:
        with tempfile.TemporaryDirectory(dir="/data/robotixx/ued_bench") as temporary:
            root = Path(temporary)
            clone = root / "source"
            subprocess.run(
                ["git", "clone", "--quiet", "--no-hardlinks", str(SOURCE), str(clone)],
                check=True,
            )
            overlay = json.loads((SOURCE / ".frontierrl_overlay.json").read_text(encoding="utf-8"))
            for relative in overlay["overlay_files"]:
                destination = clone / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(SOURCE / relative, destination)
            shutil.copy2(SOURCE / ".frontierrl_overlay.json", clone / ".frontierrl_overlay.json")
            context = _context("frontier", _sha(SCRIPTS / "run_matched_terminal.py"), _sha(SCRIPTS / "evaluate_matched_terminal.py"))
            receipt = terminal.validate_source(clone, context)
            self.assertEqual(receipt["overlay_file_count"], len(overlay["overlay_files"]))
            with (clone / "README.md").open("a", encoding="utf-8") as stream:
                stream.write("\nunbound-test-edit\n")
            with self.assertRaisesRegex(terminal.DriverError, "worktree closure drift"):
                terminal.validate_source(clone, context)


@unittest.skipUnless(
    SOURCE.is_dir() and CPU_PYTHON.is_file(),
    "bounded minimax CPU runtime is unavailable",
)
class MatchedTerminalBoundedE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory(dir="/data/robotixx/ued_bench")
        cls.root = Path(cls._temporary.name)
        for name in ("tmp", "cache", "jax-cache"):
            (cls.root / name).mkdir()
        cls.env = os.environ.copy()
        cls.env.update(
            {
                "JAX_PLATFORM_NAME": "cpu",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": "",
                "TMPDIR": str(cls.root / "tmp"),
                "XDG_CACHE_HOME": str(cls.root / "cache"),
                "JAX_COMPILATION_CACHE_DIR": str(cls.root / "jax-cache"),
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            }
        )
        cls.training_driver_sha = _sha(SCRIPTS / "run_matched_terminal.py")
        cls.evaluation_driver_sha = _sha(SCRIPTS / "evaluate_matched_terminal.py")
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
            context_path = cls.root / f"{arm}-context.json"
            _write_json(
                context_path,
                _context(arm, cls.training_driver_sha, cls.evaluation_driver_sha),
            )
            output = cls.root / f"engineering-{arm}-s101"
            sidecar = cls.root / f"engineering-{arm}-s101-driver-sidecar"
            command = [
                str(CPU_PYTHON),
                str(SCRIPTS / "run_matched_terminal.py"),
                "--arm",
                arm,
                "--config",
                str(CONFIGS[arm]),
                "--protocol",
                str(PROTOCOL),
                "--run-context",
                str(context_path),
                "--expected-run-context-sha256",
                _sha(context_path),
                "--expected-driver-sha256",
                cls.training_driver_sha,
                "--patched-source-dir",
                str(SOURCE),
                "--output-dir",
                str(output),
                "--sidecar-dir",
                str(sidecar),
                "--engineering-test-mode",
            ]
            for override in overrides:
                command.extend(["--engineering-override", override])
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=cls.env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
            )
            if completed.returncode != 0:
                raise AssertionError(
                    f"bounded {arm} driver failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            cls.cells[arm] = {
                "context": context_path,
                "output": output,
                "sidecar": sidecar,
            }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_both_arms_terminal_checkpoint_and_counter_reconciliation(self) -> None:
        for arm, paths in self.cells.items():
            with self.subTest(arm=arm):
                output = paths["output"]
                self.assertEqual(
                    {path.name for path in output.iterdir()},
                    {"checkpoint.pkl", "endpoint.json", "logs.csv", "meta.json"},
                )
                endpoint = json.loads((output / "endpoint.json").read_text(encoding="utf-8"))
                self.assertEqual(
                    set(endpoint),
                    {
                        "schema",
                        "status",
                        "run_id",
                        "arm",
                        "training_seed",
                        "n_updates",
                        "n_grad_updates",
                        "optimizer_step_applications",
                        "outer_cycles",
                        "student_training_transitions",
                        "checkpoint_file",
                        "checkpoint_sha256",
                        "terminal_checkpoint_saved_after_training",
                        "resumed",
                        "frontier_integrity",
                    },
                )
                receipt = json.loads(
                    (paths["sidecar"] / "training-receipt.json").read_text(encoding="utf-8")
                )
                self.assertEqual(endpoint["n_updates"], 1)
                self.assertEqual(endpoint["n_grad_updates"], 1)
                self.assertEqual(endpoint["optimizer_step_applications"], 1)
                self.assertEqual(endpoint["outer_cycles"], 2)
                self.assertEqual(endpoint["student_training_transitions"], 128)
                self.assertTrue(endpoint["terminal_checkpoint_saved_after_training"])
                self.assertFalse(endpoint["resumed"])
                self.assertEqual(
                    endpoint["checkpoint_sha256"], _sha(output / "checkpoint.pkl")
                )
                self.assertEqual(
                    receipt["integrity"]["terminal"]["n_updates"],
                    receipt["integrity"]["checkpoint_round_trip"]["n_updates"],
                )
                self.assertEqual(receipt["integrity"]["terminal"]["n_iters"], 2)
                self.assertEqual(receipt["integrity"]["terminal"]["n_grad_updates"], 1)
                self.assertEqual(receipt["student_training_transitions"], 2 * 4 * 8 * 2)
                self.assertEqual(receipt["periodic_evaluation_accounting"]["calls"], 0)
                self.assertEqual(
                    receipt["periodic_evaluation_accounting"][
                        "budgeted_max_transitions"
                    ],
                    0,
                )
                self.assertEqual(len(receipt["engineering_test"]["overrides"]), 15)
                terminal.validate_training_sidecar(
                    paths["sidecar"], f"engineering-{arm}-s101", arm
                )

    def test_frontier_snapshot_schema_and_pinned_distribution_agreement(self) -> None:
        paths = self.cells["frontier"]
        snapshot = json.loads(
            (paths["sidecar"] / "frontier-buffer-snapshot.json").read_text(encoding="utf-8")
        )
        self.assertEqual(snapshot["kind"], "frontier_plr_buffer_safe_snapshot")
        self.assertEqual(snapshot["filled_count"], len(snapshot["slots"]))
        self.assertEqual(snapshot["filled_count"], 4)
        self.assertAlmostEqual(
            sum(row["normalized_replay_probability"] for row in snapshot["slots"]),
            1.0,
            places=9,
        )
        replay_validation = snapshot["replay_distribution"]
        self.assertLessEqual(
            replay_validation["pinned_implementation_max_abs_error"],
            replay_validation["float32_validation_tolerance"],
        )
        stored_validation = snapshot["stored_score_validation"]
        self.assertLessEqual(
            stored_validation["max_abs_error"],
            stored_validation["float32_validation_tolerance"],
        )
        for row in snapshot["slots"]:
            self.assertEqual(len(row["level_sha256"]), 64)
            self.assertLessEqual(row["success_count"], row["trial_count"])
            self.assertGreaterEqual(
                row["mean_plugin_score"] + terminal.FLOAT32_DIAGNOSTIC_TOLERANCE,
                row["analytic_expected_activity_score"],
            )
        maxmc_names = {path.name for path in self.cells["maxmc"]["sidecar"].iterdir()}
        self.assertNotIn("frontier-buffer-snapshot.json", maxmc_names)

    def test_actual_external_evaluator_bounded_cpu_both_arms(self) -> None:
        for arm, paths in self.cells.items():
            with self.subTest(arm=arm):
                output = self.root / f"evaluation-actual-{arm}"
                command = [
                    str(CPU_PYTHON),
                    str(SCRIPTS / "evaluate_matched_terminal.py"),
                    "--arm",
                    arm,
                    "--protocol",
                    str(PROTOCOL),
                    "--run-context",
                    str(paths["context"]),
                    "--expected-run-context-sha256",
                    _sha(paths["context"]),
                    "--expected-driver-sha256",
                    self.evaluation_driver_sha,
                    "--patched-source-dir",
                    str(SOURCE),
                    "--checkpoint",
                    str(paths["output"] / "checkpoint.pkl"),
                    "--endpoint",
                    str(paths["output"] / "endpoint.json"),
                    "--training-receipt",
                    str(paths["sidecar"] / "training-receipt.json"),
                    "--meta",
                    str(paths["output"] / "meta.json"),
                    "--output-dir",
                    str(output),
                    "--engineering-test-mode",
                    "--engineering-verify-independent-aggregate",
                ]
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=self.env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=300,
                )
                if completed.returncode != 0:
                    self.fail(
                        f"actual bounded evaluator failed\nstdout:\n{completed.stdout}\n"
                        f"stderr:\n{completed.stderr}"
                    )
                evaluator.validate_package(output, f"engineering-{arm}-s101")
                receipt = json.loads(
                    (output / "evaluation-receipt.json").read_text(encoding="utf-8")
                )
                self.assertFalse(receipt["synthetic_test_mode"])
                self.assertEqual(receipt["raw_results"]["record_count"], 30)
                self.assertEqual(receipt["provenance"]["runtime"]["backend"], "cpu")
                self.assertEqual(receipt["provenance"]["runtime"]["device_count"], 1)
                parity = receipt["provenance"]["runtime"][
                    "raw_vs_independent_evalrunner"
                ]
                self.assertTrue(parity["all_six_fields_checked"])
                self.assertEqual(len(parity["per_field_abs_error"]), 6)
                self.assertLessEqual(parity["max_abs_error"], parity["float32_tolerance"])
                accounting = receipt["evaluation_transition_accounting"]
                self.assertEqual(
                    accounting["budgeted_primary_max_transitions"], 3 * 10 * 450
                )
                self.assertEqual(
                    accounting["effective_primary_transitions"], 3 * 10 * 450
                )
                self.assertEqual(
                    accounting["engineering_independent_verification_transitions"],
                    3 * 10 * 450,
                )
                self.assertEqual(accounting["total_runtime_transitions"], 2 * 3 * 10 * 450)
                self.assertEqual(
                    receipt["provenance"]["runtime"][
                        "per_environment_max_episode_horizons"
                    ],
                    [450, 450, 450],
                )

    def test_external_evaluator_is_deterministic_and_manifest_closed(self) -> None:
        paths = self.cells["frontier"]
        outputs = []
        for suffix in ("a", "b"):
            output = self.root / f"evaluation-{suffix}"
            command = [
                str(CPU_PYTHON),
                str(SCRIPTS / "evaluate_matched_terminal.py"),
                "--arm",
                "frontier",
                "--protocol",
                str(PROTOCOL),
                "--run-context",
                str(paths["context"]),
                "--expected-run-context-sha256",
                _sha(paths["context"]),
                "--expected-driver-sha256",
                self.evaluation_driver_sha,
                "--patched-source-dir",
                str(SOURCE),
                "--checkpoint",
                str(paths["output"] / "checkpoint.pkl"),
                "--endpoint",
                str(paths["output"] / "endpoint.json"),
                "--training-receipt",
                str(paths["sidecar"] / "training-receipt.json"),
                "--meta",
                str(paths["output"] / "meta.json"),
                "--output-dir",
                str(output),
                "--engineering-test-mode",
                "--synthetic-test-mode",
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=self.env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
            if completed.returncode != 0:
                self.fail(
                    f"synthetic evaluator failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            evaluator.validate_package(output, "engineering-frontier-s101")
            outputs.append(output)
        for name in set(evaluator.PAYLOADS) | {"SHA256SUMS", "COMPLETE"}:
            self.assertEqual((outputs[0] / name).read_bytes(), (outputs[1] / name).read_bytes())
        raw_rows = [
            json.loads(line)
            for line in (outputs[0] / "evaluation-episodes.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        self.assertEqual(len(raw_rows), 30)
        self.assertEqual(
            [(row["environment"], row["episode"]) for row in raw_rows],
            [
                (environment, episode)
                for environment in evaluator.ENVIRONMENTS
                for episode in range(evaluator.N_EPISODES)
            ],
        )
        receipt = json.loads(
            (outputs[0] / "evaluation-receipt.json").read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["evaluation_seed"], 100101)
        self.assertEqual(receipt["raw_results"]["record_count"], 30)
        self.assertEqual(
            receipt["evaluation_transition_accounting"][
                "effective_primary_transitions"
            ],
            0,
        )
        (outputs[0] / "extra.txt").write_text("closure violation\n", encoding="utf-8")
        with self.assertRaisesRegex(evaluator.EvaluationError, "closure drift"):
            evaluator.validate_package(outputs[0], "engineering-frontier-s101")

    def test_evaluator_rejects_resealed_training_protocol_drift(self) -> None:
        paths = self.cells["maxmc"]
        drifted_sidecar = self.root / "maxmc-drifted-training-sidecar"
        shutil.copytree(paths["sidecar"], drifted_sidecar)
        receipt_path = drifted_sidecar / "training-receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["provenance"]["protocol_sha256"] = "0" * 64
        _write_json(receipt_path, receipt)
        manifest_path = drifted_sidecar / "SHA256SUMS"
        manifest_path.write_text(
            f"{_sha(receipt_path)}  training-receipt.json\n", encoding="utf-8"
        )
        complete_path = drifted_sidecar / "COMPLETE"
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        complete["sha256sums_sha256"] = _sha(manifest_path)
        _write_json(complete_path, complete)

        context = json.loads(paths["context"].read_text(encoding="utf-8"))
        source_receipt = terminal.validate_source(SOURCE, context)
        with self.assertRaisesRegex(
            evaluator.EvaluationError, "training/evaluation protocol hash drift"
        ):
            evaluator._validate_training_inputs(
                paths["output"] / "checkpoint.pkl",
                paths["output"] / "endpoint.json",
                receipt_path,
                paths["output"] / "meta.json",
                context,
                _sha(paths["context"]),
                _sha(PROTOCOL),
                self.training_driver_sha,
                source_receipt,
                True,
                False,
            )


if __name__ == "__main__":
    unittest.main()
