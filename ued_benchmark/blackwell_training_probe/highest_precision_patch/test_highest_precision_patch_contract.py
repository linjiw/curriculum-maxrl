import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
BASE_SOURCE = Path(
    "/data/robotixx/ued_bench/src/minimax-frontier-blackwell-training-jax062-5868d346-d053054"
)
APPLIED_SOURCE = Path(
    "/data/robotixx/ued_bench/src/minimax-frontier-blackwell-highest-lstm-v1-d053054"
)
REPRO_SOURCE = Path(
    "/data/robotixx/ued_bench/src/minimax-frontier-blackwell-highest-lstm-repro-v2-d053054"
)
REFERENCE_RECEIPT = Path(
    "/data/robotixx/ued_bench/runs/blackwell_training_probe_b7c865/"
    "reference-jax0431-cpu-protocol-v6/receipt.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HighestPrecisionPatchContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "HIGHEST_PRECISION_ONE_UPDATE_PROTOCOL.json").read_text()
        )
        cls.contract = json.loads((ROOT / "PATCH_CONTRACT.json").read_text())

    def test_patch_and_protocol_are_content_addressed(self):
        self.assertEqual(
            sha256(ROOT / "HIGHEST_PRECISION_ONE_UPDATE_PROTOCOL.json"),
            "ba0b6fd30de472554d732308017cb8d3c28f7ddef0549631fc5fe907610ec4c3",
        )
        self.assertEqual(
            sha256(ROOT / "PATCH_CONTRACT.json"),
            "7d8744ff34d064bd324cdc3d92b972b8050f492ff580edc6e44870bbf4aa969e",
        )
        self.assertEqual(
            sha256(ROOT / "minimax-highest-lstm.patch"),
            "a16f4394af0d89289314ab4a11ea43d3334ecba36a22e3c86ed11633d15fb9db",
        )
        self.assertEqual(
            sha256(ROOT / "apply_highest_precision_patch.py"),
            "4fef0fdb4bee747b9794b06832db2ba87345e54e2d21fb1881536521104abd57",
        )

    def test_protocol_keeps_prior_tolerances_and_is_bounded(self):
        parent = json.loads(
            (REPO_ROOT / "ued_benchmark/blackwell_training_probe/PARITY_PROTOCOL.json").read_text()
        )
        self.assertEqual(self.protocol["tolerances"], parent["tolerances"])
        self.assertEqual(self.protocol["schedule"], parent["schedule"])
        self.assertFalse(self.protocol["paper_evidence"])
        self.assertFalse(self.protocol["performance_endpoint"])
        self.assertEqual(self.protocol["execution_budget"]["cpu_candidate_runs"], 1)
        self.assertEqual(self.protocol["execution_budget"]["gpu_candidate_runs"], 1)
        self.assertEqual(self.protocol["execution_budget"]["maximum_gpu_ppo_updates"], 1)
        self.assertTrue(self.protocol["execution_budget"]["gpu_requires_cpu_gate_pass"])
        self.assertTrue(self.protocol["decision"]["fail_closed"])
        self.assertTrue(self.protocol["decision"]["no_tolerance_relaxation_after_execution"])

    def test_patch_is_one_file_and_applies_cleanly_to_frozen_parent(self):
        self.assertEqual(list(self.contract["files"]), ["src/minimax/models/common.py"])
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(BASE_SOURCE),
                "apply",
                "--check",
                str(ROOT / "minimax-highest-lstm.patch"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        patch_text = (ROOT / "minimax-highest-lstm.patch").read_text()
        self.assertEqual(patch_text.count("--- a/"), 1)
        self.assertEqual(patch_text.count("+++ b/"), 1)
        self.assertEqual(sum(line.startswith("@@ ") for line in patch_text.splitlines()), 1)

    def test_applied_source_is_reproducible_and_scope_preserving(self):
        relative = Path("src/minimax/models/common.py")
        expected = self.contract["files"][str(relative)]
        self.assertEqual(sha256(BASE_SOURCE / relative), expected["source_sha256"])
        self.assertEqual(sha256(APPLIED_SOURCE / relative), expected["applied_sha256"])
        self.assertEqual(sha256(REPRO_SOURCE / relative), expected["applied_sha256"])
        self.assertEqual(
            sha256(APPLIED_SOURCE / ".blackwell_highest_lstm_overlay.json"),
            sha256(REPRO_SOURCE / ".blackwell_highest_lstm_overlay.json"),
        )
        text = (APPLIED_SOURCE / relative).read_text()
        self.assertEqual(text.count("with jax.default_matmul_precision('highest'):"), 1)
        expected_block = """\
\t\tif self.recurrent_arch == 'lstm':
\t\t\trnn_cell = nn.OptimizedLSTMCell(**rnn_kwargs) # defaults to orth init
\t\t\twith jax.default_matmul_precision('highest'):
\t\t\t\tnew_rnn_state, y = rnn_cell(rnn_state, x)
\t\telif self.recurrent_arch == 'gru':
\t\t\trnn_cell = nn.GRUCell(**rnn_kwargs)
\t\t\tnew_rnn_state, y = rnn_cell(rnn_state, x)
"""
        self.assertIn(expected_block, text)

    def test_runner_imports_no_accelerator_library_before_backend_checks(self):
        tree = ast.parse((ROOT / "run_highest_precision_one_update.py").read_text())
        imported_roots = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertNotIn("jax", imported_roots)
        self.assertNotIn("numpy", imported_roots)
        runner_text = (ROOT / "run_highest_precision_one_update.py").read_text()
        self.assertIn("base_cli.reference_receipt = None", runner_text)
        self.assertIn("maximum_gpu_ppo_updates", runner_text)
        self.assertNotIn("experiment.step", runner_text)

    def test_cycle_api_shim_preserves_optimizer_count_assertion(self):
        from types import SimpleNamespace
        from ued_benchmark.blackwell_training_probe.highest_precision_patch import (
            assert_cycle_compat,
        )

        calls = []

        def summary(_state):
            return {"n_grad_updates": 1, "optimizer_step_applications": 1}

        def assertion(value, *, cycle, expected_optimizer_step_applications):
            calls.append((value, cycle, expected_optimizer_step_applications))

        module = SimpleNamespace(_state_summary=summary, _assert_cycle=assertion)
        state = assert_cycle_compat.install(module)
        self.assertEqual(module._state_summary(None), {"n_grad_updates": 1})
        module._assert_cycle({"n_grad_updates": 1}, cycle=2)
        self.assertEqual(state["optimizer_step_observations"], [1])
        self.assertEqual(calls[0][1:], (2, 1))
        self.assertEqual(calls[0][0]["optimizer_step_applications"], 1)

        failing = SimpleNamespace(
            _state_summary=lambda _state: {
                "n_grad_updates": 1,
                "optimizer_step_applications": 2,
            },
            _assert_cycle=assertion,
        )
        assert_cycle_compat.install(failing)
        with self.assertRaisesRegex(AssertionError, "optimizer application count drift"):
            failing._state_summary(None)

    def test_cpu_recovery_is_read_only_with_respect_to_training(self):
        tree = ast.parse((ROOT / "recover_cpu_one_update.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(
            any(
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "step"
                for call in calls
            )
        )
        text = (ROOT / "recover_cpu_one_update.py").read_text()
        self.assertIn('"additional_cpu_updates_during_recovery": 0', text)
        self.assertIn('"gpu_attempted": False', text)
        self.assertIn('"project_gate": "HOLD"', text)

    def test_final_manifest_binds_recovered_artifacts_and_holds_gpu(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        run = Path(manifest["cpu_result"]["run_directory"])
        self.assertEqual(manifest["decision"]["cpu_numerical_parity"], "GO")
        self.assertEqual(manifest["decision"]["cpu_bounded_training_gate"], "INCOMPLETE")
        self.assertEqual(manifest["decision"]["rtx5090_bounded_training_gate"], "HOLD")
        self.assertFalse(manifest["gpu"]["attempted"])
        self.assertEqual(sha256(run / "receipt.json"), manifest["cpu_result"]["raw_receipt_sha256"])
        self.assertEqual(
            sha256(run / "aggregate-comparison-recovered.json"),
            manifest["cpu_result"]["aggregate_report_sha256"],
        )
        self.assertEqual(
            sha256(run / "read-only-recovery.json"),
            manifest["cpu_result"]["read_only_recovery_sha256"],
        )
        script_paths = {
            "cycle_api_shim_sha256": "assert_cycle_compat.py",
            "fixed_future_wrapper_sha256": "run_highest_precision_one_update.py",
            "aggregate_auditor_sha256": "audit_one_update_aggregates.py",
            "read_only_recovery_sha256": "recover_cpu_one_update.py",
        }
        for field, name in script_paths.items():
            self.assertEqual(sha256(ROOT / name), manifest["scripts"][field])

    def test_exhaustive_auditor_enumerates_every_aggregate_and_fails_closed(self):
        from ued_benchmark.blackwell_training_probe.highest_precision_patch import (
            audit_one_update_aggregates as auditor,
        )

        reference = json.loads(REFERENCE_RECEIPT.read_text())
        candidate = copy.deepcopy(reference)
        candidate.update(
            {
                "lane": "modern",
                "backend": "cpu",
                "paper_evidence": False,
                "ood_evaluation": False,
                "max_student_updates": 1,
                "actual_student_updates": 1,
            }
        )
        candidate["initial_checkpoint"]["source_sha256"] = self.protocol["reference"][
            "initial_checkpoint_sha256"
        ]
        report = auditor.compare_documents(
            reference, candidate, backend="cpu", protocol=self.protocol
        )
        self.assertEqual(report["summary"]["status"], "passed")
        self.assertEqual(report["summary"]["aggregate_gate_count"], 546)
        self.assertEqual(report["summary"]["initial_exact_leaf_hash_count"], 91)
        changed = copy.deepcopy(candidate)
        changed["numerical"]["final"]["leaves"][0]["abs_sum"] += 1.0
        failed = auditor.compare_documents(
            reference, changed, backend="cpu", protocol=self.protocol
        )
        self.assertEqual(failed["summary"]["status"], "failed")
        self.assertGreaterEqual(failed["summary"]["failure_count"], 1)
        self.assertIn("final.", failed["summary"]["first_failure"])


if __name__ == "__main__":
    unittest.main()
