import ast
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
FORWARD_ROOT = ROOT.parent
PROTOCOL_SHA256 = "0abdb46a7b56986756a31f3d4cc1793af20fc6ca53d2b397720386aab7f5b820"
PAYLOAD_SHA256 = "845a34ae40fb762e72b4c6ec569ef16ab6531b241eeaf6cecbc0523059f3bc78"
BASE_CAPTURE_SHA256 = "437e65d445b42d78430c7f84f2e2c4dfe8e2d31ad0973acf031f8831ae40d5a4"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrecisionProbeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ROOT / "PRECISION_PROTOCOL.json").read_text())
        cls.capture_source = (ROOT / "capture_precision_probe.py").read_text()
        cls.capture_tree = ast.parse(cls.capture_source)
        cls.compare_source = (ROOT / "compare_precision_probe.py").read_text()
        cls.manifest = json.loads((ROOT / "manifest.json").read_text())

    def test_frozen_protocol_payload_and_base_hashes(self):
        self.assertEqual(sha256(ROOT / "PRECISION_PROTOCOL.json"), PROTOCOL_SHA256)
        self.assertEqual(sha256(FORWARD_ROOT / "FORWARD_PAYLOAD.json"), PAYLOAD_SHA256)
        self.assertEqual(sha256(FORWARD_ROOT / "capture_forward_only.py"), BASE_CAPTURE_SHA256)
        self.assertEqual(self.protocol["payload"]["sha256"], PAYLOAD_SHA256)
        self.assertEqual(
            self.protocol["parent_forward_capture_script_sha256"], BASE_CAPTURE_SHA256
        )

    def test_zero_execution_budget_and_single_gpu_capture(self):
        limits = self.protocol["execution_limits"]
        for field in (
            "training_steps",
            "experiment_step_calls",
            "agent_update_calls",
            "gradient_calculations",
            "gradient_transformation_proposals",
            "optimizer_applications",
            "parameter_mutations",
            "gpu_ppo_updates",
            "ood_evaluations",
            "throughput_measurements",
            "performance_endpoints",
            "paper_evidence_endpoints",
        ):
            self.assertEqual(limits[field], 0)
        self.assertEqual(limits["gpu_forward_only_captures"], 1)
        self.assertEqual(limits["seeds"], 1)

    def test_precision_change_is_lstm_dot_only(self):
        intervention = self.protocol["precision_intervention"]
        self.assertEqual(
            intervention["highest"],
            "jnp.dot with precision=jax.lax.Precision.HIGHEST",
        )
        self.assertFalse(intervention["convolution_precision_changed"])
        self.assertFalse(intervention["parameters_changed"])
        self.assertFalse(intervention["inputs_changed"])
        self.assertIn("jax.lax.Precision.HIGHEST", self.capture_source)
        self.assertGreaterEqual(self.capture_source.count("precision=precision"), 2)

    def test_tolerances_are_not_relaxed(self):
        comparison = self.protocol["comparison"]
        self.assertEqual(comparison["cpu"], {"rtol": 0.000001, "atol": 0.0000001})
        self.assertEqual(comparison["gpu"], {"rtol": 0.0005, "atol": 0.00005})
        self.assertFalse(comparison["tolerance_relaxed"])

    def test_capture_imports_jax_only_after_environment_gate(self):
        capture_function = next(
            node
            for node in self.capture_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "capture"
        )
        statements = capture_function.body
        gate_index = next(
            index
            for index, node in enumerate(statements)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "validate_environment"
        )
        import_index = next(
            index
            for index, node in enumerate(statements)
            if isinstance(node, ast.Import)
            and any(alias.name == "jax" for alias in node.names)
        )
        self.assertLess(gate_index, import_index)

    def test_capture_contains_no_training_gradient_or_optimizer_api(self):
        for forbidden in (
            "experiment.step",
            "agent.update",
            "apply_gradients",
            "optax.apply_updates",
            "value_and_grad",
            "grad_fn",
            "tx.update",
            "import optax",
        ):
            self.assertNotIn(forbidden, self.capture_source)

    def test_closure_never_opens_training_gate(self):
        closure = self.protocol["closure_gate"]
        self.assertFalse(closure["training_gate_opened_by_this_protocol"])
        self.assertIn('"training_gate_open": False', self.compare_source)
        self.assertNotIn('"training_gate_open": True', self.compare_source)
        self.assertIn("future_training_gate_may_be_reconsidered", self.compare_source)

    def test_manifest_records_precision_closure_but_not_training(self):
        self.assertTrue(self.manifest["closure"]["closure_pass"])
        self.assertTrue(
            self.manifest["closure"]["future_training_gate_may_be_reconsidered"]
        )
        self.assertFalse(self.manifest["closure"]["training_gate_open"])
        self.assertEqual(
            self.manifest["cross_backend_results"]["default"]["final_carry"]["status"],
            "fail",
        )
        self.assertEqual(
            self.manifest["cross_backend_results"]["highest"]["final_carry"]["status"],
            "pass",
        )
        self.assertEqual(self.manifest["execution"]["training_steps"], 0)
        self.assertEqual(self.manifest["execution"]["optimizer_applications"], 0)

    def test_manifest_script_hashes(self):
        scripts = self.manifest["scripts"]
        for name in ("capture", "compare", "test"):
            self.assertEqual(sha256(ROOT / scripts[name]), scripts[f"{name}_sha256"])


if __name__ == "__main__":
    unittest.main()
