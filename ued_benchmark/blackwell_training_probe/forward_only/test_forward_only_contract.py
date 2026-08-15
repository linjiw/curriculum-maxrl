import ast
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
PROTOCOL_SHA256 = "024239a6b659097198a6d902b1bb63698849d38e340ac033fa21537b0e5888ce"
PAYLOAD_SHA256 = "845a34ae40fb762e72b4c6ec569ef16ab6531b241eeaf6cecbc0523059f3bc78"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ForwardOnlyContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ROOT / "FORWARD_ONLY_PROTOCOL.json").read_text())
        cls.payload = json.loads((ROOT / "FORWARD_PAYLOAD.json").read_text())
        cls.capture_source = (ROOT / "capture_forward_only.py").read_text()
        cls.capture_tree = ast.parse(cls.capture_source)
        cls.compare_source = (ROOT / "compare_forward_only.py").read_text()
        cls.manifest = json.loads((ROOT / "manifest.json").read_text())

    def test_frozen_protocol_and_payload_hashes(self):
        self.assertEqual(sha256(ROOT / "FORWARD_ONLY_PROTOCOL.json"), PROTOCOL_SHA256)
        self.assertEqual(sha256(ROOT / "FORWARD_PAYLOAD.json"), PAYLOAD_SHA256)
        self.assertEqual(self.protocol["payload"]["sha256"], PAYLOAD_SHA256)
        self.assertFalse(self.protocol["paper_evidence"])
        self.assertFalse(self.protocol["performance_endpoint"])

    def test_zero_training_and_single_gpu_forward_budget(self):
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

    def test_payload_is_bound_to_exact_checkpoint_capture_and_records(self):
        source = self.payload["source_capture"]
        self.assertEqual(
            source["sha256"],
            "9aefe688d1630f97799220455fdbae32205874e0b8a0971860d3d9bca0ec6382",
        )
        self.assertEqual(
            self.payload["initial_checkpoint"]["sha256"],
            "4dd07bf02eeb7ec072e4ec72b3aa02180c3ae84284ba20b27174f3dfa9886187",
        )
        selectors = self.payload["record_selectors"]
        self.assertEqual(selectors["population_parameters"]["indices"], list(range(51, 74)))
        self.assertEqual(selectors["image"]["index"], 288)
        self.assertEqual(selectors["agent_dir"]["index"], 289)
        self.assertEqual(selectors["dones"]["index"], 292)
        self.assertEqual(selectors["carry_c"]["index"], 297)
        self.assertEqual(selectors["carry_h"]["index"], 298)
        self.assertEqual(selectors["loss_rng"]["index"], 283)

    def test_operation_order_localizes_features_then_lstm(self):
        self.assertEqual(
            [item["name"] for item in self.protocol["ordered_stages"]],
            [
                "input_payload",
                "convolution_preactivation",
                "convolution_activation",
                "visual_flatten",
                "scalar_embedding",
                "concatenated_features",
                "reset_selected_carry",
                "lstm_input_affine",
                "lstm_hidden_affine",
                "lstm_gate_preactivation",
                "lstm_gate_activation",
                "lstm_cell_terms",
                "lstm_cell_state",
                "lstm_hidden_state",
                "manual_final_carry",
                "canonical_model_output",
            ],
        )

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
        self.assertIn("jax.lax.conv_general_dilated", self.capture_source)
        self.assertIn("jax.lax.scan", self.capture_source)
        self.assertIn("lstm_gate_preactivation", self.capture_source)

    def test_comparator_always_keeps_training_gate_closed(self):
        self.assertIn('"training_gate_open": False', self.compare_source)
        self.assertNotIn('"training_gate_open": True', self.compare_source)
        self.assertIn("input_gemm_diagnostic", self.compare_source)
        self.assertIn("propagated_feature_delta", self.compare_source)

    def test_manifest_is_fail_closed_at_lstm_input_gemm(self):
        self.assertEqual(self.manifest["payload"]["sha256"], PAYLOAD_SHA256)
        self.assertEqual(self.manifest["protocol"]["sha256"], PROTOCOL_SHA256)
        primary = self.manifest["comparisons"]["modern_cpu_vs_rtx5090"]
        self.assertEqual(primary["status"], "fail_closed")
        self.assertEqual(primary["earliest_failing_stage"], "lstm_input_affine")
        self.assertEqual(primary["classification"], "lstm_input_gemm")
        self.assertTrue(
            self.manifest["read_only_gemm_counterfactual"][
                "all_gate_dominance_checks_pass"
            ]
        )
        self.assertEqual(self.manifest["execution"]["training_steps"], 0)
        self.assertEqual(self.manifest["execution"]["optimizer_applications"], 0)
        self.assertFalse(self.manifest["claims"]["gpu_training_gate_open"])

    def test_manifest_script_hashes(self):
        scripts = self.manifest["scripts"]
        for name in ("capture", "compare", "test"):
            self.assertEqual(sha256(ROOT / scripts[name]), scripts[f"{name}_sha256"])


if __name__ == "__main__":
    unittest.main()
