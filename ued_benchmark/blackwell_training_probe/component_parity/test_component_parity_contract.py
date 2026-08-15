import ast
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from .compare_component_parity import (
    first_step_adam_diagnostic,
    first_step_adam_proposal,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_SHA256 = "0f8c083202a189ec234f32c0e1c15e7c09753892fb05af0d6262b9ff0bf9f1a5"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ComponentParityContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ROOT / "COMPONENT_PARITY_PROTOCOL.json").read_text())
        cls.manifest = json.loads((ROOT / "manifest.json").read_text())
        cls.capture_source = (ROOT / "capture_component_parity.py").read_text()
        cls.capture_tree = ast.parse(cls.capture_source)

    def test_protocol_digest_and_zero_update_budget(self):
        self.assertEqual(sha256(ROOT / "COMPONENT_PARITY_PROTOCOL.json"), PROTOCOL_SHA256)
        limits = self.protocol["execution_limits"]
        self.assertEqual(limits["optimizer_applications"], 0)
        self.assertEqual(limits["parameter_mutations"], 0)
        self.assertEqual(limits["cycle_two_agent_update_calls"], 0)
        self.assertEqual(limits["cycle_two_experiment_step_calls"], 0)
        self.assertEqual(limits["gpu_component_captures"], 1)
        self.assertEqual(limits["gpu_ppo_updates"], 0)
        self.assertFalse(limits["permit_optax_apply_updates"])
        self.assertFalse(limits["permit_train_state_apply_gradients"])
        self.assertFalse(self.protocol["paper_evidence"])
        self.assertFalse(self.protocol["performance_endpoint"])

    def test_stage_order_is_complete(self):
        self.assertEqual(
            [item["name"] for item in self.protocol["ordered_stages"]],
            [
                "initial_state",
                "cycle_one_control",
                "task_stream",
                "rollout_observation_stream",
                "rollout_action_stream",
                "rollout_forward_stream",
                "rollout_return_batch",
                "minibatch_stream",
                "ppo_forward",
                "ppo_loss_elements",
                "ppo_loss_terms",
                "unclipped_gradients",
                "clipping_and_global_norm",
                "adam_proposal",
            ],
        )

    def test_prior_tolerances_are_not_relaxed(self):
        self.assertEqual(
            self.protocol["comparison"]["cpu"],
            {"rtol": 0.000001, "atol": 0.0000001},
        )
        self.assertEqual(
            self.protocol["comparison"]["gpu"],
            {"rtol": 0.0005, "atol": 0.00005},
        )

    def test_capture_has_one_warmup_step_and_no_update_application(self):
        calls = [node for node in ast.walk(self.capture_tree) if isinstance(node, ast.Call)]
        attributes = [
            node.func.attr
            for node in calls
            if isinstance(node.func, ast.Attribute)
        ]
        experiment_steps = [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "experiment"
        ]
        self.assertEqual(len(experiment_steps), 1)
        for forbidden in (
            "apply_gradients",
            "apply_updates",
            "_efficient_grad_update",
        ):
            self.assertNotIn(forbidden, attributes)
        self.assertNotIn("student_pop.update", self.capture_source)
        self.assertNotIn("agent.update", self.capture_source)
        self.assertIn("single_train_state.tx.update", self.capture_source)

    def test_capture_imports_jax_only_after_runtime_gate(self):
        capture_function = next(
            node
            for node in self.capture_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "capture"
        )
        statements = capture_function.body
        validate_index = next(
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
        self.assertLess(validate_index, import_index)

    def test_manifest_fails_closed_at_forward_carry(self):
        self.assertEqual(self.manifest["protocol"]["sha256"], PROTOCOL_SHA256)
        primary = self.manifest["comparisons"]["modern_cpu_vs_rtx5090"]
        self.assertEqual(primary["status"], "fail_closed")
        self.assertEqual(primary["earliest_failing_stage"], "cycle_one_control")
        self.assertEqual(primary["classification"], "forward_or_gemm_recurrent_carry")
        self.assertEqual(self.manifest["execution"]["optimizer_applications"], 0)
        self.assertEqual(self.manifest["execution"]["gpu_component_captures"], 1)
        self.assertEqual(self.manifest["execution"]["gpu_ppo_updates"], 0)
        self.assertTrue(self.manifest["diagnosis"]["task_stream_exact"])
        self.assertTrue(self.manifest["diagnosis"]["action_stream_exact"])
        self.assertTrue(self.manifest["diagnosis"]["adam_formula_matches_proposals"])
        self.assertFalse(self.manifest["claims"]["gpu_training_gate_open"])

    def test_manifest_script_hashes(self):
        scripts = self.manifest["scripts"]
        self.assertEqual(
            sha256(ROOT / "capture_component_parity.py"), scripts["capture_sha256"]
        )
        self.assertEqual(
            sha256(ROOT / "compare_component_parity.py"), scripts["compare_sha256"]
        )

    def test_adam_diagnostic_uses_captured_clipped_gradient(self):
        raw_gradient = np.asarray([3.0, 4.0], dtype=np.float32)
        clip_factor = np.asarray([0.1], dtype=np.float32)
        clipped_gradient = raw_gradient * clip_factor
        proposed_update = first_step_adam_proposal(np, clipped_gradient)

        diagnostic = first_step_adam_diagnostic(
            np,
            raw_gradient,
            clipped_gradient,
            proposed_update,
            clip_factor,
        )

        self.assertAlmostEqual(float(np.linalg.norm(raw_gradient)), 5.0)
        self.assertAlmostEqual(float(np.linalg.norm(clipped_gradient)), 0.5)
        self.assertTrue(diagnostic["captured_clipping_matches_raw_times_factor"])
        self.assertTrue(diagnostic["first_step_adam_formula_matches"])
        self.assertEqual(diagnostic["analytic_input"], "captured_clipped_gradient_tree")
        self.assertFalse(
            np.allclose(
                proposed_update,
                first_step_adam_proposal(np, raw_gradient),
                rtol=1e-6,
                atol=1e-9,
            )
        )


if __name__ == "__main__":
    unittest.main()
