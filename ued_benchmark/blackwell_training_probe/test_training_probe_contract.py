import ast
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BlackwellTrainingProbeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((ROOT / "MODERNIZATION_CONTRACT.json").read_text())
        cls.protocol = json.loads((ROOT / "PARITY_PROTOCOL.json").read_text())
        cls.manifest = json.loads((ROOT / "manifest.json").read_text())

    def test_modernization_is_mechanical_and_complete(self):
        self.assertEqual(self.contract["removed_api"], "jax.tree_map")
        self.assertEqual(self.contract["replacement_api"], "jax.tree_util.tree_map")
        self.assertEqual(self.contract["total_replacements"], 35)
        self.assertEqual(len(self.contract["files"]), 10)
        self.assertEqual(
            sum(item["replacements"] for item in self.contract["files"].values()),
            35,
        )
        self.assertFalse(self.contract["paper_evidence"])
        self.assertEqual(
            sha256(ROOT / "MODERNIZATION_CONTRACT.json"),
            self.manifest["modernization"]["contract_sha256"],
        )

    def test_protocol_is_bounded_and_frozen(self):
        self.assertFalse(self.protocol["paper_evidence"])
        self.assertFalse(self.protocol["source_era_prng"]["jax_threefry_partitionable"])
        self.assertEqual(self.protocol["schedule"]["student_updates"], 1)
        self.assertEqual(self.protocol["schedule"]["outer_cycles"], 2)
        self.assertEqual(self.protocol["schedule"]["n_parallel"], 4)
        self.assertEqual(self.protocol["schedule"]["n_eval"], 8)
        self.assertEqual(self.protocol["schedule"]["frontier_n_rollouts"], 8)
        self.assertEqual(self.protocol["schedule"]["total_transitions"], 128)
        self.assertEqual(
            sha256(ROOT / "PARITY_PROTOCOL.json"),
            self.manifest["protocol"]["sha256"],
        )

    def test_manifest_fails_gpu_gate_closed(self):
        self.assertEqual(
            self.manifest["scope"],
            "engineering_cpu_one_update_compatible_gpu_parity_failed_no_evidence",
        )
        self.assertEqual(self.manifest["gpu_probe"]["updates_executed"], 1)
        self.assertEqual(self.manifest["gpu_probe"]["parity"]["failure_count"], 1)
        self.assertFalse(self.manifest["gpu_probe"]["gate_open"])
        self.assertFalse(self.manifest["claims"]["gpu_training_lane_ready"])
        self.assertFalse(self.manifest["claims"]["benchmark_evidence"])
        self.assertFalse(self.manifest["claims"]["paper_evidence"])

    def test_recovery_is_read_only_with_respect_to_training(self):
        source = (ROOT / "recover_gpu_checkpoint.py").read_text()
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(
            any(
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "step"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "experiment"
                for call in calls
            )
        )
        self.assertEqual(
            sha256(ROOT / "recover_gpu_checkpoint.py"),
            self.manifest["scripts"]["recovery_sha256"],
        )

    def test_script_digests_are_frozen(self):
        expected = self.manifest["scripts"]
        self.assertEqual(
            sha256(ROOT / "apply_blackwell_training_overlay.py"),
            expected["applicator_sha256"],
        )
        self.assertEqual(
            sha256(ROOT / "run_parity_one_update.py"),
            expected["parity_runner_sha256"],
        )
        self.assertEqual(
            sha256(ROOT / "run_upstream_tests.py"),
            expected["upstream_test_runner_sha256"],
        )
        self.assertEqual(
            sha256(ROOT / "pytest-target.freeze.txt"),
            self.manifest["environment"]["pytest_target_freeze_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
