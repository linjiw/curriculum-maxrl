import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BlackwellProbeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "manifest.json").read_text())

    def test_patch_digest_and_scope(self):
        patch = ROOT / "minimax-jax062.patch"
        self.assertEqual(
            sha256(patch), self.manifest["modernization_patch"]["patch_sha256"]
        )
        text = patch.read_text()
        touched = re.findall(r"^\+\+\+ b/(.+)$", text, flags=re.MULTILINE)
        self.assertEqual(touched, ["src/minimax/envs/environment.py"])
        self.assertEqual(text.count("+\t\t\tstate = jax.tree_util.tree_map("), 1)
        self.assertEqual(text.count("+\t\t\tobs = jax.tree_util.tree_map("), 1)

    def test_freeze_digest(self):
        self.assertEqual(
            sha256(ROOT / "environment.freeze.txt"),
            self.manifest["environment"]["freeze_sha256"],
        )

    def test_lane_is_not_evidence(self):
        self.assertEqual(self.manifest["scope"], "engineering_probe_no_training_no_evidence")
        self.assertFalse(self.manifest["claims"]["training_validated"])
        self.assertFalse(self.manifest["claims"]["benchmark_evidence"])


if __name__ == "__main__":
    unittest.main()
