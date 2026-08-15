"""Bounded runtime receipt test for the v4 grouped one-update driver."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("TIE_AWARE_MINIMAX_SOURCE", "/nonexistent"))
CPU_PYTHON = Path("/data/robotixx/ued_bench/envs/minimax-jax0431-cpu/bin/python")
CONFIG = ROOT/"ued_benchmark/configs/maze_frontier_exact_grouped_n8_tie_aware_v4.json"
CONTRACT = ROOT/"ued_benchmark/OVERLAY_CONTRACT_V4.json"
DRIVER = ROOT/"ued_benchmark/scripts/run_grouped_one_update_v4.py"
CONFIG_SHA256 = "0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2"
CONTRACT_SHA256 = "3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b"
XPID = "eng1-ca-ovv4ch3d5f3827_N8ne8a1.0b1.0th0.0eastrict-rt-4p-b8-rp1-mf0.5-seed1"


def _provenance() -> dict[str, object]:
    hashes = {
        field: "a"*64
        for field in (
            "bundle_manifest_sha256",
            "upstream_git_bundle_sha256",
            "overlay_manifest_sha256",
            "applied_overlay_manifest_sha256",
            "sbatch_sha256",
            "environment_lock_sha256",
            "environment_freeze_sha256",
            "environment_manifest_sha256",
            "environment_setup_script_sha256",
            "conda_explicit_sha256",
            "environment_json_sha256",
            "import_smoke_manifest_sha256",
            "import_smoke_bundle_manifest_sha256",
            "import_smoke_sbatch_sha256",
        )
    }
    hashes.update({
        "config_sha256": CONFIG_SHA256,
        "overlay_contract_sha256": CONTRACT_SHA256,
        "upstream_commit": "d053054c5290a04c1c4cd8b55704d999cad73e30",
        "upstream_tree_git_sha1": "b0cace1fc54984e21a842f12d15d0b899e33d270",
    })
    return {
        "provenance_schema": 1,
        "purpose": "bounded Frontier grouped one-update engineering validation",
        "paper_evidence": False,
        "endpoint_class": "bounded_engineering_one_update",
        "max_student_updates": 1,
        "git": "git version 2.45.2",
        "xpid": XPID,
        "job_id": "local-test",
        "resources": {
            "partition": "gpuq",
            "qos": "gpu",
            "gres": "gpu:1g.10gb:1",
            "cpus_per_task": 2,
            "memory": "15G",
            "walltime": "00:30:00",
        },
        "hashes": hashes,
    }


@unittest.skipUnless(
    SOURCE.is_dir() and CPU_PYTHON.is_file(),
    "set TIE_AWARE_MINIMAX_SOURCE to a v4 applied clone for bounded runtime tests",
)
class GroupedOneUpdateV4E2E(unittest.TestCase):
    def test_warmup_then_exact_with_replacement_replay_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/"output"
            output.mkdir()
            provenance = root/"provenance.json"
            provenance.write_text(
                json.dumps(_provenance(), sort_keys=True)+"\n", encoding="utf-8")
            environment = os.environ.copy()
            environment.update({
                "PYTHONPATH": str(SOURCE/"src"),
                "PYTHONNOUSERSITE": "1",
                "JAX_PLATFORMS": "cpu",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
            })
            completed = subprocess.run(
                [
                    str(CPU_PYTHON), "-B", str(DRIVER),
                    "--config", str(CONFIG),
                    "--contract", str(CONTRACT),
                    "--provenance", str(provenance),
                    "--patched-source-dir", str(SOURCE),
                    "--output-dir", str(output),
                    "--local-test-mode",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
            )
            if completed.returncode:
                self.fail(
                    f"one-update driver failed\nstdout:\n{completed.stdout}"
                    f"\nstderr:\n{completed.stderr}")
            receipt = json.loads((output/"run-result.json").read_text())
            self.assertFalse(receipt["paper_evidence"])
            self.assertEqual(receipt["xpid"], XPID)
            self.assertTrue(receipt["checkpoint"]["fresh_runner_static_signature_validated"])
            warmup, replay = receipt["cycles"]
            for key in (
                "replay_group_draw_count",
                "replay_distinct_group_count",
                "replay_duplicate_group_count",
                "last_replay_group_count",
                "last_replay_distinct_group_count",
                "last_replay_duplicate_group_count",
            ):
                self.assertEqual(warmup["state"][key], 0)
            self.assertEqual(replay["state"]["replay_group_draw_count"], 4)
            self.assertEqual(replay["state"]["last_replay_group_count"], 4)
            distinct = replay["state"]["replay_distinct_group_count"]
            self.assertGreaterEqual(distinct, 1)
            self.assertLessEqual(distinct, 4)
            self.assertEqual(replay["state"]["last_replay_distinct_group_count"], distinct)
            self.assertEqual(replay["state"]["replay_duplicate_group_count"], 4-distinct)
            self.assertEqual(replay["state"]["last_replay_duplicate_group_count"], 4-distinct)
            self.assertTrue(replay["state"]["tie_aware_score_ranks"])
            self.assertEqual(replay["state"]["nonfinite_filled_score_count"], 0)
            self.assertEqual(replay["state"]["nonfinite_score_rejection_count"], 0)


if __name__ == "__main__":
    unittest.main()
