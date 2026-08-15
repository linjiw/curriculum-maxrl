#!/usr/bin/env python3
"""Focused CPU-only adversarial tests for the v4h remote contracts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


environment_tree = load(HERE / "environment_tree.py", "v4h_test_environment")
gpu_probe = load(HERE / "gpu_runtime_probe.py", "v4h_test_gpu")
job_guard = load(HERE / "job_guard.py", "v4h_test_guard")
pair_plan = load(HERE / "pair_plan.py", "v4h_test_pair")
slurm = load(HERE / "slurm_integrity.py", "v4h_test_slurm")
launcher = load(ROOT / "hopper/hopper_v4_remote_hardened.py", "v4h_test_launcher")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


class EnvironmentClosureTests(unittest.TestCase):
    def test_byte_closure_detects_mutation_and_escape(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v4h-env-") as raw:
            root = Path(raw).resolve(); environment = root / "environment"
            (environment / "bin").mkdir(parents=True); (environment / "lib").mkdir()
            shutil.copy2(Path(sys.executable).resolve(), environment / "bin/python")
            shutil.copy2(Path("/bin/true").resolve(), root / "conda")
            (environment / "lib/payload.txt").write_text("closed\n", encoding="utf-8")
            (environment / "lib/payload-link").symlink_to("payload.txt")
            closure = root / "closure"
            tool_sha = digest(HERE / "environment_tree.py")
            _, manifest, receipt = environment_tree.create(
                environment, root / "conda", closure, tool_sha,
                platform.python_version(),
            )
            environment_tree.verify(
                environment, root / "conda", closure, tool_sha,
                platform.python_version(), manifest, receipt,
            )
            (environment / "lib/payload.txt").write_text("mutated\n", encoding="utf-8")
            with self.assertRaisesRegex(environment_tree.EnvironmentClosureError, "byte tree drift"):
                environment_tree.verify(
                    environment, root / "conda", closure, tool_sha,
                    platform.python_version(), manifest, receipt,
                )
            (environment / "lib/payload.txt").write_text("closed\n", encoding="utf-8")
            (environment / "escape").symlink_to("/bin/true")
            with self.assertRaisesRegex(environment_tree.EnvironmentClosureError, "escapes prefix"):
                environment_tree.inventory(environment)


class GPUContractTests(unittest.TestCase):
    @staticmethod
    def environment(rung: str = "terminal") -> dict[str, str]:
        return {
            "SLURM_JOB_NAME": {"import": "ued-v4h-import", "one_update": "ued-v4h-one-update", "terminal": "ued-v4h-terminal"}[rung],
            "SLURM_JOB_PARTITION": "gpuq", "SLURM_JOB_QOS": "gpu",
            "SLURM_CPUS_PER_TASK": "2", "SLURM_JOB_NUM_NODES": "1",
            "SLURM_NTASKS": "1", "SLURM_MEM_PER_NODE": "15360",
            "SLURM_GPUS_ON_NODE": "1", "SLURM_TRES_PER_NODE": "gres/gpu:1g.10gb:1",
            "SLURM_EXPORT_ENV": "NIL", "SLURM_RESTART_COUNT": "0",
            "CUDA_VISIBLE_DEVICES": "MIG-good", "SLURM_JOB_GPUS": "0",
        }

    @staticmethod
    def mock() -> dict:
        return {
            "cuda_driver": {"device_count": 1, "device_name": "NVIDIA A100 MIG 1g.10gb", "total_memory_bytes": 10_000 * 1024 * 1024},
            "jax": {"backend": "gpu", "device_count": 1, "device_id": 0, "device_kind": "NVIDIA A100 MIG 1g.10gb", "platform": "gpu"},
        }

    def test_exact_mig_and_allocation_negatives(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v4h-gpu-") as raw:
            root = Path(raw).resolve(); mock = root / "mock.json"; write_json(mock, self.mock())
            output = root / "runtime.json"; tool_sha = digest(HERE / "gpu_runtime_probe.py")
            gpu_probe.probe(
                output, "terminal", tool_sha, "1" * 64,
                local_test_mode=True, mock_input=mock,
                environment=self.environment(),
            )
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt["slurm"]["slurm_tres_per_node"], "gres/gpu:1g.10gb:1")
            for key, value, message in (
                ("SLURM_RESTART_COUNT", "1", "restarted"),
                ("SLURM_MEM_PER_NODE", "16000", "memory"),
                ("SLURM_TRES_PER_NODE", "gres/gpu:1", "typed MIG"),
            ):
                environment = self.environment(); environment[key] = value
                with self.assertRaisesRegex(gpu_probe.GPUProbeError, message):
                    gpu_probe._slurm_identity(environment, "terminal", True)
            environment = self.environment(); environment["SLURM_ARRAY_JOB_ID"] = "9"
            with self.assertRaisesRegex(gpu_probe.GPUProbeError, "arrays"):
                gpu_probe._slurm_identity(environment, "terminal", True)
            bad = self.mock(); bad["cuda_driver"]["total_memory_bytes"] = 12_000 * 1024 * 1024
            with self.assertRaisesRegex(gpu_probe.GPUProbeError, "not exactly"):
                gpu_probe.validate_measurements(bad["cuda_driver"], bad["jax"])


class SlurmReceiptTests(unittest.TestCase):
    def fixture(self, root: Path):
        job = "12345"; sbatch = root / "bundle/hopper/sbatch/terminal.sbatch"
        sbatch.parent.mkdir(parents=True); sbatch.write_text("#!/usr/bin/bash\n", encoding="utf-8")
        envelope = root / "input.nul"; envelope.write_bytes(b"UED_A=value\0")
        work = "/scratch/test/maxrl"; stdout = f"{work}/tests/logs/ued-v4h-terminal_{job}.out"
        receipt = {
            "schema": 1, "status": "submitted", "job_id": job,
            "created_utc": "2026-08-14T15:59:59Z", "paper_evidence": False,
            "production_authorized": False, "ambient_environment": "env-i-empty",
            "export_mode": "NIL", "get_user_env": False,
            "runtime_input_mode": "NUL_argument_envelope", "sbatch_path": str(sbatch),
            "sbatch_sha256": digest(sbatch), "input_envelope_path": str(envelope),
            "input_envelope_sha256": digest(envelope), "input_keys": ["UED_A"],
            "work_dir": work, "expected_stdout_path": stdout,
            "argv": ["/usr/bin/sbatch", "--parsable", f"--chdir={work}", "--export=NIL", str(sbatch), f"--ued-input-envelope={envelope}", f"--ued-bundle-dir={root / 'bundle'}", f"--ued-submitted-sbatch={sbatch}"],
            "remote_submission_authorized": True,
        }
        submission_path = root / "submission.json"; write_json(submission_path, receipt)
        submission, _ = slurm.validate_submission(
            submission_path, envelope, {"UED_A"}, job_id=job,
            sbatch_path=sbatch, sbatch_sha256=digest(sbatch),
        )
        zone = ZoneInfo("America/New_York")
        submit = datetime(2026, 8, 14, 11, 59, 58, tzinfo=zone)
        start = datetime(2026, 8, 14, 12, 0, 0, tzinfo=zone)
        end = start + timedelta(seconds=30); retrieved = end.astimezone(timezone.utc) + timedelta(seconds=2)
        submit_line = " ".join(receipt["argv"])
        row = "|".join((
            job, "ued-v4h-terminal", "gpuq", "COMPLETED", "0:0", "30", "2", "15G",
            "node01", submit.strftime("%Y-%m-%dT%H:%M:%S"), start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S"),
            "billing=2,cpu=2,gres/gpu=1,gres/gpu:1g.10gb=1,mem=15G,node=1", "gpu", "45", "0",
            work, stdout, stdout, submit_line,
        ))
        terminal = root / "terminal.tsv"
        terminal.write_text(
            "terminal_receipt_schema\t2\n"
            f"retrieved_utc\t{retrieved.strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
            f"retrieved_epoch\t{int(retrieved.timestamp())}\n"
            f"terminal_end_epoch\t{int(end.timestamp())}\n"
            f"terminal_header\t{slurm.TERMINAL_HEADER}\nterminal_row\t{row}\n"
            f"resource_header\t{slurm.RESOURCE_HEADER}\n"
            f"resource_row\t{job}.batch|900M|cpu=2,gres/gpumem=9G\n"
            f"resource_row\t{job}.extern|2G|cpu=1\n",
            encoding="utf-8",
        )
        runtime = root / "gpu.json"; write_json(runtime, {
            "status": "complete", "requested_gres": "gpu:1g.10gb:1", "rung": "terminal",
            "slurm": {"job_id": job, "array_job": False, "slurm_restart_count": 0},
        })
        return submission, terminal, runtime, job

    def test_authoritative_rows_and_numeric_units(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v4h-slurm-") as raw:
            root = Path(raw).resolve(); submission, terminal, runtime, job = self.fixture(root)
            summary = slurm.validate_terminal(terminal, runtime, digest(runtime), submission, job_id=job)
            self.assertEqual(summary["max_rss_bytes"], 2 * 1024 ** 3)
            self.assertEqual({row["job_id"] for row in summary["resource_rows"]}, {f"{job}.batch", f"{job}.extern"})
            self.assertEqual(slurm.slurm_size_bytes("900M"), 900 * 1024 ** 2)
            for malformed in ("", "900MB", "nan", "-1G"):
                with self.assertRaises(slurm.SlurmIntegrityError): slurm.slurm_size_bytes(malformed)
            missing = root / "missing.tsv"
            missing.write_text("\n".join(line for line in terminal.read_text().splitlines() if ".extern|" not in line) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(slurm.SlurmIntegrityError, "batch/extern"):
                slurm.validate_terminal(missing, runtime, digest(runtime), submission, job_id=job)


class AllowlistTests(unittest.TestCase):
    def test_slurm_spool_copy_bootstraps_from_explicit_bound_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v4h-spool-") as raw:
            root = Path(raw).resolve(); bundle = root / "bundle"
            sbatch_dir = bundle / "hopper/sbatch"; sbatch_dir.mkdir(parents=True)
            runner = bundle / "hopper/run_ued_minimax_v4_remote_hardened.sh"
            runner.write_text("#!/usr/bin/bash\nprintf 'FAKE_RUNNER:%s\\n' \"$*\"\n", encoding="utf-8")
            runner.chmod(0o755)
            source = ROOT / "hopper/sbatch/ued_minimax_v4_remote_hardened_gpu_smoke.sbatch"
            submitted = sbatch_dir / source.name; shutil.copy2(source, submitted)
            spool_dir = root / "var/spool/slurmd/job123"; spool_dir.mkdir(parents=True)
            spool = spool_dir / "slurm_script"; shutil.copy2(submitted, spool)
            envelope = root / "input.nul"; envelope.write_bytes(b"UED_A=x\0")
            argv = ["/usr/bin/bash", str(spool), f"--ued-input-envelope={envelope}",
                    f"--ued-bundle-dir={bundle}", f"--ued-submitted-sbatch={submitted}"]
            environment = {"PATH": "/usr/bin:/bin", "SLURM_EXPORT_ENV": "NIL"}
            result = subprocess.run(argv, text=True, capture_output=True, env=environment, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"FAKE_RUNNER:import {submitted} --ued-input-envelope={envelope}", result.stdout)
            with spool.open("a", encoding="utf-8") as stream:
                stream.write("# tamper\n")
            result = subprocess.run(argv, text=True, capture_output=True, env=environment, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn("spool sbatch byte drift", result.stderr)

    def test_launcher_and_job_guard_share_exact_keysets(self) -> None:
        self.assertEqual(launcher.EXPECTED_KEYS, job_guard.EXPECTED_KEYS)
        for rung, keys in launcher.EXPECTED_KEYS.items():
            values = [f"{key}={'1' * 64 if key.endswith('_SHA256') else 'x'}" for key in sorted(keys)]
            self.assertEqual(set(launcher.parse_assignments(values, keys)), keys)
            with self.assertRaisesRegex(launcher.LauncherError, "extra"):
                launcher.parse_assignments(values + ["UED_EXTRA=x"], keys)
            with self.assertRaisesRegex(launcher.LauncherError, "duplicate"):
                launcher.parse_assignments(values + [values[0]], keys)

    def test_pair_receipt_keysets_are_exact(self) -> None:
        values = {key: "x" for key in pair_plan.IMPORT_RECEIPT_KEYS}
        text = "field\tvalue\n" + "".join(f"{key}\t{values[key]}\n" for key in sorted(values))
        self.assertEqual(set(pair_plan._parse_receipt(text, pair_plan.IMPORT_RECEIPT_KEYS, "import")), pair_plan.IMPORT_RECEIPT_KEYS)
        with self.assertRaisesRegex(pair_plan.PairPlanError, "exact keyset"):
            pair_plan._parse_receipt(text + "extra\tx\n", pair_plan.IMPORT_RECEIPT_KEYS, "import")

    def test_source_files_keep_remote_actions_disabled(self) -> None:
        launcher_text = (ROOT / "hopper/hopper_v4_remote_hardened.py").read_text()
        state_stage = (ROOT / "hopper/stage_ued_minimax_v4_remote_hardened.sh").read_text()
        self.assertIn("remote submission is HOLD", launcher_text)
        self.assertNotIn("subprocess.run(\n        [\"/usr/bin/sbatch\"", launcher_text)
        self.assertNotIn("ssh ", state_stage)
        self.assertNotIn("scp ", state_stage)
        self.assertNotIn("rsync ", state_stage)
        for path in (ROOT / "hopper/sbatch").glob("ued_minimax_v4_remote_hardened_*.sbatch"):
            text = path.read_text()
            self.assertIn("#SBATCH --gres=gpu:1g.10gb:1", text)
            self.assertIn("#SBATCH --no-requeue", text)
            self.assertIn("#SBATCH --export=NIL", text)
            self.assertNotIn("#SBATCH --array", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
