"""Synthetic, endpoint-free tests for the matched UED development gate."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ued_benchmark.analysis import preregistered_dev_analysis as gate
from ued_benchmark.scripts import assemble_matched_run as assembler


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SyntheticCampaign:
    def __init__(self, root: Path, *, mismatch_budget: bool = False,
                 frontier_counter: int = 0, meta_drift: bool = False,
                 optimizer_step_drift: bool = False) -> None:
        self.root = root
        self.runs = root / "runs"
        self.runs.mkdir()
        self.protocol, self.protocol_sha = gate._load_protocol()
        fake_hashes = iter("123456789abcdef")
        self.provenance = {
            "base_commit": self.protocol["provenance"]["base_commit"],
            "base_tree": self.protocol["provenance"]["base_tree"],
            "overlay_contract_sha256": self.protocol["provenance"]["overlay_contract_sha256"],
            "bundle_manifest_sha256": next(fake_hashes) * 64,
            "overlay_manifest_sha256": next(fake_hashes) * 64,
            "applied_overlay_manifest_sha256": next(fake_hashes) * 64,
            "environment_manifest_sha256": next(fake_hashes) * 64,
            "training_driver_sha256": next(fake_hashes) * 64,
            "evaluation_driver_sha256": next(fake_hashes) * 64,
            "assembler_driver_sha256": _sha(Path(assembler.__file__).resolve()),
            "sbatch_sha256": next(fake_hashes) * 64,
        }
        self.hardware = {
            "partition": "gpuq",
            "gpu_model": "NVIDIA A100-SXM4-80GB",
            "gpu_profile": "3g.40gb",
            "gpu_count": 1,
            "n_devices": 1,
        }
        submissions = []
        job_number = 8000000
        for seed in self.protocol["training_seeds"]:
            for arm in gate.ARMS:
                job_number += 1
                submissions.append({
                    "arm": arm,
                    "training_seed": seed,
                    "evaluation_seed": 100000 + seed,
                    "run_id": gate._run_id(seed, arm),
                    "job_id": str(job_number),
                    "attempt": 1,
                })
        self.campaign = {
            "schema": 1,
            "protocol_id": gate.PROTOCOL_ID,
            "purpose": gate.PURPOSE,
            "created_utc": "2026-08-14T12:00:00Z",
            "frozen_before_endpoint_access": True,
            "protocol_sha256": self.protocol_sha,
            "analyzer_sha256": gate._sha256(Path(gate.__file__).resolve()),
            "provenance": self.provenance,
            "hardware": self.hardware,
            "submissions": submissions,
        }
        self.campaign_path = root / "campaign-manifest.json"
        _write_json(self.campaign_path, self.campaign)
        self.campaign_sha = _sha(self.campaign_path)

        for submission in submissions:
            outer = 60000
            if (mismatch_budget and submission["training_seed"] == 101
                    and submission["arm"] == "maxmc"):
                outer = 60010
            self._make_run(
                submission,
                outer=outer,
                frontier_counter=(
                    frontier_counter
                    if submission["training_seed"] == 101 and submission["arm"] == "frontier"
                    else 0
                ),
                meta_drift=(
                    meta_drift
                    if submission["training_seed"] == 101 and submission["arm"] == "frontier"
                    else False
                ),
                optimizer_step_drift=(
                    optimizer_step_drift
                    if submission["training_seed"] == 101 and submission["arm"] == "frontier"
                    else False
                ),
            )

    def _make_run(self, submission: dict[str, object], *, outer: int,
                  frontier_counter: int, meta_drift: bool,
                  optimizer_step_drift: bool) -> None:
        arm = str(submission["arm"])
        seed = int(submission["training_seed"])
        run_id = str(submission["run_id"])
        job_id = str(submission["job_id"])
        root = self.runs / run_id
        root.mkdir()

        (root / "checkpoint.pkl").write_bytes(f"synthetic-{run_id}".encode())
        checkpoint_sha = _sha(root / "checkpoint.pkl")
        (root / "command.txt").write_text(f"python -m minimax.train --seed={seed}\n", encoding="utf-8")
        (root / "stdout.log").write_text("synthetic stdout\n", encoding="utf-8")
        (root / "stderr.log").write_text("", encoding="utf-8")

        run_context = {
            "schema": 1,
            "protocol_id": gate.PROTOCOL_ID,
            "purpose": gate.PURPOSE,
            "run_id": run_id,
            "arm": arm,
            "training_seed": seed,
            "job_id": job_id,
            "campaign_manifest_sha256": self.campaign_sha,
            "provenance": {
                key: value for key, value in self.provenance.items()
                if key in gate.RUN_CONTEXT_PROVENANCE_KEYS
            },
        }
        _write_json(root / "run-context.json", run_context)

        environments = self.protocol["evaluation"]["environments"]
        maxmc_values = {
            "Maze-SixteenRooms": 0.5,
            "Maze-Labyrinth": 0.4,
            "Maze-StandardMaze": 0.6,
        }
        frontier_values = {
            "Maze-SixteenRooms": 0.6,
            "Maze-Labyrinth": 0.4,
            "Maze-StandardMaze": 0.6,
        }
        solved = frontier_values if arm == "frontier" else maxmc_values
        columns = [
            key
            for env in environments
            for key in (
                f"eval/a0:test_return:{env}",
                f"eval/a0:test_solved_rate:{env}",
            )
        ]
        with (root / "evaluation.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            row = {}
            for env in environments:
                row[f"eval/a0:test_return:{env}"] = solved[env] * 0.9
                row[f"eval/a0:test_solved_rate:{env}"] = solved[env]
            writer.writerow(row)

        raw_records = []
        for env in environments:
            success_count = round(solved[env] * 10)
            for episode in range(10):
                is_solved = episode < success_count
                raw_records.append({
                    "environment": env,
                    "episode": episode,
                    "agent_index": 0,
                    "solved": is_solved,
                    "return": 0.9 if is_solved else 0.0,
                })
        (root / "evaluation-episodes.jsonl").write_text(
            "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in raw_records
            ),
            encoding="utf-8",
        )

        endpoint = {
            "schema": 1,
            "status": "completed",
            "run_id": run_id,
            "arm": arm,
            "training_seed": seed,
            "n_updates": 30000,
            "n_grad_updates": 30000,
            "optimizer_step_applications": 149999 if optimizer_step_drift else 150000,
            "outer_cycles": outer,
            "student_training_transitions": outer * 8192,
            "checkpoint_file": "checkpoint.pkl",
            "checkpoint_sha256": checkpoint_sha,
            "terminal_checkpoint_saved_after_training": True,
            "resumed": False,
            "frontier_integrity": (
                {
                    "n_rollouts": 8,
                    "n_eval": 8,
                    "group_size_match": True,
                    "incomplete_group_count": frontier_counter,
                    "duplicate_new_group_count": 0,
                    "buffer_total_trials": 4000,
                    "buffer_total_successes": 1000,
                }
                if arm == "frontier" else None
            ),
        }
        _write_json(root / "endpoint.json", endpoint)

        log_fields = ["_tick", "n_updates", "steps", "real_steps"]
        if arm == "frontier":
            log_fields += [
                "plr/frontier_n_rollouts",
                "plr/frontier_n_eval",
                "plr/frontier_group_size_match",
                "plr/frontier_incomplete_group_count",
                "plr/frontier_duplicate_new_group_count",
            ]
        with (root / "logs.csv").open("w", encoding="utf-8", newline="") as stream:
            stream.write("# " + ",".join(log_fields) + "\n")
            writer = csv.DictWriter(stream, fieldnames=log_fields)
            row = {
                "_tick": outer,
                "n_updates": 30000,
                "steps": outer * 8192,
                "real_steps": outer * 8192,
            }
            if arm == "frontier":
                row.update({
                    "plr/frontier_n_rollouts": 8,
                    "plr/frontier_n_eval": 8,
                    "plr/frontier_group_size_match": 1,
                    "plr/frontier_incomplete_group_count": frontier_counter,
                    "plr/frontier_duplicate_new_group_count": 0,
                })
            writer.writerow(row)

        config = copy.deepcopy(self.protocol["expected_static_meta_config"])
        config["train_runner_args"]["ued_score"] = self.protocol["arms"][arm]["ued_score"]
        if meta_drift:
            config["train_runner_args"]["temp"] = 0.31
        config.update({
            "seed": seed,
            "xpid": run_id,
            "log_dir": "/scratch/lwang44/maxrl/synthetic",
            "wandb_args": {"api_key": None, "base_url": "https://api.wandb.ai"},
        })
        meta = {
            "xpid": run_id,
            "config": config,
            "git": {"commit": self.protocol["provenance"]["base_commit"], "is_dirty": True},
            "slurm": {"job_id": job_id},
            "env": {"JAX_PLATFORM_NAME": "gpu"},
        }
        _write_json(root / "meta.json", meta)

        source_receipt = {
            "base_commit": self.protocol["provenance"]["base_commit"],
            "base_tree": self.protocol["provenance"]["base_tree"],
            "applied_overlay_manifest_sha256": self.provenance[
                "applied_overlay_manifest_sha256"
            ],
            "overlay_file_count": 1,
            "worktree_status": {
                ".frontierrl_overlay.json": "??",
                "src/minimax/util/rl/frontier_activity.py": " M",
            },
            "git_executable": "/scratch/lwang44/envs/synthetic/bin/git",
            "git_executable_sha256": "c" * 64,
            "git_version": "git version 2.45.2",
        }
        training_receipt = {
            "schema": 1,
            "status": "completed",
            "protocol_id": gate.PROTOCOL_ID,
            "purpose": gate.PURPOSE,
            "paper_evidence": False,
            "endpoint_class": "matched_development",
            "run_id": run_id,
            "arm": arm,
            "training_seed": seed,
            "job_id": job_id,
            "resumed": False,
            "outer_cycles": outer,
            "student_training_transitions": outer * 8192,
            "transitions_per_outer_cycle": 8192,
            "n_updates": 30000,
            "upstream_n_grad_updates": 30000,
            "optimizer_step_applications": 149999 if optimizer_step_drift else 150000,
            "periodic_evaluation_accounting": {
                "calls": outer // 100,
                "test_interval_outer_cycles": 100,
                "environment_count": 3,
                "episodes_per_environment": 10,
                "max_episode_horizon": 450,
                "per_environment_max_episode_horizons": [450, 450, 450],
                "budgeted_max_transitions": (outer // 100) * 13500,
                "runner_scans_full_horizon": True,
                "excluded_from_student_training_transitions": True,
            },
            "optimizer_step_formula": {
                "n_updates": 30000,
                "student_n_epochs": 5,
                "student_n_minibatches": 1,
            },
            "integrity": {
                "initial": {},
                "terminal": {},
                "checkpoint_round_trip": {},
                "last_step_stats": {},
                "max_outer_cycles": 301000,
            },
            "engineering_test": {
                "enabled": False,
                "execution_mode": "production",
                "overrides": [],
            },
            "terminal_checkpoint": {
                "path": "checkpoint.pkl",
                "sha256": checkpoint_sha,
                "saved_after_loop_termination": True,
                "periodic_checkpoint_used": False,
                "round_trip_counter_freshness": True,
            },
            "config": {
                "authored_path": Path(self.protocol["arms"][arm]["config_path"]).name,
                "authored_sha256": self.protocol["arms"][arm]["config_sha256"],
                "resolved": config,
                "resolved_canonical_sha256": gate._canonical_sha256(config),
                "meta_sha256": _sha(root / "meta.json"),
                "logs_sha256": _sha(root / "logs.csv"),
            },
            "provenance": {
                "run_context": run_context,
                "run_context_sha256": _sha(root / "run-context.json"),
                "protocol_sha256": self.protocol_sha,
                "training_driver_sha256": self.provenance["training_driver_sha256"],
                "source": source_receipt,
                "minimax_module": "/scratch/lwang44/synthetic/src/minimax/__init__.py",
                "backend": "gpu",
                "devices": [
                    {"id": 0, "platform": "gpu", "device_kind": "synthetic"}
                ],
            },
            "endpoint": {"path": "endpoint.json", "sha256": _sha(root / "endpoint.json")},
            "wall_seconds": 3600.0,
            "frontier_snapshot": None,
        }
        if arm == "frontier":
            snapshot = {
                "schema": 1,
                "status": "completed",
                "kind": "frontier_plr_buffer_safe_snapshot",
                "protocol_id": gate.PROTOCOL_ID,
                "purpose": gate.PURPOSE,
                "paper_evidence": False,
                "run_id": run_id,
                "arm": arm,
                "training_seed": seed,
                "checkpoint_sha256": checkpoint_sha,
                "buffer_size": 500,
                "filled_count": 0,
                "n_rollouts": 8,
                "n_eval": 8,
                "prior_alpha": 1.0,
                "prior_beta": 1.0,
                "replay_distribution": {},
                "stored_score_validation": {},
                "level_identity": "synthetic fixture",
                "level_identity_materialization": {},
                "slots": [],
            }
            _write_json(root / "training-frontier-buffer-snapshot.json", snapshot)
            training_receipt["frontier_snapshot"] = {
                "path": "frontier-buffer-snapshot.json",
                "sha256": _sha(root / "training-frontier-buffer-snapshot.json"),
            }
        _write_json(root / "training-receipt.json", training_receipt)
        training_source_names = ["training-receipt.json"]
        if arm == "frontier":
            training_source_names.append("frontier-buffer-snapshot.json")
        training_source_paths = {
            "training-receipt.json": root / "training-receipt.json",
            "frontier-buffer-snapshot.json": (
                root / "training-frontier-buffer-snapshot.json"
            ),
        }
        (root / "training-SHA256SUMS").write_text(
            "".join(
                f"{_sha(training_source_paths[name])}  {name}\n"
                for name in sorted(training_source_names)
            ),
            encoding="utf-8",
        )
        _write_json(root / "training-COMPLETE", {
            "schema": 1,
            "status": "complete",
            "run_id": run_id,
            "arm": arm,
            "sha256sums_sha256": _sha(root / "training-SHA256SUMS"),
            "file_count": len(training_source_names),
        })

        aggregate_values = {
            f"eval/a0:test_{metric}:{env}": (
                solved[env] if metric == "solved_rate" else solved[env] * 0.9
            )
            for env in environments
            for metric in ("return", "solved_rate")
        }
        evaluation_receipt = {
            "schema": 1,
            "status": "completed",
            "protocol_id": gate.PROTOCOL_ID,
            "purpose": gate.PURPOSE,
            "paper_evidence": False,
            "run_id": run_id,
            "arm": arm,
            "training_seed": seed,
            "evaluation_seed": 100000 + seed,
            "environments": environments,
            "n_episodes_per_environment": 10,
            "agent_indices": [0],
            "synthetic_test_mode": False,
            "evaluation_transition_accounting": {
                "environment_count": 3,
                "episodes_per_environment": 10,
                "max_episode_horizon": 450,
                "per_environment_max_episode_horizons": [450, 450, 450],
                "budgeted_primary_max_transitions": 13500,
                "effective_primary_transitions": 13500,
                "primary_runner_scans_full_horizon": True,
                "engineering_independent_verification_transitions": 0,
                "total_runtime_transitions": 13500,
                "excluded_from_student_training_transitions": True,
            },
            "terminal_checkpoint": {"sha256": checkpoint_sha},
            "training_receipt_sha256": _sha(root / "training-receipt.json"),
            "meta_sha256": _sha(root / "meta.json"),
            "provenance": {
                "run_context": run_context,
                "run_context_sha256": _sha(root / "run-context.json"),
                "protocol_sha256": self.protocol_sha,
                "evaluation_driver_sha256": self.provenance["evaluation_driver_sha256"],
                "source": source_receipt,
                "runtime": {
                    "backend": "gpu",
                    "device_count": 1,
                    "devices": [{"id": 0, "platform": "gpu", "device_kind": "synthetic"}],
                    "minimax_module": "/scratch/lwang44/synthetic/src/minimax/__init__.py",
                    "per_environment_max_episode_horizons": [450, 450, 450],
                    "raw_vs_independent_evalrunner": {
                        "checked": False,
                        "all_six_fields_checked": False,
                        "max_abs_error": None,
                        "per_field_abs_error": None,
                        "float32_tolerance": 2e-6,
                    },
                },
            },
            "raw_results": {
                "path": "evaluation-episodes.jsonl",
                "sha256": _sha(root / "evaluation-episodes.jsonl"),
                "record_count": 30,
            },
            "aggregate_results": {
                "path": "evaluation.csv",
                "sha256": _sha(root / "evaluation.csv"),
                "values": aggregate_values,
            },
        }
        _write_json(root / "evaluation-receipt.json", evaluation_receipt)
        evaluation_source_names = [
            "evaluation-episodes.jsonl", "evaluation.csv", "evaluation-receipt.json"
        ]
        (root / "evaluation-SHA256SUMS").write_text(
            "".join(
                f"{_sha(root / name)}  {name}\n"
                for name in sorted(evaluation_source_names)
            ),
            encoding="utf-8",
        )
        _write_json(root / "evaluation-COMPLETE", {
            "schema": 1,
            "status": "complete",
            "run_id": run_id,
            "sha256sums_sha256": _sha(root / "evaluation-SHA256SUMS"),
            "file_count": 3,
        })

        scheduler = {
            "schema": 1,
            "job_id": job_id,
            "state": "COMPLETED",
            "exit_code": "0:0",
            "partition": self.hardware["partition"],
            "gpu_model": self.hardware["gpu_model"],
            "gpu_profile": self.hardware["gpu_profile"],
            "gpu_count": 1,
            "elapsed_seconds": 3600,
            "max_rss_bytes": 1024,
            "peak_gpu_memory_bytes": 2048,
            "terminal_sacct_retrieved_utc": "2026-08-14T13:00:00Z",
        }
        _write_json(root / "scheduler.json", scheduler)

        run_manifest = {
            "schema": 2,
            "protocol_id": gate.PROTOCOL_ID,
            "purpose": gate.PURPOSE,
            "paper_evidence": False,
            "analyzer_eligible": True,
            "endpoint_class": "matched_development",
            "campaign_manifest_sha256": self.campaign_sha,
            "run_id": run_id,
            "arm": arm,
            "training_seed": seed,
            "evaluation_seed": 100000 + seed,
            "job_id": job_id,
            "config_template_path": self.protocol["arms"][arm]["config_path"],
            "config_template_sha256": self.protocol["arms"][arm]["config_sha256"],
            "provenance": self.provenance,
            "run_context": {
                "file": "run-context.json",
                "sha256": _sha(root / "run-context.json"),
            },
            "training_source_package": {
                "receipt_file": "training-receipt.json",
                "receipt_sha256": _sha(root / "training-receipt.json"),
                "sha256sums_file": "training-SHA256SUMS",
                "sha256sums_sha256": _sha(root / "training-SHA256SUMS"),
                "complete_file": "training-COMPLETE",
                "complete_sha256": _sha(root / "training-COMPLETE"),
                "source_payload_count": 2 if arm == "frontier" else 1,
                "frontier_snapshot_file": (
                    "training-frontier-buffer-snapshot.json" if arm == "frontier" else None
                ),
                "frontier_snapshot_sha256": (
                    _sha(root / "training-frontier-buffer-snapshot.json")
                    if arm == "frontier" else None
                ),
            },
            "evaluation_source_package": {
                "receipt_file": "evaluation-receipt.json",
                "receipt_sha256": _sha(root / "evaluation-receipt.json"),
                "raw_results_file": "evaluation-episodes.jsonl",
                "raw_results_sha256": _sha(root / "evaluation-episodes.jsonl"),
                "aggregate_results_file": "evaluation.csv",
                "aggregate_results_sha256": _sha(root / "evaluation.csv"),
                "sha256sums_file": "evaluation-SHA256SUMS",
                "sha256sums_sha256": _sha(root / "evaluation-SHA256SUMS"),
                "complete_file": "evaluation-COMPLETE",
                "complete_sha256": _sha(root / "evaluation-COMPLETE"),
                "source_payload_count": 3,
            },
            "evaluation": {
                "seed": 100000 + seed,
                "n_episodes": 10,
                "environments": environments,
                "checkpoint_sha256": checkpoint_sha,
                "results_file": "evaluation.csv",
                "results_sha256": _sha(root / "evaluation.csv"),
                "raw_results_file": "evaluation-episodes.jsonl",
                "raw_results_sha256": _sha(root / "evaluation-episodes.jsonl"),
                "raw_record_count": 30,
                "receipt_file": "evaluation-receipt.json",
                "receipt_sha256": _sha(root / "evaluation-receipt.json"),
            },
        }
        _write_json(root / "run-manifest.json", run_manifest)
        self._seal(root, run_id)

    @staticmethod
    def _seal(root: Path, run_id: str) -> None:
        arm = "frontier" if run_id.endswith("-frontier") else "maxmc"
        payloads = gate._package_payloads(arm)
        lines = [f"{_sha(root / name)}  {name}\n" for name in sorted(payloads)]
        (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
        _write_json(root / "COMPLETE", {
            "schema": 2,
            "status": "complete",
            "run_id": run_id,
            "sha256sums_sha256": _sha(root / "SHA256SUMS"),
            "file_count": len(payloads),
        })


class PreregisteredDevelopmentGateTest(unittest.TestCase):
    def test_repository_preflight_is_score_isolating(self) -> None:
        result = gate.repository_preflight()
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["endpoint_accessed"])
        self.assertEqual(result["authored_template_difference_keys"][-1], "ued_score")

    def test_complete_synthetic_campaign_advances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticCampaign(Path(directory))
            result = gate.analyze(fixture.campaign_path, fixture.campaign_sha, fixture.runs)
        self.assertEqual(result["integrity_gate"], "PASS")
        self.assertTrue(result["all_ten_packages_validated_before_metric_parse"])
        self.assertTrue(result["decision"]["advance_exact_grouped_frontier"])
        self.assertEqual(result["budget_semantics"]["student_ppo_updates_per_arm"], 30000)
        self.assertEqual(
            result["budget_semantics"]["optimizer_step_applications_per_arm"], 150000)
        self.assertAlmostEqual(result["primary"]["mean_difference"], 1.0 / 30.0)
        self.assertEqual(len(result["paired_runs"]), 5)

    def test_pair_budget_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticCampaign(Path(directory), mismatch_budget=True)
            with self.assertRaisesRegex(gate.GateError, "outer-cycle budget mismatch"):
                gate.analyze(fixture.campaign_path, fixture.campaign_sha, fixture.runs)

    def test_nonzero_frontier_delivery_counter_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticCampaign(Path(directory), frontier_counter=1)
            with self.assertRaisesRegex(gate.GateError, "nonzero Frontier delivery counters"):
                gate.analyze(fixture.campaign_path, fixture.campaign_sha, fixture.runs)

    def test_resolved_config_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticCampaign(Path(directory), meta_drift=True)
            with self.assertRaisesRegex(gate.GateError, "resolved meta config drift"):
                gate.analyze(fixture.campaign_path, fixture.campaign_sha, fixture.runs)

    def test_optimizer_step_application_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticCampaign(Path(directory), optimizer_step_drift=True)
            with self.assertRaisesRegex(
                    gate.GateError, "optimizer-step application accounting drift"):
                gate.analyze(fixture.campaign_path, fixture.campaign_sha, fixture.runs)

    def test_missing_run_is_rejected_before_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticCampaign(Path(directory))
            missing = fixture.runs / gate._run_id(105, "maxmc")
            for path in missing.iterdir():
                path.unlink()
            missing.rmdir()
            with self.assertRaisesRegex(gate.GateError, "missing or extra run packages"):
                gate.analyze(fixture.campaign_path, fixture.campaign_sha, fixture.runs)

    def test_unsealed_nonnumeric_csv_is_rejected_as_closure_before_parse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticCampaign(Path(directory))
            root = fixture.runs / gate._run_id(101, "frontier")
            (root / "evaluation.csv").write_text("poison,not,numeric\n", encoding="utf-8")
            with self.assertRaisesRegex(gate.GateError, "payload hash mismatch"):
                gate.analyze(fixture.campaign_path, fixture.campaign_sha, fixture.runs)

    def test_fully_resealed_raw_episode_reordering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticCampaign(Path(directory))
            run_id = gate._run_id(101, "frontier")
            root = fixture.runs / run_id
            raw_path = root / "evaluation-episodes.jsonl"
            lines = raw_path.read_text(encoding="utf-8").splitlines()
            lines[0], lines[1] = lines[1], lines[0]
            raw_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            receipt_path = root / "evaluation-receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["raw_results"]["sha256"] = _sha(raw_path)
            _write_json(receipt_path, receipt)
            source_manifest = root / "evaluation-SHA256SUMS"
            source_manifest.write_text(
                "".join(
                    f"{_sha(root / name)}  {name}\n"
                    for name in sorted((
                        "evaluation-episodes.jsonl", "evaluation.csv",
                        "evaluation-receipt.json",
                    ))
                ),
                encoding="utf-8",
            )
            complete = json.loads((root / "evaluation-COMPLETE").read_text(encoding="utf-8"))
            complete["sha256sums_sha256"] = _sha(source_manifest)
            _write_json(root / "evaluation-COMPLETE", complete)
            manifest_path = root / "run-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["evaluation_source_package"].update({
                "receipt_sha256": _sha(receipt_path),
                "raw_results_sha256": _sha(raw_path),
                "sha256sums_sha256": _sha(source_manifest),
                "complete_sha256": _sha(root / "evaluation-COMPLETE"),
            })
            manifest["evaluation"].update({
                "raw_results_sha256": _sha(raw_path),
                "receipt_sha256": _sha(receipt_path),
            })
            _write_json(manifest_path, manifest)
            fixture._seal(root, run_id)
            with self.assertRaisesRegex(gate.GateError, "raw evaluation order drift"):
                gate.analyze(fixture.campaign_path, fixture.campaign_sha, fixture.runs)


if __name__ == "__main__":
    unittest.main()
