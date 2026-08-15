"""Endpoint-free integration and adversarial tests for schema-2 assembly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from ued_benchmark.analysis import preregistered_dev_analysis as gate
from ued_benchmark.analysis.test_preregistered_dev_analysis import SyntheticCampaign
from ued_benchmark.scripts import assemble_matched_run as assembler


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy(source: Path, destination: Path) -> None:
    destination.write_bytes(source.read_bytes())


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_source_packages(
    fixture: SyntheticCampaign,
    run_id: str,
    workspace: Path,
) -> argparse.Namespace:
    final = fixture.runs / run_id
    arm = "frontier" if run_id.endswith("-frontier") else "maxmc"
    components = workspace / "components"
    components.mkdir()
    training = components / "training"
    sidecar = components / "training-sidecar"
    evaluation = components / "evaluation"
    ancillary = components / "ancillary"
    for directory in (training, sidecar, evaluation, ancillary):
        directory.mkdir()
    for name in assembler.TRAINING_OUTPUT_NAMES:
        _copy(final / name, training / name)
    _copy(final / "training-receipt.json", sidecar / "training-receipt.json")
    _copy(final / "training-SHA256SUMS", sidecar / "SHA256SUMS")
    _copy(final / "training-COMPLETE", sidecar / "COMPLETE")
    if arm == "frontier":
        _copy(
            final / "training-frontier-buffer-snapshot.json",
            sidecar / "frontier-buffer-snapshot.json",
        )
    for name in ("evaluation-episodes.jsonl", "evaluation.csv", "evaluation-receipt.json"):
        _copy(final / name, evaluation / name)
    _copy(final / "evaluation-SHA256SUMS", evaluation / "SHA256SUMS")
    _copy(final / "evaluation-COMPLETE", evaluation / "COMPLETE")
    for key, name in assembler.ANCILLARY_OUTPUT_NAMES.items():
        _copy(final / name, ancillary / name)
    context = components / "run-context.json"
    _copy(final / "run-context.json", context)

    shutil.rmtree(final)
    return argparse.Namespace(
        campaign_manifest=fixture.campaign_path.resolve(),
        expected_campaign_sha256=fixture.campaign_sha,
        run_context=context.resolve(),
        expected_run_context_sha256=_sha(context),
        expected_assembler_sha256=_sha(Path(assembler.__file__).resolve()),
        training_output_dir=training.resolve(),
        training_sidecar_dir=sidecar.resolve(),
        evaluation_package_dir=evaluation.resolve(),
        command=(ancillary / "command.txt").resolve(),
        scheduler=(ancillary / "scheduler.json").resolve(),
        stdout=(ancillary / "stdout.log").resolve(),
        stderr=(ancillary / "stderr.log").resolve(),
        output_dir=(fixture.runs / run_id).resolve(),
        engineering_test_mode=False,
    )


def _convert_to_local_engineering(
    fixture: SyntheticCampaign,
    cli: argparse.Namespace,
    workspace: Path,
) -> argparse.Namespace:
    run_id = "engineering-ued-terminal-s101-maxmc"
    job_id = "8123456"
    campaign = dict(fixture.campaign)
    campaign["submissions"] = [{
        "arm": "maxmc",
        "training_seed": 101,
        "evaluation_seed": 100101,
        "run_id": run_id,
        "job_id": job_id,
        "attempt": 1,
    }]
    campaign_path = workspace / "engineering-campaign.json"
    _write_json(campaign_path, campaign)
    campaign_sha = _sha(campaign_path)

    context = json.loads(cli.run_context.read_text(encoding="utf-8"))
    context.update({
        "run_id": run_id,
        "job_id": job_id,
        "campaign_manifest_sha256": campaign_sha,
    })
    _write_json(cli.run_context, context)
    context_sha = _sha(cli.run_context)

    endpoint_path = cli.training_output_dir / "endpoint.json"
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    endpoint.update({
        "run_id": run_id,
        "n_updates": 1,
        "n_grad_updates": 1,
        "optimizer_step_applications": 1,
        "outer_cycles": 1,
        "student_training_transitions": 8192,
    })
    _write_json(endpoint_path, endpoint)

    meta_path = cli.training_output_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["xpid"] = run_id
    meta["slurm"]["job_id"] = job_id
    meta["config"].update({"xpid": run_id, "n_total_updates": 1})
    meta["config"]["student_rl_args"]["n_epochs"] = 1
    _write_json(meta_path, meta)

    training_receipt_path = cli.training_sidecar_dir / "training-receipt.json"
    training_receipt = json.loads(training_receipt_path.read_text(encoding="utf-8"))
    training_receipt.update({
        "endpoint_class": "bounded_engineering_test",
        "run_id": run_id,
        "job_id": job_id,
        "outer_cycles": 1,
        "student_training_transitions": 8192,
        "n_updates": 1,
        "upstream_n_grad_updates": 1,
        "optimizer_step_applications": 1,
    })
    training_receipt["optimizer_step_formula"] = {
        "n_updates": 1,
        "student_n_epochs": 1,
        "student_n_minibatches": 1,
    }
    training_receipt["engineering_test"] = {
        "enabled": True,
        "execution_mode": "local",
        "overrides": [{"field": "n_total_updates", "authored": 30000, "value": 1}],
    }
    training_receipt["config"].update({
        "resolved": meta["config"],
        "resolved_canonical_sha256": gate._canonical_sha256(meta["config"]),
        "meta_sha256": _sha(meta_path),
        "logs_sha256": _sha(cli.training_output_dir / "logs.csv"),
    })
    training_receipt["provenance"].update({
        "run_context": context,
        "run_context_sha256": context_sha,
        "backend": "cpu",
        "devices": [
            {"id": 0, "platform": "cpu", "device_kind": "synthetic-cpu"}
        ],
    })
    training_receipt["endpoint"] = {
        "path": "endpoint.json", "sha256": _sha(endpoint_path)
    }
    _write_json(training_receipt_path, training_receipt)
    training_manifest = cli.training_sidecar_dir / "SHA256SUMS"
    training_manifest.write_text(
        f"{_sha(training_receipt_path)}  training-receipt.json\n", encoding="utf-8")
    training_complete = json.loads(
        (cli.training_sidecar_dir / "COMPLETE").read_text(encoding="utf-8"))
    training_complete.update({
        "run_id": run_id,
        "sha256sums_sha256": _sha(training_manifest),
        "file_count": 1,
    })
    _write_json(cli.training_sidecar_dir / "COMPLETE", training_complete)

    evaluation_receipt_path = cli.evaluation_package_dir / "evaluation-receipt.json"
    evaluation_receipt = json.loads(evaluation_receipt_path.read_text(encoding="utf-8"))
    evaluation_receipt.update({
        "run_id": run_id,
        "synthetic_test_mode": True,
        "training_receipt_sha256": _sha(training_receipt_path),
        "meta_sha256": _sha(meta_path),
    })
    evaluation_receipt["evaluation_transition_accounting"].update({
        "effective_primary_transitions": 0,
        "primary_runner_scans_full_horizon": False,
        "total_runtime_transitions": 0,
    })
    evaluation_receipt["provenance"].update({
        "run_context": context,
        "run_context_sha256": context_sha,
        "runtime": {
            "backend": "deterministic_synthetic",
            "device_count": 0,
            "devices": [],
            "minimax_module": None,
        },
    })
    _write_json(evaluation_receipt_path, evaluation_receipt)
    evaluation_manifest = cli.evaluation_package_dir / "SHA256SUMS"
    evaluation_manifest.write_text(
        "".join(
            f"{_sha(cli.evaluation_package_dir / name)}  {name}\n"
            for name in sorted((
                "evaluation-episodes.jsonl", "evaluation.csv", "evaluation-receipt.json"
            ))
        ),
        encoding="utf-8",
    )
    evaluation_complete = json.loads(
        (cli.evaluation_package_dir / "COMPLETE").read_text(encoding="utf-8"))
    evaluation_complete.update({
        "run_id": run_id,
        "sha256sums_sha256": _sha(evaluation_manifest),
    })
    _write_json(cli.evaluation_package_dir / "COMPLETE", evaluation_complete)

    scheduler = json.loads(cli.scheduler.read_text(encoding="utf-8"))
    scheduler["job_id"] = job_id
    _write_json(cli.scheduler, scheduler)
    cli.campaign_manifest = campaign_path.resolve()
    cli.expected_campaign_sha256 = campaign_sha
    cli.expected_run_context_sha256 = context_sha
    cli.output_dir = (fixture.runs / run_id).resolve()
    cli.engineering_test_mode = True
    return cli


class MatchedRunAssemblerTest(unittest.TestCase):
    def test_synthetic_sources_assemble_then_full_analyzer_advances(self) -> None:
        """Synthetic train/eval sources exercise the complete handoff without endpoints."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            fixture_root = workspace / "fixture"
            fixture_root.mkdir()
            fixture = SyntheticCampaign(fixture_root)
            run_id = gate._run_id(101, "frontier")
            cli = _extract_source_packages(fixture, run_id, workspace)
            result = assembler.assemble(cli)
            cli.expected_package_sha256sums_sha256 = result[
                "package_sha256sums_sha256"
            ]
            validation = assembler.validate_output(cli)
            analyzed = gate.analyze(
                fixture.campaign_path, fixture.campaign_sha, fixture.runs)

            self.assertFalse(result["paper_evidence"])
            self.assertTrue(result["analyzer_eligible"])
            self.assertEqual(result["raw_record_count"], 30)
            self.assertEqual(validation["status"], "PASS")
            self.assertEqual(result["package_sha256sums_sha256"],
                             _sha(cli.output_dir / "SHA256SUMS"))
            self.assertEqual(analyzed["integrity_gate"], "PASS")
            self.assertTrue(
                analyzed["all_300_raw_episode_records_validated_before_aggregate_csv_parse"])

    def test_engineering_package_is_closed_but_analyzer_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            fixture_root = workspace / "fixture"
            fixture_root.mkdir()
            fixture = SyntheticCampaign(fixture_root)
            cli = _extract_source_packages(fixture, gate._run_id(101, "maxmc"), workspace)
            cli = _convert_to_local_engineering(fixture, cli, workspace)
            result = assembler.assemble(cli)
            cli.expected_package_sha256sums_sha256 = result[
                "package_sha256sums_sha256"
            ]
            validation = assembler.validate_output(cli)
            manifest = json.loads(
                (cli.output_dir / "run-manifest.json").read_text(encoding="utf-8"))

            self.assertFalse(result["analyzer_eligible"])
            self.assertFalse(validation["analyzer_eligible"])
            self.assertFalse(manifest["analyzer_eligible"])
            self.assertEqual(manifest["endpoint_class"], "bounded_engineering_test")
            self.assertEqual(
                gate._verify_package_closure(cli.output_dir, manifest["run_id"], "maxmc"),
                _sha(cli.output_dir / "SHA256SUMS"),
            )
            protocol, protocol_sha = gate._load_protocol()
            campaign = json.loads(cli.campaign_manifest.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(gate.GateError, "manifest identity drift"):
                gate._validate_run(
                    cli.output_dir,
                    campaign["submissions"][0],
                    campaign,
                    cli.expected_campaign_sha256,
                    protocol,
                    protocol_sha,
                )

    def test_extra_training_artifact_refuses_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            fixture_root = workspace / "fixture"
            fixture_root.mkdir()
            fixture = SyntheticCampaign(fixture_root)
            cli = _extract_source_packages(fixture, gate._run_id(101, "maxmc"), workspace)
            (cli.training_output_dir / "unexpected.txt").write_text("extra\n", encoding="utf-8")
            with self.assertRaisesRegex(assembler.AssemblyError, "closure drift"):
                assembler.assemble(cli)
            self.assertFalse(cli.output_dir.exists())

    def test_partial_evaluation_package_refuses_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            fixture_root = workspace / "fixture"
            fixture_root.mkdir()
            fixture = SyntheticCampaign(fixture_root)
            cli = _extract_source_packages(fixture, gate._run_id(101, "maxmc"), workspace)
            (cli.evaluation_package_dir / "evaluation-receipt.json").unlink()
            with self.assertRaisesRegex(assembler.AssemblyError, "closure drift"):
                assembler.assemble(cli)
            self.assertFalse(cli.output_dir.exists())

    def test_slurm_engineering_refuses_synthetic_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            fixture_root = workspace / "fixture"
            fixture_root.mkdir()
            fixture = SyntheticCampaign(fixture_root)
            cli = _extract_source_packages(fixture, gate._run_id(101, "maxmc"), workspace)
            cli = _convert_to_local_engineering(fixture, cli, workspace)
            training_path = cli.training_sidecar_dir / "training-receipt.json"
            training = json.loads(training_path.read_text(encoding="utf-8"))
            training["engineering_test"]["execution_mode"] = "slurm"
            training["provenance"]["backend"] = "gpu"
            training["provenance"]["devices"] = [
                {"id": 0, "platform": "gpu", "device_kind": "synthetic-gpu"}
            ]
            _write_json(training_path, training)
            training_manifest = cli.training_sidecar_dir / "SHA256SUMS"
            training_manifest.write_text(
                f"{_sha(training_path)}  training-receipt.json\n", encoding="utf-8")
            training_complete_path = cli.training_sidecar_dir / "COMPLETE"
            training_complete = json.loads(training_complete_path.read_text(encoding="utf-8"))
            training_complete["sha256sums_sha256"] = _sha(training_manifest)
            _write_json(training_complete_path, training_complete)
            evaluation_path = cli.evaluation_package_dir / "evaluation-receipt.json"
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            evaluation["training_receipt_sha256"] = _sha(training_path)
            _write_json(evaluation_path, evaluation)
            evaluation_manifest = cli.evaluation_package_dir / "SHA256SUMS"
            evaluation_manifest.write_text(
                "".join(
                    f"{_sha(cli.evaluation_package_dir / name)}  {name}\n"
                    for name in sorted((
                        "evaluation-episodes.jsonl", "evaluation.csv",
                        "evaluation-receipt.json",
                    ))
                ),
                encoding="utf-8",
            )
            evaluation_complete_path = cli.evaluation_package_dir / "COMPLETE"
            evaluation_complete = json.loads(
                evaluation_complete_path.read_text(encoding="utf-8"))
            evaluation_complete["sha256sums_sha256"] = _sha(evaluation_manifest)
            _write_json(evaluation_complete_path, evaluation_complete)
            with self.assertRaisesRegex(
                    assembler.AssemblyError, "Slurm engineering evaluation cannot be synthetic"):
                assembler.assemble(cli)
            self.assertFalse(cli.output_dir.exists())

    def test_symlinked_input_is_refused_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            fixture_root = workspace / "fixture"
            fixture_root.mkdir()
            fixture = SyntheticCampaign(fixture_root)
            cli = _extract_source_packages(fixture, gate._run_id(101, "maxmc"), workspace)
            real = cli.command
            link = real.with_name("command-link.txt")
            link.symlink_to(real)
            cli.command = link
            with self.assertRaisesRegex(assembler.AssemblyError, "symlink"):
                assembler.assemble(cli)
            self.assertFalse(cli.output_dir.exists())

    def test_manifest_path_traversal_is_refused_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            fixture_root = workspace / "fixture"
            fixture_root.mkdir()
            fixture = SyntheticCampaign(fixture_root)
            cli = _extract_source_packages(fixture, gate._run_id(101, "maxmc"), workspace)
            manifest = cli.training_sidecar_dir / "SHA256SUMS"
            digest = manifest.read_text(encoding="utf-8").split()[0]
            manifest.write_text(f"{digest}  ../training-receipt.json\n", encoding="utf-8")
            complete_path = cli.training_sidecar_dir / "COMPLETE"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["sha256sums_sha256"] = _sha(manifest)
            complete_path.write_text(
                json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(assembler.AssemblyError, "unsafe training source"):
                assembler.assemble(cli)
            self.assertFalse(cli.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
