"""Synthetic-only tests for outcome-blind BARN retry selection."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from icra2027 import merge_barn_campaign as merger
from icra2027 import select_barn_attempts as selector
from icra2027.test_merge_barn_campaign import HASHES, _artifact, _protocol


def _write_artifact(tmp_path: Path, artifact: dict, name: str):
    path = tmp_path / name
    path.write_text(json.dumps(artifact) + "\n")
    return path


def _ledger_row(path: Path, artifact: dict, *, complete=True, hashes=None,
                include_job_id=False):
    execution = artifact["execution"]
    seed = artifact["config"]["campaign_seed"]
    hashes = HASHES if hashes is None else hashes
    return {
        "campaign_id": execution["campaign_id"],
        "campaign_cell": artifact["config"]["campaign_cell"],
        "attempt_id": execution["attempt_id"],
        "seed": seed,
        "submitted_utc": execution["submitted_utc"],
        "slurm_array_job_id": execution["slurm_array_job_id"],
        "slurm_array_task_id": seed,
        "slurm_job_id": execution["slurm_job_id"] if include_job_id else None,
        "artifact_path": path.name,
        "artifact_complete": complete,
        "artifact_sha256": merger.sha256_path(path) if complete else None,
        "expected_hashes": copy.deepcopy(hashes),
    }


def _fixture(tmp_path, *, hashes=None):
    hashes = HASHES if hashes is None else hashes
    protocol = _protocol()
    early_1 = _artifact(
        1, protocol, attempt="attempt-early",
        submitted="2026-08-14T00:00:00Z", hashes=hashes)
    late_1 = _artifact(
        1, protocol, attempt="attempt-late",
        submitted="2026-08-15T00:00:00Z", hashes=hashes)
    only_2 = _artifact(
        2, protocol, attempt="attempt-complete",
        submitted="2026-08-14T02:00:00Z", hashes=hashes)
    paths = [
        _write_artifact(tmp_path, early_1, "seed1-early.json"),
        _write_artifact(tmp_path, late_1, "seed1-late.json"),
        _write_artifact(tmp_path, only_2, "seed2-complete.json"),
    ]
    rows = [
        _ledger_row(paths[1], late_1, hashes=hashes),
        _ledger_row(paths[2], only_2, hashes=hashes),
        _ledger_row(paths[0], early_1, hashes=hashes),
        {
            **_ledger_row(paths[2], only_2, complete=False, hashes=hashes),
            "attempt_id": "attempt-incomplete",
            "submitted_utc": "2026-08-13T00:00:00+00:00",
            "artifact_path": "seed2-incomplete.json",
        },
    ]
    return protocol, paths, {"schema_version": 1, "submissions": rows}


def _select(tmp_path, protocol, paths, ledger, *, hashes=None):
    hashes = HASHES if hashes is None else hashes
    return selector.select_attempts(
        ledger=ledger, ledger_base=tmp_path, artifact_paths=paths,
        campaign_id="campaign-1", campaign_cell="primary",
        protocol=protocol, expected_seeds=(1, 2), ledger_sha256="8" * 64,
        **{f"expected_{field}": value for field, value in hashes.items()})


def _write_finalized_ledger(tmp_path: Path, ledger: dict, *, canonical=True):
    payload = (json.dumps(ledger, indent=2) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    name = (f"SUBMISSION_LEDGER.finalized-{digest}.json"
            if canonical else "ledger.json")
    directory = tmp_path / "campaign-1" / "finalized_ledgers"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(payload)
    return path


def _production_cli_fixture(tmp_path: Path):
    seeds = selector.PRODUCTION_SEEDS
    protocol = _protocol(seeds)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    hashes = {**HASHES, "protocol_sha256": merger.sha256_path(protocol_path)}
    rows = []
    primary_paths = []
    for cell_index, cell in enumerate(selector.PRODUCTION_CAMPAIGN_CELLS):
        for seed in seeds:
            artifact = _artifact(
                seed, protocol, campaign_cell=cell,
                attempt=f"attempt-{cell}", hashes=hashes)
            artifact["execution"]["slurm_array_job_id"] = str(
                50000 + cell_index)
            artifact["execution"]["slurm_job_id"] = str(
                60000 + 10 * cell_index + seed)
            path = (
                tmp_path / "campaign-1" / "cells" / cell / "attempts"
                / f"attempt-{cell}" / f"seed-{seed}" / "results"
                / f"seed-{seed}.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(artifact) + "\n")
            row = _ledger_row(
                path, artifact, hashes=hashes, include_job_id=True)
            row["artifact_path"] = str(path)
            rows.append(row)
            if cell == "primary":
                primary_paths.append(path)
    return (
        protocol_path, primary_paths,
        {"schema_version": 1, "submissions": rows}, hashes)


def _production_argv(*, protocol_path, paths, ledger_path, output, hashes):
    argv = [*(str(path) for path in paths), "--ledger", str(ledger_path),
            "--campaign-id", "campaign-1", "--campaign-cell", "primary",
            "--protocol", str(protocol_path), "--expected-seeds", "1,2,3,4,5",
            "--output", str(output)]
    for field, value in hashes.items():
        argv.extend(["--expected-" + field.replace("_", "-"), value])
    return argv


def test_normalize_row_preserves_array_task_execution_identity(tmp_path):
    _, _, ledger = _fixture(tmp_path)
    rows = selector.normalize_ledger(ledger, ledger_base=tmp_path)
    assert all(row["slurm_array_task_id"] == row["seed"] for row in rows)


def test_selects_earliest_complete_valid_attempt_without_outcomes(tmp_path):
    protocol, paths, ledger = _fixture(tmp_path)
    receipt = _select(tmp_path, protocol, list(reversed(paths)), ledger)
    assert [(row["seed"], row["attempt_id"]) for row in receipt["selected"]] == [
        (1, "attempt-early"), (2, "attempt-complete")]
    assert [row["reason"] for row in receipt["excluded"]] == [
        "later_complete_attempt", "incomplete_no_artifact"]
    serialized = json.dumps(receipt).lower()
    assert "mean_success" not in serialized and "auc" not in serialized


@pytest.mark.parametrize("mode,match", [
    ("omitted", "omitted"),
    ("unknown", "unknown"),
    ("hash", "artifact hash differs"),
    ("identity", "execution identity differs"),
    ("bindings", "expected_hashes differ"),
])
def test_closure_hash_identity_and_bindings_fail_closed(tmp_path, mode, match):
    protocol, paths, ledger = _fixture(tmp_path)
    supplied = list(paths)
    if mode == "omitted":
        supplied.pop()
    elif mode == "unknown":
        unknown = tmp_path / "unknown.json"
        unknown.write_text("{}\n")
        supplied.append(unknown)
    elif mode == "hash":
        ledger["submissions"][0]["artifact_sha256"] = "a" * 64
    elif mode == "identity":
        ledger["submissions"][0]["attempt_id"] = "different-attempt"
    else:
        ledger["submissions"][0]["expected_hashes"]["source_sha256"] = "a" * 64
    with pytest.raises(selector.AttemptSelectionError, match=match):
        _select(tmp_path, protocol, supplied, ledger)


def test_nullable_element_job_id_binds_by_array_and_seed(tmp_path):
    protocol, paths, ledger = _fixture(tmp_path)
    assert all(row["slurm_job_id"] is None for row in ledger["submissions"])
    receipt = _select(tmp_path, protocol, paths, ledger)
    assert len(receipt["selected"]) == 2

    ledger["submissions"][0]["slurm_job_id"] = "99999_1"
    with pytest.raises(selector.AttemptSelectionError, match="execution identity"):
        _select(tmp_path, protocol, paths, ledger)


def test_ambiguous_earliest_timestamp_is_rejected(tmp_path):
    protocol, paths, ledger = _fixture(tmp_path)
    ledger["submissions"][0]["submitted_utc"] = ledger["submissions"][2][
        "submitted_utc"]
    # Keep the artifact identity matched so ambiguity, not identity drift, wins.
    late = json.loads(paths[1].read_text())
    late["execution"]["submitted_utc"] = ledger["submissions"][0]["submitted_utc"]
    paths[1].write_text(json.dumps(late) + "\n")
    ledger["submissions"][0]["artifact_sha256"] = merger.sha256_path(paths[1])
    with pytest.raises(selector.AttemptSelectionError, match="ambiguous earliest"):
        _select(tmp_path, protocol, paths, ledger)


def test_incomplete_row_cannot_hide_existing_artifact(tmp_path):
    protocol, paths, ledger = _fixture(tmp_path)
    hidden = tmp_path / "seed2-incomplete.json"
    hidden.write_text("{}\n")
    with pytest.raises(selector.AttemptSelectionError, match="existing artifact"):
        _select(tmp_path, protocol, paths, ledger)


def test_cli_outputs_only_selection_metadata_after_full_campaign_closure(
        tmp_path, capsys):
    protocol_path, paths, ledger, hashes = _production_cli_fixture(tmp_path)
    ledger_path = _write_finalized_ledger(tmp_path, ledger)
    output = tmp_path / "selection.json"
    argv = _production_argv(
        protocol_path=protocol_path, paths=paths, ledger_path=ledger_path,
        output=output, hashes=hashes)
    assert selector.main(argv) == 0
    stdout = capsys.readouterr().out.lower()
    assert "selected=5" in stdout
    assert "mean_success" not in stdout and "auc" not in stdout
    receipt = json.loads(output.read_text())
    assert receipt["outcome_blind"] is True
    assert receipt["ledger_sha256"] == merger.sha256_path(ledger_path)
    assert {row["artifact_path"] for row in receipt["selected"]} <= {
        str(path.resolve()) for path in paths}


def test_cli_refuses_primary_only_ledger(tmp_path):
    protocol_path, paths, ledger, hashes = _production_cli_fixture(tmp_path)
    ledger["submissions"] = [
        row for row in ledger["submissions"]
        if row["campaign_cell"] == "primary"]
    ledger_path = _write_finalized_ledger(tmp_path, ledger)
    with pytest.raises(selector.AttemptSelectionError,
                       match="campaign-cell closure differs"):
        selector.main(_production_argv(
            protocol_path=protocol_path, paths=paths, ledger_path=ledger_path,
            output=tmp_path / "selection.json", hashes=hashes))


def test_cli_refuses_nonterminal_ledger_row(tmp_path):
    protocol_path, paths, ledger, hashes = _production_cli_fixture(tmp_path)
    ledger["submissions"][0]["slurm_job_id"] = None
    ledger_path = _write_finalized_ledger(tmp_path, ledger)
    with pytest.raises(selector.AttemptSelectionError,
                       match="numeric slurm_job_id"):
        selector.main(_production_argv(
            protocol_path=protocol_path, paths=paths, ledger_path=ledger_path,
            output=tmp_path / "selection.json", hashes=hashes))


def test_cli_refuses_cell_seed_without_complete_attempt(tmp_path):
    protocol_path, paths, ledger, hashes = _production_cli_fixture(tmp_path)
    row = next(
        item for item in ledger["submissions"]
        if item["campaign_cell"] == "ablation_n2" and item["seed"] == 1)
    row["artifact_complete"] = False
    row["artifact_sha256"] = None
    Path(row["artifact_path"]).unlink()
    ledger_path = _write_finalized_ledger(tmp_path, ledger)
    with pytest.raises(selector.AttemptSelectionError,
                       match="lacks a complete artifact attempt"):
        selector.main(_production_argv(
            protocol_path=protocol_path, paths=paths, ledger_path=ledger_path,
            output=tmp_path / "selection.json", hashes=hashes))


def test_cli_refuses_noncanonical_finalized_ledger_filename(tmp_path):
    protocol_path, paths, ledger, hashes = _production_cli_fixture(tmp_path)
    ledger_path = _write_finalized_ledger(tmp_path, ledger, canonical=False)
    with pytest.raises(selector.AttemptSelectionError,
                       match="canonical content-addressed finalized ledger"):
        selector.main(_production_argv(
            protocol_path=protocol_path, paths=paths, ledger_path=ledger_path,
            output=tmp_path / "selection.json", hashes=hashes))


@pytest.mark.parametrize("mode,match", [
    ("campaign", "campaign ID differs"),
    ("hash", "expected_hashes differ"),
    ("seed", "attempt group must contain exactly"),
])
def test_cli_refuses_cross_campaign_hash_or_seed_drift(tmp_path, mode, match):
    protocol_path, paths, ledger, hashes = _production_cli_fixture(tmp_path)
    row = ledger["submissions"][-1]
    if mode == "campaign":
        row["campaign_id"] = "campaign-2"
    elif mode == "hash":
        row["expected_hashes"]["source_sha256"] = "a" * 64
    else:
        row["seed"] = 6
        row["slurm_array_task_id"] = 6
    ledger_path = _write_finalized_ledger(tmp_path, ledger)
    with pytest.raises(selector.AttemptSelectionError, match=match):
        selector.main(_production_argv(
            protocol_path=protocol_path, paths=paths, ledger_path=ledger_path,
            output=tmp_path / "selection.json", hashes=hashes))


def test_cli_refuses_seed_fragmented_attempt_groups(tmp_path):
    protocol_path, paths, ledger, hashes = _production_cli_fixture(tmp_path)
    for row in ledger["submissions"]:
        if row["campaign_cell"] == "primary" and row["seed"] in {4, 5}:
            row["attempt_id"] = "attempt-primary-retry"
            row["slurm_array_job_id"] = "50004"
    ledger_path = _write_finalized_ledger(tmp_path, ledger)
    with pytest.raises(selector.AttemptSelectionError,
                       match="attempt group must contain exactly"):
        selector.main(_production_argv(
            protocol_path=protocol_path, paths=paths, ledger_path=ledger_path,
            output=tmp_path / "selection.json", hashes=hashes))


def test_cli_refuses_array_id_reused_across_attempt_groups(tmp_path):
    protocol_path, paths, ledger, hashes = _production_cli_fixture(tmp_path)
    for row in ledger["submissions"]:
        if row["campaign_cell"] == "ablation_n2":
            row["slurm_array_job_id"] = "50000"
    ledger_path = _write_finalized_ledger(tmp_path, ledger)
    with pytest.raises(selector.AttemptSelectionError,
                       match="array ID is reused"):
        selector.main(_production_argv(
            protocol_path=protocol_path, paths=paths, ledger_path=ledger_path,
            output=tmp_path / "selection.json", hashes=hashes))


def test_cli_refuses_submission_time_drift_within_attempt(tmp_path):
    protocol_path, paths, ledger, hashes = _production_cli_fixture(tmp_path)
    ledger["submissions"][0]["submitted_utc"] = "2026-08-15T00:00:00Z"
    ledger_path = _write_finalized_ledger(tmp_path, ledger)
    with pytest.raises(selector.AttemptSelectionError,
                       match="submission-time or expected-hash drift"):
        selector.main(_production_argv(
            protocol_path=protocol_path, paths=paths, ledger_path=ledger_path,
            output=tmp_path / "selection.json", hashes=hashes))


@pytest.mark.parametrize("mode,match", [
    ("missing", "complete artifact is missing"),
    ("tampered", "complete artifact hash differs"),
])
def test_cli_authenticates_non_target_complete_artifacts(
        tmp_path, mode, match):
    protocol_path, paths, ledger, hashes = _production_cli_fixture(tmp_path)
    row = next(
        item for item in ledger["submissions"]
        if item["campaign_cell"] == "ablation_n4" and item["seed"] == 3)
    artifact_path = Path(row["artifact_path"])
    if mode == "missing":
        artifact_path.unlink()
    else:
        artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
    ledger_path = _write_finalized_ledger(tmp_path, ledger)
    with pytest.raises(selector.AttemptSelectionError, match=match):
        selector.main(_production_argv(
            protocol_path=protocol_path, paths=paths, ledger_path=ledger_path,
            output=tmp_path / "selection.json", hashes=hashes))


def test_cli_refuses_noncanonical_artifact_path(tmp_path):
    protocol_path, paths, ledger, hashes = _production_cli_fixture(tmp_path)
    row = next(
        item for item in ledger["submissions"]
        if item["campaign_cell"] == "ablation_n16" and item["seed"] == 2)
    row["artifact_path"] += ".alias"
    ledger_path = _write_finalized_ledger(tmp_path, ledger)
    with pytest.raises(selector.AttemptSelectionError,
                       match="artifact path is not canonical"):
        selector.main(_production_argv(
            protocol_path=protocol_path, paths=paths, ledger_path=ledger_path,
            output=tmp_path / "selection.json", hashes=hashes))


def test_cli_refuses_existing_artifacts_for_incomplete_attempt(tmp_path):
    protocol_path, paths, ledger, hashes = _production_cli_fixture(tmp_path)
    failed_rows = []
    campaign_root = tmp_path / "campaign-1"
    for original in ledger["submissions"]:
        if original["campaign_cell"] != "ablation_n2":
            continue
        row = copy.deepcopy(original)
        seed = row["seed"]
        row["attempt_id"] = "attempt-ablation_n2-failed"
        row["slurm_array_job_id"] = "50999"
        row["slurm_job_id"] = str(60990 + seed)
        row["artifact_complete"] = False
        row["artifact_sha256"] = None
        path = (
            campaign_root / "cells" / "ablation_n2" / "attempts"
            / row["attempt_id"] / f"seed-{seed}" / "results"
            / f"seed-{seed}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(original["artifact_path"]).read_bytes())
        row["artifact_path"] = str(path)
        failed_rows.append(row)
    ledger["submissions"].extend(failed_rows)
    ledger_path = _write_finalized_ledger(tmp_path, ledger)
    with pytest.raises(selector.AttemptSelectionError,
                       match="incomplete artifact path must be absent"):
        selector.main(_production_argv(
            protocol_path=protocol_path, paths=paths, ledger_path=ledger_path,
            output=tmp_path / "selection.json", hashes=hashes))
