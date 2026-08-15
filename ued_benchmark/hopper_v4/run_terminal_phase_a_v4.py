#!/usr/bin/env python3
"""Run and close one arm of the bounded tie-aware v4 terminal Phase A."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import re

import evaluate_matched_terminal_v4 as evaluator
import run_matched_terminal_v4 as training


HASHES = {
    "protocol": "1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269",
    "contract": "3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b",
    "applied": "9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae",
    "frontier_config": "0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2",
    "maxmc_config": "a3cc3ddf387a3bb7cf3d9759c3f13e6a74f2c7e32de311ac344d54ef5e703ec6",
}
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
INPUT_CLOSURE_KEYS = {
    "schema", "status", "purpose", "endpoint_class", "paper_evidence",
    "analyzer_eligible", "endpoint_access_authorized", "production_authorized",
    "cost100_implemented", "max_student_updates", "arm", "training_seed",
    "job_id", "attempt", "from_last_checkpoint", "archive_interval",
    "periodic_checkpoint_used", "no_requeue", "source_commit", "source_tree",
    "bundle_manifest_sha256", "overlay_manifest_sha256",
    "applied_overlay_manifest_sha256", "environment_manifest_sha256",
    "sbatch_sha256", "phase_a_driver_sha256", "training_driver_sha256",
    "evaluation_driver_sha256", "assembler_sha256", "protocol_sha256",
    "finalizer_sha256", "config_path", "config_sha256", "prerequisites",
}


class PhaseAError(RuntimeError):
    """Raised when Phase A cannot publish a closed component tree."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PhaseAError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"unsafe {label}")
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs
    )
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    require(not temporary.exists(), f"stale temporary file: {temporary}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_config(path: Path, arm: str) -> str:
    digest = sha256(path)
    expected = HASHES[f"{arm}_config"]
    require(digest == expected, "arm/config hash mismatch")
    args = _load(path, "authored config").get("args")
    require(isinstance(args, dict), "config args missing")
    score = "coefficient_activity" if arm == "frontier" else "max_mc"
    require(args.get("ued_score") == [score], "score arm drift")
    require(args.get("plr_tie_aware_score_ranks") == [True], "tie-aware ranks disabled")
    require(args.get("n_parallel") == [4] and args.get("n_eval") == [8], "4x8 layout drift")
    temperature = float(args.get("plr_temp", [float("nan")])[0])
    require(math.isfinite(temperature) and temperature > 0.0, "temperature must be finite and positive")
    require(args.get("from_last_checkpoint") == [False], "resume-enabled config forbidden")
    return digest


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_archived_receipt(text: str, rung: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    lines = text.splitlines()
    require(lines and lines[0] == "field\tvalue", f"{rung} archived receipt header drift")
    for number, line in enumerate(lines[1:], 2):
        fields = line.split("\t", 1)
        require(
            len(fields) == 2 and fields[0] and fields[1] and fields[0] not in rows,
            f"{rung} archived receipt line {number} drift",
        )
        rows[fields[0]] = fields[1]
    return rows


def _validate_archived_prerequisite(
    rung: str,
    prerequisite: Mapping[str, Any],
    *,
    bundle_manifest_sha256: str,
) -> None:
    archive = prerequisite.get("archived_provenance")
    require(
        isinstance(archive, dict)
        and set(archive) == {"schema", "files"}
        and archive.get("schema") == 1,
        f"{rung} archived provenance schema drift",
    )
    files = archive.get("files")
    require(
        isinstance(files, dict)
        and set(files) == {"receipt.tsv", "SHA256SUMS", "COMPLETE"},
        f"{rung} archived provenance files drift",
    )
    texts: dict[str, str] = {}
    for name, record in files.items():
        require(
            isinstance(record, dict)
            and set(record) == {"encoding", "text", "sha256"}
            and record.get("encoding") == "utf-8"
            and isinstance(record.get("text"), str)
            and HASH_RE.fullmatch(str(record.get("sha256"))) is not None,
            f"{rung} archived {name} record drift",
        )
        encoded = record["text"].encode("utf-8")
        require(
            hashlib.sha256(encoded).hexdigest() == record["sha256"],
            f"{rung} archived {name} byte hash drift",
        )
        texts[name] = record["text"]
    require(
        files["SHA256SUMS"]["sha256"] == prerequisite["manifest_sha256"]
        and files["COMPLETE"]["sha256"] == prerequisite["complete_sha256"],
        f"{rung} archived closure hash binding drift",
    )
    listed: dict[str, str] = {}
    for line in texts["SHA256SUMS"].splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"{rung} archived manifest row drift")
        digest, raw_name = match.groups()
        name = raw_name.removeprefix("./")
        relative = PurePosixPath(name)
        require(
            name
            and not relative.is_absolute()
            and all(part not in {"", ".", ".."} for part in relative.parts)
            and relative.as_posix() not in listed,
            f"{rung} archived manifest path drift",
        )
        listed[relative.as_posix()] = digest
    require(
        listed.get("receipt.tsv") == files["receipt.tsv"]["sha256"],
        f"{rung} archived receipt/manifest binding drift",
    )
    receipt = _parse_archived_receipt(texts["receipt.tsv"], rung)
    result_path = PurePosixPath(prerequisite["result_dir"])
    require(
        result_path.is_absolute()
        and all(part not in {"", ".", ".."} for part in result_path.parts)
        and receipt.get("job_id") == result_path.name
        and receipt.get("result_dir") == prerequisite["result_dir"]
        and receipt.get("bundle_manifest_sha256") == bundle_manifest_sha256
        and receipt.get("applied_overlay_manifest_sha256") == HASHES["applied"],
        f"{rung} archived receipt provenance drift",
    )
    if rung == "import":
        require(
            receipt.get("training_endpoint") == "false"
            and re.fullmatch(
                r"complete\t[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\n",
                texts["COMPLETE"],
            ) is not None,
            "import archived completion semantics drift",
        )
    else:
        require(
            receipt.get("endpoint_class") == "bounded_engineering_one_update"
            and receipt.get("actual_student_updates") == "1"
            and receipt.get("paper_evidence") == "false"
            and receipt.get("config_sha256") == HASHES["frontier_config"],
            "one_update archived receipt semantics drift",
        )
        try:
            complete = json.loads(
                texts["COMPLETE"], object_pairs_hook=_unique_json_pairs
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise PhaseAError(
                "one_update archived completion is invalid JSON"
            ) from exc
        require(
            isinstance(complete, dict)
            and set(complete) == {
                "complete_schema", "artifact_type", "job_id", "paper_evidence",
                "actual_ppo_updates", "n_grad_updates", "ppo_epochs",
                "ppo_minibatches", "optimizer_step_applications",
                "resource_accounting_source", "external_accounting_authority",
                "terminal_sacct_included", "input_closure_sha256",
                "sha256sums_sha256",
            }
            and complete.get("complete_schema") == 2
            and complete.get("artifact_type")
            == "frontier_exact_grouped_one_update_engineering"
            and complete.get("paper_evidence") is False
            and complete.get("job_id") == receipt.get("job_id")
            and complete.get("input_closure_sha256")
            == receipt.get("input_closure_sha256")
            and complete.get("actual_ppo_updates")
            == complete.get("n_grad_updates") == 1
            and complete.get("terminal_sacct_included") is False
            and complete.get("sha256sums_sha256")
            == prerequisite["manifest_sha256"],
            "one_update archived completion semantics drift",
        )


def _launcher_command(
    python: Path,
    source: Path,
    script: Path,
    arguments: Sequence[str],
) -> list[str]:
    launcher = (
        "import runpy,sys; from pathlib import Path; "
        "source=sys.argv.pop(1); script=sys.argv.pop(1); "
        "sys.path[:0]=[source,str(Path(script).parent)]; sys.argv[0]=script; "
        "runpy.run_path(script,run_name='__main__')"
    )
    return [
        str(python), "-I", "-B", "-c", launcher,
        str(source / "src"), str(script), *arguments,
    ]


def _run(command: Sequence[str], *, environment: Mapping[str, str], stdout: Path, stderr: Path) -> None:
    with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
        completed = subprocess.run(
            list(command), env=dict(environment), cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.DEVNULL, stdout=out, stderr=err, check=False,
        )
    if completed.returncode != 0:
        diagnostic = stderr.read_text(encoding="utf-8", errors="replace")[-4000:].strip()
        raise PhaseAError(
            f"subprocess refused with exit {completed.returncode}: {diagnostic}"
        )


def _campaign_context(
    cli: argparse.Namespace,
    root: Path,
    config_sha: str,
) -> tuple[Path, str, Path, str, str]:
    protocol_sha = sha256(cli.protocol)
    require(protocol_sha == HASHES["protocol"], "protocol hash drift")
    driver_root = Path(__file__).resolve().parent
    train_driver = driver_root / "run_matched_terminal_v4.py"
    eval_driver = driver_root / "evaluate_matched_terminal_v4.py"
    assembler = driver_root / "assemble_matched_run_v4.py"
    expected = {
        "training": cli.expected_training_driver_sha256,
        "evaluation": cli.expected_evaluation_driver_sha256,
        "assembler": cli.expected_assembler_sha256,
    }
    actual = {
        "training": sha256(train_driver),
        "evaluation": sha256(eval_driver),
        "assembler": sha256(assembler),
    }
    require(actual == expected, "terminal helper hash drift")
    run_id = (
        f"engineering-{cli.arm}-s101"
        if cli.local_test_mode
        else f"engineering-slurm-{cli.job_id}-{cli.arm}-s101"
    )
    analyzer = driver_root.parents[0] / "analysis" / "preregistered_dev_analysis.py"
    provenance = {
        "base_commit": training.BASE_COMMIT,
        "base_tree": training.BASE_TREE,
        "overlay_contract_sha256": HASHES["contract"],
        "bundle_manifest_sha256": cli.bundle_manifest_sha256,
        "overlay_manifest_sha256": cli.overlay_manifest_sha256,
        "applied_overlay_manifest_sha256": cli.applied_overlay_manifest_sha256,
        "environment_manifest_sha256": cli.environment_manifest_sha256,
        "training_driver_sha256": actual["training"],
        "evaluation_driver_sha256": actual["evaluation"],
        "sbatch_sha256": cli.sbatch_sha256,
        "assembler_driver_sha256": actual["assembler"],
    }
    campaign = {
        "schema": 1,
        "protocol_id": training.PROTOCOL_ID,
        "purpose": training.PURPOSE,
        "created_utc": _utc(),
        "frozen_before_endpoint_access": True,
        "protocol_sha256": protocol_sha,
        "analyzer_sha256": sha256(analyzer),
        "provenance": provenance,
        "hardware": {
            "partition": "local" if cli.local_test_mode else "gpuq",
            "gpu_model": "CPU local fixture" if cli.local_test_mode else "NVIDIA A100",
            "gpu_profile": "local-cpu" if cli.local_test_mode else "1g.10gb",
            "gpu_count": 1,
            "n_devices": 1,
        },
        "submissions": [{
            "arm": cli.arm,
            "training_seed": 101,
            "evaluation_seed": 100101,
            "run_id": run_id,
            "job_id": cli.job_id,
            "attempt": 1,
        }],
    }
    campaign_path = root / "campaign-manifest.json"
    _write(campaign_path, campaign)
    campaign_sha = sha256(campaign_path)
    context_provenance = {
        key: provenance[key]
        for key in ({"base_commit", "base_tree", "overlay_contract_sha256"} | training.HASH_KEYS)
    }
    context = {
        "schema": 1,
        "protocol_id": training.PROTOCOL_ID,
        "purpose": training.PURPOSE,
        "run_id": run_id,
        "arm": cli.arm,
        "training_seed": 101,
        "job_id": cli.job_id,
        "campaign_manifest_sha256": campaign_sha,
        "provenance": context_provenance,
    }
    context_path = root / "run-context.json"
    _write(context_path, context)
    return campaign_path, campaign_sha, context_path, sha256(context_path), run_id


def _postcheck_sidecar(
    root: Path, run_id: str, arm: str, *, local_test_mode: bool
) -> tuple[str, dict[str, Any]]:
    manifest_sha = training.validate_training_sidecar(root, run_id, arm)
    receipt = _load(root / "training-receipt.json", "training receipt")
    snapshot_path = root / "plr-replay-snapshot.json"
    snapshot = _load(snapshot_path, "PLR replay snapshot")
    require(receipt.get("plr_snapshot") == {
        "path": "plr-replay-snapshot.json", "sha256": sha256(snapshot_path)
    }, "snapshot receipt binding drift")
    provenance = receipt.get("provenance")
    require(isinstance(provenance, dict), "training provenance missing")
    expected_backend = "cpu" if local_test_mode else "gpu"
    require(provenance.get("backend") == expected_backend, "training backend drift")
    devices = provenance.get("devices")
    require(isinstance(devices, list) and len(devices) == 1, "training device cardinality drift")
    device = devices[0]
    require(
        isinstance(device, dict)
        and device.get("platform") == expected_backend
        and isinstance(device.get("device_kind"), str)
        and device["device_kind"],
        "training device receipt drift",
    )
    if not local_test_mode:
        require("A100" in device["device_kind"], "terminal rung requires an A100 MIG device")
    require(snapshot.get("protocol_id") == training.PROTOCOL_ID, "snapshot protocol drift")
    require(snapshot.get("purpose") == training.PURPOSE, "snapshot purpose drift")
    require(snapshot.get("paper_evidence") is False, "snapshot evidence label drift")
    require(
        snapshot.get("schema") == 1
        and snapshot.get("status") == "completed"
        and snapshot.get("kind") == "tie_aware_plr_buffer_safe_snapshot",
        "snapshot schema/kind drift",
    )
    require(snapshot.get("run_id") == run_id and snapshot.get("arm") == arm, "snapshot run/arm drift")
    require(snapshot.get("training_seed") == 101, "snapshot seed drift")
    require(snapshot.get("checkpoint_sha256") == receipt["terminal_checkpoint"]["sha256"], "snapshot checkpoint drift")
    distribution = snapshot.get("replay_distribution")
    require(isinstance(distribution, dict), "replay distribution receipt missing")
    require(distribution.get("tie_aware_score_ranks") is True, "tie-aware mode disabled")
    temperature = float(distribution.get("temperature", float("nan")))
    require(math.isfinite(temperature) and temperature > 0.0, "snapshot temperature invalid")
    require(distribution.get("score_normalization_order") == "canonical_ascending_unnormalized_mass", "snapshot normalization-order drift")
    require(distribution.get("tie_equality") == "exact filled-score equality; +0 and -0 tie", "snapshot tie equality drift")
    filled_count = snapshot.get("filled_count")
    require(isinstance(filled_count, int) and 1 <= filled_count <= snapshot.get("buffer_size", 0), "snapshot filled count drift")
    blocks = distribution.get("tie_block_sizes_descending_score_order")
    require(
        isinstance(blocks, list)
        and blocks
        and all(isinstance(size, int) and size > 0 for size in blocks)
        and sum(blocks) == filled_count
        and distribution.get("distinct_filled_score_count") == len(blocks),
        "snapshot tie-block receipt drift",
    )
    tied = distribution.get("tied_block_sizes_descending_score_order")
    require(tied == [size for size in blocks if size > 1], "snapshot tied-block receipt drift")
    for key in (
        "normalization_sum", "filled_slot_float64_normalization_sum",
        "pinned_implementation_normalization_sum",
    ):
        value = float(distribution.get(key, float("nan")))
        require(math.isfinite(value) and abs(value - 1.0) <= 2e-6, f"invalid {key}")
    for key in ("score_effective_support", "replay_effective_support"):
        value = float(distribution.get(key, float("nan")))
        require(math.isfinite(value) and 1.0 <= value <= snapshot["filled_count"] + 2e-6, f"invalid {key}")
    replay = snapshot.get("sampling_diagnostics")
    require(isinstance(replay, dict), "sampling diagnostics missing")
    training.validate_replay_integrity({"replay_integrity": replay})
    terminal_replay = receipt.get("integrity", {}).get("terminal", {}).get("replay_integrity")
    checkpoint_replay = receipt.get("integrity", {}).get("checkpoint_round_trip", {}).get("replay_integrity")
    require(replay == terminal_replay == checkpoint_replay, "snapshot/terminal/checkpoint replay drift")
    require(replay.get("nonfinite_filled_score_count") == 0, "nonfinite filled score")
    require(replay.get("nonfinite_score_rejection_count") == 0, "nonfinite score rejection")
    require(replay.get("force_unique_resamples_replay") is False, "force_unique sampling semantics drift")
    require(replay.get("replay_group_draw_count") == 4, "cumulative replay draw drift")
    require(replay.get("last_replay_group_count") == 4, "last replay draw drift")
    require(
        replay.get("replay_distinct_group_count") + replay.get("replay_duplicate_group_count") == 4
        and replay.get("last_replay_distinct_group_count") + replay.get("last_replay_duplicate_group_count") == 4,
        "replay distinct/duplicate accounting drift",
    )
    require(
        snapshot.get("sampling_semantics")
        == {
            "replacement": "with_replacement",
            "force_unique_effect": "buffer-update deduplication only; no replay resampling",
            "distinct_identity": "replay buffer slot index",
            "last_counts_persist_until_next_actual_replay_batch": True,
        },
        "snapshot sampling semantics drift",
    )
    slots = snapshot.get("slots")
    require(isinstance(slots, list) and len(slots) == snapshot["filled_count"], "snapshot slots drift")
    posterior_keys = {
        "success_count", "trial_count", "analytic_expected_activity_score",
        "mean_plugin_score", "jensen_gap", "stored_score_abs_error",
    }
    for slot in slots:
        require(slot.get("normalized_score_probability") is not None, "score mass missing")
        require(slot.get("normalized_replay_probability") is not None, "replay mass missing")
        for key in ("stored_score", "normalized_score_probability", "normalized_replay_probability"):
            require(math.isfinite(float(slot[key])), f"nonfinite snapshot slot {key}")
        if arm == "frontier":
            require(posterior_keys <= set(slot), "Frontier posterior fields missing")
        else:
            require(posterior_keys.isdisjoint(slot), "MaxMC received Frontier posterior fields")
    require((snapshot.get("stored_score_validation") is None) is (arm == "maxmc"), "arm score validation drift")
    require(
        abs(sum(float(slot["normalized_score_probability"]) for slot in slots) - 1.0) <= 1e-10
        and abs(sum(float(slot["normalized_replay_probability"]) for slot in slots) - 1.0) <= 1e-10,
        "filled-slot probability normalization drift",
    )
    return manifest_sha, receipt


def _write_evaluation_integrity(
    evaluation_root: Path, run_id: str, arm: str, package_sha: str, output: Path,
    *, local_test_mode: bool,
) -> str:
    """Parse only the evaluator's separately closed value-free projection."""
    receipt_path = evaluation_root / "evaluation-receipt.json"
    integrity_receipt_path = evaluation_root / "evaluation-integrity-receipt.json"
    receipt = _load(integrity_receipt_path, "evaluation integrity receipt")
    require(
        set(receipt) == {
            "schema", "status", "protocol_id", "purpose", "paper_evidence",
            "performance_fields_included", "run_id", "arm", "training_seed",
            "evaluation_seed", "synthetic_test_mode", "raw_results",
            "evaluation_transition_accounting", "runtime",
            "evaluation_receipt_sha256",
        },
        "evaluation integrity receipt fields drift",
    )
    require(
        receipt.get("schema") == 1
        and receipt.get("status") == "completed"
        and receipt.get("protocol_id") == training.PROTOCOL_ID
        and receipt.get("purpose") == training.PURPOSE
        and receipt.get("paper_evidence") is False,
        "evaluation receipt protocol/evidence drift",
    )
    require(
        receipt.get("performance_fields_included") is False
        and receipt.get("evaluation_receipt_sha256") == sha256(receipt_path),
        "evaluation integrity/full-receipt binding drift",
    )
    require(
        receipt.get("run_id") == run_id
        and receipt.get("arm") == arm
        and receipt.get("training_seed") == 101
        and receipt.get("evaluation_seed") == 100101,
        "evaluation receipt identity drift",
    )
    require(receipt.get("synthetic_test_mode") is False, "synthetic evaluation forbidden")
    raw = receipt.get("raw_results")
    require(
        isinstance(raw, dict)
        and set(raw) == {"path", "sha256", "record_count"}
        and raw.get("path") == "evaluation-episodes.jsonl"
        and raw.get("record_count") == 30
        and raw.get("sha256") == sha256(evaluation_root / "evaluation-episodes.jsonl"),
        "evaluation raw-result closure drift",
    )
    accounting = receipt.get("evaluation_transition_accounting")
    require(
        isinstance(accounting, dict)
        and set(accounting) == {
            "environment_count", "episodes_per_environment",
            "max_episode_horizon", "per_environment_max_episode_horizons",
            "budgeted_primary_max_transitions", "effective_primary_transitions",
            "primary_runner_scans_full_horizon",
            "engineering_independent_verification_transitions",
            "total_runtime_transitions",
            "excluded_from_student_training_transitions",
        }
        and accounting.get("environment_count") == 3
        and accounting.get("episodes_per_environment") == 10
        and accounting.get("max_episode_horizon") == 450
        and accounting.get("per_environment_max_episode_horizons") == [450, 450, 450]
        and accounting.get("budgeted_primary_max_transitions") == 13_500
        and accounting.get("effective_primary_transitions") == 13_500
        and accounting.get("primary_runner_scans_full_horizon") is True
        and accounting.get("engineering_independent_verification_transitions") == 0
        and accounting.get("total_runtime_transitions") == 13_500
        and accounting.get("excluded_from_student_training_transitions") is True,
        "evaluation transition accounting drift",
    )
    runtime = receipt.get("runtime")
    expected_backend = "cpu" if local_test_mode else "gpu"
    require(
        isinstance(runtime, dict)
        and set(runtime) == {"backend", "device_count", "devices"}
        and runtime.get("backend") == expected_backend
        and runtime.get("device_count") == 1
        and isinstance(runtime.get("devices"), list)
        and len(runtime["devices"]) == 1
        and isinstance(runtime["devices"][0], dict)
        and set(runtime["devices"][0]) == {"id", "platform", "device_kind"}
        and runtime["devices"][0].get("platform") == expected_backend,
        "evaluation runtime device drift",
    )
    if not local_test_mode:
        require(
            "A100" in str(runtime["devices"][0].get("device_kind", "")),
            "evaluation requires an A100 MIG device",
        )
    integrity = {
        "schema": 1,
        "status": "complete",
        "paper_evidence": False,
        "analyzer_eligible": False,
        "performance_values_copied": False,
        "run_id": run_id,
        "arm": arm,
        "training_seed": 101,
        "evaluation_seed": 100101,
        "synthetic_test_mode": False,
        "raw_record_count": 30,
        "primary_max_transitions": 13_500,
        "evaluation_receipt_sha256": sha256(receipt_path),
        "evaluation_package_manifest_sha256": package_sha,
    }
    _write(output, integrity)
    return sha256(output)


def _validate_input_closure(
    closure: Mapping[str, Any], cli: argparse.Namespace, config_sha: str
) -> None:
    require(set(closure) == INPUT_CLOSURE_KEYS, "input closure keys drift")
    require(closure.get("schema") == 1 and closure.get("status") == "frozen_before_phase_a", "input closure schema drift")
    require(closure.get("purpose") == training.PURPOSE, "input closure purpose drift")
    require(closure.get("endpoint_class") == "bounded_engineering_terminal_chain_components_v4", "input closure endpoint class drift")
    for key in (
        "paper_evidence", "analyzer_eligible", "endpoint_access_authorized",
        "production_authorized", "cost100_implemented", "from_last_checkpoint",
        "periodic_checkpoint_used",
    ):
        require(closure.get(key) is False, f"input closure must disable {key}")
    require(closure.get("max_student_updates") == 1, "input closure update ceiling drift")
    require(closure.get("training_seed") == 101 and closure.get("attempt") == 1, "input closure seed/attempt drift")
    require(closure.get("archive_interval") == 0, "input closure archive drift")
    require(closure.get("no_requeue") is True, "input closure requeue drift")
    require(closure.get("arm") == cli.arm and closure.get("job_id") == cli.job_id, "input closure execution identity drift")
    require(closure.get("source_commit") == training.BASE_COMMIT, "input closure source commit drift")
    require(closure.get("source_tree") == training.BASE_TREE, "input closure source tree drift")
    expected = {
        "bundle_manifest_sha256": cli.bundle_manifest_sha256,
        "overlay_manifest_sha256": cli.overlay_manifest_sha256,
        "applied_overlay_manifest_sha256": cli.applied_overlay_manifest_sha256,
        "environment_manifest_sha256": cli.environment_manifest_sha256,
        "sbatch_sha256": cli.sbatch_sha256,
        "phase_a_driver_sha256": cli.expected_phase_a_driver_sha256,
        "training_driver_sha256": cli.expected_training_driver_sha256,
        "evaluation_driver_sha256": cli.expected_evaluation_driver_sha256,
        "assembler_sha256": cli.expected_assembler_sha256,
        "finalizer_sha256": cli.expected_finalizer_sha256,
        "protocol_sha256": HASHES["protocol"],
        "config_sha256": config_sha,
    }
    for key, value in expected.items():
        require(HASH_RE.fullmatch(str(value)) is not None, f"malformed expected hash: {key}")
        require(closure.get(key) == value, f"input closure hash drift: {key}")
    expected_name = (
        "maze_frontier_exact_grouped_n8_tie_aware_v4.json"
        if cli.arm == "frontier"
        else "maze_maxmc_group_matched_4x8_b500_tie_aware_v4.json"
    )
    require(closure.get("config_path") == f"ued_benchmark/configs/{expected_name}", "input closure config path drift")
    prerequisites = closure.get("prerequisites")
    require(isinstance(prerequisites, dict) and set(prerequisites) == {"import", "one_update"}, "input closure prerequisite keys drift")
    for rung, expected_updates in (("import", 0), ("one_update", 1)):
        receipt = prerequisites[rung]
        require(
            isinstance(receipt, dict)
            and set(receipt) == {
                "result_dir", "manifest_sha256", "complete_sha256",
                "bundle_manifest_sha256", "actual_student_updates",
                "paper_evidence", "analyzer_eligible", "archived_provenance",
            },
            f"{rung} prerequisite receipt keys drift",
        )
        require(isinstance(receipt["result_dir"], str) and receipt["result_dir"], f"{rung} result path missing")
        require(receipt["bundle_manifest_sha256"] == cli.bundle_manifest_sha256, f"{rung} cross-bundle prerequisite")
        require(receipt["actual_student_updates"] == expected_updates, f"{rung} update receipt drift")
        require(receipt["paper_evidence"] is False and receipt["analyzer_eligible"] is False, f"{rung} evidence label drift")
        for key in ("manifest_sha256", "complete_sha256", "bundle_manifest_sha256"):
            require(HASH_RE.fullmatch(str(receipt[key])) is not None, f"malformed {rung} hash: {key}")
        _validate_archived_prerequisite(
            rung, receipt, bundle_manifest_sha256=cli.bundle_manifest_sha256
        )


def run(cli: argparse.Namespace) -> tuple[Path, str]:
    for name in (
        "protocol", "config", "patched_source_dir", "input_closure", "output_dir",
    ):
        setattr(cli, name, getattr(cli, name).resolve())
    require(cli.python.is_absolute() and cli.python.is_file() and os.access(cli.python, os.X_OK), "unsafe Python executable")
    require(cli.git_executable.is_absolute() and cli.git_executable.is_file() and os.access(cli.git_executable, os.X_OK), "unsafe Git executable")
    require(cli.arm in training.ARMS, "invalid arm")
    require(sha256(Path(__file__).resolve()) == cli.expected_phase_a_driver_sha256, "Phase-A driver hash drift")
    require(cli.job_id == ("local-test" if cli.local_test_mode else os.environ.get("SLURM_JOB_ID")), "job identity drift")
    require(cli.applied_overlay_manifest_sha256 == HASHES["applied"], "applied overlay drift")
    applied_manifest = cli.patched_source_dir / ".frontierrl_overlay.json"
    require(
        applied_manifest.is_file()
        and not applied_manifest.is_symlink()
        and sha256(applied_manifest) == cli.applied_overlay_manifest_sha256,
        "patched source applied-manifest drift",
    )
    require(sha256(cli.input_closure) == cli.expected_input_closure_sha256, "input closure hash drift")
    closure = _load(cli.input_closure, "input closure")
    config_sha = _validate_config(cli.config, cli.arm)
    _validate_input_closure(closure, cli, config_sha)
    require(sha256(cli.protocol) == HASHES["protocol"], "protocol drift")
    require(cli.output_dir.is_absolute(), "output path must be absolute")
    require(not cli.output_dir.exists() and not cli.output_dir.is_symlink(), "output exists")
    require(cli.output_dir.parent.is_dir() and not cli.output_dir.parent.is_symlink(), "unsafe output parent")
    temporary = Path(tempfile.mkdtemp(prefix=f".{cli.output_dir.name}.", dir=cli.output_dir.parent))
    try:
        shutil.copy2(cli.input_closure, temporary / "INPUT_CLOSURE.json", follow_symlinks=False)
        campaign, campaign_sha, context, context_sha, run_id = _campaign_context(cli, temporary, config_sha)
        # The frozen training driver requires the output basename to be the
        # exact run ID.  Phase A maps that closed source directory to the
        # arm-neutral package name only after evaluation consumes it.
        output = temporary / run_id
        sidecar = temporary / "training-sidecar"
        evaluation_output = temporary / "evaluation-package"
        environment = os.environ.copy()
        environment.update({
            "JAX_PLATFORMS": "cpu" if cli.local_test_mode else "cuda",
            "JAX_PLATFORM_NAME": "cpu" if cli.local_test_mode else "gpu",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "WANDB_MODE": "disabled",
        })
        train_args = [
            "--arm", cli.arm, "--config", str(cli.config),
            "--protocol", str(cli.protocol),
            "--campaign-manifest", str(campaign),
            "--expected-campaign-manifest-sha256", campaign_sha,
            "--run-context", str(context),
            "--expected-run-context-sha256", context_sha,
            "--expected-driver-sha256", cli.expected_training_driver_sha256,
            "--patched-source-dir", str(cli.patched_source_dir),
            "--git-executable", str(cli.git_executable),
            "--output-dir", str(output), "--sidecar-dir", str(sidecar),
            "--engineering-test-mode" if cli.local_test_mode else "--slurm-engineering-test-mode",
        ]
        overrides = [
            "n_total_updates=1", "test_interval=0", "log_interval=1",
            "train_runner_args.buffer_size=8", "train_runner_args.replay_prob=1.0",
            "train_runner_args.min_fill_ratio=0.5", "driver.max_outer_cycles=4",
        ]
        if cli.local_test_mode:
            overrides.extend([
                "train_runner_args.n_rollout_steps=2", "train_runner_args.n_unroll_rollout=1",
                "env_args.max_episode_steps=2", "student_rl_args.n_unroll_update=1",
                "student_rl_args.n_epochs=1", "student_model_args.hidden_dim=16",
                "student_model_args.recurrent_hidden_dim=16", "student_model_args.n_conv_filters=4",
            ])
        for override in overrides:
            train_args.extend(("--engineering-override", override))
        _run(
            _launcher_command(cli.python, cli.patched_source_dir, Path(__file__).resolve().parent / "run_matched_terminal_v4.py", train_args),
            environment=environment, stdout=temporary / "training.stdout", stderr=temporary / "training.stderr",
        )
        sidecar_sha, training_receipt = _postcheck_sidecar(
            sidecar, run_id, cli.arm, local_test_mode=cli.local_test_mode
        )
        require(training_receipt.get("resumed") is False, "resume forbidden")
        require(training_receipt.get("n_updates") == 1, "training update count drift")
        eval_args = [
            "--arm", cli.arm, "--protocol", str(cli.protocol),
            "--campaign-manifest", str(campaign),
            "--expected-campaign-manifest-sha256", campaign_sha,
            "--run-context", str(context), "--expected-run-context-sha256", context_sha,
            "--expected-driver-sha256", cli.expected_evaluation_driver_sha256,
            "--patched-source-dir", str(cli.patched_source_dir),
            "--git-executable", str(cli.git_executable),
            "--checkpoint", str(output / "checkpoint.pkl"),
            "--endpoint", str(output / "endpoint.json"),
            "--training-receipt", str(sidecar / "training-receipt.json"),
            "--meta", str(output / "meta.json"), "--output-dir", str(evaluation_output),
            "--engineering-test-mode" if cli.local_test_mode else "--slurm-engineering-test-mode",
        ]
        _run(
            _launcher_command(cli.python, cli.patched_source_dir, Path(__file__).resolve().parent / "evaluate_matched_terminal_v4.py", eval_args),
            environment=environment, stdout=temporary / "evaluation.stdout", stderr=temporary / "evaluation.stderr",
        )
        evaluation_sha = evaluator.validate_package(evaluation_output, run_id)
        # The evaluation receipt/CSV/episode JSONL remain sealed here.
        evaluation_integrity_sha = _write_evaluation_integrity(
            evaluation_output,
            run_id,
            cli.arm,
            evaluation_sha,
            temporary / "evaluation-integrity.json",
            local_test_mode=cli.local_test_mode,
        )
        os.replace(output, temporary / "training-output")
        payloads = sorted(
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file()
            and path.relative_to(temporary).as_posix()
            not in {"SHA256SUMS", "COMPONENTS_COMPLETE.json"}
        )
        (temporary / "SHA256SUMS").write_text(
            "".join(f"{sha256(temporary / name)}  {name}\n" for name in payloads),
            encoding="utf-8",
        )
        manifest_sha = sha256(temporary / "SHA256SUMS")
        _write(
            temporary / "COMPONENTS_COMPLETE.json",
            {
                "schema": 1, "status": "complete", "paper_evidence": False,
                "analyzer_eligible": False,
                "endpoint_class": "bounded_engineering_terminal_chain_components_v4",
                "job_id": cli.job_id, "run_id": run_id, "arm": cli.arm,
                "bundle_manifest_sha256": cli.bundle_manifest_sha256,
                "input_closure_sha256": cli.expected_input_closure_sha256,
                "campaign_manifest_sha256": campaign_sha,
                "run_context_sha256": context_sha,
                "training_sidecar_manifest_sha256": sidecar_sha,
                "evaluation_package_manifest_sha256": evaluation_sha,
                "evaluation_integrity_sha256": evaluation_integrity_sha,
                "actual_student_updates": 1, "actual_external_evaluation": True,
                "raw_evaluation_records": 30,
                "student_training_transitions": training_receipt["student_training_transitions"],
                "optimizer_step_applications": training_receipt["optimizer_step_applications"],
                "outer_cycles": training_receipt["outer_cycles"],
                "training_wall_seconds": training_receipt["wall_seconds"],
                "primary_evaluation_max_transitions": 13_500,
                "terminal_sacct_included": False, "phase_b_required": True,
                "config_sha256": config_sha,
            },
        )
        os.replace(temporary, cli.output_dir)
        return cli.output_dir, manifest_sha
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=training.ARMS, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--patched-source-dir", type=Path, required=True)
    parser.add_argument("--git-executable", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--input-closure", type=Path, required=True)
    parser.add_argument("--expected-input-closure-sha256", required=True)
    parser.add_argument("--bundle-manifest-sha256", required=True)
    parser.add_argument("--overlay-manifest-sha256", required=True)
    parser.add_argument("--applied-overlay-manifest-sha256", required=True)
    parser.add_argument("--environment-manifest-sha256", required=True)
    parser.add_argument("--sbatch-sha256", required=True)
    parser.add_argument("--expected-phase-a-driver-sha256", required=True)
    parser.add_argument("--expected-training-driver-sha256", required=True)
    parser.add_argument("--expected-evaluation-driver-sha256", required=True)
    parser.add_argument("--expected-assembler-sha256", required=True)
    parser.add_argument("--expected-finalizer-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-test-mode", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        output, digest = run(parse_cli(argv))
    except (PhaseAError, training.DriverError, evaluator.EvaluationError, OSError, ValueError, KeyError) as exc:
        print(f"V4_TERMINAL_PHASE_A_REFUSED: {exc}", file=os.sys.stderr)
        return 1
    print(
        "V4_TERMINAL_PHASE_A_COMPLETE "
        f"manifest={digest} result={output} analyzer_eligible=false phase_b_required=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
